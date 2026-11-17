import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import subprocess
import sys

from cosmosr.cli import main, prepare_output
from cosmosr.config import build_command, load_config
from cosmosr.data_check import validate_data


ROOT = Path(__file__).resolve().parents[1]


def write_case(root, name="case_0", n=8):
    rng = np.random.default_rng(7)
    for split, size in (("LR_selected_100_train", n), ("HR_selected_100_train", 2 * n)):
        directory = root / split / name
        directory.mkdir(parents=True)
        for field, scale in (("disp.npy", 1000.0), ("vel.npy", 100.0)):
            np.save(
                directory / field,
                (rng.standard_normal((3, size, size, size)) * scale).astype(np.float32),
            )
    style = np.array([1.0, 0.3, 0.05, 0.7, 0.96, 0.8], dtype=np.float32)
    np.save(root / "LR_selected_100_train" / name / "style.npy", style)


def tiny_config():
    config = load_config(ROOT / "configs/state56.json")
    config.update(
        crop=8,
        loader_workers=0,
        epochs=4,
        save_interval=1,
        gp_interval=1,
        log_interval=1,
    )
    return config


class LauncherTests(unittest.TestCase):
    def test_dry_run_needs_no_torch_or_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "train.py",
                    "--config",
                    "configs/state56.json",
                    "--data-root",
                    directory,
                    "--output",
                    str(output),
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Dry run only", result.stdout)
            self.assertFalse(output.exists())

    def test_only_state56_recipe_is_distributed(self):
        self.assertEqual(
            [path.name for path in (ROOT / "configs").glob("*.json")], ["state56.json"]
        )
        config = load_config(ROOT / "configs/state56.json")
        command = build_command(config, "/unused")
        self.assertEqual(config["crop"], 72)
        self.assertEqual(config["aug_shift"], 36)
        self.assertEqual(config["loader_workers"], 24)
        self.assertEqual(config["betas"], [0.0, 0.99])
        self.assertEqual(config["gp_interval"], 16)
        self.assertEqual(config["div_shuffle_dist"], 3)
        self.assertIn("--div-data", command)
        self.assertEqual(command[command.index("--tgt-pad") + 1], "0")
        self.assertEqual(command[command.index("--lr") + 1], "1e-05")
        self.assertEqual(command[command.index("--adv-lr") + 1], "2e-05")
        self.assertEqual(command[command.index("--adv-start") + 1], "1")

    def test_configuration_typos_and_invalid_types_are_rejected(self):
        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({**config, "mse_weight": 1.0}))
            with self.assertRaisesRegex(ValueError, "Configuration keys"):
                load_config(path)
            for key, value in (
                ("lr", True),
                ("in_pad", 3.0),
                ("div_data", "false"),
                ("batch_size", 2),
            ):
                path.write_text(json.dumps({**config, key: value}))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_headers_style_and_validation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_case(root)
            config = tiny_config()
            manifest = validate_data(config, root)
            self.assertEqual(manifest["cases"][0]["id"], "case_0")
            self.assertEqual(
                manifest["cases"][0]["shapes"]["HR/disp.npy"], [3, 16, 16, 16]
            )
            path = root / "config.json"
            path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(
                    [
                        "--config",
                        str(path),
                        "--data-root",
                        directory,
                        "--output",
                        str(root / "not_created"),
                        "--validate-only",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertFalse((root / "not_created").exists())
            np.save(root / "LR_selected_100_train/case_0/style.npy", np.ones(2))
            with self.assertRaisesRegex(ValueError, "six finite"):
                validate_data(config, root)

    def test_pairing_uses_ids_not_just_file_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_case(root)
            (root / "HR_selected_100_train/case_0").rename(
                root / "HR_selected_100_train/wrong_id"
            )
            with self.assertRaisesRegex(ValueError, "Unpaired"):
                validate_data(tiny_config(), root)

    def test_output_cannot_overwrite_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint.pt").write_bytes(b"do not replace")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                prepare_output(root, {}, {}, ["python", "train.py"], {})
            self.assertEqual((root / "checkpoint.pt").read_bytes(), b"do not replace")


if __name__ == "__main__":
    unittest.main()
