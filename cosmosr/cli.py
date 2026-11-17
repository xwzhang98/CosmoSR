"""Validate a run, then launch state56 training in a fresh directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .config import build_command, load_config


RELEASE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = RELEASE_ROOT / "configs" / "state56.json"
if not DEFAULT_CONFIG.is_file():
    DEFAULT_CONFIG = Path(sys.prefix) / "share" / "cosmosr" / "configs" / "state56.json"


def validate_checkpoint(path, config):
    import torch
    from .models.styled_srsgan import G, D

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Checkpoint not found: {path}")
    # Full training states contain Python objects; only accept trusted files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "epoch",
        "model",
        "adv_model",
        "min_loss",
        "rng",
        "optimizer",
        "scheduler",
        "adv_optimizer",
        "adv_scheduler",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError(
            f"Expected a full training checkpoint with keys {sorted(required)}; an inference-only export cannot be resumed"
        )
    if type(state["epoch"]) is not int or not 0 <= state["epoch"] < config["epochs"]:
        raise ValueError(
            "epochs is a total endpoint and must exceed the checkpoint's stored epoch"
        )
    generator = G(6, 6, 6, scale_factor=2)
    generator.load_state_dict(state["model"], strict=True)
    critic = D(12, 1, 6, scale_factor=2)
    critic.load_state_dict(state["adv_model"], strict=True)
    for name in ("model", "adv_model"):
        if name in state and any(
            not torch.isfinite(value).all() for value in state[name].values()
        ):
            raise ValueError(f"Non-finite {name} parameters")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "stored_epoch": state["epoch"],
        "mode": "resume",
        "discriminator_input_channels": 20,
    }


def prepare_output(path, config, manifest, command, checkpoint):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be new or empty, including for resume: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("run_config.json", config),
        ("data_manifest.json", manifest),
        ("initialization.json", checkpoint),
    ):
        with (path / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    with (path / "command.txt").open("x") as stream:
        stream.write(shlex.join(command) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="CosmoSR single-node CUDA training")
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="trusted full checkpoint; restore optimizer/scheduler state",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command without importing torch, validating data or training",
    )
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="validate headers/styles and any supplied checkpoint; do not train",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        data_root = args.data_root.expanduser().resolve()
        if any(char in str(data_root) for char in ",*?[]"):
            raise ValueError("Data root cannot contain commas or glob metacharacters")
        checkpoint_path = args.resume
        command = build_command(config, data_root, checkpoint_path)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            print(
                "Dry run only: data, checkpoint and CUDA availability were not validated."
            )
            return 0
        from .data_check import validate_data

        manifest = validate_data(config, data_root)
        checkpoint = (
            validate_checkpoint(checkpoint_path, config)
            if checkpoint_path is not None
            else {"mode": "scratch"}
        )
        print(
            f"Validated {len(manifest['cases'])} paired simulation headers and styles.",
            flush=True,
        )
        if args.validate_only:
            print("Validation only: no training or output directory was created.")
            return 0
        if (
            int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) != 1
            or "LOCAL_RANK" in os.environ
            or int(os.environ.get("WORLD_SIZE", "1")) != 1
        ):
            raise ValueError(
                "This launcher supports one node and spawns CUDA workers itself; do not launch it with torchrun"
            )
        import torch

        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is required for training; use --validate-only for CPU preflight"
            )
        output = args.output.expanduser().resolve()
        prepare_output(output, config, manifest, command, checkpoint)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(RELEASE_ROOT) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(command, cwd=output, env=env, check=False)
        return result.returncode
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"CosmoSR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
