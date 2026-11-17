"""Load the single state56 training recipe."""

import json
from pathlib import Path
import sys


FIELDS = {
    "lr_directory",
    "hr_directory",
    "crop",
    "in_pad",
    "tgt_pad",
    "aug_shift",
    "batch_size",
    "loader_workers",
    "epochs",
    "lr",
    "adv_lr",
    "betas",
    "adv_start",
    "gp_interval",
    "div_data",
    "div_shuffle_dist",
    "save_interval",
    "log_interval",
    "seed",
}


def load_config(path):
    with Path(path).open() as stream:
        config = json.load(stream)
    supplied = set(config) if isinstance(config, dict) else set()
    if not isinstance(config, dict) or supplied != FIELDS:
        raise ValueError(
            f"Configuration keys: missing={FIELDS - supplied}, unknown={supplied - FIELDS}"
        )
    for key in ("crop", "batch_size", "epochs", "gp_interval", "log_interval"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["crop"] < 8:
        raise ValueError("crop must be at least 8 for the discriminator")
    if config["batch_size"] != 1:
        raise ValueError("The state56 model requires batch_size=1 per GPU")
    for key in (
        "loader_workers",
        "save_interval",
        "aug_shift",
        "adv_start",
        "seed",
        "in_pad",
        "tgt_pad",
    ):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["in_pad"] != 3 or config["tgt_pad"] != 0:
        raise ValueError("The state56 recipe uses in_pad=3 and tgt_pad=0")
    for key in ("lr", "adv_lr"):
        if type(config[key]) not in (int, float) or not 0 < config[key] < float("inf"):
            raise ValueError(f"{key} must be finite and positive")
    if type(config["div_data"]) is not bool:
        raise ValueError("div_data must be a JSON boolean")
    distance = config["div_shuffle_dist"]
    if type(distance) not in (int, float) or not 0 <= distance < float("inf"):
        raise ValueError("div_shuffle_dist must be finite and nonnegative")
    betas = config["betas"]
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or not all(type(b) in (int, float) and 0 <= b < 1 for b in betas)
    ):
        raise ValueError("betas must contain two numbers in [0, 1)")
    for key in ("lr_directory", "hr_directory"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty relative directory")
        path = Path(config[key])
        if (
            path.is_absolute()
            or ".." in path.parts
            or any(c in str(path) for c in ",*?[]")
        ):
            raise ValueError(f"{key} must be a literal relative directory")
    return config


def build_command(config, data_root, checkpoint=None):
    data_root = Path(data_root).expanduser().resolve()
    if any(c in str(data_root) for c in ",*?[]"):
        raise ValueError("data-root cannot contain commas or glob metacharacters")
    lr_root = data_root / config["lr_directory"]
    hr_root = data_root / config["hr_directory"]
    command = [sys.executable, "-m", "cosmosr.engine", "train"]

    def option(name, value):
        command.extend(["--" + name, str(value)])

    option(
        "train-in-patterns",
        ",".join(str(lr_root / "*" / name) for name in ("disp.npy", "vel.npy")),
    )
    option(
        "train-tgt-patterns",
        ",".join(str(hr_root / "*" / name) for name in ("disp.npy", "vel.npy")),
    )
    option("train-style-pattern", lr_root / "*" / "style.npy")
    option("in-norms", "cosmology.dis,cosmology.vel")
    option("tgt-norms", "cosmology.dis,cosmology.vel")
    option("model", "styled_srsgan.G")
    option("adv-model", "styled_srsgan.D")
    option("scale-factor", 2)
    option("crop-step", config["crop"])
    option("pad", config["in_pad"])
    option("optimizer-args", json.dumps({"betas": config["betas"]}))
    option(
        "load-state",
        str(Path(checkpoint).expanduser().resolve()) if checkpoint is not None else "",
    )
    for key, flag in (
        ("crop", "crop"),
        ("tgt_pad", "tgt-pad"),
        ("aug_shift", "aug-shift"),
        ("batch_size", "batch-size"),
        ("loader_workers", "loader-workers"),
        ("epochs", "epochs"),
        ("lr", "lr"),
        ("adv_lr", "adv-lr"),
        ("adv_start", "adv-start"),
        ("gp_interval", "adv-wgan-gp-interval"),
        ("save_interval", "save-interval"),
        ("log_interval", "log-interval"),
        ("seed", "seed"),
        ("div_shuffle_dist", "div-shuffle-dist"),
    ):
        option(flag, config[key])
    command.append("--augment")
    if config["div_data"]:
        command.append("--div-data")
    return command
