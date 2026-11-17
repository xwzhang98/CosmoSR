"""Validate paired field headers and styles without loading full simulation boxes."""

from pathlib import Path

import numpy as np


STYLE_ORDER = ["a", "Omega_m", "Omega_b", "h", "n_s", "sigma_8"]


def validate_data(config, data_root):
    root = Path(data_root).expanduser().resolve()
    lr_root = root / config["lr_directory"]
    hr_root = root / config["hr_directory"]
    if not lr_root.is_dir() or not hr_root.is_dir() or lr_root == hr_root:
        raise ValueError("LR and HR must be distinct, existing directories")
    lr_cases = {path.name: path for path in lr_root.iterdir() if path.is_dir()}
    hr_cases = {path.name: path for path in hr_root.iterdir() if path.is_dir()}
    if not lr_cases or lr_cases.keys() != hr_cases.keys():
        raise ValueError(f"Unpaired simulation IDs: LR-only={lr_cases.keys() - hr_cases.keys()}, HR-only={hr_cases.keys() - lr_cases.keys()}")
    manifest = {"style_order": STYLE_ORDER, "channel_order": ["sx", "sy", "sz", "vx", "vy", "vz"],
                "normalization": "cosmology.dis,cosmology.vel", "cases": [],
                "scope": "headers and styles only; full-array finiteness, shared phases and train/test disjointness are not certified"}
    for case in sorted(lr_cases):
        shapes = {}
        for kind, directory in (("LR", lr_cases[case]), ("HR", hr_cases[case])):
            for filename in ("disp.npy", "vel.npy"):
                field = np.load(directory / filename, mmap_mode="r", allow_pickle=False)
                if field.ndim != 4 or field.shape[0] != 3 or len(set(field.shape[1:])) != 1:
                    raise ValueError(f"{case}/{kind}/{filename}: expected (3,N,N,N), got {field.shape}")
                if field.dtype not in (np.dtype("float32"), np.dtype("float64")):
                    raise ValueError(f"{case}/{kind}/{filename}: expected float32 or float64")
                shapes[f"{kind}/{filename}"] = list(field.shape)
        n = shapes["LR/disp.npy"][1]
        if shapes["LR/vel.npy"] != shapes["LR/disp.npy"] or shapes["HR/disp.npy"] != [3, 2*n, 2*n, 2*n] or shapes["HR/vel.npy"] != shapes["HR/disp.npy"]:
            raise ValueError(f"{case}: displacement/velocity shapes or factor-two pairing disagree")
        if n < config["crop"]:
            raise ValueError(f"{case}: LR grid is smaller than crop={config['crop']}")
        style = np.load(lr_cases[case] / "style.npy", allow_pickle=False)
        if style.shape != (6,) or not np.isfinite(style).all():
            raise ValueError(f"{case}: style must be six finite values in {STYLE_ORDER} order")
        if not (0 < style[0] <= 1 and 0 < style[1] < 1 and 0 < style[2] < style[1] and np.all(style[3:] > 0)):
            raise ValueError(f"{case}: unphysical scale factor or cosmological parameters")
        hr_style = hr_cases[case] / "style.npy"
        if hr_style.exists() and not np.allclose(np.load(hr_style, allow_pickle=False), style, rtol=1e-6, atol=1e-8):
            raise ValueError(f"{case}: LR and HR styles disagree")
        manifest["cases"].append({"id": case, "lr_directory": str(lr_cases[case]),
                                  "hr_directory": str(hr_cases[case]), "shapes": shapes,
                                  "style": style.tolist()})
    return manifest
