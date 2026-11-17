# CosmoSR

Cosmology-conditioned super-resolution of cosmological N-body simulations.

Companion code for [AI-assisted super-resolution cosmological simulations V: Cosmology-aware super-resolution](https://arxiv.org/abs/2610.06710).

## Model

The generator maps six-channel Lagrangian displacement and velocity fields to twice the resolution along each spatial axis. Styled convolutions condition on `(a, Omega_m, Omega_b, h, n_s, sigma_8)`; learned noise amplitudes provide stochasticity.

This repository contains the model and training code for **`state_56.pt`**, with one configuration: [configs/state56.json](configs/state56.json). The discriminator receives upsampled LR fields (6 channels), CIC Eulerian density (8), and candidate SR/HR fields (6).

| Setting | Value |
| --- | --- |
| LR crop / input padding / target padding | 72 / 3 / 0 |
| Resolution factor | 2 per axis |
| Batch size | 1 per GPU |
| Generator / discriminator learning rate | `1e-5` / `2e-5` |
| Optimizer | Adam, betas `(0, 0.99)` |
| Gradient penalty | WGAN-GP, coefficient 10, every 16 batches |
| Precision | float32 |

## Installation

Use Python 3.11+ and a CUDA-compatible PyTorch build:

```bash
python -m pip install -e .
```

## Data

Prepare matching simulation IDs:

```text
quijote/
  LR_selected_100_train/<id>/{disp.npy,vel.npy,style.npy}
  HR_selected_100_train/<id>/{disp.npy,vel.npy}
```

Fields have shape `(3,N,N,N)`, with HR side length `2*N`. Style has shape `(6,)`. See [Data Format](docs/DATA_FORMAT.md) for units, normalization and pairing requirements.

## Training

Check inputs without training:

```bash
python train.py --data-root /path/to/quijote \
  --output runs/state56 --validate-only
```

Start training, or resume a trusted full checkpoint:

```bash
python train.py --data-root /path/to/quijote --output runs/state56

python train.py --data-root /path/to/quijote --output runs/resumed \
  --resume /path/to/state_56.pt
```

The launcher defaults to `configs/state56.json`. Add `--dry-run` to inspect the command without importing PyTorch or reading data. Each invocation requires a new or empty output directory.

Training spawns one worker per visible GPU on one node; do not wrap it in `torchrun`. The reference run used two GPUs (global batch size 2). The original setting `adv_start=1` means epoch zero uses paired MSE when starting from scratch; later epochs use adversarial feedback and log MSE only. The data sampler partitions fields across GPUs with shuffle distance 3. Full-resolution training needs substantial GPU memory.

Adapt [scripts/train.slurm](scripts/train.slurm) to your cluster. [Checkpoint Notes](docs/CHECKPOINTS.md) describe restart semantics and saved outputs.

## Layout and Tests

- `cosmosr/models/`: styled generator, discriminator and CIC conversion.
- `cosmosr/data/`: periodic crops, augmentation, normalization and sampling.
- `cosmosr/trainer.py`: distributed training and checkpoint saving.
- `tests/`: configuration, data and reduced-width CPU training checks.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

CPU regressions are not full-resolution CUDA or scientific-accuracy validation. Simulation data, checkpoints, preprocessing, full-box inference and halo finding are not distributed.

## Citation

```bibtex
@misc{zhang2026cosmosr,
  title = {AI-assisted super-resolution cosmological simulations V: Cosmology-aware super-resolution},
  author = {Zhang, Xiaowen and Lachance, Patrick and Croft, Rupert A. C. and Di Matteo, Tiziana and Villaescusa-Navarro, Francisco},
  year = {2026},
  eprint = {2610.06710},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.CO}
}
```

## Attribution

Based on the project's modified map2map implementation and the Quijote simulations. The GPLv3 [license](LICENSE) and [source attribution](THIRD_PARTY.md) are retained.
