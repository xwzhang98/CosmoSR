# Checkpoints

Only load trusted checkpoint files: full training states require `torch.load(..., weights_only=False)`.

## Initialization

Without `--resume`, training starts from scratch using `configs/state56.json`; an unrelated local `checkpoint.pt` is never loaded implicitly. The original `adv_start=1` retains one initial MSE epoch before adversarial training.

With `--resume /path/to/state_56.pt`, the launcher checks the six-channel generator, 20-channel discriminator and required training-state keys. The loop restores G/D weights, optimizers, schedulers, CPU RNG and stored epoch. Restored optimizer learning rates supersede values in the JSON configuration. The total stopping epoch must exceed the stored epoch.

The configuration matches the settings saved with `state_56.pt` and its corresponding training log. It does not guarantee bitwise reproduction of a historical run: GPU count, CUDA kernels and data-loader state also affect training.

## Outputs

Each run records `run_config.json`, `data_manifest.json`, `initialization.json` and `command.txt`. The trainer saves `state_<epoch>.pt`, a `checkpoint.pt` symlink, and TensorBoard events. Optional `save_interval` exports intermediate `state_<epoch>_b<batch>.pt` files; the state56 default is zero.

Use a new or empty output directory for every invocation, including resume. The checkpoint format does not preserve data-loader cursors or every worker's CUDA RNG, so mid-epoch restarts may replay data and are not exact continuations. An inference-only weight export cannot resume training.
