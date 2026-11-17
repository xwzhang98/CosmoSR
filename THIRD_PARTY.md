# Source Attribution

The model and numerical training code derive from the project's modified [map2map](https://github.com/sagasv5-xw/map2map) implementation, originally authored by Yin Li et al. The upstream GNU GPL version 3 license is retained in `LICENSE`.

The 2026-11-17 source release renames the Python package to `cosmosr`, retains the state56 model and training path, and removes unused experimental branches. Model state-dict names remain compatible with `state_56.pt`. Portable launch, configuration validation, data preflight and regression tests were added for this repository.
