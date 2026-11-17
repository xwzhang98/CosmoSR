# Prepared Training Fields

The loader consumes prepared NumPy arrays, not raw Gadget snapshots. No data conversion is performed implicitly.

| File | Shape | Values |
| --- | --- | --- |
| `disp.npy` | `(3,N,N,N)` | Lagrangian displacement components in comoving kpc/h |
| `vel.npy` | `(3,N,N,N)` | Peculiar velocity components in km/s |
| `style.npy` | `(6,)` | `(a, Omega_m, Omega_b, h, n_s, sigma_8)` |

Use float32 or float64 fields; the crop loader converts them to float32. The HR side length must be twice LR's. Style is passed to G/D without an additional standardization; the paper's training snapshots have `a=1`. If HR also contains a style file, the preflight checks that it agrees with LR.

LR and HR subdirectories must have identical simulation IDs. Before preparing these arrays, align each field to the same Lagrangian lattice convention and ensure the paired simulations share the intended large-scale initial modes. Matching filenames and array shapes do not certify shared phases or particle identities.

The channel order after concatenation is `(sx,sy,sz,vx,vy,vz)`. Each crop is read with periodic wrapping. The state56 recipe uses a 72-cell LR crop, three LR padding cells per face, and a 144-cell HR target without target padding. The generator's extra output cells are center-cropped to the target during training.

`cosmosr/data/norms/cosmology.py` applies the original normalization in place:

- Displacement denominator: `60000 * D(z)` kpc/h.
- Velocity denominator: `60 * D(z) * H(z) * f(z) / (1+z)` km/s.
- `D(0)=1`, and `H(z)` uses h km/s/Mpc units. Do not add another numerical factor of h.

Do not pre-normalize the fields and then apply these normalizers again. Do not substitute a density grid for the six-channel displacement/velocity representation.

The preflight reads array headers and small style files through NumPy's structured APIs; it does not scan full multi-gigabyte fields for NaN/Inf. It records a manifest of the selected training cases but cannot infer which cases were used in earlier work. Keep evaluation/confirmation cases outside the training roots and audit their simulation IDs explicitly.

Changing crop size, dataset membership, normalization or augmentation is a new experiment, not an incidental way to make an existing run fit in memory.
