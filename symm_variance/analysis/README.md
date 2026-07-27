# Analysis tier

Plain numpy + h5py + matplotlib. No DCA imports, so this runs anywhere the driver's HDF5 lands.

## Notebooks (open these to review)

| notebook | question it answers |
|---|---|
| `01_validation_ladder.ipynb` | Does the pipeline measure what it claims? Singleton null control, per-orbit `r` vs `m/[1+(m−1)ρ]`, ω-flatness, headline `R`, and mean preservation against production. |
| `02_noise_mechanism.ipynb` | *Why* is `ρ` what it is? Noise correlation matrices, real-space `σ²(r)` profiles, within- vs across-symmetry-shell correlation. |

Both are committed **with outputs and figures already executed**, so they can be read without running
anything. They read from `../runs/*.hdf5`.

## Modules

| file | role |
|---|---|
| `symm_variance_lib.py` | Core: loads a run, applies the serialized operator `P`, computes variances/ratios, extracts orbits and signs from `P`'s support, reproduces the full production symmetrization (spin + cluster + frequency). |
| `noise_diagnostics.py` | Mechanism: noise residuals, correlation matrices, real-space noise profile, D4 shell decomposition, within/across-shell correlation. |
| `reduction_map.py` | `r` resolved by entry class and orbit size, the exact harmonic decomposition `R = 1/Σ(w_C/R_C)`, plus `R_ideal` and `efficiency`. Run directly: `python reduction_map.py`. |
| `validate.py` | Command-line version of the ladder: `python validate.py ../runs/square_16rank.hdf5` |
| `build_notebooks.py` | Regenerates the notebooks from source. Edit **this**, not the `.ipynb`, then re-execute. |

The libraries stay as `.py` on purpose — they are imported by the notebooks, and logic that lives in
one place is logic that can't drift between copies. The notebooks exercise and explain them.

## Running

A Jupyter kernel named **`symm-variance (py)`** is registered and already selected in both notebooks.

Regenerate and re-execute after changing `build_notebooks.py`:

```bash
cd /home/tsax10/dca/analysis/symm_variance/analysis
V=/home/tsax10/dca/analysis/.venv/bin/python
$V build_notebooks.py
$V -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 01_validation_ladder.ipynb
$V -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 02_noise_mechanism.ipynb
```

Note `build_notebooks.py` rewrites **both** notebooks, clearing outputs — re-execute both after any
edit.

## Data contract

Written by `symm_variance_main.inc`; see `orbit_table.hpp` for the operator.

```
metadata/{model,seed,n_ops,n_ranks,measurements_per_rank,local_size,k_elements}
raw/{G_raw_re,G_raw_im}   [n_rank][local_size]  raw per-rank G, C++ leaf order
raw/{sign_re,sign_im}     [n_rank]              per-rank accumulated phase
functions/cluster_greens_function_G_k_w         production symmetrized mean, (w,k,s1,b1,s0,b0)
symmetrization/{P,flat_labels,nb,nk,n_ops,flat_index_order}
    P            [E][E] real, P[out][in] — the point-group orbit average on (b0,b1,k)
    flat_labels  [E][3] = (b0,b1,k),  flat index = (b0*nb + b1)*nk + k
```

⚠️ Rung 2 (mean preservation) is only meaningful for runs made with
`error-computation-type = NONE`. Under `JACK_KNIFE`, `finalize` writes a leave-one-out replicate into
`G_k_w` and the comparison shows a spurious ~3% low-frequency gap.
