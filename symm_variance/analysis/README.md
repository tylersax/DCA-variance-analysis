# Analysis tier

Plain numpy + h5py + matplotlib. No DCA imports, so this runs anywhere the driver's HDF5 lands.

## Notebooks (open these to review)

| notebook | question it answers |
|---|---|
| `01_validation_ladder.ipynb` | Does the pipeline measure what it claims? Singleton null control, per-orbit `r` vs `m/[1+(m−1)ρ]`, ω-flatness, headline `R`, and mean preservation against production. |
| `02_noise_mechanism.ipynb` | *Why* is `ρ` what it is? Noise correlation matrices, real-space `σ²(r)` profiles, within- vs across-symmetry-shell correlation. |
| `03_m_scaling.ipynb` | Is `R` actually independent of run depth? The depth floor, its two mechanisms (autocorrelation vs sign problem), and the bootstrap CIs. Reads `../runs/m_scaling_summary.json`, not the raw ladder. |
| `04_beta_ladder.ipynb` | Is square intrinsically ρ-limited, or only at β=1? `ρ` and `R` across β ∈ {1,2,4,8} on a model that is sign-free at every β, so the correlation-length effect is isolated from the competing sign channel. Reads `../runs/beta_ladder_square.json`. |
| `05_beta_cross_model.ipynb` | Does the β trend reproduce on a second model? The FeAs ladder beside square's, with the sign channel live. Reads `../runs/beta_ladder_{square,fe_as}.json`. |
| `06_model_sweep.ipynb` | Does `R` grow with symmetry-**equivalent** orbitals? Four design points at fixed β=5, the within-model inequivalent-orbital control, and the cluster axis. Reads `../runs/model_sweep.json`. |

They are committed **with outputs and figures already executed**, so they can be read without
running anything. 01 and 02 read `../runs/*.hdf5`; 03 and 04 read the summary JSON beside them, since
their raw ladders live in scratch and are not committed.

## Modules

| file | role |
|---|---|
| `symm_variance_lib.py` | Core: loads a run, applies the serialized operator `P`, computes variances/ratios, extracts orbits and signs from `P`'s support, reproduces the full production symmetrization (spin + cluster + frequency). |
| `noise_diagnostics.py` | Mechanism: noise residuals, correlation matrices, real-space noise profile, D4 shell decomposition, within/across-shell correlation. |
| `reduction_map.py` | `r` resolved by entry class and orbit size, the exact harmonic decomposition `R = 1/Σ(w_C/R_C)`, plus `R_ideal` and `efficiency`. Run directly: `python reduction_map.py`. |
| `m_scaling.py` | Depth control: `R` vs measurements-per-rank, bootstrap-over-ranks CIs, a **paired** CI on the step-to-step `ΔR`, sign-health diagnostics, and the across-seed cross-check on those CIs. Run directly on a ladder directory: `python m_scaling.py <dir> --out summary.json`. |
| `seed_ensemble.py` | Precision: the headline `R` and its interval from many **independent base seeds** at one fixed depth, plus a paired depth check, an across-whole-runs oracle, and the contamination gate. Run directly: `python seed_ensemble.py <dir> --deep <deeper_dir> --out summary.json`. |
| `beta_ladder.py` | Temperature axis: `R`, `ρ`, per-orbit `r` and the sign diagnostics across a β ladder, one seed ensemble per rung. Carries the two controls the axis needs — a check that the sign channel stays off, and an `r`-vs-ω check, since a fixed frequency count means the absolute window shrinks as β grows. Run directly: `python beta_ladder.py <ladder_root> --out summary.json`. |
| `bath_drift.py` | Bath axis (ROADMAP 3e): `R` at every DCA iteration of a real `DcaLoop`, so the bare-bath convention can be checked against the bath a production run actually samples. Reports the **paired within-seed** drift (iterations are serially dependent; ranks are matched streams), the cost-weighted mean over iterations, the bath's own step size per iteration, and — with `--ref` — the coarse-graining gap between the loop's iteration 0 and the single-iteration driver. Run directly: `python bath_drift.py <drift_root> --ref <ref_dir> --out summary.json`. |
| `validate.py` | Command-line version of the ladder: `python validate.py ../runs/square_16rank.hdf5` |
| `build_notebooks.py` | Regenerates the notebooks from source. Edit **this**, not the `.ipynb`, then re-execute. |

### Which estimator to quote

`m_scaling.bootstrap_R` resamples ranks *within* one run; `seed_ensemble` uses whole independent runs
as the replicate unit. They answer the same question with different assumptions, and where a model
has a sign problem they do **not** agree — the rank bootstrap runs optimistic, because a resample of
64 ranks cannot see the heavy tail the sign denominator creates. **Quote the seed ensemble.** The rank
bootstrap stays useful for its paired form (`paired_bootstrap_dR`), where the shared randomness
cancels and the assumption does no work.

Two constraints that are easy to violate and invisible afterwards:

- **Base seeds must be spaced.** A walker's stream is `hash(global_id + base_seed)` with
  `global_id = local_id*n_ranks + proc_id`, so a run occupies the key range
  `[S, S + n_ranks*n_walkers)`. Base seeds closer than that share chains and the runs are not
  independent replicates. `run_seed_ensemble.sh` enforces it; `seed_ensemble.check_seed_spacing`
  re-checks it and voids the report if violated.
- **Depth must clear the model's floor** before seeds buy anything (see `03_m_scaling.ipynb`). More
  seeds at shallow depth is more chances at a near-zero sign denominator, not more precision.

The libraries stay as `.py` on purpose — they are imported by the notebooks, and logic that lives in
one place is logic that can't drift between copies. The notebooks exercise and explain them.

## Running

A Jupyter kernel named **`symm-variance (py)`** is registered and already selected in every notebook
`build_notebooks.py` emits. Its spec lives at `~/.local/share/jupyter/kernels/symm-variance/kernel.json` and holds an
**absolute** interpreter path, so it breaks whenever this repo moves (it pointed at
`dca/analysis_deprecated/.venv` until 2026-07-27). If nbconvert dies with `FileNotFoundError` on a
`.venv/bin/python`, that file is what to fix.

Regenerate and re-execute after changing `build_notebooks.py`:

```bash
cd /home/tsax10/dca/analysis/symm_variance/analysis
V=/home/tsax10/dca/analysis/.venv/bin/python
$V build_notebooks.py
for nb in *.ipynb; do
  $V -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$nb"
done
```

Note `build_notebooks.py` rewrites **every** notebook it emits, clearing outputs — so re-execute all
of them after any edit, not just the one you changed. The loop above globs rather than naming them,
which is why it no longer goes stale. (This text said "all four" while five existed; `ls *.ipynb` is
the authority. 07 is planned for the migration scatter.) Verify by counting `execution_count` per cell, not by trusting the exit code —
nbconvert can exit 0 having executed nothing (Gotcha 12).

## Data contract

Written by `symm_variance_main.inc`; see `orbit_table.hpp` for the operator. `bath_drift_main.inc`
writes the **same schema, one file per DCA iteration** (`<stem>_iter<k>.hdf5`), which is why every
module above reads a drift iteration with no changes — it adds keys, it does not change any.

```
metadata/{model,seed,beta,sweeps_per_measurement,warm_up_sweeps,n_ops,n_ranks,
          measurements_total,measurements_per_rank,local_size,k_elements}
raw/{G_raw_re,G_raw_im}   [n_rank][local_size]  raw per-rank G, C++ leaf order
raw/{sign_re,sign_im}     [n_rank]              per-rank accumulated phase
functions/cluster_greens_function_G_k_w         production symmetrized mean, (w,k,s1,b1,s0,b0)
symmetrization/{P,flat_labels,nb,nk,n_ops,flat_index_order}
    P            [E][E] real, P[out][in] — the point-group orbit average on (b0,b1,k)
    flat_labels  [E][3] = (b0,b1,k),  flat index = (b0*nb + b1)*nk + k
```

⚠️ **Depth is per rank.** DCA's `measurements` input key is the TOTAL over ranks (it goes through
`parallel::util::getWorkload`), so the driver records both figures and `run_symm_variance.sh` takes
the per-rank one. Files written before 2026-07-27 have only `measurements_per_rank` and it holds the
**total** — the committed 16-rank "4000" runs are 250 per rank. `m_scaling.read_depth` handles both.

⚠️ `beta`, `sweeps_per_measurement` and `warm_up_sweeps` are recorded from 2026-07-28 on. Earlier
files lack them; `m_scaling.read_beta` returns `None` there rather than guessing, and
`beta_ladder.build` refuses such a run instead of grouping it under a β it cannot verify.

⚠️ **`bath_drift_main.inc` adds** `metadata/{dca_iteration,dca_iterations_total,
self_energy_mixing_factor,chemical_potential}` and three more `functions/`: the bath the walker
actually sampled (`cluster_excluded_greens_function_G0_k_w`), the bare cluster `G0`
(`free_cluster_greens_function_G0_k_w`), and `Self_Energy`. The bath is written so drift is measured
rather than inferred from `R` — and because at `Σ=0` the loop's bath is the **coarse-grained** `G`,
not the bare `G0` the single-iteration driver uses, a distinction nothing else in the tree exposes.

⚠️ Rung 2 (mean preservation) is only meaningful for runs made with
`error-computation-type = NONE`. Under `JACK_KNIFE`, `finalize` writes a leave-one-out replicate into
`G_k_w` and the comparison shows a spurious ~3% low-frequency gap.
