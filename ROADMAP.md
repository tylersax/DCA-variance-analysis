# Roadmap — symmetrization variance reduction in DCA++

**This file is the task list.** It carries what to do next, the current state, and the conventions
that govern future work — nothing else. **Findings and the evidence for them live in
[`TAKEAWAYS.md`](TAKEAWAYS.md)**; design rationale in [`symm-variance-plan.md`](symm-variance-plan.md).
If you want to know *why* a number is what it is, go there, not here.

**Goal.** Measure how much Monte-Carlo variance the point-group symmetrization removes from the
single-particle Green's function, report it as `R = Var(G)/Var(Sym G)`, and eventually make it a
standard solver output.

**Current priority: broaden model coverage (task 2, then 3). The estimator itself is now exercised.**

---

## State

| # | Milestone | Result |
|---|---|---|
| 1 | `local_G_k_w(symmetrize=false)` + orbit-table serializer | shipped (plus a latent-bug fix) |
| 2 | `symm_variance/` driver, multi-rank raw per-rank dump | shipped |
| 3 | numpy validator + figures; validation rungs 1 & 2 | all rungs pass |
| 4 | FeAs through the same pipeline | all rungs pass |
| 5 | M-scaling control, bootstrap CIs, depth floors | `R` scale-free above a per-model depth floor |

| model | **`R`** | orbits | ρ (mates) | depth floor, per rank |
|---|---|---|---|---|
| square / D4, β=1 | **1.041 ± 0.003** | m = 1, 2, 4 | ~0.94 | ≈64 — autocorrelation |
| FeAs 2-band, β=5 | **3.17 ± 0.14** | m = 2, 4, 8 (+24 forced nulls) | ~0.06 | ≳1000 — **sign problem** |

64 ranks × 3 base seeds, 4×4 clusters, single DCA iteration, `±` = standard error over seeds. FeAs
declares 2 ops → derives 8. Validation rungs pass on both models: singletons pin at exactly `r=1`,
per-orbit `r` matches `m/[1+(m−1)ρ]`, `r` flat in ω, mean preservation vs production to 2e-16.
Evidence and mechanism: **TAKEAWAYS §1a** and `03_m_scaling.ipynb`.

> ⚠️ Earlier drafts quoted **square 1.034 / FeAs 2.482** from single 16-rank runs. Both are
> superseded. The FeAs one especially — its 95% CI was `[2.13, 3.51]`, and 2.482 was a low draw from
> it. Do not reuse either number; anything still citing them is stale.

---

## Conventions — settled, do not re-litigate

**`R` means the FULL support, and it is the only headline number** (decided 2026-07-27). It tracks
what a user actually experiences: run unsymmetrized and you carry the symmetry-forbidden entries too,
and their noise propagates downstream. Non-null `R` is a secondary structural note — related exactly
by `R_full = R_non-null/(1 − w_null)`, so it adds only `w_null`, the forced-null share of raw
variance. Report `w_null` where noise *structure* is the point, above all in the cross-model sweep
(task 3), since it varies by model and would otherwise absorb part of any trend. **Efficiency
`R/R_ideal` is a dev diagnostic and does not go in a writeup.** Standing principle behind all of
this: *only take on complexity that buys something worth its cost.*

**Depth is measurements PER RANK, and every model has a floor** that must be re-established whenever
β or the model changes — the sign floor rises with β independently of the autocorrelation time.
Rules that follow:

- **More ranks do not substitute for depth.** Precision on `R` comes from ranks; the sign floor does
  not care — more ranks at shallow depth is just more chances to draw a near-zero denominator.
- **Where a sign problem exists, quote `R` from across-seed scatter, not the rank bootstrap** — the
  bootstrap ran 1.4–1.9× optimistic for FeAs (well calibrated for square). Tighten a number with
  **more base seeds**, not more ranks inside one seed.
- **Check measured vs predicted per-orbit `r` on every new run.** They agree to 3 decimals at
  adequate depth; divergence flags a run contaminated by sign outliers. Free contamination detector.

---

## Task list — in priority order

### 1. Finish statistical hardening

Mostly done — `R_ideal`/efficiency and the by-class reduction map in `reduction_map.py`, bootstrap
CIs and the across-seed check in `m_scaling.py`. What is left:

- **A tight FeAs `R`.** Current best `3.17 ± 0.14` from 3 seeds — a 4.4% standard error, dominated by
  between-seed scatter. Tightening it means **more base seeds**. Depth must stay above the sign floor
  (`m ≳ 1000`/rank) regardless.
- Optional independent oracle: many independent whole runs, sample-variance across runs, sharing no
  code with the per-rank path.

### 2. β ladder — test the central hypothesis  ▸ needs GPU

Square's `R = 1.04` was measured at **β = 1**, exactly the regime where noise is dominated by
symmetric scalar channels. Prediction: as β rises, correlation length grows, noise acquires
symmetry-breaking structure, `ρ` falls and `R` rises. If so, square is not intrinsically ρ-limited.

- Ladder β = 1, 2, 4, 8 on the small 4×4 square cluster first — isolates the β axis cheaply.
- Track `ρ`, per-orbit `r`, `R`, **and** the mechanism diagnostics at each β.
- **Two competing effects — this is the interesting part, and the main result on offer.** Growing
  correlation length should push `ρ` *down* (`R` up). But `G = ⟨sign·M⟩/⟨sign⟩`, and a denominator
  fluctuation is a **global scalar** giving `δG(k) = −G(k)·δs/s` with `w(k) = −G(k)` symmetric — so
  **sign-problem noise is a fully symmetric channel with mate-correlation exactly 1**, pushing `ρ`
  *up* (`R` → 1). Which dominates, and where the crossover sits, is unknown. Track the sign
  distribution alongside `ρ` at every β so the two can be separated. Mapping where symmetrization
  pays off — **including the regimes where it collapses** — is the research output here.
- **Sign data already exists and is instrumented.** Square at β=1 is sign-free (⟨sign⟩ = 1 exactly);
  FeAs at β=5 has ⟨sign⟩ ≈ 0.25, verified physical (smooth β-decay to 1 at β→0, square exactly
  sign-free as control — see TAKEAWAYS §1a). `m_scaling.sign_health` reports the per-rank sign
  distribution on every run; the β ladder inherits it. `symm_variance/run_sign_sweep.sh <outdir>
  <n_ranks> <meas_per_rank> <beta...>` runs the β-sweep-of-⟨sign⟩ check on both models cheaply — use
  it to scout the β range before committing to full ladder runs.
- Cost climbs steeply: expansion order ~ β·U·Nc, walker linear algebra ~ cubic in expansion order.

**GPU build.** 4× RTX A5000 (24 GB, idle), CUDA 12.2. `-DDCA_WITH_CUDA=ON` +
`CUDA_GPU_ARCH=sm_86`. `symm_variance_setup.hpp` already selects `linalg::GPU` under `DCA_HAVE_GPU`,
so no driver change is needed. **Oversubscribe — several ranks per GPU**: the statistics want many
ranks at adequate depth, not 4 heavy ranks. Re-validate rungs 1 & 2 on a GPU build before trusting
new physics from it.

### 3. Broaden model coverage — vary ONE axis at a time  ▸ carries the headline claim

**The claim this task establishes:** symmetrization pays off most where the physics is most
interesting — multi-orbital models, and *within* them the interband components that carry the
distinctive physics. See TAKEAWAYS §4c.

**Four runs turn that framing into a defensible claim:**

| run | establishes |
|---|---|
| single-band square at FeAs's β | **the keystone** — de-confounds temperature from band count |
| a 3-orbital model | the trend in `nb`; exercises `U_S` as a genuine permutation, not a swap |
| an inequivalent-orbital model | benefit should **vanish** — confirms the gain comes from symmetry-*equivalent* orbitals, not orbital count |
| `R` split by entry class, on all of the above | surfaces the interband concentration that aggregate `R` buries |

Prediction to test: `R` grows with the number of symmetry-equivalent orbitals
(nb=1 → 1.04, nb=2 → 3.2, nb=3 → ?).

**Design constraint (important):** square and FeAs differ in β, band count, interaction and filling
*simultaneously*, so the ρ difference between them **cannot be attributed to any single cause**. The
mechanism is measured; the causation is not. The sweep must isolate axes.

Axes, and which knob in `r = m/[1+(m−1)ρ]` each moves:

| axis | moves | note |
|---|---|---|
| cluster size 4×4 → 8×8 | `m` | first *free* orbit-8 k-point only exists at 8×8 |
| point group `D4`=8 → `D6`=12 → 3D `O_h`=48 | `m` | needs non-square lattices |
| band count / band-permuting ops | `m` | enlarges the `(b0,b1,k)` index space |
| β | `ρ` | task 2 |
| U, filling | `ρ` | |
| band-diagonal vs interband entries | `ρ` | interband has no local scalar channel |

- Open empirical question: **does `ρ` fall with cluster size?** That decides whether the `m`-ceiling
  is reachable in practice.
- Needs a sweep driver + results aggregator; record per-orbit `m`, `ρ`, `r`, the headline `R`, and
  the mechanism diagnostics for every run. **Record `w_null` and non-null `R` here specifically** —
  this is the task where they earn their place (see Conventions). Efficiency stays in the aggregator
  as a dev diagnostic only.
- ⚠️ **Check what CT-AUX actually simulates before adding a multi-orbital model.** It silently drops
  spin-flip (`J`) and pair-hopping (`Jp`) terms — see Gotchas — so `fe_as`, `hund_lattice`,
  `La3Ni2O7_bilayer` and `twoband_Cu` all run as density-density models regardless of input.

### 4. Migration scatter figure

The plan's signature visualization (§4.3), still to build. For a chosen orbit at fixed ω, plot mates
in the complex plane: raw cloud of `m×P` points (signs applied first so odd mates align), then arrows
to their sign-aware orbit average. The cloud contracts toward its own centroid by `1/√r` while the
**centroid does not move** — variance reduction, mean preservation, and the ρ mechanism all legible
at once. Singletons show no migration: the visual null control.

Small-multiples grid, one panel per orbit. Ideal contrast: square (ρ≈0.94, barely moves) beside FeAs
`m=8` (ρ≈0.08, collapses) — makes ρ-limited vs m-limited visual. Add as **`04_migration_scatter.ipynb`**
(03 is taken by the M-scaling notebook) via `analysis/build_notebooks.py`.

### 5. Production reporting  ▸ deferred until the above is settled

Fork the raw replicate **before** `symmetrize_measurements()` mutates `M`, form the unsymmetrized
`G`, and write the on/off error pair plus `R` as a standard solver output. Requires `JACK_KNIFE` and
`n_rank > 1`; off by default; cost is one extra G-formation plus one reduction.

This is the **only correct route for Σ and G4**: symmetrization of `G` is linear, but `Σ = G0⁻¹ − G⁻¹`
is not, so a post-hoc linear operator cannot recover the gain — it must be re-symmetrized inside each
jackknife replicate. numpy cannot do this.

⚠️ Do **not** read the `STANDARD_DEVIATION` accumulator for this: `symmetrize_measurements()`
symmetrizes `M_r_w_squared_`, so that accumulator is already arm-dependent.

### 6. Σ / G4 reporting  ▸ OUT OF SCOPE for this analysis

Follows from task 5 by construction, but observable-level variance is outside the boundary set with
the advisor (see Scope boundaries). Listed only so the path is known, not as planned work.

---

## Design principle worth not re-deriving

`R` is **scale-free**: `Var(Ĝ_N) = Σ/N` and `Var(P·Ĝ_N) = PΣPᵀ/N`, so `1/N` cancels and `R` depends
only on the noise covariance *structure*, not on compute spent. Therefore precision on `R` comes from
the **number of independent samples**, not run depth, and the measured `R` transfers to any
production scale.

**With one correction the M-scaling control forced:** the slogan used to be "many ranks, modest
depth". *Modest has a floor* — below it the per-rank sample is not in the CLT regime and `R` is
biased, and for a model with a sign problem that floor is set by the sign, not the autocorrelation
time. So: **many ranks, at or above the model's depth floor.** See Conventions.

The estimator is also **paired** (`P` is deterministic and linear, so one sample set gives both
numerator and denominator), which is why ~16 samples reach precision comparable to a ~128-per-arm
ON/OFF ensemble. Note this buys accuracy *per sample*, not precision from nowhere — 16 ranks was
still far too few for a quotable FeAs `R`.

---

## Where things live

This repo (`DCA-variance-analysis`) is checked out at `/home/tsax10/dca/analysis`. It holds **both**
the current project and the superseded prototype:

| path | what |
|---|---|
| `symm_variance/` | **current work** — driver, orbit-table serializer, inputs, `run_symm_variance.sh`, `run_m_ladder.sh`, `run_sign_sweep.sh` |
| `symm_variance/analysis/` | numpy libs + notebooks (`01_validation_ladder`, `02_noise_mechanism`, `03_m_scaling`) |
| `symm_variance/runs/` | run data the notebooks read, plus `m_scaling_summary.json` |
| `patches/symm-variance-dca.patch` | **our DCA source edits** — see warning below |
| `ROADMAP.md`, `TAKEAWAYS.md`, `symm-variance-plan.md` | this file, headline claims, design doc |
| `variance_demo/`, `notebooks/`, `SETUP.md`, `variance-demo-plan.md` | **superseded prototype** — reference only, be skeptical |
| `.venv/` | python env (gitignored). Its `jupyter` launcher has a stale shebang — use `python -m nbconvert` |

Outside the repo: `../source/DCA` (the DCA checkout, a separate git repo), `../build_symm` (build
tree; `../build` is the stale prototype build).

> ⚠️ **The DCA source edits live in a different repo and are uncommitted there.**
> `source/DCA` is its own git checkout; our changes to `ctaux_cluster_solver.hpp`
> (`local_G_k_w(symmetrize)`, a latent-bug fix, `local_accumulated_sign()`) sit in its working tree.
> `patches/symm-variance-dca.patch` is the authoritative copy — reapply with
> `git -C ../source/DCA apply patches/symm-variance-dca.patch` if that tree is ever reset.

**Build:** standalone CMake pulling DCA as a subdirectory; deps from `/home/tsax10/conda/envs/qe`
(OpenMPI 4.1.6, HDF5 1.14.3, FFTW, LAPACK), system g++ 13.3. Only the two driver objects are ours —
everything under `build_symm/dca/` is stock DCA.
**Run:** `symm_variance/run_symm_variance.sh <square|fe_as> <n_ranks> <measurements_per_rank> [seed] [outdir]`
(caps BLAS threads; shared 128-core box — check load first). Depth sweeps:
`symm_variance/run_m_ladder.sh <model> <n_ranks> <outroot> <m,...> <seed,...>` — one HDF5 per design
point, ~30 MB each at 64 ranks, so point it at scratch and commit only the summary JSON.
**Notebooks:** Jupyter kernel `symm-variance (py)`; regenerate via
`symm_variance/analysis/build_notebooks.py`, which rewrites **all three** and clears outputs, so
re-execute all three after any edit.

---

## Scope boundaries (state these in any writeup)

- **We measure `Var(G)` and stop there.** ← decided with advisor, 2026-07-27. Observable-level
  variance (Σ, susceptibilities/`Tc`, spectra) is **out of scope**. State the caveat — the reduction a
  user experiences can be higher or lower than the `R` we quote, since each downstream transform
  reweights the noise — but do not chase it. This boundary exists to keep the analysis finite.
- Variance of a **single-iteration** estimator, not a converged DCA solution.
- **Plain DCA, not DCA+** (symmetrization imposed on the CLUSTER family).
- **CT-AUX only** — CT-INT does not currently compile in this tree. Note CT-AUX also drops
  non-density interaction terms (see Gotchas), so multi-orbital models run as density-density.

---

## Gotchas that cost time once already

1. **`measurements` in the input JSON is the TOTAL over ranks, not per rank.** Every solver routes it
   through `parallel::util::getWorkload`; `mci_parameters.hpp` computes `local_meas = measurements /
   mpi_size`. Depth per rank governs whether a sample is asymptotic, so the axis was off by a factor
   of `n_ranks` until 2026-07-27. Fixed at the source: `run_symm_variance.sh` takes **measurements
   per rank** and multiplies up, the driver records `measurements_total` *and* `measurements_per_rank`
   and refuses a total not divisible by `n_ranks`. **The committed 16-rank runs named "4000" are 250
   per rank**; `m_scaling.read_depth` divides legacy files down automatically.
2. **CT-AUX silently drops spin-flip (`J`) and pair-hopping (`Jp`) terms.** Its vertices come from
   `CV::get_H_interaction()` (density-density only); nothing in `cluster_solver/ctaux/` references
   `non_density_interaction`, which only CT-INT and the two-particle accumulators read. **No warning
   is emitted.** So FeAs with `U=4, V=1, J=1, Jp=1` actually runs `U=4`, inter-band same-spin `V−J=0`,
   inter-band opposite-spin `V+J=2` — an Ising-Hund model. Also affects `hund_lattice`,
   `La3Ni2O7_bilayer`, `twoband_Cu`. Describe the model accordingly in any writeup.
3. **Changing the rank count re-seeds the walkers.** `getGlobalId = local_id*num_procs + proc_id`
   (`src/math/random/random_utils.cpp`), so runs at different `n_ranks` are near-independent, not
   nested — measured stream overlap between the 16- and 64-rank FeAs runs is −0.065. Nesting across
   *depths* at fixed rank count does hold, which is what the paired bootstrap relies on.
4. **Needs `n_rank > 1`.** DCA has *no* single-rank error bars: `JACK_KNIFE` returns empty at `n==1`,
   and `STANDARD_DEVIATION` (`average_and_compute_stddev`) yields identically zero.
5. **Call `local_G_k_w()` before `finalize()`** — it throws once `averaged_` is set.
6. **Rung 2 requires `error-computation-type = NONE`.** Under `JACK_KNIFE`, `finalize` writes a
   leave-one-out replicate into `G_k_w`, producing a ~3% low-frequency gap that is a jackknife
   artifact, not a bug.
7. **Never detect orbits by G-value equality** — signed orbits split into ± classes and fabricate
   `ρ=1`. Use the serialized table.
8. `<cmath>` must precede DCA's point-group headers (`Cn_2d`/`Sn_2d` use `std::cos`/`M_PI` without it).
9. HDF5 axis order of `cluster_greens_function_G_k_w` is reversed C++ leaf order: `(w,k,s1,b1,s0,b0)`.
10. **Absolute paths break when this repo moves.** Two bit us on the move to `dca/analysis/`:
    `build_symm`'s cached CMake source dir (retargeted in place — a fresh configure rebuilds all of
    DCA for a two-file change), and the Jupyter kernelspec at
    `~/.local/share/jupyter/kernels/symm-variance/kernel.json`, which pointed at a `.venv` that no
    longer existed. If nbconvert dies with `FileNotFoundError` on a `python`, that file is the fix.
11. **The FeAs input has `adjust-chemichal-potential`** (misspelled). DCA reads
    `adjust-chemical-potential`, its reader swallows unknown keys silently, and the default is
    `true`. Inert for our driver — it uses `DcaLoopData` but never `DcaLoop`, which is what calls the
    adjuster, so μ=1.45 stands (≈half filling, n≈2.02 of 4). But run that input through `main_dca`
    and the physics changes with no warning.
