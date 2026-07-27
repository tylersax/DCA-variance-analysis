# Roadmap — symmetrization variance reduction in DCA++

**This file is the single task list.** Edit it freely; it is the source of truth for what's next.
Design rationale lives in [`analysis/symm-variance-plan.md`](analysis/symm-variance-plan.md) (long
reference document). Presentation-ready claims accumulate in [`TAKEAWAYS.md`](TAKEAWAYS.md).

**Goal.** Measure how much Monte-Carlo variance the point-group symmetrization removes from the
single-particle Green's function, report it as `R = Var(G)/Var(Sym G)`, and eventually make it a
standard solver output.

**Current priority: exercise the statistics and broaden model coverage. Production comes later.**

---

## Done

| # | Milestone | Result |
|---|---|---|
| 1 | `local_G_k_w(symmetrize=false)` + orbit-table serializer | shipped (plus a latent-bug fix; see gotchas) |
| 2 | `symm_variance/` driver, multi-rank raw per-rank dump | shipped, square |
| 3 | numpy validator + figures; validation rungs 1 & 2 | all rungs pass |
| 4 | FeAs through the same pipeline | all rungs pass |

| model | orbits | ρ (mates) | R (full / non-null) |
|---|---|---|---|
| square / D4, β=1 | m = 1, 2, 4 | ~0.94 | **1.034** / 1.034 |
| FeAs 2-band, β=5 (declared 2 ops → derived 8) | m = 2, 4, 8 (+24 forced nulls) | ~0.06 | **2.482** / 2.156 |

16 ranks each, 4×4 clusters, single DCA iteration. Singletons pin at exactly `r=1`; per-orbit `r`
matches `m/[1+(m−1)ρ]` to ≤2%; `r` flat in ω; mean preservation vs production to 2e-16.

---

## Task list — in priority order

### 1. M-scaling control  ▸ do first

**Why first:** it licenses the entire budget strategy. `R` is scale-free (`Var` and `Var∘P` both go
as `1/N`, so the ratio is depth-independent) *only if* each rank's sample is in the CLT regime —
past thermalization and well beyond the autocorrelation time. Everything else assumes this.

- At fixed `P`, measure `R` at `M` and `4M`; confirm flat within the paired estimator's precision.
- Compare mechanism diagnostics (`σ²(r)` profile, within-shell correlation) across the two `M` —
  drift there is the early warning.
- Cheap on the current 4×4 runs. **Re-run at each β** in task 3: autocorrelation times grow with β,
  so a depth that is safely asymptotic at β=1 may not be at β=8.

### 2. Statistical hardening

Exercise the estimator itself before trusting it across many models.

- **Headroom metric.** Add `R_ideal = ΣVar / Σ(Var/m)` — the `ρ=0` ceiling for the model's actual
  orbit structure — and report **efficiency = R / R_ideal**. Separates "small because orbits are
  small" from "small because the noise is symmetric", which the bare `R` conflates.
- **Reduction map, not a lone scalar.** Ship `r` resolved by orbit and entry class rather than a
  single `R`. It is the transferable object: it describes the noise covariance structure, so anyone
  who later wants an observable-level number can reweight it. (Observable-level reduction itself is
  **out of scope** — see Scope boundaries.)
- **`R` split by entry class** (band-diagonal vs interband; local- vs nonlocal-dominated). `R` is
  variance-weighted, so it is dominated by the noisiest entries — which are exactly the
  local-dominated, high-ρ ones symmetrization helps least. Aggregate `R` understates the benefit to
  the physically interesting components.
- **Confidence intervals**: bootstrap over ranks; scale ranks up (~64) before quoting `R` with error
  bars. Confirm `R` is stable across base seeds.
- Optional independent oracle: many independent whole runs, sample-variance across runs, sharing no
  code with the per-rank path.

### 3. β ladder — test the central hypothesis  ▸ needs GPU

Square's `R = 1.03` was measured at **β = 1**, exactly the regime where noise is dominated by
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
  pays off — **including the regimes where it collapses** — is the research output here; a boundary
  is more useful to a practitioner than an unqualified endorsement.
- Per-rank signs were exactly equal at β=1 (no sign problem); once they diverge, the phase-weighted
  mean path matters.
- Cost climbs steeply: expansion order ~ β·U·Nc, walker linear algebra ~ cubic in expansion order.

**GPU build.** 4× RTX A5000 (24 GB, idle), CUDA 12.2. `-DDCA_WITH_CUDA=ON` +
`CUDA_GPU_ARCH=sm_86`. `symm_variance_setup.hpp` already selects `linalg::GPU` under `DCA_HAVE_GPU`,
so no driver change is needed. **Oversubscribe — several ranks per GPU**: the statistics want many
ranks at modest depth, not 4 heavy ranks. Re-validate rungs 1 & 2 on a GPU build before trusting new
physics from it.

### 4. Broaden model coverage — vary ONE axis at a time  ▸ carries the headline claim

**The claim this task establishes:** symmetrization pays off most where the physics is most
interesting — multi-orbital models, and *within* them the interband components that carry the
distinctive physics. That is the strongest selling point for the work, because it is where people
actually want to use DCA++. See TAKEAWAYS §4c.

**Four runs turn that framing into a defensible claim:**

| run | establishes |
|---|---|
| single-band square at FeAs's β | **the keystone** — de-confounds temperature from band count |
| a 3-orbital model | the trend in `nb`; exercises `U_S` as a genuine permutation, not a swap |
| an inequivalent-orbital model | benefit should **vanish** — confirms the gain comes from symmetry-*equivalent* orbitals, not orbital count |
| `R` split by entry class, on all of the above | surfaces the interband concentration that aggregate `R` buries |

Prediction to test: `R` grows with the number of symmetry-equivalent orbitals
(nb=1 → 1.03, nb=2 → ~2.5, nb=3 → ?).

**Design constraint (important):** square and FeAs currently differ in β, band count, interaction and
filling *simultaneously*, so the ρ difference between them **cannot be attributed to any single
cause**. The mechanism is measured; the causation is not. The sweep must isolate axes.

Axes, and which knob in `r = m/[1+(m−1)ρ]` each moves:

| axis | moves | note |
|---|---|---|
| cluster size 4×4 → 8×8 | `m` | first *free* orbit-8 k-point only exists at 8×8 |
| point group `D4`=8 → `D6`=12 → 3D `O_h`=48 | `m` | needs non-square lattices |
| band count / band-permuting ops | `m` | enlarges the `(b0,b1,k)` index space |
| β | `ρ` | task 3 |
| U, filling | `ρ` | |
| band-diagonal vs interband entries | `ρ` | interband has no local scalar channel |

- **Single-band square at FeAs's β** (and vice versa where meaningful) is the key de-confounding run.
- Open empirical question: **does `ρ` fall with cluster size?** That decides whether the `m`-ceiling
  is reachable in practice.
- Needs a sweep driver + results aggregator; record per-orbit `m`, `ρ`, `r`, `R` (both supports),
  efficiency, and the mechanism diagnostics for every run.

### 5. Migration scatter figure

The plan's signature visualization (§4.3), still to build. For a chosen orbit at fixed ω, plot mates
in the complex plane: raw cloud of `m×P` points (signs applied first so odd mates align), then arrows
to their sign-aware orbit average. The cloud contracts toward its own centroid by `1/√r` while the
**centroid does not move** — variance reduction, mean preservation, and the ρ mechanism all legible
at once. Singletons show no migration: the visual null control.

Small-multiples grid, one panel per orbit. Ideal contrast: square (ρ≈0.94, barely moves) beside FeAs
`m=8` (ρ≈0.08, collapses) — makes ρ-limited vs m-limited visual. Add as `03_migration_scatter.ipynb`
via `analysis/build_notebooks.py`.

### 6. Production reporting  ▸ deferred until the above is settled

Fork the raw replicate **before** `symmetrize_measurements()` mutates `M`, form the unsymmetrized
`G`, and write the on/off error pair plus `R` as a standard solver output. Requires `JACK_KNIFE` and
`n_rank > 1`; off by default; cost is one extra G-formation plus one reduction.

This is the **only correct route for Σ and G4**: symmetrization of `G` is linear, but `Σ = G0⁻¹ − G⁻¹`
is not, so a post-hoc linear operator cannot recover the gain — it must be re-symmetrized inside each
jackknife replicate. numpy cannot do this.

⚠️ Do **not** read the `STANDARD_DEVIATION` accumulator for this: `symmetrize_measurements()`
symmetrizes `M_r_w_squared_`, so that accumulator is already arm-dependent.

### 7. Σ / G4 reporting  ▸ OUT OF SCOPE for this analysis

Follows from task 6 by construction, but observable-level variance is outside the boundary set with
the advisor (see Scope boundaries). Listed only so the path is known, not as planned work.

---

## Design principle worth not re-deriving

`R` is **scale-free**: `Var(Ĝ_N) = Σ/N` and `Var(P·Ĝ_N) = PΣPᵀ/N`, so `1/N` cancels and `R` depends
only on the noise covariance *structure*, not on compute spent. Therefore:

- Precision on `R` comes from the **number of independent samples (ranks)**, not run depth.
- Measurements beyond the autocorrelation threshold buy **nothing** for `R`.
- **Many ranks, modest depth** — and the measured `R` transfers to any production scale.

The estimator is also **paired** (`P` is deterministic and linear, so one sample set gives both
numerator and denominator), which is why 16 ranks beat the prior ~128-runs-per-arm ensemble design.

---

## Where things live

This repo (`DCA-variance-analysis`) is checked out at `/home/tsax10/dca/analysis`. It holds **both**
the current project and the superseded prototype:

| path | what |
|---|---|
| `symm_variance/` | **current work** — driver, orbit-table serializer, inputs, run script |
| `symm_variance/analysis/` | numpy libs + notebooks (`01_validation_ladder`, `02_noise_mechanism`) |
| `symm_variance/runs/` | run data the notebooks read |
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
(OpenMPI 4.1.6, HDF5 1.14.3, FFTW, LAPACK), system g++ 13.3.
**Run:** `symm_variance/run_symm_variance.sh <square|fe_as> <n_ranks> <measurements> [seed] [outdir]`
(caps BLAS threads; shared 128-core box — check load first).
**Notebooks:** Jupyter kernel `symm-variance (py)`; regenerate via
`symm_variance/analysis/build_notebooks.py`.

---

## Scope boundaries (state these in any writeup)

- **We measure `Var(G)` and stop there.** ← decided with advisor, 2026-07-27. Observable-level
  variance (Σ, susceptibilities/`Tc`, spectra) is **out of scope**. State the caveat — the reduction a
  user experiences can be higher or lower than the `R` we quote, since each downstream transform
  reweights the noise — but do not chase it. This boundary exists to keep the analysis finite.
- Variance of a **single-iteration** estimator, not a converged DCA solution.
- **Plain DCA, not DCA+** (symmetrization imposed on the CLUSTER family).
- **CT-AUX only** — CT-INT does not currently compile in this tree.
- Runs so far are 16 ranks — fine for the deterministic rungs, light for tight `R` intervals.

---

## Gotchas that cost time once already

1. **Needs `n_rank > 1`.** DCA has *no* single-rank error bars: `JACK_KNIFE` returns empty at `n==1`,
   and `STANDARD_DEVIATION` (`average_and_compute_stddev`) yields identically zero.
2. **Call `local_G_k_w()` before `finalize()`** — it throws once `averaged_` is set.
3. **Rung 2 requires `error-computation-type = NONE`.** Under `JACK_KNIFE`, `finalize` writes a
   leave-one-out replicate into `G_k_w`, producing a ~3% low-frequency gap that is a jackknife
   artifact, not a bug.
4. **Never detect orbits by G-value equality** — signed orbits split into ± classes and fabricate
   `ρ=1`. Use the serialized table.
5. `<cmath>` must precede DCA's point-group headers (`Cn_2d`/`Sn_2d` use `std::cos`/`M_PI` without it).
6. HDF5 axis order of `cluster_greens_function_G_k_w` is reversed C++ leaf order: `(w,k,s1,b1,s0,b0)`.
