# Implementation plan: measuring (and reporting) the variance reduction from symmetrization

**Audience:** an engineer picking this up cold. This document is self-contained — it states the goal, the
statistics, the relevant facts about the DCA++ code (with file:line pointers), the implementation across three
layers, how to validate it, and which existing files to mine as reference implementations.

---

## 1. What we're building

DCA++ runs a stochastic (quantum Monte Carlo) cluster solver, so its output — the single-particle Green's
function `G(k, ω)` — carries Monte-Carlo noise. The code can impose the lattice's point-group symmetry on `G`.
Symmetrization **averages `G` over each symmetry orbit** (the set of `(band, k)` indices the group maps into
each other), and because orbit-mates are equal in expectation, averaging them **reduces variance without
changing the mean.**

**Goal:** measure how much variance symmetrization removes — symmetry **ON vs OFF** — and ultimately report it
as a standard part of the solver's output, across multiple models (start with `square_lattice` and FeAs; the
design generalizes).

**Headline number (per model):** the total-variance ratio
$$R \;=\; \frac{\sum_x \mathrm{Var}(G[x])}{\sum_x \mathrm{Var}(\mathrm{Sym}\,G[x])}, \qquad x = (\nu,\nu,K,\omega),$$
read as *"symmetrization removes MC variance equivalent to ~R× more measurements, for free."* Always
accompanied by a **per-orbit breakdown** (below) — the scalar is the summary, the breakdown is the evidence.

---

## 2. The statistics (why this works, and the one formula)

Symmetrization of `G` is a **deterministic linear operator** `P` (a sign-aware orbit average — see §4.3 on
signs). For an orbit of `m` mates, each with per-sample variance `σ²` and mean pairwise correlation `ρ`
between mates' noise, the symmetrized entry is a mean of `m` correlated samples, whose variance is textbook:

$$\mathrm{Var}(\mathrm{Sym}\,G) = \frac{\sigma^2}{m}\big[1+(m-1)\rho\big] \;\;\Rightarrow\;\;
r \equiv \frac{\mathrm{Var}(G)}{\mathrm{Var}(\mathrm{Sym}\,G)} = \frac{m}{1+(m-1)\rho}.$$

Consequences you will use as correctness checks:

- **Singleton orbits (`m=1`) give `r = 1` exactly** — `P` is the identity there. This is a free, built-in null
  control: any deviation from 1 on a singleton means the pipeline is wrong.
- `ρ=0` (independent mates) → full `r = m`; `ρ=1` (perfectly correlated) → `r = 1`. Real cluster cells are
  ρ-limited, so expect `r` well **below** `m`. A measured `r ≥ m` is a red flag, not a triumph.
- The point group does not touch Matsubara frequency, so `r` is **ω-independent** — a slope in `r` vs ω signals
  contamination.

Because `P` is deterministic and linear, you do **not** need two separate experiments (on-run vs off-run):
from one run's raw samples you compute both `Var(G)` and `Var(P·G)`, and their **ratio is paired** (same
samples in numerator and denominator), which determines `r` far more tightly than either variance alone.

---

## 3. Facts about the DCA++ code you need (verified in-tree)

Paths are under `source/DCA/`. The solver of record is **CT-AUX**
([`include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp)).

### 3.1 The sampling unit is the MPI rank, and DCA already resamples over it

Error bars are computed by resampling across MPI ranks. `ErrorComputationType {NONE, STANDARD_DEVIATION,
JACK_KNIFE}` ([`phys/error_computation_type.hpp:21`](../source/DCA/include/dca/phys/error_computation_type.hpp#L21)).
The jackknife (`mpi_collective_sum.hpp` `jackknifeError`, real at
[`:364`](../source/DCA/include/dca/parallel/mpi_concurrency/mpi_collective_sum.hpp#L364)) resamples the
per-rank estimates; **it returns empty when there is only one rank** (`if (n==1) return err;`,
[`:370`](../source/DCA/include/dca/parallel/mpi_concurrency/mpi_collective_sum.hpp#L370)). So this measurement
**requires a multi-rank run** (`mpirun -n P`, with `P` = number of samples).

Jackknife matters (vs a naive stddev) because the self-energy Σ is a **nonlinear** function of `G`
(`Σ = G0⁻¹ − G⁻¹`); leave-one-out replicates propagate correctly through that nonlinearity.

### 3.2 Per-rank RNG streams are independent by construction

`global_id = local_id * num_procs + proc_id`
([`src/math/random/random_utils.cpp` `getGlobalId`](../source/DCA/src/math/random/random_utils.cpp)), then
`seed = hash(global_id + offset)` (`generateSeed`, same file). Within **one** run (one base seed), every rank
has a distinct `global_id` ⇒ distinct stream. So a single multi-rank run yields independent per-rank samples
automatically — **no seed-list management, no cross-run seed-collision hazard.** (That hazard only exists when
manufacturing independence across *separate* runs with hand-chosen seeds; we don't.)

### 3.3 `finalize()` flow — where symmetrization and error live

`finalize()` ([`:293`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L293)):

1. `collect_measurements()` ([`:295`→`:457`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L457)):
   with jackknife enabled, `leaveOneOutSum(M_r_w_, …)`
   ([`:476`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L476)) leaves
   **each rank holding a leave-one-out replicate of M**; `resolveSums()`
   ([`:518`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L518));
   `M_r_w_ /= accumulated_sign_` (the **global** sign,
   [`:521`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L521)); sets
   `averaged_ = true` ([`:565`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L565)).
2. `symmetrize_measurements()` ([`:296`→`:569`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L569))
   symmetrizes `M_r_w_` **and** `M_r_w_squared_` (the stddev accumulator) at
   [`:573`–574`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L573).
3. `compute_G_k_w_from_M_r_w()` ([`:299`→`:605`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L605))
   forms `G = G0 − G0·M·G0/β`, then `Symmetrize::execute(data_.G_k_w)`
   ([`:636`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L636)).
4. `compute_S_k_w_from_G_k_w()` ([`:304`→`:640`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L640))
   forms Σ, then (jackknife on) **writes the error functions**:
   `get_G_k_w_error() = jackknifeError(data_.G_k_w, …)` and the same for `G_r_w`, `Σ`
   ([`:680`–682`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L680)).

**Key takeaway:** production **already computes and writes `G_k_w_error` *with* symmetrization every run.** We
need its *without*-symmetrization twin and the ratio — that's the whole production change (§4.4). The error
getters already exist (`get_G_k_w_error()` at
[`dca_data.hpp:205`](../source/DCA/include/dca/phys/dca_data/dca_data.hpp#L205)).

### 3.4 `local_G_k_w()` gives the raw per-rank sample

`local_G_k_w()` ([`:897`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L897))
returns a **single rank's** G, built from that rank's raw accumulator
(`get_sign_times_M_r_w()`, [`:904`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L904)),
divided by the **local** sign ([`:906`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L906)),
through `compute_G_k_w_new()` — whose **only** symmetrization is one call at
[`:729`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L729). It throws
if `averaged_` is set, so it must be called **after `integrate()` and before `finalize()`**.

This is exactly the per-rank raw sample we want. Add a `symmetrize` flag that skips the
[`:729`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L729) call and it
returns **truly raw** samples, **independent of whatever point group the binary declares** — so you never need
a special "no-symmetry" build of any model.

### 3.5 Symmetrization of G is linear; of Σ is not

`Symmetrize::execute(data_.G_k_w)` ([`:636`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L636))
is a linear orbit-average on `G`. The group action commutes with the `G0`-fixed Dyson map, so applying our `P`
to raw `G` reproduces the production symmetrized `G` (validated in §6). **This linearity holds only for the
single-particle G.** For Σ and G4 you cannot recover the symmetrization gain by a post-hoc linear `P`; you must
re-symmetrize inside each jackknife replicate through the nonlinearity — which the production path (§4.4) does
and numpy cannot. This is *the* reason the production layer exists.

### 3.6 Domain layout of `G_k_w`

Type `func::function<complex, dmn_variadic<nu, nu, KDmn, w>>`, where `nu = b·s` (bands × spins). Square
single-band: `b=1, s=2 → nu=2`. FeAs: `b=2, s=2 → nu=4`. Storage is `G_k_w(b1,s1,b2,s2,k,w)` with the leading
`nu×nu` a matrix of size `b·s`. The point group acts on the flattened `(nu,nu,K)` index **per fixed ω**; orbits
partition that index set (FeAs's interesting orbits mix band and k indices).

---

## 4. Implementation — three layers

Build them in order; each is validated before the next (§6).

### 4.1 DCA source change (one small, shared edit)

In `ctaux_cluster_solver.hpp`:

- **`auto local_G_k_w(bool symmetrize = false) const;`** — thread the flag through to skip the
  `Symmetrize::execute` at [`:729`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L729)
  (in `compute_G_k_w_new`). Default `false` = raw. ~1 line plus signature plumbing.

Separately, a **sign-aware orbit-table serializer**: emit the full point-group orbit table the symmetrization
machinery already builds — a list of orbits, each `[(flat_index, sign), …]` over the `(nu,nu,K)` index for one
ω — to a small sidecar file (HDF5 or JSON). Find where the derivation constructs the orbits (the
`Symmetrize`/point-group-imposition code that #374 introduced) and dump it. This is the **single source of
truth** shared by production `Symmetrize` and numpy `P` (see §4.3 on why not to re-derive).

### 4.2 The driver (new tree `symm_variance/`)

A thin standalone binary per model. Flow — **one QMC iteration**, dump raw per-rank G **before** finalize:

```
read input JSON  ->  data.initialize()
ThreadedSolver<CT_AUX> qmc(parameters, data, nullptr)
qmc.initialize(0);  qmc.integrate();
// NEW: on every rank, grab the raw per-rank sample
auto G_i = qmc.local_G_k_w(/*symmetrize=*/false);
// write G_i tagged by rank  (gather to rank 0, or one file per rank)
qmc.finalize(loop_data);            // optional: also dump the production symmetrized G_k_w for §6.2
```

Dump the stack `[n_rank, nu, nu, K, w]` complex plus metadata (`n_ranks`, `k_elements`, `measurements`,
model/group id). Keep the config minimal: single DCA iteration, plain DCA (not DCA+), CT-AUX, small cluster
(e.g. Nc=16). The declared point group is **irrelevant** to the raw dump (§3.4) — run each model's stock setup.

**Sampling budget:** rank = one sample. Use `mpirun -n P` with `P ≈ 64` to start (variance estimate relative
SE ≈ √(2/(P−1)) ≈ 18% at P=64; the *ratio* is much tighter because it's paired). Each rank may run its own
walker/accumulator threads.

### 4.3 numpy — two roles, one data contract

**Data contract:** the **full raw per-rank tensor** (not summary stats) + the serialized orbit table. The
playground must be able to compute anything later (ρ, covariance, ω-slices), which pre-reduced variances
foreclose. This data stays in the numpy tier; it is never a production output.

**Role 1 — validator (temporary).** Compute `R` and per-orbit `r` on `G`; confirm the self-consistency checks
(§6) and, if you run one, the independent-ensemble cross-check. Retire once the production path matches.

**Role 2 — statistical playground (permanent).** The fine-grained figures that do not belong in the HDF5:
the orbit "migration" scatter (described below — the signature figure), per-orbit `r` vs the `m/[1+(m−1)ρ]`
overlay, ρ maps, singleton nulls, ω-flatness, per-model comparison bars, bias histograms, "which entries
benefit most."

Core computation (real and imaginary parts treated independently, matching DCA's jackknife convention):

```python
# Graw[i, x, w] complex; i: rank, x: flat (nu,nu,K); orbits: [ [(x, sign), ...], ... ]
Gsym = Graw.copy()
for orb in orbits:                                   # sign-aware orbit average = P
    xs, sgn = [x for x,_ in orb], np.array([s for _,s in orb])
    avg = (sgn[None,:,None] * Graw[:, xs, :]).mean(axis=1)      # [n, w]
    for x, s in orb:
        Gsym[:, x, :] = s * avg

def var_r(G): return G.real.var(0, ddof=1), G.imag.var(0, ddof=1)   # over rank axis
(vr_re, vr_im), (vs_re, vs_im) = var_r(Graw), var_r(Gsym)
r_re, r_im = vr_re/vs_re, vr_im/vs_im                # (n-1) cancels in the ratio
R = (vr_re.sum()+vr_im.sum()) / (vs_re.sum()+vs_im.sum())          # headline (see note)
# per orbit: measure rho from sign-aligned mates, overlay m/(1+(m-1)*rho) vs measured r
```

**Signature figure — the orbit "migration" scatter.** The most intuitive single view of the result, and the
one to build first once data lands. For a chosen orbit (its `m` sign-related `(band, k)` mates) at a fixed ω,
plot in the **complex plane** (Re `G` on x, Im `G` on y):

- **Before (raw):** one point per rank per orbit-mate — a scattered cloud of `m × P` points. Orbit-mates share
  the same expectation by symmetry, so the `m` sub-clouds sit on top of each other, spread out by the raw MC
  noise. (Apply the mate's sign first so odd-representation mates align rather than landing on the opposite
  side.)
- **After (symmetrized):** replace each rank's mate values with their sign-aware orbit average `P·G_i`, and
  draw an arrow from every raw point to its symmetrized position.

The points **"migrate in":** the cloud contracts toward its own centroid by the reduction factor, its radius
shrinking as `1/√r` with `r = m/[1+(m−1)ρ]`, while the centroid **does not move**. Three claims are legible at
a glance:

- **Variance reduction** — the cloud visibly shrinks.
- **Mean preservation** — the centroid stays put; arrows point inward, not sideways (no net drift). A centroid
  that shifts means a bias, not a reduction.
- **The mechanism (ρ)** — how far the points travel encodes the correlation: near-independent mates (ρ→0)
  collapse dramatically toward the mean; strongly-correlated mates (ρ→1) barely move. **Singleton orbits
  (`m=1`) show no migration at all** — the built-in null control, visually static.

Present it as a **small-multiples grid**, one panel per orbit (or per orbit-size class), so
singletons-static / large-orbits-collapse reads immediately; optionally animate the raw→symmetrized
interpolation so the "migration" is literal. This figure carries the intuition that the summary scalar `R`
compresses — pair the two in any writeup.

**`R` support note.** `R` is variance-weighted (noisy entries dominate — a fair "total noise removed"
reading). Report it **twice**: full support and non-null support. Some entries are **symmetry-forced nulls**
(the group sends them to ~0, e.g. FeAs's largest interband orbit); there `Var(Sym G) ≈ 0`, i.e. infinite
per-entry reduction. The summed form tolerates them, but they can inflate `R`; showing both makes their
contribution visible instead of hidden.

**On the orbit table — do not re-derive orbits from G values.** Orbit-mates can carry a relative **sign**
(e.g. an interband component odd under a band swap). Detecting mates by equality of `G` values silently splits
a signed orbit into two ± classes, corrupting orbit sizes and yielding a spurious `ρ=1`. Use the serialized
sign-aware table (§4.1) as authoritative. An independent geometric re-derivation of orbit **membership** (from
the k-mesh) is fine as a cross-check, but signs come from the table.

### 4.4 Production reporting (end state)

Report the on/off error pair as a standard output. The leave-one-out replicates already exist (§3.3); fork
once, **before** symmetrization mutates M:

```cpp
collect_measurements();                 // per-rank leave-one-out M replicates
if (report_symm_gain_) {                // new flag; requires jackknife + n_rank > 1
  auto M_raw = M_r_w_;                   // raw replicate, pre-symmetrization
  compute_G_k_w(M_raw, G_raw, /*symmetrize=*/false);   // skip :573 and :636
  data_.get_G_k_w_error_unsym() = concurrency_.jackknifeError(G_raw);   // new Data member
}
symmetrize_measurements();              // existing
compute_G_k_w_from_M_r_w();             // existing -> G_k_w_error (with symmetrization) at :680
// write R = ||error_unsym|| / ||error_sym|| (per entry + total) alongside the errors
```

This is the **production estimator run twice on identical replicates** — nothing new to justify, cost is one
extra G-formation + one reduction (negligible), off by default. It is the **only correct route for Σ and G4**
(§3.5). **Trap:** use the JACK_KNIFE path and fork *before* `symmetrize_measurements()`. Do **not** try to read
the STANDARD_DEVIATION output for this — `symmetrize_measurements()` symmetrizes `M_r_w_squared_`
([`:574`](../source/DCA/include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp#L574)), so that
error accumulator is already arm-dependent.

---

## 5. Predictions / expected results

- Singletons pinned at exactly 1.0 (hard control).
- `r` per orbit below `m`, tracking `m/[1+(m−1)ρ]` with ρ measured from the same samples; agreement across the
  whole orbit spectrum on a `y=x` line is the real evidence.
- `r` flat in ω.
- Square Nc=16 (4×4): max orbit size is 4 (a generic orbit-8 point first appears at 8×8), and single-band
  orbit-mates are strongly correlated (expect ρ high, so `r` modest, ~1–1.4). FeAs Nc=16 has richer orbits
  (band-mixing), including the more interesting reductions.
- Bias: `⟨Sym G⟩ − ⟨G⟩` within MC error everywhere; per-orbit χ² / z-scores standard-normal.

---

## 6. Validation ladder

Each rung licenses the next.

1. **Self-consistency (internal, no external oracle needed).** Singletons = 1.0 exactly; `r` matches
   `m/[1+(m−1)ρ]` across orbits; `r` flat in ω; bias z-scores ~ N(0,1). These alone strongly constrain
   correctness.
2. **Mean preservation / table check.** `mean_rank(P·raw)` equals a stock full-group run's finalized
   `data.G_k_w` to numerical precision (expected from §3.5). Confirms numpy `P` ≡ production `Symmetrize` and
   the serialized table. (Dump the finalized `G_k_w` from the same driver run.)
3. **Production vs numpy.** The `finalize()` dual-jackknife `R` on `G` matches the numpy `R` on square and
   FeAs. This is where you confirm the local-vs-global sign normalization (§3.3–3.4) doesn't bite — it cancels
   in the ratio, but verify.
4. **Production Σ / G4.** Validated *by construction* once rung 3 holds; numpy cannot check these.

**Optional independent cross-check.** For extra assurance on rung 1 you can measure `R` a completely
independent way: many independent *whole* runs (one sample each), sample-variance across runs with and without
the declared group, no per-rank machinery. This shares no code with the per-rank path, so agreement is strong
evidence. A working reference implementation of this ensemble approach exists (see §7) — you can run it as an
oracle, but the plan does not depend on keeping it.

---

## 7. Reference implementations already in this repo

There is a prior prototype you can mine. Treat these as **reference code to adapt**, not as constraints.

- **Driver skeleton + HDF5 metadata pattern:** [`variance_demo/variance_demo_main.inc`](variance_demo/variance_demo_main.inc)
  — read→initialize→integrate→finalize→dump, with the metadata-tagging pattern. Adapt by inserting the
  pre-finalize `local_G_k_w(false)` per-rank dump.
- **Standalone build (pulls DCA in as a subdirectory, no DCA modification needed):**
  [`variance_demo/CMakeLists.txt`](variance_demo/CMakeLists.txt) and [`SETUP.md`](SETUP.md). The build
  *mechanism* carries over ~1:1; you just need one target per model instead of four.
- **Thin test-binary instantiation (lattice/solver/typedefs in a standalone binary):**
  [`variance_demo/symmetry_variance_setup.hpp`](variance_demo/symmetry_variance_setup.hpp) +
  [`variance_demo/ctaux_variance_demo_on.cpp`](variance_demo/ctaux_variance_demo_on.cpp). Shows how to stand up
  CT-AUX against a chosen lattice outside `applications/dca`.
- **Input config:** [`variance_demo/square_variance_input.template.json`](variance_demo/square_variance_input.template.json)
  — Nc, single DCA iteration, CT-AUX walker/accumulator counts. Set
  `Monte-Carlo-integration.error-computation-type` to `JACK_KNIFE` for the production layer.
- **numpy plumbing + sign-aware orbit handling + ρ/prediction math:**
  [`notebooks/variance_demo_lib.py`](notebooks/variance_demo_lib.py) and
  [`variance_demo_feas_lib.py`](notebooks/variance_demo_feas_lib.py). The sign-aware orbit logic and the
  `m/[1+(m−1)ρ]` computation are directly reusable for the playground.
- **Operational lessons for the run script:** [`variance_demo/run_replicas.sh`](variance_demo/run_replicas.sh)
  — **cap the BLAS thread pool** (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`; otherwise each rank
  spawns an `nproc` pool and the box thrashes) and tear down child processes on interrupt. The concurrency
  model changes (you want `mpirun -n P`, not many single-rank invocations), but these lessons carry.
- **Independent-ensemble oracle (§6 optional):** the same `variance_demo/` binaries + notebooks implement the
  many-independent-runs measurement; runnable as a cross-check.

---

## 8. Build & run mechanics

- Build as a standalone CMake project that adds the DCA checkout as a subdirectory (see `SETUP.md`) — **no DCA
  source modification is required for the build itself**; the `local_G_k_w` flag and serializer are ordinary
  edits to the DCA headers you compile against.
- Run multi-rank: `mpirun -n P ./<model>_symm_variance <input.json> <out.hdf5>`.
- **Shared machine:** this box is shared with other jobs. Check load before launching; size `P × threads/rank`
  to the cores you may use; keep the BLAS cap (§7). Don't benchmark timing on a loaded box.

---

## 9. Milestones

1. `local_G_k_w(symmetrize=false)` + orbit-table serializer (§4.1).
2. `symm_variance/` driver: multi-rank raw per-rank dump for **square** (§4.2).
3. numpy validator + first playground figures on square; pass rung 1 self-consistency (§6.1) and rung 2
   mean-preservation (§6.2).
4. FeAs via the same pipeline (§10 template).
5. Production dual-jackknife (§4.4); pass rungs 3–4.
6. Extend: more models, 8×8, Σ/G4 reporting.

**Per-model template** (steps 2–4 repeat): (a) a stock binary/setup, (b) the full-group sign-aware orbit
table, (c) the same numpy. Adding a model is filling those three slots.

---

## 10. Scope boundaries (state these in any writeup)

- Variance of a **single-iteration** estimator, not a converged DCA solution (one iteration only —
  symmetrization inside the self-consistency loop entangles noise reduction with a changed convergence path).
- **Plain DCA, not DCA+** (symmetrization imposed on the CLUSTER family; DCA+ routes through LATTICE).
- **CT-AUX** (its real-space symmetrization branch is the one to exercise; CT-INT is a possible later
  cross-check but does not currently compile in this tree).
- numpy covers the **single-particle G** only; Σ/G4 exist solely via the production path (§3.5, §4.4).

---

## 11. Gotchas, in one place

1. **Needs `n_rank > 1`.** Jackknife/variance over ranks is empty for a single rank (§3.1). Always `mpirun -n P`.
2. **Call `local_G_k_w()` before `finalize()`** — it throws once `averaged_` is set (§3.4).
3. **Raw means `symmetrize=false`** — otherwise you get the declared group's symmetrized G, not raw (§3.4).
4. **Orbit signs are real** — never detect orbits by G-value equality; use the serialized sign-aware table
   (§4.3).
5. **Production fork point** — JACK_KNIFE path, fork the raw replicate *before* `symmetrize_measurements()`;
   avoid the STANDARD_DEVIATION accumulator (`M_r_w_squared_` is symmetrized) (§4.4).
6. **Symmetry-forced nulls** — some entries go to ~0 under the group; handle in `R` (report full and non-null
   support) and exclude from per-entry `r` (0/0) (§4.3).
7. **`R` is a summary, not the whole story** — always ship it with the per-orbit structure panel that backs it
   (§1, §5).

---

## 12. Open questions

- `R` primary support: full vs non-null (lean: show both).
- Ranks vs walkers/rank split of the sampling budget.
- Orbit-table serialization format and exactly where the derivation exposes it.
- Naming of the tree (`symm_variance/`) and this file.
