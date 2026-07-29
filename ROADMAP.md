# Roadmap — symmetrization variance reduction in DCA++

**This file is the task list.** It carries what to do next, the current state, and the conventions
that govern future work — nothing else. **Findings and the evidence for them live in
[`TAKEAWAYS.md`](TAKEAWAYS.md)**; design rationale in [`symm-variance-plan.md`](symm-variance-plan.md).
If you want to know *why* a number is what it is, go there, not here.

**Goal.** Measure how much Monte-Carlo variance the point-group symmetrization removes from the
single-particle Green's function, report it as `R = Var(G)/Var(Sym G)`, and eventually make it a
standard solver output.

**Current priority: task 4 — broaden model coverage.** §3e is **done** (2026-07-28): `R` does not
drift across the iterations a production run performs, so the bare-bath caveat is closed and every
committed number reads as what a user experiences (TAKEAWAYS §4e). The β ladders are measured on both
models and the trend reproduces with a live sign problem (§3a). Task 4 is what the ladders point at
directly: FeAs ends m-limited, so orbit size — not colder physics — is what buys more.

> **Starting fresh on task 4?** Read *§4 → "Starting this task cold"* first — it has the model
> candidates, the three files needed to add one, the environment paths, and the one fact that decides
> the cost of this whole task (band-permuting symmetry is **derived from `H0`**, so a new
> multi-orbital model needs no symmetry code). Then Conventions, then Gotchas.

---

## State

| # | Milestone | Result |
|---|---|---|
| 1 | `local_G_k_w(symmetrize=false)` + orbit-table serializer | shipped (plus a latent-bug fix) |
| 2 | `symm_variance/` driver, multi-rank raw per-rank dump | shipped |
| 3 | numpy validator + figures; validation rungs 1 & 2 | all rungs pass |
| 4 | FeAs through the same pipeline | all rungs pass |
| 5 | M-scaling control, bootstrap CIs, depth floors | `R` scale-free above a per-model depth floor |
| 6 | Seed ensemble: 32 independent base seeds per model, whole-run oracle, paired depth check | `R` pinned to 1.6% (FeAs) and 0.13% (square) |
| 7 | β ladder on square, β ∈ {1,2,4,8}, 32 seeds/rung, per-β depth floors | `ρ` falls, `R` rises, both resolved — square is **not** intrinsically ρ-limited |
| 8 | β ladder on FeAs, β ∈ {1..5}, 32 seeds/rung, sign channel live | trend reproduces; `ρ` falls despite ⟨s⟩ → 0.148; `R` saturates against the **m**-ceiling |
| 9 | Bath drift: real `DcaLoop`, 10 iterations × 8 seeds, square β=8 | `R` flat across iterations — **the bare-bath caveat is closed**; found a second gap (coarse-graining) the scope had missed |

| model | **`R`** | orbits | ρ (mates) | depth floor, per rank |
|---|---|---|---|---|
| square / D4, β=1 | **1.0425 ± 0.0013** | m = 1, 2, 4 | ~0.94 | ≈64 — autocorrelation |
| square / D4, β=8 | **1.3429 ± 0.0088** | m = 1, 2, 4 | ~0.56 | ≈256 — autocorrelation |
| FeAs 2-band, β=5 | **3.042 ± 0.049** | m = 2, 4, 8 (+24 forced nulls) | ~0.06 | ≳1000 — **sign problem** |

**32 independent base seeds × 64 ranks × 2048 measurements/rank**, 4×4 clusters, **single DCA
iteration at the bare (Σ=0) bath** — see the bath convention below, and §3e for the measurement that
these numbers transfer to the bath a production run actually samples; `±` = standard error over seeds
(95% t-CI: square `[1.0398, 1.0452]`, FeAs
`[2.943, 3.141]`). FeAs declares 2 ops → derives 8. Validation rungs pass on both models: singletons
pin at exactly `r=1`, per-orbit `r` matches `m/[1+(m−1)ρ]`, `r` flat in ω, mean preservation vs
production to 2e-16. Evidence and mechanism: **TAKEAWAYS §1a** and `03_m_scaling.ipynb`; the
intervals and every cross-check below in `runs/seed_ensemble_{square,fe_as}.json`.

Each headline survives four checks that do not share the estimator's assumptions
(`analysis/seed_ensemble.py`, milestone 6):

| check | square | FeAs |
|---|---|---|
| whole-run oracle — variance across 32 whole runs, replicate 64× deeper, no per-rank machinery | 1.047 `[1.028, 1.090]` ✓ | 3.28 `[2.98, 3.65]` ✓ |
| paired depth check — same seeds rerun at 4× depth | `+0.0007 ± 0.0019`, flat | `+0.025 ± 0.100`, flat |
| median / pooled / distribution-free bootstrap vs the t-interval | agree | agree |
| contamination gate — unusable runs, worst measured-vs-predicted `r` | 0 runs, 0.0004 | 0 runs, 0.108 |

> ⚠️ Earlier drafts quoted **square 1.034 / FeAs 2.482** (single 16-rank runs), then **1.041 ± 0.003
> / 3.17 ± 0.14** (3 seeds). All four are superseded by the seed-ensemble values above. The 16-rank
> FeAs number especially — its 95% CI was `[2.13, 3.51]`, and 2.482 was a low draw from it. Every
> superseded value is statistically consistent with its replacement, so no *claim* has been
> overturned; the intervals just got honest. Do not reuse the old numbers.

---

## Conventions — settled, do not re-litigate

**`R` means the FULL support, and it is the only headline number** (decided 2026-07-27). It tracks
what a user actually experiences: run unsymmetrized and you carry the symmetry-forbidden entries too,
and their noise propagates downstream. Non-null `R` is a secondary structural note — related exactly
by `R_full = R_non-null/(1 − w_null)`, so it adds only `w_null`, the forced-null share of raw
variance. Report `w_null` where noise *structure* is the point, above all in the cross-model sweep
(tasks 3 and 4), since it varies by model and would otherwise absorb part of any trend. **Efficiency
`R/R_ideal` is a dev diagnostic and does not go in a writeup.** Standing principle behind all of
this: *only take on complexity that buys something worth its cost.*

**Solver knobs that are not the swept axis get ONE sensible production-representative value, held
fixed — we do not map `R` against them** (settled 2026-07-28, with the advisor). `R` is a property of
the noise *structure*, and that structure responds to many things: `beta`, depth, the sign, and —
measured below — warm-up. Characterizing that whole surface is a different project, and a much larger
one. The finding worth reporting is that **the noise structure in DCA++ is rich and not obvious on
the surface**, illustrated with a few clear, well-controlled examples (`R` rising with `beta` is the
lead one). Chasing every knob that moves `R` is parameter-hacking, not a result.

So: **`warm-up-sweeps = 200` and `sweeps-per-measurement = 2` for BOTH models, everywhere.** Square's
ladder already used 200; the FeAs template's 80 is aligned up to it. Both values are recorded per run
in the HDF5 metadata, so any run's provenance is checkable rather than remembered.

- **What that costs, stated honestly:** the FeAs `beta=5` rung is no longer a bit-exact reproduction
  of the committed `3.042 +/- 0.049` headline, which ran at warm-up 80. It stays a meaningful
  cross-check — 80 and 320 agree within error (below), so 200 is in the same flat region — just not
  an exact one.
- **The measured sensitivity, recorded so nobody rediscovers it and re-opens this.** FeAs, `beta=4`,
  2048/rank, 8 seeds per arm: warm-up 80 → `R = 2.834 +/- 0.060`; 320 → `2.836 +/- 0.071`
  (`+0.003 +/- 0.105`, flat); 1280 → `2.625 +/- 0.030` (`-0.208 +/- 0.064`, 1/8 seeds positive), and
  the across-seed SD halves, 0.170 → 0.084. Flat then a step is the wrong shape for smooth
  thermalization, so this is **an observation, not a mechanism** — it is not chased further by
  deliberate choice, not because it was resolved.
- ⚠️ An earlier 3-seed pass at `beta=4` showed `+0.34 +/- 0.13` (3/3 same sign) and `beta=5` showed
  `-0.25 +/- 0.23` with both large differences traceable to a single contaminated run (outlier index
  3.66). At 8 seeds the `beta=4` signal vanished. **Three seeds cannot resolve a parameter effect on a
  sign-problem model** — do not re-run that comparison at n=3 and believe it. The same trap produced a
  false floor call at β=3 and a false "the committed headline is biased low" alarm in one session.

**Every number here is measured at the BARE bath — `G0_cluster_excluded = G0`, the bare cluster
non-interacting Green's function** (settled 2026-07-28). The driver does not instantiate a `DcaLoop`
at all: `symm_variance_main.inc` calls `data.initialize()` → `qmc_solver.integrate()` directly, and
`DcaData::initialize()` builds only `H0`/`H_int` and the non-interacting `G0` (with
`G0_cluster_excluded = G0`). The templates set `"iterations": 1` and no `initial-self-energy`. So we
skip `perform_cluster_mapping`, `adjust_coarsegrained_self_energy` and
`perform_cluster_exclusion_step` (`dca_loop.hpp` `execute()`) — the steps that turn a converged `Σ`
into the bath the walker samples.

> ⚠️ **This is NOT "iteration 1 of a cold-started DCA loop", and that earlier wording was wrong**
> (corrected 2026-07-28 by §3e, which measured it). A real loop runs cluster exclusion *before* its
> first solve, and at `Σ_cluster = 0` that computes `G0_cluster_excluded = G_k_w`, the
> **coarse-grained** cluster Green's function (`cluster_exclusion.hpp`: `one_plus_S_G` is the
> identity, so `G0_excl = G`) — not the bare `G0(K)` that `DcaData::initialize()` assigns
> (`dca_data.hpp:599`). The two differ by the dispersion's variation within each coarse-graining
> patch, which on square β=8 is a **73%** relative difference in the bath at identical `Σ = 0`. So
> there are two gaps between the committed numbers and a production run, not one: a fixed
> coarse-graining offset **and** the self-consistency drift. §3e measures both; the bare bath is a
> well-defined and reproducible reference point, it is just not the loop's first iteration.

Consequences, none of them accidents:

- **`R` is a property of a single solver call at a given (β, bath) — but it is now measured to be
  the same at every call a production run makes.** §3e ran the real loop for 10 iterations at square
  β=8: the compute-weighted mean `R` over those iterations is `1.3422 ± 0.0064` against the committed
  bare-bath `1.3429 ± 0.0088`, agreeing to 0.05%. **So the committed numbers may be quoted as what a
  user experiences**, and the old "we characterize the first call only" hedge is retired. The one
  thing square cannot settle is a model with a live sign problem (TAKEAWAYS §4e).
- **The β ladder is a FIXED-BARE-BATH temperature axis, and that is a feature.** It varies β and
  nothing else. A self-consistent ladder varies β *and* `Σ(β)` together, so a monotone `R(β)` there
  could not be attributed to correlation length. Relabel, don't rebuild.
- **Do not chain rungs the way the production Tc workflow chains temperatures.** Seeding `Σ` from the
  previous temperature is a convergence accelerator: it changes how many iterations reach the fixed
  point, not the fixed point itself, and it carries `Σ` — not walker state, so it is not an RNG
  concern either. Adopting it would make rungs statistically dependent, and every trend interval we
  quote (`ΔR`, `Δρ`) rests on independent rungs. Rung independence is deliberate.

**Depth is measurements PER RANK, and every model has a floor** that must be re-established whenever
β or the model changes — the sign floor rises with β independently of the autocorrelation time.
**Measured on the β ladder, in the sign-free case: the floor rises with β through autocorrelation
alone** — 64 per rank at β=1,2 but 256 at β=4,8 on square, where ⟨sign⟩ = 1 exactly. So the
re-establish rule is not just about the sign; a model with no sign problem at all still moves its
floor with β. **A depth in "measurements" is only meaningful at a stated `sweeps-per-measurement`**
(fixed at 2 in the templates); the driver now records it and `warm_up_sweeps` in the HDF5 metadata,
and both are overridable per run. Rules that follow:

- **More ranks do not substitute for depth.** Precision on `R` comes from ranks; the sign floor does
  not care — more ranks at shallow depth is just more chances to draw a near-zero denominator.
- **Measured floors, per rank** (`floor` mode, 3 seeds/rung): square 64 at β=1,2 and 256 at β=4,8;
  **FeAs 1024 at β=1,2 (autocorrelation, ⟨s⟩≈1), rising to ≥4096 from β=3 (sign-set)**. The handover
  from autocorrelation-set to sign-set happens between β=2 and β=3 as ⟨s⟩ falls 0.96 → 0.72.
- **Chasing outlier index ≈ 1 is the wrong target and costs hours.** It would demand 16384/rank at
  FeAs β=4 and more at β=5 (~14 h of ensembles). Milestone 6's paired depth check already settled the
  question it purports to answer — same 32 seeds at 4× depth moved `R` by `+0.025 ± 0.100`, flat — so
  depth clears *bias* at 2048/rank even where the estimator is heavy-tailed. `R` is a ratio of
  across-rank variance sums, so one outlying rank inflates numerator and denominator together: heavy
  tails cost **precision**, which seeds fix, not accuracy, which they would not.
- **Quote `R` from across-seed scatter, not the rank bootstrap** — and now with the calibration
  measured rather than guessed. At 32 seeds the ratio (across-seed SD ÷ mean bootstrap SE) is
  **1.44 `[0.96, 1.79]` for FeAs** and **1.08 `[0.81, 1.27]` for square**. So the direction stands —
  the rank bootstrap is optimistic where a sign problem makes per-rank `G` heavy-tailed, well
  calibrated where it does not — but **the FeAs interval includes 1, so the earlier "1.4–1.9×" is a
  tendency, not an established factor.** The 3-seed numbers it came from were far too noisy to carry
  a factor. Tighten a number with **more base seeds**, not more ranks inside one seed.
- **Seeds beat depth for precision, and it is measured, not assumed.** At 4× depth the per-seed SD of
  `R` falls by only 1.78× `[1.04, 2.36]` (FeAs) and 1.23× `[0.71, 2.18]` (square), both short of the
  √4 = 2 that `k`× the seeds delivers for the same compute. Extra depth beyond the floor buys
  estimator *safety*, not efficiency.
- **Base seeds must be spaced ≥ `n_ranks × n_walkers` apart.** A walker's stream is
  `hash(global_id + base_seed)` with `global_id = local_id*n_ranks + proc_id`, so a run occupies the
  contiguous key range `[S, S + n_ranks*n_walkers)`. Consecutive base seeds therefore **replay each
  other's chains** and are not independent replicates — the one failure mode that makes an interval
  too narrow rather than too wide. `run_seed_ensemble.sh` enforces the spacing and
  `seed_ensemble.check_seed_spacing` re-checks it.
- **Check measured vs predicted per-orbit `r` on every new run.** They agree to 3 decimals at
  adequate depth; divergence flags a run contaminated by sign outliers. Free contamination detector.
  ⚠️ **But it does NOT detect sign heavy-tails, and it is not a floor criterion.** `r_pred` is built
  from `ρ` measured on the *same* sample, so the identity is near-algebraic and breaks only when mate
  *variances* become unequal — not when one rank's near-zero denominator makes `G` heavy-tailed. At
  FeAs β=5 it called 1024/rank clean and 16384 contaminated, which is backwards. Three instruments,
  three different questions, do not substitute one for another:
  **paired depth test → bias** (the floor criterion), **outlier index → conditioning** (largest
  per-rank `max|G|` ÷ median; this is the sign diagnostic), **per-orbit law → orbit bookkeeping**
  (membership and signs). Conflating the last two cost most of a session on 2026-07-28.

---

## Task list — in priority order

### 1. Finish statistical hardening  ▸ DONE (2026-07-27)

Closed by milestone 6. `R_ideal`/efficiency and the by-class reduction map in `reduction_map.py`,
bootstrap CIs and the paired drift test in `m_scaling.py`, and now `analysis/seed_ensemble.py` +
`run_seed_ensemble.sh` for the ensemble itself.

- **A tight FeAs `R`: done.** `3.17 ± 0.14` (3 seeds, 4.4%) → **`3.042 ± 0.049`** (32 seeds, 1.6%),
  a 2.9× tightening; the two are statistically consistent. Square moved `1.041 ± 0.003` →
  **`1.0425 ± 0.0013`**. Both at 64 ranks × 2048 measurements/rank, above each model's depth floor.
- **Independent oracle: done, and it agrees.** Variance taken across 32 *whole runs* — replicate unit
  a whole run rather than a rank, so 64× deeper per replicate and none of the per-rank machinery —
  gives `3.28 [2.98, 3.65]` (FeAs) and `1.047 [1.028, 1.090]` (square), each containing its headline.
  It shares `P` with the main path by construction; the operator itself is what rungs 1 and 2 check.
- Also established: the bootstrap calibration and seeds-vs-depth trade-off now quoted in Conventions,
  and a contamination gate (0 unusable runs in 88 runs across both models).

**One loose end, deliberately left.** `R` correlates mildly with the per-run outlier index across
FeAs seeds (Spearman +0.35, p = 0.05). The paired depth check says this is **not** a depth-induced
bias — the mean is flat to `±0.10` at 4× depth — so it reads as within-sample co-fluctuation: a seed
that happens to draw a large per-rank excursion has more symmetry-breaking noise for `P` to remove,
which raises `R` legitimately. Worth re-checking on any model whose sign problem is worse than
FeAs's, since there the same correlation could mean something else.

### 2. β ladder — test the central hypothesis  ▸ DONE (2026-07-28), and no GPU was needed

**Result: `ρ` is a property of the regime, not of the model.** Square is not intrinsically ρ-limited.
Numbers, controls and caveats: **TAKEAWAYS §3a**; notebook `04_beta_ladder.ipynb`;
`runs/beta_ladder_square.json`.

| β | **`R`** | ρ (mates) |
|---|---|---|
| 1 | 1.0414 ± 0.0015 | 0.9389 ± 0.0021 |
| 2 | 1.1406 ± 0.0029 | 0.8246 ± 0.0030 |
| 4 | 1.2122 ± 0.0065 | 0.7240 ± 0.0053 |
| 8 | **1.3429 ± 0.0088** | 0.5575 ± 0.0062 |

4×4 square, D4, 64 ranks × 2048 measurements/rank, **32 independent base seeds per rung**, disjoint
seed range per rung, **each rung at the bare bath** (Conventions) — an isolated temperature axis, not
a self-consistent sweep. Endpoint change `ΔR = +0.302 [+0.284, +0.319]`, `Δρ = −0.381 [−0.395, −0.369]`,
both monotone and resolved against across-seed scatter. Per-orbit `r` matches `m/[1+(m−1)ρ]` to
≤3e-3 at every rung. The β=1 rung independently reproduces the milestone-6 headline (1.0414 ± 0.0015
vs 1.0425 ± 0.0013, disjoint seeds) — a free cross-check on the whole pipeline.

**The GPU was not needed, and the cost estimate that said otherwise was wrong.** Square at β=8 costs
**36 s** per production run (64 ranks × 2048/rank) on CPU; the full 4-rung × 32-seed ladder is ~26
min. Cost does climb steeply in β (expansion order ~ β·U·Nc, walker algebra cubic in it) — 2 s at
β=1 to 36 s at β=8 — but from a base small enough that it does not matter for this cluster. Measure
before building: a GPU build rebuilds all of DCA for a two-file change. (The GPU option is still
there if a later task needs it: 4× RTX A5000, CUDA 12.2, `-DDCA_WITH_CUDA=ON` + `CUDA_GPU_ARCH=sm_86`;
`symm_variance_setup.hpp` already selects `linalg::GPU` under `DCA_HAVE_GPU`. Oversubscribe several
ranks per GPU, and re-validate rungs 1 & 2 before trusting new physics from it.)

**What this deliberately does NOT settle — 1: the production bath.** Every rung is iteration 1 from
`Σ=0` (Conventions). At β=8 the converged 4×4 square has strong AF correlations feeding back into the
bath; the bare problem is an easier one. Direction of the gap is a hypothesis, not a measurement:
self-consistency feeds cluster correlations into the bath, which should push mate-`ρ` *down* further
at fixed β — the same direction as the measured trend, making this ladder a conservative floor at low
T if so. Carried into task 3 as a two-point control.

**What this deliberately does NOT settle — 2: the competing sign channel.** `G = ⟨sign·M⟩/⟨sign⟩`, so a
denominator fluctuation is a global scalar giving `δG(k) = −G(k)·δs/s` with `w(k) = −G(k)` symmetric
— **sign-problem noise is a fully symmetric channel with mate-correlation exactly 1**, pushing `ρ`
*up* and `R` → 1, against the correlation-length effect measured above. Square is provably sign-free
at any β (⟨sign⟩ = 1 exactly on every rank at every rung, verified), which is *why* it cleanly
isolates the correlation-length axis — and equally why it cannot locate the crossover. **That needs a
β ladder on a model with a sign problem** (FeAs: ⟨sign⟩ ≈ 0.25 at β=5). Cost is the obstacle, not
method: FeAs is ~125 s per run at 64×2048 and rises steeply with β, so a 5-rung × 32-seed ladder is
several hours, and its depth floor is sign-set and rises with β on top of that. Carried as a
follow-on. **It is now task 3.**

**Infrastructure this task added** (all reusable for the task 3 / 4 sweeps):
- `run_symm_variance.sh` takes `BETA`, `SWEEPS_PER_MEAS`, `WARMUP` env overrides, each verified after
  substitution; `run_m_ladder.sh` and `run_seed_ensemble.sh` inherit them for free.
- The driver records `beta`, `sweeps_per_measurement`, `warm_up_sweeps` in HDF5 metadata, so the
  file stays the authority over the filename.
- `run_beta_ladder.sh <floor|ensemble>` — per-β output dirs (keeps filenames parseable) and disjoint
  per-β seed ranges.
- `analysis/beta_ladder.py` — per-rung aggregation, the trend test, and the two axis controls.

### 3. FeAs β ladder — does the β trend reproduce on a second model?  ▸ MEASURED; §3e outstanding

**Status (2026-07-28).** The ladder itself is **done and the answer is yes** — numbers in
**TAKEAWAYS §3a**, chart in `05_beta_cross_model.ipynb`, data in `runs/beta_ladder_fe_as.json`.
Depth floors were established per rung, the warm-up confound was tested and closed into a Convention,
and all five rungs ran at 32 seeds. **What remains is the bath-drift check in §3e** — everything
measured so far is at the bare bath.

| β | **`R`** | ρ (mates) | ⟨s⟩ | `w_null` |
|---|---|---|---|---|
| 1 | 1.5259 ± 0.0157 | 0.5221 | 0.998 | 0.031 |
| 2 | 1.8021 ± 0.0186 | 0.4376 | 0.961 | 0.065 |
| 3 | 2.3427 ± 0.0267 | 0.2320 | 0.723 | 0.112 |
| 4 | 2.7971 ± 0.0188 | 0.1150 | 0.378 | 0.152 |
| 5 | **2.9679 ± 0.0592** | 0.0759 | 0.148 | 0.153 |

`ΔR = +1.442 [+1.322, +1.561]`, `Δρ = −0.446 [−0.463, −0.430]`, both monotone and resolved.
**ρ falls even as ⟨s⟩ collapses to 0.148** — the perfectly-mate-correlated sign channel loses to the
correlation-length effect everywhere FeAs is simulable. Two results the ladder added beyond the
trend: `R` is **saturating** against the orbit-size ceiling (steps +0.276, +0.541, +0.454, +0.171,
with ρ still falling), and FeAs **crosses from ρ-limited to m-limited between β=2 and β=3**
(TAKEAWAYS §3). Measured floors: 1024/rank at β=1,2 (autocorrelation, ⟨s⟩≈1) rising to ≥4096 from
β=3 (sign-set). The β=5 rung reproduces the milestone-6 headline across disjoint seeds and a
different warm-up (2.968 ± 0.059 vs 3.042 ± 0.049).

⚠️ **β=5 is at the edge of measurability** — outlier index 6.4, across-seed SEM tripling
(0.019 → 0.059). Report it as the method's boundary, not as a comfortable interior point. Nothing
here locates where the sign channel finally wins; the β=6 scouting turnover was contamination.

The original plan, kept for the design rationale:

**The claim this establishes:** *`R` grows as you reach deeper into the more interesting
low-temperature physics* — and it is a general statement about the method, not a quirk of one model.
Task 2 measured it on square (`R` 1.04 → 1.34 over β=1→8, `ρ` 0.94 → 0.56). One model is an anecdote;
a **β ladder per model, plotted together, is the result** — the small-multiples/overlay chart is the
deliverable in its own right.

Second, and only available here: square is sign-free, so it could not exercise the **competing sign
channel** (`δG(k) = −G(k)·δs/s`, weight `−G(k)` symmetric → mate-correlation exactly 1, pushing `ρ`
*up* and `R` → 1). FeAs has `⟨sign⟩ ≈ 0.25` at β=5. So this ladder can show where the two effects
cross over — the regime where symmetrization **stops** paying off. Mapping that boundary honestly,
including the regimes where the method collapses, is the research output.

**Scouting is already done** (2026-07-28, 16 ranks × 512/rank — *below FeAs's floor, so these numbers
are indicative only and must not be quoted*):

| β | `R` (below floor) | ⟨s⟩ min | ⟨s⟩ max | outlier idx | cost @16×512 |
|---|---|---|---|---|---|
| 1 | 1.53 | 0.996 | 1.000 | 1.02 | 1.1 s |
| 2 | 1.52 | 0.965 | 0.992 | 1.04 | 2.0 s |
| 3 | 2.21 | 0.715 | 0.836 | 1.11 | 3.4 s |
| 4 | 2.72 | 0.383 | 0.570 | 1.15 | 6.1 s |
| 5 | 2.96 | 0.082 | 0.336 | 1.48 | 10.1 s |
| 6 | 2.45 | 0.031 | 0.191 | **2.52** | 14.3 s |

What that already tells us, and what it does not:
- **The trend looks like it reproduces**: `R` climbs 1.5 → 3.0 across β=1→5. The β=5 probe (2.96)
  lands on the pinned headline (3.042) despite being shallow and 16-rank — a good sign for the
  pipeline, not evidence for the trend.
- **The apparent turnover at β=6 is almost certainly contamination, not the sign channel.** Outlier
  index 2.52 and `⟨s⟩min = 0.031` at 512/rank is far below the sign floor. **Do not read a crossover
  off this table** — resolving whether `R` genuinely turns over is the single most interesting thing
  this task can settle, and it needs depth at β≥5, not a shallow probe. Resist the temptation.
- `ρ_generic` falls 0.19 → 0.011 over β=1→5, i.e. the correlation-length effect appears to dominate
  the sign channel at least that far.

**Design as executed** (all of this is done — kept for the rationale).
- Rungs β ∈ {1, 2, 3, 4, 5} to start; add 6 only if its floor turns out affordable. **β sets the
  ceiling here, not compute** — the sign floor rises steeply with β, and a rank whose accumulated
  sign hits zero yields `G = 0/0` and an unusable run.
- **Run `floor` mode first at every rung.** FeAs's floor is *sign-set* (≳1000/rank at β=5) and rises
  with β on top of the autocorrelation rise task 2 measured. Expect ≥4096/rank at the top rungs.
- Then `ensemble` mode, 32 base seeds per rung, at a depth above each measured floor.
- **Record `w_null` and non-null `R` at every rung** (Conventions: this is the task where they earn
  their place — `w_null` varies with model and would otherwise absorb part of any trend). Already in
  `beta_ladder.support_table`.
- **Report the intraband/interband split** — `beta_ladder.mechanism_table` does this per rung.
  Interband has no local scalar channel, so it should carry lower `ρ_generic`; whether the *β trend*
  differs between blocks is new and is where TAKEAWAYS §4c would get its temperature axis.

**Cost, as estimated then and measured since.** The five-rung × 32-seed ladder took **2 h 13 m**
(rung wall times 176 s, 458 s, 1033 s, 2179 s, 4207 s — a clean ~2× per unit β), against a ~1.5–2 h
estimate. Floor ladders added ~1 h 50 m. Original note kept: roughly 8× the probe times for 8× depth, and 16→64 ranks is near-free in wall time on a
128-core box. Estimate ~3 min/seed for all five rungs → **~1.5–2 h for the ensembles**, plus floor
ladders. Comfortable overnight; no GPU (see task 2 — measure before assuming, and §3e for the one-off that
remains).

**Gotchas specific to this run.**
- The committed 32-seed FeAs ensemble at β=5 **cannot be reused as a rung**: those files predate the
  `beta` metadata key, and `beta_ladder.build` refuses a run whose β it cannot verify. Re-run it.
- `beta_ladder.sign_channel_check` will report `sign_free = False` here. That is the *point*, not a
  failure — for FeAs, `⟨s⟩` is the second variable, tracked alongside `ρ`.
- CT-AUX drops `J`/`Jp`, so this is an Ising-Hund density-density model regardless of the input —
  describe it that way (Gotcha 2).

**Done:** the cross-model chart is `05_beta_cross_model.ipynb` — `R(β)` and ρ(β) for both models on
shared axes, two panels rather than a dual y-axis. The axes are not directly comparable (different
band count, `U`, filling, `Nc` orbits); the claim is about **direction and mechanism**, not that the
two curves coincide.

### 3e. Bath drift — does `R` hold across the iterations a production run performs?  ▸ DONE (2026-07-28)

**Answer: yes, and the bare-bath caveat is now closed.** Numbers and the full reading in
**TAKEAWAYS §4e**; data in `runs/bath_drift_square.json`; module `analysis/bath_drift.py`.

Square, β=8 (the worst case — the converged 4×4 square has the strongest AF correlations feeding back
into the bath), a real `DcaLoop` for **10 iterations × 8 independent base seeds × 64 ranks × 2048
measurements/rank**, undamped `Σ` update. Wall time 41 min + 5 min for the reference arm.

| quantity | `R` |
|---|---|
| committed bare-bath headline (32 seeds) | **1.3429 ± 0.0088** |
| compute-weighted mean over the 10 iterations a production run performs | **1.3422 ± 0.0064** |
| drift, iteration 0 → 9 | **−0.5%**, CI `[−7.4%, +6.3%]`, unresolved |

**Agreement to 0.05%**, comfortably inside the ≲5% tolerance this task fixed in advance. Per its own
decision rule: the frozen converged-bath fallback **has been deleted** and is not to be rebuilt for
any model. Do not re-run this check on square.

**What it also found, which the original scope did not anticipate — there are two gaps, not one.**
The convention said "bare bath = iteration 1 of a cold-started loop". That is false, and measurement
is what caught it: a real loop runs cluster exclusion *before* its first solve, and at `Σ_cluster = 0`
that computes `G0_cluster_excluded = G_k_w`, the **coarse-grained** cluster Green's function
(`cluster_exclusion.hpp` — `one_plus_S_G` is the identity, so `G0_excl = G`), not the bare `G0(K)`
that `DcaData::initialize()` assigns (`dca_data.hpp:599`). At β=8 the two baths differ by **73%** at
identical `Σ = 0`. So:

- **gap A — coarse-graining, a fixed offset, not drift:** `R` = 1.3258 ± 0.0100 (bare `G0`) vs
  1.3814 ± 0.0179 (loop iteration 0), paired by base seed: **+4.2%** `[+0.7%, +7.7%]`, **resolved**.
  No iteration count reveals this one — it is invisible to the experiment as originally scoped.
- **gap B — self-consistency:** the −0.5% above, unresolved.

They have **opposite signs and largely cancel**, which is *why* the production average lands back on
the committed headline. State it that way: the agreement is a partial cancellation, not evidence that
either effect is individually zero.

**Three controls that make the flat result meaningful rather than vacuous:**
- **The bath genuinely converged.** Per-iteration bath motion falls `1.1e-1 → 1.8e-2 → 3.6e-3 →
  7.5e-4`, then plateaus at `~4e-4` — and that plateau is Monte-Carlo noise in `Σ`, not further
  convergence. A flat `R` across a bath that never moved would have proved nothing.
- **No confounds moved.** `⟨sign⟩ = 1.0000` exactly at every iteration (the interacting bath does not
  introduce a sign problem on square) and `μ` stays pinned at 0 by particle-hole symmetry.
- **Rungs 1 & 2 pass on the loop path**, at both the first and the converged iteration — mean
  preservation to 2e-16, singletons at exactly `r=1`, `r` flat in ω.

⚠️ **Pairing does NOT work across DCA iterations, unlike across depths — do not reuse that design
here.** The milestone-6 paired depth test works because the deeper run reuses the same chain prefix.
Across iterations the walker is re-warmed at a changed bath every time, so by the last iteration
nothing survives from the first but the base seed: measured `corr(R_first, R_last) = −0.17`, pairing
gain **0.93×**, i.e. none. The resolved statement above therefore comes from the compute-weighted
mean (SEM 0.48%), **not** from the paired difference (±7%), whose interval is ~15× wider. Quoting the
paired interval as the drift bound would be the natural mistake.

⚠️ **`self-energy-mixing-factor` must be > 0 or this experiment silently measures nothing.**
`mix_self_energy` computes `Σ = α·Σ_new + (1−α)·Σ_cluster`, so at α=0 — which is what the
single-iteration templates carry, harmlessly, because it is irrelevant at one iteration — the measured
`Σ` is discarded and the bath never moves. `bath_drift_main.inc` refuses α ≤ 0 rather than trust the
input. The 3e templates use α=1 (DCA's own default): undamped reaches the self-consistent bath
fastest in a fixed iteration budget, so it is the *strongest* drift test; the damped 0.75–0.8 that
production inputs use walks a slower path to the same fixed point.

⚠️ **FeAs only:** the single-iteration template's misspelled `adjust-chemichal-potential` (Gotcha 11)
is inert for the single-iteration driver, which never instantiates the adjuster — but it goes **live**
under the real loop and would silently turn μ adjustment on. `fe_as_bath_drift_input.template.json`
spells it correctly.

**FeAs β=5 was deliberately NOT run.** The decision rule fixed in advance was "escalate to FeAs
(~3 h) only if square shows drift", and it does not. The `fe_as_bath_drift` binary and template exist
so that escalation is a run, not a build, should a later task want it. What square cannot settle: a
model with a live sign problem could behave differently, since `Σ` moves `⟨sign⟩` and the sign channel
is what decides where symmetrization stops paying. Square is sign-free *precisely* so that the drift
is isolated from that — the same reason it was the right first target.

**How it was built** (reusable, and the reason this cost ~1 h rather than a day):
- Two opt-in hooks on `DcaLoop` (`patches/symm-variance-dca.patch`): `pre_finalize_hook_` fires
  between `integrate()` and `finalize()` — the only gap where `local_G_k_w()` is legal, since
  `finalize` sets `averaged_` and the getter throws after — and `post_finalize_hook_` just after,
  where the finalized symmetrized mean exists. Both default empty, so stock DCA is unchanged.
  Hooks rather than a subclass because `execute()` calls `solve_cluster_problem` non-virtually and
  its body touches several private members.
- `bath_drift_main.inc` writes **one HDF5 per DCA iteration in exactly the existing schema**, which is
  the whole trick: `Run`, `m_scaling` and `seed_ensemble` read a drift iteration with no changes, and
  the validation rungs apply per iteration for free. It adds keys; it changes none.
- It also writes **the bath itself** (`G0_k_w_cluster_excluded`, plus bare `G0` and `Σ`), so
  convergence is measured rather than inferred from `R` — which is how gap A was found at all.
- `run_bath_drift.sh <model> <ranks> <meas/rank> <iterations> <seeds> <outroot> [seed0] [stride]`,
  then `analysis/bath_drift.py <root> --ref <ref_dir> --out runs/bath_drift_<model>.json`. The
  `--ref` arm is a matched-seed `run_seed_ensemble.sh` run, which is what makes gap A paired.



### 4. Broaden model coverage — vary ONE axis at a time  ▸ carries the headline claim

**The claim this task establishes:** symmetrization pays off most where the physics is most
interesting — multi-orbital models, and *within* them the interband components that carry the
distinctive physics. See TAKEAWAYS §4c.

**The β ladders now point at this task with a measurement, not a hunch** (2026-07-28). Which ceiling
binds *moves with the regime*: square stays ρ-limited at every β it was run (`1/ρ` only reaches 1.79
at β=8, under its ~3.0 m-ceiling), while **FeAs crosses to m-limited between β=2 and β=3** and ends at
`1/ρ = 13.2` against a ~3.7 m-ceiling. So FeAs at β≥3 has noise structure supporting a 13× reduction
and a 4×4 geometry that can only cash 3.7× of it — **orbit size, not colder physics, is what buys more
there.** Napkin ceiling for a realistic best case is `R` ≈ 5–10, not 20+, because `r → min(m, 1/ρ)`
and the large orbits are variance-poor: FeAs's two m=8 orbits carry only **6%** of raw variance, and
square 4×4 has no m=8 orbit at all (orbit size is `|G|/|stabilizer|`, and every k-point on a 4×4 mesh
has a nontrivial stabilizer — the first free orbit-8 k-point needs 8×8). Reaching a higher ceiling
means the variance has to *live* in large orbits, which is what this task should test first.

**Four runs turn that framing into a defensible claim:**

| run | establishes |
|---|---|
| single-band square at FeAs's β | **the keystone** — de-confounds temperature from band count. Task 2 already brackets it: square reaches `R = 1.34` at β=8 against FeAs's 3.04 at β=5, so **temperature alone does not account for the gap** — but run β=5 itself to close it cleanly |
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
| β | `ρ` | task 2 — **done** on square (sign-free); `ρ` 0.94→0.56 over β=1→8 |
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

**GPU: decide it here, by benchmark, not on the β axis** (assessed 2026-07-28). Not worth it at 4×4:
matrices are small (expansion order ~ β·U·Nc), and the design — 64 MPI ranks = 64 independent samples
— already maps perfectly onto 128 CPU cores at rank granularity, whereas 4 GPUs would serialize 16
ranks each. The axis that could change this is **cluster size**: 4×4 → 8×8 quadruples the matrix
dimension, ~64× the algebra. So if this task commits to 8×8, build once and benchmark a single 8×8
FeAs run CPU vs GPU before committing the sweep; under ~3×, skip it. Two costs to budget beyond the
build: re-validating rungs 1 & 2 on the GPU binary, and a **confounding risk** — if square runs on CPU
and 8×8 on GPU, the code path is silently confounded with the axis, so keep one reference point run on
both. Cheaper first move that nobody has measured: the thread config. Runs are currently 64 ranks × (2
walkers + 2 accumulators) = 256 threads on 128 cores, 2× oversubscribed; 1+1 at 128 ranks may be
faster for free. Hardware if needed: 4× RTX A5000 (24 GB, idle), CUDA 12.2, `-DDCA_WITH_CUDA=ON`,
`CUDA_GPU_ARCH=sm_86`; `symm_variance_setup.hpp` already selects `linalg::GPU` under `DCA_HAVE_GPU`.


#### Progress (2026-07-28): infrastructure + gates 0–2 done, ensembles outstanding

**Scope, chosen with the advisor:** four measured points at fixed **β=5** — `square_b5_c4` (keystone),
`threeband_b5_c4`, `square_b5_c8` (cluster axis), and `kagome_b5_c4` — reusing the committed FeAs β=5
rung as a reference rather than re-running it. `twoband_Cu` deliberately dropped: threeband's d–p
block is a *within-model, single-axis* inequivalent-orbital control, which is better evidence than a
whole second model differing on dispersions and filling too. **Follow-up task 4b** (below) carries the
"push β higher" question.

**What is built and verified** (`symm_variance/`, `analysis/`, manifest in
`runs/model_sweep_manifest.json`):
- `threeband_symm_variance` and `kagome_symm_variance` targets + templates + CMake tokens.
- `CLUSTER` / `NW` / `MU` / `EP_P` / `U_PP` env overrides on `run_symm_variance.sh`, each verified by
  `grep -q` after substitution. **`CLUSTER` must use `sed -z`** — square's template writes `"cluster"`
  over two lines while FeAs's is on one, so a line-oriented sed silently misses square (new Gotcha).
- Driver records `metadata/chemical_potential` and `functions/H_DCA` (both additive, both optional on
  read). `H_DCA` buys the **analytic-`G0` oracle**.
- `analysis/model_sweep.py` (manifest verification, `band_equivalence`, `band_pair_classes`,
  `band_permutation_content`, `square_mesh` guard, `axis_pairs`, tables, `plan` command) and
  `analysis/scout_point.py` (the gate-1 readout).
- `validate.py` **rung 1d** (`P²=P`, `P=Pᵀ`, `trace(P)` = non-null orbit count) and the oracle.

**Regression, run before the new code was trusted on new data:** `model_sweep` reproduces TAKEAWAYS
§4c exactly on the committed runs — `n_offblock` 104 (FeAs) / 0 (square), orbits mixing blocks 10-of-10
/ 0-of-6, band-equivalence classes `{0,1}` / `{0}` — and the harmonic reconstruction is exact
(≤4e-16) in all three partition modes. Rung 1d is **exactly 0.0** on both, with trace 10 and 6.

**Three findings that change the task's framing:**

1. **Rung 2 PASSES on threeband (1.2e-15) — the expected failure did not happen.** DCA's
   characterization test marks `ThreebandD4` with `expectedFailingReps = {k_iw, r_iw, r_tau}`, so the
   plan was built around production's multi-band imposition disagreeing with our `P`. At nk=16 through
   this driver it does not: production's `Symmetrize::execute` and our derive-path `P` agree to 1e-15,
   **and** the analytic-`G0` oracle passes at 1.4e-16, so `P` is independently correct. **Threeband
   numbers therefore need no imposition caveat** — they are both "what the symmetry buys" and "what
   DCA delivers". The prepared caveat wording is not needed. Do not delete rung 1d or the oracle: they
   are what makes this statement checkable rather than lucky.

2. **`U_pp = 0` makes threeband measure nothing, and this is not a matter of degree.** At the repo
   test's `U_pp = 0` the `|H_int| > 1e-3` filter leaves the p orbitals with no CT-AUX vertices, and the
   consequence is that the p-block `G` comes out **deterministic** — across-rank σ ≈ 1e-17 against
   |G| ≈ 1.6 — while the d–p block is identically zero in the measured `G`. Every class except d–d
   then carries ~1e-30 of the raw variance and `R` collapses to the single-band answer (1.03–1.12
   against square's 1.04). **The operating point is `U_pp = 2`, `μ = 3`** (μ at the p site energy),
   measured to give 48% filling and an equivalent-orbital variance share of **0.708 vs 0.000**.
   `U_pp = 8` diverges outright (band occupancy 28.7, `R` = nan). Symmetry is untouched by any of
   this — the derive path gates on `H0` alone, so `P` and `n_ops` are bit-identical at any `U_pp`.
   ⚠️ Never pair `U_pp = 0` with `interacting-orbitals: [1,2]`: empty `correlated_orbitals` is a
   **segfault**, not an error (new Gotcha).

3. **Kagome is BLOCKED by a DCA-side segfault — out of scope, not broken.** It was the only model with
   three symmetry-equivalent orbitals and the only route to `|G| = 12`, so the diagnosis is recorded in
   full in the manifest's `blocked_points`. Summary: the target *builds* (`KagomeHubbard<D6>` passes
   `CanDeriveSymmetry`), but every run segfaults inside DCA's own
   `SymmetrizeSingleParticleFunction::executeCluster`, called from `DcaData::initialize_G0()` — before
   any driver code runs (gdb backtrace confirmed). Kagome's `flavors() = {0,1,2}` are all distinct, so
   `set_symmetry_matrices` records band images of `-1` which are then dereferenced. **Declaring D4
   instead of D6 does not help**, so it is the derive/record path, not the declared group. DCA's own
   test reaches 12 derived ops only by calling `verifiedSymmetryOps(H0)` directly on a `no_symmetry<2>`
   instantiation, which never exercises this path. Fixing it means patching DCA's symmetrization,
   which this task scopes out. The `.cpp` and template are kept so the attempt is not repeated.

**Gate-1 scouting results** (16 ranks × 512/rank, β=5, `NW=64` — **indicative only, below any floor**):

| point | wall | `n_ops` | orbit sizes (non-null) | occupancy | oracle | `R` (scout) |
|---|---|---|---|---|---|---|
| `square_b5_c4` | 2 s | 8 | `{1:2, 2:1, 4:3}` | 1.00 of 2 | — | — |
| `threeband_b5_c4` | 15 s | 8 | `{1:2, 2:10, 4:15, 8:4}` | 2.88 of 6 | 1.4e-16 ✓ | 3.22 |
| `square_b5_c8` | 50 s | 8 | **`{1:2, 2:1, 4:9, 8:3}`** | 1.00 of 2 | 7.8e-16 ✓ | 1.51 |
| `fe_as` (calibration) | 9 s | 8 | — | — | — | — |

- **The 8×8 point already confirms TAKEAWAYS §6 geometrically:** three **free** `m=8` orbits appear at
  8×8 that a 4×4 mesh cannot have (`{1:2, 2:1, 4:3}` → `{1:2, 2:1, 4:9, 8:3}`). Whether `ρ` falls with
  cluster size still needs the ensemble.
- **threeband's band-equivalence classes come out `[[0], [1,2]]`** — d inequivalent, p_x ~ p_y
  equivalent — with exactly **two** realized band permutations (identity and the p_x↔p_y
  transposition, **no d↔p**). That is the inequivalent-orbital control confirmed structurally, before
  any statistics.

**Cost calibration, measured not guessed.** `t(64 ranks × 2048/rank) / t(16 ranks × 512/rank) = 13.9`,
pinned against FeAs's known 125 s at 64×2048 (its scout took 9 s). A 32-seed ensemble is 32× that.
Budgets follow: square β=5 **~15 min**, threeband **~1.9 h**, square 8×8 **6.2 h at 32 seeds** — over
the 4 h budget rule, so the manifest cuts it to **16 seeds (~3.1 h)**.

**Gate 3 (depth floors) — measured 2026-07-28, and it forced a Convention change on threeband.**

| point | floor | evidence |
|---|---|---|
| `square_b5_c4` | **≤512/rank** | every step flat in all 3 seeds; ⟨s⟩ = 1.0000 exactly; `R` ≈ 1.25 |
| `square_b5_c8` | **≤512/rank** | every step flat in all 3 seeds; ⟨s⟩ = 1.0000; outlier 1.02 |
| `threeband_b5_c4` | 2048/rank **at warm-up 8000** | see the thermalization finding below |

**The cluster-size question is ANSWERED: `ρ` falls with `Nc`** (3 seeds at m=8192, ≫ the ≤512 floor,
sign-free, so this is well controlled even before the ensembles pin it):

| | 4×4 (nk=16) | 8×8 (nk=64) |
|---|---|---|
| **`R`** | 1.2506 ± 0.0013 | **1.4014 ± 0.0026** |
| **`ρ` (mates)** | 0.6777 ± 0.0013 | **0.5777 ± 0.0012** |
| `1/ρ` | 1.48 | 1.73 |
| `R_ideal` (m-ceiling) | 3.015 | 4.611 |
| efficiency | 41.5% | **30.4%** |
| orbit sizes | `{1:2, 2:1, 4:3}` | `{1:2, 2:1, 4:9, 8:3}` |

So the roadmap's open question — *does `ρ` fall with cluster size?*, which decides whether the
m-ceiling is reachable in practice — gets **yes**: `Δρ = −0.100`, and `R` rises 12%.

**But the shape of the answer matters more than the sign, and it cuts against the obvious reading.**
Both ceilings rise, and the m-ceiling rises *faster*: 3.02 → 4.61 (the three **free** `m=8` orbits that
only exist at 8×8, TAKEAWAYS §6) against `1/ρ` 1.48 → 1.73. Square stays **ρ-limited at both sizes**
(1.48 < 3.02 and 1.73 < 4.61), so **efficiency falls, 41.5% → 30.4%**. The bigger cluster does help —
but it helps through the *noise structure*, not through the larger orbits it unlocks, because those
orbits cannot be cashed while `ρ` still binds. Stated for a practitioner: *on square at β=5, buying a
bigger cluster buys you a better `ρ`, and the extra orbit structure you also bought sits idle.*
Whether a model that is already m-limited (FeAs at β≥3) converts the 8×8 orbit structure into `R` is
the obvious follow-on and is **not** answered here.

Square's β=5 `R` ≈ **1.25** sits between the committed β=4 (1.2122) and β=8 (1.3429) rungs — a free
consistency check on the whole pipeline at a temperature it had never been run at.

⚠️ **A "depth floor" can be a THERMALIZATION artifact, and on threeband it was.** The first threeband
ladder looked like an unconverged floor: `R` = 2.9653 ± 0.0055 at m=2048 rising to 3.3218 ± 0.0089 at
m=8192, with the 2048→8192 step resolved in **all three** seeds. Two things said this was not the
sign problem — ⟨s⟩ = 0.60 and the outlier index *fell* to **1.06**, cleaner conditioning than FeAs ever
had — and `R` converged from **below**, opposite to the upward bias square and FeAs show below their
floors (TAKEAWAYS §1a). Raising **warm-up 200 → 2000 at fixed m=2048** moves `R` by **+0.407**,
reproducing the +0.357 that 4× depth bought and landing within **0.05** of the deep run
(3.3724 ± 0.0203 vs 3.3218 ± 0.0089). The chain was simply not thermalized; extra depth was diluting
an unthermalized prefix rather than curing it.

- **This overrides "warm-up 200 everywhere" for threeband only.** The Convention was established on
  square and FeAs; threeband's expansion order (`U_dd = 8` across 3 bands) is far larger. Square at
  the *same* β shows every depth step flat at warm-up 200, which is the control that makes this
  model-specific rather than a general flaw.
- **This is a mechanism, not just an observation** — unlike the FeAs warm-up entry in Conventions
  ("flat then a step is the wrong shape for smooth thermalization"). Here raising warm-up at fixed
  depth reproduces the effect of raising depth at fixed warm-up, which is precisely what an
  unthermalized prefix predicts.
- **Diagnostic rule worth keeping:** if `R` rises with depth while the outlier index *falls*, suspect
  thermalization before buying more depth. Testing warm-up at fixed depth costs a fraction of the next
  depth rung — here 3 runs (~17 min) against a 32768/rank ladder that would have been ~2.5 h and, at
  ensemble scale, 26 h and unaffordable.

**What remains: gate 4.** The 32-seed ensembles (`model_sweep.py plan --mode ensemble`), then
`model_sweep.py build`, notebook 06, and the TAKEAWAYS edits. All ladders are skip-if-exists and
resumable, so re-running a planned command is always safe. **The build to use is
`/home/tsax10/dca/build_task4`** (configured against this worktree); `build_symm` still targets the
main checkout and does not have the new targets.

### 4b. Push β higher on whichever model maximizes `R`  ▸ follow-up to task 4

Decided with the advisor 2026-07-28: run the sweep at a common β=5 first, then push one model colder.
**Which model is decided by the sweep's own diagnostic, not guessed now** — `model_sweep.binding_table`
reports `1/ρ` against each point's m-ceiling. Pick the model that is still **ρ-limited** with the
**largest m-ceiling**: there colder physics still pays *and* there is headroom to cash it, whereas a
model already m-limited (FeAs by β=3) gains little from more β. Costs a β ladder plus per-rung floor
ladders on one model. Note kagome — the likely winner on m-ceiling — is blocked above, so on current
evidence this is threeband or the 8×8 square.

#### Starting this task cold — what a fresh session needs (written 2026-07-28, at the close of §3e)

**You can measure at the bare bath and stop there.** §3e settled that `R` does not drift across the
iterations a production run performs, so this task needs **no `DcaLoop` runs at all** — the ordinary
`*_symm_variance` single-iteration driver is sufficient and its numbers transfer. Do not reach for
`*_bath_drift` here.

**Adding a model is three files and a rebuild** (verified this session by adding two targets):

1. `symm_variance/<model>_symm_variance.cpp` — `#include <cmath>` **first** (Gotcha 8), then the
   lattice header, then `#define SYMM_VARIANCE_LATTICE <type>`, `#define SYMM_VARIANCE_MODEL_LABEL
   "<label>"`, then `#include "symm_variance_main.inc"`. Copy `fe_as_symm_variance.cpp`.
2. `symm_variance/<model>_input.template.json` — `__SEED__` and `__MEASUREMENTS__` placeholders,
   `interacting-orbitals` listing **every** band, `error-computation-type: NONE` (Gotcha 6), and
   warm-up 200 / sweeps-per-measurement 2 per Conventions.
3. Add the target name to the `foreach` list in `symm_variance/CMakeLists.txt`, then rebuild.

**Band-permuting symmetry is DERIVED from `H0`, not declared — so a new multi-orbital model needs no
symmetry code.** This is the single most useful fact for this task and it is easy to get wrong:
`Lattice::orbitalPermutations()` returns an **empty** list on *every* stock lattice **including
FeAs** (it inherits `bilayer_lattice`'s), so that is not where FeAs's 8 ops come from. The real
mechanism is `deriveOrbitalOpForOp` (`solve_orbital_op_signs.hpp`), which *solves* for the orbital
operator `U_S` from `H0` for each candidate spatial op; `verifiedSymmetryOps` keeps the ops that pass
and silently drops the rest. FeAs declares {identity, C4} and derives 8 that way. Two preconditions
before choosing a model:

- **The derive candidate pool is `holohedry_pool_2D`** — 2D lattices only on this path.
- **`U_S` must come out a SIGNED PERMUTATION (±1) or the op is silently dropped**, shrinking the
  derived group with no error — `set_symmetry_matrices` rejects non-±1 fold phases, and
  `orbit_table.hpp`'s `static_assert` separately pins the whole serializer to the derive-authoritative
  path (`CanDeriveSymmetry`). So a 3-fold model (`Kagome_hubbard`, `triangular_lattice`) is a real
  risk: a C3 that mixes
  three orbitals with fractional coefficients fails the gate. **Check the derived op count (`n_ops`
  in the HDF5 metadata) on the very first run of any new model** — if it collapses toward 1, this is
  why, and the model is out of scope rather than broken.

**Candidate models actually in the tree**, band counts verified from `BANDS`:

| lattice | `nb` | note |
|---|---|---|
| `threeband_hubbard` | 3 | **the recommended 3-orbital run.** Emery d-p model (`ep_d`, `ep_p`, `t_pd`, `t_pp`, `U_dd`, `U_pp`) — the file's "two-orbital" header comment is stale, `BANDS = 3`. |
| `Kagome_hubbard` | 3 | 3-fold — check the ±1 gate above before committing |
| `fourband_lattice`, `La3Ni2O7_bilayer`, `square_plaquette_hubbard` | 4 | larger index space |
| `twoband_lattice`, `twoband_Cu`, `twoband_chain`, `bilayer_lattice` | 2 | `bilayer_lattice` is FeAs's base class |

**`threeband_hubbard` may collapse two of the four planned runs into one.** Its d orbital is
inequivalent to its two p orbitals, while the two p orbitals are equivalent to each other — so a
single run should show the band-permutation benefit *within* the p-p block and **none** in the d-p
block. That is the "benefit should vanish for inequivalent orbitals" control and the `nb=3` trend
point in one measurement, with the entry-class split (`reduction_map.py`) doing the separating.
Verify the orbit table actually has that structure before relying on it.

**Environment, so nothing is rediscovered:**
- Build: `/home/tsax10/dca/build_symm` (`BIN_DIR` default in `run_symm_variance.sh`). ⚠️
  `/home/tsax10/dca/build_bath_drift` is **orphaned** — it was configured against a §3e worktree that
  no longer exists, so its compiled binaries still run but any rebuild fails until it is reconfigured
  with `-S /home/tsax10/dca/analysis/symm_variance`. Use `build_symm` and ignore it, or delete it.
- ⚠️ **The DCA source edits are uncommitted in `../source/DCA`'s working tree** and the build depends
  on them. If that tree is ever reset, reapply `patches/symm-variance-dca.patch` (see the warning in
  *Where things live*) — this now includes the `dca_loop.hpp` hooks as well as the `ctaux` changes.
- Python: `/home/tsax10/dca/analysis/.venv/bin/python` (no scipy — `seed_ensemble` tabulates its own
  t-quantiles). MPI: `/home/tsax10/conda/envs/qe/bin/mpirun`.
- **Shared 128-core box — check load before launching** (`uptime`), and never time a run on a loaded
  machine. Runs are 64 ranks × 4 threads = 2× oversubscribed by design; that is the established
  config, not a mistake.
- Point run output at `/home/tsax10/dca/scratch/` and commit only the summary JSON — a 64-rank run is
  ~17-34 MB.

**Order of operations for any new model**, each step gating the next: one run → check `n_ops` and the
orbit table → validation rungs 1 & 2 (`analysis/validate.py`) → `floor` mode to establish the depth
floor (it must be re-established per model, Conventions) → only then a 32-seed ensemble. Skipping the
floor is what produced the false results catalogued in Conventions.


### 5. Migration scatter figure

The plan's signature visualization (§4.3), still to build. For a chosen orbit at fixed ω, plot mates
in the complex plane: raw cloud of `m×P` points (signs applied first so odd mates align), then arrows
to their sign-aware orbit average. The cloud contracts toward its own centroid by `1/√r` while the
**centroid does not move** — variance reduction, mean preservation, and the ρ mechanism all legible
at once. Singletons show no migration: the visual null control.

Small-multiples grid, one panel per orbit. Ideal contrast: square (ρ≈0.94, barely moves) beside FeAs
`m=8` (ρ≈0.08, collapses) — makes ρ-limited vs m-limited visual. Add as **`04_migration_scatter.ipynb`**
(03 is taken by the M-scaling notebook) via `analysis/build_notebooks.py`.

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

**And precision is bought with seeds, not depth — measured, in Conventions.** Above the floor, 4×
depth narrows the per-seed spread of `R` by less than the √4 that 4× the seeds gives for the same
compute. Depth clears the floor; seeds do the rest.

---

## Where things live

This repo (`DCA-variance-analysis`) is checked out at `/home/tsax10/dca/analysis`. It holds **both**
the current project and the superseded prototype:

| path | what |
|---|---|
| `symm_variance/` | **current work** — driver, orbit-table serializer, inputs, `run_symm_variance.sh`, `run_m_ladder.sh`, `run_seed_ensemble.sh`, `run_sign_sweep.sh`, `run_beta_ladder.sh`, `run_bath_drift.sh` |
| `symm_variance/bath_drift_main.inc` | the §3e driver: runs the **real `DcaLoop`** and dumps raw per-rank `G` at every iteration, one HDF5 per iteration in the *same schema* as the single-iteration driver (so the whole analysis tier reads it unchanged) |
| `symm_variance/analysis/` | numpy libs + notebooks (`01_validation_ladder`, `02_noise_mechanism`, `03_m_scaling`, `04_beta_ladder`, `05_beta_cross_model`) |
| `symm_variance/runs/` | run data the notebooks read, plus `m_scaling_summary.json`, `seed_ensemble_{square,fe_as}.json`, `beta_ladder_{square,fe_as}.json` and `bath_drift_square.json` |
| `patches/symm-variance-dca.patch` | **our DCA source edits** — see warning below |
| `ROADMAP.md`, `TAKEAWAYS.md`, `symm-variance-plan.md` | this file, headline claims, design doc |
| `variance_demo/`, `notebooks/`, `SETUP.md`, `variance-demo-plan.md` | **superseded prototype** — reference only, be skeptical |
| `.venv/` | python env (gitignored). Its `jupyter` launcher has a stale shebang — use `python -m nbconvert` |

Outside the repo: `../source/DCA` (the DCA checkout, a separate git repo), `../build_symm` (build
tree; `../build` is the stale prototype build).

> ⚠️ **The DCA source edits live in a different repo and are uncommitted there.**
> `source/DCA` is its own git checkout; our changes to `ctaux_cluster_solver.hpp`
> (`local_G_k_w(symmetrize)`, a latent-bug fix, `local_accumulated_sign()`) and to `dca_loop.hpp`
> (the two opt-in per-iteration hooks §3e needs — `pre_finalize_hook_` fires in the only gap where a
> raw per-rank sample is legal, `post_finalize_hook_` where the finalized mean exists) sit in its
> working tree.
> `patches/symm-variance-dca.patch` is the authoritative copy — reapply with
> `git -C ../source/DCA apply patches/symm-variance-dca.patch` if that tree is ever reset.

**Build:** standalone CMake pulling DCA as a subdirectory; deps from `/home/tsax10/conda/envs/qe`
(OpenMPI 4.1.6, HDF5 1.14.3, FFTW, LAPACK), system g++ 13.3. Only the two driver objects are ours —
everything under `build_symm/dca/` is stock DCA.
**Run:** `symm_variance/run_symm_variance.sh <square|fe_as> <n_ranks> <measurements_per_rank> [seed] [outdir]`
(caps BLAS threads; shared 128-core box — check load first). Depth sweeps:
`symm_variance/run_m_ladder.sh <model> <n_ranks> <outroot> <m,...> <seed,...>`. Precision:
`symm_variance/run_seed_ensemble.sh <model> <n_ranks> <m_per_rank> <n_seeds> <outroot> [seed0] [stride]`,
then `analysis/seed_ensemble.py <dir> --deep <deeper_dir> --out runs/seed_ensemble_<model>.json`.
One HDF5 per run, ~30 MB at 64 ranks (~34 MB at m=8192), so point these at scratch and commit only
the summary JSON. **Cost is wildly model-dependent** — at 64 ranks × 2048/rank a square run is ~1 s
and a FeAs run ~125 s, because expansion order goes as β·U·Nc and the walker algebra is cubic in it.
The 32-seed FeAs ensemble is ~70 min; the square one is under a minute.
β ladder: `symm_variance/run_beta_ladder.sh <floor|ensemble> <model> <n_ranks> <outroot> <m|m,...>
<seeds|n_seeds> <beta...>`, then `analysis/beta_ladder.py <root> --out runs/beta_ladder_<model>.json`.
Run `floor` mode first — the depth floor moves with β — and only then `ensemble` above it. Square is
cheap (~26 min for 4 rungs × 32 seeds at 2048/rank, β=8 dominating at 36 s/run); the FeAs 5-rung ×
32-seed ladder is **2 h 13 m**, plus ~1 h 50 m of floor ladders. **No GPU needed** (§4 for when to
revisit). `WARMUP`/`SWEEPS_PER_MEAS` are env overrides on `run_symm_variance.sh` and are inherited by
every ladder script; Conventions fix them at 200 and 2.
Bath drift (§3e): `symm_variance/run_bath_drift.sh <model> <n_ranks> <m_per_rank> <n_iterations>
<n_seeds> <outroot> [seed0] [stride]`, then `analysis/bath_drift.py <root> --ref <ref_dir> --out
runs/bath_drift_<model>.json`, where `<ref_dir>` is a **matched-seed** `run_seed_ensemble.sh` run —
that is what makes the coarse-graining gap a paired comparison. Needs its own build (the `*_bath_drift`
targets); square β=8 at 10 iterations × 8 seeds is **41 min** plus 5 min for the reference arm, and
each iteration file is ~17 MB, so point it at scratch and commit only the summary JSON.
**Notebooks:** Jupyter kernel `symm-variance (py)`; regenerate via
`symm_variance/analysis/build_notebooks.py`, which rewrites **all five** and clears outputs, so
re-execute all five after any edit — and check `execution_count` per cell afterwards, since nbconvert
can exit 0 having executed nothing (Gotcha 12).

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
12. **`build_notebooks.py` escapes backslashes TWICE, and the second stage is easy to miss.** Cell
    source is written through a non-raw Python `"""` string, so LaTeX in a `CODE(...)` block needs
    `\\rho`. But that yields `\rho` *in the generated cell*, which is eaten a second time unless the
    literal there is also raw — so a bare `f'...$1/\\rho$...'` still breaks while `rf'...'` works.
    Both forms produce `$1/ho$` and a matplotlib mathtext `ParseException` **at execute time, not
    build time**, and nbconvert can report exit 0 while a notebook executed zero cells. Verify
    execution by counting `execution_count` per cell, not by trusting the exit code. MD cells are
    `MD(r"""...""")` and safe.
13. **Consecutive base seeds are NOT independent runs.** Walker streams are
    `hash(global_id + base_seed)` with `global_id = local_id*n_ranks + proc_id`
    (`src/math/random/random_utils.cpp`), so base seeds `S` and `S+1` at 64 ranks share 63/64 of
    their walker streams. Nothing warns; the runs simply look like agreeing replicates and every
    interval built from them is too narrow. Space base seeds by at least `n_ranks × n_walkers`.

14. **`CanDeriveSymmetry` rejects a `no_symmetry` declaration, and it fails at COMPILE time.**
    `derive_point_group.hpp` gates on `!is_no_symmetry<Lattice::DCA_point_group>` and
    `orbit_table.hpp:52` static_asserts on `CanDeriveSymmetry`, so a model must be declared with
    `D4`/`D6` even where DCA's own tests instantiate it with `no_symmetry<2>`. The declared group is
    only an on/off switch on this path — `deriveAndPopulateRecord` installs `holohedry_pool_2D`
    regardless — so declaring the real holohedry costs nothing and turns the derive path on.
15. **An empty `correlated_orbitals` is a SEGFAULT, not an error.** `general_interaction.hpp:79` does
    `rng() * correlated_orbitals.size()` and then indexes it unguarded; the `assert` that would catch
    it is commented out. It is reached by naming only zero-`U` orbitals in `interacting-orbitals` —
    e.g. threeband with `U_pp = 0` and `interacting-orbitals: [1,2]`.
16. **`square_input.template.json` writes `"cluster"` over two lines while `fe_as`'s is on one.** A
    line-oriented `sed -i -E` matches nothing on square and still exits 0, so the `CLUSTER` override
    uses `sed -z` and verifies through `tr -d '[:space:]' | grep -q`. Only the verify catches it.
17. **A model can pass every structural gate and still measure nothing.** Check the per-class share of
    RAW variance (`scout_point.py`), not just `n_ops` and the orbit table. Threeband at `U_pp = 0` has
    the right 8 ops, the right band-equivalence classes and the right orbit sizes, yet its p-block `G`
    is deterministic to 1e-17 and every class but d–d carries ~1e-30 of the variance — so `R` is
    silently just the single-band answer. `w` is a gate, not a report.
