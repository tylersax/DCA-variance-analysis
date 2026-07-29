# Headline takeaways — symmetrization variance reduction

Running list of presentation-worthy points. Each entry: the claim, the evidence, and the caveat that
keeps it honest. Add to this as results land; prune anything that doesn't survive scrutiny.

---

## Notation — three levels of the same ratio

All three are `Var(G) / Var(Sym G)`; they differ only in what they are summed over. Do not rely on
capitalization alone to disambiguate in prose — name the scope.

| symbol | scope | definition |
|---|---|---|
| `r` | **one orbit** | `r = m / [1 + (m-1) rho]` — the per-orbit reduction |
| `R_C` | **one class of entries** | variance-weighted ratio within a class (interband, `m=8`, ...) |
| `R` | **all entries** | the headline: `sum_x Var(G[x]) / sum_x Var(Sym G[x])` |

They are related exactly by `R = 1 / sum_C (w_C / R_C)` with `w_C` the class share of raw variance —
a **harmonic** mean, so `R` is pinned by the worst well-populated class and always sits below the
best orbits. FeAs, concretely: individual `m=8` orbits reach `r = 5.12` and `4.56`; the `m=8` *class*
gives `R_C = 4.86`; the aggregate is `R = 3.04`, dragged down by the intraband class at `R_C = 2.53`
carrying 82% of the variance.

**`R` is the single headline number, and it is summed over ALL entries (full support).** That is the
reduction a user actually experiences: run unsymmetrized and you genuinely carry every entry,
including the symmetry-forbidden ones, and their noise propagates into everything downstream. One
number, no qualifier.

**Every `R` in this document is measured at the bare bath** — the first iteration of a cold-started
DCA loop, `Sigma = 0` — i.e. one solver call, not a converged self-consistent run. See §4e; it is a
scope note, not a caveat on the estimator.

Two internal quantities — useful, **not for external reporting**:
- **Non-null `R`** drops the symmetry-forced-zero entries. Exactly `R_full = R_non-null/(1 - w_null)`
  with `w_null` the forced-null share of raw variance, so it adds precisely one fact: `w_null`. Use it
  when the *structure* of the noise is the point (§4b), especially across models, where `w_null`
  varies (FeAs 38% of entries; square none) and would otherwise absorb part of a cross-model trend.
- `R_ideal = sum(Var)/sum(Var/m)` and `efficiency = R/R_ideal` — the `rho = 0` ceiling and the share
  of it captured. Definable only on the non-null support. A development diagnostic.

---

## 1. The benefit does not decay with scale — it holds all the way to supercomputer runs

`R = Var(G)/Var(P·G)` is a ratio of variances of the **same linear functionals of the same noise
distribution**. Scale the total sampling `N` (more measurements per rank, more ranks, both) and
`Var(Ĝ_N) = Σ/N` while `Var(P·Ĝ_N) = PΣPᵀ/N` — the `1/N` cancels. **`R` depends only on the noise
covariance *structure*, never on how much compute you spend.**

Framing: symmetrization is worth a constant factor of `R`× more measurements, *forever*. It is not a
small-run trick that washes out once you have enough statistics — the multiplier at 10^9 measurements
on a leadership-class machine is the same as at 10^4 on a workstation.

**Practical corollary:** precision on `R` is governed by the number of independent samples (ranks),
not by run depth. Cheap to measure, and the measurement transfers to any production scale.

*Caveat — now measured, not assumed.* The cancellation requires each rank's sample to be in the
CLT/asymptotic regime. The M-scaling control (64 ranks, 3 seeds, `m ∈ {4…4096}` per rank) confirms
`R` is flat in depth **above a model-dependent floor**, and finds that the floor is set by different
mechanisms in the two models — see §1a. Below the floor `R` is biased, and the bias is **upward**, so
a too-shallow run flatters symmetrization rather than penalizing it.

## 1a. The depth floor is real, and a sign problem sets it — not the autocorrelation time

Two distinct ways a per-rank sample can fail to be asymptotic, one found in each model:

- **square/D4 at β=1 — autocorrelation-limited, floor ≈ 64 measurements/rank.** `R` measures ≈1.11 at
  `m=4` against ≈1.04 asymptotically; every `4→16` step drifts significantly, and from `m ≳ 64` every
  paired step is flat across all three seeds. *Why upward:* a single CT-AUX configuration is not
  symmetric. A rank that has averaged only a handful of them still carries raw configuration-level
  asymmetry — pure symmetry-**breaking** noise, exactly what `P` removes in full. It self-averages
  away *within* each rank as `m` grows, and the mechanism diagnostics track it: mate-`ρ` on the `m=2`
  orbits climbs 0.65 → 0.85, within-shell real-space correlation 0.74 → 0.90, then both plateau.
- **FeAs at β=5 — sign-limited, floor ≳ 1000 measurements/rank.** `G_i = ⟨sign·M⟩_i/⟨sign⟩_i` is a
  ratio with a **stochastic denominator**, and FeAs has `⟨sign⟩ ≈ 0.25`. At `m ≤ 64` several of 64
  ranks accumulate a sign of **exactly zero**, making `G_i` literally `0/0`. Even at `m=256`, where
  every rank is finite, one rank can carry an 18× outlier and single-handedly set `ΣVar`. The
  estimator goes **heavy-tailed well before it goes undefined**, which is the dangerous part: nothing
  in the output announces it.

**The practical rule, and it cuts against the obvious one:** precision on `R` comes from ranks, but
the *sign* floor does not. Adding ranks at fixed shallow depth makes it **worse** — more draws, more
chances one of them has a near-zero denominator. Depth per rank has to clear the floor first; only
then do ranks buy precision.

**Above the floor, `R` is flat** — FeAs was pushed a further 4x to `m=16384` and `4096 -> 16384` is
flat on both seeds tested. The values that survive this, replacing the single-run numbers everywhere
else in this file (per-seed means over all runs above each floor, +/- the standard error over 3 base
seeds):

| model | depth | **`R`** | *non-null (internal)* | *`w_null` (internal)* | *efficiency (internal)* |
|---|---|---|---|---|---|
| square / D4, beta=1 | `m >= 64`/rank | **1.0425 +/- 0.0013** | 1.0425 +/- 0.0013 | 0 | 34.20% +/- 0.06% |
| FeAs 2-band, beta=5 | `m >= 1024`/rank | **3.042 +/- 0.049** | 2.569 +/- 0.037 | 0.1545 +/- 0.0036 | 83.94% +/- 1.02% |

Those are the **seed-ensemble** values: 32 independent base seeds per model, 64 ranks x 2048
measurements/rank, `+/-` the standard error over seeds (95% t-CI square `[1.0398, 1.0452]`, FeAs
`[2.943, 3.141]`). They supersede the 3-seed values `1.041 +/- 0.003` and `3.17 +/- 0.14` quoted in
earlier drafts, which they are statistically consistent with -- the point estimates barely moved, the
intervals shrank ~3x. See §1b for how hard they were pushed.

**FeAs's move from the previously quoted 2.48 is sampling noise, not the depth floor.** `R` is
statistically flat in depth from `m=256` through `m=16384` (per-depth means 3.19, 3.33, 3.04, 3.07),
and the committed run was not contaminated (outlier index 2.13, worst rank sign 0.12). It was just
imprecise: four disjoint 16-rank blocks of one 64-rank sample give `R` = 2.86, 2.96, 2.82, 3.40
against 3.36 for all 64. The old interval `[2.13, 3.51]` contains both 3.17 and the current 3.04, so
**the shift is not statistically significant** — 2.48 was a low draw. The depth floor governs estimator *safety*, not
bias in `R`.

Why FeAs is so much noisier than square at equal rank count: `sum Var` is dominated by a few
high-variance entries and the sign denominator makes each rank's `G` heavy-tailed, so the ~36%
per-entry error of a 16-sample variance barely averages down. Square's 16-rank interval is +/-1.6%.

**Where the error bar comes from matters, and the size of the effect is now measured.** Comparing
the mean bootstrap-over-ranks SE against the sample SD of `R` across base seeds at fixed depth, at 32
seeds: **square 1.08 `[0.81, 1.27]`** — well calibrated; **FeAs 1.44 `[0.96, 1.79]`** — optimistic,
consistent with the heavy-tailed sign denominator that a resample of 64 ranks cannot see. The
direction holds, but note **FeAs's interval includes 1**: earlier drafts quoted "1.39, 1.43, 1.90"
from three seeds, and an SD from `n=3` was never able to carry a factor. Treat it as a tendency. The
reporting rule follows either way, since it costs nothing to obey: **quote `R` from across-seed
scatter, not from the rank bootstrap.**

**A free contamination detector.** The identity `r = m/[1+(m-1)rho]` reproduces measured `r` to three
decimals at depth. Where the sign outliers bite it visibly fails (FeAs seed 12345 at `m=256`:
measured `r = 4.85` against predicted `5.50`, one `m=4` orbit even showing `rho < 0`). The prediction
assumes every pair within an orbit shares a single `rho`; one outlier rank makes the pairwise
correlations heterogeneous and breaks that. **Divergence between measured and predicted `r` flags a
contaminated run** — worth checking on every new model.

**Consequence for the β ladder.** `⟨sign⟩` falls with β, so the depth floor *rises* with β
independently of the autocorrelation time. The floor must be re-established at every β, and the same
measurement that certifies it also supplies the sign data needed to separate the two competing
effects on `ρ` (§4c).

## 1b. How hard the headline numbers were pushed

`R = 3.042 +/- 0.049` is a 1.6% standard error on a quantity whose per-seed values range 2.61 to 3.76.
That gap is worth stating plainly, because it is what the error bar is made of, and it is why the
number needed 32 whole independent runs rather than one long one.

**Precision comes from independent replicates, and a base seed is the only unimpeachable one.** A
rank is nominally an independent sample, and resampling ranks is cheap -- but where a sign problem
makes per-rank `G` heavy-tailed, a resample of 64 ranks cannot see the tail it is drawn from, and the
interval comes out too narrow (see the calibration in §1a). A fresh base seed reruns the entire
measurement with new walker streams everywhere, so the scatter across seeds estimates the sampling
distribution directly, assuming nothing about the ranks.

**Four checks, each able to fail independently, and none did:**

| check | what it would catch | square | FeAs |
|---|---|---|---|
| whole-run oracle: variance across 32 *whole runs*, replicate 64x deeper, none of the per-rank machinery | a defect in how the per-rank variance is constructed; a failure of scale-freeness | 1.047 `[1.028, 1.090]` | 3.28 `[2.98, 3.65]` |
| paired depth check: the same seeds rerun at 4x depth | contamination surviving the depth floor | `+0.0007 +/- 0.0019` | `+0.025 +/- 0.100` |
| median, pooled ratio-of-means, distribution-free bootstrap over seeds | the mean being set by tail draws; the t-interval's normality assumption | all agree | all agree |
| contamination gate: unusable runs, worst measured-vs-predicted `r`, worst rank sign | a near-zero sign denominator distorting a run | 0 runs, 0.0004, `<s>` = 1 exactly | 0 runs, 0.108, `<s>_min` = 0.147 |

Each headline sits inside its oracle's interval, which is the substantive one: the oracle's replicate
is a whole 131072-measurement run rather than a 2048-measurement rank, so agreement across that 64x
span is a direct measurement of the scale-freeness the whole budget argument assumes (§1).

**Depth buys safety; seeds buy precision — and the trade is now measured.** At 4x depth the per-seed
spread of `R` falls by only **1.78x `[1.04, 2.36]`** (FeAs) and 1.23x `[0.71, 2.18]` (square), while
the mean does not move. Both fall short of the `sqrt(4) = 2` that 4x the seeds delivers for the same
compute, so above the floor **seeds are the better buy**. Depth's job is to clear the floor; after
that it is the less efficient axis.

**One honest loose end.** Across FeAs seeds, `R` correlates mildly with the per-run outlier index
(Spearman +0.35, p = 0.05) and with the measured-vs-predicted `r` deviation (+0.32, p = 0.07). The
paired depth check rules out a depth-induced bias, so the reading is within-sample co-fluctuation: a
seed that happens to draw a large per-rank excursion carries more symmetry-*breaking* noise, which is
exactly what `P` removes, so its `R` is legitimately higher. It is not evidence of contamination
here, but it is the kind of correlation that would mean something else in a model with a worse sign
problem, so it is worth re-checking rather than assuming.

## 2. The ceiling is the orbit size `m`, not `sqrt(m)`

Symmetrization averages `m` orbit-mates: it divides **variance** by up to `m`, and the **error bar**
by up to `sqrt(m)`. These are variance ratios, so `m` is the ceiling.
*Evidence:* FeAs's two `m=8` orbits measured `r = 5.12` and `4.56` — both impossible under a
`sqrt(8) = 2.83` ceiling.

## 3. Two ceilings, and which one binds tells you what to do

`r = m/[1+(m-1)rho]`, bounded by `m` (orbit size) and by `1/rho` (noise-correlation limit); roughly
`r ~ min(m, 1/rho)`.
- **square/D4 at beta=1: rho-limited** (`1/rho ~ 1.1`) -> more symmetry buys nothing; must change the
  noise structure (e.g. lower temperature). **That prescription is now tested and it works — see 3a.**
- **FeAs: m-limited** (`1/rho ~ 20`) -> bigger clusters / larger groups / band-permuting ops pay off
  almost directly.

This is the single most useful diagnostic for "should I bother symmetrizing this system?"

**Which ceiling binds is not fixed per model — it MOVES with the regime, and the crossover is
measured** (both beta ladders, 2026-07-28). The `m`-ceiling is ~3.0 for square and ~3.7 for FeAs:

| model | `beta` | `1/rho` | binding |
|---|---|---|---|
| square | 1 -> 8 | 1.07 -> 1.79 | **`rho`, at every rung** |
| FeAs | 1, 2 | 1.92, 2.28 | `rho` |
| FeAs | 3, 4, 5 | 4.31, 8.70, **13.18** | **`m` (geometry)** |

So square never escapes being rho-limited over the range it was run: colder physics keeps paying
there, and more symmetry still would not. FeAs **crosses over between beta=2 and beta=3** and ends
the ladder firmly m-limited — by beta=5 the noise structure would support a 13x reduction and the 4x4
geometry can only cash 3.7x of it. The prescription flips with the crossover: below it, chase
temperature; above it, chase orbit size (bigger cluster, larger point group, more equivalent bands).
**Check which side you are on before spending anything** — this is a one-run diagnostic.

## 3a. `R` grows as you reach deeper into the interesting low-temperature physics

*Measured on TWO models (2026-07-28). Square isolates the mechanism; FeAs tests it with a live sign
problem. Both ladders are below, and the limit they expose is in 3b.*

`rho` is a property of the REGIME, not of the model — square is not intrinsically rho-limited.

Square's `R = 1.04` was measured at `beta = 1`, which is also the regime where the noise is dominated
by symmetric scalar channels — so "this model has symmetric noise" and "this temperature has
symmetric noise" were confounded. Sweeping `beta` with everything else fixed separates them.

**4x4 square, D4, 64 ranks x 2048 measurements/rank, 32 independent base seeds per rung:**

| `beta` | **`R`** | `rho` (mates) |
|---|---|---|
| 1 | 1.0414 +/- 0.0015 | 0.9389 +/- 0.0021 |
| 2 | 1.1406 +/- 0.0029 | 0.8246 +/- 0.0030 |
| 4 | 1.2122 +/- 0.0065 | 0.7240 +/- 0.0053 |
| 8 | **1.3429 +/- 0.0088** | 0.5575 +/- 0.0062 |

`rho` falls and `R` rises monotonically; across the ladder `R` changes by `+0.302 [+0.284, +0.319]`
and `rho` by `-0.381 [-0.395, -0.369]`, both resolved against across-seed scatter. So **the `beta=1`
value of `R ~ 1.04` is a property of that temperature regime, not of the single-band square model** —
growing correlation length gives the noise symmetry-breaking structure, exactly as predicted. The
`m/[1+(m-1)rho]` law holds rung by rung (measured vs predicted `r` agree to <=3e-3), so the gain is
fully accounted for by the mechanism in section 4.

**This is a fixed-bare-bath temperature axis — say so.** Every rung is the FIRST iteration of a
cold-started DCA loop: CT-AUX sampling the non-interacting bath at that `beta`, with `Sigma = 0`. It
is not a self-consistent sweep, and in particular it is not the chained sweep a production `Tc` run
does (each temperature seeded from the converged `Sigma` of the one above). That chaining is a
convergence accelerator — it changes how many iterations reach the fixed point, not the fixed point
itself — so its absence costs nothing here, and adopting it would make the rungs statistically
dependent and invalidate the trend intervals below.

What the bare bath *does* cost is directness: `R` is a property of a single solver call, and a
production run makes ~10-20 of them per temperature at a converged bath. **So quote this as `R` at
the bare bath, not as what a converged run experiences on every iteration.** The upside is that this
ladder varies `beta` and nothing else; a self-consistent ladder varies `beta` and `Sigma(beta)`
together, and a monotone `R(beta)` there could not be attributed to correlation length. The gap is
widest at the top rung — the converged 4x4 square at `beta=8` has strong AF correlations feeding back
into the bath, where the bare problem does not.

**This has now been measured at that worst-case rung, and the gap is negligible (§4e).** Running a
real `DcaLoop` for 10 iterations at `beta=8`, 8 seeds, the compute-weighted mean `R` over the
iterations a production run performs is **`1.3422 +/- 0.0064`** against the committed bare-bath
`1.3429 +/- 0.0088` — agreement to **0.05%**. The earlier hypothesis that self-consistency would push
mate-`rho` further down, making this ladder a conservative floor, is **not** what happens: `R` does
not move materially in either direction. So the `beta`-ladder numbers can be read as what a
production run experiences, not merely as a bare-bath reference.

**Why this ladder is clean, and what it therefore cannot say.** Two effects compete in general:
growing correlation length pushes `rho` DOWN, but `G = <sign*M>/<sign>` means a denominator
fluctuation gives `dG(k) = -G(k)*ds/s` -- weight `-G(k)`, itself symmetric -- so **sign-problem noise
is a fully symmetric channel with mate-correlation exactly 1**, pushing `rho` UP and `R` toward 1. A
model with a sign problem mixes the two and a bare `R(beta)` curve cannot say which is acting.

Square is provably sign-free in CT-AUX at any `beta` (nearest-neighbour hopping, bipartite lattice,
half filling), and the runs confirm it: `<sign>` is exactly 1 on every rank at every rung. So this
ladder isolates the correlation-length effect **by construction** — and by the same token it cannot
say which effect wins when both are live. **That is what the FeAs ladder below settles.**

Two controls, because the axis has two ways to lie:
- **The depth floor rises with `beta`** and was re-established per rung, not inherited: 64 per rank at
  `beta=1,2` but 256 at `beta=4,8` (the `64->256` step drifts negative in all three floor seeds at
  both). The ladder ran at 2048, clearing every rung by 8x.
- **The absolute frequency window shrinks as `beta` grows** — `sp-fermionic-frequencies` is fixed at
  128 while Matsubara spacing is `pi/beta` — so each rung sums `R` over a different slice of frequency
  space. Harmless only if `r` is flat in `omega`, which is checked at every rung: slope `~1e-18` per
  step, i.e. flat to machine precision.

Evidence: `04_beta_ladder.ipynb`, `runs/beta_ladder_square.json`. The `beta=1` rung independently
reproduces the milestone-6 headline (`1.0414 +/- 0.0015` here vs `1.0425 +/- 0.0013`, disjoint seeds).

### The second model, with the sign channel live

**4x4 FeAs, 2-band, 64 ranks x 2048 measurements/rank, 32 independent base seeds per rung**
(warm-up 200, per Conventions; disjoint seed range per rung):

| `beta` | **`R`** | `rho` (mates) | `<s>` | `w_null` |
|---|---|---|---|---|
| 1 | 1.5259 +/- 0.0157 | 0.5221 +/- 0.0075 | 0.998 | 0.031 |
| 2 | 1.8021 +/- 0.0186 | 0.4376 +/- 0.0065 | 0.961 | 0.065 |
| 3 | 2.3427 +/- 0.0267 | 0.2320 +/- 0.0033 | 0.723 | 0.112 |
| 4 | 2.7971 +/- 0.0188 | 0.1150 +/- 0.0028 | 0.378 | 0.152 |
| 5 | **2.9679 +/- 0.0592** | 0.0759 +/- 0.0034 | 0.148 | 0.153 |

`dR = +1.442 [+1.322, +1.561]`, `drho = -0.446 [-0.463, -0.430]`, both monotone and resolved.

**The finding: `rho` falls monotonically even as `<s>` collapses to 0.148.** Sign noise is a
*perfectly* mate-correlated channel and should drive `rho` UP; across the entire range where FeAs is
simulable, it loses to the correlation-length effect. So the mechanism measured on square is not an
artifact of a sign-free corner — **the trend is now established on two models, one of them with a
live sign problem**, not extrapolated from one. The `beta=5` rung also reproduces the milestone-6
headline (`2.968 +/- 0.059` vs `3.042 +/- 0.049`) across disjoint seeds AND a different warm-up.

**Two honest qualifications.**
- **Part of the `R_full` rise is `w_null`, not mechanism.** `R_full = R_non-null/(1 - w_null)`, and
  `w_null` grows 0.031 -> 0.153. Over `beta=1->5` `R_full` rises x1.945 while `R_non-null` rises
  x1.698, so roughly a quarter of the headline rise is the growing forced-null share. Both columns
  belong in any cross-model plot, which is why Conventions require it here specifically.
- **`beta=5` is at the edge of measurability**, not a comfortable interior point: outlier index 6.4
  and the across-seed SEM triples (0.019 -> 0.059). That is the method's boundary showing, and it is
  worth reporting as such.

Evidence: `05_beta_cross_model.ipynb`, `runs/beta_ladder_fe_as.json`. **Still bare-bath on both
models** — the bath-drift check (ROADMAP §3e) is task 3's remaining piece.

## 3c. A bigger cluster buys a better `rho` -- and the orbits it unlocks sit idle

*Measured 2026-07-29, ROADMAP task 4. Square/D4 at `beta = 5`, 64 ranks x 2048 measurements/rank,
independent base seeds, both points floored at `<=512`/rank and provably sign-free
(`<s>` = 1.0000 exactly).*

The open question this settles: **does `rho` fall with cluster size?** It decides whether the
`m`-ceiling is reachable in practice, because `r ~ min(m, 1/rho)` and a bigger cluster buys `m`.

| | 4x4 (`nk` = 16), 32 seeds | 8x8 (`nk` = 64), 16 seeds |
|---|---|---|
| **`R`** | 1.2619 +/- 0.0072 | **1.5006 +/- 0.0130** |
| **`rho` (mates)** | 0.6623 +/- 0.0049 | **0.5309 +/- 0.0056** |
| `1/rho` | 1.51 | 1.88 |
| `R_ideal` (m-ceiling) | 2.974 | 4.615 |
| *efficiency (internal)* | *42.4%* | *32.5%* |
| orbit sizes | `{1:2, 2:1, 4:3}` | `{1:2, 2:1, 4:9, 8:3}` |

**Answer: yes.** `drho = -0.131` and `R` rises **19%** on a single axis -- only `Nc` changes, at fixed
`beta`, band count, interaction and filling. This is the cleanest single-axis result in the project.

**But the shape of it corrects the natural expectation, and that is the part worth saying.** The 8x8
mesh delivers exactly what section 6 predicted it would: **three FREE `m=8` orbits**, which a 4x4 mesh
cannot host at all (every k-point there lies on a mirror line). So the m-ceiling rises hard, 2.97 ->
4.62. But `1/rho` only reaches 1.88, so **square stays `rho`-limited at BOTH sizes** (1.51 < 2.97 and
1.88 < 4.62) -- and because the ceiling rose faster than the achieved reduction, **efficiency falls,
42.4% -> 32.5%**.

Put plainly, for anyone deciding what to spend on: *on square at `beta = 5`, a bigger cluster buys you
a better `rho`, and the extra orbit structure you also paid for does nothing.* Both effects are real;
only one of them is being cashed. This is the same lesson as section 4b -- a model's symmetry
structure and the symmetry structure of its NOISE are different things, and it is the second that
decides `R`.

**What this does not establish.** Whether a model already **`m`-limited** converts 8x8 orbit structure
into `R` is untested and is the obvious follow-on: FeAs at `beta >= 3` has `1/rho` up to 13.2 against a
3.7 ceiling (section 3), i.e. it is starved of exactly what 8x8 supplies. On current evidence that is
where a bigger cluster should pay, and square is precisely the model that cannot show it.

> ⚠️ **Superseded values, kept because the mistake is instructive.** An earlier pass quoted this axis
> as `R` = 1.2506 +/- 0.0013 / 1.4014 +/- 0.0026 with `drho = -0.100`, read off the **floor-ladder**
> runs. Those intervals were void: `run_m_ladder.sh` does not enforce seed spacing -- deliberately,
> since floor ladders nest at fixed seed -- so its seeds sat **1 apart where 4096 is required**, and at
> 64 ranks consecutive base seeds share 63/64 of their walker streams (Gotcha 13). Three runs, one
> effective sample, fake precision. The 8x8 point estimate then moved **1.4014 -> 1.5006** once
> properly sampled, a 7% shift far outside the interval originally quoted. **Floor ladders answer
> depth-flatness only; never read an interval or a headline off one.**

## 4. The correlation that defeats symmetrization is already in the real-space noise

Symmetrization is an orthogonal projector: it deletes symmetry-**breaking** noise and leaves
symmetry-**preserving** noise untouched. So `rho` = the fraction of noise already living in the
symmetric subspace, and it is visible directly in real space as `corr(deltaM(r), deltaM(S.r))`:
- square/D4: **+0.905** within a symmetry shell vs **-0.183** across shells (specifically
  symmetry-aligned, not generic); `sigma^2(r=0)` is 49% of all noise power.
- FeAs: only +0.07..+0.10 within-shell; `sigma^2(r=0)` is 10% (intraband) / 5.4% (interband ~ white).

Physical reading: noise dominated by fluctuations in globally **symmetric scalars** (expansion order,
sign, density) is entirely symmetric — symmetrization cannot touch it. Good predictor for generic
k-pairs: `rho ~= sigma^2(r=0)/total - 1/Nk` (matched to a few percent on all blocks tested).

## 4a. Scalar fluctuations are mate-correlated *by construction* — the sharp mechanism

Any k-**independent** (scalar) fluctuation — local self-energy, expansion order, sign, density —
produces noise `delta G(k) = w(k)*delta X` with `w(k) = dG(k)/dX`. Since the model and `G` are
symmetric, `w(k)` is a **symmetric function of k**, so symmetry mates share it *exactly*:
`w(k_a) = w(k_b)`. **Every scalar channel therefore contributes correlation exactly 1 between
orbit-mates and is completely invisible to symmetrization.** Only momentum-resolved,
symmetry-breaking fluctuations differ between mates — the only thing `P` can remove.

This also explains the generic-vs-mate gap: for a generic pair `w(k_a) != w(k_b)`, so the shared
scalar channel is diluted by the weight mismatch. **That gap is the cleanest single diagnostic of
symmetry-aligned noise** — square **+0.543** (0.398 -> 0.941) vs FeAs intraband **+0.019**.

The three square measurements are one coherent picture, not three separate facts:
`sigma^2(r=0)` share 0.49 (fully-symmetric local part) + the symmetric content of the `r != 0`
shells, each ~93% symmetric internally (`[1+(m_s-1)c]/m_s = [1+3(0.905)]/4`), gives
`0.49 + 0.51*0.93 ~= 0.96` against a **measured mate-rho of 0.94**. Note the `r=0` share *alone*
badly understates the symmetric fraction.

What makes a model local-dominated, in rough order of expected importance:
1. **Short correlation length / high temperature** — estimator noise tracks the magnitude of what is
   measured, and `G(r)` is sharply peaked at `r=0` at high T.
2. **Near-k-independent self-energy** (DMFT-like) — the dominant physical fluctuation is literally
   the scalar `delta Sigma_loc`, giving `delta G(k) = G(k)^2 * delta Sigma_loc`, mate-correlation 1.
3. **Orbital structure** — interband components often have *no local scalar channel at all*, hence
   essentially white noise (FeAs interband `sigma^2(r=0)` = 5.4%, `rho ~ 0`).
4. **More bands** = more competing channels, diluting any single symmetric one.

> **Caveat — do not overstate.** square and FeAs differ in beta, band count, interaction *and*
> filling simultaneously, so the `rho` difference **cannot yet be attributed to any single cause**.
> The mechanism (where the correlation sits) is measured; the causation is not. This is exactly why
> the sweep must vary one axis at a time.

## 4b. `R` is a harmonic mean — it hides its own best cases

Splitting the entries into classes `C` gives an **exact** decomposition:

    R = 1 / sum_C ( w_C / R_C ),    w_C = class share of RAW variance

i.e. `R` is the variance-weighted **harmonic** mean of the class-wise `R_C`. Harmonic means are
dominated by their SMALLEST terms, so a class with poor `R_C` and a large variance share pins the
total down no matter how well everything else does. Measured on FeAs:

| class | n | variance share `w_C` | `R_C` |
|---|---|---|---|
| intraband | 32 | **81.90% +/- 0.38%** | 2.532 +/- 0.036 |
| interband | 8 | **2.65% +/- 0.06%** | **4.823 +/- 0.122** |
| forced null | 24 | 15.45% +/- 0.36% | inf |

The interband entries — carrying the distinctive multi-orbital physics — get a **4.8x** reduction on
under **3%** of the variance, so they are invisible in the aggregate 3.04. Same by orbit size: `m=8`
reaches `R_C = 4.86 +/- 0.09` on 5.4% of variance while `m=2` carries 28.8% at `1.862 +/- 0.018`.
(All from the 32-seed ensemble, `+/-` the standard error over seeds. Earlier drafts gave this table
from a single 16-rank run — same structure, but the values were a few percent off and had no error
bars at all.)

> **Headroom — an internal diagnostic, do not put this in a talk.** `R_ideal = sum(Var)/sum(Var/m)`
> is the `rho=0` ceiling for a model's ACTUAL orbit structure, and **efficiency = R / R_ideal**
> separates "small because orbits are small" from "small because the noise is symmetric". It earns
> its keep for deciding *what to do next* about a model, but it needs a paragraph of setup to
> interpret and it is defined only on the non-null support, so it does not belong in external
> reporting. Kept here because the conclusion it supports (next paragraph) does.
>
> | | `R` (non-null) | `R_ideal` | efficiency |
> |---|---|---|---|
> | square | 1.035 | **3.053** | **33.9%** |
> | FeAs | 2.156 | **2.894** | **74.5%** |
>
> *(single-run values; at proper depth — see §1a — square is 34.1% +/- 0.1% and FeAs
> **87.5% +/- 3.1%**, so the qualitative gap below is if anything understated.)*

**Square's headroom is HIGHER than FeAs's.** Its orbit structure is fine (78% of variance in `m=4`
orbits); it simply captures a third of what is available, while FeAs captures three quarters.

This CORRECTS the natural assumption that multi-orbital wins via bigger orbits. On a variance-weighted
basis it does not: FeAs's `m=8` orbits carry too little variance to matter, and its many `m=2` orbits
pull `R_ideal` back down to roughly square's level. FeAs wins on two different things — **much lower
rho** (75% vs 34% efficiency) and **forced nulls** (13-16% of variance annihilated, square has none).
Sharper statement: *the models differ in the symmetry-structure of their noise, not in their symmetry
structure.*

Note the forced-null half of that now shows up **directly in the headline**, since `R` is reported on
the full support: FeAs's `3.04` already contains the `w_null = 15.5%` of its raw variance that
symmetry drives to exactly zero. That is the intended behaviour — it is noise a user really would have carried — but it
is also why a cross-model comparison of `R` alone is not purely a statement about noise correlation.

## 4c. Symmetrization pays off most where the physics is most interesting  ← lead with this

The positioning claim, and the strongest form of it: not merely "multi-orbital models benefit more",
but **within** a multi-orbital model the benefit concentrates on the **interband / off-diagonal**
components — exactly the observables carrying the distinctive multi-orbital physics
(orbital-selective behavior, interorbital pairing, orbital order). The mechanism guarantees it:
interband entries have little or no on-site weight, so the local scalar channel that dominates
single-band noise is weak or absent -> `rho -> 0` -> `r` approaches the full orbit size.

**So: symmetrization helps most on the quantities people run DCA++ to compute.** And it compounds —
multi-orbital runs cost vastly more per data point, so a factor-R saving buys more *absolute* compute
exactly where compute is scarcest.

Structural support that survives the beta/band confound (see the caveat in 4a):
- On the *identical* 4x4 mesh with the *identical* |G|=8, square caps at m=4 while FeAs reaches m=8.
- All 10 of FeAs's non-null orbits mix `(b0,b1)` blocks; 104 P entries map between band blocks.
  At nb=1 those counts are 0 — the band machinery is entirely dormant.
- 38% of FeAs entries (24/64) are symmetry-forced nulls: noise driven to exactly zero. Square: none.

**Qualifier:** the gain comes from **symmetry-EQUIVALENT orbitals related by the point group**, not
from orbital count. With inequivalent orbitals, `U_S` acts trivially on band indices, orbits collapse
back to k-orbits, and there is no multi-orbital advantage at all.

**Reporting consequence:** aggregate `R` is variance-weighted and therefore dominated by the noisy,
local-dominated, band-diagonal entries that symmetrization helps LEAST. **The current headline number
actively hides the selling point** — FeAs's 3.04 is diluted by exactly the components nobody runs a
two-band calculation to look at. Split `R` by entry class (see 4b).

**Testable prediction:** `R` should grow with the number of symmetry-equivalent orbitals
(nb=1 -> 1.04, nb=2 -> 3.04, nb=3 -> ?).

**A second, competing channel — a finding in its own right.** G is estimated as `<sign*M>/<sign>`. A
fluctuation in the denominator is a GLOBAL SCALAR, giving `delta G(k) = -G(k) * delta s/s` with
`w(k) = -G(k)` symmetric — so **sign-problem noise is a fully symmetric channel with mate-correlation
exactly 1**. As the sign problem worsens, an increasing share of noise becomes untouchable, pushing
`rho -> 1` and `R -> 1`.

So two mechanisms push `rho` in OPPOSITE directions as beta rises: growing correlation length pushes
it down, a worsening sign problem pushes it up. Which dominates, and where the crossover sits, is
unknown and is one of the more interesting things the beta ladder can establish. Mapping the boundary
of where symmetrization pays off — *including* the regimes where it collapses — is a more useful
result for a practitioner than an unqualified endorsement would be.

*Status of the evidence:* square at beta=1 is sign-free (per-rank `<sign>` = 1 exactly), but **FeAs at
beta=5 has `<sign>` ~ 0.25** — a real sign problem, surfaced by the M-scaling control (§1a). So the
channel is live in a model already in hand, and the per-rank sign distribution is now instrumented on
every run. What is still missing is the *controlled* comparison: FeAs's `rho` is low (~0.06) despite
that sign problem, but FeAs differs from square on four axes at once, so this says nothing yet about
the sign channel's strength. Isolating it needs the beta ladder on a single model.

## 4d. Scope note: we measure variance in `G`; observables may differ

**This analysis measures `Var(G)` — the single-particle Green's function — and stops there.** That is
a deliberate scope boundary, not an oversight.

Be explicit about the caveat in any writeup: **the variance reduction a user actually experiences
depends on which observable they compute.** Downstream quantities are reached by transformations that
reweight the noise — the self-energy `Sigma = G0^-1 - G^-1` is a matrix inversion, susceptibilities and
`Tc` come from a Bethe-Salpeter solve on `G4`, spectra require analytic continuation — and each of
those redistributes noise across entries differently. So the observable-level reduction can be
**higher or lower** than the `R` we quote on `G`.

We spot-checked this once on `Sigma` and confirmed it moves in both directions (up substantially for
the two-band model, slightly down for single-band), which is why the caveat is stated rather than
assumed away. Quantifying observable-level reduction is **out of scope**.

The transferable quantity is the **per-orbit / per-entry-class `r`**, since it describes the noise
covariance structure itself; anyone wanting an observable-level number can reweight that map. This is
a further reason to ship the class breakdown (§4b) rather than a lone scalar.


## 4e. The bare bath is what a production run experiences — measured, and the caveat retires

**All the numbers here are measured at the BARE bath** (`G0_cluster_excluded = G0`, `Sigma = 0`): the
measurement driver calls the cluster solver directly and never instantiates a DCA loop, so the
coarse-graining and cluster-exclusion steps that build the sampled bath never run. `R` is therefore a
property of a **single solver call**, while a production run makes ~10-20 of them per temperature on
a bath rebuilt from the previous iteration's `Sigma`. What a total-compute claim depends on is the
**cost-weighted average of `R` over the iterations actually run** — so the error in quoting the
bare-bath number is roughly `(N-1)/N x drift`.

**That number has now been measured, and it is negligible.** Square, `beta=8` (the worst case — the
converged 4x4 square has strong AF correlations feeding back into the bath), a real `DcaLoop` for
**10 iterations x 8 independent base seeds x 64 ranks x 2048 measurements/rank**, undamped
`Sigma` update:

| quantity | `R` |
|---|---|
| committed bare-bath headline (32 seeds) | **1.3429 +/- 0.0088** |
| compute-weighted mean over the 10 iterations a production run performs | **1.3422 +/- 0.0064** |
| drift, iteration 0 -> 9 | **-0.5%**, CI `[-7.4%, +6.3%]`, unresolved |

**Agreement to 0.05%.** Quoting the bare-bath `R` as what a user experiences is honest, and this
caveat is closed rather than carried. Evidence: `runs/bath_drift_square.json`,
`analysis/bath_drift.py`.

**Two gaps, not one — and the roadmap's framing of the first was wrong.** "Bare bath" was described
as *iteration 1 of a cold-started loop*. It is not. A real loop runs cluster exclusion before its
first solve, and at `Sigma_cluster = 0` that computes `G0_cluster_excluded = G_k_w`, the
**coarse-grained** cluster Green's function, not the bare `G0(K)` the driver uses. At `beta=8` the
two baths differ by **73%** (`max|dG0|/max|G0|`) at identical `Sigma = 0`. So:

- **gap A, coarse-graining (a fixed offset, not drift):** `R` = 1.3258 +/- 0.0100 at the bare `G0`
  vs 1.3814 +/- 0.0179 at the loop's iteration 0, paired by base seed —
  **+4.2%**, `[+0.7%, +7.7%]`, resolved. No iteration count reveals this one.
- **gap B, self-consistency:** iteration 0 -> 9, the -0.5% above, unresolved.

They have opposite signs and largely cancel, which is *why* the production average lands back on the
committed headline. Worth stating plainly: the agreement is a partial cancellation of two effects,
not evidence that either is individually zero.

**The bath really did converge, so the flat `R` means something.** Per-iteration bath motion falls
`1.1e-1 -> 1.8e-2 -> 3.6e-3 -> 7.5e-4` and then plateaus at `~4e-4` — that plateau is Monte-Carlo
noise in `Sigma`, not further convergence. `<sign>` stays exactly 1.000 at every iteration (the
interacting bath does not introduce a sign problem on square) and `mu` stays pinned at 0 by
particle-hole symmetry, so neither is a confound. Validation rungs 1 and 2 pass on the loop path at
both the first and the converged iteration (mean preservation to 2e-16).

⚠️ **Pairing does NOT work across DCA iterations, unlike across depths.** The milestone-6 paired
depth test works because the deeper run reuses the same chain prefix. Across iterations the walker is
re-warmed at a changed bath every time, so by the last iteration nothing survives from the first but
the base seed: measured `corr(R_first, R_last) = -0.17` across seeds, pairing gain **0.93x** — i.e.
none. The resolved statement above therefore comes from the compute-weighted mean (SEM 0.48%), not
from the paired difference (+/-7%). Do not quote the paired interval as the drift bound.

**Scope, still honest.** This is one model at one temperature. FeAs at `beta=5` was *not* run — 3e's
decision rule fixed in advance was "escalate only if square shows drift", and it does not. A model
with a live sign problem could behave differently, since `Sigma` moves `<sign>` and the sign channel
is what decides where symmetrization stops paying; square is sign-free precisely so the drift is
isolated from that. This is a different caveat from §4d and both belong in a writeup: §4d is about
*which quantity* you reduce the variance of, this one about *which problem* the solver was solving.


## 5. It is genuinely free, and provably unbiased

`P` is a deterministic linear operator applied post-measurement — no extra sampling, no extra MC cost.
Mean preservation is verified to machine precision: numpy's full symmetrization of the raw ensemble
mean reproduces production's finalized `G_k_w` to **2e-16** on both models.

## 6. Band-permuting symmetry buys orbit size that lattice geometry alone cannot

On the *same* 4x4 mesh, square/D4 tops out at `m=4` while FeAs reaches `m=8`. Reason: `|G|=8` for both,
but orbit size is `|G|/|stabilizer|`, and on a 4x4 mesh **every** k-point lies on a mirror line (the
only non-{0,pi} coordinates available are +/-pi/2, forcing `|kx|=|ky|`), so no free orbits exist. The
first free (m=8) k-orbit appears at 8x8. FeAs escapes this because its group acts on the combined
`(b0,b1,k)` index — a stabilizing operation can still permute bands.

## 7. The paired design is enormously cheaper than ON/OFF ensembles

Because `P` is deterministic and linear, one set of raw per-rank samples yields **both** numerator and
denominator, so the ratio is paired and fluctuations cancel.
- Paired: **16 ranks** gave FeAs `R = 2.48`, with per-orbit `r` within 2% of prediction.
- Unpaired ensemble (prior prototype): ~**128 runs per arm** for `R = 2.45 [2.12, 2.90]` (+/-16%).

**Correction — the pairing buys accuracy per sample, not precision from nowhere.** Bootstrapping over
the 16 ranks of that paired run gives `R = 2.48 [2.13, 3.51]`, which is *wider* than the 128-run
unpaired interval, not tighter. The honest claim is per-sample cost: the paired design reaches
comparable precision from ~16 samples instead of ~256 (two arms x 128), and needs no special
no-symmetry build. **It does not license quoting `R = 2.48` from 16 ranks** — that point estimate was
always sitting inside a factor-1.6 interval. See §1a for the separate depth requirement.

The built-in null control is exact: singleton orbits (`m=1`, where `P` is the identity) return
`r = 1` with deviation **0.000e+00** — not `1.0 +/- something`.

## 7a. What a production DCA++ calculation costs: ~19,000-38,000 V100 GPU-hours

This is the number that makes `R` worth caring about. It is **entirely from DCA++'s own published
performance work** — no extrapolation from our runs, which are far too small to reach this regime.

The Summit DCA++ paper specifies a configuration it calls "*a typical production level execution*":
single-band square Hubbard, `U/t = 4`, `T/t = 0.02` (`beta = 50`), `Nc = 36`, average expansion order
`<k> = 2600`, **80 million total measurements**. Its weak-scaling study fixes **50,000 measurements
per node**, which it reports as "*a fixed runtime of about 12 minutes*".

That last pair is the calibration. 12 min / 50,000 measurements = **0.0144 node-seconds per
measurement**, so:

| quantity | value |
|---|---|
| one DCA iteration (80M measurements) | **320 Summit node-hours** |
| same, in GPU-hours (6x V100 per node) | **~1,900 V100 GPU-hours** |
| full calculation, 10-20 self-consistency iterations | **3,200-6,400 node-hours** |
| same, in GPU-hours | **~19,000-38,000 V100 GPU-hours**  <- headline |

Only the last row is a judgement call: the iteration count is the conventional 10-20 rather than a
figure quoted in the paper. Everything above it is arithmetic on published numbers.

**Why this composes with `R` without any extrapolation.** CT-AUX cost is linear in the number of
measurements at fixed model/`beta`/`Nc`, and `R` divides the measurements needed for a fixed error bar
(sec. "Notation"). So the two numbers multiply directly: `R` from our runs, absolute scale from the
literature. We never have to claim our `t_meas` reaches production.

**And we must not make that claim.** Fitting our own run logs gives `t_meas ~ beta^1.94` (square,
`beta = 1-8`) and `beta^2.14` (FeAs, `beta = 1-5`), both at `Nc = 16`. But the paper states cost scales
as `O(k^3)` with `<k> ~ O(Nc*U/T)`. Our `<k>` is small enough that the cubic term has not taken over;
production `<k>` is 2600. Extrapolating `beta^2` from `beta = 8` to `beta = 50` at `Nc = 16 -> 36`
would be wrong by orders of magnitude, and we have **no `Nc`-scaling data at all** — every run in
`scratch/` is `nk = 16`.

**Caveat before this goes in an abstract.** Those are GPU node-hours for a run whose cost is dominated
by two-particle measurements in single precision, while our `R` is measured on the single-particle `G`
(see sec. 4d). The anchor and the multiplier are describing different channels; they compose cleanly
only if `R` is comparable in both.

Sources: [Accelerating DCA++ on Summit](https://www.osti.gov/servlets/purl/1607140) (config and
timings); [DCA++ CPC paper](https://www.osti.gov/biblio/1606982).


## 8. DCA++ has no single-rank error bars

Both error routes are across-rank: `JACK_KNIFE` returns an empty function at `n==1`
(mpi_collective_sum.hpp:370), and `STANDARD_DEVIATION` runs `average_and_compute_stddev` across ranks,
which yields identically zero at `n==1`. The within-run second moment `M_r_w_squared_` is accumulated,
reduced and symmetrized but never consumed into any output error function. So uncertainty
quantification in DCA++ is fundamentally replication-across-ranks — which is also why this measurement
requires `mpirun -n P` with `P > 1`.
