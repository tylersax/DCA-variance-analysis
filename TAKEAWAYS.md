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
gives `R_C = 4.83`; the aggregate is `R = 2.48` (full) / `2.16` (non-null), dragged down by the
intraband class at `R_C = 2.13` carrying 84.8% of the variance.

Two related quantities:
- `R_ideal = sum(Var) / sum(Var/m)` — the `rho = 0` ceiling for a model's actual orbit structure.
- `efficiency = R / R_ideal` — how much of the available reduction is actually captured.

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

*Caveat:* assumes each sample is in the CLT/asymptotic regime — measurements past thermalization and
well beyond the autocorrelation time. If `M` sits below the autocorrelation scale, the noise structure
(hence `ρ`, hence `R`) could drift with `M`. This is exactly what the M-scaling control tests, and it
matters most at large beta where autocorrelation times grow.

## 2. The ceiling is the orbit size `m`, not `sqrt(m)`

Symmetrization averages `m` orbit-mates: it divides **variance** by up to `m`, and the **error bar**
by up to `sqrt(m)`. These are variance ratios, so `m` is the ceiling.
*Evidence:* FeAs's two `m=8` orbits measured `r = 5.12` and `4.56` — both impossible under a
`sqrt(8) = 2.83` ceiling.

## 3. Two ceilings, and which one binds tells you what to do

`r = m/[1+(m-1)rho]`, bounded by `m` (orbit size) and by `1/rho` (noise-correlation limit); roughly
`r ~ min(m, 1/rho)`.
- **square/D4 at beta=1: rho-limited** (`1/rho ~ 1.1`) -> more symmetry buys nothing; must change the
  noise structure (e.g. lower temperature).
- **FeAs: m-limited** (`1/rho ~ 20`) -> bigger clusters / larger groups / band-permuting ops pay off
  almost directly.

This is the single most useful diagnostic for "should I bother symmetrizing this system?"

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

| class | n | variance share | `R_C` |
|---|---|---|---|
| intraband | 32 | **84.8%** | 2.13 |
| interband | 8 | **2.1%** | **4.56** |
| forced null | 24 | 13.1% | inf |

The interband entries — carrying the distinctive multi-orbital physics — get a **4.6x** reduction on
**2%** of the variance, so they are invisible in the aggregate 2.48. Same by orbit size: `m=8`
reaches `R_C = 4.83` on 4.4% of variance while `m=2` carries 35% at 1.76.

**Headroom.** `R_ideal = sum(Var)/sum(Var/m)` is the `rho=0` ceiling for a model's ACTUAL orbit
structure, so **efficiency = R / R_ideal** separates "small because orbits are small" from "small
because the noise is symmetric":

| | `R` (non-null) | `R_ideal` | efficiency |
|---|---|---|---|
| square | 1.035 | **3.053** | **33.9%** |
| FeAs | 2.156 | **2.894** | **74.5%** |

**Square's headroom is HIGHER than FeAs's.** Its orbit structure is fine (78% of variance in `m=4`
orbits); it simply captures a third of what is available, while FeAs captures three quarters.

This CORRECTS the natural assumption that multi-orbital wins via bigger orbits. On a variance-weighted
basis it does not: FeAs's `m=8` orbits carry too little variance to matter, and its many `m=2` orbits
pull `R_ideal` back down to roughly square's level. FeAs wins on two different things — **much lower
rho** (75% vs 34% efficiency) and **forced nulls** (13% of variance annihilated, square has none).
Sharper statement: *the models differ in the symmetry-structure of their noise, not in their symmetry
structure.*

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
actively hides the selling point** — FeAs's 2.48 is diluted by exactly the components nobody runs a
two-band calculation to look at. Split `R` by entry class (see 4b).

**Testable prediction:** `R` should grow with the number of symmetry-equivalent orbitals
(nb=1 -> 1.03, nb=2 -> ~2.5, nb=3 -> ?).

**A second, competing channel — a finding in its own right.** G is estimated as `<sign*M>/<sign>`. A
fluctuation in the denominator is a GLOBAL SCALAR, giving `delta G(k) = -G(k) * delta s/s` with
`w(k) = -G(k)` symmetric — so **sign-problem noise is a fully symmetric channel with mate-correlation
exactly 1**. As the sign problem worsens, an increasing share of noise becomes untouchable, pushing
`rho -> 1` and `R -> 1`.

So two mechanisms push `rho` in OPPOSITE directions as beta rises: growing correlation length pushes
it down, a worsening sign problem pushes it up. Which dominates, and where the crossover sits, is
unknown and is one of the more interesting things the beta ladder can establish. Mapping the boundary
of where symmetrization pays off — *including* the regimes where it collapses — is a more useful
result for a practitioner than an unqualified endorsement would be. No sign problem at beta=1
(per-rank signs exactly equal), so there is no data on this yet.

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

The built-in null control is exact: singleton orbits (`m=1`, where `P` is the identity) return
`r = 1` with deviation **0.000e+00** — not `1.0 +/- something`.

## 8. DCA++ has no single-rank error bars

Both error routes are across-rank: `JACK_KNIFE` returns an empty function at `n==1`
(mpi_collective_sum.hpp:370), and `STANDARD_DEVIATION` runs `average_and_compute_stddev` across ranks,
which yields identically zero at `n==1`. The within-run second moment `M_r_w_squared_` is accumulated,
reduced and symmetrized but never consumed into any output error function. So uncertainty
quantification in DCA++ is fundamentally replication-across-ranks — which is also why this measurement
requires `mpirun -n P` with `P > 1`.
