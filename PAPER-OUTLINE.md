# Potential paper — outline, gaps, and open questions

**Status: speculative.** Nothing here is committed work. This file captures a *reframing* of the
project and what a paper built on it would look like, so the idea is not lost while the measurements
that would support it are still outstanding.

**Scope boundaries for this file, so it does not drift against anything:**

- **This is the paper, not the talk.** The 25-minute talk is a separate, more focused artifact with
  much more background and far fewer claims. Do not let this outline set the talk's scope.
- **This is NOT a task list.** [`ROADMAP.md`](ROADMAP.md) is the single task list and stays that way.
  Work items identified here are *proposals* for it (§6); if one is adopted, it moves there and this
  file cites it rather than tracking it.
- **Measured findings live in [`TAKEAWAYS.md`](TAKEAWAYS.md).** Numbers are quoted here only where
  the reframing changes how they read. TAKEAWAYS stays the source of truth.

---

## 1. The reframe

### 1.1 The identity

`P` is an orthogonal projector (`P² = P`, `Pᵀ = P`), so with `Σ` the noise covariance of `Ĝ`:

```
Σ_x Var(P·Ĝ)[x] = tr(P Σ Pᵀ) = tr(P Σ)
```

and therefore, **exactly**:

```
R = tr Σ / tr(P Σ) = 1 / (1 − f_break)

    f_break  ≡  tr[(1−P) Σ] / tr Σ
             =  the fraction of total sampling-noise POWER living in the
                symmetry-breaking subspace
```

**Consequence: `R` is not a performance number. It is a measurement of the symmetry structure of the
Monte Carlo noise, and symmetrization is the instrument that performs it.** The variance reduction is
a *corollary* of the measurement, not the point of it.

This is the whole reframe. Everything below follows from taking it seriously.

### 1.2 What it does to the existing numbers

Every committed `R` converts with no new runs. Errors propagate as `δf = δR/R²`.

| system | `R` (committed) | **`f_break`** |
|---|---|---|
| square / D4, β=1 | 1.0425 ± 0.0013 | **4.08% ± 0.12%** |
| square / D4, β=2 | 1.1406 ± 0.0029 | **12.33% ± 0.22%** |
| square / D4, β=4 | 1.2122 ± 0.0065 | **17.51% ± 0.44%** |
| square / D4, β=8 | 1.3429 ± 0.0088 | **25.53% ± 0.49%** |
| FeAs 2-band, β=1 | 1.5259 ± 0.0157 | **34.46% ± 0.67%** |
| FeAs 2-band, β=2 | 1.8021 ± 0.0186 | **44.51% ± 0.57%** |
| FeAs 2-band, β=3 | 2.3427 ± 0.0267 | **57.31% ± 0.49%** |
| FeAs 2-band, β=4 | 2.7971 ± 0.0188 | **64.25% ± 0.24%** |
| FeAs 2-band, β=5 | 2.9679 ± 0.0592 | **66.31% ± 0.67%** |
| FeAs, milestone-6 headline | 3.042 ± 0.049 | **67.13% ± 0.53%** (15.5 pts of it symmetry-forbidden) |

Read the square column aloud: *96% of the sampling noise in a β=1 DCA calculation of the
single-band square-lattice Hubbard model is symmetry-preserving, falling to 74% by β=8.* That is a
statement about the simulation. `R = 1.04` is a disappointing benchmark. **Same measurement.**

Three secondary benefits of the change of variable, all real:

- `f_break ∈ [0,1]`, so it plots far better than an unbounded ratio and is comparable across models.
- `w_null` stops being an asterisk — symmetry-forbidden entries are simply the part of `f_break` that
  symmetry annihilates outright. This *supports* the standing convention that full-support `R` is the
  headline (ROADMAP Conventions), rather than straining against it.
- Efficiency `R/R_ideal` becomes "the fraction of the **geometrically available** `f_break` that the
  noise actually populates", which is interpretable in one line instead of a paragraph. ⚠️ This cuts
  against the settled convention that efficiency is a dev diagnostic and stays out of writeups — flagged
  as a decision to make, **not** as a reason to reopen it. Only revisit if Fig. 6 needs it as an axis.

### 1.3 The mechanistic reading

`f_break` small means **the noise covariance is low-rank and its leading directions are symmetric.**
Sampling error in `Ĝ` is not spread over the `Nc × Nω` measured entries; it is concentrated in a
handful of *group-invariant collective coordinates of the configuration* — expansion order, the sign,
total density, the local self-energy. Each is a scalar function of the auxiliary field, so each enters
as `δG(k) = w(k)·δX` with `w(k) = ∂G(k)/∂X` a symmetric function of `k`, and each is therefore
confined to the trivial sector by construction (TAKEAWAYS §4a).

So the finding is not "a standard trick underperforms." It is: **QMC noise in these algorithms is
structured, low-rank, and aligned with the symmetry of the Hamiltonian.**

---

## 2. Title candidates

Ordered by how much they commit:

1. **"Most Monte Carlo noise in cluster dynamical mean-field calculations is symmetry-preserving"**
   — states the finding, and invites the "…and when it isn't" that the results section delivers.
   *Current preference.*
2. "Symmetry structure of Monte Carlo sampling noise in dynamical cluster approximation calculations"
   — safe, descriptive, CPC / SciPost-shaped.
3. "How much quantum Monte Carlo noise can symmetry remove? Structure, ceilings, and the sign problem"
   — most readable; closest to advocacy, which we do not want (see `research-not-advocacy`).

---

## 3. Section outline

**§1 — Setup.** `P` as orthogonal projector; the identity of §1.1; the per-orbit form
`r = m/[1+(m−1)ρ]` **cited as the survey-sampling design effect, not derived as new** (Kish); two
ceilings, `r → min(m, 1/ρ)`. State plainly up front that the technique is standard and has been
documented only qualitatively — the contribution is the measurement and the mechanism, not the trick.

**§2 — The scalar-channel result.** *The core of the paper; give it its own section.* Any
k-independent fluctuation enters with `w(k)` symmetric ⇒ orbit mates share it exactly ⇒
mate-correlation 1 ⇒ contribution to `f_break` identically zero. **Whole classes of physical
fluctuation are structurally invisible to any symmetrization**, and `1/ρ` is where they cap it.
Corollary as a labelled result: sign-problem noise enters through a stochastic denominator with
weight `−G(k)`, hence is a *perfectly* symmetric channel.

**§3 — Method.** The paired estimator (one sample set gives numerator and denominator); scale-freeness
of `R`; the depth floor and the finding that it is **sign-set**, not autocorrelation-set, above β≈2 on
FeAs; seeds > depth > ranks for precision, with the rank-bootstrap calibration. Given that DCA++ has
no single-rank error bars (TAKEAWAYS §8), this reads as a UQ contribution, not housekeeping.

**§4 — Temperature axis.** `f_break` rises on both models with β; ρ falls monotonically on FeAs *even
as ⟨s⟩ collapses to 0.148*. The symmetric sign channel loses to the correlation-length effect
everywhere FeAs is simulable. Growing correlation length is what gives the noise symmetry-breaking
structure.

**§5 — Orbital axis (needs `threeband_hubbard`).** The `nb` trend, and the equivalent-vs-inequivalent
control *inside a single run* — d–p block inequivalent, p–p block equivalent.

**§6 — Geometry axis (needs 8×8).** `ρ(Nc)`, and whether the `m`-ceiling is reachable in practice.

**§7 — Consequence.** The `min(m, 1/ρ)` diagnostic: one run tells a user which ceiling binds and
therefore what to buy. Compute anchor (TAKEAWAYS §7a) confined to one paragraph, with the
single-particle-vs-`G4` mismatch stated in the same breath rather than in a footnote.

---

## 4. Figures

| # | figure | status |
|---|---|---|
| 1 | **Migration scatter.** Orbit mates in the complex plane at fixed ω: raw cloud, arrows to the sign-aware orbit average, centroid marked. Small multiples — square (ρ≈0.94, barely contracts) beside FeAs `m=8` (ρ≈0.08, collapses), singleton panel as visual null. Shows the projector, mean preservation and ρ in one image. | **ROADMAP task 5 — build regardless of publication; it is also the best available talk slide** |
| 2 | **The noise is symmetry-aligned, not generically correlated.** Real-space `corr(δM(r), δM(S·r))` within-shell vs across-shell, plus the mate-vs-generic ρ gap (square +0.543, FeAs +0.019). Rules out "everything is just correlated." | data in hand (`02_noise_mechanism`) |
| 3 | **`f_break(β)`, both models.** Two panels, `f_break` and ρ, with ⟨s⟩ annotated along the FeAs curve so the reader watches the sign problem worsen while ρ keeps falling. | replot of `05_beta_cross_model` in the new variable |
| 4 | **Orbital structure — the control inside the measurement.** Per-entry-class `r` for the p–p block (equivalent → gain), the d–p block (inequivalent → should pin at `r=1`), and the `nb` = 1→2→3 trend. Null control and effect in the same run; make the adjacency visible. | **needs `threeband_hubbard`** |
| 5 | **Geometry.** `m`, `1/ρ` and measured `r` vs `Nc` on shared axes. | **needs 8×8** |
| 6 | **The `(1/ρ, m)` plane.** Contours of `min(m, 1/ρ)`, every measured system placed as a point, diagonal separating ρ-limited from m-limited; square's β ladder walks up one side, FeAs crosses over. The figure most likely to be reused by others. | assemble from existing data |

Plus **Table 1**: every system × regime with `m`, ρ, `f_break`, `R`, ⟨s⟩, `w_null`, depth floor.

**Fig. 5 is the one whose result cannot be predicted**, and it is load-bearing for §7. Run the 8×8 ρ
measurement before committing to the outline, so it is known whether §6 is a positive result or a
bounded negative one. **The reframe survives either way** — if ρ is flat in `Nc`, the finding is that
the ceiling recedes as fast as you approach it, which is still a statement about noise locality and
still supports the diagnostic. That robustness is a reason to prefer the reframe.

---

## 5. Precedent — what is already claimed, and what is not

Verified by search, 2026-07-29. Full citations in §8.

**Already owned:**

- **The technique, qualitatively.** The original DCA-QMC paper states that point-group and
  translational averaging "can greatly reduce the statistical error in measured Green functions" —
  no number, no mechanism, no conditions. DCA++ ships it silently. *This is the gap.*
- **The general theorem.** Orbit averaging over a group leaving the distribution invariant provably
  reduces variance in the Loewner order and preserves the mean — Chen/Dobriban/Lee, explicitly framed
  as Rao–Blackwellization. **Our TAKEAWAYS §5 ("free and provably unbiased") is a rediscovery of
  this** and must cite it rather than assert it.
- **`r = m/[1+(m−1)ρ]`.** The survey-sampling design effect / effective sample size, `n_eff =
  n/[1+(m̄−1)ρ]` with ρ the intraclass correlation. Textbook. Not a result — but a good *import*, and
  nobody appears to have applied it to QMC symmetrization.
- **"Symmetry averaging saturates because of within-sample correlation."** Well known in lattice QCD:
  source-position averaging stops improving past ~8–12 sources per configuration. And
  Blum–Izubuchi–Shintani's covariant approximation averaging is a published class of symmetry-based
  variance reduction with *quantified* gains (16× for the nucleon mass). **The genre is established
  and publishable — just in a different community.**

**Apparently unclaimed** (ranked by how likely a referee calls it new):

1. **The scalar-channel ceiling.** Converts "correlated samples cap the gain" (known) into a statement
   about *which physical fluctuation channels are structurally uncorrectable*, with `1/ρ` as a
   measurable hard ceiling. Not found anywhere. **Best result in the project.**
2. **Its sign-problem corollary**, plus the measurement that the symmetric sign channel *loses* to the
   correlation-length effect everywhere FeAs is simulable. A clean, mildly counterintuitive negative.
3. **ρ as a property of the regime, not the model**, with the m-vs-1/ρ crossover as a one-run
   diagnostic. The piece with actual practitioner value.
4. **The UQ methodology**, given DCA++'s lack of single-rank error bars.

**In correlated-electron QMC the variance question appears unstudied.** Symmetry appears for
*systematic* error and sign mitigation (Shi–Zhang, AFQMC) and improved estimators exist for CT-QMC
noise (Kaufmann et al.), but nothing quantifies point-group symmetrization's variance effect.

---

## 6. What is missing — analyses that would strengthen the case

Proposals for [`ROADMAP.md`](ROADMAP.md), not a parallel task list. Roughly in order of
value-per-unit-cost.

### 6.1 Already on the roadmap, and load-bearing here

- **`threeband_hubbard`** (ROADMAP task 4). The d orbital is inequivalent to two equivalent p
  orbitals, so one run yields the `nb`=3 trend point *and* the "benefit vanishes for inequivalent
  orbitals" control. **Converts §5 / TAKEAWAYS §4c from a confounded two-model comparison into a
  causal claim.** Highest value, already scoped, no new symmetry code needed.
- **8×8 / `ρ(Nc)`** (ROADMAP task 4, cluster-size axis). Answers the question a referee asks first,
  because cluster size is the axis production runs actually push. Decides whether the `m`-ceiling is
  ever reachable. Benchmark GPU before committing (ROADMAP task 4 has the decision rule).
- **Migration scatter** (ROADMAP task 5). Figure 1, and the best talk asset.

### 6.2 New, cheap, and high-value

- **Decompose `f_break` by irreducible representation, not just symmetric vs. breaking.** `P` already
  exists; this is characters instead of orbits, a small extension of `orbit_table.hpp`. It turns one
  scalar into a spectrum and is what nearly every open question in §7 needs. **Single most valuable
  addition identified.** It is also the difference between "we quantified a known trick" and "we
  characterized the structure of Monte Carlo noise in cluster QMC."
- **Is the self-consistency noise floor the symmetric channel?** (§7.2) Testable from data already on
  disk — the §3e per-iteration bath dumps. Apply `P` to the iteration-to-iteration bath motion and
  split it. Hours, not days.
- **Leading eigenvectors of the measured noise covariance vs. the predicted `w(k)`.** Direct test of
  the §2 mechanism, and it is the constructive step toward control variates (§7.1). Pure post-processing
  on existing HDF5.

### 6.3 Known weaknesses a referee will press on

| weakness | honest response | can it be fixed cheaply? |
|---|---|---|
| **Single-particle `G` only** | The transferable object is the per-entry-class `r` map; ship it as a table. Scope decision made with the advisor, stated in the paper. | No — and it should stay out of scope. But see §7.3 for why the `G4` question is an *opening*, not just a hole. |
| **Cost anchor and multiplier describe different channels** — TAKEAWAYS §7a's 19k–38k V100-h is dominated by two-particle measurement in single precision; `R` is measured on single-particle `G` | Already flagged in TAKEAWAYS. Must be in the same paragraph as the number, never a footnote, and never in the abstract. | No |
| **One code, one solver (CT-AUX)** | State it. CT-AUX also silently drops `J`/`Jp`, so multi-orbital models must be described as density-density / Ising-Hund (Gotcha 2) — **this must be right in the paper text.** | No |
| **Two models at one cluster size** | 6.1 fixes this | Yes |
| **`R` is a harmonic mean that hides its own structure** (TAKEAWAYS §4b) | Report the class breakdown, not a lone scalar. The `f_break` reframe helps: it makes the decomposition additive in noise power rather than harmonic in ratios. | Yes, already have it |

---

## 7. Bigger questions this begs

**Clearly speculative.** Each is labelled with what would test it. None of this belongs in the paper
as a claim; some belongs in a discussion section, and one or two might be the *next* paper.

### 7.1 Shortcomings in the algorithm — the constructive reading

**Symmetrization Rao-Blackwellizes on the wrong statistic.** The group orbit is a sufficient statistic
for symmetry-breaking noise and nothing else. If the noise lives in a few scalar channels, the same
measurement that says "symmetry cannot help here" identifies the estimator that would: **control
variates on those scalars.** `⟨k⟩` has a known relation to the potential energy in the CT-AUX
expansion; the density is externally constrained by the μ condition; the sign is measured every sweep.
These are quantities whose fluctuations can be subtracted with known or estimable weights, and
`w(k) = ∂G(k)/∂X` is exactly the required coefficient. Live precedent exists for control variates
against the sign problem.

> **Test:** take the leading eigenvectors of the measured noise covariance and check whether they are
> the predicted `w(k)`. If they are, that is a result on its own and the control-variate construction
> follows immediately. Pure post-processing (§6.2).

**A better ergodicity diagnostic than we meant to build.** The depth floor — shallow runs show
inflated `f_break` because a single configuration is not symmetric — is a measurement of *how far the
walker has explored the point-group orbit of configuration space*. Dimensionless, known asymptote,
computed from data already dumped. Arguably better behaved than an integrated autocorrelation time.
A spin-off, not extra work.

**A hard negative worth stating in the field.** *Any* variance reduction that works by exploiting a
symmetry of `H` is structurally blind to the sign channel, because the sign is a scalar. Symmetry
methods are often reached for precisely in the hard regimes where they are least able to help.

### 7.2 DCA-specific — one conjecture testable from data on disk

**Conjecture: symmetric noise is the component that survives the self-consistency loop.** Noise in `Σ`
that is symmetric produces a symmetric bath perturbation, is not damped by symmetrization at *any*
iteration, and can accumulate coherently across iterations in a way symmetry-breaking noise cannot.
§3e found bath motion plateauing at ~4e-4 and identified that plateau as MC noise in `Σ` rather than
further convergence. **If that plateau is dominated by the symmetric channel, then the noise floor on
a converged DCA solution is set by exactly the component symmetrization cannot touch.** That is a far
more pointed statement about DCA than anything currently in the draft.

> **Test:** apply `P` to the iteration-to-iteration bath motion in the existing §3e per-iteration
> dumps and split it. Cheap and falsifiable (§6.2).

### 7.3 The `G4` angle — why the referee's question is the opening

DCA++'s production cost and its worst noise both live in the two-particle function, and the observable
people run it for — the d-wave pairing eigenvalue — is a **B1g** quantity, not a trivial-rep one.

**Conjecture: if noise concentrates in A1g while the interesting instabilities live in non-trivial
irreps, signal-to-noise for symmetry-breaking order parameters is structurally better than an entry
count suggests — and symmetrization helps most in exactly the channels carrying the instabilities.**

> **Stated gap, do not paper over it:** `G`'s noise structure does not automatically transfer to `G4`,
> and the BSE solve is nonlinear. This is a conjecture with an obvious test, not a result. But it
> converts "we only did single-particle" from a weakness into a motivated next step, which is worth
> more than the hole costs.

### 7.4 Physics of solids — mostly no, with one thread

Honestly: `f_break` is a property of an estimator, not of a material. **Do not oversell this.** But one
thread is not empty.

`f_break` grows with β because noise tracks the weight of what is measured: at high T, `G(r)` is peaked
at `r=0`, and the on-site component is invariant under the *entire* point group, so local noise is
maximally symmetric. As ξ grows, weight moves to `r ≠ 0` shells where symmetry-related sites can differ
within a single sample. So `f_break` is a crude correlation-length diagnostic read off the **noise**
rather than the signal.

**The speculative extension, and the highest-upside idea here:** near an instability that breaks the
point group, fluctuations in the ordering channel diverge — and those are symmetry-breaking *by
definition*. Monte Carlo sampling noise of a susceptibility grows with the susceptibility. So
`f_break`, **resolved by irrep**, should develop a peak in the ordering channel on approach to a
nematic or d-wave instability. If it holds, the noise is a fluctuation spectrometer, and part of the
measured `R(β)` rise is physics rather than estimator behaviour.

> **Caveats, firmly.** The 4×4 clusters have no transition; the observed rise is smooth, modest, and
> fully accounted for by correlation length; and the noise-to-susceptibility link is loose. But it is
> testable: go toward an instability and look at irrep-resolved `f_break`.

### 7.5 Questions for a second paper

- **Is there a scaling law?** `f_break` as a function of `ξ/L` would tie Fig. 5 to something
  universal-ish and testable across cluster sizes.
- **Is this a CT-AUX fact, a Hubbard fact, or a Monte Carlo fact?** The same measurement in DQMC,
  CT-HYB, AFQMC — and the direct analogue in lattice QCD, where correlated-source saturation is
  folklore but, as far as could be found, never decomposed by irrep either. A cross-method
  noise-structure comparison is a whole second paper and a much broader one.
- **Should the measurement be done in the symmetry-adapted basis from the start?** If most entries are
  quiet and a few are noisy, that changes accumulator design and `G4` storage allocation — a real
  production bottleneck in DCA++ — not just post-processing.

---

## 8. Venue assessment

**Not a letter.** No new algorithm, no new physics, no discovery — this is a characterization of an
existing technique in one code. PRL is not a realistic target and pitching it there costs months.

| venue | fit |
|---|---|
| **Phys. Rev. E** | best fit for mechanism + measurement as a statistical-physics-of-estimators paper |
| **SciPost Phys. Core / Codebases** | good; open review; tolerant of narrow-and-honest |
| **Computer Physics Communications** | strong **if** paired with ROADMAP task 6 — upstream the patch, make `R` a standard solver output. Converts a measurement into a tool. |
| PASC / SC / ICCS proceedings | low-risk home for the compute-savings framing |

**Career note:** this is squarely computational physics / UQ. There is **no real solid-state angle** —
nothing here is a statement about FeAs or the Hubbard model as physics, and the one hook (noise
structure tracking correlation length) is an observation, not a condensed-matter result. §7.4 is the
only path to a physics claim and it is speculative.

---

## 9. References

Verified by search on 2026-07-29 unless marked.

**DCA and DCA++**
- M. Jarrell, Th. Maier, C. Huscroft, S. Moukouri, *Quantum Monte Carlo algorithm for nonlocal
  corrections to the dynamical mean-field approximation*, Phys. Rev. B **64**, 195130 (2001).
  [cond-mat/0108140](https://arxiv.org/pdf/cond-mat/0108140) — **the qualitative point-group-averaging
  statement this paper quantifies.**
- Th. Maier, M. Jarrell, Th. Pruschke, M. H. Hettler, *Quantum cluster theories*, Rev. Mod. Phys.
  **77**, 1027 (2005). *(cited from knowledge — verify exact pages before use)*
- U. R. Hähner et al., *DCA++: A software framework to solve correlated electron problems with modern
  quantum cluster methods*, Comput. Phys. Commun. (2020).
  [OSTI 1606982](https://www.osti.gov/biblio/1606982)
- G. Balduzzi et al., *Accelerating DCA++ on Summit*.
  [OSTI 1607140](https://www.osti.gov/servlets/purl/1607140) — source of the production-cost anchor
  (TAKEAWAYS §7a).
- P. Staar, Th. Maier, T. C. Schulthess, *DCA⁺: DCA with continuous lattice self-energy*.
  [arXiv:1304.3624](https://arxiv.org/abs/1304.3624) — scope boundary: we do plain DCA.

**Group averaging as variance reduction (statistics / ML)**
- S. Chen, E. Dobriban, J. H. Lee, *A Group-Theoretic Framework for Data Augmentation*, JMLR **21**
  (2020) / NeurIPS 2020. [arXiv:1907.10905](https://arxiv.org/pdf/1907.10905) —
  [JMLR PDF](https://jmlr.csail.mit.edu/papers/volume21/20-163/20-163.pdf). **Orbit averaging reduces
  variance in the Loewner order, preserves the mean, framed as Rao–Blackwellization. TAKEAWAYS §5
  must cite this rather than assert it.**
- *Rao-Blackwell Gradient Estimators for Equivariant Denoising Diffusion*, NeurIPS 2025.
  [arXiv:2502.09890](https://arxiv.org/html/2502.09890v2) — recent instance of the same construction.
- B. Elesedy, S. Zaidi, *Provably strict generalisation benefit for equivariant models*, ICML 2021.
  *(cited from knowledge — NOT verified by search; check before citing)*
- L. Kish, *Survey Sampling* (Wiley, 1965) — the design effect `deff = 1 + (m−1)ρ` and effective
  sample size. *(cited from knowledge; the identity itself is verified textbook material —
  [summary](https://cran.r-project.org/web/packages/PracTools/vignettes/Design-effects.html))*

**Symmetry-based variance reduction in lattice QCD**
- T. Blum, T. Izubuchi, E. Shintani, *New class of variance-reduction techniques using lattice
  symmetries*, Phys. Rev. D **88**, 094503 (2013).
  [arXiv:1208.4349](https://arxiv.org/abs/1208.4349) — **the closest methodological precedent, and it
  quantifies (16× cost reduction for the nucleon mass).**
- E. Shintani, R. Arthur, T. Blum, T. Izubuchi, C. Jung, C. Lehner, *Covariant approximation
  averaging*, Phys. Rev. D **91**, 114511 (2015).
  [APS](https://link.aps.org/doi/10.1103/PhysRevD.91.114511)
- *Variance reduction strategies for lattice QCD* (2026 review).
  [arXiv:2605.00643](https://arxiv.org/html/2605.00643)
- *Sparsening Algorithm for Multi-Hadron Lattice QCD Correlation Functions*.
  [arXiv:1908.07050](https://arxiv.org/pdf/1908.07050) — source-density saturation, i.e. the
  correlated-samples ceiling in the QCD context.
- *Signal/noise enhancement strategies for stochastically estimated correlation functions*.
  [arXiv:1404.6816](https://arxiv.org/pdf/1404.6816)

**Symmetry and improved estimators in correlated-electron QMC**
- H. Shi, S. Zhang, *Symmetry in auxiliary-field quantum Monte Carlo calculations*, Phys. Rev. B
  **88**, 125132 (2013). [arXiv:1307.2147](https://arxiv.org/abs/1307.2147) — symmetry for
  *systematic* error and sign mitigation; reports smaller statistical errors but does **not** quantify
  variance reduction.
- J. Kaufmann, P. Gunacker, A. Kowalski, G. Sangiovanni, K. Held, *Symmetric improved estimators for
  continuous-time quantum Monte Carlo*, Phys. Rev. B **100**, 075119 (2019).
  [arXiv:1906.00880](https://arxiv.org/pdf/1906.00880) — the adjacent "reduce CT-QMC noise by a better
  estimator" line of work.
- H. Hafermann et al., *Improved estimators for the self-energy and vertex function in
  hybridization-expansion CT-QMC*, Phys. Rev. B **85**, 205106 (2012).
  [DOI](https://doi.org/10.1103/PhysRevB.85.205106) *(author list from knowledge — verify)*

**Control variates and sign-problem noise**
- *Neural Autoregressive Control Variates for the Quantum Monte Carlo Sign Problem*.
  [arXiv:2605.26814](https://arxiv.org/html/2605.26814) — precedent for the §7.1 constructive step.
- *Elucidating the sign problem through noise distributions*.
  [arXiv:1210.7250](https://arxiv.org/pdf/1210.7250)
- *Statistical Angles on the Lattice QCD Signal-to-Noise Problem*.
  [arXiv:1711.00062](https://arxiv.org/pdf/1711.00062)
