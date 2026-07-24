# Mini-plan: end-to-end variance-reduction demonstration (symmetrization ON vs OFF)

**Status: SIGNED OFF 2026-07-21, decisions resolved (§10). Parked in `dev/` — demo only, NOT committed, NOT part of the M3′ PR.** Sits under M5 in the roadmap but is a **pre-M5** demonstration — it does *not* need M5's runtime switch. Depends on M3′ + [#374](https://github.com/CompFUSE/DCA/pull/374) (or at least on the current `symm-m3` working tree).

## 1. The claim we want to demonstrate

Imposing the H0-derived point group on the single-particle G reduces the Monte Carlo variance of the final estimator, by an amount set by the **symmetry-orbit size of each k-point**, without shifting the mean.

Two distinct claims, worth keeping separate:

- **(A) Symmetrization reduces variance.** Demonstrated on `square_lattice<D4>`. Not novel — this is the textbook expectation, and it validates the *mechanism* and the measurement.
- **(B) Our derivation found symmetry that was never declared, and it buys real variance reduction.** Demonstrated on **FeAs**, whose declared `FeAsPointGroup` is 2 ops and whose derived group is 8. Four of those ops were unreachable before #374. **This is the project's actual story**; (A) is the control that makes it credible.

## 2. Why this is possible now, without M5

D7 made `no_symmetry` a permanent compile-time OFF switch, and M3′'s gate is
`if constexpr (DIMENSION==2 && LatticeHasInitializeH0 && !is_no_symmetry<DCA_point_group>)`.
So the A/B is a **one-token change in a test-local typedef**:

| arm | instantiation | gate | behavior |
|---|---|---|---|
| ON | `square_lattice<D4>` | true | derive-authoritative, 8 ops imposed |
| OFF | `square_lattice<no_symmetry<2>>` | false | legacy path, identity-only group = genuine no-op |

Crucially the point group is a **test-local typedef** (`square_lattice_setup.hpp` defines
`Model = TightBindingModel<square_lattice<D4>>`), *not* the `DCA_POINT_GROUP` cmake selector. So both arms are two binaries in **one build**, not two builds.

## 3. Grounded findings that shape the design

These were verified in the tree, not assumed. Each one would have silently broken a naive version of this experiment.

1. **Symmetrization is applied AFTER MPI reduction, to the ensemble-mean G.** CT-INT: `finalize()` → `collect_measurements()` → `computeG_k_w(...)` → `Symmetrize<Parameters>::execute(data_.G_k_w)` ([`ctint_cluster_solver.hpp:260`](../../include/dca/phys/dca_step/cluster_solver/ctint/ctint_cluster_solver.hpp)). CT-AUX: `collect_measurements()` → `symmetrize_measurements()` (on `M_r_w`) → `compute_G_k_w_from_M_r_w()`. Also `dca_loop.hpp:340`.
   ⇒ In production, symmetrization reduces the variance of the **final averaged estimator**, not of any per-rank quantity.
2. **`local_G_k_w()` is raw and UNSYMMETRIZED** — it is `M_r → M_k`, divide by accumulated phase, `computeG_k_w`, and returns. No symmetrize call.
   ⇒ The existing verification stat test, which samples `local_G_k_w()` per rank, would measure **exactly zero** ON/OFF difference. **Do not reuse it unmodified** — this is the trap that motivated the whole design. The demo must read G *after* `finalize()`, not the per-rank accessor.
3. **Coarse-graining does not use the point group for its k-mesh.** The only symmetry hit in `coarsegraining_sp.hpp` is `spin_symmetric_`, an unrelated flag. ⇒ Flipping the declaration changes symmetrization and not the CG quadrature. (Still verified empirically by the bias control in §6.)

## 4. Design

**Estimator under test = the production estimator itself.** One run produces
$\hat G_{\text{prod}} = \mathrm{Sym}\big(\tfrac{1}{N_{\text{rank}}}\sum_r G_r\big)$ via the solver's own `finalize()`. We do **not** re-implement or relocate the symmetrization; we run the real path and measure the spread of its output.

**Replicas = N independent runs (DECIDED — §10.1).** Each replica is one invocation with its own RNG seed; both arms use the *same* seed list, so paired runs see the same Markov chains and differ only in the group being imposed. Variance is the sample variance across the N runs, per (k, ω), per arm.

### Seeding — VERIFIED 2026-07-21, with a trap

Plumbing is intact: input JSON `Monte-Carlo-integration.seed` takes an integer or `"random"` ([`mci_parameters.hpp:261`](../../../include/dca/phys/parameters/mci_parameters.hpp)); CT-AUX constructs `rng_(concurrency.id(), num_procs, parameters_.get_seed())` ([`ctaux_cluster_solver.hpp:200`](../../../include/dca/phys/dca_step/cluster_solver/ctaux/ctaux_cluster_solver.hpp)); the threaded solver builds one RNG per walker. Then, in [`random_utils.cpp`](../../../src/math/random/random_utils.cpp):

```
global_id = local_id * num_procs + proc_id     // :21
seed_     = hash(global_id + seed)             // :31   hash = MurmurHash3 Mix13 finalizer
```

**TRAP: the stream is fixed by the SUM `global_id + seed`, not by the pair.** A run with seed $S$ and walker `global_id = g` draws the *identical* stream as a run with seed $S+\delta$ and walker $g-\delta$. With consecutive seeds, two replicas whose seeds differ by $\delta$ share $(PW-\delta)$ of their $PW$ streams ($P$ = ranks, $W$ = walkers/rank). Replicas would be correlated and the measured variance **deflated in both arms** — the ON/OFF ratio might roughly survive (both arms hit equally), but the absolute variances, the bootstrap CIs and above all the **ρ estimate in notebook §5** would be corrupted. That is the strongest evidence plot, so this is not cosmetic.

**Requirements:**
1. **Seed spacing > P·W.** Use a large prime stride (`seed_i = 1000003 * i`) or randomly drawn distinct seeds. `seed_` is an `int`, so keep the largest under 2³¹ (64 replicas × 10⁶ ≈ 6.4×10⁷ — comfortable).
2. **Explicit integers, never `"random"`.** The `square_lattice_input.json` this adapts from uses `"seed": "random"`, which would silently break the paired design (each arm drawing independently from `std::random_device`). Record the seed list to a file and have both arms read the same one.
3. **Verify, don't infer.** Debug builds print `Generated Rng with global id: X and seed: Y` for every RNG constructed ([`std_random_wrapper.hpp:41`](../../../include/dca/math/random/std_random_wrapper.hpp), `#ifndef NDEBUG`). Run two replicas in Debug and diff the printed seed sets to confirm disjointness directly.

**Why not ranks-as-replicas** (per-rank `local_G_k_w()` + a test-side `Symmetrize` call, the pattern the verification stat test uses). It was the original plan and was **rejected on cost/benefit 2026-07-21**:
- The apparent saving was illusory. It compared N single-rank replicas against N *production-sized* runs — but a replica in this design may be a **one-rank run**, and N one-rank runs is statistically the same experiment as one N-rank job. At equal total measurements the two cost the same and give the same estimator quality.
- It measured a **proxy** — Sym(G_r) — and then argued the proxy maps to production by linearity of `Sym`. Three assumptions had to be carried and checked (ranks i.i.d.; `Sym` genuinely linear, incl. no data-dependent norm/skip; and the local-vs-global sign division, since `local_G_k_w()` divides by the *local* accumulated sign whereas production divides by the global one). **N independent runs deletes all three**, because the thing measured *is* $\hat G_{\text{prod}}$.
- It needed a custom all-replica gather and a test-side re-implementation of the production symmetrize call — extra code and a fidelity risk, for an end-to-end demonstration whose entire point is fidelity.

Kept documented as the **fallback** only if replica count ever becomes compute-bound (e.g. per-run setup overhead turns out to be a large fraction of runtime — it is seconds for a 4×4 single-band cluster, so this is not expected), or if a specific question needs per-rank granularity.

**Harness (DECIDED — §10.3: parked in `dev/`, not committed).** Small binary pair under `.claude/symmetry_project/dev/variance_demo/`, adapted from `ctaux_square_lattice_verification_stattest.cpp`:

```
symmetry_variance_setup.hpp        # typedefs; the ONE token that differs per arm
ctaux_variance_demo_on.cpp         # square_lattice<D4>
ctaux_variance_demo_off.cpp        # square_lattice<no_symmetry<2>>
square_variance_input.json         # Nc=16, single DCA iteration, seed from argv/input
```

Each invocation: read input → `update_model()` / `update_domains()` → one QMC iteration (`initialize(0); integrate();`) → **`qmc_solver.finalize()`** — the real production path, which is what applies the symmetrization → dump `data_.G_k_w` to HDF5 tagged with its seed, plus the k-mesh and the live op count. Run each binary N times with the shared seed list; the notebook reads 2N files.

*Why a test binary rather than the shipped `applications/dca` executable:* the app selects its lattice and point group through the `DCA_POINT_GROUP` cmake selector, so an ON/OFF pair would need **two configured builds**. A test-local typedef gives two binaries in **one build**, which is the only reason any C++ is written here at all. The binary is otherwise thin — no custom symmetrize, no custom gather, no covariance machinery; `finalize()` does the work.

The code is **byte-identical between arms**; only the typedef differs. In the OFF arm `Symmetrize::execute` still runs inside `finalize()`, with the identity group, and is a no-op.

**Do NOT converge the DCA loop.** One iteration only. Symmetrization inside the self-consistency loop feeds the next iteration's Σ, which entangles noise reduction with a changed convergence path.

**Plain DCA, not DCA+.** M3′ imposes on the CLUSTER family only; DCA+ routes through LATTICE, still legacy.

### Solver: CT-AUX primary, CT-INT cross-check (DECIDED — §10.2)

Both solvers are driven through their own `finalize()`, so the harness is identical for either and the original "start with CT-INT because its intervention is simpler" reasoning is moot — there is no intervention.

What differs is *what production symmetrizes*, and it favors CT-AUX:

| solver | production symmetrizes | M3′ branch exercised |
|---|---|---|
| CT-INT | `data_.G_k_w` (k-space), `finalize():260` | k-space, eq:imp-k + `fold_phase` re-dressing |
| **CT-AUX** | `M_r_w_` + `M_r_w_squared_` (real space), `symmetrize_measurements():574` | **r-space, eq:imp-r + band-dependent site images** |

**CT-AUX is the better proof point for M3′ specifically**, not merely the more-used solver: the r-space branch is the one that flipped `threeband_D4`'s `r_tau` and `r_iw` green, M3′'s biggest correctness win. Running both is a bonus — matching orbit structure from the two solvers independently validates *both* M3′ branches.

*(Under the earlier ranks-as-replicas design this section had to argue that symmetrizing G is equivalent to CT-AUX's symmetrizing M — true given G0 is exactly symmetric, but it carried a `G0_k_w_cluster_excluded` verification gap. Running `finalize()` makes the argument unnecessary: CT-AUX symmetrizes M exactly as it does in production.)*

**Trap — do NOT use CT-AUX's internal error bars.** `symmetrize_measurements()` symmetrizes `M_r_w_squared_`, the jack-knife error accumulator (line 574), so the *reported* errors are themselves altered by which arm you are in. Measuring variance across replicas sidesteps this entirely; nothing in this plan reads an internal error bar.

**Free control from the A/B design:** `Symmetrize::execute` also runs `executeTimeOrFreq` (the ω↔−ω averaging), which is point-group-independent and therefore applied *identically in both arms*. It cancels in the ratio, so the measured ratio isolates the point-group contribution with no extra work.

## 5. Model + orbit structure

**Primary: `square_lattice`, Nc = 16 (4×4), single band.** Cheapest, no U_S or Bug-2 confounds, already green on the characterization spine.

D4 orbits of the 4×4 mesh, k = (π/2)(n₁,n₂):

| orbit representative | size |
|---|---|
| (0,0) | 1 |
| (π,π) | 1 |
| (π,0) | 2 |
| (π/2,0)-star | 4 |
| (π/2,π/2)-star | 4 |
| (π/2,π)-star | 4 |

6 orbits, 16 points. ✓

**There are no orbit-8 points on a 4×4** — every k sits on a high-symmetry line or point and so has a nontrivial stabilizer. A generic point (orbit 8) first appears at 8×8. **So the ceiling here is 4×, not 8×.** Do not promise 8× and then measure 3.

**Secondary: FeAs**, 2 declared ops vs 8 derived — claim (B). Verify first that FeAs's H0(k) is real symmetric so Bug 2 stays out of the frequency channel.

## 6. Predictions

Per-k, with ρ the mean pairwise correlation of the noise between orbit-mates:

$$\frac{\mathrm{Var}_{\text{off}}(k)}{\mathrm{Var}_{\text{on}}(k)} \;\approx\; \frac{n_{\text{orbit}}(k)}{1 + (n_{\text{orbit}}(k)-1)\,\rho}$$

- **Ratio exactly 1 at Γ and (π,π)** (orbit 1, nothing to average). **This is the hard control** — if these move, the setup is wrong (suspect the coarse-graining confound of §3.3).
- ≤ 2 at (π,0)/(0,π); ≤ 4 on the three stars.
- Expect meaningfully **below** the ideal, since orbit-mates are estimated from the same configurations ⇒ ρ > 0. A measured ratio at or above n_orbit is a red flag, not a triumph.
- **Bias control:** ⟨G_on⟩ − ⟨G_off⟩ must sit within MC error. Symmetrization is an exact relation; a systematic shift means something is biasing. This is the stochastic echo of what `CoarseNoOp` already proves deterministically on G₀.

**Report the structure, not a scalar.** "Ratio tracks orbit size, pinned at 1 for the singletons" is sharp and falsifiable. A single global "≈30% less noise" is weak and easy to attack.

**Statistical precision — sets the replica count.** A variance estimated from N replicas has relative SE ≈ √(2/(N−1)); a ratio of two, ≈ √(4/(N−1)). N = 64 ranks ⇒ ~18% per variance, ~25% per ratio. **Enough to distinguish 1 from 4; not enough to resolve 3.6 from 4.0.** Pool over ω and over same-size orbits, and quote bootstrap CIs. If ranks are scarce, N = 32 still separates 1 from 4 but little else.

## 7. The analysis notebook

`.claude/symmetry_project/dev/symmetry_variance_demo.ipynb` — reads the two HDF5 files, produces the figures and the summary table. Sections:

1. **Load.** Glob the 2N per-run HDF5 files, stack into `[replica, k, band, ω]` complex arrays per arm; read the k-mesh and the live op count from each file. Assert the op counts are what the arms claim (8 vs 1) — cheap guard against running the wrong binary — and that the ON and OFF seed lists match element-wise, which is what makes the arms paired.
2. **Orbit labeling — reimplemented independently in Python.** Build the D4 orbits of the cluster mesh from scratch rather than importing the C++ answer; the point is an independent check. Assert orbit sizes sum to Nc and reproduce the §5 table.
3. **Variance estimation.** Per (k, ω), sample variance across replicas for each arm, using E|G − ⟨G⟩|² (and Re/Im separately as a cross-check).
3b. **Replica-independence check.** Confirm the N runs really are independent samples: no replica is a duplicate, the per-replica mean shows no drift with run index, and — the specific hazard from §4 "Seeding" — the **inter-replica correlation matrix is consistent with zero off-diagonal**. Partially shared RNG streams (the `hash(global_id + seed)` sum collision) would show up here as structured off-diagonal correlation, typically strongest between replicas whose seeds are closest. This is the failure mode that would deflate variance in *both* arms and masquerade as a null result. Run it first; nothing downstream is trustworthy without it.
4. **Money plot** — ratio Var_off/Var_on vs n_orbit, one point per (k,ω), with bootstrap CIs; overlay the ideal `y = n` and the null `y = 1`. The expected picture is points pinned at 1 for n=1 and scattered between 1 and n above it.
5. **Correlation diagnostic — the plot that turns "less noise" into "the mechanism we claim."** Estimate ρ per orbit directly from the OFF arm (pairwise correlation of replica noise between orbit-mates), then plot the *predicted* n/(1+(n−1)ρ) against the *measured* ratio. Agreement on a y = x line is far stronger evidence than the loose bound in §6.
6. **Bias control.** ⟨G_on⟩ − ⟨G_off⟩ normalized by its own MC error, per (k,ω); should be a standard normal. Histogram + a Γ/(π,π)-only zoom.
7. **Frequency dependence.** Ratio vs ω. Should be flat — the point-group effect is ω-independent, so a slope indicates contamination (likely from the frequency channel not cancelling as §4 assumes).
8. **Summary table** — per orbit: size, mean measured ratio + CI, fitted ρ, predicted ratio, and the bias z-score. This table is what goes in the PR description.

Deliberately plain: numpy + h5py + matplotlib, no project imports, so it runs anywhere the HDF5 files land.

## 8. Staging

1. **Square Nc=16, ~64 ranks, 1 iteration, CT-INT.** Hours, not days. Either the orbit structure appears or the setup is wrong — cheap failure.
2. **FeAs 2-vs-8** — claim (B), the real result.
3. **8×8 only if the 8× headline is wanted**, and only after the structure validates at 4×4.

## 9. What this will NOT show

- Nothing about LATTICE families (DCA+/BSE) — still legacy under M3′ (D4).
- Nothing about the multiband frequency channel — Bug 2 deferred.
- Nothing two-particle; G4 symmetrization is a separate path.
- Not M5's runtime switch — this is a compile-time A/B.
- On square, declared == derived, so it measures *symmetrization*, not *derive-authoritative*. Only FeAs separates those.
- Variance of a **single-iteration** estimator, not of a converged DCA solution.

## 10. Decisions (RESOLVED 2026-07-21, user)

1. **Replica granularity → N independent runs** (reversed 2026-07-21 on user challenge; ranks-as-replicas was the original plan). The claimed 1/N saving was illusory — a replica may be a one-rank run, so the two designs cost the same at equal total measurements. N independent runs measures $\hat G_{\text{prod}}$ **directly** through the solver's own `finalize()`, which deletes all three assumptions the proxy approach had to carry (ranks i.i.d.; `Sym` linear; local-vs-global sign division), the custom gather, and the test-side symmetrize call. Full reasoning in §4. Ranks-as-replicas retained only as a compute-bound fallback.
2. **Solver → CT-AUX primary, CT-INT as cross-check.** CT-AUX is both the more-used solver *and* the one exercising M3′'s r-space branch — the branch that produced the biggest correctness win. See §4. (My earlier "start with CT-INT, it's simpler" recommendation was withdrawn once `ctaux::local_G_k_w()` turned out to be the same shape as CT-INT's.)
3. **Harness location → parked in `.claude/symmetry_project/dev/variance_demo/`.** Not committed. Revisit if M5 wants it as a real test.
4. **Not part of the M3′ PR.** Demonstration, not a correctness gate; M3′'s reviewer surface is already large (Principle 5).

## 11. Next steps

1. ~~Confirm the seed plumbing.~~ **DONE 2026-07-21 — see §4 "Seeding".** Settable per invocation and reaches the engine, but `seed_ = hash(global_id + seed)` makes the stream a function of the *sum*, so seeds must be spaced by more than ranks×walkers and must be explicit integers, not `"random"`. (The `G0_k_w_cluster_excluded` symmetry question is **moot** under N-independent-runs; it only mattered for the withdrawn proxy design.)
2. Build the `dev/variance_demo/` binary pair (CT-AUX ON/OFF) — thin wrappers around `finalize()` + an HDF5 dump.
3. Run N ≈ 64 seeds per arm, Nc = 16, one DCA iteration.
4. Write `symmetry_variance_demo.ipynb` against the real files — layout observed, not guessed. Run §3b first.
5. Square result → FeAs 2-vs-8 (claim B) → CT-INT cross-check (validates M3′'s k-space branch alongside CT-AUX's r-space).

---

## 12. RESULTS (executed 2026-07-22, CT-AUX, symm-m3 tree)

Harness + notebooks under `.claude/symmetry_project/dev/` (`variance_demo/`, `symmetry_variance_demo*.ipynb`,
`variance_demo_lib.py`, `variance_demo_feas_lib.py`). Built via `-DDCA_WITH_VARIANCE_DEMO=ON` + a guarded
`add_subdirectory` in the worktree CMakeLists (uncommitted). **CT-AUX only** — CT-INT does not compile in
this tree (ctint submatrix-walker header); CT-AUX is the plan's primary solver anyway.

### Stage 1 — square Nc=16, claim A (N=64/arm)
- ON=8 ops, OFF=1 op; seeds paired; independence checks clean (drift 0.03, lag-1 0.06, seed-proximity −0.002).
- Ratio **pinned exactly at 1.0** for the Γ and (π,π) singletons (hard control), ω-independent, mean unbiased
  (singleton |z|~1e-16).
- Above the singletons the reduction is **ρ-limited**: orbit-mate MC noise is strongly correlated on this
  cheap single-band cell (ρ≈0.86–0.94 for the 4-stars, 0.46 for the 2-fold), so measured ratios are
  1.05–1.37 — well below the ideal n, exactly as §6 predicted. Predicted n/(1+(n−1)ρ) matches measured
  (machinery-consistency check, since paired seeds make ON = orbit-average of OFF).

### Stage 2 — FeAs, claim B
- **2→8 derivation proven** (report, both cluster sizes): "declared group: 2 op(s); derived group: 8 op(s)";
  6 under-declared band-permuting ops named (mirrors + C4/C4³), reachable only via the P-search (#374).
- To get an OFF arm: `FeAsLattice` ignores its own template arg (base hardcoded `bilayer_lattice<FeAsPointGroup>`),
  so DCA_point_group is always FeAsPointGroup. OFF arm subclasses FeAs + overrides DCA_point_group→no_symmetry,
  with a demo-local `ModelParameters` specialization so the subclass is still recognized as FeAs.
- **2×2 (Nc=4) is vacuous for claim B**: interband G ≡ 0 (H0 interband ∝ sin kx·sin ky = 0 on that mesh) and
  band-diagonal k-orbits are trivial (declared C4 already reaches them). Verified interband |G|≈1e-35. Band-
  diagonal there gives a clean 2× (ρ≈0, bands independent) but the derived group buys nothing beyond declared.
- **4×4 (Nc=16), N=32/arm, is the real claim-B result.** Non-null interband (|G|≈5e-3), genuine k-stars.
  Derived orbit sizes {2:4, 4:6, 8:1, 24:1}. Headline: the **size-8 band-diagonal orbit** ((π/2,π/2) star ×
  band swap) measures **Var_off/Var_on = 2.45 [2.12, 2.90] (ρ=0.325)** — CI entirely **above the declared
  2-op ceiling of 2×**. Mean unbiased (mean|z|=0.69). One symmetry-forced-null interband orbit (the size-24)
  is excluded: the derived group sends it to ~0, making Voff/Von an ill-defined 0/0.
  ⇒ Reduction the declared group could never reach, bought purely by H0-derived, undeclared symmetry.

### Stage 2b — FeAs 4×4 at N=512/arm (executed 2026-07-24, second machine, 128 cores)

Same binaries, same seed construction (stride 1000003), N raised 32 → 512/arm. 1024/1024 runs clean;
every ON file reports ops=8, every OFF ops=1; ON/OFF seed sets identical. ~50 min at concurrency 32.

- **Headline tightens and holds: Var_off/Var_on = 2.51 [2.40, 2.64] (ρ=0.312)**, against 2.45 [2.12,
  2.90] at N=32. CI **3.3× narrower**; the lower bound moves 2.12 → 2.40, so the interval now clears
  the declared 2× ceiling with room rather than grazing it. Point estimate is well inside the old CI.
- Independence checks pass at the tighter N=512 band (±0.087): drift +0.002, lag-1 −0.004,
  seed-proximity −0.015. Mean unbiased, mean|z|=0.67. Orbit sizes unchanged, {2:4, 4:6, 8:1, 24:1}.
- **All orbits match the predicted n/(1+(n−1)ρ) to two decimals** — the whole orbit spectrum, not
  just the headline. N=32 was too noisy to show this.

#### Resolved: the interband orbits were an analysis artifact, not a symmetrization bug

The first pass at N=512 reported **two** size-4 interband orbits with ρ = exactly 1.000, measuring
1.58 / 1.74 where the prediction gives 1.00 — the only rows in the table that missed. Unequal member
variances were ruled out (max/min variance = 1.00). The actual cause, established directly against
the data:

- `ON_A = −ON_B` to **1.3e-17** — the two "orbits" are exact negatives of each other;
- `ON_A` = the **signed** mean `(ΣA − ΣB)/8` to **1.3e-11**, i.e. machine precision;
- the plain mean of either class alone is off by 1.3e-2, and the plain mean of all 8 cancels to ~0.

So there is **one size-8 interband orbit on which the interband component is odd**: half its members
enter the average with a −1, and M3′'s imposition path reproduces that signed average exactly. The
symmetrization code is correct — this is a second, independent demonstration that it handles a
nontrivial representation, not just trivial band-diagonal averaging.

The fault was in `orbits_from_on`, which defined orbit-mates by equality of the ON value and so split
the signed orbit into two ± classes, handing `n=4` downstream for what is an `n=8` orbit and
computing ρ from unsigned residuals (hence the spurious ρ=1.000: within a ± class the entries *are*
the same quantity). Fixed by matching up to sign and returning per-member signs, which
`orbit_rho_flat` now undoes before correlating. Result: orbit sizes {2:4, 4:4, 8:2, 24:1} (64 entries
conserved), the two rows collapse to one **n=8 interband orbit at 1.66 [1.61, 1.72], ρ=0.546**, and
**10/10 orbits match the ρ prediction**. Claim B's headline was never affected — its orbit is
band-diagonal and was verified an exact plain mean throughout.

### Not done (out of scope / cost)
- CT-INT cross-check (doesn't compile here); 8×8 headline (§8 stage 3); the size-24 interband channel
  quantitatively (symmetry-forced null). None affect the two headline claims.
