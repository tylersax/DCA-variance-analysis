"""RETIRED 2026-07-31 -- this is no longer the source of the hero notebook.

`symmetrization_variance.ipynb` is now edited directly and is the source of truth. Do NOT run this
file: it would overwrite the notebook with the state as of 2026-07-31 and clear its outputs.

Kept only as a plaintext record of the notebook's prose and cell code at the point of the switch,
since a .py diffs and greps more comfortably than a .ipynb. It never held any of the computation --
that lives in hero_lib.py / reduction.py / mechanism.py / summaries.py, which the notebook imports,
and which are still live.

To regenerate the notebook's *outputs* (the only regeneration that still applies):

    python -m nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=2400 symmetrization_variance.ipynb
    python check_executed.py

Verify execution by counting execution_count per cell -- nbconvert can exit 0 having executed
nothing. `check_executed.py` does that.

Chart palette (still the notebook's convention): the categorical slots of the dataviz reference palette, used in its documented
configuration -- slots 1-3 where every pair must separate (scatter), up to slot 4 for lines and bars
where only adjacent pairs must. Aqua and yellow sit below 3:1 contrast on a light surface, so every
figure using them carries direct labels and the same numbers appear in a table.
"""
import nbformat as nbf

MD, CODE = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []


def md(text):
    cells.append(MD(text.strip()))


def code(text):
    cells.append(CODE(text.strip()))


# =================================================================================================
md(r"""
# Symmetry and Monte Carlo noise in DCA++

**How much sampling variance does point-group symmetrization remove from a quantum Monte Carlo
Green's function, where does the benefit come from, and what caps it.**

Cluster dynamical mean-field calculations can impose the lattice point group on the Green's function
`G`. Symmetrization averages `G` over each symmetry orbit; orbit mates are equal in expectation, so
the operation reduces variance *without changing the mean*. It is standard, it is nearly free, and
as far as we can tell it has only ever been documented qualitatively.

This notebook measures it. The measurement is a single ratio, `R`, and the interesting content is
not that `R > 1` but **how far above 1 it gets, and what stops it going further**. Two ceilings turn
out to govern everything: the size of the symmetry orbits, and how much of the Monte Carlo noise is
*already* symmetric before you do anything.

Every number below is measured on DCA++ CT-AUX. Sections 1-3 set up the estimator and what to expect
from it; section 4 is the results; sections 5-6 take them apart; sections 7-8 cover what we could not
run and the checks that justify the configuration.
""")

code("""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, '.')
import hero_lib as HL
import reduction as RED
import mechanism as MECH
import summaries as S

# Chart palette -- dataviz reference categorical slots, light surface.
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e5e4df'

plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 120,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'font.size': 9, 'legend.frameon': False,
})

MANIFEST = json.loads(Path('data/MANIFEST.json').read_text())
print('data bundle:')
for kind in ('runs', 'ops', 'summaries'):
    for name, rec in MANIFEST[kind].items():
        size = rec.get('size_mb') and f"{rec['size_mb']:.1f} MB" or f"{rec.get('size_kb', 0):.0f} KB"
        print(f'  {kind:10s} {name:28s} {size}')
""")

# =================================================================================================
md(r"""
---
## 1. What `R` is, and why it is the number

### 1.1 Definition

Let `G[x]` be the raw Monte Carlo estimate of the Green's function at entry `x`, where `x` ranges
over every measured `(band, band, momentum, frequency, spin)` index. Let `Sym` be the point-group
symmetrization operator. Then

$$R \;=\; \frac{\sum_x \operatorname{Var}\big(G[x]\big)}{\sum_x \operatorname{Var}\big(\mathrm{Sym}\,G[x]\big)}$$

**Summed over all entries — full support, no qualifier.** That includes entries the symmetry
annihilates outright. This is deliberate: run without symmetrization and you genuinely carry every
entry, including the symmetry-forbidden ones, and their noise propagates into everything downstream.
`R` is the reduction a user actually experiences.

### 1.2 Why a variance ratio

Because it is the quantity that converts into saved compute. Monte Carlo variance falls as `1/M` in
the number of measurements `M`, so cutting variance by a factor `R` is worth exactly `R` times more
measurements — and CT-AUX cost is linear in `M`. So:

> `R` is the factor by which you would have to lengthen an unsymmetrized run to match the error bars
> of a symmetrized one of the same length.

**Where the square root goes, since this is the easy thing to get wrong.** Error bars fall as
`1/√M`, not `1/M`, so it is natural to suspect the saving should be `√R`. Write `Var_raw(M) = A/M`
and `Var_sym(M) = B/M`, so that `R = A/B`. Setting the two error bars equal gives
`A/M_raw = B/M_sym`, hence `M_raw/M_sym = R`. The square root appears when converting variance to an
error bar and is squared away again when converting that error bar back into a measurement count.
Three equivalent statements:

| quantity | factor |
|---|---|
| variance | reduced by `R` |
| **error bar** | reduced by **`√R`** |
| measurements needed for equal error | increased by `R` |

Both numbers are worth carrying, because they sound very different. FeAs's `R = 3.04` is a 3×
saving in samples and a **43% smaller error bar**; square's `R = 1.26` is a 26% saving in samples and
an **11% smaller error bar**. Neither framing is more honest than the other, but quoting `R` alone
invites the reader to hear the error-bar improvement, which is always the smaller number.

Two conditions make the compute reading valid, and both hold here: the error must be
**variance-limited** rather than dominated by a systematic, and cost must be **linear in measurement
count**. Where those fail — a run dominated by warm-up, say — `R` still measures what it says, but
stops being a statement about wall clock.

### 1.3 Why the ratio has better properties than either variance alone

- **Dimensionless and scale-free.** Both variances are estimated from the *same* samples, so run
  twice as long and both halve. `R` is a property of the noise structure, not of how long you ran —
  a claim section 6.2 tests directly rather than assuming.
- **Paired.** `Sym` is a *deterministic linear operator*, so numerator and denominator are computed
  from the identical noise realizations. Common fluctuations cancel. This is dramatically tighter
  than comparing two independent symmetrization-on/off experiments, and it needs no special
  no-symmetry build of any model.
- **Bounded below by 1.** `Sym` is an orthogonal projector, so `Var(Sym G) <= Var(G)` entrywise.
  Symmetrization cannot hurt, and `R = 1` exactly when the noise is already fully symmetric.

### 1.4 Why we measure it on `G`

`G` is the simplest possible target and the most universal one. Symmetrization of `G` is **linear**,
which is what makes the paired estimator above valid — one deterministic operator applied to stored
samples. It is also the object everything else is built from, so a reduction map on `G` is the
transferable result: anyone wanting an observable-level number can reweight it.

The self-energy is the instructive counterexample. `Σ = G₀⁻¹ − G⁻¹` is **not** linear in `G`, so no
post-hoc linear operator recovers the gain — it would have to be re-symmetrized inside each jackknife
replicate, inside the solver. That is a code change, not an analysis, and it is out of scope here.

### 1.5 Scope — what this does and does not measure

| | |
|---|---|
| **Measures** | `Var(G)`, the sampling variance of the cluster Green's function |
| **Does not measure** | observable-level variance — `Σ`, susceptibilities, spectra, `G₄` |
| **Bath** | one solver call at the **bare bath** (`Σ = 0`), not a converged self-consistent loop. Section 8.1 shows this transfers |
| **Symmetries** | ±1 signed permutations only. DCA drops any symmetry operation whose orbital part is not a signed permutation, so 3-fold models (kagome, triangular) are out of scope rather than merely untested |
| **Solver** | CT-AUX, which represents only density-density interactions — the multi-orbital models here are density-density regardless of what their Hamiltonians declare |
""")

# =================================================================================================
md(r"""
---
## 2. How `R` is computed

### 2.1 Mechanics

Three things have to be true for the measurement to mean anything, and each is arranged explicitly.

**Raw samples must be genuinely raw.** The driver calls `local_G_k_w(symmetrize=false)` on every MPI
rank *before* `finalize()`. That yields a `G` untouched by symmetrization regardless of what point
group the binary declares — so the numerator is not quietly pre-reduced. The per-rank stack is
gathered to rank 0 and written out alongside the run.

**One rank is one independent sample.** DCA seeds per-rank RNG streams by `hash(global_id + seed)`,
so the 64 ranks of a run are 64 independent replicates of the same measurement. Variance across the
rank axis is therefore an honest sampling variance.

**The operator must be the real one.** `P` — the matrix that performs the orbit average on
`(b₀, b₁, k)` — is serialized by the driver from the simulation's own `cluster_symmetry` records and
shipped inside the run. We never infer orbits by looking for equal values of `G`: orbit mates can
carry a relative **sign**, and value-matching silently splits one signed orbit into two ± classes,
corrupting every orbit size and manufacturing a spurious perfect correlation.

With those in place, `R` is one line of numpy: form `Var` over the rank axis, apply `P`, form `Var`
again, and take the ratio of the sums.

### 2.2 How much of DCA++ had to be modified

Very little, and nothing that could manufacture the result. **Two header files, +50 / −7 lines.**

| change | file | what it does |
|---|---|---|
| `local_G_k_w(bool symmetrize = false)` | `ctaux_cluster_solver.hpp` | the method already existed, marked *"For testing purposes"*; we added the flag and threaded it into `compute_G_k_w_new` so it can skip `Symmetrize::execute` |
| `local_accumulated_sign()` | `ctaux_cluster_solver.hpp` | a 3-line accessor exposing the per-rank accumulated phase, so production's *global* sign normalization can be rebuilt from per-rank samples |
| a latent-bug fix | `ctaux_cluster_solver.hpp` | the pre-existing normalization line **never compiled** — the method had never been instantiated, since every real caller goes through `finalize()`. Corrected to the same `get_accumulated_phase()` form the proven `computeErrorBars` path uses |
| two optional loop hooks | `dca_loop.hpp` | no-ops unless a driver installs them; they let an out-of-tree driver sample each DCA iteration without reimplementing `execute()`. **Used only for the section 8.1 appendix**, never by the headline measurement |

**Production behavior is unchanged by construction.** Both new flags default so that every production
path takes exactly the branch it took before — `compute_G_k_w_new` still defaults to
`symmetrize = true`. Section 2.5's mean-preservation check is the empirical confirmation: our numpy
symmetrization reproduces production's finalized `G_k_w` to `4e-16`, which it could not do if the
patch had perturbed the path that produces it.

**What was *not* touched, because it matters for the claims above:** the random number generator and
its per-rank seeding, the Monte Carlo sampling, and the symmetrization operator itself. In particular
the seeding quoted above is stock DCA++ — `generateSeed(global_id, offset) = hash(global_id + offset)`
in `dca/math/random/random_utils.cpp`, unmodified — so rank independence is a property of the
simulator, not something we arranged.

Everything else lives out of tree: the driver, the per-model translation units, and the routine that
serializes `P` — which *replays* DCA's own derive-path symmetrization formula against its
`cluster_symmetry` records rather than changing it.
""")

code("""
sq = HL.Run('square_b5_c4')     # single-band square, D4, 4x4 cluster
fe = HL.Run('fe_as_b5_c4')      # two-band FeAs, 4x4 cluster

for r in (sq, fe):
    print(f'{r.model:10s}  ranks={r.n_ranks}  live point-group ops={r.n_ops}  '
          f'nb={r.nb} nk={r.nk} nw={r.nw}  flat entries E={r.E}  '
          f'measurements/rank={r.m_per_rank}')

print()
for r in (sq, fe):
    print(f'{r.model:10s}  single-run R = {r.R():.4f}')
""")

md(r"""
### 2.3 `R` has an error bar, and where it comes from

A single run gives a single `R`. The spread of `R` across independent runs is not small — and on a
model with a sign problem it is not even symmetric. So every headline number below is an **ensemble
over independent base seeds**, not a single run.

Two estimators of that uncertainty were available and they disagree, which is worth stating plainly:

- **Resampling ranks within one run** is cheap, but assumes the ranks are exchangeable draws from a
  well-behaved distribution. Where a model has a sign problem that fails — a resample of 64 ranks
  cannot see the heavy tail that a near-zero sign denominator creates — and the interval comes out
  optimistic.
- **Whole independent runs as the replicate unit** makes no such assumption.

**Everything quoted here uses independent base seeds.** One constraint is easy to violate and
invisible afterwards: a walker's stream is `hash(global_id + base_seed)` with
`global_id = local_id·n_ranks + proc_id`, so a run occupies the key range `[S, S + n_ranks·n_walkers)`.
Base seeds closer together than that share chains and the runs are *not* independent replicates. The
campaign used a stride of 10 000; the check is re-run below.
""")

code("""
for key, label in (('square_b1', 'square / D4, beta=1'), ('fe_as_b5', 'FeAs 2-band, beta=5')):
    h = S.headline(key)
    R = h['R_full']
    print(f"{label:24s}  R = {R['mean']:.4f} +/- {R['sem']:.4f}   "
          f"(n={R['n']} seeds, spread sd={R['sd']:.3f}, range {R['min']:.3f}-{R['max']:.3f})")
    print(f"{'':24s}  seed spacing ok: {h['seed_spacing_ok']}  (min gap {h['min_seed_gap']})")

# What a single run would have told you, for the same physical point.
print()
print(f'single run  vs  32-seed ensemble, FeAs:  {fe.R():.4f}  vs  '
      f"{S.headline('fe_as_b5')['R_full']['mean']:.4f}")
""")

md(r"""
The single run lands about a third of an across-seed *standard deviation* from the ensemble mean —
which is to say comfortably inside the run-to-run spread, but nearly two *standard errors* from the
ensemble's answer. One run tells you roughly where `R` is; it cannot tell you `R` to three digits.
That gap is the whole reason for the ensembles, and it is why the run shipped with this notebook
reproduces the headline approximately rather than exactly.

### 2.4 The configuration, and the choices in it

These are stated here, not argued. Each was settled by a separate measurement that is not part of
this story; section 8 carries the one justification that would otherwise look arbitrary.

| choice | value | why it is a choice at all |
|---|---|---|
| inverse temperature | **β = 5** (production point) | `R` depends strongly on β — section 6.1 |
| bath | **bare** (`Σ = 0`), one solver call | not a converged loop; justified in section 8.1 |
| ranks per run | 64 | the replicate axis |
| measurements per rank | 2048 | must clear a per-model depth floor — section 6.2 |
| seeds per ensemble | 32 (16 for square 8×8) | the precision axis |
| sweeps per measurement | 2 | held fixed, not swept |
| warm-up sweeps | 200 square, 80 FeAs, 8000 threeband | threeband's chain is not thermalized at 200; the residual dependence is disclosed in section 4 rather than corrected for |
| error computation | `NONE` | under `JACK_KNIFE`, `finalize` writes a leave-one-out replicate into `G_k_w`, which breaks the mean-preservation check below |

⚠️ **`measurements` in a DCA input is the TOTAL across ranks**, not per rank — it is divided by
`parallel::util::getWorkload`. The driver records both figures and everything here uses the per-rank
one. The FeAs run also predates the `beta` metadata key added on 2026-07-28, so its β comes from the
input template that generated it rather than from the file.

### 2.5 Is the instrument honest?

Four checks. The first three are exact — no statistics, no tolerance to argue about — and the fourth
ties our operator to the one the simulation actually applied.
""")

code("""
rows = []
for r in (sq, fe):
    ident = HL.projector_identities(r.P)
    rows.append((r.model, ident))
    print(f"{r.model:10s}  |P@P - P| = {ident['idempotence']:.2e}   "
          f"|P - P.T| = {ident['symmetry']:.2e}   "
          f"tr(P) = {ident['trace']:.6f} vs {ident['n_nonnull_orbits']} non-null orbits   "
          f"-> {'PASS' if ident['passes'] else 'FAIL'}")
""")

md(r"""
`P² = P` says symmetrizing twice is symmetrizing once. `P = Pᵀ` says the orbit average is an
**orthogonal** projector — this is the identity that licenses `Var(P G) ≤ Var(G)` entrywise and the
entire `r = m/[1+(m−1)ρ]` law below; a non-orthogonal averaging operator would have neither. And the
rank of a projector is its trace, so `tr(P)` must equal the number of non-null orbits exactly, which
simultaneously checks the forced-null bookkeeping.

Next, two statistical checks: symmetrization must **not move the mean**, and entries that have no
orbit mates must show no reduction at all.
""")

code("""
# Mean preservation: our numpy symmetrization of the raw ensemble mean must reproduce the finalized
# G_k_w that production wrote. On a model with a sign problem the invariant mean is the PHASE-WEIGHTED
# one -- production forms sum(sign_i G_i)/sum(sign_i), not the plain average of the per-rank points.
for r in (sq, fe):
    ours = HL.full_symmetrize(r, r.phase_weighted_mean())
    gap = np.abs(ours - r.G_final).max() / np.abs(r.G_final).max()
    print(f'{r.model:10s}  max relative gap vs production G_k_w = {gap:.2e}')

print()
# Singleton null control: an orbit of size 1 has nothing to average, so r must be exactly 1.
for r in (sq, fe):
    singles = [o for o in r.orbits() if not o['forced_null'] and len(o['members']) == 1]
    vr = r.variance_ratios()
    A = (vr['var_raw'].real + vr['var_raw'].imag).sum((0, 1))
    B = (vr['var_sym'].real + vr['var_sym'].imag).sum((0, 1))
    if not singles:
        print(f'{r.model:10s}  no singleton orbits — every entry has at least one mate '
              f'(or is a forced null)')
        continue
    ratios = [A[o['members'][0]] / B[o['members'][0]] for o in singles]
    print(f'{r.model:10s}  {len(singles)} singleton orbits, r = '
          f'{min(ratios):.12f} .. {max(ratios):.12f}')
""")

md(r"""
The mean is preserved to machine precision, so the variance reduction is genuinely free rather than
bought with bias. Square's singletons pin at exactly 1 to twelve digits — the estimator reports no
reduction where none is available, which is the control that would catch a bug inflating `R` across
the board.

FeAs has no singleton orbits to test: its band permutations give every non-null entry at least one
mate. That is itself the section 5 story arriving early — a second orbital removes the unmated
entries that a single-band model is stuck with.
""")

# =================================================================================================
md(r"""
---
## 3. What to expect from `R`

### 3.1 Averaging `m` mates does not divide variance by `m`

The naive expectation is that averaging an orbit of `m` symmetry-related entries cuts variance by
`m`. That is true only if the mates fluctuate **independently**. They do not.

For an orbit of size `m` whose members have pairwise noise correlation `ρ`, the variance of the
average is reduced by

$$r \;=\; \frac{m}{1 + (m-1)\,\rho}$$

This is the survey-sampling **design effect** (Kish 1965) — averaging `m` correlated samples gives an
effective sample size of `m/[1+(m−1)ρ]`, not `m`. We cite it rather than derive it; what is new here
is measuring `ρ` for quantum Monte Carlo noise under a lattice point group.

**Two ceilings follow immediately**, and between them they explain every result in this notebook:

$$r \;\longrightarrow\; \min\!\left(m, \; \frac{1}{\rho}\right)$$

- **The geometric ceiling `m`.** Reached only when `ρ = 0`. You cannot average more than you have.
- **The noise ceiling `1/ρ`.** As `m → ∞`, `r → 1/ρ`. If the mates are 90% correlated, no amount of
  orbit size gets you past about 1.1×.

A model is **`m`-limited** or **`ρ`-limited** depending on which binds, and the two call for opposite
remedies: bigger orbits versus less-correlated noise. Section 4 reports which binds for each design
point, because that — not `R` itself — is what tells you where to look next.
""")

code("""
fig, ax = plt.subplots(figsize=(5.4, 3.4))
m = np.arange(1, 33)
for rho, color, lab in ((0.0, BLUE, r'$\\rho=0$  (independent)'),
                        (0.1, ORANGE, r'$\\rho=0.1$'),
                        (0.5, AQUA, r'$\\rho=0.5$'),
                        (0.9, YELLOW, r'$\\rho=0.9$')):
    r_vals = HL.predicted_r(m, rho)
    ax.plot(m, r_vals, lw=2, color=color)
    ax.annotate(lab, (m[-1], r_vals[-1]), xytext=(4, 0), textcoords='offset points',
                va='center', fontsize=8, color=color)
    if rho > 0:
        ax.axhline(1 / rho, color=color, lw=0.8, ls=':', alpha=0.7)

ax.set_xlabel('orbit size $m$')
ax.set_ylabel(r'variance reduction $r$')
ax.set_title('Correlated mates cap the reduction well below $m$', loc='left', fontsize=10)
ax.set_xlim(1, 40); ax.set_ylim(0, 33)
ax.set_xticks([1, 8, 16, 24, 32])
plt.tight_layout(); plt.show()
""")

md(r"""
The dotted lines are the `1/ρ` ceilings. The blue line is the naive `r = m` expectation, and it is
the only one that keeps growing.

### 3.2 The headline `R` is a *harmonic* mean, so it sits below the best orbits

`R` is variance-weighted over all entries. Split the entries into classes `C` and the decomposition
is exact:

$$R \;=\; \Big(\sum_C \frac{w_C}{R_C}\Big)^{-1}, \qquad w_C = \text{class share of raw variance}$$

`R` is the variance-weighted **harmonic** mean of the class-wise `R_C`. Harmonic means are dominated
by their smallest terms, so **a class that is both noisy and poorly reduced pins the total down no
matter how well every other class does.**

This is structural, not a reporting preference, and it has one consequence worth internalising
before reading any number below: **a lone scalar `R` always sits below a model's best components and
always understates them.** FeAs is the clean illustration — its two `m=8` orbits reach `r = 4.76` and
`4.36`, and its interband class reduces at 4.4, while the aggregate is 3.0. The reason is in the
`w` column: 82% of the variance sits in the intraband class, which reduces at only 2.45.
""")

code("""
rows, harmonic, aggregate = RED.reduction_table(fe)
print(f'FeAs, by band character (single run, seed {fe.seed}):')
print(f"  {'class':<14} {'n':>4} {'var share':>10} {'R_C':>9} {'mean m':>7}")
for r in sorted(rows, key=lambda r: -r['w']):
    rc = f"{r['R_C']:.3f}" if np.isfinite(r['R_C']) else 'inf'
    print(f"  {r['cls']:<14} {r['n']:>4} {r['w']:>10.4f} {rc:>9} {r['mean_m']:>7.2f}")
print(f'\\n  aggregate R           = {aggregate:.4f}')
print(f'  harmonic reconstruction = {harmonic:.4f}   (gap {abs(aggregate-harmonic):.2e})')
""")

md(r"""
The reconstruction reproduces the aggregate exactly, which is the self-check that the decomposition
is the right one.

### 3.3 What the reduction looks like

The algebra above has a picture. Take one orbit at one Matsubara frequency and plot every rank's
estimate of every mate in the complex plane, with the orbit's signs applied so that odd mates are
aligned rather than pointing opposite ways. Symmetrization replaces each of those points with the
orbit average. So:

- the cloud **contracts** toward its own centre by `1/√r`;
- the centre **does not move** — that is mean preservation, visually;
- a **singleton** orbit shows no contraction at all, the visual null control.

Contrast a `ρ`-limited model with an `m`-limited one and the two ceilings become something you can
see: square's mates are so strongly correlated that the cloud barely tightens, while FeAs's `m=8`
orbit collapses.

⚠️ On a model with a sign problem the invariant centre is the **phase-weighted** one; the plain
average of the per-rank points would appear to move, and the contradiction would be an artifact of
the plot rather than physics.
""")

code("""
def pick_orbit(run, m_want):
    cands = [o for o in run.orbits() if not o['forced_null'] and len(o['members']) == m_want]
    return cands[len(cands) // 2] if cands else None

panels = [
    ('square / D4, $m=4$', sq, pick_orbit(sq, 4), BLUE),
    ('FeAs, $m=8$', fe, pick_orbit(fe, 8), ORANGE),
    ('square singleton (null control)', sq, pick_orbit(sq, 1), AQUA),
]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
for ax, (title, run, orb, color) in zip(axes, panels):
    mc = MECH.migration_cloud(run, orb)
    pts, sym = mc['points'].ravel(), mc['sym'].ravel()
    ax.scatter(pts.real, pts.imag, s=22, color=color, alpha=0.32, lw=0, label='raw, per rank')
    ax.scatter(sym.real, sym.imag, s=9, color=INK, alpha=0.8, lw=0, label='after symmetrization')
    ax.scatter([mc['centroid'].real], [mc['centroid'].imag], s=170, marker='+',
               color='white', lw=3.0, zorder=5)
    ax.scatter([mc['centroid'].real], [mc['centroid'].imag], s=170, marker='+',
               color=INK, lw=1.8, zorder=6)
    ax.set_title(f"{title}\\n$m$={mc['m']}, contraction {mc['contraction']:.2f}x",
                 loc='left', fontsize=9)
    ax.set_xlabel(r'Re $G$'); ax.set_aspect('equal', adjustable='datalim')
axes[0].set_ylabel(r'Im $G$')
axes[0].legend(fontsize=7.5, loc='upper left')
fig.suptitle('The cloud contracts; the centre (+) does not move', x=0.01, ha='left', fontsize=10)
plt.tight_layout(); plt.show()

for title, run, orb, _ in panels:
    mc = MECH.migration_cloud(run, orb)
    shift = abs(mc['sym_centroid'] - mc['centroid']) / abs(mc['centroid'])
    print(f"{title:32s}  m={mc['m']}  spread {mc['spread_raw']:.3e} -> {mc['spread_sym']:.3e}"
          f"   centre moved by {shift:.2e} (relative)")
""")

# =================================================================================================
md(r"""
---
## 4. The headline numbers

Four design points, all at the production configuration: β = 5, 4×4 cluster unless stated, 64 ranks,
|G| = 8 declared point-group operations, 32 independent seeds (16 for square 8×8).

`±` is the standard error across seeds. **`err ×`** is `√R`, the factor by which error bars shrink —
carried alongside `R`, the factor by which measurement count is saved (section 1.2). `1/ρ` is the
measured noise ceiling and `m-ceiling` the geometric one; **binds** names whichever is smaller, and
therefore what limits that model.
""")

code("""
sweep = {r['label']: r for r in S.design_points()}
fe_head = S.headline('fe_as_b5')
fe_ceilings = S.feas_ceilings()   # same definitions as the sweep, from FeAs's own ensembles

table = [
    ('square / D4, 4x4', sweep['square_b5_c4'], 1),
    ('FeAs 2-band, 4x4', None, 2),
    ('threeband (Emery d-p), 4x4', sweep['threeband_b5_c4'], 3),
    ('square / D4, 8x8', sweep['square_b5_c8'], 1),
]

print(f"{'design point':<28} {'nb':>3} {'Nc':>4} {'seeds':>6} {'R':>19} "
      f"{'err x':>6} {'1/rho':>7} {'m-ceil':>7} {'binds':>6}")
print('-' * 102)
for label, rec, nb in table:
    if rec is None:   # FeAs headline comes from its own 32-seed ensemble
        R, sem, n, nk = fe_head['R_full']['mean'], fe_head['R_full']['sem'], fe_head['n_seeds'], 16
        inv_rho, m_ceil, binds = (fe_ceilings['inv_rho'], fe_ceilings['m_ceiling'],
                                  fe_ceilings['binding'])
    else:
        R, sem, n, nk = rec['R_full']['mean'], rec['R_full']['sem'], rec['n_seeds'], rec['nk']
        inv_rho, m_ceil, binds = rec['inv_rho'], rec['m_ceiling'], rec['binding']
    # sqrt(R) is the error-bar factor; R is the measurement-count factor. See section 1.2.
    print(f'{label:<28} {nb:>3} {nk:>4} {n:>6} {R:>11.4f} +/- {sem:<5.4f} '
          f'{np.sqrt(R):>5.2f}x {inv_rho:>7.2f} {m_ceil:>7.2f} {binds:>6}')
""")

md(r"""
**Read the spread first.** At one temperature, on one cluster, with the same declared point group,
`R` runs from **1.26 to 3.24**. The single-band square-lattice Hubbard model — the most-studied case
there is — gets a 26% variance reduction from symmetrization. The three-band model gets 3.2×. This
is not a technique with a characteristic performance number; it is a measurement whose answer is a
property of the model.

**And the reason splits cleanly along band count.** Both single-band points are `ρ`-limited: square
4×4's geometry could support 2.97× but its noise correlation caps it at 1.51×. Both multi-orbital
points are `m`-limited, FeAs dramatically so — its noise structure would support **13.2×** and its
4×4 geometry cashes only 3.06×.

**These call for opposite fixes.** A `ρ`-limited model needs less-correlated noise, which means
colder physics (section 6.1). An `m`-limited model needs bigger orbits, which means a bigger cluster
or a larger group (section 7). Reading `R` alone would not tell you which; reading it against its two
ceilings does, and that is the practical output of this measurement.

Three qualifications, all stated rather than corrected for:

- **threeband carries a −1.4% warm-up systematic.** It runs at warm-up 8000 because its chain is not
  thermalized at the standard 200; where warm-up converges is not established, so an 8-seed arm at
  warm-up 2000 measures the residual dependence at `ΔR = −0.044 [−0.227, +0.139]` — the same order as
  the ±1.1% statistical error, and **not resolved**.
- **square 8×8 is 16 seeds, not 32** — its ensemble at 32 would have exceeded the compute budget.
- **The FeAs row comes from its own 32-seed ensemble** rather than the sweep. An independent
  ensemble at the same physical point, from disjoint seeds and a different warm-up, gives
  `2.968 ± 0.059` — a 0.96σ difference from `3.042 ± 0.049`. Two independent estimates agreeing at
  1σ is a robustness statement; we quote the first and use the second as the β-ladder's top rung in
  section 6.1.
""")

code("""
# Only the BINDING ceiling is plotted. Each point has two, and the larger one is not a constraint --
# showing both would also put FeAs's 1/rho = 13.2 off the top of a scale the bars need to share.
rows = []
for label, rec, nb in table:
    if rec is None:
        rows.append(('FeAs 2-band\\n4x4', fe_head['R_full']['mean'], fe_head['R_full']['sem'],
                     fe_ceilings['m_ceiling'], fe_ceilings['inv_rho'], fe_ceilings['binding']))
    else:
        short = {'square_b5_c4': 'square\\n4x4', 'threeband_b5_c4': 'threeband\\n4x4',
                 'square_b5_c8': 'square\\n8x8'}[rec['label']]
        rows.append((short, rec['R_full']['mean'], rec['R_full']['sem'],
                     rec['m_ceiling'], rec['inv_rho'], rec['binding']))

fig, ax = plt.subplots(figsize=(6.8, 3.4))
x = np.arange(len(rows))
ax.bar(x, [r[1] for r in rows], width=0.5, color=BLUE, zorder=3,
       yerr=[r[2] for r in rows], ecolor=INK, capsize=3)

for i, (lab, R, sem, mc, irho, binds) in enumerate(rows):
    ceiling = min(mc, irho)
    color = ORANGE if binds == 'm' else AQUA
    ax.plot([i - 0.3, i + 0.3], [ceiling, ceiling], color=color, lw=2.2, zorder=4)
    ax.annotate(f'{ceiling:.2f}', (i + 0.32, ceiling), fontsize=8, color=color, va='center')
    ax.annotate(f'{R:.2f}', (i, R / 2), ha='center', va='center', fontsize=9, color='white')

ax.plot([], [], color=ORANGE, lw=2.2, label='bound by geometry ($m$-ceiling)')
ax.plot([], [], color=AQUA, lw=2.2, label=r'bound by noise ($1/\\rho$)')
ax.axhline(1, color=MUTED, lw=0.9, ls='--')
ax.legend(fontsize=8, loc='upper left')
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylabel('$R$'); ax.set_ylim(0, 5.4)
ax.set_title(r'$R$ against the ceiling that binds it, all at $\\beta=5$', loc='left', fontsize=10)
plt.tight_layout(); plt.show()
""")

md(r"""
The gap between a bar and the *lower* of its two ticks is the reduction the model leaves on the
table. Square nearly saturates its noise ceiling — its problem is the ceiling, not the capture.
threeband sits well under its noise ceiling because geometry stops it first.

### 4.1 Does the per-orbit law actually hold?

Everything above rests on `r = m/[1+(m−1)ρ]`. It is worth confirming that the measured per-orbit
reduction matches the prediction from the *independently measured* mate correlation, rather than
assuming the design effect transfers to this setting.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, run, color, name in ((axes[0], sq, BLUE, 'square / D4'), (axes[1], fe, ORANGE, 'FeAs')):
    vr = run.variance_ratios()
    A = (vr['var_raw'].real + vr['var_raw'].imag).sum((0, 1))
    B = (vr['var_sym'].real + vr['var_sym'].imag).sum((0, 1))
    meas, pred, ms = [], [], []
    for o in run.orbits():
        if o['forced_null']:
            continue
        mem = o['members']
        rho = run.orbit_rho(mem, o['signs'])
        r_meas = A[mem].sum() / B[mem].sum()
        meas.append(r_meas)
        pred.append(HL.predicted_r(len(mem), 0.0 if np.isnan(rho) else rho))
        ms.append(len(mem))
    meas, pred, ms = np.array(meas), np.array(pred), np.array(ms)
    lim = max(pred.max(), meas.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.9, ls='--', zorder=1)
    ax.scatter(pred, meas, s=26 + 6 * ms, color=color, alpha=0.75, lw=0, zorder=3)
    dev = np.abs(meas - pred) / pred
    ax.set_title(f'{name}  ({len(ms)} orbits, max deviation {dev.max():.1%})',
                 loc='left', fontsize=9)
    ax.set_xlabel(r'predicted $m/[1+(m-1)\\rho]$'); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
axes[0].set_ylabel('measured $r$')
plt.tight_layout(); plt.show()
""")

md(r"""
Every orbit sits on the diagonal. The design effect is the right law for this noise, and `ρ` measured
directly from mate correlations predicts the reduction actually obtained. Marker size is orbit size.
""")

# =================================================================================================
md(r"""
---
## 5. The multi-orbital case, where the gains actually are

Band count is the single largest lever in the headline table: at fixed β, fixed cluster and the same
declared point group, `R` goes from **1.26 (nb=1) to 3.04 (nb=2) to 3.24 (nb=3)**. It is worth being
precise about *why*, because the mechanism is not the obvious one and the naive version of it is
wrong.

Three things happen at once when a model gains orbitals.

### 5.1 The extra symmetry is derived, not declared — so it is free

FeAs declares two symmetry operations in its input and ends up with **eight**. Nothing in the model
file supplies band-permutation symmetry: `orbitalPermutations()` returns an empty list on every stock
lattice, including FeAs. What actually happens is that DCA *solves* for the orbital operator `U_S`
from `H₀` for each candidate spatial operation, keeps the ones that work, and silently drops the
rest.

So the band content of the symmetry group is a **measured property of the Hamiltonian**, not an
input — and a user adding a multi-orbital model gets it without knowing it happened. It is also why
the ±1 scope limit in section 1.5 bites: an operation whose `U_S` is not a signed permutation is
dropped with no error, so the group can silently shrink.
""")

code("""
tb = HL.OpsView('threeband_b5_c4')     # operator only -- raw run too large to ship
c8 = HL.OpsView('square_b5_c8')

print(f"{'design point':<20} {'nb':>3} {'ops':>4} {'band classes':>16} "
      f"{'off-block share':>16} {'forced nulls':>13}")
print('-' * 78)
for label, o in (('square 4x4', sq), ('FeAs 4x4', fe), ('threeband 4x4', tb), ('square 8x8', c8)):
    classes = HL.band_equivalence_classes(o.P, o.labels, o.nb)
    off = HL.off_block_fraction(o.P, o.labels)
    nulls = sum(1 for x in HL.orbits_from_P(o.P) if x['forced_null'])
    print(f'{label:<20} {o.nb:>3} {o.n_ops:>4} {str(classes):>16} {off:>16.3f} {nulls:>13}')
""")

md(r"""
threeband's band-equivalence classes come out `[[0], [1, 2]]` — derived from the operator, never
declared: the d orbital is alone, the two p orbitals are equivalent to each other, and exactly two
band permutations are realized (identity and the p_x↔p_y transposition). **Remember this structure;
section 5.4 uses it as a control.**

### 5.2 Orbits get bigger, so the geometric ceiling rises

The symmetry group now acts on `(b₀, b₁, k)` rather than on `k` alone, so orbits can be larger than
pure momentum orbits allow.
""")

code("""
print(f"{'design point':<20} {'orbit sizes (non-null)':<34} {'largest m':>10}")
print('-' * 66)
for label, o in (('square 4x4', sq), ('FeAs 4x4', fe), ('threeband 4x4', tb), ('square 8x8', c8)):
    h = HL.orbit_size_histogram(HL.orbits_from_P(o.P))
    txt = '  '.join(f'm={k}: {v}' for k, v in h.items())
    print(f'{label:<20} {txt:<34} {max(h):>10}')
""")

md(r"""
On a 4×4 mesh every momentum lies on a mirror line, so **single-band square tops out at `m = 4`
despite having 8 group operations**. FeAs and threeband reach `m = 8` on the identical mesh, purely
because band permutations enlarge the orbits. (Square only reaches `m = 8` by going to an 8×8
cluster, which buys three free orbits — the last row.)

The aggregate consequence is the `m-ceiling` column of section 4, which weights these orbit sizes by
how much variance each entry actually carries: 2.97 for square 4×4 against 3.06 for FeAs and 3.61 for
threeband, on the identical mesh.

### 5.3 Symmetry-forbidden entries: noise driven to exactly zero

A multi-orbital model has entries that the group maps onto *minus themselves*. Their expectation is
exactly zero, so symmetrization does not average their noise down — it **removes all of it**.

threeband has 30 such entries carrying **20.5%** of the total raw variance; FeAs has 24, carrying
16%. In the harmonic decomposition of section 3.2 these classes have `R_C = ∞` and drop out of the
sum entirely, which is precisely how they lift the aggregate:

$$R_\text{full} = \frac{R_\text{non-null}}{1 - w_\text{null}}$$

So roughly a fifth of threeband's headline is noise on entries that symmetry simply deletes. That is
a real reduction a user experiences — those entries are carried, and their noise propagates, in any
unsymmetrized run — and it is the concrete reason the headline `R` is defined on full support.
""")

code("""
print(f'FeAs      w_null = {RED.null_weight(fe):.4f}   '
      f"R_full = {fe.R():.4f}  R_nonnull = "
      f"{RED.R_over(fe, [x for x in range(fe.E) if not RED.orbit_info(fe)[1][x]]):.4f}")

tb_rows = S.class_table('threeband_b5_c4')
print('\\nthreeband, R by entry class (32 seeds):')
print(f"  {'class':<32} {'n':>4} {'var share':>10} {'R_C':>16} {'mean m':>7}")
for r in tb_rows:
    rc = 'inf (exactly 0)' if not np.isfinite(r['R_C']) else f"{r['R_C']:.3f} +/- {r['R_C_sem']:.3f}"
    print(f"  {r['cls']:<32} {r['n']:>4} {r['w']:>10.3f} {rc:>16} {r['mean_m']:>7.2f}")
""")

md(r"""
### 5.4 The control: what actually decides whether a block benefits

This is the sharpest evidence in the notebook, because it is a *within-model* comparison — one
threeband run contains blocks the point group can permute and blocks it cannot, at exactly the same
β, `U`, filling and geometry. Nothing but symmetry differs.

**The obvious expectation is wrong.** One would guess that entries between *inequivalent* orbitals
(d–p) get no band-permutation benefit, since d is in a class by itself. The measurement refutes it:

| contrast | result |
|---|---|
| off-diagonal: inequivalent (d–p) − equivalent (p–p′) | `−0.045 [−0.163, +0.074]` — **not resolved** |
| diagonal: band-orbit > 1 (p–p) − band-orbit 1 (d–d) | `+1.105 [+1.039, +1.170]` — resolved |
| off-diagonal equivalent (p–p′) − diagonal band-orbit 1 (d–d) | `+1.967 [+1.864, +2.070]` — resolved |

d–p entries reduce **just as well** as the equivalent p–p′ ones. The mechanism is visible in the
orbit sizes: d–p orbits average `m = 5.00`, *above the 4 that momentum symmetry alone can reach on
this mesh*, because the p_x↔p_y permutation maps the d–p_x block onto the d–p_y block. An entry
*between* inequivalent bands is still reachable by a band permutation.

**The block that genuinely has no band symmetry is d–d** — the one class whose orbits are pure
momentum orbits, mean `m = 3.38`, exactly single-band square's. It lands at `R_C = 1.408 ± 0.010`
against single-band square's `1.262 ± 0.007` at the same β and geometry. Both ≈ 1.3, and a factor
1.8–2.4 below *every* permutation-reachable block.

> **The corrected claim.** The gain comes from symmetry-equivalent orbitals — but a single equivalent
> pair anywhere in the model lifts **every block that pair touches**, including blocks involving the
> inequivalent orbitals. Only blocks the permutation cannot reach at all stay at the single-band
> value.

This *widens* the result rather than narrowing it. A material does not need all its orbitals
equivalent to benefit; most of the index space of a realistic multi-orbital model is lifted by
whatever equivalent pair the point group does supply. And the ordering the mechanism predicts comes
out cleanly separable: off-diagonal (3.33–3.38) > band-diagonal but permutable (2.51) >
band-diagonal and unreachable (1.41).
""")

code("""
rows = [r for r in tb_rows if np.isfinite(r['R_C'])]
rows = sorted(rows, key=lambda r: r['R_C'])
fig, ax = plt.subplots(figsize=(7.4, 3.0))
y = np.arange(len(rows))
colors = [AQUA if 'orbit 1' in r['cls'] else BLUE for r in rows]
ax.barh(y, [r['R_C'] for r in rows], height=0.55, color=colors, zorder=3,
        xerr=[r['R_C_sem'] for r in rows], ecolor=INK, capsize=2.5)
for i, r in enumerate(rows):
    ax.annotate(f"{r['R_C']:.2f}   ({r['w']:.0%} of variance)", (r['R_C'], i), xytext=(6, 0),
                textcoords='offset points', va='center', fontsize=8, color=INK)
ax.axvline(1.2619, color=ORANGE, lw=1.2, ls='--', zorder=2)
ax.annotate('single-band square,\\nsame $\\\\beta$ and cluster', (1.2619, -0.75), xytext=(4, 0),
            textcoords='offset points', fontsize=8, color=ORANGE, va='center')
ax.set_yticks(y); ax.set_yticklabels([r['cls'] for r in rows], fontsize=8)
ax.set_ylim(-1.4, len(rows) - 0.4)
ax.set_xlabel('$R_C$'); ax.set_xlim(0, 5.2)
ax.set_title('threeband: only the block symmetry cannot reach\\nstays at the single-band value',
             loc='left', fontsize=10)
plt.tight_layout(); plt.show()
""")

md(r"""
### 5.5 Where threeband's aggregate actually comes from

The class table explains why the headline understates the model. Nearly half the raw variance (48%)
sits in the band-diagonal p blocks at `R_C = 2.51`, and another 20% in forced nulls. The two
off-diagonal classes, which reduce best at 3.3–3.4, carry only 26% between them. The harmonic mean
does the rest.

The aggregate is dominated by exactly the noisy, local-dominated, band-diagonal entries that
symmetrization helps *least* — which is a general property of variance-weighted reporting, not a
quirk of threeband. **If you care about a specific block, read its `R_C`, not the headline.**
""")

# =================================================================================================
md(r"""
---
## 6. Digging deeper

### 6.1 Temperature

`ρ` is the binding constraint on square, so the obvious question is whether square is *intrinsically*
`ρ`-limited or only looks that way at high temperature. The mechanism suggests the latter: `ρ` is
high when the noise is dominated by k-independent, spatially local fluctuations (section 6.3), and
lowering the temperature grows the correlation length, which should spread noise out in real space
and de-correlate the mates.

Two ladders, one per model. Square is sign-free at every β in CT-AUX, so it isolates the
correlation-length effect; FeAs has a live sign problem, so it tests whether the trend survives the
competing channel.
""")

code("""
lad_sq, lad_fe = S.beta_ladder('square'), S.beta_ladder('fe_as')

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
for ax, key, ylab, title in ((axes[0], 'R_full', '$R$', r'$R$ rises as the system cools'),
                             (axes[1], 'rho', r'mate correlation $\\rho$',
                              r'because $\\rho$ falls')):
    for lad, color, name in ((lad_sq, BLUE, 'square / D4'), (lad_fe, ORANGE, 'FeAs 2-band')):
        b = [r['beta'] for r in lad]
        v = [r[key] for r in lad]
        e = [r.get(key + '_sem', 0) for r in lad]
        ax.errorbar(b, v, yerr=e, lw=2, color=color, marker='o', ms=5, capsize=3, label=name)
        ax.annotate(name, (b[-1], v[-1]), xytext=(6, 0), textcoords='offset points',
                    fontsize=8, color=color, va='center')
    ax.set_xlabel(r'inverse temperature $\\beta$'); ax.set_ylabel(ylab)
    ax.set_title(title, loc='left', fontsize=10); ax.set_xlim(0.5, 9.5)
axes[0].axhline(1, color=MUTED, lw=0.9, ls='--')
plt.tight_layout(); plt.show()

for name, lad in (('square', lad_sq), ('FeAs', lad_fe)):
    print(f'{name}:')
    for r in lad:
        print(f"  beta={r['beta']:.0f}  R = {r['R_full']:.4f} +/- {r['R_full_sem']:.4f}   "
              f"rho = {r['rho']:.3f}   w_null = {r['w_null']:.3f}")
""")

md(r"""
**Both trends are monotone and resolved.** Square goes `1.041 → 1.343` over β = 1 → 8 as `ρ` falls
`0.94 → 0.56`; FeAs goes `1.526 → 2.968` over β = 1 → 5 as `ρ` falls `0.52 → 0.076`. So square is
*not* intrinsically `ρ`-limited — it is `ρ`-limited at accessible temperatures, and the constraint
relaxes steadily as the system cools.

The FeAs ladder carries the more surprising result. Sign-problem noise is a *perfectly* mate-correlated
channel (section 6.3), so as `⟨s⟩` collapses it should drive `ρ` **up**. Across the entire range where
FeAs is simulable it loses to the correlation-length effect: `ρ` falls monotonically even as `⟨s⟩`
reaches 0.148. The mechanism measured on square is therefore not an artifact of a sign-free corner.

Two honest qualifications:

- **Part of the `R_full` rise is `w_null`, not mechanism.** `w_null` grows `0.031 → 0.153` across the
  FeAs ladder, so while `R_full` rises ×1.945, `R_non-null` rises only ×1.698 — roughly a quarter of
  the headline rise is the growing forced-null share rather than better averaging.
- **β = 5 is at the edge of measurability for FeAs**, not a comfortable interior point: the
  across-seed error triples from 0.019 to 0.059 at the last rung. That is the method's boundary
  showing.

### 6.2 Depth — is `R` really scale-free?

Section 1.3 claimed `R` is independent of run length because both variances come from the same
samples. That is a claim to test, not assume: it is exactly true in the asymptotic regime and can
fail below it, because a short run's per-rank estimates are still heavy-tailed.
""")

code("""
depth = S.depth_ladder()
fig, ax = plt.subplots(figsize=(6.0, 3.3))
for model, color, name in (('fe_as', ORANGE, 'FeAs, $\\\\beta=5$'),
                           ('square_D4', BLUE, 'square, $\\\\beta=1$')):
    rows = [r for r in depth if r['model'] == model]
    m = [r['m_per_rank'] for r in rows]; v = [r['R_full'] for r in rows]
    e = [0 if np.isnan(r['sem']) else r['sem'] for r in rows]
    ax.errorbar(m, v, yerr=e, lw=2, marker='o', ms=5, color=color, capsize=3)
    ax.annotate(name, (m[-1], v[-1]), xytext=(6, 0), textcoords='offset points',
                fontsize=8, color=color, va='center')
ax.axvline(2048, color=MUTED, lw=1.0, ls='--', zorder=2)
ax.annotate('production depth', (2048, 0.6), xytext=(5, 0), textcoords='offset points',
            fontsize=8, color=MUTED)
ax.set_xscale('log'); ax.set_xlabel('measurements per rank'); ax.set_ylabel('$R$')
ax.set_ylim(0.5, 3.8)
ax.set_title('$R$ is flat in run length — above a per-model floor', loc='left', fontsize=10)
plt.tight_layout(); plt.show()

for r in depth:
    sem = '   —  ' if np.isnan(r['sem']) else f"{r['sem']:.4f}"
    print(f"{r['model']:8s} m/rank={r['m_per_rank']:>6}  n={r['n']}  R = {r['R_full']:.4f} +/- {sem}")
""")

md(r"""
⚠️ **These two ladders are at different temperatures** — this control predates the β work, so square
runs at β = 1 (where its `R` is 1.04, not the 1.26 of section 4) and FeAs at β = 5. That is fine for
the question being asked, which is about *flatness in depth*, not about the level. Do not read the
two curves against each other.

Each is flat above its own floor, and the floors differ by more than an order of magnitude: square
settles by about 64 measurements per rank, FeAs not until about 1000. So scale-freeness holds — a
single production-length run measures `R`, not `R`-at-that-length — but **only above a floor that has
to be re-established per model rather than assumed.**

The reason the floors differ is the reason the FeAs one is so much higher: above about β = 2 the
floor is set by the **sign problem**, not by autocorrelation. Below it, a near-zero sign denominator
makes the per-rank estimates heavy-tailed and `R` both biased and unstable. More seeds at shallow
depth buy nothing — they are more chances at a near-zero denominator, not more precision. The
production config sits at 2048 per rank, clear of both.

### 6.3 Where `ρ` comes from

This is the part that generalizes beyond the models measured here. `P` is an orthogonal projector, so
`ρ` is exactly the fraction of the noise already living in the symmetric subspace — and the question
is what puts it there.

**The scalar-channel argument.** Consider any fluctuating scalar property of the Monte Carlo
configuration: the expansion order, the sign, the total density, the local self-energy. Each is a
single number per configuration, so it enters the Green's function as

$$\delta G(k) \;=\; w(k)\,\delta X$$

with `w(k) = ∂G(k)/∂X`. Because `X` is a scalar function of the auxiliary field and the Hamiltonian
is symmetric, **`w(k)` is a symmetric function of `k`** — so every orbit mate receives *exactly the
same* fluctuation. Mate correlation for that channel is 1, and its contribution to the reducible
variance is identically **zero**.

> Whole classes of physical fluctuation are structurally invisible to any symmetrization, and `1/ρ`
> is where they cap it. This is not an inefficiency to engineer around — it is what symmetry cannot
> see, by construction.

**The corollary for the sign problem.** Sign-problem noise enters through a stochastic *denominator*,
`G = ⟨sign·M⟩/⟨sign⟩`, with weight `−G(k)`. That is a scalar channel, so it is a **perfectly**
symmetric one: symmetrization removes none of it. This is why a model deep in a sign problem cannot
be rescued by symmetrization. It also sets up a competition rather than a prediction: cooling lowers
`ρ` through the correlation length and raises it through the sign, and which wins is an empirical
question. Section 6.1 answers it — across the whole range where FeAs is simulable, the
correlation-length effect wins and `ρ` falls monotonically even as `⟨s⟩` collapses to 0.148.

Three diagnostics make the argument concrete rather than rhetorical.

**First: are symmetry mates special?** If the noise carried no particular alignment with the point
group, symmetry mates would correlate no more than any two random momenta, and symmetrization would
be fighting an ordinary correlated-sample problem. Compare the two directly.
""")

code("""
print(f"{'':10s} {'mate rho':>10} {'generic-pair rho':>18} {'ratio':>7}")
for run, name in ((sq, 'square'), (fe, 'FeAs')):
    mates = [run.orbit_rho(o['members'], o['signs'])
             for o in run.orbits() if not o['forced_null'] and len(o['members']) > 1]
    mate, generic = float(np.mean(mates)), MECH.pairwise_rho_all_k(run)
    print(f'{name:10s} {mate:>10.3f} {generic:>18.3f} {mate/generic:>6.1f}x')
""")

md(r"""
On square, symmetry mates correlate **6.2× more strongly** than generic momentum pairs — the noise is
sharply aligned with the point group, and it is exactly the component symmetrization cannot touch. On
FeAs the same ratio is 2.4×, and both numbers are small in absolute terms. The models differ in the
*symmetry structure* of their noise, not in their symmetry structure.

**Second: where does the noise live in real space?** A fluctuation at zero separation contributes
identically to every momentum, since `exp(i k·0) = 1` — so it is perfectly correlated across all `k`
and immune to any symmetrization. More generally, noise on *any* site the point group leaves fixed is
shared identically by every orbit mate. On a 4×4 torus there are two such sites: `(0,0)` and `(2,2)`.
""")

code("""
# Noise at r=0 is identical for every momentum (exp(i k.0) = 1), hence perfectly correlated across
# all k and immune to symmetrization. A spatially white profile would put 1/Nk in every cell.
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
for ax, run, name in ((axes[0], sq, 'square / D4'), (axes[1], fe, 'FeAs')):
    prof = MECH.realspace_noise_profile(run)
    im = ax.imshow(prof, cmap='Blues', vmin=0, vmax=prof.max())
    ax.set_title(f'{name}: $\\\\sigma^2(r)$ share\\nlocal share = {prof[0,0]:.1%}, '
                 f'white would be {1/run.nk:.1%}', loc='left', fontsize=9)
    ax.set_xticks(range(prof.shape[0])); ax.set_yticks(range(prof.shape[0]))
    ax.grid(False)
    for (i, j), v in np.ndenumerate(prof):
        ax.text(j, i, f'{v:.02f}', ha='center', va='center', fontsize=7,
                color='white' if v > prof.max() * 0.55 else INK)
plt.tight_layout(); plt.show()

print('noise power on the point-group-INVARIANT sites (immune to symmetrization by construction):')
for run, name in ((sq, 'square'), (fe, 'FeAs')):
    inv = MECH.invariant_site_share(run)
    print(f"  {name:8s}  sites {inv['sites']}  carry {inv['share']:.1%}   "
          f"(white noise would put {inv['white']:.1%} there)")
""")

md(r"""
This is the mechanism in one number. On square, **53% of the sampling noise sits on the two sites the
point group cannot move** — against the 12.5% a spatially white noise field would put there. That
noise is shared identically by every orbit mate, so symmetrization removes none of it, and it alone
is enough to pin `ρ` high. On FeAs the same two sites carry 16%, barely above white, and `ρ` is
correspondingly small.

The heat maps show where the rest lives. Square's is sharply concentrated; FeAs's is nearly uniform,
which is what noise carrying no memory of the point group looks like.

**Third: do symmetry-equivalent *sites* already fluctuate together, before anything is symmetrized?**
This is the source-level version of the question, asked of the sampling itself rather than of the
Green's function.
""")

code("""
print('real-space noise correlation, within a symmetry shell vs across shells:')
for run, name in ((sq, 'square'), (fe, 'FeAs')):
    sc = MECH.shell_correlations(run)
    print(f"  {name:8s}  within {sc['within']:+.3f}   across {sc['across']:+.3f}"
          f"   shell sizes {sc['shell_sizes']}")
""")

md(r"""
Square's noise at symmetry-equivalent sites is positively correlated (+0.19) while noise at
inequivalent sites is *anti*-correlated (−0.18) — a clean separation, and precisely the alignment
that leaves symmetrization little to remove. FeAs shows no such structure: within-shell correlation
is +0.03, actually *below* its across-shell value, which is what a noise field carrying no memory of
the point group looks like.

⚠️ These are β = 5 numbers, and the effect is strongly temperature dependent — the same square
diagnostic at β = 1, where `ρ = 0.94`, gives a within-shell correlation above +0.9. Section 6.1 is
the systematic version of this observation; the point here is the mechanism, not the magnitude.

One consequence inverts the intuitive reading and is worth stating plainly: **square's orbit geometry
is not its problem.** Its headroom is *larger* than FeAs's — square captures only about a third of the
reduction its orbit structure could support, while FeAs captures 84% of what its geometry allows.
Square's noise is simply already symmetric, and no amount of orbit engineering changes that.
""")

# =================================================================================================
md(r"""
---
## 7. What we could not run

**The measurement points at one run we could not afford, and it is a specific one.**

**Both multi-orbital points are `m`-limited**, threeband by a factor of nearly two (`1/ρ = 6.21`
against an `m`-ceiling of 3.61) and FeAs by a factor of four (13.2 against 3.06). They are starved of
exactly what a bigger cluster supplies. And the cluster axis is measured — going from 4×4 to 8×8 on
square adds three free `m = 8` orbits, lifting the geometric ceiling from 2.97 to 4.62.

But every point that has been run on 8×8 is single-band and `ρ`-limited, so **no measurement here
shows whether an already `m`-limited model converts extra orbit structure into `R`.** threeband on
8×8 would answer it, and on the ceiling arithmetic alone there is reason to expect `R > 5` — well
beyond anything measured here.

It costs roughly **30 hours per run** on this hardware, so an ensemble is on the order of weeks of
wall clock on a shared workstation. That is the blocker: not a question of method, just of compute.
A GPU build does not rescue it here — the available cards have less double-precision throughput than
the CPUs already in use.

Two smaller things the design also leaves open:

- **Where the β trend ends.** Both ladders stop where the sign problem does, not where the physics
  stops being interesting. FeAs at β = 5 already sits at the edge of measurability.
- **Observable-level variance.** `Σ` and `G₄` need re-symmetrization inside each jackknife replicate,
  inside the solver — a code change rather than an analysis. Everything here is `Var(G)`.
""")

# =================================================================================================
md(r"""
---
## 8. Appendix

### 8.1 Why the bare bath is a fair place to measure

Every number in this notebook comes from one solver call at the bare bath (`Σ = 0`), not from a
converged self-consistent loop. That is a real simplification and it needs a check, because a
production run spends its life at a *self-consistent* bath, not this one.

The check is direct: run an actual `DcaLoop` and measure `R` at every iteration. This one is **square
at β = 8** — 8 seeds, 10 iterations — chosen because it is the coldest sign-free point available, and
therefore the one where the bath moves furthest from bare over the course of the loop.
""")

code("""
bd = S.bath_drift()
it = bd['by_iteration']
fig, ax = plt.subplots(figsize=(6.0, 3.2))
k = [r['iteration'] for r in it]
v = [r['R_full'] for r in it]
e = [r['R_full_sem'] for r in it]
ax.errorbar(k, v, yerr=e, lw=2, marker='o', ms=5, color=BLUE, capsize=3)
ax.axhline(np.mean(v), color=MUTED, lw=0.9, ls='--')
ax.annotate(f'mean {np.mean(v):.3f}', (k[-1], np.mean(v)), xytext=(6, 0),
            textcoords='offset points', fontsize=8, color=MUTED, va='center')
ax.set_xlabel('DCA iteration'); ax.set_ylabel('$R$')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_title('$R$ does not drift as the self-consistent bath evolves', loc='left', fontsize=10)
plt.tight_layout(); plt.show()

p = bd['paired']
print(f"paired within-seed drift across iterations: {p['mean']:+.4f} "
      f"[{p['lo']:+.4f}, {p['hi']:+.4f}]  (n={p['n']})")
print(f"cost-weighted mean R over iterations: {bd['cost_weighted']['mean']:.4f} "
      f"+/- {bd['cost_weighted']['sem']:.4f}")
""")

md(r"""
The drift is consistent with zero. Iterations are serially dependent, so the interval quoted is the
**paired within-seed** one — ranks are matched streams across iterations, and pairing is what makes
the comparison legitimate.

Two things fall out. The bare-bath convention costs nothing: a single-iteration measurement transfers
to the iterations a production run actually performs. And the loop's cost-weighted `R = 1.342` lands
on top of the β = 8 rung of the independent square ladder in section 6.1 (`1.3429 ± 0.0088`) — two
different codepaths, two different seed sets, the same answer.

### 8.2 Reproducing this notebook

The bundle in `data/` is self-contained and everything above re-executes from it.

| what | how it is obtained here |
|---|---|
| orbit structure, band classes, forced nulls, orbit sizes, ceilings | recomputed from the shipped operator `P` for **all four** design points |
| validation ladder, per-orbit `r`, noise mechanism, migration figure, single-run `R` | recomputed from the shipped **raw per-rank samples** for square 4×4 and FeAs |
| every quoted `R ± ` | recomputed here from **per-seed records**; the per-seed `R` values themselves came from the full campaign |

The one thing the bundle cannot reproduce is a 32-seed ensemble from scratch — that needed the
campaign's full output, about 20 GB of raw run data, which is why the per-seed values ship
precomputed and the aggregation happens in front of you.
""")

# =================================================================================================
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata.kernelspec = {"display_name": "symm-variance (py)", "language": "python",
                          "name": "symm-variance"}
nbf.write(nb, "symmetrization_variance.ipynb")
print(f"wrote symmetrization_variance.ipynb  ({len(cells)} cells: "
      f"{sum(1 for c in cells if c.cell_type == 'markdown')} markdown, "
      f"{sum(1 for c in cells if c.cell_type == 'code')} code)")
