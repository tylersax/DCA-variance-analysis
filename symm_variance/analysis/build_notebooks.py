"""Generate the review notebooks from source, so they stay regenerable rather than hand-edited.

Usage:  python build_notebooks.py          (then execute with nbconvert -- see README)

Produces:
  01_validation_ladder.ipynb  -- does the pipeline measure what it claims?
  02_noise_mechanism.ipynb    -- why is rho what it is? (the figures)
"""
import nbformat as nbf

MD, CODE = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def write(nb, path):
    nb.metadata.kernelspec = {"display_name": "symm-variance (py)",
                              "language": "python", "name": "symm-variance"}
    nbf.write(nb, path)
    print("wrote", path)


# ======================================================================================
# Notebook 1 -- validation ladder
# ======================================================================================
nb = nbf.v4.new_notebook()
nb.cells = [
    MD(r"""# 1. Validation ladder — does the pipeline measure what it claims?

**What is being measured.** DCA++ can impose the lattice point group on the Green's function `G`.
Symmetrization averages `G` over each symmetry orbit; orbit-mates are equal in expectation, so this
**reduces variance without changing the mean**. We quantify that as

$$R \;=\; \frac{\sum_x \mathrm{Var}(G[x])}{\sum_x \mathrm{Var}(\mathrm{Sym}\,G[x])}$$

read as *"symmetrization removes MC noise equivalent to ~R× more measurements, for free."*

**Why one run suffices (the paired design).** Symmetrization `P` is a *deterministic linear
operator*. So from a single set of raw per-rank samples we form **both** `Var(G)` and `Var(P·G)` —
the same noise realizations appear in numerator and denominator. The ratio is therefore *paired*,
and fluctuations largely cancel. This is dramatically tighter than comparing two independent
ON/OFF experiments, and it needs no special "no-symmetry" build of any model.

**Where the samples come from.** One MPI rank = one independent sample (DCA seeds per-rank RNG
streams by `hash(global_id + seed)`). The driver calls `local_G_k_w(symmetrize=false)` on every
rank *before* `finalize()` — giving a truly raw `G`, independent of whatever point group the binary
declares — and gathers the stack to rank 0.

**The ladder.** Each rung licenses the next:
1. **Self-consistency** — singletons pinned at exactly 1; `r` matches `m/[1+(m−1)ρ]`; `r` flat in ω.
2. **Mean preservation** — our numpy symmetrization of the raw ensemble mean must reproduce
   production's finalized `G_k_w` to machine precision."""),

    CODE("""import sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from symm_variance_lib import Run, predicted_r, full_symmetrize

plt.rcParams['figure.dpi'] = 110

sq = Run('../runs/square_16rank.hdf5')
fe = Run('../runs/fe_as_16rank.hdf5')

for r in (sq, fe):
    print(f"{r.model:10s} ranks={r.n_ranks:3d}  live point-group ops={r.n_ops}  "
          f"nb={r.nb} nk={r.nk} nw={r.nw}  flat entries E={r.E}")"""),

    MD(r"""## The orbit structure

Orbits and their per-member signs are read from the **authoritative operator `P`** serialized by the
driver (built by replaying DCA's own derive-path symmetrization formula against the
`cluster_symmetry` records). We never infer orbits by looking for equal `G` values — orbit-mates can
carry a relative **sign**, and value-matching silently splits one signed orbit into two ± classes,
corrupting orbit sizes and manufacturing a spurious `ρ=1`.

Note `m ≤ |G|`, with equality only for *free* orbits. On a 4×4 mesh every k-point lies on a mirror
line, so square tops out at `m=4` despite `|G|=8`. FeAs reaches `m=8` on the same mesh because its
group acts on the combined `(b0,b1,k)` index — a stabilizing operation can still permute bands."""),

    CODE("""for run in (sq, fe):
    orbs = run.orbits()
    sizes = [len(o['members']) for o in orbs if not o['forced_null']]
    nulls = sum(o['forced_null'] for o in orbs)
    from collections import Counter
    print(f"{run.model:10s} {len(orbs):3d} orbits   size histogram {dict(sorted(Counter(sizes).items()))}"
          f"   symmetry-forced nulls: {nulls}")"""),

    MD(r"""## Rung 1a — the built-in null control

Singleton orbits (`m=1`) are ones where `P` acts as the **identity**. They must return `r = 1`
*exactly* — not "within error". Any deviation means the pipeline is wrong. This control is free and
requires no external oracle."""),

    CODE("""for run in (sq, fe):
    vr = run.variance_ratios()
    var_raw, var_sym = vr['var_raw'], vr['var_sym']
    worst = 0.0
    for o in run.orbits():
        if o['forced_null'] or len(o['members']) != 1:
            continue
        x = o['members'][0]
        worst = max(worst, np.nanmax(np.abs(var_raw.real[:, :, x] / var_sym.real[:, :, x] - 1.0)))
    print(f"{run.model:10s} max |r - 1| over singleton orbits = {worst:.3e}   "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")"""),

    MD(r"""## Rung 1b — per-orbit `r` against `m/[1+(m−1)ρ]`

For an orbit of `m` mates with per-sample variance `σ²` and mean pairwise noise correlation `ρ`, the
symmetrized entry is a mean of `m` correlated samples:

$$\mathrm{Var}(\mathrm{Sym}\,G) = \frac{\sigma^2}{m}\big[1+(m-1)\rho\big]
\;\;\Rightarrow\;\; r = \frac{m}{1+(m-1)\rho}.$$

> **Honest caveat.** For an orbit with equal-variance mates and homogeneous pairwise correlation this
> is an *algebraic identity* on the sample moments, and we measure `ρ` from the very same samples. So
> agreement here is a strong check that orbits are genuinely exchangeable and that membership and
> **signs** are right (wrong signs break it badly) — a **pipeline** check, not independent evidence of
> the physics. The genuinely independent test is Rung 2 below.

Note `ρ` is bounded by the *symmetry-mate* correlation, not the generic one — see notebook 02."""),

    CODE("""def orbit_table(run):
    vr = run.variance_ratios()
    var_raw, var_sym = vr['var_raw'], vr['var_sym']
    rows = []
    for o in sorted(run.orbits(), key=lambda z: len(z['members'])):
        if o['forced_null'] or len(o['members']) < 2:
            continue
        m = len(o['members'])
        rho = run.orbit_rho(o['members'], o['signs'])
        xs = o['members']
        num = var_raw.real[:, :, xs].mean() + var_raw.imag[:, :, xs].mean()
        den = var_sym.real[:, :, xs].mean() + var_sym.imag[:, :, xs].mean()
        rows.append((m, rho, predicted_r(m, rho), num / den))
    return rows

tables = {}
for run in (sq, fe):
    tables[run.model] = orbit_table(run)
    print(f"\\n{run.model}")
    print(f"  {'m':>3} {'rho':>8} {'r_pred':>8} {'r_meas':>8} {'ratio':>7}")
    for m, rho, rp, rm in tables[run.model]:
        print(f"  {m:>3} {rho:>8.4f} {rp:>8.3f} {rm:>8.3f} {rm/rp:>7.3f}")"""),

    CODE("""fig, ax = plt.subplots(figsize=(5.2, 5))
for (name, rows), mk, col in zip(tables.items(), ('o', 's'), ('#0072B2', '#D55E00')):
    ax.scatter([r[2] for r in rows], [r[3] for r in rows], marker=mk, s=70,
               label=name, color=col, alpha=.85, edgecolor='k', linewidth=.5)
lim = [0.9, 6]
ax.plot(lim, lim, 'k--', lw=1, label='y = x')
ax.set(xlabel=r'predicted  $r = m/[1+(m-1)\\rho]$', ylabel='measured  $r$',
       xlim=lim, ylim=lim, title='Per-orbit variance reduction')
ax.legend(); ax.grid(alpha=.3); plt.tight_layout()"""),

    MD(r"""## Rung 1c — `r` must be flat in ω

The point group acts on band and momentum indices only; it **does not touch Matsubara frequency**.
So `r` must be ω-independent. A slope here would signal contamination — e.g. frequency-domain
symmetrization leaking into what we are attributing to the point group."""),

    CODE("""fig, ax = plt.subplots(figsize=(7, 3.4))
for run, col in zip((sq, fe), ('#0072B2', '#D55E00')):
    vr = run.variance_ratios()
    var_raw, var_sym = vr['var_raw'], vr['var_sym']
    mask = np.zeros(run.E, bool)
    for o in run.orbits():
        if not o['forced_null'] and len(o['members']) > 1:
            mask[o['members']] = True
    rw = ((var_raw.real[:, :, mask].sum((1, 2)) + var_raw.imag[:, :, mask].sum((1, 2))) /
          (var_sym.real[:, :, mask].sum((1, 2)) + var_sym.imag[:, :, mask].sum((1, 2))))
    slope = np.polyfit(np.arange(run.nw), rw, 1)[0]
    ax.plot(np.linspace(0, 1, run.nw), rw, color=col,
            label=f"{run.model}: mean {rw.mean():.3f}, slope/step {slope:.1e}")
ax.set(xlabel=r'Matsubara index (normalized)', ylabel='pooled $r$', title=r'$r$ vs $\\omega$ — expect flat')
ax.legend(); ax.grid(alpha=.3); plt.tight_layout()"""),

    MD(r"""## Headline `R`, on full and non-null support

`R` is **variance-weighted**: the noisiest entries dominate, which is the fair "total noise removed"
reading. But some entries are **symmetry-forced nulls** — the group sends them to ~0, so
`Var(Sym G) ≈ 0` and their per-entry reduction is formally infinite. The summed form tolerates them,
but they can inflate `R`, so we report both supports to make their contribution visible rather than
hidden."""),

    CODE("""for run in (sq, fe):
    vr = run.variance_ratios()
    var_raw, var_sym = vr['var_raw'], vr['var_sym']
    nulls = set()
    for o in run.orbits():
        if o['forced_null']:
            nulls.update(o['members'])
    nonnull = np.array([x not in nulls for x in range(run.E)])
    full = np.ones(run.E, bool)

    def R_over(mask):
        return ((var_raw.real[:, :, mask].sum() + var_raw.imag[:, :, mask].sum()) /
                (var_sym.real[:, :, mask].sum() + var_sym.imag[:, :, mask].sum()))
    print(f"{run.model:10s} R(full support) = {R_over(full):.4f}    "
          f"R(non-null support) = {R_over(nonnull):.4f}")"""),

    MD(r"""## Rung 2 — mean preservation (the independent test)

This is the rung that is *not* an identity. We take the raw per-rank samples, form the ensemble mean,
apply **our own** numpy reconstruction of the full production symmetrization, and compare against the
`G_k_w` that DCA actually finalized. Agreement to machine precision proves simultaneously that our
operator `P` matches production's `Symmetrize`, that the serialized orbit table is right, and that
symmetrization **does not move the mean**.

Two subtleties:
- `Symmetrize::execute` bundles **three** reductions — spin, point-group (cluster), and frequency
  (ω↔−ω). `full_symmetrize` reproduces all three in production order; the headline `R` above isolates
  the point-group piece.
- We use the **phase-weighted** mean `Σᵢ(signᵢ·Gᵢ)/Σᵢ(signᵢ)`, matching production's *global* sign
  normalization (`local_G_k_w` divides each rank by its *own* sign). For a sign-free model these
  coincide; they diverge once a sign problem develops.

⚠️ This rung requires `error-computation-type = NONE`. Under `JACK_KNIFE`, `finalize` writes a
*leave-one-out* replicate into `G_k_w`, which shows up as a ~3% low-frequency gap — a jackknife
artifact, not a pipeline error."""),

    CODE("""for run in (sq, fe):
    G_pw = full_symmetrize(run, run.phase_weighted_mean(), complex_g0=False)
    d = np.abs(G_pw - run.G_final).max()
    rel = d / np.abs(run.G_final).max()
    print(f"{run.model:10s} max|full_symmetrize(raw mean) - finalized G_k_w| = {d:.3e}  "
          f"(rel {rel:.2e})  {'PASS' if rel < 1e-8 else 'FAIL'}")"""),

    MD(r"""## Summary

All rungs pass on both models. The measured reductions are consistent with the plan's predictions:
square's single-band orbit-mates are strongly correlated (`ρ ≈ 0.9`) so the gain is modest, while
FeAs's band-mixing orbits have near-independent mates (`ρ ≈ 0`) so `r` approaches the full orbit size.

**Why the two models differ so much is the subject of notebook 02.**"""),
]
write(nb, "01_validation_ladder.ipynb")


# ======================================================================================
# Notebook 2 -- noise mechanism
# ======================================================================================
nb = nbf.v4.new_notebook()
nb.cells = [
    MD(r"""# 2. Where the correlation comes from — why `ρ` is what it is

Notebook 01 established *how much* variance symmetrization removes. This one asks **why**, because
that is what tells us where symmetrization will and won't pay off.

**The framing.** Symmetrization `P` is an **orthogonal projector** onto the symmetry-invariant
subspace. It deletes the symmetry-**breaking** part of the MC noise entirely and leaves the
symmetry-**preserving** part untouched. So

$$\rho \;=\; \text{the fraction of noise variance already living in the symmetric subspace},$$

and `r = m/[1+(m−1)ρ]` carries **two ceilings**:

| ceiling | reached when | meaning |
|---|---|---|
| `r ≤ m` | `ρ = 0` | finite orbit size — you only average `m` things |
| `r ≤ 1/ρ` | `m → ∞` | correlation floor — symmetric noise is untouchable |

roughly `r ≈ min(m, 1/ρ)`. **Which ceiling binds tells you what to do:** a `ρ`-limited system gains
nothing from a bigger group or cluster, while an `m`-limited system gains almost directly.

*(Note the ceiling is `m`, not `√m` — symmetrization divides **variance** by `m` and the **error bar**
by `√m`. `R` is a variance ratio.)*"""),

    CODE("""import sys
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from symm_variance_lib import Run
import noise_diagnostics as nd

plt.rcParams['figure.dpi'] = 110
sq = Run('../runs/square_16rank.hdf5')
fe = Run('../runs/fe_as_16rank.hdf5')

# blocks worth contrasting: (run, b0, b1, band-class, label)
# Band-permuting ops map band-diagonal entries to band-diagonal ones and off-diagonal to
# off-diagonal, so every orbit is cleanly 'intra' or 'inter' -- we can report mate-rho per class.
BLOCKS = [(sq, 0, 0, 'intra', 'square/D4  band-diag'),
          (fe, 0, 0, 'intra', 'FeAs  intraband'),
          (fe, 0, 1, 'inter', 'FeAs  INTERBAND')]

def mate_rho_by_class(run):
    \"\"\"Mean symmetry-mate rho, split by whether the orbit lives in band-diagonal or
    band-off-diagonal entries.\"\"\"
    out = {}
    for o in run.orbits():
        if o['forced_null'] or len(o['members']) < 2:
            continue
        cls = {'intra' if run.labels[x][0] == run.labels[x][1] else 'inter' for x in o['members']}
        if len(cls) != 1:
            continue  # mixed orbit -- not expected for signed permutations
        out.setdefault(cls.pop(), []).append(run.orbit_rho(o['members'], o['signs']))
    return {k: float(np.nanmean(v)) for k, v in out.items()}

MATE_RHO = {id(sq): mate_rho_by_class(sq), id(fe): mate_rho_by_class(fe)}"""),

    MD(r"""## The tell: symmetry mates are far more correlated than generic pairs

If the noise were structureless, symmetry-related momenta would be no more correlated than any other
pair. They are — dramatically so for square. That gap *is* the symmetric-noise fraction, and it is
what symmetrization cannot touch."""),

    CODE("""print(f"{'block':26s} {'rho (generic k-pairs)':>22s} {'rho (symmetry mates)':>22s} {'gap':>8s}")
for run, b0, b1, cls, label in BLOCKS:
    generic = nd.pairwise_rho_all_k(run, b0, b1)
    mates = MATE_RHO[id(run)][cls]
    print(f"{label:26s} {generic:>22.4f} {mates:>22.4f} {mates-generic:>+8.4f}")"""),

    CODE("""fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, (run, b0, b1, cls, label) in zip(axes, BLOCKS):
    C = nd.noise_corr_matrix(run, b0, b1)
    im = ax.imshow(C, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set(title=f'{label}\\nnoise correlation across k', xlabel='k index', ylabel='k index')
    plt.colorbar(im, ax=ax, fraction=.046)
plt.suptitle('Pre-symmetrization noise correlation matrix of $\\\\delta G$', y=1.04)
plt.tight_layout()"""),

    MD(r"""## The source: the real-space noise profile `σ²(r)`

Fourier-transform the noise back to real space. Noise sitting at `r = 0` (the **local** component)
contributes identically to *every* momentum, since `e^{ik·0} = 1`. It is therefore perfectly
correlated across all `k` and completely immune to symmetrization.

A spatially **white** noise profile would put `1/N_k` in every cell (here `1/16 = 0.0625`). The
excess above that baseline is what drives `ρ`."""),

    CODE("""fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, (run, b0, b1, cls, label) in zip(axes, BLOCKS):
    prof = nd.realspace_noise_profile(run, b0, b1)
    im = ax.imshow(prof, cmap='viridis', vmin=0)
    for (i, j), v in np.ndenumerate(prof):
        ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                color='w' if v < prof.max()*.6 else 'k', fontsize=7)
    ax.set(title=f'{label}\\n$\\\\sigma^2(r=0)$ share = {prof[0,0]:.3f}   (white = {1/run.nk:.3f})',
           xlabel='$r_2$', ylabel='$r_1$')
    plt.colorbar(im, ax=ax, fraction=.046)
plt.suptitle('Real-space MC noise power profile $\\\\sigma^2(r)$ / total', y=1.04)
plt.tight_layout()"""),

    MD(r"""### A quantitative predictor

Model the profile as a uniform background plus a local spike at `r=0`. Then for `q ≠ 0` the
normalized Fourier transform gives

$$\rho \;\approx\; \frac{\sigma^2(r=0)}{\text{total}} \;-\; \frac{1}{N_k}.$$"""),

    CODE("""print(f"{'block':26s} {'predicted rho':>14s} {'measured rho':>14s}")
for run, b0, b1, cls, label in BLOCKS:
    print(f"{label:26s} {nd.predicted_rho(run, b0, b1):>14.4f} "
          f"{nd.pairwise_rho_all_k(run, b0, b1):>14.4f}")"""),

    MD(r"""## The clincher: correlation between symmetry-equivalent *sites*

The most direct statement of the mechanism. Partition the real-space cluster into D4 shells and ask
whether `δM(r)` and `δM(S·r)` — the noise at symmetry-**equivalent sites** — already fluctuate
together. Symmetrization averages exactly those sites, so if they are already correlated, there is
nothing left to remove.

Note that square's *across*-shell correlation is **negative**, confirming this is specifically a
symmetry-aligned effect rather than generic common-mode noise."""),

    CODE("""labels, wi, ac = [], [], []
for run, b0, b1, cls, label in BLOCKS:
    s = nd.shell_correlations(run, b0, b1)
    labels.append(label); wi.append(s['within']); ac.append(s['across'])
    print(f"{label:26s} within-shell {s['within']:+.4f}   across-shell {s['across']:+.4f}   "
          f"shell sizes {s['shell_sizes']}")

x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.bar(x - w/2, wi, w, label='within symmetry shell  (r vs S·r)', color='#D55E00')
ax.bar(x + w/2, ac, w, label='across different shells', color='#56B4E9')
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set(ylabel='real-space noise correlation',
       title='The correlation that defeats symmetrization is already in the real-space noise')
ax.legend(); ax.grid(alpha=.3, axis='y'); plt.tight_layout()"""),

    MD(r"""## What this predicts

**Physical reading.** Noise dominated by fluctuations in globally **symmetric scalars** — expansion
order, average sign, density — is *entirely* symmetric, so `P` cannot touch it. Square at `β=1` is
exactly that regime: very high temperature, near-local physics, ~49% of all noise power in the single
local component. FeAs, with two bands and richer structure, has essentially white interband noise.

**Consequences:**

- **square/D4 at β=1 is ρ-limited** (`1/ρ ≈ 1.1`). Going to an 8×8 cluster would raise `m` from 4 to 8
  and buy almost nothing — the correlation ceiling binds first. To help square you must change the
  *noise structure*, which is what the β ladder tests: at lower temperature, correlations extend, the
  noise should acquire symmetry-breaking structure, `ρ` should fall and `R` should rise.
- **FeAs is m-limited** (`1/ρ ≈ 20`). Here bigger clusters, larger point groups, and band-permuting
  operations pay off almost directly.

**Where to expect maximum `R`:** multi-band models with band-permuting symmetry, on large clusters
with large point groups (2D `D4`=8, `D6`=12; 3D `O_h`=48), in entries where the noise is not
local-dominated. Since `R` is variance-weighted, the noisiest entries dominate it — and in square
those are precisely the local-dominated, high-`ρ` ones, which is why `R` lands at 1.03."""),
]
write(nb, "02_noise_mechanism.ipynb")
