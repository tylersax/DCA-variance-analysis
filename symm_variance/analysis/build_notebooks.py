"""Generate the review notebooks from source, so they stay regenerable rather than hand-edited.

Usage:  python build_notebooks.py          (then execute with nbconvert -- see README)

Produces:
  01_validation_ladder.ipynb  -- does the pipeline measure what it claims?
  02_noise_mechanism.ipynb    -- why is rho what it is? (the figures)
  03_m_scaling.ipynb          -- is R independent of run depth? (the depth floor)
  04_beta_ladder.ipynb        -- is square intrinsically rho-limited, or only at beta=1?
  05_beta_cross_model.ipynb   -- does the beta trend reproduce on a second model?
  06_model_sweep.ipynb        -- does R grow with symmetry-EQUIVALENT orbitals?
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

    MD(r"""> ### ⚠️ These are 16-rank values — **not** the numbers to quote
>
> This notebook reads the two committed 16-rank runs, so the `R` printed above is a **single draw**
> from each. Bootstrapping over those 16 ranks gives
>
> | | printed above | 95% CI from the same 16 ranks | **quote this instead** |
> |---|---|---|---|
> | square/D4 | 1.0345 | `[1.024, 1.058]` | **1.0425 ± 0.0013** |
> | FeAs | 2.4821 | `[2.13, 3.51]` | **3.042 ± 0.049** |
>
> The headline `R` is the **full-support** number — the left-hand column above, and the reduction a
> user actually experiences. Non-null is an internal structural diagnostic, not a second headline.
>
> Square's is fine — ±1.6%. **FeAs's is not**: that interval spans a factor of 1.65, and 2.4821 is a
> low draw from it. The values to quote come from the **seed ensemble** — 32 independent base seeds
> × 64 ranks × 2048 measurements/rank, each checked against a whole-run oracle and a paired depth
> test (`analysis/seed_ensemble.py`, `runs/seed_ensemble_*.json`). The shift from 2.4821 is
> *sampling noise*, not a correction of bias — FeAs's `R` is flat in depth, which is what
> **`03_m_scaling.ipynb`** establishes.
>
> The rungs below are unaffected: they are identities and machine-precision checks, not estimates."""),

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
those are precisely the local-dominated, high-`ρ` ones, which is why `R` lands at 1.04.

> **Note on numbers.** Like notebook 01, this one reads the committed 16-rank runs, so every `ρ` and
> `r` here is a single 16-rank draw. The mechanism they demonstrate is robust — the ordering,
> the mate-vs-generic gap and the shell structure all reproduce at 64 ranks — but for quotable
> values use the seed ensemble (`runs/seed_ensemble_*.json`): square `R = 1.0425 ± 0.0013`,
> FeAs `3.042 ± 0.049`."""),
]
write(nb, "02_noise_mechanism.ipynb")


# ======================================================================================
# Notebook 3 -- M-scaling control (ROADMAP task 1)
# ======================================================================================
nb = nbf.v4.new_notebook()
nb.cells = [
    MD(r"""# 3. M-scaling control — is `R` really independent of run depth?

Notebooks 01 and 02 measured `R` and explained `ρ`. Both quote a single number per model, and the
budget strategy behind the whole project — *many ranks, modest depth, and the measured `R` transfers
to any production scale* — rests on `R` being **scale-free**:

$$\mathrm{Var}(\hat G_N)=\Sigma/N,\qquad \mathrm{Var}(P\hat G_N)=P\Sigma P^{\!\top}/N
\;\Longrightarrow\; R \text{ has no } N.$$

That cancellation assumes each rank's sample is in the **CLT regime** — past thermalization and well
beyond the autocorrelation time. Below that threshold `Var(G_i)` is not `Σ/m`, the raw and
symmetrized arms need not approach their asymptotes at the same rate, and `R` can drift with depth.
This notebook measures the drift instead of assuming it away.

**Depth means measurements PER RANK.** A rank is one sample, so its own depth is what decides whether
that sample is asymptotic. DCA's `measurements` input key is the **total over ranks** (every solver
routes it through `parallel::util::getWorkload`), so the driver now records both numbers and
`run_symm_variance.sh` takes the per-rank figure. Runs made before 2026-07-27 stored the total under
the per-rank key — the committed 16-rank "4000" runs are **250 measurements per rank**.

**Design.** At fixed point group, model, and rank count (64), sweep `m ∈ {4,16,64,256,1024,4096}`
per rank at three base seeds. Two things are tracked:

- `R` itself, with a bootstrap-over-ranks CI, and a **paired** CI on the step-to-step difference.
  Pairing is legitimate here: a rank's walker streams are seeded by `hash(global_id + base_seed)` and
  do not depend on `measurements`, so at fixed seed the depth-`m` chain is a **prefix** of the
  depth-`4m` chain on the same rank. `nesting_corr` below verifies that empirically against its
  predicted value `sqrt(m/m')`.
- `m·ΣVar`, which is flat in `m` **iff** the per-rank sample obeys the `1/m` law. This is the
  stricter test: `R` is a ratio, so a departure from `1/m` common to both arms cancels in it, while
  `m·ΣVar` catches the common part too."""),

    CODE("""import sys, json
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 110
S = json.load(open('../runs/m_scaling_summary.json'))
runs = S['runs']
MODELS = sorted({r['model'] for r in runs})
SEEDS = sorted({r['seed'] for r in runs})
print(f"{len(runs)} runs | models {MODELS} | seeds {SEEDS} | "
      f"depths {sorted({r['m'] for r in runs})} | ranks {sorted({r['n_ranks'] for r in runs})}")
print(f"unusable (a rank's accumulated sign hit zero): {sum(1 for r in runs if not r['usable'])}")"""),

    MD(r"""## The headline number

**This is the canonical source for `R`.** Notebooks 01 and 02 read the two committed 16-rank runs and
print single draws — fine for square (±1.6%), badly imprecise for FeAs (95% CI `[2.13, 3.51]`). The
values below average every run above each model's depth floor and quote the standard error across
base seeds, which for FeAs is the dominant uncertainty.

**`R` is on the full support**, summed over every entry. That is the reduction a user actually
experiences: run unsymmetrized and you carry all of them, including the symmetry-forbidden entries,
and their noise propagates downstream. One number, no qualifier. The two right-hand columns are
internal diagnostics — non-null drops the forced-zero entries (related exactly by
`R = R_non-null/(1 − w_null)`, so it adds only `w_null`), and efficiency needs `R_ideal` and is
definable only on the non-null support. Useful for deciding what to do about a model; not for
reporting."""),

    CODE("""FLOOR = {'square_D4': 64, 'fe_as': 1024}   # per-rank depth floor, established below
print(f"{'model':<10} {'floor':>6} {'runs':>5} {'seeds':>6} {'R (HEADLINE)':>16} "
      f"{'non-null [int]':>16} {'effic. [int]':>14}")
for model in MODELS:
    g = [r for r in runs if r['usable'] and r['model'] == model and r['m'] >= FLOOR[model]]
    by_seed = {}
    for r in g:
        by_seed.setdefault(r['seed'], []).append(r)
    def agg(key):
        per = np.array([np.mean([r[key] for r in v]) for v in by_seed.values()])
        return per.mean(), per.std(ddof=1) / np.sqrt(len(per))
    (rf, ef), (rn, en), (ff, fe_) = agg('R_full'), agg('R_nonnull'), agg('efficiency')
    print(f"{model:<10} {FLOOR[model]:>6} {len(g):>5} {len(by_seed):>6} "
          f"{rf:>9.3f} ± {ef:<5.3f} {rn:>9.3f} ± {en:<5.3f} {ff:>8.1%} ± {fe_:<5.1%}")"""),

    MD(r"""## `R` vs depth

Each panel is one model; each line one base seed; the band is the 95% bootstrap-over-ranks CI. The
grey band marks the mean of the deepest two rungs — the working asymptote."""),

    CODE("""fig, axes = plt.subplots(1, len(MODELS), figsize=(5.6*len(MODELS), 4.0))
axes = np.atleast_1d(axes)
for ax, model in zip(axes, MODELS):
    ok = [r for r in runs if r['model'] == model and r['usable']]
    deep = sorted({r['m'] for r in ok})[-2:]
    asym = [r['R_full'] for r in ok if r['m'] in deep]
    ax.axhspan(np.mean(asym)-np.std(asym), np.mean(asym)+np.std(asym), color='0.85',
               label='deepest two rungs (working asymptote)')
    for seed, c in zip(SEEDS, ['#0072B2', '#D55E00', '#009E73']):
        g = sorted([r for r in ok if r['seed'] == seed], key=lambda r: r['m'])
        if not g:
            continue
        x = [r['m'] for r in g]
        y = [r['R_full'] for r in g]
        lo = [r['R_full_ci'][0] for r in g]
        hi = [r['R_full_ci'][1] for r in g]
        ax.fill_between(x, lo, hi, color=c, alpha=.15)
        ax.plot(x, y, 'o-', color=c, label=f'seed {seed}', ms=4)
    ax.set_xscale('log', base=2)
    ax.set(xlabel='measurements per rank', ylabel='R (all entries)', title=model)
    ax.grid(alpha=.3); ax.legend(fontsize=7)
plt.tight_layout()"""),

    MD(r"""## The `1/m` law, and where each model's estimator becomes trustworthy

`m·ΣVar` should be flat once the per-rank sample is diffusive. Rising means variance is falling
*slower* than `1/m` — the signature of a sample that has not yet decorrelated.

The second panel is the **sign diagnostic**, and it is a different failure mode entirely.
`G_i = ⟨sign·M⟩_i/⟨sign⟩_i` is a ratio with a **stochastic denominator**. Under a sign problem the
per-rank average sign is `O(⟨s⟩)` with `O(1/√m)` fluctuations, so at shallow depth some rank lands
near zero, its `G_i` blows up, and that one rank dominates `ΣVar`. The estimator becomes
**heavy-tailed long before it becomes wrong**, and at the extreme a rank's sign is exactly zero and
`G_i` is `0/0`. `outlier_index` — largest per-rank `|G|` over the median — is ~1 for a healthy
sample."""),

    CODE("""fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))
for seed, c in zip(SEEDS, ['#0072B2', '#D55E00', '#009E73']):
    for model, ls in zip(MODELS, ['-', '--']):
        g = sorted([r for r in runs if r['usable'] and r['model'] == model and r['seed'] == seed],
                   key=lambda r: r['m'])
        if not g:
            continue
        x = [r['m'] for r in g]
        lab = f'{model} s{seed}'
        # normalized so both models fit one axis: each curve against its own deepest value
        mv = np.array([r['m_times_sum_var'] for r in g])
        axes[0].plot(x, mv/mv[-1], 'o'+ls, color=c, ms=4, label=lab)
        axes[1].plot(x, [r['sign']['min_mean_sign'] for r in g], 'o'+ls, color=c, ms=4, label=lab)
        axes[2].plot(x, [r['sign']['outlier_index'] for r in g], 'o'+ls, color=c, ms=4, label=lab)
axes[0].axhline(1, color='k', lw=.8)
axes[0].set(ylabel='m·ΣVar  (÷ its deepest value)', title='the 1/m law')
axes[1].set(ylabel='min over ranks of ⟨sign⟩', title='worst denominator in the sample')
axes[2].axhline(1, color='k', lw=.8)
axes[2].set(ylabel='max |G| per rank ÷ median', title='outlier index', yscale='log')
for ax in axes:
    ax.set_xscale('log', base=2); ax.set_xlabel('measurements per rank')
    ax.grid(alpha=.3); ax.legend(fontsize=6)
plt.tight_layout()"""),

    MD(r"""## What moved underneath `R`

`R` drifting is the symptom; `ρ` is the cause. The expected signature of a sub-asymptotic sample is a
**low** mate-`ρ` at small `m`: a single CT-AUX configuration is not symmetric, so a rank that has
averaged only a handful of them still carries raw configuration-level asymmetry. That is pure
symmetry-**breaking** noise, which `P` removes in full — so `R` is biased **upward**. As `m` grows
that component self-averages away *within* each rank, leaving the symmetric scalar channels `P`
cannot touch, and `ρ` rises to its asymptote while `R` falls to its own."""),

    CODE("""fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
for seed, c in zip(SEEDS, ['#0072B2', '#D55E00', '#009E73']):
    for model, ls in zip(MODELS, ['-', '--']):
        g = sorted([r for r in runs if r['usable'] and r['model'] == model and r['seed'] == seed],
                   key=lambda r: r['m'])
        if not g:
            continue
        x = [r['m'] for r in g]
        # variance-weighted-ish summary: mean mate-rho over all non-singleton orbits
        rho = [np.mean([o['rho'] for o in r['orbits'] if o['rho'] is not None]) for r in g]
        blk = sorted(g[0]['mechanism'])[0]
        wi = [r['mechanism'][blk]['within_shell'] for r in g]
        axes[0].plot(x, rho, 'o'+ls, color=c, ms=4, label=f'{model} s{seed}')
        axes[1].plot(x, wi, 'o'+ls, color=c, ms=4, label=f'{model} {blk} s{seed}')
axes[0].set(ylabel='mean mate-ρ over orbits', title='the correlation P cannot remove')
axes[1].set(ylabel='within-shell real-space noise corr', title='same thing, at the source')
for ax in axes:
    ax.set_xscale('log', base=2); ax.set_xlabel('measurements per rank')
    ax.grid(alpha=.3); ax.legend(fontsize=6)
plt.tight_layout()"""),

    MD(r"""## Paired step-to-step drift

The marginal CIs above overlap generously, which is a weak test. The paired bootstrap is the sharp
one: resampling the **same rank indices** in both runs cancels the shared randomness of the nested
chains, so it resolves a drift far smaller than either interval suggests. `nesting` reports the
measured per-rank residual correlation against its predicted `sqrt(m/m')` — the licence for pairing
at all."""),

    CODE("""hdr = f"{'model':<10} {'seed':>9} {'step':>16} {'ΔR (paired 95% CI)':>30} {'nesting meas/pred':>18}"
print(hdr); print('-'*len(hdr))
for d in S['drift']:
    f = d['full']
    flag = 'flat' if f['consistent_with_zero'] else 'DRIFT'
    print(f"{d['model']:<10} {d['seed']:>9} {d['m_from']:>7}->{d['m_to']:<8} "
          f"{f['dR']:>+8.4f} [{f['lo']:+.4f},{f['hi']:+.4f}] {flag:<6} "
          f"{d['nesting_corr']:>8.3f} / {d['nesting_expected']:.3f}")"""),

    MD(r"""## Conclusion

**`R` is scale-free above a model-dependent depth threshold, and the two models set that threshold by
different mechanisms.**

- **square/D4, β=1 — autocorrelation-limited.** `R` is biased **high** at shallow depth (≈1.11 at
  `m=4` against ≈1.04 asymptotically) and every `4→16` step drifts significantly. From `m ≳ 64` all
  paired steps are flat across all three seeds. The mechanism diagnostics move in lockstep: mate-`ρ`
  on the `m=2` orbits climbs from ≈0.65 to ≈0.85 and within-shell real-space correlation from ≈0.74
  to ≈0.90, then plateau.
- **FeAs, β=5 — sign-limited.** Autocorrelation is not the binding constraint; the ratio estimator
  is. `⟨sign⟩ ≈ 0.25`, so at `m ≤ 64` several of the 64 ranks accumulate a sign of **exactly zero**
  and their `G_i` is `0/0`. Even at `m=256`, where every rank is finite, one rank can carry an 18×
  outlier and single-handedly set `ΣVar`. The depth FeAs needs is whatever keeps the *worst*
  denominator away from zero, which is a stricter requirement than decorrelation.

**Consequence for the design principle.** "Many ranks, modest depth" survives, but *modest* has a
floor, and for a model with a sign problem that floor is set by the sign, not the autocorrelation
time. Adding ranks at fixed shallow depth does **not** fix it — it adds more chances to draw a rank
with a near-zero denominator.

**Consequence for the β ladder (task 3).** The per-rank depth must be re-established at every β, and
the sign diagnostic is the thing to watch: `⟨sign⟩` falls with β, so the depth floor rises with β
independently of the autocorrelation time. The same measurement that certifies the depth also gives
the sign data that task 3 needs to separate the two competing effects on `ρ`."""),
]
write(nb, "03_m_scaling.ipynb")


# ======================================================================================
# Notebook 4 -- beta ladder (ROADMAP task 2)
# ======================================================================================
nb = nbf.v4.new_notebook()
nb.cells = [
    MD(r"""# 4. β ladder — is square intrinsically ρ-limited, or only at β=1?

Notebook 01 measured `R = 1.0425` for the square model and notebook 02 explained it: the mates'
noise is almost perfectly correlated, `ρ ≈ 0.94`, and

$$r \;=\; \frac{m}{1+(m-1)\rho}$$

collapses to ~1 as `ρ → 1`. The natural reading was *"the single-band square model has symmetric
noise"*. But β=1 is **also** the regime where noise is dominated by symmetric scalar channels — so
that reading confounds **the model** with **the temperature**. This notebook separates them by
sweeping β with everything else held fixed.

**The prediction.** As β rises the correlation length grows, the noise acquires symmetry-*breaking*
structure, `ρ` falls and `R` rises. If so, square is not intrinsically ρ-limited and the β=1 number
is a property of the regime, not of the model.

**The competing effect, and why this ladder does not suffer from it.** `G = ⟨sign·M⟩/⟨sign⟩`, and a
fluctuation of that scalar denominator gives `δG(k) = −G(k)·δs/s` — a weight `−G(k)` that is itself
symmetric. So **sign-problem noise is a fully symmetric channel with mate-correlation exactly 1**,
pushing `ρ` *up* and `R` toward 1. Any model with a sign problem mixes the two effects and a bare
`R(β)` curve cannot say which is acting.

Square avoids this **by construction**: nearest-neighbour hopping on a bipartite lattice at half
filling (μ=0) is provably sign-free in CT-AUX at *any* β. The runs confirm it — `⟨sign⟩` is exactly 1
on every rank at every β — so this ladder isolates the correlation-length effect with the sign
channel switched off. That is checked below, not assumed.

**Design.** 4×4 cluster, D4, 64 ranks, **2048 measurements per rank**, **32 independent base seeds
per β**, β ∈ {1, 2, 4, 8}. The depth clears every rung's floor by 8× (measured separately: the floor
is 64 at β=1,2 and 256 at β=4,8 — it *rises* with β, so it was re-established at each rung rather
than inherited). Each β gets a disjoint base-seed range, so no two rungs share walker streams."""),

    CODE("""import sys, json
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
import beta_ladder as bl

plt.rcParams['figure.dpi'] = 110
S = json.load(open('../runs/beta_ladder_square.json'))
rows = S['report']['rungs']
print(f"{len(S['runs'])} runs | betas {[r['beta'] for r in rows]} | "
      f"seeds/rung {[r['n_seeds'] for r in rows]} | depth {sorted({r['m'] for r in rows})} per rank")
print('seed-spacing violations:', S['report']['seed_spacing_violations'] or 'none')"""),

    MD(r"""## The result

`R` on the full support, averaged over base seeds, with the standard error across seeds — the same
estimator and the same conventions as the milestone-6 headline numbers."""),

    CODE("""print(bl.rung_table(rows))"""),

    MD(r"""## ρ falls, R rises

Left: the mate-correlation `ρ` against β. Right: `R` against β. The band is the 95% t-interval on
the mean across seeds."""),

    CODE("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
b = np.array([r['beta'] for r in rows], float)
for ax, key, lab in ((axes[0], 'rho', r'mate-correlation $\\rho$'),
                     (axes[1], 'R_full', r'$R$ (full support)')):
    m  = np.array([r[key]['mean'] for r in rows])
    lo = np.array([r[key]['lo'] for r in rows])
    hi = np.array([r[key]['hi'] for r in rows])
    ax.fill_between(b, lo, hi, alpha=.25, color='#0072B2')
    ax.plot(b, m, 'o-', color='#0072B2')
    ax.set(xscale='log', xticks=b, xlabel=r'$\\beta$', ylabel=lab)
    ax.set_xticklabels([f'{x:g}' for x in b])
    ax.grid(alpha=.3)
axes[1].axhline(1.0, ls=':', c='k', lw=1)
fig.suptitle(r'Square 4$\\times$4, D4 — sign-free at every $\\beta$')
plt.tight_layout()"""),

    MD(r"""## Is the trend resolved?

Four rungs is too few for a regression to carry weight, so two things four points *can* support are
reported: the endpoint change with an interval built by resampling **seeds within each rung** (the
level of independent replication the design actually has), and whether every rung-to-rung step shares
one sign. Monotonicity alone is weak — three steps agreeing has probability 0.25 under an
exchangeable null — so it is only meaningful beside an endpoint interval that excludes zero."""),

    CODE("""print(bl.trend_table(S['report']['trends']))"""),

    MD(r"""## The precondition: the sign channel must stay off

The whole interpretation rests on this. If `⟨sign⟩` drifted below 1 anywhere on the ladder, a
perfectly-symmetric noise channel would start feeding `ρ` and could be mistaken for the effect
saturating."""),

    CODE("""print(bl.sign_table(rows))"""),

    MD(r"""## Control: `r` must stay flat in ω

The input template fixes `sp-fermionic-frequencies` at 128 regardless of β, and Matsubara
frequencies are spaced `π/β` — so the window in **absolute** energy units shrinks as β grows, and
each rung sums `R` over a different slice of frequency space. That would confound the ladder *if* `r`
depended on ω.

It should not: the point group acts on band and momentum indices and does not touch Matsubara
frequency. Measuring the slope at every β turns that from an assumption inherited from β=1 into a
per-rung check."""),

    CODE("""print(bl.omega_table(rows))"""),

    MD(r"""## Mechanism: per-orbit ρ and the `m/[1+(m−1)ρ]` law

The aggregate `R` is a variance-weighted mixture over orbits. Resolving it per orbit shows the law
holding rung by rung — and measured-vs-predicted `r` doubles as the contamination detector, since a
run whose variance is set by one outlying rank breaks the identity."""),

    CODE("""print(bl.orbit_table(rows))"""),

    CODE("""fig, ax = plt.subplots(figsize=(6.4, 4.4))
for r in rows:
    for o in r['orbits']:
        ax.scatter(o['r_pred']['mean'], o['r']['mean'], s=38,
                   label=f"beta={r['beta']:g}" if o is r['orbits'][0] else None,
                   color=plt.cm.viridis(np.log2(r['beta'])/3))
lim = ax.get_xlim()
ax.plot(lim, lim, 'k:', lw=1)
ax.set(xlabel=r'predicted  $m/[1+(m-1)\\rho]$', ylabel=r'measured  $r$',
       title=r'Per-orbit law holds at every $\\beta$')
ax.legend(); ax.grid(alpha=.3); plt.tight_layout()"""),

    MD(r"""## What this establishes, and what it does not

**Establishes.** `ρ` falls and `R` rises monotonically across β ∈ {1,2,4,8} on the square model, with
the endpoint change resolved against across-seed scatter. **Square is not intrinsically ρ-limited** —
its β=1 value of `R ≈ 1.04` is a property of that temperature regime, not of the single-band square
model. The `m/[1+(m−1)ρ]` law holds rung by rung, so the gain is fully accounted for by the mechanism
notebook 02 identified.

**Does not establish.** This ladder says nothing about the *competing* sign channel, because square
has no sign problem to exercise it — that is exactly why it is the clean instrument for the
correlation-length axis, and exactly why it cannot settle where the two effects cross over. That
needs a β ladder on a model that *has* a sign problem, where `⟨sign⟩` and `ρ` can be tracked
together.

**Scope, unchanged.** This is `Var(G)` for a single-iteration estimator on a 4×4 cluster in plain
DCA under CT-AUX. Observable-level variance stays out of scope."""),
]
write(nb, "04_beta_ladder.ipynb")


# ======================================================================================
# Notebook 5 -- cross-model beta ladder (ROADMAP task 3)
# ======================================================================================
# Colour is assigned per MODEL and fixed, never by rank -- a filter that dropped a rung must not
# repaint the survivor. The two hues are Okabe-Ito blue/vermillion, validated for CVD separation
# (OKLab dE 21.9 under protan/deutan against a target of 8) rather than assumed safe. `R` and `rho`
# live on different scales and therefore get their own panels: a dual y-axis would be the single
# most common way to make this chart lie.
nb = nbf.v4.new_notebook()
nb.cells = [
    MD(r"""# 5. Does the β trend reproduce on a second model?

Notebook 04 measured, on the square model, that `R` rises and the mate-correlation `ρ` falls as β
grows — so square's `R ≈ 1.04` at β=1 is a property of that **temperature regime**, not of the
single-band square model. That is one model. **One model is an anecdote**, and the claim we want to
make — *`R` grows as you reach deeper into the more interesting low-temperature physics* — is a
statement about the method, not about square.

So this notebook puts a second ladder beside it. FeAs adds something square structurally **could
not** provide:

**Square is sign-free at every β by construction** (nearest-neighbour hopping, bipartite lattice,
half filling), which is exactly why it isolates the correlation-length effect cleanly — and exactly
why it cannot exercise the competing channel. `G = ⟨sign·M⟩/⟨sign⟩`, so a fluctuation in that scalar
denominator gives `δG(k) = −G(k)·δs/s`: a weight `−G(k)` that is itself symmetric, making
**sign-problem noise a fully symmetric channel with mate-correlation exactly 1**. It pushes `ρ` *up*
and `R` toward 1, against the correlation-length effect pushing `ρ` down.

FeAs has `⟨s⟩` falling from 0.998 to 0.148 across its ladder. So this pairing answers a question
neither model answers alone: **when both effects are live, which one wins?**

⚠️ **The two curves are not expected to coincide, and the axes are not directly comparable.** The
models differ in band count, `U`, filling, orbit structure and β range *simultaneously*. The claim
is about **direction and mechanism**, not about the two curves lying on top of each other."""),

    CODE("""import sys, json
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
import beta_ladder as bl

plt.rcParams['figure.dpi'] = 110
SQ, FE = '#0072B2', '#D55E00'          # Okabe-Ito; CVD-validated pair, fixed per model

L = {'square': json.load(open('../runs/beta_ladder_square.json')),
     'fe_as':  json.load(open('../runs/beta_ladder_fe_as.json'))}
R = {k: v['report']['rungs'] for k, v in L.items()}

for k, v in L.items():
    rr = R[k]
    print(f"{k:>7}: {len(v['runs']):>3} runs | beta {[r['beta'] for r in rr]} | "
          f"{rr[0]['n_seeds']} seeds/rung | {rr[0]['m']} meas/rank | "
          f"spacing violations: {v['report']['seed_spacing_violations'] or 'none'}")"""),

    MD(r"""## The two ladders

`R` on the full support, with the standard error across base seeds — the project's settled estimator
(the rank bootstrap is optimistic where a sign problem makes per-rank `G` heavy-tailed). `ρ` is the
**mate** correlation, the one that enters `r = m/[1+(m−1)ρ]`."""),

    CODE("""for k in ('square', 'fe_as'):
    print(f'=== {k} ===')
    print(bl.rung_table(R[k]))
    print()"""),

    MD(r"""## The overlay

Two panels, not two y-axes on one: `R` and `ρ` live on different scales, and a dual-axis chart would
let the eye read a crossing that is an artifact of the scaling choice. Bands are 95% t-intervals on
the mean across seeds. Endpoints are labelled directly, so identity never rests on colour alone."""),

    CODE("""fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for ax, key, lab in ((axes[0], 'R_full', r'$R$ (full support)'),
                     (axes[1], 'rho',    r'mate-correlation $\\rho$')):
    for name, colour, disp in (('square', SQ, 'square / D4'), ('fe_as', FE, 'FeAs 2-band')):
        rows = R[name]
        b  = np.array([r['beta'] for r in rows], float)
        m  = np.array([r[key]['mean'] for r in rows])
        lo = np.array([r[key]['lo'] for r in rows])
        hi = np.array([r[key]['hi'] for r in rows])
        ax.fill_between(b, lo, hi, alpha=.20, color=colour, lw=0)
        ax.plot(b, m, 'o-', color=colour, lw=2, ms=8, label=disp)
        # Direct-label the endpoint only -- a number on every point is noise, but identity should
        # not depend on the legend alone.
        ax.annotate(f'{m[-1]:.2f}', (b[-1], m[-1]), textcoords='offset points',
                    xytext=(8, -3), color=colour, fontsize=9, fontweight='bold')
    ax.set(xscale='log', xlabel=r'$\\beta$', ylabel=lab)
    ax.set_xticks([1, 2, 3, 4, 5, 8]); ax.set_xticklabels(['1', '2', '3', '4', '5', '8'])
    ax.grid(alpha=.25, lw=.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].axhline(1.0, ls=':', c='0.5', lw=1)
axes[0].legend(frameon=False)
fig.suptitle(r'$R$ rises and $\\rho$ falls with $\\beta$ on both models', y=1.02)
plt.tight_layout()"""),

    MD(r"""## The result the pairing exists to produce

**`ρ` falls monotonically on FeAs even as `⟨s⟩` collapses to 0.148.** Sign noise is a *perfectly*
mate-correlated channel — it should drive `ρ` up. Across the entire range where FeAs is simulable, it
does not win. So the correlation-length mechanism measured on square is not an artifact of working in
a sign-free corner: it survives with the competing channel fully live.

That upgrades the claim from *"measured on square, expected to generalize"* to a two-model result."""),

    CODE("""print('=== FeAs: the sign channel is live at every rung ===')
print(bl.sign_table(R['fe_as']))
print()
print('=== square: sign-free by construction, verified per rung ===')
print(bl.sign_table(R['square']))"""),

    MD(r"""## Two ceilings, and which one binds

`r = m/[1+(m−1)ρ]` has two limits: an orbit can never beat its size `m`, and as `m → ∞` it tends to
`1/ρ`. So each orbit is capped near **min(m, 1/ρ)** — and *which* ceiling binds tells you what to do
next. A ρ-limited model needs colder physics; an m-limited model needs more symmetry.

Both models start ρ-limited. FeAs does not end that way."""),

    CODE("""fig, ax = plt.subplots(figsize=(6.6, 4.4))
for name, colour, disp in (('square', SQ, 'square / D4'), ('fe_as', FE, 'FeAs 2-band')):
    rows = R[name]
    b   = np.array([r['beta'] for r in rows], float)
    inv = np.array([1.0 / r['rho']['mean'] for r in rows])
    rr  = np.array([r['R_full']['mean'] for r in rows])
    # rf-string, not f: a bare f-string is not raw, so the notebook's \\rho would be eaten as \\r.
    ax.plot(b, inv, 'o--', color=colour, lw=2, ms=7, alpha=.75, label=rf'{disp}: $1/\\rho$ ceiling')
    ax.plot(b, rr, 'o-', color=colour, lw=2, ms=8, label=f'{disp}: measured $R$')
ax.set(xscale='log', yscale='log', xlabel=r'$\\beta$', ylabel='reduction factor')
ax.set_xticks([1, 2, 3, 4, 5, 8]); ax.set_xticklabels(['1', '2', '3', '4', '5', '8'])
ax.grid(alpha=.25, lw=.6)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8.5)
ax.set_title(r'The $1/\\rho$ ceiling lifts away; $R$ stops following it')
plt.tight_layout()"""),

    CODE("""# Where each rung sits between its two ceilings. The m-ceiling is the rho=0 limit of the same
# weighted-harmonic mixture that produces R, so it is the honest "if the physics were perfect" bar.
print(f"{'model':>8} {'beta':>5} {'R':>8} {'1/rho':>8} {'binding ceiling':>16}")
for name in ('square', 'fe_as'):
    for r in R[name]:
        inv = 1.0 / r['rho']['mean']
        print(f"{name:>8} {r['beta']:>5g} {r['R_full']['mean']:>8.3f} {inv:>8.2f} "
              f"{('rho (physics)' if inv < 3.6 else 'm (geometry)'):>16}")"""),

    MD(r"""## Honesty item: `w_null` grows, and it is part of the trend

`R_full = R_non-null/(1 − w_null)`, where `w_null` is the forced-null share of raw variance —
symmetry-*forbidden* entries whose noise symmetrization annihilates completely. It is a real benefit
a user experiences, but it is **not** the mate-decorrelation mechanism, and it grows with β. So a
rising `R_full` mixes two effects, and the Conventions require reporting both columns in exactly this
cross-model setting."""),

    CODE("""for k in ('square', 'fe_as'):
    print(f'=== {k} ===')
    print(bl.support_table(R[k]))
    print()

fe = R['fe_as']
f0, f1 = fe[0], fe[-1]
gf = f1['R_full']['mean'] / f0['R_full']['mean']
gn = f1['R_nonnull']['mean'] / f0['R_nonnull']['mean']
print(f"FeAs beta=1 -> {f1['beta']:g}:  R_full x{gf:.3f}   R_non-null x{gn:.3f}"
      f"   -> {100*(gf-gn)/(gf-1):.0f}% of the rise is w_null, not mechanism")"""),

    MD(r"""## Controls

The frequency window shrinks in absolute units as β grows (`sp-fermionic-frequencies` is fixed while
Matsubara spacing is `π/β`), so each rung sums `R` over a different slice — harmless only if `r` is
flat in ω, which is measured per rung rather than inherited from β=1. And the per-orbit law is the
free contamination detector: measured vs predicted `r` diverges exactly when one rank's near-zero
sign dominates the variance sum."""),

    CODE("""for k in ('square', 'fe_as'):
    print(f'=== {k}: r vs omega ===')
    print(bl.omega_table(R[k]))
    print()
print('=== FeAs: per-orbit law, worst gap per rung ===')
for r in R['fe_as']:
    print(f"  beta={r['beta']:g}  max|r - r_pred| over orbits = "
          f"{max(o['max_abs_gap'] for o in r['orbits']):.3g}")"""),

    MD(r"""## What this establishes, and what it does not

**Establishes.** The β trend reproduces on a second, structurally different model: FeAs `R` rises
`1.526 ± 0.016 → 2.968 ± 0.059` while `ρ` falls `0.522 → 0.076` over β=1→5, both monotone and
resolved against across-seed scatter (`ΔR = +1.442 [+1.322, +1.561]`). Critically, this happens with
a **live sign problem** — `⟨s⟩` falls to 0.148 — so the perfectly-mate-correlated sign channel loses
to the correlation-length effect across the whole simulable range. The mechanism measured on square
is not an artifact of a sign-free corner.

**Also establishes a limit, which is the more useful half.** `R` is *saturating*: FeAs's steps are
`+0.276, +0.541, +0.454, +0.171`. And it is the **orbit-size ceiling** doing it, not the sign
channel — `ρ` is still falling, and by β=5 `1/ρ = 13.2` against an m-ceiling of ~3.7. **FeAs ends the
ladder m-limited**: the physics is delivering decorrelation that the 4×4 geometry cannot cash in. The
implication is concrete — at that point more symmetry (bigger cluster, larger point group, more
equivalent bands) buys more than colder physics does.

**Does not establish.** Where the sign channel finally wins. It must eventually — as `⟨s⟩ → 0` the
noise is *all* symmetric channel and `R → 1` — but FeAs at β=5 is already at the edge of
measurability (outlier index 6.4, the across-seed SEM tripling), and nothing here locates the
turnover. Shallow scouting at β=6 appeared to show one; it was contamination, not physics.

**Does not establish causation between the models either.** Square and FeAs differ in band count,
`U`, filling and orbit structure simultaneously, so the gap between the curves cannot be attributed
to any single cause. The mechanism is measured; the attribution is not. That is what ROADMAP task 4's
one-axis-at-a-time sweep is for.

**Scope, unchanged.** `Var(G)` for a single-iteration estimator on a 4×4 cluster, plain DCA, CT-AUX
at the bare bath. Observable-level variance stays out of scope. Note CT-AUX drops `J`/`Jp`, so FeAs
runs here as an Ising-Hund density-density model."""),
]
write(nb, "05_beta_cross_model.ipynb")


# ======================================================================================
# Notebook 6 -- model sweep
# ======================================================================================
nb = nbf.v4.new_notebook()
nb.cells = [
    MD(r"""# 6. Does `R` grow with symmetry-**equivalent** orbitals?

Every number in this project so far rests on two models — square (`nb`=1) and FeAs (`nb`=2) — that
differ in β, band count, interaction **and** filling simultaneously. Notebook 05 is explicit about
the consequence: the mechanism is measured, the attribution is not. `R` = 1.04 on square and 3.04 on
FeAs is a gap with four candidate causes and no way to separate them.

This notebook is the sweep that separates them. Four design points at a **fixed β = 5**, each moving
as few axes as the model tree allows:

| point | what it is for |
|---|---|
| `square_b5_c4` | the **keystone** — square at *FeAs's* temperature, so the nb comparison is not also a β comparison |
| `fe_as` β=5 | reused from notebook 05's ladder, not re-run |
| `threeband_b5_c4` | the `nb`=3 point **and** the inequivalent-orbital control, in one run |
| `square_b5_c8` | the cluster axis — a genuine single-axis pair with `square_b5_c4` |

**Why `threeband` carries two questions at once.** Its d orbital is inequivalent to its two p
orbitals, while the two p orbitals are equivalent to each other. So a single run contains both a
band-pair block where the point group *does* permute bands (p–p′) and blocks where it cannot (d–d,
d–p). If the claim of TAKEAWAYS 4c is right — that the benefit comes from symmetry-**equivalent**
orbitals rather than from orbital count — those blocks must behave differently **within one model**,
with β, `U`, filling and geometry held exactly fixed. That is a far stronger control than a second
whole model would be.

Crucially, the band-equivalence classes are **derived from `P`**, never declared: two bands are
equivalent iff some symmetry orbit contains band-diagonal entries of both. Whether threeband really
has the structure claimed above is therefore something the run reports, not something we assert."""),

    CODE("""import sys, json
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 110
# Okabe-Ito, assigned per MODEL and never cycled: colour follows the entity, not its rank.
C  = {'square': '#0072B2', 'fe_as': '#D55E00', 'threeband': '#009E73'}
MK = {'square': 'o',       'fe_as': 's',       'threeband': 'D'}
INK, MUTED = '#222222', '#666666'

S    = json.load(open('../runs/model_sweep.json'))
ROWS = S['rows']
BY   = {r['label']: r for r in ROWS}
# The nb=2 point is the committed beta=5 rung of notebook 05's FeAs ladder -- reused, not re-run.
FE = [r for r in json.load(open('../runs/beta_ladder_fe_as.json'))['report']['rungs']
      if abs(r['beta'] - 5.0) < 1e-9][0]

for r in ROWS:
    print(f"{r['label']:<20} nb={r['nb']} nk={r['nk']:>3} |G|={r['n_ops']} "
          f"seeds={r['n_seeds']:>3}  R={r['R_full']['mean']:.4f} +/- {r['R_full']['sem']:.4f}")
print(f"{'fe_as_b5_c4 (ref)':<20} nb={FE['nb'] if 'nb' in FE else 2} nk=16 |G|=8 "
      f"seeds={FE['n_seeds']:>3}  R={FE['R_full']['mean']:.4f} +/- {FE['R_full']['sem']:.4f}")
viol = [v for r in ROWS for v in (r.get('seed_spacing_violations') or [])]
print('\\nseed-spacing violations:', viol or 'none')"""),

    MD(r"""## The measured sweep

Everything below is read from `runs/model_sweep.json`, which `model_sweep.py build` produced by
verifying **every run's own HDF5 metadata against the declared design point** and raising on any
disagreement — so a run staged into the wrong directory is a hard error rather than a silent
contribution to an average."""),

    CODE("""print(S['report'])"""),

    MD(r"""## The `nb` trend — and why it is a trend, not a controlled axis

⚠️ **Read the caveat before the figure.** `nb` cannot be varied while holding "model" fixed: there is
no such thing as the same lattice with a different band count. The aggregator enforces the
one-axis-at-a-time rule mechanically (`axis_pairs` refuses to pair points that differ in more than
the named axis), and it emits **no pair for `nb`** for exactly this reason. The single-axis result in
this notebook is the *cluster* axis, where square 4×4 and 8×8 differ in `nk` and nothing else.

So the three points below are a **trend across models at fixed β**, which is strictly weaker than a
controlled axis. What makes it worth plotting anyway is that the one confound the earlier comparison
could not shed — temperature — is now held fixed by construction, and the *within-model* control on
the next figure carries the causal weight."""),

    CODE("""pts = [(BY['square_b5_c4'], 1, 'square', 'square / D4'),
       (FE,                    2, 'fe_as', 'FeAs 2-band'),
       (BY['threeband_b5_c4'], 3, 'threeband', 'threeband (d-p)')]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
for r, nb, key, disp in pts:
    st = r['R_full']
    ax.errorbar(nb, st['mean'], yerr=[[st['mean'] - st['lo']], [st['hi'] - st['mean']]],
                fmt=MK[key], ms=9, lw=2, capsize=5, color=C[key], label=disp, zorder=3)
    ax.annotate(f"{st['mean']:.2f}", (nb, st['mean']), textcoords='offset points',
                xytext=(11, -4), color=INK, fontsize=10, fontweight='bold')
ax.plot([p[1] for p in pts], [p[0]['R_full']['mean'] for p in pts],
        '-', color=MUTED, lw=1.2, alpha=.7, zorder=1)

# The threeband point carries a warm-up systematic. It is SMALLER than the plotted 95% interval, so
# drawing it as a band would render it invisible and imply it does not exist -- it is stated instead.
sysd = {s['parent']: s for s in S.get('systematics', [])}.get('threeband_b5_c4')
if sysd and sysd.get('R_full'):
    v = sysd['R_full']
    ax.annotate(f"warm-up systematic {100 * v['frac']:+.1f}%\\n(within the interval shown)",
                (3, BY['threeband_b5_c4']['R_full']['lo']), textcoords='offset points',
                xytext=(0, -24), color=MUTED, fontsize=8.5, ha='center')

ax.axhline(1.0, ls=':', c='0.5', lw=1)
ax.set(xlabel='number of bands  $n_b$', ylabel=r'$R$ (full support)', xticks=[1, 2, 3],
       xlim=(0.6, 3.6))
ax.set_title(r'$R$ at fixed $\\beta=5$, 4$\\times$4 cluster, $|G|=8$', fontsize=11)
ax.grid(alpha=.25, lw=.6)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc='upper left')
plt.tight_layout(); plt.show()"""),

    MD(r"""## The control that carries the causal weight

Now **inside one model**, with every other axis frozen. The equivalence-aware partition splits every
entry of threeband's `(b0, b1, k)` index space four ways:

- **diagonal, band-orbit 1** — the d–d block. `d` is alone in its equivalence class, so no group
  element maps it anywhere: symmetrization can only average it over k-orbits.
- **diagonal, band-orbit >1** — the p–p and p′–p′ blocks, which the group *can* interchange.
- **off-diagonal, inequivalent** — d–p. Off-diagonal, but between bands the group cannot relate.
- **off-diagonal, equivalent** — p–p′. Off-diagonal **and** band-permutable: the block where both
  effects are live at once.

If band **count** were what mattered, all four would improve together. If symmetry-**equivalence** is
what matters, the two `equivalent` classes must separate from the two `inequivalent` ones. The
harmonic reconstruction printed underneath is a self-check: the per-class `R_C` must recombine to the
aggregate `R` exactly, so a mis-specified partition shows up as a nonzero gap rather than as a
plausible-looking table."""),

    CODE("""tb = BY['threeband_b5_c4']
cs = tb['class_stats']['equivalence']
# Forced nulls have R_C = inf by construction (noise driven to exactly zero); they are reported in
# w_null, not on an axis that would have to be broken to hold them.
items = [(k, v) for k, v in cs.items() if v['R_C']]
items.sort(key=lambda kv: -kv[1]['R_C']['mean'])

fig, ax = plt.subplots(figsize=(9.0, 3.9))
ys = np.arange(len(items))[::-1]
# Greyscale by whether a band permutation can REACH the class, which is the distinction under test.
# Hue is reserved for model identity in the figure above; re-using it here would imply a link.
reach = {'diagonal, band-orbit 1': False}
cols = ['#BBBBBB' if reach.get(k, True) is False else '#009E73' for k, _ in items]
vals = [v['R_C']['mean'] for _, v in items]
errs = [[v['R_C']['mean'] - v['R_C']['lo'] for _, v in items],
        [v['R_C']['hi'] - v['R_C']['mean'] for _, v in items]]
ax.barh(ys, vals, color=cols, height=.62, zorder=2)
ax.errorbar(vals, ys, xerr=errs, fmt='none', ecolor=INK, elinewidth=1.4, capsize=3, zorder=4)
for y, (k, v), val in zip(ys, items, vals):
    ax.annotate(f"{val:.3f} ± {v['R_C']['sem']:.3f}   "
                f"({100 * v['w']['mean']:.0f}% of raw variance, mean m={v['mean_m']:.1f})",
                (val, y), textcoords='offset points', xytext=(8, -4), fontsize=9, color=INK)
agg = tb['R_full']['mean']
ax.axvline(agg, ls='--', color=MUTED, lw=1.4, zorder=3)
ax.annotate(f'aggregate R = {agg:.2f}', (agg, ys[0]), textcoords='offset points', xytext=(0, 22),
            fontsize=9, color=MUTED, ha='center')
ax.set(yticks=ys, yticklabels=[k for k, _ in items], xlabel=r'$R_C$ within class',
       xlim=(0, max(vals) * 1.75), ylim=(ys[-1] - .6, ys[0] + .95))
ax.set_title(f"threeband, β=5: where the benefit lives  ({tb['n_seeds']} seeds, 95% CI)",
             fontsize=11)
ax.grid(axis='x', alpha=.25, lw=.6)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()

print('band-equivalence classes (derived from P, not declared):',
      tb['structure']['band_equivalence']['classes'])
print('realized band permutations:', sorted(tb['structure']['permutation']['permutations']))
eq1 = tb['by_class']['equivalence']
print(f"partition self-check on the reference run: harmonic reconstruction "
      f"{eq1['harmonic_reconstruction']:.6f} vs aggregate {eq1['aggregate_R']:.6f} "
      f"(gap {eq1['reconstruction_gap']:.2e})")"""),

    MD(r"""### The contrasts, with intervals

A prediction that survives only because nobody put an interval on it is not evidence. These are
differences between classes **within the one model**, so β, `U`, filling and geometry are held fixed
by construction, and the error is the scatter between seeds."""),

    CODE("""for c in S['class_contrasts']:
    if c['point'] != 'threeband_b5_c4':
        continue
    verdict = 'RESOLVED' if c['resolved'] else 'not resolved (interval spans zero)'
    print(f"{c['contrast']:<34} {c['a_mean']:.3f} - {c['b_mean']:.3f} = "
          f"{c['delta']:+.3f}  95% [{c['lo']:+.3f}, {c['hi']:+.3f}]   {verdict}")
print()
print('For reference, the single-band square point at the same beta:',
      f"R = {BY['square_b5_c4']['R_full']['mean']:.3f} "
      f"+/- {BY['square_b5_c4']['R_full']['sem']:.3f}, mean m = "
      f"{BY['square_b5_c4']['class_stats']['equivalence']['diagonal, band-orbit 1']['mean_m']:.2f}")"""),

    MD(r"""## Which ceiling binds

`r = m/[1+(m−1)ρ]` is squeezed between two ceilings: the orbit size `m`, and `1/ρ` set by how
mate-correlated the noise is. Whichever is smaller is what a practitioner is actually fighting, and
it prescribes different actions — chase colder physics when `ρ` binds, chase bigger orbits when `m`
does. This is the one-run diagnostic of TAKEAWAYS 3, evaluated across the sweep."""),

    CODE("""fig, ax = plt.subplots(figsize=(7.6, 3.4))
labs = [r['label'] for r in ROWS]
ys = np.arange(len(ROWS))[::-1]
w = .32   # < the .5 half-spacing, so the paired bars carry a visible gap rather than touching
ax.barh(ys + .18, [r['inv_rho'] for r in ROWS], height=w, color='#56B4E9',
        label=r'$1/\\rho$ (what the noise structure allows)')
ax.barh(ys - .18, [r['m_ceiling'] for r in ROWS], height=w, color='#E69F00',
        label=r'm-ceiling (what the geometry can cash)')
for y, r in zip(ys, ROWS):
    ax.annotate(f"binds: {r['binding']}", (max(r['inv_rho'], r['m_ceiling']), y),
                textcoords='offset points', xytext=(8, -4), fontsize=9, color=INK)
ax.set(yticks=ys, yticklabels=labs, xlabel='ceiling on the per-orbit reduction')
# Headroom for the legend, which sits above the bars: the annotations own the right-hand margin.
ax.set_xlim(0, max(max(r['inv_rho'], r['m_ceiling']) for r in ROWS) * 1.42)
ax.set_ylim(ys[-1] - .55, ys[0] + 1.15)
ax.grid(axis='x', alpha=.25, lw=.6)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc='upper right', ncol=1)
plt.tight_layout(); plt.show()"""),

    MD(r"""## The systematic that must be quoted with the threeband number

threeband is run at **warm-up 8000**, not the project Convention's 200. That is not a preference: at
warm-up 200 the model's depth ladder *appears* to drift, which reads as an unconverged depth floor
but is actually an unthermalized chain (raising warm-up at fixed depth reproduces what 4× depth
bought). Square at the same β is flat in depth at warm-up 200, which is the control that makes this
specific to threeband's much larger expansion order.

What is **not** established is where warm-up converges. The sensitivity arm re-runs the same design
point at warm-up 2000 with its own disjoint seed range, so the residual dependence is measured rather
than assumed. It is reported as a **disclosed systematic**, not used as a correction."""),

    CODE("""for s in S.get('systematics', []):
    v = s['R_full']
    print(f"{s['arm']}  vs  {s['parent']}")
    print(f"  knob      : {s['knobs']}  {s['arm_knobs']} vs {s['parent_knobs']}")
    print(f"  seeds     : arm n={s['n_arm']}, parent n={s['n_parent']}")
    print(f"  R         : {v['arm']:.4f} (arm)  vs  {v['parent']:.4f} (parent)")
    print(f"  delta     : {v['delta']:+.4f}  95% [{v['lo']:+.4f}, {v['hi']:+.4f}]"
          f"  -> {100 * v['frac']:+.1f}% of R, resolved={v['resolved']}")
    print(f"  statistical error on the parent for comparison: "
          f"+/- {BY[s['parent']]['R_full']['sem']:.4f}")"""),

    MD(r"""## What this establishes, and what it does not

**Establishes: the temperature confound is closed.** Square at FeAs's own β=5 gives
`R = 1.2619 ± 0.0072` against FeAs's `2.9679 ± 0.0592` — same β, same 4×4 mesh, same `|G|=8`. The
2.35× gap between the project's two founding models is not a temperature artifact. That was the
single largest hole in the positioning claim and it is now closed by measurement.

**Establishes: `R` grows with band count, and saturates.** `1.26 → 2.97 → 3.24` at fixed β. The
direction predicted in TAKEAWAYS 4c holds; the magnitude does not extrapolate — the second band buys
the effect, the third mostly buys index space.

**Establishes: the benefit is where the mechanism says it should be, and the ordering is clean.**
Within threeband, at fixed everything: off-diagonal `≈3.33–3.38` > band-diagonal but
permutation-reachable `2.51` > band-diagonal and unreachable `1.41`. The last of those is the d–d
block, whose orbits are pure k-orbits (`mean m = 3.38`, exactly single-band square's) and which lands
next to square's `1.262 ± 0.007` at the same β and geometry.

**Corrects the qualifier as it was written.** TAKEAWAYS 4c predicted the benefit would *vanish* for
inequivalent orbitals. It does not: d–p entries sit between bands in different equivalence classes
yet reduce as well as p–p′ does (`−0.045 [−0.163, +0.074]`, not resolved). The reason is legible in
the orbit sizes — d–p orbits average `m = 5.00`, above the 4 that k-symmetry alone can reach here,
because the p_x↔p_y permutation carries the d–p_x block onto d–p_y. **An entry between inequivalent
bands is still reachable by a band permutation.** The corrected claim is *wider* than the original: a
single equivalent pair lifts every block it touches, so a material need not have all its orbitals
equivalent to benefit.

**Establishes a prescription.** threeband is **m-limited** at β=5 (`1/ρ = 6.21` against a 3.61
ceiling) at a temperature where single-band square is still ρ-limited (1.51 against 2.97). Band
structure moves you across the crossover of TAKEAWAYS 3, not just temperature — and multi-orbital
models arrive on the side where more symmetry, not colder physics, is what pays.

**Does not establish the `nb` axis causally.** `nb` cannot be moved without changing the model, so
the three-point trend is a trend, not a controlled axis. The cluster axis is the only genuine
single-axis pair here, and it resolves (`Δρ = −0.131 [−0.147, −0.116]`).

**Does not establish a model with three *mutually* equivalent orbitals.** threeband has one
equivalent pair, not a 3-cycle. Kagome was the only route to that and to a `|G| = 12` point group,
and it is blocked by a DCA-side segfault in production's own G0 symmetrization — diagnosed in the
manifest's `blocked_points`, out of scope here.

**Does not resolve threeband's warm-up dependence.** It is measured at n=8 and disclosed (−1.4%,
interval spanning zero), not corrected for.

**Scope, unchanged.** `Var(G)` for a single-iteration estimator at the bare bath, plain DCA, CT-AUX —
which drops `J`/`Jp`, so every multi-orbital model here runs as density-density. Observable-level
variance stays out of scope (4d)."""),
]
write(nb, "06_model_sweep.ipynb")
