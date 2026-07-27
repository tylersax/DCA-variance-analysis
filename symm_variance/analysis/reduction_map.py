"""The reduction map: `r` resolved by entry class and orbit size, plus headroom.

Companion to symm_variance_lib.py. That module computes the ratio; this one *resolves* it, which is
the shippable object per ROADMAP task 2.

WHY THIS EXISTS. The headline `R = sum_x Var(G[x]) / sum_x Var(Sym G[x])` is variance-weighted, and
splitting entries into classes C gives the EXACT decomposition

    R = 1 / sum_C ( w_C / R_C ),    w_C = class share of RAW variance

i.e. `R` is the variance-weighted HARMONIC mean of the class-wise `R_C`. Harmonic means are dominated
by their smallest terms, so a class that is both noisy and poorly-reduced pins the total down no
matter how well every other class does. That is a structural fact, not a reporting preference: a lone
scalar `R` will always sit below a model's best-performing components and will always understate them.

Measured on FeAs: interband entries reach `R_C = 4.56` while carrying only 2.1% of the variance, so
they are invisible in the aggregate 2.48; intraband carries 84.8% at `R_C = 2.13` and sets the total.

HEADROOM. `R_ideal = sum(Var) / sum(Var/m)` is the `rho = 0` ceiling for a model's ACTUAL orbit
structure, so `efficiency = R / R_ideal` separates "small because orbits are small" from "small
because the noise is already symmetric". Measured: square 1.035/3.053 = 33.9%; FeAs 2.156/2.894 =
74.5%. Note square's HEADROOM IS LARGER than FeAs's -- its orbit structure is fine, its noise is the
problem. The models differ in the symmetry-structure of their noise, not in their symmetry structure.

SCOPE: everything here is a property of Var(G). Observable-level variance (Sigma, susceptibilities,
spectra) is out of scope by decision -- see ROADMAP "Scope boundaries". The per-entry-class `r` is the
transferable object: anyone wanting an observable-level number can reweight this map.
"""
import numpy as np


def entry_variance(run):
    """Raw and symmetrized variance per flat entry, summed over Matsubara frequency and spin."""
    vr = run.variance_ratios()
    A = (vr["var_raw"].real + vr["var_raw"].imag).sum((0, 1))
    B = (vr["var_sym"].real + vr["var_sym"].imag).sum((0, 1))
    return A, B


def orbit_info(run):
    """Per flat entry: its orbit size `m`, and whether it is a symmetry-forced null."""
    m_of = np.zeros(run.E, int)
    null = np.zeros(run.E, bool)
    for o in run.orbits():
        for x in o["members"]:
            m_of[x] = len(o["members"])
            null[x] = o["forced_null"]
    return m_of, null


def band_classes(run):
    """Partition entries by band character: forced null / intraband / interband."""
    _, null = orbit_info(run)
    lab = run.labels
    cls = {}
    for x in range(run.E):
        if null[x]:
            key = "forced null"
        elif run.nb == 1:
            key = "band-diagonal"
        elif lab[x][0] == lab[x][1]:
            key = "intraband"
        else:
            key = "interband"
        cls.setdefault(key, []).append(x)
    return cls


def orbit_classes(run, exclude_null=True):
    """Partition non-null entries by orbit size `m`."""
    m_of, null = orbit_info(run)
    cls = {}
    for x in range(run.E):
        if exclude_null and null[x]:
            continue
        cls.setdefault(f"m={m_of[x]}", []).append(x)
    return dict(sorted(cls.items(), key=lambda kv: int(kv[0][2:])))


def R_over(run, subset, A=None, B=None):
    """Variance-weighted R restricted to a subset of flat entries."""
    if A is None:
        A, B = entry_variance(run)
    den = B[subset].sum()
    return float(A[subset].sum() / den) if den > 0 else np.inf


def R_ideal(run, subset=None, A=None):
    """`rho = 0` ceiling for the actual orbit structure: sum(Var) / sum(Var/m).

    Forced nulls are excluded by default -- their true ceiling is infinite (Var(Sym G) == 0), so
    including them makes the number meaningless rather than optimistic.
    """
    m_of, null = orbit_info(run)
    if A is None:
        A, _ = entry_variance(run)
    if subset is None:
        subset = [x for x in range(run.E) if not null[x]]
    subset = [x for x in subset if not null[x]]
    return float(A[subset].sum() / (A[subset] / m_of[subset]).sum())


def efficiency(run, subset=None):
    """R / R_ideal -- the share of available reduction actually captured."""
    A, B = entry_variance(run)
    m_of, null = orbit_info(run)
    if subset is None:
        subset = [x for x in range(run.E) if not null[x]]
    subset = [x for x in subset if not null[x]]
    return R_over(run, subset, A, B) / R_ideal(run, subset, A)


def reduction_table(run, groups=None):
    """Rows of (class, n, variance share w_C, R_C, mean m) plus the harmonic reconstruction of R.

    The reconstruction is a self-check: 1/sum(w_C/R_C) must reproduce the aggregate R exactly.
    """
    A, B = entry_variance(run)
    m_of, _ = orbit_info(run)
    if groups is None:
        groups = band_classes(run)
    rows, harmonic = [], 0.0
    for name, xs in groups.items():
        w = float(A[xs].sum() / A.sum())
        RC = R_over(run, xs, A, B)
        if np.isfinite(RC):
            harmonic += w / RC
        rows.append(dict(cls=name, n=len(xs), w=w, R_C=RC, mean_m=float(m_of[xs].mean())))
    return rows, (1.0 / harmonic if harmonic > 0 else np.inf), float(A.sum() / B.sum())


def report(run):
    """Print the full reduction map for one run."""
    A, B = entry_variance(run)
    _, null = orbit_info(run)
    nn = [x for x in range(run.E) if not null[x]]
    print(f"=== {run.model}  ({run.n_ranks} ranks, |G|={run.n_ops} ops, nb={run.nb}, nk={run.nk}) ===")
    for label, groups in (("by band character", band_classes(run)),
                          ("by orbit size", orbit_classes(run))):
        rows, recon, agg = reduction_table(run, groups)
        print(f"  -- {label} --")
        print(f"  {'class':<14} {'n':>4} {'var share':>10} {'R_C':>9} {'mean m':>7}")
        for r in rows:
            print(f"  {r['cls']:<14} {r['n']:>4} {r['w']:>10.4f} {r['R_C']:>9.3f} {r['mean_m']:>7.2f}")
        if label == "by band character":
            print(f"  aggregate R = {agg:.4f}   harmonic reconstruction = {recon:.4f}")
    print(f"  non-null: R = {R_over(run, nn, A, B):.3f}   R_ideal = {R_ideal(run, nn, A):.3f}   "
          f"efficiency = {efficiency(run, nn):.1%}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from symm_variance_lib import Run
    for p in (sys.argv[1:] or ["../runs/square_16rank.hdf5", "../runs/fe_as_16rank.hdf5"]):
        report(Run(p))
        print()
