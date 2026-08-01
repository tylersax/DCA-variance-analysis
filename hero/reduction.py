"""Resolving R: which entries reduce well, which pin the total, and how much headroom is left.

The headline R is variance-weighted over every entry. Split the entries into classes C and the
decomposition is EXACT:

    R = 1 / sum_C (w_C / R_C),      w_C = class share of RAW variance

i.e. R is the variance-weighted HARMONIC mean of the class-wise R_C. Harmonic means are dominated by
their smallest terms, so a class that is both noisy and poorly reduced pins the total down no matter
how well every other class does. That is a structural fact rather than a reporting preference: a lone
scalar R will always sit below a model's best-performing components.
"""
import numpy as np

from hero_lib import orbits_from_P


def entry_variance(run):
    """Raw and symmetrized variance per flat entry, summed over Matsubara frequency and spin."""
    vr = run.variance_ratios()
    A = (vr["var_raw"].real + vr["var_raw"].imag).sum((0, 1))
    B = (vr["var_sym"].real + vr["var_sym"].imag).sum((0, 1))
    return A, B


def orbit_info(obj):
    """Per flat entry: its orbit size m, and whether it is a symmetry-forced null.

    Takes a Run or an OpsView -- it only needs P.
    """
    m_of = np.zeros(obj.E, int)
    null = np.zeros(obj.E, bool)
    for o in orbits_from_P(obj.P):
        for x in o["members"]:
            m_of[x] = len(o["members"])
            null[x] = o["forced_null"]
    return m_of, null


def band_classes(obj):
    """Partition entries by band character: forced null / intraband / interband.

    For a single-band model there is only one non-null class, which is the point: the classes below
    are exactly what a multi-orbital model adds.
    """
    _, null = orbit_info(obj)
    lab = obj.labels
    cls = {}
    for x in range(obj.E):
        if null[x]:
            key = "forced null"
        elif obj.nb == 1:
            key = "band-diagonal"
        elif lab[x][0] == lab[x][1]:
            key = "intraband"
        else:
            key = "interband"
        cls.setdefault(key, []).append(x)
    return cls


def orbit_classes(obj, exclude_null=True):
    """Partition non-null entries by orbit size m."""
    m_of, null = orbit_info(obj)
    cls = {}
    for x in range(obj.E):
        if exclude_null and null[x]:
            continue
        cls.setdefault(int(m_of[x]), []).append(x)
    return dict(sorted(cls.items()))


def R_over(run, subset, A=None, B=None):
    """Variance-weighted R restricted to a subset of flat entries."""
    if A is None:
        A, B = entry_variance(run)
    den = B[subset].sum()
    return float(A[subset].sum() / den) if den > 0 else np.inf


def R_ideal(run, subset=None, A=None):
    """The rho = 0 ceiling for the model's ACTUAL orbit structure: sum(Var) / sum(Var/m).

    Forced nulls are excluded: their true ceiling is infinite, since Var(Sym G) is exactly zero
    there, and including them makes the number meaningless rather than merely optimistic.
    """
    m_of, null = orbit_info(run)
    if A is None:
        A, _ = entry_variance(run)
    subset = range(run.E) if subset is None else subset
    subset = [x for x in subset if not null[x]]
    return float(A[subset].sum() / (A[subset] / m_of[subset]).sum())


def m_ceiling(obj, A=None):
    """The same rho = 0 ceiling, available for an operator-only point by weighting orbits equally.

    With sample data, prefer R_ideal, which weights by measured variance.
    """
    m_of, null = orbit_info(obj)
    keep = ~null
    if A is None:
        return float(keep.sum() / (1.0 / m_of[keep]).sum())
    return float(A[keep].sum() / (A[keep] / m_of[keep]).sum())


def reduction_table(run, groups=None):
    """Rows of (class, n, variance share w_C, R_C, mean m), the harmonic reconstruction, and R.

    The reconstruction is a self-check: 1/sum(w_C/R_C) must reproduce the aggregate R exactly.
    Classes whose symmetrized variance is zero (the forced nulls) have R_C = infinity and drop out of
    the sum -- which is precisely how they raise the total.
    """
    A, B = entry_variance(run)
    m_of, _ = orbit_info(run)
    groups = band_classes(run) if groups is None else groups
    rows, harmonic = [], 0.0
    for name, xs in groups.items():
        w = float(A[xs].sum() / A.sum())
        RC = R_over(run, xs, A, B)
        if np.isfinite(RC):
            harmonic += w / RC
        rows.append(dict(cls=name, n=len(xs), w=w, R_C=RC, mean_m=float(m_of[xs].mean())))
    return rows, (1.0 / harmonic if harmonic > 0 else np.inf), float(A.sum() / B.sum())


def null_weight(run):
    """w_null: the share of raw variance sitting on symmetry-forbidden entries.

    These are entries the group maps onto minus themselves, so their expectation is exactly zero and
    symmetrization removes ALL of their noise. R_full = R_nonnull / (1 - w_null).
    """
    A, _ = entry_variance(run)
    _, null = orbit_info(run)
    return float(A[null].sum() / A.sum())
