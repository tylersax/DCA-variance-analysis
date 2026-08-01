"""Where rho comes from -- and the migration picture that makes it visible.

hero_lib answers "how much variance does symmetrization remove". This module answers "why is rho what
it is", which is the part that generalizes beyond the models measured here.

The framing: P is an orthogonal projector onto the symmetry-invariant subspace. It deletes the
symmetry-BREAKING part of the sampling noise and leaves the symmetry-PRESERVING part untouched. So
rho is the fraction of noise already living in the symmetric subspace, and the question is what puts
it there.

The real-space shell decomposition hard-codes the 8 D4 operations and assumes a square LxL cluster
mesh, which is what every design point in this bundle uses.
"""
import numpy as np

# E, C4, C2, C4^3, sigma_x, sigma_y, sigma_diag, sigma_antidiag as 2x2 integer matrices on (n1, n2).
D4 = [(1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0),
      (1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0)]


def noise_residuals(run, b0=0, b1=0, spin=0):
    """Per-rank sampling noise on one band block: delta_i = G_i - mean(G). Returns [rank, w, k]."""
    X = run.G_raw[:, :, :, spin, b1, spin, b0]
    return X - X.mean(0, keepdims=True)


def noise_corr_matrix(run, b0=0, b1=0, spin=0):
    """The full (nk, nk) correlation matrix of the noise BEFORE symmetrization."""
    d = noise_residuals(run, b0, b1, spin)
    nk = run.nk
    C = np.zeros((nk, nk))
    for a in range(nk):
        for b in range(nk):
            xa, xb = d[:, :, a].ravel(), d[:, :, b].ravel()
            den = np.sqrt(np.vdot(xa, xa) * np.vdot(xb, xb))
            C[a, b] = (np.vdot(xa, xb) / den).real if abs(den) > 0 else np.nan
    return C


def pairwise_rho_all_k(run, b0=0, b1=0, spin=0):
    """Mean noise correlation over ALL distinct k-pairs, pooled over ranks and frequency.

    The contrast against the mate-correlation of SYMMETRY-RELATED pairs is the signature that
    matters: if generic pairs and symmetry mates correlate equally, the noise carries no special
    alignment with the point group and symmetrization is fighting an ordinary correlated-sample
    problem. If mates correlate much more strongly, the noise is symmetry-aligned and symmetrization
    has little left to remove.
    """
    d = noise_residuals(run, b0, b1, spin)
    cs = []
    for a in range(run.nk):
        for b in range(a + 1, run.nk):
            xa, xb = d[:, :, a].ravel(), d[:, :, b].ravel()
            den = np.sqrt(np.vdot(xa, xa) * np.vdot(xb, xb))
            if abs(den) > 0:
                cs.append((np.vdot(xa, xb) / den).real)
    return float(np.mean(cs))


def _k_to_grid_index(run):
    """Map each cluster momentum onto integer (n1, n2) mesh coordinates for a square LxL cluster."""
    L = int(round(np.sqrt(run.nk)))
    idx = [tuple(int(round(c * L / (2 * np.pi))) % L for c in k) for k in run.k_elements]
    return idx, L


def realspace_noise(run, b0=0, b1=0, spin=0):
    """Inverse-FFT the k-space noise onto the real-space cluster: delta M(r) per rank, [rank,w,r1,r2]."""
    d = noise_residuals(run, b0, b1, spin)
    idx, L = _k_to_grid_index(run)
    grid = np.zeros(d.shape[:2] + (L, L), dtype=complex)
    for i, (n1, n2) in enumerate(idx):
        grid[:, :, n1, n2] = d[:, :, i]
    return np.fft.ifft2(grid, axes=(2, 3))


def realspace_noise_profile(run, b0=0, b1=0, spin=0):
    """sigma^2(r) / total: the share of noise power at each real-space separation.

    The number that matters is the r = 0 share. Noise concentrated at r = 0 is identical for every
    momentum, since exp(i k . 0) = 1 -- hence perfectly correlated across all k and completely immune
    to any symmetrization. A spatially white profile would put 1/Nk in every cell.
    """
    dr = realspace_noise(run, b0, b1, spin)
    s2 = (dr.real.var(0, ddof=1) + dr.imag.var(0, ddof=1)).mean(0)
    return s2 / s2.sum()


def predicted_rho(run, b0=0, b1=0, spin=0):
    """rho ~= sigma^2(r=0)/total - 1/Nk, from a local-spike-plus-white-background model of the noise."""
    prof = realspace_noise_profile(run, b0, b1, spin)
    return float(prof[0, 0] - 1.0 / run.nk)


def invariant_site_share(run, b0=0, b1=0, spin=0):
    """Share of real-space noise power sitting on sites the point group leaves FIXED.

    On an even LxL torus those are (0,0) and (L/2, L/2). They are the sharpest form of the scalar
    channel: a fluctuation localized on a D4-invariant site contributes identically to every orbit
    mate, so orbit-mate correlation for that component is exactly 1 and symmetrization removes none
    of it. Compare against the share a spatially white noise field would put there.
    """
    prof = realspace_noise_profile(run, b0, b1, spin)
    _, L = _k_to_grid_index(run)
    fixed = [s[0] for s in d4_shells(L) if len(s) == 1]
    share = float(sum(prof[r1, r2] for r1, r2 in fixed))
    return dict(sites=fixed, share=share, white=len(fixed) / run.nk)


def d4_shells(L):
    """Partition the LxL real-space mesh into D4 orbits (shells)."""
    def orbit(s):
        return frozenset((((a * s[0] + b * s[1]) % L), ((c * s[0] + e * s[1]) % L))
                         for (a, b, c, e) in D4)
    shells = {}
    for r1 in range(L):
        for r2 in range(L):
            shells.setdefault(orbit((r1, r2)), set()).add((r1, r2))
    return [sorted(s) for s in shells.values()]


def shell_correlations(run, b0=0, b1=0, spin=0):
    """Noise correlation WITHIN a real-space symmetry shell versus ACROSS different shells.

    The source-level diagnostic: it asks whether delta M(r) and delta M(S.r) at symmetry-equivalent
    SITES already fluctuate together. If they do, symmetrization -- which averages exactly those
    sites -- has almost nothing left to remove, and it is a statement about the sampling itself
    rather than about the Green's function.
    """
    dr = realspace_noise(run, b0, b1, spin)
    _, L = _k_to_grid_index(run)

    def corr(a, b):
        xa, xb = dr[:, :, a[0], a[1]].ravel(), dr[:, :, b[0], b[1]].ravel()
        den = np.sqrt(np.vdot(xa, xa) * np.vdot(xb, xb))
        return (np.vdot(xa, xb) / den).real if abs(den) > 0 else np.nan

    shells = d4_shells(L)
    within, across = [], []
    for s in shells:
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                within.append(corr(s[i], s[j]))
    for a in range(len(shells)):
        for b in range(a + 1, len(shells)):
            if shells[a][0] != (0, 0) and shells[b][0] != (0, 0):
                across.append(corr(shells[a][0], shells[b][0]))
    return dict(within=float(np.nanmean(within)), across=float(np.nanmean(across)),
                shell_sizes=[len(s) for s in shells])


# =================================================================================================
# The migration picture
# =================================================================================================
def migration_cloud(run, orbit, w_index=None, spin=0):
    """Per-rank orbit mates in the complex plane, and the point they migrate to.

    For one orbit at one Matsubara frequency, returns every rank's estimate of every mate with the
    orbit's signs applied first, so that odd mates are aligned rather than pointing opposite ways.
    The symmetrized value is the sign-aware orbit average, which is what each of those points moves
    to when symmetrization is switched on.

    Two things are legible at once: the cloud contracts by 1/sqrt(r), and its centre does not move.

    ON A MODEL WITH A SIGN PROBLEM THE INVARIANT CENTRE IS THE PHASE-WEIGHTED ONE. Production forms
    sum_i(sign_i G_i) / sum_i(sign_i), not the plain average of the per-rank points. Draw the plain
    centroid on FeAs and it appears to move, contradicting the figure's own claim -- and the
    contradiction is an artifact of the plot, not physics.
    """
    members = np.asarray(orbit["members"])
    signs = np.asarray(orbit["signs"])
    w = run.nw // 2 if w_index is None else w_index  # lowest positive Matsubara frequency

    pts = run.vec[:, w, spin, :][:, members] * signs[None, :]        # (n_rank, m) sign-aligned
    sym = run.apply_P(run.vec)[:, w, spin, :][:, members] * signs[None, :]

    ph = run.sign / run.sign.sum()
    centroid = complex((ph[:, None] * pts).sum())                    # phase-weighted, over all mates
    sym_centroid = complex((ph[:, None] * sym).sum() / len(members))

    spread_raw = float(np.sqrt(np.var(pts.real, ddof=1) + np.var(pts.imag, ddof=1)))
    spread_sym = float(np.sqrt(np.var(sym.real, ddof=1) + np.var(sym.imag, ddof=1)))
    return dict(points=pts, sym=sym, m=len(members), w_index=w,
                centroid=centroid / len(members), sym_centroid=sym_centroid,
                spread_raw=spread_raw, spread_sym=spread_sym,
                contraction=spread_raw / spread_sym if spread_sym > 0 else np.inf)
