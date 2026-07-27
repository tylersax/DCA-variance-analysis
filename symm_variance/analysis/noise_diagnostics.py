"""Mechanism diagnostics: WHERE the noise correlation that limits symmetrization comes from.

Companion to symm_variance_lib.py. That module answers "how much variance does symmetrization
remove" (the ratio r, R); this one answers "why is rho what it is".

The framing: symmetrization P is an ORTHOGONAL PROJECTOR onto the symmetry-invariant subspace. It
deletes the symmetry-BREAKING part of the MC noise and leaves the symmetry-PRESERVING part untouched.
So rho is exactly the fraction of noise variance already living in the symmetric subspace, and the
interesting question is what puts it there.

Assumes a 2D square cluster mesh (both current models: square/D4 and FeAs are on 4x4). The real-space
shell decomposition hard-codes the 8 D4 operations; revisit for hexagonal/3D meshes.
"""
import numpy as np

# The 8 D4 operations as 2x2 integer matrices (a,b,c,d) acting on (n1,n2) mesh coordinates:
#   E, C4, C2, C4^3, sigma_x, sigma_y, sigma_diag, sigma_antidiag
D4 = [(1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0),
      (1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0)]


def noise_residuals(run, b0=0, b1=0, spin=0):
    """Per-rank MC noise on G for one (band, band, spin) block: delta = G_i - mean_i(G_i).

    Returns [rank, w, k] complex. This is the raw material for every diagnostic below.
    """
    X = run.G_raw[:, :, :, spin, b1, spin, b0]
    return X - X.mean(0, keepdims=True)


def pairwise_rho_all_k(run, b0=0, b1=0, spin=0):
    """Mean noise correlation over ALL distinct k-pairs (pooled over ranks and w).

    Contrast this with the mate-rho that symm_variance_lib.Run.orbit_rho reports for
    SYMMETRY-RELATED pairs. The gap between the two is the signature of symmetry-aligned noise:
    for square/D4 the generic-pair value is ~0.40 while symmetry mates sit at ~0.9.
    """
    d = noise_residuals(run, b0, b1, spin)
    nk = run.nk
    cs = []
    for a in range(nk):
        for b in range(a + 1, nk):
            xa, xb = d[:, :, a].ravel(), d[:, :, b].ravel()
            den = np.sqrt(np.vdot(xa, xa) * np.vdot(xb, xb))
            if abs(den) > 0:
                cs.append((np.vdot(xa, xb) / den).real)
    return float(np.mean(cs))


def noise_corr_matrix(run, b0=0, b1=0, spin=0):
    """Full (nk, nk) noise correlation matrix of delta-G BEFORE symmetrization."""
    d = noise_residuals(run, b0, b1, spin)
    nk = run.nk
    C = np.zeros((nk, nk))
    for a in range(nk):
        for b in range(nk):
            xa, xb = d[:, :, a].ravel(), d[:, :, b].ravel()
            den = np.sqrt(np.vdot(xa, xa) * np.vdot(xb, xb))
            C[a, b] = (np.vdot(xa, xb) / den).real if abs(den) > 0 else np.nan
    return C


def _k_to_grid_index(run):
    """Map each cluster momentum onto integer (n1, n2) mesh coordinates for a square LxL cluster."""
    L = int(round(np.sqrt(run.nk)))
    idx = [tuple(int(round(c * L / (2 * np.pi))) % L for c in k) for k in run.k_elements]
    return idx, L


def realspace_noise(run, b0=0, b1=0, spin=0):
    """Inverse-FFT the k-space noise onto the real-space cluster: delta M(r) per rank.

    Returns [rank, w, r1, r2] complex.
    """
    d = noise_residuals(run, b0, b1, spin)
    idx, L = _k_to_grid_index(run)
    grid = np.zeros(d.shape[:2] + (L, L), dtype=complex)
    for i, (n1, n2) in enumerate(idx):
        grid[:, :, n1, n2] = d[:, :, i]
    return np.fft.ifft2(grid, axes=(2, 3))


def realspace_noise_profile(run, b0=0, b1=0, spin=0):
    """sigma^2(r) / total: the share of MC noise power sitting at each real-space separation.

    The key number is the r=0 (local) share. Noise concentrated at r=0 is identical for every k
    (e^{ik.0}=1), hence perfectly correlated across all momenta and completely immune to
    symmetrization. A spatially white profile would put 1/Nk in every cell.
    """
    dr = realspace_noise(run, b0, b1, spin)
    s2 = (dr.real.var(0, ddof=1) + dr.imag.var(0, ddof=1)).mean(0)  # average over w -> [r1, r2]
    return s2 / s2.sum()


def predicted_rho(run, b0=0, b1=0, spin=0):
    """rho ~= sigma^2(r=0)/total - 1/Nk.

    Model: sigma^2(r) = uniform background c + a local spike A at r=0. Then for q != 0 the
    normalized FT is A/total = sigma^2(0)/total - c/total, and c*Nk/total ~ 1 when the background
    dominates. Matched the measured generic-pair rho to a few percent on every block tested.
    """
    prof = realspace_noise_profile(run, b0, b1, spin)
    return float(prof[0, 0] - 1.0 / run.nk)


def d4_shells(L):
    """Partition the LxL real-space mesh into D4 orbits (shells). Returns list of sorted site lists."""
    def orbit(s):
        return frozenset((((a * s[0] + b * s[1]) % L), ((c * s[0] + e * s[1]) % L))
                         for (a, b, c, e) in D4)
    shells = {}
    for r1 in range(L):
        for r2 in range(L):
            shells.setdefault(orbit((r1, r2)), set()).add((r1, r2))
    return [sorted(s) for s in shells.values()]


def shell_correlations(run, b0=0, b1=0, spin=0):
    """Real-space noise correlation WITHIN a symmetry shell vs ACROSS different shells.

    This is the source-level diagnostic: it asks whether delta M(r) and delta M(S.r) at
    symmetry-equivalent SITES already fluctuate together. If they do, symmetrization -- which
    averages exactly those sites -- has almost nothing left to remove.

    square/D4: within +0.905, across -0.183  (strongly symmetry-aligned -> rho high -> R ~ 1.03)
    FeAs:      within +0.07..0.10            (little alignment      -> rho ~ 0 -> r approaches m)
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
            # compare shell representatives, skipping the r=0 singleton (trivially symmetric)
            if shells[a][0] != (0, 0) and shells[b][0] != (0, 0):
                across.append(corr(shells[a][0], shells[b][0]))
    return dict(within=float(np.nanmean(within)), across=float(np.nanmean(across)),
                shell_sizes=[len(s) for s in shells], shells=shells)
