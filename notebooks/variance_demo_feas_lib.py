"""Analysis helpers for the FeAs variance-reduction demo (claim B: derived 8 ops vs declared 2).

FeAs is two-band, so the symmetrization orbits live in (band0, band1, k) space, not just k — the
derived C4 op swaps bands 0<->1, and the two bands are degenerate at the C4-fixed momenta (Γ, (π,π)),
so even those k-points get band-space averaging. Rather than re-derive the band-permutation orbits in
Python, we define orbits *empirically*: two flattened entries are orbit-mates iff the ON arm collapses
them to the same value (symmetrization made them equal). This is model-agnostic and reflects exactly
what the imposition did.
"""
import glob, os
import numpy as np
import h5py

def load_arm(run_dir, arm):
    files = sorted(glob.glob(os.path.join(run_dir, arm, "G_*.hdf5")),
                   key=lambda f: int(f.split("_")[-1].split(".")[0]))
    G, seeds, nops = [], [], set()
    kel = None
    for f in files:
        with h5py.File(f, "r") as h:
            g = np.array(h["functions/cluster_greens_function_G_k_w"])   # (w,k,s1,b1,s0,b0)
            # spin-up block, keep both bands: -> (w, k, b1, b0)
            G.append(g[:, :, 0, :, 0, :])
            seeds.append(int(h["metadata/seed"][()][0]))
            nops.add(int(h["metadata/n_ops"][()][0]))
            kel = np.array([np.array(x) for x in h["metadata/k_elements"]])
    return np.array(G), np.array(seeds), kel, nops   # G: (rep, w, k, b1, b0)

def flatten_entries(G):
    """(rep,w,k,b1,b0) -> (rep,w,E) with E = k*b1*b0, plus the list of (k,b1,b0) labels."""
    rep, w, nk, nb, _ = G.shape
    F = G.reshape(rep, w, nk*nb*nb)
    labels = [(k, b1, b0) for k in range(nk) for b1 in range(nb) for b0 in range(nb)]
    return F, labels

def orbits_from_on(Gon, tol=1e-9):
    """Group flattened entries into symmetrization orbits: entries the ON arm makes equal UP TO
    SIGN (per replica, all w). Uses replica 0 across all w as the fingerprint.

    The sign matters. A component that is *odd* under some derived op is averaged with a -1 on half
    its orbit, so the ON arm collapses those members to equal magnitude and opposite sign. Matching
    on value alone therefore splits one orbit into two +/- classes -- for FeAs 4x4 it reported two
    n=4 interband orbits where there is really one n=8 orbit, and the n it hands downstream then
    disagrees with the measured reduction. Verified against the data: the ON value reproduces the
    signed mean (sum(+members) - sum(-members))/8 to ~1e-11, while the plain mean of either class
    is off by ~1e-2. Returns the per-member sign so callers can undo it before correlating.
    """
    F, labels = flatten_entries(Gon)
    fp = F[0]                       # (w, E) fingerprint from replica 0
    E = fp.shape[1]
    scale = np.abs(fp).mean() + 1e-30
    assigned = [-1]*E
    groups, signs = [], []
    for i in range(E):
        if assigned[i] != -1:
            continue
        gid = len(groups); assigned[i] = gid; members = [i]; member_signs = [1.0]
        for j in range(i+1, E):
            if assigned[j] != -1:
                continue
            if np.max(np.abs(fp[:, i] - fp[:, j])) < tol*scale:
                assigned[j] = gid; members.append(j); member_signs.append(1.0)
            elif np.max(np.abs(fp[:, i] + fp[:, j])) < tol*scale:
                assigned[j] = gid; members.append(j); member_signs.append(-1.0)
        groups.append(members); signs.append(member_signs)
    return groups, labels, signs

def replica_variance_flat(G):
    F, _ = flatten_entries(G)
    return F.var(axis=0, ddof=1).real     # (w, E)

def orbit_rho_flat(G, members, signs=None):
    # signs: undo the orbit's +/- structure before correlating. Without it, members that enter the
    # symmetrization with opposite sign look anti-correlated (or, within one +/- class, spuriously
    # perfectly correlated), and the n/(1+(n-1)rho) prediction no longer describes the orbit.
    if len(members) < 2:
        return np.nan
    F, _ = flatten_entries(G)
    X = F[:, :, members]
    if signs is not None:
        X = X * np.asarray(signs, dtype=float)[None, None, :]
    res = X - X.mean(axis=0, keepdims=True)
    cs = []
    for a in range(len(members)):
        for b in range(a+1, len(members)):
            xa = res[:, :, a].ravel(); xb = res[:, :, b].ravel()
            d = np.sqrt(np.vdot(xa, xa)*np.vdot(xb, xb))
            if abs(d) > 0:
                cs.append((np.vdot(xa, xb)/d).real)
    return float(np.mean(cs)) if cs else np.nan

def predicted_ratio(n, rho):
    return n / (1.0 + (n-1.0)*rho)

# ---- Declared FeAsPointGroup = {identity, C4}, applied in (k,b1,b0) space, for the claim-B contrast.
# C4: (kx,ky)->(ky,-kx); the band-permuting C4 also swaps the two bands (b->1-b). The declared orbits
# are the closure of {id, C4} on each (k,b1,b0) entry. Comparing this (coarser) partition with the
# derived-8 partition isolates what the 6 undeclared ops bought. Works for any square k-mesh: the C4
# image index is found by matching the rotated momentum modulo 2*pi.
def _c4_kmap(kel):
    def key(v): return tuple(round(((c) % (2*np.pi)), 4) for c in v)
    idx = {key(k): i for i, k in enumerate(kel)}
    kmap = []
    for k in kel:
        kx, ky = k
        kmap.append(idx[key((ky, -kx))])
    return kmap

def declared_orbits(kel, nb=2):
    nk = len(kel); kmap = _c4_kmap(kel)
    labels = [(k, b1, b0) for k in range(nk) for b1 in range(nb) for b0 in range(nb)]
    index = {lab: i for i, lab in enumerate(labels)}
    def c4(lab):
        k, b1, b0 = lab
        return (kmap[k], 1-b1, 1-b0)
    seen = [False]*len(labels); groups = []
    for i, lab in enumerate(labels):
        if seen[i]:
            continue
        orbit = set(); cur = lab
        for _ in range(4):                 # C4 has order 4; closure terminates
            if cur in orbit: break
            orbit.add(cur); cur = c4(cur)
        members = sorted(index[o] for o in orbit)
        for m in members: seen[m] = True
        groups.append(members)
    return groups, labels
