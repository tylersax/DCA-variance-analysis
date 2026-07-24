"""Analysis helpers for the symmetrization variance-reduction demo.

Deliberately plain: numpy + h5py + matplotlib, no DCA/project imports, so it runs anywhere the
HDF5 replica files land. Imported by symmetry_variance_demo.ipynb; kept as a module so the notebook
cells stay short and the logic is unit-checkable from the shell.
"""
import glob, os
import numpy as np
import h5py

# HDF5 axis order of cluster_greens_function_G_k_w is the C++ domain reversed:
#   (w, k, s1, b1, s0, b0). We take the spin-up (s0=s1=0), band-diagonal (b0=b1=band) slice.
def load_arm(run_dir, arm, band=0):
    files = sorted(glob.glob(os.path.join(run_dir, arm, "G_*.hdf5")),
                   key=lambda f: int(f.split("_")[-1].split(".")[0]))
    G, seeds, nops = [], [], set()
    kel = None
    for f in files:
        with h5py.File(f, "r") as h:
            g = np.array(h["functions/cluster_greens_function_G_k_w"])
            G.append(g[:, :, 0, band, 0, band])           # (w, k)
            seeds.append(int(h["metadata/seed"][()][0]))
            nops.add(int(h["metadata/n_ops"][()][0]))
            kel = np.array([np.array(x) for x in h["metadata/k_elements"]])
    return np.array(G), np.array(seeds), kel, nops   # G: (replica, w, k)

# ---- D4 orbits of the cluster momentum mesh, reimplemented independently in Python (plan §7.2) ----
_D4 = [(1,0,0,1),(0,-1,1,0),(-1,0,0,-1),(0,1,-1,0),      # E, C4, C2, C4^3
       (1,0,0,-1),(-1,0,0,1),(0,1,1,0),(0,-1,-1,0)]      # sigma_x, sigma_y, sigma_/, sigma_\

def _wrap(v):
    return tuple(round(((c + np.pi) % (2*np.pi)) - np.pi, 5) for c in v)

def _orbit_key(k):
    kx, ky = k
    return min(_wrap((a*kx+b*ky, c*kx+d*ky)) for (a,b,c,d) in _D4)

def build_orbits(kel):
    from collections import defaultdict
    orb = defaultdict(list)
    for i, k in enumerate(kel):
        orb[_orbit_key(k)].append(i)
    # sort members, return list of (representative-array, members) sorted by size then key
    out = [(np.array(key), sorted(idx)) for key, idx in orb.items()]
    out.sort(key=lambda t: (len(t[1]), tuple(t[0])))
    return out

# ---- variance across replicas, per (w,k). np.var on complex data gives E|G-<G>|^2. ----
def replica_variance(G):          # G:(rep,w,k) -> (w,k) real
    return G.var(axis=0, ddof=1).real

# ---- pairwise noise correlation between orbit-mates in one arm, pooled over replicas and w ----
def orbit_rho(G, members):
    if len(members) < 2:
        return np.nan
    res = G[:, :, members] - G[:, :, members].mean(axis=0, keepdims=True)   # (rep,w,m)
    cs = []
    for a in range(len(members)):
        for b in range(a+1, len(members)):
            xa = res[:, :, a].ravel(); xb = res[:, :, b].ravel()
            cs.append((np.vdot(xa, xb) / np.sqrt(np.vdot(xa, xa) * np.vdot(xb, xb))).real)
    return float(np.mean(cs))

def predicted_ratio(n, rho):
    return n / (1.0 + (n - 1.0) * rho)

# ---- bootstrap CI for the pooled Var_off/Var_on ratio over a set of (k) columns ----
def bootstrap_ratio_ci(Goff, Gon, cols, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    nrep = Goff.shape[0]
    vals = []
    for _ in range(n_boot):
        bs = rng.integers(0, nrep, nrep)
        voff = Goff[bs][:, :, cols].var(axis=0, ddof=1).real.mean()
        von  = Gon[bs][:, :, cols].var(axis=0, ddof=1).real.mean()
        vals.append(voff / von)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
