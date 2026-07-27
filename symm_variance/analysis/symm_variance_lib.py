"""Analysis tier for the symmetrization-variance measurement.

Plain numpy + h5py, no DCA imports, so it runs anywhere the driver's HDF5 lands. It consumes the
authoritative point-group operator P serialized by the driver (orbit_table.hpp) -- it never detects
orbits from G values (that silently splits signed orbits into +/- classes).

Data contract (one file per model, written by symm_variance_main.inc):
  metadata/{model,seed,n_ops,n_ranks,measurements_per_rank,local_size,k_elements}
  raw/{G_raw_re,G_raw_im}          [n_rank] of [local_size]  -- raw per-rank G, C++ leaf order
  functions/cluster_greens_function_G_k_w   (w,k,s1,b1,s0,b0) -- production symmetrized ensemble mean
  symmetrization/{P,flat_labels,nb,nk,n_ops,flat_index_order}
      P           [E][E] real operator, P[out][in], the point-group orbit average on (b0,b1,k)
      flat_labels [E][3] = (b0,b1,k), flat index = (b0*nb+b1)*nk+k
"""
import numpy as np
import h5py


def _stack_object(arr):
    """h5py reads vector<vector<T>> as a (n,) object array of variable-length rows -> dense (n,m)."""
    return np.array([np.asarray(row) for row in arr])


class Run:
    def __init__(self, path):
        with h5py.File(path, "r") as h:
            self.model = h["metadata/model"][()][0].decode() if h["metadata/model"].dtype.kind == "S" \
                else str(h["metadata/model"][()][0])
            self.n_ops = int(h["metadata/n_ops"][()][0])
            self.n_ranks = int(h["metadata/n_ranks"][()][0])
            self.nb = int(h["symmetrization/nb"][()][0])
            self.nk = int(h["symmetrization/nk"][()][0])
            self.P = _stack_object(h["symmetrization/P"][()])          # (E, E)
            self.labels = np.asarray(h["symmetrization/flat_labels"][()])  # (E, 3) = (b0,b1,k)
            re = _stack_object(h["raw/G_raw_re"][()])                  # (R, local_size)
            im = _stack_object(h["raw/G_raw_im"][()])
            self.sign = np.asarray(h["raw/sign_re"][()]) + 1j * np.asarray(h["raw/sign_im"][()])  # (R,)
            self.G_final = np.asarray(h["functions/cluster_greens_function_G_k_w"][()])  # (w,k,s1,b1,s0,b0)
            self.k_elements = np.array([np.asarray(x) for x in h["metadata/k_elements"][()]])

        R = re.shape[0]
        nb, nk = self.nb, self.nk
        ns = 2
        nw = re.shape[1] // (nk * ns * ns * nb * nb)
        self.nw, self.ns = nw, ns
        # C++ leaf order fastest->slowest is (b0,s0,b1,s1,k,w); a C-order reshape of the flat row is
        # therefore the reversed tuple (w,k,s1,b1,s0,b0).
        self.G_raw = (re + 1j * im).reshape(R, nw, nk, ns, nb, ns, nb)  # [r,w,k,s1,b1,s0,b0]

        self.E = self.P.shape[0]
        assert self.E == nb * nb * nk, (self.E, nb, nk)
        self._vec_cache = None

    @property
    def vec(self):
        """The per-rank raw samples in P's (b0,b1,k) convention: (n_rank, nw, ns, E).

        Cached: `_to_vec` is a Python-level scatter and every variance/orbit query needs the same
        array, so rebuilding it per call dominates the runtime on a ladder of runs. `G_raw` is never
        mutated after load, so the cache cannot go stale.
        """
        if self._vec_cache is None:
            self._vec_cache = self._to_vec(self.G_raw)
        return self._vec_cache

    # ---- flatten a spin-diagonal (b0,b1,k) block to the P convention -------------------------------
    def _to_vec(self, G):
        """G[...,k,s1,b1,s0,b0] spin-diagonal blocks -> Gvec[...,s,E], E flat over (b0,b1,k)."""
        nb, nk = self.nb, self.nk
        lead = G.shape[:-5]  # e.g. (r, w) or (w,)
        out = np.zeros(lead + (self.ns, self.E), dtype=complex)
        for b0 in range(nb):
            for b1 in range(nb):
                for k in range(nk):
                    e = (b0 * nb + b1) * nk + k
                    for s in range(self.ns):
                        out[..., s, e] = G[..., k, s, b1, s, b0]
        return out

    def _from_vec(self, vec, like):
        """Inverse of _to_vec: scatter Gvec[...,s,E] back into a (...,k,s1,b1,s0,b0) array."""
        nb, nk = self.nb, self.nk
        out = np.zeros_like(like)
        for b0 in range(nb):
            for b1 in range(nb):
                for k in range(nk):
                    e = (b0 * nb + b1) * nk + k
                    for s in range(self.ns):
                        out[..., k, s, b1, s, b0] = vec[..., s, e]
        return out

    def apply_P(self, vec):
        """Point-group orbit average: Gsym[...,out] = sum_in P[out,in] G[...,in]."""
        return np.einsum("oi,...i->...o", self.P, vec)

    def phase_weighted_mean(self):
        """Sum_i(sign_i G_i) / Sum_i(sign_i): reproduces production's GLOBAL sign normalization
        (M_bar = sum signM_i / sum sign_i), whereas G_raw.mean(0) uses each rank's own phase. The two
        agree only in the infinite-sample limit; the difference is largest at low Matsubara frequency."""
        w = self.sign.reshape((-1,) + (1,) * (self.G_raw.ndim - 1))
        return (w * self.G_raw).sum(0) / self.sign.sum()

    # ---- variance ratios (real & imag independently, over the rank axis) ---------------------------
    def variance_ratios(self):
        """Returns dict with raw/sym per-entry variances and the ratio r, all shaped (nw, ns, E)."""
        vec = self.vec                                # (R, nw, ns, E)
        vsym = self.apply_P(vec)                      # (R, nw, ns, E)
        vr = vec.real.var(0, ddof=1) + 1j * vec.imag.var(0, ddof=1)
        vs = vsym.real.var(0, ddof=1) + 1j * vsym.imag.var(0, ddof=1)
        return dict(var_raw=vr, var_sym=vs, vec=vec, vsym=vsym)

    # ---- orbits & signs derived from the SUPPORT of P (not from G values) --------------------------
    def orbits(self):
        """Connected components of P's support -> list of (members, signs). Rows that are all-zero
        are symmetry-forced nulls (returned as singleton 'null' orbits with sign 0)."""
        E = self.E
        tol = 1e-9
        seen = [False] * E
        orbs = []
        for x in range(E):
            if seen[x]:
                continue
            support = np.where(np.abs(self.P[x]) > tol)[0]
            if support.size == 0:  # forced null: Sym G[x] == 0 identically
                seen[x] = True
                orbs.append(dict(members=[x], signs=[0.0], forced_null=True))
                continue
            signs = np.sign(self.P[x, support])
            members = list(support)
            for m in members:
                seen[m] = True
            orbs.append(dict(members=members, signs=list(signs), forced_null=False))
        return orbs

    def orbit_rho(self, members, signs):
        """Mean pairwise noise correlation between sign-aligned orbit-mates, pooled over ranks, w, s."""
        if len(members) < 2:
            return np.nan
        X = self.vec[..., members] * np.asarray(signs)[None, None, None, :]
        res = X - X.mean(0, keepdims=True)
        cs = []
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                xa = res[..., a].ravel()
                xb = res[..., b].ravel()
                d = np.sqrt(np.vdot(xa, xa) * np.vdot(xb, xb))
                if abs(d) > 0:
                    cs.append((np.vdot(xa, xb) / d).real)
        return float(np.mean(cs)) if cs else np.nan


def predicted_r(m, rho):
    return m / (1.0 + (m - 1.0) * rho)


# ---- full production symmetrization reproduced in numpy (for the mean-preservation rung) -----------
def full_symmetrize(run, G_mean, complex_g0=False):
    """Apply the three reductions Symmetrize::execute bundles, in production order (spin, cluster,
    frequency), to an ensemble-mean G[w,k,s1,b1,s0,b0]. Should reproduce the finalized G_k_w."""
    nb, nk, ns, nw = run.nb, run.nk, run.ns, run.nw
    G = G_mean.copy()

    # (1) spin: average the two spin-diagonal blocks, zero the spin-off-diagonal.
    diag = 0.5 * (G[:, :, 0, :, 0, :] + G[:, :, 1, :, 1, :])
    G[:, :, 0, :, 0, :] = diag
    G[:, :, 1, :, 1, :] = diag
    G[:, :, 0, :, 1, :] = 0.0
    G[:, :, 1, :, 0, :] = 0.0

    # (2) cluster point-group orbit average P, per spin, per w.
    vec = run._to_vec(G)                       # (nw, ns, E)
    vec = run.apply_P(vec)
    G = run._from_vec(vec, G)

    # (3) frequency: pair w with w0-w via conjugation + band transpose (momentum: same k).
    w0 = nw - 1
    Gn = G.copy()
    for w in range(nw // 2):
        for k in range(nk):
            for b0 in range(nb):
                for b1 in range(nb):
                    for s in range(ns):
                        if complex_g0:
                            t1 = G[w, k, s, b1, s, b0]
                            t2 = G[w0 - w, k, s, b0, s, b1]  # F(w)=conj(F^t(-w))
                            tmp = (t1 + np.conj(t2)) / 2.0
                            Gn[w, k, s, b1, s, b0] = tmp
                            Gn[w0 - w, k, s, b0, s, b1] = np.conj(tmp)
                        else:
                            t1 = G[w, k, s, b1, s, b0]
                            t2 = G[w0 - w, k, s, b0, s, b1]
                            t3 = G[w0 - w, k, s, b1, s, b0]
                            t4 = G[w, k, s, b0, s, b1]
                            tmp = (t1 + np.conj(t2) + np.conj(t3) + t4) / 4.0
                            Gn[w, k, s, b1, s, b0] = tmp
                            Gn[w0 - w, k, s, b0, s, b1] = np.conj(tmp)
                            Gn[w0 - w, k, s, b1, s, b0] = np.conj(tmp)
                            Gn[w, k, s, b0, s, b1] = tmp
    return Gn
