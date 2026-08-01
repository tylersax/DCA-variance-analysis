"""Core: load a run, apply the point-group operator P, compute variances and ratios.

Plain numpy + h5py. No DCA imports, so this runs anywhere the shipped data lands.

This directory is a FROZEN, TRIMMED SNAPSHOT of the working analysis tier that produced the
measurements (`../symm_variance/analysis/`). It carries only what the hero notebook needs. It is not
a second live copy: fixes belong upstream first, then get re-snapshotted here.

The one invariant worth stating up front, because everything else depends on it: orbits and their
per-member signs are read from the SUPPORT OF P, the operator the simulation itself serialized --
never inferred by looking for equal values of G. Orbit mates can carry a relative sign, and
value-matching silently splits one signed orbit into two +/- classes, corrupting every orbit size and
manufacturing a spurious perfect correlation.

Data contract (written by symm_variance_main.inc):
  metadata/{model,seed,beta,n_ops,n_ranks,measurements_per_rank,local_size,k_elements}
  raw/{G_raw_re,G_raw_im}       [n_rank] of [local_size]  raw per-rank G, C++ leaf order
  raw/{sign_re,sign_im}         [n_rank]                  per-rank accumulated phase
  functions/cluster_greens_function_G_k_w   (w,k,s1,b1,s0,b0)  production symmetrized mean
  symmetrization/{P,flat_labels,nb,nk,n_ops}
      P            [E][E] real, P[out][in] -- the orbit average on (b0,b1,k)
      flat_labels  [E][3] = (b0,b1,k),  flat index = (b0*nb + b1)*nk + k
"""
from pathlib import Path

import h5py
import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def _stack_object(arr):
    """h5py reads vector<vector<T>> as a (n,) object array of variable-length rows -> dense (n,m)."""
    return np.array([np.asarray(row) for row in arr])


# =================================================================================================
# Orbit structure -- a property of P alone, so it works for a full run and for an operator-only point
# =================================================================================================
def orbits_from_P(P, tol=1e-9):
    """Connected components of P's support -> list of dicts with members and per-member signs.

    A row that is entirely zero is a SYMMETRY-FORCED NULL: symmetrization sends that entry to exactly
    zero, because the symmetry group maps it onto minus itself. Its expectation is zero, so all of its
    sampling noise is removed outright rather than merely averaged down.
    """
    E = P.shape[0]
    seen = [False] * E
    orbs = []
    for x in range(E):
        if seen[x]:
            continue
        support = np.where(np.abs(P[x]) > tol)[0]
        if support.size == 0:
            seen[x] = True
            orbs.append(dict(members=[x], signs=[0.0], forced_null=True))
            continue
        signs = np.sign(P[x, support])
        members = list(support)
        for m in members:
            seen[m] = True
        orbs.append(dict(members=members, signs=list(signs), forced_null=False))
    return orbs


def projector_identities(P, orbs=None):
    """P must be an exact orthogonal projector. Three identities, all exact to machine zero.

    There is no statistics here and no dependence on anything production computes -- these either
    hold or the operator is wrong:

      P @ P == P     idempotence: symmetrizing twice is symmetrizing once.
      P == P.T       self-adjointness. This is what licenses Var(P G) <= Var(G) entrywise and the
                     whole r = m/[1 + (m-1)rho] law; a non-orthogonal averaging operator has neither.
      tr(P) == number of non-null orbits
                     the rank of a projector is its trace, and the invariant subspace has exactly one
                     dimension per non-null orbit. Forced-null rows are all-zero and contribute
                     nothing, so this simultaneously checks the null bookkeeping.
    """
    orbs = orbs if orbs is not None else orbits_from_P(P)
    n_orbits = sum(1 for o in orbs if not o["forced_null"])
    idem = float(np.abs(P @ P - P).max())
    symm = float(np.abs(P - P.T).max())
    tr = float(np.trace(P))
    return dict(idempotence=idem, symmetry=symm, trace=tr, n_nonnull_orbits=n_orbits,
                trace_gap=abs(tr - n_orbits),
                passes=bool(idem < 1e-9 and symm < 1e-9 and abs(tr - n_orbits) < 1e-9))


def orbit_size_histogram(orbs):
    """{m: count} over non-null orbits -- the geometry available to symmetrization."""
    h = {}
    for o in orbs:
        if not o["forced_null"]:
            h[len(o["members"])] = h.get(len(o["members"]), 0) + 1
    return dict(sorted(h.items()))


def band_equivalence_classes(P, labels, nb, tol=1e-9):
    """Which bands the symmetry group actually permutes into each other.

    Two bands are in the same class iff P has support connecting an entry of one to an entry of the
    other. This is DERIVED from the operator, not declared: DCA solves for the orbital operator from
    H0 for each candidate spatial op and silently keeps the ones that work, so the band content of the
    symmetry group is a measured property of the model, not an input.
    """
    parent = list(range(nb))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    E = P.shape[0]
    for out in range(E):
        for inp in np.where(np.abs(P[out]) > tol)[0]:
            for col in (0, 1):  # b0 and b1 of the (b0, b1, k) label
                union(int(labels[out][col]), int(labels[inp][col]))
    groups = {}
    for b in range(nb):
        groups.setdefault(find(b), []).append(b)
    return [sorted(v) for v in groups.values()]


def off_block_fraction(P, labels, tol=1e-9):
    """Share of P's nonzero entries that connect DIFFERENT band pairs.

    Zero for a single-band model by construction. Nonzero only when the group genuinely permutes
    bands, so it is the cleanest one-number answer to "is the band machinery live in this model?"
    """
    nz = off = 0
    E = P.shape[0]
    for out in range(E):
        for inp in np.where(np.abs(P[out]) > tol)[0]:
            nz += 1
            if (labels[out][0], labels[out][1]) != (labels[inp][0], labels[inp][1]):
                off += 1
    return off / nz if nz else 0.0


# =================================================================================================
# An operator-only design point: structure without sample data
# =================================================================================================
class OpsView:
    """P and its labels for a design point whose raw run is too large to ship.

    Every STRUCTURAL claim -- orbit sizes, band classes, forced nulls, off-block content, the m-ceiling
    -- is a property of P, so those stay live for these points. Nothing statistical is available.
    """

    def __init__(self, label):
        z = np.load(DATA / "ops" / f"{label}.npz", allow_pickle=True)
        self.label = label
        self.P = z["P"]
        self.labels = z["flat_labels"]
        self.nb = int(z["meta_nb"])
        self.nk = int(z["meta_nk"])
        self.n_ops = int(z["meta_n_ops"])
        self.beta = float(z["meta_beta"])
        self.model = str(z["meta_model"])
        self.E = self.P.shape[0]

    def orbits(self):
        return orbits_from_P(self.P)


# =================================================================================================
# A full run
# =================================================================================================
class Run:
    """One design point with its raw per-rank samples.

    One MPI rank is one independent sample: DCA seeds per-rank RNG streams by
    hash(global_id + base_seed), so the 64 ranks of a run are 64 independent replicates of the same
    measurement. The driver calls local_G_k_w(symmetrize=false) on every rank BEFORE finalize(),
    giving a genuinely raw G regardless of what point group the binary declares, and gathers the
    stack to rank 0.
    """

    def __init__(self, label, path=None):
        path = Path(path) if path else DATA / "runs" / f"{label}.hdf5"
        self.label = label
        with h5py.File(path, "r") as h:
            self.model = h["metadata/model"][()][0].decode()
            self.n_ops = int(h["metadata/n_ops"][()][0])
            self.n_ranks = int(h["metadata/n_ranks"][()][0])
            self.seed = int(h["metadata/seed"][()][0])
            self.m_per_rank = int(h["metadata/measurements_per_rank"][()][0])
            self.nb = int(h["symmetrization/nb"][()][0])
            self.nk = int(h["symmetrization/nk"][()][0])
            self.P = _stack_object(h["symmetrization/P"][()])
            self.labels = np.asarray(h["symmetrization/flat_labels"][()])
            re = _stack_object(h["raw/G_raw_re"][()])
            im = _stack_object(h["raw/G_raw_im"][()])
            self.sign = np.asarray(h["raw/sign_re"][()]) + 1j * np.asarray(h["raw/sign_im"][()])
            self.G_final = np.asarray(h["functions/cluster_greens_function_G_k_w"][()])
            self.k_elements = np.array([np.asarray(x) for x in h["metadata/k_elements"][()]])
            # beta was added to the driver on 2026-07-28; the FeAs ensemble predates it. The bundle
            # manifest carries the value from the input template in that case -- see notebook section 2.
            self.beta = float(h["metadata/beta"][()][0]) if "metadata/beta" in h else None

        R = re.shape[0]
        ns = 2
        nw = re.shape[1] // (self.nk * ns * ns * self.nb * self.nb)
        self.nw, self.ns = nw, ns
        # C++ leaf order fastest->slowest is (b0,s0,b1,s1,k,w), so a C-order reshape of the flat row
        # is the reversed tuple.
        self.G_raw = (re + 1j * im).reshape(R, nw, self.nk, ns, self.nb, ns, self.nb)
        self.E = self.P.shape[0]
        assert self.E == self.nb * self.nb * self.nk, (self.E, self.nb, self.nk)
        self._vec_cache = None

    # ---- layout -------------------------------------------------------------------------------
    @property
    def vec(self):
        """Per-rank raw samples in P's (b0,b1,k) convention: (n_rank, nw, ns, E). Cached."""
        if self._vec_cache is None:
            self._vec_cache = self._to_vec(self.G_raw)
        return self._vec_cache

    def _to_vec(self, G):
        """G[..., k, s1, b1, s0, b0] spin-diagonal blocks -> Gvec[..., s, E], E flat over (b0,b1,k)."""
        nb, nk = self.nb, self.nk
        out = np.zeros(G.shape[:-5] + (self.ns, self.E), dtype=complex)
        for b0 in range(nb):
            for b1 in range(nb):
                for k in range(nk):
                    e = (b0 * nb + b1) * nk + k
                    for s in range(self.ns):
                        out[..., s, e] = G[..., k, s, b1, s, b0]
        return out

    def _from_vec(self, vec, like):
        nb, nk = self.nb, self.nk
        out = np.zeros_like(like)
        for b0 in range(nb):
            for b1 in range(nb):
                for k in range(nk):
                    e = (b0 * nb + b1) * nk + k
                    for s in range(self.ns):
                        out[..., k, s, b1, s, b0] = vec[..., s, e]
        return out

    # ---- the operator -------------------------------------------------------------------------
    def apply_P(self, vec):
        """Orbit average: Gsym[..., out] = sum_in P[out, in] G[..., in]."""
        return np.einsum("oi,...i->...o", self.P, vec)

    def orbits(self):
        return orbits_from_P(self.P)

    # ---- variance -----------------------------------------------------------------------------
    def variance_ratios(self):
        """Raw and symmetrized per-entry variance over the rank axis, shaped (nw, ns, E).

        Real and imaginary parts are treated as separate real random variables, which is what the
        variance of a complex estimator means here.

        THE PAIRED DESIGN. P is a deterministic linear operator, so both variances come from the SAME
        set of raw samples -- the identical noise realizations appear in numerator and denominator.
        The ratio is therefore paired and common fluctuations largely cancel, which is dramatically
        tighter than comparing two independent symmetrization-on / symmetrization-off experiments and
        needs no special no-symmetry build of any model.
        """
        vec = self.vec
        vsym = self.apply_P(vec)
        vr = vec.real.var(0, ddof=1) + 1j * vec.imag.var(0, ddof=1)
        vs = vsym.real.var(0, ddof=1) + 1j * vsym.imag.var(0, ddof=1)
        return dict(var_raw=vr, var_sym=vs, vec=vec, vsym=vsym)

    def R(self):
        """The headline ratio for this single run, summed over every entry (full support)."""
        vr = self.variance_ratios()
        A = vr["var_raw"].real + vr["var_raw"].imag
        B = vr["var_sym"].real + vr["var_sym"].imag
        return float(A.sum() / B.sum())

    def orbit_rho(self, members, signs):
        """Mean pairwise noise correlation between sign-aligned orbit mates, pooled over ranks, w, s.

        Signs are applied first, so odd mates are aligned before correlating -- otherwise a signed
        orbit reports a spurious -1.
        """
        if len(members) < 2:
            return np.nan
        X = self.vec[..., members] * np.asarray(signs)[None, None, None, :]
        res = X - X.mean(0, keepdims=True)
        cs = []
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                xa, xb = res[..., a].ravel(), res[..., b].ravel()
                d = np.sqrt(np.vdot(xa, xa) * np.vdot(xb, xb))
                if abs(d) > 0:
                    cs.append((np.vdot(xa, xb) / d).real)
        return float(np.mean(cs)) if cs else np.nan

    # ---- means --------------------------------------------------------------------------------
    def phase_weighted_mean(self):
        """sum_i(sign_i G_i) / sum_i(sign_i) -- production's GLOBAL sign normalization.

        Not the same as G_raw.mean(0), which uses each rank's own phase. On a model with a sign
        problem the difference is real and largest at low Matsubara frequency, and it is the
        phase-weighted mean that symmetrization preserves.
        """
        w = self.sign.reshape((-1,) + (1,) * (self.G_raw.ndim - 1))
        return (w * self.G_raw).sum(0) / self.sign.sum()


def predicted_r(m, rho):
    """The design effect. r = m / [1 + (m-1) rho].

    Standard survey-sampling result (Kish 1965): averaging m samples whose pairwise correlation is
    rho gives an effective sample size of m/[1 + (m-1)rho], not m. Cited, not derived -- what is new
    here is measuring rho for quantum Monte Carlo noise under a lattice point group.
    """
    return m / (1.0 + (m - 1.0) * rho)


def full_symmetrize(run, G_mean):
    """Reproduce production's Symmetrize::execute in numpy: spin, then cluster, then frequency.

    Used for exactly one purpose -- checking that our symmetrization of the raw ensemble mean
    reproduces the finalized G_k_w the simulation wrote. If that holds to machine precision, the
    operator we measure with is the operator production applies.
    """
    nb, nk, ns, nw = run.nb, run.nk, run.ns, run.nw
    G = G_mean.copy()

    # (1) spin: average the two spin-diagonal blocks, zero the spin-off-diagonal.
    diag = 0.5 * (G[:, :, 0, :, 0, :] + G[:, :, 1, :, 1, :])
    G[:, :, 0, :, 0, :] = diag
    G[:, :, 1, :, 1, :] = diag
    G[:, :, 0, :, 1, :] = 0.0
    G[:, :, 1, :, 0, :] = 0.0

    # (2) cluster: the point-group orbit average P, per spin, per frequency.
    G = run._from_vec(run.apply_P(run._to_vec(G)), G)

    # (3) frequency: pair w with -w via conjugation plus band transpose, at the same k.
    w0 = nw - 1
    Gn = G.copy()
    for w in range(nw // 2):
        for k in range(nk):
            for b0 in range(nb):
                for b1 in range(nb):
                    for s in range(ns):
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
