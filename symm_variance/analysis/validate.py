"""Validation ladder for a symm_variance run. Usage: validate.py <run.hdf5> [--complex-g0]"""
import sys
import numpy as np
from symm_variance_lib import (Run, predicted_r, full_symmetrize, projector_identities,
                               g0_invariance)

path = sys.argv[1]
complex_g0 = "--complex-g0" in sys.argv
run = Run(path)
print(f"model={run.model}  n_ranks={run.n_ranks}  n_ops={run.n_ops}  "
      f"nb={run.nb} nk={run.nk} nw={run.nw}  E={run.E}")

vr = run.variance_ratios()
var_raw, var_sym = vr["var_raw"], vr["var_sym"]   # (nw, ns, E) complex (re-var + i im-var)

orbs = run.orbits()
sizes = [len(o["members"]) for o in orbs if not o["forced_null"]]
print(f"\norbits: {len(orbs)}  sizes(non-null)={sorted(set(sizes))}  "
      f"forced_null={sum(o['forced_null'] for o in orbs)}")

# ---- RUNG 1a: singletons pinned at r = 1 exactly --------------------------------------------------
print("\n[rung 1a] singleton orbits (m=1) must give r = 1 exactly:")
max_singleton_dev = 0.0
for o in orbs:
    if o["forced_null"] or len(o["members"]) != 1:
        continue
    x = o["members"][0]
    # ratio over all (w,s); raw/sym should be identical -> r = 1
    r_re = var_raw.real[:, :, x] / var_sym.real[:, :, x]
    dev = np.nanmax(np.abs(r_re - 1.0))
    max_singleton_dev = max(max_singleton_dev, dev)
print(f"  max |r-1| over singletons = {max_singleton_dev:.3e}  "
      f"-> {'PASS' if max_singleton_dev < 1e-9 else 'FAIL'}")

# ---- RUNG 1b: per-orbit r vs m/[1+(m-1)rho] -------------------------------------------------------
print("\n[rung 1b] per-orbit measured r vs predicted m/[1+(m-1)rho]:")
print(f"  {'m':>3} {'rho':>8} {'r_pred':>8} {'r_meas':>8} {'ratio':>7}")
for o in sorted(orbs, key=lambda z: len(z["members"])):
    if o["forced_null"] or len(o["members"]) < 2:
        continue
    m = len(o["members"])
    rho = run.orbit_rho(o["members"], o["signs"])
    rp = predicted_r(m, rho)
    # measured r pooled over the orbit's members and all (w,s), real+imag
    xs = o["members"]
    vr_pool = var_raw.real[:, :, xs].mean() + var_raw.imag[:, :, xs].mean()
    vs_pool = var_sym.real[:, :, xs].mean() + var_sym.imag[:, :, xs].mean()
    rm = vr_pool / vs_pool
    print(f"  {m:>3} {rho:>8.4f} {rp:>8.3f} {rm:>8.3f} {rm/rp:>7.3f}")

# ---- RUNG 1c: r flat in omega (point group does not touch Matsubara) ------------------------------
# pool non-singleton, non-null entries per w; report slope of r(w) vs w.
mask = np.zeros(run.E, bool)
for o in orbs:
    if not o["forced_null"] and len(o["members"]) > 1:
        for x in o["members"]:
            mask[x] = True
rw = (var_raw.real[:, :, mask].sum((1, 2)) + var_raw.imag[:, :, mask].sum((1, 2))) / \
     (var_sym.real[:, :, mask].sum((1, 2)) + var_sym.imag[:, :, mask].sum((1, 2)))
w_idx = np.arange(run.nw)
slope = np.polyfit(w_idx, rw, 1)[0]
print(f"\n[rung 1c] r(omega): mean={rw.mean():.3f} std={rw.std():.3f} "
      f"slope/step={slope:.2e}  (flat expected)")

# ---- RUNG 1d: P is an exact orthogonal projector --------------------------------------------------
# Pure algebra on P alone -- no statistics, no dependence on anything production computes. This is the
# rung that carries the weight when rung 2 fails: 1d passing while 2 fails localizes the disagreement
# to production's Symmetrize::execute rather than to our operator.
pid = projector_identities(run)
print(f"\n[rung 1d] P is an orthogonal projector:")
print(f"  max|P@P - P|      = {pid['idempotence']:.3e}   (idempotence)")
print(f"  max|P - P^T|      = {pid['symmetry']:.3e}   (self-adjoint)")
print(f"  trace(P)          = {pid['trace']:.6f}  vs {pid['n_nonnull_orbits']} non-null orbits "
      f"(gap {pid['trace_gap']:.3e})")
print(f"  -> {'PASS' if pid['passes'] else 'FAIL'}")

# ---- ORACLE: P leaves the deterministic analytic G0 invariant -------------------------------------
# G0(k,iw) = [(iw+mu)I - H0(k)]^-1 is exactly invariant under every symmetry of H0, so this tests P
# against an object production never touched. Requires the H_DCA / chemical_potential driver keys;
# runs predating them report n/a rather than failing.
g0_dev = g0_invariance(run)
if g0_dev is None:
    print("\n[oracle]  n/a -- run predates the H_DCA / chemical_potential driver keys")
else:
    print(f"\n[oracle]  max|P@G0_analytic - G0_analytic| / max|G0| = {g0_dev:.3e}  "
          f"-> {'PASS' if g0_dev < 1e-12 else 'FAIL'}")
    print("          (G0 rebuilt from H0; the STORED G0_k_w is production-symmetrized and cannot serve)")

# ---- headline R (point-group), full and non-null support ------------------------------------------
def R_over(mask_e):
    num = var_raw.real[:, :, mask_e].sum() + var_raw.imag[:, :, mask_e].sum()
    den = var_sym.real[:, :, mask_e].sum() + var_sym.imag[:, :, mask_e].sum()
    return num / den
full_mask = np.ones(run.E, bool)
nonnull_mask = np.array([not any(x in o["members"] and o["forced_null"] for o in orbs)
                         for x in range(run.E)])
print(f"\nHEADLINE R (point-group): full-support={R_over(full_mask):.4f}  "
      f"non-null-support={R_over(nonnull_mask):.4f}")

# ---- RUNG 2: mean preservation -- full symmetrize(phase-weighted mean raw) == finalized G ---------
# NB: the finalized G_k_w is a valid full-ensemble reference only under error-computation-type NONE.
# Under JACK_KNIFE, finalize writes a leave-one-out replicate (rank 0's), so this rung shows a ~1/N
# low-frequency gap by construction -- that is the jackknife, not a pipeline error.
# The phase-weighted mean reproduces production's GLOBAL sign normalization (needed when per-rank
# signs differ, e.g. a model with a sign problem); for a sign-free model it equals the plain mean.
den = np.abs(run.G_final).max()
G_pw = full_symmetrize(run, run.phase_weighted_mean(), complex_g0=complex_g0)
diff_pw = np.abs(G_pw - run.G_final).max()
print(f"\n[rung 2] max|full_symmetrize(phase-weighted mean raw) - finalized G_k_w| = "
      f"{diff_pw:.3e} (rel {diff_pw/den:.2e})  -> {'PASS' if diff_pw/den < 1e-8 else 'FAIL'}")
print("         (requires error-computation-type=NONE; JACK_KNIFE writes a leave-one-out replicate)")
