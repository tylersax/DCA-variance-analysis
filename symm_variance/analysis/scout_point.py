"""Gate-1 scouting readout for one run: structure, filling, and where the variance actually lives.

Usage: scout_point.py <run.hdf5> [<run.hdf5> ...]

This answers the questions the ROADMAP's gate 1 asks before any ensemble is worth launching, and one
the original gate list did not: **which classes carry the raw variance**. A model can pass every
structural check -- right n_ops, right band-equivalence classes, orbits the expected sizes -- and
still be useless, because the block whose behaviour the run is meant to measure carries ~0% of the
variance. Threeband at the repo test's (ep_p = 3, mu = 0) does exactly that: the p orbitals sit ~3
above the d level, come out nearly empty, and the d-d block takes 100.00% of the raw variance, so the
equivalent-orbital block that is the entire point of the run has no signal in it.

So `w` (the class share of raw variance) is a GATE, not a report. Numbers printed here are scouting
numbers: at 16 ranks x 512/rank they are below any plausible depth floor and must not be quoted.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_sweep as msw
import reduction_map as rm
from symm_variance_lib import Run, projector_identities, g0_invariance


def scout(path, verbose=True):
    run = Run(path)
    eq = msw.band_equivalence(run)
    bpc = msw.band_permutation_content(run)
    occ = msw.band_occupancy(run)
    ok, why = msw.square_mesh(run)
    pid = projector_identities(run)
    g0 = g0_invariance(run)

    rows, recon, agg = rm.reduction_table(run, msw.band_pair_classes(run, "equivalence"))
    rows_p, _, _ = rm.reduction_table(run, msw.band_pair_classes(run, "pair"))

    out = dict(
        path=os.path.basename(path), model=run.model, nb=run.nb, nk=run.nk, n_ops=run.n_ops,
        beta=run.beta, mu=run.mu, n_ranks=run.n_ranks,
        band_classes=eq["classes"], n_permutations=bpc["n_permutations"],
        permutations=bpc["permutations"], n_offblock=bpc["n_offblock"],
        orbit_sizes=msw.orbit_size_histogram(run), n_forced_null=bpc["n_forced_null"],
        occupancy=occ, square_mesh=ok, mesh_note=why,
        projector=pid, g0_invariance=g0, R_full=agg,
        by_class={r["cls"]: dict(w=r["w"], R_C=r["R_C"], n=r["n"]) for r in rows},
        by_pair={r["cls"]: dict(w=r["w"], R_C=r["R_C"], n=r["n"]) for r in rows_p},
    )
    if verbose:
        print(f"=== {out['model']}  nb={run.nb} nk={run.nk} |G|={run.n_ops} "
              f"beta={run.beta} mu={run.mu}  ({run.n_ranks} ranks) ===")
        print(f"  band-equivalence classes : {eq['classes']}")
        print(f"  band permutations        : {sorted(bpc['permutations'])}  "
              f"(n={bpc['n_permutations']}; 1 == band machinery dormant)")
        print(f"  orbit sizes (non-null)   : {out['orbit_sizes']}   forced nulls: {bpc['n_forced_null']}")
        if occ:
            print(f"  occupancy per band       : {[round(x, 3) for x in occ['per_band']]}  "
                  f"total={occ['total']:.3f} of {2 * run.nb}  "
                  f"({occ['filled_fraction']:.1%} of a filled cell)")
        print(f"  projector identities     : {'PASS' if pid['passes'] else 'FAIL'} "
              f"(idem {pid['idempotence']:.1e}, sym {pid['symmetry']:.1e}, "
              f"trace {pid['trace']:.0f} vs {pid['n_nonnull_orbits']} orbits)")
        print(f"  analytic-G0 oracle       : "
              + ("n/a" if g0 is None else f"{g0:.2e} -> {'PASS' if g0 < 1e-12 else 'FAIL'}"))
        print(f"  R (scouting only)        : {agg:.4f}")
        print(f"  {'class':<32}{'n':>4}{'var share w':>13}{'R_C':>10}")
        for r in rows:
            rc = "inf" if not np.isfinite(r["R_C"]) else f"{r['R_C']:.3f}"
            flag = "  <-- carries no variance" if r["w"] < 1e-4 else ""
            print(f"  {r['cls']:<32}{r['n']:>4}{r['w']:>13.4f}{rc:>10}{flag}")
        # The gate: is there signal in the block this run exists to measure? Only meaningful for a
        # model that HAS equivalent orbitals -- at nb=1 there is no such block by construction, and
        # flagging its absence would be a false alarm on the single-band reference points.
        if len(eq["classes"]) < run.nb or any(len(c) > 1 for c in eq["classes"]):
            eqw = sum(v["w"] for k, v in out["by_class"].items()
                      if "equivalent" in k or "orbit >1" in k)
            verdict = ("USABLE" if eqw > 0.02 else
                       "VACUOUS -- equivalent-orbital block carries no variance")
            print(f"  equivalent-orbital variance share = {eqw:.4f}  -> {verdict}")
            out["equivalent_orbital_share"] = eqw
        else:
            print(f"  (nb={run.nb}, no symmetry-equivalent orbital pairs -- "
                  f"equivalent-orbital gate not applicable)")
        print()
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        scout(p)
