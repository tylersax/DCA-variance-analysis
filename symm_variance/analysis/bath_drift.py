"""Bath drift (ROADMAP 3e): does `R` hold across the iterations a production run performs?

THE QUESTION. Every other number in this project is measured at the BARE bath -- iteration 1 of a
cold-started loop, where the walker samples the non-interacting G0 (ROADMAP Conventions). A
production run instead calls the solver ~N times at similar cost each, on a bath rebuilt from the
previous iteration's measured Sigma. So a total-compute claim depends on the cost-weighted average
of `R` over the iterations actually run, and the error in quoting the bare-bath `R` is ~(N-1)/N x
drift. That is one measurable number and it is the whole point of this module.

TWO GAPS, NOT ONE -- and the second was found by measurement, not by design. The roadmap scoped 3e
as "bare bath vs iteration N", treating loop iteration 0 as the bare bath. It is not:

  gap A (coarse-graining, both at Sigma = 0)
      The single-iteration driver takes its bath from DcaData::initialize(), which sets
      G0_cluster_excluded = G0_k_w -- the BARE cluster G0 (dca_data.hpp:599). The real loop runs
      cluster exclusion first, and at Sigma_cluster = 0 that computes G0_cluster_excluded = G_k_w,
      the COARSE-GRAINED cluster Green's function (cluster_exclusion.hpp: one_plus_S_G is the
      identity, so G0_excl = G). The two differ by the dispersion's variation within each
      coarse-graining patch. So iteration 0 of a real loop is already a different bath, with
      identical Sigma. This is a FIXED OFFSET, not drift, and no iteration count reveals it.

  gap B (self-consistency)
      iteration 0 -> iteration N, the drift 3e was scoped to measure.

What a user actually experiences relative to the committed headline is A + B, so both are reported.
`reference_gap` measures A (needs matched-seed single-iteration runs); `paired_drift` measures B.

THE DESIGN IS PAIRED, TWICE OVER, because neither axis gives independent samples:
  * Iterations within a seed are serially dependent -- iteration k's bath is built from k-1's noisy
    Sigma -- so they are not independent measurements of anything. But rank i is the SAME walker
    stream at every iteration of a fixed base seed, so `m_scaling.paired_bootstrap_dR` resamples
    matched ranks and the shared randomness cancels in the difference.
  * Across seeds, the per-seed paired difference IS an independent replicate (fresh streams
    throughout), so the across-seed t-interval on those differences is the quotable one. This is the
    same two-level structure the milestone-6 paired depth test uses.

CLI:  python bath_drift.py <drift_root> [--ref <reference_dir>] [--out summary.json]
      python bath_drift.py --summary existing_summary.json
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symm_variance_lib import Run  # noqa: E402
import m_scaling as ms  # noqa: E402
import reduction_map as rm  # noqa: E402
from seed_ensemble import across_seed, t975  # noqa: E402

_ITER_RE = re.compile(r"^(?P<model>.+)_iter(?P<iter>\d+)\.hdf5$")

# The decision rule 3e fixed IN ADVANCE, kept here so the analysis cannot quietly move it: drift
# below this fraction of R retires the bare-bath caveat with one sentence; above it, the frozen-bath
# fallback earns its complexity.
DRIFT_TOLERANCE = 0.05


def parse_iter(path):
    g = _ITER_RE.match(os.path.basename(path))
    return None if not g else dict(model=g["model"], iteration=int(g["iter"]))


# ---- the bath itself, so drift is measured rather than inferred from R ------------------------------

def bath_functions(path):
    """The bath the walker sampled this iteration, plus the two functions that bracket gap A."""
    import h5py
    with h5py.File(path, "r") as h:
        f = h["functions"]
        out = dict(bath=np.asarray(f["cluster_excluded_greens_function_G0_k_w"][()]),
                   G0=np.asarray(f["free_cluster_greens_function_G0_k_w"][()]),
                   Sigma=np.asarray(f["Self_Energy"][()]))
        md = h["metadata"]
        out["mu"] = float(md["chemical_potential"][()][0])
        out["iteration"] = int(md["dca_iteration"][()][0])
        out["mixing"] = float(md["self_energy_mixing_factor"][()][0])
    return out


def _rel(a, b):
    """max|a-b| / max|b| -- a scale-free size for 'how far did this function move'."""
    d = np.abs(np.asarray(a) - np.asarray(b)).max()
    n = np.abs(np.asarray(b)).max()
    return float(d / n) if n > 0 else float("nan")


def bath_track(paths):
    """Per-iteration bath diagnostics for one seed, in iteration order.

    `bath_vs_bare_G0` is gap A expressed on the bath (nonzero already at iteration 0, at Sigma = 0);
    `bath_step` is how far the bath moved since the previous iteration, i.e. whether the loop has
    stopped moving by the last iteration. Reporting the step matters because a flat `R` across
    iterations means nothing if the bath never converged -- the two have to be read together.
    """
    rows, prev = [], None
    for p in paths:
        bf = bath_functions(p)
        rows.append(dict(iteration=bf["iteration"], mu=bf["mu"],
                         bath_vs_bare_G0=_rel(bf["bath"], bf["G0"]),
                         sigma_norm=float(np.abs(bf["Sigma"]).max()),
                         bath_step=_rel(bf["bath"], prev) if prev is not None else None))
        prev = bf["bath"]
    return rows


# ---- per-seed assembly -----------------------------------------------------------------------------

def seed_paths(seed_dir):
    """Iteration files for one seed, ordered by DCA iteration (never by glob order)."""
    ps = [p for p in glob.glob(os.path.join(seed_dir, "*_iter*.hdf5")) if parse_iter(p)]
    return sorted(ps, key=lambda p: parse_iter(p)["iteration"])


def summarize_seed(seed_dir, n_boot=400, verbose=True):
    """Summarize every iteration of one seed, plus the paired iteration-0 -> iteration-N difference."""
    paths = seed_paths(seed_dir)
    if not paths:
        return None
    iters = []
    for p in paths:
        if verbose:
            print(f"[bath_drift]   {os.path.basename(seed_dir)}/{os.path.basename(p)}", flush=True)
        s = ms.summarize(p, n_boot)
        s["iteration"] = parse_iter(p)["iteration"]
        s["path"] = os.path.abspath(p)
        iters.append(s)
    iters.sort(key=lambda s: s["iteration"])

    track = {r["iteration"]: r for r in bath_track(paths)}
    for s in iters:
        s.update({k: v for k, v in track[s["iteration"]].items() if k != "iteration"})

    out = dict(seed_dir=os.path.abspath(seed_dir), seed=iters[0]["seed"],
               model=iters[0]["model"], beta=iters[0].get("beta"),
               n_ranks=iters[0]["n_ranks"], m=iters[0]["m"], iterations=iters)

    # Paired within-seed difference, first vs last iteration, resampling matched ranks.
    if len(iters) > 1 and iters[0]["usable"] and iters[-1]["usable"]:
        a, b = Run(iters[0]["path"]), Run(iters[-1]["path"])
        out["paired_last_vs_first"] = ms.paired_bootstrap_dR(a, b, n_boot, seed=out["seed"])
        out["dR"] = iters[-1]["R_full"] - iters[0]["R_full"]
    return out


def build(root, n_boot=400, verbose=True):
    """Summarize every seed under a drift root (one subdirectory per base seed)."""
    seed_dirs = sorted(d for d in glob.glob(os.path.join(root, "seed_*")) if os.path.isdir(d))
    seeds = []
    for d in seed_dirs:
        if verbose:
            print(f"[bath_drift] {os.path.basename(d)}", flush=True)
        s = summarize_seed(d, n_boot, verbose)
        if s:
            seeds.append(s)
    return dict(root=os.path.abspath(root), seeds=seeds)


# ---- the headline ----------------------------------------------------------------------------------

def by_iteration(summary):
    """`R` per DCA iteration, averaged across seeds -- the drift curve.

    Each row's interval is the across-seed t-interval AT that iteration. Those are not independent of
    each other (a seed's iterations share a trajectory), so read the curve for shape and take the
    resolved statement from `paired_drift`, which removes exactly that shared variation.
    """
    per = {}
    for s in summary["seeds"]:
        for it in s["iterations"]:
            if it["usable"]:
                per.setdefault(it["iteration"], []).append(it)
    rows = []
    for k in sorted(per):
        g = per[k]
        rows.append(dict(
            iteration=k, n_seeds=len(g),
            R_full=across_seed([x["R_full"] for x in g]),
            R_nonnull=across_seed([x["R_nonnull"] for x in g]),
            w_null=across_seed([x["by_band"].get("forced null", {}).get("w", 0.0) for x in g]),
            bath_step=across_seed([x["bath_step"] for x in g if x["bath_step"] is not None]),
            bath_vs_bare_G0=across_seed([x["bath_vs_bare_G0"] for x in g]),
            sigma_norm=across_seed([x["sigma_norm"] for x in g]),
            mu=across_seed([x["mu"] for x in g]),
            min_sign=float(min(x["sign"]["min_mean_sign"] for x in g)),
            outlier=float(max(x["sign"]["outlier_index"] for x in g))))
    return rows


def paired_drift(summary):
    """THE headline: the within-seed change in `R` from the first to the last DCA iteration.

    Pairing within seed removes the across-seed scatter that dominates a raw iteration-vs-iteration
    comparison, exactly as the milestone-6 paired depth test does. Each seed contributes ONE
    difference; those differences are independent across seeds, so a t-interval on them is honest.
    """
    ds, firsts, lasts = [], [], []
    for s in summary["seeds"]:
        us = [it for it in s["iterations"] if it["usable"]]
        if len(us) > 1:
            ds.append(us[-1]["R_full"] - us[0]["R_full"])
            firsts.append(us[0]["R_full"])
            lasts.append(us[-1]["R_full"])
    st = across_seed(ds)
    base = float(np.mean(firsts)) if firsts else float("nan")
    st["R_first"] = across_seed(firsts)
    st["R_last"] = across_seed(lasts)
    # DID PAIRING ACTUALLY BUY ANYTHING? Measured, not assumed -- and on this axis it does not.
    # The milestone-6 paired DEPTH test works because the deeper run reuses the same chain prefix, so
    # rank i's randomness genuinely is shared. Across DCA ITERATIONS it is not: the walker is
    # re-warmed at a changed bath every iteration, so by the last one nothing survives from the first
    # except the base seed. Measured here at corr = -0.17 and a gain of 0.94x (1.0 = pairing bought
    # nothing), which is why the resolving statement below comes from `cost_weighted`, whose interval
    # is ~15x tighter, and not from this difference.
    if len(ds) > 1:
        a, c = np.array(firsts), np.array(lasts)
        st["corr_first_last"] = float(np.corrcoef(a, c)[0, 1])
        sd_indep = float(np.hypot(a.std(ddof=1), c.std(ddof=1)))
        st["pairing_gain"] = float(sd_indep / np.std(ds, ddof=1)) if np.std(ds, ddof=1) > 0 else float("nan")
    st["rel_to_first"] = float(st["mean"] / base) if st.get("n") and base else float("nan")
    st["n_same_sign"] = int(max((np.array(ds) > 0).sum(), (np.array(ds) < 0).sum())) if ds else 0
    # Resolved = the interval on the paired difference excludes zero. Small = the drift is within the
    # tolerance 3e fixed in advance. They are different questions: a drift can be real and still
    # negligible, which is the outcome that retires the caveat.
    st["resolved"] = bool(st.get("n", 0) > 1 and not (st["lo"] <= 0.0 <= st["hi"]))
    st["within_tolerance"] = bool(abs(st.get("rel_to_first", np.nan)) < DRIFT_TOLERANCE)
    return st


def cost_weighted(summary):
    """What a total-compute claim actually depends on, and the error in quoting the bare-bath `R`.

    A production run performs N solver calls at similar cost each, so the aggregate reduction is the
    cost-weighted mean of `R` over the iterations run. Iterations here are at identical depth and
    rank count by construction (a scalar `measurements` is broadcast to every iteration), so the
    weights are equal and this is the plain mean over iterations -- computed per seed, then averaged
    across seeds so the interval rests on independent replicates.
    """
    means, firsts = [], []
    for s in summary["seeds"]:
        us = [it["R_full"] for it in s["iterations"] if it["usable"]]
        if us:
            means.append(float(np.mean(us)))
            firsts.append(us[0])
    st = across_seed(means)
    st["R_iter0"] = across_seed(firsts)
    if st.get("n") and firsts:
        st["rel_error_of_quoting_iter0"] = float((np.mean(firsts) - st["mean"]) / st["mean"])
    return st


def reference_gap(summary, ref_dir, n_boot=400):
    """Gap A: `R` at the loop's iteration 0 vs the single-iteration BARE-G0 driver, paired by seed.

    Both have Sigma = 0; they differ only in whether the bath went through cluster exclusion (see the
    module docstring). Matched base seeds mean matched walker streams, so this is a paired comparison
    of two baths and not two samples -- which is what makes an 8-seed difference resolvable at all.

    `ref_dir` holds one single-iteration run per seed, named as run_symm_variance.sh writes them.
    """
    refs = {}
    for p in glob.glob(os.path.join(ref_dir, "**", "*.hdf5"), recursive=True):
        if parse_iter(p):
            continue  # a drift file, not a reference run
        try:
            import h5py
            with h5py.File(p, "r") as h:
                refs[int(h["metadata/seed"][()][0])] = p
        except (OSError, KeyError):
            continue
    if not refs:
        return None

    rows, ds = [], []
    for s in summary["seeds"]:
        p = refs.get(s["seed"])
        us = [it for it in s["iterations"] if it["usable"]]
        if not p or not us:
            continue
        r_ref = ms.summarize(p, n_boot)
        if not r_ref["usable"]:
            continue
        loop0 = us[0]
        d = loop0["R_full"] - r_ref["R_full"]
        ds.append(d)
        rows.append(dict(seed=s["seed"], R_bare_G0=r_ref["R_full"], R_loop_iter0=loop0["R_full"],
                         dR=d, bath_rel_diff=loop0["bath_vs_bare_G0"],
                         paired=ms.paired_bootstrap_dR(Run(p), Run(loop0["path"]), n_boot,
                                                       seed=s["seed"])))
    st = across_seed(ds)
    base = float(np.mean([r["R_bare_G0"] for r in rows])) if rows else float("nan")
    st["rel_to_bare"] = float(st["mean"] / base) if st.get("n") and base else float("nan")
    st["resolved"] = bool(st.get("n", 0) > 1 and not (st["lo"] <= 0.0 <= st["hi"]))
    return dict(per_seed=rows, across_seed=st, R_bare_G0=across_seed([r["R_bare_G0"] for r in rows]),
                R_loop_iter0=across_seed([r["R_loop_iter0"] for r in rows]))


def report(summary, ref=None):
    """Human-readable verdict, in the terms 3e's decision rule is written in."""
    L = []
    s0 = summary["seeds"][0]
    L.append(f"BATH DRIFT (ROADMAP 3e) -- {s0['model']}, beta={s0['beta']}, "
             f"{len(summary['seeds'])} seeds x {len(s0['iterations'])} DCA iterations, "
             f"{s0['n_ranks']} ranks x {s0['m']} meas/rank")
    L.append("")
    L.append("R per DCA iteration (across-seed mean +/- SEM); bath_step = how far the bath moved")
    L.append("since the previous iteration, i.e. whether the loop has actually converged:")
    L.append(f"  {'iter':>4}  {'R':>18}  {'w_null':>8}  {'bath_step':>10}  "
             f"{'|Sigma|':>9}  {'mu':>8}  {'<sign>':>7}")
    for r in by_iteration(summary):
        step = r["bath_step"].get("mean")
        L.append(f"  {r['iteration']:>4}  {r['R_full']['mean']:>9.4f} +/- {r['R_full']['sem']:<6.4f}"
                 f"  {r['w_null']['mean']:>8.4f}  "
                 f"{('%.2e' % step) if step is not None and np.isfinite(step) else '-':>10}  "
                 f"{r['sigma_norm']['mean']:>9.4f}  {r['mu']['mean']:>8.4f}  "
                 f"{r['min_sign']:>7.4f}")
    L.append("")

    pd = paired_drift(summary)
    L.append("PAIRED within-seed drift, last iteration vs first (the headline):")
    L.append(f"  R(first) = {pd['R_first']['mean']:.4f} +/- {pd['R_first']['sem']:.4f}   "
             f"R(last) = {pd['R_last']['mean']:.4f} +/- {pd['R_last']['sem']:.4f}")
    L.append(f"  dR = {pd['mean']:+.4f} +/- {pd['sem']:.4f}  "
             f"95% CI [{pd['lo']:+.4f}, {pd['hi']:+.4f}]  "
             f"({pd['n_same_sign']}/{pd['n']} seeds same sign)")
    L.append(f"  relative to R(first): {pd['rel_to_first']:+.2%}   "
             f"resolved={pd['resolved']}  within {DRIFT_TOLERANCE:.0%} tolerance="
             f"{pd['within_tolerance']}")
    if "pairing_gain" in pd:
        L.append(f"  pairing gain {pd['pairing_gain']:.2f}x (1.0 = none), "
                 f"corr(first,last) = {pd['corr_first_last']:+.2f} -- the walker re-warms at a "
                 f"changed bath\n  every iteration, so pairing does NOT work across iterations the "
                 f"way it does across depths.\n  Take the resolved statement from the "
                 f"cost-weighted mean below, whose interval is far tighter.")
    L.append("")

    cw = cost_weighted(summary)
    L.append("Cost-weighted mean R over the iterations a production run performs:")
    L.append(f"  mean over iterations = {cw['mean']:.4f} +/- {cw['sem']:.4f}   "
             f"R(iteration 0) = {cw['R_iter0']['mean']:.4f}")
    L.append(f"  error in quoting iteration 0 instead: "
             f"{cw.get('rel_error_of_quoting_iter0', float('nan')):+.2%}")

    if ref:
        L.append("")
        a = ref["across_seed"]
        L.append("GAP A -- loop iteration 0 vs the single-iteration BARE-G0 driver (both at Sigma=0),")
        L.append("paired by base seed. A fixed coarse-graining offset, not drift:")
        L.append(f"  R(bare G0) = {ref['R_bare_G0']['mean']:.4f} +/- {ref['R_bare_G0']['sem']:.4f}   "
                 f"R(loop iter 0) = {ref['R_loop_iter0']['mean']:.4f} +/- "
                 f"{ref['R_loop_iter0']['sem']:.4f}")
        L.append(f"  dR = {a['mean']:+.4f} +/- {a['sem']:.4f}  95% CI [{a['lo']:+.4f}, {a['hi']:+.4f}]"
                 f"   relative {a['rel_to_bare']:+.2%}  resolved={a['resolved']}")
        if ref["per_seed"]:
            L.append(f"  bath differs by {ref['per_seed'][0]['bath_rel_diff']:.2%} "
                     f"(max|dG0|/max|G0|) with identical Sigma = 0")
    return "\n".join(L)


def main(argv):
    if "--summary" in argv:
        with open(argv[argv.index("--summary") + 1]) as f:
            blob = json.load(f)
        print(report(blob["summary"], blob.get("reference_gap")))
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 1
    root = argv[1]
    n_boot = int(argv[argv.index("--boot") + 1]) if "--boot" in argv else 400
    summary = build(root, n_boot)
    if not summary["seeds"]:
        print(f"[bath_drift] no seed_* directories with iteration files under {root}")
        return 1
    ref = None
    if "--ref" in argv:
        ref = reference_gap(summary, argv[argv.index("--ref") + 1], n_boot)
    txt = report(summary, ref)
    print("\n" + txt)
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(summary=summary, by_iteration=by_iteration(summary),
                           paired_drift=paired_drift(summary), cost_weighted=cost_weighted(summary),
                           reference_gap=ref, report=txt), f, indent=1, default=float)
        print(f"\n[bath_drift] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
