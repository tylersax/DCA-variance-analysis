"""Beta ladder (ROADMAP task 2): is square intrinsically rho-limited, or only at beta=1?

THE QUESTION. The per-orbit reduction obeys `r = m/[1+(m-1)rho]`, so an orbit of size `m` delivers
its full `m`-fold reduction only when its mates' noise is uncorrelated. At beta=1 the square model
measured `rho ~ 0.94` and `R = 1.0425` -- almost no reduction, and the natural reading was that the
single-band square model simply has symmetric noise. But beta=1 is also the regime where the noise is
dominated by symmetric scalar channels, so that reading confounds "this model" with "this
temperature". This module separates them by sweeping beta at everything else fixed.

THE PREDICTION, AND THE COMPETING EFFECT. Growing correlation length should give the noise
symmetry-BREAKING structure, pushing `rho` down and `R` up. Working against it: `G = <sign*M>/<sign>`,
and a fluctuation in that scalar denominator gives `dG(k) = -G(k) * ds/s` -- a weight `-G(k)` that is
itself symmetric, so sign-problem noise is a fully symmetric channel with mate-correlation exactly 1,
pushing `rho` UP and `R` toward 1. Any model with a sign problem therefore mixes the two effects, and
a bare `R(beta)` curve cannot say which is acting.

WHY THIS LADDER USES SQUARE. Nearest-neighbour hopping on a bipartite lattice at half filling (mu=0)
is provably sign-free in CT-AUX at any beta, and the runs confirm it -- `<s>` is exactly 1 at every
beta on the ladder, on every rank. So the sign channel is switched OFF by construction and the ladder
measures the correlation-length effect alone. `sign_channel_check` re-verifies this per run rather
than trusting the argument, because the entire interpretation rests on it.

WHAT IT REPORTS, per beta: the headline `R` across seeds (from `seed_ensemble`), the mate-correlation
`rho`, the per-orbit `r` against its `m/[1+(m-1)rho]` prediction, and the sign diagnostics. Plus a
monotonicity test on `rho(beta)` and `R(beta)` that uses the across-seed scatter, so "rho falls" is a
claim with an interval on it rather than an eyeball reading of four points.

CLI:  python beta_ladder.py <ladder_root> [--out summary.json] [--boot 400]
      python beta_ladder.py --summary existing_summary.json
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m_scaling as ms  # noqa: E402
import seed_ensemble as se  # noqa: E402


def build(root, n_boot=400, verbose=True):
    """Summarize every run under `<root>/b<beta>/`, one seed ensemble per temperature.

    Beta comes from each file's own HDF5 metadata, never from the directory name -- the layout is a
    convenience for keeping filenames parseable, not a source of truth. A run whose recorded beta
    disagrees with its directory is a staging bug (a stale `.stage` file moved into the wrong place),
    so it is raised rather than quietly grouped.
    """
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "b*"))):
        if not os.path.isdir(d):
            continue
        want = float(os.path.basename(d)[1:])
        for p in sorted(x for x in glob.glob(os.path.join(d, "*.hdf5")) if ms.parse_name(x)):
            if verbose:
                print(f"[beta_ladder] {os.path.basename(d)}/{os.path.basename(p)}", flush=True)
            s = ms.summarize(p, n_boot)
            if s.get("beta") is None:
                raise ValueError(f"{p}: no beta in metadata -- rebuilt driver needed for this ladder")
            if abs(s["beta"] - want) > 1e-9:
                raise ValueError(f"{p}: metadata beta={s['beta']} but lives in {d}")
            runs.append(s)
    return dict(runs=runs, dirpath=os.path.abspath(root))


def sign_channel_check(g):
    """Is the sign channel actually off across this beta's runs?

    The ladder's interpretation -- that any fall in `rho` is the correlation-length effect -- holds
    only while `<sign>` is pinned at 1. A model that develops even a mild sign problem starts feeding
    a perfectly-symmetric noise channel into `rho`, which would push `R` down and could be mistaken
    for the effect saturating. So this is a precondition, checked per beta, not a footnote.
    """
    lo = min(r["sign"]["min_mean_sign"] for r in g)
    hi = max(r["sign"]["max_mean_sign"] for r in g)
    return dict(min_mean_sign=lo, max_mean_sign=hi, sign_free=bool(abs(lo - 1) < 1e-12 and
                                                                   abs(hi - 1) < 1e-12),
                worst_outlier_index=max(r["sign"]["outlier_index"] for r in g))


def omega_flatness(paths):
    """Pooled `r` against Matsubara index, per run: mean, slope per step, and spread.

    THIS IS THE CONTROL FOR A CONFOUND THE LADDER WOULD OTHERWISE CARRY. The input template fixes
    `sp-fermionic-frequencies` at 128 regardless of beta, and Matsubara frequencies are spaced
    pi/beta -- so the window in ABSOLUTE energy units shrinks by the same factor beta grows. The
    rungs are therefore not quite "everything else fixed": each one sums `R` over a different slice
    of frequency space. That only matters if `r` depends on omega.

    It should not. The point group acts on band and momentum indices and does not touch Matsubara
    frequency, so `r` must be omega-independent -- rung 1c of the validation ladder. Measuring the
    slope at every beta turns that from an assumption inherited from beta=1 into a per-rung check: if
    `r` is flat at each beta, the changing frequency window cannot be driving the trend in `R`.

    Restricted to non-singleton, non-null orbits, matching how rung 1c pools in notebook 01.
    """
    out = []
    for p in paths:
        run = ms.Run(p)
        vr = run.variance_ratios()
        mask = np.zeros(run.E, bool)
        for o in run.orbits():
            if not o["forced_null"] and len(o["members"]) > 1:
                mask[o["members"]] = True
        num = vr["var_raw"].real[:, :, mask].sum((1, 2)) + vr["var_raw"].imag[:, :, mask].sum((1, 2))
        den = vr["var_sym"].real[:, :, mask].sum((1, 2)) + vr["var_sym"].imag[:, :, mask].sum((1, 2))
        rw = num / den
        slope = float(np.polyfit(np.arange(run.nw), rw, 1)[0])
        out.append(dict(mean=float(rw.mean()), slope_per_step=slope,
                        # Total drift across the whole window, as a fraction of the mean: the
                        # scale-free way to ask "does the omega window matter for R?".
                        rel_drift=float(abs(slope) * (run.nw - 1) / rw.mean()),
                        spread=float(rw.std())))
    return out


def mate_rho(run_summary):
    """Aggregate MATE correlation for one run: the rho that actually enters the law.

    Two different quantities are both called "rho" in this project and they are not interchangeable.
    `mechanism.rho_generic` is the pairwise noise correlation over ALL k-pairs -- a description of the
    noise field at large. The one in `r = m/[1+(m-1)rho]`, and the one quoted as square's "rho ~ 0.94",
    is the correlation between ORBIT MATES specifically. They differ by a lot (0.95 vs 0.39 for square
    at beta=1), so reporting the generic one under the bare name `rho` next to `R` would invite
    exactly the wrong reading.

    Averaged over non-singleton, non-null orbits weighted by orbit size, so an orbit contributes in
    proportion to how many entries it actually governs.
    """
    num = den = 0.0
    for o in run_summary["orbits"]:
        if o["rho"] is None or o["m"] < 2:
            continue
        num += o["m"] * o["rho"]
        den += o["m"]
    return num / den if den else float("nan")


def orbit_rho(g):
    """Per-orbit `m`, across-seed mean `rho`, measured `r` and predicted `m/[1+(m-1)rho]`.

    Orbits are matched across seeds by their member tuple, not by position: the orbit table is
    deterministic given the point group, but keying on the members makes that an assumption the code
    does not have to make.
    """
    acc = {}
    for r in g:
        for o in r["orbits"]:
            if o["rho"] is None:
                continue
            acc.setdefault(tuple(o["members"]), dict(m=o["m"], rho=[], r=[], r_pred=[]))
            acc[tuple(o["members"])]["rho"].append(o["rho"])
            acc[tuple(o["members"])]["r"].append(o["r"])
            acc[tuple(o["members"])]["r_pred"].append(o["r_pred"])
    rows = []
    for members, v in acc.items():
        rows.append(dict(m=v["m"], n_orbit_seeds=len(v["rho"]), members=list(members),
                         rho=se.across_seed(v["rho"]), r=se.across_seed(v["r"]),
                         r_pred=se.across_seed(v["r_pred"]),
                         # Measured vs predicted is the free contamination detector (Conventions):
                         # they agree to ~3 decimals at adequate depth, and diverge on a run whose
                         # variance is being set by a single outlying rank.
                         max_abs_gap=float(np.max(np.abs(np.array(v["r"], float) -
                                                         np.array(v["r_pred"], float))))))
    return sorted(rows, key=lambda x: (-x["m"], x["members"]))


def rungs(summary, n_boot=4000):
    """One row per beta: the headline `R`, `rho`, the sign check, and the per-orbit breakdown."""
    out = []
    for (model, ranks, m, beta), g in se.groups(summary).items():
        g = [r for r in g if r["usable"]]
        if not g:
            continue
        # Every band block, not just the first. Single-band square has one ("band-diagonal"), but a
        # multi-orbital model has "intraband" and "interband" -- and for those the interband block is
        # where the distinctive physics sits (TAKEAWAYS 4c), so collapsing to the first block would
        # silently discard the more interesting half of the mechanism.
        blocks = list(g[0]["mechanism"])
        # The omega control needs the raw HDF5s, which live in scratch and are not committed. A
        # reloaded summary is the normal case once that scratch is gone, so degrade to "not
        # available" rather than dying -- same policy as seed_ensemble's pooled/oracle checks.
        paths = [os.path.join(summary["dirpath"], f"b{beta:g}", r["path"]) for r in g]
        flat = omega_flatness(paths) if all(os.path.exists(p) for p in paths) else []
        out.append(dict(
            model=model, beta=beta, n_ranks=ranks, m=m, n_seeds=len(g),
            R_full=se.across_seed([r["R_full"] for r in g]),
            R_full_boot=se.seed_bootstrap([r["R_full"] for r in g], n_boot),
            R_nonnull=se.across_seed([r["R_nonnull"] for r in g]),
            w_null=se.across_seed([se.w_null(r) for r in g]),
            # `rho` (mate correlation) stays top-level and band-agnostic: it is defined over orbits,
            # which span the whole (b0,b1,k) index space, and it is the one in r = m/[1+(m-1)rho].
            rho=se.across_seed([mate_rho(r) for r in g]),
            mechanism={b: dict(
                rho_generic=se.across_seed([r["mechanism"][b]["rho_generic"] for r in g]),
                local_share=se.across_seed([r["mechanism"][b]["local_share"] for r in g]),
                within_shell=se.across_seed([r["mechanism"][b]["within_shell"] for r in g]),
                across_shell=se.across_seed([r["mechanism"][b]["across_shell"] for r in g]))
                for b in blocks},
            sign=sign_channel_check(g),
            calibration=se.calibration(g),
            omega=dict(n=len(flat),
                       rel_drift=se.across_seed([f["rel_drift"] for f in flat]),
                       slope=se.across_seed([f["slope_per_step"] for f in flat])) if flat else None,
            orbits=orbit_rho(g)))
    return sorted(out, key=lambda r: r["beta"])


def trend(rows, key, n_boot=20000, seed=0):
    """Does `key` move monotonically with beta, beyond the across-seed scatter?

    Four rungs is far too few for a regression to mean much, so this reports two things that four
    points can support: the endpoint change with a bootstrap interval built by resampling SEEDS
    within each rung (the level of independent replication the design actually has), and whether the
    rung-to-rung steps all share one sign. A monotone run of 3 steps under an exchangeable null has
    probability 2*(1/2)^3 = 0.25, so monotonicity ALONE is weak evidence -- it is worth reporting
    only next to an endpoint interval that excludes zero, which is why both appear together.
    """
    betas = [r["beta"] for r in rows]
    # One rung per beta, or the "trend" is partly a trend in DEPTH. A floor-finding directory has
    # several depths per beta and would otherwise produce a confident-looking interval on a mixture
    # of two axes -- silently the wrong number, which is the failure this codebase guards hardest
    # against. Point this at an ensemble directory (one depth per beta) instead.
    if len(set(betas)) != len(betas):
        dupes = sorted({b for b in betas if betas.count(b) > 1})
        raise ValueError(f"trend() needs one rung per beta; betas {dupes} appear at several depths. "
                         f"This looks like a floor-finding directory -- use m_scaling.py on it.")
    mean = np.array([r[key]["mean"] for r in rows], float)
    sd = np.array([r[key]["sd"] for r in rows], float)
    n = np.array([r[key]["n"] for r in rows], int)
    if len(rows) < 2:
        return dict(key=key, n_rungs=len(rows))
    steps = np.diff(mean)
    rng = np.random.default_rng(seed)
    # Resample each rung's mean from its own across-seed sampling distribution. Rungs are independent
    # runs, so their errors do not share randomness and can be drawn separately.
    sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), 0.0)
    draws = mean + rng.normal(size=(n_boot, len(rows))) * sem
    delta = draws[:, -1] - draws[:, 0]
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return dict(key=key, n_rungs=len(rows), betas=betas, means=mean.tolist(),
                delta=float(mean[-1] - mean[0]), lo=float(lo), hi=float(hi),
                excludes_zero=bool(lo > 0 or hi < 0),
                monotone=bool(np.all(steps > 0) or np.all(steps < 0)),
                direction="up" if mean[-1] > mean[0] else "down",
                steps=steps.tolist())


# ---- text report ----------------------------------------------------------------------------------

def rung_table(rows):
    lines = [f"{'beta':>5} {'seeds':>5} {'m/rank':>7} {'R (mean +/- sem)':>22} {'95% t-CI':>18} "
             f"{'rho (mates)':>16} {'<s>':>9} {'outl':>6}"]
    for r in rows:
        f, p = r["R_full"], r["rho"]
        lines.append(f"{r['beta']:>5g} {r['n_seeds']:>5} {r['m']:>7} "
                     f"{f['mean']:>14.4f} +/- {f['sem']:<5.4f} [{f['lo']:.4f},{f['hi']:.4f}] "
                     f"{p['mean']:>9.4f} +/- {p['sem']:<5.4f} "
                     f"{r['sign']['min_mean_sign']:>9.5f} {r['sign']['worst_outlier_index']:>6.2f}")
    return "\n".join(lines)


def orbit_table(rows):
    lines = [f"{'beta':>5} {'m':>3} {'rho (mean +/- sem)':>21} {'r measured':>18} "
             f"{'r predicted':>18} {'max|gap|':>9}"]
    for r in rows:
        for o in r["orbits"]:
            rho, rr, rp = o["rho"], o["r"], o["r_pred"]
            lines.append(f"{r['beta']:>5g} {o['m']:>3} {rho['mean']:>13.4f} +/- {rho['sem']:<5.4f} "
                         f"{rr['mean']:>10.4f} +/- {rr['sem']:<5.4f} "
                         f"{rp['mean']:>10.4f} +/- {rp['sem']:<5.4f} {o['max_abs_gap']:>9.2e}")
    return "\n".join(lines)


def trend_table(trends):
    lines = [f"{'quantity':<14} {'beta range':>14} {'change':>12} {'95% CI':>22} "
             f"{'monotone':>9} {'verdict':>9}"]
    for t in trends:
        if t.get("n_rungs", 0) < 2:
            continue
        rng = f"{t['betas'][0]:g}->{t['betas'][-1]:g}"
        verdict = "resolved" if t["excludes_zero"] else "unresolved"
        lines.append(f"{t['key']:<14} {rng:>14} {t['delta']:>+12.4f} "
                     f"[{t['lo']:+.4f},{t['hi']:+.4f}] {str(t['monotone']):>9} {verdict:>9}")
    return "\n".join(lines)


def sign_table(rows):
    lines = [f"{'beta':>5} {'min <s>':>10} {'max <s>':>10} {'sign-free':>10} {'worst outlier':>14}"]
    for r in rows:
        s = r["sign"]
        lines.append(f"{r['beta']:>5g} {s['min_mean_sign']:>10.6f} {s['max_mean_sign']:>10.6f} "
                     f"{str(s['sign_free']):>10} {s['worst_outlier_index']:>14.3f}")
    return "\n".join(lines)


def mechanism_table(rows):
    """Noise structure per band block. For a multi-orbital model the intraband/interband split is
    the point: interband entries have no local scalar channel, so they should carry a lower
    `rho_generic` and a smaller `sigma^2(r=0)` share than intraband at the same beta."""
    lines = [f"{'beta':>5} {'block':<14} {'rho_generic':>18} {'local share':>18} "
             f"{'within shell':>16} {'across shell':>16}"]
    for r in rows:
        for b, v in r["mechanism"].items():
            lines.append(f"{r['beta']:>5g} {b:<14} "
                         f"{v['rho_generic']['mean']:>11.4f} +/- {v['rho_generic']['sem']:<5.4f} "
                         f"{v['local_share']['mean']:>11.4f} +/- {v['local_share']['sem']:<5.4f} "
                         f"{v['within_shell']['mean']:>16.4f} {v['across_shell']['mean']:>16.4f}")
    return "\n".join(lines)


def support_table(rows):
    """Full vs non-null `R`, and the forced-null share that separates them.

    Square has no forced nulls so this is degenerate there, but a multi-orbital model does, and
    `w_null` varies with the model -- Conventions require reporting it wherever noise STRUCTURE is
    the point, so that it cannot silently absorb part of a trend in `R`."""
    lines = [f"{'beta':>5} {'R full':>20} {'R non-null':>20} {'w_null':>18}"]
    for r in rows:
        f, n, w = r["R_full"], r["R_nonnull"], r["w_null"]
        lines.append(f"{r['beta']:>5g} {f['mean']:>12.4f} +/- {f['sem']:<6.4f} "
                     f"{n['mean']:>12.4f} +/- {n['sem']:<6.4f} "
                     f"{w['mean']:>10.4f} +/- {w['sem']:<6.4f}")
    return "\n".join(lines)


def omega_table(rows):
    lines = [f"{'beta':>5} {'runs':>5} {'slope per step':>22} {'drift across window':>24} {'verdict':>9}"]
    for r in rows:
        o = r["omega"]
        if not o:
            lines.append(f"{r['beta']:>5g} {'-':>5} {'raw HDF5s not available':>22}")
            continue
        s, d = o["slope"], o["rel_drift"]
        # 1% total drift across the whole window is far below the beta-to-beta changes in R, so the
        # shrinking absolute frequency window cannot account for the trend.
        verdict = "flat" if d["mean"] < 0.01 else "SLOPED"
        lines.append(f"{r['beta']:>5g} {o['n']:>5} {s['mean']:>14.2e} +/- {s['sem']:<6.1e} "
                     f"{d['mean']:>17.4%} +/- {d['sem']:<5.4%} {verdict:>9}")
    return "\n".join(lines)


def report(summary, n_boot=4000):
    rows = rungs(summary, n_boot)
    try:
        trends = [trend(rows, k) for k in ("R_full", "rho")]
    except ValueError as e:  # floor-finding layout: the per-beta tables below are still meaningful
        print(f"\n[beta_ladder] skipping trend: {e}")
        trends = []
    spacing = se.check_seed_spacing(summary["runs"])
    print("\n=== R and rho vs beta ===")
    print(rung_table(rows))
    print("\n=== trend across the ladder ===")
    print(trend_table(trends))
    # For a sign-free model this is a precondition (the ladder isolates correlation length only while
    # <s> = 1). For a model WITH a sign problem it is the second variable, not a nuisance: sign noise
    # is a fully symmetric channel, so <s> falling and rho rising together is the competing effect
    # showing up rather than a fault.
    print("\n=== sign channel ===")
    print(sign_table(rows))
    print("\n=== full vs non-null support ===")
    print(support_table(rows))
    print("\n=== noise structure by band block ===")
    print(mechanism_table(rows))
    print("\n=== r vs omega (fixed 128 frequencies means the absolute window shrinks with beta) ===")
    print(omega_table(rows))
    print("\n=== per-orbit rho, measured r vs m/[1+(m-1)rho] ===")
    print(orbit_table(rows))
    print("\n=== seed independence ===")
    print("OK: no overlapping walker-stream ranges" if not spacing else f"VIOLATION: {spacing}")
    return dict(rungs=rows, trends=trends, seed_spacing_violations=spacing)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=None)
    ap.add_argument("--summary", default=None, help="reload a previously written summary JSON")
    ap.add_argument("--out", default=None, help="write the summary as JSON here")
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    if args.summary:
        with open(args.summary) as fh:
            summary = json.load(fh)
    elif args.root:
        summary = build(args.root, args.boot)
    else:
        ap.error("need <root> or --summary")

    summary["report"] = report(summary)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=1)
        print(f"\n[beta_ladder] wrote {args.out}")
