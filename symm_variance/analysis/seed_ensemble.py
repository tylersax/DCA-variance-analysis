"""Seed ensemble (ROADMAP task 1): pin `R` with an interval that rests on nothing.

WHY THIS EXISTS, AND WHY IT IS NOT JUST "RUN MORE RANKS". `R` is scale-free above the model's depth
floor, so precision comes from independent samples, not from compute per sample. The obvious source
of independent samples is ranks -- and `m_scaling.bootstrap_R` resamples them. But the M-scaling
control found that interval running optimistic on FeAs -- measured here at 1.44x `[0.96, 1.79]`
across 32 seeds -- i.e. the rank bootstrap understates the uncertainty where a sign problem makes the
per-rank `G` heavy-tailed. Its assumption
(ranks are iid draws whose resampling reproduces the sampling distribution of a RATIO of variance
sums) is the part that fails; no amount of extra ranks repairs it.

An independent base seed sidesteps the whole question. It reruns the entire measurement -- fresh
walker streams on every rank -- so each seed's `R` is one draw from the estimator's actual sampling
distribution, and the sample SD across seeds estimates that distribution's width using no structural
assumption at all. That is what this module aggregates.

WHAT IT REPORTS:
  * the headline: mean `R` over seeds with a Student-t interval on the mean;
  * three robustness cross-checks that would disagree if the mean were being set by tail draws --
    the median, a percentile bootstrap over seeds, and the POOLED estimator (all seeds' ranks in one
    variance sum, which is a ratio-of-means where the headline is a mean-of-ratios);
  * a calibration of the rank bootstrap against the across-seed SD, now at a seed count where that
    ratio is actually resolved (the original 1.4-1.9x figure came from three seeds);
  * a contamination gate per run, since a sign outlier inflates `R` silently -- see `contamination`.

SEED INDEPENDENCE IS A PRECONDITION, NOT A GIVEN. Walker streams are hash(global_id + base_seed) with
global_id = local_id*n_ranks + proc_id, so base seeds spaced closer than n_ranks*n_walkers replay each
other's chains. `run_seed_ensemble.sh` enforces the spacing; `check_seed_spacing` re-checks it here,
because every interval below is void if it fails.

CLI:  python seed_ensemble.py <ensemble_dir> [--out summary.json] [--boot 400]
      python seed_ensemble.py --summary existing_summary.json
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symm_variance_lib import Run  # noqa: E402
import m_scaling as ms  # noqa: E402
import reduction_map as rm  # noqa: E402

# Student-t 97.5th percentile by degrees of freedom, for the interval on the mean. Tabulated rather
# than imported: the analysis venv has no scipy, and this is the only special function needed.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
         18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
         26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 31: 2.040, 35: 2.030, 40: 2.021,
         50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984}


def t975(df):
    if df < 1:
        return float("nan")
    if df in _T975:
        return _T975[df]
    if df > 100:
        return 1.960
    return _T975[min(k for k in sorted(_T975) if k > df)]  # round UP: never anti-conservative


# ---- across-seed statistics -----------------------------------------------------------------------

def across_seed(values):
    """Mean, its Student-t interval, and the shape diagnostics that say whether the mean is safe.

    `skew` and `max_share` are the guards. A heavy right tail -- one seed contributing a large part of
    the spread -- is the signature of sign contamination surviving the depth floor, and it would make
    the t interval (which assumes an approximately normal mean) optimistic. Both are reported so that
    assumption is checked rather than asserted.
    """
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    n = x.size
    if n == 0:
        return dict(n=0)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else float("nan")
    half = t975(n - 1) * sem if n > 1 else float("nan")
    dev = np.abs(x - mean)
    return dict(n=n, mean=mean, sd=sd, sem=float(sem), lo=float(mean - half), hi=float(mean + half),
                rel_sem=float(sem / mean) if mean else float("nan"),
                median=float(np.median(x)), min=float(x.min()), max=float(x.max()),
                skew=float(np.mean((x - mean) ** 3) / sd ** 3) if sd > 0 else 0.0,
                max_share=float(dev.max() / dev.sum()) if dev.sum() > 0 else 0.0)


def seed_bootstrap(values, n_boot=4000, seed=0):
    """Percentile bootstrap over SEEDS -- an interval on the mean that assumes no distribution shape.

    Cross-check on the t interval, not a replacement: it is the same data resampled, so agreement only
    rules out the normality approximation as the source of a wrong width.
    """
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if x.size < 2:
        return dict(lo=float("nan"), hi=float("nan"), se=float("nan"))
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(lo=float(lo), hi=float(hi), se=float(draws.std(ddof=1)))


def pooled_R(paths, nonnull=False):
    """`R` from every seed's ranks in ONE variance sum: a ratio of means, not a mean of ratios.

    The headline averages per-seed `R`, so it is a mean of ratios and carries the usual small upward
    ratio bias. Pooling forms the two variance sums over all seeds' ranks first and divides once, so
    the bias enters differently. If the two agree the bias is negligible at this seed count, which is
    the only thing this check is for.

    Pooling is legitimate here because the seeds are draws from the same distribution: they share the
    model, depth and rank count, and differ only in RNG stream, so their per-rank samples are
    exchangeable and a pooled variance is estimating the same Sigma.
    """
    As, Bs = [], []
    for p in paths:
        run = Run(p)
        subset = None
        if nonnull:
            _, null = rm.orbit_info(run)
            subset = [x for x in range(run.E) if not null[x]]
        a, b = ms._arms(run, subset)
        As.append(a)
        Bs.append(b)
    A, B = np.concatenate(As), np.concatenate(Bs)
    return float(A.var(0, ddof=1).sum() / B.var(0, ddof=1).sum()), A.shape[0]


def check_seed_spacing(runs, n_walkers=64):
    """Do any two base seeds occupy overlapping walker-stream key ranges?

    A run with base seed S uses keys [S, S + n_ranks*n_walkers). Overlap means shared chains, which
    would make the across-seed SD an underestimate of the true spread -- the one failure mode that
    makes every interval here too narrow rather than too wide. `n_walkers` is an upper bound; the
    inputs use 2.
    """
    bad = []
    for key in {(r["model"], r["n_ranks"]) for r in runs}:
        seeds = sorted({r["seed"] for r in runs if (r["model"], r["n_ranks"]) == key})
        width = key[1] * n_walkers
        for a, b in zip(seeds, seeds[1:]):
            if b - a < width:
                bad.append(dict(model=key[0], seeds=[a, b], gap=b - a, need=width))
    return bad


# ---- contamination gate ---------------------------------------------------------------------------

def contamination(run_summary):
    """Per-run evidence that the sample is clean, from three independent angles.

    `r_dev` is the free detector: `r = m/[1+(m-1)rho]` reproduces the measured per-orbit `r` to three
    decimals when every rank is well-conditioned, and visibly fails when one rank's near-zero sign
    denominator makes the pairwise correlations heterogeneous -- the prediction assumes ONE rho per
    orbit. It needs no threshold tuning because the clean value is ~0.

    `min_sign` is the worst denominator anywhere in the sample, and `outlier` the largest per-rank |G|
    over the median. These catch the same failure earlier in the causal chain, before it has distorted
    the correlation structure.
    """
    devs = [abs(o["r"] / o["r_pred"] - 1.0) for o in run_summary.get("orbits", [])
            if o.get("r") and o.get("r_pred")]
    sg = run_summary["sign"]
    return dict(seed=run_summary["seed"], m=run_summary["m"], R=run_summary["R_full"],
                r_dev=float(max(devs)) if devs else float("nan"),
                min_sign=sg["min_mean_sign"], outlier=sg["outlier_index"],
                nonfinite=sg["n_ranks_nonfinite"])


# ---- assembly -------------------------------------------------------------------------------------

def build(dirpath, n_boot=400, verbose=True):
    """Summarize every ensemble run in a directory. Same per-run summary the M-ladder uses."""
    paths = sorted(p for p in glob.glob(os.path.join(dirpath, "*.hdf5")) if ms.parse_name(p))
    runs = []
    for p in paths:
        if verbose:
            print(f"[seed_ensemble] {os.path.basename(p)}", flush=True)
        runs.append(ms.summarize(p, n_boot))
    return dict(runs=runs, dirpath=os.path.abspath(dirpath))


def groups(summary):
    """Ensemble runs keyed by design point (model, ranks, depth, beta).

    `beta` is in the key because averaging `R` across temperatures would silently produce a number
    that is not any model's `R` -- and the beta ladder (ROADMAP task 2) exists precisely because `R`
    moves with beta. Runs written before beta was recorded carry None, so a pre-ladder summary still
    forms exactly one group and reloads unchanged.
    """
    g = {}
    for r in summary["runs"]:
        g.setdefault((r["model"], r["n_ranks"], r["m"], r.get("beta")), []).append(r)
    # None sorts against float in py3, so key on a sentinel rather than the raw beta.
    return dict(sorted(g.items(), key=lambda kv: (kv[0][:3], kv[0][3] if kv[0][3] is not None else -1)))


def w_null(run_summary):
    """Forced-null share of RAW variance -- the only thing separating full from non-null `R`."""
    return run_summary["by_band"].get("forced null", {}).get("w", 0.0)


def calibration(g):
    """Across-seed SD vs the mean rank-bootstrap SE, at the same design point.

    Ratio > 1 means the rank bootstrap is optimistic. The M-scaling control saw 1.4-1.9 on FeAs from
    three seeds -- an SD from n=3 is barely an estimate, so that was a signal, not a factor. At 32
    seeds it resolves to **1.44 `[0.96, 1.79]`** for FeAs and **1.08 `[0.81, 1.27]`** for square: the
    direction survives, the magnitude does not -- FeAs's interval still includes 1. Report the
    interval, not the ratio alone.
    """
    R = np.array([r["R_full"] for r in g], float)
    se = float(np.mean([r["R_full_se"] for r in g]))
    if R.size < 2 or se <= 0:
        return dict(ratio=float("nan"))
    rng = np.random.default_rng(0)
    draws = R[rng.integers(0, R.size, size=(4000, R.size))].std(ddof=1, axis=1) / se
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(boot_se=se, seed_sd=float(R.std(ddof=1)), ratio=float(R.std(ddof=1) / se),
                ratio_lo=float(lo), ratio_hi=float(hi))


def headline(summary, n_boot=4000):
    """The quotable number per design point, with every cross-check beside it."""
    out = []
    for (model, ranks, m, beta), g in groups(summary).items():
        g = [r for r in g if r["usable"]]
        if not g:
            continue
        paths = [os.path.join(summary["dirpath"], r["path"]) for r in g]
        full = across_seed([r["R_full"] for r in g])
        nn = across_seed([r["R_nonnull"] for r in g])
        wn = across_seed([w_null(r) for r in g])
        # Everything above comes from the per-run summaries; only the pooled cross-check needs the
        # raw HDF5s, which live in scratch and are not committed. Losing it degrades the report by
        # one column rather than making a reloaded summary unusable.
        have_raw = all(os.path.exists(p) for p in paths)
        pooled, n_pooled = pooled_R(paths) if have_raw else (float("nan"), 0)
        out.append(dict(
            model=model, n_ranks=ranks, m=m, n_seeds=len(g),
            R_full=full, R_nonnull=nn, w_null=wn,
            R_full_boot=seed_bootstrap([r["R_full"] for r in g], n_boot),
            R_pooled=pooled, n_pooled_ranks=n_pooled,
            # R_full = R_nonnull / (1 - w_null) is an identity, not a fit. Checking it on the
            # ensemble mean catches a support/mask mismatch between the two arms.
            identity_gap=float(full["mean"] - nn["mean"] / (1.0 - wn["mean"])),
            efficiency=across_seed([r["efficiency"] for r in g]),
            calibration=calibration(g)))
    return out


# ---- text report ----------------------------------------------------------------------------------

def headline_table(rows):
    lines = [f"{'model':<10} {'ranks':>5} {'m/rank':>7} {'seeds':>5} "
             f"{'R (mean +/- sem)':>22} {'95% t-CI':>18} {'rel sem':>8}"]
    for h in rows:
        f = h["R_full"]
        lines.append(f"{h['model']:<10} {h['n_ranks']:>5} {h['m']:>7} {h['n_seeds']:>5} "
                     f"{f['mean']:>14.4f} +/- {f['sem']:<5.4f} "
                     f"[{f['lo']:.4f},{f['hi']:.4f}] {f['rel_sem']:>7.2%}")
    return "\n".join(lines)


def robustness_table(rows):
    lines = [f"{'model':<10} {'m/rank':>7} {'mean':>8} {'median':>8} {'pooled':>8} "
             f"{'seed-boot 95%':>19} {'skew':>7} {'max share':>10} {'identity':>11}"]
    for h in rows:
        f, b = h["R_full"], h["R_full_boot"]
        pooled = f"{h['R_pooled']:.4f}" if np.isfinite(h["R_pooled"]) else "n/a"
        lines.append(f"{h['model']:<10} {h['m']:>7} {f['mean']:>8.4f} {f['median']:>8.4f} "
                     f"{pooled:>8} [{b['lo']:.4f},{b['hi']:.4f}] "
                     f"{f['skew']:>+7.2f} {f['max_share']:>10.3f} {h['identity_gap']:>+11.2e}")
    return "\n".join(lines)


def secondary_table(rows):
    """Non-null R, w_null and efficiency. Internal diagnostics -- see ROADMAP Conventions."""
    lines = [f"{'model':<10} {'m/rank':>7} {'R_nonnull':>19} {'w_null':>17} {'efficiency':>18}"]
    for h in rows:
        nn, wn, ef = h["R_nonnull"], h["w_null"], h["efficiency"]
        lines.append(f"{h['model']:<10} {h['m']:>7} {nn['mean']:>11.4f} +/-{nn['sem']:<6.4f} "
                     f"{wn['mean']:>9.4f} +/-{wn['sem']:<6.4f} "
                     f"{ef['mean']:>10.2%} +/-{ef['sem']:<6.2%}")
    return "\n".join(lines)


def calibration_table(rows):
    lines = [f"{'model':<10} {'m/rank':>7} {'seeds':>5} {'boot SE':>9} {'seed SD':>9} "
             f"{'ratio':>7} {'95% CI on ratio':>18}"]
    for h in rows:
        c = h["calibration"]
        ci = f"[{c['ratio_lo']:.2f},{c['ratio_hi']:.2f}]"
        lines.append(f"{h['model']:<10} {h['m']:>7} {h['n_seeds']:>5} {c['boot_se']:>9.4f} "
                     f"{c['seed_sd']:>9.4f} {c['ratio']:>7.2f} {ci:>18}")
    return "\n".join(lines)


def contamination_table(summary, worst=8):
    """The `worst` runs by r-deviation, plus the ensemble-wide extremes."""
    rows = [contamination(r) for r in summary["runs"] if r["usable"]]
    unusable = [r for r in summary["runs"] if not r["usable"]]
    if not rows:
        return "[gate] no usable runs"
    rows.sort(key=lambda d: -(d["r_dev"] if np.isfinite(d["r_dev"]) else -1))
    lines = [f"{'seed':>10} {'m/rank':>7} {'R':>8} {'max |r/r_pred-1|':>17} {'min <s>':>9} "
             f"{'outlier':>8} {'nonfinite':>10}"]
    for d in rows[:worst]:
        lines.append(f"{d['seed']:>10} {d['m']:>7} {d['R']:>8.4f} {d['r_dev']:>17.4f} "
                     f"{d['min_sign']:>9.4f} {d['outlier']:>8.2f} {d['nonfinite']:>10}")
    if len(rows) > worst:
        lines.append(f"... {len(rows) - worst} cleaner runs omitted")
    lines.append(f"[gate] worst r-deviation {max(d['r_dev'] for d in rows):.4f}, "
                 f"worst min<s> {min(d['min_sign'] for d in rows):.4f}, "
                 f"worst outlier index {max(d['outlier'] for d in rows):.2f}, "
                 f"unusable runs {len(unusable)}")
    return "\n".join(lines)


def _ranks(x):
    """Ranks with ties averaged -- the only ingredient Spearman needs, and numpy has no rankdata."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    r[order] = np.arange(1, x.size + 1, dtype=float)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            r[tie] = r[tie].mean()
    return r


def contamination_coupling(summary, n_perm=20000, seed=0):
    """Is the spread in `R` driven by sign contamination, or is it the estimator's own shape?

    `R` is a ratio of variance sums, so its sampling distribution is right-skewed by construction; a
    positive skew is therefore NOT by itself evidence that anything is wrong. What would be evidence
    is the high-`R` seeds also being the badly-conditioned ones -- that is what an inadequate depth
    looks like, a few ranks with near-zero sign denominators inflating the raw arm.

    Spearman rank correlation between `R` and each diagnostic separates the two, with a permutation
    p-value so no distributional assumption is smuggled in. Near zero across all three: the skew is
    the estimator's shape, the depth floor is cleared, and the mean is quotable. Strongly positive
    with `r_dev` or `outlier` (or negative with `min_sign`, since SMALL signs are the bad ones): the
    mean is being pulled by contamination and the depth is too shallow to quote.
    """
    rows = [contamination(r) for r in summary["runs"] if r["usable"]]
    if len(rows) < 4:
        return []
    R = _ranks([d["R"] for d in rows])
    rng = np.random.default_rng(seed)
    out = []
    for name in ("r_dev", "min_sign", "outlier"):
        v = np.array([d[name] for d in rows], float)
        if not np.isfinite(v).all() or v.std() == 0:
            continue
        y = _ranks(v)
        rc = float(np.corrcoef(R, y)[0, 1])
        # Permuting one ranking against the other IS the null of no association, exactly. Both
        # vectors are rank vectors with fixed mean and SD, so the correlation over all permutations
        # reduces to their inner product and the whole null distribution is one matrix product.
        perm = np.argsort(rng.random((n_perm, R.size)), axis=1)
        n = R.size
        cen = (R - R.mean()) / R.std()
        ceny = (y - y.mean()) / y.std()
        null = cen[perm] @ ceny / n
        out.append(dict(diagnostic=name, n=n, spearman=rc, n_perm=n_perm,
                        p=float((np.abs(null) >= abs(rc)).mean())))
    return out


def coupling_table(rows):
    lines = [f"{'diagnostic':<12} {'n':>4} {'Spearman vs R':>15} {'perm p':>9}  interpretation"]
    note = {"r_dev": "predicted-vs-measured r mismatch",
            "min_sign": "worst rank denominator (NEGATIVE would be the bad sign)",
            "outlier": "largest per-rank |G| over median"}
    for d in rows:
        lines.append(f"{d['diagnostic']:<12} {d['n']:>4} {d['spearman']:>+15.3f} {d['p']:>9.3f}  "
                     f"{note.get(d['diagnostic'], '')}")
    return "\n".join(lines)


# ---- paired depth check ---------------------------------------------------------------------------

def depth_pairing(shallow, deep):
    """Does the ENSEMBLE MEAN of `R` move when every seed is rerun deeper?

    This is the check the seed ensemble needs and the M-ladder cannot supply. The ladder established
    that `R` is flat in depth within a seed; what a headline needs is that the mean over MANY seeds is
    not biased at the depth actually used. Contamination that survives the floor would show up here as
    a systematic shift, because a near-zero sign denominator is exactly what extra depth cures.

    PAIRED, because it can be. A rank's walker stream does not depend on `measurements`, so at a fixed
    base seed the shallow run's chain is a PREFIX of the deep one's on the same rank. Per-seed
    differences therefore share most of their randomness and cancel it: the paired interval on the
    shift is far tighter than comparing two independent ensemble means, which at this seed count is
    the difference between a real test and a vacuous one.

    Requires the two ensembles to have been run at the SAME base seeds; unmatched seeds are dropped.

    HOW MUCH the pairing actually buys is reported rather than assumed, because it is not guaranteed:
    the shared prefix is only `m_shallow/m_deep` of the deep run, and `R` is a ratio of variance
    sums, so at a 4x depth step most of what sets the deep `R` is data the shallow run never saw. If
    the reported gain is ~1x the comparison is effectively unpaired and its resolution is simply that
    of the seed count -- which is why the resolution is printed next to the verdict. `nesting_audit`
    confirms separately that the prefix relation itself holds, so a gain near 1 is a statement about
    the estimator, not a failed setup.
    """
    by = {}
    for tag, summ in (("shallow", shallow), ("deep", deep)):
        for r in summ["runs"]:
            if r["usable"]:
                by.setdefault((r["model"], r["n_ranks"], r["seed"]), {})[tag] = r
    pairs = [(v["shallow"], v["deep"]) for v in by.values() if "shallow" in v and "deep" in v]
    if len(pairs) < 2:
        return dict(n=0)
    d = np.array([b["R_full"] - a["R_full"] for a, b in pairs], float)
    n = d.size
    sem = float(d.std(ddof=1) / np.sqrt(n))
    half = t975(n - 1) * sem
    a0, b0 = pairs[0]
    # Bias and DISPERSION are separate questions, and depth answers them differently. The mean can be
    # flat while the seed-to-seed SD still falls, which is what a heavy tail looks like when it is
    # symmetric enough not to shift the centre. That distinction decides how to spend compute: if
    # depth only narrows the per-seed spread by less than sqrt(depth factor), seeds are the better
    # buy, since k times the seeds always narrows the mean by sqrt(k).
    ra = np.array([a["R_full"] for a, _ in pairs], float)
    rb = np.array([b["R_full"] for _, b in pairs], float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(20000, n))
    sd_draws = ra[idx].std(ddof=1, axis=1) / rb[idx].std(ddof=1, axis=1)
    sd_lo, sd_hi = np.percentile(sd_draws, [2.5, 97.5])
    return dict(n=n, m_shallow=a0["m"], m_deep=b0["m"],
                sd_shallow=float(ra.std(ddof=1)), sd_deep=float(rb.std(ddof=1)),
                sd_ratio=float(ra.std(ddof=1) / rb.std(ddof=1)),
                sd_ratio_lo=float(sd_lo), sd_ratio_hi=float(sd_hi),
                depth_factor=float(b0["m"] / a0["m"]),
                R_shallow=float(np.mean([a["R_full"] for a, _ in pairs])),
                R_deep=float(np.mean([b["R_full"] for _, b in pairs])),
                dR=float(d.mean()), sem=sem, lo=float(d.mean() - half), hi=float(d.mean() + half),
                flat=bool(d.mean() - half <= 0.0 <= d.mean() + half),
                unpaired_sem=float(np.sqrt(
                    np.array([a["R_full"] for a, _ in pairs]).var(ddof=1) / n
                    + np.array([b["R_full"] for _, b in pairs]).var(ddof=1) / n)))


def nesting_audit(shallow, deep, max_pairs=4):
    """Verify the prefix relation the pairing above assumes, rather than trusting it.

    If rank i's shallow chain really is a prefix of its deep one, their per-rank noise vectors share
    the first `m_shallow` measurements and correlate at ~sqrt(m_shallow/m_deep). Independent runs
    would give ~0. Checked on a few seeds because it is a structural property, not a noisy one.
    """
    by = {}
    for tag, summ in (("shallow", shallow), ("deep", deep)):
        for r in summ["runs"]:
            if r["usable"]:
                by.setdefault(r["seed"], {})[tag] = (os.path.join(summ["dirpath"], r["path"]), r["m"])
    out = []
    for seed in sorted(k for k, v in by.items() if len(v) == 2)[:max_pairs]:
        (pa, ma), (pb, mb) = by[seed]["shallow"], by[seed]["deep"]
        out.append(dict(seed=seed, measured=ms.nesting_check(Run(pa), Run(pb)),
                        expected=float(np.sqrt(ma / mb))))
    return out


def depth_table(pairing, audit):
    if not pairing.get("n"):
        return "[depth] no matched seeds between the two ensembles"
    lines = [f"paired over {pairing['n']} seeds, m={pairing['m_shallow']} -> {pairing['m_deep']}/rank",
             f"  mean R  {pairing['R_shallow']:.4f} -> {pairing['R_deep']:.4f}",
             f"  shift   {pairing['dR']:+.4f} +/- {pairing['sem']:.4f} "
             f"[{pairing['lo']:+.4f},{pairing['hi']:+.4f}]  "
             f"{'FLAT (no depth bias detected)' if pairing['flat'] else 'SHIFTED -- depth matters'}",
             f"  pairing bought {pairing['unpaired_sem'] / pairing['sem']:.2f}x over an unpaired "
             f"comparison (sem {pairing['unpaired_sem']:.4f} -> {pairing['sem']:.4f})",
             f"  spread   per-seed SD {pairing['sd_shallow']:.4f} -> {pairing['sd_deep']:.4f}  "
             f"(ratio {pairing['sd_ratio']:.2f} "
             f"[{pairing['sd_ratio_lo']:.2f},{pairing['sd_ratio_hi']:.2f}] "
             f"against sqrt(depth factor) = {np.sqrt(pairing['depth_factor']):.2f}) -> "
             + ("SEEDS are the better buy" if pairing['sd_ratio'] < np.sqrt(pairing['depth_factor'])
                else "DEPTH is the better buy"),
             f"  resolution: this check excludes a depth shift larger than "
             f"{max(abs(pairing['lo']), abs(pairing['hi'])):.3f} "
             f"({max(abs(pairing['lo']), abs(pairing['hi'])) / pairing['R_shallow']:.1%} of R), "
             f"no finer"]
    if audit:
        lines.append("  nesting audit (measured/expected corr): "
                     + "  ".join(f"s{a['seed']}: {a['measured']:.3f}/{a['expected']:.3f}"
                                 for a in audit))
    return "\n".join(lines)


# ---- independent oracle ---------------------------------------------------------------------------

def whole_run_oracle(summary, n_boot=4000, seed=0):
    """`R` from the variance ACROSS WHOLE RUNS, not across ranks inside one run.

    WHAT THIS IS FOR. The headline estimator forms both arms from the per-rank samples of a single
    run: rank i is one replicate, and `Var` runs over the rank axis. Everything downstream inherits
    that one construction, so a defect in it -- a mis-scaled sample, an arm accidentally sharing a
    reduction with the other -- would be invisible to any amount of resampling within it.

    This is the arms-length version of the same measurement. A whole run is one replicate, `Var` runs
    over the SEED axis, and the per-rank machinery is not involved at any point: no rank-axis
    variance, no bootstrap, no orbit table beyond `P` itself. It is also a 64x change in the depth of
    a single replicate (m per rank, against n_ranks*m per run), so agreement is simultaneously a
    direct test of the scale-freeness the whole budget argument rests on.

    WHAT IT DOES NOT ISOLATE, stated plainly: both arms still apply the same numpy `P`. This checks
    the variance construction and the scale-free claim, not the operator -- that is what rung 1
    (singletons pin at r=1) and rung 2 (mean preservation against DCA's own C++ symmetrization to
    2e-16) are for. Between the three, the estimator is checked from every side that matters.

    The whole-run mean is the phase-weighted one, `sum_i(sign_i G_i)/sum_i(sign_i)`, which is how
    production normalizes; the naive rank mean would use each rank's own phase and is a different
    quantity at finite depth.
    """
    out = []
    for (model, ranks, m, beta), g in groups(summary).items():
        g = [r for r in g if r["usable"]]
        if len(g) < 3:
            continue
        # The oracle needs the raw HDF5s, which live in scratch and are not committed -- only the
        # summary JSON is. Reloading a summary after that scratch is gone is the normal case, not an
        # error, so say what is missing and move on rather than dying on the first open().
        if not all(os.path.exists(os.path.join(summary["dirpath"], r["path"])) for r in g):
            out.append(dict(model=model, n_ranks=ranks, m=m, n_runs=len(g),
                            depth_per_replicate=m * ranks, R_oracle=None,
                            unavailable=summary["dirpath"]))
            continue
        vecs = []
        for r in g:
            run = Run(os.path.join(summary["dirpath"], r["path"]))
            vecs.append(run._to_vec(run.phase_weighted_mean()))   # (nw, ns, E), one per whole run
            P = run.P
        V = np.stack(vecs)                                        # (n_seeds, nw, ns, E)
        S = np.einsum("oi,...i->...o", P, V)

        def flat(x):
            n = x.shape[0]
            return np.concatenate([x.real.reshape(n, -1), x.imag.reshape(n, -1)], axis=1)

        A, B = flat(V), flat(S)
        R = float(A.var(0, ddof=1).sum() / B.var(0, ddof=1).sum())
        # Bootstrap over whole runs -- the replicate unit here, exactly as ranks are there.
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, A.shape[0], size=(n_boot, A.shape[0]))
        draws = np.array([A[i].var(0, ddof=1).sum() / B[i].var(0, ddof=1).sum() for i in idx])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out.append(dict(model=model, n_ranks=ranks, m=m, n_runs=len(g),
                        depth_per_replicate=m * ranks, R_oracle=R,
                        lo=float(lo), hi=float(hi), se=float(draws.std(ddof=1))))
    return out


def oracle_table(oracle, rows):
    by = {(h["model"], h["n_ranks"], h["m"]): h for h in rows}
    lines = [f"{'model':<10} {'m/rank':>7} {'runs':>5} {'depth/replicate':>16} "
             f"{'R_oracle':>9} {'95% CI':>18} {'headline R':>11} {'agree?':>8}"]
    for o in oracle:
        if o.get("R_oracle") is None:
            lines.append(f"{o['model']:<10} {o['m']:>7} -- raw runs not present at "
                         f"{o.get('unavailable')}; oracle needs them, the summary JSON is not enough")
            continue
        h = by.get((o["model"], o["n_ranks"], o["m"]))
        hr = h["R_full"]["mean"] if h else float("nan")
        ok = "yes" if (o["lo"] <= hr <= o["hi"]) else "NO"
        ci = f"[{o['lo']:.4f},{o['hi']:.4f}]"
        lines.append(f"{o['model']:<10} {o['m']:>7} {o['n_runs']:>5} {o['depth_per_replicate']:>16} "
                     f"{o['R_oracle']:>9.4f} {ci:>18} {hr:>11.4f} {ok:>8}")
    return "\n".join(lines)


def report(summary, n_boot=4000, deep=None):
    rows = headline(summary, n_boot)
    bad = check_seed_spacing(summary["runs"])
    parts = []
    if bad:
        parts.append("!! SEED SPACING VIOLATION -- runs share walker streams, intervals are void:\n"
                     + json.dumps(bad, indent=1))
    for title, text in (("headline R (across independent base seeds)", headline_table(rows)),
                        ("robustness cross-checks", robustness_table(rows)),
                        ("secondary / internal", secondary_table(rows)),
                        ("rank-bootstrap calibration", calibration_table(rows)),
                        ("contamination gate", contamination_table(summary)),
                        ("is the skew contamination or estimator shape?",
                         coupling_table(contamination_coupling(summary)))):
        parts.append(f"=== {title} ===\n{text}")
    parts.append("=== independent oracle: variance across whole runs ===\n"
                 + oracle_table(whole_run_oracle(summary), rows))
    if deep is not None:
        parts.append("=== paired depth check ===\n"
                     + depth_table(depth_pairing(summary, deep), nesting_audit(summary, deep)))
    return rows, "\n\n".join(parts)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirpath", nargs="?", help="directory of ensemble HDF5 runs")
    ap.add_argument("--summary", default=None, help="reuse an existing per-run summary JSON")
    ap.add_argument("--out", default=None, help="write the summary + headline as JSON here")
    ap.add_argument("--boot", type=int, default=400, help="rank-bootstrap draws per run")
    ap.add_argument("--deep", default=None,
                    help="second ensemble at greater depth, SAME base seeds, for the paired depth check")
    args = ap.parse_args()

    if args.summary:
        with open(args.summary) as fh:
            summary = json.load(fh)
        summary.setdefault("dirpath", os.path.dirname(os.path.abspath(args.summary)))
    else:
        summary = build(args.dirpath, args.boot)

    deep = build(args.deep, args.boot) if args.deep else None
    rows, text = report(summary, deep=deep)
    print("\n" + text)
    if args.out:
        blob = dict(runs=summary["runs"], dirpath=summary["dirpath"], headline=rows,
                    oracle=whole_run_oracle(summary))
        if deep is not None:
            blob["deep_runs"] = deep["runs"]
            blob["depth_pairing"] = depth_pairing(summary, deep)
        with open(args.out, "w") as fh:
            json.dump(blob, fh, indent=1)
        print(f"\n[seed_ensemble] wrote {args.out}")
