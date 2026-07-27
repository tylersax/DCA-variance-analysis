"""M-scaling control (ROADMAP task 1): is `R` independent of run depth?

WHY THIS EXISTS. The whole budget strategy -- "many ranks, modest depth, and the measured `R`
transfers to any production scale" -- rests on `R` being scale-free. The argument is that
`Var(G_N) = Sigma/N` and `Var(P G_N) = P Sigma P^T / N`, so the `1/N` cancels and `R` depends only on
the noise covariance STRUCTURE. That argument assumes each rank's sample is in the CLT regime: past
thermalization and well beyond the autocorrelation time. Below that threshold `Var(G_i)` is not
`Sigma/M`, and the raw and symmetrized arms need not approach their asymptotes at the same rate --
so `R` can drift with `M`. This module measures the drift instead of assuming it away.

WHAT IT REPORTS, per run:
  * `R` on the full and non-null supports, with a bootstrap-over-ranks CI;
  * `M * sum(Var_raw)` -- flat in `M` iff the per-rank sample is diffusive (the 1/M law). This is the
    stricter test: `R` is a ratio, so a common departure from 1/M cancels in it and only a
    DIFFERENTIAL departure between the arms shows up. `M*sum(Var)` catches the common part too, and
    it is the direct fingerprint of autocorrelation/thermalization.
  * the mechanism diagnostics (real-space `sigma^2(r=0)` share, within/across-shell correlation,
    mate-rho) -- drift THERE is the early warning that the noise structure itself is still moving.

PAIRING. At fixed base seed a rank's walker streams do not depend on `measurements`, so the M-run's
chain is a prefix of the 4M-run's on the same rank (see run_m_ladder.sh). `paired_bootstrap_dR`
resamples the SAME rank indices in both runs, so the shared randomness cancels and the interval on
`R(4M) - R(M)` is much tighter than the two marginal intervals would suggest. `nesting_check`
verifies the prefix relation empirically before that pairing is relied on.

CLI:  python m_scaling.py <ladder_dir> [--out summary.json] [--boot 400]
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symm_variance_lib import Run, predicted_r  # noqa: E402
import noise_diagnostics as nd  # noqa: E402
import reduction_map as rm  # noqa: E402

_NAME_RE = re.compile(r"^(?P<model>.+)_r(?P<ranks>\d+)_m(?P<m>\d+)_s(?P<seed>\d+)\.hdf5$")


def parse_name(path):
    """Ladder filenames carry the design point; the HDF5 metadata is the authority we cross-check."""
    g = _NAME_RE.match(os.path.basename(path))
    if not g:
        return None
    return dict(model=g["model"], n_ranks=int(g["ranks"]), m=int(g["m"]), seed=int(g["seed"]))


def read_depth(h, n_ranks):
    """Measurements PER RANK -- the axis that matters, since a rank is one sample.

    DCA's `measurements` input is the TOTAL over ranks (every solver routes it through
    parallel::util::getWorkload). Runs written before 2026-07-27 stored that total under the key
    `measurements_per_rank`, so for those files the depth has to be divided down; newer files carry
    both keys and are read directly.
    """
    if "metadata/measurements_total" in h:
        return int(h["metadata/measurements_per_rank"][()][0]), int(h["metadata/measurements_total"][()][0])
    total = int(h["metadata/measurements_per_rank"][()][0])  # legacy: mislabeled total
    return total // n_ranks, total


# ---- bootstrap over ranks -------------------------------------------------------------------------

def _arms(run, subset=None):
    """Raw and symmetrized per-rank samples as real matrices (n_rank, N), ready for resampling.

    Real and imaginary parts are concatenated as separate coordinates: `R` is defined on the sum of
    their variances, and treating them as one real vector makes the bootstrap a plain resample.
    """
    vec = run.vec                         # (R, nw, ns, E)
    vsym = run.apply_P(vec)
    if subset is not None:
        vec, vsym = vec[..., subset], vsym[..., subset]

    def flat(v):
        n = v.shape[0]
        return np.concatenate([v.real.reshape(n, -1), v.imag.reshape(n, -1)], axis=1)

    return flat(vec), flat(vsym)


def _counts(idx, n):
    """Bootstrap resamples as multiplicity counts (n_boot, n_rank) rather than index lists."""
    C = np.zeros((idx.shape[0], n))
    for b in range(idx.shape[0]):
        C[b] = np.bincount(idx[b], minlength=n)
    return C


def _var_sum(A, C):
    """sum_entries Var over each bootstrap resample, from sufficient statistics.

    Materializing A[idx] is (n_boot, n_rank, n_entries) and dominates the runtime. A resample is
    fully described by its multiplicity vector c, and
        sum_i x_i = c.A,  sum_i x_i^2 = c.(A*A),  Var = (sum x^2 - (sum x)^2/n) / (n-1),
    so both moments come from one matrix product each. Only the first-moment matrix has to be kept at
    full width, because the entry-sum of the second moment collapses to a vector first.
    """
    n = A.shape[0]
    S1 = C @ A                       # (n_boot, n_entries)
    s2 = C @ (A * A).sum(1)          # (n_boot,)  -- entry-summed second moment
    return (s2 - (S1 * S1).sum(1) / n) / (n - 1)


def _ratio(A, B, C):
    """R over bootstrap resamples described by the multiplicity matrix C."""
    return _var_sum(A, C) / _var_sum(B, C)


def bootstrap_R(run, n_boot=400, subset=None, seed=0):
    """Percentile bootstrap CI for `R`, resampling RANKS (the independent replicates)."""
    A, B = _arms(run, subset)
    n = A.shape[0]
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    draws = _ratio(A, B, _counts(idx, n))
    point = float(A.var(0, ddof=1).sum() / B.var(0, ddof=1).sum())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(R=point, lo=float(lo), hi=float(hi), se=float(draws.std(ddof=1)))


def paired_bootstrap_dR(run_a, run_b, n_boot=400, subset=None, seed=0):
    """CI on `R(b) - R(a)` resampling the SAME rank indices in both runs.

    Valid because rank i means the same RNG stream in both (nested chains at a fixed base seed), so
    the shared randomness cancels in the difference. Requires equal rank counts.
    """
    assert run_a.n_ranks == run_b.n_ranks, "paired bootstrap needs matching rank counts"
    Aa, Ba = _arms(run_a, subset)
    Ab, Bb = _arms(run_b, subset)
    n = Aa.shape[0]
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    C = _counts(idx, n)
    d = _ratio(Ab, Bb, C) - _ratio(Aa, Ba, C)
    point = float(Ab.var(0, ddof=1).sum() / Bb.var(0, ddof=1).sum()
                  - Aa.var(0, ddof=1).sum() / Ba.var(0, ddof=1).sum())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(dR=point, lo=float(lo), hi=float(hi), se=float(d.std(ddof=1)),
                consistent_with_zero=bool(lo <= 0.0 <= hi))


def nesting_check(run_a, run_b, b0=0, b1=0, spin=0):
    """Empirical test of the prefix relation between a depth-M and a depth-M' run at the same seed.

    If rank i's chain at M is a prefix of its chain at M' > M, the two per-rank noise vectors share
    the first M measurements, giving corr ~= sqrt(M/M'). Independent runs would give ~0. Reported as
    (measured, expected) so the pairing above is used only when it is actually earned.
    """
    da = nd.noise_residuals(run_a, b0, b1, spin).reshape(run_a.n_ranks, -1)
    db = nd.noise_residuals(run_b, b0, b1, spin).reshape(run_b.n_ranks, -1)
    n = min(da.shape[0], db.shape[0])
    cs = []
    for i in range(n):
        xa, xb = da[i], db[i]
        den = np.sqrt(np.vdot(xa, xa) * np.vdot(xb, xb))
        if abs(den) > 0:
            cs.append((np.vdot(xa, xb) / den).real)
    return float(np.mean(cs))


# ---- per-run summary ------------------------------------------------------------------------------

def sign_health(run):
    """Is the per-rank ratio estimator well-conditioned at this depth?

    `G_i = <sign*M>_i / <sign>_i` is a RATIO with a stochastic denominator. Under a sign problem the
    per-rank average sign is O(<s>) with fluctuations O(1/sqrt(m)), so at shallow depth some rank
    lands near zero, its `G_i` blows up, and that single rank dominates the variance sum -- the
    estimator becomes heavy-tailed long before it becomes wrong. At the extreme the rank's sign is
    exactly zero and `G_i` is 0/0 = NaN.

    So the depth a model needs is set by whichever binds first: autocorrelation (square) or keeping
    every rank's denominator away from zero (FeAs, where <s> ~ 0.25 at beta=5).

    Reports the mean sign per rank, its worst value over ranks, and an outlier index -- the ratio of
    the largest per-rank |G| to the median, which is ~1 for a healthy sample.
    """
    s = np.abs(run.sign)
    amp = np.abs(run.G_raw).reshape(run.n_ranks, -1)
    with np.errstate(invalid="ignore"):
        peak = np.nanmax(np.where(np.isfinite(amp), amp, np.nan), axis=1)
    finite = np.isfinite(run.G_raw).reshape(run.n_ranks, -1).all(1)
    med = float(np.median(peak[finite])) if finite.any() else float("nan")
    return dict(n_ranks_nonfinite=int((~finite).sum()),
                n_ranks_zero_sign=int((s == 0).sum()),
                min_abs_sign=float(s.min()), max_abs_sign=float(s.max()),
                outlier_index=float(np.nanmax(peak[finite]) / med) if finite.any() and med > 0
                else float("nan"))


def mechanism(run, b0=0, b1=0, spin=0):
    """The noise-structure diagnostics, so drift in the MECHANISM is visible, not just in `R`."""
    prof = nd.realspace_noise_profile(run, b0, b1, spin)
    sh = nd.shell_correlations(run, b0, b1, spin)
    return dict(local_share=float(prof[0, 0]), rho_pred=float(prof[0, 0] - 1.0 / run.nk),
                within_shell=sh["within"], across_shell=sh["across"],
                rho_generic=nd.pairwise_rho_all_k(run, b0, b1, spin))


def orbit_rows(run):
    """Per-orbit m, measured rho, measured r and the predicted m/[1+(m-1)rho]."""
    A, B = rm.entry_variance(run)
    rows = []
    for o in run.orbits():
        if o["forced_null"]:
            continue
        mem = o["members"]
        rho = run.orbit_rho(mem, o["signs"])
        den = B[mem].sum()
        rows.append(dict(m=len(mem), members=[int(x) for x in mem],
                         rho=None if np.isnan(rho) else float(rho),
                         r=float(A[mem].sum() / den) if den > 0 else None,
                         r_pred=None if np.isnan(rho) else float(predicted_r(len(mem), rho))))
    return rows


def summarize(path, n_boot=400, blocks=None):
    """Everything the ladder needs from one run, as plain JSON-able types."""
    run = Run(path)
    meta = parse_name(path) or {}
    with __import__("h5py").File(path, "r") as h:
        m_per_rank, m_total = read_depth(h, run.n_ranks)
        seed = int(h["metadata/seed"][()][0])
    if meta:  # the filename is a convenience; the file's own metadata wins
        assert meta["m"] == m_per_rank and meta["seed"] == seed, f"name/metadata mismatch: {path}"

    sh = sign_health(run)
    # Average sign per rank = accumulated sign / that rank's measurement count. The MINIMUM over
    # ranks is the number that matters: it is the worst denominator anywhere in the sample.
    sh["min_mean_sign"] = sh["min_abs_sign"] / m_per_rank
    sh["max_mean_sign"] = sh["max_abs_sign"] / m_per_rank
    # A rank whose accumulated sign vanished carries G = 0/0. Such a run cannot yield a variance at
    # all, so report the diagnosis instead of propagating NaN into every downstream number.
    usable = sh["n_ranks_nonfinite"] == 0

    _, null = rm.orbit_info(run)
    nn = [x for x in range(run.E) if not null[x]]
    A, B = rm.entry_variance(run) if usable else (None, None)
    if not usable:
        return dict(path=os.path.basename(path), model=run.model, m=m_per_rank, m_total=m_total,
                    seed=seed, n_ranks=run.n_ranks, n_ops=run.n_ops, nb=run.nb, nk=run.nk,
                    nw=run.nw, usable=False, sign=sh, R_full=None, R_nonnull=None)

    if blocks is None:
        blocks = {"band-diagonal": (0, 0, 0)} if run.nb == 1 else \
                 {"intraband": (0, 0, 0), "interband": (0, 1, 0)}

    boot_full = bootstrap_R(run, n_boot, None, seed=seed)
    boot_nn = bootstrap_R(run, n_boot, nn, seed=seed)
    sumvar = float(A.sum())

    rows, recon, agg = rm.reduction_table(run, rm.band_classes(run))
    rows_m, _, _ = rm.reduction_table(run, rm.orbit_classes(run))

    return dict(
        path=os.path.basename(path), model=run.model, m=m_per_rank, m_total=m_total, seed=seed,
        n_ranks=run.n_ranks, n_ops=run.n_ops, nb=run.nb, nk=run.nk, nw=run.nw,
        usable=True, sign=sh,
        R_full=agg, R_full_ci=[boot_full["lo"], boot_full["hi"]], R_full_se=boot_full["se"],
        R_nonnull=rm.R_over(run, nn, A, B), R_nonnull_ci=[boot_nn["lo"], boot_nn["hi"]],
        R_nonnull_se=boot_nn["se"],
        R_ideal=rm.R_ideal(run, nn, A), efficiency=rm.efficiency(run, nn),
        sum_var_raw=sumvar, m_times_sum_var=m_per_rank * sumvar,
        harmonic_reconstruction=recon,
        by_band={r["cls"]: dict(n=r["n"], w=r["w"], R_C=r["R_C"]) for r in rows},
        by_orbit_size={r["cls"]: dict(n=r["n"], w=r["w"], R_C=r["R_C"]) for r in rows_m},
        orbits=orbit_rows(run),
        mechanism={name: mechanism(run, *idx) for name, idx in blocks.items()},
    )


# ---- ladder assembly ------------------------------------------------------------------------------

def build_ladder(dirpath, n_boot=400, verbose=True):
    """Summarize every ladder run in a directory, then add the paired M-to-M drift tests."""
    paths = sorted(p for p in glob.glob(os.path.join(dirpath, "*.hdf5")) if parse_name(p))
    runs = []
    for p in paths:
        if verbose:
            print(f"[m_scaling] {os.path.basename(p)}", flush=True)
        runs.append(summarize(p, n_boot))

    # Paired drift between consecutive depths, within (model, seed, ranks).
    drift = []
    key = lambda s: (s["model"], s["seed"], s["n_ranks"])  # noqa: E731
    groups = {}
    for s in runs:
        groups.setdefault(key(s), []).append(s)
    cache = {}

    def load(name):  # each run appears in two consecutive drift steps; load and vectorize it once
        if name not in cache:
            cache[name] = Run(os.path.join(dirpath, name))
        return cache[name]

    for k, g in groups.items():
        # Unusable runs (a rank with zero accumulated sign) have no variance to compare against, so
        # they drop out of the drift chain rather than poisoning it; the ladder table still lists
        # them, since knowing WHERE the estimator breaks is half the answer here.
        g = sorted((s for s in g if s["usable"]), key=lambda s: s["m"])
        for a, b in zip(g, g[1:]):
            ra, rb = load(a["path"]), load(b["path"])
            _, null = rm.orbit_info(ra)
            nn = [x for x in range(ra.E) if not null[x]]
            d_full = paired_bootstrap_dR(ra, rb, n_boot, None, seed=a["seed"])
            d_nn = paired_bootstrap_dR(ra, rb, n_boot, nn, seed=a["seed"])
            drift.append(dict(model=k[0], seed=k[1], n_ranks=k[2], m_from=a["m"], m_to=b["m"],
                              full=d_full, nonnull=d_nn,
                              nesting_corr=nesting_check(ra, rb),
                              nesting_expected=float(np.sqrt(a["m"] / b["m"]))))
            if verbose:
                print(f"[m_scaling] drift {k[0]} s{k[1]} {a['m']}->{b['m']}: "
                      f"dR={d_full['dR']:+.4f} [{d_full['lo']:+.4f},{d_full['hi']:+.4f}]", flush=True)
    return dict(runs=runs, drift=drift)


def ladder_table(summary):
    """Compact text table of R and the 1/m law, one line per run."""
    lines = [f"{'model':<10} {'seed':>9} {'ranks':>5} {'m/rank':>7} {'R_full':>18} {'R_nonnull':>18} "
             f"{'m*sumVar':>11} {'eff':>7} {'min<s>':>7} {'outlier':>8}"]
    for s in sorted(summary["runs"], key=lambda s: (s["model"], s["seed"], s["m"])):
        sg = s["sign"]
        if not s["usable"]:
            lines.append(f"{s['model']:<10} {s['seed']:>9} {s['n_ranks']:>5} {s['m']:>7} "
                         f"{'-- unusable: ' + str(sg['n_ranks_zero_sign']) + ' rank(s) with zero accumulated sign --':<57} "
                         f"{0.0:>7.4f}")
            continue
        lo, hi = s["R_full_ci"]
        lo2, hi2 = s["R_nonnull_ci"]
        lines.append(f"{s['model']:<10} {s['seed']:>9} {s['n_ranks']:>5} {s['m']:>7} "
                     f"{s['R_full']:>7.4f} [{lo:.3f},{hi:.3f}] "
                     f"{s['R_nonnull']:>7.4f} [{lo2:.3f},{hi2:.3f}] "
                     f"{s['m_times_sum_var']:>11.4g} {s['efficiency']:>6.1%} "
                     f"{sg['min_mean_sign']:>7.4f} {sg['outlier_index']:>8.2f}")
    return "\n".join(lines)


def mechanism_table(summary, seed=None):
    """Noise-structure diagnostics vs depth.

    `R` drifting is the symptom; these say what moved underneath it. The expected signature of a
    sub-asymptotic sample is a LOW mate-rho at small `m`: a single CT-AUX configuration is not
    symmetric, so a rank that has averaged only a few of them still carries raw configuration-level
    asymmetry -- pure symmetry-BREAKING noise, which `P` removes in full and which therefore inflates
    `R`. As `m` grows that part self-averages away within each rank, leaving the symmetric scalar
    channels `P` cannot touch, and rho rises to its asymptote.
    """
    rows = [r for r in summary["runs"] if r["usable"] and (seed is None or r["seed"] == seed)]
    blocks = sorted({b for r in rows for b in r["mechanism"]})
    head = f"{'model':<10} {'seed':>9} {'m/rank':>7} {'block':<14} " \
           f"{'local share':>11} {'rho_pred':>9} {'within':>8} {'across':>8} {'rho_generic':>11}"
    lines = [head]
    for r in sorted(rows, key=lambda s: (s["model"], s["seed"], s["m"])):
        for b in blocks:
            mm = r["mechanism"].get(b)
            if mm is None:
                continue
            lines.append(f"{r['model']:<10} {r['seed']:>9} {r['m']:>7} {b:<14} "
                         f"{mm['local_share']:>11.4f} {mm['rho_pred']:>9.4f} "
                         f"{mm['within_shell']:>+8.4f} {mm['across_shell']:>+8.4f} "
                         f"{mm['rho_generic']:>+11.4f}")
    return "\n".join(lines)


def orbit_rho_table(summary, seed=None):
    """Measured mate-rho and r per orbit size, vs depth -- the same story at the orbit level."""
    rows = [r for r in summary["runs"] if r["usable"] and (seed is None or r["seed"] == seed)]
    lines = [f"{'model':<10} {'seed':>9} {'m/rank':>7}  per-orbit (size m: rho -> r [pred])"]
    for r in sorted(rows, key=lambda s: (s["model"], s["seed"], s["m"])):
        by_m = {}
        for o in r["orbits"]:
            if o["rho"] is not None:
                by_m.setdefault(o["m"], []).append(o)
        parts = []
        for msz in sorted(by_m):
            os_ = by_m[msz]
            rho = float(np.mean([o["rho"] for o in os_]))
            rr = float(np.mean([o["r"] for o in os_]))
            rp = float(np.mean([o["r_pred"] for o in os_]))
            parts.append(f"m={msz}: rho={rho:+.3f} -> r={rr:.3f} [{rp:.3f}]")
        lines.append(f"{r['model']:<10} {r['seed']:>9} {r['m']:>7}  " + "   ".join(parts))
    return "\n".join(lines)


def ci_validation(summary, min_seeds=3):
    """Does the bootstrap-over-ranks SE match the scatter actually seen across base seeds?

    The bootstrap treats ranks as iid draws and resamples them; independent base seeds give a whole
    fresh ladder, so the sample SD of `R` across seeds at fixed depth is an EXTERNAL check on that
    interval, sharing no assumption with it. They should agree up to the large sampling error of an
    SD from a handful of seeds. Bootstrap SE much smaller than the seed scatter would mean the ranks
    are not as independent as the resampling assumes.
    """
    groups = {}
    for r in summary["runs"]:
        if r["usable"]:
            groups.setdefault((r["model"], r["n_ranks"], r["m"]), []).append(r)
    lines = [f"{'model':<10} {'ranks':>5} {'m/rank':>7} {'seeds':>5} {'mean R':>8} "
             f"{'boot SE':>8} {'seed SD':>8} {'ratio':>6}"]
    for (model, ranks, m), g in sorted(groups.items()):
        if len(g) < min_seeds:
            continue
        Rs = np.array([r["R_full"] for r in g])
        se = float(np.mean([r["R_full_se"] for r in g]))
        sd = float(Rs.std(ddof=1))
        lines.append(f"{model:<10} {ranks:>5} {m:>7} {len(g):>5} {Rs.mean():>8.4f} "
                     f"{se:>8.4f} {sd:>8.4f} {sd/se if se > 0 else np.nan:>6.2f}")
    return "\n".join(lines)


def drift_table(summary):
    lines = [f"{'model':<10} {'seed':>9} {'step':>16} {'dR_full':>26} {'nesting':>16}"]
    for d in summary["drift"]:
        f = d["full"]
        flag = "flat" if f["consistent_with_zero"] else "DRIFT"
        lines.append(f"{d['model']:<10} {d['seed']:>9} {d['m_from']:>7}->{d['m_to']:<8} "
                     f"{f['dR']:>+8.4f} [{f['lo']:+.4f},{f['hi']:+.4f}] {flag:<6} "
                     f"{d['nesting_corr']:>7.3f}/{d['nesting_expected']:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirpath")
    ap.add_argument("--out", default=None, help="write the summary as JSON here")
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    summary = build_ladder(args.dirpath, args.boot)
    for title, fn in (("R vs depth", ladder_table), ("paired drift between depths", drift_table),
                      ("bootstrap SE vs across-seed scatter", ci_validation),
                      ("per-orbit rho and r vs depth", orbit_rho_table),
                      ("noise-structure diagnostics vs depth", mechanism_table)):
        print(f"\n=== {title} ===")
        print(fn(summary))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=1)
        print(f"\n[m_scaling] wrote {args.out}")
