"""Readers for the shipped ensemble summaries.

Every number this module returns is recomputed here from PER-SEED records, not copied from a
precomputed field: each summary file stores one R per independent run, and the means and intervals
below are formed from those. What cannot be recomputed from the bundle is the per-seed R itself,
which required the full 20 GB of raw output the campaign produced.

WHY SEEDS AND NOT RANKS. Two estimators of the uncertainty on R are available. Resampling ranks
within one run is cheap but assumes the ranks are exchangeable draws from a well-behaved
distribution; where a model has a sign problem that assumption fails, because a resample of 64 ranks
cannot see the heavy tail a near-zero sign denominator creates, and the interval comes out
optimistic. Whole independent runs as the replicate unit make no such assumption. Everything quoted
here uses independent base seeds.

BASE SEEDS MUST BE SPACED. A walker's stream is hash(global_id + base_seed) with
global_id = local_id * n_ranks + proc_id, so a run occupies the key range [S, S + n_ranks*n_walkers).
Base seeds closer together than that share chains, and the runs are not independent replicates. The
campaign enforced a stride of 10000; the check is re-run in `seed_spacing_ok` below.
"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "summaries"


def _load(name):
    with open(DATA / name) as f:
        return json.load(f)


def _stats(values):
    """Mean and standard error over independent replicates."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    n = len(v)
    # A single replicate has a mean but no spread; report nan rather than warning on ddof=1.
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    return dict(n=n, mean=float(v.mean()), sem=sd / np.sqrt(n) if n > 1 else float("nan"),
                sd=sd, min=float(v.min()), max=float(v.max()), values=v)


def _from_runs(runs, key="R_full"):
    return _stats([r[key] for r in runs if r.get("usable", True)])


def seed_spacing_ok(runs, stride=10000):
    """Independent replicates require base seeds further apart than one run's key range."""
    seeds = sorted(r["seed"] for r in runs)
    gaps = np.diff(seeds)
    return bool(len(gaps) == 0 or gaps.min() >= stride), (int(gaps.min()) if len(gaps) else None)


# =================================================================================================
# The headline ensembles
# =================================================================================================
def headline(model):
    """The 32-seed ensemble for one of the two headline points.

    'square_b1' is the cold-start reference at beta = 1; 'fe_as_b5' is the multi-orbital headline at
    the production temperature.
    """
    name = {"square_b1": "seed_ensemble_square.json", "fe_as_b5": "seed_ensemble_fe_as.json"}[model]
    d = _load(name)
    runs = d["runs"]
    ok, gap = seed_spacing_ok(runs)
    return dict(model=model, n_seeds=len(runs), seed_spacing_ok=ok, min_seed_gap=gap,
                R_full=_from_runs(runs, "R_full"), R_nonnull=_from_runs(runs, "R_nonnull"),
                w_null=_stats([1.0 - r["R_nonnull"] / r["R_full"] for r in runs]))


# =================================================================================================
# The beta=5 design points
# =================================================================================================
def design_points():
    """The three sweep points at the production config, each aggregated over its own seed ensemble.

    Returns one row per point with the ensemble R, the measured noise correlation ceiling 1/rho, the
    geometric ceiling set by orbit sizes, and which of the two binds.
    """
    d = _load("model_sweep.json")
    rows = []
    by_label = {r["label"]: r for r in d["rows"]}
    for p in d["points"]:
        pt, runs = p["point"], p["runs"]
        st = p["structure"]
        meta = by_label[pt["label"]]
        rows.append(dict(
            label=pt["label"], model=pt["model_label"], beta=pt["beta"], nb=pt["nb"],
            nk=pt["nk"], L=pt["L"], n_ops=pt["n_ops"], n_seeds=len(runs),
            R_full=_from_runs(runs, "R_full"), R_nonnull=_from_runs(runs, "R_nonnull"),
            orbit_sizes={int(k): v for k, v in st["orbit_sizes"].items()},
            band_classes=st["band_equivalence"]["classes"],
            frac_offblock=st["permutation"]["frac_offblock"],
            n_forced_null=st["permutation"]["n_forced_null"],
            inv_rho=meta["inv_rho"], m_ceiling=meta["m_ceiling"], binding=meta["binding"],
        ))
    return rows


def feas_ceilings():
    """The two ceilings for FeAs, built to the same definitions the sweep uses for its own points.

    FeAs is not one of the three sweep points, so its row would otherwise be blank. `m_ceiling` is
    the across-seed mean of the per-run `R_ideal` (the rho = 0 ceiling for the actual orbit
    structure, variance-weighted); `1/rho` uses the mate correlation from the beta = 5 rung of its
    own ladder ensemble. Both are ensemble quantities, exactly as for the sweep points.
    """
    d = _load("seed_ensemble_fe_as.json")
    ideal = _stats([r["R_ideal"] for r in d["runs"] if r.get("usable", True)])
    rho = [r for r in beta_ladder("fe_as") if r["beta"] == 5][0]["rho"]
    inv_rho = 1.0 / rho
    return dict(m_ceiling=ideal["mean"], m_ceiling_sem=ideal["sem"], rho=rho, inv_rho=inv_rho,
                binding="rho" if inv_rho < ideal["mean"] else "m",
                efficiency=_stats([r["efficiency"] for r in d["runs"]])["mean"])


def class_contrasts():
    """Within-model contrasts between entry classes -- the controls that isolate what symmetry does."""
    return _load("model_sweep.json")["class_contrasts"]


def systematics():
    """Disclosed systematics measured as a separate arm rather than corrected for."""
    return _load("model_sweep.json")["systematics"]


def class_table(label, scheme="equivalence"):
    """Per-class R_C and variance share for one design point, averaged across the seed ensemble.

    `class_stats` carries the across-seed version; `by_class` holds the single-run one. Use the
    ensemble, so a class-level number carries the same replicate unit as the headline.

    A class of forced nulls has R_C = infinity, because its symmetrized variance is exactly zero.
    That is not a divergence to guard against -- it is the whole contribution: such a class drops out
    of the harmonic sum entirely and lifts the aggregate by its variance share.
    """
    d = _load("model_sweep.json")
    for p in d["points"]:
        if p["point"]["label"] == label:
            out = []
            for cls, v in p["class_stats"][scheme].items():
                # A forced-null class stores R_C as null: it is infinite, which JSON cannot hold.
                rc = v["R_C"]
                out.append(dict(cls=cls, n=v["n"], mean_m=v["mean_m"],
                                R_C=rc["mean"] if rc else float("inf"),
                                R_C_sem=rc["sem"] if rc else float("nan"),
                                w=v["w"]["mean"], w_sem=v["w"]["sem"]))
            return sorted(out, key=lambda r: -r["w"])
    raise KeyError(label)


# =================================================================================================
# The axes
# =================================================================================================
def beta_ladder(model):
    """One row per temperature rung: R, the mate correlation rho, the forced-null share, the sign."""
    d = _load(f"beta_ladder_{model}.json")
    rows = []
    for r in d["report"]["rungs"]:
        rows.append(dict(beta=r["beta"], n_seeds=r["n_seeds"],
                         R_full=r["R_full"]["mean"], R_full_sem=r["R_full"]["sem"],
                         R_nonnull=r["R_nonnull"]["mean"], R_nonnull_sem=r["R_nonnull"]["sem"],
                         rho=r["rho"]["mean"], rho_sem=r["rho"]["sem"],
                         w_null=r["w_null"]["mean"],
                         min_sign=r["sign"].get("min_abs_sign") if isinstance(r["sign"], dict) else None))
    return sorted(rows, key=lambda r: r["beta"])


def depth_ladder():
    """R against measurements per rank -- the control for whether R depends on how long you run.

    R is a ratio of two variances measured from the same samples, so it should be scale-free: run
    twice as long and both variances halve. That holds only ABOVE a per-model depth floor, below
    which the per-rank estimates are still heavy-tailed.
    """
    d = _load("m_scaling_summary.json")
    rows = {}
    for r in d["runs"]:
        if not r.get("usable", True):
            continue
        rows.setdefault((r["model"], r["m"]), []).append(r["R_full"])
    out = []
    for (model, m), vals in sorted(rows.items()):
        s = _stats(vals)
        out.append(dict(model=model, m_per_rank=m, n=s["n"], R_full=s["mean"],
                        sem=s["sem"] if s["n"] > 1 else float("nan")))
    return out


def bath_drift():
    """R at every iteration of a real self-consistent DCA loop.

    The justification for measuring at the bare bath: if R does not drift as the loop's bath evolves,
    a single-iteration measurement transfers to the iterations a production run actually performs.
    """
    d = _load("bath_drift_square.json")
    # bath_step is absent at iteration 0 -- there is no previous bath to have stepped from.
    rows = [dict(iteration=r["iteration"], n_seeds=r["n_seeds"],
                 R_full=r["R_full"]["mean"], R_full_sem=r["R_full"]["sem"],
                 bath_step=(r["bath_step"] or {}).get("mean"), min_sign=r["min_sign"])
            for r in d["by_iteration"]]
    return dict(by_iteration=sorted(rows, key=lambda r: r["iteration"]),
                paired=d["paired_drift"], cost_weighted=d["cost_weighted"],
                reference_gap=d["reference_gap"])
