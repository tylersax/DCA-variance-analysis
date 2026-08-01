"""Stage the shipped data bundle for the hero notebook.

Run once from this directory. Everything it copies comes from artifacts that already exist -- this
script performs no computation that the notebook then trusts, it only selects and packages.

    python stage_data.py

The bundle is deliberately small (~45 MB). What is shipped, and why that split:

  data/runs/*.hdf5        FULL raw per-rank G for TWO design points (square 4x4 and FeAs, both at
                          the production config beta=5, 64 ranks, at/above the depth floor). These
                          carry every result the notebook recomputes live: the validation ladder,
                          the noise mechanism, the migration scatter, and a single-run R.

  data/ops/*.npz          The serialized point-group operator P and its flat labels ONLY, for the
                          two design points whose raw runs are too large to ship (threeband at
                          77 MB, square 8x8 at 34 MB). P is the authoritative object -- orbit
                          structure, orbit sizes, band-equivalence classes, off-block content and
                          the symmetry-forced nulls are all properties of P alone, so every
                          STRUCTURAL claim stays recomputable for all four design points.

  data/summaries/*.json   The 32-seed ensemble summaries. Every quoted R +/- comes from here.

The division is the honest one: structure is recomputed everywhere, single-run statistics are
recomputed where the raw data is shipped, and ensemble statistics are tabulated -- because no single
run can reproduce a 32-seed interval no matter how much data we ship.

Provenance for each item is recorded in data/MANIFEST.json so the bundle is self-describing.
"""
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SCRATCH = Path("/home/tsax10/dca/scratch")
RUNS = HERE.parent / "symm_variance" / "runs"

# ---------------------------------------------------------------------------------------------
# Full raw runs. One seed each -- the FIRST seed of the ensemble, not a hand-picked one; see the
# note in the notebook where the single-run R is compared against the ensemble mean.
# ---------------------------------------------------------------------------------------------
FULL_RUNS = {
    "square_b5_c4": SCRATCH / "model_sweep/square_b5_c4/square_r64_m2048_s20000000.hdf5",
    "fe_as_b5_c4": SCRATCH / "seed_ensemble/fe_as_r64_m2048_s1000000.hdf5",
}

# Operator-only, for the points whose raw runs are too big to ship.
OPS_ONLY = {
    "threeband_b5_c4": SCRATCH / "model_sweep/threeband_b5_c4/threeband_r64_m2048_s21000000.hdf5",
    "square_b5_c8": SCRATCH / "model_sweep/square_b5_c8/square_r64_m2048_s22000000.hdf5",
}

SUMMARIES = [
    "seed_ensemble_square.json",     # square 4x4 beta=1 -- the cold-start reference rung
    "seed_ensemble_fe_as.json",      # FeAs beta=5 -- the headline multi-orbital ensemble
    "model_sweep.json",              # the three beta=5 design points + class decompositions
    "beta_ladder_square.json",       # beta axis, sign-free throughout
    "beta_ladder_fe_as.json",        # beta axis with the sign channel live
    "m_scaling_summary.json",        # the depth floor
    "bath_drift_square.json",        # the bare-bath justification (appendix)
]


def stage_full_runs(manifest):
    for label, src in FULL_RUNS.items():
        dst = DATA / "runs" / f"{label}.hdf5"
        shutil.copy2(src, dst)
        with h5py.File(dst, "r") as h:
            meta = {
                "n_ranks": int(h["metadata/n_ranks"][()][0]),
                "n_ops": int(h["metadata/n_ops"][()][0]),
                "beta": float(h["metadata/beta"][()][0]) if "metadata/beta" in h else None,
                "seed": int(h["metadata/seed"][()][0]),
                "measurements_per_rank": int(h["metadata/measurements_per_rank"][()][0]),
                "measurements_total": (int(h["metadata/measurements_total"][()][0])
                                       if "metadata/measurements_total" in h else None),
                "warm_up_sweeps": (int(h["metadata/warm_up_sweeps"][()][0])
                                   if "metadata/warm_up_sweeps" in h else None),
                "nb": int(h["symmetrization/nb"][()][0]),
                "nk": int(h["symmetrization/nk"][()][0]),
            }
        manifest["runs"][label] = {
            "file": f"data/runs/{label}.hdf5",
            "source": str(src),
            "size_mb": round(dst.stat().st_size / 1e6, 1),
            **meta,
        }
        print(f"  runs/{label}.hdf5   {dst.stat().st_size/1e6:6.1f} MB   "
              f"beta={meta['beta']} ranks={meta['n_ranks']} nb={meta['nb']} nk={meta['nk']}")


def stage_ops(manifest):
    """Extract P, flat_labels and the metadata that describes them. No sample data."""
    for label, src in OPS_ONLY.items():
        with h5py.File(src, "r") as h:
            P = np.array([np.asarray(row) for row in h["symmetrization/P"][()]])
            labels = np.asarray(h["symmetrization/flat_labels"][()])
            meta = dict(
                nb=int(h["symmetrization/nb"][()][0]),
                nk=int(h["symmetrization/nk"][()][0]),
                n_ops=int(h["symmetrization/n_ops"][()][0]),
                n_ranks=int(h["metadata/n_ranks"][()][0]),
                beta=float(h["metadata/beta"][()][0]) if "metadata/beta" in h else None,
                model=h["metadata/model"][()][0].decode(),
            )
        dst = DATA / "ops" / f"{label}.npz"
        np.savez_compressed(dst, P=P, flat_labels=labels, **{f"meta_{k}": v for k, v in meta.items()})
        manifest["ops"][label] = {
            "file": f"data/ops/{label}.npz",
            "source": str(src),
            "size_kb": round(dst.stat().st_size / 1e3, 1),
            **meta,
        }
        print(f"  ops/{label}.npz     {dst.stat().st_size/1e3:6.1f} KB   "
              f"E={P.shape[0]} nb={meta['nb']} nk={meta['nk']}")


def stage_summaries(manifest):
    for name in SUMMARIES:
        src = RUNS / name
        dst = DATA / "summaries" / name
        shutil.copy2(src, dst)
        manifest["summaries"][name] = {
            "file": f"data/summaries/{name}",
            "source": str(src),
            "size_kb": round(dst.stat().st_size / 1e3, 1),
        }
        print(f"  summaries/{name:32s} {dst.stat().st_size/1e3:6.1f} KB")


if __name__ == "__main__":
    manifest = {"runs": {}, "ops": {}, "summaries": {},
                "note": "Written by stage_data.py. Sources are paths on the machine the "
                        "measurements were made on; they are recorded for provenance, not needed "
                        "to run the notebook."}
    print("staging full raw runs:")
    stage_full_runs(manifest)
    print("staging operator-only points:")
    stage_ops(manifest)
    print("staging ensemble summaries:")
    stage_summaries(manifest)

    (DATA / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file())
    print(f"\nbundle total: {total/1e6:.1f} MB  ->  {DATA}")
