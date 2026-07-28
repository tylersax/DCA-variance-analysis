"""Cross-model sweep aggregation -- ROADMAP task 4.

WHY THIS EXISTS. Every number in this project so far rests on two models (square nb=1, FeAs nb=2)
that differ in beta, band count, interaction AND filling simultaneously. TAKEAWAYS 4a states the
consequence plainly: the mechanism is measured, the causation is not. This module aggregates a sweep
designed to move ONE axis at a time -- band count, orbital equivalence, point group, cluster size --
so the headline claim of 4c ("symmetrization pays off most where the physics is most interesting")
becomes a measurement rather than a comparison of two anecdotes.

WHAT IS NEW HERE, AND WHAT IS DELEGATED. The per-run and per-design-point layers already exist and
are model-agnostic; this module adds only what they cannot do:

  delegated   m_scaling.summarize          per-run everything (R, rho, orbits, sign, mechanism)
              seed_ensemble.across_seed    across-seed statistics, t-intervals, bootstraps
              seed_ensemble.calibration    rank-bootstrap vs across-seed-SD calibration
              beta_ladder.mate_rho         orbit-size-weighted mate correlation
              beta_ladder.sign_channel_check / orbit_rho / omega_flatness
              reduction_map.reduction_table(run, groups=...)   the class decomposition itself

  new here    band_equivalence             which bands the point group maps onto each other,
                                           DERIVED from P -- never hardcoded per model
              band_pair_classes            partitions finer than reduction_map's binary intra/inter,
                                           including the equivalence-aware one that lets a single
                                           threeband run carry both the nb point and the
                                           inequivalent-orbital control
              band_permutation_content     is the band machinery live or dormant? Reproduces
                                           TAKEAWAYS 4c's "104 P entries map between band blocks"
                                           (FeAs) and 0 (nb=1) from P's support alone
              square_mesh                  guard: noise_diagnostics assumes an LxL Cartesian k-mesh
                                           and silently returns nonsense on a hexagonal one
              manifest verification        every design point's metadata checked against a declared
                                           manifest, raising on mismatch

WHY NOT beta_ladder.trend. Its shape is monotonicity plus an endpoint delta over an ORDERED axis.
`model` and `point group` are CATEGORICAL -- a monotone verdict on them is meaningless. Ordered axes
(nk, nb) get a delta with an interval; categorical ones get a contrast table and nothing more.

SERIALIZATION RULE. Anything needing `run.P` or per-entry variances is computed in `build` and
written into the summary JSON; `report` then runs from the JSON alone. The raw HDF5s live in
uncommitted scratch, so a committed artifact that needed them would not be reproducible.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import beta_ladder as bl
import m_scaling as ms
import reduction_map as rm
import seed_ensemble as se
from symm_variance_lib import Run

TOL = 1e-9

# Fields verified against the manifest on every single run. The directory name and the filename are
# conveniences; the file's own metadata is the authority. Same policy as beta_ladder.build, which
# raises rather than grouping when a file's beta disagrees with the directory it sits in.
_CHECKED = ("model", "beta", "nb", "nk", "n_ops", "n_ranks", "m")
# Checked only when the run records them (the driver gained these keys after runs already existed).
_CHECKED_OPTIONAL = ("sweeps_per_measurement", "warm_up_sweeps", "chemical_potential")


# ---- manifest -------------------------------------------------------------------------------------

def load_manifest(path):
    with open(path) as f:
        man = json.load(f)
    defaults = man.get("defaults", {})
    for p in man["points"]:
        for k, v in defaults.items():
            p.setdefault(k, v)
    return man


def validate_manifest(man):
    """Raise on anything that would make the sweep quietly wrong. Runs BEFORE anything launches.

    The seed-range check is the one that matters most: base seeds S and S+1 at 64 ranks share 63/64
    of their walker streams (Gotcha 13), so two points drawing from overlapping ranges are not
    independent replicates -- and nothing downstream can detect it. The runs simply look like
    agreeing replicates and every interval built from them is too narrow.
    """
    problems = []
    labels = [p["label"] for p in man["points"]]
    dupes = {x for x in labels if labels.count(x) > 1}
    if dupes:
        problems.append(f"duplicate point labels: {sorted(dupes)}")

    spans = []
    for p in man["points"]:
        need = ("label", "model", "beta", "nk", "nb", "n_ops", "m", "n_seeds", "seed0", "n_ranks")
        missing = [k for k in need if k not in p]
        if missing:
            problems.append(f"{p.get('label', '?')}: missing fields {missing}")
            continue
        stride = p.get("stride", 10000)
        min_stride = p["n_ranks"] * 64
        if stride < min_stride:
            problems.append(f"{p['label']}: stride {stride} < n_ranks*64 = {min_stride} "
                            f"(walker streams would overlap between seeds)")
        lo = p["seed0"]
        hi = p["seed0"] + p["n_seeds"] * stride
        spans.append((lo, hi, p["label"]))
        if p.get("mesh", "square") == "square" and "L" in p and p["L"] ** 2 != p["nk"]:
            problems.append(f"{p['label']}: L={p['L']} implies nk={p['L']**2}, manifest says {p['nk']}")

    spans.sort()
    for (lo1, hi1, l1), (lo2, hi2, l2) in zip(spans, spans[1:]):
        if lo2 < hi1:
            problems.append(f"seed ranges overlap: {l1} [{lo1},{hi1}) vs {l2} [{lo2},{hi2}) "
                            f"-- these points would share walker streams")

    if problems:
        raise ValueError("manifest is not runnable:\n  " + "\n  ".join(problems))


def point_dir(man, point):
    return os.path.join(man["root"], point["label"])


def _extra_metadata(path):
    """The metadata keys m_scaling.summarize does not surface. Absent keys are omitted, not None."""
    import h5py
    out = {}
    with h5py.File(path, "r") as h:
        for key in _CHECKED_OPTIONAL:
            if f"metadata/{key}" in h:
                out[key] = float(h[f"metadata/{key}"][()][0])
    return out


def verify_point(point, summary, extra):
    """Raise if a run's own metadata disagrees with the design point it is filed under."""
    bad = []
    for key in _CHECKED:
        want = point.get(key)
        got = summary.get(key)
        if want is None or got is None:
            continue
        if isinstance(want, str):
            ok = want == got
        else:
            ok = abs(float(want) - float(got)) < 1e-9
        if not ok:
            bad.append(f"{key}: manifest={want} file={got}")
    for key in _CHECKED_OPTIONAL:
        if key in extra and key in point:
            if abs(float(point[key]) - extra[key]) > 1e-9:
                bad.append(f"{key}: manifest={point[key]} file={extra[key]}")
    if bad:
        raise ValueError(f"{summary.get('path')} filed under point '{point['label']}' but its "
                         f"metadata disagrees:\n    " + "\n    ".join(bad))


# ---- P structure: the genuinely new analysis ------------------------------------------------------

def band_equivalence(run, tol=TOL):
    """Which bands the point group maps onto each other -- derived from P, not declared.

    Two bands b, b' are symmetry-equivalent iff some P-orbit contains BOTH a band-diagonal entry
    (b,b,k) and a band-diagonal entry (b',b',k'). That is exactly the statement that a group element
    carries band b onto band b': the orbit is the set of entries the group identifies.

    Deriving this rather than hardcoding it is what makes the inequivalent-orbital control an
    observation. Expected: square {0}; fe_as {0,1}; threeband {0},{1,2}; kagome {0,1,2}.
    """
    nb = run.nb
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

    lab = run.labels
    for o in run.orbits():
        if o["forced_null"]:
            continue
        diag_bands = {int(lab[x][0]) for x in o["members"] if lab[x][0] == lab[x][1]}
        diag_bands = sorted(diag_bands)
        for b in diag_bands[1:]:
            union(diag_bands[0], b)

    classes = {}
    for b in range(nb):
        classes.setdefault(find(b), []).append(b)
    class_list = [sorted(v) for v in classes.values()]
    class_list.sort()
    class_of = [0] * nb
    for i, members in enumerate(class_list):
        for b in members:
            class_of[b] = i
    return dict(classes=class_list,
                class_of=class_of,
                size_of=[len(class_list[class_of[b]]) for b in range(nb)])


def band_pair_classes(run, mode="equivalence"):
    """A PARTITION of every flat entry, for reduction_map.reduction_table(run, groups=...).

    'forced null' is a class in every mode, so the harmonic reconstruction still reproduces the
    aggregate R exactly -- that self-check is what makes a wrong partition detectable rather than
    silently plausible.

    binary       reduction_map.band_classes verbatim; keeps new points comparable to the committed
                 FeAs intraband/interband table (TAKEAWAYS 4b).
    pair         one class per unordered (b0,b1) band pair. nb=3 -> 6 classes. Finest model-agnostic
                 split; needed because binary intra/inter lumps (0,1) and (0,2) together, and for
                 threeband those are the d-p and d-p' blocks whose whole point is that they differ
                 from the p-p' block.
    equivalence  the 4-way, equivalence-aware partition. THIS is the one that makes a single
                 threeband run answer two questions at once: p-p and p_x-p_y entries land in the
                 'equivalent' classes, d-d and every d-p entry in the singleton / 'inequivalent'
                 classes, and the contrast between their R_C is the inequivalent-orbital control.
    """
    _, null = rm.orbit_info(run)
    lab = run.labels
    if mode == "binary":
        return rm.band_classes(run)

    eq = band_equivalence(run)
    cls = {}
    for x in range(run.E):
        b0, b1 = int(lab[x][0]), int(lab[x][1])
        if null[x]:
            key = "forced null"
        elif mode == "pair":
            a, b = min(b0, b1), max(b0, b1)
            key = f"b{a}{b}"
        elif mode == "equivalence":
            if b0 == b1:
                key = ("diagonal, band-orbit 1" if eq["size_of"][b0] == 1
                       else "diagonal, band-orbit >1")
            elif eq["class_of"][b0] == eq["class_of"][b1]:
                key = "off-diagonal, equivalent"
            else:
                key = "off-diagonal, inequivalent"
        else:
            raise ValueError(f"unknown mode: {mode}")
        cls.setdefault(key, []).append(x)
    return dict(sorted(cls.items()))


def _permutations(nb):
    from itertools import permutations as _p
    return list(_p(range(nb)))


def band_permutation_content(run, tol=TOL):
    """Is the band machinery live or dormant? The diagnostic that makes the qualifier in TAKEAWAYS
    4c ("the gain comes from symmetry-EQUIVALENT orbitals, not orbital count") quantitative.

    Everything is read off P's support -- no statistics, no model knowledge:

      n_offblock       nonzero P[out][in] whose (b0,b1) block differs between out and in. TAKEAWAYS
                       4c quotes 104 for FeAs and 0 at nb=1; both are reproduced exactly.
      n_orbits_mixing  non-null orbits spanning more than one (b0,b1) block. FeAs 10 of 10.
      permutations     band permutations pi actually realized: a nonzero P[out][in] is consistent
                       with pi iff (b0_in, b1_in) == (pi[b0_out], pi[b1_out]). nb <= 3 here, so all
                       <= 6 candidates are enumerated exactly rather than sampled.
      n_permutations   == 1 (identity only) IS the dormant-band-machinery finding. Expected 1 for
                       square (nb=1), 2 for fe_as and threeband, up to 6 for kagome.
    """
    P, lab, nb = run.P, run.labels, run.nb
    nz = np.argwhere(np.abs(P) > tol)
    n_nonzero = int(len(nz))

    n_offblock = 0
    for out, inn in nz:
        if lab[out][0] != lab[inn][0] or lab[out][1] != lab[inn][1]:
            n_offblock += 1

    orbs = run.orbits()
    nonnull = [o for o in orbs if not o["forced_null"]]
    n_orbits_mixing = 0
    for o in nonnull:
        blocks = {(int(lab[x][0]), int(lab[x][1])) for x in o["members"]}
        if len(blocks) > 1:
            n_orbits_mixing += 1

    # Which band permutations the group actually realizes.
    #
    # A per-entry consistency test alone OVER-COUNTS, and the failure is not obvious: a band-diagonal
    # entry (b,b) -> (b',b') only constrains pi at b, so it is "consistent with" every pi sending
    # b -> b' -- for nb=3 that is two permutations per entry, and threeband reported all 6 of S_3
    # including d<->p swaps that no symmetry performs. The fix is to require pi to preserve the
    # derived band-equivalence partition setwise, which is a necessary condition for pi to be induced
    # by a group element: a symmetry can only exchange bands the point group identifies, and
    # band_equivalence measures exactly which those are.
    eq = band_equivalence(run, tol)
    class_of = eq["class_of"]
    perm_counts = {}
    for pi in _permutations(nb):
        if any(class_of[pi[b]] != class_of[b] for b in range(nb)):
            continue  # would move a band out of its equivalence class -- no group element does this
        c = 0
        for out, inn in nz:
            if (lab[inn][0], lab[inn][1]) == (pi[lab[out][0]], pi[lab[out][1]]):
                c += 1
        if c > 0:
            perm_counts["".join(str(i) for i in pi)] = c

    return dict(
        n_nonzero=n_nonzero,
        n_offblock=n_offblock,
        frac_offblock=(n_offblock / n_nonzero) if n_nonzero else 0.0,
        n_orbits=len(nonnull),
        n_orbits_mixing=n_orbits_mixing,
        n_forced_null=int(sum(o["forced_null"] for o in orbs)),
        permutations=dict(sorted(perm_counts.items())),
        n_permutations=len(perm_counts),
    )


def orbit_size_histogram(run):
    """{m: count} over non-null orbits -- the geometry side of r ~ min(m, 1/rho)."""
    h = {}
    for o in run.orbits():
        if o["forced_null"]:
            continue
        m = len(o["members"])
        h[m] = h.get(m, 0) + 1
    return dict(sorted(h.items()))


def p_fingerprint(run):
    """P is deterministic given the design point, so it is computed once -- but VERIFIED per seed.
    A disagreement means two different physical setups were filed under one label."""
    bpc = band_permutation_content(run)
    return dict(n_nonzero=bpc["n_nonzero"], n_offblock=bpc["n_offblock"],
                trace=round(float(np.trace(run.P)), 6),
                hist={str(k): v for k, v in orbit_size_histogram(run).items()})


# ---- mesh guard -----------------------------------------------------------------------------------

def square_mesh(run, tol=1e-6):
    """Is the k-mesh an LxL grid at 2*pi/L spacing on Cartesian axes?

    noise_diagnostics._k_to_grid_index assumes exactly that (L = round(sqrt(nk)), then
    round(c*L/2pi) % L on the Cartesian components) and d4_shells hardcodes the 8 D4 matrices on top.
    On a hexagonal Bravais lattice neither holds: the reciprocal basis is not axis-aligned, so
    several k alias onto the same grid cell while others are never filled. realspace_noise_profile
    and shell_correlations then return finite numbers that mean nothing, and NOTHING RAISES. Hence
    this guard -- the mechanism block is skipped and the omission is recorded, rather than a wrong
    number being reported as a right one.

    R, rho, orbits, the reduction map, w_null and sign health are all mesh-agnostic and unaffected.
    """
    nk = run.nk
    L = int(round(np.sqrt(nk)))
    if L * L != nk:
        return False, f"nk={nk} is not a perfect square"
    ks = np.asarray(run.k_elements, dtype=float)
    if ks.ndim != 2 or ks.shape[1] != 2:
        return False, f"k_elements has shape {ks.shape}, expected (nk, 2)"
    cells = set()
    for c in ks:
        idx = c * L / (2.0 * np.pi)
        if np.abs(idx - np.round(idx)).max() > tol:
            return False, "k-points are not on a 2*pi/L Cartesian grid (non-square Bravais lattice)"
        cells.add(tuple(int(v) % L for v in np.round(idx)))
    if len(cells) != nk:
        return False, f"k-points alias: {len(cells)} distinct cells for {nk} points"
    return True, "square LxL mesh"


def default_blocks(run):
    """Which (b0,b1) blocks the mechanism diagnostics sample.

    m_scaling's own default falls into its nb=2 branch for ANY nb > 1, i.e. it samples only (0,0) and
    (0,1). For threeband that means d-d and d-p_x and never the p-p block where the band permutation
    actually lives -- unrepresentative rather than wrong, but exactly the wrong omission for this
    task. So nb >= 3 gets every unordered band pair.
    """
    if run.nb == 1:
        return {"band-diagonal": (0, 0, 0)}
    if run.nb == 2:
        return {"intraband": (0, 0, 0), "interband": (0, 1, 0)}
    return {f"b{a}{b}": (a, b, 0) for a in range(run.nb) for b in range(a, run.nb)}


# ---- filling --------------------------------------------------------------------------------------

def band_occupancy(run):
    """Per-band occupancy from the finalized G_k_w, with the leading 1/(i*w) tail removed:

        n_b = 1/2 + (1/(beta*nk)) * sum_{k,w} Re[ G_bb(k,iw) - 1/(iw) ]

    Truncation error is O(1/w_max) AFTER the tail subtraction, which is ample to place mu to within
    a few percent filling -- all the scouting gate needs. mu is a free per-model choice here (the
    driver never instantiates DcaLoop, so no chemical-potential adjuster ever runs, Gotcha 11), so
    without this the filling of a new model is an assertion rather than a measurement.

    Returns None if the run predates the `beta` metadata key.
    """
    if run.beta is None:
        return None
    nw, nk, ns, nb = run.nw, run.nk, run.ns, run.nb
    n = np.arange(-(nw // 2), nw - nw // 2)
    w = (2.0 * n + 1.0) * np.pi / run.beta
    tail = 1.0 / (1j * w)                                   # (nw,)
    out = []
    for b in range(nb):
        acc = 0.0
        for s in range(ns):
            g = run.G_final[:, :, s, b, s, b]               # (nw, nk)
            acc += (g - tail[:, None]).real.sum()
        # acc sums both spins; per-spin occupancy is the average, total per band is the sum.
        n_b = ns * 0.5 + acc / (run.beta * nk)
        out.append(float(n_b))
    return dict(per_band=out, total=float(sum(out)), n_bands=nb, filled_fraction=float(sum(out) / (ns * nb)))


# ---- assembly -------------------------------------------------------------------------------------

def summarize_point(man, point, n_boot=400, verbose=True):
    """All seeds of one design point, verified, plus everything that needs the raw P/variances."""
    d = point_dir(man, point)
    paths = sorted(p for p in glob.glob(os.path.join(d, "*.hdf5")) if ms.parse_name(p))
    if not paths:
        if verbose:
            print(f"[model_sweep] {point['label']}: no runs in {d}")
        return dict(point=point, dirpath=d, runs=[], structure=None, by_class=None, paths=[])

    runs, fingerprints = [], []
    structure, by_class, mesh_note = None, None, None
    for i, p in enumerate(paths):
        run = Run(p)
        blocks = point.get("blocks") or default_blocks(run)
        blocks = {k: tuple(v) for k, v in blocks.items()}
        ok, why = square_mesh(run)
        if not ok:
            blocks = {}
            mesh_note = why
        # summarize(blocks={}) yields mechanism={} without ever entering noise_diagnostics -- which
        # is exactly what a non-square mesh needs. Only blocks=None triggers its internal default.
        s = ms.summarize(p, n_boot, blocks=blocks)
        if not blocks:
            s["mechanism_skipped"] = why
        verify_point(point, s, _extra_metadata(p))
        runs.append(s)
        fingerprints.append(p_fingerprint(run))
        if i == 0:
            structure = dict(
                band_equivalence=band_equivalence(run),
                permutation=band_permutation_content(run),
                orbit_sizes={str(k): v for k, v in orbit_size_histogram(run).items()},
                mesh=dict(square=ok, note=why),
                occupancy=band_occupancy(run),
            )
            by_class = {}
            for mode in ("binary", "pair", "equivalence"):
                rows_c, recon, agg = rm.reduction_table(run, band_pair_classes(run, mode))
                by_class[mode] = dict(
                    rows=[dict(cls=r["cls"], n=r["n"], w=r["w"], R_C=r["R_C"],
                               mean_m=r["mean_m"]) for r in rows_c],
                    harmonic_reconstruction=recon, aggregate_R=agg,
                    reconstruction_gap=abs(recon - agg) if np.isfinite(recon) else None)
        if verbose:
            print(f"[model_sweep] {point['label']}: {os.path.basename(p)}  "
                  f"R={s.get('R_full')}")

    # P is deterministic per design point; a disagreement means two setups under one label.
    base = fingerprints[0]
    for p, fp in zip(paths, fingerprints):
        if fp != base:
            raise ValueError(f"{p}: P fingerprint differs from the first run of point "
                             f"'{point['label']}'\n  first={base}\n  this ={fp}")

    return dict(point=point, dirpath=d, runs=runs, structure=structure, by_class=by_class,
                paths=paths, mesh_note=mesh_note)


def build(manifest_path, n_boot=400, verbose=True):
    man = load_manifest(manifest_path)
    validate_manifest(man)
    points = [summarize_point(man, p, n_boot, verbose) for p in man["points"]]
    return dict(manifest=man, root=man["root"], points=points)


def _stat(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return se.across_seed(vals) if vals else None


def rows(summary, n_boot=4000):
    """One quotable row per design point. Delegates every statistic; adds the structural columns."""
    out = []
    for pt in summary["points"]:
        runs = [r for r in pt["runs"] if r.get("usable")]
        if not runs:
            continue
        point = pt["point"]
        Rf = [r["R_full"] for r in runs]
        row = dict(
            label=point["label"], model=point["model"], role=point.get("role"),
            beta=point["beta"], nb=point["nb"], nk=point["nk"], n_ops=point["n_ops"],
            n_ranks=point["n_ranks"], m=point["m"], n_seeds=len(runs),
            R_full=_stat(Rf),
            R_full_boot=se.seed_bootstrap(Rf, n_boot=n_boot) if len(Rf) > 2 else None,
            R_nonnull=_stat([r["R_nonnull"] for r in runs]),
            w_null=_stat([se.w_null(r) for r in runs]),
            efficiency=_stat([r["efficiency"] for r in runs]),
            rho=_stat([bl.mate_rho(r) for r in runs]),
            sign=bl.sign_channel_check(runs),
            calibration=se.calibration(runs) if len(runs) > 2 else None,
            orbits=bl.orbit_rho(runs),
            structure=pt["structure"],
            by_class=pt["by_class"],
            seed_spacing_violations=se.check_seed_spacing(runs),
            contamination_worst=max((se.contamination(r) for r in runs),
                                    key=lambda c: c.get("outlier") or 0, default=None),
        )
        # 1/rho against the m-ceiling: the one-run diagnostic that says whether to chase colder
        # physics (rho-limited) or bigger orbits (m-limited). TAKEAWAYS 3.
        rho_mean = row["rho"]["mean"] if row["rho"] else None
        if rho_mean and rho_mean > 0:
            row["inv_rho"] = 1.0 / rho_mean
            ideal = _stat([r["R_ideal"] for r in runs])
            row["m_ceiling"] = ideal["mean"] if ideal else None
            if row["m_ceiling"]:
                row["binding"] = "rho" if row["inv_rho"] < row["m_ceiling"] else "m"
        # omega flatness needs the raw HDF5; degrade rather than fail on a committed-only summary.
        paths = pt.get("paths") or []
        row["omega"] = (bl.omega_flatness(paths[:4])
                        if paths and all(os.path.exists(p) for p in paths[:4]) else None)
        out.append(row)
    return out


def axis_pairs(rows_, axis, keys=("model", "beta", "nb", "nk", "n_ops")):
    """Point pairs differing in EXACTLY `axis` and matching on every other key.

    This enforces the roadmap's one-axis-at-a-time design constraint mechanically rather than by eye.
    Ordered axes (nk, nb) get a delta with a 95% interval built by resampling each side's across-seed
    sampling distribution INDEPENDENTLY -- the points are independent runs, so the errors add in
    quadrature. Categorical axes get the contrast and no monotonicity verdict, which is why
    beta_ladder.trend is not reused here.
    """
    others = [k for k in keys if k != axis]
    pairs = []
    for i, a in enumerate(rows_):
        for b in rows_[i + 1:]:
            if a.get(axis) == b.get(axis):
                continue
            if any(a.get(k) != b.get(k) for k in others):
                continue
            entry = dict(axis=axis, a=a["label"], b=b["label"],
                         a_value=a.get(axis), b_value=b.get(axis),
                         held_fixed={k: a.get(k) for k in others})
            for key in ("R_full", "rho", "w_null"):
                sa, sb = a.get(key), b.get(key)
                if not sa or not sb:
                    entry[key] = None
                    continue
                d = sb["mean"] - sa["mean"]
                sem = float(np.hypot(sa["sem"], sb["sem"]))
                df = max(1, min(sa["n"], sb["n"]) - 1)
                half = se.t975(df) * sem
                entry[key] = dict(delta=d, sem=sem, lo=d - half, hi=d + half,
                                  resolved=bool(abs(d) > half))
            pairs.append(entry)
    return pairs


# ---- tables ---------------------------------------------------------------------------------------

def _fmt(stat, prec=4):
    if not stat:
        return "n/a"
    return f"{stat['mean']:.{prec}f} +/- {stat['sem']:.{prec}f}"


def point_table(rows_):
    out = [f"  {'point':<20} {'nb':>3} {'|G|':>4} {'nk':>4} {'seeds':>6} "
           f"{'R_full':>20} {'R_nonnull':>20} {'w_null':>16} {'rho':>16} {'<s>min':>8}",
           "  " + "-" * 130]
    for r in rows_:
        s = r["sign"]
        out.append(f"  {r['label']:<20} {r['nb']:>3} {r['n_ops']:>4} {r['nk']:>4} {r['n_seeds']:>6} "
                   f"{_fmt(r['R_full']):>20} {_fmt(r['R_nonnull']):>20} "
                   f"{_fmt(r['w_null'], 4):>16} {_fmt(r['rho'], 4):>16} "
                   f"{s.get('min_mean_sign', float('nan')):>8.3f}")
    return "\n".join(out)


def permutation_table(rows_):
    """Is the band machinery live? n_permutations == 1 means dormant."""
    out = [f"  {'point':<20} {'nb':>3} {'nonzero P':>10} {'off-block':>10} {'frac':>7} "
           f"{'orbits':>7} {'mixing':>7} {'nulls':>6} {'perms':>6}  realized",
           "  " + "-" * 110]
    for r in rows_:
        st = r.get("structure") or {}
        p = st.get("permutation")
        if not p:
            continue
        eq = st.get("band_equivalence", {}).get("classes")
        out.append(f"  {r['label']:<20} {r['nb']:>3} {p['n_nonzero']:>10} {p['n_offblock']:>10} "
                   f"{p['frac_offblock']:>7.3f} {p['n_orbits']:>7} {p['n_orbits_mixing']:>7} "
                   f"{p['n_forced_null']:>6} {p['n_permutations']:>6}  "
                   f"{sorted(p['permutations'])}  band classes={eq}")
    return "\n".join(out)


def class_table(rows_, mode="equivalence"):
    out = []
    for r in rows_:
        bc = (r.get("by_class") or {}).get(mode)
        if not bc:
            continue
        out.append(f"  -- {r['label']}  ({mode}) --")
        out.append(f"  {'class':<30} {'n':>4} {'var share':>10} {'R_C':>10} {'mean m':>7}")
        for row in bc["rows"]:
            rc = row["R_C"]
            rc_s = "inf" if not np.isfinite(rc) else f"{rc:.3f}"
            out.append(f"  {row['cls']:<30} {row['n']:>4} {row['w']:>10.4f} {rc_s:>10} "
                       f"{row['mean_m']:>7.2f}")
        gap = bc.get("reconstruction_gap")
        out.append(f"  aggregate R = {bc['aggregate_R']:.4f}   harmonic reconstruction = "
                   f"{bc['harmonic_reconstruction']:.4f}"
                   + (f"   (gap {gap:.2e})" if gap is not None else ""))
        out.append("")
    return "\n".join(out)


def orbit_size_table(rows_):
    out = [f"  {'point':<20} orbit-size histogram (non-null)", "  " + "-" * 60]
    for r in rows_:
        st = r.get("structure") or {}
        h = st.get("orbit_sizes")
        if h:
            out.append(f"  {r['label']:<20} " + "  ".join(f"m={k}: {v}" for k, v in sorted(h.items())))
    return "\n".join(out)


def binding_table(rows_):
    """Which ceiling binds -- the one-run diagnostic of TAKEAWAYS 3."""
    out = [f"  {'point':<20} {'1/rho':>8} {'m-ceiling':>10} {'binding':>8}  prescription",
           "  " + "-" * 80]
    for r in rows_:
        if "binding" not in r:
            continue
        pres = ("chase temperature (colder physics still pays)" if r["binding"] == "rho"
                else "chase orbit size (bigger cluster / group / equivalent bands)")
        out.append(f"  {r['label']:<20} {r['inv_rho']:>8.2f} {r['m_ceiling']:>10.2f} "
                   f"{r['binding']:>8}  {pres}")
    return "\n".join(out)


def axis_table(pairs):
    out = [f"  {'axis':<8} {'from':<20} {'to':<20} {'dR':>22} {'drho':>22}", "  " + "-" * 96]
    for p in pairs:
        def s(key):
            v = p.get(key)
            if not v:
                return "n/a"
            mark = "*" if v["resolved"] else " "
            return f"{v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]{mark}"
        out.append(f"  {p['axis']:<8} {p['a']:<20} {p['b']:<20} {s('R_full'):>22} {s('rho'):>22}")
    out.append("  (* = resolved against across-seed scatter)")
    return "\n".join(out)


def report(summary, n_boot=4000):
    rows_ = rows(summary, n_boot)
    parts = ["=== model sweep: design points ===", point_table(rows_), "",
             "=== band-permutation content (is the band machinery live?) ===", permutation_table(rows_), "",
             "=== orbit sizes ===", orbit_size_table(rows_), "",
             "=== which ceiling binds ===", binding_table(rows_), "",
             "=== class decomposition (equivalence-aware) ===", class_table(rows_, "equivalence"),
             "=== class decomposition (per band pair) ===", class_table(rows_, "pair")]
    for axis in ("nb", "nk", "n_ops", "model"):
        pairs = axis_pairs(rows_, axis)
        if pairs:
            parts += [f"=== axis: {axis} ===", axis_table(pairs), ""]
    viol = [v for r in rows_ for v in (r.get("seed_spacing_violations") or [])]
    if viol:
        parts += ["!!! SEED SPACING VIOLATIONS -- intervals are void !!!", str(viol), ""]
    return "\n".join(parts), rows_


# ---- planning: emit the run commands ---------------------------------------------------------------

def plan(man, mode):
    """One fully-formed command per line. JSON parsing stays in python, launching stays in bash."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lines = []
    for p in man["points"]:
        if p.get("reference"):
            continue
        env = f"BETA={p['beta']}"
        if "L" in p:
            env += f" CLUSTER={p['L']}"
        if p.get("nw"):
            env += f" NW={p['nw']}"
        stride = p.get("stride", 10000)
        if mode == "scout":
            d = os.path.join(man["root"] + "_scout", p["label"])
            lines.append(f"{env} {here}/run_m_ladder.sh {p['model']} 16 {d} 512 {p['seed0'] + 7}")
        elif mode == "floor":
            d = os.path.join(man["root"], p["label"] + "_floor")
            ml = p.get("floor_ladder", "512,2048,8192")
            seeds = ",".join(str(p["seed0"] + 1 + i) for i in range(3))
            lines.append(f"{env} {here}/run_m_ladder.sh {p['model']} {p['n_ranks']} {d} {ml} {seeds}")
        elif mode == "ensemble":
            d = point_dir(man, p)
            lines.append(f"{env} {here}/run_seed_ensemble.sh {p['model']} {p['n_ranks']} {p['m']} "
                         f"{p['n_seeds']} {d} {p['seed0']} {stride}")
        else:
            raise ValueError(f"unknown mode: {mode}")
    return lines


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("usage: model_sweep.py build  <manifest.json> --out <summary.json> [--boot N] [--slim]\n"
              "       model_sweep.py report <summary.json> [--mode equivalence]\n"
              "       model_sweep.py plan   <manifest.json> --mode {scout|floor|ensemble}")
        return 1
    cmd = argv[1]
    if cmd == "build":
        path = argv[2]
        n_boot = int(argv[argv.index("--boot") + 1]) if "--boot" in argv else 400
        summary = build(path, n_boot)
        text, rows_ = report(summary, n_boot=4000)
        print(text)
        summary["report"] = text
        summary["rows"] = rows_
        if "--slim" in argv:
            for pt in summary["points"]:
                for r in pt["runs"]:
                    r.pop("orbits", None)
        if "--out" in argv:
            out = argv[argv.index("--out") + 1]
            with open(out, "w") as f:
                json.dump(summary, f, indent=1, default=float)
            print(f"\nwrote {out}")
    elif cmd == "report":
        with open(argv[2]) as f:
            summary = json.load(f)
        mode = argv[argv.index("--mode") + 1] if "--mode" in argv else "equivalence"
        text, _ = report(summary)
        print(text)
    elif cmd == "plan":
        man = load_manifest(argv[2])
        validate_manifest(man)
        mode = argv[argv.index("--mode") + 1] if "--mode" in argv else "ensemble"
        for line in plan(man, mode):
            print(line)
    else:
        print(f"unknown command: {cmd}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
