#!/usr/bin/env bash
# Multi-rank symmetrization-variance run: rank == one sample, one QMC iteration, raw per-rank G
# dumped before finalize (see symm_variance_main.inc). Unlike the deprecated variance_demo (which ran
# many single-rank replica processes), this uses ONE mpirun with n_ranks == number of samples; the
# per-rank RNG streams are independent by construction (seed = hash(global_id + base_seed)), so a
# single base seed is correct and there is no cross-run collision hazard.
#
# DEPTH SEMANTICS: <measurements> here is PER RANK, which is the axis that matters -- a rank is one
# sample, so its depth sets whether that sample is in the CLT regime. DCA's input key of the same
# name is the TOTAL over ranks (every solver routes it through parallel::util::getWorkload, and
# mci_parameters.hpp computes `local_meas = measurements/mpi_size`), so this script multiplies up.
# Runs made before 2026-07-27 passed the total here and were therefore n_ranks times shallower than
# their filenames suggest -- the committed 16-rank "4000" runs are 250 measurements per rank.
#
# BETA (env var, optional): overrides the template's "beta". This is the swept axis of the beta
# ladder (ROADMAP task 2); putting it here rather than in a separate script means run_m_ladder.sh and
# run_seed_ensemble.sh -- which both call this one -- become beta-aware for free. Unset means "use
# the template's beta", so every existing call site is unaffected.
#
# SWEEPS_PER_MEAS / WARMUP (env vars, optional): same mechanism, for the two knobs that decide
# whether a rank's measurements are actually independent draws. Both matter once beta is swept,
# because autocorrelation AND thermalization times grow with beta while the template's values (2 and
# 200) are fixed. A depth floor quoted in measurements is only meaningful at a stated
# sweeps-per-measurement -- raising it decorrelates consecutive measurements, so it trades wall-clock
# for effective sample size rather than just buying more of the same correlated samples.
#
# usage: run_symm_variance.sh <square|fe_as|threeband|kagome> <n_ranks> <measurements_per_rank> [seed] [outdir]
set -euo pipefail

MODEL=${1:?usage: run_symm_variance.sh <square|fe_as|threeband|kagome> <n_ranks> <measurements_per_rank> [seed] [outdir]}
NRANKS=${2:?need n_ranks (== number of samples)}
MEAS=${3:?need measurements PER RANK}
SEED=${4:-42}
OUTDIR=${5:-runs}

# Total handed to DCA. Kept an exact multiple so getWorkload gives every rank the same depth; unequal
# depth would make the per-rank samples non-identically-distributed and bias the variance estimate.
MEAS_TOTAL=$(( MEAS * NRANKS ))

# Cap the BLAS thread pool: each rank already spawns walker+accumulator threads; without this every
# rank would also open an nproc-wide BLAS pool and the box thrashes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR=${BIN_DIR:-/home/tsax10/dca/build_symm}
MPIRUN=${MPIRUN:-/home/tsax10/conda/envs/qe/bin/mpirun}

BIN="${BIN_DIR}/${MODEL}_symm_variance"
TEMPLATE="${HERE}/${MODEL}_input.template.json"
[[ -x "$BIN" ]] || { echo "missing binary: $BIN (build first, or set BIN_DIR)"; exit 1; }
[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE"; exit 1; }

mkdir -p "$OUTDIR"
INPUT="${OUTDIR}/${MODEL}_input.json"
OUT="${OUTDIR}/${MODEL}.hdf5"
sed -e "s/__SEED__/${SEED}/" -e "s/__MEASUREMENTS__/${MEAS_TOTAL}/" "$TEMPLATE" > "$INPUT"
if [[ -n "${BETA:-}" ]]; then
  # Rewrite the physics beta in place. The driver records parameters.get_beta() into the HDF5
  # metadata, so the written file -- not this substitution -- is what the analysis trusts.
  sed -i -E "s/(\"beta\"[[:space:]]*:[[:space:]]*)[0-9.]+/\1${BETA}/" "$INPUT"
  grep -q "\"beta\"[[:space:]]*:[[:space:]]*${BETA}" "$INPUT" \
    || { echo "[run] BETA=${BETA} substitution failed in ${INPUT}"; exit 1; }
fi
# Each substitution is verified rather than assumed: sed reports success even when it matched
# nothing, so a renamed key would silently leave the template's value in place and the run would be
# mislabeled in exactly the way that produces a wrong number with no error.
if [[ -n "${SWEEPS_PER_MEAS:-}" ]]; then
  sed -i -E "s/(\"sweeps-per-measurement\"[[:space:]]*:[[:space:]]*)[0-9.]+/\1${SWEEPS_PER_MEAS}/" "$INPUT"
  grep -q "\"sweeps-per-measurement\"[[:space:]]*:[[:space:]]*${SWEEPS_PER_MEAS}" "$INPUT" \
    || { echo "[run] SWEEPS_PER_MEAS=${SWEEPS_PER_MEAS} substitution failed in ${INPUT}"; exit 1; }
fi
if [[ -n "${WARMUP:-}" ]]; then
  sed -i -E "s/(\"warm-up-sweeps\"[[:space:]]*:[[:space:]]*)[0-9.]+/\1${WARMUP}/" "$INPUT"
  grep -q "\"warm-up-sweeps\"[[:space:]]*:[[:space:]]*${WARMUP}" "$INPUT" \
    || { echo "[run] WARMUP=${WARMUP} substitution failed in ${INPUT}"; exit 1; }
fi
# MU (env var, optional): overrides "chemical-potential". This driver never instantiates DcaLoop, so
# DCA's chemical-potential adjuster never runs (Gotcha 11) and mu is a free per-model choice that the
# input alone decides. For a NEW model that makes mu a parameter to be measured rather than assumed:
# model_sweep.band_occupancy reads the resulting filling back out of the finalized G, so a scan over
# this variable is how a model's operating point gets chosen. Negative values are permitted.
if [[ -n "${MU:-}" ]]; then
  sed -i -E "s/(\"chemical-potential\"[[:space:]]*:[[:space:]]*)-?[0-9.]+/\1${MU}/" "$INPUT"
  grep -q "\"chemical-potential\"[[:space:]]*:[[:space:]]*${MU}" "$INPUT" \
    || { echo "[run] MU=${MU} substitution failed in ${INPUT}"; exit 1; }
fi
# EP_P (env var, optional): threeband only -- the p-orbital site energy, i.e. the d-p splitting that
# decides whether the p orbitals sit near the Fermi level at all. Exposed for the same reason as MU:
# at the repo test's ep_p=3 with mu=0 the p orbitals are nearly empty and carry ~0% of the raw
# variance, which would make the equivalent-orbital block -- the whole point of the threeband run --
# unmeasurable. Symmetry is untouched by it: the derive path gates on H0's STRUCTURE, and ep_p enters
# both p diagonals identically, so p_x <-> p_y remains an exact symmetry at any value.
if [[ -n "${EP_P:-}" ]]; then
  sed -i -E "s/(\"ep_p\"[[:space:]]*:[[:space:]]*)-?[0-9.]+/\1${EP_P}/" "$INPUT"
  grep -q "\"ep_p\"[[:space:]]*:[[:space:]]*${EP_P}" "$INPUT" \
    || { echo "[run] EP_P=${EP_P} substitution failed in ${INPUT}"; exit 1; }
fi
# U_PP (env var, optional): threeband only. At U_pp = 0 (standard Emery, the repo-test value) the
# |H_int| > 1e-3 filter in general_interaction.hpp leaves the p orbitals with NO CT-AUX vertices, so
# their noise arrives only through hybridization with d. Setting it > 0 puts vertices on all three
# bands. Symmetry is unaffected either way -- H_int does not enter the point-group derivation at all.
# WARNING: never pair U_pp = 0 with "interacting-orbitals": [1,2] -- correlated_orbitals comes out
# empty and general_interaction.hpp:79 indexes it unguarded, which is a SEGFAULT, not an error.
if [[ -n "${U_PP:-}" ]]; then
  sed -i -E "s/(\"U_pp\"[[:space:]]*:[[:space:]]*)-?[0-9.]+/\1${U_PP}/" "$INPUT"
  grep -q "\"U_pp\"[[:space:]]*:[[:space:]]*${U_PP}" "$INPUT" \
    || { echo "[run] U_PP=${U_PP} substitution failed in ${INPUT}"; exit 1; }
fi
# CLUSTER (env var, optional): rewrites the superlattice basis to the LxL square [[L,0],[0,L]]. This
# is the orbit-size axis of ROADMAP task 4 -- 4x4 -> 8x8 quadruples nk and is the first mesh carrying
# a FREE orbit-8 k-point (on a 4x4 every k lies on a mirror line, so no free orbit exists). Doing it
# here rather than in a per-size template means the cluster axis needs no new model target and no new
# template, and run_m_ladder/run_seed_ensemble/run_beta_ladder inherit it for free.
#
# There is no Nc or k-point key in DCA: the count is |det(basis)|, emergent from this matrix alone.
# The REALIZED nk is recorded in the HDF5 (symmetrization/nk) and the analysis verifies against that,
# never against this variable or a directory name.
if [[ -n "${CLUSTER:-}" ]]; then
  [[ "$CLUSTER" =~ ^[0-9]+$ ]] || { echo "[run] CLUSTER must be an integer L (the LxL basis)"; exit 1; }
  # -z slurps the whole file so [[:space:]] can span newlines: fe_as's template writes the basis on
  # one line and square's over two, and the axis is most valuable on square. Without -z the square
  # substitution silently matches nothing -- sed still exits 0 -- and only the verify below catches it.
  sed -z -i -E \
    "s/(\"cluster\"[[:space:]]*:[[:space:]]*)\[\[[0-9]+,[[:space:]]*[0-9]+\],[[:space:]]*\[[0-9]+,[[:space:]]*[0-9]+\]\]/\1[[${CLUSTER}, 0], [0, ${CLUSTER}]]/" \
    "$INPUT"
  # Verified, not assumed. Whitespace is collapsed first so this one check covers both template
  # layouts. A silent miss here would mislabel the run's cluster size, which is exactly the class of
  # failure that yields a wrong number with no error anywhere.
  tr -d '[:space:]' < "$INPUT" | grep -q "\"cluster\":\[\[${CLUSTER},0\],\[0,${CLUSTER}\]\]" \
    || { echo "[run] CLUSTER=${CLUSTER} substitution failed in ${INPUT}"; exit 1; }
fi
# NW (env var, optional): moves the fermionic-frequency count and the imaginary-time interval count
# TOGETHER. The committed templates disagree -- square is 128/128, fe_as 64/64 -- so a cross-model
# comparison would otherwise carry that difference as an uncontrolled second axis. Setting NW=64
# everywhere makes the band-count axis exactly one axis. (The omega-flatness control says r is flat
# in frequency to machine precision, so this is belt-and-braces rather than a correction.)
if [[ -n "${NW:-}" ]]; then
  sed -i -E "s/(\"sp-fermionic-frequencies\"[[:space:]]*:[[:space:]]*)[0-9]+/\1${NW}/" "$INPUT"
  grep -q "\"sp-fermionic-frequencies\"[[:space:]]*:[[:space:]]*${NW}" "$INPUT" \
    || { echo "[run] NW=${NW} substitution failed in ${INPUT}"; exit 1; }
  sed -i -E "s/(\"sp-time-intervals\"[[:space:]]*:[[:space:]]*)[0-9]+/\1${NW}/" "$INPUT"
  grep -q "\"sp-time-intervals\"[[:space:]]*:[[:space:]]*${NW}" "$INPUT" \
    || { echo "[run] NW=${NW} (sp-time-intervals) substitution failed in ${INPUT}"; exit 1; }
fi

echo "[run] ${MODEL}: ${NRANKS} ranks x ${MEAS} meas/rank (total ${MEAS_TOTAL}), seed ${SEED}${BETA:+, beta ${BETA}}${CLUSTER:+, cluster ${CLUSTER}x${CLUSTER}}${NW:+, nw ${NW}} -> ${OUT}"
# --bind-to none: each rank is itself multi-threaded (walkers + accumulators), so do not pin a rank
# to a single core.
"$MPIRUN" -n "$NRANKS" --bind-to none "$BIN" "$INPUT" "$OUT"
echo "[run] done: ${OUT}"
