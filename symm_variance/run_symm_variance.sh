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
# usage: run_symm_variance.sh <square|fe_as> <n_ranks> <measurements_per_rank> [seed] [outdir]
set -euo pipefail

MODEL=${1:?usage: run_symm_variance.sh <square|fe_as> <n_ranks> <measurements_per_rank> [seed] [outdir]}
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

echo "[run] ${MODEL}: ${NRANKS} ranks x ${MEAS} meas/rank (total ${MEAS_TOTAL}), seed ${SEED}${BETA:+, beta ${BETA}} -> ${OUT}"
# --bind-to none: each rank is itself multi-threaded (walkers + accumulators), so do not pin a rank
# to a single core.
"$MPIRUN" -n "$NRANKS" --bind-to none "$BIN" "$INPUT" "$OUT"
echo "[run] done: ${OUT}"
