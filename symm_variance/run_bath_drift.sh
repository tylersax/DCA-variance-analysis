#!/usr/bin/env bash
# Bath-drift check (ROADMAP 3e): run the REAL DcaLoop for N iterations and dump the raw per-rank G at
# every one, so R can be compared at iteration 1 vs iteration N *paired within seed*.
#
# WHY THIS EXISTS. Every other number in this project is measured at the bare bath (iteration 1 of a
# cold-started loop). A production run performs ~N solver calls at similar cost each, on a bath
# rebuilt from the previous iteration's Sigma, so a total-compute claim depends on the cost-weighted
# average of R over those iterations -- the error in quoting the bare-bath number is ~(N-1)/N x drift.
#
# HOW THE INPUT DIFFERS from <model>_input.template.json, all of it deliberate:
#   iterations                 1 -> N            the axis
#   self-energy-mixing-factor  0. -> 1.          MANDATORY. mix_self_energy() computes
#                                                Sigma = a*Sigma_new + (1-a)*Sigma_cluster, so a=0
#                                                DISCARDS the measured Sigma and the bath never
#                                                moves -- a silent flat result. The driver refuses
#                                                a <= 0 rather than trust the input. a=1 (DCA's own
#                                                default) is the undamped update, which reaches the
#                                                self-consistent bath fastest in a fixed iteration
#                                                budget and so is the STRONGEST drift test; the
#                                                damped 0.75-0.8 production values walk a slower path
#                                                toward the same fixed point.
#   dca-accuracy               (unset) -> 0.     explicit: the loop breaks early when
#                                                L2_Sigma_difference < dca-accuracy, and we want all
#                                                N iterations every time so seeds stay comparable.
#   output.directory           (unset) -> outdir keeps the loop's own dca_loop.hdf5 out of cwd.
# Unchanged and pinned by Conventions: warm-up-sweeps 200, sweeps-per-measurement 2, cluster, domains.
#
# FeAs only: the single-iteration template carries "adjust-chemichal-potential" (misspelled, Gotcha
# 11). That typo is INERT for the single-iteration driver, which never instantiates the adjuster --
# but this driver runs the real loop, which does, so the misspelling would silently turn mu
# adjustment ON and move the filling. The bath-drift template spells it correctly.
#
# DEPTH SEMANTICS: <measurements_per_rank> is PER RANK, as everywhere else in this tree; DCA's input
# key of the same name is the TOTAL over ranks, so this script multiplies up. A scalar is broadcast
# to every DCA iteration by MciParameters::solveDcaIterationConflict, so all N iterations run at
# identical depth -- which is what makes iteration k comparable to iteration 1.
#
# usage: run_bath_drift.sh <square|fe_as> <n_ranks> <measurements_per_rank> <n_iterations> \
#                          <n_seeds> <outroot> [seed0] [stride]
set -euo pipefail

MODEL=${1:?usage: run_bath_drift.sh <square|fe_as> <n_ranks> <meas_per_rank> <n_iterations> <n_seeds> <outroot> [seed0] [stride]}
NRANKS=${2:?need n_ranks}
MEAS=${3:?need measurements PER RANK}
NITER=${4:?need n_iterations}
NSEEDS=${5:?need n_seeds}
OUTROOT=${6:?need outroot}
SEED0=${7:-1000}
# Base seeds must be spaced >= n_ranks * n_walkers apart: a walker's stream is
# hash(global_id + base_seed) with global_id = local_id*n_ranks + proc_id, so a run occupies the
# contiguous key range [S, S + n_ranks*n_walkers) and consecutive base seeds REPLAY each other's
# chains (Gotcha 13). Default stride leaves headroom over 64 ranks x 2 walkers.
STRIDE=${8:-100000}

MEAS_TOTAL=$(( MEAS * NRANKS ))

MIN_STRIDE=$(( NRANKS * 2 ))
if (( STRIDE < MIN_STRIDE )); then
  echo "[drift] ERROR: stride ${STRIDE} < n_ranks*n_walkers = ${MIN_STRIDE}; base seeds would share walker streams"
  exit 1
fi

# Cap the BLAS thread pool: each rank already spawns walker+accumulator threads; without this every
# rank would also open an nproc-wide BLAS pool and the box thrashes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR=${BIN_DIR:-/home/tsax10/dca/build_bath_drift}
MPIRUN=${MPIRUN:-/home/tsax10/conda/envs/qe/bin/mpirun}

BIN="${BIN_DIR}/${MODEL}_bath_drift"
TEMPLATE="${HERE}/${MODEL}_bath_drift_input.template.json"
[[ -x "$BIN" ]] || { echo "missing binary: $BIN (build first, or set BIN_DIR)"; exit 1; }
[[ -f "$TEMPLATE" ]] || { echo "missing template: $TEMPLATE"; exit 1; }

mkdir -p "$OUTROOT"
echo "[drift] ${MODEL}: ${NSEEDS} seeds x ${NITER} iterations, ${NRANKS} ranks x ${MEAS} meas/rank${BETA:+, beta ${BETA}} -> ${OUTROOT}"
START_ALL=$(date +%s)

for (( i = 0; i < NSEEDS; ++i )); do
  SEED=$(( SEED0 + i * STRIDE ))
  DIR="${OUTROOT}/seed_${SEED}"
  mkdir -p "$DIR"
  INPUT="${DIR}/${MODEL}_input.json"
  STEM="${DIR}/${MODEL}"

  sed -e "s/__SEED__/${SEED}/" \
      -e "s/__MEASUREMENTS__/${MEAS_TOTAL}/" \
      -e "s/__ITERATIONS__/${NITER}/" \
      -e "s|__OUTDIR__|${DIR}|" "$TEMPLATE" > "$INPUT"

  # Every substitution is verified rather than assumed: sed exits 0 even when it matched nothing, so
  # a renamed key would silently leave the placeholder (or the template default) in place and the run
  # would be mislabeled in exactly the way that produces a wrong number with no error.
  grep -q "__" "$INPUT" && { echo "[drift] unsubstituted placeholder left in ${INPUT}"; exit 1; }
  grep -q "\"iterations\"[[:space:]]*:[[:space:]]*${NITER}" "$INPUT" \
    || { echo "[drift] ITERATIONS=${NITER} substitution failed in ${INPUT}"; exit 1; }

  if [[ -n "${BETA:-}" ]]; then
    sed -i -E "s/(\"beta\"[[:space:]]*:[[:space:]]*)[0-9.]+/\1${BETA}/" "$INPUT"
    grep -q "\"beta\"[[:space:]]*:[[:space:]]*${BETA}" "$INPUT" \
      || { echo "[drift] BETA=${BETA} substitution failed in ${INPUT}"; exit 1; }
  fi
  if [[ -n "${WARMUP:-}" ]]; then
    sed -i -E "s/(\"warm-up-sweeps\"[[:space:]]*:[[:space:]]*)[0-9.]+/\1${WARMUP}/" "$INPUT"
    grep -q "\"warm-up-sweeps\"[[:space:]]*:[[:space:]]*${WARMUP}" "$INPUT" \
      || { echo "[drift] WARMUP=${WARMUP} substitution failed in ${INPUT}"; exit 1; }
  fi
  if [[ -n "${SWEEPS_PER_MEAS:-}" ]]; then
    sed -i -E "s/(\"sweeps-per-measurement\"[[:space:]]*:[[:space:]]*)[0-9.]+/\1${SWEEPS_PER_MEAS}/" "$INPUT"
    grep -q "\"sweeps-per-measurement\"[[:space:]]*:[[:space:]]*${SWEEPS_PER_MEAS}" "$INPUT" \
      || { echo "[drift] SWEEPS_PER_MEAS=${SWEEPS_PER_MEAS} substitution failed in ${INPUT}"; exit 1; }
  fi

  echo "[drift] seed ${SEED} ($((i+1))/${NSEEDS})"
  START=$(date +%s)
  # --bind-to none: each rank is itself multi-threaded (walkers + accumulators), so do not pin a rank
  # to a single core.
  "$MPIRUN" -n "$NRANKS" --bind-to none "$BIN" "$INPUT" "$STEM" > "${DIR}/run.log" 2>&1 \
    || { echo "[drift] seed ${SEED} FAILED, see ${DIR}/run.log"; tail -20 "${DIR}/run.log"; exit 1; }
  echo "[drift]   done in $(( $(date +%s) - START ))s -> $(ls "${DIR}"/${MODEL}_iter*.hdf5 | wc -l) iteration files"
done

echo "[drift] all ${NSEEDS} seeds done in $(( $(date +%s) - START_ALL ))s -> ${OUTROOT}"
