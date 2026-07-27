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

echo "[run] ${MODEL}: ${NRANKS} ranks x ${MEAS} meas/rank (total ${MEAS_TOTAL}), seed ${SEED} -> ${OUT}"
# --bind-to none: each rank is itself multi-threaded (walkers + accumulators), so do not pin a rank
# to a single core.
"$MPIRUN" -n "$NRANKS" --bind-to none "$BIN" "$INPUT" "$OUT"
echo "[run] done: ${OUT}"
