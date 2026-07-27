#!/usr/bin/env bash
# Multi-rank symmetrization-variance run: rank == one sample, one QMC iteration, raw per-rank G
# dumped before finalize (see symm_variance_main.inc). Unlike the deprecated variance_demo (which ran
# many single-rank replica processes), this uses ONE mpirun with n_ranks == number of samples; the
# per-rank RNG streams are independent by construction (seed = hash(global_id + base_seed)), so a
# single base seed is correct and there is no cross-run collision hazard.
#
# usage: run_symm_variance.sh <square|fe_as> <n_ranks> <measurements> [seed] [outdir]
set -euo pipefail

MODEL=${1:?usage: run_symm_variance.sh <square|fe_as> <n_ranks> <measurements> [seed] [outdir]}
NRANKS=${2:?need n_ranks (== number of samples)}
MEAS=${3:?need measurements per rank}
SEED=${4:-42}
OUTDIR=${5:-runs}

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
sed -e "s/__SEED__/${SEED}/" -e "s/__MEASUREMENTS__/${MEAS}/" "$TEMPLATE" > "$INPUT"

echo "[run] ${MODEL}: ${NRANKS} ranks x ${MEAS} meas, seed ${SEED} -> ${OUT}"
# --bind-to none: each rank is itself multi-threaded (walkers + accumulators), so do not pin a rank
# to a single core.
"$MPIRUN" -n "$NRANKS" --bind-to none "$BIN" "$INPUT" "$OUT"
echo "[run] done: ${OUT}"
