#!/usr/bin/env bash
# Diagnostic: does <sign> behave like physics? Two tests.
#   (a) beta sweep on FeAs -- a physical sign problem must go to 1 smoothly as beta -> 0 and decay
#       monotonically, roughly exp(-c*beta*Nc). Erratic or non-monotonic behaviour would point at code.
#   (b) square at the SAME betas as a sign-free control: nearest-neighbour-only hopping on a bipartite
#       lattice at half filling (mu=0) is provably sign-free in CT-AUX at ANY beta. If square stays
#       exactly 1 where FeAs decays, the sign problem is a property of the MODEL, not of DCA or of
#       this driver.
set -euo pipefail
OUT=${1:?usage: run_sign_sweep.sh <outdir> [n_ranks] [meas_per_rank] <beta...>}
TPL_DIR=/home/tsax10/dca/analysis/symm_variance
BIN_DIR=/home/tsax10/dca/build_symm
MPIRUN=/home/tsax10/conda/envs/qe/bin/mpirun
NRANKS=${2:-16}
MPR=${3:-1024}          # measurements per rank
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
TOTAL=$(( MPR * NRANKS ))

run_one() {  # model beta
  local model=$1 beta=$2
  local dest="${OUT}/${model}_b${beta}.hdf5"
  [[ -f "$dest" ]] && { echo "[sign] skip $(basename "$dest")"; return; }
  local inp="${OUT}/${model}_b${beta}.json"
  # substitute seed/measurements as usual, then override beta only
  sed -e "s/__SEED__/4242/" -e "s/__MEASUREMENTS__/${TOTAL}/" \
      "${TPL_DIR}/${model}_input.template.json" \
    | sed -E "s/(\"beta\"[[:space:]]*:[[:space:]]*)[0-9.]+/\1${beta}/" > "$inp"
  local t0=$(date +%s)
  "$MPIRUN" -n "$NRANKS" --bind-to none "${BIN_DIR}/${model}_symm_variance" "$inp" "$dest" \
     > "${OUT}/${model}_b${beta}.log" 2>&1
  echo "[sign] ${model} beta=${beta}  ($(( $(date +%s) - t0 ))s)"
}

for b in "${@:4}"; do run_one fe_as "$b"; done
for b in "${@:4}"; do run_one square "$b"; done
echo "[sign] done -> $OUT"
