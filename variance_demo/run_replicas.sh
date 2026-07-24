#!/usr/bin/env bash
# Drive N paired replicas of the variance demo. Both arms read the SAME explicit-integer seed list
# (prime stride 1000003, so consecutive seeds are spaced far beyond ranks*walkers -- see the plan's
# "Seeding" trap). One JSON per seed is generated from the template; each run dumps its own HDF5.
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Where the built demo binaries live. Override per machine:  BIN_DIR=/path/to/build/variance_demo ./run_replicas.sh
BIN_DIR="${BIN_DIR:-/home/tyler-sax/projects/dca_m3_build/variance_demo}"
OUT_DIR="${DEMO_DIR}/runs"
TEMPLATE="${DEMO_DIR}/square_variance_input.template.json"

N="${1:-64}"          # replicas per arm
STRIDE=1000003
CONCURRENCY="${2:-3}"  # simultaneous runs (each ~4 threads)

mkdir -p "${OUT_DIR}/inputs" "${OUT_DIR}/on" "${OUT_DIR}/off"

# Build the shared seed list once and record it.
SEEDS=()
for ((i=1; i<=N; i++)); do SEEDS+=( $(( STRIDE * i )) ); done
printf "%s\n" "${SEEDS[@]}" > "${OUT_DIR}/seeds.txt"
echo "seeds: ${SEEDS[0]} .. ${SEEDS[$((N-1))]}  (N=${N})"

run_one() {
  local arm="$1" seed="$2"
  local inp="${OUT_DIR}/inputs/input_${seed}.json"
  sed "s/__SEED__/${seed}/" "${TEMPLATE}" > "${inp}"
  "${BIN_DIR}/ctaux_variance_demo_${arm}" "${inp}" "${OUT_DIR}/${arm}/G_${seed}.hdf5" \
    > "${OUT_DIR}/${arm}/log_${seed}.txt" 2>&1
}
export -f run_one
export OUT_DIR TEMPLATE BIN_DIR

# Launch all (arm,seed) jobs through xargs -P for bounded concurrency.
for arm in on off; do
  for seed in "${SEEDS[@]}"; do echo "${arm} ${seed}"; done
done | xargs -P "${CONCURRENCY}" -I{} bash -c 'run_one $0 $1' {}

echo "done: $(ls ${OUT_DIR}/on/*.hdf5 | wc -l) ON, $(ls ${OUT_DIR}/off/*.hdf5 | wc -l) OFF files"
