#!/usr/bin/env bash
# Seed ensemble (ROADMAP task 1): many INDEPENDENT base seeds at ONE fixed depth, to pin `R` with an
# honest across-seed standard error.
#
# Why not more ranks, and why not more depth:
#   * `R` is scale-free above the model's depth floor, so extra depth buys nothing once the floor is
#     cleared -- it only makes each replicate more expensive.
#   * Precision comes from independent samples, but the rank bootstrap runs 1.4-1.9x optimistic where
#     there is a sign problem, so ranks inside one seed do not give a trustworthy interval. An
#     independent base seed is a whole fresh replicate; the sample SD across seeds needs no
#     assumption about rank independence at all.
# Hence: fix the depth just above the floor, and spend everything on seeds.
#
# SEED SPACING IS NOT COSMETIC. A walker's stream is hash(global_id + base_seed) with
# global_id = local_id*n_ranks + proc_id (src/math/random/random_utils.cpp), so a run with base seed
# S occupies the contiguous key range [S, S + n_ranks*n_walkers). Two base seeds closer together than
# that width therefore SHARE walker streams, and the runs are not independent replicates -- which is
# the one thing this ensemble exists to provide. This script enforces the spacing.
#
# usage: run_seed_ensemble.sh <square|fe_as> <n_ranks> <m_per_rank> <n_seeds> <outroot> [seed0] [stride]
# e.g.   run_seed_ensemble.sh fe_as 64 2048 32 /home/tsax10/dca/scratch/seed_ensemble
set -euo pipefail

MODEL=${1:?usage: run_seed_ensemble.sh <square|fe_as> <n_ranks> <m_per_rank> <n_seeds> <outroot> [seed0] [stride]}
NRANKS=${2:?need n_ranks}
M=${3:?need measurements per rank}
NSEEDS=${4:?need n_seeds}
OUTROOT=${5:?need outroot}
SEED0=${6:-1000000}
STRIDE=${7:-10000}

# Widest plausible key range one run can occupy: n_ranks * n_walkers, with a generous walker bound.
# The stride must clear it, or distinct "seeds" replay each other's chains.
MIN_STRIDE=$(( NRANKS * 64 ))
if (( STRIDE < MIN_STRIDE )); then
  echo "[ensemble] stride ${STRIDE} is too small for ${NRANKS} ranks: runs would share walker streams." >&2
  echo "[ensemble] use a stride of at least ${MIN_STRIDE}." >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTROOT"
STAGE="${OUTROOT}/.stage"

for (( i = 0; i < NSEEDS; i++ )); do
  seed=$(( SEED0 + i * STRIDE ))
  DEST="${OUTROOT}/${MODEL}_r${NRANKS}_m${M}_s${seed}.hdf5"
  if [[ -f "$DEST" ]]; then
    echo "[ensemble] skip (exists): $(basename "$DEST")"
    continue
  fi
  rm -rf "$STAGE"
  start=$(date +%s)
  "${HERE}/run_symm_variance.sh" "$MODEL" "$NRANKS" "$M" "$seed" "$STAGE" > "${STAGE}.log" 2>&1 \
    || { echo "[ensemble] FAILED ${MODEL} seed=${seed}; see ${STAGE}.log"; exit 1; }
  mv "${STAGE}/${MODEL}.hdf5" "$DEST"
  echo "[ensemble] $(( i + 1 ))/${NSEEDS} $(basename "$DEST")  ($(( $(date +%s) - start ))s)"
done
rm -rf "$STAGE" "${STAGE}.log"
echo "[ensemble] done -> ${OUTROOT}"
