#!/usr/bin/env bash
# M-scaling control (ROADMAP task 1): sweep measurements-per-rank at FIXED point group, ranks and
# seed, so the only thing that changes is run depth.
#
# Why this matters: R = Var(G)/Var(Sym G) is scale-free ONLY if each rank's sample is in the CLT
# regime. Below that, Var(G_i) is not simply Sigma/M, the two arms can pick up depth-dependent
# structure at different rates, and R drifts. Flatness of R in M is the licence for "many ranks,
# modest depth".
#
# NESTING (deliberate): a rank's walker seeds are hash(global_id + base_seed) and do not depend on
# `measurements`, so at fixed seed the M-run's chain is a PREFIX of the 4M-run's chain on the same
# rank. Runs at different M are therefore positively correlated, and the paired bootstrap in
# analysis/m_scaling.py exploits that -- common randomness cancels, so drift in R is easier to
# resolve than it would be from independent runs. Use several base seeds to recover independent
# replicates of the whole ladder.
#
# The swept quantity is measurements PER RANK -- a rank is one sample, so its own depth is what
# decides whether that sample is asymptotic. (DCA's input key is the total over ranks;
# run_symm_variance.sh does the conversion.)
#
# usage: run_m_ladder.sh <square|fe_as> <n_ranks> <outroot> <m[,m,...]> <seed[,seed,...]>
# e.g.   run_m_ladder.sh square 64 /scratch/mladder 4,16,64,256,1024,4096 12345,777,20260727
set -euo pipefail

MODEL=${1:?usage: run_m_ladder.sh <square|fe_as> <n_ranks> <outroot> <m_per_rank,...> <seed,...>}
NRANKS=${2:?need n_ranks}
OUTROOT=${3:?need outroot}
MLIST=${4:-4,16,64,256,1024,4096}
SEEDLIST=${5:-12345}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTROOT"
STAGE="${OUTROOT}/.stage"

IFS=',' read -r -a MS <<< "$MLIST"
IFS=',' read -r -a SEEDS <<< "$SEEDLIST"

for seed in "${SEEDS[@]}"; do
  for m in "${MS[@]}"; do
    DEST="${OUTROOT}/${MODEL}_r${NRANKS}_m${m}_s${seed}.hdf5"
    if [[ -f "$DEST" ]]; then
      echo "[ladder] skip (exists): $(basename "$DEST")"
      continue
    fi
    rm -rf "$STAGE"
    start=$(date +%s)
    "${HERE}/run_symm_variance.sh" "$MODEL" "$NRANKS" "$m" "$seed" "$STAGE" > "${STAGE}.log" 2>&1 \
      || { echo "[ladder] FAILED ${MODEL} M=${m} seed=${seed}; see ${STAGE}.log"; exit 1; }
    mv "${STAGE}/${MODEL}.hdf5" "$DEST"
    echo "[ladder] $(basename "$DEST")  ($(( $(date +%s) - start ))s)"
  done
done
rm -rf "$STAGE" "${STAGE}.log"
echo "[ladder] done -> ${OUTROOT}"
