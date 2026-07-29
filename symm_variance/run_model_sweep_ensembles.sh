#!/usr/bin/env bash
# Gate 4 ensembles, run SEQUENTIALLY: two 64-rank jobs at once is 4x oversubscribed on 128 cores and
# slows both. Cheapest first, so the cheap points land early even if the long ones are interrupted.
# Every step is skip-if-exists, so re-running this script resumes rather than repeats.
set -uo pipefail
cd /home/tsax10/dca/analysis/.claude/worktrees/task4-model-sweep/symm_variance
export BIN_DIR=/home/tsax10/dca/build_task4
R=/home/tsax10/dca/scratch/model_sweep

# Wait out the 8x8 floor ladder (file-existence, not pgrep -- a pgrep pattern matches this script's
# own command line and would spin forever).
while [ ! -f /home/tsax10/dca/scratch/task4_floor/square_b5_c8/square_r64_m8192_s22000003.hdf5 ]; do
  sleep 30
done
echo "[chain] 8x8 floor done, starting ensembles $(date)"

echo "[chain] === square_b5_c4 (32 seeds) $(date) ==="
BETA=5 CLUSTER=4 NW=64 ./run_seed_ensemble.sh square 64 2048 32 $R/square_b5_c4 20000000 10000
echo "[chain] === square_b5_c4 DONE $(date) ==="

echo "[chain] === threeband_b5_c4 w2000 sensitivity arm (8 seeds) $(date) ==="
BETA=5 CLUSTER=4 NW=64 WARMUP=2000 MU=3.0 ./run_seed_ensemble.sh threeband 64 2048 8 $R/threeband_b5_c4_w2000 21500000 10000
echo "[chain] === sensitivity arm DONE $(date) ==="

echo "[chain] === square_b5_c8 (16 seeds) $(date) ==="
BETA=5 CLUSTER=8 NW=64 ./run_seed_ensemble.sh square 64 2048 16 $R/square_b5_c8 22000000 10000
echo "[chain] === square_b5_c8 DONE $(date) ==="

echo "[chain] === threeband_b5_c4 (32 seeds, warm-up 8000 -- the long one) $(date) ==="
BETA=5 CLUSTER=4 NW=64 WARMUP=8000 MU=3.0 ./run_seed_ensemble.sh threeband 64 2048 32 $R/threeband_b5_c4 21000000 10000
echo "[chain] === threeband_b5_c4 DONE $(date) ==="
echo "[chain] ALL ENSEMBLES COMPLETE $(date)"
