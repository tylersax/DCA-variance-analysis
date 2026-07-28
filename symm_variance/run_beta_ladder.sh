#!/usr/bin/env bash
# Beta ladder (ROADMAP task 2): sweep the inverse temperature at fixed model, cluster, ranks and
# depth, to test whether square is intrinsically rho-limited or only looks that way at beta=1.
#
# THE HYPOTHESIS. `r = m/[1+(m-1)rho]`, so the reduction is capped by the mate-correlation rho, not
# just by orbit size m. At beta=1 the noise is dominated by symmetric scalar channels, rho ~ 0.95,
# and `R` is stuck near 1. As beta rises the correlation length grows, the noise acquires
# symmetry-BREAKING structure, rho should fall and `R` rise.
#
# WHY SQUARE IS THE RIGHT MODEL FOR THIS AXIS. Two effects compete in general (see ROADMAP task 2):
# growing correlation length pushes rho DOWN, but sign-problem noise enters as a global scalar
# (`dG(k) = -G(k) ds/s`, weight -G(k), fully symmetric, mate-correlation exactly 1) and pushes rho
# UP. Square with nearest-neighbour hopping on a bipartite lattice at half filling (mu=0) is provably
# sign-free in CT-AUX at ANY beta -- measured <s> = 1 exactly at every beta on this ladder -- so it
# isolates the correlation-length effect with the sign channel switched off. The sign channel is a
# separate experiment on a model that has one.
#
# DEPTH FLOORS ARE PER-BETA. Conventions require re-establishing the floor whenever beta changes:
# autocorrelation time grows with beta at fixed sweeps-per-measurement, so a depth that was
# comfortably asymptotic at beta=1 need not be at beta=8. Mode `floor` runs an m-ladder at each beta
# for exactly that check; run it, read m_scaling's drift/flatness tables, and only then run mode
# `ensemble` at a depth above the measured floor.
#
# Each beta gets its OWN directory, so run filenames keep the `<model>_r<ranks>_m<M>_s<seed>.hdf5`
# form that m_scaling/seed_ensemble already parse -- no naming changes, and beta is read back from
# the HDF5 metadata (the driver records parameters.get_beta()).
#
# usage: run_beta_ladder.sh floor    <model> <n_ranks> <outroot> <m,...>  <seed,...> <beta...>
#        run_beta_ladder.sh ensemble <model> <n_ranks> <outroot> <m>      <n_seeds>  <beta...>
set -euo pipefail

MODE=${1:?usage: run_beta_ladder.sh <floor|ensemble> <model> <n_ranks> <outroot> <m|m,...> <seeds|n_seeds> <beta...>}
MODEL=${2:?need model}
NRANKS=${3:?need n_ranks}
OUTROOT=${4:?need outroot}
MSPEC=${5:?need m (ensemble) or m-list (floor)}
SEEDSPEC=${6:?need seed-list (floor) or n_seeds (ensemble)}
shift 6
(( $# > 0 )) || { echo "need at least one beta"; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTROOT"

for b in "$@"; do
  # A directory name must survive being a shell path: beta 0.5 -> b0.5 is fine, but keep it explicit.
  DEST="${OUTROOT}/b${b}"
  echo "=== beta=${b} -> ${DEST}"
  start=$(date +%s)
  case "$MODE" in
    floor)
      BETA="$b" "${HERE}/run_m_ladder.sh" "$MODEL" "$NRANKS" "$DEST" "$MSPEC" "$SEEDSPEC"
      ;;
    ensemble)
      # DISJOINT seed range per beta. An ensemble of N seeds at stride 10000 spans N*10000 keys, so
      # the per-beta offsets must be separated by more than that or two RUNGS share base seeds --
      # identical walker streams at different temperatures. The rungs would then be correlated, and
      # `beta_ladder.trend` treats them as independent when it builds the interval on the change
      # across the ladder. That is the failure mode that makes an interval too NARROW rather than too
      # wide, which is the one worth engineering against. 1e6 per unit beta clears any sane N.
      SEED0=$(( 2000000 + $(printf '%.0f' "$(echo "$b * 1000000" | bc -l)") ))
      SPAN=$(( SEEDSPEC * 10000 ))   # ensemble mode: SEEDSPEC is n_seeds
      (( SPAN < 1000000 )) || { echo "[beta-ladder] ${SEEDSPEC} seeds span ${SPAN} keys, exceeding" \
        "the 1e6 gap between beta offsets: rungs would share walker streams." >&2; exit 1; }
      BETA="$b" "${HERE}/run_seed_ensemble.sh" "$MODEL" "$NRANKS" "$MSPEC" "$SEEDSPEC" "$DEST" \
           "$SEED0" 10000
      ;;
    *) echo "unknown mode: $MODE"; exit 1 ;;
  esac
  echo "=== beta=${b} done ($(( $(date +%s) - start ))s)"
done
echo "[beta-ladder] ${MODE} complete -> ${OUTROOT}"
