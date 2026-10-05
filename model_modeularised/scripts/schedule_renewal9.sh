#!/bin/zsh
# Scheduled 2026-09-28 (user): the 9-state renewal control, every model (transformer, feedback GRU,
# read-out GRU) with and without Gumbel-ST, both arms:
#
#   F = (0.04, 0.08, 0.12, 0.16, 0.20, 0.16, 0.12, 0.08, 0.04)   9 causal states both ways, C+ = C- = 2.829 bits
#   K = 8V = 16                                               (the default 4V = 8 is below the 9 states)
#
#   results/renewal9          the argmax head (the default)
#   results/gumbel_renewal9   Gumbel-ST for every model
#
#   scripts/schedule_renewal9.sh [PID ...]      waits for every PID given, then runs both grids in parallel
set -u
cd "$(dirname "$0")/.."
for pid in "$@"; do
  echo "waiting for PID $pid to finish ($(date))"
  while kill -0 $pid 2>/dev/null; do sleep 30; done
done
F='F=[0.04,0.08,0.12,0.16,0.2,0.16,0.12,0.08,0.04]'
export ARCHS="transformer gru_feedback gru" PROCESSES=renewal
echo "9-state renewal grids starting $(date)"
scripts/launch_grid.sh renewal9 --num-states-mult 8 --params "$F" &
scripts/launch_grid.sh gumbel_renewal9 --num-states-mult 8 --params "$F" --gumbel-transformer --gumbel-gru &
wait
echo "9-state renewal grids done $(date)"
