#!/bin/zsh
# Scheduled 2026-09-28 (user): once the argmax grid in results/default has finished, rerun every
# process with Gumbel-ST for the transformer and the feedback GRU, one folder per process:
#
#   results/gumbel_coin    results/gumbel_flower    results/gumbel_renewal
#
# The three grids run in parallel (three transformer lanes on MPS, six 1-thread GRU processes);
# each draws its own figures when it finishes.
#
#   scripts/schedule_gumbel.sh [PID to wait for]
set -u
cd "$(dirname "$0")/.."
if [[ -n ${1:-} ]]; then
  echo "waiting for PID $1 to finish ($(date))"
  while kill -0 $1 2>/dev/null; do sleep 30; done
fi
echo "Gumbel grids starting $(date)"
for proc in coin flower renewal; do
  PROCESSES=$proc scripts/launch_grid.sh gumbel_$proc --gumbel-transformer --gumbel-gru &
done
wait
echo "Gumbel grids done $(date)"
