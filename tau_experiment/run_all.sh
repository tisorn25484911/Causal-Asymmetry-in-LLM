#!/bin/bash
# Drives every remaining phase to completion, in dependency order.
#
#   ./run_all.sh              from tau_experiment/
#
# Phases, and why they are ordered this way:
#
#   1A  direction / endpoint / range, on a second process family.  Launched
#       separately (study0's script, 6 arms) -- this script waits for it and
#       files the JSON under study1/results_1A.
#   1B  levels and shape, 12 arms x 3 cells.  Cell 2 is already running; cells
#       0 and 1 go in parallel here.
#   ->  study 1 is analysed, and the WINNER IS READ OFF THE DATA (pick_winner.py)
#       rather than typed in, so study 2 cannot inherit a hand-picked schedule.
#   1D  robustness of the top two arms on four cells they did not choose.
#   2   Adam momentum, tau pinned at study 1's winner.  Runs in parallel with 1D
#       -- 1D is a check on study 1, not an input to study 2.
#
# Two threads per job: the models are small, so throughput comes from running
# four jobs at once on 11 cores rather than from threading one.
set -u
PY=/opt/anaconda3/envs/qdrug/bin/python
TE="$(cd "$(dirname "$0")" && pwd)"
SCRATCH="${TAUCHECK_DIR:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

say() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1A: wait for the two already-running direction cells, then file them ──
if [ -n "$SCRATCH" ]; then
  say "waiting for phase 1A (direction/endpoint/range) ..."
  while [ "$(ls "$SCRATCH"/*.json 2>/dev/null | wc -l)" -lt 2 ]; do sleep 30; done
  cp "$SCRATCH"/*.json "$TE/study1_tau_schedules/results_1A/"
  cp "$SCRATCH"/*.log  "$TE/study1_tau_schedules/results_1A/" 2>/dev/null
  say "1A filed: $(ls "$TE/study1_tau_schedules/results_1A"/*.json | wc -l) cells"
fi

# ── 1B: cells 0 and 1 in parallel; cell 2 is already in flight ───────────
cd "$TE/study1_tau_schedules" || exit 1
say "launching 1B cells 0 and 1"
$PY run.py --phase 1B --cell 0 > results/1B_cell0_flower_n2m6_fw.log 2>&1 &
P0=$!
$PY run.py --phase 1B --cell 1 > results/1B_cell1_flower_n6m4_bw.log 2>&1 &
P1=$!
wait $P0 $P1
say "1B cells 0,1 done"

say "waiting for 1B cell 2 ..."
while [ "$(ls results/1B_*.json 2>/dev/null | wc -l)" -lt 3 ]; do sleep 30; done
say "all three 1B cells present"

# ── analyse study 1, and let the data name the winner ────────────────────
$PY analyse.py > results/analysis_study1.txt 2>&1
WINNERS=$($PY pick_winner.py --n 2)
W1=$(echo "$WINNERS" | awk '{print $1}')
say "study 1 winner: $W1   (runner-up: $(echo "$WINNERS" | awk '{print $2}'))"

# ── 1D and study 2 in parallel ───────────────────────────────────────────
say "launching study 2 (3 cells) with tau=$W1, and 1D (4 cells)"
cd "$TE/study2_momentum" || exit 1
for c in 0 1 2; do
  $PY run.py --tau "$W1" --cell $c > "results/S2_cell${c}.log" 2>&1 &
done
S2PIDS=$(jobs -p)

cd "$TE/study1_tau_schedules" || exit 1
(
  for c in 0 1 2 3; do
    $PY run.py --phase 1D --arms $WINNERS --cell $c \
        >> results/1D_robustness.log 2>&1
  done
) &
P1D=$!

wait $S2PIDS $P1D
say "1D and study 2 done"

# ── final analyses ───────────────────────────────────────────────────────
cd "$TE/study1_tau_schedules" && $PY analyse.py > results/analysis_study1.txt 2>&1
cd "$TE/study2_momentum"      && $PY analyse.py > results/analysis_study2.txt 2>&1
say "FIGURES FINAL"
ls "$TE"/study1_tau_schedules/figures "$TE"/study2_momentum/figures
