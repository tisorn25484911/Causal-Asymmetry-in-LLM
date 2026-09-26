#!/bin/zsh
# Train a whole grid in parallel, one log per run, then draw every figure.
#
#   scripts/launch_grid.sh <name> [extra run_model.py flags ...]
#   scripts/launch_grid.sh default
#   scripts/launch_grid.sh det_gru --no-gumbel-gru       (the deterministic-head ablation)
#
# Transformers share one lane (they run on MPS and would fight over it); every GRU
# run is its own 1-thread CPU process.  Results, logs and figures all go to
# results/<name>/ (layout: run_model.py's docstring).  Override the grid with
# ARCHS / PROCESSES / ARMS / SEED in the environment.
set -u
cd "$(dirname "$0")/.."
NAME=${1:?usage: scripts/launch_grid.sh <name> [run_model.py flags ...]}; shift
PY=${PY:-/opt/anaconda3/envs/qdrug/bin/python}
ARCHS=(${=ARCHS:-transformer gru_feedback})
PROCESSES=(${=PROCESSES:-coin flower})
ARMS=(${=ARMS:-forward backward})
SEED=${SEED:-0}
LOGS=results/$NAME/logs
mkdir -p $LOGS

for arch in $ARCHS; do
  if [[ $arch == transformer ]]; then
    $PY run_model.py --name $NAME --seed $SEED --arch transformer --process $PROCESSES --arm $ARMS \
        --threads 2 --no-plots "$@" > $LOGS/transformer_all_s$SEED.log 2>&1 &
    continue
  fi
  for proc in $PROCESSES; do for arm in $ARMS; do
    $PY run_model.py --name $NAME --seed $SEED --arch $arch --process $proc --arm $arm \
        --threads 1 --no-plots "$@" > $LOGS/${arch}_${proc}_${arm}_s$SEED.log 2>&1 &
  done; done
done
wait
$PY run_model.py --name $NAME --plots-only
echo "done: results/$NAME  ($(date))"
