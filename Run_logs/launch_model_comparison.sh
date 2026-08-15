#!/bin/bash
# Sequential launcher for the two-architecture comparison.
#
# Sequential, not parallel: these are CPU/MPS-bound and contending would make
# every ETA meaningless.  Every runner resumes from its own pickle, so killing
# this script and re-running it continues rather than restarting.
#
# Measured cost per process-repeat (2 arms x 5 folds):
#     onehot   (QUICK,    lr 1e-2, 10 epochs)    ~15 s
#     discrete (DISCRETE, lr 1e-3, 230 epochs)   ~320 s
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/asym/bin/python
LOG=Run_logs
ts() { date "+%Y-%m-%d %H:%M:%S"; }
run() { echo "[$(ts)] START $1"; shift; "$@" ; echo "[$(ts)] EXIT $?"; }

echo "[$(ts)] ===== model comparison launch ====="

# 1. onehot baseline into the new tree (~20 min)
run "onehot run_experiments" \
    $PY Experimental_setup/run_experiments.py --config QUICK --model onehot \
    > $LOG/onehot_experiments.log 2>&1

# 2. onehot repeats (~2 h)
run "onehot run_statistical_trj" \
    $PY Experimental_setup/run_statistical_trj.py --config QUICK --model onehot \
    --repeats 30 > $LOG/onehot_trajectories.log 2>&1

# 3. discrete baseline (~8 h)
run "discrete run_experiments" \
    $PY Experimental_setup/run_experiments.py --config DISCRETE \
    > $LOG/discrete_experiments.log 2>&1

# 4. discrete repeats, 7 processes x 30 (~19-24 h)
run "discrete run_statistical_trj" \
    $PY Experimental_setup/run_statistical_trj.py --config DISCRETE --repeats 30 \
    > $LOG/discrete_trajectories.log 2>&1

# 5. discrete sweep on the REDUCED grid: 5x5 coin + 5x5 flower = 50 processes,
#    5 repeats.  50 x 5 x 320 s ~ 22 h.  The full 181-cell grid at 30 repeats
#    would be ~10-20 days at this budget, which is why the grid is coarser and
#    the sem correspondingly wider -- state that with the results.
run "discrete sweep (reduced grid)" \
    $PY Experimental_setup/run_sweep_experiment.py --config DISCRETE \
    --sweep-coin 0.15 0.35 0.55 0.75 0.95 \
    --sweep-flower 2 4 6 8 10 --repeats 5 \
    > $LOG/discrete_sweep.log 2>&1

echo "[$(ts)] ===== all runs finished ====="
