#!/bin/bash
# Every discrete experiment, with the sweep run THREE times: one fixed beta each
# at 0.01 and 0.20, and once with the per-process rule.
#
# Training parameters, all measured (HOW_TO_RUN 14A.3):
#   lr 1e-3, weight_decay 0.01, 150 epochs, usage_target uniform,
#   n_states = the process's theoretical causal-state count, per arm.
#
# The three beta arms exist because no single value serves every process:
#   0.01   flower_n2_m6 collapses (+0.70 above H_inf); everything else converges
#   0.20   flower_n6_m4 misses (+0.107); flower_n2_m6 is rescued (+0.059)
#   rule   beta = 0.20 if min(log2K_fw - C+, log2K_bw - C-) <= 0.1 else 0.01,
#          SHARED by both arms -- 7 of 7 converge.  Shared, not per-arm, because
#          delta_CE is a paired difference and an arm-dependent hyperparameter is
#          precisely the artefact the pairing exists to exclude.
#
# PARALLELISM.  The three sweeps are independent -- separate out_roots, separate
# pickles -- so they run concurrently.  This machine has 11 cores; each job is
# pinned to 3 threads (3x3 = 9) with 2 left for the baseline chain.  Running them
# sequentially would take ~45 h; concurrently, ~20-25 h.
#
# Every runner resumes from its own pickle: killing this script and re-running it
# continues rather than restarting.
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/asym/bin/python
LOG=Run_logs
M=$LOG/launch_3way_master.log
ts() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a $M; }

say "===== discrete suite: 3-way sweep + baseline ====="

# ---- the three sweeps, concurrently ---------------------------------------
GRID_C="0.15 0.35 0.55 0.75 0.95"
GRID_F="2 4 6 8 10"

OMP_NUM_THREADS=3 $PY Experimental_setup/run_sweep_experiment.py --config DISCRETE \
    --sweep-coin $GRID_C --sweep-flower $GRID_F --repeats 5 --usage-beta 0.01 \
    --out-root All_Results/discrete/sweep_beta001 \
    > $LOG/sweep_beta001.log 2>&1 &
P1=$!; say "started sweep beta=0.01   pid $P1"

OMP_NUM_THREADS=3 $PY Experimental_setup/run_sweep_experiment.py --config DISCRETE \
    --sweep-coin $GRID_C --sweep-flower $GRID_F --repeats 5 --usage-beta 0.20 \
    --out-root All_Results/discrete/sweep_beta020 \
    > $LOG/sweep_beta020.log 2>&1 &
P2=$!; say "started sweep beta=0.20   pid $P2"

OMP_NUM_THREADS=3 $PY Experimental_setup/run_sweep_experiment.py --config DISCRETE \
    --sweep-coin $GRID_C --sweep-flower $GRID_F --repeats 5 \
    --out-root All_Results/discrete/sweep_betarule \
    > $LOG/sweep_betarule.log 2>&1 &
P3=$!; say "started sweep beta=rule   pid $P3"

# ---- the baseline chain on the two spare cores, alongside ------------------
(
  OMP_NUM_THREADS=2 $PY Experimental_setup/run_experiments.py --config DISCRETE \
      > $LOG/discrete_experiments.log 2>&1
  echo "[$(ts)] baseline run_experiments EXIT $?" >> $M
  OMP_NUM_THREADS=2 $PY Experimental_setup/run_statistical_trj.py --config DISCRETE \
      --repeats 30 > $LOG/discrete_trajectories.log 2>&1
  echo "[$(ts)] baseline run_statistical_trj EXIT $?" >> $M
) &
P4=$!; say "started baseline chain    pid $P4"

wait $P1; say "sweep beta=0.01 EXIT $?"
wait $P2; say "sweep beta=0.20 EXIT $?"
wait $P3; say "sweep beta=rule EXIT $?"
wait $P4; say "baseline chain  EXIT $?"
say "===== all finished ====="
