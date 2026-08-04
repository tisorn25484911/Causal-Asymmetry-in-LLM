#!/bin/zsh
# Wait for the QUICK run to finish, then start LARGE.  Sequential rather than
# parallel: both train on the same single MPS device, so overlapping them just
# thrashes it.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate qdrug
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version
while pgrep -f "run_experiments.py --config QUICK" > /dev/null; do sleep 30; done
exec python run_experiments.py --config LARGE > run_large.log 2>&1
