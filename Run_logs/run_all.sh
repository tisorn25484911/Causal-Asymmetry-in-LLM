#!/bin/zsh
# Full reproduction run: controls, then QUICK, then LARGE.
#
# REORGANISATION_FIX_PLAN.md 5.9.  Script paths carry their new directory and the
# logs are written into Run_logs/ rather than the repo root.  The `run_*.log`
# gitignore rule matches at any depth, so they stay out of git here too.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate qdrug
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version
python Experimental_setup/sanity_check.py                    > Run_logs/run_sanity.log 2>&1
python Experimental_setup/run_experiments.py --config QUICK   > Run_logs/run_quick.log  2>&1
python Experimental_setup/run_experiments.py --config LARGE   > Run_logs/run_large.log  2>&1
echo "ALL RUNS COMPLETE" > Run_logs/run_all_done.marker
