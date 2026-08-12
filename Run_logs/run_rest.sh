#!/bin/zsh
# Controls + LARGE only, for resuming after QUICK has already been run.
#
# REORGANISATION_FIX_PLAN.md 5.9.  See run_all.sh for the path notes.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate qdrug
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version
python Experimental_setup/sanity_check.py                  > Run_logs/run_sanity.log 2>&1
python Experimental_setup/run_experiments.py --config LARGE > Run_logs/run_large.log  2>&1
echo done > Run_logs/run_all_done.marker
