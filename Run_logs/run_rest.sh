#!/bin/zsh
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate qdrug
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version
python sanity_check.py                   > run_sanity.log 2>&1
python run_experiments.py --config LARGE > run_large.log  2>&1
echo done > run_all_done.marker
