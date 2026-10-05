#!/bin/zsh
# The continuous-mode rerun of the existing result grid (user, 2026-10-03): coin, flower and the
# 5-state renewal at K = 4V -> results/cont_default/; the 9-state renewal at K = 16 ->
# results/cont_renewal9/.  All three models, both arms, seeds 0-2: 72 runs.  The GRUs run in a CPU
# pool (one thread each), the transformers in an MPS pool; each run logs to results/<name>/logs/.
# Figures are drawn once at the end, from the pickles.
#
#   scripts/schedule_continuous.sh            (from model_modeularised/)
#   CPU_JOBS=6 MPS_JOBS=4 SEEDS="0 1 2" PY=python scripts/schedule_continuous.sh
set -u
cd "${0:A:h}/.."
PY=${PY:-python}
CPU_JOBS=${CPU_JOBS:-6}
MPS_JOBS=${MPS_JOBS:-4}
SEEDS=(${=SEEDS:-0 1 2})
F9='F=[0.04,0.08,0.12,0.16,0.2,0.16,0.12,0.08,0.04]'

typeset -a cpu_jobs mps_jobs
for seed in $SEEDS; do
  for arm in forward backward; do
    for spec in "cont_default coin" "cont_default flower" "cont_default renewal" "cont_renewal9 renewal9"; do
      name=${spec%% *}; proc=${spec##* }
      extra=""
      [[ $proc == renewal9 ]] && extra="--params '$F9' --num-states-mult 8" && proc=renewal
      mkdir -p results/$name/logs
      for arch in transformer gru_feedback gru; do
        threads=1; [[ $arch == transformer ]] && threads=2
        log=results/$name/logs/${arch}_${proc}_${arm}_s${seed}.log
        cmd="$PY run_model.py --name $name --data-mode continuous --process $proc $extra --arch $arch --arm $arm --seed $seed --threads $threads --no-plots > $log 2>&1"
        if [[ $arch == transformer ]]; then mps_jobs+=("$cmd"); else cpu_jobs+=("$cmd"); fi
      done
    done
  done
done

echo "$(date '+%H:%M') ${#cpu_jobs} CPU jobs (x$CPU_JOBS), ${#mps_jobs} MPS jobs (x$MPS_JOBS)"
print -rN -- $mps_jobs | xargs -0 -P $MPS_JOBS -S 4096 -I{} zsh -c '{}' &
mps_pid=$!
print -rN -- $cpu_jobs | xargs -0 -P $CPU_JOBS -S 4096 -I{} zsh -c '{}' &
cpu_pid=$!
wait $mps_pid $cpu_pid
echo "$(date '+%H:%M') all runs finished; drawing figures"
for name in cont_default cont_renewal9; do
  $PY run_model.py --name $name --plots-only 2>&1 | grep "figures ->"
done
echo "$(date '+%H:%M') done"
