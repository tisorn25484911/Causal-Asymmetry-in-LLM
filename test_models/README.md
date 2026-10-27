# test_models

Every model of `model_modeularised/hybrid_model_test.ipynb` (sections 13–17) as its own file, with shared training,
shared analysis, and a walkthrough notebook. Run everything from this folder with the **qdrug** environment.

| file | what it is |
|---|---|
| `process.py` | every process and its analytical results: a verbatim copy of `HMM_processes.py` and `process_generator.py`, plus `Process` (one process with its parameters) and `WindowSampler` (fresh training windows) |
| `causal_state_matrix.py` | the part every model shares: the table of K state vectors, the output layer (state vector → next-token probabilities), and the choice of a state from scores |
| `training.py` | settings, training loop, validation measures, saved runs, longer training, loading runs |
| `model_<name>.py` | one model each, self-contained (it imports only the two files above) |
| `main.py` | the command line: `list`, `train`, `train-many`, `extend`, `report`, `import-runs` |
| `analysis.py` | everything after training: test results, machines, comparison with the true machine, all figures; its docstring says which analysis applies to which model |
| `walkthrough.ipynb` | choose a process → choose a model → train or load → plots; then one section per model (section 6); section 7: five models on renewal-60 and the flower process read forward and backward |
| `plans/` | lists of trainings for `train-many`; `new_processes.json` is section 7's 45 runs |
| `tests/` | `test_equivalence.py`: the same results as the code of sections 13–17 (needs `../model_modeularised`); `test_models.py`: every model end to end |
| `runs/` | saved runs: `runs/model_<name>/...` for new runs, `runs/imported/...` for the runs of sections 13, 16, 17, `runs/logs/` for `train-many`'s logs (another folder: set `TEST_MODELS_RUNS`) |

## The models

| name | loss (all CE in bits) | from |
|---|---|---|
| `1` | the GRU's predictions of the future tokens | section 16 `gru_loss` (1 + `--original-setup`: section 13's settings) |
| `2` | 1 + the Transformer's own next-token predictions | `both_losses` |
| `3` | 2 + weight × (entropy + CE) of a next-state network following the Transformer's next state | `next_state_net` |
| `4` | 3 without the GRU | `next_state_net_no_gru` |
| `A` | 3 with the Transformer's next state as a fixed target | `next_state_fixed_target` |
| `5a` | 1 + one- and two-step predictions + weight × entropy of the next state worked out from them | `two_step` |
| `5b` | 5a without the GRU | `two_step_no_gru` |
| `C` | 5b with the next state worked out from two predicted tokens (a three-step head) | `three_step_no_gru` |
| `B` | 1 + weight × CE of the Transformer's starting state against the GRU's own state after the history | `copy_gru_handoff` |
| `1_L2` | 1 + weight × D over all L future tokens (the notebook's L2) | sections 7, 13, 15 |
| `2_L2` | 2 + weight × D over the L − 1 GRU steps | section 17 `l2_both_losses` |
| `feedback_GRU` | a GRU alone whose state is a row of the state table; its own next-token predictions | section 17 `feedback_gru` |
| `discrete_Transformer` | a Transformer alone choosing a state at every position; its own next-token predictions | section 17 `discrete_transformer` |

D = the distance between the Transformer's and the GRU's state probabilities after the same tokens.

## Quick start

    python main.py list
    python main.py train --model 2 --seed 0                  # section 16's settings, 50,000 updates
    python main.py train --model 5b --steps 10000 --weight 0.1
    python main.py train --model C --json '{"process": "flower", "params": {"n": 8, "m": 12, "dice_seed": 0, "direction": "backward"}}'
    python main.py train-many --plan plans/new_processes.json --workers 8 --memory-gb 13
    python main.py report --run runs/model_2/<folder>
    python main.py import-runs                               # the trained runs of sections 13, 16, 17
    python -m pytest tests -q

`--json` takes any settings as a JSON object; `"direction": "backward"` in a process's params gives the same process
read from right to left (its generating machine is then the reverse ε-machine).

## Many runs at once

`train-many` runs the trainings of a plan (a JSON list of `{"model": ..., "settings": {...}, "memory_gb": ...}`), each in
its own process with one thread, in the order listed. A new one starts while fewer than `--workers` run and the
`memory_gb` of those running stays within `--memory-gb` (`memory_gb` is the run's peak memory; typical use is about a
third of it). Finished runs are skipped, so the same command continues an interrupted plan. A run whose `progress.json`
changed in the last 30 minutes is taken to be training in another process: it is watched until it finishes and not
started again, so a stopped `train-many` can be restarted (for example with another order) without stopping its
trainings. Each training's output goes to `runs/logs/`.

On macOS the runner sets `MallocLargeExpandedCacheThreshold=1`: freed large blocks are kept for reuse instead of being
returned to the system after every update. Model C on the flower (an output of 20³ = 8,000 numbers per position) then
trains 16% faster; the results are bit-identical (checked on four models). Do not change `training.py`, `process.py`,
`causal_state_matrix.py` or a model file while trainings run: a run's folder name includes a fingerprint of that code.
`TR.find_runs(model=..., process=..., seed=...)` finds saved runs by their `settings.json`, whatever their fingerprint.

## A new model

Copy `model_2.py` to `model_<name>.py` and change the loss or the parts. Keep the interface that `training.py`
describes (`Model(settings)`, `loss`, `predict`, and optionally `step_rules`, `handoff_states`, `measures`,
`after_update`); then `python main.py train --model <name>` trains it and every analysis that applies works on it.
