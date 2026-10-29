"""Build and execute test_models/walkthrough.ipynb.  The per-model findings text comes from walkthrough_findings.py
(written after the cached reports of the imported runs were read)."""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
OUT = Path("/Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version/test_models/walkthrough.ipynb")
sys.path.insert(0, str(HERE))
from walkthrough_findings import FINDINGS                                   # noqa: E402
import new_processes_section as S7                                          # noqa: E402
import de_bruijn_section as S10                                             # noqa: E402

cells = []


def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbformat.v4.new_code_cell(text.strip("\n")))


md(r"""
# Walkthrough: choose a process, choose a model, train or load it, look at the results

This notebook uses the files of this folder:

* `process.py` — every process and its analytical results;
* `training.py` — the shared training code;
* `causal_state_matrix.py` — the state table every model shares;
* `model_<name>.py` — one file per model;
* `analysis.py` — every measure and figure.

Sections 1–5 go through the steps once: choose a process, choose a model, train or load it, choose the plots. Section 6
then has one demonstration per model. Each one shows how well the model predicts on held-out data, how it converges
during training, and what machine it ends up with. Sections 7–9 train five of the models on new processes:
renewal-60 (section 7), and the flower process with 8 dice of 12 faces (section 8) and with 20 dice of 30 faces
(section 9), each flower read forward and backward. Section 10 tests a new model, the de Bruijn hybrid
(`model_de_bruijn.py`), on renewal processes with 5 to 60 states and two more flower processes. The demonstrations use the trained runs of sections 13, 16 and 17 of
`model_modeularised/hybrid_model_test.ipynb`, copied into `runs/imported/` by `python main.py import-runs`. Nothing is
retrained here except a short example in section 4.

**Terms used throughout**
* **Excess CE.** A predictor's cross-entropy minus the true process's cross-entropy on the same tokens, in bits per
  token. 0 is perfect.
* **Test windows.** 4,096 fresh windows of 40 history + 20 future tokens that never affect training or any choice.
* **Seeds.** Each model has three runs (seeds 0, 1, 2), which differ in their starting weights and training windows.
""")

md(r"""
## 1. Setup
""")
code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))          # run this notebook from test_models/

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

import process as P
import training as TR
import analysis as A
import main as MAIN

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
runs = MAIN.imported_runs()                  # {(model, label, seed): run folder}


def runs_of(model, label=None):
    # The imported runs of one model (all seeds; optionally one label such as 'section16' or 'w0.03').
    return {f"{m} {l} s{s}": f for (m, l, s), f in runs.items() if m == model and (label is None or l == label)}


def performance(r, which="latest"):
    # The test results of some runs: every predictor's excess CE on the 20 future tokens and over 200 future tokens,
    # true states with their own learned state, and the size of every step rule's machine.
    table = A.test_table(r, which)
    keep = ["seed", "weight", "update"] + [c for c in table if c.startswith("excess_") and c.endswith("_future")] \
        + [c for c in table if c.startswith("long_excess_")] + [c for c in table if c.endswith("_true_states_with_own_state")] \
        + [c for c in table if c.endswith("_long_run_states")] + [c for c in ("distance_D",) if c in table]
    return table[keep].set_index("seed").round(4)
""")

md(r"""
## 2. Choose the process

`process.py` contains every process of the project (`PROCESS`), copied from `HMM_processes.py` and
`process_generator.py`. `P.Process(name, params)` turns one of them into an object:

* **The generating machine** (`T`, `E`, `M`): the probability of each token and next state from each state.
* **The long-run state distribution** `pi`.
* **A window sampler** (`sampler(seed)`): fresh windows, each started from `pi`, as in section 16.
* **The true next-token probabilities** of any window (`exact_probs`), used to compute every excess CE.
* **The analytical results** (`analytics()`):
  * the number of causal states;
  * the statistical complexity C_μ, the entropy of `pi`;
  * the entropy rate h;
  * the excess entropy E, how much the past tells about the future;
  * the crypticity χ = C_μ − E, the part of the state that no amount of future reveals (Mahoney, Ellison & Crutchfield 2009).
* **`is_epsilon_machine`**: whether the generating machine is its own ε-machine. When it is, the sampler's states are
  the true causal states, and the analyses that compare learned states with true ones can be used.

All demonstrations use renewal-15. After each 1, the number of 0s before the next 1 is equally likely to be any of 0–14.
Its 15 causal states are the counts of 0s since the last 1 (written a0 … a14; Marzen & Crutchfield 2015). To use another
process, give `process=` and `params=` to `TR.settings_for` (last line of the next cell).
""")
code(r"""
print("processes:", ", ".join(sorted(P.PROCESS)))
proc = P.Process("renewal", {"F": [1 / 15] * 15})
print(proc.label, "| its own epsilon-machine:", proc.is_epsilon_machine)
display(pd.Series({k: v for k, v in proc.analytics().items()
                   if k in ("n_causal", "statistical_c", "entropy_rate", "excess_entropy", "crypticity")}).round(4))
settings = TR.settings_for("2")              # section 16's settings for model 2, on renewal-15
fig = A.plot_true_machine(settings)
plt.show()
# another process, e.g.:  TR.settings_for("2", process="even", params={"p": 0.5})
""")

md(r"""
## 3. Choose the model

Every model chooses its states from the same kind of table (`causal_state_matrix.py`): K = 32 learned state vectors and
one output layer that turns a state vector into next-token probabilities. A state is chosen from scores. During training
random noise is added and the highest noisy score is taken; at evaluation the highest score is taken.

The models differ in which parts choose the states and in what they are trained on.

| model | parts | loss | idea |
|---|---|---|---|
| 1 | Transformer + GRU | the GRU's predictions of the 20 future tokens | the starting point (sections 13 and 16) |
| 2 | Transformer + GRU | 1 + the Transformer's own next-token predictions | train the Transformer's state choice at every position |
| 3 | + next-state network | 2 + weight × (entropy + CE) of a network predicting the Transformer's next state | if the states follow a fixed rule, the network can learn it |
| 4 | Transformer + network | 3 without the GRU | can the network replace the GRU? |
| A | as 3 | 3, with the Transformer's next state a fixed target | stop the Transformer being pulled toward easy states |
| 5a | Transformer + GRU + two-step head | 1 + one- and two-step predictions + weight × entropy of the next state worked out from them | the predictions themselves say which state comes next |
| 5b | Transformer + two-step head | 5a without the GRU | as 4, for this way |
| C | + three-step head | 5b with the next state worked out from two predicted tokens | two tokens tell states apart better |
| B | Transformer + GRU | 1 + weight × CE of the Transformer's starting state against the GRU's own state after the history | the GRU continues from a state it would itself be in |
| 1_L2 | Transformer + GRU | 1 + weight × D over all future tokens | the notebook's L2 term (sections 13 and 15) |
| 2_L2 | Transformer + GRU | 2 + weight × D over the GRU's steps | section 17's L2 term |
| feedback_GRU | a GRU alone | its own next-token predictions | the GRU's state is a table row, fed back every step |
| discrete_Transformer | a Transformer alone | its own next-token predictions | a state choice at every position |

D is the distance between the Transformer's and the GRU's state probabilities after the same tokens: 0 when equal,
at most √2.

**The predictors** are what a model's predictions are measured on. They all use the same output layer and differ only
in how the state at each future position is chosen:

* **`transformer`**: at every position the Transformer reads all the real tokens so far and picks a state afresh. It
  never has to carry a state forward.
* **`gru_from_handoff`**: the Transformer picks one state after the 40 history tokens (the code calls it *the
  handoff*). From then on the GRU moves from state to state on each real token.
* **`next_state_from_handoff`**: the same, but each step is made by the next-state network (models 3, 4, A).
* **`gru_from_start`**: the feedback GRU alone reads every token of the window from a blank start.

A low error for `transformer` only shows that the Transformer can tell, from the whole history, which state to use. A
low error for a predictor that steps from state to state shows that the states and the step rule work together as a
machine (state + token → next state), as the ε-machine does.

The table below lists, for every model file, the analyses of `analysis.py` that apply to it. The module docstring of
`analysis.py` explains each one. To make a new model, copy `model_2.py` to `model_<name>.py` and change it. It then
appears here, and in `python main.py list`.
""")
code(r"""
display(A.applicability())
""")

md(r"""
## 4. Train a model, or load a trained one

`TR.settings_for(model, **changes)` builds a run's settings in three layers:
1. section 16's setup:
   * windows of 40 + 20 tokens;
   * 50,000 updates, batch 128, learning rate 0.001;
   * temperature 2.5 → 0.5, multiplied by the same factor at every update;
   * LayerNorm on the Transformer's output, AdamW weight decay 0.0005, gradient clipping at 1;
2. then the model file's `DEFAULTS` (e.g. the weight of its extra term);
3. then your changes.

`TR.train(settings)` trains and saves the run in `runs/model_<name>/<process>_w<weight>_n<steps>_s<seed>_<key>/`. A
finished run with the same settings and code is reloaded instead. The folder holds:
* `settings.json`;
* `completed.pt` (with the optimizer and random states);
* `best.pt` (the parameters with the best validation value);
* the training and validation histories.

Three more functions:
* `TR.extend(folder, n)` trains n more updates at the final temperature.
* `TR.train_until_settled` repeats `extend` by 25,000 while the main measure still improves, up to 100,000 updates
  (section 16's rule).
* `TR.load_run(folder)` reads a run back.

From a terminal: `python main.py train --model 2 --seed 0`.

The next cell trains model 2 for 300 updates, only to show the steps; that is far too few to learn the process.
""")
code(r"""
example = TR.settings_for("2", steps=300, eval_every=100)
model, folder = TR.train(example, verbose=False)
print("saved in", folder.relative_to(Path.cwd()))
display(TR.history(folder)[["update", "tau", "excess_transformer_future", "excess_gru_from_handoff_future",
                            "transformer_states_used"]].round(4))
""")

# Section 4 protocols: Markdown examples, not additional executable notebook cells.
md(r"""
Section 4 connects **the process that generates data, the training settings, the model, and a folder that records
the run**. The protocols below are copyable examples: they are displayed as Markdown, so running the notebook does
not execute them. The 300-update example above remains the executable training demonstration. Use the `qdrug`
environment or notebook kernel, with `test_models/` as the working directory.

```text
Choose process + parameters + window lengths
    -> choose model + training settings
    -> determine the run folder
    -> generate a batch -> calculate loss -> update parameters (repeat)
    -> save checkpoints and metric histories
    -> load, evaluate, plot, or extend
```

The raw training sequences are generated during training. `runs/` stores settings, checkpoints and measurements;
it does not store the sequence batches themselves.

Your project's run directory is
`/Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version/test_models/runs`. In code, use `TR.RUNS` to refer
to it. A different root can be selected with the `TEST_MODELS_RUNS` environment variable **before importing
`training`**.

```text
runs/
├── imported/                 Experiments copied from the old project
│   ├── index.json
│   ├── model_1/
│   └── ...
├── model_2/                  Runs created by the current training code
│   └── renewal-15_w0_n300_s0_<hash>/
├── model_5b/
├── model_B/
├── ...
└── logs/                     Output logs from main.py train-many
```

For a newly created run, `renewal-15_w0_n300_s0_<hash>` encodes the process label, extra-loss weight 0, 300 planned
updates and seed 0. The final hash depends on all settings and the tracked source files: the selected model,
`training.py`, `process.py` and `causal_state_matrix.py`. Changing batch size, learning rate, seed or one of those
files normally produces a different folder. The exact process parameters are in `settings.json`; the short process
label does not encode every parameter. Imported runs keep the importer's older naming convention.

| File in a new run | What it contains | When it is written |
|---|---|---|
| `settings.json` | Process, process parameters, model and optimizer settings, seeds, code fingerprint | At the start |
| `validation_history.csv` | Held-out metrics, temperature, timing and recorded gradient norms | At update 0, every `eval_every` updates, and the final update |
| `progress.json` | Most recent validation record | At each evaluation during the initial training |
| `best.pt` | Parameters with the best main validation metric | Whenever that metric improves |
| `training_history.csv` | Per-update training loss, its components and temperature | After the initial training finishes |
| `completed.pt` | Final parameters, optimizer state, settings, sampler state and PyTorch random state | After the initial training finishes |
| `report_latest.json` / `report_best.json` | Detailed held-out evaluation results | When analysis requests the corresponding report |

At each update the training loss is accumulated in memory. The validation CSV and `progress.json` are refreshed at
evaluation points, but the initial training-loss CSV is written only at completion. A folder with `settings.json`
and some validation records can therefore be unfinished. `completed.pt` marks completion of the initial schedule.
Training extensions add their own files, described below; `progress.json` is not refreshed by `TR.extend()`.
""")

md(r"""
The **first protocol is to define and start one controlled run**. Define the data source and optimization settings
separately so that it is clear which change affects the experiment. These variable names are separate from the
walkthrough's existing `settings`, `model` and `folder` variables.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import training as TR
import process as P
import analysis as A

data_options = dict(
    process="renewal",
    params={"F": [1 / 15] * 15},
    history_len=40,
    rollout_len=20,
    context=19,
)

train_options = dict(
    steps=5_000,
    batch_size=64,
    lr=1e-3,
    weight_decay=5e-4,
    grad_clip=1.0,
    tau_start=2.5,
    tau_end=0.5,
    eval_every=250,
    validation_rows=512,
    seed=0,
)

protocol_settings = TR.settings_for("2", **data_options, **train_options)
display(pd.Series(protocol_settings.to_dict()))
print("Will save to:", TR.run_folder(protocol_settings))

# This line starts training, or loads the matching completed run.
protocol_model, protocol_folder = TR.train(protocol_settings)
```

`settings_for()` applies general `Settings` defaults, then the selected model file's `DEFAULTS`, then your explicit
arguments. Your values take precedence. The example changes the default batch size from 128 to 64; settings that
are not specified keep their model/default values.

| Control | Meaning |
|---|---|
| `"2"` in `settings_for("2", ...)` | Selects `model_2.py` |
| `process`, `params` | The statistical source of the sequences |
| `history_len` | History tokens before the hybrid's handoff |
| `rollout_len` | Future tokens predicted after that history |
| `context` | Position from which the Transformer's own next-token predictions are scored |
| `batch_size` | Independent windows used for one optimizer update |
| `steps` | Number of optimizer updates |
| `lr` | Learning rate |
| `weight` | Coefficient of a model's extra loss term, where applicable |
| `weight_decay`, `grad_clip` | Optimizer regularization and gradient clipping |
| `tau_start`, `tau_end` | Start and end of the state-selection temperature schedule |
| `eval_every`, `validation_rows` | Validation frequency and number of validation windows |
| `n_states`, `state_dim` | Available learned states and their vector dimension |
| `d_model`, `n_layers` | Transformer size, for models that use it |
| `seed` | Run seed; controls model initialization and training-window randomness |

`weight` and `weight_decay` are separate controls. A model's extra-loss coefficient does not select the optimizer's
weight decay. The process's true number of states and the model's `n_states` capacity are also separate quantities.

In this example every update draws a tensor of shape `(64, 60)`: 64 independent windows, each with 40 history tokens
and 20 future tokens. The data-generation part of the loop is:

```python
# Illustration of the sampler calls inside TR.train(), not a second training loop.
sampler = protocol_settings.process_object().sampler(protocol_settings.train_seed)
for step in range(protocol_settings.steps):
    batch = sampler.draw(protocol_settings.batch_size, protocol_settings.total_len)
    # TR.train() then computes the loss, backpropagates, and updates the model.
```

There is no fixed training dataset that gets revisited for multiple epochs. This configuration draws
5,000 × 64 = **320,000 windows**. Each window starts from the generating machine's stationary state distribution;
the rows are not consecutive chunks of one long sequence. Random sampling can produce repeated token sequences.
True generating-state IDs are used for analysis, not as training labels.

By default the temperature schedule spans `steps`. The executable 300-update example above therefore changes
temperature from 2.5 to 0.5 within those 300 updates. To study only the beginning of a longer schedule, pass, for
example, `schedule_steps=50_000`; it must cover the requested initial updates. `TR.extend()` instead uses the fixed
final temperature `tau_end`.
""")

md(r"""
The **second protocol is to choose and inspect a different data source before training**. List the available models
and processes with:

```python
print(TR.available_models())
print(sorted(P.PROCESS))
```

Some concrete process configurations are:

| Data source | `process` | `params` |
|---|---|---|
| Uniform renewal, up to 14 zeros between ones | `"renewal"` | `{"F": [1 / 15] * 15}` |
| Uniform renewal, up to 59 zeros between ones | `"renewal"` | `{"F": [1 / 60] * 60}` |
| Even process | `"even"` | `{"p": 0.5}` |
| Flower: 8 dice, 12 faces | `"flower"` | `{"n": 8, "m": 12, "dice_seed": 0}` |
| The same flower read backward | `"flower"` | `{"n": 8, "m": 12, "dice_seed": 0, "direction": "backward"}` |

For renewal, `F[n]` specifies the probability of exactly `n` zeros between successive ones (the implementation
normalizes the supplied weights). For the flower, `dice_seed` fixes the randomly drawn face probabilities of the
dice. Changing it changes the process being studied.

Using `data_options` and `train_options` from the first protocol, switch to the flower and inspect four windows:

```python
flower_data = {
    **data_options,
    "process": "flower",
    "params": {"n": 8, "m": 12, "dice_seed": 0},
}
flower_settings = TR.settings_for("2", **flower_data, **train_options)
flower_process = flower_settings.process_object()

# An independent random stream for this preview.
preview, generating_states = flower_process.sampler(9001).draw(
    batch_size=4,
    total_len=flower_settings.total_len,
    states=True,
)
print("Vocabulary size:", flower_process.vocab_size)
print("Token shape:", preview.shape)
print(preview)

# Uncomment in your own code cell to train this configuration.
# flower_model, flower_folder = TR.train(flower_settings)
```

The preview sampler does not advance the training sampler: `TR.train()` creates its own sampler. `vocab_size` is
filled automatically from the chosen process. Changing the process does not automatically increase window lengths
or model capacity. For renewal-60, the walkthrough explicitly chooses:

```python
renewal60_settings = TR.settings_for(
    "2",
    process="renewal",
    params={"F": [1 / 60] * 60},
    history_len=120,
    rollout_len=60,
    context=59,
    n_states=128,
    **train_options,
)
```

The run seeds control different random streams:

* Starting model weights use `model_seed_base + seed` (default base 17).
* Training windows use `train_seed_base + seed` (default base 1101).
* Validation uses `validation_seed` (default 2202), which stays fixed across run seeds by default.
* Flower's `dice_seed` belongs to the process parameters and is independent of these run seeds.

Set `model_seed_base` or `train_seed_base` explicitly when you want to vary initialization and sampled data
separately. Matching the process, training seed, batch size and window length across models gives them the same
training-window stream under this sampler. To repeat an experiment with seed 1, build a new settings object using
`**{**train_options, "seed": 1}`.

The command-line equivalent accepts repeated `--set` arguments for any settings fields:

```bash
python main.py train \
  --model 2 \
  --process even \
  --params '{"p": 0.5}' \
  --steps 5000 \
  --seed 0 \
  --set batch_size=64 \
  --set history_len=40 \
  --set rollout_len=20 \
  --set lr=0.001 \
  --set eval_every=250
```
""")

md(r"""
The **third protocol is to choose between a new experiment and continuing an existing run**. With the default
automatically generated folder, these actions have different effects:

| Action | Effect |
|---|---|
| `TR.train(settings)` with the same settings/code and an existing `completed.pt` | Loads the saved run, using its latest checkpoint |
| Change a setting and call `TR.train(new_settings)` | Normally creates a separate folder and starts from initialization |
| `TR.extend(folder, extra_steps=...)` | Continues the saved run inside the same folder |
| Rerun an initial run that has no `completed.pt` | Starts the initial training again; automatic resumption from an interrupted update is not implemented |
| `TR.train(settings, force=True)` | Retrains and overwrites initial-run files in that folder; this is not continuation |

Increasing `steps` in a new settings object creates a different experiment. To add 1,000 updates to the first
protocol's completed 5,000-update run, use its returned folder:

```python
print("Before:", TR.checkpoint(protocol_folder, which="latest"))
protocol_model = TR.extend(protocol_folder, extra_steps=1_000)
print("After:", TR.checkpoint(protocol_folder, which="latest"))
```

`extend()` restores parameters, optimizer state, the training sampler's random state and PyTorch's random state.
It continues at `tau_end` and writes these additional files after the extension completes:

```text
extended_6000.pt
training_history_extended_6000.csv
validation_history_extended.csv
extensions.json
```

The extended validation CSV is also updated during extension evaluations, and `best.pt` is replaced whenever the
main validation metric improves. The original `completed.pt` stays available. The folder name and `settings.steps`
still describe the original 5,000-update schedule; `TR.checkpoint(folder, "latest")` gives the current checkpoint and
total update count. Another 1,000-update extension would produce `extended_7000.pt` and its training-loss CSV.

To load rather than train:

```python
loaded_model, loaded_settings, loaded_update = TR.load_run(protocol_folder, which="latest")
```

`"completed"` selects the end of the original schedule, `"latest"` includes the latest recorded extension, and
`"best"` selects the checkpoint with the best main validation metric. For model 2 that metric is the GRU's future
excess CE. `TR.train_until_settled(settings)` can automate extensions using the validation-improvement rule documented
above. Imported checkpoints can be extended only if their original files included the required optimizer and
random-generator states; not every imported run has those fields.
""")

md(r"""
The **fourth protocol is to analyse saved training and evaluation records**. You can use the `folder` returned by
the executable 300-update example, `protocol_folder` from the optional training protocol, or an explicitly selected
existing path. The following defaults to the already executed demonstration:

```python
import json

target = Path(folder)  # Change to Path(protocol_folder) to inspect the optional 5,000-update run.
saved_settings = TR.read_settings(target)
display(pd.Series(saved_settings.to_dict()))

validation = TR.history(target)
metric = TR.main_metric(saved_settings.model)
display(validation[["update", "tau", metric]].tail())
validation.plot(x="update", y=metric, ylabel="Validation excess CE (bits/token)")
plt.show()
```

**`TR.history()` returns validation measurements**, including any extension validation history. It does not return
the per-update training losses. Read those from the training CSVs:

```python
metadata = json.loads((target / "settings.json").read_text())
training_paths = list(target.glob("training_history*.csv"))

# Imported runs keep references to the original, potentially large training CSVs.
if not training_paths:
    training_paths = [Path(p) for p in metadata.get("training_histories", [])]
training_paths = [p for p in training_paths if p.exists()]

if training_paths:
    training_history = pd.concat(
        [pd.read_csv(p) for p in training_paths], ignore_index=True,
    ).sort_values("update")
    display(training_history.tail())
    print("Available columns:", training_history.columns.tolist())
    training_history.plot(x="update", y="loss")
    plt.show()
else:
    print("No accessible training-history CSVs.")
```

For a new model 2 run, the training columns include `update`, `tau`, `ce_gru`, `ce_t1` and `loss`. Its loss combines
the two prediction losses. Validation excess CE subtracts the true process's cross-entropy on held-out tokens, so
its numerical value has a different meaning from the training loss. Imported CSV column names may follow the old
code's naming; inspect the available columns before selecting loss components.

To discover runs without remembering their hashes:

```python
found = TR.find_runs(model="2", process="renewal", steps=5_000, batch_size=64)
for path, run_settings in found.items():
    print(path, "seed =", run_settings.seed)

# find_runs returns {folder: Settings}; test_table expects {display_name: folder}.
selected = {path.name: path for path in found}
if selected:
    display(A.test_table(selected, which="latest"))

# Imported runs are under a separate root.
imported_found = TR.find_runs(root=TR.RUNS / "imported", model="2")
```

By default `find_runs()` includes only folders with `completed.pt`. Pass `finished=False` to include unfinished runs.
Its filters match saved settings, so `steps=5_000` means the original planned schedule even if the run was later
extended. Runs with different code hashes can coexist and match the same settings filters.

For one run's full held-out report:

```python
display(A.test_table({"my run": target}, which="latest"))
```

`test_table()` loads the saved model and reuses the cached report when its update and report version match.
Otherwise it evaluates the model and saves a report; this can take time but does not train it. The notebook's
`performance({"my run": target})` helper then selects a smaller set of columns, indexes by seed and rounds to four
decimal places. Each run remains a separate row.
""")

md(r"""
The **fifth protocol is to reconstruct and analyse the actual token sequences**. For runs produced by this
`training.py`, reconstruct past batches from the saved process, parameters, training seed, original batch size,
window length and sampler-call order. The checkpoint's sampler state is the state at its saved update: it supports
continuation from that point, rather than storing all preceding token batches.

The example below reconstructs only the first ten batches of the executable demonstration. It also requests
generating-state IDs for analysis; `states=True` does not change which tokens the sampler draws.

```python
import numpy as np

target = Path(folder)  # Or another run produced by the current training sampler.
replay_settings = TR.read_settings(target)
replay_sampler = replay_settings.process_object().sampler(replay_settings.train_seed)
token_batches, state_batches = [], []

for _ in range(min(10, replay_settings.steps)):
    tokens, states = replay_sampler.draw(
        replay_settings.batch_size,
        replay_settings.total_len,
        states=True,
    )
    token_batches.append(tokens.numpy())
    state_batches.append(states)

replayed_tokens = np.stack(token_batches)
replayed_states = np.stack(state_batches)
print("Tokens:", replayed_tokens.shape)
print("Generating states:", replayed_states.shape)

export_dir = Path("exports") / target.name
export_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    export_dir / "first_training_batches.npz",
    tokens=replayed_tokens,
    generating_states=replayed_states,
)
```

For the 300-update demonstration, token shape is `(10, 128, 60)`: update, window, token position. State shape is
`(10, 128, 61)`. `states[..., j]` is the generating machine's state **before** token `j`; the extra last entry is its
state after the final token. These generating states are causal states when the process's generating machine is its
epsilon-machine (`process_object().is_epsilon_machine`). The model sees the tokens, not these state IDs.

Inspect a window and the empirical token frequencies:

```python
first_window = replayed_tokens[0, 0]
print("History:", first_window[:replay_settings.history_len])
print("Future: ", first_window[replay_settings.history_len:])

counts = np.bincount(replayed_tokens.ravel(), minlength=replay_settings.vocab_size)
display(pd.Series(counts / counts.sum(), name="observed_frequency"))
```

To reproduce the batch at update 100, restart from the original seed and advance through 99 batches:

```python
replay_sampler = replay_settings.process_object().sampler(replay_settings.train_seed)
for _ in range(99):
    replay_sampler.draw(replay_settings.batch_size, replay_settings.total_len)
batch_100 = replay_sampler.draw(replay_settings.batch_size, replay_settings.total_len)
```

Keep the original batch size, window length and call order. One giant draw does not reproduce the same arrangement
of random draws. Exact reconstruction also requires compatible process/sampler code and random-generator behavior;
the saved settings and code fingerprint record provenance but do not archive the source code. Imported experiments
must be checked against their original sampler, especially the older Section 13 experiment. This protocol
reconstructs input windows, not the full optimizer trajectory. For a complete training-data export, write batches
incrementally instead of retaining every batch in memory.

Validation and test windows have direct reconstruction helpers and use separate streams from training:

```python
validation_tokens, validation_states = TR.validation_windows(replay_settings)
test_tokens, test_states, exact_ce = A.windows(replay_settings, "test")
```

The validation windows use the run's own `validation_rows` and `validation_seed`. The test set currently uses
4,096 windows and the fixed test seed defined in `analysis.py`. `exact_ce` contains the true process's token-level
cross-entropies used to calculate excess CE. These helpers let you inspect or export held-out data separately from
the batches used for parameter updates.
""")
# End Section 4 protocols.

md(r"""
The trained runs of sections 13, 16 and 17 are in `runs/imported/` (`MAIN.imported_runs()`):
* models 1–C: three seeds each, section 16;
* model 1 in the original setup: three seeds, section 16;
* model 2 + L2 at weights 0.01, 0.03 and 0.1, the feedback GRU and the discrete Transformer: three seeds each, section 17;
* model 1 + L2: one run, section 13.

Each was checked against the old code: the same settings, the same parameters, and the same test results
(`tests/test_equivalence.py`).
""")
code(r"""
display(pd.DataFrame([dict(model=m, label=l, seed=s, update=TR.checkpoint(f)[1]) for (m, l, s), f in runs.items()])
        .pivot_table(index=["model", "label"], columns="seed", values="update"))
""")

md(r"""
## 5. Choose the plots

| function | shows | models |
|---|---|---|
| `A.plot_learning_curves(runs)` | the validation excess CE of every predictor over training, with the two reference levels | all |
| `A.plot_states_over_training(runs)` | states in use by each predictor, and the gap between the two highest state scores at the last history position (above about 10 the choice no longer changes in training) | all |
| `A.plot_extra_terms(runs)` | the model's extra measures over training: next-state entropy (3, 4, A), inferred-next-state entropy (5a, 5b, C), copy agreement (B), D (L2 models) | models with an extra term |
| `A.plot_test_summary(A.test_table(runs), levels)` | every predictor's test excess CE, per seed | all |
| `A.plot_token_probs_by_true_state(runs)` | each predictor's predicted P(next token = 1) at each true state, against the truth | all (true states needed) |
| `A.plot_state_tables(runs)` | which learned state each true state is sent to | all (true states needed) |
| `A.plot_machines(runs)` | the learning curves, and for each run the machine of its main predictor, its transition matrix on the true states, and the difference from the true matrix | all (true states needed) |
| `A.plot_machine_summary(A.machine_table(runs), levels)` | every run's machine size, true states covered, mean difference, test excess | all |
| `A.lock_in_trace(settings)` | the first updates of a run, measured after each (section 16.1) | hybrids |

**Reference levels.** Always predicting each token's long-run frequency is 0.0549 bits above exact on the test windows
("remembers nothing"). Predicting from the last token alone is 0.0513 ("last symbol only"). A predictor near these
levels does not use the count.

**How a machine is read off.**
* *From a step rule* (the GRU, the next-state network, the feedback GRU): every (state, token) pair is tried. The result
  is reduced to the states it keeps returning to from the model's states after the history, and identical states are
  merged. This is exact.
* *From a state sequence* (the Transformer): the machine is read off the states picked on 4,096 windows, keeping how
  often each next state follows. It need not be deterministic.

The overview of all imported runs follows.
""")
code(r"""
overview = {k: f for k, f in runs.items() if k[1] in ("section16", "section17", "original_setup", "w0.03")}
levels = A.reference_levels(settings, "test")
fig = A.plot_learning_curves(list(overview.values())); plt.show()
fig = A.plot_test_summary(A.test_table(overview), levels); plt.show()
fig = A.plot_machine_summary(A.machine_table(overview), levels); plt.show()
""")

md(r"""
## 6. Demonstrations, one section per model

Each section shows three things for the model's runs:
* **Performance**: test excess CE of each predictor, over 20 and over 200 future tokens.
* **Convergence**: learning curves, states in use, and when the main predictor first fell below 0.04 / 0.02 / 0.01 bits.
* **Properties**: the machine each run ends with, its transitions against the true ones, and what its states stand for.
""")

for section in FINDINGS:
    md(section["markdown"])
    code(section["code"])
    if section.get("after"):
        md(section["after"])

for module in (S7, S10):
    for kind, text in module.CELLS:
        (code if kind == "code" else md)(text)
    for text in getattr(module, "FINDINGS", []):
        md(text)

md(r"""
## References

* Chen, X. & He, K. (2021). Exploring Simple Siamese Representation Learning. CVPR 2021. arXiv:2011.10566.
* Crutchfield, J. P., Ellison, C. J. & Mahoney, J. R. (2009). Time's Barbed Arrow: Irreversibility, Crypticity, and
  Stored Information. Physical Review Letters 103, 094101. arXiv:0902.1209.
* Gloeckle, F. et al. (2024). Better & Faster Large Language Models via Multi-token Prediction. ICML 2024. arXiv:2404.19737.
* Hinton, G., Vinyals, O. & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
* Jang, E., Gu, S. & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. ICLR 2017. arXiv:1611.01144.
* Mahoney, J. R., Ellison, C. J. & Crutchfield, J. P. (2009). Information Accessibility and Cryptic Processes. arXiv:0905.4787.
* Marzen, S. E. & Crutchfield, J. P. (2015). Informational and Causal Architecture of Discrete-Time Renewal Processes.
  Entropy 17(7), 4891–4917. arXiv:1408.6876.
* Shalizi, C. R. & Crutchfield, J. P. (2001). Computational Mechanics: Pattern and Prediction, Structure and Simplicity.
  Journal of Statistical Physics 104, 817–879.
* Tang, Y. et al. (2023). Understanding Self-Predictive Learning for Reinforcement Learning. ICML 2023. arXiv:2212.03319.
* Thompson, J., Garner, A. J. P., Mahoney, J. R., Crutchfield, J. P., Vedral, V. & Gu, M. (2018). Causal Asymmetry in
  a Quantum World. Physical Review X 8, 031013. arXiv:1712.02368.
""")

nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "qdrug", "language": "python",
                                                                     "name": "python3"}})
start = time.time()
NotebookClient(nb, kernel_name="python3", timeout=7200,
               resources={"metadata": {"path": str(OUT.parent)}}).execute()
nbformat.write(nb, OUT)
print(f"executed and written {OUT} in {time.time() - start:.0f} s")
