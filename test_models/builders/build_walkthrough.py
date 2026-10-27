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
during training, and what machine it ends up with. Section 7 trains five of the models on three new processes:
renewal-60, and the flower process read forward and backward. The demonstrations use the trained runs of sections 13, 16 and 17 of
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

for kind, text in S7.CELLS:
    (code if kind == "code" else md)(text)
for text in getattr(S7, "FINDINGS", []):
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
