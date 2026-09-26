# model_modeularised — discrete causal-state models

Two sequence models are trained on data from a hidden-Markov process. Each is forced to
predict through a small set of **discrete states**. Those states are then compared with the
process's theoretical **causal states** (its ε-machine), in both reading directions:
the forward arm (predict the next token) and the backward arm (predict the previous token).
The research question is causal asymmetry: forward vs backward memory, C⁺ vs C⁻
([Thompson et al. 2018](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031013)).

The main models are the **transformer** and the **feedback GRU**. The feedback GRU trains
with Gumbel-softmax sampling. Both use one shared causal-state head, one training loop, the
same data and the same scoring. Every theoretical quantity comes from `HMM_processes.py`.

---

## 1. Quick start

Use the `qdrug` conda environment (python 3.11, torch 2.12, lightning 2.6). Run everything
from this folder.

```bash
scripts/launch_grid.sh default                   # the default grid in parallel -> results/default/
python run_model.py --name default               # the same grid, one run after another
python run_model.py --name default --arch gru_feedback --process coin --arm backward   # one run
python run_model.py --name default --plots-only  # redraw every figure from the saved runs
python run_model.py --smoke                      # 2-epoch plumbing check -> checks/results/smoke/
pytest tests/ -q                                 # unit tests
```

The default grid is `transformer` and `gru_feedback`, on `coin` and `flower`, in the
`forward` and `backward` arms, with seed 0. Choose other settings with `--arch`,
`--process`, `--arm`, `--seed` and `--params`. `launch_grid.sh` takes `ARCHS`,
`PROCESSES`, `ARMS`, `SEED` and `PY` from the environment:

```bash
PROCESSES="even rip" SEED=1 scripts/launch_grid.sh my_run
```

Run times (Apple M-series; 25 updates per epoch):

| model | device | standard length | time per run |
|---|---|---|---|
| transformer | MPS | 1000 epochs = 25,000 updates | ~20 min |
| feedback GRU | CPU, 1 thread | 800 epochs = 20,000 updates | ~30–40 min |

`launch_grid.sh` runs every GRU as its own process and all transformers in one lane.

## 2. Folder layout

```
model_modeularised/
  run_model.py            entry point for real runs: config (RunConfig), data, train, score, save, plot
  scripts/launch_grid.sh  a whole grid in parallel, one log per run
  Transformer.py  GRU.py  the architectures
  causal_matrix.py        the shared discrete causal-state head
  decoder_base.py         loss, training step, optimiser, shared by every model
  training.py             train_model, Recorder (curves, best checkpoint), CE helpers
  tau_schedue.py          the straight-through temperature schedule
  extraction.py           states, transition matrices, empirical/exact machines, scoring
  theory.py               a process's closed forms, from HMM_processes.py
  plotting.py             every figure
  HMM_processes.py        the processes and their theory (epsilon-machines, entropies, ...)
  process_generator.py    one process as an object: sampling, reverse(), exact predictions
  tests/                  pytest unit tests
  checks/                 verification and diagnosis scripts (section 8)
  results/<name>/         REAL results only, one folder per experiment (git-ignored)
  checks/results/         everything the checks write: smoke/, crosscheck/, diagnostics/, ablations/
```

**What a run needs:** only the 12 `.py` files at the top level. It needs none of `tests/`,
`checks/` or `scripts/`, and nothing outside this folder. This was verified by running from a
copy that holds those 12 files alone. The packages are numpy, scipy, torch, lightning,
matplotlib and networkx (plus pytest for `tests/`).
- `scripts/` is a convenience: it runs `run_model.py` in parallel.
- `tests/` re-verifies the code after you change it.
- `checks/` holds the verification and diagnosis behind the design, and its evidence.
  Two of its scripts need `Experimental_pipeline/` or `tau_experiment/`.

Each experiment `results/<name>/` contains:

```
runs/<arch>_<process>_<arm>_s<seed>.pkl     everything a run produced, including the weights
runs/<arch>_<process>_<arm>_s<seed>.json    its config and headline metrics, readable
logs/<arch>_<process>_<arm>_s<seed>.log     (launch_grid.sh)
figures/scorecard_s<seed>.png               every run of one seed on one page
figures/<process>/theory.png                the process's theory (section 6)
figures/<process>/theory_comparison_s<seed>.png   every model against that theory
figures/<process>/<arch>/<arm>_s<seed>_{training,states,machine}.png
figures/<process>/<arch>/s<seed>_{per_position,transitions,complexity}.png
                                            (+ s<seed>_transitions_exact.png for gru_feedback)
```

Rule: `results/` only ever holds runs of the real pipeline. Smoke runs, cross-checks,
diagnostics and ablations write under `checks/results/`.

## 3. The data and the two arms

`process_generator.generator` draws `num_samples` sequences of `seq_len + 1` tokens,
at stationarity after a burn-in. `SequenceDataset` turns each sequence into
`(input, target) = (x[:-1], x[1:])`. Every model sees the same draw and the same
seeded 80/20 split; each run records the data's SHA-1.

- **Forward arm:** the sequences as drawn. The model predicts the next token, and its
  states are compared with M⁺, the forward ε-machine.
- **Backward arm:** the same sequences reversed by `generator.reverse()`. That call also
  switches the theory to M⁻, the reverse ε-machine
  ([Ellison, Mahoney & Crutchfield 2009](https://arxiv.org/abs/0905.3587)). The model still
  reads left to right (`mode="forward"`), so its "next token" is the previous token of
  the original process.

  Every model also has a `mode="backward"` (a `triu` mask, or reading right to left).
  Never combine it with reversed data: the two flips cancel and give the forward arm again.

## 4. How each architecture works

### Common interface

| | |
|---|---|
| **input** | `tokens`: a `(B, T)` long tensor with values in `[0, V)` |
| **output** | logits `(B, T, V)`; position `t` predicts token `t + 1` in the reading order |
| **causal states** | `model.state_assignment(tokens)` gives `(B, T)` state indices; `model.emission_table()` gives `(K, V)` = P(next token \| state) |
| **loss** | cross-entropy in bits plus `usage_beta · H(p̄)`, where p̄ is the batch's state occupancy (pushes towards fewer states) |

### The shared causal-state head (`causal_matrix._learn_causal_state`)

```
value vector v_t (D) ─ state_head ─► logits over K states ─► s_t = argmax      (one-hot, straight-through)
                                                               │
                                              state_matrix[s_t] (S) ─ emission ─► token logits (V)
```

- The forward value is always an exact one-hot. Gradients reach the logits through
  softmax(logits / τ): the straight-through estimator
  ([Bengio et al. 2013](https://arxiv.org/abs/1308.3432)).
- τ follows `geom:5:0.5` over the run (`tau_schedue.py`). It changes the gradient, never
  which state is chosen.
- With `gumbel` on (training only), the state is **sampled** as argmax(logits + Gumbel noise),
  i.e. drawn from softmax(logits)
  ([Jang et al. 2017](https://arxiv.org/abs/1611.01144),
  [Maddison et al. 2017](https://arxiv.org/abs/1611.00712)). A state that is losing still
  gets chosen, and so trained, now and then. Evaluation always uses the plain argmax.

### Transformer (`Transformer.DiscreteCausalDecoder`)

```
tokens ─► one_hot · rand_prj (fixed, V×16) + sinusoidal PE ─► 4 × [pre-LN single-head attention + pre-LN FFN(4·16)]
       ─► x_t (16) at every position ─► shared head ─► logits
```

- A causal (`tril`) mask means position `t` sees tokens `0..t`
  ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762); pre-LN as in
  [Xiong et al. 2020](https://arxiv.org/abs/2002.04745)).
- The state is **read off** the top layer at each position and never fed back. It depends
  on the whole prefix, and unifilarity is not guaranteed; it is measured (`determinism`).
- Size: 12,164 (coin) to 12,424 (flower) parameters. Trained with Gumbel off. This is
  verified bit-identical to `Experimental_pipeline`'s transformer (section 8).

### Feedback GRU (`GRU.DiscreteFeedbackGRU`, `--arch gru_feedback`)

```
h_t = GRUCell(one_hot(x_t), m_{t-1})        m_{-1} = 0 (blank start)
s_t = shared head's state for h_t           (Gumbel sample while training)
m_t = state_matrix[s_t]                     fed back as the next hidden state
logits_t = emission(m_t)                    predicts x_{t+1}
```

- The only memory carried forward is the discrete state, one of K values. So
  `s_t = f(s_{t-1}, x_t)`: a **unifilar machine by construction**
  ([Cho et al. 2014](https://arxiv.org/abs/1406.1078) for the GRU; a discretised
  recurrent state as in [Zeng et al. 1993](https://direct.mit.edu/neco/article-abstract/5/6/976/5759/Learning-Finite-State-Machines-With-Self?redirectedFrom=PDF)
  and [Koul et al. 2019](https://arxiv.org/abs/1811.12530)).
- `model.machine()` returns the **exact** machine by enumerating every (state, token):
  `next_state[s, x]`, `emission_probs[s]`, `start_state[x]`.
- `state_dim` must equal `d_model` (16), because the state vector *is* the hidden state.
- Size: 1,257 (coin) to 1,781 (flower) parameters.

### Read-out GRU (`GRU.DiscreteCausalGRU`, `--arch gru`)

Kept for comparison: `nn.GRU` over the sequence, with the state read off h_t and never
fed back. It is the transformer's design with a GRU encoder. Even with Gumbel it finds
only 3 of flower backward's 5 states. Removing the usage penalty recovers all 5, but then
over-splits coin backward (see `checks/results/crosscheck/summary.md` and
`checks/results/diagnostics/`).

### Using a model directly

```python
import torch, GRU, Transformer
m = GRU.build_feedback_model("discrete", token_size=3, d_model=16, max_len=300, lr=1e-3,
                             mode="forward", n_layers=1, weight_decay=0.01,
                             n_states=6, state_dim=None, tau=1.0, usage_beta=1e-4)
m.gumbel = True                            # training only; evaluation is always argmax
logits = m(torch.randint(0, 3, (4, 50)))   # (4, 50, 3)
states = m.state_assignment()              # (4, 50)
machine = m.machine()                      # exact: next_state (6,3), emission_probs (6,3), start_state (3,)
```

Or train and score one arm as `run_model.py` does:
`res = run_model.run_arm(RunConfig(), "gru_feedback", "coin", "backward")`.

## 5. Training and what is measured

`RunConfig` in `run_model.py` holds every setting.

| setting | value |
|---|---|
| data | 1000 × 300 tokens, batch 32 |
| width | `d_model` 16 |
| states | K = 2V; S = V for the transformer, 16 for the feedback GRU |
| τ | transformer `geomhold:5:0.5:0.8`; GRUs `geom:5:0.5` (or `--tau`, below) |
| optimiser | AdamW, lr 1e-3, weight decay 0.01 ([Loshchilov & Hutter 2019](https://arxiv.org/abs/1711.05101)) |
| usage penalty | β = 1/(batch · seq_len) |
| length | transformer 1000 epochs; GRUs 800 epochs |
| checkpoint | best validation CE (checked every 25 steps) |
| Gumbel | on for the GRUs, off for the transformer |

### Changing the schedule, the length or the budget

```bash
python run_model.py --name gh --tau geomhold:5:0.5                       # descend 5 -> 0.5 over the first half, hold 0.5
python run_model.py --name gh25 --tau geomhold:5:0.5:0.25 --max-epochs 800   # descent over the first quarter
python run_model.py --name c1 --tau const:1                             # constant
scripts/launch_grid.sh gh --tau geomhold:5:0.5                           # a whole grid with it
```

`--tau` and `--max-epochs` set every architecture. `--tau-transformer` and
`--max-epochs-transformer` set the transformer alone. In Python, `RunConfig(tau=...,
max_epochs=...)` sets the GRUs and `RunConfig(tau_transformer=..., max_epochs_transformer=...)`
the transformer. `run_model.arch_settings(cfg, arch, V)` shows what a run will use.

| `--tau` spec | schedule (f = fraction of training done) |
|---|---|
| `X`, `const:X` | constant X |
| `geom:A:B` | geometric A → B over the whole run (the default is `geom:5:0.5`) |
| `geomhold:A:B[:DESCENT]` | geometric A → B over the first DESCENT fraction of the run (default 0.5), then held at B for the rest |

**The fourth number is how long the descent takes, not how long the hold lasts.** The
transformer's standard is `geomhold:5:0.5:0.8` over 1000 epochs: τ descends 5 → 0.5
during the first 800 epochs, then holds at 0.5 for the last 200. `geomhold` is
`tau_experiment/study5`'s rule. On flower (3,8) backward, 5 → 0.5 descended
over 2000 epochs and then held for 2000 more gave 9, 8, 8, 8, 7 of 9 states. The plain
2000-epoch descent averaged 7.1. The extra updates helped only once τ was low.

The schedule follows the training fraction, so `--max-epochs` rescales it: at 800 epochs,
`geomhold:5:0.5:0.5` descends over 400 epochs and holds for 400. `--num-states-mult`
sets the budget K (section 6).

Metrics for each run (`runs/*.json`, and the scorecard):

| metric | meaning |
|---|---|
| **discovered** | theoretical states matched by some learned state's next-token row (nearest in total variation) |
| **FULL** | every theoretical state discovered **and** \|S_emp − C\| < 0.05 bits |
| S_emp | entropy of the learned states' occupancy along the data; compared with C⁺ or C⁻ |
| emission TV | occupancy-weighted distance from each learned row to its matched theory row |
| T err | max \|learned − theory\| of the state-to-state matrix (from free-running generation; `exact_T_err` from enumeration for `gru_feedback`) |
| CE − exact | test cross-entropy minus the exact model's on the same tokens |
| determinism | fraction of (state, token) visits that follow the majority successor; 1 means unifilar |

## 6. Choosing another process from `HMM_processes.py`

Any process in `HMM_processes.PROCESS` can be trained:

```bash
python run_model.py --name even --process even                        # default parameters
python run_model.py --name even_p03 --process even --params p=0.3     # your parameters
python run_model.py --name rgm3 --process rgm --params k=3 burn_in=500
python run_model.py --name mix --process even gm --params even.p=0.3 gm.p=0.7   # per process
```

- `--params` values are read as JSON: numbers and lists, e.g. `'probs=[0.5,0.3,0.2]'`.
- A bare `key=value` applies to every `--process` in the run; `process.key=value` to one.
- `burn_in` is set the same way.
- Runs are named by process only, so **use one `--name` per parameter setting**.
- **State budget:** K = `num_states_mult` × V (default 2V) is set from the vocabulary
  alone. A binary process (V = 2) with more than 4 causal states needs
  `--num-states-mult 4` or more. A run prints a warning, and records `budget_below_k`,
  whenever K is below the theory's k: such a run cannot be FULL.

Registered processes, with their defaults (`run_model.DEFAULT_PARAMS`, or `RunConfig` for
coin and flower). V is the vocabulary; k⁺/k⁻ the forward/backward causal-state counts;
C in bits. Values are from `theory.process_theory`.

| process | parameters (default) | V | k⁺/k⁻ | C⁺ / C⁻ | note |
|---|---|---|---|---|---|
| `coin` | p=0.7, q=0.8 | 3 | 2/3 | 0.997 / 1.467 | |
| `flower` | n=3, m=4, dice_seed=0 | 7 | 4/5 | 1.793 / 1.843 | |
| `pcoin` | p=0.3, q=0.4 | 2 | 2/2 | 0.985 / 0.985 | |
| `iid` | probs=[0.5,0.3,0.2] | 3 | 1/1 | 0 / 0 | zero-memory control |
| `rip` | p=0.4, q=0.3 | 2 | 3/4 | 1.483 / 1.704 | |
| `rnc` | p=0.5, q=0.3 | 2 | 3/3 | 1.500 / 1.467 | |
| `butterfly` | — | 8 | 5/7 | 2.322 / 2.746 | |
| `explosive` | — | 3 | 2/∞ | 0.971 / 2.117 | ⚠ infinite M⁻; shared rows |
| `t3` | — | 2 | 3/∞ | 1.522 / 3.004 | ⚠ infinite M⁻; shared rows |
| `even` | p=0.5 | 2 | 2/2 | 0.918 / 0.918 | |
| `gm` | p=0.5 | 2 | 2/2 | 0.918 / 0.918 | |
| `nemo` | p=0.5, q=0.5 | 2 | 3/3 | 1.500 / 1.500 | ⚠ shared rows |
| `rgm` | k=2 | 2 | 3/3 | 1.500 / 1.500 | ⚠ shared rows |
| `sgm` | k=2, p=0.5 | 2 | 3/3 | 1.500 / 1.500 | |
| `ising` | — (J1=1, J2=0.25, B=0, temp=1) | 2 | 4/4 | 1.310 / 1.310 | |
| `sns` | p=0.5, q=0.5 | 2 | ∞/∞ | 2.712 (truncated) | ⚠ infinite both ways |
| `renewal` | F=[0.1,0.2,0.3,0.25,0.15] | 2 | 5/5 | 2.111 / 2.111 | time-symmetric: the null control for asymmetry |
| `edges` | **required** | | | | any machine, as an edge list (below) |
| `tilt` | **required** | | | | β-map of another process |

### The renewal process: a control with no causal asymmetry

`renewal` ([Marzen & Crutchfield 2015](https://arxiv.org/abs/1408.6876)) emits isolated
events (1) separated by runs of 0s. The run lengths are i.i.d. draws from the inter-event
count distribution `F` (F[n] = P(n 0s between consecutive 1s)). Its state is the count of
0s since the last event, and an event happens with the hazard F(s) / Σ_{n≥s} F(n).

Because the intervals are i.i.d., the process reads the same backwards: M⁻ is isomorphic
to M⁺, and C⁺ = C⁻ exactly (`tests/test_theory_and_config.py` checks this). A model that
learns an asymmetry here has learned an artefact.

```bash
python run_model.py --name renewal_K8 --process renewal --arch transformer gru_feedback gru --num-states-mult 4
python run_model.py --name renewal_F4 --process renewal --params 'F=[0.4,0.3,0.2,0.1]' --num-states-mult 4
```

With V = 2 the default budget is K = 2V = 4, below the default F's 5 causal states.
Raise it with `--num-states-mult`, as above. The runs in `results/renewal/` used K = 4, so
they could not be FULL.

### Any machine from a paper, without writing code

Use `edges`. This is the golden mean; it reproduces `gm(p=0.5)` exactly:

```bash
python run_model.py --name golden --process edges \
    --params n_states=2 vocab=2 'edges=[[0,1,0,0.5],[0,0,1,0.5],[1,1,0,1.0]]'
```

Each entry is `[state, token, next_state, probability]`.

### Adding a new process

In `HMM_processes.py`, write three functions and register them:

- `<name>_transition(params)` → `T` (S×S), with `T[s, s']` = P(s → s');
- `<name>_state_map(params)` → `E` (S×S×V), with `E[s, s', x]` = P(token x | s → s'), each live edge's row summing to 1;
- `<name>_vocab_size(params)` → V;

then add `"<name>": {"transition": ..., "state_map": ..., "vocab_size": ...}` to `PROCESS`.
The generator builds M⁺ and M⁻ from any presentation, including non-unifilar ones.
Optionally add defaults to `run_model.DEFAULT_PARAMS`.

### The theoretical comparison

`theory.process_theory(process, params, burn_in, horizon)` computes everything from
`HMM_processes.py`, through `process_generator.generator`:

| quantity | source |
|---|---|
| the generating HMM exactly as the process defines it (possibly non-unifilar) | `<name>_transition`, `<name>_state_map` |
| M⁺ and M⁻ as (next_state, emission) machines, with occupancy, k and C | `epsilon_machine`, `reverse_machine` |
| C⁺, C⁻, h, E, χ⁺ = H[S⁺\|S⁻], χ⁻ | the generator's closed forms |
| the entropy-convergence floor h(L) = H[X_L \| L − 1 tokens of context] | `conditional_entropies` |

The floor is what a perfect model's per-position CE converges along. Its excess over h
sums to E ([Crutchfield & Feldman 2003](https://pubs.aip.org/aip/cha/article/13/1/25/510735/)),
and this is checked in every theory card.

It shows up in these figures:

| figure | shows |
|---|---|
| `figures/<process>/theory.png` | the generating HMM, M⁺ and M⁻ drawn as graphs, the h(L) curve, and every closed form |
| `figures/<process>/theory_comparison_s<seed>.png` | every model against theory: S_emp vs C⁺/C⁻, the causal asymmetry S_emp(bw) − S_emp(fw) vs C⁻ − C⁺, states discovered vs k, CE − exact |
| `…/<arm>_s<seed>_machine.png` | the machine the model became beside the theoretical ε-machine |
| `…/s<seed>_per_position.png` | the model's CE at each position, the exact model's on the same tokens, and the theory floor |
| `…_states.png`, `…_transitions.png` | occupancy, emission rows and state-to-state matrices against theory |

**Two caveats, flagged on every theory figure:**
1. **Infinite or truncated ε-machines** (`explosive` and `t3` backward, `sns`): the theory
   is a finite approximation, and machines above 20 states are summarised, not drawn.
2. **Causal states that share a next-token distribution** (`nemo`, `rgm`, `t3`, and
   `explosive` backward): matching by next-token rows cannot tell them apart, so
   "discovered" undercounts. Judge those runs by CE − exact, S_emp vs C, and the machine plot.

## 7. Figures, per run

- **training:** CE against H∞ with the restored checkpoint; τ; states used in the
  training batch; gradient norm. With Gumbel on, the states count includes *sampled*
  states, so it runs above the true k while training.
- **states:** learned occupancy vs theory; learned vs true emission rows; determinism.
- **machine:** learned vs theoretical machine graph.
- **per_position, transitions, complexity:** see the table above.
- **scorecard:** every run of a seed, as a table.

## 8. Checks and diagnostics (`checks/`)

| script | what it verifies | output |
|---|---|---|
| `transformer_bit_identical.py` | the modular transformer and training loop equal `Experimental_pipeline`'s bit for bit (CPU, both modes) | printed |
| `old_gru_same_data.py` | runs the notebook's GRU (before modularisation) on the same data and split, scored the same way | `checks/results/crosscheck/` |
| `diagnose_one_knob.py` | one setting at a time (τ, β, K, width, clipping, Gumbel, …), combinable with `+` | `checks/results/diagnostics/` |
| `summarise.py` | one table of every model on every arm, including the ablations, with each run's τ and epochs | `checks/results/crosscheck/summary.{md,csv,png}` |
| `mps_layernorm_bug.py` | the MPS LayerNorm bug (section 9): `repro` compares gradients, MPS vs CPU, for a bare LayerNorm and for the transformer; `train` retrains one arm as-is, with the workaround, or on CPU; `saved` reads the `ln_attn.0` drift from saved runs | `checks/results/diagnostics/mps_layernorm_*.log` |

`checks/results/ablations/` holds the runs that led to the current design:
- `deterministic_head/`: every model with the deterministic head;
- `gumbel_gru/`: both GRUs with Gumbel;
- `transformer_geom800/{default,renewal_K8}/`: the transformer at the earlier standard,
  `geom:5:0.5` for 800 epochs.

The finding: with a deterministic straight-through head, a state that loses early is never
chosen again, so merged states stay merged. This is the "dead codeword" problem of
discrete bottlenecks ([Łańcucki et al. 2020](https://arxiv.org/abs/2005.08520)).
Gumbel-ST fixes it for the feedback GRU on every arm tested.

## 9. Known issues

- **MPS LayerNorm bug (PyTorch 2.12.1), not fixed.** On MPS, `LayerNorm`'s backward pass
  returns wrong weight and bias gradients when its input does not require a gradient. In the
  transformer that is `ln_attn.0`, whose input is built from buffers only. Every MPS-trained
  transformer, the pipeline's included, trains that one LayerNorm's weight and bias on
  garbage gradients. The other parameters match CPU; the key biases differ only by rounding
  noise, because their true gradient is zero. Workaround, verified: make that input require a gradient.
  It is not applied, so as not to change results relative to earlier runs. Training on CPU
  is correct.

  What it does to training (measured in every saved MPS run; `checks/mps_layernorm_bug.py saved`):
  AdamW turns the consistently-signed wrong gradient into a step of about the learning rate
  every update, so `ln_attn.0`'s gain drifts to roughly `1 − lr × step`. It reaches −15 by step
  16,000 and has bias entries up to 18; the other seven LayerNorms keep mean gains of 0.84–1.31.
  Validation CE gets worse late in training. In all 16 transformer runs, the best checkpoint
  that `restore_best` keeps comes from during the descent (τ 0.56–4.1). In the six runs with a
  hold, it never comes from the hold at τ = 0.5. On `renewal`, all four runs end at the
  memoryless predictor: val CE 0.9026–0.9029, where H(1/3.15) = 0.9016 bits. The logged
  gradient norm (10⁵–10⁷) grows linearly with the step and does not depend on the data: coin
  and flower give identical values. The first run in each lane sits near 3·10⁵ and every later
  run near 1.9·10⁷, which is the signature of a data-independent term.

  Retraining without the bug (`checks/mps_layernorm_bug.py train`; seed 0, standard settings;
  logs in `checks/results/diagnostics/mps_layernorm_*.log`):

  | arm | variant | states found | used | S − C | CE − exact | max \|g\| |
  |---|---|---|---|---|---|---|
  | renewal fw (K = 8) | MPS as-is | 4/5 | 5 | +0.001 | +0.027 | 3·10⁵ |
  | | MPS + workaround | 4/5 | 5 | −0.038 | +0.040 | 1.4 |
  | | CPU | 3/5 | 3 | −0.752 | +0.047 | 1.9 |
  | | MPS + workaround + Gumbel-ST | **5/5** | 8 | +0.393 | **+0.003** | 1.5 |
  | flower bw (K = 14) | MPS as-is | 3/5 | 5 | −0.427 | +0.050 | 2·10⁷ |
  | | MPS + workaround | 2/5 | 2 | −0.843 | +0.106 | 47 |
  | | MPS + workaround + Gumbel-ST | 2/5 | 2 | −0.843 | +0.106 | 51 |

  Without the bug, the gain stays near 1 and training no longer collapses late. The workaround
  and CPU behave alike, since both are correct. But fixing the bug does not, by itself, improve
  state recovery, and on flower backward it is worse. With the bug, early training fit faster on
  both arms, so the current settings were in effect tuned with the bug in place. Gumbel-ST, which
  fixed the GRUs, finds every renewal state but over-splits, and does nothing for flower backward.
  All of this is one seed.
- The default results are **one seed**. Use `--seed` / `SEED=` for repeats before
  drawing conclusions.

## References

- Thompson, Garner, Mahoney, Crutchfield, Vedral & Gu (2018), *Causal asymmetry in a quantum world*, [PRX 8, 031013](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031013)
- Ellison, Mahoney & Crutchfield (2009), *Prediction, retrodiction, and the amount of information stored in the present*, J. Stat. Phys. 136, 1005, [arXiv:0905.3587](https://arxiv.org/abs/0905.3587)
- Crutchfield & Feldman (2003), *Regularities unseen, randomness observed: levels of entropy convergence*, [Chaos 13, 25](https://pubs.aip.org/aip/cha/article/13/1/25/510735/)
- Cho et al. (2014), *Learning phrase representations using RNN encoder–decoder*, [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
- Vaswani et al. (2017), *Attention is all you need*, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Xiong et al. (2020), *On layer normalization in the transformer architecture*, [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Loshchilov & Hutter (2019), *Decoupled weight decay regularization*, [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Bengio, Léonard & Courville (2013), *Estimating or propagating gradients through stochastic neurons*, [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
- Jang, Gu & Poole (2017), *Categorical reparameterization with Gumbel-softmax*, [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
- Maddison, Mnih & Teh (2017), *The Concrete distribution*, [arXiv:1611.00712](https://arxiv.org/abs/1611.00712)
- Marzen & Crutchfield (2015), *Informational and causal architecture of discrete-time renewal processes*, Entropy 17, 4891, [arXiv:1408.6876](https://arxiv.org/abs/1408.6876)
- Zeng, Goodman & Smyth (1993), *Learning finite state machines with self-clustering recurrent networks*, [Neural Computation 5, 976](https://direct.mit.edu/neco/article-abstract/5/6/976/5759/Learning-Finite-State-Machines-With-Self?redirectedFrom=PDF)
- Zeng, Goodman & Smyth (1994), *Discrete recurrent neural networks for grammatical inference*, [IEEE TNN 5, 320](https://pubmed.ncbi.nlm.nih.gov/18267800)
- Koul, Greydanus & Fern (2019), *Learning finite state representations of recurrent policy networks*, [arXiv:1811.12530](https://arxiv.org/abs/1811.12530)
- Łańcucki et al. (2020), *Robust training of vector quantized bottleneck models*, [arXiv:2005.08520](https://arxiv.org/abs/2005.08520)
