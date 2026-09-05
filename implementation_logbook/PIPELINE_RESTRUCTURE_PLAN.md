# Pipeline Restructure — Plan (rev. 2)

Turns the exploratory tree into a **definite transformer-based causal-extraction
pipeline**: one standard config, two entry points, four figures.
Rev 1: 2026-09-05. Rev 2: 2026-09-05, after verifying rev 1 against the code.

Purpose beyond tidiness: the next stage of the work compares this method against
other causal-state extraction methods. That comparison needs a pipeline that
takes a process and returns *the extracted causal structure* under fixed,
stated settings — not a parameter search. Everything that exists only to explore
a parameter axis is out of scope here.

## What changed in rev 2

Rev 1 was checked function-by-function against the tree. The extraction manifest
(§5) is accurate — every function it names exists. The β reasoning matches
`discrete_hparams`'s own docstring. The §4 arithmetic is right. Four corrections
and four resolved decisions:

| | Rev 1 said | Actually |
|---|---|---|
| C1 | `pipeline.py` has two live bugs (`_HERE` paths, missing `--` prefixes) | **Stale.** Both already fixed; it resolves against `_PIPE` and the flags are correct. |
| C2 | F3 draws "the theoretical transition matrix" | **No such closed form exists anywhere in the tree.** `coin_true_conditional` is the token-emission conditional, a different object. It has to be derived — see §7a. |
| C3 | `main_results/<process_tag>/` + `sweep_coin/` | Conflicts with the three folders already on disk. Now §2. |
| C4 | Sweep sizing left open | Measured: ~156 s per repeat at 4 models. The 10-value crossed grid is ~108 h. Now §8a. |

Resolved (were §9 "open decisions"):

- **Both arms stay.** The backward arm is where the state count differs — coin 3
  vs 2, flower's backward count depends on dice merging — so it is where an
  extraction method is actually stressed.
- **Both architectures every run.** F4 needs both estimators on the *same*
  realisation or its two empirical bars are not comparable.
- **Sweep: pilot first.** 5×5 coin + 5 flower at 5 repeats, measure the spread of
  `S_emp_states` / `S_hat`, then set the repeat count from that.
- **F3 theory: forward arms only.** Both forward closed forms are simple and
  exactly checkable. The backward ones are new derivation work with real room
  for error, and a wrong theory panel is worse than none.

---

## 1. Scope

**In** — the four quantities, for every process the pipeline is run on:

| # | Quantity | Source |
|---|---|---|
| 1 | Causal states extracted by the **discrete bottleneck** (occupancy + emission table) | `embed_type="discrete"` |
| 2 | Causal states extracted from the **standard model by clustering** predictive distributions | `embed_type="onehot"` |
| 3 | The **learned state-to-state transition matrix** | discrete |
| 4 | **Theoretical vs empirical statistical complexity** | both |

**Out** — dropped entirely: ΔCE and the paired forward/backward *asymmetry
statistic*, the K sweep, state_dim sweep, β sweep, capacity sweep, budget sweep,
seed-spread study, F0–F12 figure set, attention heatmaps, perplexity/KL
diagnostics, UMAP-for-its-own-sake, `pq_experiment`, `run_dice_experiment`,
`analyse_*`, `sanity_check`, `LLM_asymmetry_testing`, the notebook builder.

Both arms are still **trained** — dropping ΔCE drops the *statistic*, not the
backward model, which is quantity 1–4's harder half. CE is still recorded, but
only as a **convergence check** — is this model trained enough for its extracted
states to mean anything — not as a result.

---

## 2. Target layout

Everything self-contained. `Experimental_pipeline/` imports **nothing** from
`Transformer_model/` or `Experimental_setup/`.

```
Experimental_pipeline/
├── config.py        master config + per-process spec builders      [helper]
├── processes.py     coin & flower: generation, Datasets, closed forms  [helper]
├── models.py        OneHotDecoder + DiscreteCausalDecoder           [helper]
├── training.py      train loop, split, eval, seeding, divergence    [helper]
├── extraction.py    the three extractors                            [helper]
├── figures.py       the four figures                                [helper]
├── pipeline.py      the experiment itself: one_repeat, run_process, run_sweep
├── run_process.py   ENTRY: one process x N repeats
└── run_sweep.py     ENTRY: (p,q) and (n,m) grids
```

Three layers, as requested: `run_*.py` are the terminal commands, `pipeline.py`
runs the experiment, the rest are helpers. The old `pipeline.py` was a
*subprocess dispatcher*; that role disappears, because the two entry points are
themselves the terminal commands and a dispatcher over two commands earns
nothing.

```
main_results/
├── run_config.json
├── trainings/                       run_process.py
│   └── coin_p030_q040/
│       ├── repeats.pkl
│       ├── F1_causal_states_discrete.png
│       ├── F2_causal_states_clustered.png
│       ├── F3_transition_matrix.png
│       └── F4_complexity.png
├── sweep_params/                    run_sweep.py
│   ├── sweep_coin/
│   │   ├── grid.pkl
│   │   ├── F4_complexity_grid.png
│   │   └── coin_p015_q035/  F1–F4
│   └── sweep_flower/
└── arc_comparision/                 cross-architecture summary
    └── F4_all_processes.png
```

`arc_comparision/` holds the figure that pools every process onto one
theory-vs-discrete-vs-clustered plot. That is the figure the *next* stage
extends: a new extraction method is one more series against the same theory line.

`processes.py` deliberately holds both the sampler and the theory for each
process: "how to draw it" and "what its causal structure provably is" are the
same subject, and keeping them together stops a generator change from silently
invalidating a closed form.

---

## 3. The master config

One config, in `config.py`. Values are `DISCRETE` (the old `00_base`) with the
sequence geometry changed per §4.

```python
CONFIG = dict(
    seed              = 0,
    # ── sequence geometry  (§4) ────────────────────────────────────────
    seq_len           = 300,    # TOKENS per training sequence, both processes
    burn_in           = 250,    # TOKENS drawn and discarded before the window
    num_samples       = 500,    # sequences per repeat
    batch             = 32,
    # ── model ──────────────────────────────────────────────────────────
    d_model           = 32,
    n_layers          = 2,
    # ── discrete bottleneck ────────────────────────────────────────────
    n_states_mult     = 5,      # K = 5V  — a BUDGET, not an estimate
    state_dim_mult    = 1,      # S = V   — carries no expressive power
    tau               = 1.0,
    usage_beta        = None,   # -> 1/(batch*seq_len) = 1.0417e-4
    # ── optimiser ──────────────────────────────────────────────────────
    lr                = 1e-3,
    weight_decay      = 0.01,
    max_epochs        = 150,
    # ── analysis ───────────────────────────────────────────────────────
    ana_batch         = 32,
    max_batches       = 20,
    state_min_pos     = 5,
    state_tol         = 0.10,
    cluster_metric    = "euclidean",
    n_pts             = 1000,
    conv_tol          = 0.10,
    val_every_n_steps = 25,
    accelerator       = "auto",
    trans_total_run   = 5000,
)
```

Two derived quantities, computed and never written as literals:

- `K = 5V` → coin (V=3) K=15; flower(2,8) (V=10) K=50.
- `β = 1/(batch · seq_len) = 1/9600 = 1.0417e-4`.

**β is 1/(the training average).** `discrete_hparams`'s docstring states the rule
exactly: "beta is naturally per-token: 1/N with N = batch * chunk_len, the count
the cross-entropy itself averages over." The penalty is `β·H(p̄)` and `p̄` is the
occupancy averaged over every scored token, so β must track that same count.
Under the new geometry the scored count per step is `batch · seq_len`, so
β = 1/9600. Written as a literal it would decouple from N and drift silently
relative to the collapse cliff, which is why it stays derived.

**β check.** The measured collapse cliff is ~6e-4 at lr=1e-3, and the cliff
scales with 1/lr rather than with N. 1.0417e-4 sits **5.8× below** it (the old
32×256 geometry sat 4.9× below). Safe — but this is the number to re-check if
`seq_len` or `batch` ever moves.

`conv_tol` earns its place: at 150 epochs with wd=0.01 every arm of both process
families measured inside 0.1 bits of H∞. A repeat outside it has an untrustworthy
state assignment and must be **excluded from the figures and counted in the
caption**, not silently averaged in.

---

## 4. Sequence generation — the change

**Now:** coin generates 2000 tokens per sequence and flower ~4000 (its `seq_len`
counts *cycles*, 2 tokens each), then `ChunckDataset` keeps one random 256-token
window and discards the rest — 87% for coin, 94% for flower. Two gotchas ride
along: the cycles-vs-tokens mismatch, and the model only ever seeing positional
indices `[0, chunk)` so any analysis at full length measures extrapolation.

**New:** generate exactly what is trained on.

```python
def generate(kind, params, num_samples, seq_len, burn_in, rng):
    """Return (num_samples, seq_len + 1) int array of tokens.

    seq_len is in TOKENS for both processes.  The +1 is the shift: the
    Dataset emits (x[:-1], x[1:]), so seq_len+1 tokens give seq_len
    scored positions.

    burn_in tokens are drawn and discarded first, so every kept token is
    at stationarity regardless of how the chain was initialised.
    """
```

Arithmetic at `burn_in=250, seq_len=300`:

- **Coin** — draw `250 + 301 = 551` steps, keep the last 301. The sampler already
  initialises from the stationary distribution (`P(state=1) = p/(p+q)`), so the
  burn-in is belt-and-braces here; it costs 250 steps and makes the two processes
  behave identically.
- **Flower** — one cycle emits 2 tokens, so draw `ceil(551/2) = 276` cycles →
  552 tokens, slice `[250:551]` → 301 tokens. An odd `burn_in` or `seq_len` is
  handled by the slice, not by the caller.

What this buys:

1. **Nothing is discarded.** 500 × 300 = 150k training tokens, all used, against
   128k used out of 2M generated. Generation cost drops ~13×.
2. **`ChunckDataset` / `make_chunked_loader` / `make_analysis_loader` disappear.**
   Training and analysis run at the same length by construction, so the
   extrapolation hazard cannot occur. `max_len = seq_len = 300`.
3. **The cycles-vs-tokens trap is gone.** `seq_len=300` is 300 tokens for both.
4. **`num_samples` becomes the only data knob.**

**Flower phase — deliberate, and unchanged from current behaviour.** A flower
sequence alternates select/roll, so token *type* is determined by index parity.
With an even `burn_in` every kept sequence starts on a select token, exactly as
the current `pre_depth` (whole cycles) does. This is preserved rather than fixed:
randomising the phase would change the process the existing results were measured
on. It is worth knowing when reading F2, because a model *could* in principle key
on positional parity rather than token content — `h_state_given_token` is the
diagnostic that would catch it.

**Statistical note.** Test split is 20% = 100 sequences × 300 = 30k tokens. Fine
for state occupancy (which converges fast), thin for a CE *difference* — another
reason ΔCE is out of scope rather than merely deprioritised.

---

## 5. Extraction manifest

Copy these and nothing else. Where a source function carries branches for dropped
experiments, strip them on the way over. **Verified: every function named here
exists at the stated location.**

### `processes.py`
| From | Take |
|---|---|
| `Data_generation.py` | `CoinDataset`, `coin_generation` → **rewrite** per §4 |
| `Flower_process_generation.py` | `FlowerDataset`, `flower_process_generation` → **rewrite** per §4 |
| `Model_analysis.py` | `statistical_complexity` (coin C±), `flower_complexity`, `_merged_outcome_mass`, `flower_entropy_rate`, `causal_state_count`, `causal_state_occupancy` |
| `utils.py` | `entropy_bits`, `entropy_rate_coin`, `coin_tag`, `flower_tag` |
| **new** | `coin_transition_matrix`, `flower_transition_matrix` — §7a |

Drop: `Rev_HMM_generation`, `Rev_flower_process_generation`, `make_loader`,
`make_flower_loader` (the reversed-*data* notion of backward is a third, unused
mechanism and a known trap — backward is the attention mask plus the batch swap).

### `models.py`
| From | Take |
|---|---|
| `OneHot_model.py` | `PositionalEncoding`, `AttentionModel`, `OneHotDecoder`, `cross_ent_onehot` |
| `DiscreteCausal_model.py` | `DiscreteCausalDecoder` |

### `training.py`
| From | Take |
|---|---|
| `Training_model.py` | `train_model`, `Record_training`, `set_seed`, `test_train_validation`, `_eval_loss_on_loader`, `diagnose_divergence` |
| `utils.py` | `cleanup`, `to_cpu_for_analysis`, `mkdir`, `save_pkl`, `save_run_config` |
| `run_experiments.py` | `full_seq_len` (6 lines — inline it) |

Drop: `ChunckDataset`, `make_chunked_loader`, `make_analysis_loader`, `_loader`,
`train_test_val_pipeline` (5-fold CV is asymmetry machinery; repeats replace it),
`save_weights` / `load_weight_meta` / `check_weight_meta`.

### `extraction.py`
| From | Take |
|---|---|
| `DiscreteCausal_analysis.py` | `causal_state_report`, `s_emp_from_states`, `transition_matrix_extraction` |
| `Model_analysis.py` | `recover_causal_states`, `latent_extraction`, `_sample_latents`, `_resolve_t`, the distance matrices + `DISTANCE_MATRICES` / `DEFAULT_STATE_TOL`, `discrete_hparams` |

Drop `statistical_complexity_empirical` (k-means at an *assumed* k, so
`S ≤ log2 k` by construction — it cannot report a state count it was not told).
The two kept estimators are strictly better and are exactly the pipeline's two
extraction methods:

- **discrete** → `causal_state_report` → `S_emp_states`, exact occupancy entropy,
  no clustering, no free threshold.
- **onehot** → `recover_causal_states` → `k_hat`, `S_hat`, plus `stability` and
  `plateau` across the tolerance grid.

Keep `h_state_given_token`. It needs no ground truth (the true causal state is a
deterministic function of the current token in both arms of both processes, so
truth scores exactly 0) and it is the only unsupervised warning that the
bottleneck merged states — measured 0.000 on every run that recovered its state
set, 0.11–0.48 on every run that did not.

### `figures.py`
| From | Take |
|---|---|
| `DiscreteCausal_analysis.py` | `plot_causal_states`, `plot_causal_states_pair`, `print_state_summary`, `plot_transition_matrix_pair` |
| `plot_state_clusters.py` | `draw_panel` |
| `Model_analysis.py` | `savefig` |

Drop the whole `DiscreteCausal_figures.py` F0–F12 set — every one plots a swept
axis.

---

## 6. The two entry points

### 6.1 `run_process.py` — one process, N repeats

```
python Experimental_pipeline/run_process.py --process coin --p 0.3 --q 0.4 --repeats 30
python Experimental_pipeline/run_process.py --process flower --n 2 --m 8 --repeats 30
python Experimental_pipeline/run_process.py --all --repeats 30      # the seven, = old 00_base
python Experimental_pipeline/run_process.py --all --plots-only      # redraw from repeats.pkl
```

Per repeat, seeded by the repeat index:

```
data = generate(...)                       # §4, ONE realisation, shared by all 4 models
for arch in ("discrete", "onehot"):
    for arm in ("fw", "bw"):
        train                              # identical init & split across arch/arm
        record CE, converged?
        if arch == "discrete":  causal_state_report   -> occupancy, emissions, S_emp_states
        if arch == "onehot":    recover_causal_states -> k_hat, S_hat, stability, plateau
transition_matrix_extraction on the discrete arms of the best repeat only
```

**4 models per repeat.** Measured on the finished `08_pqsweep`: 78 s per repeat at
2 models ⇒ ~39 s/model ⇒ **~156 s per repeat**. The seven processes at 30 repeats
is therefore **~9 h**. (Rev 1 said "time one repeat before committing"; it has now
been timed, from a completed 10.4 h run.)

`transition_matrix_extraction` runs on the **best repeat only** (lowest
max|CE − H∞| among converged), with live models held via `keep_models` — holding
30 repeats × 4 models is what that flag exists to avoid. Set
`burn_in = seq_len + 50 = 350` and `window_size = seq_len = 300`; the seed
sequence must be at least `window_size` long or the roll-out starts on a short
context.

### 6.2 `run_sweep.py` — the grids

```
python Experimental_pipeline/run_sweep.py --sweep-coin   0.15 0.35 0.55 0.75 0.95 --repeats 5
python Experimental_pipeline/run_sweep.py --sweep-flower 2 4 6 8 10 --repeats 5
```

The coin grid is **crossed with itself**: k values → k² cells. See §8a for sizing.

---

## 7. The four figures

Drawn identically by both entry points — per-process for `run_process`, per-cell
plus one aggregate for `run_sweep`.

**F1 — `F1_causal_states_discrete.png`** (`plot_causal_states_pair`)
Forward and backward side by side. Per arm: state-occupancy bars with theoretical
occupancy overlaid, and the K×V emission table as a heatmap. Caption carries
`n_states_used` vs `true_k`, `S_emp_states` vs C±, and `h_state_given_token`.
States are identified only up to a **permutation** — match before comparing.

**F2 — `F2_causal_states_clustered.png`** (`draw_panel`)
The onehot model's predictive distributions, 2-D projected, coloured by
agglomerative cluster, forward and backward. Caption carries `k_hat` at
`state_tol` **and the plateau across the tolerance grid** — `k_hat` at a single
threshold is not defensible alone, since the smallest true backward-state
separation ranges 0.135 (p=0.1,q=0.9) to 0.612 (p=q=0.5).

**F3 — `F3_transition_matrix.png`** (`plot_transition_matrix_pair`)
Learned T[i][j] = P(s_{t+1}=j | s_t=i), row-normalised, visited states only,
forward and backward. **Forward panels carry a third theory sub-panel** under the
matched permutation (§7a); backward panels show the learned matrix alone.

**F4 — `F4_complexity.png`** — the headline
Per arm, three bars: **theory** (C+ / C−), **discrete** (`S_emp_states`),
**clustered** (`S_hat` at the PLATEAU tolerance), with error bars over converged
repeats and n annotated. This is the figure that generalises to the method
comparison.

**Why the clustered bar uses the plateau, measured.** At the inherited
`state_tol = 0.10` the clusterer over-splits badly on this geometry — coin
p=0.3,q=0.4 gives `k_hat = 5` against a true 2 forward and 8 against a true 3
backward, biasing `S_hat` by **+1.00** and **+1.11** bits. The plateau — the k
that survives the widest span of the tolerance grid — recovers 2 and 3 exactly,
and its `S_hat` lands at **−0.002** and **−0.027** of the closed form. The
plateau is chosen without reference to the truth, so this is a mitigation for
the free threshold rather than tuning on the answer. `S_hat` at `state_tol` is
still recorded in the pickle; it is simply not what F4 plots.

Sweep version `F4_complexity_grid.png`: x = theory, y = empirical, one series per
estimator, with the y=x line. Excluded non-converged repeats in the caption.

Aggregate `arc_comparision/F4_all_processes.png`: every process on one
theory-vs-empirical plot, both estimators.

### 7a. The theoretical transition matrices — new work

Nothing in the tree computes these; `coin_true_conditional` is P(next token |
current token), a different object. Both forward closed forms are simple:

**Coin, forward.** 2 states = the value of `cur_state`. From the generator,
`0 → 1` w.p. `p` and `1 → 0` w.p. `q`:

```
T+ = [[1-p,  p ],
      [ q,  1-q]]
```

Already validated empirically: median max-cell error **0.009** (worst 0.025)
across the 24 two-state cells of the finished `08_pqsweep`.

**Flower, forward.** n+1 states: one "a roll just happened" state R, plus one per
die. `R → die i` w.p. `1/n`; `die i → R` w.p. 1. Stationary mass ½ on R and
`1/(2n)` on each die, which is what `flower_complexity` already assumes.

**Backward is deferred.** The backward ε-machine transition structure — coin's 3
states, and flower's `1 + #distinguishable outcomes` with the dice-merging rule
from `causal_state_count` — is genuine derivation work with real room for error.
A wrong theory panel is worse than none, so backward panels show the learned
matrix without an overlay until the derivation is done and tested.

---

## 8. Build order

Each step ends in a check that can fail loudly.

1. **Commit the pending move first.** `run_statistical_trj.py` /
   `run_sweep_experiment.py` / `replot_complexity.py` show as deleted from
   `Experimental_setup/` with `Experimental_pipeline/` untracked. Commit as-is so
   the restructure reads as a reviewable diff. (Note the moved copies were also
   *edited* — a docstring block removed, imports repointed — so this is not a
   pure rename.)
2. **`processes.py`** — new generators + closed forms + §7a matrices.
   *Check:* empirical transition counts from 10⁵ generated tokens match the
   analytic chain to <1e-2; empirical entropy rate matches `entropy_rate_coin` /
   `flower_entropy_rate` to <1e-2; sequences are exactly `seq_len+1` long; a
   fixed seed reproduces byte-identical output; `coin_transition_matrix` matches
   the empirical count matrix to <1e-2.
3. **`models.py` + `training.py`** — copy, strip, inline `full_seq_len`.
   *Check:* a 5-epoch run of each architecture completes and CE falls.
4. **`extraction.py`** — the three extractors against the new loaders.
   *Check (the important one):* on coin p=0.3,q=0.4, `S_emp_states` lands within
   ~0.01 bits of C+ on the forward arm, as it did under the old geometry. If not,
   the geometry change moved something real — stop and find it.
5. **`figures.py`** — four figures from a saved pickle, so `--plots-only` works
   without retraining.
6. **`pipeline.py`**, then **`run_process.py`**, then **`run_sweep.py`**.
7. **Parity run:** one process, 5 repeats, against the archived
   `All_Results/discrete_v2/00_base` numbers for the same process. Expect *close,
   not identical* — the geometry changed on purpose. Beyond ~0.02 bits on
   `S_emp_states` needs an explanation before the full run.
8. Only then delete the superseded files.

Until step 8, `Transformer_model/`, `Experimental_setup/` and `All_Results/` stay
untouched as a frozen reference to diff against.

### 8a. Sizing the runs

Measured: **~156 s per repeat** (4 models).

| Run | Cells | Repeats | Models | Wall |
|---|---|---|---|---|
| `run_process --all` (= old 00_base) | 7 | 30 | 840 | ~9 h |
| Sweep pilot: coin 5×5 + flower 5 | 50 | 5 | 1000 | ~11 h |
| Sweep at 5×5, 20 repeats | 50 | 20 | 4000 | ~43 h |
| Sweep at 10×10 coin + 5 flower, 20 repeats | 125 | 20 | 10000 | ~108 h |

**Decision: pilot first.** Run the 5-repeat pilot, measure the across-repeat
spread of `S_emp_states` and `S_hat`, and set the final repeat count from that
spread rather than from the old schedule's inherited 20/30. A repeat count chosen
because the previous schedule used it is not a measurement.

---

## 9. Residual risks

- **`state_tol = 0.10` stays a free parameter, and on this geometry it is the
  WRONG value.** Measured above: it over-splits by 2.5x and biases `S_hat` by a
  full bit. The pipeline reports the plateau instead, which is unsupervised and
  lands within 0.03 bits — but "the threshold that happened to be inherited is
  off by that much" is exactly the fragility the comparison against other
  extraction methods will turn on, and it should be stated wherever F2 or F4's
  clustered bar appears.
- **Flower phase is fixed by construction** (§4). Not a regression, but a
  confound to keep in view.
- **The two estimators are not like-for-like.** `S_emp_states` is an exact
  occupancy entropy over a *budgeted* K; `S_hat` is an entropy over clusters at a
  chosen threshold. F4 puts them on one axis, which is the point — but they fail
  differently, and the caption should say so.
