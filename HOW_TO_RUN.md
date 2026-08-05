# Running the Causal-Asymmetry Experiments: A Tutorial

This document is a complete operating guide to the experiment suite in this
directory. It is written to be read in order by someone who has not run the
code before, but each section is self-contained enough to be consulted on its
own later.

Every command shown here has been executed and verified against the current
state of the repository.

**Contents**

1. [Prerequisites](#1-prerequisites)
2. [What the experiments measure](#2-what-the-experiments-measure)
3. [How an experiment is structured](#3-how-an-experiment-is-structured)
4. [The configuration system](#4-the-configuration-system)
5. [Running the training suite](#5-running-the-training-suite)
6. [Running the controls](#6-running-the-controls)
7. [Generating the causal-state figures](#7-generating-the-causal-state-figures)
8. [The post-hoc evaluators](#8-the-post-hoc-evaluators)
9. [Verification and documentation](#9-verification-and-documentation)
10. [Understanding the outputs](#10-understanding-the-outputs)
11. [Interpreting the results](#11-interpreting-the-results)
12. [Troubleshooting](#12-troubleshooting)
13. [Command reference](#13-command-reference)

---

## 1. Prerequisites

### 1.1 Environment

All work is performed in the `qdrug` Conda environment. Activate it and change
to the project directory before running anything:

```bash
conda activate qdrug
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version
```

The environment provides the following versions, which are the ones the results
were produced with:

| Package | Version |
|---|---|
| Python | 3.11.14 |
| PyTorch | 2.12.1 |
| Lightning | 2.6.5 |
| NumPy | 2.4.2 |
| scikit-learn | 1.8.0 |

To reproduce the environment elsewhere, install from `requirements.txt`; for a
byte-exact reproduction, use `requirements-lock.txt`.

### 1.2 Hardware and the accelerator

The code selects an accelerator automatically. On this machine that is Apple
MPS, which is approximately six times faster than the CPU at these model sizes.

The reader should be aware of one consequence. **MPS is not bit-reproducible
across runs.** Model initialisation is identical and the first loss value
matches to eight decimal places, after which trajectories diverge because of
non-deterministic reductions in the Metal kernels. The identical test on CPU is
exactly repeatable.

This does **not** affect the pairing of the forward and backward arms, which is
what the scientific comparison depends upon: within a single run the two arms
share their data splits, fold membership, batch order and initial weights.
Should exact run-to-run reproducibility be required — for instance when
preparing final reported figures — set `accelerator = "cpu"` in `configs.py`
and accept roughly six times the wall-clock time.

### 1.3 Verifying the installation

Before running any experiment, confirm that the test suite passes:

```bash
pytest tests/ -q
```

The expected result is `61 passed` in approximately twenty seconds. These tests
pin the correctness fixes that are *silent* when broken — a numerically wrong
loss function, for example, produces plausible-looking numbers rather than an
error. If any test fails, do not run experiments until the cause is understood.

---

## 2. What the experiments measure

### 2.1 The hypothesis

Two transformer decoders are trained on **the same forward-generated data**.
They differ in only two respects: the attention mask, and which element of the
batch is treated as input.

| Model | Mask | At position *t* it sees | It predicts |
|---|---|---|---|
| forward | `tril` (lower triangular) | `x[0 … t]` | `x[t+1]` |
| backward | `triu` (upper triangular) | `x[t … T−1]` | `x[t−1]` |

The backward model is therefore **not** trained on reversed sequences. It
receives the same batch with `(inputs, targets)` swapped.

Computational mechanics (Crutchfield; Ellison–Mahoney; Thompson et al., 2017)
associates with a stochastic process two quantities: the **forward statistical
complexity** C⁺, the entropy of the minimal set of states required to predict
the future from the past, and the **backward statistical complexity** C⁻, the
same quantity for retrodicting the past from the future. The hypothesis under
test is that when C⁻ > C⁺, the backward model should converge to a measurably
higher cross-entropy.

### 2.2 The quantity that is measured

The headline measurement is

```
ΔCE = CE_backward − CE_forward
```

evaluated on held-out sequences, paired fold by fold.

### 2.3 The principal caveat, stated before any results

The entropy rate of a stationary process is **invariant under time reversal**:
H∞(forward) = H∞(backward). Consequently

```
CE_forward  → H∞ + residual_forward
CE_backward → H∞ + residual_backward
ΔCE = residual_backward − residual_forward
```

**ΔCE is a difference of residuals, not a difference of entropy rates.** A
predictor with unbounded capacity, trained to convergence, attains ΔCE = 0 in
both directions *regardless* of C⁻ − C⁺. A non-zero ΔCE is an artefact of a
**memory-bounded** predictor: the direction requiring more causal states leaves
a larger residual at fixed `d_model` and fixed epoch budget.

Two consequences follow, and both must accompany any reported result:

1. Only the **sign** of ΔCE is predicted, and only when C⁻ ≠ C⁺. The magnitude
   has no theoretical anchor against which to be compared.
2. A null result (ΔCE ≈ 0) is **ambiguous**. It may mean "no asymmetry" or
   "capacity was sufficient to absorb the asymmetry". Distinguishing these
   requires the `d_model` sweep, which has not yet been performed.

### 2.4 The two processes

**The coin process** is a three-token hidden Markov model with parameters
*p* and *q*. It is asymmetric for *every* (*p*, *q*): the three backward causal
states refine the two forward states, so C⁻ − C⁺ ≥ 0 identically. Forward it has
2 causal states; backward, 3.

**The flower process** alternates between selecting one of *n* dice and rolling
it, producing a vocabulary of *n* + *m* tokens. Its closed forms are

```
C⁺ = 1 + ½·log₂(n)
C⁻ = 1 + ½·H(π_merged)
```

where the merge in C⁻ is over *distinguishable* outcomes: two outcomes belong
to the same backward causal state precisely when they induce the same posterior
over dice, that is, when their `dice_probs` columns are proportional. Because
C⁻ ≤ 1 + ½·log₂(m), the condition **C⁻ > C⁺ requires m > n**. Configurations
with n > m test the *negation* of the hypothesis and are included deliberately,
so that the sign of ΔCE can be checked against the sign of C⁻ − C⁺.

---

## 3. How an experiment is structured

Understanding the data flow is necessary in order to know which reported
numbers are held-out and which are not.

### 3.1 Data generation and windowing

A configuration specifies a number of sequences and a sequence length. Each
sequence is converted to an (input, target) pair by offsetting it by one token:
`input = x[:-1]`, `target = x[1:]`.

Because the sequences are long (typically 2000 tokens) and attention is
quadratic in length, each sequence is reduced to a single fixed-length
**window** of `train_chunk_len` tokens. The window offset is chosen once at
dataset construction from the seed, making window selection a pure function of
the sequence index. This is what allows the forward and backward arms to be
compared: both see identical windows.

### 3.2 The splits

```
N sequences  →  one fixed window each
   │
   ├── 20 %   HELD-OUT TEST SET   never trained on, never used for selection
   │
   └── 80 %   train + validation pool
         └── 5-fold cross-validation
             ├── each fold: 4/5 train, 1/5 validation
             ├── validation loss selects the best fold
             └── all five fold models are scored on the held-out test set
```

For the QUICK configuration with 500 sequences this yields 100 held-out
sequences, a pool of 400, and per fold 320 training and 80 validation
sequences.

Both arms are trained with the same seed, and therefore share the hold-out
split, the fold membership, the batch ordering and the initial weights. The
only difference between them is the mask and the batch convention.

### 3.3 Which reported quantities are held-out

This distinction is important and easily overlooked.

| Reported quantity | Computed on | Held out? |
|---|---|---|
| **Paired ΔCE** | the 20 % held-out sequences | **Yes** |
| `S_emp`, `S_hat`, `k̂`, latent UMAPs | all sequences | **No** — 80 % were trained on |
| `sanity_check.py` scalar ΔCE | all sequences | **No** |
| p-q sweep heatmaps | the training loader itself | **No split at all** |

The p-q sweep trains a model and then evaluates it on the same loader. Its
heatmaps are training-set quantities. This is pre-existing behaviour and is
documented rather than corrected.

The figures produced by `plot_state_clusters.py` (Section 7) address the second
row of this table directly, by separating the train-seen and held-out
sequences into different panels.

---

## 4. The configuration system

### 4.1 Where configurations live

All experiment parameters are defined in `configs.py`. A configuration is a
dictionary; `BASE` holds the values shared by all of them, and each named
configuration overrides what it needs.

Four configurations are defined:

| Name | Output directory | `d_model` | Learning rate | Epochs | Gradient steps per fold | Approximate runtime |
|---|---|---|---|---|---|---|
| `SMOKE` | `results_smoke/` | 16 | 1e-2 | 3 | ≈ 15 | 2 minutes |
| `QUICK` | `results_quick/` | 32 | 1e-2 | 10 | 100 | 9 minutes |
| `QUICK_LARGE_HMM` | `results_quick/` | 32 | 1e-2 | 10 | 100 | 3 minutes |
| `LARGE` | `results_large/` | 64 | 5e-3 | 80 | 1600 | 3.5 hours |

### 4.2 The purpose of each configuration

**`SMOKE`** is not a scientific configuration. It trains for three epochs on
forty sequences, which measures nothing, and exists solely to exercise every
code path quickly. Run it after modifying the code and before committing to a
long run.

**`QUICK`** is the configuration whose results should be reported. Its models
converge to within 0.001 to 0.008 bits of H∞, and all five folds converge in
every one of its seven experiments.

**`QUICK_LARGE_HMM`** applies QUICK's scale, capacity and learning rate to
LARGE's *processes*. It exists because QUICK and LARGE differ along several axes
simultaneously — data volume, capacity, learning rate, and the processes
themselves — which makes them non-comparable. Holding the process fixed converts
the comparison into a controlled one. It writes into `results_quick/` alongside
the QUICK results, and only runs the processes QUICK does not already cover.

**`LARGE`** increases every dimension. Its cross-validation results are **not
usable**; see Section 11.3.

### 4.3 Output directories and tags

Each configuration carries its own `out_root`, so two configurations cannot
overwrite one another's weights.

Experiment tags are **derived from the parameter values** rather than written as
literals: a coin experiment at p = 0.4, q = 0.8 is tagged
`exp1_coin_p040_q080`. A tag therefore cannot disagree with the parameters that
produced it. This was a genuine defect previously: two runners used the same tag
literal at different parameters and overwrote one another's checkpoints.

---

## 5. Running the training suite

### 5.1 The command

`run_experiments.py` is the only file that trains the main experiments. It is
configured entirely by command-line flags; no editing is required.

```bash
python run_experiments.py --config QUICK
```

### 5.2 Available options

```
--config {SMOKE,QUICK,LARGE,QUICK_LARGE_HMM}
        Which configuration from configs.py to run.  Default: QUICK.

--seed N
        Override the configuration's seed.  Both arms always share it, so
        pairing is preserved.  Default: 0.

--out-root DIR
        Override the output directory.  Useful for keeping seed repeats apart.

--only {all,exp1,exp1_2,exp2}
        Run one experiment instead of the whole suite.  Default: all.
```

### 5.3 What each experiment does

| `--only` value | Contents |
|---|---|
| `exp1` | Coin HMM at (`coin_p1`, `coin_q1`); forward and backward arms; five-fold cross-validation; latent, attention and complexity analysis. |
| `exp1_2` | Coin HMM at (`coin_p2`, `coin_q2`), as above, **and additionally the p-q sweep**. |
| `exp2` | One flower experiment for each entry in `flower_configs`; presently two, one with m > n and one with n > m. |
| `all` | All of the above. This is the default. |

The p-q sweep within `exp1_2` trains an additional 2 × |grid|² models — 200 for
QUICK, 512 for LARGE — and is the dominant cost of the LARGE configuration,
accounting for approximately 2.5 of its 3.5 hours. It also does not benefit from
the validation-cadence optimisation, because it trains without a validation
loader.

To obtain the asymmetry numbers without paying for the sweep:

```bash
python run_experiments.py --config LARGE --only exp1
python run_experiments.py --config LARGE --only exp2
```

### 5.4 Worked examples

Run a second seed into a separate directory, so that the two are not confused:

```bash
python run_experiments.py --config QUICK --seed 1 --out-root results_quick_seed1
```

Run only the flower experiments:

```bash
python run_experiments.py --config QUICK --only exp2
```

Add LARGE's processes to the QUICK results directory:

```bash
python run_experiments.py --config QUICK_LARGE_HMM --only exp1
python run_experiments.py --config QUICK_LARGE_HMM --only exp2
```

### 5.5 Behaviour on repeated runs

Repeated and partial runs are safe. `all_results.pkl` is **merged by tag**
rather than overwritten, so invoking `--only exp2` will not discard the record
of `exp1`. The runner prints which earlier experiments it preserved. Because
tags are derived from parameters, re-running an experiment replaces its own
entry and nothing else.

Each configuration also writes `run_config_<CONFIG>.json` into the output
directory, recording the complete parameter set and a UTC timestamp.

### 5.6 Reading the console output

During the run the following are printed for each experiment:

- the theoretical quantities: H∞, C⁺, C⁻, and the predicted sign of ΔCE;
- per-fold validation and test losses;
- a warning for any fold that **diverged** — that is, reached a good loss and
  then blew up (Section 11.3);
- the recovered number of causal states k̂ with its stability profile;
- the paired ΔCE, reported both over all folds and over converged folds only.

At the end, a summary table lists every experiment in the output directory with
its C⁺, C⁻, ΔCE, standard error, and a verdict of `match`, `MISMATCH` or `n.s.`.
The verdict `n.s.` denotes |ΔCE| < 2 × standard error, that is, not
distinguishable from zero.

---

## 6. Running the controls

```bash
python sanity_check.py
```

Runtime is approximately seven minutes; output is written to
`sanity_check_flower_process/`.

### 6.1 Configuration

This script has **no command-line interface**. Its parameters are the `CFG`
dictionary near the top of the file (approximately line 106). The tolerance
`NULL_TOL` (approximately line 150) defines the band within which a null result
is considered upheld; it is presently 0.02 bits.

### 6.2 What the controls establish

Two processes are run, with opposite predictions.

**The positive control** is the coin process at p = q = 0.5, for which
H∞ = 1.0 bit exactly, C⁺ = 1.0 and C⁻ = 1.5. The prediction is ΔCE > 0.

**The null control** is the flower process at n = 1, m = 2 with a fair die. With
a single die every outcome induces the same posterior over dice, so all outcomes
collapse into one backward causal state and C⁻ = C⁺ = 1.0. The sequence
`0, X, 0, X, …` with X uniform on {1, 2} reads identically in reverse: the
process is **exactly time-reversible**. The prediction is ΔCE ≈ 0.

The null control is the more important of the two. A pipeline that reports
ΔCE > 0 on a time-reversible process is measuring an artefact of the
architecture rather than causal asymmetry. This control has already earned its
place: it detected a defect in the analysis pipeline that produced
ΔCE = −0.4210 on a process whose true value is zero.

### 6.3 Interpretation of the verdicts

Each control is scored against its own prediction, using a three-way verdict:

| Verdict | Meaning |
|---|---|
| `PASS` | The predicted effect was observed. |
| `INCONCLUSIVE` | \|ΔCE\| < `NULL_TOL` on the *positive* control. This is not a refutation: a converged model with spare capacity yields ΔCE ≈ 0 whatever C⁻ − C⁺ may be. Interpretation requires the `d_model` sweep. |
| `FAIL` | The effect was observed with the wrong sign, by more than the tolerance. |

---

## 7. Generating the causal-state figures

```bash
python plot_state_clusters.py
```

This script must be run **after** a training run. It loads saved weights and
does not retrain, so it is inexpensive.

### 7.1 Options

```
--config CFG          Which configuration's experiments to plot.  Default: QUICK.
--also-config CFG     A second configuration sharing the same out_root.
                      Default: QUICK_LARGE_HMM.  Pass '' to skip.
--out-root DIR        Where to read weights from and write figures to.
--metrics ...         Any of: euclidean, tv, js.  Default: all three.
```

Example:

```bash
python plot_state_clusters.py --out-root results_large --metrics js
```

### 7.2 What is produced

For every experiment and every metric, one 2 × 2 figure is written to
`results_quick/<tag>/<tag>_states_<metric>.png`:

```
forward / TRAIN          backward / TRAIN
forward / HELD-OUT       backward / HELD-OUT
```

Each panel is a UMAP projection of the model's **predictive distribution** at
the maximum-context position of every sequence. Colour encodes the input
**token**; marker shape encodes the **recovered causal state**. The two
encodings are independent, which permits the central claim to be read directly
off the figure: on the coin's forward arm, tokens 0 and 2 appear in *different
colours* but with the *same marker*, within a single group — two tokens, one
causal state.

### 7.3 Why the train/held-out split appears here

As noted in Section 3.3, the complexity and latent quantities reported elsewhere
are computed on the full dataset, roughly 80 % of which the model was trained
on. These figures quantify what that costs. If the recovered state structure is
genuine rather than memorised, the held-out row must recover the same k̂ as the
train row.

The split is reproduced **exactly**, not approximated. PyTorch's `random_split`
draws `randperm(N, generator)` and slices it, so the partition depends only on N
and the seed rather than on the dataset contents; the reproduction is verified
against the training pipeline's own indices by a test.

The measured outcome: **k̂ agrees between train and held-out on 12 of 14 arms.**
The exception is the flower configuration with n = 2, m = 8 backward, which has
the largest number of states (9) and therefore the fewest held-out sequences per
state.

### 7.4 The three distance metrics

The predictive distributions are points on the probability simplex. Three
distances are implemented:

| Metric | Definition | Range | Default threshold |
|---|---|---|---|
| `euclidean` | √(Σ(pᵢ − qᵢ)²) | [0, √2] | 0.100 |
| `tv` | ½·Σ\|pᵢ − qᵢ\| | [0, 1] | 0.075 |
| `js` | √(Jensen–Shannon divergence) | [0, 1] bit | 0.075 |

The thresholds are **not interchangeable**, because the ranges differ. Each is
the value that maximised agreement with the closed forms across the fourteen
arms.

Jensen–Shannon has the strongest theoretical justification: it is symmetric,
bounded, finite in the presence of zero probabilities (where Kullback–Leibler
would diverge), and its square root satisfies the triangle inequality. On this
data, however, it performs marginally *worse* than plain Euclidean distance —
12 of 14 arms correct against 13 of 14. All three agree on 12 or 13 arms and
disagree only where the model itself under-resolves the states.

---

## 8. The post-hoc evaluators

```bash
python Test_data_eval.py
python LLM_asymmetry_testing.py
```

These load saved checkpoints and score them on freshly generated data.

### 8.1 Configuration

Neither script has a command-line interface. Each is configured by a `RUN`
dictionary — near the bottom of `Test_data_eval.py`, and at approximately line
61 of `LLM_asymmetry_testing.py`:

```python
RUN = dict(
    config      = "LARGE",   # which configs.py entry produced the weights
    results_dir = None,      # None → that configuration's out_root
    out_dir     = None,      # None → <results_dir>/eval
    exp         = "all",     # "all" | "exp1" | "exp1_2" | "exp2"
)
```

The `config` field must correspond to the run that produced the weights being
scored. Every checkpoint is accompanied by a JSON sidecar recording its
architecture and the process it was trained on, and the loaders verify against
it; a mismatch produces an explicit warning rather than a confusing shape error
or, worse, silently incorrect numbers.

### 8.2 A note on metrics

`Test_data_eval.py` reports two perplexities and they must not be conflated.
The **teacher-forced** perplexity scores both models on the same ground-truth
sequences and is the comparison metric. The **self-generated** perplexity scores
each model on its own sampled sequence; it estimates that model's own entropy
rate, is minimised by a degenerate over-confident model, and is labelled
informational only.

---

## 9. Verification and documentation

### 9.1 The test suite

```bash
pytest tests/ -q
```

Sixty-one tests, approximately twenty seconds. They should be run after any
modification to `Model_analysis.py`, `Training_model.py` or `OneHot_model.py`.

The suite pins the properties that fail *silently* when broken: that the loss
function remains finite at extreme logit magnitudes; that perplexity is
token-weighted; that the two arms receive identical windows; that the flower
closed forms reproduce their reference values; that vectorised code produces the
same numbers as the loops it replaced; and that the causal-state estimator
merges tokens with equal futures while separating those with different ones.

### 9.2 The walkthrough notebook

```bash
python build_walkthrough.py
```

This regenerates `walkthrough.ipynb`, a thirty-seven-cell notebook that
demonstrates every change made across Phases 0 to 3. It is generated from a
script rather than edited by hand so that it can be regenerated after a re-run.
Each defect is recomputed side by side with its correction in the same cell, so
that the claims may be checked rather than taken on trust. It closes with a
consolidated list of caveats.

---

## 10. Understanding the outputs

### 10.1 Directory layout

```
results_quick/
├── run_config_QUICK.json                  complete parameter set + timestamp
├── run_config_QUICK_LARGE_HMM.json
├── all_results.pkl                        every experiment, merged by tag
├── models/
│   ├── exp1_coin_p040_q080_fw.pt          weights
│   └── exp1_coin_p040_q080_fw.json        sidecar: d_model, n_layers, p, q, seed, …
└── exp1_coin_p040_q080/
    ├── results.pkl                        all metrics and curves
    ├── ..._fw_cv.png                      six-panel cross-validation figure
    ├── ..._loss_theory.png                loss curves against H∞
    ├── ..._fw_umap.png                    latent UMAP, three panels
    ├── ..._umap_compare.png               forward against backward
    ├── ..._states_{euclidean,tv,js}.png   causal states, train against held-out
    ├── ..._attn.png, ..._attn_compare.png attention heatmaps
    └── ..._complexity_compare.png         empirical against theoretical complexity
```

Everything other than the source files is excluded from version control and
should be regenerated rather than committed.

### 10.2 The schema of `results.pkl`

```python
import pickle
r = pickle.load(open("results_quick/exp1_coin_p040_q080/results.pkl", "rb"))
```

| Key | Contents |
|---|---|
| `tag`, `seed`, `p`, `q` (or `n`, `m`) | Identification and parameters. |
| `theory` | H∞ for the process. |
| `C_plus`, `C_minus` | Theoretical forward and backward complexity. |
| `cv_fw`, `cv_bw` | Cross-validation results for each arm. |
| `ana_fw`, `ana_bw` | Analysis results for each arm. |
| `paired` | The paired ΔCE statistics. |

Within `cv_fw` and `cv_bw`:

| Key | Contents |
|---|---|
| `fold_test_loss`, `fold_test_ppl` | Per-fold performance on the held-out set. |
| `fold_val_loss`, `fold_val_ppl` | Per-fold validation performance. |
| `best_fold`, `test_loss`, `test_ppl` | The selected fold and its performance. |
| `best_curves`, `fold_curves` | Training curves: `step_loss`, `step_ppl`, `step_at`, `step_val_loss`, `step_val_ppl`, `step_val_at`, `epoch_loss`. |
| `fold_divergence` | Per-fold divergence diagnosis. |

Note that `step_val_at` exists because validation is recorded on a sparser
cadence than the training loss; plotting the validation series against its list
index rather than against `step_val_at` would compress the run into the first
few percent of the axis.

Within `ana_fw` and `ana_bw`:

| Key | Contents |
|---|---|
| `S_emp` | Complexity from k-means at an **assumed** k. |
| `S_hat`, `k_hat` | Complexity and state count from the **recovered** clustering. |
| `k_stability`, `k_plateau` | k̂ across a grid of thresholds, and the most persistent value. |
| `S_theory` | The closed-form value. |
| `umap_coords` | Cached projections. |

Within `paired`:

| Key | Contents |
|---|---|
| `fold_ce_fw`, `fold_ce_bw`, `fold_delta` | Per-fold values; nothing is discarded. |
| `mean`, `sd`, `sem`, `t`, `n` | The headline statistics. |
| `all_folds`, `converged_only` | The same statistics computed both ways. |
| `converged`, `conv_tol` | Which folds passed the convergence filter, and the tolerance used. |

Because every fold is retained, an alternative convergence criterion may be
applied to existing results without re-running anything.

---

## 11. Interpreting the results

### 11.1 The current findings

The QUICK configuration converges: the forward cross-entropy exceeds H∞ by
between 0.0010 and 0.0082 bits across the seven experiments. The measured ΔCE
values are of order 0.001 to 0.003 bits, with fold-level standard errors of
comparable magnitude:

| Experiment | ΔCE | Standard error | Verdict |
|---|---|---|---|
| `exp1_2_coin_p010_q090` | +0.0026 | 0.0004 | `match` |
| `exp1_coin_p030_q040` | −0.0001 | 0.0012 | `n.s.` |
| `exp1_coin_p040_q080` | −0.0004 | 0.0013 | `n.s.` |
| `exp2_flower_n2_m6` | −0.0006 | 0.0025 | `n.s.` |
| `exp2_flower_n2_m8` | +0.0016 | 0.0004 | `match` |
| `exp2_flower_n4_m2` | −0.0015 | 0.0017 | `n.s.` |
| `exp2_flower_n6_m4` | +0.0015 | 0.0023 | `n.s.` |

Five of the seven are `n.s.` — not distinguishable from zero. Two are marked
`match`, meaning the sign of ΔCE agrees with the sign of C⁻ − C⁺ and the
magnitude exceeds twice the standard error.

**Those two verdicts should not be over-read.** The effects concerned are
+0.0026 and +0.0016 bits — a few thousandths of a bit. They clear the threshold
only because the fold-level standard error is itself very small (0.0004), and
that standard error is computed over five folds that **share a training set**.
It therefore measures fold-to-fold variability rather than sampling variability
over datasets, and is a lower bound on the true uncertainty. Establishing an
effect of this size would require the seed-repeat harness, which has not yet
been built.

The overall picture is a **null result**, which is precisely what Section 2.3
predicts for a converged predictor with capacity to spare. It should not be
reported as "no causal asymmetry", because it remains ambiguous until the
`d_model` sweep distinguishes that conclusion from "capacity absorbed the
asymmetry".

### 11.2 Caveats attaching to every reported number

1. ΔCE is a difference of residuals; a converged, sufficient-capacity model
   yields approximately zero regardless of C⁻ − C⁺.
2. The standard error is computed over five folds **that share a training
   set**. It measures fold-to-fold variability, not sampling variability over
   datasets, and is therefore a lower bound on the true uncertainty. It must not
   be presented as a confidence interval.
3. `S_emp` uses a **pre-specified** k and satisfies S ≤ log₂k, so it will
   confirm whatever k it is given. The recovered `S_hat` is preferable: across
   the fourteen arms its mean absolute error against the closed forms is 0.0260
   bits, against 0.1104 for the fixed-k estimator.
4. The clustering threshold underlying k̂ is a genuine free parameter. The
   smallest true separation between backward states ranges from 0.135 to 0.612
   across the configurations in this repository, so no single value is correct
   everywhere. The stability profile is therefore always reported, and
   "k̂ = 2, stable across tolerances 0.10 to 0.60" is the defensible claim.
5. UMAP is a non-linear, **non-distance-preserving** embedding. The number of
   visually apparent groups is not a state count; k̂ is computed in the full
   space and annotated on the figures for this reason.
6. Results are not bit-reproducible across runs on MPS, though the pairing of
   the two arms is unaffected.

### 11.3 The LARGE configuration

**The cross-validation results from the LARGE configuration are not usable.**
In the completed run, **all forty folds diverged** — five of five on each of its
eight arms. The paired ΔCE consequently carries standard errors of 0.11 to 0.15,
with one value reported as `nan`.

The cause has been isolated by single-factor ablation and is **not** the scale
of the data. Holding the process fixed and varying one factor at a time, only
those factors that increase the **number of gradient steps** produce divergence;
learning rate, chunk length and batch size have no effect in isolation. The
onset occurs consistently at around step 90 to 255. QUICK's 100 steps per fold
stop before that window opens; LARGE's 1600 are well beyond it.

The mechanism has been measured directly. Cross-entropy on a **deterministic
transition** — and both processes contain them — has no finite minimiser, so
gradient descent continues to enlarge the logit gap indefinitely even after the
loss has stopped improving. Over 80 epochs the training loss remains flat at
H∞ from step 100 to step 800 while the maximum logit magnitude grows from 2.9 to
8.7 and peaks at 12.8, after which a single step overshoots, the loss spikes,
and the model settles at a worse solution than it had already reached. Nothing
in the loss curve gives warning.

What LARGE still provides is usable: the trained weights and their sidecars, the
figures, and the p-q heatmaps, the last of these being unaffected because the
sweep does not use cross-validation.

Should usable LARGE numbers be required, the appropriate correction is
`AdamW(weight_decay=0.01)`, which restores a finite optimum and reduced the
measured divergence rate from four in six to one in six. Adopting it would
change every number, so QUICK and the controls would require re-running under
the same optimiser to remain comparable.

### 11.4 Which run to report

**QUICK.** All five folds converge, in all seven experiments.

---

## 12. Troubleshooting

**A test fails.** Do not proceed to experiments. The tests cover failures that
are silent in ordinary use.

**`load_state_dict` raises a shape error.** The checkpoint was produced by a
different configuration. Check the JSON sidecar beside the `.pt` file and set
`RUN["config"]` accordingly.

**A run reports `k̂ != theory`.** This is frequently correct behaviour: the
estimator measures what the *model* represents, and a model that has not
resolved all the states of its process will legitimately return a smaller k̂.
Consult the stability profile before concluding that the estimator is at fault.

**A run reports many diverged folds.** See Section 11.3. This is expected for
long runs and is reported rather than concealed.

**The paired standard error is `nan`.** Fewer than two folds survived the
convergence filter. The paired statistic is not meaningful for that arm.

**Results differ slightly between identical runs.** Expected on MPS; see
Section 1.2.

---

## 13. Command reference

```bash
# Environment
conda activate qdrug
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version

# Verification
pytest tests/ -q                                    # 61 tests, ~20 s
python run_experiments.py --config SMOKE            # ~2 min, exercises all paths

# Training
python run_experiments.py --config QUICK            # ~9 min   ← the reportable run
python run_experiments.py --config QUICK_LARGE_HMM --only exp1
python run_experiments.py --config QUICK_LARGE_HMM --only exp2
python run_experiments.py --config LARGE            # ~3.5 h   ← CV not usable
python run_experiments.py --config QUICK --seed 1 --out-root results_quick_seed1

# Controls
python sanity_check.py                              # ~7 min

# Analysis of saved weights
python plot_state_clusters.py                       # all metrics, results_quick
python plot_state_clusters.py --out-root results_large --metrics js

# Post-hoc evaluation (edit the RUN dict first)
python Test_data_eval.py
python LLM_asymmetry_testing.py

# Documentation
python build_walkthrough.py                         # regenerate walkthrough.ipynb
```

### Further reading within this repository

| Document | Contents |
|---|---|
| `IMPROVEMENT_PLAN.md` | The audit that motivated the work, with findings by tier. |
| `PHASE0_HANDOFF.md` | Repository hygiene and environment. |
| `PHASE1_2_HANDOFF.md` | Correctness fixes and the consolidation of the runners. |
| `PHASE3_HANDOFF.md` | Performance work, the latent-sampling correction, and causal-state recovery. |
| `walkthrough.ipynb` | An executed demonstration of every change, with caveats. |
