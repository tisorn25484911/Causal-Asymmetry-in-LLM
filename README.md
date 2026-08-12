# Thermodynamic Consequences of Causal Asymmetry in Transformer Language Models

This repository tests, empirically and on processes whose answer is known
analytically, whether a transformer trained to predict a stochastic process
*backwards* pays a measurable price when the backward direction requires more
memory than the forward one.

The mechanism under test comes from computational mechanics and from Thompson et
al. (2018): a stationary process has a forward statistical complexity **C⁺** and
a backward one **C⁻**, the minimum memory an optimal predictor needs in each
direction. When C⁺ ≠ C⁻ the process is **causally asymmetric** — one direction is
intrinsically cheaper to model. The question here is whether a finite-capacity
neural predictor exhibits that asymmetry as excess cross-entropy.

**The headline result, stated up front.** Two halves, with opposite outcomes.

- **The representational half is confirmed.** The models demonstrably build the
  ε-machine of the direction they are trained in. Across fourteen trained arms
  the recovered number of causal states and their occupancy entropy match the
  closed forms to a mean absolute error of **0.026 bits**, and they do so
  *asymmetrically in the right direction*: a coin's backward arm recovers three
  states where its forward arm recovers two, and the flower arms swap when m > n
  becomes n > m.
- **The thermodynamic half is not confirmed.** Across 1,400 trainings — seven
  processes, 100 independent datasets each — ΔCE = CE_BW − CE_FW is between
  0.0003 and 0.0024 bits, that is **0.03 % to 0.13 % of the entropy rate**. At
  that sample size the effect is resolvable, and it does **not** follow the
  theory: two processes come out significantly in the predicted direction and
  **two significantly in the opposite direction**. **The extra memory is real,
  and at this capacity it is free.**

That second outcome is what the theory predicts for an over-provisioned,
converged model — Section 1.2 explains why — so it is a null on the hypothesis
rather than a refutation of it, and it is not yet interpretable as either.
Section 11 sets out what it does and does not license, and what would settle it.

---

## Contents

1. [The question](#1-the-question)
2. [The two processes](#2-the-two-processes)
3. [Model architecture](#3-model-architecture)
4. [Experimental design](#4-experimental-design)
5. [The experiments](#5-the-experiments)
6. [The analysis](#6-the-analysis)
7. [Configurations](#7-configurations)
8. [Repository structure](#8-repository-structure)
9. [Running](#9-running)
10. [Results](#10-results)
11. [What the results mean](#11-what-the-results-mean)
12. [Limitations and open work](#12-limitations-and-open-work)
13. [References and related work](#13-references-and-related-work)

For operating instructions — every flag, every output file, troubleshooting —
see **`HOW_TO_RUN.md`**. This document is about *what* is being done and *why*.

---

## 1. The question

### 1.1 The hypothesis

For a stationary process, the **forward causal states** are the equivalence
classes of pasts that induce the same distribution over futures (Shalizi &
Crutchfield 2001). C⁺ is the Shannon entropy of the stationary distribution over
those states: the memory, in bits, that an optimal forward predictor must carry.
C⁻ is the same quantity for the time-reversed process. Generically C⁺ ≠ C⁻ —
prediction is not a symmetric activity.

Applied to a neural predictor, the conjecture is:

```
CE_FW  →  H∞ + residual_FW
CE_BW  →  H∞ + residual_BW
ΔCE    =  CE_BW − CE_FW  should track  sign(C⁻ − C⁺)
```

### 1.2 What the design can and cannot show

**This is the single most important section in this document.** The entropy rate
H∞ is **time-reversal invariant**: a process and its reverse have the same H∞.
The code relies on this — one `theory` value is passed to both arms. Therefore

```
ΔCE = CE_BW − CE_FW = residual_BW − residual_FW
```

ΔCE is a difference of **residuals**, not of entropy rates. An
unbounded-capacity, fully-converged predictor achieves residual ≈ 0 in *both*
directions and hence ΔCE ≈ 0 **regardless of C⁻ − C⁺**. ΔCE > 0 is a prediction
about *memory-bounded* predictors only: the backward direction needs more causal
states, so at fixed `d_model` and a fixed step budget it leaves a larger
residual.

Three consequences constrain every claim this repository can make:

1. **Only the sign of ΔCE is predicted**, and only when C⁻ ≠ C⁺. The magnitude
   has no theoretical anchor, so there is nothing to compare an effect size
   against.
2. **A null result is ambiguous.** "No asymmetry" and "capacity was large enough
   to absorb it" produce the same number. Distinguishing them requires varying
   capacity — the `d_model` sweep, not yet run.
3. **The measurement must be paired.** The predicted effect is a few hundredths
   of a bit at most. Any nuisance difference between the two arms — a different
   data split, different windows, different initialisation — is of that order or
   larger, so an unpaired comparison cannot resolve the effect it is looking for.

### 1.3 What is actually measured

Four quantities, in decreasing directness:

| Quantity | How | What it establishes |
|---|---|---|
| **ΔCE** | Held-out cross-entropy of both arms on the same sequences, paired per fold | The thermodynamic claim |
| **C⁺, C⁻ recovered** | Clustering the model's predictive distribution | Whether the ε-machine was learned at all |
| **Convergence** | CE against the analytic H∞ | Whether the residuals are small enough for ΔCE to be about asymmetry rather than about optimisation |
| **p-q map** | Coarse sweep over coin parameters | Where in parameter space the asymmetry is largest |

---

## 2. The two processes

Both are hidden Markov models with an analytically known H∞, C⁺ and C⁻. That is
the point of using them: the answer is known before the model is trained.

### 2.1 The coin process

Two hidden states, 0 and 1, with `p = P(0→1)` and `q = P(1→0)`. Three
observable tokens (`Data_generation.coin_generation`):

| Token | Emitted when |
|---|---|
| `0` | in state 0, and was in state 0 |
| `1` | in state 1 |
| `2` | in state 0, having just arrived from state 1 |

The true conditionals (`Model_analysis.coin_true_conditional`) are what make this
a useful test case:

```
forward  P(next | token)          backward  P(previous | token)
  0 →  [1−p,  p,   0 ]              0 →  [1−p,     0,    p  ]
  1 →  [ 0,  1−q,  q ]              1 →  [q(1−p), 1−q,  pq  ]
  2 →  [1−p,  p,   0 ]              2 →  [ 0,      1,    0  ]
```

Tokens 0 and 2 have **identical forward futures** — they are two tokens forming
**one** forward causal state. Backwards, all three rows differ, so there are
three. Hence

```
C⁺ = H( q/(p+q),  p/(p+q) )                        2 states
C⁻ = H( q(1−p)/(p+q),  p/(p+q),  pq/(p+q) )        3 states
H∞ = π₀·H₂(p) + π₁·H₂(q)
```

The backward states **refine** the forward ones — token 1 is state 1; tokens 0
and 2 split forward-state 0 according to the previous state — so **C⁻ ≥ C⁺ for
every (p, q)**. The coin is a positive case everywhere in its parameter space,
which a test asserts.

### 2.2 The flower process

*n* dice with *m* faces each (`Flower_process_generation.flower_process_generation`).
Each cycle: choose a die *i* uniformly and emit token *i*; roll it and emit token
*n + j*. Vocabulary n + m, and the sequence strictly alternates
selection, outcome, selection, outcome.

**`seq_len` is measured in cycles, not tokens** — a sequence is `2 × seq_len`
tokens long. The first `pre_depth = 10` cycles are discarded.

Forward causal states: one "a roll just happened, the next token is a uniform
selection" state, plus one state per die (the next token is that die's
distribution). Their probabilities are ½ and 1/(2n) each, so

```
C⁺ = 1 + ½·log₂ n
```

Backward causal states: one "the current token is a selection" state, plus one
state per **distinguishable** outcome. Two outcomes *j* and *j′* are the *same*
backward state exactly when they induce the same posterior P(die | outcome) —
that is, when columns *j* and *j′* of `dice_probs` are proportional. Merging
those,

```
C⁻ = 1 + ½·H(π_merged),     π_j = (1/n)·Σᵢ dice_probs[i, j]
```

That word *distinguishable* is a correction to an earlier version of this
repository, which counted m + 1 backward states unconditionally. Two consequences
follow from the corrected form:

- **n = 1 degenerates.** With a single die, P(die | outcome) = 1 for every
  outcome, so *all* m outcomes collapse into one state and C⁻ = C⁺ = 1. The
  process `0, X, 0, X, …` with X uniform is exactly time-reversible — a **null**
  control, not a positive one.
- **C⁻ ≤ 1 + ½·log₂ m, so C⁻ > C⁺ requires m > n.** Every flower configuration
  originally in this repository had n > m, which is to say all of them tested the
  *negation* of the hypothesis. Both signs are now run deliberately, because a
  result in which the sign of ΔCE follows the sign of C⁻ − C⁺ *across*
  configurations is far stronger than a single positive case.

Entropy rate: `H∞ = ½·log₂ n + ½·meanᵢ H(dice_probs[i])`.

### 2.3 Reference values

Verified by `tests/test_theory.py` against the closed forms and, for the
conditionals, against a Monte-Carlo estimate over the generator itself.

| Process | C⁺ | C⁻ | C⁻ − C⁺ | H∞ |
|---|---|---|---|---|
| coin p=0.5, q=0.5 | 1.0000 | 1.5000 | +0.5000 | 1.0000 |
| coin p=0.1, q=0.9 | 0.4690 | 0.8911 | +0.4221 | 0.4690 |
| coin p=0.3, q=0.4 | 0.9852 | 1.4888 | +0.5036 | 0.9197 |
| coin p=0.4, q=0.8 | 0.9183 | 1.5656 | +0.6473 | 0.8879 |
| flower n=1, m=2 (fair) | 1.0000 | 1.0000 | 0.0000 | 0.5000 |
| flower n=2, m=6 | 1.5000 | 2.1137 | +0.6137 | 1.5343 |
| flower n=2, m=8 | 1.5000 | 2.4765 | +0.9765 | 1.7800 |
| flower n=4, m=2 | 2.0000 | 1.4952 | −0.5048 | 1.3364 |
| flower n=6, m=4 | 2.2925 | 1.9899 | −0.3026 | 2.0985 |

Flower values are for dice drawn from `default_rng(flower_dice_seed = 42)`,
which is what every run uses, so these are the actual processes trained on.

---

## 3. Model architecture

One class, `OneHot_model.OneHotDecoder`, with a single flag selecting the causal
direction.

```python
model_fw = OneHotDecoder(token_size=3, d_model=32, max_len=1999, mode="forward")
model_bw = OneHotDecoder(token_size=3, d_model=32, max_len=1999, mode="backward")
```

### 3.1 The two arms

| | Forward | Backward |
|---|---|---|
| Attention mask | lower-triangular (`tril`) | upper-triangular (`triu`) |
| Position *t* attends to | x[0…t] — the past | x[t…T−1] — the future |
| `training_step` unpacks | `inputs, targets = batch` | `targets, inputs = batch` |
| Predicts | x[t+1] from x[≤t] | x[t−1] from x[≥t] |
| Maximum-context position | `T − 1` (last) | `0` (first) |
| Context accumulates | rightward | leftward |

The masks are **exact mirror images**: summed over positions the total context
budget is identical, and a test asserts `tril(T) == flip(triu(T))`. The mask is
therefore *not* a confound between the arms.

The maximum-context row of that table is load-bearing and was once wrong. With a
`triu` mask, position T−1 attends to *itself alone*; the backward model's
richest representation is at position **0**. Reading the backward arm's
complexity at the last position — as an earlier version did — clustered latents
that encode a single token.

The masks are cached per `(T, device)` in a plain dict, deliberately not a
buffer, so they never enter a checkpoint.

### 3.2 One block

`n_layers` blocks are applied in sequence. Each is pre-norm with two residual
connections:

```python
normed = ln1(x);  x = x + attn(normed, normed, normed, mask=mask)
normed = ln2(x);  x = x + ffn(normed)
```

where `attn` is single-head scaled dot-product attention (`wq`, `wk`, `wv`, each
`d_model × d_model`) and `ffn` is `Linear(d, 4d) → ReLU → Linear(4d, d)`. The
input is a **fixed random projection** of the one-hot token, and the output is a
`Linear(d_model, token_size)` read directly off the residual stream.

### 3.3 Deviations from a standard decoder, and why they matter

Four, all deliberate, and all worth knowing before comparing anything here to a
conventional transformer:

- **The input projection is frozen.** `rand_prj` is a registered *buffer*, not a
  parameter. It was an `nn.Parameter`, which made `one_hot @ rand_prj`
  mathematically a learned embedding table despite the class name and the "fixed
  random projection" comment. Freezing it is what makes a `d_model` sweep
  interpretable: varying `d_model` then varies representational capacity alone,
  rather than capacity plus the size of a learned input code.
- **Single head, and no output projection.** Attention returns `prob @ V`
  directly; there is no `W_O`.
- **No dropout, no weight decay, no gradient clipping, no learning-rate
  schedule.** Plain Adam. This matters: it is the direct cause of the training
  divergence documented in Section 4.6.
- **No final LayerNorm.** `output_prj` reads the unnormalised residual stream.

`reverse_pos_for_backward` exists and is **never set anywhere**. Both arms
receive `PE[t]` increasing left to right, so for the forward model the positional
index increases with context length and for the backward model it decreases.
This is the one remaining architectural asymmetry between the arms and the
most plausible source of a spurious ΔCE. Testing it is open work
(Section 12).

### 3.4 The loss

`cross_ent_onehot` returns cross-entropy in **bits** and perplexity as `2**CE`:

```python
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1)) / math.log(2)
```

Everything downstream is in bits, comparable directly against H∞, C⁺ and C⁻.
The implementation matters. The natural-looking form

```python
-(target_prob * logits.softmax(-1).log2()).sum(dim=1)
```

returns **NaN** as soon as any non-target class probability underflows to exactly
zero: that term is `0 · −inf`, and the NaN propagates through the sum and poisons
the whole batch even though the target class was fine. This is reachable rather
than hypothetical — both processes contain deterministic transitions, and
cross-entropy on a deterministic transition drives the logit gap toward infinity.
Measured: at logit scale 200 the old expression returns NaN where the true value
is 0.0. `F.cross_entropy` uses log-sum-exp internally and is exact at any logit
scale.

### 3.5 Capacity

Trainable parameters are `n_layers · (11·d² + 12·d) + d·V + V`:

| `d_model` | `n_layers` | Vocabulary | Parameters |
|---|---|---|---|
| 32 | 2 | 3 | 23,395 |
| 32 | 2 | 10 | 23,626 |
| 64 | 2 | 3 | 91,843 |

Hold this beside the quantity being measured: an optimal predictor of the coin
needs **0.5 to 1.6 bits** of memory, and the model carries a 32-dimensional float
residual stream. The mismatch of scales is, as Section 11 argues, the whole story
of the null result.

`n_layers` is 2 in every configuration and is deliberately never varied, because
depth is part of the same capacity budget as `d_model`; varying both would make a
change in ΔCE attributable to neither.

---

## 4. Experimental design

### 4.1 Both arms train on the same forward data

The backward model is **not** trained on reversed sequences. It sees the same
forward-generated sequences as the forward model and differs only in its mask and
its batch convention. Reversing the data would confound the comparison with a
change of dataset; the point is to hold the process fixed and change only the
causal direction of prediction.

An earlier version of the repository contained a third notion of "backward" — a
`mode="backward"` branch in the data loader that reversed the sequences
themselves — which no experiment used. It has been removed rather than left as a
trap.

### 4.2 Pairing the two arms

Per Section 1.2, the effect is smaller than plausible nuisance differences, so
the two arms are made **paired** at every level. Given the same seed, fold *k* of
the forward run and fold *k* of the backward run share:

- the same generated sequences;
- the same chunk windows (`ChunckDataset` draws one offset per sequence once at
  construction, so `__getitem__` is a pure function of the index);
- the same hold-out test set and the same fold membership (`random_split` and
  `randperm` are given explicit seeded generators);
- the same batch order (each fold loader gets `Generator().manual_seed(seed + fold)`);
- the same weight initialisation (`torch.manual_seed(seed·1000 + fold)` before
  each fit).

Each of these was once absent, and each mattered. The window bug is the
instructive one: both arms were handed the *same loader object*, and offsets were
drawn from a stateful RNG at access time, so the forward run advanced the
generator and the backward run trained on **different windows of the same
sequences**. Validation also redrew fresh windows at every recorded step, so the
validation curve carried window noise on top of learning signal and "best fold"
was chosen partly on which fold happened to draw easy windows.

The cost of the fix: a sequence now contributes one window for the whole run
rather than a fresh one each epoch. That is a deliberate trade — exact pairing is
what the study rests on, and window diversity is recoverable by generating more
sequences.

Pairing does **not** give bit-reproducibility across invocations on Apple MPS.
Measured: initialisation is identical and the first loss matches to eight
decimals, then trajectories diverge from the first backward pass. The same test
on CPU is exactly repeatable. CPU is ~6× slower, so MPS is the default and
`accelerator="cpu"` is available when exact reproducibility is worth the wall
clock. Pairing is unaffected either way.

### 4.3 Chunked training

Attention is O(T²) in memory, and sequences are 2,000–4,000 tokens.
`ChunckDataset` therefore takes one fixed-length window per sequence
(`train_chunk_len` = 256 or 512).

`max_len` — the positional-encoding table size — is nevertheless set to the
**full** input length, so the table covers the whole range regardless of what is
fed in.

### 4.4 Splits, and what is held out

```
500 sequences
├── 100  (20 %)  held-out test set, never trained on, shared by both arms
└── 400  (80 %)  train+val pool
    └── 5 folds: each fold trains on 320 and validates on 80
```

Every fold model is then evaluated on the held-out 100. The **paired ΔCE is
computed from those held-out per-fold cross-entropies**, so the headline number
is out-of-sample.

Two quantities are *not* out-of-sample, and this is stated wherever they are
reported: the empirical complexity and the latent figures are computed over the
whole dataset, roughly 80 % of which the model trained on.
`plot_state_clusters.py` exists to quantify what that costs, by drawing the same
figure for train-seen and held-out sequences side by side.

### 4.5 Analysis must run at the training chunk length

This was the most consequential finding of the rewrite, and the null control is
what caught it.

An earlier version ran analysis on **full-length** sequences, on the stated
grounds that "analysis on full sequences remains in-distribution". **That claim
is false whenever `chunk_len` < full sequence length.** A chunk is fed to the
model as a standalone sequence, so the positional-encoding index is the position
*within the chunk*: the model only ever sees PE indices [0, chunk). Evaluating at
full length asks it to extrapolate to positions it has never been trained on.

Measured on the control coin checkpoint (chunk 512, evaluated at T = 999,
H∞ = 1.0):

| positions | CE |
|---|---|
| 0–511 (trained) | **1.0249** — converged |
| 512–998 (never seen) | 1.5563 |
| whole sequence | **1.2840** ← what was being reported |

So the coin arm had converged all along; the 0.28-bit "failure to converge" was
extrapolation error.

Why this is fatal rather than merely inaccurate: it biases the two arms
**asymmetrically**. The forward arm's complexity is read at the last position and
the backward arm's at position 0 — correctly, those are the maximum-context
positions. At full length the forward read is at an untrained position and the
backward read at a trained one, so the two arms are measured under different
amounts of extrapolation, and ΔCE is a difference between them.

The first re-run of the controls after this was noticed reported **ΔCE = −0.4210
on a process that is exactly time-reversible** — precisely the failure a null
control exists to detect. `make_analysis_loader` now emits deterministic windows
at the training chunk length, from a seed offset from the training one. On the
same checkpoint:

| | before | after | theory |
|---|---|---|---|
| CE | 1.2859 | **1.0235** | 1.0000 |
| C⁺ empirical | 0.9863 | **0.9982** | 1.0000 |
| C⁻ empirical | 1.5458 | **1.5061** | 1.5000 |

### 4.6 Training divergence

Training on these processes is only marginally stable, and the failure mode is
not a bad local optimum. Folds **reach H∞ and then blow up** — to CE of 15 to 58
bits on a 3-token process, far worse than the 1.585 a uniform predictor achieves
— then partially recover. Because they recover, the failure cannot be diagnosed
from the final loss; `Training_model.diagnose_divergence` detects the
reach-then-rise shape instead.

**The cause is the same deterministic transitions that caused the NaN of Section
3.4.** Cross-entropy on a deterministic transition has no finite minimiser, so
gradient descent keeps enlarging the logit gap even after the loss has stopped
improving, until one step overshoots. Over 80 epochs the training loss sits flat
at H∞ from step 100 to step 800 while the maximum logit magnitude grows from 2.9
to 8.7 and peaks at 12.8. Nothing in the loss curve gives warning.

Single-factor ablation, holding the process fixed and changing one thing at a
time from a QUICK baseline (3 seeds each, recorded in `factorial.log`):

| Configuration | Steps | Diverged | First blow-up |
|---|---|---|---|
| A QUICK baseline | 160 | 0/3 | — |
| B + epochs 80 | 1280 | **3/3** | step 255 |
| C + d_model 64 | 160 | 1/3 | step 88 |
| D + lr 5e-3 | 160 | 0/3 | — |
| E + chunk 512 | 160 | 0/3 | — |
| F + batch 64 | 80 | 0/3 | — |
| G + n_seq 2000 | 630 | 2/3 | step 192 |
| H LARGE-like (all) | 2560 | **3/3** | step 145 |

Only the factors that increase the **number of gradient steps** produce
divergence. Learning rate, chunk length and batch size have no effect in
isolation. The onset window is roughly step 90 to 255.

Optimiser variants on the control flower process (6 seeds each, `divtest.log`):

| Configuration | Diverged |
|---|---|
| MPS, Adam, lr 5e-3 (current) | 4/6 |
| MPS, AdamW, weight decay 0.01 | 1/6 |
| MPS, Adam, lr 1e-3 | 0/6 |
| CPU, Adam, lr 5e-3 | 0/6 |

So it is an *interaction*: the landscape is sharp enough that MPS's numerical
noise tips it over where CPU's does not. Weight decay is the principled fix,
since it restores a finite optimum.

**It is left unfixed deliberately**, because changing the optimiser would change
every number in the study and require re-running everything. The consequence is
recorded rather than concealed: QUICK's 100 steps per fold stop *before* the
onset window and all its folds converge, which is why QUICK is the run that gets
reported, and why LARGE's cross-validation does not (Section 10.5).

---

## 5. The experiments

### 5.1 Experiment 1 — coin, cross-validated

Forward and backward arms on the same coin sequences at `(coin_p1, coin_q1)`,
five-fold CV, then: attention heatmaps, latent UMAPs, recovered causal states,
empirical complexity against the closed form, and the paired ΔCE.

### 5.2 Experiment 1.2 — a second coin, plus the p-q sweep

The same, at `(coin_p2, coin_q2)`, and additionally a sweep over a grid of
(p, q) values training a fresh pair of models per grid point. Four heatmaps are
produced: empirical complexity forward and backward, theoretical complexity
forward and backward, their differences, and ΔPPL.

**The sweep is exploratory, not a measurement**, for three reasons that should be
read before any heatmap is quoted: it uses **no** held-out split, so its
perplexities are in-sample; it trains on **full-length** 199-token sequences with
no chunking, a different regime from the main experiments; and it runs a single
short fit per grid point with no cross-validation. It maps *where* in parameter
space the asymmetry is largest; it does not measure how large.

The sweep is also the dominant cost of the LARGE configuration — 512 models,
about 2.5 of its 3.5 hours — and it is the one part that gains nothing from the
validation-cadence optimisation, because it trains without a validation loader.

### 5.3 Experiment 2 — flower, both signs

One experiment per entry in `flower_configs`, currently one with m > n (predicting
ΔCE > 0) and one with n > m (predicting ΔCE < 0). The runner prints the
prediction *before* training and scores the outcome against that configuration's
own prediction rather than against a blanket expectation.

`k` for the fixed-k complexity estimator is n+1 forward and m+1 backward — the
actual causal-state counts, not `n+m` for both as an earlier version used. Since
S = H(cluster occupancy) ≤ log₂ k, an inflated k inflates both numbers and
destroys the comparison. Note that m+1 is an *upper bound* backward: indistinguishable
outcomes merge (Section 2.2), which is exactly what the recovered k̂ detects.

### 5.4 The controls — `sanity_check.py`

Two processes with opposite predictions, and a result is only interpretable if
both come out right.

- **Positive control**: coin p = q = 0.5. H∞ = 1.0 exactly, C⁺ = 1.0, C⁻ = 1.5.
  Predicts ΔCE > 0.
- **Null control**: flower n = 1, m = 2, fair die. C⁻ = C⁺ = 1.0 and the process
  is exactly time-reversible. Predicts ΔCE ≈ 0.

The null control is the more important of the two: a pipeline that reports
ΔCE > 0 on a time-reversible process is measuring an artefact of the
architecture, not causal asymmetry. It has already earned its place twice — it
caught the analysis-length defect of Section 4.5, and it is what forced the
correction of the flower closed form in Section 2.2, since this arm was
originally labelled a *positive* case and its correct answer (ΔCE ≈ 0) was being
printed as `FAIL`.

Verdicts are three-way — `PASS`, `INCONCLUSIVE`, `FAIL` — because a near-zero
ΔCE on the *positive* control is not a refutation, for the reason in Section 1.2.
An earlier version scored it with a bare sign test and reported −0.0006 bits, six
ten-thousandths of a bit, as a failure.

### 5.5 The repeat harness — `run_statistical_trj.py`

Seven processes × 100 repeats × both directions. Each repeat regenerates its
process from a fresh seed and trains one model per arm.

This exists to fix a specific weakness. Every standard error reported by
`run_experiments.py` is computed over the five CV folds of a single seed, and
those folds **share a training set**: the quantity measured is fold-to-fold
variability, not sampling variability over datasets, so it is a lower bound on
the true uncertainty and cannot support a claim about the sign of ΔCE. Repeats do
not share a training set, so their spread is sampling variability.

Design points: one fit per repeat rather than a CV, because reporting the best of
five folds is a minimum over five draws and not a sample from the same
distribution as a single draw; flower dice pinned to `flower_dice_seed`, since the
dice *are* the process and resampling them would average over different
processes; and uniform settings across all seven processes so they are comparable
to each other. Within a repeat the two arms are paired exactly as in Section 4.2.

Per process it produces the complexity with error bars over repeats, the final
loss as three bars (forward, backward, and their paired difference), and every
run's trajectory with both mean curves and a shaded interval. At the root it
writes two cross-process figures: `summary_delta_ce.png` (ΔCE with its standard
error per process, and ΔCE against C⁻ − C⁺) and `summary_grid.png`, a 3 × 7 grid
carrying all three quantities for all seven processes on one sheet — rows are the
quantities, columns the processes ordered coin-then-flower by descending
C⁻ − C⁺, so scanning a row asks directly whether anything varies with the
theoretical asymmetry.

Every panel in the grid is drawn by the same `_draw_*` helper as the
corresponding standalone figure, in a compact mode, so the two cannot disagree.
Y-scales are per column, since H∞ ranges from 0.47 to 2.10 bits across these
processes and a shared axis would compress every panel to accommodate the
largest.

**The full run is complete**: 700 repeats, 1,400 trainings, ~2.3 hours with
`--khat`, zero divergences. Its results are Section 10.3.

### 5.6 The causal-state figures — `plot_state_clusters.py`

For every experiment and every distance metric, a 2 × 2 figure: forward and
backward arms, train-seen and held-out sequences. Each panel is a UMAP of the
model's predictive distribution at the maximum-context position, with **colour
encoding the token** and **marker shape the recovered causal state**.

The two encodings are independent, which lets the central claim be read straight
off the figure: on the coin's forward arm, tokens 0 and 2 appear in *different
colours* with the *same marker*, inside one group — two tokens, one causal state.

The split is reproduced exactly rather than approximated: `random_split` draws
`randperm(N, generator)` and slices it, so the partition depends only on N and the
seed, and the reproduction is verified against the training pipeline's own
indices by a test.

---

## 6. The analysis

### 6.1 Loss and perplexity

`perplexity_calculation` is the comparison metric. Both models are scored on the
**same ground-truth sequences**, using each arm's batch convention, and CE is
accumulated **token-weighted** so that `PPL = 2**CE` holds by construction.

Both properties were once wrong in ways worth recording, because both were
silent. Perplexity was computed as `mean(2**CE_batch)` rather than `2**mean(CE)`;
by Jensen's inequality that is biased high, and on a realistic spread of
per-batch CE the bias is **+0.041 bits** — the same order as the ΔCE being
measured. The accumulators were also divided by the number of batches, so a
ragged final batch counted as much as a full one.

### 6.2 Statistical complexity: two estimators, and why the second is better

**`statistical_complexity_empirical`** — k-means on the latent vectors at the
maximum-context position, at a k the caller supplies; S = H(cluster occupancy).
This is what the earlier literature in this repository reported, and it has a
structural weakness: **S ≤ log₂ k, so it confirms whatever k it is handed.** It
also cannot discover that two tokens are one state.

**`recover_causal_states`** — the better estimator, and the reasoning behind it is
the interesting part. A causal state is an equivalence class of histories that
induce the same distribution over the **future**. It is *not* a region of latent
space. So the thing to cluster is the model's **predictive distribution**,
`softmax(output_prj(latent))`, not the latent: two histories in the same causal
state must predict alike, but nothing forces the network to give them the same
latent, and in practice it does not.

Measured on the coin's forward arm, where tokens 0 and 2 are one causal state:

```
latent distance    d(tok0, tok2) = 1.59   against a true state gap of 19.10
predictive dist.   d(tok0, tok2) = 0.041  against a true state gap of  1.065
within-token spread              = 0.004
```

Both carry the state structure, but the latent also carries token identity, which
the state does not depend on. Clustering is agglomerative with a **distance
threshold** and `n_clusters=None`, so k falls out of the data; complete linkage,
so a cluster is a set whose members are *all* mutually within tolerance — the
literal reading of "the same predictive distribution to within tolerance".

**The threshold is a genuine free parameter and no single value is correct.** The
smallest true separation between backward states ranges from 0.135 at
p = 0.1, q = 0.9 to 0.612 at p = q = 0.5, so a threshold that resolves one
configuration merges another. `recover_causal_states` therefore always returns and
prints k̂ across a grid of thresholds plus the **plateau** — the k that survives
the widest span — because "k̂ = 2, stable for tolerances 0.10 to 0.60" is
defensible where a single number is not.

### 6.3 Distances on the probability simplex

Predictive distributions are points on a simplex, where Euclidean geometry is not
the natural one. Three metrics are implemented, with per-metric thresholds
because their ranges differ:

| Metric | Definition | Range | Threshold | Arms correct |
|---|---|---|---|---|
| `euclidean` | √Σ(pᵢ−qᵢ)² | [0, √2] | 0.100 | 13/14 |
| `tv` | ½·Σ\|pᵢ−qᵢ\| | [0, 1] | 0.075 | 13/14 |
| `js` | √(Jensen–Shannon divergence) | [0, 1] bit | 0.075 | 12/14 |

Jensen–Shannon has the strongest theoretical claim. Raw KL is unusable as a
clustering metric — asymmetric, unbounded, and infinite wherever q = 0 < p, which
is fatal here because a trained model drives probabilities to ~1e-6. JS repairs
all three by comparing both distributions to their mixture: it is symmetric by
construction, bounded by 1 bit, always finite, and its square root satisfies the
triangle inequality (Endres & Schindelin 2003) — which is what a
distance-threshold clusterer actually requires.

**On this data it is nevertheless marginally worse than plain L2.** That is
worth stating plainly, since an earlier note in this repository claimed JS
"doubles the safe window" — measured on two arms of a single experiment, and not
surviving the full fourteen. All three agree on 12 or 13 of 14 and disagree only
on the high-k flower backward arms, where the model itself under-resolves the
states.

### 6.4 Latent visualisation, and how it misleads

UMAP figures are kept, but annotated, because they were actively misleading in
two distinct ways.

**They used to show one trajectory, not a distribution.** The code embedded
`latents.reshape(-1, d)[:n_pts]`. The reshape is row-major, so those first rows
are every position of the first ⌈n_pts/T⌉ sequences — four sequences at chunk
256, and at T = 999 with `n_pts = 500`, a **single** sequence. Every latent figure
in the repository was one realisation.

Two panels are now drawn, because each is misleading alone:

- **`per_sequence`** — one position per sequence, at the maximum-context position
  for that direction. This is the exact slice the complexity estimator clusters,
  so the picture and the number cannot tell different stories, and since all
  points share a context length any spread is structure.
- **`random`** — uniform over (sequence, position) across all sequences, minus a
  **direction-aware** burn-in: the first positions for a forward model, the
  *last* for a backward one. Richer, but mixes context lengths.

**UMAP is scale-free, so a blob count is not a state count.** It builds a k-NN
graph, so what matters is separation relative to *local* spread, not absolute
distance. On the coin's forward arm the model has not perfectly merged tokens 0
and 2 — they sit 0.041 apart against a true state gap of 1.065, 26× larger — but
the within-token spread is 0.004, giving a separation-to-spread ratio of 10.8,
which reads to a k-NN graph as a clean boundary. So the figure drew **three**
blobs for a two-state process. Meanwhile k-means at k=2 in the full 32-dimensional
space recovers exactly {0,2} versus {1} with silhouette 0.967, *higher* than
k=3's 0.941.

Correcting the sampling did not cause this; it **exposed** it, by removing the
context-length smear that had been blurring the sub-split away. The figures now
carry k̂ and an explicit reminder, plus a third panel showing the predictive
distribution jittered at a quarter of the state tolerance, which draws causal
states rather than tokens.

### 6.5 Attention

`FW_BW_attention_comparison` draws both arms on the same short prefix with a
shared colour scale. Attention maps are retained **only** inside a
`capture_attention()` context manager: a (B, T, T) map at B = 32, T = 1999 is
511 MB retained per forward pass, which only the plotting helpers ever need. The
plotted map is the **final layer only**, so at `n_layers = 2` the figure
under-reports what the model does; every layer's map is available on
`last_attention_layers`.

### 6.6 Metrics kept for reference, not comparison

Two functions in this repository must not be used to compare the arms, and are
named and documented accordingly:

- **`self_generated_entropy_rate`** (formerly `perplexity_ind_model`) samples a
  sequence *from the model* and scores the model's log-loss on that same
  sequence. It is an estimate of the model's own entropy rate, and a degenerate,
  over-confident model **minimises** it. Each model is evaluated on its own
  sequence from its own state distribution, so it cannot say which of two models
  is better. It was once the headline chart of one evaluator.
- **`perplexity_ind_CE`** scores the model against the analytic conditional
  (a soft label) on ground-truth sequences. This is legitimate, but note the
  history: an earlier autoregressive version grew the backward model's context
  leftward using *backward* HMM transitions, giving it out-of-distribution
  context and making the backward arm look artificially accurate — reversing the
  asymmetry signal. It now uses the ground-truth loader so both arms see forward
  sequences, exactly as in training.

---

## 7. Configurations

All configurations live in `configs.py`, each with its own `out_root` so two
cannot overwrite each other's weights. Tags are **derived from parameter values**
(`utils.coin_tag`, `utils.flower_tag`), so a tag can never disagree with the run
that produced it — a real defect once, when two runners used the same tag literal
at different parameters and overwrote each other's checkpoints.

| | SMOKE | QUICK | LARGE | QUICK_LARGE_HMM |
|---|---|---|---|---|
| `out_root` | `All_Results/results_smoke` | `All_Results/results_quick` | `All_Results/results_large` | `All_Results/results_quick` |
| `d_model` | 16 | 32 | 64 | 32 |
| `lr` | 1e-2 | 1e-2 | 5e-3 | 1e-2 |
| `train_chunk_len` | 64 | 256 | 512 | 256 |
| Sequences | 40 | 500 | 2000 | 500 |
| Coin length | 200 | 2000 | 2000 | 2000 |
| Epochs | 3 | 10 | 80 | 10 |
| Batch | 8 | 32 | 64 | 32 |
| Folds | 2 | 5 | 5 | 5 |
| p-q grid | 2 pts | 10 pts | 16 pts | 10 pts |
| Flower configs | (2,3), (3,2) | (2,6), (4,2) | (2,8), (6,4) | (2,8), (6,4) |
| Runtime | ~2 min | ~9 min | ~3.5 h | ~9 min |

`n_layers = 2`, `seed = 0`, `val_every_n_steps = 25` and `state_tol = 0.10` are
shared by all.

**QUICK_LARGE_HMM** exists because QUICK and LARGE differ along four axes at
once — data volume, capacity, optimisation, *and the processes themselves*, which
have different entropy rates and different theoretical asymmetries. That last
one makes them non-comparable. Running LARGE's processes at QUICK's settings
holds the process fixed, so QUICK versus LARGE becomes a controlled comparison of
scale, capacity and learning rate. It writes into QUICK's `out_root`, and because
tags are derived, the process the two configurations share is not duplicated.

---

## 8. Repository structure

```
Transformer_model/                the primitives — no dependency on a runner
├── Data_generation.py            coin generator, CoinDataset, make_loader
├── Flower_process_generation.py  flower generator, FlowerDataset (single definition)
├── OneHot_model.py               OneHotDecoder, attention, PE, loss in bits
├── Training_model.py             chunking, loaders, CV pipeline, recorder, divergence
├── Model_analysis.py             CE/PPL, complexity (both estimators), distances,
│                                 UMAP, attention, paired ΔCE, result slimming
├── utils.py                      tags, filesystem, path anchoring, entropy rate,
│                                 checkpoint sidecars
├── Test_data_eval.py             post-hoc evaluation of saved weights
└── plot_state_clusters.py        causal states, train vs held-out, per metric

Experimental_setup/               the drivers
├── configs.py                    every configuration, one place
├── run_experiments.py            THE training runner — exp1, exp1.2, exp2
├── sanity_check.py               positive and null controls
├── run_statistical_trj.py        seed-repeat harness, 7 processes x 100 repeats
├── run_sweep_experiment.py       125-process sweep: does measured track theoretical?
├── pq_experiment.py              the p-q grid sweep and its heatmaps
└── LLM_asymmetry_testing.py      post-hoc metric suite (LaTeX figures)

Jupyter_notebooks/
├── walkthrough.ipynb             the Phase 0-3 demonstration (generated)
├── build_walkthrough.py          regenerates walkthrough.ipynb
├── metric_panel_plot.ipynb       ┐
├── sequence_prediction.ipynb     ├ read the saved pickles and weights
└── umap_analysis.ipynb           ┘

All_Results/                      every results tree
├── results_smoke|quick|large/    run_experiments.py, one per config
├── results_trajectories/         run_statistical_trj.py, 7 x 100 repeats
├── results_sweep/                run_sweep_experiment.py at weight_decay 0
├── results_sweep_wd/wd<λ>/       the weight-decay sweep, one folder per λ
└── results/                      the original pre-refactor outputs

Run_logs/                         run logs, requirements*.txt, run_all.sh
sanity_check_flower_process/      sanity_check.py's controls (at the root)

Verification and documentation
├── conftest.py                   makes the split tree importable for pytest
├── tests/test_theory.py          66 regression tests
├── HOW_TO_RUN.md                 the operating manual
└── implementation_logbook/       the audit, the phase handoffs, the fix plans
```

The source is split by role: `Transformer_model/` holds the primitives and
`Experimental_setup/` the drivers. The modules are **flat** — they import each
other as `from utils import ...` — so every runnable script carries a short
`sys.path` bootstrap anchored on its own `__file__`, and every command is run from
the repository root as `python <dir>/<script>.py`. Output directories are named
repo-relative and resolved through `utils.repo_path()`, so they mean the same
place whatever the working directory. `HOW_TO_RUN.md` §1.2 covers this, and
`implementation_logbook/REORGANISATION_FIX_PLAN.md` records why it is done this
way rather than with packages.

Two files sit in `Transformer_model/` but import `configs` from
`Experimental_setup/`, so that directory is not yet the clean leaf layer the split
intends — `Test_data_eval.py` and `plot_state_clusters.py` are both entry points
and arguably belong with the drivers. See the fix plan §7.1.

`walkthrough.ipynb` is generated from `build_walkthrough.py` rather than edited by
hand so it can be regenerated after a re-run.

Every `.pt` is written with a **JSON sidecar** recording `d_model`, `n_layers`,
`token_size`, `max_len`, `seed` and the process parameters, and both post-hoc
evaluators call `check_weight_meta` to verify rather than guess. Without it a
loader has to guess the architecture, and guessing wrong surfaces as a confusing
shape error — or, once `d_model` happens to line up, as silently scoring a
checkpoint against the wrong process.

---

## 9. Running

```bash
conda activate qdrug

pytest tests/ -q                              # 66 tests, ~20 s
python Experimental_setup/run_experiments.py --config SMOKE      # ~2 min, exercises every path

python Experimental_setup/run_experiments.py --config QUICK      # ~9 min  ← the reportable run
python Experimental_setup/sanity_check.py                        # ~7 min, the controls
python Transformer_model/plot_state_clusters.py                 # causal-state figures
python Experimental_setup/run_statistical_trj.py --khat          # ~2.3 h, repeat statistics (done)
python Experimental_setup/run_statistical_trj.py --plots-only    # redraw its figures, no training

python Experimental_setup/run_sweep_experiment.py --dry-run      # 125-process coverage table
python Experimental_setup/run_sweep_experiment.py --repeats 30   # ~7.6 h, the sweep (done)
```

Every command is run from the repository root; see §8 and `HOW_TO_RUN.md` §1.2.
Before resuming any long run, `--dry-run` first — it prints how many processes it
would skip and how many it would train, which is the difference between
continuing and silently starting over (`HOW_TO_RUN.md` §14.6).

`HOW_TO_RUN.md` documents every flag, every output file, the schema of each
pickle, and troubleshooting. Dependencies are in `Run_logs/requirements.txt`,
with exact pins in `Run_logs/requirements-lock.txt`.

---

## 10. Results

Two independent measurements of the same quantity are reported, and they should
be read in this order:

- **The cross-validated runs** (§10.2) — `All_Results/results_quick/all_results.pkl`, seven
  experiments, five folds of one seed each. Their standard error is over folds
  that share a training set, so it understates the uncertainty.
- **The repeat harness** (§10.3) — `All_Results/results_trajectories/all_trajectories.pkl`,
  seven processes × 100 independent datasets, 1,400 trainings. Its standard error
  is sampling variability. **This is the measurement to quote.**

Both used `d_model = 32`, `n_layers = 2`, base seed 0.

### 10.1 Convergence

Every arm of every run converged. Forward cross-entropy exceeds the analytic H∞
by **0.0010 to 0.0082 bits** in the cross-validated runs and by **0.0023 to
0.0117 bits** over the 100-repeat means. This matters because ΔCE is only about
asymmetry once both residuals are small; had the arms been 0.3 bits off H∞, the
difference between them would have been about optimisation instead.

The repeat harness recorded **zero divergences in 1,400 trainings**, consistent
with Section 4.6: at 130 gradient steps per run it stops just short of the onset
window.

### 10.2 The asymmetry signal — cross-validated

| Experiment | H∞ | CE_FW | CE_BW | ΔCE | sem | C⁻−C⁺ | Verdict |
|---|---|---|---|---|---|---|---|
| `exp1_2_coin_p010_q090` | 0.4690 | 0.4700 | 0.4726 | **+0.0026** | 0.0004 | +0.4221 | `match` |
| `exp1_coin_p030_q040` | 0.9197 | 0.9248 | 0.9247 | −0.0001 | 0.0012 | +0.5036 | `n.s.` |
| `exp1_coin_p040_q080` | 0.8879 | 0.8944 | 0.8941 | −0.0004 | 0.0013 | +0.6473 | `n.s.` |
| `exp2_flower_n2_m6` | 1.5343 | 1.5425 | 1.5419 | −0.0006 | 0.0025 | +0.6137 | `n.s.` |
| `exp2_flower_n2_m8` | 1.7800 | 1.7830 | 1.7846 | **+0.0016** | 0.0004 | +0.9765 | `match` |
| `exp2_flower_n4_m2` | 1.3364 | 1.3401 | 1.3387 | −0.0015 | 0.0017 | −0.5048 | `n.s.` |
| `exp2_flower_n6_m4` | 2.0985 | 2.1026 | 2.1041 | +0.0015 | 0.0023 | −0.3026 | `n.s.` |

`n.s.` denotes |ΔCE| < 2 × sem — not distinguishable from zero. Five of seven are
`n.s.`; the two marked `match` have the sign predicted by C⁻ − C⁺ and exceed
twice their standard error. Section 11.4 explains why those two should not be
read as confirmations, and §10.3 supersedes this table.

### 10.3 The asymmetry signal — 100 independent repeats

Each row is 100 independently generated datasets, each trained in both directions
from the same seed, and the difference taken within a repeat. Columns are ordered
as in `summary_grid.png`: coin then flower, each by descending C⁻ − C⁺.

| Process | C⁺ | C⁻ | C⁻−C⁺ | CE_FW | CE_BW | ΔCE | sem | t | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `coin_p040_q080` | 0.918 | 1.566 | +0.647 | 0.8913 | 0.8902 | −0.0011 | 0.0002 | −4.86 | **`MISMATCH`** |
| `coin_p030_q040` | 0.985 | 1.489 | +0.504 | 0.9236 | 0.9226 | −0.0011 | 0.0003 | −3.64 | **`MISMATCH`** |
| `coin_p010_q090` | 0.469 | 0.891 | +0.422 | 0.4713 | 0.4716 | +0.0003 | 0.0002 | +1.28 | `n.s.` |
| `flower_n2_m8` | 1.500 | 2.477 | +0.977 | 1.7870 | 1.7894 | **+0.0024** | 0.0004 | +5.56 | `match` |
| `flower_n2_m6` | 1.500 | 2.114 | +0.614 | 1.5412 | 1.5418 | +0.0006 | 0.0004 | +1.81 | `n.s.` |
| `flower_n6_m4` | 2.292 | 1.990 | −0.303 | 2.1101 | 2.1095 | −0.0006 | 0.0004 | −1.57 | `n.s.` |
| `flower_n4_m2` | 2.000 | 1.495 | −0.505 | 1.3420 | 1.3413 | −0.0007 | 0.0003 | −2.42 | `match` |

Four of the seven effects are now resolvable — **two in the predicted direction,
two against it.** Three remain indistinguishable from zero. Every magnitude lies
between 0.0003 and 0.0024 bits, which is 0.03 % to 0.13 % of the process entropy
rate.

Three aggregate statistics, none of which reaches significance:

| Statistic | Value | |
|---|---|---|
| Signs agreeing with C⁻ − C⁺ | 5 of 7 | binomial p = 0.227 |
| Pearson r(C⁻ − C⁺, ΔCE) | **+0.527** | p = 0.224, n = 7 processes |
| Range of \|ΔCE\| | 0.0003 – 0.0024 bits | no relation to \|C⁻ − C⁺\| |

The correlation point estimate is positive, i.e. in the direction the hypothesis
predicts, and with only seven processes it is nowhere near distinguishable from
zero. It is the most encouraging number in this repository and it must not be
quoted without its p-value. Section 11.4 works through what it would take to
turn it into evidence.

### 10.4 Causal states recovered — the positive result

For each arm, the assumed k, the discovered k̂, and both complexity estimates
against the closed form:

| Arm | k assumed | k̂ | plateau | S_emp | S_hat | Closed form |
|---|---|---|---|---|---|---|
| `coin_p010_q090` fw | 2 | 2 | 2 | 0.5574 | 0.5574 | 0.4690 |
| `coin_p010_q090` bw | 3 | 3 | 2 | 0.8783 | 0.8783 | 0.8911 |
| `coin_p030_q040` fw | 2 | 2 | 2 | 0.9833 | 0.9833 | 0.9852 |
| `coin_p030_q040` bw | 3 | 3 | 3 | 1.5173 | 1.5173 | 1.4888 |
| `coin_p040_q080` fw | 2 | 2 | 2 | 0.9248 | 0.9248 | 0.9183 |
| `coin_p040_q080` bw | 3 | 3 | 3 | 1.5582 | 1.5582 | 1.5656 |
| `flower_n2_m6` fw | 3 | 3 | 3 | 1.5020 | 1.5020 | 1.5000 |
| `flower_n2_m6` bw | 7 | 6 | 6 | **2.5967** | **2.0997** | 2.1137 |
| `flower_n2_m8` fw | 3 | 3 | 3 | 1.5007 | 1.5007 | 1.5000 |
| `flower_n2_m8` bw | 9 | 8 | 5 | **2.8539** | **2.2847** | 2.4765 |
| `flower_n4_m2` fw | 5 | 5 | 5 | **2.2378** | **2.0021** | 2.0000 |
| `flower_n4_m2` bw | 3 | 3 | 3 | 1.4941 | 1.4941 | 1.4952 |
| `flower_n6_m4` fw | 7 | 7 | 7 | **2.5849** | **2.2932** | 2.2925 |
| `flower_n6_m4` bw | 5 | 5 | 5 | 1.9957 | 1.9957 | 1.9899 |

**Mean absolute error against the closed forms: 0.1104 bits for `S_emp`, 0.0260
bits for the recovered `S_hat` — a 4.2× improvement.** The worst fixed-k cases
collapse: flower n=2,m=6 backward from +0.4831 to −0.0140; n=6,m=4 forward from
+0.2924 to +0.0007; n=4,m=2 forward from +0.2378 to +0.0021.

Two structural observations. First, **the two estimators return identical numbers
on 10 of 14 arms** and diverge only where the latent partition differs from the
predictive-distribution partition — which is to say the fixed-k estimator is fine
when the model has cleanly resolved its states and unreliable exactly when it has
not. Second, `S_emp` systematically **over**estimates when the true state
occupancy is unbalanced: `coin_p010_q090` forward has π = (0.9, 0.1), giving
C⁺ = 0.469, and k-means — which tends toward balanced clusters — returns 0.557.

Where k̂ ≠ theory it is the **model**, not the estimator. The flower n=2,m=8
backward arm merges several outcome tokens into one predictive distribution, so it
genuinely represents fewer states than the process has; it correctly merges the
two die *selections*, which theory also says are one backward state.

`plot_state_clusters.py` additionally reports that **k̂ agrees between train-seen
and held-out sequences on 12 of 14 arms**, the exceptions being the arms with the
most states and therefore the fewest held-out sequences per state — evidence that
the recovered structure is not memorisation.

**The repeat harness replicates all of this over 100 datasets per process**, with
standard errors on the complexity that the single cross-validated run could not
provide:

| Arm | k theory | k̂ median | S_emp | S_hat | Closed form |
|---|---|---|---|---|---|
| `coin_p040_q080` fw / bw | 2 / 3 | 2 / 3 | 0.9146 / 1.5620 | 0.9146 / 1.5620 | 0.9183 / 1.5656 |
| `coin_p030_q040` fw / bw | 2 / 3 | 2 / 3 | 0.9843 / 1.4843 | 0.9843 / 1.4916 | 0.9852 / 1.4888 |
| `coin_p010_q090` fw / bw | 2 / 3 | 2 / 3 | 0.4709 / 0.8972 | 0.4709 / 0.8020 | 0.4690 / 0.8911 |
| `flower_n2_m8` fw / bw | 3 / 9 | 3 / **7** | 1.4975 / 2.8565 | 1.5012 / 2.2219 | 1.5000 / 2.4765 |
| `flower_n2_m6` fw / bw | 3 / 7 | 3 / **6** | 1.4959 / 2.5628 | 1.5032 / 2.0253 | 1.5000 / 2.1137 |
| `flower_n6_m4` fw / bw | 7 / 5 | 7 / 5 | 2.5335 / 2.0453 | 2.2893 / 1.9882 | 2.2925 / 1.9899 |
| `flower_n4_m2` fw / bw | 5 / 3 | 5 / 3 | 2.1259 / 1.4941 | 1.9996 / 1.5099 | 2.0000 / 1.4952 |

Standard errors on the S_emp means are 0.001 to 0.013 bits, so the gaps against
the closed forms below are far larger than the sampling noise in them.

- **k̂ recovers the theoretical state count on 12 of 14 arms**, including both
  directions of every coin and the correct *inversion* of the flower arms. The
  two exceptions are the high-k flower backward arms, where the model
  under-resolves the states it should have (7 against 9, and 6 against 7).
- **Mean absolute error against the closed forms: 0.0914 bits for `S_emp`,
  0.0335 for `S_hat`** — a 2.7× improvement, and `S_hat` is the closer estimate
  on **11 of 14** arms.
- One honest exception: on `coin_p010_q090` backward, `S_emp` (error 0.0061) beats
  `S_hat` (0.0891). This is the most skewed occupancy in the set, and it is the
  case where the two estimators' biases happen to point opposite ways. `S_hat`
  is the better estimator on the evidence, not on every arm.

### 10.5 The controls

| Control | H∞ | CE_FW | CE_BW | ΔCE | Verdict | Folds diverged |
|---|---|---|---|---|---|---|
| coin p=q=0.5 (positive) | 1.0000 | 1.0059 | 1.0053 | −0.0006 | `INCONCLUSIVE` | 8/10 |
| flower n=1, m=2 (null) | 0.5000 | 0.5007 | 0.5006 | −0.0001 | `PASS` | 6/10 |

Empirical complexity on the same run: the coin gives C⁺ = 0.9991 against a theory
value of 1.0 and C⁻ = 1.4866 against 1.5; the null control gives 1.0000 in both
directions, as it must for a time-reversible process. (This run predates the k̂
estimator, so only `S_emp` is available for the controls.)

**The null control holds** — on a process that is exactly time-reversible, the
pipeline reports no asymmetry. That is the single most informative number here,
and before the analysis-length fix this same control reported −0.4210.

The positive control returns ≈ 0 as well, and is scored `INCONCLUSIVE` rather
than `FAIL`: both arms converged to within 0.006 bits of H∞, and once that
happens the residuals vanish and ΔCE → 0 whatever C⁻ − C⁺ is. Both controls carry
heavy divergence at 60 epochs, so their paired statistics rest on one or two folds
and the standard error is `nan`.

### 10.6 LARGE

**LARGE's cross-validation results are not usable.** All forty folds diverged —
five of five on each of its eight arms — leaving paired standard errors of 0.11
to 0.15 and one reported as `nan`. Section 4.6 explains the mechanism; at 80
epochs the run is far past the onset window. Its exp1, for example, reports a
converged-only mean of −0.0827 from a single surviving fold, which is one fold's
optimisation luck rather than a measurement.

What LARGE still provides is usable: the trained weights and sidecars, the
figures, and the p-q heatmaps — the last unaffected because the sweep does not
use cross-validation.

---

## 11. What the results mean

### 11.1 The magnitude of ΔCE is what the theory predicts

Section 1.2 is not a hedge written after the fact; it is a derivation. H∞ is
time-reversal invariant, so ΔCE is a difference of *residuals*, and residuals go
to zero as capacity and training suffice. At `d_model = 32` — 23,395 parameters, a
32-dimensional float residual stream — on a 3-token process whose optimal
predictor needs under 1.6 bits of memory, the model is over-provisioned by orders
of magnitude. **The correct prediction for this regime is ΔCE ≈ 0, and |ΔCE| came
out between 0.03 % and 0.13 % of the entropy rate.**

The mismatch of scales is the whole story. C⁻ − C⁺ = 0.65 bits for the coin at
p=0.4, q=0.8. Holding 1.57 bits of state instead of 0.92 costs a 32-dimensional
continuous representation nothing whatsoever. There is no reason it should show
up in the loss, and to three decimal places it does not.

What 100 repeats add is the ability to *resolve* what remains. With a standard
error of 0.0002 bits, differences of a thousandth of a bit become statistically
visible — and they turn out not to be the predicted ones (§10.3, §11.4). So the
honest statement has two parts, and both matter:

- **On magnitude, the theory is vindicated**: an over-provisioned converged model
  pays essentially nothing for the extra causal states, exactly as the residual
  argument requires.
- **On sign, at the resolution now available, the data do not support the
  hypothesis**: two processes go the predicted way, two the other way.

This may still **not** be reported as "no causal asymmetry in transformers". It
remains equally consistent with "the asymmetry was entirely absorbed by spare
capacity", and nothing in these seven processes distinguishes those two readings.
What has been established is narrower and firmer: **at this capacity the effect,
if present, is smaller than a thousandth of a bit and is not detectable by its
sign.**

### 11.2 The representational half of the hypothesis *is* confirmed

This is the finding that the ΔCE null tends to overshadow, and it is the stronger
of the two.

The models are not merely fitting a conditional table. They are constructing the
ε-machine of the direction they are trained in, and doing so accurately:

- **The state counts come out right, asymmetrically.** A coin's forward arm
  recovers 2 states; its backward arm recovers 3. The flower arms invert with the
  theory — n=2,m=6 recovers 3 forward and 6 backward; n=4,m=2 recovers 5 forward
  and 3 backward. Nothing tells the model how many states to build; k̂ is
  discovered by thresholding, and the threshold is the same 0.10 everywhere.
- **The occupancy entropies come out right**, to a mean absolute error of 0.026
  bits across the fourteen cross-validated arms, and 0.034 bits across the
  fourteen arms of the 100-repeat run — spanning C from 0.47 to 2.48.
- **It replicates.** The repeat harness recovers the same state counts on 12 of
  14 arms over 100 independent datasets per process, with standard errors on the
  complexity of 0.001 to 0.013 bits. This is not one lucky fit.
- **The structure generalises.** k̂ is stable across a wide band of thresholds,
  and agrees between train-seen and held-out sequences on 12 of 14 arms.
- **The one clean qualitative prediction is visible directly.** Tokens 0 and 2 of
  the coin have identical forward futures. The forward model merges them into one
  predictive distribution while keeping them distinguishable in its latent — and
  the state figures show them in different colours with the same marker, in one
  group.

In other words: **C⁺ and C⁻ are real, they differ, and the network finds both.**
The information-theoretic content of causal asymmetry is reproduced by a trained
transformer to two decimal places.

### 11.3 What separates the two halves

The gap between 11.1 and 11.2 is precisely the gap between *representing* extra
memory and *paying* for it. Thompson et al.'s thermodynamic argument concerns a
memory-bounded machine, where each additional causal state must be physically
instantiated and erased. A transformer with a 32-dimensional residual stream is
not that machine: within a very wide range, extra states are free.

That is not a failure of the experiment; it locates the effect. The interesting
question is no longer "does ΔCE track C⁻ − C⁺" but **"at what capacity does the
extra memory stop being free?"** That question has a shape — a curve of ΔCE
against `d_model` — and a curve is a far stronger result than any single point.
Everything in this repository is arranged so that curve can be measured: the
input projection is frozen so `d_model` varies capacity alone, `n_layers` is held
at 2 so depth does not confound it, the arms are paired so a few thousandths of a
bit is resolvable, and the harness for repeated seeds now exists.

Two further design notes follow from this reading. Depth and width both work
*against* seeing the effect, so the informative direction is downward —
`d_model` of 2 to 8 — not upward. And a transformer may be the wrong instrument
for the thermodynamic claim in the first place: the natural test is a model whose
memory is explicitly bounded and countable, where the number of states is a
parameter rather than an emergent property.

### 11.4 On the sign of ΔCE, and the one encouraging number

The cross-validated table (§10.2) marked two experiments `match`. Those two
should not be read as confirmations, for a reason that has nothing to do with
their sign: they clear the threshold only because the fold-level standard error
(0.0004) is computed over five folds that **share a training set**. That measures
fold-to-fold variability, not sampling variability over datasets, and is a lower
bound on the true uncertainty. Building the repeat harness was the correct
response to that objection, and it changed the answer.

**What the repeats show is more interesting than a flat null.** With sampling
variability and a standard error of 0.0002–0.0004 bits, four of the seven effects
resolve — and they do not agree with each other:

| | |
|---|---|
| Predicted direction, significant | `flower_n2_m8` (+0.0024), `flower_n4_m2` (−0.0007) |
| **Opposite** direction, significant | `coin_p040_q080` (−0.0011), `coin_p030_q040` (−0.0011) |
| Not resolvable | `coin_p010_q090`, `flower_n2_m6`, `flower_n6_m4` |

Where the effect is resolvable it is **as likely to point the wrong way as the
right way**, and the two wrong-way cases are the two coins with the largest
positive C⁻ − C⁺ — precisely where the hypothesis expects the clearest positive
signal. Any account of these data has to explain that, and "causal asymmetry
raises backward cross-entropy" does not.

Two aggregate statistics cut in the other direction and are worth reporting
honestly:

- **Sign agreement is 5 of 7** (binomial p = 0.227).
- **Pearson r(C⁻ − C⁺, ΔCE) = +0.527** across the seven processes (p = 0.224).

Both point the way the hypothesis predicts; neither is significant, and with
n = 7 processes neither could be. A correlation of +0.53 on seven points is what
noise routinely produces. It is nonetheless the most encouraging number in this
repository, and it identifies the cheapest experiment that could move the
question: **more processes.** The per-process precision is already ample —
0.0002 bits — so nothing is gained by more repeats; what limits the correlation
test is that the x-axis has seven values on it. Twenty or thirty (n, m) pairs at
100 repeats each would give that test real power, at a cost of roughly a day.

The honest summary of §10.3 is therefore: **the magnitudes behave exactly as the
residual argument requires, the signs do not track the theory at the resolution
now available, and the across-process trend is positive but statistically
worthless at n = 7.**

### 11.5 The negative results were worth having

Three of the most useful outcomes here are not measurements of the hypothesis at
all, and they are recorded because a reader deciding whether to trust the numbers
needs them:

- **A null control caught a −0.42-bit artefact.** Evaluating chunk-trained models
  at full sequence length biased the two arms asymmetrically, manufacturing a
  large false asymmetry on a process that has none. Without a time-reversible
  control in the suite, that number would have looked like a strong positive
  result.
- **Training diverges on these processes for a structural reason**, not through
  bad luck: cross-entropy on a deterministic transition has no finite minimiser.
  Any study of this kind on processes with deterministic transitions will hit it,
  it scales with the number of gradient steps rather than with the learning rate,
  and it is invisible in the final loss because the model partially recovers.
- **The estimator you use to count causal states determines your answer.** A
  fixed-k estimator confirms the k it is given and overestimates on unbalanced
  occupancies; the same fourteen arms give a 4.2× smaller error when clustering
  the predictive distribution instead (2.7× on the repeat run), which is also the
  object the theory actually defines states by.
- **A standard error over cross-validation folds was hiding the answer, not just
  widening it.** Replacing five shared folds with 100 independent datasets did
  not merely shrink the error bars — it turned two `n.s.` verdicts into
  significant results *pointing the wrong way*, and moved one `match` from
  +0.0026 to +0.0003. A resampling scheme that shares training data does not give
  a conservative answer; it gives a different one.

### 11.6 What would turn this into a finding

In descending order of value, updated now that the repeat harness has run:

1. **The `d_model` sweep.** ΔCE against capacity, at fixed depth and fixed
   process. This is the experiment that makes the result interpretable, and the
   only one that can. §11.3 argues the informative direction is *downward*,
   `d_model` of 2 to 8.
2. **More processes, not more repeats.** §11.4: per-process precision is already
   0.0002 bits, so the across-process correlation test is limited purely by
   having seven points on its x-axis. Twenty to thirty (n, m) pairs at 100
   repeats each would give it real power for roughly a day of compute, and
   `run_statistical_trj.py` needs only a longer `FLOWER_NM` list to do it.
3. **The reversed positional encoding.** `reverse_pos_for_backward` is the last
   architectural asymmetry between the arms, and it is now the leading suspect
   for the two significant wrong-way coins. Running the backward arm both ways is
   what separates causal asymmetry from a positional-encoding artefact.
4. **A second null control** — an i.i.d. process, for which C⁺ = C⁻ = 0 — to
   confirm that the pipeline reports zero for a reason rather than by luck.
5. **Stable long runs**, via `AdamW(weight_decay=0.01)`, if the large
   configuration is ever to contribute. Every number would need re-running under
   the same optimiser.

~~The seed-repeat harness~~ — **done**, and it is what makes items 2 and 3
worth doing next: it established that the measurement is precise enough for the
sign to be a real question rather than noise.

---

## 12. Limitations and open work

**Statistical.** Two regimes, and which applies depends on the figure:

- The **cross-validated** runs (§10.2, §10.4, §10.5) report a standard error over
  five folds of one seed sharing a training set — fold-to-fold variability, not
  sampling variability, and a lower bound on the true uncertainty. §11.5 shows it
  was not merely narrow but misleading.
- The **repeat harness** (§10.3) reports sampling variability over 100
  independent datasets per process, which is the correct quantity. Its residual
  limitations are that the seven processes are a small sample for the
  across-process test (§11.4), and that there is **no correction for testing
  seven processes** — the expected number of spurious verdicts at a two-sided
  2-sem threshold across seven comparisons is about 0.3, so any single isolated
  verdict deserves little weight, and that caution applies to the two `match`
  results as much as to the two `MISMATCH` ones.

**The estimator's free parameter.** k̂ depends on a distance threshold whose
correct value varies by an order of magnitude across these processes. The
stability plateau is always reported for this reason, and a single k̂ at a single
threshold should not be quoted alone.

**Not held out.** Complexity and latent figures are computed over the whole
dataset, ~80 % of which was trained on. `plot_state_clusters.py` quantifies the
cost; the paired ΔCE itself is fully out-of-sample.

**The p-q sweep is exploratory** — in-sample, unchunked, single-fit — as
Section 5.2 sets out.

**The two post-hoc evaluators are unreliable as written.**
`LLM_asymmetry_testing.py` and `Test_data_eval.py` both build their loader with
`Training_model._loader` at **full sequence length**, which is exactly the
extrapolation bias of Section 4.5, and hits the two arms asymmetrically. Their
flower arms are worse still: both draw `dice_probs` from `default_rng(99)` while
the runner trains with `flower_dice_seed = 42`, so they score flower checkpoints
against a **different process**, not merely fresh samples of the same one. The
figures they produce are useful as illustrations; their numbers should not be
quoted. `run_experiments.py`, `sanity_check.py` and `plot_state_clusters.py` all
use the chunk-length analysis loader correctly.

**Reproducibility.** MPS is not bit-reproducible across runs; pairing of the two
arms is unaffected. `accelerator="cpu"` is exact at ~6× the wall clock.

**Scope.** Two process families, one architecture, one depth, one seed, sequences
of a few thousand tokens, vocabularies of 3 to 14 tokens. Nothing here licenses a
claim about language models at scale.

---

## 13. References and related work

### The primary reference

**Thompson, Garner, Mahoney, Vanner, Gu & Modi (2018)** — *Causal asymmetry in a
quantum world.* Physical Review X 8(3), 031013. The theoretical foundation:
establishes that C⁺ ≠ C⁻ for causally asymmetric processes and that quantum
models can exploit the gap. This repository tests the same asymmetry signal in
classical transformer models.

### Closest empirical work

**Papadopoulos, Wenger & Hongler (2024)** — *Arrows of Time for Large Language
Models.* arXiv:2401.17505. Tests whether large models assign higher perplexity to
reversed text, finding a consistent forward-time preference. This repository is a
controlled version of that test, on processes where the ground-truth asymmetry is
analytically known — which is what allows a null to be interpreted rather than
merely reported.

**Ren, Ichter & Majumdar (2024)** — *Thinking Forward and Backward: Effective
Backward Planning with Large Language Models.* arXiv:2411.01790. Shows backward
planning is harder for LLMs on certain tasks, consistent with causal asymmetry
operating at the level of reasoning.

### Computational mechanics

**Shalizi & Crutchfield (2001)** — *Computational mechanics: Pattern and
prediction, structure and simplicity.* Journal of Statistical Physics 104(3),
817–879. The formal foundation for ε-machines, causal states and C⁺; defines the
state representation used throughout, and the connection between minimal
predictive memory and thermodynamic cost.

**Crutchfield & Feldman (1997)** — *Statistical complexity of simple
one-dimensional spin systems.* Physical Review E 55(2), R1239. Early
demonstration on spin chains, and the motivation for mapping C over parameter
space as the p-q sweep does.

**Tan et al. (2014)** — *Towards quantifying complexity with quantum mechanics.*
European Physical Journal Plus 129(9), 191. Extends the framework to quantum
models and quantifies the advantage quantum memory provides.

### Quantum models of complexity

**Gu, Wiesner, Rieper & Vedral (2012)** — *Quantum mechanics can reduce the
complexity of classical models.* Nature Communications 3(1), 762. Quantum models
can simulate a process with strictly less memory than any classical model — which
motivates asking whether a classical network can nonetheless exploit causal
structure through learned representations.

**Suen et al. (2017)** — *The classical-quantum divergence of complexity in
modelling spin chains.* Quantum 1, 25. Analyses the classical-quantum gap across
a family of processes structurally similar to the flower process.

**Elliott et al. (2022)** — *Quantum adaptive agents with efficient long-term
memories.* Physical Review X 12(1), 011007. Context for why causal asymmetry
matters to memory-bounded agents — a category that includes transformers with
finite `d_model`.

### Thermodynamics of computation

**Landauer (1961)** — *Irreversibility and heat generation in the computing
process.* IBM Journal of Research and Development 5(3), 183–191. The foundational
link between logical irreversibility and thermodynamic cost. The causal-asymmetry
hypothesis can be read as Landauer's principle applied to a predictor's memory
operations — and Section 11.3 is the observation that a transformer is not
operating anywhere near that bound.

### Architecture

**Vaswani et al. (2017)** — *Attention Is All You Need.* NeurIPS 30. The
architecture used here; the `tril` and `triu` masks are direct applications of
its masked self-attention.

**Devlin et al. (2019)** — *BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding.* arXiv:1810.04805. Establishes bidirectional
transformers as models of future context. The backward model here is related but
distinct: it is trained autoregressively on reversed causal structure rather than
by masked prediction.

### Area laws and mutual information

**Wolf, Verstraete, Hastings & Cirac (2008)** — *Area laws in quantum systems:
Mutual information and correlations.* Physical Review Letters 100(7), 070502.
Information-theoretic context for why long-range correlations impose fundamental
constraints on the memory a model requires.

**Asplund & Panciu (2024)** — *Predictive complexity of quantum subsystems.*
Entropy 26(12), 1065. Extends predictive complexity to quantum subsystems,
bridging the classical experiments here to quantum extensions.

---

*Every quantitative claim in Sections 10 and 11 is drawn from
`All_Results/results_quick/all_results.pkl` and `sanity_check_flower_process/*/results.pkl`,
and can be recomputed from those files. The divergence tables in Section 4.6 come
from `factorial.log` and `divtest.log`. The reasoning behind each correction
described here is recorded in `implementation_logbook/`, and demonstrated
side-by-side with the code it replaced in `walkthrough.ipynb`.*
