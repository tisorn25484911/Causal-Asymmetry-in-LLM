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
7. [Running the seed-repeat harness](#7-running-the-seed-repeat-harness)
8. [The parameter sweep](#8-the-parameter-sweep)
   - 8A. [The dice-axis experiment — separating C⁻−C⁺ from m−n](#8a-the-dice-axis-experiment--separating-cc-from-mn)
   - 8B. [The capacity axis — the direct test of the residual argument](#8b-the-capacity-axis--the-direct-test-of-the-residual-argument)
   - 8C. [Weight decay across λ](#8c-weight-decay-across-λ)
9. [Generating the causal-state figures](#9-generating-the-causal-state-figures)
10. [The post-hoc evaluators](#10-the-post-hoc-evaluators)
11. [Verification and documentation](#11-verification-and-documentation)
12. [Understanding the outputs](#12-understanding-the-outputs)
13. [Interpreting the results](#13-interpreting-the-results)
14. [Troubleshooting](#14-troubleshooting)
15. [Command reference](#15-command-reference)

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

To reproduce the environment elsewhere, install from `Run_logs/requirements.txt`;
for a byte-exact reproduction, use `Run_logs/requirements-lock.txt`.

### 1.2 Repository layout, and why every command starts from the root

The source is split across two directories by role, with results, notebooks and
logs in three more:

```
Transformer_model/     the primitives: process generators, the model, training,
                       the analysis library, shared utilities
Experimental_setup/    the drivers: configs.py and the six runnable experiments
Jupyter_notebooks/     the four notebooks and the script that generates one
All_Results/           every results tree (results_quick/, results_sweep/, ...)
Run_logs/              run logs, requirements files, the launcher shells
tests/                 the regression suite
implementation_logbook/  the audit trail and the phase/fix plans
```

Two consequences matter in practice.

**Run every command from the repository root, as `python <dir>/<script>.py`.**
The modules are flat — they import each other as `from utils import ...`, not
`from Transformer_model.utils import ...` — so each runnable script puts both
source directories on `sys.path` itself, anchored on its own location. That is
why `python Experimental_setup/run_experiments.py` works and why there is no
`PYTHONPATH` to set. It also means `cd Experimental_setup && python
run_experiments.py` works too, though nothing in this guide is written that way.

**Relative output paths resolve against the repository root, not the working
directory.** `--out-root All_Results/results_sweep` means the same directory
whatever your cwd is, because the runners pass it through `utils.repo_path()`.
Absolute paths are used verbatim, so `--out-root /tmp/scratch` is still
`/tmp/scratch`. The reason for anchoring rather than trusting the cwd is in
§14.6: a stale relative out-root does not raise, it silently starts the run over.

`configs.py` keeps its `out_root` values *relative* (`All_Results/results_quick`)
so the `run_config_*.json` provenance file each run writes records a portable
path rather than one specific to this machine.

### 1.3 Hardware and the accelerator

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

### 1.4 Verifying the installation

Before running any experiment, confirm that the test suite passes:

```bash
pytest tests/ -q
```

The expected result is `66 passed` in approximately twenty seconds. These tests
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

The figures produced by `plot_state_clusters.py` (Section 9) address the second
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
| `SMOKE` | `All_Results/results_smoke/` | 16 | 1e-2 | 3 | ≈ 15 | 2 minutes |
| `QUICK` | `All_Results/results_quick/` | 32 | 1e-2 | 10 | 100 | 9 minutes |
| `QUICK_LARGE_HMM` | `All_Results/results_quick/` | 32 | 1e-2 | 10 | 100 | 3 minutes |
| `LARGE` | `All_Results/results_large/` | 64 | 5e-3 | 80 | 1600 | 3.5 hours |

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
the comparison into a controlled one. It writes into `All_Results/results_quick/` alongside
the QUICK results, and only runs the processes QUICK does not already cover.

**`LARGE`** increases every dimension. Its cross-validation results are **not
usable**; see Section 13.3.

### 4.3 Output directories and tags

Each configuration carries its own `out_root`, so two configurations cannot
overwrite one another's weights.

Experiment tags are **derived from the parameter values** rather than written as
literals: a coin experiment at p = 0.4, q = 0.8 is tagged
`exp1_coin_p040_q080`. A tag therefore cannot disagree with the parameters that
produced it. This was a genuine defect previously: two runners used the same tag
literal at different parameters and overwrote one another's checkpoints.

The same derivation supplies the `traj_` tags of the repeat harness (Section 7),
which writes to `All_Results/results_trajectories/` and so cannot collide with a training
run's output either.

### 4.4 Weight decay, and the optimiser

Training uses **AdamW**, with `weight_decay` taken from the configuration and
defaulting to **0.0**. At 0.0, AdamW is bit-identical to the plain Adam that
produced every result in this repository — verified over 20 seeded steps,
`max|Δparam| = 0` — so the default changes nothing, and **a non-zero λ is a new
experimental condition, not a correction of the earlier ones.**

> **Trap.** PyTorch's `AdamW` defaults to `weight_decay=0.01` where `Adam`
> defaults to `0.0`. It is not a drop-in replacement; the value is always passed
> explicitly, and `tests/test_theory.py` asserts that every configuration ships
> 0.0 and that a non-zero value actually reaches `opt.param_groups[0]`.

λ is threaded the way `n_layers` and `val_every_n_steps` are — config → runner →
pipeline → `train_model` → model — so a runner cannot record one value in its
`run_config_*.json` while training at another. Every `.pt` sidecar now also
records `optimizer` and `weight_decay`, because a checkpoint trained at λ = 0.03
is otherwise indistinguishable from one trained at λ = 0.

Set it per run with `--weight-decay` on `run_sweep_experiment.py`, or in
`configs.py` for the other runners.

**Choosing λ requires knowing the step budget.** The decay's cumulative effect is
`(1 − lr·λ)^steps`, so the same λ bites very differently in a short run than a
long one:

| λ | shrinkage over 130 steps at lr 1e-2 | over 1600 steps at lr 5e-3 |
|---|---|---|
| 0.01 | 1.3 % | 7.7 % |
| 0.1 | 12.2 % | 55.1 % |
| 1.0 | 72.9 % | 100 % |

**Measured λ response** on the flower process at the repeat harness's 130 steps
(2 repeats per point, `δ(λ) = CE − H∞`):

| λ | shrinkage | δ(λ) | verdict |
|---|---|---|---|
| 0 – 1.0 | 0 – 73 % | 0.011 – 0.016 | **does not bind** — the model still reaches H∞ |
| 3.0 | 98 % | 0.090 – 0.178 | binds hard, and **unequally**: ΔCE reaches −0.27 and −0.41 bits, both negative despite opposite-sign C⁻ − C⁺ |
| ≥ 10 | 100 % | 0.61 – 1.11 | both arms destroyed; ΔCE collapses to noise |

Two operational conclusions. **λ ≤ 1 is safe but does not bound capacity here** —
the task needs so little of the model that a 73 % norm reduction leaves it
converging to within 0.016 bits of H∞. And **λ ≥ 3 must not be used for a ΔCE
measurement**: the artefact is a hundred times the effect, its sign does not
track C⁻ − C⁺, and δ(λ) exceeds the 0.10-bit convergence tolerance so every
repeat is flagged non-converged. That last point matters differently per runner:
`run_sweep_experiment.py` reports all-repeat statistics and is unaffected, whereas
`run_experiments.py` and `sanity_check.py` read the *converged-only* headline and
would print a confident verdict from a single surviving fold.

Weight decay is also the principled suppressant for the divergence of Section
13.3 — cross-entropy on a deterministic transition has no finite minimiser, and a
norm penalty restores one (measured 4/6 → 1/6 on the flower null control at 600
steps). But at the 130 steps these harnesses run, divergence is already rare (3 in
3,750 repeats), so that is not a reason to enable it.

---

### 4.5 When two flower outcomes count as one backward state

`flower_complexity` decides that two outcomes share a backward causal state when
they induce the same posterior over dice. The rule is a parameter:

| `merge_tol` | rule | when to use it |
|---|---|---|
| `None` (default) | round the posterior to 9 decimals, require exact equality | **always, unless you have a reason** — every C⁻ this repo has reported uses it |
| a float | group posteriors within that distance in max-norm | an explicitly operational reading, reported alongside the default |

The two are not equivalent, and the difference is not academic. At Dirichlet
α = 0.2, (n,m) = (2,8), seed 74, two outcomes have posteriors differing by
**6.4 × 10⁻¹⁰**. The rounding rule calls them two backward states (C⁻ = 2.1041); a
1e-9 tolerance calls them one (C⁻ = 2.0748) — a 0.029-bit difference in the x-axis
of the whole experiment.

Both readings are defensible. Exactly, an ε-machine is defined by exact conditional
distributions, so distinct posteriors are distinct states. Operationally,
distinguishing those two from data needs of order 10¹⁸ samples, so a finite-sample
predictor cannot represent them separately — and the hypothesis is about what a
*memory-bounded* model does.

It matters only for spiky dice. Over 3600 Dirichlet(1.0) draws the two rules agree
exactly, so nothing in the 81-cell grid or the seven-process harness is affected.
Any experiment that deliberately uses small α should state which rule it used —
`run_dice_experiment.py` writes `merge_tol` into its run config, and its selected
dice were checked to give identical answers under both.

---

## 5. Running the training suite

### 5.1 The command

`run_experiments.py` is the only file that trains the main experiments. It is
configured entirely by command-line flags; no editing is required.

```bash
python Experimental_setup/run_experiments.py --config QUICK
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
python Experimental_setup/run_experiments.py --config LARGE --only exp1
python Experimental_setup/run_experiments.py --config LARGE --only exp2
```

### 5.4 Worked examples

Run a second seed into a separate directory, so that the two are not confused:

```bash
python Experimental_setup/run_experiments.py --config QUICK --seed 1 --out-root All_Results/results_quick_seed1
```

Run only the flower experiments:

```bash
python Experimental_setup/run_experiments.py --config QUICK --only exp2
```

Add LARGE's processes to the QUICK results directory:

```bash
python Experimental_setup/run_experiments.py --config QUICK_LARGE_HMM --only exp1
python Experimental_setup/run_experiments.py --config QUICK_LARGE_HMM --only exp2
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
  then blew up (Section 13.3);
- the recovered number of causal states k̂ with its stability profile;
- the paired ΔCE, reported both over all folds and over converged folds only.

At the end, a summary table lists every experiment in the output directory with
its C⁺, C⁻, ΔCE, standard error, and a verdict of `match`, `MISMATCH` or `n.s.`.
The verdict `n.s.` denotes |ΔCE| < 2 × standard error, that is, not
distinguishable from zero.

---

## 6. Running the controls

```bash
python Experimental_setup/sanity_check.py
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

## 7. Running the seed-repeat harness

```bash
python Experimental_setup/run_statistical_trj.py
```

Seven processes, each trained one hundred times in both causal directions.
Runtime is approximately two and a quarter hours; output is written to
`All_Results/results_trajectories/`.

This is the answer to the second caveat in Section 13.2. Every standard error
reported by `run_experiments.py` is computed over the five cross-validation folds
of a single seed, and those folds share a training set: the quantity measured is
fold-to-fold variability, not sampling variability over datasets. Each repeat
here regenerates its process from a fresh seed, so the spread across repeats
**is** sampling variability, and the paired statistic can therefore support a
claim about the sign of ΔCE that the per-fold statistic cannot.

### 7.1 Options

```
--config CFG          Configuration supplying the model, data and optimiser
                      settings.  Default: QUICK.

--repeats N           Independent repeats per process.  Default: 100.

--seed N              Base seed.  Repeat i uses base + i for BOTH arms.
                      Default: the configuration's seed.

--out-root DIR        Where figures and pickles are written.
                      Default: All_Results/results_trajectories.

--only TAG [TAG ...]  Run a subset.  Matches a full tag or any substring of
                      one, so --only coin_p030 flower_n2 is accepted.

--khat                Additionally recover k̂ and S_hat from each model's
                      predictive distribution.  Slower; see 7.6.

--plots-only          Redraw every figure from the saved pickle without
                      training anything.

--verbose             Do not suppress the per-run training output.
```

### 7.2 The seven processes

All are run at one common set of settings — the configuration's `coin_*` values
for the three coin processes and its `flower_*` values for the four flower ones —
so that the processes are comparable with one another. In particular all three
coin processes use `coin_num_samples`, `coin_seq_len` and `coin_max_epochs`; the
shorter `coin_seq_len_12` of experiment 1.2 is not used here.

| Tag | Process | C⁺ | C⁻ | Prediction |
|---|---|---|---|---|
| `traj_coin_p010_q090` | coin, p = 0.1, q = 0.9 | 0.4690 | 0.8911 | ΔCE > 0 |
| `traj_coin_p030_q040` | coin, p = 0.3, q = 0.4 | 0.9852 | 1.4888 | ΔCE > 0 |
| `traj_coin_p040_q080` | coin, p = 0.4, q = 0.8 | 0.9183 | 1.5656 | ΔCE > 0 |
| `traj_flower_n2_m6` | flower, n = 2, m = 6 | 1.5000 | 2.1137 | ΔCE > 0 |
| `traj_flower_n2_m8` | flower, n = 2, m = 8 | 1.5000 | 2.4765 | ΔCE > 0 |
| `traj_flower_n4_m2` | flower, n = 4, m = 2 | 2.0000 | 1.4952 | ΔCE < 0 |
| `traj_flower_n6_m4` | flower, n = 6, m = 4 | 2.2925 | 1.9899 | ΔCE < 0 |

The last two are the reversed cases, and they are the reason for running seven
processes rather than one: a result in which the *sign* of ΔCE follows the sign
of C⁻ − C⁺ across processes is far stronger evidence than any single positive
case.

Approximate cost per repeat, measured: eight seconds for a coin process, fourteen
for a flower one, sixteen with `--khat`. One hundred repeats of all seven is
therefore about two and a quarter hours, or three with `--khat`.

### 7.3 How a repeat differs from a cross-validated run

| | `run_experiments.py` | `run_statistical_trj.py` |
|---|---|---|
| Unit of replication | Cross-validation fold | Independent repeat |
| Data | One dataset, shared by all folds | Regenerated per repeat |
| Trainings per unit | Five folds, best selected | One |
| Reported loss | Held-out CE of the best fold | Held-out CE of the single model |
| What the standard error measures | Fold-to-fold variability | **Sampling variability** |

Two consequences of the middle row deserve stating. A cross-validated run
reports the fold with the lowest validation loss, and a minimum over five draws
is not a sample from the same distribution as a single draw; using it as the
per-repeat value would bias the statistic. Running a full cross-validation
within each repeat would also multiply the cost fivefold, to seven thousand
trainings.

Within a repeat the two arms remain **paired** exactly as elsewhere in the
repository: the same seed yields identical sequences, identical chunk windows, an
identical train/held-out split, identical batch order and identical weight
initialisation, so that only the attention mask and the batch convention differ.

The flower dice are drawn from `flower_dice_seed`, **not** from the repeat seed.
The dice define the process — they determine C⁺ and C⁻ — so resampling them per
repeat would average over different processes rather than over realisations of
one. They are the same dice `run_experiments.py` trains on, which keeps the two
sets of numbers comparable.

### 7.4 What is produced

```
All_Results/results_trajectories/
├── run_config_QUICK.json                   parameters, repeat count, timestamp
├── all_trajectories.pkl                    every process, merged by tag
├── summary_delta_ce.png                    cross-process summary, two panels
└── traj_coin_p040_q080/
    ├── results.pkl                         every run of this process
    ├── ..._complexity.png                  S_emp against C⁺ / C⁻, with error bars
    ├── ..._final_loss.png                  three bars: FW, BW, BW − FW
    └── ..._trajectories.png                all runs, both means, shaded interval
```

Three figures per process, as follows.

**`_complexity.png`** places the mean empirical complexity of each arm beside its
closed form. The caption beneath restates that `S_emp` is k-means at the
pre-specified k printed on the axis, so that the figure cannot be read as
independent confirmation of that k.

**`_final_loss.png`** carries the three bars: forward, backward, and their
difference. The difference is drawn against **its own axis on the right**,
because ΔCE is two to three orders of magnitude smaller than the cross-entropies
beside it and would otherwise be a flat line at zero. Its error is the *paired*
standard deviation — the spread of the per-repeat differences — which is the
purpose of pairing the arms and is substantially smaller than the propagation of
two independent errors would suggest. Where any run diverged, a second hatched
group of bars gives the same three statistics over converged runs only.

**`_trajectories.png`** has two panels. The left shows every run at alpha 0.3
with the two mean curves in bold and a shaded ±1 standard deviation band; its
y-axis is clipped, and the number of points excluded is stated in the title,
because a diverged run reaches fifteen to sixty bits and would otherwise
compress everything of interest. The right panel is the mean of the per-step
paired difference, scaled to the second half of training for the same reason.

In every figure a **thick** error bar is ±1 standard error and a **thin** whisker
is ±1 standard deviation, stated in the caption, so that no error bar is
ambiguous as to which it is.

**`summary_delta_ce.png`** shows ΔCE with its standard error for each process,
and ΔCE against C⁻ − C⁺. A process whose ΔCE is not distinguishable from zero is
drawn in grey rather than assigned a colour, on the same |ΔCE| < 2 × standard
error rule the console table uses.

### 7.5 Interruption, subsetting and redrawing

State is written after **every** repeat, to both the per-process `results.pkl` and
the combined `all_trajectories.pkl`, so an interruption costs one repeat rather
than the run. The combined pickle is merged by tag, exactly as in Section 5.5,
so a subsequent partial run does not discard earlier processes.

```bash
# one process only, into the same directory
python Experimental_setup/run_statistical_trj.py --only flower_n6_m4

# a short shakedown before committing to the full run
python Experimental_setup/run_statistical_trj.py --repeats 3 --out-root /tmp/traj_check

# regenerate all figures after editing a plotting function
python Experimental_setup/run_statistical_trj.py --plots-only
```

### 7.6 The recovered state count

`--khat` additionally runs `recover_causal_states` on each arm of each repeat,
recording k̂, S_hat and the stability plateau. This is the better estimator —
Section 13.2, caveat 3 — but it costs an agglomerative clustering pass per arm
per repeat, and it is off by default so that the headline run is not delayed by
it. When it has been used, the complexity figure carries the median k̂ and the
mean S_hat beneath the axis for comparison with `S_emp`.

### 7.7 The schema of `results.pkl`

```python
import pickle
r = pickle.load(open("All_Results/results_trajectories/traj_coin_p040_q080/results.pkl", "rb"))
r["runs"][0]["fw"]["final_ce"]        # held-out CE of repeat 0, forward arm
```

| Key | Contents |
|---|---|
| `spec` | The process: tag, kind, parameters, dice, C⁺, C⁻, H∞, and the k assumed for each arm. |
| `config`, `base_seed`, `repeats`, `khat` | How the run was invoked. |
| `runs` | One entry per repeat; see below. |
| `paired` | The paired ΔCE statistics, in the same schema as Section 11.2, with the repeat as the unit in place of the fold. |

Each entry of `runs` holds `seed`, `delta_ce`, and an `fw` and a `bw` dictionary:

| Key | Contents |
|---|---|
| `final_ce`, `final_ppl` | Held-out cross-entropy and perplexity. |
| `traj`, `traj_at` | Training loss per gradient step, and the step indices. |
| `val_traj`, `val_at` | Held-out loss on its own sparser cadence. |
| `S_emp` | Complexity from k-means at the assumed k. |
| `k_hat`, `S_hat`, `k_plateau` | Present only when `--khat` was passed. |
| `diverged`, `divergence` | The divergence verdict and its diagnosis. |

Because every repeat is retained in full, an alternative convergence filter or a
different summary statistic may be applied without re-running anything.

### 7.8 Status

The harness has been verified end to end on short runs — one and two repeats
across a coin and two flower processes, with and without `--khat`, and through
`--plots-only`. **The full hundred-repeat run has not yet been performed**, so no
results from it are quoted in Section 12.

---

## 8. The parameter sweep

```bash
python Experimental_setup/run_sweep_experiment.py --dry-run     # spec table and coverage, no training
python Experimental_setup/run_sweep_experiment.py --repeats 30  # 125 processes, ~7.6 h
```

Where Section 7 measures seven processes precisely, this measures 125 coarsely,
to answer the stronger question: **across many processes spanning a wide range of
C⁻ − C⁺, does the measured asymmetry track the theoretical one?** A single
process cannot show a trend.

Everything that trains or measures one process is imported from
`run_statistical_trj.py`; this file only chooses the processes and aggregates
across them. Output goes to `All_Results/results_sweep/`.

### 8.1 The two grids, and why they are not equivalent

| | processes | C⁻ − C⁺ | corr with H∞ | tests |
|---|---|---|---|---|
| coin, p,q ∈ {0.05 … 0.95} | 100 | +0.014 to +0.683, **all positive** | **+0.590** | magnitude only |
| flower, n,m ∈ {2,4,6,8,10} | 25 | −1.164 to +1.069 | −0.011 | **the sign** |

Two consequences govern how the output must be read.

**Every negative-x point is a flower process**, so `corr(family, C⁻−C⁺) = −0.46`
and a **pooled** correlation over all 125 points substantially tests "do flower
processes differ from coin processes" — vocabulary 3 versus 4–20, H∞ 0.29–0.99
versus 0.87–3.07. The within-family rows are the real statistics, and the pooled
row is labelled as not being the test.

**On the coin grid C⁻ − C⁺ correlates with H∞ at +0.59**, so a partial
correlation controlling for H∞ is reported for that family. This is not
decoration: the raw coin trend for ΔCE is ρ = −0.415 (p < 1e-5), significant and
in the *wrong* direction, and it vanishes entirely once H∞ is controlled
(ρ = +0.068, p = 0.50). Within the flower grid C⁻ − C⁺ is uncorrelated with both
H∞ and vocabulary, which is what makes it the better-designed axis.

### 8.2 Options

```
--repeats N            Independent repeats per process.  Default 30.
--sweep-coin [P ...]   Coin grid; bare flag uses the default 10-point grid.
--sweep-flower [N ...] Flower grid; bare flag uses {2,4,6,8,10}.
--weight-decay LAMBDA  AdamW decay; see Section 4.4.  Omit for plain Adam.
--only TAG [TAG ...]   Restrict to tags containing these substrings.
--area-burn-coin N     Integrate the coin area from step N (see 8.4).
--area-burn-flower N   The same, for flower.
--dry-run              Print the spec table and coverage, then stop.
--plots-only           Redraw the figures from the saved pickle.
--redo                 Retrain processes that are already complete.
--khat                 Also recover k̂ per arm per repeat (adds ~1 h per 2500).
```

Runs are **resumable**: any process already holding `--repeats` repeats is
skipped, so re-issuing the same command after an interruption continues where it
stopped. State is written at least once a minute.

### 8.3 Three response variables, only one of which the theory predicts

1. **Final ΔCE — PRIMARY.** The held-out endpoint difference. The only quantity
   the hypothesis makes a statement about.
2. **Trajectory area — exploratory.** The mean of `CE_BW(s) − CE_FW(s)` over the
   settled portion of training. The intuition is that the harder direction should
   sit above the other throughout optimisation; but the integral of a loss
   difference along an Adam trajectory depends on the learning rate, the
   initialisation and the landscape's curvature, none of which appear in
   computational mechanics. A positive result here generates a hypothesis; it
   does not confirm one.
3. **Convergence-step difference — exploratory.** Steps for the backward arm to
   settle minus the forward arm's. Immune to the magnitude of the transient and
   to a diverged run's spike, both of which corrupt an integral.

### 8.4 The integration window, and why it is not a free choice

The area is computed at **analysis** time from the stored trajectories, so any
window can be recomputed in about 0.2 s for all 125 processes — no retraining.
Four are stored per process: the shared settled window (plotted), each process's
own settling step, a fixed fraction, and the whole trajectory.

The window matters because convergence time varies systematically across the
sweep: the coins settle at step 32–104 and the flowers at 58–125. Integrating
from a **fixed** step therefore folds a different amount of transient into each
family — and since every negative-x point is a flower, that contamination lands
on one side of the x-axis, which manufactures a slope. So the default window
starts at the **slowest** process's settling step, capped at 60 % of the run so
one pathological cell cannot shrink the window to nothing. When the cap binds,
the run prints `! CAPPED`.

The sensitivity is worth knowing before quoting anything from this panel:

| window | flower ρ | coin ρ given H∞ |
|---|---|---|
| whole trajectory | +0.872 | −0.763 |
| from step 20 | +0.835 | −0.871 |
| from step 30 | +0.764 | −0.851 |
| shared settled (default) | +0.805 | −0.301 |
| from step 110 | — | −0.027 (p = 0.79) |

The flower trend is insensitive — +0.69 to +0.87 across every window. The coin
column is monotone in the window: strong when the window is all transient,
vanishing once it is all settled, and +0.07 at the endpoint. Whatever the coin
trend is, it is a property of the descent, not of the converged residuals.

### 8.5 What is produced

```
All_Results/results_sweep/
├── all_sweep.pkl          every repeat's trajectory, resumable state
├── sweep_rows.pkl          the per-process scalars the figures use
├── run_config_QUICK.json   parameters, grids, λ, timestamp
├── sweep_scatter.png       4 panels against C⁻ − C⁺
└── sweep_trajectories.png  one mean D(s) curve per process, coloured by C⁻−C⁺
```

The run also prints a per-process table, the within-family correlations, and a
**baseline cross-check**: the default flower grid contains (2,6), (2,8), (4,2)
and (6,4) — every flower process the Section 7 harness ran, at the same dice
seed — so those cells must reproduce its ΔCE within combined error. They do,
with |z| ≤ 0.79 and one cell agreeing to z = +0.01.

### 8.6 The results, and the caveats that travel with them

From the completed 125-process run at 30 repeats (7.6 h, 7,500 trainings, 3
divergences):

| | ρ against C⁻ − C⁺ | p |
|---|---|---|
| **flower, final ΔCE** | **+0.826** | < 1e-5 |
| coin, final ΔCE, raw | −0.415 | < 1e-5 |
| coin, final ΔCE, controlling H∞ | +0.068 | 0.50 |
| pooled | −0.046 | 0.61 |

The flower trend is robust to every exclusion tried: dropping the two
marginal-convergence cells gives +0.878, dropping cells containing a divergence
+0.809, dropping the n = m diagonal +0.850. The five near-zero-gap diagonal cells
read ΔCE ≈ 0, which is the built-in null control behaving.

Three caveats belong with that number. Magnitudes remain 0.0003–0.0044 bits,
about 0.2 % of the entropy rate, consistent throughout with ΔCE being a
difference of residuals. `corr(C⁻−C⁺, m−n) = +0.973` on the flower family, so
this design **cannot** separate "tracks C⁻ − C⁺" from "tracks m − n" — arguably
the same mechanism, since the forward arm must represent one conditional per die
and the backward arm one per distinguishable outcome, but that is a structural
argument rather than an empirical one. And the nominal p-values are
anti-conservative: C⁻ − C⁺ is a smooth function of (p,q) and (n,m), so
neighbouring cells are near-duplicate processes and the effective degrees of
freedom are well below 125.

### 8.7 Which slope to quote

The sweep prints two fits, and they disagree about whether there is a result at
all:

| estimator | n | slope b | t |
|---|---|---|---|
| unweighted OLS | 81 | +0.00344 ± 0.00248 | **+1.38** — reads null |
| inverse-variance WLS | 81 | +0.00294 ± 0.00016 | **+18.9** |

**Quote the WLS slope.** The per-cell sems span 249× on this grid (0.00038 to
0.09442) because a few cells contain a diverged repeat. OLS assumes equal errors,
so those cells dominate the residual variance and flatten the t-statistic. WLS
gives the same estimate with or without `unstable_mask`, which is what you want
from an estimator: the answer should not depend on a hand-set exclusion
threshold.

`chi²/dof` is printed beside it. At 2.17 (flower) and 4.35 (coin) the scatter
exceeds the quoted sems, i.e. there is real process-to-process variation the per-
cell error bars do not capture — so treat the WLS error as a lower bound.

For ρ, quote the **block-bootstrap interval**, not the nominal p-value: ρ = +0.827
with a 95% CI of [+0.638, +0.878] over 9 resampled grid rows, sign stable in 100%
of resamples. Resampling rows rather than cells respects the near-duplicate
structure; the nominal p = 1.9 × 10⁻²¹ does not.

### 8.8 The coin grid is a control, not evidence

The summary now says this outright, because the raw coin correlation has the
**wrong sign** for the hypothesis and vanishes when controlled:

```
corr(C- - C+, H_inf) = +0.590
raw      rho(gap, dCE) = -0.415   p = 1.8e-05
partial  rho | H_inf   = +0.068   p = 0.50
         rho(H_inf,dCE)= -0.793   p = 8.3e-23
```

At a fixed step budget, higher-entropy processes sit further from convergence and
their two arms' residuals differ differently — so the coin grid is measuring how
hard the process is, not how asymmetric. Read positively, that makes it a
**confound control**: it demonstrates the measurement does respond to entropy
rate, which is exactly why the flower grid, where corr(gap, H∞) = −0.010, is the
axis that carries the claim. Never quote a coin ρ without its partial.

---

## 8A. The dice-axis experiment — separating C⁻−C⁺ from m−n

```bash
python Experimental_setup/run_dice_experiment.py --dry-run     # the design, no training
python Experimental_setup/run_dice_experiment.py --repeats 30  # ~2.6 h, 20 processes
python Experimental_setup/run_dice_experiment.py --plots-only
```

### 8A.1 Why it exists

The flower sweep's ρ = +0.827 cannot distinguish two explanations, because on that
grid

```
corr(C- - C+,  m - n) = +0.977      (spearman +0.986)
```

Anything that scales with `m − n` — token-frequency imbalance, how often each die
is revisited, the ratio of selection to outcome tokens — reproduces the whole
result. This is a property of the **design**, so no number of extra repeats fixes
it.

### 8A.2 How it escapes the confound

C⁻ for the flower has a closed form,

```
C- = 1 + H(pi_outcome)/2       pi_outcome_j = (1/n) * sum_i dice_probs[i,j]
so   C- - C+ = [ H(pi_outcome) - log2(n) ] / 2
```

which depends on how much the *n* dice overlap — not only on (n, m). Fix (n, m),
vary only the dice, and `m − n`, vocabulary, C⁺, the forward state count, capacity
and the step budget are all identical by construction while the gap still moves.

Selecting the dice inside a narrow H∞ band removes the entropy-rate confound too.
Measured over 3000 draws per cell:

| cell | α | gap range in band | negatives | corr(gap, H∞) in band | unrestricted |
|---|---|---|---|---|---|
| (3,5) | 0.2 | −0.206 → +0.263 | 3 | **+0.033** | +0.526 |
| (4,6) | 0.2 | −0.275 → +0.199 | 3 | **+0.055** | +0.484 |

Both cells contain **both signs**, so this design can run a controlled sign test —
the only thing the theory actually predicts — which the lattice grid cannot do
anywhere.

### 8A.3 Reading the result

The reported statistic is the **within-cell** slope, plus a pooled fit that gives
each cell its own intercept, so between-cell variation (where `m − n` lives) cannot
contribute:

- within-cell slope ≈ +0.0029 (the between-cell value) → the confounds are
  exhausted; the trend is the theoretical gap.
- within-cell slope ≈ 0 while the 81-cell grid still shows ρ = +0.83 → the flower
  trend was `m − n` all along.
- sign of ΔCE following sign of the gap inside one cell → the strongest available
  outcome.

### 8A.4 Two traps this file closes

**Tags must carry the dice seed.** `sweep_specs` re-tags flower cells as
`flower_tag("sweep", n, m)` — a function of (n, m) alone. Twenty realisations at
one cell would collide on a single key, and since the result store is keyed by tag,
nineteen would be silently overwritten. The dice runner tags as
`dice_flower_n<N>_m<M>_s<SEED>`.

**α = 0.2 is the merge-rule regime.** Spiky dice produce near-zero outcome masses,
where C⁻ can depend on whether two vanishingly-rare outcomes count as one backward
state — 0.029 bits at (2,8) seed 74. Checked for these exact dice: the two rules
agree to 0.0e+00, and `merge_tol` is recorded in the run config regardless. See
§4.5.

---

## 8B. The capacity axis — the direct test of the residual argument

```bash
for D in 8 16 32 64; do
  python Experimental_setup/run_dice_experiment.py --repeats 30 --d-model $D \
         --out-root All_Results/results_dice_cap/d$(printf %03d $D)
done
python Experimental_setup/analyse_capacity.py --root All_Results/results_dice_cap
```

§2.3 derives that ΔCE is a difference of **residuals**: H∞ is time-reversal
invariant, so an unbounded predictor trained to convergence gives ΔCE = 0
*regardless* of C⁻ − C⁺. A non-zero ΔCE is therefore a symptom of bounded memory,
and that makes a sharp prediction nobody has tested — **|ΔCE| must shrink as
`d_model` grows.**

Until it is measured, every near-null ΔCE in this repo is ambiguous between

- "this process pair is symmetric" (a result), and
- "d_model = 32 was already enough for both arms" (no result),

and nothing else here can break the tie. `sanity_check.py`'s own docstring flags
the same gap for the positive control.

**A flat profile would be the more interesting outcome**: it would falsify the
residual argument as stated and mean ΔCE is measuring something that does not
vanish with width — an optimisation asymmetry, say. So `analyse_capacity.py`
reports the shape of the profile rather than testing only for the expected sign.

Two operational points. `lr` is held **fixed** across capacities, which is what
makes this a clean capacity manipulation and also means the extremes may be
trained sub-optimally — so the divergence rate and mean settling step are printed
per capacity, and a drift there is a caveat to state rather than a confound to
hide. And each capacity **must** have its own `--out-root`: tags do not encode
`d_model`, so the runners refuse unless the out-root names it (`d064` or `d64`).

---

## 8C. Weight decay across λ

```bash
python Experimental_setup/analyse_lambda_sweep.py --like-for-like
```

Fits ΔCE = a(λ) + b(λ)·(C⁻−C⁺) per λ and per family — the decomposition
`WEIGHT_DECAY_PLAN.md` §6.2 asked for. On the shared 25 flower cells:

| λ | 0 | 0.03 | 0.10 | 0.30 | 1.00 |
|---|---|---|---|---|---|
| flower b | +.00255 | +.00265 | +.00206 | +.00302 | **+.00503** |
| coin b | −.00169 | −.00153 | −.00137 | −.00172 | **−.00404** |

**b is flat for λ ≤ 0.3** — so mild regularisation is ruled out as an explanation
for the near-null ΔCE; the signal was not being hidden by an unregularised fit.
**At λ = 1 it roughly doubles in both families** (z = +5.91). That is *not* weight
decay revealing causal structure: it amplifies the coin slope just as much, and
the coin trend is a known H∞ artefact. It is a scale effect from degraded
training, corroborated by the unstable-cell count going 2 → 7 and the coin
intercept becoming a real offset (+0.00088 ± 0.00009) only at λ = 1.

Use `--like-for-like` when comparing slopes: λ = 0 ran a 9×9 flower grid and the
others 5×5, so an 81-point slope against a 25-point one confounds λ with grid size.

---

## 9. Generating the causal-state figures

```bash
python Transformer_model/plot_state_clusters.py
```

This script must be run **after** a training run. It loads saved weights and
does not retrain, so it is inexpensive.

### 9.1 Options

```
--config CFG          Which configuration's experiments to plot.  Default: QUICK.
--also-config CFG     A second configuration sharing the same out_root.
                      Default: QUICK_LARGE_HMM.  Pass '' to skip.
--out-root DIR        Where to read weights from and write figures to.
--metrics ...         Any of: euclidean, tv, js.  Default: all three.
```

Example:

```bash
python Transformer_model/plot_state_clusters.py --out-root All_Results/results_large --metrics js
```

### 9.2 What is produced

For every experiment and every metric, one 2 × 2 figure is written to
`All_Results/results_quick/<tag>/<tag>_states_<metric>.png`:

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

### 9.3 Why the train/held-out split appears here

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

### 9.4 The three distance metrics

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

## 10. The post-hoc evaluators

```bash
python Transformer_model/Test_data_eval.py
python Experimental_setup/LLM_asymmetry_testing.py
```

These load saved checkpoints and score them on freshly generated data.

### 10.1 Configuration

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

### 10.2 A note on metrics

`Test_data_eval.py` reports two perplexities and they must not be conflated.
The **teacher-forced** perplexity scores both models on the same ground-truth
sequences and is the comparison metric. The **self-generated** perplexity scores
each model on its own sampled sequence; it estimates that model's own entropy
rate, is minimised by a degenerate over-confident model, and is labelled
informational only.

---

## 11. Verification and documentation

### 11.1 The test suite

```bash
pytest tests/ -q
```

Sixty-six tests, approximately twenty seconds. They should be run after any
modification to `Model_analysis.py`, `Training_model.py` or `OneHot_model.py`.

The suite pins the properties that fail *silently* when broken: that the loss
function remains finite at extreme logit magnitudes; that perplexity is
token-weighted; that the two arms receive identical windows; that the flower
closed forms reproduce their reference values; that vectorised code produces the
same numbers as the loops it replaced; and that the causal-state estimator
merges tokens with equal futures while separating those with different ones.

### 11.2 The walkthrough notebook

```bash
python Jupyter_notebooks/build_walkthrough.py
```

This regenerates `walkthrough.ipynb`, a thirty-seven-cell notebook that
demonstrates every change made across Phases 0 to 3. It is generated from a
script rather than edited by hand so that it can be regenerated after a re-run.
Each defect is recomputed side by side with its correction in the same cell, so
that the claims may be checked rather than taken on trust. It closes with a
consolidated list of caveats.

---

## 12. Understanding the outputs

### 12.1 Directory layout

```
All_Results/results_quick/
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

### 12.2 The schema of `results.pkl`

```python
import pickle
r = pickle.load(open("All_Results/results_quick/exp1_coin_p040_q080/results.pkl", "rb"))
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

## 13. Interpreting the results

### 13.1 The current findings

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
effect of this size requires the seed-repeat harness of Section 7, which now
exists but has not yet been run in full.

The overall picture is a **null result**, which is precisely what Section 2.3
predicts for a converged predictor with capacity to spare. It should not be
reported as "no causal asymmetry", because it remains ambiguous until the
`d_model` sweep distinguishes that conclusion from "capacity absorbed the
asymmetry".

### 13.2 Caveats attaching to every reported number

1. ΔCE is a difference of residuals; a converged, sufficient-capacity model
   yields approximately zero regardless of C⁻ − C⁺.
2. The standard error is computed over five folds **that share a training
   set**. It measures fold-to-fold variability, not sampling variability over
   datasets, and is therefore a lower bound on the true uncertainty. It must not
   be presented as a confidence interval. Section 7 is the harness that replaces
   it with a standard error over independent repeats.
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

### 13.3 The LARGE configuration

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

### 13.4 Which run to report

**QUICK.** All five folds converge, in all seven experiments.

---

## 14. Troubleshooting

**A test fails.** Do not proceed to experiments. The tests cover failures that
are silent in ordinary use.

**`load_state_dict` raises a shape error.** The checkpoint was produced by a
different configuration. Check the JSON sidecar beside the `.pt` file and set
`RUN["config"]` accordingly.

**A run reports `k̂ != theory`.** This is frequently correct behaviour: the
estimator measures what the *model* represents, and a model that has not
resolved all the states of its process will legitimately return a smaller k̂.
Consult the stability profile before concluding that the estimator is at fault.

**A run reports many diverged folds.** See Section 13.3. This is expected for
long runs and is reported rather than concealed.

**The paired standard error is `nan`.** Fewer than two folds survived the
convergence filter. The paired statistic is not meaningful for that arm.

**Results differ slightly between identical runs.** Expected on MPS; see
Section 1.3.

**`ModuleNotFoundError: No module named 'Data_generation'` (or `Model_analysis`,
`utils`, `configs`).** The script was invoked in a way that bypassed its path
bootstrap — most likely it was copied elsewhere, or a new script was added
without the bootstrap block. Every runnable file carries a short block that puts
`Transformer_model/` and `Experimental_setup/` on `sys.path`, anchored on
`__file__`; see §1.1.1 and copy the block from any existing runner. For a new
*test* file nothing is needed: `conftest.py` at the repository root does it for
the whole pytest session.

### 14.6 A resumable run appears to start over

This one is worth knowing about, because it is silent and it is expensive.

`--out-root` is created if it does not exist (`os.makedirs(exist_ok=True)`), and
a missing `all_sweep.pkl` / `all_trajectories.pkl` is treated as "no repeats
completed yet" — which is exactly what has to happen on a genuine first run. The
consequence is that a *wrong* out-root does not raise. It silently produces an
empty resume state and retrains everything. A 23-hour weight-decay sweep was lost
this way once, when the results tree was moved while the run was in progress.

Two things now guard against it:

1. Relative out-roots resolve against the repository root rather than the
   working directory (§1.1.1), so the same command means the same directory
   wherever it is typed.
2. `run_sweep_experiment.py` prints a **resume plan** before training, and
   `--dry-run` prints it too:

   ```
   resume plan : 97 complete and skipped, 3 to train
   partial     : sweep_coin_p095_q075 (18/30)
   ```

   If the out-root holds nothing, the run says so in three loud lines rather than
   quietly beginning again.

**Always `--dry-run` first when resuming.** It costs a second and it answers the
only question that matters before committing hours of compute: am I continuing,
or starting over?

Note the two harnesses resume at **different granularities**, which is easy to
get wrong:

| | `run_sweep_experiment.py` | `run_statistical_trj.py` |
|---|---|---|
| Skips processes already at `--repeats` | yes | **no** |
| Preserves *other* processes in the pickle | yes | yes |
| Effect of re-running one process | continues from the repeats it has | **discards them and retrains all `--repeats`** |

So `--only flower_n6_m4` on the trajectory harness does not continue a
half-finished process — it replaces that process's record with a fresh 100-repeat
run (~20 min) while leaving the other six untouched. Only the sweep counts
completed repeats and skips them.

```bash
python Experimental_setup/run_sweep_experiment.py --sweep-coin --repeats 30 \
       --weight-decay 1.0 --out-root All_Results/results_sweep_wd/wd1.000 --dry-run
```

Relatedly, if the baseline cross-check (§8) is missing from a sweep's output, the
run will now say `! BASELINE CROSS-CHECK SKIPPED` and name the path it tried. It
used to skip in silence, which meant a stale `--baseline` removed the sweep's only
end-to-end regression check without leaving a trace.

---

## 14A. The two model architectures

Every runner takes `--model {onehot,discrete}`. The default comes from the
config, so nothing changes unless you ask for it.

### 14A.1 What differs

| | `onehot` | `discrete` |
|---|---|---|
| class | `OneHotDecoder` | `DiscreteCausalDecoder` |
| head | `D → V` logits | `D → K` state code, hard one-hot, `K → V` emission |
| causal states | clustered out of the latents (`k̂`, silhouette, `state_tol`) | **read off directly** |
| `S_emp` | k-means on the predictive distribution | entropy of the state occupancy |
| config | `QUICK` | `DISCRETE` |

The transformer stack is shared — same fixed random projection, same positional
encoding, same masks, same blocks, imported rather than copied. Only the head
differs, which is what makes a model-to-model comparison mean anything.

### 14A.2 K is the process's theoretical state count

`n_states` is **not** a free hyperparameter and **not** the vocabulary size. The
runner computes it per process and per arm with
`Model_analysis.causal_state_count`:

```
coin     forward  2                backward  3
flower   forward  n + 1            backward  1 + #distinguishable outcomes
```

The flower backward count is **not** `m + 1` in general. Generic dice give
`m + 1`, but dice whose posterior columns are proportional merge into one
backward state, and at `n = 1` every outcome merges — the `n=1, m=2` null
control has **2** backward states, not 3, and `C⁻ = C⁺ = 1`. It is computed from
the dice actually drawn, under the same merge rule as `C⁻`.

`train_model` **refuses** `--model discrete` without `n_states` rather than
falling back to `V`. That fallback is the artefact the architecture exists to
remove: with the argmax over the `V`-dim logits the state budget is pinned to
the vocabulary, and measured on the coin the backward arm then sat **+0.237
bits** above `H∞` while the forward arm reached it — roughly 100× the effect
under study, pointing the way the hypothesis does.

### 14A.3 Why `discrete` needs its own config

`QUICK` is `lr=1e-2` over ~130 gradient steps. The discrete bottleneck does not
converge there. Measured on the coin at `d_model=32`:

| lr | steps | forward CE−H∞ | states | backward CE−H∞ | states |
|---|---|---|---|---|---|
| 1e-2 | 1500 | +0.573 | **1/2** | +0.120 | **2/3** |
| 3e-3 | 3000 | +0.197 | 2/2 | +0.084 | 3/3 |
| **1e-3** | **3000** | **+0.087** | 2/2 | **+0.088** | 3/3 |

So `DISCRETE` is `QUICK`'s processes and data at `lr=1e-3` and 230 epochs
(~23× the steps). The two architectures are each run at a working point rather
than at a shared one where one of them fails — **that difference is a real
caveat on any model-to-model comparison and belongs beside the results.**

Measured cost per process-repeat (2 arms × 5 folds): `onehot` ~15 s,
`discrete` ~320 s.

### 14A.4 `usage_beta`, and the bias it buys

`usage_beta` penalises collapse of the bottleneck onto a subset of the states.
It defaults to **0.01**, not 0 — unlike `weight_decay` — because 0 is measurably
unsafe: on the coin backward arm at `K=3` over 4 seeds, `β=0` collapsed one seed
in four to 2 of 3 states and `CE−H∞` ranged over 15×, while `β=0.01` found all
three every time at `+0.036 ± 0.003`.

It is not free — it biases `S_emp` upward, and the size of that bias is
configuration-dependent (measured between ~0.000 and ~+0.03 on the backward
arm). Re-run with `--usage-beta 0` to see the unpenalised number, and quote both.

### 14A.5 Where the output goes

New runs go to `All_Results/<model>/<experiment>/`:

```
All_Results/
  results_quick/  results_sweep/  results_dice/  results_trajectories/
        ^ the existing onehot baseline — untouched
  onehot/    quick/  trajectories/  sweep/
  discrete/  quick/  trajectories/  sweep/
```

**Replotting or resuming the historical results needs an explicit `--out-root`**,
e.g. `--out-root All_Results/results_sweep`. Without it a run against the new
default finds an empty directory, and `load_combined` reads a missing pickle as
"nothing done yet" and silently retrains everything. That failure cost a 23-hour
sweep once already. Every runner now prints the resolved `out_root` and the
resume plan before training starts — read that line.

### 14A.6 The causal-state figure

`--model discrete` adds one figure per arm to every experiment, on top of
everything the `onehot` runs already produce:

```
<tag>_causal_states_forward.png
<tag>_causal_states_backward.png
```

Three panels, in the style of `Jupyter_notebooks/test_new training method.ipynb`:

1. **occupancy** — token positions per state, the count written on each bar,
   unused states greyed;
2. **state vectors** — one point per state, PCA to 2-D, size and colour by
   occupancy, the count written on each point, hollow if unused. Its
   **geometry is not identified** — `state_matrix @ Q` with `Q⁻¹·emission` is an
   identical model for any invertible `Q` — which the figure states;
3. **transition probabilities** — `P(next | state)`, or `P(previous | state)`
   for the backward arm, with the value printed in each cell and the
   total-variation distance to the nearest closed-form row.

`S_emp` is printed in the title. On the coin it lands within ~0.001 bits of the
closed-form `C⁺`/`C⁻` with no clustering hyperparameters at all.

### 14A.7 The caveat to check before quoting ΔCE

At the DISCRETE budget the forward arm reaches `H∞` (+0.004…+0.013 measured)
but **the backward arm does not** (+0.14…+0.17). A ΔCE from that pair is
dominated by the backward arm's non-convergence, not by causal asymmetry. Check
the per-arm `CE − H∞` in the summary before reading anything into ΔCE.

---

## 15. Command reference

```bash
# Environment
conda activate qdrug
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version

# Verification
pytest tests/ -q                                    # 66 tests, ~20 s
python Experimental_setup/run_experiments.py --config SMOKE            # ~2 min, exercises all paths

# Training
python Experimental_setup/run_experiments.py --config QUICK            # ~9 min   ← the reportable run
python Experimental_setup/run_experiments.py --config QUICK_LARGE_HMM --only exp1
python Experimental_setup/run_experiments.py --config QUICK_LARGE_HMM --only exp2
python Experimental_setup/run_experiments.py --config LARGE            # ~3.5 h   ← CV not usable
python Experimental_setup/run_experiments.py --config QUICK --seed 1 --out-root All_Results/results_quick_seed1

# Controls
python Experimental_setup/sanity_check.py                              # ~7 min

# Parameter sweep (Section 8)
python Experimental_setup/run_sweep_experiment.py --dry-run            # coverage table, no training
python Experimental_setup/run_sweep_experiment.py --dry-run --repeats 30  # ALWAYS check the resume plan first
python Experimental_setup/run_sweep_experiment.py --repeats 30         # ~7.6 h, 125 processes
python Experimental_setup/run_sweep_experiment.py --sweep-flower --repeats 30 --weight-decay 0.1 \
       --out-root All_Results/results_sweep_wd/wd0.100          # one lambda, ~3 h
python Experimental_setup/run_sweep_experiment.py --plots-only         # redraw, no training

# The dice axis (Section 8A) — separates C--C+ from m-n, ~2.6 h
python Experimental_setup/run_dice_experiment.py --dry-run             # the design, no training
python Experimental_setup/run_dice_experiment.py --repeats 30
python Experimental_setup/run_dice_experiment.py --plots-only

# The capacity axis (Section 8B) — the direct test of the residual argument
for D in 8 16 32 64; do
  python Experimental_setup/run_dice_experiment.py --repeats 30 --d-model $D \
         --out-root All_Results/results_dice_cap/d$(printf %03d $D)
done
python Experimental_setup/analyse_capacity.py --root All_Results/results_dice_cap

# Cross-lambda decomposition (Section 8C) — analysis only, no training
python Experimental_setup/analyse_lambda_sweep.py --like-for-like

# Reproducibility: 'cpu' is the only bit-reproducible accelerator (Section 1.3)
python Experimental_setup/run_statistical_trj.py --repeats 4 --accelerator cpu \
       --out-root /tmp/traj_check

# The two architectures (Section 14A)
python Experimental_setup/run_experiments.py     --config QUICK    --model onehot
python Experimental_setup/run_experiments.py     --config DISCRETE                  # ~8 h
python Experimental_setup/run_statistical_trj.py --config QUICK    --model onehot --repeats 30
python Experimental_setup/run_statistical_trj.py --config DISCRETE --repeats 30     # ~19-24 h
python Experimental_setup/run_sweep_experiment.py --config DISCRETE \
    --sweep-coin 0.15 0.35 0.55 0.75 0.95 --sweep-flower 2 4 6 8 10 --repeats 5   # ~22 h
python Experimental_setup/run_statistical_trj.py --config DISCRETE --usage-beta 0 \
    --out-root All_Results/discrete/trajectories_beta0     # the unpenalised S_emp
bash Run_logs/launch_model_comparison.sh          # all of the above, in sequence

# Reaching the HISTORICAL results needs an explicit --out-root (Section 14A.5)
python Experimental_setup/run_sweep_experiment.py --plots-only --out-root All_Results/results_sweep

# Repeat statistics (Section 7)
python Experimental_setup/run_statistical_trj.py                       # ~2.25 h, 7 x 100  ← done
python Experimental_setup/run_statistical_trj.py --repeats 3 --out-root /tmp/traj_check
python Experimental_setup/run_statistical_trj.py --only flower_n6_m4   # one process, retrained (see 14.6)
python Experimental_setup/run_statistical_trj.py --khat                # ~3 h, adds k-hat per run
python Experimental_setup/run_statistical_trj.py --plots-only          # redraw, no training

# Analysis of saved weights
python Transformer_model/plot_state_clusters.py                       # all metrics, All_Results/results_quick
python Transformer_model/plot_state_clusters.py --out-root All_Results/results_large --metrics js

# Post-hoc evaluation (edit the RUN dict first)
python Transformer_model/Test_data_eval.py
python Experimental_setup/LLM_asymmetry_testing.py

# Documentation
python Jupyter_notebooks/build_walkthrough.py                         # regenerate walkthrough.ipynb
```

### Further reading within this repository

| Document | Contents |
|---|---|
| `IMPROVEMENT_PLAN.md` | The audit that motivated the work, with findings by tier. |
| `PHASE0_HANDOFF.md` | Repository hygiene and environment. |
| `PHASE1_2_HANDOFF.md` | Correctness fixes and the consolidation of the runners. |
| `PHASE3_HANDOFF.md` | Performance work, the latent-sampling correction, and causal-state recovery. |
| `walkthrough.ipynb` | An executed demonstration of every change, with caveats. |
