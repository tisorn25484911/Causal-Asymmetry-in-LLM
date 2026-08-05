# Repository Review & Improvement Plan — v2

Second full read-through of every Python file in `LLM_final_version/`, dated 2026-08-04.
This revision supersedes v1. Every claim below was re-checked against the code at the
line cited; several v1 claims were wrong and are corrected here (see §8 for the diff
against v1, so nothing already acted on is silently dropped).

**Format.** Each finding gives *Where → What → Why it is wrong → Fix*. The "why" matters
more than the "what": several of these bugs are individually small but interact, and two
of v1's recommended fixes made a different bug worse.

---

## 1. What the repo does — and what it can and cannot show

Two transformer decoders are trained on **the same forward HMM data**. The only
differences are the attention mask and the batch convention:

| | mask | `training_step` | at position *t* the model sees | predicts |
|---|---|---|---|---|
| forward | `tril` | `inputs, targets = batch` | `x[0..t]` | `x[t+1]` |
| backward | `triu` | `targets, inputs = batch` | `x[t..T-1]` | `x[t-1]` |

The hypothesis (Crutchfield/Ellison–Mahoney causal asymmetry; Thompson et al. 2017):
if a process needs more memory to predict backward than forward (C⁻ > C⁺), the backward
model should converge to a measurably higher cross-entropy.

### 1.1 The premise is not a theorem — state it before reporting any Δ

The entropy rate is **time-reversal invariant**: H∞(forward) = H∞(backward). The code
already asserts this (`Main_call.py:199`) and relies on it by passing the same `theory`
value for both directions (`Main_call.py:462`). Therefore:

```
CE_FW → H∞ + residual_FW
CE_BW → H∞ + residual_BW
ΔCE  = CE_BW − CE_FW = residual_BW − residual_FW
```

ΔCE is a difference of **residuals**, not of entropy rates. An unbounded-capacity,
fully-converged predictor achieves ΔCE = 0 in both directions regardless of C⁻ − C⁺.
ΔCE > 0 is purely an artefact of a *memory-bounded* predictor: the backward direction
needs more causal states, so a fixed `d_model` and a fixed epoch budget leave a larger
residual.

Two consequences that must be stated in any write-up:

1. A null result (ΔCE ≈ 0) is **ambiguous** — it could mean "no asymmetry" or "`d_model`
   was large enough to absorb it". The only way to disambiguate is the `d_model` sweep
   (§7, Phase 4). That sweep is not a nice-to-have; it is what makes ΔCE interpretable.
2. The size of ΔCE has no theoretical prediction. Only its **sign** is predicted, and
   only when C⁻ ≠ C⁺. Which makes §5 A1 below fatal.

---

## 2. Module flow

```
Data_generation.py            coin_generation, flower_process_generation, CoinDataset,
                              FlowerDataset (dead), Rev_HMM_generation, make_loader
Flower_process_generation.py  parametric FlowerDataset + generator (has pre_depth)
         │
OneHot_model.py               PositionalEncoding → AttentionModel → OneHotDecoder
                              WordEmbDecoder (unreachable), cross_ent_onehot (log2 → BITS)
         │
Training_model.py             ChunckDataset (random 256/512-token windows)
                              Record_training (Lightning callback)
                              train_model → train_test_val_pipeline (5-fold CV + hold-out)
         │
Model_analysis.py             latent_extraction, plot_umap, perplexity_calculation,
pq_experiment.py              statistical_complexity{,_empirical}, heatmaps
         │
RUNNERS  →  sanity_check.py, Main_call.py, main_large.py
POST-HOC →  LLM_asymmetry_testing.py, Test_data_eval.py   (load results/models/*.pt)
```

`Main_call.py` imports the flower **generator** from `Data_generation` but the flower
**Dataset** from `Flower_process_generation` (lines 82, 84). `sanity_check.py` does the
same. `Data_generation.FlowerDataset` is therefore dead in every runner.

---

## 3. Where to run

| Command | What it is | Correct on `use_t`? | Correct on `k`? |
|---|---|---|---|
| `python sanity_check.py` | Coin p=q=0.5 + Flower n=1,m=2 → `sanity_check_flower_process/` | ✅ | ❌ (see A1) |
| `python Main_call.py` | Quick suite (exp1, exp1.2 + pq sweep, exp2) → `results/` | ❌ | ❌ |
| `python main_large.py` | Same file, bigger CFG — 73 diff lines, all config + tags | ❌ | ❌ |
| `python LLM_asymmetry_testing.py` | Post-hoc metric suite on saved `.pt` | ✅ | ✅ |
| `python Test_data_eval.py` | Autoregressive generation, cumulative PPL, UMAP | ✅ | ✅ |

Note the pattern: **the two training runners are the only files that get `use_t` and `k`
wrong.** The three analysis-only consumers all get them right, and so does the README —
which documents the correct convention twice, in a code block at README:115-124 and in
prose at README:235: *"`use_t="last"` for forward, `"first"` for backward… Clustering at
any other position mixes different context lengths and produces noisy estimates."* The
runners contradict their own documentation.

### 3.1 Environment — resolved

v1 said "nothing runs right now". That is no longer true and the v1 Phase-0 instruction
to build a new env is obsolete:

- **`qdrug`** (`/opt/anaconda3/envs/qdrug`, py3.11.14, torch 2.12.1, lightning 2.6.5) is
  the working environment. All notebooks already point at its Jupyter kernel. Use it.
- `base` was broken by a numpy-2 ABI mismatch (`scipy` 1.12, `pyarrow` 14 compiled against
  numpy 1.x while base runs numpy 2.4.6). Both were upgraded; base now imports the full
  ML stack and runs `Trainer.fit`. `qutip`/`qhdopt`/`streamlit` in `base` remain broken —
  they pin numpy<2 and predate the upgrade. Keep quantum libs in their own env.
- Still worth doing: freeze `qdrug` to `requirements.txt` so the result is reproducible.

---

## 4. Reference values (ground truth for every config in the repo)

Computed analytically; use these to check any run. Coin: C⁺ = H(π_states),
C⁻ = H(π_tokens), H∞ = π₀H₂(p) + π₁H₂(q).

| config | p | q | H∞ | C⁺ | C⁻ | C⁻−C⁺ |
|---|---|---|---|---|---|---|
| `sanity_check.py` coin | 0.50 | 0.50 | 1.0000 | 1.0000 | 1.5000 | **+0.5000** |
| `Main_call.py` exp1 | 0.40 | 0.80 | 0.8879 | 0.9183 | 1.5656 | **+0.6473** |
| `Main_call.py` exp1.2 | 0.10 | 0.90 | 0.4690 | 0.4690 | 0.8911 | **+0.4221** |
| `main_large.py` exp1 | 0.30 | 0.40 | 0.9197 | 0.9852 | 1.4888 | **+0.5036** |
| `main_large.py` exp1.2 | 0.40 | 0.80 | 0.8879 | 0.9183 | 1.5656 | **+0.6473** |

The coin is a valid positive control for **every** (p,q): the three backward states refine
the two forward states (token 1 ≡ state 1; tokens 0 and 2 split state 0 by the previous
state), so C⁻ − C⁺ = P(state 0)·H(split) ≥ 0 identically.

Flower (uniform die selection, so C⁺ = 1 + ½log₂n; C⁻ = 1 + ½H(π_outcome) **only when
outcomes are distinguishable** — see A1):

| config | n | m | C⁺ | C⁻ | fwd/bwd states | verdict |
|---|---|---|---|---|---|---|
| `sanity_check.py` | 1 | 2 | 1.0000 | 1.0000 | 2 / 2 | **null — no asymmetry** |
| `Main_call.py` exp2 | 4 | 2 | 2.0000 | 1.4952 | 5 / 3 | **C⁺ > C⁻ — reversed** |
| `main_large.py` exp2 | 6 | 4 | 2.2925 | 1.9899 | 7 / 5 | **C⁺ > C⁻ — reversed** |

**C⁻ > C⁺ requires H(π_outcome) > log₂n, which requires m > n.** No flower config in the
repo satisfies this.

---

## 5. Findings

### Tier A — invalidates conclusions

#### A1. Every flower configuration has the asymmetry absent or reversed

**Where.** `sanity_check.py:22-23, 507-517, 582-586, 734`; `Main_call.py` CFG:140,
`main_large.py` CFG:149-153; `LLM_asymmetry_testing.py:450`.

**What.** The flower process is `select`/`roll` alternating: die *i* ~ Uniform{0..n−1},
then outcome *j* ~ `dice_probs[i]`, emitted as token *n+j*.

Forward causal states: one state "last token was a roll" (next is a uniform die
selection, independent of everything), plus one state per die *i* (next is `dice_probs[i]`).
That is n+1 states with probabilities ½ and ½·(1/n)·n, giving

```
C⁺ = 1 + ½·log₂(n)
```

Backward causal states: one state "current token is a selection" (the previous token is a
roll drawn from the marginal outcome distribution, independent of which die), plus one
state per **distinguishable** outcome *j*, where outcomes are distinguished by the
posterior P(die = i | outcome = j) ∝ dice_probs[i,j]/n. When all m posteriors are distinct,

```
C⁻ = 1 + ½·H(π_outcome),   π_j = (1/n)·Σᵢ dice_probs[i,j]
```

**Why the repo is wrong — two separate failures:**

1. **n=1 (sanity_check) degenerates.** With one die, P(die|outcome j) = 1 for every j, so
   all m outcome-states collapse into a single state. C⁻ = H(½,½) = 1 = C⁺. The process
   `0,X,0,X,…` with X iid uniform{1,2} reads identically backwards — it is **exactly
   time-reversible**. The module docstring at `sanity_check.py:22-23` even states the
   refuting fact while drawing the opposite conclusion: *"token 1 and 2 both
   deterministically precede token 0"* — that is precisely why they are indistinguishable
   looking backward and why the m+1 formula fails here.

2. **n=4,m=2 and n=6,m=4 have it backwards.** Since C⁻ ≤ 1 + ½log₂m, C⁻ > C⁺ requires
   m > n. Both training configs have n > m, so C⁺ > C⁻ and the *forward* direction is the
   harder one — the prediction is ΔCE < 0. `main_large.py:149` already contains the
   correct state counts in its own comment ("forward causal states=7, backward=5") without
   anyone noticing that 7 > 5 refutes the hypothesis being tested.

**Consequences.**
- `sanity_check.py:734` prints `'BW harder — PASS' if delta > 0 else 'FW harder — FAIL'`.
  For the flower arm the correct answer (ΔCE ≈ 0) is reported as **FAIL**.
- `sanity_check.py:586` forces `k = m+1 = 3` clusters onto a 2-state latent, so the
  empirical C⁻ is inflated by construction and will appear to "confirm" 3 states.
- `LLM_asymmetry_testing.plot_complexity` titles every flower chart "C⁻ > C⁺ expected"
  (line 450) while evaluating configs where C⁺ > C⁻.
- Any flower result already written up asserts the opposite of what the process does.

**Fix.**
- Relabel the `sanity_check.py` flower arm as the **null control** (it is exactly the
  control §7 Phase 4 asks for — it already exists, just mislabelled). Set `k=2` for both
  directions, change the pass criterion to `|ΔCE| < tol`, and rewrite the docstring at
  lines 16-23 and 507-517.
- For a *positive* flower case choose **m > n** (e.g. n=2, m=6 → C⁺ = 1.500,
  C⁻ ≤ 2.292) with dice whose columns are non-proportional, and update `Main_call.py`
  /`main_large.py` CFG accordingly.
- Keep the coin as the primary positive control — it is asymmetric for every (p,q) (§4).
- Add the closed forms above as `flower_complexity(n, m, dice_probs)` next to
  `statistical_complexity`, so `S_theory` is available for flower runs instead of `nan`
  (`Main_call.py:628-629`).

#### A2. Forward and backward models are compared on different data splits

**Where.** `Training_model.py:265-267` (`random_split`, no generator);
`Training_model.py:345-347` (fold permutation *is* seeded); `Main_call.py:419` →
`430`/`440` (one loader object reused for both directions).

**What.** `test_train_validation` calls `torch.utils.data.random_split(...)` with **no
`generator=`**, so it consumes the global torch RNG. `train_test_val_pipeline` is invoked
once for forward and once for backward, so the two runs draw **different hold-out test
sets and different fold memberships**. Only the fold *permutation* inside a given
train/val pool is seeded (`manual_seed(0)`), which does not help because the pool itself
differs.

Compounding it: both pipelines receive the **same** `loader_fw` object, whose
`ChunckDataset.rng` (`Training_model.py:20`) is a single stateful generator. The forward
run advances it, so the backward run draws different random windows of the same
sequences. `sample_seq = next(iter(loader_fw))[0][0]` (`Main_call.py:423`) advances it
once more before either run starts.

**Why it matters.** The headline number is a *difference* of two CE values,
`test_ppl_bw − test_ppl_fw`, expected to be a few hundredths of a bit. Right now that
difference is contaminated by split noise and window noise of unknown magnitude, and the
comparison is **unpaired** when it could trivially be paired. No amount of fixing the PPL
base (B1) or the Jensen bias (B2) repairs this — those change both arms equally; this does
not.

**Fix.** Pass an explicit generator to `random_split`; construct a **fresh**
`ChunckDataset` per direction from the same seed (or precompute the window offsets once
and share them); add a `set_seed(seed)` helper seeding `numpy`, `torch`, and the chunk RNG,
called at the top of every runner. Then run both directions from the same seed and report
paired differences. This is the highest-value single change in the repo.

#### A3. Backward complexity is read at the *minimum*-context position

**Where.** `Main_call.py:456, 526, 658` (and the same three lines in `main_large.py`).

**What.** `analyse_model(..., use_t="last")` is passed for the backward model.
`statistical_complexity_empirical` (`Model_analysis.py:445-448`) then clusters
`latents[:, -1, :]`.

**Why it is wrong.** With a `triu` mask, position *t* attends to `[t, T-1]`. Position
`T-1` attends to **itself only** — it has seen exactly one token. Maximum context for a
backward model is at position **0**. So every backward complexity number in `results/` was
computed from latents that encode a single token, which cannot express more than the
token identity. The inline comment on those lines even says "# forward model → last pos",
describing a model that is not forward.

**Corroboration.** `sanity_check.py:481, 586`, `LLM_asymmetry_testing.py:548, 662` and
`Test_data_eval.py:360, 399` all use `use_t="first"` for backward, and README:115-124 and
README:235 both document that convention as the intended one. The two training runners are
the only files that violate it.

**Fix.** `use_t="first"` for backward in both runners.

#### A4. `Main_call.py` and `main_large.py` collide inside `results/`

**Where.** `Main_call.py:406/478/599` vs `main_large.py:406/491/612`; `OUT_ROOT="results"`
hard-coded in both (`Main_call.py:693`).

**What.** Both write to `results/models/` and both use the tag `exp1_coin_p03_q04` for
experiment 1 — but at different parameters (Main_call p=0.4,q=0.8; main_large p=0.3,q=0.4)
and different `d_model` (32 vs 64). There is exactly one
`results/models/exp1_coin_p03_q04_{fw,bw}.pt` pair; whichever runner finished last owns it.

**Observed state on disk.** `results/models/` currently holds a **mixture** from at least
two runs: `exp1_2_coin_p01_q09` (a Main_call tag) and `exp1_2_coin_p04_q08` (a main_large
tag), `exp2_flower_n4_m2` (Main_call) and `exp2_flower_n6_m4` (main_large), mtimes spanning
Mar 21 → Mar 23. Every checkpoint has `rand_prj` of shape (·, 64) → **d_model = 64**, so
all surviving weights came from a `d_model=64` run; `Main_call.py` as committed uses
`d_model=32` and cannot reproduce any of them.

**Why it is dangerous.** `LLM_asymmetry_testing.py` and `Test_data_eval.py` hard-code
main_large's tags *and* its CFG (`coin_p1=0.3, coin_q1=0.4`, `d_model=64`). Run either
after `Main_call.py` and it will load `exp1_coin_p03_q04_*.pt` and score it against
p=0.3,q=0.4 data and theory — even though Main_call trained that tag at p=0.4,q=0.8.
Today the d_model mismatch (32 vs 64) makes `load_state_dict` raise, which is the **only**
reason this has not produced silent nonsense. Align `d_model` and it goes quiet.

**Fix.** This promotes the v1 "collapse the two runners" item from a maintenance nicety to
a correctness fix: one `run_experiments.py`, a `configs.py` exposing `QUICK`/`LARGE`/
`SANITY`, selected by `--config`, and **`out_root` as part of the config** so each config
writes to its own directory. Derive every tag from the config values
(`f"exp1_coin_p{p:.2f}_q{q:.2f}"`) so a tag can never disagree with the parameters again.

---

### Tier B — correctness bugs

#### B1. Train perplexity is base *e*, validation perplexity is base 2

**Where.** `Training_model.py:147` `self.step_ppl.append(math.exp(loss))`.

**Why.** `cross_ent_onehot` uses `logits_prob.log2()` (`OneHot_model.py:90`) and returns
`2**loss` (line 92), so the loss is in **bits**. `math.exp` of a bits-valued CE is
meaningless. Panel 3 (train PPL, base e) and panel 4 (val PPL, base 2, via
`_eval_loss_on_loader`) of every CV figure are on different scales and are placed side by
side as if comparable.

**Fix.** `2 ** loss`. Note the comment on that line — "PPL consistent with model's
exp(CE)" — is stale; the model has not used `exp` since the `2**loss` commit.

#### B2. `_eval_loss_on_loader` averages the wrong quantity

**Where.** `Training_model.py:53-79`.

**Why.** It accumulates `total_ppl += ppl.item()` where `ppl = 2**loss_of_this_batch`, then
returns `total_ppl / n_batches`. That is **mean(2^CE)**, not **2^mean(CE)**. By Jensen's
inequality mean(2^CE) ≥ 2^mean(CE), so reported perplexity is biased **high**, by an amount
that grows with the variance of per-batch CE. Measured on a realistic spread
(CE ∈ {0.9, 1.4, 1.1, 0.6}): 2.0411 vs 2.0000, a +0.041 bias — the same order as the ΔCE
being measured.

Second defect: both `total_loss` and `total_ppl` are divided by `n_batches`, so a ragged
final batch (DataLoader defaults to `drop_last=False`) carries the same weight as a full
one. The loss is a per-token mean inside `cross_ent_onehot`, so the aggregate should be
token-weighted.

**Why it matters beyond cosmetics.** `fold_val_loss` from this function selects the best
fold (`Training_model.py:386, 393`), and `Ppl_emp` in `pq_experiment_full:467-468` feeds
the ΔPPL heatmap (`Main_call.py:561`). By contrast `perplexity_calculation`
(`Model_analysis.py:186-195`) concatenates first and computes one CE — unbiased. So the
repo already contains a correct implementation and an incorrect one, used in different
places.

**Fix.** Accumulate `Σ CE·n_tokens` and `Σ n_tokens`, return `(Σ/Σ, 2**(Σ/Σ))`.

#### B3. `cross_ent_onehot` returns NaN on saturated logits

**Where.** `OneHot_model.py:89-91`.

**Why.** `cond_ent = -(target_prob * logits_prob.log2()).sum(dim=1)` with `target_prob`
one-hot. For non-target classes `target_prob = 0`. If any class probability underflows to
exactly 0, `log2(0) = -inf` and `0 * -inf = NaN` in IEEE-754 — and the NaN propagates
through `.sum()`, poisoning the *whole* batch loss even though the target class was fine.

**Verified.** With logits `[s, 0, 0]` and target 0:

| logit scale | `cross_ent_onehot` | `F.cross_entropy/ln2` |
|---|---|---|
| 10 | 0.000131 | 0.000131 |
| 50 | 0.000000 | 0.000000 |
| 100 | 0.000000 | 0.000000 |
| **200** | **nan** | 0.000000 |

**Why it is reachable here, not hypothetical.** The processes contain **deterministic
transitions**: coin token 2 → token 1 with probability 1 (`coin_true_conditional` row
`bw[2] = [0,1,0]`), and the flower process always follows an outcome with a die selection.
Cross-entropy training on a deterministic transition drives the corresponding logit gap to
infinity — there is no finite optimum. With Adam at lr=1e-2 for 60–80 epochs
(`main_large`, `sanity_check`), reaching a gap of 200 is realistic. The failure is silent:
the loss becomes NaN, the recorded curve breaks, and nothing raises.

v1 listed this under "optimisation opportunities" (item 3). It is a **latent correctness
bug**, and it should be fixed before the long runs, not after.

**Fix.** `loss = F.cross_entropy(flat_input, flat_target) / math.log(2)`. Identical value,
uses the log-sum-exp trick internally, no underflow, and faster. Keep `perplexity = 2**loss`.

#### B4. The analysis loaders are built and never used

**Where.** `Main_call.py:421, 495, 623-624` (and the same in `main_large.py`).

**Why.** `loader_fw_ana = _loader(ds_fw, ...)` constructs the full-sequence loader, then
`analyse_model` is called with `loader_fw` — the **chunked training loader**
(lines 450, 454, 520, 524, 652, 656). So UMAP and complexity run on 256-token windows.
README:233 states the design explicitly: *"Chunked training, full-length analysis… analysis
on full sequences remains in-distribution."* The code does the opposite of its own spec.

`experiment_2` additionally builds `seqs_bw`, `ds_bw`, `loader_bw`, `loader_bw_ana`
(lines 613, 620, 622, 624) and uses none of them — `cv_bw` trains on `loader_fw`
(line 643). The comment at line 646 says "reversed data → forward model", which describes
the exact opposite of what the code does (forward data → backward model). The `loader_bw`
parameter of `compare_fw_bw` (line 315) receives `loader_fw` at all three call sites and is
never read.

**Fix.** Pass `loader_*_ana` to `analyse_model`; delete the dead `ds_bw`/`loader_bw` chain
and the `loader_bw` parameter; correct the comment. **Do this together with D3** — see the
interaction note there.

#### B5. `max_len = chunk` contradicts the stated PE design

**Where.** `Main_call.py:425, 499, 627`.

**Why.** README:233: *"`max_len` is always set to the full sequence length so the
positional encoding table covers the entire range."* The runners set `max_len = chunk`
(256/512). `PositionalEncoding.forward` auto-grows the table when `T > pe.size(0)`
(`OneHot_model.py:44-46`), so nothing crashes — but the table is then rebuilt **at
inference time on the analysis batch**, and the model was never trained at those positions.
Combined with B4 this is currently masked (analysis also runs at T=chunk); fixing B4
without fixing B5 would expose it.

`sanity_check.py:450, 554` sets `max_len` to the full sequence length — the correct
reference.

**Fix.** `max_len = <full sequence length>` in both runners.

#### B6. Flower complexity clusters at k = n+m instead of n+1 / m+1

**Where.** `Main_call.py:654, 658` — `k=n+m` for **both** directions.

**Why.** The causal-state counts are n+1 forward and m+1 backward (§4), not n+m. For
n=4,m=2 that asks k-means for 6 clusters when the forward model has 5 states and the
backward model has 3. Since `S = H(cluster occupancy) ≤ log₂k`, forcing a larger k inflates
both numbers and destroys the comparison.

`sanity_check.py:582-586`, `LLM_asymmetry_testing.py:660-662` and
`Test_data_eval.py:395-399` all correctly use `k=n+1` / `k=m+1`. Again the runners are the
outliers.

**Fix.** `k=n+1` forward, `k=m+1` backward — and see C1 for why fixed k is a weak
estimator regardless.

#### B7. `'cfg' in dir()` is always False

**Where.** `Main_call.py:358-359`.

```python
c_fw, mlbl = _project2d(fl_fw, n_neighbors=cfg.get('umap_n_neighbors', 200) if 'cfg' in dir() else 200)
```

**Why.** `dir()` with no argument returns the names in the **current local scope**. Inside
`compare_fw_bw`, `cfg` is not a parameter and not a local, and `dir()` would not see a
global anyway. The condition is always False, so the expression is always `200`. The
`umap_n_neighbors` config key is decorative.

Note this is not the only path that ignores it: `plot_umap` calls `_project2d(sub_l)`
with no `n_neighbors` (`Model_analysis.py:123`), so it also always uses the default 200.

**Fix.** Add `cfg` to the `compare_fw_bw` signature and pass it through; thread
`n_neighbors` into `plot_umap`. Then reconsider the value — 200 neighbours on 1000 points
is a very large neighbourhood and will smear exactly the local cluster structure the plot
is meant to show. 15–50 is more appropriate (`sanity_check.py` uses 15).

#### B8. Experiment-1 tag does not match its parameters

**Where.** `Main_call.py:406` — `tag = "exp1_coin_p03_q04"` while `CFG` sets
`coin_p1=0.4, coin_q1=0.8` (line 132).

**Correction to v1.** v1 claimed "Every exp1 output directory, PNG and pickle is
mislabeled". That is **not true of what is on disk**: all existing checkpoints are
`d_model=64`, i.e. produced by `main_large.py`, where the same tag correctly matches
p=0.3,q=0.4. This is a **latent** bug that will mislabel the next `Main_call.py` run, not
existing corruption. It is still worth fixing, and A4's "derive tags from config" removes
the whole class of error.

#### B9. Three `plt.show()` calls under the `Agg` backend

**Where.** `Model_analysis.py:410` (`plot_perplexity`), `Model_analysis.py:541`
(`statistical_complexity_compare`), `pq_experiment.py:299` (`plot_heatmap`).

**Why.** Every runner sets `matplotlib.use("Agg")` (`Main_call.py:71`, `sanity_check.py:42`,
…). Under Agg, `plt.show()` is a no-op. The first two build a figure and never save it, so
those figures are **silently discarded** — `plot_perplexity` returns `None` and
`statistical_complexity_compare` returns only the numbers. `plot_heatmap` does save first
(line 297) so nothing is lost there, but it never calls `plt.close(fig)`, leaking a large
2×2 figure per call.

**Fix.** Replace with `savefig(fig, path)`; add a `save_path` parameter to the two
`Model_analysis` functions; close the figure in `plot_heatmap`.

#### B10. `FW_BW_loss_comparison` builds a figure it never returns

**Where.** `Model_analysis.py:675-693` — bare `return` on line 693.

**Why.** The figure is constructed and then unreachable. Additional defects in the same
function: both x-axes are labelled "Epoch" while `step_loss` is indexed by gradient step;
panel 2's title says "Backward Model Training Loss" while it plots the BW−FW difference;
panel 1's title says "Forward Model Training Loss" while it plots both curves.

**Fix.** `return fig`; relabel axes to "Gradient step"; retitle panel 2 "Loss difference
(BW − FW)". (`Main_call.plot_loss_theory` and `sanity_check.plot_loss_theory` already do
this correctly and are what the runners actually call — this function is currently used by
nothing, so consider deleting it instead.)

#### B11. `n_layers` is never plumbed through training

**Where.** `Training_model.py:192-220` — `train_model` has no `n_layers` parameter and
does not forward one to `OneHotDecoder`.

**Why.** `OneHotDecoder.__init__` defaults to `n_layers=2` (`OneHot_model.py:116`), so every
trained model is 2-layer regardless of intent. Meanwhile `LLM_asymmetry_testing.py:75` and
`Test_data_eval.py:170` both instantiate with `n_layers=cfg["n_layers"]`. Both configs
happen to say 2, so it works — silently. Change either config and `load_state_dict` fails
with a confusing shape error rather than a clear message.

**Fix.** Add `n_layers` to `train_model` and `train_test_val_pipeline`, thread it from CFG,
and record it next to the weights (a small JSON sidecar per `.pt` naming
`d_model`/`n_layers`/`p`/`q`/`mode`) so loaders can validate instead of guessing.

#### B12. Raising `record_every_n_steps` silently corrupts `epoch_loss`

**Where.** `Training_model.py:134` (early return) vs `149-150` (accumulation).

```python
if trainer.global_step % self.record_every_n_steps != 0:
    return                          # ← line 134
...
self._epoch_loss_sum   += loss      # ← line 149, AFTER the return
self._epoch_loss_count += 1
```

**Why it matters.** `_epoch_loss_sum` accumulates only on **recorded** steps. At the
current `record_every_n_steps=1` that is every step, so `epoch_loss` is a true epoch mean.
The moment D1 is applied (`record_every_n_steps=25`), `epoch_loss` becomes the mean over
every 25th step — a different estimator, changed silently, with no error. This is a direct
interaction between two of v1's own recommendations that v1 did not flag.

**Fix.** Move the accumulation above the modulo check, so it always runs.

#### B13. `record_val_loss` is a dead parameter

**Where.** `Training_model.py:102` (in the signature) — never assigned in `__init__`
(lines 106-110 assign the other five). `train_model:228` passes `record_val_loss=True`,
which does nothing. Validation recording is actually gated on `val_loader is not None`
(line 153).

**Fix.** Delete the parameter and the call-site argument.

#### B14. `Test_data_eval.evaluate_one` computes a different metric than its comment claims

**Where.** `Test_data_eval.py:192-198`.

```python
# perplexity_calculation collects all logits then computes global CE loss
# → true dataset-level perplexity (not mean of batch means)
ppl = perplexity_autoregressive(model, num_token, ...)   # ← not that function
```

**Why.** `perplexity_autoregressive` samples the model's **own** sequence and scores the
model on it — the self-generated entropy-rate metric that `LLM_asymmetry_testing.py:94-95`
explicitly labels *"informational only — not used for asymmetry comparison because each
model evaluates on its OWN generated sequence"*. `compare_plot` (line 297) then presents it
as the headline FW-vs-BW comparison. So `Test_data_eval.py`'s primary chart is the one
metric the rest of the repo says must not be used for the comparison.

**Fix.** Either call `perplexity_calculation(model, loader_ana, ...)` as the comment says,
or keep the autoregressive number and retitle the plot "informational only", adding the
ground-truth CE as the actual comparison.

---

### Tier C — design and conceptual weaknesses

#### C1. `statistical_complexity_empirical` cannot measure what it claims

**Where.** `Model_analysis.py:428-463`.

**Why.** With k fixed by the caller, the function returns `S = H(cluster occupancy) ≤ log₂k`.
It measures **cluster balance at a pre-specified k**; it cannot discover the number of
causal states, and it will "confirm" whatever k it is handed. Comparing forward at k=2
against backward at k=3 compares two different ceilings (1 bit vs 1.585 bits), so part of
the observed C⁻ > C⁺ gap is guaranteed by the parameterisation rather than measured.

This is the weakest link in the complexity story and the first thing a referee will attack.

**Fix.** Model selection over k (silhouette, BIC on a GMM, or the elbow of within-cluster
inertia) over a range k ∈ [1, 8], and report the **recovered** k̂ alongside the entropy.
That converts the claim from "we assumed 2 vs 3 and got 2 vs 3" into "the estimator
recovered 2 vs 3 unprompted", which is the actual result worth having.

#### C2. The positional encoding is the real forward/backward confound

**Where.** `OneHot_model.py:115` (`reverse_pos_for_backward: bool = False`), `177`, and
`Training_model.py:213-220` — the flag is **never set True anywhere in the repo**.

**Why.** v1 claimed the mask itself is the confound ("position 0 sees 1 token, position T
sees T"). That is wrong: `tril` and `triu` are exact mirror images — forward position *t*
sees t+1 tokens, backward position *t* sees T−t, and summed over positions the context
budget is identical. The mask is symmetric under reversal.

What is *not* symmetric is the positional encoding. Both models receive `PE[t]` increasing
left-to-right. For the forward model, "amount of available context" is increasing in the PE
index; for the backward model it is decreasing. The backward model must therefore learn a
different (and arguably harder) mapping from PE to context-length, which contributes to
residual_BW independently of C⁻.

**Fix.** Either set `reverse_pos_for_backward=True` for backward models so both see
"context grows with PE index", or run both settings and report the difference. Combined
with the null control (A1), this is what separates a genuine causal-asymmetry signal from
an architectural artefact.

#### C3. `perplexity_ind_model` scores the model on its own samples

**Where.** `Model_analysis.py:330-378`.

**Why.** It generates a sequence by sampling from the model and then evaluates the model's
log-loss on that same sequence. That is an estimate of the **model's own entropy rate**,
which a degenerate (over-confident, low-entropy) model minimises. It cannot be used to
compare two models. `LLM_asymmetry_testing.py:94-95` correctly labels it informational;
the function name does not, and `Test_data_eval.py` uses it as its headline (B14).

**Fix.** Rename to `self_generated_entropy_rate` and give it a docstring stating it is not
a comparison metric.

#### C4. Duplicated definitions that will drift

**Where.**
- `FlowerDataset` and `flower_process_generation` exist in **both** `Data_generation.py`
  (hard-coded n=4/m=2 attributes, no `pre_depth`) and `Flower_process_generation.py`
  (parametric, has `pre_depth`). Every runner imports the generator from the first and the
  Dataset from the second, so `Data_generation.FlowerDataset` is dead.
- `coin_true_conditional` and `stepwise_kl_coin` are byte-identical in
  `Model_analysis.py:268-327` and `LLM_asymmetry_testing.py:183-238`.
- `entropy_rate_coin` exists three times (`Main_call.py:196`, `sanity_check.py:146`,
  `LLM_asymmetry_testing.py:85`).
- `_project2d`, `mkdir`, `save_pkl`, `save_weights`, `cleanup` are duplicated between
  `Model_analysis.py` and `sanity_check.py`.

**Verified correct, so safe to consolidate rather than re-derive:** I checked all six rows
of `coin_true_conditional` against the HMM by hand — forward
`[[1−p,p,0],[0,1−q,q],[1−p,p,0]]` and backward `[[1−p,0,p],[q(1−p),1−q,pq],[0,1,0]]` are
both correct, in both copies. The backward state probabilities in `statistical_complexity`
(`Model_analysis.py:418-421`) equal the token marginals P(obs=0/1/2) exactly, which is the
right identification for this process.

**Fix.** Delete the `Data_generation` flower duplicates; import the KL helpers from
`Model_analysis`; move the shared utilities into a `utils.py`.

#### C5. `rand_prj` is a learned parameter in the "one-hot" model

**Where.** `OneHot_model.py:129-131` (`nn.Parameter`) vs `OneHot_model.py:249-251`
(`register_buffer`), with the comment "Fixed random projection" on the buffer version.

**Why.** In `OneHotDecoder`, `one_hot @ self.rand_prj` with `rand_prj` an `nn.Parameter`
**is** a learned embedding table — mathematically identical to `nn.Embedding`. So "onehot"
mode is not a fixed random projection at all. Separately, `OneHotDecoder.we`
(`OneHot_model.py:124`) is an `nn.Embedding` that is never used in `forward`: dead
parameters that receive no gradient, are serialised into every `.pt`, and pad the
checkpoints.

**Correction to v1.** v1 concluded from this that "the two classes are far closer than
their names suggest". They are not: `OneHotDecoder` is n-layer with pre-norm LayerNorm,
FFN and residuals; `WordEmbDecoder` is a **single** attention layer with no FFN, no
LayerNorm and no `n_layers` argument. They are architecturally quite different — which is
why the right action is v1's *other* recommendation: delete `WordEmbDecoder` (no runner
sets `embed_type="wordemb"`), and either freeze `rand_prj` or rename the mode to reflect
that it is learned.

#### C6. Every UMAP plots a prefix, and often a single sequence

**Where.** `Model_analysis.py:58-61` (`_sub` returns `arr[:n]`), used at line 120;
`sanity_check.py:170-171` and `Main_call.py:356-357` do the same slicing inline.

**Why.** `flat_l = latents.reshape(-1, d)` flattens (N, T, d) row-major, so the first
`n_pts` rows are **all positions of the first ⌈n_pts/T⌉ sequences**. With `n_pts=500` and
T=1999 (`sanity_check.py`), the UMAP shows positions 0-499 of **sequence 0 only** — one
trajectory, not a sample of the latent distribution. Any cluster structure seen is
conditioned on one realisation.

**Fix.** Random subsample without replacement across the flattened array, or better,
sample one position per sequence (which is what `statistical_complexity_empirical` already
does correctly, and why its estimator is better-founded than the plot's).

#### C7. `make_loader`'s `states` argument is a no-op

**Where.** `Data_generation.py:122` — `seqs, _ = data, states`.

**Why.** This is a tuple unpack that assigns `seqs = data` and discards `states`, written
in a form that makes `states` look load-bearing. In backward mode (line 119) it calls
`Rev_HMM_generation(data, states)` and again discards the reversed states. No caller uses
the state sequences at all.

**Note.** `make_loader(mode="backward")` reverses the *data* — a **third**, unused notion
of "backward" alongside the `triu` mask and the batch swap. Only `pq_experiment` calls
`make_loader`, and always with `mode="forward"`. Delete the backward branch or document
that it is not the mechanism the experiments use.

#### C8. `last_attention` records only the final layer

**Where.** `OneHot_model.py:189-201` — `attn_prob` is the loop variable, read after the
loop ends.

**Why.** Every attention heatmap therefore shows layer 2 only, with no label saying so, and
with `n_layers=0` the line would raise `NameError`. Not wrong, but under-specified for a
figure that is meant to be evidence.

**Fix.** Store a list of per-layer attentions (gated by a flag — see D3) and title the plot
with the layer index.

---

### Tier D — performance

Ordered by measured benefit, not by guess.

#### D1. Per-step validation (~2–4×, the largest single win)

**Where.** `Training_model.py:225` — `record_every_n_steps=1`, with `val_loader` supplied,
so `_eval_loss_on_loader` runs a **full pass over the validation set at every gradient
step** (line 153-156).

**Sizing it honestly.** `Main_call` exp1: 400 train+val samples → 320 train (10 batches of
32) and 80 val (3 batches) per fold. Each training step costs ≈3 forward-equivalents
(fwd+bwd); each recorded step adds 3 val forward passes. Overhead ≈ (3+3)/3 = **2×**.
`main_large`: 20 train batches, 5 val batches per fold → ≈**2.7×**. Real and worth fixing,
but v1's "dominates wall-clock… costing hours per run" overstates it; budget a 2–4×
speedup, not an order of magnitude.

**Fix.** `record_every_n_steps=25` — **and apply B12 in the same commit**, or `epoch_loss`
changes meaning silently.

#### D2. Python double-loops over every token

**Where.** `perplexity_ind_CE` (`Model_analysis.py:253-261`) and `stepwise_kl_coin`
(`Model_analysis.py:307-317`, duplicated in `LLM_asymmetry_testing.py:218-228`) both run
`for b: for t:` in pure Python. `LLM_asymmetry_testing` calls them with
`max_batches=None`, i.e. 500 sequences × 1999 positions ≈ 10⁶ iterations **per call**, four
calls per experiment.

**Fix.** Fully vectorisable: `p_true = target_prob[inp_np]` gives (B,T,V) by fancy
indexing, then `ce = -(p_true * np.log2(p_model + 1e-12)).sum(-1)`. Expect 100–1000×.
Same shape of fix for `heatmap_theory` (`pq_experiment.py:36-41`), which double-loops a
100×100 grid to evaluate a closed form.

#### D3. `last_attention` retains a (B,T,T) tensor on every forward pass

**Where.** `OneHot_model.py:201`, `294`.

**Sizing it, and the interaction that matters.** At training time T = chunk, so this is
32·256²·4 B = 8 MB (Main_call) or 32·512²·4 B = 34 MB (main_large) — negligible. v1's
"512 MB at T=2000" scenario does **not** occur in the runners today, precisely *because* of
bug B4 (analysis also runs on chunks). Where it does bite is `sanity_check.py`, which
correctly uses the full-length loader at T=999: 32·999²·4 B ≈ **128 MB retained per
forward pass**.

**Therefore: fixing B4 will create the memory problem D3 describes.** These two must land
in the same change. Gate the assignment behind `self.store_attention = False` and set it
True only for the plotting call sites.

#### D4. Cheaper wins

- `cross_ent_onehot` → `F.cross_entropy/ln2` (also fixes B3) — removes a softmax + log over
  (B·T, V) per step.
- The `tril`/`triu` mask is rebuilt on every forward (`OneHot_model.py:183-185`). Cache per
  `(T, device, mode)`.
- `perplexity_calculation` (`Model_analysis.py:186`) concatenates **all** logits on CPU
  before computing the loss — 80 MB for the flower config. Accumulate CE incrementally.
- The UMAP JIT warm-up runs at *import* of `Model_analysis.py` (line 15, `n_neighbors=200`
  on 20 points — the parameter is clamped to 19 anyway), **again** in `Main_call.py:107`,
  and **again** in `sanity_check.py:67`. Make it lazy and do it once.
- `plot_attention_heatmap` (`Model_analysis.py:149`) and `FW_BW_attention_comparison`
  (lines 594, 598) call the model without `torch.no_grad()`, building an autograd graph for
  a plot. Transient rather than a leak (the graph frees when `logits` goes out of scope),
  but free to fix.
- Local re-imports of `PCA` (`Model_analysis.py:110`) and `KMeans` (line 455) inside
  functions that already import them at module top (lines 7, 10).

#### D5. Stop writing checkpoints and logs you never read

**Where.** `Training_model.py:233-239` — `L.Trainer` is constructed with default
`logger=True` and `enable_checkpointing=True`.

**Why.** Every `train_model` call creates a `lightning_logs/version_N/` with a `.ckpt`.
One `Main_call.py` run makes 5 folds × 2 directions × 3 experiments = 30, plus 2 × |pq
grid| = 200 more from the sweep. Nothing in the repo ever reads them — `save_weights`
writes the models that are actually used.

**Fix.** `L.Trainer(..., logger=False, enable_checkpointing=False)`. This is the real fix;
`.gitignore` only hides the symptom.

---

### Tier E — naming

| Current | Suggested | Why |
|---|---|---|
| `ChunckDataset`, `chunck_len` | `ChunkDataset`, `chunk_len` | typo |
| `flower_process_generation(seq_len=N)` | `n_cycles=N` | emits **2N** tokens; `flower_seq_len=2000` yields 4000-token sequences |
| `statistical_complexity` | `coin_statistical_complexity` | coin-only (takes p,q); silently inapplicable to flower |
| `statistical_complexity_empirical` | `latent_cluster_entropy` | it is H(k-means occupancy) at fixed k, not complexity (C1) |
| `perplexity_ind_CE` | `soft_label_ce_ppl_coin` | coin-specific; "ind" means nothing |
| `perplexity_ind_model` | `self_generated_entropy_rate` | scores its own samples (C3) |
| `perplexity_calculation` | `teacher_forced_ppl` | says nothing about its convention |
| `_eval_loss_on_loader` | `evaluate_ce_ppl` | public API — but note it is imported by 3 files and **actually called by only one** (`pq_experiment.py`); the imports in `Main_call.py:100` and `main_large.py:101` are unused |
| `_loader` | `full_sequence_loader` | docstring "FIX-3 only" is meaningless |
| `_sub` | `take_first_n` | it is a prefix, not a subsample (C6) |
| `test_train_validation` | `holdout_split` | word order reversed; it makes no validation set |
| `train_test_val_pipeline` | `cross_validate` | "pipeline" says nothing |
| `Rev_HMM_generation` | `reverse_sequences` | does not generate and does not touch an HMM |
| `Record_training` | `MetricRecorder` | |
| `cross_ent_onehot` | `cross_entropy_bits` | "onehot" is an implementation detail being deleted (B3) |
| `pq_experiment_full` | `pq_sweep` | and `pq_experiment` (the non-full variant) is reachable only via `python pq_experiment.py` (line 334, behind an `input()` prompt) — not strictly dead, but delete it and the `__main__` block together |
| `Main_call.py` | `run_experiments.py` | README already points at `main.py` and `run_experiments.ipynb`, neither of which exists |

---

## 6. Repository hygiene

`.git` is **6.2 GB**. The cause is not what v1 said:

| blob | size |
|---|---|
| `results/all_results.pkl` | 1873 MB |
| `results/training set/*/results.pkl` ×4 | 843 / 639 / 631 / 603 MB |
| largest `lightning_logs` blob (`metrics.csv`) | 3 MB |

So **~4.6 GB is committed pickles**; `lightning_logs` checkpoints are a rounding error
(the working tree holds 30 dirs / 33 MB). v1 attributed it to "checkpoints and a 1.8 GB
all_results.pkl"; only the second half was right.

**Why the pickles are enormous.** `save_pkl(res, ...)` (`Main_call.py:467`) stores
`cv_fw`/`cv_bw`, which contain `all_recorders` — every fold's **full model** plus its
per-step latents — **and** `ana_fw["latents"]`, an (N, T, d_model) float array
(`Main_call.py:292`). At N=640, T=256, d=64 that is 42 M floats ≈ 168 MB per experiment
arm, before the models.

**Fix, in order:**
1. `.gitignore`: `lightning_logs/`, `results/`, `sanity_check_flower_process/`,
   `__pycache__/`, `*.pt`, `*.ckpt`, `*.pkl`, `.DS_Store`.
2. Strip `all_recorders` and `latents` before pickling — persist metrics and best weights
   only. Latents are cheap to regenerate from the saved `.pt`.
3. Commit the ~5745 pending deletions, then `git filter-repo` the five large blobs out of
   history. The repo is effectively unclonable today.
4. Apply D5 so the logs stop being generated in the first place.

---

## 7. Phased plan

### Phase 0 — Hygiene (~1 hour)

0.1 `.gitignore` + strip pickles + `filter-repo` (§6). Do this first so every later commit
    is cheap.
0.2 `pip freeze > requirements.txt` from `qdrug`. (The environment itself is already fixed
    — see §3.1.)
0.3 Add `L.Trainer(logger=False, enable_checkpointing=False)` (D5).

### Phase 1 — Make the comparison valid (~half a day)

*Rationale: A2 and A1 determine whether the numbers mean anything. B1/B2 determine whether
they are on the right scale. Nothing else is worth doing first.*

1.1 **A2** — seed `random_split`; fresh `ChunckDataset` per direction; add `set_seed()`
    to every runner. Report paired ΔCE.
1.2 **A1** — relabel the `sanity_check.py` flower arm as the null control (`k=2`,
    pass criterion `|ΔCE| < tol`, rewrite docstrings). Choose **m > n** for any positive
    flower experiment. Add `flower_complexity(n, m, dice_probs)`.
1.3 **A3** — `use_t="first"` for backward in both runners.
1.4 **B3** — `F.cross_entropy/ln2`. Do this before the long runs; it is a silent-NaN risk
    on deterministic transitions.
1.5 **B1, B2** — `2**loss`; token-weighted `_eval_loss_on_loader`.
1.6 **B6** — `k=n+1` / `k=m+1`.
1.7 **B4 + D3 together** — pass `loader_*_ana`, and gate `last_attention` in the same
    commit. **B5** — `max_len` = full sequence length.
1.8 **B7, B8, B12, B13** — the small silent ones.
1.9 **Re-run everything.** Results in `results/` were produced with A1, A2, A3, B2, B4,
    B5, B6 active. (Not B8 — see the correction there.)

### Phase 2 — One runner (~half a day)

2.1 **A4** — merge `Main_call.py` + `main_large.py` into `run_experiments.py`; `configs.py`
    with `QUICK`/`LARGE`/`SANITY`; `--config` flag; `out_root` in the config; tags derived
    from config values. This is a correctness fix, not tidying.
2.2 **C4** — delete the `Data_generation` flower duplicates, the duplicated KL helpers,
    the triplicated `entropy_rate_coin`; move shared utilities to `utils.py`.
2.3 **C5** — delete `WordEmbDecoder` and `OneHotDecoder.we`; decide whether `rand_prj` is
    frozen or learned and name it accordingly.
2.4 **B11** — thread `n_layers`; write a JSON sidecar next to every `.pt`.
2.5 **B9, B10, B14, C7** — the plotting and API-honesty fixes.

### Phase 3 — Make it fast (~half a day)

3.1 **D1 + B12** (same commit). 3.2 **D2** vectorisation. 3.3 **D4** cheap wins.

### Phase 4 — Strengthen the science (where new experiments live)

4.1 **C1** — model selection over k; report recovered k̂ next to the entropy. This is the
    single most attackable claim in the current write-up.
4.2 **Seed-repeat harness** — N seeds per condition; report mean ± CI on ΔCE and a paired
    test. One ΔCE from one best-fold model is not evidence, and §1.1 explains why the
    effect size has no theoretical anchor to compare against.
4.3 **Null control** — already written (A1: `sanity_check.py` flower n=1,m=2, C⁺=C⁻=1);
    relabel it and add an i.i.d. process as a second null. Must yield ΔCE ≈ 0.
4.4 **C2** — run backward models with and without `reverse_pos_for_backward=True`. With
    4.3 this is what separates causal asymmetry from a PE artefact.
4.5 **`d_model` sweep** — per §1.1, ΔCE should **shrink** as capacity grows. That curve is
    a far stronger result than a single point, and it is the only thing that makes a null
    result interpretable.
4.6 **`tests/`** — assert conditional matrices sum to 1 and match a Monte-Carlo estimate;
    `entropy_rate_coin(0.5,0.5) == 1.0`; `statistical_complexity(0.5,0.5,'backward') == 1.5`;
    flower generator emits alternating select/roll tokens; `flower_complexity(1,2,·)`
    returns C⁺ == C⁻; `2**cross_entropy_bits(uniform_logits) == vocab_size`;
    `cross_entropy_bits` is finite at logit scale 200 (B3 regression test).

### Phase 5 — Docs

5.1 README: `main.py` and `run_experiments.ipynb` do not exist; add `sanity_check.py`;
    state that flower `seq_len` is in **cycles**, not tokens; correct the
    `perplexity_ind_CE` note; add §1.1 (ΔCE is a residual difference) and the §4 reference
    tables.

---

## 8. Corrections to v1 (so nothing acted on is silently dropped)

| v1 claim | Status | Correction |
|---|---|---|
| §3 "Nothing runs right now" (numpy/scipy break) | **obsolete** | `qdrug` was always working; `base` has since been repaired. §3.1 |
| §4.1 exp1 tag → "every exp1 output is mislabeled" | **overstated** | Bug is real; all existing outputs are `d_model=64`, i.e. from `main_large.py`, where the tag is correct. Latent, not retroactive. B8 |
| §4.1 "results were produced with bugs 4–6" | **partly wrong** | Bugs 4 and 5 yes; bug 6 (tag) no. Phase 1.9 |
| §4.2 "`pq_experiment` is dead code" | **imprecise** | Not called by the runners, but reachable via `python pq_experiment.py:334`. Tier E |
| §4.2 "the two decoder classes are far closer than their names suggest" | **wrong** | `OneHotDecoder` is n-layer + FFN + LayerNorm; `WordEmbDecoder` is single-layer, none of those. Delete rather than unify. C5 |
| §4.2 "No global seeding anywhere" | **imprecise** | `ChunckDataset` takes `seed=0`. The real defect is unseeded `random_split` + a shared stateful RNG. A2 |
| §4.3 #1 "dominates wall-clock, hours per run" | **overstated** | Measured 2–4×, not an order of magnitude. D1 |
| §4.3 #3 `softmax().log2()` listed as an optimisation | **misclassified** | It is a latent **NaN bug**, reachable via deterministic transitions. Verified. B3 |
| §4.3 #4 "512 MB at B=32,T=2000" | **wrong scenario** | Does not occur in the runners (T=chunk). Occurs in `sanity_check.py` at ~128 MB, and fixing B4 will create it. D3 |
| §4.4 "`_eval_loss_on_loader` imported by 4 modules" | **wrong** | Imported by 3; called by 1. The other two imports are unused. Tier E |
| §5 Phase 0.2 ".git is 6.2 GB because checkpoints and a 1.8 GB pickle were committed" | **half wrong** | ~4.6 GB is pickles; checkpoints are a rounding error. §6 |
| §5 Phase 4.18 "tril/triu asymmetry (position 0 sees 1 token, position T sees T)" | **wrong mechanism** | The masks are exact mirror images and symmetric under reversal. The real confound is the positional encoding. C2 |
| §5 Phase 4.18 "add a null control" | **already exists** | `sanity_check.py` flower n=1,m=2 *is* a null (C⁺=C⁻=1) — mislabelled as a positive case. A1, 4.3 |
| — | **new, Tier A** | A1 flower asymmetry absent/reversed in every config; A2 unpaired splits; A4 runner collision in `results/` |
| — | **new, Tier B** | B3 NaN; B6 wrong k; B12 `epoch_loss` interaction; B13 dead param; B14 `Test_data_eval` metric mismatch |
| — | **new** | C6 UMAP plots one sequence; C8 last-layer-only attention; D5 unused checkpoints |

---

## 9. Do these three first

1. **A2** — seed the splits. Until the two directions see the same data, every ΔCE in the
   repo is contaminated by an unmeasured amount of split noise.
2. **A1** — no flower configuration currently tests the hypothesis; two test its negation.
   Fix the configs before spending another overnight run on them.
3. **B3 + B1 + B2** — one silent-NaN risk and two wrong-scale metrics, all in the loss path,
   all cheap. Do them before the long runs, not after.
