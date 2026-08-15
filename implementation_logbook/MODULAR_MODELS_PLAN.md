# Two Model Architectures, One Set of Experiments — Plan

Makes the transformer a choice rather than a constant, so every experiment can
run on the existing continuous decoder or on the discrete causal-state decoder,
with results in separate comparable trees.

Revised 2026-08-15: the state budget `K` is now the **theoretical causal-state
count for the process and arm**, not the vocabulary size. Everything below that
depends on it has been re-measured.

Model already written: `Transformer_model/DiscreteCausal_model.py`.

---

## 0. What this answers

ΔCE is a difference of *residuals*, so a converged model with capacity to spare
gives ≈ 0 whatever `C⁻ − C⁺` is. The discrete decoder attacks that differently:
it forces the prediction through a hard one-hot over a bounded state set, so the
states are **read off directly** and `S_emp` stops depending on a silhouette
score and a `state_tol`.

---

## 1. The insertion point already exists

`embed_type` is threaded end to end and gated in exactly one place:

```
configs.py:28                    embed_type = "onehot"        (BASE)
run_experiments.py:484,497,...   embed_type=cfg["embed_type"] (6 call sites)
sanity_check.py:492,504,...      embed_type=cfg["embed_type"] (4 call sites)
run_statistical_trj.py:301       embed_type=cfg["embed_type"]
Training_model.py:428,559,675    threaded through the pipeline
Training_model.py:459-463        THE GATE — raises unless "onehot"
```

Narrowed to one value when `WordEmbDecoder` was deleted (IMPROVEMENT_PLAN C5).
Widening it into a registry is the whole of the plumbing. **Do not add a
parallel `--model` path that bypasses `embed_type`** — exactly one place decides
which class is constructed.

---

## 2. The state budget K — the central change

### 2.1 The counts, verified

| process | arm | K |
|---|---|---|
| coin | forward | **2** |
| coin | backward | **3** |
| flower | forward | **n + 1** |
| flower | backward | **1 + #distinguishable outcomes** (≤ m + 1) |

The coin counts are exact for every `(p, q)` — checked at (0.6,0.4), (0.1,0.9),
(0.3,0.4), (0.05,0.05), (0.95,0.95): forward always 2 (rows 0 and 2 of the
forward conditional coincide), backward always 3 (all rows distinct).

**The flower backward count must be COMPUTED, not assumed to be `m + 1`.**
Verified:

```
generic Dirichlet dice, 40 draws at alpha=0.2:  n=2,m=8 -> 9 every time
                                                n=3,m=5 -> 6 every time
                                                n=4,m=6 -> 7 every time
n=1, m=2   -> 2    every outcome collapses (the null control, C- = C+ = 1)
n=1, m=5   -> 2    same
n=2, m=2 identical dice -> 2   (posterior columns proportional -> merge)
n=2, m=2 disjoint dice  -> 3
```

So the generic case is `m + 1`, but degenerate dice merge and `n = 1` collapses
entirely. The count is `1 + |{distinct rounded posterior columns}|`, exactly the
merge `flower_complexity` performs.

### 2.2 Where the count lives

`Model_analysis.causal_state_count(process, mode, **kw)`, beside
`flower_complexity`, sharing its `MERGE_ROUND_DP` and its optional `merge_tol`
so the two can never drift. **The model must not compute this** — it does not
know which process it is training on. The runner computes it and passes
`n_states`.

---

## 3. The model interface

A protocol, not a base class with behaviour:

```python
class CausalDecoder(Protocol):
    mode: str; token_size: int
    def forward(self, tokens) -> Tensor: ...
    def training_step(self, batch, idx) -> Tensor: ...
    last_encodings: Tensor
    def state_assignment(self, tokens=None) -> Tensor | None: ...
    def emission_table(self) -> Tensor | None: ...
```

`OneHotDecoder` returns `None` from both new methods and the analysis layer
falls back to clustering. `DiscreteCausalDecoder` returns real values. **One
branch, in `Model_analysis`** — never scattered through the runners.

---

## 4. `DiscreteCausalDecoder` — as written

```
x          (B,T,D)   transformer stack, imported from OneHot_model
state_lgt  (B,T,K)   state_head    D -> K
onehot     (B,T,K)   straight-through argmax
rep        (B,T,S)   onehot @ state_matrix
out        (B,T,V)   emission      S -> V     <- the cross-entropy is here
```

### 4.1 Why the argmax is on a dedicated head

Taking it over the `V`-dim logits pins the budget to the vocabulary, and `V` is
a property of the process that cannot be set per arm. Measured on the coin,
where `H∞ = 0.9710` is time-reversal invariant and identical for both arms:

| | CE − H∞ | states |
|---|---|---|
| forward, V-pinned | +0.0016 | 2 |
| backward, V-pinned | **+0.2371** | 3 (only after 120 epochs) |

That is ~100× the effect being measured, pointing the way the hypothesis does.

### 4.2 There is no `output_prj`

Once the argmax moves to `state_head`, the alphabet logits have no job — the
loss never touches them. Measured: `output_prj` received **exactly zero
gradient**, so `last_logits` would be an untrained random projection. Removed;
the causal-state figures replace the logits plot.

### 4.3 The straight-through estimator and the emission layer

Both load-bearing, both already argued and both measured: without the estimator
2 of 33 tensors train while the loss still falls plausibly; without the emission
layer the CE has a floor of `1/(1 + (V−1)/e)` that **grows with the vocabulary**
(0.576 at V=3, 0.198 at V=12), i.e. an artefact along the sweep's own axis.

### 4.4 Identifiability

`state_matrix` ∘ `emission` compose into one `(K, V)` map, so `state_matrix @ Q`
with `Q⁻¹·emission` is an identical model for any invertible `Q` (max output
deviation 3.6e-07). The state-vector **scatter geometry is not identified**;
`emission_table()` is. Both are plotted, and the figure says which is which.

---

## 5. Measured behaviour at K = theory

Coin, `p=0.6 q=0.4`, `H∞ = 0.9710`, 60 epochs, 4 seeds, `d_model=20`.

| β | arm | K | CE − H∞ (per seed) | states used |
|---|---|---|---|---|
| 0.00 | forward | 2 | +0.0026, +0.0004, +0.0211, +0.0015 | 2/2 always |
| 0.00 | backward | 3 | +0.2681, +0.0225, +0.1794, +0.0179 | **2/3 on seed 0** |
| 0.01 | forward | 2 | +0.0016, +0.0006, +0.0184, +0.0015 | 2/2 always |
| 0.01 | backward | 3 | +0.0387, +0.0377, +0.0362, +0.0329 | **3/3 always** |

Emission tables at K = theory match the closed form closely — forward
TV = 0.0039 and 0.0071 against `[0.4,0.6,0]` and `[0,0.6,0.4]`.

**Three things follow.**

1. **The dedicated head plus K = theory cuts the backward penalty from +0.237 to
   ~+0.036**, a 6–7× reduction.
2. **`usage_beta = 0` is not safe for the backward arm.** One seed in four
   collapses to 2 of 3 states, and CE − H∞ ranges over 15×. At β = 0.01 it is
   stable to ±0.003 and always finds all three.
3. **A residual arm asymmetry of ~+0.031 bits survives** (forward +0.006,
   backward +0.036). That is still ~10× the real ΔCE (0.001–0.003 bits). See §9.1.

### 5.1 `S_emp` from occupancy

`S_emp = −Σ p_k log₂ p_k` over the occupancy distribution. No clustering, no
`state_tol`, no silhouette. Measured against the closed form (`C⁺ = 0.9710`,
`C⁻ = 1.3593`), 3 seeds:

| β | arm | S_emp − theory |
|---|---|---|
| 0.00 | forward | −0.0008, −0.0008, −0.0055 → mean **−0.0024** |
| 0.00 | backward | −0.4438, −0.0010, +0.2007 → mean −0.0814, **unusable spread** |
| 0.01 | forward | −0.0008, −0.0008, −0.0013 → mean **−0.0010** |
| 0.01 | backward | +0.0321, +0.0059, +0.0345 → mean **+0.0242** |

The recovered backward occupancy at β=0.01 is `[0.576, 0.164, 0.260]` against a
true `[0.6, 0.16, 0.24]` (labels are arbitrary, so compare sorted).

**The penalty buys stability at the price of a small upward bias, and the bias is
arm-dependent** — forward −0.001, backward +0.024. A measured `C⁻ − C⁺` would
therefore be inflated by ~0.025 bits out of 0.388, about 6%, in the direction
the hypothesis predicts. Report `S_emp` at β = 0 alongside, and quote the bias.

`recover_causal_states` stays for `OneHotDecoder`, and is available for the
discrete model behind `--khat` as a cross-check, off by default.

---

## 6. Output layout

New runs, **both** models:

```
All_Results/
  results_quick/  results_sweep/  results_dice/  results_trajectories/
        ^ existing, untouched, the historical onehot baseline
  onehot/    quick/  trajectories/  sweep/
  discrete/  quick/  trajectories/  sweep/
```

`out_root` becomes a function of `(model, experiment)` rather than a constant in
`configs.py:112,139,164`. Absolute `--out-root` still passes through untouched.

**Document, do not let anyone discover:** replotting or resuming the *existing*
results now needs an explicit `--out-root All_Results/results_sweep`. Without
it, `load_combined` reads a missing pickle as "nothing done yet" and silently
retrains everything — the failure that cost a 23-hour sweep once already. The
runners must print the resolved out_root and the resume plan **before** training.

---

## 7. Figures

### 7.1 Notebook style — the causal-state plots and the transition probabilities

These three come straight from `Jupyter_notebooks/test_new training method.ipynb`
and keep its presentation, because that is the form the states are legible in:

1. **occupancy** — one bar per state, the token count written on each bar,
   unused states greyed;
2. **state scatter** — one point per state, PCA to 2-D when `S > 2`, marker size
   and colour by occupancy, **the count written on each point**, hollow marker
   for unused states, identifiability caveat in the subtitle;
3. **transition-probability readout** — the `P(next | state)` heatmap
   (`P(prev | state)` in backward mode) with the value printed in each cell,
   beside the closed-form rows where they exist, exactly as the notebook prints
   `learned` against `theory` with a TV distance per state.

They replace the logits UMAP for the discrete model, which no longer exists
(§4.2). `OneHotDecoder` keeps `plot_umap` unchanged. Dispatch on
`state_assignment() is None`, inside `Model_analysis`.

### 7.2 Repo style — everything else

`plot_complexity`, `plot_final_loss`, `plot_trajectories`, `plot_summary`,
`plot_grid_summary`, `plot_sweep_scatter`, `plot_sweep_trajectories` stay exactly
as they are, in the existing experiment-file style. They consume the record, not
the model, and must stay model-agnostic.

### 7.3 One new cross-model figure

`model_comparison.png`: ΔCE and area against `C⁻ − C⁺`, both architectures
overlaid, one marker shape per model. The figure the exercise exists to produce.

---

## 8. Steps

Each step ends with the 74 tests green and `embed_type="onehot"` **bit-identical
to HEAD** — that is the control.

**Step 1 — the class.** `DiscreteCausal_model.py`. *Done:* written and tested —
33/33 tensors receive gradient in both arms, K decoupled from V (K=7 with V=3,
K=3 with V=12 both work), emissions match the closed form.

**Step 2 — the count.** `Model_analysis.causal_state_count()`, sharing the merge
rule with `flower_complexity`. *Done when:* it returns 2/3 for every coin, n+1
for flower forward, and the computed backward count for the dice actually drawn.

**Step 3 — widen the gate.** `Training_model.py:459-463` becomes a registry;
`configs.py` gains `n_states`, `state_dim`, `tau`, `usage_beta`. Runners compute
`n_states` per arm and pass it. *Done when:* onehot is byte-identical on a SMOKE
run and discrete trains through the existing pipeline.

**Step 4 — the analysis surface.** `state_assignment` / `emission_table` on both
classes; `S_emp` from occupancy, dispatching on `None`. *Done when:* forward
`S_emp` equals `C⁺` to 3 dp.

**Step 5 — output layout**, resolved path and resume plan printed first.

**Step 6 — figures.** §7.1 in notebook style, §7.2 untouched.

**Step 7 — the runs.** `run_experiments` (7) → `run_statistical_trj` (7 × 30) →
`run_sweep_experiment` (181). Timing probe at `--repeats 2` first.

**Step 8 — `model_comparison.png`** and the write-up.

### Budget

Measured: the discrete head costs **0.907×** the baseline per step and adds ~21
parameters, so the budget is the existing one re-run — ~20 min, ~2 h and ~23 h
for the three experiments. Out of scope by decision: dice, λ, capacity.

---

## 9. What I am not sure about — decide these before Step 3

**9.1 Is the discrete model a ΔCE instrument at all?** At K = theory and
β = 0.01 a residual arm asymmetry of ~+0.031 bits survives, ~10× the effect. It
may be optimisation difficulty — the backward arm has more states to discover —
rather than representation. The discriminating test is convergence: run both arms
far past the budget and see whether the gap closes. **I recommend treating the
discrete model as a state-discovery and `S_emp` instrument first, and its ΔCE as
secondary until that test is done.** Tell me if you want ΔCE reported anyway.

**9.2 `usage_beta` default — 0.01 or 0.0?** 0.01 is required for the backward
arm to reliably find its states (§5) but biases `S_emp` upward by +0.024 on the
backward arm and −0.001 on the forward one — arm-dependent, in the hypothesis's
direction. 0.0 is unbiased but unstable. **I lean 0.01 with the β=0 number
reported alongside**, but this is a science call, not an implementation one.

**9.3 K = theory injects the answer.** The model can no longer *discover* how
many states the process needs. That was one of the architecture's attractions.
Worth also supporting `--n-states {theory, vocab, <int>}` so a discovery run is
possible? I would add the flag and default it to `theory`.

**9.4 K varies per cell across the sweep.** With K = theory, K is `n+1` / `m+1`
per flower cell, so K becomes a per-cell covariate **correlated with the x-axis
by construction**. Unavoidable if K = theory. It means a sweep trend cannot
separate "asymmetry" from "more states to fit". A fixed-K sweep would separate
them but reintroduces unequal slack. **I do not have a clean answer here and it
affects what the sweep can conclude.**

**9.5 Does `state_matrix` earn its place?** It has no expressive power (§4.4) and
its geometry is unidentified. Dropping it makes `emission.weight` literally the
`(K, V)` emission table. It survives only to give the scatter plot something to
draw. Keep, or drop and lose that figure?

**9.6 `merge_tol` for the flower count.** `flower_complexity` defaults to the
rounding rule and offers a tolerance. `causal_state_count` must use the *same*
rule as the `C⁻` it is paired with, or K and the theory disagree. I will pass it
through; confirm the default stays the rounding rule.

---

## 10. What will bite

1. **The backward arm may not converge at the shared budget.** Both arms must
   keep the *same* budget — that is what makes ΔCE paired — so the summary must
   report per-arm convergence and refuse to score ΔCE when either arm has not
   settled. An unconverged backward arm reads as causal asymmetry.
2. **Seed sensitivity.** At β=0 one backward seed in four collapsed. Report the
   per-repeat state count, not just the mean.
3. **Tags do not encode the model**, and `combined` is keyed by tag, so a
   discrete run into an onehot out_root would silently overwrite. Step 5 makes
   the trees disjoint; the run config must record `model`, `n_states`, `β`.
4. **Divergence interacts with the bottleneck.** The backward coin conditional
   has a point-mass row (`P(prev | 2) = [0,1,0]`); CE on a deterministic
   transition has no finite minimiser. Watch divergence per arm, not pooled.
5. **Do not migrate the existing results.** They are the onehot baseline.
6. **The mask cache is not keyed by `mode`** (inherited from `OneHot_model`, and
   documented there). Never mutate `.mode` after a forward pass.

---

## 11. Verification checklist

- [ ] `pytest tests/ -q` green after every step (74 now)
- [ ] `embed_type="onehot"` bit-identical to HEAD on a SMOKE run
- [ ] gradient reaches **all** tensors in both arms — asserted, not eyeballed
- [ ] `causal_state_count` returns 2/3 for the coin, n+1 for flower forward, and
      the computed count for the dice drawn — including the n=1 null control
- [ ] discrete forward `S_emp` equals `C⁺` to 3 dp; backward bias quoted
- [ ] discrete and onehot out_roots disjoint; neither resumes from the other
- [ ] run config records `model`, `n_states`, `state_dim`, `tau`, `usage_beta`
- [ ] per-arm convergence reported; ΔCE suppressed when either arm is unsettled
- [ ] existing `All_Results/results_*` trees byte-unchanged at the end
