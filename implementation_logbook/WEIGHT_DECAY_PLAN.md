# Weight Decay — Implementation Plan

Adds an optional weight-decay term to training, threaded through the same
config → runner → pipeline → model chain every other setting uses. Date:
2026-08-10.

Two purposes, in increasing order of value:

1. **Suppress the training divergence** that makes LARGE's cross-validation
   unusable (all 40 folds diverged) and costs SANITY 6–8 folds of 10.
2. **Provide a continuous capacity axis** so a null ΔCE can be told apart from
   "capacity absorbed the asymmetry" — the disambiguation `IMPROVEMENT_PLAN.md`
   §1.1 says the study needs before any ΔCE is interpretable.

The second is the real reason to do this. The first is a side benefit.

---

## 1. Why weight decay and not the alternatives

Divergence has a known cause: both processes contain deterministic transitions,
cross-entropy on a deterministic transition has **no finite minimiser**, so
training enlarges the logit gap indefinitely until a step overshoots. Measured
over 80 epochs: training loss flat at H∞ from step 100 to 800 while max|logit|
grows 2.9 → 8.7, peaks at 12.8, then a single step spikes the loss.

Anything that restores a finite optimum fixes this. The candidates are **not**
interchangeable, because this study measures a *difference between two arms* and
the arms carry structurally different amounts of determinism.

### The disqualifying measurement

The coin's backward conditional contains an exactly-deterministic row (token 2 is
always preceded by token 1) carrying stationary mass `pq/(p+q)` — 2.5% to 47.5%
across the sweep grid. The forward conditional contains **no** point-mass row.

Any regulariser that penalises *confidence* therefore taxes the two arms
unequally. Computed in closed form for label smoothing, where the optimum is
`p* = (1−ε)p_true + ε/V` and the reported excess over H∞ is
`Σ_i π_i KL(p_i ‖ p*_i)`:

| ε | spurious ΔCE range (bits) | corr with C⁻ − C⁺ |
|---|---|---|
| 0.01 | [−0.00383, −0.00001] | **+0.662** |
| 0.05 | [−0.01421, −0.00005] | +0.561 |
| 0.10 | [−0.02472, −0.00008] | +0.466 |

At ε = 0.01, with no causal asymmetry involved at all, you would obtain a
Spearman ρ of **+0.66** against C⁻ − C⁺ — the sweep's headline statistic — plus a
negative offset flipping individual verdicts toward `MISMATCH`. The variation
across the x-range (0.0038 bits) **exceeds the real effect** (|ΔCE| ~ 0.001–0.003).

This is structurally the same error as the fixed burn-in caught in
`run_sweep_experiment.py`: a nuisance that (a) hits the arms unequally and (b)
varies along x manufactures a slope indistinguishable from the effect.

### The classification

| method | acts in | moves CE floor | arm-symmetric | verdict |
|---|---|---|---|---|
| gradient clipping | optimiser step | **no** — loss untouched | **yes** | safe; orthogonal, do alongside |
| **AdamW weight decay** | parameter space | slightly, indirectly | mostly | **this plan** |
| dropout | architecture | slightly | yes | invalidates every checkpoint |
| logit-norm penalty | output space | yes | **no** | confounded |
| entropy / confidence penalty | output space | yes | **no** | confounded |
| label smoothing | output space | yes, computed above | **no** | **do not use** |

Rule: **parameter-space** regularisers are roughly arm-agnostic; **output-space**
regularisers penalise confidence, and confidence requirements differ between the
arms.

Weight decay is not perfectly immune — the backward arm needs a larger logit gap
for its point-mass row, so shrinking weights costs it somewhat more. The coupling
is indirect and unstructured rather than a per-token tax proportional to
determinism. §6 says how to measure what remains.

### Measured effect on divergence

From `divtest.log`, flower n=1 m=2, 60 epochs, 6 seeds each:

| configuration | diverged |
|---|---|
| MPS, Adam, lr=5e-3 (current) | 4/6 |
| **MPS, AdamW, wd=0.01** | **1/6** |
| MPS, Adam, lr=1e-3 | 0/6 |
| CPU, Adam, lr=5e-3 | 0/6 |

Weight decay is the principled entry in that table: lowering the learning rate
only delays the overshoot, whereas weight decay restores a finite optimum.

---

## 2. Backward compatibility — the three guarantees

These are what make the change safe to land before any results are re-run.

**`AdamW(weight_decay=0.0)` is bit-identical to `Adam`.** Verified: 20 steps on a
seeded Linear, `max|Δparam| = 0.000e+00`, `torch.allclose` True. So a default of
`0.0` preserves every existing number exactly.

> **Trap:** PyTorch's `AdamW` defaults to `weight_decay=0.01`, whereas `Adam`
> defaults to `0.0`. `AdamW` is **not** a drop-in — the value must be passed
> explicitly, or every existing result silently changes.

**`state_dict` is unchanged.** Adding a keyword argument to `OneHotDecoder.__init__`
adds no parameters and no buffers, so every `.pt` in `results_quick/models/`,
`results_large/models/` and `sanity_check_flower_process/models/` still loads.

**Old sidecars will not false-alarm.** `utils.check_weight_meta` compares only
keys present in the sidecar (`if k in meta`), so checkpoints written before this
change are skipped on the new field rather than reported as a mismatch.

---

## 3. The threading chain

Same pattern as `n_layers` (B11) and `val_every_n_steps` (D1): config key →
runner → pipeline → `train_model` → model. Every call site below must be updated
in the same commit, or a runner will silently train at the default while its
`run_config_*.json` records something else.

### 3.1 Definitions

| file | line | change |
|---|---|---|
| `configs.py` | `BASE` | add `weight_decay = 0.0` with a comment naming the 4/6 → 1/6 measurement |
| `OneHot_model.py` | `:125` `__init__` | add `weight_decay: float = 0.0`; store as `self.weight_decay`. `save_hyperparameters()` at `:188` records it automatically |
| `OneHot_model.py` | `:320` `configure_optimizers` | `AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)` |
| `Training_model.py` | `:420` `train_model` | add the parameter; forward to `OneHotDecoder` |
| `Training_model.py` | `:548` `train_test_val_pipeline` | add the parameter; forward to `train_model` at `:659` |

### 3.2 Call sites — all seventeen

```
Training_model.py:659         train_model      (inside train_test_val_pipeline)
run_statistical_trj.py:265    train_model      (one_repeat)
pq_experiment.py:97,108       train_model      (pq_experiment)
pq_experiment.py:429,440      train_model      (pq_experiment_full)
run_experiments.py:462,474    train_test_val_pipeline  (experiment_1)
run_experiments.py:556,566    train_test_val_pipeline  (experiment_1_2)
run_experiments.py:728,740    train_test_val_pipeline  (experiment_2)
sanity_check.py:474,485       train_test_val_pipeline  (exp_coin)
sanity_check.py:594,605       train_test_val_pipeline  (exp_flower)
```

Each passes `weight_decay=cfg["weight_decay"]` (or `cfg.get("weight_decay", 0.0)`
in `sanity_check.py`, whose `CFG` is a local literal rather than a `configs.py`
entry — add the key there too).

`run_sweep_experiment.py` needs **no change**: it calls `one_repeat`, which reads
`cfg`, so the value flows through once `run_statistical_trj.py:265` forwards it.

### 3.3 Provenance

`run_experiments.weight_meta:142` must record `weight_decay` **and**
`optimizer` (`"adamw"`). Without both, a checkpoint trained at λ=0.03 is
indistinguishable from one trained at λ=0 — exactly the A4 failure the sidecar
exists to prevent, and worse here because the difference is invisible in the
architecture.

Both `run_config_*.json` writers already splat `**cfg`, so they pick the key up
for free.

### 3.4 Decision: which parameters get decayed

Two options.

- **Decay everything** (recommended to start). Simplest, and it is what the
  4/6 → 1/6 measurement used, so the plan's one quantitative claim about
  effectiveness stays valid. `rand_prj` is a buffer, not a parameter, so it is
  excluded automatically and "onehot" stays a fixed random projection.
- **Exclude LayerNorm gains and biases** via two parameter groups — standard
  practice in large-scale LM training. Defensible, but it is an optimisation
  heuristic imported from a different regime, and it would invalidate the
  measured divergence rate above.

For a capacity-bounding *axis*, decaying everything is also the cleaner knob: it
shrinks the whole function class rather than a hand-chosen subset. Record the
choice in the sidecar if the second option is ever taken.

---

## 4. Tests

Added to `tests/test_theory.py`, which currently holds 61 passing tests.

1. **`test_adamw_zero_decay_equals_adam`** — 20 seeded steps, assert
   `torch.allclose` on every parameter. This is the test that licenses the claim
   in §2.
2. **`test_weight_decay_defaults_to_zero`** — `configs.BASE["weight_decay"] == 0.0`
   and `OneHotDecoder().weight_decay == 0.0`. Guards against someone "helpfully"
   adopting PyTorch's 0.01 default.
3. **`test_weight_decay_reaches_the_optimiser`** — build with
   `weight_decay=0.03`, call `configure_optimizers()`, assert
   `opt.param_groups[0]["weight_decay"] == 0.03`. The threading bug this whole
   plan risks is a value that is accepted and then dropped; `n_layers` had
   exactly that bug (B11) and it went unnoticed because both configs happened to
   agree.
4. **`test_old_checkpoint_still_loads`** — instantiate at the sidecar's
   architecture and `load_state_dict` an existing `results_quick/models/*.pt`.
   Skip if absent, so the suite still runs on a clean clone.
5. **Regression, CPU only** — a short `train_model` run at `weight_decay=0.0`,
   `accelerator="cpu"`, must reproduce its own stored `step_loss` to 1e-9. CPU
   because MPS is not bit-reproducible across runs.

---

## 5. The experiment this enables

Do **not** land weight decay as a bug fix and then leave it at one value. Land it
as an axis.

`IMPROVEMENT_PLAN.md` §1.1: ΔCE is a difference of *residuals*, so an
unbounded-capacity converged predictor gives ΔCE = 0 whatever C⁻ − C⁺ is, and a
null is therefore ambiguous until capacity is varied. The plan names a `d_model`
sweep. **A λ sweep is the same experiment, better instrumented:**

| | `d_model` sweep | λ sweep |
|---|---|---|
| resolution | discrete, coarse (16, 32, 64) | continuous |
| changes parameter count | yes | no |
| changes the input code size | yes | no |
| invalidates checkpoints | yes | no |
| cost to implement | new architecture handling | one config key |

Prediction, and it is falsifiable:

```
λ = 0          unbounded          both arms reach H∞      ΔCE ≈ 0    (observed today)
λ increasing   bound tightens     residuals reappear      |ΔCE| grows, sign tracks C⁻ − C⁺
λ large        both underfit      optimisation dominates  ΔCE swamped
```

An inverted-U in |ΔCE| against λ whose peak *sign* matches C⁻ − C⁺ is far
stronger than anything a single λ can give. A flat zero across every λ would be
genuine evidence **against** the hypothesis — which the current null cannot
claim.

**Grid:** λ ∈ {0, 0.001, 0.01, 0.03, 0.1, 0.3}.
**Processes:** the **flower** grid. Two reasons — it spans both signs of
C⁻ − C⁺, and (§1) it has no exactly-deterministic row, so the determinism
confound is far weaker there than on the coin.
**Repeats:** 20, per the sweep plan's budget reasoning.

Reuses `one_repeat` untouched; the harness needs only a loop over λ writing to
`results_wd/<lambda>/`.

---

## 6. What will bite

1. **The convergence filter breaks at λ > 0.** `paired_delta_ce` calls a repeat
   converged when both arms land within `conv_tol = 0.10` bits of H∞, and
   `run_sweep_experiment` uses the same `CONV_TOL`. Weight decay raises the
   achievable CE above H∞ by some δ(λ). Once δ(λ) > 0.10 **every repeat is
   flagged non-converged**, the converged-only statistic returns `nan`, and — via
   the `np.isfinite(sem)` fallthrough in `_verdict` — a confident `match` /
   `MISMATCH` verdict gets printed from a single fold with no uncertainty at all.
   *Fix before running:* measure δ(λ) with a 3-repeat probe at each λ, then either
   widen `conv_tol` per λ or switch the reference from H∞ to the minimum CE
   actually achieved at that λ. Do not discover this after ten hours.
2. **The residual weight-decay asymmetry.** The backward arm needs a larger logit
   gap for its point-mass row, so it pays more decay. Measure it rather than
   assume it away: the flower `n = m` diagonal sits at C⁻ − C⁺ ≈ −0.04, so
   ΔCE(λ) there **is** the artifact. Report intercept and slope separately, as
   `run_sweep_experiment` already does for the area.
3. **H∞ lines in every figure become wrong-ish.** `plot_loss_theory`,
   `_draw_final_loss` and the trajectory panels all draw `axhline(theory)`. At
   λ > 0 the curves settle above it by δ(λ). Not incorrect — H∞ is still the
   process entropy rate — but the figures should say so, or a reader will read
   the offset as a failure to converge.
4. **`settled_from` is unaffected.** It measures a plateau relative to the
   curve's own final value, not to H∞, so a shifted floor changes nothing there.
5. **Every published number becomes conditional on λ = 0.** The existing QUICK
   results, the controls and the baseline harness were all produced with plain
   Adam. Keeping the default at `0.0` means they stay valid and comparable; a
   λ > 0 run is a *new* condition, not a correction of the old one. Say so
   wherever both appear.

---

## 7. Verification checklist

- [ ] `pytest tests/ -q` — 61 existing plus the 5 new all pass
- [ ] `AdamW(wd=0)` regression is bit-identical on CPU
- [ ] `python run_experiments.py --config SMOKE` completes unchanged
- [ ] an existing `results_quick/models/*.pt` still loads
- [ ] `weight_decay` and `optimizer` appear in a freshly written sidecar
- [ ] a non-zero λ actually reaches `opt.param_groups[0]["weight_decay"]`
- [ ] δ(λ) measured at every λ in the grid **before** the long run
- [ ] `conv_tol` recalibrated, or the reference switched, per §6.1
