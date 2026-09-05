# Discrete Decoder — Working Parameters, Config Changes, and Runthrough

Plan only. Nothing here has been applied.

Covers `All_Results/discrete/*`, the config that produced them, and what to
change before re-running against the current
`Transformer_model/DiscreteCausal_model.py`.

---

## 0a. The regulariser changed — what that invalidates

`DiscreteCausal_model.usage_penalty` is now:

```python
p_bar   = occ.reshape(-1, n_states).mean(dim=0)      # occupancy over ALL scored tokens
H_usage = -(p_bar * log2(p_bar)).sum()               # = S_emp, the empirical C
return H_usage, usage_beta * H_usage                 # penalty = beta * S_emp
```

`log2(K) − H(p_bar)` is gone. The KL-against-`target_occupancy` branch is gone.
The objective is now **`CE + β · S_emp`** — a minimum-complexity prior, which is
the ε-machine's own trade-off (predictive optimality at least complexity).

**Everything in `configs.py` that tunes β from the uniformity gap is now dead.**

| key / function | status | why |
|---|---|---|
| `usage_target` (`configs.py:73`) | **dead** | only decides whether to pass `target_occupancy`, which `usage_penalty` never reads |
| `target_occupancy` | **inert** | accepted at `DiscreteCausal_model.py:81`, buffered at `:109`, threaded through `Training_model.py:439/494/609/723`, set by `run_experiments.py:507` and `run_statistical_trj.py:355/385` — and used by nothing |
| `usage_beta_gap/high/low` | **dead** | inputs to `usage_beta_shared` |
| `usage_beta_shared` (`Model_analysis.py:947`) | **meaningless** | derives β from `log2(K) − C`, the uniformity gap of the *old* penalty. Called at `run_statistical_trj.py:221, 279`, so the trajectory runner is currently choosing β from a quantity that no longer appears in the objective |

**Decision, per instruction: β is fixed at 1/N everywhere**, where N is the number
of scored tokens the cross-entropy averages over — `N = B · T`. β sweeps sweep
*around* 1/N rather than around 0.01.

**N measured, not assumed.** Building the real loader and reading the batches:

```
config: train_chunk_len=256  coin_batch=32  coin_num_samples=500  coin_seq_len=2000
  CoinDataset item: input (1999,) target (1999,)        <- shift happens FIRST
  batches/epoch = 16
    batch 0..14: B=32  T=256  N = B*T = 8192   1/N = 1.2207e-4
    batch 15   : B=20  T=256  N = B*T = 5120   1/N = 1.9531e-4
```

**T = 256, not 255.** `ChunckDataset.__getitem__` slices the *already-shifted*
`(input, target)` pair, so the chunk length is T. Therefore

```
N = 32 x 256 = 8192        beta = 1/N = 1.2207e-4
```

For scale: the retired per-arm rule emits 0.01 (82x) or 0.2 (1638x). The measured
collapse cliff at lr=1e-3 is ~6e-4, so **1/N sits ~5x below the cliff** — inside
the safe band, with less margin than is comfortable (§1.4).

**Open decision — nominal or per-batch?** The final batch of each epoch is
partial (B=20 here), so a literal per-step `1/(B·T)` is 1.95e-4 on 1 step in 16
and 1.22e-4 on the other 15. Both readings are defensible:

* **nominal** `1/(batch · chunk_len)` — one constant, recorded once, reproducible;
* **per-batch** `1/(B·T)` from the actual tensor — the literal reading of "1/N
  where N is the tokens scored at this step".

Here the difference is 1.6x on one step in sixteen and both values sit below the
cliff, so it likely does not matter — but it did matter at T=99, where the
remainder batch's 1/N was 8.4e-4, *past* the cliff. **This needs a decision
before implementation; it is not safe to assume.**

### Validation of beta = 1/N against the retired values

Small-case test at the real geometry (d_model=32, lr=1e-3, wd=0.01, tau=1.0,
150 epochs, K = the exact theoretical count, seeded loader):

```
    process       arm   K |  beta=1/N            |  beta=0.01           |  beta=0.2
                          | CE-Hinf  k   ARI     | CE-Hinf  k   ARI     | CE-Hinf  k   ARI
       coin   forward   2 | +0.3867  1  0.000    | +0.3867  1  0.000    | +0.3867  1  0.000
       coin  backward   3 | +0.2583  2  0.156    | +0.3869  1  0.000    | +0.3869  1  0.000
  flower2,3   forward   3 | +0.0167  3  1.000    | +0.2697  2  0.750    | +1.1907  1  0.000
  flower2,3  backward   4 | +0.2887  2  0.691    | +1.1907  1  0.000    | +1.1907  1  0.000
```

**beta = 1/N wins or ties in every cell**, and `flower(2,3)` forward is exact:
ARI 1.000, S_emp 1.5002 against C+ = 1.5000. The retired values collapse the
bottleneck to one state almost everywhere — which is expected, since under the
NEW penalty `beta * S_emp` a large beta is a large pressure toward *fewer*
states, the opposite of what those values were tuned for.

The coin failures in this table are **budget, not beta**: 1500 steps, where the
K=2 forward arm was previously measured to need ~2700. Longer-budget results
supersede this table.

### Sections superseded by this change

- §1.1 — drop `usage_target` from the keep-list.
- §1.2 — the per-arm β rule is retired, not kept. `sweep_beta001` / `sweep_beta020`
  / `sweep_betarule` were an ablation of a penalty that no longer exists; they
  become historical record, and the β sweep is re-specified in §2.2.
- §1.4 row 1 stands and is now the *governing* rule: β is calibrated against lr,
  and `β_cliff · lr ≈ 6e-7`. Because β is now pinned to 1/N, **any change to lr,
  batch size or chunk length moves β relative to the cliff** and must be
  re-checked.

---

## 0. Two sources of evidence, and which one wins

Numbers in this document come from two places and they do **not** carry equal
weight:

| source | harness | status |
|---|---|---|
| **Repo-measured** — the comments in `configs.py` BASE and DISCRETE | real: d_model=32, 5-fold CV, full QUICK data, `conv_tol` | **authoritative** |
| **Scratchpad** — the β / K / lr / step sweeps run in this session | simplified: d_model=20, single split, 300×100 tokens, no folds, no weight decay | **candidate hypotheses only** |

Where they disagree, the repo wins and the scratchpad result becomes a test to
run, not a change to make. The clearest disagreement is §1.3.

---

## 1. Working parameters

### 1.1 Settled, keep as-is

| parameter | value | why |
|---|---|---|
| `embed_type` | `"discrete"` | selects this architecture |
| `lr` | `1e-3` | measured: at 1e-2 the forward coin arm stays collapsed on 1 of 2 states after 1500 steps, +0.573 above H∞. At 1e-3 **both arms converge together** (+0.087 / +0.088), which is what keeps ΔCE paired |
| `coin_max_epochs`, `flower_max_epochs` | `150` | a **ceiling, not a floor**. CE on a deterministic transition has no finite minimiser, so the backward coin arm converges and then *diverges past* it: held-out CE−H∞ goes +0.0231 (75) → +0.0222 (100) → +0.0621 (150) → +0.2006 (300) at wd=0 |
| `weight_decay` | `0.01` | AdamW restores a finite optimum on exactly the deterministic transitions that cause the above. At 150 epochs with wd=0.01 every arm of both families is inside `conv_tol=0.1` |
| `tau` | `1.0` | measured: τ ≤ 0.2 collapses the bottleneck. Confirmed independently in the scratchpad (τ=0.2 → 1 state; τ=1.0 → 2 states) |
| `d_model` | `32` | inherited from QUICK; untested for this architecture (§4.1) |
| `state_dim` | `None` → K | carries no expressive power; `state_matrix ∘ emission` composes to one K×V map |
| `usage_target` | `"uniform"` | `"theory"` makes S_emp **circular** — penalising the model for not having the occupancy whose entropy is C installs S_emp ≈ C rather than measuring it |

### 1.2 Settled, but the value is a live experiment

`usage_beta` — the per-arm rule `usage_beta_shared(...)`, keyed on the
uniformity gap `log2(K) − C`:

```
usage_beta_gap  = 0.1     threshold on log2(K) - C
usage_beta_high = 0.2     gap <= threshold: uniform is nearly the truth
usage_beta_low  = 0.01    gap >  threshold: uniform is wrong, tread lightly
```

Repo-measured: the rule puts **14/14** arm-process combinations inside
`conv_tol`; a fixed β of 0.01 or 0.2 puts 13/14, failing on *different*
processes. That is why `sweep_beta001`, `sweep_beta020` and `sweep_betarule`
exist as three separate result trees — they are the ablation, and all three
should be kept.

Known cost, and it must be reported with any result: β biases S_emp
**arm-dependently** — forward −0.001, backward +0.024, roughly 6% of the true
C⁻−C⁺ gap and *in the direction the hypothesis predicts*.

### 1.3 The one place the scratchpad disagrees with the config

`n_states = causal_state_count(process, mode, ...)` — the **exact** theoretical
count.

Repo-measured: at K=3, coin backward, β=0.01 finds all 3 states every seed at
+0.036 ± 0.003. Works.

Scratchpad-measured: K = the exact count is precisely the value that fails —
coin backward at K=3 never opens its third state at any β tried, while K=4
reaches ARI 0.980 and K=16 reaches ARI 1.000 with `H_occ` matching C⁻ to 0.01
bits.

These are compatible: the repo runs β=0.01, which is ~30× the scratchpad's
β=1/N, at d_model=32 with weight decay and 5 folds. A larger β is exactly what
suppresses the over-splitting that made slack necessary in the scratchpad.

**So do not change `n_states`. Test it instead** — §3, run E2. The proposal is a
`state_slack` config key (default 0, i.e. current behaviour) so that
`K = causal_state_count(...) + state_slack` can be swept without editing any
runner.

### 1.4 What the scratchpad establishes that the config does not yet encode

| finding | evidence | proposed config expression |
|---|---|---|
| **β does not scale with N; it scales with 1/lr.** Energetic prediction for the collapse cliff is 670× too high; measured `β_cliff·lr ≈ 6e-7` at two lr values | coin fw, β cliff in (5e-4,7e-4) at lr=1e-3 and (1e-4,3e-4) at lr=3e-3, with β=0 controls confirming lr=3e-3 is not itself broken | record β as an absolute number in the run config (already done); **add a comment forbidding a `1/N` formula**, and re-calibrate if `lr` or the budget changes |
| ~~Evaluation perturbs training~~ — **CORRECTED, the training path is already seeded** | `make_chunked_loader:150` passes `generator=torch.Generator().manual_seed(seed)`; the fold loaders at `:690` use `manual_seed(seed + fold)`. Verified on the live loader: `ld.generator is not None` → True | **no change needed.** The unseeded loader is `Data_generation.make_loader`, used by `pq_experiment.py` and by my scratchpad harness — *not* by the discrete training path. My earlier "highest-priority fix" was a defect in my own test code, not in the repo |
| **State count and state quality move in opposite directions with training.** k_found rises to and past the truth while ARI falls; CE falls monotonically throughout and gives no warning | 4 probes to 30k steps: ARI peaks at 3–6k then declines; CE ends *below* H∞ in all four | nothing to change — but it is the reason `coin_max_epochs=150` is a ceiling, and it means **`conv_tol` on CE is not a state-recovery criterion** |
| **No reliable stopping rule exists.** Mean ARI lost: CE≈H∞+0.10 → 0.071 (worst 0.545); `H(state\|token)`>0.01 → 0.217; k stable → 0.103 (worst 0.222) | 8 live runs | add `H(state\|token)` to `causal_state_report` as a **reported diagnostic**, not a criterion (§4.2) |

---

## 2. Proposed config changes, per result folder

### 2.1 Folders to retire, not re-run

| folder | why |
|---|---|
| `quick_shortbudget230` | 230 epochs, wd=0.0 — the superseded budget. Its own failure is what motivated 150/0.01 |
| `sweep_shortbudget230` | same |
| `trajectories_shortbudget230` | same |
| `quick_pre08161545` | predates `usage_target` / `usage_beta_*`; its config cannot be reproduced by the current code |

Keep them on disk as the record of *why* the current settings exist — the
230-epoch divergence table in `configs.py` is derived from them — but do not
re-run and do not include them in any comparison.

### 2.2 Folders to re-run, and what changes

All four keep `embed_type=discrete, lr=1e-3, tau=1.0, weight_decay=0.01,
d_model=32, n_layers=2, epochs=150, usage_target=uniform, n_folds=5`.

| folder | keep | change | add |
|---|---|---|---|
| `quick` | geometry, lr, epochs, wd | **β → 1/N** | `state_slack=0` recorded explicitly |
| `trajectories` | `repeats=30` | **β → 1/N** | report `H(state\|token)` per arm |
| `sweep_beta001` | `repeats=5` | **retire** — ablation of the old penalty | superseded by `sweep_beta_1overN` |
| `sweep_beta020` | `repeats=5` | **retire** — same | — |
| `sweep_betarule` | `repeats=5` | **retire** — the rule is meaningless now (§0a) | — |

New β sweep replacing the three above, centred on 1/N rather than on 0.01:

| folder | `usage_beta` | ratio to 1/N |
|---|---|---|
| `sweep_beta_0` | 0 | — (unpenalised reference; S_emp here is the unbiased measurement) |
| `sweep_beta_0p3N` | 4.1e-5 | 0.33× |
| `sweep_beta_1N` | **1.2255e-4** | **1× (the default)** |
| `sweep_beta_3N` | 3.7e-4 | 3× |
| `sweep_beta_10N` | 1.2e-3 | 10× — expected to be **past the cliff**, included so the cliff is bracketed rather than assumed |

Five points spanning 0 → 10×, log-spaced around 1/N. The top point is deliberately
past the measured cliff so the sweep locates it instead of relying on the T=99
scratchpad number.

### 2.3 New config keys proposed

```python
# configs.py BASE
usage_beta    = None,   # None -> 1/(batch * (train_chunk_len - 1)), computed by
                        # the runner from the geometry actually in use.  An
                        # explicit float overrides, which is how the sweep works.
                        # NOT a fixed literal: N changes with batch and chunk,
                        # and beta must move with it.  The resolved value is
                        # written to run_config so the run is reproducible.
state_slack   = 0,      # K = causal_state_count(...) + state_slack.  0 is the
                        # current behaviour.  Exists so the exact-count-vs-slack
                        # question (section 1.3) is a sweep, not an edit.
report_hcond  = True,   # add H(state|token) to causal_state_report.  0 iff the
                        # assignment is a pure function of the current token,
                        # which the truth is for both processes in both arms.
```

### 2.4 Keys to delete

`usage_target`, `usage_beta_gap`, `usage_beta_high`, `usage_beta_low`, and the
`target_occupancy` threading through `Training_model.py`, `run_experiments.py`
and `run_statistical_trj.py`. `usage_beta_shared` in `Model_analysis.py` goes
with them. Leaving them in place is worse than removing them: `run_statistical_trj`
currently *calls* the dead rule and writes its output to `run_config`, so the
provenance records a β that was derived from a penalty the model no longer uses.

---

## 3. Runthrough

Ordered so that each step's result can invalidate the next — cheapest and most
decisive first. Budgets assume the measured ~48 s per 300-epoch flower arm at
d_model=20; d_model=32 with 5 folds is roughly 4× that, so these are estimates
and step E0 exists to replace them with measurements.

### E0 — reproducibility and timing baseline (no science, ~20 min)

```bash
python Experimental_setup/run_statistical_trj.py --config DISCRETE \
       --only traj_coin_p030_q040 --repeats 3 --accelerator cpu \
       --out-root All_Results/discrete/_e0_check
```

Run twice. **Done when:** the two runs are bit-identical (the training loaders
are already seeded — §1.4 — so this should pass; it is a regression check, not a
fix). Also time one repeat, to replace every estimate below.

*Blocking on timing only.* Every budget in this section is extrapolated from a
d_model=20 harness and needs replacing with a measurement.

### E1 — regenerate the five current trees (~4–6 h)

```bash
python Experimental_setup/run_experiments.py     --config DISCRETE --accelerator cpu
python Experimental_setup/run_statistical_trj.py --config DISCRETE --repeats 30
python Experimental_setup/run_sweep_experiment.py --config DISCRETE --repeats 5 \
       --usage-beta 0.01 --out-root All_Results/discrete/sweep_beta001
python Experimental_setup/run_sweep_experiment.py --config DISCRETE --repeats 5 \
       --usage-beta 0.2  --out-root All_Results/discrete/sweep_beta020
python Experimental_setup/run_sweep_experiment.py --config DISCRETE --repeats 5 \
       --out-root All_Results/discrete/sweep_betarule
```

**Done when:** every arm inside `conv_tol=0.1`, and the β ablation reproduces
14/14 for the rule against 13/14 for each fixed β. If it does not reproduce,
the seeded loader has changed the result and §1.2 needs re-measuring.

### E2 — the state-slack question (~2 h)

`state_slack ∈ {0, 1, 2}` on the seven baseline processes, β at the rule.
Settles §1.3 on the real harness.

**Done when:** we can say whether K = exact count is adequate at β=0.01, or
whether the scratchpad's slack requirement survives d_model=32 and weight decay.
Report `n_states_used`, `S_emp_states`, and `H(state|token)` per arm.

### E3 — the capacity axis (~8 h, only if E2 leaves a deficit)

`d_model ∈ {16, 32, 64}`. **This is the only completely untested axis.** Every
scratchpad result is at d_model=20 and every repo result at 32. The scratchpad
found processes needing ≥4 states are never fully recovered — whether that is
capacity is unknown.

Guard: the runners already refuse unless `out_root` names the capacity.

### E4 — dice and capacity runners under DISCRETE (blocked, see §4.1)

`run_dice_experiment.py` cannot express β and produces no state figures. Do not
run it under DISCRETE until §4.1 item 2 is resolved.

---

## 4. Sanity check against the current `DiscreteCausal_model.py`

### 4.1 Findings

**1. `run_sweep_experiment.py` plots the wrong S_emp.** Line 480 reads
`r["fw"]["S_emp"]` — the k-means clustering estimate — while `one_repeat`
stores the discrete one as `S_emp_states` (`run_statistical_trj.py:436`). The
axis note at `:903` still says *"S_emp is k-means at an assumed k, so S ≤
log2(k)"*, which is **false for a discrete run**. So the sweep's S_emp panel
silently reports the continuous estimator even in discrete mode.
*Fix: prefer `S_emp_states` when present, and make the caption conditional.*

**2. `run_dice_experiment.py` is half-wired.** It has `--config` and `--d-model`
but **no `--usage-beta`**, and does not import `DiscreteCausal_analysis`. It
calls `one_repeat`, so `S_emp_states` and the emission tables *are* captured in
the pickle — but β cannot be overridden and no state figure is produced.
*Fix: add `--usage-beta` and the two figure calls, mirroring
`run_statistical_trj.py:1411`.*

**3. `sanity_check.py` and `pq_experiment.py` have no `--config` at all**, so
neither can exercise the discrete path. `sanity_check.py` is the null/positive
control — it is the one place a discrete regression would be caught cheaply.
*Fix: add `--config`, or document that they are onehot-only.*

**4. `analyse_capacity.py` / `analyse_lambda_sweep.py` never reference the
discrete keys.** They read the sweep pickles, so they will run, but they will
report `S_emp` (clustered) rather than `S_emp_states` — same defect as item 1.

**5. Correctly wired, no action:** `run_experiments.py` and
`run_statistical_trj.py` both import `causal_state_report`,
`plot_causal_states`, `plot_causal_states_pair`, call them unconditionally, and
store the K×V emission table and K×S vectors as plain lists so `--plots-only`
works from the pickle alone.

**6. `causal_state_report` handles the arm asymmetry correctly** — `min_pos`
trims the *start* for forward and the *finish* for backward, and the backward
model is scored on the swapped batch. Both are easy to get wrong and both are
right.

### 4.2 Proposed addition

`causal_state_report` returns occupancy, vectors and emissions but **not**
`H(state|token)`, which is the most diagnostic quantity found in this session:
0.0000 in every successful run, 0.6–1.2 in every failure, and computable with no
ground truth. Adding it costs one `bincount` per token in the existing loop.

Also absent: the **state→state transition matrix**. `emission_table()` is K×V
(what a state emits), not K×K (where a state goes). Everything needed is in
`last_states`; cross-tabulating consecutive positions gives the other half of
the ε-machine, which is the object comparable to Thompson's construction.

---

## 5. Figures — minimal without losing clarity

Reviewing `plot_causal_states` (`DiscreteCausal_analysis.py:111`). The panels
are right; the labelling is roughly 2× longer than it needs to be, and three
elements are **duplicated across panels**.

### 5.1 Remove — redundant, not terse

| element | line | why it can go |
|---|---|---|
| Panel 2 colorbar (“Token positions”) | 175–177 | encodes count, which panel 1 already shows as bars *and* prints as numbers, and which panel 2 also prints under each point. Third encoding of one variable |
| Panel 2 per-point count `s0\n12,345` | 179 | same count again. Keep `s0` only |
| Panel 3 colorbar (“Probability”) | 213–215 | the value is printed in every cell and the scale is fixed 0–1 |
| Panel 3 `n=` in the y-labels | 202 | count, for the third time |
| Panel 3 xlabel “Next token” | 211 | the title already says `P(next token | state)` |

That removes two colorbars and two label lines, and widens the actual data.

### 5.2 Shorten

| now | proposed |
|---|---|
| `Occupancy — 3 of 3 states used` | `Occupancy · 3/3 used` |
| ylabel `Token positions assigned` | `Positions` |
| `PCA component 1` / `2` | `PC1` / `PC2` |
| `s0  n=12,345\nTV=0.003` | `s0` / `TV 0.003` |
| suptitle `traj_coin_p030_q040 — backward  [arm]   S_emp = 1.3421 bits   (K=3, beta=0.01)` | `coin p030 q040 · backward · S_emp 1.342 (K=3, β=0.01)` |
| footnote *“state-vector geometry is identified only up to an invertible K x K map (plan 4.4); the emission table is the identified view”* | `geometry identified only up to a K×K map` — the caveat matters, the citation does not belong on the axes |

### 5.3 Keep — these are clarity, not decoration

- **Grey bars / hollow markers for unused states.** The distinction between
  "K states exist" and "k are used" is the headline of the whole architecture.
- **Values printed in the emission cells.** They are what gets compared to the
  closed form; a colour alone cannot be read to 3 decimals.
- **`TV=` per row.** The learned-vs-theory number, and the only quantitative
  claim on the figure.
- **`constrained_layout=True`.** Panel widths are data-dependent; fixed spacing
  collides on some inputs.

### 5.4 One addition worth its ink

The theory occupancy is currently invisible. Overlaying it on panel 1 as a thin
rule per bar turns "3 of 3 used" into "3 of 3 used, **and in the right
proportions**" — which is the actual claim, and is otherwise only checkable by
reading `S_emp` against C in the title.

---

## 6. Order of work

1. **Decide nominal vs per-batch beta** (§0a) — blocking, and a question for the
   author rather than something to assume.
2. **Delete the dead beta machinery** (§2.4) — `usage_target`, `usage_beta_*`,
   `usage_beta_shared`, `target_occupancy`. Until this is done,
   `run_statistical_trj` writes a beta into provenance that the model never used.
3. **Sweep S_emp fix** (§4.1 item 1) — currently reports the wrong estimator.
4. **Figure trim** (§5) — cheap, no effect on numbers.
5. **`H(state|token)` in the report** (§4.2).
6. **E1** regenerate the trees at beta = 1/N.
7. **E2** state slack.
8. `run_dice_experiment` wiring (§4.1 item 2), then E3/E4.

Items 3–5 change no measured value. Items 1–2 change beta everywhere, which is
the point. Item 6 onward is compute.
