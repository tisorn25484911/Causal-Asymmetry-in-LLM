# Phase 3 (+ C6) — Handoff

Date: 2026-08-05
Branch: `phase0-unblock` (6 new commits on top of Phase 1/2, **not pushed**)
Follows: `IMPROVEMENT_PLAN.md` §7 Phase 3, plus C6 from Tier C
Previous: `PHASE1_2_HANDOFF.md`, `PHASE0_HANDOFF.md`

**Status: C6 and all of Phase 3 complete, 52 tests passing.** SANITY and QUICK
are complete on this code; LARGE is running and its cross-validation results
are dominated by training divergence — see §7.

---

## 0. Read this first

Two findings this phase, both of which change how you should read the results:

1. **The UMAP figures in every previous phase showed one trajectory, not a
   distribution.** C6 is fixed; all latent figures are regenerated.
2. **Training diverges on a large fraction of folds in the long configs.** Not
   a bad local optimum — folds *reach* H∞ and then blow up to CE 15–58 bits.
   Measured 4/6 seeds on MPS at sanity_check's settings, 0/6 on CPU. You chose
   to leave training unchanged and report it, so the code now detects and
   prints it. **SANITY and LARGE paired statistics are weak because of this.
   QUICK is unaffected.**

---

## 1. Decisions you made this phase

| Decision | Chosen | Consequence |
|---|---|---|
| C6 sampling | Both panels side by side | Every per-model UMAP is now 2 panels: one position per sequence (matching `S_emp`) and random across sequences+positions. 2× UMAP cost, 2× panels to read. |
| D1 cadence | Split train and val cadences | Full-resolution train curve kept (free) while validation is sub-sampled. Measured 3.06× on a LARGE-shaped fold. Deviates from the plan's literal `record_every_n_steps=25`, which would have left QUICK with 4 points per curve. |
| Training instability | Change nothing, report it | No CE bias introduced and all numbers stay comparable across phases, but SANITY/LARGE keep only 1–2 usable folds, so their converged-only `sem` is unreliable and can be `nan`. |
| LARGE | Run in full | ~10 h, of which ~8 h is the pq sweep that D1 cannot speed up. |

---

## 2. The commits

### `a0d9761` — C6: UMAPs sample across sequences

`plot_umap` embedded `latents.reshape(-1, d)[:n_pts]`. The reshape is
row-major, so those first rows are **every position of the first ⌈n_pts/T⌉
sequences** — 4 sequences at chunk 256, and at T=999 in sanity_check with
`n_pts=500`, a *single* sequence. Verified on tagged latents (500 seqs × 256
positions):

| mode | points | distinct sequences |
|---|---|---|
| `prefix` (old) | 1000 | **4** |
| `per_sequence` | 500 | **500** |
| `random` | 1000 | **426** |

Two panels are now drawn because each is misleading alone:

- **`per_sequence`** — one position per sequence at the max-context position
  for the direction (`last` for tril, `first` for triu). This is the exact
  slice `statistical_complexity_empirical` clusters, so the picture and the
  `S_emp` number cannot tell different stories. All points share a context
  length, so spread is structure.
- **`random`** — uniform without replacement over (sequence, position) across
  all sequences, excluding a **direction-aware** burn-in: the first `burn_in`
  positions for a forward model (context grows left-to-right) and the *last*
  `burn_in` for a backward one. Shows much more of the distribution but mixes
  context lengths.

Both panels are labelled with what they sampled. `mode="prefix"` is retained
so the notebook can *show* the old behaviour rather than assert it.
`sanity_check.py` had its own copy of the bug in `plot_umap_tokens`; it now
delegates. `compare_fw_bw` in both runners sampled the same way and now takes
each arm at its own max-context position.

### `d4720f2` — Phase 3.1: D1, split the two cadences

The plan says `record_every_n_steps=25`. Applied literally that throws away 96%
of a series that costs **nothing** — the training loss is already computed and
returned by `training_step` — to save the expensive one. QUICK has 100 steps
per fold, so it would have produced **four points on every curve**.

So the cadences are separate now: `record_every_n_steps=1` for the free train
series, `val_every_n_steps=25` for the validation pass.

| | time | train pts | val pts |
|---|---|---|---|
| val every step | 14.9 s | 80 | 80 |
| val every 25 | 4.9 s | 80 | 3 |

**3.06×** on a LARGE-shaped fold. Standalone overhead measurements of
validating every step vs not validating at all: **1.63×** at QUICK sizes,
**2.58×** at LARGE sizes — the plan's 2–4× estimate is right for LARGE and high
for QUICK.

**`pq_experiment_full` calls `train_model` without a `val_loader`, so the ~8 h
sweep never paid this cost and D1 does not speed it up.** The saving applies to
the four CV blocks only. End to end: QUICK 10:24 → 8:44, SANITY 12.7 → 6.9 min.

Val points are now sparser than train points, so they need their own x-axis:
the recorder stores `step_at`/`step_val_at`, every plot that draws a val series
uses it, and both are carried into the slim pickle. Plotting against the list
index would have compressed the run into the first few percent of the axis.

### `b4c3cbc` — Phase 3.2: D2, vectorise the double-loops

`perplexity_ind_CE` and `stepwise_kl_coin` ran `for b: for t:` in pure Python
over every token — 500 × 1999 ≈ 10⁶ iterations per call, four calls per
experiment. `heatmap_theory` double-looped a 100×100 grid to evaluate a closed
form.

| | before | after | speedup | max abs diff |
|---|---|---|---|---|
| `heatmap_theory` (100×100) | 30 ms | 0.4 ms | 83× | **0** |
| soft-label CE (8×500) | 11 ms | 0.15 ms | 78× | 1.8e-11 |
| per-token KL (8×500) | — | — | — | **0**, counts equal |

Two are bit-identical; the CE differs by float summation order over 4000 terms.
Three tests pin the equivalence against explicit reimplementations of the loops
that were replaced.

### `c230d28` — Phase 3.3: D4, the cheap wins

- **Mask caching.** The tril/triu mask was rebuilt on every forward pass though
  it depends only on `(T, device, mode)`. Cached in a plain dict, deliberately
  *not* a buffer, so it stays out of `state_dict`. The test also asserts tril
  and triu are exact mirror images with equal total context — the C2 point that
  the **mask is not** the forward/backward confound.
- **`perplexity_calculation`** appended every batch's logits and `torch.cat`ed
  the lot before computing one loss (80 MB held on CPU for the flower config).
  Now accumulates CE incrementally, token-weighted — arithmetically the same
  mean. Agrees with the old path to 4e-07. Also fixes the `pad_id` branch,
  which flattened to `(n_keep, V)` and handed that to a function expecting
  `(B, T, V)`.
- **UMAP warm-up** ran at *import* of `Model_analysis`, again in the runner and
  again in `sanity_check` — three JIT compiles per process, one of them on 20
  points with `n_neighbors=200`, which umap clamps to 19 while warning on every
  run. Now a lazy idempotent `warm_up_umap()` called once from each `main()`.
  The spurious warning is gone.
- Removed `sanity_check._project2d`, a fourth duplicate (C4).

The other D4 items were already done in Phase 1: `F.cross_entropy` (via B3),
`no_grad` around the attention plots, the local `PCA`/`KMeans` re-imports.

### `33a2511` — Report training divergence

See §3. Detects the reach-then-blow-up shape, warns per fold, counts them in
the CV summary, and stores the diagnosis in the results bundle.

---

## 3. The training instability — the important caveat

Investigating why only 1 of 5 folds survived the convergence filter turned up
something worse than stuck optimisation. Flower null control, forward arm:

| fold | min reached | peak after min | final | |
|---|---|---|---|---|
| 1 | 0.499 | 0.501 | 0.501 | stable |
| 2 | 0.498 | **15.5** | 2.07 | diverged |
| 3 | 0.499 | **40.6** | 1.08 | diverged |
| 4 | 0.499 | **58.7** | 1.46 | diverged |
| 5 | 0.498 | 0.502 | 0.499 | stable |

CE of 40–58 bits on a 3-token process is far worse than the 1.585 a *uniform*
predictor achieves. The folds reach H∞ = 0.499 first and then blow up, so this
is divergence, not a bad optimum — and because it partially recovers, it cannot
be diagnosed from the final value.

**Root cause is B3's.** Both processes contain deterministic transitions (coin
token 2 → 1 with probability 1; a flower roll is always followed by a
selection). Cross-entropy on a deterministic transition has **no finite
optimum**, so training drives the logit gap toward infinity until a step
overshoots. Before B3 was fixed this appeared as NaN; it now appears as a large
finite loss.

Measured rates, 6 seeds each, flower n=1 m=2, 60 epochs:

| configuration | diverged |
|---|---|
| MPS, Adam, lr=5e-3 (**current**) | **4/6** |
| MPS, AdamW, wd=0.01 | 1/6 |
| MPS, Adam, lr=1e-3 | 0/6 |
| CPU, Adam, lr=5e-3 | 0/6 |

It is an **interaction**: the landscape is sharp enough that MPS's numerical
noise tips it over where CPU's does not. Both a lower learning rate and weight
decay suppress it — weight decay is the principled one, since it restores a
finite optimum.

**Left unfixed by your decision**, because changing the optimiser or the
learning rate would change every number in the study. Consequences to carry
into any write-up:

- SANITY and LARGE keep only 1–2 converged folds, so the converged-only `sem`
  is unreliable and can be `nan`. Do not quote it as a confidence interval.
- QUICK is unaffected — 10 epochs is not long enough to drive the logits that
  far, and all 5 folds converge.
- If you later want stable long runs, `AdamW(weight_decay=0.01)` is the change
  to make, and everything would need re-running.

---

## 4. What is still not done

- **C1** — `S_emp` is still `H(k-means occupancy)` at a *pre-specified* k. It
  measures cluster balance at an assumed k and cannot discover the number of
  causal states. Phase 4.1. **And there is now direct evidence it matters**:
  see §5.
- **C2** — `reverse_pos_for_backward` still never set anywhere. This is the
  remaining architectural asymmetry between the arms and Phase 4.4 exists to
  test it.
- **C3/C5 naming**, Tier E renames beyond the ones already done.
- **Phase 4** — model selection over k, the seed-repeat harness, the `d_model`
  sweep, the second null control.
- **Phase 5** — README still points at `main.py` and `run_experiments.ipynb`,
  neither of which exists.
- The `git filter-repo` history rewrite from Phase 0 §5.

---

## 5. New evidence that C1 matters

`S_emp` against the closed form, from the QUICK re-run:

| experiment | S_emp FW | C⁺ | S_emp BW | C⁻ |
|---|---|---|---|---|
| `exp1_coin_p040_q080` | 0.9248 | 0.9183 | 1.5582 | 1.5656 |
| `exp1_2_coin_p010_q090` | **0.5574** | **0.4690** | 0.8783 | 0.8911 |
| `exp2_flower_n2_m6` | 1.5020 | 1.5000 | **2.5967** | **2.1137** |
| `exp2_flower_n4_m2` | 2.0021 | 2.0000 | 1.4941 | 1.4952 |

Agreement is excellent where the true causal-state distribution is close to
uniform, and poor where it is skewed. `exp1_2` has π = (0.9, 0.1), giving
C⁺ = 0.469, and k-means — which tends toward balanced clusters — returns 0.557.
`exp2_flower_n2_m6` overshoots by 0.48 bits against a ceiling of log₂7 = 2.807.

**So `S_emp` systematically overestimates complexity when the true state
occupancy is unbalanced.** That is a concrete, measured instance of exactly what
C1 predicts, and it should be stated wherever `S_emp` is reported.

---

## 6. How to verify

```bash
conda activate qdrug
pytest tests/ -q                              # 52 passed
python run_experiments.py --config SMOKE      # ~2 min, all paths

# C6: what the old sampling actually covered
python -c "
import numpy as np; from Model_analysis import _sample_latents
N,T=500,256
lat=np.zeros((N,T,4),dtype=np.float32); lat[...,0]=np.arange(N)[:,None]
inp=np.zeros((N,T),dtype=int)
for m in ('prefix','per_sequence','random'):
    p,_,l=_sample_latents(lat,inp,mode=m,n_pts=1000)
    print(f'{m:13} {len(p):4} pts, {len(np.unique(p[:,0])):3} sequences')
"

# the divergence shape, from the real run
python -c "
import pickle,numpy as np
r=pickle.load(open('sanity_check_flower_process/flower_n1_m2_eq/results.pkl','rb'))
for i,fc in enumerate(r['cv_fw']['fold_curves']):
    sl=np.array(fc['step_loss'])
    if sl.size: print(f'fold {i+1}: min={sl.min():.3f} peak-after={sl[sl.argmin():].max():8.2f}')
"
```

---

## 7. Runs

| Run | Command | Log | Status |
|---|---|---|---|
| QUICK | `python run_experiments.py --config QUICK` | `run_quick.log` | complete, 8:44 |
| SANITY | `python sanity_check.py` | `run_sanity.log` | complete, 6:54 (was 12.7 min) |
| LARGE | `python run_experiments.py --config LARGE` | `run_large.log` | running, ~9 h remaining |

### LARGE — divergence is near-universal at 80 epochs

**exp1 completed in 17.6 min** (was 69.3 min before D1 — a 3.9x speedup, better
than the 2.58x measured in isolation). But **all 10 of its folds diverged**
(5/5 forward, 5/5 backward), against H∞ = 0.9197:

```
fold 1: CE_FW=1.0063  CE_BW=0.9236  delta=-0.0827
fold 2: CE_FW=1.2622  CE_BW=0.9430  delta=-0.3192   NOT CONVERGED
fold 3: CE_FW=1.2780  CE_BW=1.3311  delta=+0.0532   NOT CONVERGED
fold 4: CE_FW=1.5470  CE_BW=0.9242  delta=-0.6228   NOT CONVERGED
fold 5: CE_FW=0.9434  CE_BW=1.1713  delta=+0.2279   NOT CONVERGED

all folds      : mean=-0.1487  sem=0.1485
converged only : mean=-0.0827  sem=nan   (1/5 folds, 4 dropped)
```

So at 80 epochs the rate is worse than the 4/6 measured at sanity_check's 60
epochs — longer training means more opportunity for the logit gap to grow until
a step overshoots.

**LARGE's paired ΔCE is therefore not a measurement.** The surviving −0.0827 is
one fold's optimisation luck. Decision taken (again) to let the run finish
rather than change training: the weights, their sidecars, the figures and the
pq heatmaps are all still produced and usable, and the pq sweep does not use
cross-validation at all so it is unaffected.

**QUICK remains the only run where all five folds converge, and is the one to
report.**

### SANITY, on this code

| control | H∞ | CE_FW | CE_BW | ΔCE | verdict | folds diverged |
|---|---|---|---|---|---|---|
| Coin p=q=0.5 (positive) | 1.0000 | 1.0059 | 1.0053 | **−0.0006** | **INCONCLUSIVE** | 8/10 |
| Flower n=1,m=2 (null) | 0.5000 | 0.5007 | 0.5006 | **−0.0001** | **PASS** | 6/10 |

The null control holds: no asymmetry on a time-reversible process.

The positive control returns ≈ 0 as well. That is **inconclusive, not a
refutation** — both arms converged to within 0.006 bits of H∞, and once that
happens the residuals vanish and ΔCE → 0 whatever C⁻ − C⁺ is. `sanity_check`
originally scored this with a bare sign test and printed **FAIL** for
−0.0006 bits; commit `9ac1626` gives it the same NULL_TOL band the null control
uses, with a three-way PASS / INCONCLUSIVE / FAIL verdict.

Both controls carry heavy divergence at these settings (8/10 and 6/10 folds),
so their paired statistics rest on one or two folds and the `sem` is `nan`.
**QUICK is the trustworthy run.**

`./run_rest.sh` drives SANITY then LARGE. A `run_all_done.marker` file appears
when both finish.

QUICK results on this code, all four experiments converged to within ~0.005
bits of H∞:

| experiment | C⁻−C⁺ | ΔCE | sem | verdict |
|---|---|---|---|---|
| `exp1_coin_p040_q080` | +0.6473 | −0.0005 | 0.0012 | n.s. |
| `exp1_2_coin_p010_q090` | +0.4221 | +0.0022 | 0.0002 | match |
| `exp2_flower_n2_m6` | +0.6137 | +0.0005 | 0.0016 | n.s. |
| `exp2_flower_n4_m2` | −0.5048 | −0.0019 | 0.0019 | n.s. |

**Do not over-read the one "match".** +0.0022 bits is two thousandths of a bit;
it clears 2 sem only because the fold-level sem (0.0002) is tiny, and that sem
is computed over five folds that **share a training set**, so it measures
fold-to-fold variability, not sampling variability over datasets. It is a lower
bound on the true uncertainty. The overall picture remains a null, which §1.1
of the plan predicts for a converged predictor with spare capacity, and which
stays ambiguous until the `d_model` sweep.
