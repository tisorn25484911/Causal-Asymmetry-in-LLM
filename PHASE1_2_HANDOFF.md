# Phase 1 & 2 — Handoff

Date: 2026-08-04
Branch: `phase0-unblock` (6 new commits on top of Phase 0, **not pushed**)
Follows: `IMPROVEMENT_PLAN.md` §7 Phase 1 and Phase 2
Previous: `PHASE0_HANDOFF.md`

**Status: Phase 1 and Phase 2 complete in code, verified by 37 tests and a
full SMOKE run.** Phase 1.9 ("re-run everything") is in progress — see §8.

---

## 0. The one thing to read if you read nothing else

**Every checkpoint in `results/models/` is now unloadable, and every number in
`results/` should be discarded.**

Two independent reasons:

1. `rand_prj` moved from `nn.Parameter` to a registered buffer (C5, your call),
   so old `state_dict`s no longer match the model.
2. The numbers were produced with A1, A2, A3, B1, B2, B4, B5, B6 live. B2 alone
   biased perplexity high by ~0.041 bits — the same order as the ΔCE being
   measured.

New runs write to `results_quick/` and `results_large/`. The old `results/` is
untouched on disk; nothing reads it any more.

---

## 1. Decisions you made, and what they cost

Four questions were put to you before any code changed. Recording them here
because each one closed off an alternative that a later reader might otherwise
assume was overlooked.

| Decision | Chosen | Consequence |
|---|---|---|
| Chunk windowing | Fixed window per sequence | Exact FW/BW pairing and noise-free validation curves. **Cost: each sequence now contributes one 256/512-token window instead of a fresh one each epoch.** Recoverable by generating more sequences, not by more epochs. |
| Flower configs | Add m>n, keep n>m | Twice the flower compute, but the sign of ΔCE can now be checked against the sign of C⁻−C⁺ across configurations — much stronger than a single positive case. |
| `rand_prj` | Freeze (buffer) | "onehot" mode really is a fixed random projection, so the Phase 4 `d_model` sweep varies capacity alone. **Invalidates all existing checkpoints.** |
| Re-run scope | Everything | SANITY + QUICK + LARGE launched; see §8. |

---

## 2. The commits, and why each is a unit

Six commits. Each is self-contained and the tree runs at every one.

### `f68437e` — Phase 1.4/1.5: fix the loss path (B3, B1, B2)

Grouped because all three are in the path that produces the headline number,
and the plan's §9 says do them together before any long run.

- **B3** was the only *silent NaN* in the repo. `cross_ent_onehot` built the
  one-hot target and evaluated `-(target_prob * softmax.log2()).sum(dim=1)`;
  for non-target classes `target_prob = 0`, so a class probability underflowing
  to exactly 0 gives `0 * -inf = NaN`, which propagates through `.sum()` and
  poisons the whole batch. **Measured: at logit scale 200 the old expression
  returns `nan` where the true value is `0.0`** (10/50/100 are fine). Reachable
  because both processes contain deterministic transitions — coin token 2 → 1
  with probability 1, and a flower roll is always followed by a selection —
  and CE on a deterministic transition has no finite optimum, so training drives
  the logit gap toward infinity. Replaced with `F.cross_entropy / ln 2`:
  identical value, log-sum-exp internally, and faster.
- **B1** recorded `math.exp(loss)` as train perplexity while the loss is in
  bits, putting CV panel 3 on a different scale from panel 4 beside it.
- **B2** returned `mean(2**CE_batch)` instead of `2**mean(CE)`. **Measured
  Jensen bias +0.0411 bits** on a realistic CE spread — the same order as ΔCE.
  It also divided by `n_batches`, so a ragged final batch counted as a full one.
  Now token-weighted, and PPL is derived from the aggregate CE so `PPL = 2**CE`
  holds by construction.
- **B12** moved epoch-loss accumulation above the `record_every_n_steps` gate.
  Harmless at the current stride of 1 — which is exactly why it would have gone
  unnoticed when Phase 3 raises it to 25 and `epoch_loss` silently becomes a
  different estimator.
- **B13** deleted `record_val_loss`, a parameter never assigned in `__init__`.

### `93de0ef` — Phase 1.1: pair the two arms (A2)

The plan calls this "the highest-value single change in the repo" and it is.
ΔCE was an **unpaired** difference between two runs that did not see the same
data. Three separate causes, all fixed:

1. `random_split` had no `generator=`, so it consumed the global torch RNG.
   The pipeline runs once per direction → different hold-out test sets and
   different train/val pools. Seeding the fold *permutation* (already done)
   does not help when the pool being permuted differs.
2. `ChunckDataset` drew windows from a stateful RNG at access time, and both
   arms were handed the *same loader object*. The forward run advanced the
   generator, so the backward run trained on different windows.
   `sample_seq = next(iter(loader_fw))[0][0]` advanced it once more first.
3. Fold batch order and weight init came from wherever the global RNG was.

Now: offsets drawn once at construction so `__getitem__` is a pure function of
`idx`; explicit generators everywhere; `torch.manual_seed(seed*1000 + fold)`
before each `train_model` so fold *k* of both arms starts from identical
weights.

**A side effect worth knowing:** validation used to re-draw fresh windows at
*every recorded step*, so `step_val_loss` carried window noise on top of
learning signal — and "best fold" was chosen partly on which fold drew easy
windows. That is now deterministic.

Adds `set_seed()`, exposes `fold_test_loss`/`seed` from the CV pipeline, and
adds `paired_delta_ce()` reporting per-fold ΔCE with sem and a paired *t*.

### `d03a55d` — Phase 1.2: flower closed forms, null control relabelled (A1)

`flower_complexity(n, m, dice_probs)` and `flower_entropy_rate(...)`.

The formulas: C⁺ = 1 + ½log₂n. C⁻ = 1 + ½H(π_merged), where outcomes merge when
they induce the same posterior P(die | outcome) — i.e. when their `dice_probs`
columns are proportional. **That word "distinguishable" is what the repo's m+1
count missed**, and it has two consequences:

- **n=1 degenerates.** One die ⇒ P(die|outcome)=1 for every outcome ⇒ all m
  outcomes collapse to one backward state ⇒ C⁻ = C⁺ = 1. The `sanity_check`
  flower process `0,X,0,X,…` is exactly time-reversible. Its old docstring
  stated the refuting fact — *"token 1 and 2 both deterministically precede
  token 0"* — while drawing the opposite conclusion.
- **C⁻ ≤ 1 + ½log₂m, so C⁻ > C⁺ requires m > n.** No flower config in the repo
  satisfied that.

Verified against `IMPROVEMENT_PLAN.md` §4 to 4 decimals: (1,2)→(1.0000, 1.0000),
(4,2)→(2.0000, 1.4952), (6,4)→(2.2925, 1.9899). `flower_entropy_rate(1,2,fair)`
reproduces the 0.5 that `sanity_check.py` had hard-coded.

`sanity_check.py` is rewritten around this: the flower arm is now the **null
control**, scored `|ΔCE| < NULL_TOL` instead of `ΔCE > 0 else FAIL` — which was
reporting the correct answer as a failure. It clusters at k=2 for *both*
directions instead of k_bw = m+1 = 3; since S = H(cluster occupancy) ≤ log₂k,
handing the backward arm a larger k inflates C⁻ by construction and then
"confirms" the 3 states it assumed.

### `f6d2812` — Phase 1.3/1.6/1.7/1.8: runner correctness

- **A3** — the backward arm read complexity at the **minimum**-context
  position. With a `triu` mask position *t* attends to `[t, T-1]`, so `T-1`
  attends to itself alone; maximum context is at position 0. Every backward
  complexity number in `results/` came from latents encoding a single token.
  The inline comment even read `# forward model → last pos` on a backward model.
- **B4 + D3 together**, because either alone is wrong. The runners built
  `loader_fw_ana` and then passed the *chunked* loader to `analyse_model`, so
  UMAP and complexity ran on 256-token windows while README:233 promises
  full-length analysis. Fixing that takes T from 256 to ~2000, and attention is
  O(T²): **0.48 GB retained per layer per pass at B=32, T=1999**. So attention
  is now gated behind `store_attention` (default off) with a
  `capture_attention()` context manager, only the two plotting helpers enable
  it, and analysis runs at `ana_batch=8`. Verified logits are identical with
  and without capture.
- **B5** — `max_len` is the full input length, not the chunk. Masked while B4
  was unfixed; fixing B4 alone would have exposed it.
- **B6** — k = n+1 / m+1 instead of n+m for both.
- **B7** — `cfg.get(...) if 'cfg' in dir() else 200` is always 200: `dir()` with
  no argument lists the *local* scope. `cfg` is now a real parameter. Value
  lowered 200 → 15, because 200 neighbours on 1000 points smears the local
  structure the plot exists to show.
- **B8** — tags derived from config values.
- Deleted the dead `ds_bw`/`loader_bw` chain in `experiment_2` and the
  `loader_bw` parameter of `compare_fw_bw`, which received `loader_fw` at all
  three call sites and was never read.

### `87ad17c` — Phase 2: one runner, one config file (A4, C4, C5, B11, B9, B10, B14, C7)

**A4** is the structural one. `Main_call.py` and `main_large.py` both wrote to
`results/` and both used the tag `exp1_coin_p03_q04` — at different parameters
*and* different `d_model`. One `.pt` pair existed for that tag; whichever runner
finished last owned it. The post-hoc consumers hard-coded main_large's tags
*and* CFG, so running either after `Main_call.py` would score p=0.4,q=0.8
weights against p=0.3,q=0.4 theory. Only the `d_model` mismatch raising in
`load_state_dict` kept that from being silent.

Fixed so the *class* of error is gone: `out_root` is part of the config; tags
are derived; the consumers build their config from `configs.py`; every `.pt`
gets a JSON sidecar and both loaders call `check_weight_meta` to verify rather
than guess.

The rest: **C4** single definitions in `utils.py` (five helpers were duplicated,
`entropy_rate_coin` existed three times, the KL helpers were byte-identical in
two files, and the flower Dataset/generator existed in two files with every
runner importing one from each). **C5** frozen `rand_prj`, deleted
`OneHotDecoder.we` and `WordEmbDecoder`. **B11** `n_layers` threaded + sidecars.
**B9** three `plt.show()` under Agg where show() is a no-op — two figures were
built and silently discarded. **B10** `FW_BW_loss_comparison` ended in a bare
`return`. **B14** `Test_data_eval`'s headline chart used the one metric the rest
of the repo says must not be used for comparison; now reports both, correctly
labelled. **C7** `make_loader`'s no-op `states` argument.

### `f6f9ef1` — Phase 4.6 brought forward: 37 regression tests

Brought forward because almost everything Phase 1 fixed was *silent* — nothing
raised, the numbers were just wrong. Without tests a later edit re-introduces
them invisibly.

Includes the case that would have caught each bug, and checks theory against
the *generator* rather than only against itself: `coin_true_conditional` is
compared to a Monte-Carlo estimate over 40×4000 tokens at three (p,q).

```
conda activate qdrug && pytest tests/ -q      # 37 passed
```

---

## 3. What changed for you as a user

```bash
conda activate qdrug

python run_experiments.py --config SMOKE     # ~2 min, exercises every path
python run_experiments.py --config QUICK     # was Main_call.py
python run_experiments.py --config LARGE     # was main_large.py
python run_experiments.py --config QUICK --seed 1 --only exp1
python sanity_check.py                       # controls
pytest tests/ -q
```

`Main_call.py` → `run_experiments.py`; `main_large.py` deleted (it was a
73-line-diff copy). Both live on in git history.

Output goes to `results_smoke/`, `results_quick/`, `results_large/`, each with
a `run_config.json` recording exactly what produced it.

---

## 4. A reproducibility limit you should know about

Seeding gives **pairing** — within one run the two arms share splits, folds,
batch order and initialisation — and that is what ΔCE actually needs.

It does **not** give bit-reproducibility across runs on Apple MPS. Measured:
model init is identical and the first loss matches to 8 decimals, then the
trajectories diverge from the first backward pass (~1e-3 by step 2). The same
test on CPU is exactly repeatable, so this is the MPS backend, not the code.

CPU is ~6× slower at these sizes (LARGE: 304 vs 49 ms/step, i.e. ~13 hr →
~78 hr), so MPS stays the default. Set `accelerator="cpu"` in the config when
an exactly reproducible run is worth the wall-clock — worth doing for the final
reported numbers if a referee may ask.

---

## 5. Things I deliberately did **not** do

- **Phase 3** (D1 per-step validation, D2 vectorisation, D4 cheap wins) — not
  in scope. Note D1 and B12 interact and B12 is already done, so D1 is now safe
  to apply alone.
- **Phase 4** beyond 4.6: C1 (model selection over k), the seed-repeat harness,
  the `d_model` sweep, C2 (`reverse_pos_for_backward`). These are the
  *scientific* strengthening and they need the re-run results first.
- **Phase 5** docs. README still points at `main.py` and
  `run_experiments.ipynb`, neither of which exists, and now also needs updating
  for `run_experiments.py --config`.
- **C6** — UMAP still plots a prefix, which at T=1999 means positions 0-499 of
  sequence 0 only. Listed as Tier C, not in Phase 1 or 2. **This is the next
  thing I would fix**, because it means the latent figures currently show one
  trajectory rather than a sample of the distribution.
- The `git filter-repo` history rewrite from Phase 0 §5 — still yours to run.
- Your in-flight edits to `IMPROVEMENT_PLAN.md` and the four notebooks are
  still uncommitted in the working tree, as Phase 0 left them.

---

## 6. Interpreting what comes out

From `IMPROVEMENT_PLAN.md` §1.1, and it bears repeating because it constrains
every claim: H∞ is time-reversal invariant, so

```
ΔCE = CE_BW − CE_FW = residual_BW − residual_FW
```

is a difference of **residuals**, not of entropy rates. An unbounded, fully
converged predictor gives ΔCE = 0 regardless of C⁻ − C⁺. So:

- Only the **sign** of ΔCE is predicted, and only when C⁻ ≠ C⁺. The magnitude
  has no theoretical anchor.
- A null result is **ambiguous** — "no asymmetry" or "`d_model` was large enough
  to absorb it" — and only the `d_model` sweep (Phase 4.5) disambiguates.
- The paired ΔCE now reported has sem over **5 folds of one seed**, which share
  a training set. That is fold-to-fold variability, not sampling variability.
  Phase 4.2 is what turns it into evidence. `paired_delta_ce`'s docstring says
  so, so it cannot be quoted out of context.

The new cross-experiment summary table prints whether sign(ΔCE) matches
sign(C⁻−C⁺) per experiment. With the positive and reversed flower configs both
running, **four sign predictions** are now on the table instead of one.

---

## 7. How to verify any of this yourself

```bash
conda activate qdrug
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version

pytest tests/ -q                              # 37 passed
python run_experiments.py --config SMOKE      # full suite, ~2 min

# B3: the bug that was silent
python -c "
import torch; from OneHot_model import cross_ent_onehot
for s in (100, 200, 1000):
    l,_ = cross_ent_onehot(torch.tensor([[[float(s),0.,0.]]]), torch.tensor([[0]]))
    print(s, l.item())          # all finite; the old code gave nan at 200
"

# A2: the two arms see identical windows
python -c "
import torch
from Training_model import ChunckDataset
from Data_generation import CoinDataset, coin_generation
d,_ = coin_generation(num_samples=20, seq_len=200, p=.4, q=.8)
ds = CoinDataset(d, seq_len=200)
a, b = ChunckDataset(ds,64,seed=0), ChunckDataset(ds,64,seed=0)
print(all(torch.equal(a[i][0], b[i][0]) for i in range(len(ds))))
"

# A1: the null control really is null
python -c "
from Model_analysis import flower_complexity
print(flower_complexity(1, 2, [[0.5, 0.5]]))   # (1.0, 1.0), not (1.0, 1.5)
"
```

---

## 8. The re-runs (Phase 1.9)

Launched on 2026-08-04, in this order, sequentially — they share one MPS device,
so overlapping them only thrashes it:

| Run | Command | Log | Output |
|---|---|---|---|
| SANITY | `python sanity_check.py` | `run_sanity.log` | `sanity_check_flower_process/` |
| QUICK | `python run_experiments.py --config QUICK` | `run_quick.log` | `results_quick/` |
| LARGE | queued behind QUICK by `run_large_when_quick_done.sh` | `run_large.log` | `results_large/` |

Check progress with `tail -f run_quick.log`, or jump to the verdict:

```bash
grep -A12 "ASYMMETRY SUMMARY" run_quick.log
grep -A30 "SANITY CHECK SUMMARY" run_sanity.log
```

**What to look for, in priority order:**

1. **The null control.** `sanity_check` flower n=1,m=2 must give
   |ΔCE| < 0.02. If it does not, the pipeline manufactures asymmetry where the
   process has none, and no positive result from it means anything. This is the
   single most informative number in the whole re-run.
2. **The positive control.** Coin p=q=0.5 must give ΔCE > 0.
3. **Sign tracking.** In the QUICK/LARGE summary table, `exp2_flower_n2_m6`
   (C⁻ > C⁺) should give ΔCE > 0 and `exp2_flower_n4_m2` (C⁺ > C⁻) should give
   ΔCE < 0. The signs flipping *with the theory* is the strongest evidence this
   design can produce.
4. Convergence: CE should approach H∞ from above in both arms.

If the null control fails, **stop and look at C2** (`IMPROVEMENT_PLAN.md`) before
anything else. The positional encoding is the remaining architectural asymmetry:
both arms get `PE[t]` increasing left-to-right, so for the forward model context
length is increasing in the PE index and for the backward model it is
decreasing. `reverse_pos_for_backward=True` exists and is never set anywhere.
That is the most likely source of a spurious ΔCE, and Phase 4.4 is written to
test exactly it.
