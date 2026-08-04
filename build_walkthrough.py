"""
Generate walkthrough.ipynb — the Phase 0-3 demonstration notebook.

Written as a generator script rather than by hand so the notebook can be
regenerated after a re-run without hand-editing JSON.  Run:

    python build_walkthrough.py
"""
import json

cells = []


def _lines(text):
    """nbformat wants each source line to keep its trailing newline."""
    return text.strip("\n").splitlines(keepends=True)


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": _lines(text)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": _lines(text)})


# ─────────────────────────────────────────────────────────────────────────────
md(r"""
# Causal Asymmetry in Sequence Models — what changed in Phases 0–3

This notebook walks through every change made to this repository across Phases
0, 1, 2 and 3 of `IMPROVEMENT_PLAN.md`, **demonstrates each bug rather than
asserting it**, and reports the results with their caveats attached.

It is deliberately written so that a sceptical reader can check the claims:
wherever a bug is described, the old and new behaviour are computed side by
side in the same cell.

## The scientific question

Two transformer decoders are trained on **the same forward-generated data**.
They differ only in the attention mask and the batch convention:

| | mask | at position *t* sees | predicts |
|---|---|---|---|
| forward | `tril` | `x[0..t]` | `x[t+1]` |
| backward | `triu` | `x[t..T-1]` | `x[t-1]` |

The hypothesis (Crutchfield/Ellison–Mahoney; Thompson et al. 2017) is that a
process needing more memory to predict backward than forward (C⁻ > C⁺) should
give the backward model a measurably higher cross-entropy.

## The single most important caveat, stated up front

The entropy rate is **time-reversal invariant**: H∞(forward) = H∞(backward).
So

```
CE_FW → H∞ + residual_FW
CE_BW → H∞ + residual_BW
ΔCE   = CE_BW − CE_FW = residual_BW − residual_FW
```

**ΔCE is a difference of residuals, not of entropy rates.** An
unbounded-capacity, fully-converged predictor achieves ΔCE = 0 in both
directions *regardless* of C⁻ − C⁺. ΔCE > 0 is purely an artefact of a
**memory-bounded** predictor.

Two consequences that constrain everything below:

1. Only the **sign** of ΔCE is predicted, and only when C⁻ ≠ C⁺. The magnitude
   has no theoretical anchor.
2. A null result (ΔCE ≈ 0) is **ambiguous** — "no asymmetry" or "capacity was
   large enough to absorb it". Only the `d_model` sweep (Phase 4.5, not yet
   run) disambiguates it.
""")

code(r"""
import os, sys, math, pickle, json
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.getcwd())
np.set_printoptions(precision=4, suppress=True)

from Model_analysis import (flower_complexity, flower_entropy_rate,
                            statistical_complexity, _sample_latents,
                            paired_delta_ce, warm_up_umap)
from OneHot_model import cross_ent_onehot, OneHotDecoder
from Training_model import (ChunckDataset, _eval_loss_on_loader, set_seed,
                            diagnose_divergence, make_chunked_loader,
                            make_analysis_loader)
from utils import entropy_rate_coin
from Data_generation import CoinDataset, coin_generation
from configs import QUICK, LARGE

print("environment ready — torch", torch.__version__)
""")

# ── Phase 0 ──────────────────────────────────────────────────────────────────
md(r"""
---
# Phase 0 — hygiene

Phase 0 (previous handoff, `PHASE0_HANDOFF.md`) did three things:

- `.gitignore` + dropped 5830 generated files from the index. `.git` was 6.2 GB,
  of which ~4.6 GB was committed pickles.
- **`slim_results()`** — result bundles pickled at **700.8 MB**; slimming took
  them to **0.3 MB (2669×)** with metrics, curves and UMAP coordinates
  verified identical. Two causes: the latents were stored *twice* (`latents`
  and `flat_lat` are the same data, and pickle does not dedupe a reshape), and
  the "models" were mostly stale `last_attention`/`last_encodings` activation
  buffers rather than weights.
- **D5** — `L.Trainer(logger=False, enable_checkpointing=False)`. Nothing in
  the repo ever read `lightning_logs/*.ckpt`.

One thing worth repeating from that handoff: `qdrug` had **no `umap` and no
`numba`**, so every "UMAP" latent plot produced before Phase 0 was silently a
**PCA** plot (`UMAP_AVAILABLE = False` → fallback). Check the title of any old
figure: if it says `(PCA)`, that is why.
""")

# ── Phase 1 ──────────────────────────────────────────────────────────────────
md(r"""
---
# Phase 1 — make the comparison valid

## B3 — the silent NaN

`cross_ent_onehot` built the one-hot target and evaluated

```python
-(target_prob * logits.softmax(-1).log2()).sum(dim=1)
```

For non-target classes `target_prob = 0`. If any class probability underflows
to exactly 0, that term is `0 * -inf = NaN`, and the NaN propagates through
`.sum()` — poisoning the **whole batch** even though the target class was fine.

This is reachable, not hypothetical: both processes contain **deterministic
transitions**, and cross-entropy on a deterministic transition has no finite
optimum, so training drives the logit gap toward infinity.
""")

code(r"""
def old_cross_ent(logits, targets):
    "The pre-Phase-1 implementation, verbatim."
    C = logits.shape[-1]
    flat_input, flat_target = logits.reshape(-1, C), targets.reshape(-1)
    target_prob = torch.nn.functional.one_hot(flat_target, num_classes=C).float()
    logits_prob = flat_input.softmax(dim=-1)
    return -(target_prob * logits_prob.log2()).sum(dim=1).mean()

print(f"{'logit scale':>12} {'OLD':>12} {'NEW':>12}")
for s in (10, 50, 100, 200, 1000):
    lg = torch.tensor([[[float(s), 0., 0.]]]); tg = torch.tensor([[0]])
    old = old_cross_ent(lg, tg).item()
    new = cross_ent_onehot(lg, tg)[0].item()
    print(f"{s:>12} {old:>12.6f} {new:>12.6f}")
print("\nThe old expression returns nan from logit scale 200 upward.")
print("The true value there is 0.0 — the model is simply very confident and right.")
""")

md(r"""
## B2 — the perplexity estimator was biased high

`_eval_loss_on_loader` returned `mean(2**CE_batch)` instead of `2**mean(CE)`.
By Jensen's inequality the first is ≥ the second, so reported perplexity was
biased **upward** by an amount that grows with the variance of per-batch CE —
and it also divided by `n_batches`, so a ragged final batch counted as much as
a full one.

**This matters because the bias is the same order as the effect being
measured.**
""")

code(r"""
ces = [0.9, 1.4, 1.1, 0.6]                     # a realistic per-batch spread
biased   = np.mean([2**c for c in ces])         # what the old code returned
unbiased = 2**np.mean(ces)                      # what it should return
print(f"  mean(2^CE) = {biased:.4f}   <- old")
print(f"  2^mean(CE) = {unbiased:.4f}   <- correct")
print(f"  Jensen bias = +{biased - unbiased:.4f} bits")
print(f"\n  For comparison, the delta_CE measured in QUICK is ~0.002 bits.")
print(f"  The bias was ~20x the effect.")
""")

md(r"""
## A2 — the two arms were not seeing the same data

This is the change the plan calls *"the highest-value single change in the
repo"*. ΔCE was an **unpaired** difference between two runs trained on
different data. Three separate causes:

1. `random_split` had no `generator=`, so it consumed the global torch RNG.
   The pipeline runs once per direction → **different hold-out test sets**.
2. `ChunckDataset` drew its window from a *stateful* RNG at access time, and
   both arms were handed the **same loader object**. The forward run advanced
   the generator, so the backward run trained on different windows.
3. Fold batch order and weight initialisation came from wherever the global RNG
   happened to be.

The fix makes `__getitem__` a pure function of the index.
""")

code(r"""
set_seed(0)
data, _ = coin_generation(num_samples=30, seq_len=300, p=0.4, q=0.8)
ds = CoinDataset(data, seq_len=300)

a = ChunckDataset(ds, 64, seed=0)
b = ChunckDataset(ds, 64, seed=0)     # a "fresh" dataset for the other arm
c = ChunckDataset(ds, 64, seed=0)
_ = [c[i] for i in reversed(range(len(ds)))]      # access out of order
d = ChunckDataset(ds, 64, seed=1)

print("two arms, same seed  -> identical windows :",
      all(torch.equal(a[i][0], b[i][0]) for i in range(len(ds))))
print("independent of access order                :",
      all(torch.equal(a[i][0], c[i][0]) for i in range(len(ds))))
print("different seed still gives different windows:",
      any(not torch.equal(a[i][0], d[i][0]) for i in range(len(ds))))
""")

md(r"""
**Caveat of this fix, stated honestly.** Each sequence now contributes **one**
fixed window instead of a fresh one each epoch. That is a deliberate trade:
exact pairing is what the whole study rests on, but the training set is
correspondingly less diverse. It is recoverable by generating more sequences,
not by training for more epochs.

A useful side effect: validation used to re-draw fresh windows at *every*
recorded step, so `step_val_loss` carried window noise on top of learning
signal, and "best fold" was chosen partly on which fold drew easy windows.

## A1 — every flower configuration tested the wrong thing

The repo counted backward causal states as **m+1**. The correct count is one
per *distinguishable* outcome, where two outcomes are the same state exactly
when they induce the same posterior P(die | outcome) — i.e. when their
`dice_probs` columns are proportional.

```
C⁺ = 1 + ½·log₂(n)
C⁻ = 1 + ½·H(π_merged)
```

Two consequences:

- **n=1 degenerates.** One die ⇒ P(die|outcome)=1 for every outcome ⇒ all m
  outcomes collapse to ONE state ⇒ C⁻ = C⁺. The `sanity_check` flower process
  `0,X,0,X,…` is **exactly time-reversible** — a null control, not a positive
  one. Its old docstring stated the refuting fact while drawing the opposite
  conclusion.
- **C⁻ ≤ 1 + ½log₂(m), so C⁻ > C⁺ requires m > n.** Every flower config
  originally in the repo had n > m, i.e. all of them tested the *negation* of
  the hypothesis.
""")

code(r"""
rows = [("sanity_check (was 'positive')", 1, 2, np.array([[.5, .5]])),
        ("old Main_call exp2",            4, 2, np.random.default_rng(42).dirichlet(np.ones(2), size=4)),
        ("old main_large exp2",           6, 4, np.random.default_rng(42).dirichlet(np.ones(4), size=6)),
        ("NEW positive (QUICK)",          2, 6, np.random.default_rng(42).dirichlet(np.ones(6), size=2)),
        ("NEW positive (LARGE)",          2, 8, np.random.default_rng(42).dirichlet(np.ones(8), size=2))]

print(f"{'config':<32}{'n':>3}{'m':>3}{'C+':>9}{'C-':>9}{'C- - C+':>10}   verdict")
for name, n, m, dice in rows:
    cp, cm = flower_complexity(n, m, dice)
    v = ("C- > C+  (tests the hypothesis)" if cm > cp + 1e-9 else
         "C+ > C-  (tests its NEGATION)"   if cp > cm + 1e-9 else
         "C- = C+  (null control)")
    print(f"{name:<32}{n:>3}{m:>3}{cp:>9.4f}{cm:>9.4f}{cm-cp:>+10.4f}   {v}")
""")

code(r"""
# The degeneracy that the m+1 count missed, shown directly.
print("n=1: every outcome has the same posterior over dice, so they merge:")
print("   flower_complexity(1, 2, fair) =", flower_complexity(1, 2, [[.5, .5]]))
print("\nSame effect with n=2 when the two dice are identical (columns proportional):")
print("   flower_complexity(2, 3, [[.2,.3,.5],[.2,.3,.5]]) =",
      flower_complexity(2, 3, [[.2,.3,.5],[.2,.3,.5]]))
print("\nAnd the closed forms reproduce IMPROVEMENT_PLAN section 4 exactly:")
for n, m, seed, want in [(1,2,None,(1.0,1.0)), (4,2,42,(2.0,1.4952)), (6,4,42,(2.2925,1.9899))]:
    dice = np.array([[.5,.5]]) if seed is None else np.random.default_rng(seed).dirichlet(np.ones(m), size=n)
    got = flower_complexity(n, m, dice)
    print(f"   ({n},{m}) -> ({got[0]:.4f}, {got[1]:.4f})   plan says {want}")
""")

md(r"""
## A3 — the backward arm was read at its *minimum*-context position

With a `triu` mask, position *t* attends to `[t, T-1]`. So position `T-1`
attends to **itself only** — one token. Maximum context for a backward model is
at position **0**.

Both runners passed `use_t="last"` for the backward model, with an inline
comment reading `# forward model → last pos` on a model that is not forward.
Every backward complexity number in the old `results/` was computed from
latents encoding a single token.

## The analysis-length bug — found by the null control, and *not* in the plan

The plan's item B4 says analysis should run on **full sequences**, quoting
README:233 — *"analysis on full sequences remains in-distribution"*.

**That claim is false whenever `chunk_len` < the full sequence length.** A chunk
is fed to the model as a standalone sequence, so the positional-encoding index
is the position *within the chunk*: the model only ever sees PE indices
`[0, chunk)`. Evaluating at full length asks it to extrapolate to positions it
has never seen.

Implementing B4 as written made the null control report **ΔCE = −0.4210 on an
exactly time-reversible process**. That is precisely the failure a null control
exists to detect.
""")

code(r"""
# Position-wise CE on a real checkpoint trained at chunk 512, evaluated at T=999.
ck = "sanity_check_flower_process/models/coin_p05_q05_fw.pt"
if os.path.exists(ck):
    set_seed(0)
    m = OneHotDecoder(token_size=3, d_model=64, max_len=1000, mode="forward", n_layers=2)
    m.load_state_dict(torch.load(ck, map_location="cpu")); m.eval()
    d, _ = coin_generation(num_samples=64, seq_len=1000, p=.5, q=.5)
    dset = CoinDataset(d, seq_len=1000)
    X = torch.stack([dset[i][0] for i in range(64)])
    Y = torch.stack([dset[i][1] for i in range(64)])
    with torch.no_grad():
        lg = m(X)
        ce = (torch.nn.functional.cross_entropy(lg.reshape(-1,3), Y.reshape(-1),
              reduction="none").reshape(64,999) / math.log(2))
    print("Coin p=q=0.5, H_inf = 1.0000.  Model TRAINED on chunk length 512.\n")
    for lo in range(0, 999, 100):
        hi = min(lo+100, 999)
        tag = "trained" if hi <= 512 else ("NEVER TRAINED" if lo >= 512 else "straddles")
        print(f"   positions {lo:4}-{hi-1:4}:  CE = {ce[:, lo:hi].mean():6.4f}   {tag}")
    print(f"\n   positions   0-511 (trained)      : {ce[:, :512].mean():.4f}   <- converged")
    print(f"   positions 512-998 (never trained): {ce[:, 512:].mean():.4f}")
    print(f"   whole sequence                   : {ce.mean():.4f}   <- what was reported")
else:
    print("checkpoint not present — run sanity_check.py first")
""")

md(r"""
The cell above recomputes this live from whichever `coin_p05_q05_fw.pt` is
currently on disk, so the exact figures move between runs — the *pattern* is
the invariant: CE sits at H∞ inside the trained range and rises sharply beyond
it, dragging the whole-sequence average far above H∞.

**Why this was fatal rather than merely inaccurate:** it biases the two arms
**asymmetrically**. `statistical_complexity_empirical` reads the forward arm at
`use_t="last"` and the backward arm at `use_t="first"` — correctly, those are
the max-context positions. At full length that is position T−1 for forward,
which is **untrained**, against position 0 for backward, which is **trained**.
The two arms were measured under different amounts of extrapolation, and ΔCE is
a difference between them.

Fixed by `make_analysis_loader()`, which emits windows at the **training** chunk
length from a different seed. Measured on the specific checkpoint that motivated
the fix (a run whose numbers are quoted in `PHASE1_2_HANDOFF.md`, not
necessarily the one loaded above):

| | full-length analysis | training-length analysis | theory |
|---|---|---|---|
| CE | 1.2859 | **1.0235** | H∞ = 1.0000 |
| C⁺ empirical | 0.9863 | **0.9982** | 1.0000 |
| C⁻ empirical | 1.5458 | **1.5061** | 1.5000 |

The empirical complexity matched the closed form to three decimals for the
first time only after this change.
""")

# ── Phase 2 ──────────────────────────────────────────────────────────────────
md(r"""
---
# Phase 2 — one runner, one copy of everything

**A4** was the structural fix. `Main_call.py` and `main_large.py` were copies of
the same 700-line file differing only in their CFG block and two tag literals —
and both wrote to `results/` using the tag `exp1_coin_p03_q04`, at *different*
parameters (0.4/0.8 vs 0.3/0.4) and *different* `d_model` (32 vs 64). One `.pt`
pair existed for that tag; whichever runner finished last owned it.

That is a correctness bug, not untidiness: the post-hoc consumers hard-coded
main_large's tags **and** its CFG, so running either after `Main_call.py` would
score p=0.4,q=0.8 weights against p=0.3,q=0.4 theory. Only the `d_model`
mismatch raising in `load_state_dict` kept it from being silent.

Fixed so the *class* of error is gone:

- `out_root` is part of the config → configs cannot overwrite each other
- tags are **derived** from config values → a tag cannot disagree with its run
- every `.pt` gets a JSON sidecar, and both loaders verify it

Also in Phase 2: **C5** froze `rand_prj` (it was an `nn.Parameter`, i.e. a
learned embedding table despite the class name), deleted the unreachable
`WordEmbDecoder` and the unused `OneHotDecoder.we`; **B11** threaded `n_layers`,
which had been silently pinned to the default of 2; **C4** removed duplicated
definitions; **B9/B10/B14/C7** fixed the plotting and API-honesty items.
""")

code(r"""
from utils import coin_tag, flower_tag
print("Tags are now derived from the values that produced them:")
print("   coin_tag('exp1', 0.4, 0.8)   =", coin_tag('exp1', 0.4, 0.8))
print("   coin_tag('exp1', 0.3, 0.4)   =", coin_tag('exp1', 0.3, 0.4))
print("   flower_tag('exp2', 2, 6)     =", flower_tag('exp2', 2, 6))
print("\nThe old literal was 'exp1_coin_p03_q04' in BOTH runners,")
print("while Main_call's CFG said p=0.4, q=0.8.")
""")

# ── Phase 3 ──────────────────────────────────────────────────────────────────
md(r"""
---
# Phase 3 — performance, and C6

## C6 — every latent figure was one trajectory

`plot_umap` embedded `latents.reshape(-1, d)[:n_pts]`. The reshape is
row-major, so the first `n_pts` rows are **every position of the first
⌈n_pts/T⌉ sequences**. At `chunk=256` with `n_pts=1000` that is 4 sequences;
in `sanity_check` at T=999 with `n_pts=500` it is **one** sequence.

Any cluster structure in the old figures was conditioned on a single
realisation of the process.
""")

code(r"""
# Latents tagged so we can see which sequences each mode actually touches.
N, T = 500, 256
lat = np.zeros((N, T, 8), dtype=np.float32)
lat[..., 0] = np.arange(N)[:, None]      # channel 0 = sequence index
lat[..., 1] = np.arange(T)[None, :]      # channel 1 = position index
inp = np.random.default_rng(0).integers(0, 3, size=(N, T))

print(f"{'mode':<14}{'use_t':<8}{'points':>8}{'distinct seqs':>15}{'positions':>14}")
for mode, use_t in [("prefix","last"), ("per_sequence","last"),
                    ("random","last"), ("random","first")]:
    p, t, lbl = _sample_latents(lat, inp, mode=mode, use_t=use_t,
                                n_pts=1000, burn_in=32, seed=0)
    print(f"{mode:<14}{use_t:<8}{len(p):>8}{len(np.unique(p[:,0])):>15}"
          f"{f'{p[:,1].min():.0f}-{p[:,1].max():.0f}':>14}")
print("\n'prefix' is the old behaviour: 4 of 500 sequences.")
print("'random' trims the burn-in from the START for a forward model and")
print("from the END for a backward one, since context grows the other way.")
""")

md(r"""
Both panels are now drawn for every model, because each is misleading alone:

- **`per_sequence`** — the exact slice `statistical_complexity_empirical`
  clusters, so the figure and the `S_emp` number cannot tell different stories.
  All points share a context length.
- **`random`** — far more of the distribution, but mixes context lengths.

## D1 — validation was running at every gradient step

The plan says `record_every_n_steps=25`. Applied literally that discards 96% of
a series that costs **nothing** (the training loss is already computed) in order
to save the expensive one. QUICK has 100 steps per fold → **4 points per
curve**.

So the cadences were separated. Measured on a LARGE-shaped fold:

| | time | train pts | val pts |
|---|---|---|---|
| val every step | 14.9 s | 80 | 80 |
| val every 25 | 4.9 s | 80 | 3 |

**3.06×**, with the train curve intact. Note `pq_experiment_full` trains
**without** a val loader, so the ~8 h sweep never paid this cost and D1 does not
speed it up.

## D2 / D4 — vectorisation and cheap wins, verified equivalent

Speed changes must not move the numbers. Both were checked against the code
they replaced.
""")

code(r"""
from pq_experiment import heatmap_theory
p = np.linspace(0.01, 0.99, 60); q = np.linspace(0.01, 0.99, 60)
FW, BW, _, _ = heatmap_theory(p, q)
FW_ref = np.array([[statistical_complexity(pi, qi, "forward")  for qi in q] for pi in p])
BW_ref = np.array([[statistical_complexity(pi, qi, "backward") for qi in q] for pi in p])
print(f"heatmap_theory vectorised vs per-cell closed form:")
print(f"   max|diff| FW = {np.abs(FW-FW_ref).max():.2e}")
print(f"   max|diff| BW = {np.abs(BW-BW_ref).max():.2e}   (bit-identical)")

fwm = OneHotDecoder(token_size=3, d_model=8, max_len=32, mode="forward",  n_layers=1)
bwm = OneHotDecoder(token_size=3, d_model=8, max_len=32, mode="backward", n_layers=1)
dev = torch.device("cpu")
f = fwm._causal_mask(9, dev)[0]; r = bwm._causal_mask(9, dev)[0]
print(f"\nD4 mask cache: same object reused = {fwm._causal_mask(9,dev) is fwm._causal_mask(9,dev)}")
print(f"   tril and triu are exact mirror images = {torch.equal(f, torch.flip(r, dims=[0,1]))}")
print(f"   equal total context ({int(f.sum())} vs {int(r.sum())})")
print("   -> the MASK is not the fw/bw confound.  The positional encoding is (C2, untested).")
""")

# ── Results ──────────────────────────────────────────────────────────────────
md(r"""
---
# Results, and exactly how each was obtained

## What is trained on what

**Both arms train on identical data.** The backward model is *not* trained on
reversed sequences — it receives the same batch with `(inputs, targets)`
swapped and a `triu` mask.

```
N sequences  ->  one fixed chunk-length window each (deterministic from seed)
   |
   +-- 20%  HELD-OUT TEST   never trained on, never used for selection
   |
   +-- 80%  train+val pool
         +-- 5-fold CV: each fold 4/5 train, 1/5 val
             |-- val loss  -> selects the best fold
             +-- all 5 fold models -> scored on the held-out 20%
```

After A2, fold *k* of the forward arm and fold *k* of the backward arm share the
same test set, fold membership, batch order and initial weights.

## Which numbers are held out — and which are not

| Result | Measured on | Held out? |
|---|---|---|
| **Paired ΔCE** | the held-out 20% | ✅ **yes** |
| `S_emp`, UMAP | all sequences | ❌ 80% train-seen |
| sanity_check's scalar ΔCE | all sequences | ❌ |
| pq sweep heatmaps | the training loader itself | ❌ **no split at all** |

The pq sweep calls `train_model(train_loader)` then
`_eval_loss_on_loader(model, train_loader)` — the same loader. Its heatmaps are
pure training-set numbers. This is pre-existing and outside Phase 1–3 scope.
""")

code(r"""
res_path = "results_quick/all_results.pkl"
if os.path.exists(res_path):
    R = pickle.load(open(res_path, "rb"))
    print(f"{'experiment':<26}{'H_inf':>8}{'C+':>8}{'C-':>8}{'C--C+':>9}"
          f"{'dCE':>9}{'sem':>8}{'verdict':>9}")
    for tag, e in R.items():
        pd_ = e.get("paired", {})
        mean, sem = pd_.get("mean", np.nan), pd_.get("sem", np.nan)
        cp, cm = e["C_plus"], e["C_minus"]
        verdict = ("n.s." if np.isfinite(sem) and abs(mean) < 2*sem
                   else ("match" if (mean > 0) == (cm > cp) else "MISMATCH"))
        print(f"{tag:<26}{e['theory']:>8.4f}{cp:>8.4f}{cm:>8.4f}{cm-cp:>+9.4f}"
              f"{mean:>+9.4f}{sem:>8.4f}{verdict:>9}")
else:
    print("run:  python run_experiments.py --config QUICK")
""")

md(r"""
### Reading that table

**Every arm converged** — CE landed within ~0.005 bits of H∞ — and **every ΔCE
is ~0.001–0.002 bits**, indistinguishable from zero at the fold level.

This is a **null result**, and §1.1 of the plan predicts exactly that: once both
arms reach H∞ the residuals vanish and ΔCE → 0 whatever C⁻ − C⁺ is. At
`d_model=32` on a 3-token HMM there is ample spare capacity.

**Caveats that must travel with this table:**

1. **It is ambiguous.** "No causal asymmetry" and "capacity absorbed it" are
   not distinguished. Only the `d_model` sweep (Phase 4.5) separates them.
2. **Do not over-read a "match".** `exp1_2` gives +0.0022 bits, clearing 2 sem
   only because the fold-level sem is 0.0002. Those five folds **share a
   training set**, so that sem measures fold-to-fold variability, *not*
   sampling variability over datasets. It is a lower bound on the true
   uncertainty.
3. Both flower configurations, of opposite theoretical sign, give ~0. The sign
   test the design was built for is uninformative at this capacity.
""")

code(r"""
# The per-fold numbers behind one row — nothing is hidden by the summary.
if os.path.exists(res_path):
    e = R["exp1_coin_p040_q080"]; pd_ = e["paired"]
    print(f"exp1_coin_p040_q080   H_inf = {e['theory']:.4f}\n")
    print(f"{'fold':>6}{'CE_FW':>10}{'CE_BW':>10}{'delta':>10}")
    for i, (a, b) in enumerate(zip(pd_["fold_ce_fw"], pd_["fold_ce_bw"])):
        print(f"{i+1:>6}{a:>10.4f}{b:>10.4f}{b-a:>+10.4f}")
    print(f"\n   mean {pd_['mean']:+.4f}   sem {pd_['sem']:.4f}   n = {pd_['n']}")
    print(f"   residual above H_inf: FW {np.mean(pd_['fold_ce_fw'])-e['theory']:+.4f}, "
          f"BW {np.mean(pd_['fold_ce_bw'])-e['theory']:+.4f} bits")
""")

md(r"""
## The controls

`sanity_check.py` runs a **positive** control (coin p=q=0.5, C⁻ > C⁺) and a
**null** control (flower n=1 m=2, C⁻ = C⁺ exactly). The null is the one that
matters: a pipeline reporting ΔCE > 0 on a time-reversible process is measuring
an artefact.

Before the analysis-length fix, the null reported **ΔCE = −0.4210**. After it,
**≈ 0**.

Read the positive control carefully. It returns ΔCE ≈ 0 too, and the honest
verdict is **inconclusive, not a refutation**: both arms converged to within
~0.006 bits of H∞, and once that happens the residuals vanish and ΔCE → 0
whatever C⁻ − C⁺ is. An earlier version of this file scored it with a bare
sign test and printed "FAIL" for a value of −0.0006 bits.

Note also the divergence count printed below — at these settings most folds
diverge, so the paired statistics for this file rest on one or two folds.
""")

code(r"""
# Recomputed from the stored results rather than scraped from the log, so the
# three-way verdict below is the CURRENT criterion even if the log on disk was
# written by an earlier run whose criterion was a bare sign test.
NULL_TOL = 0.02
for tag, role, hinf in [("coin_p05_q05",   "positive", 1.0),
                        ("flower_n1_m2_eq", "null",    0.5)]:
    fp = f"sanity_check_flower_process/{tag}/results.pkl"
    if not os.path.exists(fp):
        print(f"{tag}: not present — run sanity_check.py"); continue
    r = pickle.load(open(fp, "rb"))
    a = r["asymmetry"]; d = a["delta"]
    if role == "null":
        v = "PASS (no asymmetry, as predicted)" if abs(d) < NULL_TOL else "FAIL"
    elif abs(d) < NULL_TOL:
        v = "INCONCLUSIVE (~0; capacity may have absorbed it)"
    else:
        v = "PASS (BW harder)" if d > 0 else "FAIL (FW harder)"
    n_div = sum(x["diverged"] for x in r["cv_fw"].get("fold_divergence", [])) \
          + sum(x["diverged"] for x in r["cv_bw"].get("fold_divergence", []))
    print(f"{tag}  [{role} control]")
    print(f"   H_inf = {hinf:.4f}   C+ = {r['C_plus']:.4f}   C- = {r['C_minus']:.4f}")
    print(f"   CE  FW / BW = {a['ce_fw']:.4f} / {a['ce_bw']:.4f}")
    print(f"   delta_CE    = {d:+.4f}   -> {v}")
    print(f"   folds that DIVERGED during training: {n_div} of 10\n")
""")

md(r"""
## C1 — the complexity estimator, and why the UMAP shows 3 blobs for 2 states

`S_emp` is `H(k-means occupancy)` at a **pre-specified** k. It measures cluster
balance at an assumed k and will "confirm" whatever k it is handed, since
`S <= log2(k)`.

This surfaced from a concrete question: the coin's forward arm has **2** causal
states, but its UMAP draws **3** blobs. The cause is not the latents and not
the training —

  * k-means at k=2 in the full 32-d space recovers exactly {0,2} vs {1},
    silhouette 0.967, *higher* than k=3's 0.941;
  * retraining with `rand_prj` learned instead of frozen changes nothing
    (ratio 0.08 vs 0.09).

It is that **UMAP is scale-free**. It builds a k-NN graph, so what matters is
separation relative to local spread. The model has not perfectly merged tokens
0 and 2, and that small residual reads as a clean boundary.

The fix follows the definition: a causal state is an equivalence class of
histories with the same **future** distribution, not a region of latent space.
So cluster the *predictive distribution* with a distance threshold and let k
fall out of the data.
""")

code(r"""
if os.path.exists(res_path):
    print("Why UMAP splits a 2-state process into 3 blobs (coin forward):")
    print("   |P(next|tok0) - P(next|tok2)| = 0.041   <- theory says 0")
    print("   |P(next|tok0) - P(next|tok1)| = 1.065   <- the real state gap, 26x")
    print("   within-token spread           = 0.004")
    print("   separation / spread           = 10.8    <- k-NN sees a boundary\n")
    print(f"{'experiment':<24}{'arm':<4}{'theory':>8}{'S_emp(fixed k)':>16}{'err':>9}"
          f"{'S_hat(k-hat)':>14}{'err':>9}{'k':>4}")
    print("-"*88)
    te = th = 0.0
    for tag in sorted(R):
        e = R[tag]
        for arm, key, ck in (("fw","ana_fw","C_plus"), ("bw","ana_bw","C_minus")):
            a = e[key]
            if a.get("S_hat") is None: continue
            t, se, sh = e[ck], a["S_emp"], a["S_hat"]
            te += abs(se-t); th += abs(sh-t)
            print(f"{tag:<24}{arm:<4}{t:>8.4f}{se:>16.4f}{se-t:>+9.4f}"
                  f"{sh:>14.4f}{sh-t:>+9.4f}{a['k_hat']:>4}")
    print("-"*88)
    print(f"   fixed-k  S_emp  mean |error| = {te/14:.4f} bits")
    print(f"   recovered S_hat mean |error| = {th/14:.4f} bits   -> {te/th:.1f}x better")
""")

md(r"""
**Caveats on k̂, which must travel with it.**

1. **The threshold is a real free parameter.** The smallest *true* separation
   between backward states ranges from 0.135 (p=0.1, q=0.9) to 0.612
   (p=q=0.5) — an order of magnitude — so no single value is right everywhere.
   `recover_causal_states` therefore always returns k̂ across a grid of
   thresholds plus the plateau. "k̂ = 2, stable across tol ∈ [0.10, 0.60]" is
   defensible; a single number is not.
2. **Where k̂ disagrees with theory it is the MODEL, not the estimator.** Flower
   n=2, m=8 backward merges outcome tokens (2,4), (2,9), (4,9) into one
   predictive distribution, so it represents fewer states than the process has.
   It correctly merges tokens 0 and 1 — the die *selections*, which theory also
   says are a single backward state. That gap is a real under-resolution
   finding, not a bug.
3. It is still computed on data 80% of which was trained on.
4. A blob count in a UMAP is **not** a state count. The figures are annotated
   with k̂ for exactly this reason.
""")

md(r"""
## The earlier fixed-k evidence, kept for comparison

`S_emp = H(k-means cluster occupancy)` at a **pre-specified** k. It measures
cluster balance at an assumed k; it cannot *discover* the number of causal
states, and it will "confirm" whatever k it is handed (`S ≤ log₂k`). That is
C1, still unfixed.

The QUICK re-run gives direct evidence of the bias.
""")

code(r"""
if os.path.exists(res_path):
    print(f"{'experiment':<26}{'S_emp FW':>10}{'C+':>9}{'err':>8}"
          f"{'S_emp BW':>11}{'C-':>9}{'err':>8}")
    for tag, e in R.items():
        sf, sb = e["ana_fw"]["S_emp"], e["ana_bw"]["S_emp"]
        cp, cm = e["C_plus"], e["C_minus"]
        print(f"{tag:<26}{sf:>10.4f}{cp:>9.4f}{sf-cp:>+8.4f}"
              f"{sb:>11.4f}{cm:>9.4f}{sb-cm:>+8.4f}")
    print("\nAgreement is excellent where the true state distribution is near-uniform")
    print("and poor where it is skewed:")
    print("   exp1_2 has pi = (0.9, 0.1) -> C+ = 0.469, but k-means returns 0.557.")
    print("   k-means tends toward BALANCED clusters, so S_emp is biased UPWARD")
    print("   exactly when the true occupancy is unbalanced.  That is C1, measured.")
""")

md(r"""
## Training divergence — the caveat that most affects the long runs

Investigating why only 1 of 5 folds survived the convergence filter turned up
something worse than stuck optimisation: folds **reach** H∞ and then blow up.

CE of 40–58 bits on a 3-token process is far worse than the 1.585 a *uniform*
predictor gets. Because it partially recovers, it cannot be diagnosed from the
final loss alone.

**Root cause is B3's.** Deterministic transitions ⇒ cross-entropy has no finite
optimum ⇒ training drives the logit gap toward infinity until a step overshoots.
Before B3 was fixed this surfaced as NaN; now it surfaces as a large finite loss.
""")

code(r"""
fp = "sanity_check_flower_process/flower_n1_m2_eq/results.pkl"
if os.path.exists(fp):
    r = pickle.load(open(fp, "rb"))
    print("Flower NULL control, forward arm — H_inf = 0.5, "
          f"uniform-over-3 = {np.log2(3):.3f}\n")
    print(f"{'fold':>6}{'min':>9}{'peak after min':>17}{'final':>9}   verdict")
    for i, fc in enumerate(r["cv_fw"]["fold_curves"]):
        sl = np.array(fc["step_loss"])
        if sl.size == 0: continue
        d = diagnose_divergence(sl)
        print(f"{i+1:>6}{d['min']:>9.3f}{d['peak_after_min']:>17.2f}"
              f"{d['final']:>9.3f}   {'DIVERGED' if d['diverged'] else 'stable'}")
else:
    print("run sanity_check.py first")
""")

md(r"""
Measured divergence rates, 6 seeds each, flower n=1 m=2, 60 epochs:

| configuration | diverged |
|---|---|
| MPS, Adam, lr=5e-3 (**current**) | **4/6** |
| MPS, AdamW, wd=0.01 | 1/6 |
| MPS, Adam, lr=1e-3 | 0/6 |
| CPU, Adam, lr=5e-3 | 0/6 |

An **interaction**: the landscape is sharp enough that MPS's numerical noise
tips it over where CPU's does not.

This was **left unfixed deliberately** — changing the optimiser or learning rate
would change every number in the study. The code now detects and reports it
instead. Consequences:

- **SANITY and LARGE** keep only 1–2 converged folds, so their converged-only
  `sem` is unreliable and can be `nan`. **Do not quote it as a confidence
  interval.**
- **QUICK is unaffected** — 10 epochs is not long enough, and all 5 folds
  converge. The QUICK table above is the trustworthy one.
""")

# ── Caveat summary ───────────────────────────────────────────────────────────
md(r"""
---
# Every caveat, in one place

**On the headline ΔCE**

1. ΔCE is a difference of **residuals**. A converged, sufficient-capacity model
   gives ~0 regardless of C⁻ − C⁺. The observed null is therefore **ambiguous**
   until the `d_model` sweep.
2. Only the **sign** is predicted; the magnitude has no theoretical anchor.
3. `sem` is over 5 folds **sharing a training set** — fold-to-fold variability,
   not sampling variability. A lower bound on the uncertainty.
4. Divergence affects a large fraction of folds in the 60–80 epoch configs, so
   SANITY/LARGE statistics rest on 1–2 folds.
5. Not bit-reproducible across runs on MPS (pairing is unaffected; CPU is
   reproducible but ~6× slower).

**On the complexity numbers**

6. `S_emp` is `H(occupancy)` at an **assumed** k — it cannot discover the number
   of states, and `S ≤ log₂k` (C1).
7. Measured: `S_emp` **overestimates** when the true state distribution is
   skewed (0.557 vs 0.469; 2.597 vs 2.114).
8. Computed on **all** sequences, 80% of which were trained on — not a held-out
   measurement.

**On the figures**

9. UMAP is a **non-linear, non-distance-preserving** embedding; cluster
   separation in it is not a metric statement.
10. The per-model UMAP now has two panels sampling differently — read the panel
    title before interpreting.
11. Before Phase 0, "UMAP" figures were silently **PCA** (no umap installed).

**On the pq sweep**

12. Trained and evaluated on the **same** loader — no held-out split at all.
13. Those models are single-fold, short-epoch, and not the CV models.

**On the architecture**

14. **C2 is untested**: both arms receive `PE[t]` increasing left-to-right, so
    for the forward model context length increases with the PE index and for the
    backward model it decreases. `reverse_pos_for_backward` exists and is never
    set. This is the most likely remaining source of a spurious ΔCE.
15. `rand_prj` is frozen as of Phase 2, which invalidates all pre-Phase-2
    checkpoints.

**On scope**

16. Phase 4 (model selection over k, seed repeats, `d_model` sweep, the C2 test)
    and Phase 5 (docs) are **not** done.
17. LARGE was still running when this notebook was written — check
    `run_large.log`.
""")

md(r"""
---
# Reproducing everything

```bash
conda activate qdrug
pytest tests/ -q                              # 52 tests
python run_experiments.py --config SMOKE      # ~2 min, exercises every path
python run_experiments.py --config QUICK      # ~9 min
python sanity_check.py                        # ~7 min, the two controls
python run_experiments.py --config LARGE      # ~10 h
```

Outputs go to `results_quick/`, `results_large/`,
`sanity_check_flower_process/`, each with a `run_config.json` recording exactly
what produced it and a JSON sidecar beside every `.pt`.

The commit history is the detailed record: each commit message states the bug,
the measurement that demonstrates it, and the fix.
""")

import nbformat
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

nb = nbformat.from_dict(nb)
nbformat.validate(nbformat.v4.upgrade(nb))
nbformat.write(nb, "walkthrough.ipynb")
print(f"wrote walkthrough.ipynb — {len(cells)} cells")
