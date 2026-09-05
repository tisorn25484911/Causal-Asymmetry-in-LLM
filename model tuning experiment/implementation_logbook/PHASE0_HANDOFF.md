# Phase 0 — Handoff

Date: 2026-08-04
Branch: `phase0-unblock` (3 new commits, not pushed)
Follows: `IMPROVEMENT_PLAN.md` §7 Phase 0

**Status: complete except the `git filter-repo` history rewrite**, which is
left for you to run deliberately — it force-pushes over published history.
Exact procedure in §5.

---

## 1. A note on which plan this followed

The plan was rewritten from v1 to v2 partway through this work, and v2 changed
Phase 0. The reconciliation:

| v1 said | v2 says | What was done |
|---|---|---|
| 0.1 Build a fresh `conda create -n asym python=3.12` | §3.1: obsolete — use the existing `qdrug` env, the notebooks already point at its kernel | Built `asym` first, then switched to `qdrug`. See §2. |
| 0.2 `.gitignore` + commit deletions + maybe filter-repo | 0.1 same, filter-repo now explicit | Done except filter-repo |
| 0.3 Stop pickling `all_recorders` / `latents` | 0.1 same | Done, §4 |
| — | 0.3 **new**: D5, `L.Trainer(logger=False, enable_checkpointing=False)` | Done, §4 |

Commit `603aaa8` ("Phase 0.1: pin a working environment") was written under v1
and describes the `asym` env. Commit `d5644fd` supersedes it with `qdrug`.
The message on `603aaa8` is now partly stale — harmless, but don't read it as
the current environment story.

---

## 2. Environment

**Use `qdrug`.** `/opt/anaconda3/envs/qdrug`, Python 3.11.14.

```
conda activate qdrug
pip install -r requirements.txt      # already satisfied; listed for reproduction
```

Verified working there: torch 2.12.1, lightning 2.6.5, numpy 2.4.2,
scipy 1.17.1, scikit-learn 1.8.0, umap-learn 0.5.12, numba 0.66.0,
matplotlib 3.10.8, pandas 3.0.1.

### The one thing that was actually broken in `qdrug`

v2 §3.1 calls `qdrug` "the working environment". It was — for training. But it
had **no `umap` and no `numba` at all**, which means:

```python
# Model_analysis.py:13-23
try:
    import umap as _umap_mod
    ...
except Exception:
    UMAP_AVAILABLE = False        # ← this is the branch qdrug was taking
    print("UMAP unavailable — PCA fallback active")
```

So every "UMAP" latent plot produced from `qdrug` was silently a **PCA** plot.
This is worth checking before you trust any existing latent figure: `_project2d`
returns a label and `plot_umap_tokens` puts it in the title, so open the saved
PNGs and look at whether the title says `(UMAP)` or `(PCA)`. Any that say
`(PCA)` were made under the fallback.

Fixed by `pip install umap-learn` into `qdrug`. Purely additive — it pulled in
`llvmlite`, `numba`, `pynndescent`, `umap-learn` and downgraded nothing
(confirmed with `--dry-run` first, since `qdrug` is shared with your quantum
work).

### Leftover: the `asym` env

`/opt/anaconda3/envs/asym` (1.3 GB) was built under v1's now-obsolete
instruction. Nothing references it. It is a clean py3.12 room if you want one;
otherwise:

```
conda env remove -n asym
```

---

## 3. Git hygiene

`.gitignore` added: `lightning_logs/`, `results/`, `sanity_check_flower_process/`,
`__pycache__/`, `*.pt`, `*.ckpt`, `*.pkl`, `.DS_Store`, `.claude/settings.local.json`,
`*.code-workspace`.

5830 files dropped from the index (`git rm --cached`, so **everything is still
on disk** — `results/` 4.5 GB, `sanity_check_flower_process/` 1.4 GB,
`lightning_logs/` 33 MB are all untouched). Tracked file count went 5845 → 20:
11 `.py`, 3 notebooks, README, IMPROVEMENT_PLAN, requirements ×2, `.gitignore`.

Your own in-flight edits were deliberately **not** committed and are still
sitting in the working tree: `IMPROVEMENT_PLAN.md` (the v2 rewrite),
`metric_panel_plot.ipynb`, `sequence_prediction.ipynb`, `testing.ipynb`, and
untracked `umap_analysis.ipynb`.

---

## 4. Code changes

### `Model_analysis.py` — new `slim_results(obj)`

The measured breakdown of `sanity_check_flower_process/coin_p05_q05/results.pkl`
(700.8 MB), by re-pickling each component:

| | MB |
|---|---|
| `ana_fw` + `ana_bw` | 495.4 |
| ├ `latents` (500, 999, 64) float32, ×2 arms | 243.8 |
| ├ `flat_lat` (499500, 64) — *the same array again*, ×2 arms | 243.8 |
| └ `flat_inp`, ×2 arms | 7.6 |
| `cv_fw` + `cv_bw` | 206.2 |
| └ `all_recorders` (5 fold models + recorders), ×2 arms | 206.2 |

Two things worth knowing beyond what the plan says:

1. **The latents are stored twice.** `latents` and `flat_lat` are the same data,
   and pickle does not dedupe them because `flat_lat` is a reshape, not the same
   object. Half of the 495 MB is pure duplication.
2. **The 81.6 MB "model" is mostly not weights.** The saved `.pt` files are
   0.3 MB. The rest is stale `last_attention` / `last_encodings` activation
   buffers still hanging off each module from the last forward pass — which is
   audit item D4 showing up as a disk problem, not just a memory one.

`slim_results` drops models, recorder objects and latent arrays; lifts the
per-step curves off the recorders into plain lists (`best_curves`,
`fold_curves`); keeps every scalar, per-fold metric, UMAP coordinate and
complexity number. It recurses, so it works on both a single `res` and on
`all_results`. It is **non-destructive** — returns a copy, so callers can still
use `cv_*["best_model"]` for `save_weights` afterwards.

Measured: **700.8 MB → 0.3 MB (2669×)**, with metrics, curves and UMAP
coordinates verified identical after a round-trip.

Wired into all 9 save sites across `Main_call.py`, `main_large.py`,
`sanity_check.py`. The runners now also store the *slim* copy in `all_results`,
which frees each experiment's models and latents instead of holding all three
arms in memory at once (`all_results` is only membership-tested and pickled, so
nothing else needed it).

`save_pkl` now prints the size it wrote, so a regression here is visible in the
run log instead of only on disk.

### `Training_model.py` — D5

`L.Trainer(..., logger=False, enable_checkpointing=False)`. This is the only
`L.Trainer` construction in the repo, so it covers every runner.

### What was deliberately *not* touched

Everything in Phase 1 and later. In particular `math.exp(loss)` at
`Training_model.py:147` is still there — it is Phase 1's first item, and
changing it here would have mixed a numbers-affecting fix into a hygiene commit.

---

## 5. Not done: `git filter-repo`

`.git` is still **6.2 GB**. `.gitignore` stops the bleeding but does not touch
history. The blobs, by on-disk (compressed) size from
`git cat-file --batch-check`:

```
1345.7 MB  results/all_results.pkl
 562.6 MB  results/training set/exp1_2_coin_p01_q09/results.pkl
 453.1 MB  results/training set/exp1_coin_p03_q04/results.pkl
 451.3 MB  results/training set/exp2_flower_n6_m4/results.pkl
 441.3 MB  results/training set/exp1_2_coin_p04_q08/results.pkl
```

(v2 §6 quotes 1873/843/639/631/603 MB — those are the uncompressed blob sizes.
Both are right; the ~3.2 GB above is what the pack actually costs you.)

`git-filter-repo` is **not installed**. Procedure:

```bash
# 0. Back up first. This is the step you cannot skip.
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA
cp -R LLM_final_version LLM_final_version.backup    # ~12 GB; you have ~45 GB free

# 1. Install
pip install git-filter-repo        # or: brew install git-filter-repo

# 2. Purge. Run from the repo root, on a clean tree.
cd LLM_final_version
git filter-repo --invert-paths \
  --path 'results/all_results.pkl' \
  --path-glob 'results/training set/*/results.pkl'

# 3. filter-repo deliberately drops the remote. Put it back:
git remote add origin https://github.com/tisorn25484911/Causal-Asymmetry-in-LLM.git

# 4. Check the damage is gone
du -sh .git        # expect tens of MB, not 6.2 GB

# 5. Only when you are satisfied:
git push --force --all origin
```

**Before you run step 5**, know what it costs: every commit hash changes, so any
existing clone of this repo — including one on another machine of yours — can no
longer pull and has to be re-cloned. There are currently **9 unpushed commits**
on `phase0-unblock` and `origin/main` still carries the blobs, so nobody else is
depending on the rewritten range yet. This is about as clean a moment to do it
as you will get.

---

## 6. Things noticed that belong to later phases

Not acted on, but confirmed while working — these are free confirmations of the
audit, from Pylance on the actual files:

- `Main_call.py:360` — `"cfg" is not defined`. The static checker agrees with the
  audit that `cfg.get(...) if 'cfg' in dir() else 200` always takes the `else`.
- `Main_call.py:422, 496, 624, 625` — `loader_fw_ana` / `loader_bw_ana` assigned
  and never read. Confirms the analysis loaders are built and discarded.
- `Model_analysis.py:735` — `fig` assigned and never used, i.e. the unreachable
  figure in `FW_BW_loss_comparison` after the bare `return`.
- `Model_analysis.py:166, 511` — the local re-imports of `PCA` / `KMeans` inside
  functions that already import them at module top (audit 4.3.10).

---

## 7. How to verify any of this yourself

```bash
conda activate qdrug
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version

# every module imports
python -c "import Model_analysis, Training_model, Main_call"

# UMAP is real now, not the PCA fallback
python -c "import Model_analysis"     # prints "umap-learn JIT warm-up succeeded"

# slimming works on a real bundle
python - <<'PY'
import pickle, io
from Model_analysis import slim_results
r = pickle.load(open("sanity_check_flower_process/coin_p05_q05/results.pkl","rb"))
b = io.BytesIO(); pickle.dump(slim_results(r), b, protocol=4)
print(f"{b.tell()/1024**2:.1f} MB slim")
PY

# D5: a training run should add no lightning_logs/version_* dirs
ls -d lightning_logs/version_* | wc -l    # before and after
```

---

## 8. Next

Phase 1 (`IMPROVEMENT_PLAN.md` §7), in its stated order: A2 and A1 first — they
decide whether the numbers mean anything — then B1/B2 for the scale. Then re-run
everything; per both v1 and v2, the results currently in `results/` were produced
with those bugs live and should not be reported.
