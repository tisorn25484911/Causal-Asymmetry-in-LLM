# Repository Review & Improvement Plan

Full read-through of every Python file in `LLM_final_version/`, dated 2026-08-04.
Covers: what the repo does, how the modules fit together, where to run it, a
function-by-function audit (bugs, conceptual weaknesses, optimisations, rename
candidates), and a phased plan for continued experimental development.

---

## 1. What the repo does

You train two transformer decoders on **the same forward HMM data**. The only
difference is the attention mask: the forward model gets `tril` (attends to the
past), the backward model gets `triu` (attends to the future), and the backward
model's `training_step` swaps the batch so it predicts `x[:-1]` from `x[1:]`.

The hypothesis under test (Thompson et al. 2017): if a process requires more
memory to predict backward than forward (C− > C+), the backward model should
converge to a measurably higher cross-entropy.

```
CE_FW → H∞ + residual_FW
CE_BW → H∞ + residual_BW
delta_CE = CE_BW − CE_FW > 0   (backward is harder)
```

---

## 2. Module flow

```
Data_generation.py            coin_generation, flower_process_generation, CoinDataset
Flower_process_generation.py  parametric FlowerDataset (duplicate of the one above)
         │
OneHot_model.py               PositionalEncoding → AttentionModel → OneHotDecoder
                              cross_ent_onehot   (log2 → loss is in BITS)
         │
Training_model.py             ChunckDataset      (random 256/512-token windows)
                              Record_training    (Lightning callback)
                              train_model → train_test_val_pipeline (5-fold CV + hold-out test)
         │
Model_analysis.py             latent_extraction, plot_umap, perplexity_calculation,
pq_experiment.py              statistical_complexity{,_empirical}, heatmaps
         │
RUNNERS  →  Main_call.py, main_large.py, sanity_check.py
POST-HOC →  LLM_asymmetry_testing.py, Test_data_eval.py   (load results/models/*.pt)
```

---

## 3. Where to run

| Command | What it is |
|---|---|
| `python sanity_check.py` | **The most correct runner.** Coin p=q=0.5 + Flower n=1,m=2 → `sanity_check_flower_process/` |
| `python Main_call.py` | Quick full suite (exp1, exp1.2 + pq sweep, exp2) → `results/` |
| `python main_large.py` | The same file with a bigger CFG — only ~73 diff lines, all config |
| `python LLM_asymmetry_testing.py` | Post-hoc metric suite on saved `.pt` weights |
| `python Test_data_eval.py` | Autoregressive generation, cumulative PPL, UMAP on fresh data |

Configuration lives in the `CFG = dict(...)` block at the top of each runner.

> **Nothing runs right now.** The `/opt/anaconda3` base env has numpy 2.4.6
> against scipy 1.12, so `import scipy` / `sklearn` / `umap` raises
> `numpy.core.multiarray failed to import` — `Model_analysis.py` will not even
> import. The Python 3.12 interpreter that produced `__pycache__/*.cpython-312.pyc`
> was not locatable on this machine. Fix this before anything else.

---

## 4. Function audit

### 4.1 Real bugs

| Where | Problem |
|---|---|
| `Training_model.py:147` | `math.exp(loss)` — but `cross_ent_onehot` uses `log2`, so PPL must be `2**loss`. The train-PPL curve is base *e* while the val-PPL curve beside it (via `_eval_loss_on_loader`) is base 2. Panels 3 and 4 of every CV figure are not comparable. |
| `Training_model.py:53-79` | `_eval_loss_on_loader` averages `2**CE` **per batch** instead of exponentiating the token-weighted mean CE. Jensen bias, plus the ragged last batch gets equal weight. |
| `Main_call.py:456,526,658` | Backward model analysed with `use_t="last"`. With a `triu` mask, maximum context is at position **0**. The inline comment says "forward model → last pos", which is not what that model is. `sanity_check.py` correctly uses `"first"`. The BW complexity numbers currently in `results/` were read at the *minimum*-context position. |
| `Main_call.py:421,495,623` | `loader_fw_ana` is built and **never used** — `analyse_model` receives `loader_fw`, the 256-token chunked *training* loader. UMAP and complexity therefore run on chunks, contradicting the README's "full-length analysis" design principle. |
| `Main_call.py:425` | `max_len = chunk`, contradicting the README's stated principle that `max_len` is always the full sequence length. The PE table auto-grows so it does not crash, but the design intent is broken. |
| `Main_call.py:358-359` | `cfg.get(...) if 'cfg' in dir() else 200` — `cfg` is not in scope and `dir()` with no args returns local names, so this **always** evaluates to 200. Fake configurability. |
| `Main_call.py:406` | `tag = "exp1_coin_p03_q04"` while `CFG` sets p=0.4, q=0.8. Every exp1 output directory, PNG and pickle is mislabeled. |
| `Model_analysis.py:410,541` | `plt.show()` inside `plot_perplexity` and `statistical_complexity_compare`, but every runner sets `matplotlib.use("Agg")` → those figures are silently discarded, never saved. |
| `Model_analysis.py:693` | `FW_BW_loss_comparison` builds a figure then hits a bare `return` — the figure is unreachable. Its x-labels say "Epoch" for step-indexed data, and panel 2's title says "Backward Model Training Loss" when it actually plots the difference. |
| `Training_model.py` (train_model) | `n_layers` is never plumbed through, so every trained model is 2-layer regardless of config — yet `LLM_asymmetry_testing.load_model` reads `cfg["n_layers"]`. Silently fine at 2, silently wrong if changed. |

### 4.2 Conceptual weaknesses / doesn't make sense

- **`statistical_complexity_empirical` cannot measure what it claims.** With `k`
  fixed by hand, `S = H(cluster sizes) ≤ log2(k)`. It measures cluster *balance*
  at a pre-specified k — it cannot discover the number of causal states, and it
  will "confirm" whatever k you pass. Comparing FW at k=2 against BW at k=3
  compares two different ceilings. This is the weakest link in the complexity
  story and the first thing a reviewer will attack.
- **`perplexity_ind_model` scores the model on its own samples** → that is the
  model's entropy rate, which a degenerate model minimises. `LLM_asymmetry_testing`
  flags it "informational only"; the name does not.
- **`FlowerDataset` and `flower_process_generation` are each defined twice** —
  `Data_generation.py` (hardcoded n=4, m=2; no `pre_depth`) versus
  `Flower_process_generation.py` (parametric; has `pre_depth`). `Main_call.py`
  imports the *generator* from one file and the *Dataset* from the other.
- **`coin_true_conditional` + `stepwise_kl_coin` are copy-pasted verbatim** into
  both `Model_analysis.py` and `LLM_asymmetry_testing.py`. (The conditional
  matrices were checked by hand against the HMM — the maths is correct in both
  copies, but they will drift.)
- **`OneHotDecoder.we`** is an `nn.Embedding` that is never used — dead
  parameters, trained into nothing, serialised into every `.pt`.
- **`rand_prj` is an `nn.Parameter` in `OneHotDecoder` but a buffer in
  `WordEmbDecoder`**, despite the comment "fixed random projection". So "onehot"
  mode is in fact a *learned* embedding table — precisely what `WordEmbDecoder`
  claims to be. The two classes are far closer than their names suggest.
- **`make_loader`**: `seqs, _ = data, states` is a no-op assignment that makes the
  `states` argument look meaningful; it is ignored in forward mode.
- **No global seeding anywhere.** `coin_generation` uses global `np.random`; only
  the fold permutation is seeded. Runs are not reproducible.
- **`pq_experiment` (the non-`_full` variant) is dead code** — both runners call
  `pq_experiment_full`, which duplicates the entire training loop.

### 4.3 Optimisation opportunities

1. **Biggest win by far:** `Record_training` runs a **full validation pass at
   every gradient step** (`record_every_n_steps=1`, `Training_model.py:225`).
   That is ~n_steps × n_val_batches extra forward passes and dominates
   wall-clock. Set it to 25–50.
2. `perplexity_ind_CE` and `stepwise_kl_coin` loop `for b: for t:` in pure Python
   over every token (~10⁶ iterations per call). Fully vectorisable → 100–1000×.
3. `cross_ent_onehot` does `softmax().log2()` — use `F.cross_entropy(...) / ln(2)`:
   identical value, faster, and no `-inf` on saturated logits.
4. `self.last_attention` retains a `(B, T, T)` tensor on **every** forward pass.
   At B=32, T=2000 that is ~512 MB held after each call. Gate it behind a flag.
5. The `tril`/`triu` mask is rebuilt on every forward — cache per `(T, device)`.
6. `heatmap_theory` double-loops a 100×100 grid in Python for a closed-form
   expression — vectorise with numpy.
7. `perplexity_calculation` concatenates all logits on CPU before computing the
   loss; accumulate CE incrementally instead.
8. The UMAP JIT warm-up runs at *import* of `Model_analysis.py`, **and again** in
   `Main_call.py`, **and again** in `sanity_check.py` — with `n_neighbors=200` on
   20 points. Make it lazy and do it once.
9. `plot_attention_heatmap` and `FW_BW_attention_comparison` call the model
   without `torch.no_grad()` — they build an autograd graph for a plot.
10. Local re-imports of `PCA` and `KMeans` inside functions that already import
    them at module top.

### 4.4 Rename candidates

| Current | Suggested | Why |
|---|---|---|
| `ChunckDataset`, `chunck_len` | `ChunkDataset`, `chunk_len` | typo |
| `flower_process_generation(seq_len=N)` | `n_cycles=N` | produces **2N** tokens; `flower_seq_len=2000` yields 4000-token sequences |
| `statistical_complexity` | `coin_statistical_complexity` | coin-only, silently wrong if called for flower |
| `statistical_complexity_empirical` | `latent_cluster_entropy` | it is H(k-means assignment) at fixed k, not complexity |
| `perplexity_ind_CE` | `soft_label_ce_ppl_coin` | coin-specific (takes p, q); "ind" means nothing |
| `perplexity_ind_model` | `self_generated_entropy_ppl` | scores its own samples |
| `perplexity_calculation` | `teacher_forced_ppl` | says nothing about the convention it uses |
| `_eval_loss_on_loader` | `evaluate_ce_ppl` | imported by 4 modules — should not be `_`-private |
| `_loader` | `full_sequence_loader` | ditto; docstring "FIX-3 only" is meaningless |
| `_sub` | `take_first_n` | it is not a subsample but a prefix — and that biases every UMAP |
| `test_train_validation` | `holdout_split` | word order backwards; it makes no validation set |
| `train_test_val_pipeline` | `cross_validate` | "pipeline" says nothing |
| `Rev_HMM_generation` | `reverse_sequences` | does not generate, does not touch an HMM |
| `Record_training` | `MetricRecorder` | |
| `cross_ent_onehot` | `cross_entropy_bits` | "onehot" is an implementation detail to delete |
| `pq_experiment_full` | `pq_sweep` | |
| `Main_call.py` | `run_experiments.py` | the README already points at files that do not exist (`main.py`, `run_experiments.ipynb`) |

---

## 5. Improvement plan

### Phase 0 — Unblock (do first, ~1 hour)

1. Create a clean environment (`conda create -n asym python=3.12`), install
   dependencies, freeze to `requirements.txt`. The numpy/scipy break blocks
   everything else.
2. Add `.gitignore`: `lightning_logs/`, `results/`, `sanity_check_flower_process/`,
   `__pycache__/`, `*.pt`, `*.ckpt`, `*.pkl`, `.DS_Store`. `.git` is currently
   **6.2 GB** because checkpoints and a 1.8 GB `all_results.pkl` were committed.
   Commit the 5745 pending deletions, then decide whether to `git filter-repo`
   them out of history — worth it, the repo is effectively unclonable today.
3. Fix why the pickles are enormous: `save_pkl(res, ...)` stores `cv_fw` (all five
   fold models plus recorders) **and** `ana_fw["latents"]` (an N×T×d_model array).
   Save metrics and best weights only; drop `all_recorders` and `latents` before
   pickling.

### Phase 1 — Trust the numbers (~half a day)

4. Fix `math.exp` → `2**` in `Record_training`, and make `_eval_loss_on_loader`
   token-weighted.
5. Fix `use_t="first"` for the backward model in `Main_call.py` / `main_large.py`;
   pass `loader_*_ana` to `analyse_model`; set `max_len` to the full sequence
   length. Use `sanity_check.py` as the reference — it already does all three
   correctly.
6. Fix the exp1 tag; delete the `'cfg' in dir()` line.
7. Add a `set_seed(seed)` called at the top of every runner, seeding `numpy`,
   `torch`, and `ChunckDataset.rng`.
8. **Re-run everything.** Results currently in `results/` were produced with bugs
   4–6 and should not be reported.

### Phase 2 — Deduplicate (~half a day)

9. Delete `Data_generation.FlowerDataset` and
   `Data_generation.flower_process_generation`; keep the parametric versions in
   `Flower_process_generation.py`.
10. Delete `coin_true_conditional` / `stepwise_kl_coin` from
    `LLM_asymmetry_testing.py`; import them from `Model_analysis.py`.
11. Delete `pq_experiment` (keep `_full`), and `WordEmbDecoder` if unused.
12. **Collapse `main_large.py` into `Main_call.py`.** They are one file with two
    configs. Move `CFG` into a `configs.py` exposing `QUICK` / `LARGE` / `SANITY`
    dicts, selected via an `--config` argparse flag. This removes the worst
    maintenance hazard in the repo: every fix currently has to be applied twice.

### Phase 3 — Make it fast (~half a day)

13. `record_every_n_steps=25` in `train_model`. Expect the largest single speedup.
14. Vectorise `perplexity_ind_CE`, `stepwise_kl_coin`, `heatmap_theory`.
15. `F.cross_entropy / ln2`; gate `last_attention` behind a flag; cache masks.

### Phase 4 — Strengthen the science (where new experiments live)

16. Replace fixed-k clustering with **model selection over k** (silhouette / BIC /
    elbow) so the estimator can actually *recover* C+=2 versus C−=3 instead of
    being told the answer. Report the recovered k alongside the entropy.
17. Add a **seed-repeat harness**: N seeds per condition, report mean ± CI on
    `delta_CE`, and a paired test. A single `delta_CE` from one best-fold model is
    not evidence.
18. Add a **null control**: an i.i.d. process where C+ = C−, which must yield
    `delta_CE ≈ 0`. Without it you cannot rule out that the tril/triu asymmetry
    itself (position 0 sees 1 token, position T sees T) produces the gap.
19. Sweep `d_model` — the theory is about *memory-bounded* predictors, so
    `delta_CE` should shrink as capacity grows. That curve is a far stronger
    result than a single point.
20. Add a `tests/` directory: assert the conditional matrices sum to 1 and match a
    Monte-Carlo estimate; assert `entropy_rate_coin(0.5, 0.5) == 1.0`; assert the
    flower generator emits alternating select/roll tokens; assert
    `2**cross_ent_onehot(uniform_logits) == vocab_size`.

### Phase 5 — Docs

21. Update the README: `main.py` and `run_experiments.ipynb` do not exist; add
    `sanity_check.py`; correct the `perplexity_ind_CE` note; state explicitly that
    the flower `seq_len` is measured in cycles, not tokens.

---

## 6. Do these three today

- **Phase 0.1** — the environment, because nothing runs without it.
- **Phase 1.4–1.5** — the PPL base and `use_t` bugs, because they affect numbers
  you may already have written up.
- **Phase 3.13** — the per-step validation pass, because it is costing hours per run.