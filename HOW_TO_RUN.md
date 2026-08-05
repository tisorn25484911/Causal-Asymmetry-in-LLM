# How to run

Everything runs from this directory with the `qdrug` environment:

```bash
conda activate qdrug
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version
```

---

## 1. The one command you probably want

```bash
python run_experiments.py --config QUICK
```

~9 minutes. Trains and analyses all four QUICK experiments, writes everything to
`results_quick/`, and prints an asymmetry summary at the end.

---

## 2. `run_experiments.py` — the training runner

This is the **only** file that trains the main experiments. Everything is
selected by flag; nothing needs editing.

```
--config {SMOKE,QUICK,LARGE,QUICK_LARGE_HMM}   which configuration (default QUICK)
--seed  N                                      override the seed (default 0)
--out-root DIR                                 override the output directory
--only  {all,exp1,exp1_2,exp2}                 run one experiment instead of all
```

### The configurations

| `--config` | writes to | d_model | lr | epochs | steps/fold | time | use it for |
|---|---|---|---|---|---|---|---|
| `SMOKE` | `results_smoke/` | 16 | 1e-2 | 3 | ~15 | **~2 min** | checking nothing is broken |
| `QUICK` | `results_quick/` | 32 | 1e-2 | 10 | 100 | **~9 min** | **the reportable results** |
| `QUICK_LARGE_HMM` | `results_quick/` | 32 | 1e-2 | 10 | 100 | ~3 min | LARGE's processes at QUICK's settings |
| `LARGE` | `results_large/` | 64 | 5e-3 | 80 | 1600 | **~3.5 h** | scale, but see the warning |

### What each experiment inside a config is

| `--only` | what it trains |
|---|---|
| `exp1` | coin HMM at `(coin_p1, coin_q1)`, forward + backward, 5-fold CV |
| `exp1_2` | coin HMM at `(coin_p2, coin_q2)`, **plus the p-q sweep** (the slow part) |
| `exp2` | one flower experiment per entry in `flower_configs` — currently two |
| `all` | all of the above (default) |

`exp1_2` is what makes LARGE take 3.5 h: the sweep trains 512 extra models
(2 × 16 × 16) and is ~2.5 h of that on its own. If you only want the asymmetry
numbers, skip it:

```bash
python run_experiments.py --config LARGE --only exp1
python run_experiments.py --config LARGE --only exp2
```

### Examples

```bash
# a second seed, so you get more than one draw of the effect
python run_experiments.py --config QUICK --seed 1 --out-root results_quick_seed1

# just the flower experiments
python run_experiments.py --config QUICK --only exp2

# add LARGE's processes to the QUICK folder (already done once)
python run_experiments.py --config QUICK_LARGE_HMM --only exp1
python run_experiments.py --config QUICK_LARGE_HMM --only exp2
```

### Re-running is safe

`all_results.pkl` is **merged by tag**, not overwritten, so `--only` will not
destroy experiments it did not run. Tags are derived from the parameter values
(`exp1_coin_p040_q080`), so re-running the same experiment replaces its own
entry and nothing else. Each config also drops a `run_config_<CONFIG>.json`
recording exactly what produced the directory.

---

## 3. `sanity_check.py` — the two controls

```bash
python sanity_check.py
```

~7 minutes, writes to `sanity_check_flower_process/`.

**No CLI flags** — settings are the `CFG` dict at the top of the file
(line ~106), and `NULL_TOL` (line ~150) is the band inside which a null result
counts as upheld.

Runs two controls:

- **positive** — coin p=q=0.5, where C⁻ = 1.5 > C⁺ = 1.0, so ΔCE > 0 is predicted
- **null** — flower n=1, m=2, which is *exactly time-reversible* (C⁻ = C⁺ = 1.0),
  so ΔCE ≈ 0 is predicted

The null is the one that matters. A pipeline reporting ΔCE > 0 there is
measuring an artefact, not causal asymmetry.

---

## 4. `plot_state_clusters.py` — causal-state figures

```bash
python plot_state_clusters.py
```

Run this **after** a training run. It reads the saved weights, so it does not
retrain. Writes `results_quick/<tag>/<tag>_states_<metric>.png` — one 2×2 figure
per experiment per metric:

```
forward / TRAIN        backward / TRAIN
forward / HELD-OUT     backward / HELD-OUT
```

```
--config CFG        which config's experiments to plot (default QUICK)
--also-config CFG   a second config sharing the same out_root (default QUICK_LARGE_HMM, '' to skip)
--out-root DIR      where to read weights and write figures
--metrics ...       any of: euclidean tv js   (default: all three)
```

```bash
python plot_state_clusters.py --out-root results_large --metrics js
```

---

## 5. Post-hoc evaluators

```bash
python Test_data_eval.py
python LLM_asymmetry_testing.py
```

These load saved `.pt` weights and score them on freshly generated data. **No
CLI** — edit the `RUN` dict near the bottom of each file:

```python
RUN = dict(
    config      = "LARGE",   # which configs.py entry produced the weights
    results_dir = None,      # None -> that config's out_root
    out_dir     = None,      # None -> <results_dir>/eval
    exp         = "all",     # "all" | "exp1" | "exp1_2" | "exp2"
)
```

Set `config` to match the run whose weights you want to score, or the sidecar
check will warn you that the architecture does not match.

---

## 6. Tests and the notebook

```bash
pytest tests/ -q            # 61 tests, ~20 s
python build_walkthrough.py # regenerate walkthrough.ipynb
```

Run the tests after any change to `Model_analysis.py`, `Training_model.py` or
`OneHot_model.py` — they pin the fixes that are silent when broken.

---

## 7. What lands on disk

```
results_quick/
├── run_config_QUICK.json              exactly what produced this directory
├── run_config_QUICK_LARGE_HMM.json
├── all_results.pkl                    every experiment, merged by tag
├── models/
│   ├── exp1_coin_p040_q080_fw.pt      weights
│   └── exp1_coin_p040_q080_fw.json    sidecar: d_model, n_layers, p, q, seed...
└── exp1_coin_p040_q080/
    ├── results.pkl                    metrics, per-fold curves, k_hat, S_hat
    ├── ..._fw_cv.png                  5-fold CV curves
    ├── ..._loss_theory.png            loss vs H_inf
    ├── ..._fw_umap.png                latent UMAP, 3 panels
    ├── ..._umap_compare.png           forward vs backward
    ├── ..._states_{euclidean,tv,js}.png   causal states, train vs held-out
    └── ..._attn*.png                  attention heatmaps
```

Everything except the `.py` files is gitignored — regenerate rather than commit.

---

## 8. Two warnings before you trust a run

**LARGE's cross-validation numbers are not usable.** In the completed 3.5 h run,
**40 of 40 folds diverged** — every fold reached H∞ and then blew up. Its paired
ΔCE has sem 0.11–0.15 and one `nan`. The cause is measured: it is the **number
of gradient steps**, not the data scale. Cross-entropy on a deterministic
transition has no finite optimum, so logits grow without bound (2.9 → 8.7 → 12.8
over 800 steps) while the loss sits flat at H∞, until a step overshoots. QUICK's
100 steps/fold stops before that window opens; LARGE's 1600 is well past it.
The weights, figures and p-q heatmaps from LARGE are still fine — only the
CV/ΔCE numbers are not.

**QUICK is the run to report.** All 5 folds converge, on all 7 experiments.

If you want usable LARGE numbers, the fix is `AdamW(weight_decay=0.01)`, which
restores a finite optimum (measured 1/6 divergence vs 4/6). That would change
every number, so QUICK and SANITY would need re-running on the same optimiser
to stay comparable.
