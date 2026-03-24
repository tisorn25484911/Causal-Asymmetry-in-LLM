# Thermodynamic Consequences of Causal Asymmetry in Transformer Language Models

This repository empirically tests the **causal asymmetry** hypothesis from Thompson et al. (2017) using transformer decoder models trained on Hidden Markov Model (HMM) sequences.

The central question: if a stochastic process requires more memory to predict in one causal direction than the other (i.e. C+ ≠ C−), do transformer models trained in the harder direction incur a measurably higher cross-entropy loss at convergence?

---

## Background

For a stationary stochastic process, the **statistical complexity** C+ (forward) and C− (backward) measure the minimum memory required by an optimal predictor in each direction. When C+ < C− the process is **causally asymmetric** — the forward direction is intrinsically easier to model.

The key theoretical prediction (Thompson et al. 2017):

```
CE_FW → H∞ + residual_FW
CE_BW → H∞ + residual_BW
delta_CE = CE_BW − CE_FW > 0   (backward is harder)
```

This repository tests that prediction empirically on two processes:

- **Coin HMM** — a two-state HMM parameterised by flip probabilities p, q with 3 observable tokens
- **Flower HMM** — an n-die, m-face process where forward causal states = n+1 and backward causal states = m+1

---

## Repository Structure

```
.
├── Data_generation.py           # Coin HMM and Flower process sequence generators
├── Flower_process_generation.py # Extended Flower process DataLoader
├── OneHot_model.py              # OneHotDecoder transformer (forward + backward)
├── Training_model.py            # Training pipeline, CV, callbacks
├── Model_analysis.py            # All analysis functions (perplexity, UMAP, complexity)
├── pq_experiment.py             # p-q grid sweep for heatmap experiments
│
├── main.py                      # Quick experiment runner (~1.5 hr)
├── main_large.py                # Overnight high-quality runner
├── Main_call.py                 # Extended main with additional plots
│
├── LLM_asymmetry_testing.py     # Causal asymmetry test suite (all metrics)
├── Test_data_eval.py            # Model evaluation on freshly generated data
│
├── run_experiments.ipynb        # Interactive experiment launcher
├── sequence_prediction.ipynb    # Token-by-token prediction visualisation
├── umap_analysis.ipynb          # UMAP latent space analysis
└── testing.ipynb                # Scratch / development notebook
```

---

## Model Architecture

Both forward and backward models use the same `OneHotDecoder` class with a single flag changing the causal structure.

```python
# Forward model: causal (lower-triangular) mask
model_fw = OneHotDecoder(token_size=3, d_model=64, mode="forward")

# Backward model: anti-causal (upper-triangular) mask
model_bw = OneHotDecoder(token_size=3, d_model=64, mode="backward")
```

**Critical design principle**: both models are always trained on the **same forward-data DataLoader**. Only the attention mask and batch convention differ. This is the principled comparison — same data, different causal structure.

The backward model's `training_step` swaps the batch so it predicts `x[:-1]` given `x[1:]`, learning to predict each token from its future context.

| | Forward model | Backward model |
|---|---|---|
| Attention mask | Lower-triangular (tril) | Upper-triangular (triu) |
| Position t attends to | x[0..t] (past) | x[t..T] (future) |
| Max-context position | logits[:, -1, :] | logits[:, 0, :] |
| Context grows | Rightward | Leftward |
| Causal states (coin) | C+ = 2 | C− = 3 |

---

## Experiments

### Experiment 1 — Coin HMM (p=0.3, q=0.4)

Baseline comparison. Both models trained on the same coin sequences, evaluated across 5-fold cross-validation.

Expected: `delta_CE > 0` (BW harder), empirical C− > C+ matching theory.

### Experiment 1.2 — Coin HMM p-q heatmap

Sweeps a grid of (p, q) values to show where causal asymmetry is strongest in parameter space. Produces complexity and perplexity difference heatmaps comparing empirical values to theoretical predictions.

### Experiment 2 — Flower HMM (n=6, m=4)

Tests a more complex process where forward causal states = n+1 = 7 and backward causal states = m+1 = 5. For this process C− < C+ (backward direction is actually easier), providing an important control case where the asymmetry reverses.

---

## Metrics

Four metrics are computed per experiment, in increasing strength of signal:

| Metric | Function | What it measures |
|---|---|---|
| Perplexity on ground-truth | `perplexity_calculation` | CE loss on same sequences for both models — primary asymmetry signal |
| CE vs true conditional | `perplexity_ind_CE` | CE against the analytical HMM conditional at every step on ground-truth sequences |
| Stepwise KL | `stepwise_kl_coin` | D_KL(P_true(·\|cur_tok) || P_model) averaged over all positions |
| Stationary KL | `kl_from_true_coin` | D_KL(pi_true || pi_model) — weaker sanity check on token frequencies |

`perplexity_calculation` is the primary metric. It evaluates both models on the same ground-truth sequences using the correct batch-swap convention for the backward model.

### Note on `perplexity_ind_CE`

An earlier autoregressive version of this function had a critical bug: for the backward model, context was grown leftward using backward HMM transitions. This gave the model out-of-distribution right-context (backward dynamics) while training always used forward-sequence context. The biased context made the backward model appear artificially accurate, reversing the asymmetry signal. The corrected version uses the ground-truth loader so both models see forward-sequence context exactly as during training.

### Statistical complexity

Empirical complexity is estimated by k-means clustering of the model's latent representations at the max-context position:

```python
# Forward model: last position has maximum past context
S_emp_fw = statistical_complexity_empirical(model_fw, loader, use_t="last", k=2)

# Backward model: first position has maximum future context
S_emp_bw = statistical_complexity_empirical(model_bw, loader, use_t="first", k=3)
```

---

## Outputs

Running `main.py` produces, per experiment:

```
results/
├── models/
│   ├── exp1_coin_p03_q04_fw.pt            # best forward model weights
│   └── exp1_coin_p03_q04_bw.pt            # best backward model weights
├── exp1_coin_p03_q04/
│   ├── *_fw_cv.png                        # forward model CV training curves
│   ├── *_bw_cv.png                        # backward model CV training curves
│   ├── *_umap_fw.png                      # forward latent UMAP coloured by token
│   ├── *_umap_bw.png                      # backward latent UMAP coloured by token
│   ├── *_complexity_heatmap.png           # empirical vs theoretical complexity
│   └── *_diff_ppl.png                     # delta perplexity heatmap
└── all_results.pkl
```

Running `LLM_asymmetry_testing.py` additionally produces:

```
results/asymmetry_test/exp1_coin_p03_q04/
├── *_dual_ppl.png                         # 4-panel: both perplexity metrics FW vs BW
├── *_kl_distribution.png                  # true vs model stationary token distribution
├── *_stepwise_kl.png                      # per-token stepwise KL breakdown
└── *_complexity.png                       # empirical + theoretical C+ / C−
```

---

## Quick Start

```bash
# Install dependencies
pip install torch lightning numpy matplotlib scikit-learn umap-learn

# Quick run (~1.5 hours, 500 samples, 15 epochs)
python main.py

# Overnight high-quality run (2000 samples, 80 epochs, 16-pt p-q grid)
python main_large.py

# Causal asymmetry test suite (requires trained models in results/models/)
python LLM_asymmetry_testing.py

# Evaluate models on freshly generated data with UMAP and cumulative PPL
python Test_data_eval.py
```

Or run interactively:

```bash
jupyter notebook run_experiments.ipynb
```

---

## Configuration

All experiment parameters are in the `CFG` dict at the top of each runner file:

```python
CFG = dict(
    d_model         = 64,     # transformer hidden dimension
    n_layers        = 2,      # attention + FFN layers
    lr              = 5e-3,
    train_chunk_len = 512,    # sequence chunk length (OOM prevention)
    n_folds         = 5,      # cross-validation folds

    # Coin HMM
    coin_p1         = 0.3,  coin_q1 = 0.4,
    coin_num_samples = 1500,
    coin_seq_len    = 2000,
    coin_max_epochs = 60,

    # Flower HMM
    flower_n = 6, flower_m = 4,

    # p-q heatmap
    pq_grid = [0.05, 0.15, ..., 0.95],  # 12 points → 144 models
    pq_epochs = 20,
)
```

The overnight config in `main_large.py` uses 2000 samples, 80 epochs, and a 16-point p-q grid (256 models per direction).

---

## Causal State Counts

| Process | Forward C+ | Backward C− | Asymmetry direction |
|---|---|---|---|
| Coin HMM (any p, q) | 2 states | 3 states | C− > C+ (forward easier) |
| Flower n=6, m=4 | 7 states (n+1) | 5 states (m+1) | C+ > C− (backward easier) |

The Flower experiment serves as a control: theory predicts `delta_CE < 0` (FW harder), the opposite of the coin result.

---

## Key Design Decisions

**Same DataLoader for both models.** Any difference in training data would confound the comparison. The backward model receives the same sequences as the forward model — only the mask and batch swap differ.

**Chunked training, full-length analysis.** Full T=2000 sequences allocate O(T²) memory per attention layer. `ChunkedDataset` samples random 512-token windows per batch. However, `max_len` is always set to the full sequence length so the positional encoding table covers the entire range — analysis on full sequences remains in-distribution.

**`use_t="last"` for forward, `"first"` for backward in complexity estimation.** The causal mask means maximum context is accumulated at the last position for forward models and the first position for backward models. Clustering at any other position mixes different context lengths and produces noisy estimates.

**`perplexity_calculation` as the primary metric.** It uses the existing batch-swap convention, evaluates both models on the same ground-truth data, and directly produces CE values comparable to H∞. It is the most direct and unambiguous measure of causal asymmetry in this setting.

---

## Reference

Thompson, J., Garner, A. R., Mahoney, J. R., Vanner, M. R., Gu, M., & Modi, K. (2017). Causal asymmetry in a quantum world. *Physical Review X*, 8(3), 031013.
