
"""
    Experiment Plan:
    Experiment 1: Train forward and backward models on the coin process data. (500 samples, each len = 2000)
        - Train forward model on original coin data (observations in {0,1,2}).
            - 5 fold cross-validation to ensure robustness. 
            - plot Loss and perplexity curves (training and validation)
            - **get the best model from the 5 folds for analysis.
            - Analyze the best forward model using:
                - Attention heatmaps on sample sequences.
                - UMAP visualization of latent representations.
                - Statistical complexity estimation (empirical and theoretical).
                - *Approximation into Automata*
        - Train backward model on reversed coin data.
            - Train forward model on original coin data (observations in {0,1,2}).
            - 5 fold cross-validation to ensure robustness. 
            - plot Loss and perplexity curves (training and validation)
            - **get the best model from the 5 folds for analysis.
            - Analyze the best forward model using:
                - Attention heatmaps on sample sequences.
                - 2D UMAP visualization of latent representations.
                - Statistical complexity estimation (empirical and theoretical).
                - *Approximation into Automata*
        - Analyze and compare models using:
            - Loss and perplexity curves from *the best model in each fold*.
            - *calculate the theoretical convergence of the loss function and compare with the trained models' loss curves*
            - Attention heatmaps on the same sample sequences.
            - 2D UMAP visualizations of latent representations on the same sample sequences with the same. plotting range for size comparison.
            - Statistical complexity comparison (empirical and theoretical).
    Experiment 1.2: Train forward and backward models on the coin process data with different parameters (p = 0.1, q = 0.9). (500 samples, each len = 500)
        - Repeat the same steps as in Experiment 1 for the new coin process data.
        - calculate the empirical and theoretical statistical complexity for the new coin process and compare with the previous one (plot a heat map).
        - plot a heat map of "the difference in statistical complexity" for the new coin process and compare with the previous one (plot a heat map)
        - plot a heat map of "the difference in perplexity" between forward and backward models across different (p, q) values on the same 2000 seq long test data.

    Experiment 2: Train forward and backward models on the flower process data. (n = 4, m = 2, 500 samples, each len = 2000)
        - Train forward model on original coin data (observations in {0,1,2}).
            - 5 fold cross-validation to ensure robustness. 
            - plot Loss and perplexity curves (training and validation)
            - **get the best model from the 5 folds for analysis.
            - Analyze the best forward model using:
                - Attention heatmaps on sample sequences.
                - 2D UMAP visualization of latent representations.
                - Statistical complexity estimation (empirical and theoretical).
                - *Approximation into Automata*
        - Train backward model on reversed coin data.
            - Train forward model on original coin data (observations in {0,1,2}).
            - 5 fold cross-validation to ensure robustness. 
            - plot Loss and perplexity curves (training and validation)
            - **get the best model from the 5 folds for analysis.
            - Analyze the best forward model using:
                - Attention heatmaps on sample sequences.
                - 2D UMAP visualization of latent representations.
                - Statistical complexity estimation (empirical and theoretical).
                - *Approximation into Automata*
        - Analyze and compare models using:
            - Loss and perplexity curves from *the best model in each fold
            - *calculate the theoretical convergence of the loss function and compare with the trained models' loss curves*
            - Attention heatmaps on the same sample sequences.
            - 2D UMAP visualizations of latent representations on the same sample sequences with the same. plotting range for size comparison.
            - Statistical complexity comparison (empirical and theoretical).
"""
# ── stdlib ─────────────────────────────────────────────────────────────────
import gc
import os
import pickle
import time

# ── third-party ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as tud

# ── project ────────────────────────────────────────────────────────────────
from Data_generation import (
    CoinDataset,
    Rev_HMM_generation,
    coin_generation,
    flower_process_generation,
)
from Flower_process_generation import FlowerDataset          # FIX-5
from Model_analysis import (
    mkdir,
    save_pkl,
    slim_results,
    _sub,
    savefig,
    save_weights,
    _project2d,
    flower_complexity,
    flower_entropy_rate,
    paired_delta_ce,
    plot_umap,
    plot_diff_heatmap,
    FW_BW_attention_comparison,
    latent_extraction,
    plot_attention_heatmap,
    statistical_complexity,
    statistical_complexity_empirical,
)
from Training_model import (
    ChunckDataset, _loader, make_chunked_loader, set_seed,
    train_test_val_pipeline,
)
from pq_experiment import heatmap_theory, plot_heatmap, pq_experiment, pq_experiment_full

# ── UMAP — warm up JIT here so segfault (if any) happens at startup ───────
# FIX-2: pre-compile numba kernels before training begins
try:
    import umap as _umap_mod
    _warmup = _umap_mod.UMAP(n_components=2, n_neighbors=20).fit_transform(
        np.random.rand(20, 4)
    )
    del _warmup
    UMAP_AVAILABLE = True
    print("umap-learn JIT warm-up succeeded")
except Exception as _e:
    UMAP_AVAILABLE = False
    print(f"UMAP unavailable ({_e}) — PCA fallback active")


# ══════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════
CFG = dict(
    # ── reproducibility (A2) ───────────────────────────────────────────
    seed          = 0,
    # ── model ──────────────────────────────────────────────────────────
    d_model       = 32,
    embed_type    = "onehot",
    n_folds       = 5,
    lr            = 1e-2,
    # ── FIX-1: limit T to avoid O(T²) OOM ─────────────────────────────
    train_chunk_len = 256,   # tokens seen per training step
    attn_vis_len    = 64,    # tokens used for attention heatmap plots
    # Analysis runs on FULL sequences (B4), where attention is O(T^2).  A
    # smaller batch there keeps the transient (B, T, T) score matrix bounded:
    # at T=1999, B=8 is ~0.12 GB per layer instead of ~0.48 GB at B=32.
    ana_batch       = 8,
    # B7: 200 neighbours on 1000 points smears the local cluster structure the
    # plot exists to show.  This value was also unreachable until B7 was fixed.
    umap_n_neighbors = 15,
    umap_n_pts       = 1000,
    # ── coin exp 1 ─────────────────────────────────────────────────────
    coin_p1          = 0.4,  coin_q1          = 0.8,
    coin_num_samples = 500,  coin_seq_len     = 2000,
    coin_max_epochs  = 10,   coin_batch       = 32,
    coin_num_token   = 3,
    # ── coin exp 1.2 ───────────────────────────────────────────────────
    coin_p2             = 0.1,  coin_q2             = 0.9,
    coin_num_samples_12 = 500,  coin_seq_len_12     = 500,
    # ── flower exp 2 ───────────────────────────────────────────────────
    # Two configurations, because the sign of delta_CE should track the sign
    # of (C- - C+) -- IMPROVEMENT_PLAN.md A1.  C- <= 1 + (1/2)log2(m) while
    # C+ = 1 + (1/2)log2(n), so C- > C+ requires m > n.  Every flower config
    # originally in this repo had n > m, i.e. all of them tested the negation
    # of the hypothesis while being labelled as confirming it.
    flower_configs = [
        # (n, m, role)   role drives the predicted sign, not the analysis
        (2, 6, "positive"),   # C+=1.500  C-~2.111   predict delta_CE > 0
        (4, 2, "reversed"),   # C+=2.000  C-~1.495   predict delta_CE < 0
    ],
    flower_num_samples = 500, flower_seq_len    = 2000,
    flower_max_epochs  = 10,  flower_batch      = 32,
    flower_dice_seed   = 42,
    # ── pq heatmap ─────────────────────────────────────────────────────
    pq_grid   = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    pq_epochs = 5,  pq_samples = 500,  pq_len = 200,
)


# ══════════════════════════════════════════════════════════════════════════
#  safe cleanup + CPU-offload for analysis
# ══════════════════════════════════════════════════════════════════════════

def cleanup():
    """
    GC + accelerator flush.
    NOTE: torch.mps.empty_cache() segfaults on PyTorch ≤ 2.1 (known Apple bug).
    We intentionally do NOT call it.  Models are moved to CPU before analysis
    (via to_cpu_for_analysis) so the Metal heap is freed that way instead.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # MPS: only synchronise — never call empty_cache
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def to_cpu_for_analysis(model: torch.nn.Module) -> torch.nn.Module:
    """
    Move a trained model to CPU in-place and return it.

    Why: torch.mps.empty_cache() segfaults on PyTorch ≤ 2.1 on macOS.
    Moving the model to CPU is the only reliable way to release the Metal
    heap between the forward and backward analysis passes.  CPU inference
    on analysis batches is fast enough (< 5 s per pass).
    """
    model.cpu()
    gc.collect()
    # Best-effort MPS sync — ignore if not available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    return model


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def coin_tag(prefix: str, p: float, q: float) -> str:
    """
    Derive an experiment tag from the parameters it was actually run at.

    IMPROVEMENT_PLAN.md B8/A4.  Tags used to be string literals, so
    Main_call.py wrote "exp1_coin_p03_q04" while its CFG said p=0.4, q=0.8 --
    main_large's parameters.  Deriving the tag makes that class of error
    impossible: a tag can no longer disagree with the config that produced it.
    p=0.4, q=0.8 -> "exp1_coin_p040_q080".
    """
    return f"{prefix}_coin_p{round(p*100):03d}_q{round(q*100):03d}"


def flower_tag(prefix: str, n: int, m: int) -> str:
    return f"{prefix}_flower_n{n}_m{m}"


def full_seq_len(dataset) -> int:
    """
    Length of the model's input sequence, i.e. len(seq) - 1 given that the
    Datasets emit (x[:-1], x[1:]).  Used for max_len so the positional
    encoding table covers the whole range at analysis time (B5).
    """
    inp, _ = dataset[0]
    return int(inp.shape[0])


def entropy_rate_coin(p: float, q: float) -> float:
    """
    Entropy rate H∞ (bits/token) of the coin-process HMM.
    H∞_forward = H∞_backward  (time-reversal invariance of entropy rate).
    H∞ = π₀·H₂(p) + π₁·H₂(q),  π₀=q/(p+q),  π₁=p/(p+q).
    """
    def _h2(a):
        b = 1.0 - a
        a, b = max(a, 1e-12), max(b, 1e-12)
        return -a * np.log2(a) - b * np.log2(b)
    pi0 = q / (p + q);  pi1 = p / (p + q)
    return pi0 * _h2(p) + pi1 * _h2(q)


def plot_loss_theory(rec_fw, rec_bw, theory_fw, theory_bw, title="", save_path=None):
    """
    3-panel loss figure:
      Left  : FW train loss + BW train loss on the same axes (like reference image 3)
      Middle: Loss difference BW − FW (like reference image 4)
      Right : Individual val-loss curves with theory H∞ lines
    """
    # Align lengths (different folds may produce slightly different step counts)
    sl_fw = np.array(rec_fw.step_loss)
    sl_bw = np.array(rec_bw.step_loss)
    min_len = min(len(sl_fw), len(sl_bw))
    sl_fw, sl_bw = sl_fw[:min_len], sl_bw[:min_len]
    steps = np.arange(min_len)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # ── Panel 1: FW + BW training loss overlaid ──────────────────────────
    ax0 = axes[0]
    ax0.plot(steps, sl_fw, color="steelblue",  lw=1.2, alpha=0.85, label="Forward Loss")
    ax0.plot(steps, sl_bw, color="darkorange", lw=1.2, alpha=0.85, label="Backward Loss")
    if theory_fw == theory_fw:   # not NaN
        ax0.axhline(theory_fw, color="crimson", ls="--", lw=1.8, label=f"H∞={theory_fw:.4f}")
    ax0.set_xlabel("Training Steps"); ax0.set_ylabel("Loss (bits)")
    ax0.set_title("Training Loss Comparison", fontweight="bold")
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)

    # ── Panel 2: BW − FW difference ──────────────────────────────────────
    ax1 = axes[1]
    diff = sl_bw - sl_fw
    ax1.plot(steps, diff, color="crimson", lw=1.2, alpha=0.9, label="Loss difference (BW − FW)")
    ax1.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax1.set_xlabel("Training Steps"); ax1.set_ylabel("Loss Difference")
    ax1.set_title("Loss Difference (BW − FW)", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # ── Panel 3: Val loss + theory lines ─────────────────────────────────
    ax2 = axes[2]
    for rec, th, lbl, col in zip(
        [rec_fw, rec_bw], [theory_fw, theory_bw],
        ["FW val", "BW val"], ["steelblue", "darkorange"]
    ):
        if rec.step_val_loss:
            ax2.plot(rec.step_val_loss, color=col, lw=1.0, alpha=0.75, label=lbl)
        if th == th:
            ax2.axhline(th, color=col, ls="--", lw=1.5, alpha=0.6, label=f"H∞ {lbl[:2]}={th:.4f}")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Val Loss (bits)")
    ax2.set_title(f"Validation Loss — {title}", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


def analyse_model(tag, model, loader, num_token, out_dir,
                  sample_seq=None, p=None, q=None, mode="forward",
                  k=2, use_t="last", attn_vis_len=64,
                  umap_n_pts=1000, umap_n_neighbors=15):
    """
    `loader` must be the FULL-SEQUENCE analysis loader, not the chunked
    training loader — IMPROVEMENT_PLAN.md B4.  README:233 states the design
    ("Chunked training, full-length analysis... analysis on full sequences
    remains in-distribution"); the runners used to build loader_*_ana and then
    hand analyse_model the chunked loader instead, so UMAP and complexity ran
    on 256-token windows.

    `use_t` selects the position the complexity is read at, and it must match
    the direction (A3).  With a tril mask the last position has seen the whole
    past; with a triu mask position t attends to [t, T-1], so the LAST position
    attends to itself alone and maximum context is at position 0.
    """
    res = {"tag": tag}

    # ── Move to CPU so Metal heap is freed before analysis ────────────────
    to_cpu_for_analysis(model)
    # Also move sample_seq to CPU
    if sample_seq is not None:
        sample_seq = sample_seq.cpu()
    if sample_seq is not None:
        try:
            fig_a = plot_attention_heatmap(model, sample_seq[:attn_vis_len])
            savefig(fig_a, os.path.join(out_dir, f"{tag}_attn.png"))
        except Exception as e:
            print(f"  attn failed: {e}")

    try:
        latents, inp_arr, _ = latent_extraction(model, loader, max_batches=20)
        #is_bw = (getattr(model, "mode", "forward") == "backward")
        lat_slice = latents.reshape(-1, latents.shape[-1])
        inp_slice = inp_arr.reshape(-1)
        lat_for_plot = lat_slice
        inp_for_plot = inp_slice
        _, coords = plot_umap(lat_for_plot, inp_for_plot, num_token, title=tag,
                              save_path=os.path.join(out_dir, f"{tag}_umap.png"),
                              n_pts=umap_n_pts, n_neighbors=umap_n_neighbors)
        res.update({"latents": latents, "inputs_arr": inp_arr, "umap_coords": coords})
    except Exception as e:
        print(f"  UMAP failed: {e}")
        res.update({"latents": None, "inputs_arr": None, "umap_coords": None})

    # Statistical complexity — model on CPU so no MPS issue inside the fn
    try:
        S_emp = statistical_complexity_empirical(model, loader, max_batches=20,
                                                 use_t=use_t, k=k)
        res["S_emp"] = S_emp
        if p is not None and q is not None:
            S_th = statistical_complexity(p, q, mode=mode)
            res["S_theory"] = S_th
            print(f"  [{tag}] S_emp={S_emp:.4f}  S_theory={S_th:.4f}  "
                  f"err={abs(S_emp-S_th):.4f}")
        else:
            print(f"  [{tag}] S_emp={S_emp:.4f}")
    except Exception as e:
        print(f"  complexity failed: {e}")
        res["S_emp"] = float("nan")
    return res


def compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw, loader_ana, num_token, out_dir,
                  sample_seq, theory_fw, theory_bw, attn_vis_len=64, p=None, q=None,
                  cfg=None):
    """
    `loader_ana` is the full-sequence analysis loader (B4).  The old signature
    took both loader_fw and loader_bw; every call site passed loader_fw twice
    and the loader_bw parameter was never read, so it has been removed.

    `cfg` is now a real parameter (B7).  The UMAP call used to read
    `cfg.get('umap_n_neighbors', 200) if 'cfg' in dir() else 200` — dir() with
    no argument lists the *local* scope, cfg was neither a parameter nor a
    local, and dir() would not see a global anyway, so the condition was always
    False and the value was always the hard-coded 200.
    """
    cfg = cfg or {}
    n_nbr = cfg.get("umap_n_neighbors", 15)
    n_pts_cfg = cfg.get("umap_n_pts", 1000)
    mfw = cv_fw["best_model"];  mbw = cv_bw["best_model"]
    rfw = cv_fw["best_recorder"];  rbw = cv_bw["best_recorder"]

    # Ensure both models are on CPU (they should already be from analyse_model)
    to_cpu_for_analysis(mfw)
    to_cpu_for_analysis(mbw)
    if sample_seq is not None:
        sample_seq = sample_seq.cpu()

    # (a) loss + theory
    plot_loss_theory(rfw, rbw, theory_fw, theory_bw, title=tag,
                     save_path=os.path.join(out_dir, f"{tag}_loss_theory.png"))

    # (b) attention side-by-side (short prefix — FIX-1)
    try:
        fig_a = FW_BW_attention_comparison(mfw, mbw, sample_seq[:attn_vis_len],
                                           title_prefix=f"{tag} ")
        savefig(fig_a, os.path.join(out_dir, f"{tag}_attn_compare.png"))
    except Exception as e:
        print(f"  attn compare failed: {e}")

    # (c) side-by-side UMAP of the two models' latents on the same data.
    # Both arms see the same forward-generated sequences; they differ in the
    # attention mask and the batch convention.  Latents from every position are
    # flattened together here (the per-direction max-context slice is what the
    # complexity estimator uses, via use_t).
    try:
        lfw, ifw, _ = latent_extraction(mfw, loader_ana, max_batches=20)
        lbw, ibw, _ = latent_extraction(mbw, loader_ana, max_batches=20)
        lat_fw = lfw.reshape(-1, lfw.shape[-1])
        lat_bw = lbw.reshape(-1, lbw.shape[-1])
        inp_fw = ifw.reshape(-1)
        inp_bw = ibw.reshape(-1)
        n_pts = min(n_pts_cfg, len(lat_fw), len(lat_bw))
        fl_fw, fl_bw = lat_fw[:n_pts], lat_bw[:n_pts]
        si_fw, si_bw = inp_fw[:n_pts], inp_bw[:n_pts]
        c_fw, mlbl = _project2d(fl_fw, n_neighbors=n_nbr)
        c_bw, _    = _project2d(fl_bw, n_neighbors=n_nbr)

        # Independent projections — no shared axis range
        cmap = plt.cm.tab10
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, coords, raw_inp, lbl in zip(
            axes, [c_fw, c_bw], [si_fw, si_bw],
            [f"Forward model ({mlbl})", f"Backward model ({mlbl})"]
        ):
            for tok in range(num_token):
                mask = raw_inp == tok
                if not mask.any(): continue
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=[cmap(tok / max(num_token - 1, 1))],
                           label=f"Token {tok}", alpha=0.7, s=10)
            ax.set_title(lbl, fontsize=13, fontweight="bold")
            ax.legend(fontsize=8, markerscale=3)
            ax.grid(True, alpha=0.2)
        fig.suptitle(tag, fontsize=12, fontweight="bold")
        fig.tight_layout()
        savefig(fig, os.path.join(out_dir, f"{tag}_umap_compare.png"))
    except Exception as e:
        print(f"  UMAP compare failed: {e}")

    # (d) complexity bar
    if p is not None and q is not None:
        try:
            fw_v = [ana_fw.get("S_emp", 0), ana_fw.get("S_theory", 0)]
            bw_v = [ana_bw.get("S_emp", 0), ana_bw.get("S_theory", 0)]
            x = np.arange(2)
            fig_c, ax_c = plt.subplots(figsize=(8, 5))
            ax_c.bar(x-0.2, fw_v, 0.35, label="Forward",  color="#4c72b0", alpha=0.85)
            ax_c.bar(x+0.2, bw_v, 0.35, label="Backward", color="#dd8452", alpha=0.85)
            ax_c.set_xticks(x); ax_c.set_xticklabels(["Empirical", "Theoretical"])
            ax_c.set_ylabel("Statistical Complexity (bits)")
            ax_c.set_title(f"{tag} — complexity comparison")
            ax_c.legend(); ax_c.grid(True, alpha=0.3, axis="y")
            fig_c.tight_layout()
            savefig(fig_c, os.path.join(out_dir, f"{tag}_complexity_compare.png"))
        except Exception as e:
            print(f"  complexity bar failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Coin HMM  p=0.4, q=0.8
# ══════════════════════════════════════════════════════════════════════════
def experiment_1(cfg, out_root, all_results):
    p, q = cfg["coin_p1"], cfg["coin_q1"]
    # B8: derive the tag from the parameters instead of hard-coding it.  The
    # literal used to be "exp1_coin_p03_q04" while CFG said p=0.4, q=0.8, so
    # this runner's output was labelled with main_large's parameters.
    tag  = coin_tag("exp1", p, q)
    odir = mkdir(os.path.join(out_root, tag))
    seed = cfg["seed"]
    t0   = time.time()
    print(f"\n{'='*70}\n  EXP 1 — Coin HMM  p={p}  q={q}\n{'='*70}")

    data, states = coin_generation(
        num_samples=cfg["coin_num_samples"], seq_len=cfg["coin_seq_len"], p=p, q=q)
    chunk     = cfg["train_chunk_len"]
    num_token = cfg["coin_num_token"]
    ds_fw = CoinDataset(data,     seq_len=cfg["coin_seq_len"])
    loader_fw     = make_chunked_loader(ds_fw, chunk, cfg["coin_batch"], seed=seed)
    loader_fw_ana = _loader(ds_fw, cfg["ana_batch"])
    sample_seq    = next(iter(loader_fw))[0][0]

    # B5: max_len is the FULL input length, not the chunk.  README:233 says the
    # PE table must cover the entire range; with max_len=chunk the table was
    # rebuilt at inference time on the analysis batch, at positions the model
    # was never trained on.  Harmless while B4 was unfixed (analysis also ran
    # at T=chunk) — fixing B4 without this would have exposed it.
    max_len = full_seq_len(ds_fw)
    theory  = entropy_rate_coin(p, q)
    print(f"  H∞={theory:.4f} bits  |  chunk={chunk}  max_len={max_len}  seed={seed}")

    print("\n  -- 1a Forward CV --")
    cv_fw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"), seed=seed,
    )
    cleanup()  # FIX-4

    print("\n  -- 1b Backward CV --")
    cv_bw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="backward",  # forward data → backward model
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"), seed=seed,
    )
    cleanup()  # FIX-4

    print("\n  -- 1c Analysis --")
    ana_fw = analyse_model(f"{tag}_fw", cv_fw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, p, q, "forward",
                           k=2, use_t="last", attn_vis_len=cfg["attn_vis_len"],
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"])
    cleanup()  # GC between fw and bw analysis (model already moved to CPU)
    # A3: use_t="first" for the backward arm.  With a triu mask position T-1
    # attends to itself only, so reading complexity at "last" read latents that
    # encode a single token — which cannot express more than token identity.
    ana_bw = analyse_model(f"{tag}_bw", cv_bw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, p, q, "backward",
                           k=3, use_t="first", attn_vis_len=cfg["attn_vis_len"],
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"])
    cleanup()

    print("\n  -- 1d Comparison --")
    compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw,
                  loader_fw_ana, num_token, odir, sample_seq,
                  theory, theory, cfg["attn_vis_len"], p, q, cfg=cfg)
    paired = paired_delta_ce(cv_fw, cv_bw, label=tag)

    print(f"\n  Exp 1 done in {(time.time()-t0)/60:.1f} min")
    res = dict(tag=tag, p=p, q=q, theory=theory, seed=seed, paired=paired,
               C_plus=statistical_complexity(p, q, "forward"),
               C_minus=statistical_complexity(p, q, "backward"),
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw)
    save_pkl(slim_results(res), os.path.join(odir, "results.pkl"))
    save_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw.pt"))
    save_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw.pt"))
    all_results[tag] = slim_results(res)   # keep only metrics; frees models+latents
    cleanup()


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1.2 — Coin HMM  p=0.1, q=0.9  + heatmaps
# ══════════════════════════════════════════════════════════════════════════
def experiment_1_2(cfg, out_root, all_results):
    p, q = cfg["coin_p2"], cfg["coin_q2"]
    tag  = coin_tag("exp1_2", p, q)                     # B8
    odir = mkdir(os.path.join(out_root, tag))
    seed = cfg["seed"]
    t0   = time.time()
    print(f"\n{'='*70}\n  EXP 1.2 — Coin HMM  p={p}  q={q}\n{'='*70}")

    data, states = coin_generation(
        num_samples=cfg["coin_num_samples_12"], seq_len=cfg["coin_seq_len_12"], p=p, q=q)

    chunk     = min(cfg["train_chunk_len"], cfg["coin_seq_len_12"] - 1)
    num_token = cfg["coin_num_token"]

    ds_fw = CoinDataset(data,     seq_len=cfg["coin_seq_len_12"])
    loader_fw     = make_chunked_loader(ds_fw, chunk, cfg["coin_batch"], seed=seed)
    loader_fw_ana = _loader(ds_fw, cfg["ana_batch"])
    sample_seq    = next(iter(loader_fw))[0][0]

    max_len = full_seq_len(ds_fw)                       # B5
    theory  = entropy_rate_coin(p, q)
    print(f"  H∞={theory:.4f} bits  |  chunk={chunk}  max_len={max_len}  seed={seed}")

    cv_fw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"), seed=seed,
    )
    cleanup()
    cv_bw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="backward",  # forward data → backward model
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"), seed=seed,
    )
    cleanup()

    ana_fw = analyse_model(f"{tag}_fw", cv_fw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, p, q, "forward",
                           k=2, use_t="last", attn_vis_len=cfg["attn_vis_len"],
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"])
    cleanup()
    ana_bw = analyse_model(f"{tag}_bw", cv_bw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, p, q, "backward",
                           k=3, use_t="first", attn_vis_len=cfg["attn_vis_len"],  # A3
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"])
    cleanup()
    compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw,
                  loader_fw_ana, num_token, odir, sample_seq,
                  theory, theory, cfg["attn_vis_len"], p, q, cfg=cfg)
    paired = paired_delta_ce(cv_fw, cv_bw, label=tag)

    # p-q heatmaps
    print("\n  -- Theoretical heatmap --")
    Ss_th_fw, Ss_th_bw, p_th, q_th = heatmap_theory(
        p_values=np.linspace(0.01, 0.99, 50),
        q_values=np.linspace(0.01, 0.99, 50),
    )
    print("\n  -- Empirical pq-grid --")
    # pq_experiment_full returns BOTH complexity and true PPL=exp(CE_loss)
    # Do NOT use np.exp(Ss_emp) for PPL — that is statistical complexity,
    # not perplexity.  Ppl_emp contains the real exp(CE_loss) values.
    Ss_emp, Ppl_emp, p_emp, q_emp = pq_experiment_full(
        num_token=num_token, d_model=cfg["d_model"],
        max_len=cfg["pq_len"], batch_size=cfg["coin_batch"],
        num_samples=cfg["pq_samples"], max_epochs=cfg["pq_epochs"],
        lr=cfg["lr"], p_values=cfg["pq_grid"], q_values=cfg["pq_grid"],
    )
    cleanup()

    plot_heatmap(Ss_emp, Ss_th_fw, Ss_th_bw,
                 p_emp=p_emp, q_emp=q_emp, p_theory=p_th, q_theory=q_th,
                 save_path=os.path.join(odir, f"{tag}_complexity_heatmap.png"))
    plot_diff_heatmap(Ss_th_fw - Ss_th_bw, p_th, q_th,
                      "Theoretical Complexity Diff (FW-BW)", "dC (bits)",
                      save_path=os.path.join(odir, f"{tag}_diff_complexity_theory.png"))
    plot_diff_heatmap(Ss_emp[0] - Ss_emp[1], p_emp, q_emp,
                      "Empirical Complexity Diff (FW-BW)", "dC (bits)",
                      save_path=os.path.join(odir, f"{tag}_diff_complexity_emp.png"))
    # Ppl_emp[0] = FW perplexity, Ppl_emp[1] = BW perplexity
    # Both computed as exp(CE_loss) from _eval_loss_on_loader
    plot_diff_heatmap(Ppl_emp[0] - Ppl_emp[1], p_emp, q_emp,
                        "Perplexity Diff (FW - BW)  [exp(CE loss)]", "ΔPPL",
                        save_path=os.path.join(odir, f"{tag}_diff_ppl.png"),
                        vcenter=0)

    if coin_tag("exp1", cfg["coin_p1"], cfg["coin_q1"]) in all_results:
        p1, q1 = cfg["coin_p1"], cfg["coin_q1"]
        bar_vals = [statistical_complexity(p1, q1, "forward"),
                    statistical_complexity(p1, q1, "backward"),
                    statistical_complexity(p,  q,  "forward"),
                    statistical_complexity(p,  q,  "backward")]
        bar_lbls = [f"FW({p1},{q1})", f"BW({p1},{q1})", f"FW({p},{q})", f"BW({p},{q})"]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(bar_lbls, bar_vals,
                      color=["#4c72b0","#dd8452","#55a868","#c44e52"],
                      alpha=0.85, edgecolor="k")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_ylabel("Statistical Complexity (bits)")
        ax.set_title("Theoretical Complexity: Exp1 vs Exp1.2")
        ax.grid(True, alpha=0.3, axis="y"); fig.tight_layout()
        savefig(fig, os.path.join(odir, f"{tag}_complexity_exp1_vs_12.png"))

    print(f"\n  Exp 1.2 done in {(time.time()-t0)/60:.1f} min")
    res = dict(tag=tag, p=p, q=q, theory=theory, seed=seed, paired=paired,
               C_plus=statistical_complexity(p, q, "forward"),
               C_minus=statistical_complexity(p, q, "backward"),
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw,
               Ss_emp=Ss_emp, Ppl_emp=Ppl_emp, p_emp=p_emp, q_emp=q_emp,
               Ss_th_fw=Ss_th_fw, Ss_th_bw=Ss_th_bw, p_th=p_th, q_th=q_th)
    save_pkl(slim_results(res), os.path.join(odir, "results.pkl"))
    save_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw.pt"))
    save_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw.pt"))
    all_results[tag] = slim_results(res)   # keep only metrics; frees models+latents
    cleanup()


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Flower HMM  n=4, m=2
# ══════════════════════════════════════════════════════════════════════════
def experiment_2(cfg, out_root, all_results, n, m, role):
    """
    Flower experiment at a given (n, m).

    `role` records what the theory predicts for this configuration, so the
    output can be scored against its own prediction rather than against a
    blanket "C- > C+ expected" (A1):

        "positive"  m > n  ->  C- > C+  ->  predict delta_CE > 0
        "reversed"  n > m  ->  C+ > C-  ->  predict delta_CE < 0

    Both are run, because a result where the *sign* of delta_CE tracks the sign
    of (C- - C+) across configurations is much stronger evidence than a single
    positive case.
    """
    tag       = flower_tag("exp2", n, m)                 # A4/B8: derived
    odir      = mkdir(os.path.join(out_root, tag))
    num_token = n + m
    seed      = cfg["seed"]
    t0        = time.time()

    rng        = np.random.default_rng(cfg["flower_dice_seed"])
    dice_probs = rng.dirichlet(np.ones(m), size=n)

    # A1: real closed forms instead of nan.  C+ = 1 + (1/2)log2(n);
    # C- = 1 + (1/2)H(pi_outcome) over *distinguishable* outcomes.
    C_plus, C_minus = flower_complexity(n, m, dice_probs)
    theory = flower_entropy_rate(n, m, dice_probs)       # H_inf, same both ways
    predicted = "delta_CE > 0" if C_minus > C_plus else "delta_CE < 0"

    print(f"\n{'='*70}\n  EXP 2 — Flower HMM  n={n}  m={m}  vocab={num_token}"
          f"  [{role}]\n{'='*70}")
    print(f"  dice_probs:\n{dice_probs}")
    print(f"  H∞ = {theory:.4f} bits")
    print(f"  C+ = {C_plus:.4f}   C- = {C_minus:.4f}   C- - C+ = {C_minus-C_plus:+.4f}")
    print(f"  prediction: {predicted}")

    data, states = flower_process_generation(
        num_samples=cfg["flower_num_samples"], seq_len=cfg["flower_seq_len"],
        n=n, m=m, dice_probs=dice_probs)

    chunk     = cfg["train_chunk_len"]
    seq_len_f = len(data[0])

    # FIX-5: parametric FlowerDataset from Flower_process_generation.
    # B4: the ds_bw / loader_bw / loader_bw_ana chain that used to be built
    # here was never used — cv_bw trained on loader_fw, and the comment
    # "reversed data -> forward model" described the exact opposite of what
    # the code did (forward data -> backward model).  Deleted.
    ds_fw = FlowerDataset(data, seq_len=seq_len_f)
    loader_fw     = make_chunked_loader(ds_fw, chunk, cfg["flower_batch"], seed=seed)
    loader_fw_ana = _loader(ds_fw, cfg["ana_batch"])
    sample_seq    = next(iter(loader_fw))[0][0]

    max_len = full_seq_len(ds_fw)                        # B5

    print("\n  -- 2a Forward CV --")
    cv_fw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"), seed=seed,
    )
    cleanup()

    print("\n  -- 2b Backward CV --")
    cv_bw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="backward",  # forward data → backward model
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"), seed=seed,
    )
    cleanup()

    # B6: k = n+1 forward and m+1 backward, the actual causal-state counts.
    # This used to be k=n+m for both, which for n=4,m=2 asks k-means for 6
    # clusters when the forward model has 5 states and the backward model 3.
    # Since S = H(cluster occupancy) <= log2(k), an inflated k inflates both
    # numbers and destroys the comparison.
    print("\n  -- 2c Analysis --")
    ana_fw = analyse_model(f"{tag}_fw", cv_fw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, None, None, "forward",
                           k=n+1, use_t="last", attn_vis_len=cfg["attn_vis_len"],
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"])
    ana_fw["S_theory"] = C_plus
    cleanup()
    ana_bw = analyse_model(f"{tag}_bw", cv_bw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, None, None, "backward",
                           k=m+1, use_t="first", attn_vis_len=cfg["attn_vis_len"],  # A3
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"])
    ana_bw["S_theory"] = C_minus
    cleanup()

    print("\n  -- 2d Comparison --")
    compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw,
                  loader_fw_ana, num_token, odir, sample_seq,
                  theory, theory, cfg["attn_vis_len"], cfg=cfg)
    paired = paired_delta_ce(cv_fw, cv_bw, label=tag)
    if paired:
        agrees = (paired["mean"] > 0) == (C_minus > C_plus)
        print(f"  sign of delta_CE {'MATCHES' if agrees else 'CONTRADICTS'} "
              f"sign of (C- - C+)  [{role}]")

    try:
        fig_cx, ax_cx = plt.subplots(figsize=(8, 5))
        x = np.arange(2)
        emp = [ana_fw.get("S_emp", 0), ana_bw.get("S_emp", 0)]
        th  = [C_plus, C_minus]
        b1 = ax_cx.bar(x - 0.2, emp, 0.35, label="Empirical",
                       color=["#4c72b0", "#dd8452"], alpha=0.85, edgecolor="k")
        b2 = ax_cx.bar(x + 0.2, th, 0.35, label="Theoretical",
                       color=["#4c72b0", "#dd8452"], alpha=0.45, edgecolor="k",
                       hatch="//")
        ax_cx.bar_label(b1, fmt="%.4f", padding=3, fontsize=9)
        ax_cx.bar_label(b2, fmt="%.4f", padding=3, fontsize=9)
        ax_cx.set_xticks(x); ax_cx.set_xticklabels(["Forward (C+)", "Backward (C-)"])
        ax_cx.set_ylabel("Statistical Complexity (bits)")
        ax_cx.set_title(f"{tag} — complexity  (n={n}, m={m}: "
                        f"{'C- > C+' if C_minus > C_plus else 'C+ > C-'})")
        ax_cx.legend(); ax_cx.grid(True, alpha=0.3, axis="y"); fig_cx.tight_layout()
        savefig(fig_cx, os.path.join(odir, f"{tag}_complexity.png"))
    except Exception as e:
        print(f"  complexity bar: {e}")

    print(f"\n  Exp 2 done in {(time.time()-t0)/60:.1f} min")
    res = dict(tag=tag, n=n, m=m, dice_probs=dice_probs, role=role,
               theory=theory, C_plus=C_plus, C_minus=C_minus,
               seed=seed, paired=paired,
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw)
    save_pkl(slim_results(res), os.path.join(odir, "results.pkl"))
    save_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw.pt"))
    save_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw.pt"))
    all_results[tag] = slim_results(res)   # keep only metrics; frees models+latents
    cleanup()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    OUT_ROOT = "results"
    set_seed(CFG["seed"])                    # A2: reproducible end to end
    mkdir(OUT_ROOT)
    mkdir(os.path.join(OUT_ROOT, "models"))
    all_results = {}
    t_start = time.time()

    experiment_1  (CFG, OUT_ROOT, all_results)
    experiment_1_2(CFG, OUT_ROOT, all_results)
    for n, m, role in CFG["flower_configs"]:
        experiment_2(CFG, OUT_ROOT, all_results, n=n, m=m, role=role)

    save_pkl(all_results, os.path.join(OUT_ROOT, "all_results.pkl"))  # already slim

    total = (time.time() - t_start) / 60
    print(f"\n{'='*70}\n  ALL COMPLETE — {total:.1f} min")
    for root, dirs, files in os.walk(OUT_ROOT):
        lvl    = root.replace(OUT_ROOT, "").count(os.sep)
        indent = "  " * (lvl + 1)
        print(f"{'  '*lvl}{os.path.basename(root)}/")
        for f in sorted(files):
            print(f"{indent}{f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()