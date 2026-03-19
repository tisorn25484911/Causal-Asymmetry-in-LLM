
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
    FW_BW_attention_comparison,
    latent_extraction,
    plot_attention_heatmap,
    statistical_complexity,
    statistical_complexity_empirical,
)
from Training_model import _eval_loss_on_loader, train_test_val_pipeline
from pq_experiment import heatmap_theory, plot_heatmap, pq_experiment
 
# ── UMAP — warm up JIT here so segfault (if any) happens at startup ───────
# FIX-2: pre-compile numba kernels before training begins
try:
    import umap as _umap_mod
    _warmup = _umap_mod.UMAP(n_components=2, n_neighbors=5).fit_transform(
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
    # ── model ──────────────────────────────────────────────────────────
    d_model       = 32,
    embed_type    = "onehot",
    n_folds       = 5,
    lr            = 1e-2,
    # ── FIX-1: limit T to avoid O(T²) OOM ─────────────────────────────
    train_chunk_len = 256,   # tokens seen per training step
    attn_vis_len    = 64,    # tokens used for attention heatmap plots
    # ── coin exp 1 ─────────────────────────────────────────────────────
    coin_p1          = 0.3,  coin_q1          = 0.4,
    coin_num_samples = 500,  coin_seq_len     = 2000,
    coin_max_epochs  = 30,   coin_batch       = 32,
    coin_num_token   = 3,
    # ── coin exp 1.2 ───────────────────────────────────────────────────
    coin_p2             = 0.1,  coin_q2             = 0.9,
    coin_num_samples_12 = 500,  coin_seq_len_12     = 500,
    # ── flower exp 2 ───────────────────────────────────────────────────
    flower_n           = 4,  flower_m           = 2,
    flower_num_samples = 500, flower_seq_len    = 2000,
    flower_max_epochs  = 30,  flower_batch      = 32,
    # ── pq heatmap ─────────────────────────────────────────────────────
    pq_grid   = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    pq_epochs = 15,  pq_samples = 500,  pq_len = 200,
)
 
 
# ══════════════════════════════════════════════════════════════════════════
# FIX-1: ChunkedDataset — caps T at train_chunk_len
# ══════════════════════════════════════════════════════════════════════════
class ChunkedDataset(tud.Dataset):
    """
    Wraps a sequence dataset and returns a random contiguous chunk of
    length `chunk_len` instead of the full sequence.
 
    Without this, training on 2000-token sequences allocates
      B × T × T × 4 bytes ≈ 32 × 1999 × 1999 × 4 ≈ 512 MB
    per attention layer per batch, causing OOM / kernel kill.
    """
 
    def __init__(self, base: tud.Dataset, chunk_len: int, seed: int = 0):
        self.base  = base
        self.chunk = chunk_len
        self.rng   = np.random.default_rng(seed)
 
    def __len__(self):
        return len(self.base)
 
    def __getitem__(self, idx):
        inp, tgt = self.base[idx]          # shape (T,) each
        T = inp.shape[0]
        if T <= self.chunk:
            return inp, tgt
        start = int(self.rng.integers(0, T - self.chunk + 1))
        return inp[start: start + self.chunk], tgt[start: start + self.chunk]
 
 
def make_chunked_loader(
    dataset: tud.Dataset, chunk_len: int, batch_size: int, shuffle: bool = True
) -> tud.DataLoader:
    """FIX-1 + FIX-3 combined."""
    return tud.DataLoader(
        ChunkedDataset(dataset, chunk_len),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,             # FIX-3
        persistent_workers=False,  # FIX-3
    )
 
 
def _loader(dataset: tud.Dataset, batch_size: int, shuffle: bool = False) -> tud.DataLoader:
    """Plain loader — FIX-3 only (no chunking; used for analysis)."""
    return tud.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=0, persistent_workers=False,
    )
 
 
# ══════════════════════════════════════════════════════════════════════════
# FIX-4 / FIX-7 / FIX-8: safe cleanup + CPU-offload for analysis
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
 
 
def latent_extraction(model, data_loader, max_batches=None):   # noqa: F811
    """
    CPU-only latent extraction (FIX-7/8).
 
    Model MUST already be on CPU when this is called (ensured by
    to_cpu_for_analysis).  This avoids all MPS command-buffer races.
    """
    model.eval()
    latents_all, inputs_all, target_all = [], [], []
 
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            # inputs / targets already on CPU — no .to(device) needed
            _  = model(inputs)
            z  = model.last_encodings.detach().numpy()
            latents_all.append(z)
            inputs_all.append(inputs.numpy())
            target_all.append(targets.numpy())
 
    latents = np.concatenate(latents_all, axis=0)
    inps    = np.concatenate(inputs_all,  axis=0)
    tgts    = np.concatenate(target_all,  axis=0)
    return latents, inps, tgts
 
 
# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════
def mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
 
 
def save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  pickle saved -> {path}")
 
 
def save_weights(model, path: str):
    torch.save(model.state_dict(), path)
    print(f"  weights saved -> {path}")
 
 
def savefig(fig, path: str, dpi: int = 120):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved -> {path}")
 
 
# FIX-6: single function — rate is direction-symmetric
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
 
 
# ── 2-D projection ───────────────────────────────────────────────────────
# n_neighbors=100 matches the reference notebook and produces tight,
# well-separated clusters.  min(100, N-1) keeps it valid for small N.
def _project2d(flat: np.ndarray, n_neighbors: int = 100) -> tuple:
    if UMAP_AVAILABLE:
        try:
            c = _umap_mod.UMAP(
                n_components=2, random_state=42,
                n_neighbors=min(n_neighbors, len(flat) - 1),
                min_dist=0.1,
                metric="euclidean",
            ).fit_transform(flat)
            return c, "UMAP"
        except Exception as e:
            print(f"  UMAP failed ({e}), using PCA")
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(flat), "PCA"
 
 
# Take the FIRST n consecutive points (same as reference notebook).
# Random subsampling shuffles temporal order and breaks the neighbourhood
# structure that UMAP relies on to form clusters.
def _sub(arr: np.ndarray, n: int = 1000) -> tuple:
    """Return (latent_subset, index_array) — first n rows, no shuffle."""
    n = min(n, len(arr))
    return arr[:n], np.arange(n)
 
 
def plot_umap(latents, inputs_arr, num_token, title="", save_path=None,
              xlim=None, ylim=None, n_pts: int = 1000):
    """
    Plot 2-D UMAP coloured by token id.
    n_pts: number of consecutive latent vectors to embed (default 1000).
    """
    flat_l = latents.reshape(-1, latents.shape[-1])
    flat_i = inputs_arr.reshape(-1)
    sub_l, idx = _sub(flat_l, n_pts)
    sub_i = flat_i[idx]
 
    coords, mlbl = _project2d(sub_l)
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(7, 6))
    for tok in range(num_token):
        mask = sub_i == tok
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(tok / max(num_token - 1, 1))],
                   label=f"Token {tok}", alpha=0.7, s=10)
    ax.set_title(f"{title} ({mlbl})", fontsize=11, fontweight="bold")
    ax.legend(markerscale=3, fontsize=8)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2)
    if save_path:
        savefig(fig, save_path)
    return fig, coords
 
 
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
 
 
def plot_diff_heatmap(Z, p_vals, q_vals, title, cbar_label, save_path=None, cmap="RdBu_r"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ext = [float(np.min(q_vals)), float(np.max(q_vals)),
           float(np.min(p_vals)), float(np.max(p_vals))]
    im = ax.imshow(Z, origin="lower", extent=ext, cmap=cmap, aspect="auto")
    Qm, Pm = np.meshgrid(q_vals, p_vals)
    ax.contour(Qm, Pm, Z, levels=8, colors="white", alpha=0.4, linewidths=0.8)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("q", fontsize=12); ax.set_ylabel("p", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if save_path: savefig(fig, save_path)
    return fig
 
 
def analyse_model(tag, model, loader, num_token, out_dir,
                  sample_seq=None, p=None, q=None, mode="forward",
                  k=2, use_t="last", attn_vis_len=64):
    res = {"tag": tag}
 
    # ── Move to CPU so Metal heap is freed before analysis ────────────────
    to_cpu_for_analysis(model)
    # Also move sample_seq to CPU
    if sample_seq is not None:
        sample_seq = sample_seq.cpu()
 
    # Attention — short prefix only (FIX-1)
    if sample_seq is not None:
        try:
            fig_a = plot_attention_heatmap(model, sample_seq[:attn_vis_len])
            savefig(fig_a, os.path.join(out_dir, f"{tag}_attn.png"))
        except Exception as e:
            print(f"  attn failed: {e}")
 
    # UMAP — model already on CPU, loader returns CPU tensors (num_workers=0)
    try:
        latents, inp_arr, _ = latent_extraction(model, loader, max_batches=20)
        _, coords = plot_umap(latents, inp_arr, num_token, title=tag,
                              save_path=os.path.join(out_dir, f"{tag}_umap.png"))
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
 
 
def compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw, loader, num_token, out_dir,
                  sample_seq, theory_fw, theory_bw, attn_vis_len=64, p=None, q=None):
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
 
    # (c) shared-range UMAP
    try:
        lfw, ifw, _ = latent_extraction(mfw, loader, max_batches=20)
        lbw, ibw, _ = latent_extraction(mbw, loader, max_batches=20)
        # _sub now returns (array, index) — unpack correctly
        fl_fw, idx_fw = _sub(lfw.reshape(-1, lfw.shape[-1]))
        fl_bw, idx_bw = _sub(lbw.reshape(-1, lbw.shape[-1]))
        si_fw = ifw.reshape(-1)[idx_fw]
        si_bw = ibw.reshape(-1)[idx_bw]
        c_fw, mlbl = _project2d(fl_fw)
        c_bw, _    = _project2d(fl_bw)
        xlim = (min(c_fw[:,0].min(), c_bw[:,0].min()) - 0.5,
                max(c_fw[:,0].max(), c_bw[:,0].max()) + 0.5)
        ylim = (min(c_fw[:,1].min(), c_bw[:,1].min()) - 0.5,
                max(c_fw[:,1].max(), c_bw[:,1].max()) + 0.5)
        cmap = plt.cm.tab10
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        for ax, coords, raw_inp, lbl in zip(
            axes, [c_fw, c_bw], [si_fw, si_bw], ["Forward", "Backward"]
        ):
            for tok in range(num_token):
                mask = raw_inp == tok
                if not mask.any(): continue
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=[cmap(tok / max(num_token - 1, 1))],
                           label=f"Token {tok}", alpha=0.7, s=10)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_title(f"{lbl} ({mlbl})", fontweight="bold")
            ax.legend(markerscale=3, fontsize=8); ax.grid(True, alpha=0.2)
        fig.suptitle(f"{tag} — shared-range {mlbl}", fontweight="bold")
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
# EXPERIMENT 1 — Coin HMM  p=0.3, q=0.4
# ══════════════════════════════════════════════════════════════════════════
def experiment_1(cfg, out_root, all_results):
    tag  = "exp1_coin_p03_q04"
    odir = mkdir(os.path.join(out_root, tag))
    p, q = cfg["coin_p1"], cfg["coin_q1"]
    t0   = time.time()
    print(f"\n{'='*70}\n  EXP 1 — Coin HMM  p={p}  q={q}\n{'='*70}")
 
    data, states = coin_generation(
        num_samples=cfg["coin_num_samples"], seq_len=cfg["coin_seq_len"], p=p, q=q)
    data_rev, states_rev = Rev_HMM_generation(data, states)
 
    chunk     = cfg["train_chunk_len"]
    num_token = cfg["coin_num_token"]
 
    ds_fw = CoinDataset(data,     seq_len=cfg["coin_seq_len"])
    ds_bw = CoinDataset(data_rev, seq_len=cfg["coin_seq_len"])
    loader_fw     = make_chunked_loader(ds_fw, chunk, cfg["coin_batch"])
    loader_bw     = make_chunked_loader(ds_bw, chunk, cfg["coin_batch"])
    loader_fw_ana = _loader(ds_fw, cfg["coin_batch"])
    loader_bw_ana = _loader(ds_bw, cfg["coin_batch"])
    sample_seq    = next(iter(loader_fw_ana))[0][0]
 
    max_len = chunk
    theory  = entropy_rate_coin(p, q)
    print(f"  H∞={theory:.4f} bits  |  chunk={chunk}")
 
    print("\n  -- 1a Forward CV --")
    cv_fw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"),
    )
    cleanup()  # FIX-4
 
    print("\n  -- 1b Backward CV --")
    cv_bw = train_test_val_pipeline(
        loader_bw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="backward",
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"),
    )
    cleanup()  # FIX-4
 
    print("\n  -- 1c Analysis --")
    ana_fw = analyse_model(f"{tag}_fw", cv_fw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, p, q, "forward",
                           k=2, use_t="last", attn_vis_len=cfg["attn_vis_len"])
    cleanup()  # GC between fw and bw analysis (model already moved to CPU)
    ana_bw = analyse_model(f"{tag}_bw", cv_bw["best_model"], loader_bw_ana,
                           num_token, odir, sample_seq, p, q, "backward",
                           k=3, use_t="first", attn_vis_len=cfg["attn_vis_len"])
    cleanup()
 
    print("\n  -- 1d Comparison --")
    compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw,
                  loader_fw_ana, num_token, odir, sample_seq,
                  theory, theory, cfg["attn_vis_len"], p, q)
 
    print(f"\n  Exp 1 done in {(time.time()-t0)/60:.1f} min")
    res = dict(tag=tag, p=p, q=q, theory=theory,
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw)
    save_pkl(res, os.path.join(odir, "results.pkl"))
    save_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw.pt"))
    save_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw.pt"))
    all_results[tag] = res
    cleanup()
 
 
# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1.2 — Coin HMM  p=0.1, q=0.9  + heatmaps
# ══════════════════════════════════════════════════════════════════════════
def experiment_1_2(cfg, out_root, all_results):
    tag  = "exp1_2_coin_p01_q09"
    odir = mkdir(os.path.join(out_root, tag))
    p, q = cfg["coin_p2"], cfg["coin_q2"]
    t0   = time.time()
    print(f"\n{'='*70}\n  EXP 1.2 — Coin HMM  p={p}  q={q}\n{'='*70}")
 
    data, states = coin_generation(
        num_samples=cfg["coin_num_samples_12"], seq_len=cfg["coin_seq_len_12"], p=p, q=q)
    data_rev, _ = Rev_HMM_generation(data, states)
 
    chunk     = min(cfg["train_chunk_len"], cfg["coin_seq_len_12"] - 1)
    num_token = cfg["coin_num_token"]
 
    ds_fw = CoinDataset(data,     seq_len=cfg["coin_seq_len_12"])
    ds_bw = CoinDataset(data_rev, seq_len=cfg["coin_seq_len_12"])
    loader_fw     = make_chunked_loader(ds_fw, chunk, cfg["coin_batch"])
    loader_bw     = make_chunked_loader(ds_bw, chunk, cfg["coin_batch"])
    loader_fw_ana = _loader(ds_fw, cfg["coin_batch"])
    loader_bw_ana = _loader(ds_bw, cfg["coin_batch"])
    sample_seq    = next(iter(loader_fw_ana))[0][0]
 
    max_len = chunk
    theory  = entropy_rate_coin(p, q)
    print(f"  H∞={theory:.4f} bits")
 
    cv_fw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"),
    )
    cleanup()
    cv_bw = train_test_val_pipeline(
        loader_bw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="backward",
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"),
    )
    cleanup()
 
    ana_fw = analyse_model(f"{tag}_fw", cv_fw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, p, q, "forward",
                           k=2, use_t="last", attn_vis_len=cfg["attn_vis_len"])
    cleanup()
    ana_bw = analyse_model(f"{tag}_bw", cv_bw["best_model"], loader_bw_ana,
                           num_token, odir, sample_seq, p, q, "backward",
                           k=3, use_t="first", attn_vis_len=cfg["attn_vis_len"])
    cleanup()
    compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw,
                  loader_fw_ana, num_token, odir, sample_seq,
                  theory, theory, cfg["attn_vis_len"], p, q)
 
    # p-q heatmaps
    print("\n  -- Theoretical heatmap --")
    Ss_th_fw, Ss_th_bw, p_th, q_th = heatmap_theory(
        p_values=np.linspace(0.01, 0.99, 50),
        q_values=np.linspace(0.01, 0.99, 50),
    )
    print("\n  -- Empirical pq-grid --")
    Ss_emp, p_emp, q_emp = pq_experiment(
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
    plot_diff_heatmap(np.exp(Ss_emp[0]) - np.exp(Ss_emp[1]), p_emp, q_emp,
                      "Empirical Perplexity Diff (FW-BW)", "dPPL",
                      save_path=os.path.join(odir, f"{tag}_diff_ppl.png"))
 
    if "exp1_coin_p03_q04" in all_results:
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
    res = dict(tag=tag, p=p, q=q, theory=theory,
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw,
               Ss_emp=Ss_emp, p_emp=p_emp, q_emp=q_emp,
               Ss_th_fw=Ss_th_fw, Ss_th_bw=Ss_th_bw, p_th=p_th, q_th=q_th)
    save_pkl(res, os.path.join(odir, "results.pkl"))
    save_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw.pt"))
    save_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw.pt"))
    all_results[tag] = res
    cleanup()
 
 
# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Flower HMM  n=4, m=2
# ══════════════════════════════════════════════════════════════════════════
def experiment_2(cfg, out_root, all_results):
    tag       = "exp2_flower_n4_m2"
    odir      = mkdir(os.path.join(out_root, tag))
    n, m      = cfg["flower_n"], cfg["flower_m"]
    num_token = n + m
    t0        = time.time()
    print(f"\n{'='*70}\n  EXP 2 — Flower HMM  n={n}  m={m}  vocab={num_token}\n{'='*70}")
 
    rng        = np.random.default_rng(42)
    dice_probs = rng.dirichlet(np.ones(m), size=n)
    print(f"  dice_probs:\n{dice_probs}")
 
    data, states = flower_process_generation(
        num_samples=cfg["flower_num_samples"], seq_len=cfg["flower_seq_len"],
        n=n, m=m, dice_probs=dice_probs)
    seqs_bw = [list(reversed(s)) for s in data]
 
    chunk    = cfg["train_chunk_len"]
    seq_len_f = len(data[0])
 
    # FIX-5: parametric FlowerDataset from Flower_process_generation
    ds_fw = FlowerDataset(data,    seq_len=seq_len_f)
    ds_bw = FlowerDataset(seqs_bw, seq_len=seq_len_f)
    loader_fw     = make_chunked_loader(ds_fw, chunk, cfg["flower_batch"])
    loader_bw     = make_chunked_loader(ds_bw, chunk, cfg["flower_batch"])
    loader_fw_ana = _loader(ds_fw, cfg["flower_batch"])
    loader_bw_ana = _loader(ds_bw, cfg["flower_batch"])
    sample_seq    = next(iter(loader_fw_ana))[0][0]
 
    max_len   = chunk
    theory_fw = float("nan")
    theory_bw = float("nan")
 
    print("\n  -- 2a Forward CV --")
    cv_fw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"),
    )
    cleanup()
 
    print("\n  -- 2b Backward CV --")
    cv_bw = train_test_val_pipeline(
        loader_bw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="backward",
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"),
    )
    cleanup()
 
    print("\n  -- 2c Analysis --")
    ana_fw = analyse_model(f"{tag}_fw", cv_fw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, None, None, "forward",
                           k=n+m, use_t="last", attn_vis_len=cfg["attn_vis_len"])
    cleanup()
    ana_bw = analyse_model(f"{tag}_bw", cv_bw["best_model"], loader_bw_ana,
                           num_token, odir, sample_seq, None, None, "backward",
                           k=n+m, use_t="first", attn_vis_len=cfg["attn_vis_len"])
    cleanup()
 
    print("\n  -- 2d Comparison --")
    compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw,
                  loader_fw_ana, num_token, odir, sample_seq,
                  theory_fw, theory_bw, cfg["attn_vis_len"])
 
    try:
        fig_cx, ax_cx = plt.subplots(figsize=(7, 5))
        vals = [ana_fw.get("S_emp", 0), ana_bw.get("S_emp", 0)]
        bars = ax_cx.bar(["Forward", "Backward"], vals,
                         color=["#4c72b0", "#dd8452"], alpha=0.85, edgecolor="k")
        ax_cx.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
        ax_cx.set_ylabel("Empirical Complexity (bits)")
        ax_cx.set_title(f"{tag} — empirical complexity")
        ax_cx.grid(True, alpha=0.3, axis="y"); fig_cx.tight_layout()
        savefig(fig_cx, os.path.join(odir, f"{tag}_complexity.png"))
    except Exception as e:
        print(f"  complexity bar: {e}")
 
    print(f"\n  Exp 2 done in {(time.time()-t0)/60:.1f} min")
    res = dict(tag=tag, n=n, m=m, dice_probs=dice_probs,
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw)
    save_pkl(res, os.path.join(odir, "results.pkl"))
    save_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw.pt"))
    save_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw.pt"))
    all_results[tag] = res
    cleanup()
 
 
# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    OUT_ROOT = "results"
    mkdir(OUT_ROOT)
    mkdir(os.path.join(OUT_ROOT, "models"))
    all_results = {}
    t_start = time.time()
 
    experiment_1  (CFG, OUT_ROOT, all_results)
    experiment_1_2(CFG, OUT_ROOT, all_results)
    experiment_2  (CFG, OUT_ROOT, all_results)
 
    save_pkl(all_results, os.path.join(OUT_ROOT, "all_results.pkl"))
 
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