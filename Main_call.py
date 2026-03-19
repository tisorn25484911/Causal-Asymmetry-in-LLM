
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

"""
main.py — Full Experiment Runner
=================================
Experiments
  1   : Coin HMM   p=0.3, q=0.4 — FW + BW, 5-fold CV, analysis, comparison
  1.2 : Coin HMM   p=0.1, q=0.9 — same + p-q heatmaps (diff complexity, diff PPL)
  2   : Flower HMM n=4,   m=2   — FW + BW, 5-fold CV, analysis, comparison

All figures are saved under  results/<exp_tag>/
All trained models + data    → results/models/<tag>.pt
Full results dict (every exp) → results/all_results.pkl   (load in notebook)
"""

# ── stdlib & third-party ───────────────────────────────────────────────────
import os, sys, pickle, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# ── project imports ────────────────────────────────────────────────────────
from Data_generation import (
    coin_generation, Rev_HMM_generation, make_loader,
    flower_process_generation, CoinDataset, FlowerDataset,
)
from torch.utils.data import DataLoader
from Training_model import (
    train_test_val_pipeline,
    train_model,
    _eval_loss_on_loader,
)
from Model_analysis import (
    statistical_complexity,
    statistical_complexity_empirical,
    statistical_complexity_compare,
    FW_BW_attention_comparison,
    FW_BW_loss_comparison,
    compare_FW_BW_latents,
    plot_attention_heatmap,
    plot_perplexity,
    latent_extraction,
)
from pq_experiment import heatmap_theory, pq_experiment, plot_heatmap

# ── UMAP with PCA/tSNE fallback ───────────────────────────────────────────
try:
    import umap as umap_lib
    UMAP_AVAILABLE = True
    print("✓ umap-learn available — using UMAP for 2D visualisations")
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠ umap-learn not installed — falling back to PCA for 2D visualisations")
    print("  (install with: pip install umap-learn)")

# ══════════════════════════════════════════════════════════════════════════
# GLOBAL HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════
CFG = dict(
    # ── model ──────────────────────────────────────────────────────────
    d_model    = 32,
    embed_type = "onehot",
    n_folds    = 5,
    lr         = 1e-2,
    # ── coin exp 1 ─────────────────────────────────────────────────────
    coin_p1 = 0.3,   coin_q1 = 0.4,
    coin_num_samples = 500,  coin_seq_len = 2000,
    coin_max_epochs  = 30,   coin_batch   = 32,
    coin_num_token   = 3,
    # ── coin exp 1.2 ───────────────────────────────────────────────────
    coin_p2 = 0.1,   coin_q2 = 0.9,
    coin_num_samples_12 = 500, coin_seq_len_12 = 500,
    # ── flower exp 2 ───────────────────────────────────────────────────
    flower_n = 4,    flower_m = 2,
    flower_num_samples = 500, flower_seq_len = 2000,
    flower_max_epochs  = 30,  flower_batch   = 32,
    # ── pq heatmap ─────────────────────────────────────────────────────
    pq_grid    = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    pq_epochs  = 15,  pq_samples = 500,  pq_len = 200,
)

# ══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  ✓ Saved pickle → {path}")


def save_model_weights(model, path: str):
    torch.save(model.state_dict(), path)
    print(f"  ✓ Model weights → {path}")


def savefig(fig, path: str, dpi: int = 130):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure → {path}")


# ── Theoretical entropy rate (coin HMM) ───────────────────────────────────
def entropy_coin_fw(p: float, q: float) -> float:
    """
    Entropy rate H∞ (bits) of the coin-process HMM.
    This is the asymptotic lower bound on the CE loss of a perfect forward model.
    """
    def _h2(a: float) -> float:
        b = 1.0 - a
        a, b = max(a, 1e-12), max(b, 1e-12)
        return -a * np.log2(a) - b * np.log2(b)
    pi0 = q / (p + q)
    return _h2(pi0)
def entropy_coin_bw(p: float, q: float) -> float:
    """
    Entropy rate H∞ (bits) of the coin-process HMM.
    This is the asymptotic lower bound on the CE loss of a perfect backward model.
    """
    def _h1(a: float) -> float:
        b = 1.0 - a
        a, b = max(a, 1e-12), max(b, 1e-12)
        return -a * np.log2(a)
    pi0 = q / (p + q)
    pi1 = p*q / (p + q)
    pi2 = 1 - pi1 - pi0
    return  _h1(pi0) + _h1(pi1) + _h1(pi2)
def entropy_rate_coin_fw(p: float, q: float) -> float:
    """
    Entropy rate H∞ (bits) of the coin-process HMM.
    This is the asymptotic lower bound on the CE loss of a perfect forward model.
    """
    def _h2(a: float) -> float:
        b = 1.0 - a
        a, b = max(a, 1e-12), max(b, 1e-12)
        return -a * np.log2(a) - b * np.log2(b)
    pi0 = q / (p + q)
    pi1 = p / (p + q)
    return pi0 * _h2(p) + pi1 * _h2(q)
def entropy_rate_coin_bw(p: float, q: float) -> float:
    """
    Entropy rate H∞ (bits) of the coin-process HMM.
    This is the asymptotic lower bound on the CE loss of a perfect backward model.
    """
    def _h2(a: float) -> float:
        b = 1.0 - a
        a, b = max(a, 1e-12), max(b, 1e-12)
        return -a * np.log2(a) - b * np.log2(b)
    pi0 = q / (p + q)
    pi1 = p / (p + q)
    return pi0 * _h2(p) + pi1 * _h2(q)

# ── 2-D projection helper ─────────────────────────────────────────────────
def project_2d(latents_flat: np.ndarray, method: str = "umap") -> np.ndarray:
    if method == "umap" and UMAP_AVAILABLE:
        return umap_lib.UMAP(n_components=2, random_state=42,
                             n_neighbors=15, min_dist=0.1).fit_transform(latents_flat)
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(latents_flat)


def plot_umap(latents: np.ndarray, inputs_arr: np.ndarray,
              num_token: int, title: str = "",
              save_path: str = None, xlim=None, ylim=None):
    """
    2-D UMAP (or PCA) scatter coloured by token id.
    Returns (fig, coords_2d).
    """
    flat_l = latents.reshape(-1, latents.shape[-1])
    flat_i = inputs_arr.reshape(-1)

    # subsample if huge to keep UMAP tractable
    if flat_l.shape[0] > 8000:
        idx = np.random.default_rng(0).choice(flat_l.shape[0], 8000, replace=False)
        flat_l_fit = flat_l[idx]
        flat_i_fit = flat_i[idx]
    else:
        flat_l_fit, flat_i_fit = flat_l, flat_i
        idx = np.arange(flat_l.shape[0])

    method = "umap" if UMAP_AVAILABLE else "pca"
    coords = project_2d(flat_l_fit, method=method)
    method_label = "UMAP" if UMAP_AVAILABLE else "PCA"

    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(7, 6))
    for tok in range(num_token):
        mask = flat_i_fit == tok
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(tok / max(num_token - 1, 1))],
                   label=f"Token {tok}", alpha=0.35, s=5)
    ax.set_title(f"{title}\n({method_label} 2D)", fontsize=11, fontweight="bold")
    ax.legend(markerscale=4, fontsize=8, loc="best")
    if xlim:  ax.set_xlim(xlim)
    if ylim:  ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2)
    if save_path:
        savefig(fig, save_path)
    return fig, coords


# ── Loss curve with theoretical limit ─────────────────────────────────────
def plot_loss_with_theory(rec_fw, rec_bw,
                          theory_fw: float, theory_bw: float,
                          title: str = "", save_path: str = None):
    """Step-loss + val-loss curves overlaid with horizontal theory limits."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, rec, theory, lbl in zip(
        axes,
        [rec_fw, rec_bw],
        [theory_fw, theory_bw],
        ["Forward", "Backward"],
    ):
        ax.plot(rec.step_loss, color="steelblue", lw=1.2,
                alpha=0.75, label="Train loss")
        if rec.step_val_loss:
            ax.plot(rec.step_val_loss, color="darkorange", lw=1.2,
                    alpha=0.75, label="Val loss")
        ax.axhline(theory, color="crimson", ls="--", lw=2,
                   label=f"Theory H = {theory:.4f} bits")
        ax.set_xlabel("Gradient Step")
        ax.set_ylabel("Loss (bits)")
        ax.set_title(f"{lbl} — {title}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


# ── Difference heatmap helper ─────────────────────────────────────────────
def plot_diff_heatmap(Z_diff: np.ndarray, p_vals, q_vals,
                      title: str, cbar_label: str, save_path: str = None,
                      cmap: str = "RdBu_r"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ext = [float(np.min(q_vals)), float(np.max(q_vals)),
           float(np.min(p_vals)), float(np.max(p_vals))]
    im = ax.imshow(Z_diff, origin="lower", extent=ext, cmap=cmap,
                   aspect="auto")
    Qm, Pm = np.meshgrid(q_vals, p_vals)
    ax.contour(Qm, Pm, Z_diff, levels=8, colors="white", alpha=0.4, linewidths=0.8)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("q", fontsize=12);  ax.set_ylabel("p", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if save_path:
        savefig(fig, save_path)
    return fig


# ── Single-model analysis helper ──────────────────────────────────────────
def analyse_model(tag: str, model, loader, num_token: int, out_dir: str,
                  sample_seq=None, p=None, q=None, mode="forward",
                  k_complexity=2, use_t="last") -> dict:
    """
    Produces:  attention heatmap, 2D UMAP, empirical & theoretical complexity.
    Returns a dict with all metrics and coords (for shared-range comparison).
    """
    res = {"tag": tag}

    # attention
    if sample_seq is not None:
        fig_a = plot_attention_heatmap(model, sample_seq)
        savefig(fig_a, os.path.join(out_dir, f"{tag}_attn.png"))
        res["attn_path"] = os.path.join(out_dir, f"{tag}_attn.png")

    # latents + UMAP
    latents, inp_arr, _ = latent_extraction(model, loader, max_batches=25)
    _, coords = plot_umap(latents, inp_arr, num_token,
                          title=tag,
                          save_path=os.path.join(out_dir, f"{tag}_umap.png"))
    res.update({"latents": latents, "inputs_arr": inp_arr, "umap_coords": coords})

    # statistical complexity
    S_emp = statistical_complexity_empirical(
        model, loader, max_batches=25, use_t=use_t, k=k_complexity)
    res["S_emp"] = S_emp
    if p is not None and q is not None:
        S_th = statistical_complexity(p, q, mode=mode)
        res["S_theory"] = S_th
        print(f"  [{tag}] S_emp={S_emp:.4f}  S_theory={S_th:.4f}  "
              f"err={abs(S_emp - S_th):.4f} bits")
    else:
        print(f"  [{tag}] S_emp={S_emp:.4f}")
    return res


# ── FW + BW comparison plots ──────────────────────────────────────────────
def compare_fw_bw(tag: str,
                  cv_fw: dict, cv_bw: dict,
                  ana_fw: dict, ana_bw: dict,
                  loader, num_token: int, out_dir: str,
                  sample_seq, theory_fw: float, theory_bw: float,
                  p=None, q=None):
    """Loss curves, attention comparison, UMAP with shared range, complexity bar."""
    model_fw = cv_fw["best_model"]
    model_bw = cv_bw["best_model"]
    rec_fw   = cv_fw["best_recorder"]
    rec_bw   = cv_bw["best_recorder"]

    # (a) loss curves + theory limit
    plot_loss_with_theory(
        rec_fw, rec_bw, theory_fw, theory_bw,
        title=tag,
        save_path=os.path.join(out_dir, f"{tag}_loss_theory.png"),
    )

    # (b) joint attention heatmap
    fig_a = FW_BW_attention_comparison(model_fw, model_bw, sample_seq,
                                       title_prefix=f"{tag} ")
    savefig(fig_a, os.path.join(out_dir, f"{tag}_attn_compare.png"))

    # (c) shared-range UMAP
    lfw, ifw, _  = latent_extraction(model_fw, loader, max_batches=25)
    lbw, ibw, _  = latent_extraction(model_bw, loader, max_batches=25)
    flat_lfw = lfw.reshape(-1, lfw.shape[-1])
    flat_lbw = lbw.reshape(-1, lbw.shape[-1])
    method = "umap" if UMAP_AVAILABLE else "pca"
    c_fw = project_2d(flat_lfw[:min(len(flat_lfw), 8000)], method)
    c_bw = project_2d(flat_lbw[:min(len(flat_lbw), 8000)], method)
    method_lbl = "UMAP" if UMAP_AVAILABLE else "PCA"

    xlim = (min(c_fw[:, 0].min(), c_bw[:, 0].min()) - 0.5,
            max(c_fw[:, 0].max(), c_bw[:, 0].max()) + 0.5)
    ylim = (min(c_fw[:, 1].min(), c_bw[:, 1].min()) - 0.5,
            max(c_fw[:, 1].max(), c_bw[:, 1].max()) + 0.5)

    cmap = plt.cm.tab10
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, inp, lbl in zip(axes, [c_fw, c_bw],
                                    [ifw.reshape(-1)[:len(c_fw)],
                                     ibw.reshape(-1)[:len(c_bw)]],
                                    ["Forward", "Backward"]):
        for tok in range(num_token):
            mask = inp == tok
            if not mask.any():
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap(tok / max(num_token - 1, 1))],
                       label=f"Tok {tok}", alpha=0.3, s=4)
        ax.set_xlim(xlim);  ax.set_ylim(ylim)
        ax.set_title(f"{lbl} Latents ({method_lbl})", fontweight="bold")
        ax.legend(markerscale=4, fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"{tag} — Shared-range {method_lbl}", fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{tag}_umap_compare.png"))

    # (d) complexity bar
    if p is not None and q is not None:
        fig_c, _ = plt.subplots(figsize=(8, 5))
        fw_vals = [ana_fw.get("S_emp", 0), ana_fw.get("S_theory", 0)]
        bw_vals = [ana_bw.get("S_emp", 0), ana_bw.get("S_theory", 0)]
        x = np.arange(2)
        plt.bar(x - 0.2, fw_vals, 0.35, label="Forward",  color="#4c72b0", alpha=0.85)
        plt.bar(x + 0.2, bw_vals, 0.35, label="Backward", color="#dd8452", alpha=0.85)
        plt.xticks(x, ["Empirical", "Theoretical"])
        plt.ylabel("Statistical Complexity (bits)")
        plt.title(f"{tag} — Complexity comparison")
        plt.legend();  plt.grid(True, alpha=0.3, axis="y");  plt.tight_layout()
        savefig(fig_c, os.path.join(out_dir, f"{tag}_complexity_compare.png"))


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Coin process  p=0.3, q=0.4
# ══════════════════════════════════════════════════════════════════════════

def experiment_1(cfg: dict, out_root: str, all_results: dict):
    tag  = "exp1_coin_p03_q04"
    odir = mkdir(os.path.join(out_root, tag))
    p, q = cfg["coin_p1"], cfg["coin_q1"]

    print(f"\n{'═'*70}")
    print(f"  EXPERIMENT 1 — Coin HMM  p={p}  q={q}")
    print(f"{'═'*70}")

    # ── data ──────────────────────────────────────────────────────────────
    t0 = time.time()
    data, states = coin_generation(
        num_samples=cfg["coin_num_samples"],
        seq_len=cfg["coin_seq_len"], p=p, q=q)
    loader_fw = make_loader(data, states, batch_size=cfg["coin_batch"], mode="forward")
    # backward loader: reversed sequences
    data_rev, states_rev = Rev_HMM_generation(data, states)
    loader_bw = make_loader(data_rev, states_rev, batch_size=cfg["coin_batch"], mode="forward")

    sample_seq = next(iter(loader_fw))[0][0]   # first sequence from first batch

    num_token  = cfg["coin_num_token"]
    max_len    = cfg["coin_seq_len"] - 1        # T-1 (input length after shift)
    theory_fw  = entropy_rate_coin_fw(p, q)
    theory_bw  = entropy_rate_coin_bw(p, q)     # same entropy rate, diff causal structure

    print(f"  Theoretical entropy rate H∞ = {theory_fw:.4f} bits")

    # ── 1a Forward CV ─────────────────────────────────────────────────────
    print("\n  ── 1a: Forward model (5-fold CV) ──")
    cv_fw = train_test_val_pipeline(
        loader_fw,
        test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"),
    )

    # ── 1b Backward CV ────────────────────────────────────────────────────
    print("\n  ── 1b: Backward model (5-fold CV) ──")
    cv_bw = train_test_val_pipeline(
        loader_bw,
        test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="backward",
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"),
    )

    # ── 1c Individual analysis ────────────────────────────────────────────
    print("\n  ── 1c: Analysing best forward model ──")
    ana_fw = analyse_model(
        f"{tag}_fw", cv_fw["best_model"], loader_fw,
        num_token=num_token, out_dir=odir,
        sample_seq=sample_seq, p=p, q=q, mode="forward",
        k_complexity=2, use_t="last",
    )

    print("\n  ── 1c: Analysing best backward model ──")
    ana_bw = analyse_model(
        f"{tag}_bw", cv_bw["best_model"], loader_bw,
        num_token=num_token, out_dir=odir,
        sample_seq=sample_seq, p=p, q=q, mode="backward",
        k_complexity=3, use_t="first",
    )

    # ── 1d Comparison ─────────────────────────────────────────────────────
    print("\n  ── 1d: FW/BW comparison ──")
    compare_fw_bw(
        tag=tag, cv_fw=cv_fw, cv_bw=cv_bw,
        ana_fw=ana_fw, ana_bw=ana_bw,
        loader=loader_fw, num_token=num_token, out_dir=odir,
        sample_seq=sample_seq,
        theory_fw=theory_fw, theory_bw=theory_bw,
        p=p, q=q,
    )

    elapsed = time.time() - t0
    print(f"\n  ✓ Experiment 1 done in {elapsed/60:.1f} min")

    # ── save ──────────────────────────────────────────────────────────────
    exp_results = dict(
        tag=tag, p=p, q=q,
        cv_fw=cv_fw, cv_bw=cv_bw,
        ana_fw=ana_fw, ana_bw=ana_bw,
        theory_fw=theory_fw, theory_bw=theory_bw,
    )
    save_pkl(exp_results, os.path.join(odir, "results.pkl"))
    save_model_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw_best.pt"))
    save_model_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw_best.pt"))
    all_results[tag] = exp_results


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1.2 — Coin process  p=0.1, q=0.9  + heatmaps
# ══════════════════════════════════════════════════════════════════════════

def experiment_1_2(cfg: dict, out_root: str, all_results: dict):
    tag  = "exp1_2_coin_p01_q09"
    odir = mkdir(os.path.join(out_root, tag))
    p, q = cfg["coin_p2"], cfg["coin_q2"]

    print(f"\n{'═'*70}")
    print(f"  EXPERIMENT 1.2 — Coin HMM  p={p}  q={q}")
    print(f"{'═'*70}")

    t0 = time.time()

    # ── data ──────────────────────────────────────────────────────────────
    data, states = coin_generation(
        num_samples=cfg["coin_num_samples_12"],
        seq_len=cfg["coin_seq_len_12"], p=p, q=q)
    loader_fw = make_loader(data, states, batch_size=cfg["coin_batch"], mode="forward")
    data_rev, states_rev = Rev_HMM_generation(data, states)
    loader_bw = make_loader(data_rev, states_rev, batch_size=cfg["coin_batch"], mode="forward")
    sample_seq = next(iter(loader_fw))[0][0]

    num_token = cfg["coin_num_token"]
    max_len   = cfg["coin_seq_len_12"] - 1
    theory_fw = entropy_rate_coin_fw(p, q)
    theory_bw = entropy_rate_coin_bw(p, q)
    print(f"  Theoretical entropy rate H∞ = {theory_fw:.4f} bits")

    # ── CV ─────────────────────────────────────────────────────────────────
    cv_fw = train_test_val_pipeline(
        loader_fw,
        test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"),
    )
    cv_bw = train_test_val_pipeline(
        loader_bw,
        test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="backward",
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"),
    )

    # ── individual analysis ────────────────────────────────────────────────
    ana_fw = analyse_model(
        f"{tag}_fw", cv_fw["best_model"], loader_fw,
        num_token=num_token, out_dir=odir,
        sample_seq=sample_seq, p=p, q=q, mode="forward",
        k_complexity=2, use_t="last",
    )
    ana_bw = analyse_model(
        f"{tag}_bw", cv_bw["best_model"], loader_bw,
        num_token=num_token, out_dir=odir,
        sample_seq=sample_seq, p=p, q=q, mode="backward",
        k_complexity=3, use_t="first",
    )

    # ── comparison ────────────────────────────────────────────────────────
    compare_fw_bw(
        tag=tag, cv_fw=cv_fw, cv_bw=cv_bw,
        ana_fw=ana_fw, ana_bw=ana_bw,
        loader=loader_fw, num_token=num_token, out_dir=odir,
        sample_seq=sample_seq,
        theory_fw=theory_fw, theory_bw=theory_bw,
        p=p, q=q,
    )

    # ── p-q heatmaps (Exp 1.2 specific) ──────────────────────────────────
    print("\n  ── Computing theoretical complexity heatmaps ──")
    pq = cfg["pq_grid"]
    Ss_th_fw, Ss_th_bw, p_th, q_th = heatmap_theory(
        p_values=np.linspace(0.01, 0.99, 50),
        q_values=np.linspace(0.01, 0.99, 50),
    )

    print("\n  ── Running empirical pq-grid experiment ──")
    print("     (trains 2×len(pq)² models — may take a while)")
    Ss_emp, p_emp, q_emp = pq_experiment(
        num_token=num_token,
        d_model=cfg["d_model"],
        max_len=cfg["pq_len"],
        batch_size=cfg["coin_batch"],
        num_samples=cfg["pq_samples"],
        max_epochs=cfg["pq_epochs"],
        lr=cfg["lr"],
        p_values=pq, q_values=pq,
    )

    # main 2×2 heatmap
    plot_heatmap(
        Ss_emp, Ss_th_fw, Ss_th_bw,
        p_emp=p_emp, q_emp=q_emp,
        p_theory=p_th, q_theory=q_th,
        save_path=os.path.join(odir, f"{tag}_complexity_heatmap.png"),
    )

    # difference in statistical complexity (FW - BW, theory)
    diff_complexity_theory = Ss_th_fw - Ss_th_bw
    plot_diff_heatmap(
        diff_complexity_theory, p_th, q_th,
        title="Theoretical Complexity Difference (FW − BW)",
        cbar_label="ΔC (bits)",
        save_path=os.path.join(odir, f"{tag}_diff_complexity_theory.png"),
    )

    # difference in statistical complexity (empirical)
    diff_complexity_emp = Ss_emp[0] - Ss_emp[1]
    plot_diff_heatmap(
        diff_complexity_emp, p_emp, q_emp,
        title="Empirical Complexity Difference (FW − BW)",
        cbar_label="ΔC (bits)",
        save_path=os.path.join(odir, f"{tag}_diff_complexity_emp.png"),
    )

    # compare with Exp 1 (p=0.3, q=0.4) theoretical complexity
    prev = all_results.get("exp1_coin_p03_q04", {})
    if "ana_fw" in prev:
        p1, q1 = cfg["coin_p1"], cfg["coin_q1"]
        S_fw_prev = statistical_complexity(p1, q1, mode="forward")
        S_bw_prev = statistical_complexity(p1, q1, mode="backward")
        S_fw_curr = statistical_complexity(p, q, mode="forward")
        S_bw_curr = statistical_complexity(p, q, mode="backward")
        fig, ax = plt.subplots(figsize=(8, 5))
        bar_lbls = ["FW prev", "BW prev", "FW curr", "BW curr"]
        bar_vals = [S_fw_prev, S_bw_prev, S_fw_curr, S_bw_curr]
        bars     = ax.bar(bar_lbls, bar_vals,
                          color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"],
                          alpha=0.85, edgecolor="k", linewidth=0.8)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_ylabel("Statistical Complexity (bits)")
        ax.set_title("Theoretical Complexity: Exp 1 vs Exp 1.2")
        ax.grid(True, alpha=0.3, axis="y");  plt.tight_layout()
        savefig(fig, os.path.join(odir, f"{tag}_complexity_exp1_vs_exp12.png"))

    # difference in model perplexity (FW - BW) over p-q grid
    diff_ppl_emp = np.exp(Ss_emp[0]) - np.exp(Ss_emp[1])
    plot_diff_heatmap(
        diff_ppl_emp, p_emp, q_emp,
        title="Empirical Perplexity Difference (FW − BW)",
        cbar_label="ΔPPL",
        save_path=os.path.join(odir, f"{tag}_diff_ppl.png"),
    )

    elapsed = time.time() - t0
    print(f"\n  ✓ Experiment 1.2 done in {elapsed/60:.1f} min")

    exp_results = dict(
        tag=tag, p=p, q=q,
        cv_fw=cv_fw, cv_bw=cv_bw,
        ana_fw=ana_fw, ana_bw=ana_bw,
        theory_fw=theory_fw, theory_bw=theory_bw,
        Ss_emp=Ss_emp, p_emp=p_emp, q_emp=q_emp,
        Ss_th_fw=Ss_th_fw, Ss_th_bw=Ss_th_bw, p_th=p_th, q_th=q_th,
    )
    save_pkl(exp_results, os.path.join(odir, "results.pkl"))
    save_model_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw_best.pt"))
    save_model_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw_best.pt"))
    all_results[tag] = exp_results


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Flower process  n=4, m=2
# ══════════════════════════════════════════════════════════════════════════

def experiment_2(cfg: dict, out_root: str, all_results: dict):
    tag   = "exp2_flower_n4_m2"
    odir  = mkdir(os.path.join(out_root, tag))
    n, m  = cfg["flower_n"], cfg["flower_m"]
    num_token = n + m      # 6 total token types

    print(f"\n{'═'*70}")
    print(f"  EXPERIMENT 2 — Flower HMM  n={n}  m={m}  vocab={num_token}")
    print(f"{'═'*70}")

    t0 = time.time()

    # ── data ──────────────────────────────────────────────────────────────
    # Use consistent dice probabilities
    rng = np.random.default_rng(42)
    dice_probs = rng.dirichlet(np.ones(m), size=n)
    print(f"  Dice probabilities:\n{dice_probs}")

    data, states = flower_process_generation(
        num_samples=cfg["flower_num_samples"],
        seq_len=cfg["flower_seq_len"],
        n=n, m=m, dice_probs=dice_probs,
    )

    # Build datasets manually so both FW and BW loaders share same data split
    seqs_fw = data
    seqs_bw = [list(reversed(s)) for s in data]

    ds_fw = FlowerDataset(seqs_fw, seq_len=len(seqs_fw[0]))
    ds_bw = FlowerDataset(seqs_bw, seq_len=len(seqs_bw[0]))
    loader_fw = DataLoader(ds_fw, batch_size=cfg["flower_batch"], shuffle=True)
    loader_bw = DataLoader(ds_bw, batch_size=cfg["flower_batch"], shuffle=True)

    sample_seq_fw = next(iter(loader_fw))[0][0]

    max_len = len(seqs_fw[0]) - 1

    # No closed-form for flower complexity → use None for p,q in analyse_model
    theory_fw = float("nan")
    theory_bw = float("nan")

    # ── 2a Forward CV ─────────────────────────────────────────────────────
    print("\n  ── 2a: Forward model (5-fold CV) ──")
    cv_fw = train_test_val_pipeline(
        loader_fw,
        test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"),
    )

    # ── 2b Backward CV ────────────────────────────────────────────────────
    print("\n  ── 2b: Backward model (5-fold CV) ──")
    cv_bw = train_test_val_pipeline(
        loader_bw,
        test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="backward",
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"),
    )

    # ── 2c Analysis ───────────────────────────────────────────────────────
    print("\n  ── 2c: Analysing best forward model ──")
    # k for flower: n+m causal states at most
    ana_fw = analyse_model(
        f"{tag}_fw", cv_fw["best_model"], loader_fw,
        num_token=num_token, out_dir=odir,
        sample_seq=sample_seq_fw,
        p=None, q=None, mode="forward",
        k_complexity=n + m, use_t="last",
    )

    print("\n  ── 2c: Analysing best backward model ──")
    ana_bw = analyse_model(
        f"{tag}_bw", cv_bw["best_model"], loader_bw,
        num_token=num_token, out_dir=odir,
        sample_seq=sample_seq_fw,
        p=None, q=None, mode="backward",
        k_complexity=n + m, use_t="first",
    )

    # ── 2d Comparison ─────────────────────────────────────────────────────
    print("\n  ── 2d: FW/BW comparison ──")
    # loss curves (no theory line since flower has no closed form here)
    rec_fw = cv_fw["best_recorder"]
    rec_bw = cv_bw["best_recorder"]
    fig_l, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, rec, lbl in zip(axes, [rec_fw, rec_bw], ["Forward", "Backward"]):
        ax.plot(rec.step_loss,     color="steelblue",  lw=1.2, alpha=0.8, label="Train loss")
        if rec.step_val_loss:
            ax.plot(rec.step_val_loss, color="darkorange", lw=1.2, alpha=0.8, label="Val loss")
        ax.set_xlabel("Gradient Step");  ax.set_ylabel("Loss (bits)")
        ax.set_title(f"{lbl} — Flower process", fontweight="bold")
        ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    fig_l.tight_layout()
    savefig(fig_l, os.path.join(odir, f"{tag}_loss_curves.png"))

    fig_a = FW_BW_attention_comparison(
        cv_fw["best_model"], cv_bw["best_model"],
        sample_seq_fw, title_prefix=f"{tag} ",
    )
    savefig(fig_a, os.path.join(odir, f"{tag}_attn_compare.png"))

    # shared-range UMAP
    lfw, ifw, _ = latent_extraction(cv_fw["best_model"], loader_fw, max_batches=25)
    lbw, ibw, _ = latent_extraction(cv_bw["best_model"], loader_bw, max_batches=25)
    method = "umap" if UMAP_AVAILABLE else "pca"
    c_fw = project_2d(lfw.reshape(-1, lfw.shape[-1])[:8000], method)
    c_bw = project_2d(lbw.reshape(-1, lbw.shape[-1])[:8000], method)
    xlim = (min(c_fw[:, 0].min(), c_bw[:, 0].min()) - 0.5,
            max(c_fw[:, 0].max(), c_bw[:, 0].max()) + 0.5)
    ylim = (min(c_fw[:, 1].min(), c_bw[:, 1].min()) - 0.5,
            max(c_fw[:, 1].max(), c_bw[:, 1].max()) + 0.5)
    method_lbl = "UMAP" if UMAP_AVAILABLE else "PCA"
    cmap2 = plt.cm.tab10
    fig_u, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, inp, lbl in zip(axes2,
                                    [c_fw, c_bw],
                                    [ifw.reshape(-1)[:len(c_fw)],
                                     ibw.reshape(-1)[:len(c_bw)]],
                                    ["Forward", "Backward"]):
        for tok in range(num_token):
            mask = inp == tok
            if not mask.any():
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[cmap2(tok / max(num_token - 1, 1))],
                       label=f"Tok {tok}", alpha=0.3, s=4)
        ax.set_xlim(xlim);  ax.set_ylim(ylim)
        ax.set_title(f"{lbl} Latents ({method_lbl})", fontweight="bold")
        ax.legend(markerscale=4, fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)
    fig_u.suptitle(f"{tag} — Shared-range {method_lbl}", fontweight="bold")
    fig_u.tight_layout()
    savefig(fig_u, os.path.join(odir, f"{tag}_umap_compare.png"))

    # complexity bar (empirical only for flower)
    fig_cx, ax_cx = plt.subplots(figsize=(7, 5))
    vals = [ana_fw["S_emp"], ana_bw["S_emp"]]
    bars = ax_cx.bar(["Forward", "Backward"], vals,
                     color=["#4c72b0", "#dd8452"], alpha=0.85,
                     edgecolor="k", linewidth=0.8)
    ax_cx.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax_cx.set_ylabel("Empirical Statistical Complexity (bits)")
    ax_cx.set_title(f"{tag} — Empirical complexity")
    ax_cx.grid(True, alpha=0.3, axis="y");  plt.tight_layout()
    savefig(fig_cx, os.path.join(odir, f"{tag}_complexity.png"))

    elapsed = time.time() - t0
    print(f"\n  ✓ Experiment 2 done in {elapsed/60:.1f} min")

    exp_results = dict(
        tag=tag, n=n, m=m, dice_probs=dice_probs,
        cv_fw=cv_fw, cv_bw=cv_bw,
        ana_fw=ana_fw, ana_bw=ana_bw,
    )
    save_pkl(exp_results, os.path.join(odir, "results.pkl"))
    save_model_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw_best.pt"))
    save_model_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw_best.pt"))
    all_results[tag] = exp_results


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    OUT_ROOT = "results"
    mkdir(OUT_ROOT)
    mkdir(os.path.join(OUT_ROOT, "models"))

    all_results: dict = {}
    t_start = time.time()

    # ── Run experiments in sequence ───────────────────────────────────────
    experiment_1(CFG, OUT_ROOT, all_results)
    experiment_1_2(CFG, OUT_ROOT, all_results)
    experiment_2(CFG, OUT_ROOT, all_results)

    # ── Persist master results dict ───────────────────────────────────────
    save_pkl(all_results, os.path.join(OUT_ROOT, "all_results.pkl"))

    total = (time.time() - t_start) / 60
    print(f"\n{'═'*70}")
    print(f"  ALL EXPERIMENTS COMPLETE  —  {total:.1f} min total")
    print(f"  Results tree:")
    for root, dirs, files in os.walk(OUT_ROOT):
        level = root.replace(OUT_ROOT, "").count(os.sep)
        indent = "  " + "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            print(f"{indent}  {f}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()