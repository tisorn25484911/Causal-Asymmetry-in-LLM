import os
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs import CONFIGS
from Data_generation import CoinDataset, coin_generation
from Flower_process_generation import FlowerDataset, flower_process_generation
from OneHot_model import OneHotDecoder
from Training_model import _loader
from Model_analysis import (
    coin_true_conditional,
    flower_complexity,
    perplexity_calculation,
    perplexity_ind_CE,
    self_generated_entropy_rate,
    statistical_complexity,
    statistical_complexity_empirical,
    stepwise_kl_coin,
    savefig,
)
from utils import check_weight_meta, coin_tag, entropy_rate_coin, flower_tag, mkdir

# ── LaTeX rendering ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":       True,
    "font.family":       "serif",
    "font.serif":        ["Computer Modern Roman"],
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        120,
})


# =============================================================================
# CONFIG
# =============================================================================
# Evaluation-only knobs.  Everything describing the *models* and the
# *processes* comes from configs.py, so this file can no longer disagree with
# the run that produced the weights it loads (IMPROVEMENT_PLAN.md A4).  It
# previously hard-coded main_large's tags and CFG, so running it after
# Main_call.py would have scored p=0.4,q=0.8 weights against p=0.3,q=0.4 data;
# only the d_model mismatch raising in load_state_dict prevented that.
EVAL_CFG = dict(
    max_len        = 2000,
    num_samples    = 500,
    batch_size     = 32,
    max_batches    = None,
    coin_seq_len_eval = 2000,   # eval sequence length, independent of training
    # KL generation
    kl_gen_len  = 2000,
    kl_burn_in  = 200,
)

RUN = dict(
    config      = "LARGE",  # which configs.py entry produced the weights
    results_dir = None,     # None → the config's out_root
    out_dir     = None,
    exp         = "all",    # "all" | "exp1" | "exp1_2" | "exp2"
)


def build_cfg():
    """Config that trained the weights, plus the evaluation-only knobs."""
    cfg = dict(CONFIGS[RUN["config"]])
    cfg.update(EVAL_CFG)
    return cfg


# =============================================================================
# MODEL LOADER
# =============================================================================
def load_model(path, num_token, cfg, mode):
    # A4: verify the checkpoint is what we think it is, instead of guessing.
    # Running an evaluator against weights trained with a different config used
    # to be caught only by load_state_dict raising on a d_model mismatch --
    # align d_model and it would silently score the wrong process.
    check_weight_meta(path, {
        "d_model": cfg["d_model"], "n_layers": cfg["n_layers"],
        "token_size": num_token, "mode": mode,
    })
    model = OneHotDecoder(
        token_size=num_token, d_model=cfg["d_model"],
        max_len=cfg["max_len"], mode=mode, n_layers=cfg["n_layers"])
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"  loaded {path}  [{mode}]")
    return model


# =============================================================================
# METRIC 0 — Standard autoregressive perplexity
# Informational only — not used for asymmetry comparison because each model
# evaluates on its OWN generated sequence (different state distributions).
# =============================================================================
def compute_standard_ppl(model, len_seq=2000):
    # FIX [1]: removed (loader, max_batches) — perplexity_ind_model is
    # autoregressive and does not accept a loader.
    return self_generated_entropy_rate(model, len_seq=len_seq, start_token=0)


# =============================================================================
# METRIC 1 — perplexity_calculation (primary asymmetry metric)
# Both models evaluated on the SAME ground-truth sequences.
# Respects FW/BW batch-swap convention.
# CE = log2(PPL) → H_inf.  delta_CE = CE_BW - CE_FW is the asymmetry signal.
# =============================================================================
def compute_ce(model, loader, max_batches=None):
    """Returns (PPL, CE bits) via perplexity_calculation."""
    ppl = perplexity_calculation(model, loader, max_batches=max_batches)
    ce  = float(np.log2(ppl))
    return ppl, ce


# =============================================================================
# METRIC 2 — perplexity_ind_CE (soft-label CE on ground-truth sequences)
# Same loader as Metric 1 — no distribution shift.
# CE_t = -sum_x P_true(x | cur_tok) * log2 P_model(x | context)
# vs Metric 1 which uses one-hot P_true = delta(x_true).
# =============================================================================
def compute_ind_ce(model, loader, p, q, cfg):
    # FIX [2]: added loader argument; passes loader to corrected
    # perplexity_ind_CE API instead of old (len_seq, start_token) params.
    ppl = perplexity_ind_CE(
        model, loader, p, q,
        max_batches=cfg.get("max_batches"),
    )
    ce = float(np.log2(ppl))
    return ppl, ce


# =============================================================================
# KL DIVERGENCE — stationary distribution (coin only, autoregressive)
# =============================================================================
def coin_true_distribution(p, q, num_token=3):
    pi = np.zeros(num_token)
    pi[1] = p / (p + q)
    pi[0] = q * (1 - p) / (p + q)
    pi[2] = p * q / (p + q)
    return pi


def kl_from_true_coin(model, p, q, cfg, num_token=3):
    """D_KL(pi_true || pi_model) from model's own generated sequence."""
    burn_in = cfg.get("kl_burn_in", 200)
    gen_len = cfg.get("kl_gen_len", 2000)
    model.eval()
    is_bw  = (getattr(model, "mode", "forward") == "backward")
    device = next(model.parameters()).device

    context      = [0]
    token_counts = np.zeros(num_token)

    with torch.no_grad():
        for i in range(gen_len + burn_in):
            x      = torch.tensor([context], device=device)
            out    = model(x)
            logits = out[:, 0, :] if is_bw else out[:, -1, :]
            probs  = torch.softmax(logits[0], dim=-1).cpu().numpy()
            next_token = int(np.random.choice(num_token, p=probs))
            if is_bw:
                context = [next_token] + context
            else:
                context = context + [next_token]
            if i >= burn_in:
                token_counts[next_token] += 1

    pi_model = token_counts / token_counts.sum()
    pi_true  = coin_true_distribution(p, q, num_token)
    kl_div   = float(np.sum(
        pi_true * np.log2(pi_true / (pi_model + 1e-12) + 1e-12)))

    print(f"    pi_true  = {np.round(pi_true, 4)}")
    print(f"    pi_model = {np.round(pi_model, 4)}")
    print(f"    D_KL     = {kl_div:.6f} bits")
    return kl_div, pi_true, pi_model


# =============================================================================
# STEPWISE KL -- per-token average KL on ground-truth sequences.
# coin_true_conditional and stepwise_kl_coin used to be byte-identical
# copies of the Model_analysis versions (IMPROVEMENT_PLAN.md C4); they are
# now imported.  Verified identical before deleting.
# =============================================================================


# =============================================================================
# PLOTS — LaTeX font throughout
# =============================================================================
def plot_dual_ppl_comparison(tag, results, h_inf=None, out_dir=None):
    """
    Four-panel 2x2 figure.
      [0,0] Standard PPL (autoregressive, informational)
      [0,1] 2^CE from perplexity_ind_CE (soft-label, ground-truth sequences)
      [1,0] CE in bits from perplexity_ind_CE
      [1,1] Mean stepwise KL divergence

    FIX [3]: changed subplots(2,2) axes indexing from broken 1D
             (axes[1], axes[2], axes[3]) to correct 2D (axes[0,1] etc.)
    FIX [4]: every panels tuple now carries ref_val and ref_lbl (6 elements)
             to match the 6-element loop unpacking.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    labels = [r"$\leftarrow$ Forward", r"Backward $\rightarrow$"]
    colors = ["#4c72b0", "#dd8452"]

    h_ref  = 2**h_inf if h_inf is not None else None

    panels = [
        # (ax, values, ylabel, title, ref_val, ref_lbl)
        (axes[0, 0],
         [results["ppl_std_fw"],  results["ppl_std_bw"]],
         r"Perplexity  $2^{\mathcal{L}}$",
         r"\textit{Metric 0}: Standard PPL (informational only)",
         h_ref,
         r"$2^{H_\infty}$"),
        (axes[0, 1],
         [results["exp_ind_fw"],  results["exp_ind_bw"]],
         r"$2^{H(P_\mathrm{true},\,P_\mathrm{model})}$",
         r"\textit{Metric 2}: $\exp(\mathrm{soft-label CE})$",
         h_ref,
         r"$2^{H_\infty}$"),
        (axes[1, 0],
         [results["ce_ind_fw"],   results["ce_ind_bw"]],
         r"$H(P_\mathrm{true},\,P_\mathrm{model})$ (bits/token)",
         r"\textit{Metric 2}: Soft-label cross-entropy",
         h_inf,
         r"$H_\infty$"),
        (axes[1, 1],
         [results["skl_fw"],      results["skl_bw"]],
         r"$\overline{D_\mathrm{KL}}\bigl(P_\mathrm{true}\,\|\,P_\mathrm{model}\bigr)$ (bits)",
         r"\textit{Metric 3}: Mean stepwise $D_\mathrm{KL}$",
         None, None),
    ]

    for ax, vals, ylabel, title, ref_val, ref_lbl in panels:
        delta = vals[1] - vals[0]
        bars  = ax.bar(labels, vals, color=colors, alpha=0.85,
                       edgecolor="k", width=0.45)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        if ref_val is not None:
            ax.axhline(ref_val, color="crimson", ls="--", lw=1.5,
                       label=rf"{ref_lbl} $= {ref_val:.4f}$")
            ax.legend(fontsize=8)
        sign = (r"$\mathcal{L}_\mathrm{BW} > \mathcal{L}_\mathrm{FW}$"
                if delta > 0 else
                r"$\mathcal{L}_\mathrm{FW} > \mathcal{L}_\mathrm{BW}$")
        ax.set_ylabel(ylabel)
        ax.set_title(
            rf"{title}"
            "\n"
            rf"$\Delta = {delta:+.4f}$  ({sign})",
            fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    tag_safe = tag.replace("_", r"\_")
    fig.suptitle(
        rf"\textbf{{{tag_safe}}} --- "
        r"Causal Asymmetry: $\mathcal{L}_\mathrm{BW} - \mathcal{L}_\mathrm{FW}$",
        fontsize=11)
    fig.tight_layout()
    if out_dir:
        savefig(fig, os.path.join(out_dir, f"{tag}_dual_ppl.png"))
    return fig


def plot_kl_distribution(tag, pi_true, pi_model_fw, kl_fw,
                         pi_model_bw, kl_bw, num_token, out_dir=None):
    x            = np.arange(num_token)
    width        = 0.35
    token_labels = [rf"$x_{i}$" for i in range(num_token)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, pi_model, kl, lbl, col in zip(
        axes,
        [pi_model_fw,  pi_model_bw],
        [kl_fw,        kl_bw],
        [r"$\leftarrow$ Forward model", r"Backward model $\rightarrow$"],
        ["#4c72b0",    "#dd8452"],
    ):
        b_true  = ax.bar(x - width/2, pi_true,  width,
                         label=r"$\pi_\mathrm{true}$ (theoretical)",
                         color="#2ecc71", alpha=0.85, edgecolor="k")
        b_model = ax.bar(x + width/2, pi_model, width,
                         label=rf"$\hat{{\pi}}$ {lbl}",
                         color=col, alpha=0.85, edgecolor="k")
        ax.bar_label(b_true,  fmt="%.3f", padding=2, fontsize=8)
        ax.bar_label(b_model, fmt="%.3f", padding=2, fontsize=8)
        ax.text(0.97, 0.95,
                rf"$D_{{\mathrm{{KL}}}}(\pi_\mathrm{{true}}\,\|\,"
                rf"\hat{{\pi}}) = {kl:.5f}$ bits",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="grey", alpha=0.9))
        ax.set_xticks(x); ax.set_xticklabels(token_labels, fontsize=10)
        ax.set_ylabel(r"Token probability  $\pi(x)$")
        ax.set_title(
            rf"{lbl}"
            "\n"
            rf"$D_{{\mathrm{{KL}}}}(\pi_\mathrm{{true}}\,\|\,\hat{{\pi}}) = {kl:.5f}$ bits")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    tag_safe = tag.replace("_", r"\_")
    fig.suptitle(
        rf"\textbf{{{tag_safe}}} --- "
        r"Stationary distribution: $\pi_\mathrm{true}$ vs $\hat{\pi}_\mathrm{model}$",
        fontsize=11)
    fig.tight_layout()
    if out_dir:
        savefig(fig, os.path.join(out_dir, f"{tag}_kl_distribution.png"))
    return fig


def plot_stepwise_kl(tag, mean_kl_fw, mean_kl_bw,
                     per_tok_fw, per_tok_bw,
                     per_tok_count_fw, num_token, out_dir=None):
    fig, axes  = plt.subplots(1, 2, figsize=(13, 5))
    colors     = ["#4c72b0", "#dd8452"]
    x_tok      = np.arange(num_token)
    delta      = mean_kl_bw - mean_kl_fw
    hard       = (r"$D_\mathrm{KL}^\mathrm{BW} > D_\mathrm{KL}^\mathrm{FW}$"
                  if delta > 0 else
                  r"$D_\mathrm{KL}^\mathrm{FW} > D_\mathrm{KL}^\mathrm{BW}$")

    ax = axes[0]
    bars = ax.bar([r"$\leftarrow$ Forward", r"Backward $\rightarrow$"],
                  [mean_kl_fw, mean_kl_bw],
                  color=colors, alpha=0.85, edgecolor="k", width=0.45)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=10)
    ax.text(0.97, 0.95,
            rf"$\Delta D_{{\mathrm{{KL}}}} = {delta:+.5f}$ bits"
            "\n" + hard,
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.9))
    ax.set_ylabel(
        r"$\overline{D_\mathrm{KL}}\bigl(P_\mathrm{true}(\cdot\mid x_t)"
        r"\,\|\,P_\mathrm{model}\bigr)$ (bits)")
    ax.set_title(r"Mean stepwise $D_\mathrm{KL}$ over ground-truth sequences")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    width     = 0.35
    tok_labels = [rf"$x_{i}$" for i in range(num_token)]
    b_fw = ax.bar(x_tok - width/2, per_tok_fw, width,
                  label=r"$\leftarrow$ Forward",
                  color=colors[0], alpha=0.85, edgecolor="k")
    b_bw = ax.bar(x_tok + width/2, per_tok_bw, width,
                  label=r"Backward $\rightarrow$",
                  color=colors[1], alpha=0.85, edgecolor="k")
    ax.bar_label(b_fw, fmt="%.4f", padding=2, fontsize=8)
    ax.bar_label(b_bw, fmt="%.4f", padding=2, fontsize=8)
    for i, n in enumerate(per_tok_count_fw.astype(int)):
        ax.text(i, -0.008, rf"$n\!=\!{n}$", ha="center", fontsize=7,
                color=colors[0], transform=ax.get_xaxis_transform())
    ax.set_xticks(x_tok); ax.set_xticklabels(tok_labels, fontsize=10)
    ax.set_ylabel(
        r"$D_\mathrm{KL}\bigl(P_\mathrm{true}(\cdot\mid x_i)"
        r"\,\|\,P_\mathrm{model}\bigr)$ (bits)")
    ax.set_title(r"Per-token $D_\mathrm{KL}$ --- which transitions are hardest?")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    tag_safe = tag.replace("_", r"\_")
    fig.suptitle(
        rf"\textbf{{{tag_safe}}} --- "
        r"Stepwise $D_{\mathrm{KL}}\bigl(P_\mathrm{true}(\cdot\mid x_t)"
        r"\,\|\,P_\mathrm{model}\bigr)$",
        fontsize=11)
    fig.tight_layout()
    if out_dir:
        savefig(fig, os.path.join(out_dir, f"{tag}_stepwise_kl.png"))
    return fig


def plot_complexity(tag, S_emp_fw, S_emp_bw, S_th_fw=None, S_th_bw=None,
                    out_dir=None):
    labels = [r"$\leftarrow$ Forward ($\mathcal{C}^+$)",
              r"Backward ($\mathcal{C}^-$) $\rightarrow$"]
    colors = ["#4c72b0", "#dd8452"]
    x      = np.arange(2)
    fig, ax = plt.subplots(figsize=(8, 5))
    b_emp = ax.bar(x - 0.2, [S_emp_fw, S_emp_bw], 0.35,
                   label=r"Empirical $\hat{\mathcal{C}}$",
                   color=colors, alpha=0.85, edgecolor="k")
    ax.bar_label(b_emp, fmt="%.4f", padding=3, fontsize=9)
    if S_th_fw is not None and S_th_bw is not None:
        b_th = ax.bar(x + 0.2, [S_th_fw, S_th_bw], 0.35,
                      label=r"Theoretical $\mathcal{C}$",
                      color=colors, alpha=0.45, edgecolor="k", hatch="//")
        ax.bar_label(b_th, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"Statistical Complexity $\mathcal{C}$ (bits)")
    tag_safe = tag.replace("_", r"\_")
    ax.set_title(
        rf"\textbf{{{tag_safe}}} --- "
        r"Causal State Complexity  ($\mathcal{C}^- > \mathcal{C}^+$ expected)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    if out_dir:
        savefig(fig, os.path.join(out_dir, f"{tag}_complexity.png"))
    return fig


# =============================================================================
# EXPERIMENT EVALUATORS
# =============================================================================
def eval_coin(tag, p, q, seq_len, models_dir, out_root, cfg):
    odir      = mkdir(os.path.join(out_root, tag))
    num_token = cfg["coin_num_token"]
    print(f"\n{'='*60}\n  EVAL — {tag}  p={p}  q={q}\n{'='*60}")

    data, _ = coin_generation(
        num_samples=cfg["num_samples"], seq_len=seq_len, p=p, q=q)
    ds     = CoinDataset(data, seq_len=seq_len)
    loader = _loader(ds, cfg["batch_size"])

    fw_path = os.path.join(models_dir, f"{tag}_fw.pt")
    bw_path = os.path.join(models_dir, f"{tag}_bw.pt")
    if not os.path.exists(fw_path) or not os.path.exists(bw_path):
        print(f"  weights not found — skipping {tag}")
        return

    model_fw = load_model(fw_path, num_token, cfg, mode="forward")
    model_bw = load_model(bw_path, num_token, cfg, mode="backward")

    h_inf = entropy_rate_coin(p, q)

    # --- Metric 0: Standard PPL (informational, not used for asymmetry) ------
    # FIX [1]: call with len_seq only — no loader argument.
    print("\n  [Metric 0] Standard PPL (autoregressive self-generated, informational):")
    ppl_std_fw = compute_standard_ppl(model_fw, len_seq=seq_len)
    print(f"    FW: PPL={ppl_std_fw:.4f}")
    ppl_std_bw = compute_standard_ppl(model_bw, len_seq=seq_len)
    print(f"    BW: PPL={ppl_std_bw:.4f}")
    print(f"    delta_PPL={ppl_std_bw-ppl_std_fw:+.4f}  (informational only)")

    # --- Metric 1: CE on ground-truth (primary asymmetry metric) -------------
    print("\n  [Metric 1] perplexity_calculation (ground-truth loader, one-hot targets):")
    ppl_calc_fw, ce_calc_fw = compute_ce(model_fw, loader, cfg["max_batches"])
    print(f"    FW: PPL={ppl_calc_fw:.4f}  CE={ce_calc_fw:.4f} bits")
    ppl_calc_bw, ce_calc_bw = compute_ce(model_bw, loader, cfg["max_batches"])
    print(f"    BW: PPL={ppl_calc_bw:.4f}  CE={ce_calc_bw:.4f} bits")
    print(f"    H_inf={h_inf:.4f}  delta_CE={ce_calc_bw-ce_calc_fw:+.4f}")

    # --- Metric 2: Soft-label CE on ground-truth loader ----------------------
    # FIX [2]: passes loader to compute_ind_ce (corrected API).
    print("\n  [Metric 2] perplexity_ind_CE (soft-label CE, ground-truth loader):")
    exp_ind_fw, ce_ind_fw = compute_ind_ce(model_fw, loader, p, q, cfg)
    print(f"    FW: exp(CE)={exp_ind_fw:.4f}  CE={ce_ind_fw:.4f} bits")
    exp_ind_bw, ce_ind_bw = compute_ind_ce(model_bw, loader, p, q, cfg)
    print(f"    BW: exp(CE)={exp_ind_bw:.4f}  CE={ce_ind_bw:.4f} bits")
    print(f"    H_inf={h_inf:.4f}  delta_CE={ce_ind_bw-ce_ind_fw:+.4f}")

    # --- Metric 3: Stepwise KL on ground-truth loader ------------------------
    # FIX [5]: passes loader; removed old (p, q, cfg, num_token) call order.
    # FIX [7]: label corrected to [Metric 3].
    print("\n  [Metric 3] Stepwise KL divergence per token (ground-truth sequences):")
    print("    Forward ...")
    skl_fw, skl_per_tok_fw, skl_count_fw = stepwise_kl_coin(
        model_fw, loader, p, q, num_token, cfg["max_batches"])
    print("    Backward ...")
    skl_bw, skl_per_tok_bw, _ = stepwise_kl_coin(
        model_bw, loader, p, q, num_token, cfg["max_batches"])
    print(f"    delta_KL={skl_bw-skl_fw:+.6f}")

    # FIX [6]: results dict uses consistent variable names matched to plots.
    # FIX [8]: removed dead triple-quoted comment block.
    results = dict(
        ppl_std_fw=ppl_std_fw,   ppl_std_bw=ppl_std_bw,
        exp_ind_fw=exp_ind_fw,   exp_ind_bw=exp_ind_bw,
        ce_ind_fw=ce_ind_fw,     ce_ind_bw=ce_ind_bw,
        skl_fw=skl_fw,           skl_bw=skl_bw,
    )
    plot_dual_ppl_comparison(tag, results, h_inf=h_inf, out_dir=odir)

    # --- KL divergence — stationary distribution -----------------------------
    print("\n  KL divergence from true stationary distribution:")
    print("    Forward ...")
    kl_fw, pi_true, pi_model_fw = kl_from_true_coin(model_fw, p, q, cfg, num_token)
    print("    Backward ...")
    kl_bw, _,       pi_model_bw = kl_from_true_coin(model_bw, p, q, cfg, num_token)
    plot_kl_distribution(tag, pi_true, pi_model_fw, kl_fw,
                         pi_model_bw, kl_bw, num_token, out_dir=odir)

    # Plot stepwise KL breakdown
    plot_stepwise_kl(tag, skl_fw, skl_bw,
                     skl_per_tok_fw, skl_per_tok_bw,
                     skl_count_fw, num_token, out_dir=odir)

    # --- Statistical complexity -----------------------------------------------
    S_emp_fw = statistical_complexity_empirical(
        model_fw, loader, max_batches=cfg["max_batches"], use_t="last",  k=2)
    S_emp_bw = statistical_complexity_empirical(
        model_bw, loader, max_batches=cfg["max_batches"], use_t="first", k=3)
    S_th_fw  = statistical_complexity(p, q, mode="forward")
    S_th_bw  = statistical_complexity(p, q, mode="backward")
    plot_complexity(tag, S_emp_fw, S_emp_bw, S_th_fw, S_th_bw, out_dir=odir)

    # --- Summary --------------------------------------------------------------
    sep = "=" * 60
    print(f"\n{sep}\n  SUMMARY — {tag}\n{sep}")
    print(f"  H_inf (theory)            = {h_inf:.4f} bits")
    print()
    print(f"  [Metric 0] Standard PPL (informational only)")
    print(f"    PPL  FW / BW            = {ppl_std_fw:.4f} / {ppl_std_bw:.4f}")
    print(f"    delta_PPL               = {ppl_std_bw-ppl_std_fw:+.4f}")
    print()
    print(f"  [Metric 1] perplexity_calculation (one-hot CE, ground-truth loader)")
    print(f"    PPL  FW / BW            = {ppl_calc_fw:.4f} / {ppl_calc_bw:.4f}")
    print(f"    CE   FW / BW            = {ce_calc_fw:.4f} / {ce_calc_bw:.4f} bits")
    print(f"    delta_CE                = {ce_calc_bw-ce_calc_fw:+.4f}  "
          f"({'BW harder' if ce_calc_bw > ce_calc_fw else 'FW harder'})")
    print()
    print(f"  [Metric 2] perplexity_ind_CE (soft-label CE, ground-truth loader)")
    print(f"    exp(CE) FW / BW         = {exp_ind_fw:.4f} / {exp_ind_bw:.4f}")
    print(f"    CE      FW / BW         = {ce_ind_fw:.4f} / {ce_ind_bw:.4f} bits")
    print(f"    delta_CE                = {ce_ind_bw-ce_ind_fw:+.4f}  "
          f"({'BW harder' if ce_ind_bw > ce_ind_fw else 'FW harder'})")
    print()
    print(f"  [Metric 3] Stepwise KL (ground-truth loader)")
    print(f"    KL   FW / BW            = {skl_fw:.6f} / {skl_bw:.6f} bits")
    print(f"    delta_KL                = {skl_bw-skl_fw:+.6f}  "
          f"({'BW harder' if skl_bw > skl_fw else 'FW harder'})")
    print()
    print(f"  D_KL stationary FW / BW  = {kl_fw:.6f} / {kl_bw:.6f} bits")
    print(f"  C+ emp / th              = {S_emp_fw:.4f} / {S_th_fw:.4f}")
    print(f"  C- emp / th              = {S_emp_bw:.4f} / {S_th_bw:.4f}")
    print(sep)

    gc.collect()
    return dict(
        ppl_std_fw=ppl_std_fw,   ppl_std_bw=ppl_std_bw,
        ppl_calc_fw=ppl_calc_fw, ce_calc_fw=ce_calc_fw,
        ppl_calc_bw=ppl_calc_bw, ce_calc_bw=ce_calc_bw,
        exp_ind_fw=exp_ind_fw,   ce_ind_fw=ce_ind_fw,
        exp_ind_bw=exp_ind_bw,   ce_ind_bw=ce_ind_bw,
        skl_fw=skl_fw,           skl_bw=skl_bw,
        kl_fw=kl_fw,             kl_bw=kl_bw,
        S_emp_fw=S_emp_fw,       S_emp_bw=S_emp_bw,
        S_th_fw=S_th_fw,         S_th_bw=S_th_bw,
    )


def eval_flower(tag, n, m, models_dir, out_root, cfg):
    """Flower process: no closed-form true conditional, Metric 1 only."""
    odir      = mkdir(os.path.join(out_root, tag))
    num_token = n + m
    print(f"\n{'='*60}\n  EVAL — {tag}  n={n}  m={m}\n{'='*60}")

    rng        = np.random.default_rng(99)
    dice_probs = rng.dirichlet(np.ones(m), size=n)
    data, _ = flower_process_generation(
        num_samples=cfg["num_samples"], seq_len=cfg["flower_seq_len"],
        n=n, m=m, dice_probs=dice_probs)
    ds     = FlowerDataset(data, seq_len=len(data[0]))
    loader = _loader(ds, cfg["batch_size"])

    fw_path = os.path.join(models_dir, f"{tag}_fw.pt")
    bw_path = os.path.join(models_dir, f"{tag}_bw.pt")
    if not os.path.exists(fw_path) or not os.path.exists(bw_path):
        print(f"  weights not found — skipping {tag}")
        return

    model_fw = load_model(fw_path, num_token, cfg, mode="forward")
    model_bw = load_model(bw_path, num_token, cfg, mode="backward")

    print("\n  [Metric 1] perplexity_calculation (ground-truth sequences):")
    ppl_calc_fw, ce_calc_fw = compute_ce(model_fw, loader, cfg["max_batches"])
    print(f"    FW: PPL={ppl_calc_fw:.4f}  CE={ce_calc_fw:.4f} bits")
    ppl_calc_bw, ce_calc_bw = compute_ce(model_bw, loader, cfg["max_batches"])
    print(f"    BW: PPL={ppl_calc_bw:.4f}  CE={ce_calc_bw:.4f} bits")
    print(f"    delta_CE = {ce_calc_bw-ce_calc_fw:+.4f}")
    print("  [Metric 2] perplexity_ind_CE: N/A (no closed-form conditional for flower)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [r"$\leftarrow$ Forward", r"Backward $\rightarrow$"]
    colors = ["#4c72b0", "#dd8452"]
    for ax, vals, ylabel in zip(
        axes,
        [[ppl_calc_fw, ppl_calc_bw], [ce_calc_fw, ce_calc_bw]],
        [r"Perplexity  $2^{\mathcal{L}}$",
         r"Cross-entropy $\mathcal{L}$ (bits/token)"],
    ):
        delta = vals[1] - vals[0]
        bars  = ax.bar(labels, vals, color=colors, alpha=0.85,
                       edgecolor="k", width=0.45)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
        ax.set_ylabel(ylabel)
        sign = (r"$\mathcal{L}_\mathrm{BW} > \mathcal{L}_\mathrm{FW}$"
                if delta > 0 else
                r"$\mathcal{L}_\mathrm{FW} > \mathcal{L}_\mathrm{BW}$")
        ax.set_title(
            r"\texttt{perplexity\_calculation}"
            "\n"
            rf"$\Delta\mathcal{{L}} = {delta:+.4f}$  ({sign})")
        ax.grid(True, alpha=0.3, axis="y")
    tag_safe = tag.replace("_", r"\_")
    fig.suptitle(
        rf"\textbf{{{tag_safe}}} --- "
        r"Causal Asymmetry (\texttt{perplexity\_calculation} only)",
        fontsize=11)
    fig.tight_layout()
    savefig(fig, os.path.join(odir, f"{tag}_ppl_comparison.png"))

    S_emp_fw = statistical_complexity_empirical(
        model_fw, loader, max_batches=cfg["max_batches"], use_t="last",  k=n+1)
    S_emp_bw = statistical_complexity_empirical(
        model_bw, loader, max_batches=cfg["max_batches"], use_t="first", k=m+1)
    plot_complexity(tag, S_emp_fw, S_emp_bw, out_dir=odir)

    sep = "=" * 60
    print(f"\n{sep}\n  SUMMARY — {tag}\n{sep}")
    print(f"  [Metric 1] perplexity_calculation")
    print(f"    PPL  FW / BW        = {ppl_calc_fw:.4f} / {ppl_calc_bw:.4f}")
    print(f"    CE   FW / BW        = {ce_calc_fw:.4f} / {ce_calc_bw:.4f} bits")
    print(f"    delta_CE (BW-FW)    = {ce_calc_bw-ce_calc_fw:+.4f}  "
          f"({'BW harder' if ce_calc_bw > ce_calc_fw else 'FW harder'})")
    print(f"  C+ empirical          = {S_emp_fw:.4f}")
    print(f"  C- empirical          = {S_emp_bw:.4f}")
    print(sep)

    gc.collect()
    return dict(ppl_calc_fw=ppl_calc_fw, ce_calc_fw=ce_calc_fw,
                ppl_calc_bw=ppl_calc_bw, ce_calc_bw=ce_calc_bw,
                S_emp_fw=S_emp_fw, S_emp_bw=S_emp_bw)


# =============================================================================
# MAIN
# =============================================================================
def main():
    cfg          = build_cfg()
    results_dir  = RUN["results_dir"] or cfg["out_root"]
    models_dir   = os.path.join(results_dir, "models")
    out_root     = RUN["out_dir"] or os.path.join(results_dir, "asymmetry_test")
    mkdir(out_root)
    exp = RUN["exp"].lower()
    print(f"  config={RUN['config']}  models_dir={models_dir}")

    if exp in ("all", "exp1"):
        eval_coin(
            tag        = coin_tag("exp1", cfg["coin_p1"], cfg["coin_q1"]),
            p          = cfg["coin_p1"],
            q          = cfg["coin_q1"],
            seq_len    = cfg["coin_seq_len_eval"],
            models_dir = models_dir,
            out_root   = out_root,
            cfg        = cfg,
        )

    if exp in ("all", "exp1_2"):
        eval_coin(
            tag        = coin_tag("exp1_2", cfg["coin_p2"], cfg["coin_q2"]),
            p          = cfg["coin_p2"],
            q          = cfg["coin_q2"],
            seq_len    = cfg["coin_seq_len_eval"],
            models_dir = models_dir,
            out_root   = out_root,
            cfg        = cfg,
        )

    if exp in ("all", "exp2"):
        # Every flower configuration the runner trained, not one hard-coded
        # (n, m) — the runner now trains both a positive and a reversed case.
        for n, m, role in cfg["flower_configs"]:
            eval_flower(
                tag        = flower_tag("exp2", n, m),
                n          = n,
                m          = m,
                models_dir = models_dir,
                out_root   = out_root,
                cfg        = cfg,
            )

    print(f"\n  DONE — outputs in {out_root}/")


if __name__ == "__main__":
    main()