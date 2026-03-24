import os
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Data_generation import CoinDataset, coin_generation, flower_process_generation
from Flower_process_generation import FlowerDataset
from OneHot_model import OneHotDecoder
from Training_model import _loader
from Model_analysis import (
    perplexity_calculation,
    perplexity_ind_CE,
    statistical_complexity,
    statistical_complexity_empirical,
    savefig,
    mkdir,
)


# =============================================================================
# CONFIG
# =============================================================================
CFG = dict(
    d_model        = 64,
    max_len        = 2000,
    n_layers       = 2,
    num_samples    = 500,
    batch_size     = 32,
    max_batches    = None,
    # coin
    coin_p1        = 0.3,  coin_q1 = 0.4,
    coin_seq_len   = 2000,
    coin_num_token = 3,
    coin_p2        = 0.4,  coin_q2 = 0.8,
    coin_seq_len_12 = 2000,
    # flower
    flower_n       = 6,    flower_m = 4,
    flower_seq_len = 2000,
    # ind_CE generation
    ind_ce_len    = 2000,
    ind_ce_burnin = 200,
    # KL generation
    kl_gen_len  = 2000,
    kl_burn_in  = 200,
)

RUN = dict(
    results_dir = "results",
    out_dir     = None,
    exp         = "all",    # "all" | "exp1" | "exp1_2" | "exp2"
)


# =============================================================================
# MODEL LOADER
# =============================================================================
def load_model(path, num_token, cfg, mode):
    model = OneHotDecoder(
        token_size=num_token, d_model=cfg["d_model"],
        max_len=cfg["max_len"], mode=mode, n_layers=cfg["n_layers"])
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"  loaded {path}  [{mode}]")
    return model


# =============================================================================
# ENTROPY RATE
# =============================================================================
def entropy_rate_coin(p, q):
    def _h2(a):
        b = max(1-a, 1e-12); a = max(a, 1e-12)
        return -a*np.log2(a) - b*np.log2(b)
    return (q/(p+q))*_h2(p) + (p/(p+q))*_h2(q)


# =============================================================================
# METRIC 1 — perplexity_calculation
#
# Evaluates BOTH models on the SAME ground-truth sequences from the loader.
# Respects the FW/BW batch-swap convention so each model is evaluated in its
# natural causal direction.
# CE = log2(PPL) converges to H_inf.  delta_CE = CE_BW - CE_FW is the
# empirical causal asymmetry signal.
# =============================================================================
def compute_ppl_ce(model, loader, max_batches=None):
    """Returns (PPL, CE bits) via perplexity_calculation."""
    ppl = perplexity_calculation(model, loader, max_batches=max_batches)
    ce  = float(np.log2(ppl))
    return ppl, ce


# =============================================================================
# METRIC 2 — perplexity_ind_CE
#
# Autoregressively generates a sequence driven by the TRUE HMM transitions
# (not the model), so both FW and BW models are evaluated at the same true
# stationary state distribution.
# At each step: CE_t = -sum_x P_true(x|cur_tok) * log2 P_model(x|context)
# PPL = 2^(mean CE_t) converges to H_inf for a perfect model.
# Using the true process to advance means the comparison is fair regardless
# of how accurate the model's own distribution is.
# =============================================================================
def compute_ind_ce(model, p, q, cfg):
    """Returns (PPL, CE bits) via perplexity_ind_CE."""
    ppl = perplexity_ind_CE(
        model,
        len_seq=cfg.get("ind_ce_len", 2000),
        start_token=0,
        p=p, q=q,
    )
    ce = float(np.log2(ppl))
    return ppl, ce


# =============================================================================
# KL DIVERGENCE — stationary distribution (coin only)
# =============================================================================
def coin_true_distribution(p, q, num_token=3):
    pi = np.zeros(num_token)
    pi[1] = p / (p + q)
    pi[0] = q * (1 - p) / (p + q)
    pi[2] = p * q / (p + q)
    return pi


def kl_from_true_coin(model, p, q, cfg, num_token=3):
    """D_KL(pi_true || pi_model) using model's autoregressive generation."""
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
# STEPWISE KL — per-token average KL on ground-truth sequences
# =============================================================================
def coin_true_conditional(p, q, num_token=3):
    fw = np.zeros((num_token, num_token))
    fw[0] = [1-p, p,       0  ]
    fw[1] = [0,   1-q,     q  ]
    fw[2] = [1-p, p,       0  ]
    bw = np.zeros((num_token, num_token))
    bw[0] = [1-p,      0,   p  ]
    bw[1] = [q*(1-p),  1-q, p*q]
    bw[2] = [0,        1,   0  ]
    return fw, bw


def stepwise_kl_coin(model, loader, p, q, num_token=3, max_batches=None):
    """
    For every position in every ground-truth sequence:
        KL_t = D_KL( P_true(.|cur_tok) || P_model(.|context_t) )
    Returns mean_kl, per_tok_avg (V,), per_tok_count (V,).
    """
    model.eval()
    is_bw  = (getattr(model, "mode", "forward") == "backward")
    device = next(model.parameters()).device
    fw_cond, bw_cond = coin_true_conditional(p, q, num_token)
    true_cond = bw_cond if is_bw else fw_cond

    total_kl      = 0.0
    per_tok_kl    = np.zeros(num_token)
    per_tok_count = np.zeros(num_token)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if is_bw:
                targets, inputs = batch
            else:
                inputs, targets = batch
            inputs  = inputs.to(device)
            logits  = model(inputs)
            p_model = torch.softmax(logits, dim=-1).cpu().numpy()
            inp_np  = inputs.cpu().numpy()
            B, T, _ = p_model.shape
            for b in range(B):
                for t in range(T):
                    cur_tok = inp_np[b, t]
                    p_true  = true_cond[cur_tok]
                    p_mod   = p_model[b, t]
                    kl_t    = float(np.sum(
                        p_true * np.log2(p_true / (p_mod + 1e-12) + 1e-12)))
                    total_kl              += kl_t
                    per_tok_kl[cur_tok]   += kl_t
                    per_tok_count[cur_tok] += 1

    mean_kl     = total_kl / max(per_tok_count.sum(), 1)
    per_tok_avg = np.where(per_tok_count > 0,
                           per_tok_kl / per_tok_count, 0.0)

    print(f"    mean KL = {mean_kl:.6f} bits")
    for tok in range(num_token):
        print(f"      token {tok}: avg KL={per_tok_avg[tok]:.6f}"
              f"  (n={int(per_tok_count[tok])})")
    return mean_kl, per_tok_avg, per_tok_count


# =============================================================================
# PLOTS
# =============================================================================
def plot_dual_ppl_comparison(tag, results, h_inf=None, out_dir=None):
    """
    Four-panel figure comparing both metrics side by side.
    Left pair : perplexity_calculation (ground-truth loader)
    Right pair: perplexity_ind_CE (true-process-driven generation)

    Panel layout:
      [PPL calc]  [CE calc]  [PPL ind_CE]  [CE ind_CE]
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    labels = ["Forward", "Backward"]
    colors = ["#4c72b0", "#dd8452"]

    panels = [
        # (ax, values, ylabel, title_suffix, ref_val)
        (axes[0],
         [results["ppl_calc_fw"], results["ppl_calc_bw"]],
         "Perplexity",
         "perplexity_calculation",
         2**h_inf if h_inf else None,
         "2^H_inf"),
        (axes[1],
         [results["ce_calc_fw"],  results["ce_calc_bw"]],
         "CE loss (bits/token)",
         "perplexity_calculation",
         h_inf, "H_inf"),
        (axes[2],
         [results["ppl_ind_fw"],  results["ppl_ind_bw"]],
         "Perplexity",
         "perplexity_ind_CE",
         2**h_inf if h_inf else None,
         "2^H_inf"),
        (axes[3],
         [results["ce_ind_fw"],   results["ce_ind_bw"]],
         "CE loss (bits/token)",
         "perplexity_ind_CE",
         h_inf, "H_inf"),
    ]

    for ax, vals, ylabel, method, ref_val, ref_lbl in panels:
        delta = vals[1] - vals[0]
        bars  = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="k")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        if ref_val is not None:
            ax.axhline(ref_val, color="crimson", ls="--", lw=1.5,
                       label=f"{ref_lbl}={ref_val:.4f}")
            ax.legend(fontsize=8)
        sign = "BW harder" if delta > 0 else "FW harder"
        ax.set_ylabel(ylabel)
        ax.set_title(f"{method}\ndelta={delta:+.4f} ({sign})",
                     fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{tag} — Causal Asymmetry: perplexity_calculation vs perplexity_ind_CE",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    if out_dir:
        savefig(fig, os.path.join(out_dir, f"{tag}_dual_ppl.png"))
    return fig


def plot_kl_distribution(tag, pi_true, pi_model_fw, kl_fw,
                         pi_model_bw, kl_bw, num_token, out_dir=None):
    x            = np.arange(num_token)
    width        = 0.35
    token_labels = [f"Token {i}" for i in range(num_token)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, pi_model, kl, lbl, col in zip(
        axes,
        [pi_model_fw,  pi_model_bw],
        [kl_fw,        kl_bw],
        ["Forward model", "Backward model"],
        ["#4c72b0",    "#dd8452"],
    ):
        b_true  = ax.bar(x - width/2, pi_true,  width,
                         label="True (theoretical)",
                         color="#2ecc71", alpha=0.85, edgecolor="k")
        b_model = ax.bar(x + width/2, pi_model, width,
                         label=f"{lbl} (sampled)",
                         color=col, alpha=0.85, edgecolor="k")
        ax.bar_label(b_true,  fmt="%.3f", padding=2, fontsize=8)
        ax.bar_label(b_model, fmt="%.3f", padding=2, fontsize=8)
        ax.text(0.97, 0.95, f"D_KL = {kl:.5f} bits",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="grey", alpha=0.9))
        ax.set_xticks(x); ax.set_xticklabels(token_labels, fontsize=9)
        ax.set_ylabel("Token probability")
        ax.set_title(f"{lbl}\nD_KL(true || model) = {kl:.5f} bits",
                     fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{tag} — Stationary distribution: True vs Model",
                 fontsize=11, fontweight="bold")
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
    hard       = "BW harder" if delta > 0 else "FW harder"

    ax = axes[0]
    bars = ax.bar(["Forward", "Backward"], [mean_kl_fw, mean_kl_bw],
                  color=colors, alpha=0.85, edgecolor="k")
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=10)
    ax.text(0.97, 0.95, f"delta = {delta:+.5f} bits ({hard})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.9))
    ax.set_ylabel("Mean D_KL(P_true || P_model)  (bits)")
    ax.set_title("Average stepwise KL over ground-truth sequences",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    width = 0.35
    b_fw = ax.bar(x_tok - width/2, per_tok_fw, width,
                  label="Forward",  color=colors[0], alpha=0.85, edgecolor="k")
    b_bw = ax.bar(x_tok + width/2, per_tok_bw, width,
                  label="Backward", color=colors[1], alpha=0.85, edgecolor="k")
    ax.bar_label(b_fw, fmt="%.4f", padding=2, fontsize=8)
    ax.bar_label(b_bw, fmt="%.4f", padding=2, fontsize=8)
    for i, n in enumerate(per_tok_count_fw.astype(int)):
        ax.text(i, -0.008, f"n={n}", ha="center", fontsize=7,
                color=colors[0], transform=ax.get_xaxis_transform())
    ax.set_xticks(x_tok)
    ax.set_xticklabels([f"Token {i}" for i in range(num_token)])
    ax.set_ylabel("Avg D_KL per token  (bits)")
    ax.set_title("Per-token stepwise KL (which transitions are hardest?)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{tag} -- Stepwise KL: D_KL(P_true(.|cur_tok) || P_model)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    if out_dir:
        savefig(fig, os.path.join(out_dir, f"{tag}_stepwise_kl.png"))
    return fig


def plot_complexity(tag, S_emp_fw, S_emp_bw, S_th_fw=None, S_th_bw=None,
                    out_dir=None):
    labels = ["Forward", "Backward"]
    colors = ["#4c72b0", "#dd8452"]
    x      = np.arange(2)
    fig, ax = plt.subplots(figsize=(8, 5))
    b_emp = ax.bar(x - 0.2, [S_emp_fw, S_emp_bw], 0.35,
                   label="Empirical", color=colors, alpha=0.85, edgecolor="k")
    ax.bar_label(b_emp, fmt="%.4f", padding=3, fontsize=9)
    if S_th_fw is not None and S_th_bw is not None:
        b_th = ax.bar(x + 0.2, [S_th_fw, S_th_bw], 0.35,
                      label="Theoretical", color=colors, alpha=0.45,
                      edgecolor="k", hatch="//")
        ax.bar_label(b_th, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Statistical Complexity (bits)")
    ax.set_title(f"{tag} — Causal State Complexity  (C- > C+ expected)",
                 fontweight="bold")
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

    # --- METRIC 1: perplexity_calculation on ground-truth loader --------------
    print("\n  [Metric 1] perplexity_calculation (ground-truth sequences, same loader):")
    ppl_calc_fw, ce_calc_fw = compute_ppl_ce(model_fw, loader, cfg["max_batches"])
    print(f"    FW: PPL={ppl_calc_fw:.4f}  CE={ce_calc_fw:.4f} bits")
    ppl_calc_bw, ce_calc_bw = compute_ppl_ce(model_bw, loader, cfg["max_batches"])
    print(f"    BW: PPL={ppl_calc_bw:.4f}  CE={ce_calc_bw:.4f} bits")
    print(f"    H_inf={h_inf:.4f}  delta_CE={ce_calc_bw-ce_calc_fw:+.4f}")

    # --- METRIC 2: perplexity_ind_CE with true-process-driven generation ------
    # Sequences are driven by the TRUE HMM (not the model), so both models are
    # evaluated at the same true stationary distribution — fair comparison.
    print("\n  [Metric 2] perplexity_ind_CE (true-process generation, per-step CE vs true conditional):")
    ppl_ind_fw, ce_ind_fw = compute_ind_ce(model_fw, p, q, cfg)
    print(f"    FW: PPL={ppl_ind_fw:.4f}  CE={ce_ind_fw:.4f} bits")
    ppl_ind_bw, ce_ind_bw = compute_ind_ce(model_bw, p, q, cfg)
    print(f"    BW: PPL={ppl_ind_bw:.4f}  CE={ce_ind_bw:.4f} bits")
    print(f"    H_inf={h_inf:.4f}  delta_CE={ce_ind_bw-ce_ind_fw:+.4f}")

    results = dict(
        ppl_calc_fw=ppl_calc_fw, ce_calc_fw=ce_calc_fw,
        ppl_calc_bw=ppl_calc_bw, ce_calc_bw=ce_calc_bw,
        ppl_ind_fw=ppl_ind_fw,   ce_ind_fw=ce_ind_fw,
        ppl_ind_bw=ppl_ind_bw,   ce_ind_bw=ce_ind_bw,
    )
    plot_dual_ppl_comparison(tag, results, h_inf=h_inf, out_dir=odir)

    # --- KL divergence — stationary distribution ------------------------------
    print("\n  KL divergence from true stationary distribution:")
    print("    Forward ...")
    kl_fw, pi_true, pi_model_fw = kl_from_true_coin(model_fw, p, q, cfg, num_token)
    print("    Backward ...")
    kl_bw, _,       pi_model_bw = kl_from_true_coin(model_bw, p, q, cfg, num_token)
    plot_kl_distribution(tag, pi_true, pi_model_fw, kl_fw,
                         pi_model_bw, kl_bw, num_token, out_dir=odir)

    # --- Stepwise KL per token ------------------------------------------------
    print("\n  Stepwise KL divergence per token (ground-truth sequences):")
    print("    Forward ...")
    skl_fw, skl_per_tok_fw, skl_count_fw = stepwise_kl_coin(
        model_fw, loader, p, q, num_token, cfg["max_batches"])
    print("    Backward ...")
    skl_bw, skl_per_tok_bw, _            = stepwise_kl_coin(
        model_bw, loader, p, q, num_token, cfg["max_batches"])
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
    print(f"  [perplexity_calculation]  (ground-truth loader, same sequences)")
    print(f"    PPL  FW / BW            = {ppl_calc_fw:.4f} / {ppl_calc_bw:.4f}")
    print(f"    CE   FW / BW            = {ce_calc_fw:.4f} / {ce_calc_bw:.4f} bits")
    print(f"    delta_CE                = {ce_calc_bw-ce_calc_fw:+.4f}  "
          f"({'BW harder' if ce_calc_bw > ce_calc_fw else 'FW harder'})")
    print()
    print(f"  [perplexity_ind_CE]  (true-process generation, CE vs true conditional)")
    print(f"    PPL  FW / BW            = {ppl_ind_fw:.4f} / {ppl_ind_bw:.4f}")
    print(f"    CE   FW / BW            = {ce_ind_fw:.4f} / {ce_ind_bw:.4f} bits")
    print(f"    delta_CE                = {ce_ind_bw-ce_ind_fw:+.4f}  "
          f"({'BW harder' if ce_ind_bw > ce_ind_fw else 'FW harder'})")
    print()
    print(f"  D_KL stationary FW / BW  = {kl_fw:.6f} / {kl_bw:.6f} bits")
    print(f"  Stepwise KL    FW / BW   = {skl_fw:.6f} / {skl_bw:.6f} bits")
    print(f"  C+ emp / th              = {S_emp_fw:.4f} / {S_th_fw:.4f}")
    print(f"  C- emp / th              = {S_emp_bw:.4f} / {S_th_bw:.4f}")
    print(sep)

    gc.collect()
    return dict(
        ppl_calc_fw=ppl_calc_fw, ce_calc_fw=ce_calc_fw,
        ppl_calc_bw=ppl_calc_bw, ce_calc_bw=ce_calc_bw,
        ppl_ind_fw=ppl_ind_fw,   ce_ind_fw=ce_ind_fw,
        ppl_ind_bw=ppl_ind_bw,   ce_ind_bw=ce_ind_bw,
        kl_fw=kl_fw, kl_bw=kl_bw,
        skl_fw=skl_fw, skl_bw=skl_bw,
        S_emp_fw=S_emp_fw, S_emp_bw=S_emp_bw,
        S_th_fw=S_th_fw, S_th_bw=S_th_bw,
    )


def eval_flower(tag, n, m, models_dir, out_root, cfg):
    """
    Flower process: no closed-form true conditional, so perplexity_ind_CE
    is not applicable. Only perplexity_calculation is used.
    """
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
    ppl_calc_fw, ce_calc_fw = compute_ppl_ce(model_fw, loader, cfg["max_batches"])
    print(f"    FW: PPL={ppl_calc_fw:.4f}  CE={ce_calc_fw:.4f} bits")
    ppl_calc_bw, ce_calc_bw = compute_ppl_ce(model_bw, loader, cfg["max_batches"])
    print(f"    BW: PPL={ppl_calc_bw:.4f}  CE={ce_calc_bw:.4f} bits")
    print(f"    delta_CE = {ce_calc_bw-ce_calc_fw:+.4f}")
    print("  [Metric 2] perplexity_ind_CE: not applicable (no closed-form true conditional for flower)")

    results = dict(
        ppl_calc_fw=ppl_calc_fw, ce_calc_fw=ce_calc_fw,
        ppl_calc_bw=ppl_calc_bw, ce_calc_bw=ce_calc_bw,
        ppl_ind_fw=float("nan"),  ce_ind_fw=float("nan"),
        ppl_ind_bw=float("nan"),  ce_ind_bw=float("nan"),
    )
    # For flower, only show the calc metric panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Forward", "Backward"]; colors = ["#4c72b0", "#dd8452"]
    for ax, vals, ylabel in zip(
        axes,
        [[ppl_calc_fw, ppl_calc_bw], [ce_calc_fw, ce_calc_bw]],
        ["Perplexity", "CE loss (bits/token)"],
    ):
        delta = vals[1] - vals[0]
        bars  = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="k")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(f"perplexity_calculation\ndelta={delta:+.4f} "
                     f"({'BW harder' if delta > 0 else 'FW harder'})",
                     fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"{tag} — Causal Asymmetry (perplexity_calculation only)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(odir, f"{tag}_ppl_comparison.png"))

    S_emp_fw = statistical_complexity_empirical(
        model_fw, loader, max_batches=cfg["max_batches"], use_t="last",  k=n+1)
    S_emp_bw = statistical_complexity_empirical(
        model_bw, loader, max_batches=cfg["max_batches"], use_t="first", k=m+1)
    plot_complexity(tag, S_emp_fw, S_emp_bw, out_dir=odir)

    sep = "=" * 60
    print(f"\n{sep}\n  SUMMARY — {tag}\n{sep}")
    print(f"  [perplexity_calculation]")
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
    cfg        = CFG.copy()
    models_dir = os.path.join(RUN["results_dir"], "models")
    out_root   = RUN["out_dir"] or os.path.join(RUN["results_dir"], "asymmetry_test")
    mkdir(out_root)
    exp = RUN["exp"].lower()

    if exp in ("all", "exp1"):
        eval_coin(
            tag        = "exp1_coin_p03_q04",
            p          = cfg["coin_p1"],
            q          = cfg["coin_q1"],
            seq_len    = cfg["coin_seq_len"],
            models_dir = models_dir,
            out_root   = out_root,
            cfg        = cfg,
        )

    if exp in ("all", "exp1_2"):
        eval_coin(
            tag        = "exp1_2_coin_p04_q08",
            p          = cfg["coin_p2"],
            q          = cfg["coin_q2"],
            seq_len    = cfg["coin_seq_len_12"],
            models_dir = models_dir,
            out_root   = out_root,
            cfg        = cfg,
        )

    if exp in ("all", "exp2"):
        eval_flower(
            tag        = "exp2_flower_n6_m4",
            n          = cfg["flower_n"],
            m          = cfg["flower_m"],
            models_dir = models_dir,
            out_root   = out_root,
            cfg        = cfg,
        )

    print(f"\n  DONE — outputs in {out_root}/")


if __name__ == "__main__":
    main()