import os
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── project imports ───────────────────────────────────────────────────────────
from Data_generation import CoinDataset, coin_generation, flower_process_generation
from Flower_process_generation import FlowerDataset
from OneHot_model import OneHotDecoder
from Training_model import make_chunked_loader, _loader
from Model_analysis import (
    latent_extraction,
    perplexity_calculation,
    plot_umap,
    statistical_complexity,
    statistical_complexity_empirical,
    savefig,
    mkdir,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — mirror the values used during training
# ══════════════════════════════════════════════════════════════════════════════
# =========================================================================
# AUTOREGRESSIVE GENERATION HELPERS
# =========================================================================
def _get_probs(model, context, num_token):
    is_bw = (getattr(model, "mode", "forward") == "backward")
    x = torch.tensor([context])
    with torch.no_grad():
        logits = model(x)
    logits = logits[:, 0, :] if is_bw else logits[:, -1, :]
    return torch.softmax(logits[0], dim=-1).numpy()


def generate_sequence(model, num_token, start_token=0, gen_len=1000, burn_in=200):
    is_bw = (getattr(model, "mode", "forward") == "backward")
    context = [start_token]
    sequence, prob_hist, chosen = [], [], []
    for i in range(gen_len + burn_in):
        probs = _get_probs(model, context, num_token)
        next_token = int(np.random.choice(num_token, p=probs))
        if i >= burn_in:
            sequence.append(next_token)
            prob_hist.append(probs)
            chosen.append(next_token)
        if is_bw:
            context = [next_token] + context
        else:
            context = context + [next_token]
    return sequence, prob_hist, chosen


def perplexity_autoregressive(model, num_token, start_token=0,
                               gen_len=1000, burn_in=200):
    _, prob_hist, chosen = generate_sequence(
        model, num_token, start_token, gen_len, burn_in)
    log2_probs = np.array([
        -np.log2(prob_hist[t][tok] + 1e-12)
        for t, tok in enumerate(chosen)
    ])
    return float(2 ** np.mean(log2_probs))


def cumulative_ppl(prob_hist, chosen):
    log2_probs = np.array([
        -np.log2(prob_hist[t][tok] + 1e-12)
        for t, tok in enumerate(chosen)
    ])
    return 2 ** (np.cumsum(log2_probs) / np.arange(1, len(log2_probs) + 1))


def plot_colored_tokens(sequence, prob_hist, chosen, num_token,
                        title="", save_path=None, n_show=100):
    n = min(n_show, len(sequence))
    mat = np.array(prob_hist[:n]).T
    cmap = plt.cm.tab10
    fig, axes = plt.subplots(2, 1, figsize=(max(14, n // 5), 6),
                              gridspec_kw={"height_ratios": [3, 1]})
    ax0 = axes[0]
    im = ax0.imshow(mat, aspect="auto", origin="lower",
                    cmap="Blues", vmin=0, vmax=1)
    for t, tok in enumerate(chosen[:n]):
        ax0.scatter(t, tok, color="red", s=20, zorder=5, marker="x")
    ax0.set_yticks(range(num_token))
    ax0.set_yticklabels([f"tok {i}" for i in range(num_token)], fontsize=8)
    ax0.set_ylabel("Token")
    ax0.set_title(f"{title} -- P(token) at each step  [red x = sampled]",
                  fontweight="bold")
    plt.colorbar(im, ax=ax0, label="P(token)")
    ax1 = axes[1]
    for t, tok in enumerate(chosen[:n]):
        ax1.bar(t, 1, color=cmap(tok / max(num_token - 1, 1)),
                width=1.0, edgecolor="none")
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_yticks([])
    ax1.set_xlabel("Generation step")
    ax1.set_ylabel("Token", fontsize=8)
    for tok in range(num_token):
        ax1.bar(0, 0, color=cmap(tok / max(num_token - 1, 1)), label=f"Token {tok}")
    ax1.legend(fontsize=7, loc="upper right", ncol=num_token)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


def plot_cumulative_ppl(ppl_fw, ppl_bw, title="", save_path=None):
    steps = np.arange(len(ppl_fw))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, ppl_fw, color="steelblue",  lw=1.5, label="Forward  cumulative PPL")
    ax.plot(steps, ppl_bw, color="darkorange", lw=1.5, label="Backward cumulative PPL")
    ax.axhline(ppl_fw[-1], color="steelblue",  ls="--", lw=1, alpha=0.6,
               label=f"FW final = {ppl_fw[-1]:.4f}")
    ax.axhline(ppl_bw[-1], color="darkorange", ls="--", lw=1, alpha=0.6,
               label=f"BW final = {ppl_bw[-1]:.4f}")
    ax.set_xlabel("Generation step t")
    ax.set_ylabel("PPL_t = 2^(-mean log2P up to t)")
    ax.set_title(f"{title} -- Cumulative standard perplexity (converges to 2^H_inf)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


EVAL_CFG = dict(
    # model architecture — must match what was trained
    d_model        = 64,
    max_len        = 2000,       # full sequence length (PE table covers this)
    n_layers       = 2,
    # new data generation
    num_samples    = 500,        # sequences to generate for evaluation
    batch_size     = 32,
    # coin
    coin_p1        = 0.3,  coin_q1 = 0.4,
    coin_seq_len   = 2000,
    coin_num_token = 3,
    coin_p2        = 0.4,  coin_q2 = 0.8,
    coin_seq_len_12 = 2000,
    # flower
    flower_n       = 6,    flower_m = 4,
    flower_seq_len = 2000,
    # analysis
    umap_n_pts       = 1000,
    umap_n_neighbors = 200,
    max_batches      = 20,
    ppl_gen_len      = 1000,
    ppl_burn_in      = 200,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_model(weights_path: str, num_token: int, cfg: dict,
               mode: str) -> OneHotDecoder:
    """Instantiate OneHotDecoder and load saved weights onto CPU."""
    model = OneHotDecoder(
        token_size = num_token,
        d_model    = cfg["d_model"],
        max_len    = cfg["max_len"],
        mode       = mode,
        n_layers   = cfg["n_layers"],
    )
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"  loaded  {weights_path}  ({mode})")
    return model


def evaluate_one(tag: str, model: OneHotDecoder, loader_ana,
                 num_token: int, out_dir: str, cfg: dict,
                 p=None, q=None, mode="forward",
                 k_complexity=2, use_t="last"):
    """
    Run all analysis for a single model on loader_ana (freshly generated data).

    Returns dict with perplexity, S_emp, S_theory.
    """
    res = {"tag": tag, "mode": mode}
    mkdir(out_dir)

    # ── Perplexity ────────────────────────────────────────────────────────────
    # perplexity_calculation collects all logits then computes global CE loss
    # → true dataset-level perplexity (not mean of batch means)
    ppl = perplexity_autoregressive(
        model, num_token, start_token=0,
        gen_len=cfg.get("ppl_gen_len", 1000),
        burn_in=cfg.get("ppl_burn_in", 200),
    )
    res["perplexity"] = ppl
    print(f"  [{tag}]  perplexity = {ppl:.4f}")

    # ── Colored token prediction + store prob_hist for cumulative PPL ────────
    try:
        seq, ph, ch = generate_sequence(
            model, num_token, start_token=0,
            gen_len=cfg.get("ppl_gen_len", 1000),
            burn_in=cfg.get("ppl_burn_in", 200),
        )
        plot_colored_tokens(seq, ph, ch, num_token, title=tag, n_show=200,
            save_path=os.path.join(out_dir, f"{tag}_token_pred.png"))
        res["prob_hist"] = ph
        res["chosen"]    = ch
    except Exception as e:
        print(f"  token pred plot failed: {e}")
        res["prob_hist"] = None
        res["chosen"]    = None

    # ── UMAP ─────────────────────────────────────────────────────────────────
    try:
        latents, inp_arr, _ = latent_extraction(
            model, loader_ana, max_batches=cfg["max_batches"])
        _, coords = plot_umap(
            latents, inp_arr, num_token,
            title=tag,
            save_path=os.path.join(out_dir, f"{tag}_umap.png"),
            n_pts=cfg["umap_n_pts"],
        )
        res["umap_coords"] = coords
    except Exception as e:
        print(f"  UMAP failed: {e}")
        res["umap_coords"] = None

    # ── Statistical complexity ────────────────────────────────────────────────
    try:
        S_emp = statistical_complexity_empirical(
            model, loader_ana,
            max_batches=cfg["max_batches"],
            use_t=use_t,
            k=k_complexity,
        )
        res["S_emp"] = S_emp
        if p is not None and q is not None:
            S_th = statistical_complexity(p, q, mode=mode)
            res["S_theory"] = S_th
            print(f"  [{tag}]  S_emp={S_emp:.4f}  S_theory={S_th:.4f}  "
                  f"err={abs(S_emp - S_th):.4f}")
        else:
            print(f"  [{tag}]  S_emp={S_emp:.4f}")
    except Exception as e:
        print(f"  complexity failed: {e}")
        res["S_emp"] = float("nan")

    return res


def compare_plot(tag: str, res_fw: dict, res_bw: dict,
                 out_dir: str, p=None, q=None):
    """
    Side-by-side bar: perplexity + complexity for FW vs BW.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Perplexity bar
    ax = axes[0]
    vals = [res_fw["perplexity"], res_bw["perplexity"]]
    bars = ax.bar(["Forward", "Backward"], vals,
                  color=["#4c72b0", "#dd8452"], alpha=0.85, edgecolor="k")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Perplexity on model's generated sequence")
    ax.grid(True, alpha=0.3, axis="y")

    # Complexity bar
    ax = axes[1]
    fw_v = [res_fw.get("S_emp", 0), res_fw.get("S_theory", 0)]
    bw_v = [res_bw.get("S_emp", 0), res_bw.get("S_theory", 0)]
    x = np.arange(2)
    ax.bar(x - 0.2, fw_v, 0.35, label="Forward",  color="#4c72b0", alpha=0.85)
    ax.bar(x + 0.2, bw_v, 0.35, label="Backward", color="#dd8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Empirical", "Theoretical"])
    ax.set_ylabel("Statistical Complexity (bits)")
    ax.set_title(f"Statistical complexity on model's generated sequence")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Experimental Results: {tag}", fontsize=12)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{tag}_eval_compare.png"))

    if res_fw.get("prob_hist") is not None and res_bw.get("prob_hist") is not None:
        ppl_fw_curve = cumulative_ppl(res_fw["prob_hist"], res_fw["chosen"])
        ppl_bw_curve = cumulative_ppl(res_bw["prob_hist"], res_bw["chosen"])
        plot_cumulative_ppl(ppl_fw_curve, ppl_bw_curve, title=tag,
            save_path=os.path.join(out_dir, f"{tag}_cumulative_ppl.png"))


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT EVALUATORS
# ══════════════════════════════════════════════════════════════════════════════
def eval_coin(tag: str, p: float, q: float, seq_len: int,
              models_dir: str, out_root: str, cfg: dict):
    odir      = mkdir(os.path.join(out_root, tag))
    num_token = cfg["coin_num_token"]
    print(f"\n{'='*60}\n  EVAL — {tag}  (p={p}, q={q})\n{'='*60}")

    # Generate fresh data
    data, _ = coin_generation(
        num_samples=cfg["num_samples"], seq_len=seq_len, p=p, q=q)
    ds         = CoinDataset(data, seq_len=seq_len)
    loader_ana = _loader(ds, cfg["batch_size"])

    # Load weights
    fw_path = os.path.join(models_dir, f"{tag}_fw.pt")
    bw_path = os.path.join(models_dir, f"{tag}_bw.pt")
    if not os.path.exists(fw_path) or not os.path.exists(bw_path):
        print(f"  weights not found — skipping {tag}")
        return

    model_fw = load_model(fw_path, num_token, cfg, mode="forward")
    model_bw = load_model(bw_path, num_token, cfg, mode="backward")

    res_fw = evaluate_one(f"{tag}_fw", model_fw, loader_ana, num_token,
                          odir, cfg, p=p, q=q, mode="forward",
                          k_complexity=2, use_t="last")

    res_bw = evaluate_one(f"{tag}_bw", model_bw, loader_ana, num_token,
                          odir, cfg, p=p, q=q, mode="backward",
                          k_complexity=3, use_t="first")

    compare_plot(tag, res_fw, res_bw, odir, p=p, q=q)
    gc.collect()
    return res_fw, res_bw


def eval_flower(tag: str, n: int, m: int,
                models_dir: str, out_root: str, cfg: dict):
    odir      = mkdir(os.path.join(out_root, tag))
    num_token = n + m
    print(f"\n{'='*60}\n  EVAL — {tag}  (n={n}, m={m})\n{'='*60}")

    # Generate fresh data (fixed dice for reproducibility)
    rng        = np.random.default_rng(99)   # different seed from training (42)
    dice_probs = rng.dirichlet(np.ones(m), size=n)
    data, _ = flower_process_generation(
        num_samples=cfg["num_samples"],
        seq_len=cfg["flower_seq_len"],
        n=n, m=m, dice_probs=dice_probs)
    seq_len_f  = len(data[0])
    ds         = FlowerDataset(data, seq_len=seq_len_f)
    loader_ana = _loader(ds, cfg["batch_size"])

    fw_path = os.path.join(models_dir, f"{tag}_fw.pt")
    bw_path = os.path.join(models_dir, f"{tag}_bw.pt")
    if not os.path.exists(fw_path) or not os.path.exists(bw_path):
        print(f"  weights not found — skipping {tag}")
        return

    model_fw = load_model(fw_path, num_token, cfg, mode="forward")
    model_bw = load_model(bw_path, num_token, cfg, mode="backward")

    res_fw = evaluate_one(f"{tag}_fw", model_fw, loader_ana, num_token,
                          odir, cfg, mode="forward",
                          k_complexity=n + 1, use_t="last")   # n+1 forward causal states

    res_bw = evaluate_one(f"{tag}_bw", model_bw, loader_ana, num_token,
                          odir, cfg, mode="backward",
                          k_complexity=m + 1, use_t="first")  # m+1 backward causal states

    compare_plot(tag, res_fw, res_bw, odir)
    gc.collect()
    return res_fw, res_bw


# ══════════════════════════════════════════════════════════════════════════════
# RUN CONFIG — edit these instead of passing CLI arguments
# Works in both scripts (python evaluate_model.py) and Jupyter notebooks
# ══════════════════════════════════════════════════════════════════════════════
RUN = dict(
    results_dir  = "results",   # change to "results_large" for overnight run
    out_dir      = None,        # None → <results_dir>/eval
    exp          = "all",       # "all" | "exp1" | "exp1_2" | "exp2"
    num_samples  = None,        # None → use EVAL_CFG default (500)
)


def main():
    cfg = EVAL_CFG.copy()
    if RUN["num_samples"] is not None:
        cfg["num_samples"] = RUN["num_samples"]

    models_dir = os.path.join(RUN["results_dir"], "models")
    out_root   = RUN["out_dir"] or os.path.join(RUN["results_dir"], "eval")
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

    print(f"\n{'='*60}")
    print(f"  EVAL COMPLETE — outputs in {out_root}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()