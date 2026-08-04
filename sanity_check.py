"""
sanity_check.py
===============
Sanity check: verify that the coin HMM p=q=0.5 and the flower process n=1, m=2
produce structurally consistent causal asymmetry results.

Why these two processes?
------------------------
Coin p=q=0.5
  - Fully symmetric transition probabilities, so H∞ = 1 bit exactly.
  - C+ = 1 bit (2 states, equal probability).
  - C- = 1.5 bits (3 backward causal states: the process is still causally
    asymmetric even at p=q because the backward causal states are not equally
    probable). delta_CE > 0 is still expected but smaller than for p≠q.

Flower n=1, m=2 with dice_probs = [0.5, 0.5]
  - 1 die (token 0 is always selected, deterministic), 2 equally likely outcomes
    (tokens 1 and 2).
  - Sequence structure: 0, X, 0, X, 0, X ... where X ∈ {1,2} with P=0.5 each.
  - Forward causal states = n+1 = 2.
  - Backward causal states = m+1 = 3 (token 1 and 2 both deterministically
    precede token 0, while token 0 has uncertain history).
  - This produces the same causal state count asymmetry (2 vs 3) as the coin.

Sanity checks
  1. Both models (FW and BW) converge to H∞ — their CE loss approaches the
     entropy rate of the respective process from above.
  2. delta_CE = CE_BW - CE_FW > 0 in both cases (BW harder).
  3. Empirical C+ ≈ theoretical C+ for the coin (2 clusters).
  4. The flower FW model needs 2 latent clusters and BW needs 3 — matching
     forward and backward causal state counts n+1=2 and m+1=3.
  5. Both processes produce consistent UMAP structure: forward model latents
     separate into 2 clusters regardless of input token label.
"""

import gc
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as tud

from Data_generation import CoinDataset, coin_generation, flower_process_generation
from Flower_process_generation import FlowerDataset
from Model_analysis import (
    FW_BW_attention_comparison,
    latent_extraction,
    plot_attention_heatmap,
    perplexity_calculation,
    statistical_complexity,
    statistical_complexity_empirical,
    savefig,
    slim_results,
)
from Training_model import (
    _loader,
    make_chunked_loader,
    train_test_val_pipeline,
)

try:
    import umap as _umap_mod
    _umap_mod.UMAP(n_components=2, n_neighbors=15).fit_transform(
        np.random.rand(20, 4))
    UMAP_AVAILABLE = True
    print("umap-learn warm-up OK")
except Exception as _e:
    UMAP_AVAILABLE = False
    print(f"UMAP unavailable ({_e}) — PCA fallback")


# =============================================================================
# CONFIG
# =============================================================================
CFG = dict(
    d_model        = 64,
    embed_type     = "onehot",
    n_folds        = 5,
    lr             = 5e-3,
    train_chunk_len = 512,
    attn_vis_len   = 64,
    umap_n_neighbors = 15,   # small: datasets are smaller than main experiments
    umap_n_pts     = 500,
    max_batches    = 20,

    # Coin p=q=0.5
    coin_p         = 0.5,
    coin_q         = 0.5,
    coin_num_samples = 500,
    coin_seq_len   = 1000,
    coin_max_epochs = 60,
    coin_batch     = 32,
    coin_num_token = 3,

    # Flower n=1, m=2  (equal dice probs → max symmetry within this structure)
    flower_n       = 1,
    flower_m       = 2,
    flower_num_samples = 500,
    flower_seq_len = 500,   # 500 cycles × 2 obs = 1000 tokens per sequence
    flower_max_epochs = 60,
    flower_batch   = 32,
)

OUT_ROOT = "sanity_check_flower_process"


# =============================================================================
# HELPERS  (self-contained so this file runs independently)
# =============================================================================
def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    mb = os.path.getsize(path) / 1024**2
    print(f"  pickle -> {path}  ({mb:.1f} MB)")


def save_weights(model, path):
    torch.save(model.state_dict(), path)
    print(f"  weights -> {path}")


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def to_cpu(model):
    model.cpu(); gc.collect()
    return model


def entropy_rate_coin(p, q):
    def _h2(a):
        b = max(1-a, 1e-12); a = max(a, 1e-12)
        return -a*np.log2(a) - b*np.log2(b)
    return (q/(p+q))*_h2(p) + (p/(p+q))*_h2(q)


def _project2d(flat, n_neighbors=15):
    if UMAP_AVAILABLE:
        try:
            c = _umap_mod.UMAP(
                n_components=2, random_state=42,
                n_neighbors=min(n_neighbors, len(flat)-1),
                min_dist=0.1, metric="euclidean",
            ).fit_transform(flat)
            return c, "UMAP"
        except Exception as e:
            print(f"  UMAP failed ({e}), using PCA")
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(flat), "PCA"


def plot_umap_tokens(flat_lat, flat_inp, num_token, title="",
                     save_path=None, n_pts=500, n_neighbors=15):
    reduced  = flat_lat[:n_pts]
    tok_ids  = flat_inp[:n_pts]
    coords, mlbl = _project2d(reduced, n_neighbors=n_neighbors)
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(7, 6))
    for tok in range(num_token):
        mask = tok_ids == tok
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(tok / max(num_token-1, 1))],
                   label=f"Token {tok}", alpha=0.7, s=10)
    ax.set_title(f"{title} ({mlbl})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.2)
    if save_path:
        savefig(fig, save_path)
    return fig, coords


def plot_loss_theory(rec_fw, rec_bw, h_inf, title="", save_path=None):
    """Training loss curves for FW and BW with H∞ reference."""
    sl_fw = np.array(rec_fw.step_loss)
    sl_bw = np.array(rec_bw.step_loss)
    n = min(len(sl_fw), len(sl_bw))
    sl_fw, sl_bw = sl_fw[:n], sl_bw[:n]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(sl_fw, color="#4c72b0", lw=1,   label="Forward")
    ax.plot(sl_bw, color="#dd8452", lw=1,   label="Backward")
    ax.axhline(h_inf, color="crimson", ls="--", lw=1.5,
               label=f"H∞ = {h_inf:.4f}")
    ax.set_xlabel("Gradient step"); ax.set_ylabel("CE loss (bits)")
    ax.set_title("Training loss FW vs BW", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff = sl_bw - sl_fw
    ax.plot(diff, color="purple", lw=1)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Gradient step"); ax.set_ylabel("CE_BW − CE_FW (bits)")
    ax.set_title("Loss difference (positive = BW harder)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    vl_fw = np.array(rec_fw.step_val_loss)
    vl_bw = np.array(rec_bw.step_val_loss)
    nv = min(len(vl_fw), len(vl_bw))
    ax.plot(vl_fw[:nv], color="#4c72b0", lw=1,   label="Forward val")
    ax.plot(vl_bw[:nv], color="#dd8452", lw=1,   label="Backward val")
    ax.axhline(h_inf, color="crimson", ls="--", lw=1.5,
               label=f"H∞ = {h_inf:.4f}")
    ax.set_xlabel("Step"); ax.set_ylabel("Val CE loss (bits)")
    ax.set_title("Validation loss", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig


def analyse_model(tag, model, loader_ana, num_token, out_dir,
                  sample_seq=None, p=None, q=None, mode="forward",
                  k=2, use_t="last", cfg=None):
    """Attention + UMAP + complexity for a single trained model."""
    cfg = cfg or {}
    res = {"tag": tag}
    to_cpu(model)
    if sample_seq is not None:
        sample_seq = sample_seq.cpu()

    if sample_seq is not None:
        try:
            fig_a = plot_attention_heatmap(model, sample_seq[:cfg.get("attn_vis_len", 64)])
            savefig(fig_a, os.path.join(out_dir, f"{tag}_attn.png"))
        except Exception as e:
            print(f"  attn: {e}")

    try:
        latents, inp_arr, _ = latent_extraction(
            model, loader_ana, max_batches=cfg.get("max_batches", 20))
        flat_lat = latents.reshape(-1, latents.shape[-1])
        flat_inp = inp_arr.reshape(-1)
        fig_u, coords = plot_umap_tokens(
            flat_lat, flat_inp, num_token,
            title=tag,
            save_path=os.path.join(out_dir, f"{tag}_umap.png"),
            n_pts=cfg.get("umap_n_pts", 500),
            n_neighbors=cfg.get("umap_n_neighbors", 15),
        )
        res.update({"latents": latents, "flat_lat": flat_lat,
                    "flat_inp": flat_inp, "umap_coords": coords})
    except Exception as e:
        print(f"  UMAP: {e}")
        res.update({"latents": None, "flat_lat": None,
                    "flat_inp": None, "umap_coords": None})

    try:
        S_emp = statistical_complexity_empirical(
            model, loader_ana,
            max_batches=cfg.get("max_batches", 20),
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
        print(f"  complexity: {e}")
        res["S_emp"] = float("nan")

    return res


def compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw,
                  loader_ana, num_token, out_dir, h_inf,
                  sample_seq=None, p=None, q=None, cfg=None):
    """UMAP comparison + complexity bar + PPL summary for FW vs BW."""
    cfg  = cfg or {}
    mfw  = cv_fw["best_model"]
    mbw  = cv_bw["best_model"]
    rfw  = cv_fw["best_recorder"]
    rbw  = cv_bw["best_recorder"]
    to_cpu(mfw); to_cpu(mbw)
    if sample_seq is not None:
        sample_seq = sample_seq.cpu()

    # ── (a) Loss curves ───────────────────────────────────────────────────
    plot_loss_theory(rfw, rbw, h_inf, title=tag,
                     save_path=os.path.join(out_dir, f"{tag}_loss_theory.png"))

    # ── (b) Attention comparison ──────────────────────────────────────────
    if sample_seq is not None:
        try:
            fig_a = FW_BW_attention_comparison(
                mfw, mbw, sample_seq[:cfg.get("attn_vis_len", 64)],
                title_prefix=f"{tag} ")
            savefig(fig_a, os.path.join(out_dir, f"{tag}_attn_compare.png"))
        except Exception as e:
            print(f"  attn compare: {e}")

    # ── (c) UMAP side-by-side ─────────────────────────────────────────────
    try:
        lfw, ifw, _ = latent_extraction(mfw, loader_ana,
                                         max_batches=cfg.get("max_batches", 20))
        lbw, ibw, _ = latent_extraction(mbw, loader_ana,
                                         max_batches=cfg.get("max_batches", 20))
        n_pts = cfg.get("umap_n_pts", 500)
        n_nbr = cfg.get("umap_n_neighbors", 15)
        fl_fw = lfw.reshape(-1, lfw.shape[-1])[:n_pts]
        fl_bw = lbw.reshape(-1, lbw.shape[-1])[:n_pts]
        si_fw = ifw.reshape(-1)[:n_pts]
        si_bw = ibw.reshape(-1)[:n_pts]
        c_fw, mlbl = _project2d(fl_fw, n_neighbors=n_nbr)
        c_bw, _    = _project2d(fl_bw, n_neighbors=n_nbr)

        cmap = plt.cm.tab10
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, coords, raw_inp, lbl in zip(
            axes, [c_fw, c_bw], [si_fw, si_bw],
            [f"Forward model ({mlbl})", f"Backward model ({mlbl})"]
        ):
            for tok in range(num_token):
                mask = raw_inp == tok
                if not mask.any():
                    continue
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=[cmap(tok / max(num_token-1, 1))],
                           label=f"Token {tok}", alpha=0.7, s=10)
            ax.set_title(lbl, fontsize=12, fontweight="bold")
            ax.legend(fontsize=8, markerscale=3); ax.grid(True, alpha=0.2)
        fig.suptitle(f"{tag} — UMAP latent comparison", fontsize=11,
                     fontweight="bold")
        fig.tight_layout()
        savefig(fig, os.path.join(out_dir, f"{tag}_umap_compare.png"))
    except Exception as e:
        print(f"  UMAP compare: {e}")

    # ── (d) Complexity bar ────────────────────────────────────────────────
    fw_v = [ana_fw.get("S_emp", 0), ana_fw.get("S_theory", 0)]
    bw_v = [ana_bw.get("S_emp", 0), ana_bw.get("S_theory", 0)]
    x = np.arange(2)
    fig_c, ax_c = plt.subplots(figsize=(8, 5))
    ax_c.bar(x - 0.2, fw_v, 0.35, label="Forward",  color="#4c72b0", alpha=0.85, edgecolor="k")
    ax_c.bar(x + 0.2, bw_v, 0.35, label="Backward", color="#dd8452", alpha=0.85, edgecolor="k")
    ax_c.set_xticks(x); ax_c.set_xticklabels(["Empirical", "Theoretical"])
    ax_c.set_ylabel("Statistical Complexity (bits)")
    ax_c.set_title(f"{tag} — Causal State Complexity", fontweight="bold")
    ax_c.legend(); ax_c.grid(True, alpha=0.3, axis="y")
    fig_c.tight_layout()
    savefig(fig_c, os.path.join(out_dir, f"{tag}_complexity_compare.png"))

    # ── (e) Perplexity asymmetry bar ─────────────────────────────────────
    ppl_fw = perplexity_calculation(mfw, loader_ana,
                                    max_batches=cfg.get("max_batches", 20))
    ppl_bw = perplexity_calculation(mbw, loader_ana,
                                    max_batches=cfg.get("max_batches", 20))
    ce_fw  = float(np.log2(ppl_fw))
    ce_bw  = float(np.log2(ppl_bw))
    delta  = ce_bw - ce_fw

    fig_p, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Forward", "Backward"]
    colors = ["#4c72b0", "#dd8452"]

    ax = axes[0]
    bars = ax.bar(labels, [ppl_fw, ppl_bw], color=colors, alpha=0.85, edgecolor="k")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.axhline(2**h_inf, color="crimson", ls="--", lw=1.5,
               label=f"2^H∞ = {2**h_inf:.4f}")
    ax.set_ylabel("Perplexity"); ax.legend(fontsize=9)
    ax.set_title("Perplexity on ground-truth sequences", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar(labels, [ce_fw, ce_bw], color=colors, alpha=0.85, edgecolor="k")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.axhline(h_inf, color="crimson", ls="--", lw=1.5,
               label=f"H∞ = {h_inf:.4f}")
    ax.set_ylabel("CE loss (bits/token)")
    sign = "BW harder" if delta > 0 else "FW harder"
    ax.set_title(f"Cross-entropy  delta = {delta:+.4f} ({sign})",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    fig_p.suptitle(f"{tag} — Causal Asymmetry", fontsize=11, fontweight="bold")
    fig_p.tight_layout()
    savefig(fig_p, os.path.join(out_dir, f"{tag}_ppl_asymmetry.png"))

    print(f"\n  [{tag}] PPL  FW={ppl_fw:.4f}  BW={ppl_bw:.4f}")
    print(f"  [{tag}] CE   FW={ce_fw:.4f}  BW={ce_bw:.4f}  delta={delta:+.4f}")
    print(f"  [{tag}] H∞  = {h_inf:.4f}  ({sign})")

    return dict(ppl_fw=ppl_fw, ppl_bw=ppl_bw,
                ce_fw=ce_fw, ce_bw=ce_bw, delta=delta)


# =============================================================================
# EXPERIMENT A — Coin HMM  p=q=0.5
# =============================================================================
def exp_coin(cfg, out_root):
    """
    Symmetric coin (p=q=0.5).
    H∞ = 1.0 bit exactly.
    C+ = 1.0 bit (2 forward causal states, both equally probable).
    C- = 1.5 bits (3 backward causal states with unequal probabilities).
    Expected: both models converge to H∞=1.0; delta_CE > 0 (BW harder).
    """
    p, q  = cfg["coin_p"], cfg["coin_q"]
    tag   = f"coin_p{int(p*10):02d}_q{int(q*10):02d}"
    odir  = mkdir(os.path.join(out_root, tag))
    t0    = time.time()
    h_inf = entropy_rate_coin(p, q)

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT A — Coin HMM  p={p}  q={q}")
    print(f"  H∞ = {h_inf:.4f} bits")
    print(f"  C+ = {statistical_complexity(p, q, 'forward'):.4f} bits  "
          f"(2 forward causal states)")
    print(f"  C- = {statistical_complexity(p, q, 'backward'):.4f} bits  "
          f"(3 backward causal states)")
    print(f"{'='*65}")

    # Data
    data, _ = coin_generation(
        num_samples=cfg["coin_num_samples"],
        seq_len=cfg["coin_seq_len"], p=p, q=q)
    ds           = CoinDataset(data, seq_len=cfg["coin_seq_len"])
    loader_train = make_chunked_loader(ds, cfg["train_chunk_len"],
                                       cfg["coin_batch"])
    loader_ana   = _loader(ds, cfg["coin_batch"])
    sample_seq   = next(iter(loader_ana))[0][0]
    num_token    = cfg["coin_num_token"]
    max_len      = cfg["coin_seq_len"]

    # Forward model
    print("\n  -- Forward model --")
    cv_fw = train_test_val_pipeline(
        loader_train, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"),
    ); cleanup()

    # Backward model
    print("\n  -- Backward model (same forward data) --")
    cv_bw = train_test_val_pipeline(
        loader_train, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="backward",
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"),
    ); cleanup()

    # Analyse
    print("\n  -- Analysis --")
    ana_fw = analyse_model(
        f"{tag}_fw", cv_fw["best_model"], loader_ana, num_token, odir,
        sample_seq, p=p, q=q, mode="forward",
        k=2, use_t="last", cfg=cfg); cleanup()
    ana_bw = analyse_model(
        f"{tag}_bw", cv_bw["best_model"], loader_ana, num_token, odir,
        sample_seq, p=p, q=q, mode="backward",
        k=3, use_t="first", cfg=cfg); cleanup()

    # Compare
    print("\n  -- Comparison --")
    asym = compare_fw_bw(
        tag, cv_fw, cv_bw, ana_fw, ana_bw,
        loader_ana, num_token, odir, h_inf,
        sample_seq=sample_seq, p=p, q=q, cfg=cfg)

    # Save
    save_weights(cv_fw["best_model"],
                 os.path.join(out_root, "models", f"{tag}_fw.pt"))
    save_weights(cv_bw["best_model"],
                 os.path.join(out_root, "models", f"{tag}_bw.pt"))
    res = dict(tag=tag, p=p, q=q, h_inf=h_inf,
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw,
               asymmetry=asym)
    save_pkl(slim_results(res), os.path.join(odir, "results.pkl"))
    print(f"\n  Exp A done in {(time.time()-t0)/60:.1f} min")
    return res


# =============================================================================
# EXPERIMENT B — Flower n=1, m=2  (equal dice probabilities)
# =============================================================================
def exp_flower(cfg, out_root):
    """
    Flower n=1, m=2 with dice_probs = [[0.5, 0.5]].
    Sequence: 0, X, 0, X, ... where X ∈ {token 1, token 2} with P=0.5 each.
    Token 0 is always selected (n=1, only one die).

    Forward causal states = n+1 = 2  (same count as coin p=q=0.5)
    Backward causal states = m+1 = 3 (same count as coin p=q=0.5)

    Sanity check: both forward and backward models should converge to H∞
    of the flower process. delta_CE > 0 expected (BW harder, C- > C+).
    """
    n, m    = cfg["flower_n"], cfg["flower_m"]
    tag     = f"flower_n{n}_m{m}_eq"
    odir    = mkdir(os.path.join(out_root, tag))
    t0      = time.time()
    num_token = n + m

    # Equal probability dice → max symmetry within this structure
    dice_probs = np.array([[0.5, 0.5]], dtype=float)   # shape (1, 2)

    # H∞ of this flower process (empirical estimate)
    # Token 0 always follows tokens 1,2 (entropy 0)
    # After token 0: P(token 1)=0.5, P(token 2)=0.5 (entropy 1 bit)
    # π(token 0) = 0.5, π(token 1) = π(token 2) = 0.25
    # H∞ = π(0)*H(0→?) + π(1)*H(1→?) + π(2)*H(2→?)
    #     = 0.5 * 1.0   + 0.25 * 0.0  + 0.25 * 0.0  = 0.5 bits
    h_inf_flower = 0.5

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT B — Flower n={n}  m={m}  (equal dice)")
    print(f"  Vocabulary: token 0 = die selection, tokens 1-{n+m-1} = outcomes")
    print(f"  H∞ (analytic) = {h_inf_flower:.4f} bits")
    print(f"  Forward causal states  = n+1 = {n+1}")
    print(f"  Backward causal states = m+1 = {m+1}")
    print(f"{'='*65}")

    # Data
    data, _ = flower_process_generation(
        num_samples=cfg["flower_num_samples"],
        seq_len=cfg["flower_seq_len"],
        n=n, m=m, dice_probs=dice_probs)
    seq_len_actual = len(data[0])
    ds           = FlowerDataset(data, seq_len=seq_len_actual)
    loader_train = make_chunked_loader(ds, cfg["train_chunk_len"],
                                       cfg["flower_batch"])
    loader_ana   = _loader(ds, cfg["flower_batch"])
    sample_seq   = next(iter(loader_ana))[0][0]
    max_len      = seq_len_actual

    # Forward model
    print("\n  -- Forward model --")
    cv_fw = train_test_val_pipeline(
        loader_train, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"),
    ); cleanup()

    # Backward model
    print("\n  -- Backward model (same forward data) --")
    cv_bw = train_test_val_pipeline(
        loader_train, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="backward",
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"),
    ); cleanup()

    # Analyse
    print("\n  -- Analysis --")
    # n+1=2 forward causal states, m+1=3 backward
    ana_fw = analyse_model(
        f"{tag}_fw", cv_fw["best_model"], loader_ana, num_token, odir,
        sample_seq, mode="forward",
        k=n+1, use_t="last", cfg=cfg); cleanup()
    ana_bw = analyse_model(
        f"{tag}_bw", cv_bw["best_model"], loader_ana, num_token, odir,
        sample_seq, mode="backward",
        k=m+1, use_t="first", cfg=cfg); cleanup()

    # Compare
    print("\n  -- Comparison --")
    asym = compare_fw_bw(
        tag, cv_fw, cv_bw, ana_fw, ana_bw,
        loader_ana, num_token, odir, h_inf_flower,
        sample_seq=sample_seq, cfg=cfg)

    # Save
    save_weights(cv_fw["best_model"],
                 os.path.join(out_root, "models", f"{tag}_fw.pt"))
    save_weights(cv_bw["best_model"],
                 os.path.join(out_root, "models", f"{tag}_bw.pt"))
    res = dict(tag=tag, n=n, m=m, h_inf=h_inf_flower,
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw,
               asymmetry=asym, dice_probs=dice_probs)
    save_pkl(slim_results(res), os.path.join(odir, "results.pkl"))
    print(f"\n  Exp B done in {(time.time()-t0)/60:.1f} min")
    return res


# =============================================================================
# CROSS-EXPERIMENT COMPARISON PLOT
# =============================================================================
def plot_cross_comparison(res_coin, res_flower, out_root):
    """
    Side-by-side summary comparing both experiments.
    Sanity checks visualised:
      - Both models converge toward their respective H∞
      - delta_CE > 0 in both cases
      - Forward complexity matches theory (2 states)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels_c = ["Coin FW", "Coin BW"]
    labels_f = ["Flower FW", "Flower BW"]
    colors   = ["#4c72b0", "#dd8452"]

    ac = res_coin["asymmetry"]
    af = res_flower["asymmetry"]

    # Panel 1: CE loss for both experiments
    ax = axes[0]
    x  = np.arange(2)
    w  = 0.3
    coin_ces   = [ac["ce_fw"],   ac["ce_bw"]]
    flower_ces = [af["ce_fw"],   af["ce_bw"]]
    b1 = ax.bar(x - w/2, coin_ces,   w, label="Coin p=q=0.5",
                color=colors, alpha=0.85, edgecolor="k")
    b2 = ax.bar(x + w/2, flower_ces, w, label="Flower n=1 m=2",
                color=colors, alpha=0.45, edgecolor="k", hatch="//")
    ax.bar_label(b1, fmt="%.4f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.4f", padding=2, fontsize=8)
    ax.axhline(res_coin["h_inf"],   color="#4c72b0", ls="--", lw=1.2,
               label=f"H∞ coin = {res_coin['h_inf']:.4f}")
    ax.axhline(res_flower["h_inf"], color="#dd8452", ls="--", lw=1.2,
               label=f"H∞ flower = {res_flower['h_inf']:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(["Forward", "Backward"])
    ax.set_ylabel("CE loss (bits/token)")
    ax.set_title("CE loss: coin vs flower", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: delta CE comparison
    ax = axes[1]
    deltas = [ac["delta"], af["delta"]]
    names  = ["Coin\np=q=0.5", "Flower\nn=1, m=2"]
    bar_c  = ["#4c72b0" if d > 0 else "#c44e52" for d in deltas]
    bars   = ax.bar(names, deltas, color=bar_c, alpha=0.85, edgecolor="k")
    ax.bar_label(bars, fmt="%+.4f", padding=3, fontsize=10)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("delta CE = CE_BW - CE_FW (bits)")
    ax.set_title("Causal asymmetry signal\n(positive = BW harder, theory predicts > 0)",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: empirical complexity
    ax = axes[2]
    S_coin_fw  = res_coin["ana_fw"].get("S_emp", 0)
    S_coin_bw  = res_coin["ana_bw"].get("S_emp", 0)
    S_fl_fw    = res_flower["ana_fw"].get("S_emp", 0)
    S_fl_bw    = res_flower["ana_bw"].get("S_emp", 0)
    x = np.arange(2)
    ax.bar(x - w/2, [S_coin_fw,  S_coin_bw],  w,
           label="Coin p=q=0.5", color=colors, alpha=0.85, edgecolor="k")
    ax.bar(x + w/2, [S_fl_fw,    S_fl_bw],    w,
           label="Flower n=1 m=2", color=colors, alpha=0.45,
           edgecolor="k", hatch="//")
    if res_coin["ana_fw"].get("S_theory") is not None:
        ax.axhline(res_coin["ana_fw"]["S_theory"], color="#4c72b0",
                   ls=":", lw=1.2,
                   label=f"C+ theory = {res_coin['ana_fw']['S_theory']:.4f}")
    if res_coin["ana_bw"].get("S_theory") is not None:
        ax.axhline(res_coin["ana_bw"]["S_theory"], color="#dd8452",
                   ls=":", lw=1.2,
                   label=f"C- theory = {res_coin['ana_bw']['S_theory']:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(["Forward (C+)", "Backward (C-)"])
    ax.set_ylabel("Empirical statistical complexity (bits)")
    ax.set_title("Causal state complexity\n(both processes: C- > C+ expected)",
                 fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Sanity Check: Coin p=q=0.5  vs  Flower n=1, m=2\n"
                 "Both processes have 2 forward and 3 backward causal states",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(out_root, "cross_comparison.png"))
    print(f"  cross-comparison saved -> {out_root}/cross_comparison.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t_total = time.time()
    mkdir(OUT_ROOT)
    mkdir(os.path.join(OUT_ROOT, "models"))

    print("\n" + "="*65)
    print("  SANITY CHECK — Coin p=q=0.5  vs  Flower n=1, m=2")
    print("  Both processes have C+(FW) = 2 states, C-(BW) = 3 states.")
    print("  Both should show delta_CE > 0 and converge to their H∞.")
    print("="*65)

    # Run Experiment A: coin p=q=0.5
    res_coin   = exp_coin(CFG, OUT_ROOT)

    # Run Experiment B: flower n=1, m=2
    res_flower = exp_flower(CFG, OUT_ROOT)

    # Cross-experiment comparison
    plot_cross_comparison(res_coin, res_flower, OUT_ROOT)

    # Final summary
    sep = "=" * 65
    print(f"\n{sep}")
    print("  SANITY CHECK SUMMARY")
    print(sep)
    ac = res_coin["asymmetry"]
    af = res_flower["asymmetry"]
    for label, res, asym in [
        ("Coin p=q=0.5", res_coin,   ac),
        ("Flower n=1,m=2", res_flower, af),
    ]:
        print(f"\n  {label}")
        print(f"    H∞              = {res['h_inf']:.4f} bits")
        print(f"    CE  FW / BW     = {asym['ce_fw']:.4f} / {asym['ce_bw']:.4f} bits")
        print(f"    delta_CE        = {asym['delta']:+.4f}  "
              f"({'BW harder — PASS' if asym['delta'] > 0 else 'FW harder — FAIL'})")
        conv_fw = abs(asym['ce_fw'] - res['h_inf'])
        conv_bw = abs(asym['ce_bw'] - res['h_inf'])
        print(f"    FW convergence  = CE_FW - H∞ = {conv_fw:.4f}  "
              f"({'OK' if conv_fw < 0.3 else 'NOT CONVERGED'})")
        print(f"    BW convergence  = CE_BW - H∞ = {conv_bw:.4f}  "
              f"({'OK' if conv_bw < 0.5 else 'NOT CONVERGED'})")
    print(f"\n  Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Outputs in: {OUT_ROOT}/")
    print(sep)


if __name__ == "__main__":
    main()
