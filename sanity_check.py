"""
sanity_check.py
===============
Two contrasting controls for the causal-asymmetry experiment:

  * a POSITIVE control (coin p=q=0.5) where C- > C+ and delta_CE > 0 is
    predicted, and
  * a NULL control (flower n=1, m=2) where C- = C+ exactly and delta_CE ~ 0
    is predicted.

A result is only interpretable if both come out right.  A method that reports
delta_CE > 0 on the null is measuring an artefact of the architecture, not
causal asymmetry.

Coin p=q=0.5  — POSITIVE control
  - Symmetric transition probabilities, so H∞ = 1 bit exactly.
  - C+ = 1 bit    (2 forward causal states, equally probable).
  - C- = 1.5 bits (3 backward causal states, unequally probable).
  - The coin is asymmetric for EVERY (p,q): the three backward states refine
    the two forward states (token 1 == state 1; tokens 0 and 2 split state 0
    by the previous state), so C- - C+ = P(state 0)*H(split) >= 0 identically.
  - Predicted: both arms converge to H∞ = 1.0; delta_CE > 0.

Flower n=1, m=2 with dice_probs = [[0.5, 0.5]]  — NULL control
  - 1 die, so token 0 is always the selection; 2 equally likely outcomes
    (tokens 1 and 2).  Sequence: 0, X, 0, X, ... with X ~ Uniform{1,2}.
  - C+ = 1 + (1/2)log2(n) = 1.0 bit.
  - C- = 1.0 bit as well, NOT 1.5.  This is the correction in
    IMPROVEMENT_PLAN.md A1.  Backward causal states are one per
    *distinguishable* outcome, where outcomes are distinguished by the
    posterior P(die | outcome).  With a single die that posterior is 1 for
    every outcome, so tokens 1 and 2 are indistinguishable looking backward
    and collapse into ONE state -- giving 2 backward states, not 3.
  - The old docstring here stated the refuting fact while drawing the opposite
    conclusion: "token 1 and 2 both deterministically precede token 0" is
    precisely why they cannot be told apart in reverse, and why the m+1 count
    fails at n=1.
  - The process 0,X,0,X,... reads identically backwards: it is exactly
    time-reversible.  Predicted: |delta_CE| ~ 0.

  This arm was previously labelled a positive case and its correct answer
  (delta_CE ~ 0) was printed as "FAIL".  It is now scored as a null.

Sanity checks
  1. Both models converge to H∞ from above (H∞ is time-reversal invariant, so
     the same H∞ applies to both arms of each process).
  2. Coin:   delta_CE > +NULL_TOL          (C- > C+, BW harder)
             |delta_CE| < NULL_TOL  -> INCONCLUSIVE, not a failure: a
             converged model with spare capacity gives ~0 whatever C- - C+ is,
             so this needs the d_model sweep to interpret (plan 1.1 / 4.5).
     Flower: |delta_CE| < NULL_TOL         (C- = C+, no asymmetry)
  3. Empirical C+ ~ theoretical C+ for the coin (2 clusters).
  4. Flower clusters at k=2 for BOTH directions, since C+ and C- both
     correspond to 2 causal states.  Forcing k=m+1=3 on a 2-state latent, as
     this file used to, inflates the empirical C- by construction and then
     "confirms" the 3 states it assumed.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Data_generation import CoinDataset, coin_generation
from Flower_process_generation import FlowerDataset, flower_process_generation
from Model_analysis import (
    FW_BW_attention_comparison,
    flower_complexity,
    flower_entropy_rate,
    latent_extraction,
    paired_delta_ce,
    plot_umap,
    warm_up_umap,
    _project2d,
    _sample_latents,
    _scatter_tokens,
    plot_attention_heatmap,
    perplexity_calculation,
    statistical_complexity,
    statistical_complexity_empirical,
    savefig,
    slim_results,
)
from Training_model import (
    make_analysis_loader,
    make_chunked_loader,
    set_seed,
    train_test_val_pipeline,
)
from utils import (
    cleanup, entropy_rate_coin, mkdir, save_pkl, save_weights,
    to_cpu_for_analysis as to_cpu,
)

# D4: one lazy warm-up, called from main().  UMAP_AVAILABLE / _project2d now
# come from Model_analysis rather than being re-implemented here (C4).


# =============================================================================
# CONFIG
# =============================================================================
CFG = dict(
    seed           = 0,
    accelerator    = "auto",   # see configs.py / train_model on MPS determinism
    val_every_n_steps = 25,    # D1: validation cadence (train curve stays at 1)
    d_model        = 64,
    embed_type     = "onehot",
    n_folds        = 5,
    lr             = 5e-3,
    train_chunk_len = 512,
    attn_vis_len   = 64,
    umap_n_neighbors = 15,   # small: datasets are smaller than main experiments
    umap_n_pts     = 500,
    max_batches    = 20,

    # Coin p=q=0.5  — POSITIVE control: C+=1.0, C-=1.5, predict delta_CE > 0
    coin_p         = 0.5,
    coin_q         = 0.5,
    coin_num_samples = 500,
    coin_seq_len   = 1000,
    coin_max_epochs = 60,
    coin_batch     = 32,
    coin_num_token = 3,
    coin_k_fw      = 2,      # 2 forward causal states
    coin_k_bw      = 3,      # 3 backward causal states

    # Flower n=1, m=2 (fair die) — NULL control: C+ = C- = 1.0, predict ~0.
    # See IMPROVEMENT_PLAN.md A1: with a single die every outcome induces the
    # same posterior over dice, so the m outcomes collapse to ONE backward
    # causal state and the process is exactly time-reversible.
    flower_n       = 1,
    flower_m       = 2,
    flower_num_samples = 500,
    flower_seq_len = 500,   # 500 cycles × 2 obs = 1000 tokens per sequence
    flower_max_epochs = 60,
    flower_batch   = 32,
    flower_k_fw    = 2,      # 2 forward causal states
    flower_k_bw    = 2,      # 2 backward causal states, NOT m+1=3
)

# Tolerance for the null control.  delta_CE is a difference of *residuals*
# (IMPROVEMENT_PLAN.md 1.1), so a finite-capacity, finite-epoch model will not
# hit exactly 0 even on a time-reversible process; this is the band inside
# which the null is considered upheld.
NULL_TOL = 0.02

OUT_ROOT = "sanity_check_flower_process"


# =============================================================================
# HELPERS  (self-contained so this file runs independently)
# =============================================================================
# C4: mkdir, save_pkl, save_weights, cleanup, to_cpu and entropy_rate_coin
# used to be redefined here "so this file runs independently".  They now come
# from utils.py -- one definition, so a fix to one cannot miss the others.


# C4: _project2d was a fourth duplicated helper -- byte-equivalent to the one
# in Model_analysis.  Imported now.


# C6: sanity_check used to carry its own copy of the prefix-sampling bug --
# `flat_lat[:n_pts]` on an array flattened from (N, T, d).  At T=999 with
# n_pts=500 that was positions 0-499 of sequence 0 ALONE.  It now delegates to
# the shared Model_analysis.plot_umap, which draws both the per-sequence and
# the random-across-sequences panels.
def plot_umap_tokens(latents, inputs_arr, num_token, title="",
                     save_path=None, n_pts=500, n_neighbors=15,
                     use_t="last", burn_in=32, seed=0):
    return plot_umap(latents, inputs_arr, num_token, title=title,
                     save_path=save_path, n_pts=n_pts,
                     n_neighbors=n_neighbors, use_t=use_t,
                     burn_in=burn_in, seed=seed)


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
    # D1: validation has its own cadence -> plot against the recorded step.
    for rec, col, lbl in [(rec_fw, "#4c72b0", "Forward val"),
                          (rec_bw, "#dd8452", "Backward val")]:
        vl = np.asarray(rec.step_val_loss, dtype=float)
        if vl.size == 0:
            continue
        xs = getattr(rec, "step_val_at", None) or list(range(len(vl)))
        ax.plot(list(xs)[:len(vl)], vl, color=col, lw=1, marker=".", ms=3,
                label=lbl)
    ax.axhline(h_inf, color="crimson", ls="--", lw=1.5,
               label=f"H∞ = {h_inf:.4f}")
    ax.set_xlabel("Gradient step"); ax.set_ylabel("Val CE loss (bits)")
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
        # C6: raw (N, T, d) and the direction, so plot_umap can sample one
        # position per sequence at max context AND a random spread.
        fig_u, coords = plot_umap_tokens(
            latents, inp_arr, num_token,
            title=tag,
            save_path=os.path.join(out_dir, f"{tag}_umap.png"),
            n_pts=cfg.get("umap_n_pts", 500),
            n_neighbors=cfg.get("umap_n_neighbors", 15),
            use_t=use_t, burn_in=cfg.get("umap_burn_in", 32),
            seed=cfg.get("seed", 0),
        )
        res.update({"latents": latents, "umap_coords": coords})
    except Exception as e:
        print(f"  UMAP: {e}")
        res.update({"latents": None, "umap_coords": None})

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
        # C6: each arm at ITS OWN max-context position, one point per
        # sequence, across all sequences.
        fl_fw, si_fw, lab_fw = _sample_latents(lfw, ifw, mode="per_sequence",
                                               use_t="last",  n_pts=n_pts,
                                               seed=cfg.get("seed", 0))
        fl_bw, si_bw, lab_bw = _sample_latents(lbw, ibw, mode="per_sequence",
                                               use_t="first", n_pts=n_pts,
                                               seed=cfg.get("seed", 0))
        c_fw, mlbl = _project2d(fl_fw, n_neighbors=n_nbr)
        c_bw, _    = _project2d(fl_bw, n_neighbors=n_nbr)

        cmap = plt.cm.tab10
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, coords, raw_inp, lbl in zip(
            axes, [c_fw, c_bw], [si_fw, si_bw],
            [f"Forward model, max context ({mlbl})\n{lab_fw}",
             f"Backward model, max context ({mlbl})\n{lab_bw}"]
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
    POSITIVE control — symmetric coin (p=q=0.5).
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
                                       cfg["coin_batch"], seed=cfg["seed"])
    loader_ana   = make_analysis_loader(ds, cfg["train_chunk_len"],
                                       cfg["coin_batch"], seed=cfg["seed"] + 1000)
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
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"), seed=cfg["seed"], accelerator=cfg["accelerator"],
        val_every_n_steps=cfg["val_every_n_steps"],
    ); cleanup()

    # Backward model
    print("\n  -- Backward model (same forward data) --")
    cv_bw = train_test_val_pipeline(
        loader_train, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="backward",
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"), seed=cfg["seed"], accelerator=cfg["accelerator"],
        val_every_n_steps=cfg["val_every_n_steps"],
    ); cleanup()

    # Analyse
    print("\n  -- Analysis --")
    ana_fw = analyse_model(
        f"{tag}_fw", cv_fw["best_model"], loader_ana, num_token, odir,
        sample_seq, p=p, q=q, mode="forward",
        k=cfg["coin_k_fw"], use_t="last", cfg=cfg); cleanup()
    ana_bw = analyse_model(
        f"{tag}_bw", cv_bw["best_model"], loader_ana, num_token, odir,
        sample_seq, p=p, q=q, mode="backward",
        k=cfg["coin_k_bw"], use_t="first", cfg=cfg); cleanup()

    # Compare
    print("\n  -- Comparison --")
    asym = compare_fw_bw(
        tag, cv_fw, cv_bw, ana_fw, ana_bw,
        loader_ana, num_token, odir, h_inf,
        sample_seq=sample_seq, p=p, q=q, cfg=cfg)
    paired = paired_delta_ce(cv_fw, cv_bw, label=tag, theory=h_inf)

    # Save
    save_weights(cv_fw["best_model"],
                 os.path.join(out_root, "models", f"{tag}_fw.pt"))
    save_weights(cv_bw["best_model"],
                 os.path.join(out_root, "models", f"{tag}_bw.pt"))
    res = dict(tag=tag, p=p, q=q, h_inf=h_inf,
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw,
               asymmetry=asym, paired=paired,
               role="positive",
               C_plus=statistical_complexity(p, q, "forward"),
               C_minus=statistical_complexity(p, q, "backward"))
    save_pkl(slim_results(res), os.path.join(odir, "results.pkl"))
    print(f"\n  Exp A done in {(time.time()-t0)/60:.1f} min")
    return res


# =============================================================================
# EXPERIMENT B — Flower n=1, m=2  (equal dice probabilities)
# =============================================================================
def exp_flower(cfg, out_root):
    """
    NULL CONTROL — flower n=1, m=2 with dice_probs = [[0.5, 0.5]].

    Sequence: 0, X, 0, X, ... where X ∈ {token 1, token 2} with P=0.5 each.
    Token 0 is always the die selection (n=1, only one die).

    C+ = 1 + (1/2)log2(n) = 1.0 bit  (2 forward causal states)
    C- = 1.0 bit                     (2 backward causal states)

    Both directions have TWO causal states, not 2 vs 3 — see the module
    docstring and IMPROVEMENT_PLAN.md A1.  With one die every outcome induces
    the same posterior over dice, so tokens 1 and 2 are indistinguishable
    backward and merge into a single causal state.  The process is exactly
    time-reversible.

    Predicted: both arms converge to H∞ = 0.5 bits and |delta_CE| < NULL_TOL.
    A large positive delta_CE here would mean the pipeline manufactures
    asymmetry where the process has none — which is exactly what this control
    exists to detect.
    """
    n, m    = cfg["flower_n"], cfg["flower_m"]
    tag     = f"flower_n{n}_m{m}_eq"
    odir    = mkdir(os.path.join(out_root, tag))
    t0      = time.time()
    num_token = n + m

    # Equal probability dice → max symmetry within this structure
    dice_probs = np.array([[0.5, 0.5]], dtype=float)   # shape (1, 2)

    # Closed forms rather than the hand-derived constants that used to sit
    # here (A1).  flower_entropy_rate(1, 2, fair) reproduces the old
    # hard-coded 0.5 exactly; flower_complexity corrects C- from 1.5 to 1.0.
    h_inf_flower = flower_entropy_rate(n, m, dice_probs)
    C_plus, C_minus = flower_complexity(n, m, dice_probs)
    k_fw, k_bw = cfg["flower_k_fw"], cfg["flower_k_bw"]

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT B — Flower n={n}  m={m}  (equal dice)  [NULL CONTROL]")
    print(f"  Vocabulary: token 0 = die selection, tokens 1-{n+m-1} = outcomes")
    print(f"  H∞ (analytic) = {h_inf_flower:.4f} bits")
    print(f"  C+ = {C_plus:.4f} bits  ({k_fw} forward causal states)")
    print(f"  C- = {C_minus:.4f} bits  ({k_bw} backward causal states)")
    print(f"  C- - C+ = {C_minus - C_plus:+.4f}  → predicted |delta_CE| < {NULL_TOL}")
    print(f"{'='*65}")

    # Data
    data, _ = flower_process_generation(
        num_samples=cfg["flower_num_samples"],
        seq_len=cfg["flower_seq_len"],
        n=n, m=m, dice_probs=dice_probs)
    seq_len_actual = len(data[0])
    ds           = FlowerDataset(data, seq_len=seq_len_actual)
    loader_train = make_chunked_loader(ds, cfg["train_chunk_len"],
                                       cfg["flower_batch"], seed=cfg["seed"])
    loader_ana   = make_analysis_loader(ds, cfg["train_chunk_len"],
                                       cfg["flower_batch"], seed=cfg["seed"] + 1000)
    sample_seq   = next(iter(loader_ana))[0][0]
    max_len      = seq_len_actual

    # Forward model
    print("\n  -- Forward model --")
    cv_fw = train_test_val_pipeline(
        loader_train, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"), seed=cfg["seed"], accelerator=cfg["accelerator"],
        val_every_n_steps=cfg["val_every_n_steps"],
    ); cleanup()

    # Backward model
    print("\n  -- Backward model (same forward data) --")
    cv_bw = train_test_val_pipeline(
        loader_train, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="backward",
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"), seed=cfg["seed"], accelerator=cfg["accelerator"],
        val_every_n_steps=cfg["val_every_n_steps"],
    ); cleanup()

    # Analyse
    # k=2 for BOTH directions: C+ and C- both correspond to 2 causal states
    # here (A1).  This used to force k=m+1=3 on the backward arm, which
    # inflates the empirical C- by construction — S = H(cluster occupancy) is
    # bounded by log2(k), so handing the backward arm a larger k guarantees
    # part of the "C- > C+" gap regardless of what the latents contain.
    print("\n  -- Analysis --")
    ana_fw = analyse_model(
        f"{tag}_fw", cv_fw["best_model"], loader_ana, num_token, odir,
        sample_seq, mode="forward",
        k=k_fw, use_t="last", cfg=cfg); cleanup()
    ana_bw = analyse_model(
        f"{tag}_bw", cv_bw["best_model"], loader_ana, num_token, odir,
        sample_seq, mode="backward",
        k=k_bw, use_t="first", cfg=cfg); cleanup()

    # Compare
    print("\n  -- Comparison --")
    asym = compare_fw_bw(
        tag, cv_fw, cv_bw, ana_fw, ana_bw,
        loader_ana, num_token, odir, h_inf_flower,
        sample_seq=sample_seq, cfg=cfg)
    paired = paired_delta_ce(cv_fw, cv_bw, label=tag, theory=h_inf_flower)

    # Save
    save_weights(cv_fw["best_model"],
                 os.path.join(out_root, "models", f"{tag}_fw.pt"))
    save_weights(cv_bw["best_model"],
                 os.path.join(out_root, "models", f"{tag}_bw.pt"))
    res = dict(tag=tag, n=n, m=m, h_inf=h_inf_flower,
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw,
               asymmetry=asym, paired=paired, dice_probs=dice_probs,
               role="null", C_plus=C_plus, C_minus=C_minus)
    save_pkl(slim_results(res), os.path.join(odir, "results.pkl"))
    print(f"\n  Exp B done in {(time.time()-t0)/60:.1f} min")
    return res


# =============================================================================
# CROSS-EXPERIMENT COMPARISON PLOT
# =============================================================================
def plot_cross_comparison(res_coin, res_flower, out_root):
    """
    Side-by-side summary comparing the positive and null controls.
    Sanity checks visualised:
      - Both models converge toward their respective H∞
      - delta_CE > 0 for the coin (C- > C+) and ~0 for the flower (C- = C+)
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
    b1 = ax.bar(x - w/2, coin_ces,   w, label="Coin p=q=0.5 (positive)",
                color=colors, alpha=0.85, edgecolor="k")
    b2 = ax.bar(x + w/2, flower_ces, w, label="Flower n=1 m=2 (null)",
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

    # Panel 2: delta CE — each control against its own prediction
    ax = axes[1]
    deltas = [ac["delta"], af["delta"]]
    names  = ["Coin\np=q=0.5\n(C- > C+)", "Flower\nn=1, m=2\n(C- = C+)"]
    # Same three-way verdict as the text summary: a near-zero delta on the
    # positive control is inconclusive, not a failure.
    def _colour(d, role):
        if role == "null":
            return "#4c72b0" if abs(d) < NULL_TOL else "#c44e52"
        if abs(d) < NULL_TOL:
            return "#999999"                 # inconclusive
        return "#4c72b0" if d > 0 else "#c44e52"
    bar_c = [_colour(deltas[0], "positive"), _colour(deltas[1], "null")]
    bars   = ax.bar(names, deltas, color=bar_c, alpha=0.85, edgecolor="k")
    ax.bar_label(bars, fmt="%+.4f", padding=3, fontsize=10)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    # Null band: the flower arm passes by landing inside this, not above it.
    ax.axhspan(-NULL_TOL, NULL_TOL, color="grey", alpha=0.18, zorder=0,
               label=f"null band ±{NULL_TOL}")
    ax.set_ylabel("delta CE = CE_BW - CE_FW (bits)")
    ax.set_title("Causal asymmetry signal\n"
                 "coin: predicted > 0   |   flower: predicted ≈ 0\n"
                 "grey = inside the null band, i.e. inconclusive",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: empirical complexity
    ax = axes[2]
    S_coin_fw  = res_coin["ana_fw"].get("S_emp", 0)
    S_coin_bw  = res_coin["ana_bw"].get("S_emp", 0)
    S_fl_fw    = res_flower["ana_fw"].get("S_emp", 0)
    S_fl_bw    = res_flower["ana_bw"].get("S_emp", 0)
    x = np.arange(2)
    ax.bar(x - w/2, [S_coin_fw,  S_coin_bw],  w,
           label="Coin p=q=0.5 (positive)", color=colors, alpha=0.85,
           edgecolor="k")
    ax.bar(x + w/2, [S_fl_fw,    S_fl_bw],    w,
           label="Flower n=1 m=2 (null)", color=colors, alpha=0.45,
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
    ax.set_title("Causal state complexity\ncoin: C- > C+   |   flower: C- = C+",
                 fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Controls: Coin p=q=0.5 (positive)  vs  Flower n=1, m=2 (null)\n"
                 "Coin has 2 forward / 3 backward causal states; "
                 "the flower has 2 / 2 and is exactly time-reversible",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, os.path.join(out_root, "cross_comparison.png"))
    print(f"  cross-comparison saved -> {out_root}/cross_comparison.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t_total = time.time()
    warm_up_umap(CFG["umap_n_neighbors"])   # D4: once, up front
    set_seed(CFG["seed"])          # A2: reproducible end to end
    mkdir(OUT_ROOT)
    mkdir(os.path.join(OUT_ROOT, "models"))

    print("\n" + "="*65)
    print("  CONTROLS — Coin p=q=0.5 (positive)  vs  Flower n=1, m=2 (null)")
    print("  Coin:   C+ = 1.0, C- = 1.5  → predict delta_CE > 0")
    print(f"  Flower: C+ = 1.0, C- = 1.0  → predict |delta_CE| < {NULL_TOL}")
    print("  Both should converge to their own H∞ from above.")
    print(f"  seed = {CFG['seed']}  (forward and backward arms are paired)")
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
        ("Coin p=q=0.5  [POSITIVE control]", res_coin,   ac),
        ("Flower n=1,m=2  [NULL control]",   res_flower, af),
    ]:
        print(f"\n  {label}")
        print(f"    H∞              = {res['h_inf']:.4f} bits")
        print(f"    C+ / C-         = {res['C_plus']:.4f} / {res['C_minus']:.4f} bits"
              f"   (C- - C+ = {res['C_minus'] - res['C_plus']:+.4f})")
        print(f"    CE  FW / BW     = {asym['ce_fw']:.4f} / {asym['ce_bw']:.4f} bits")

        # Each control is scored against its OWN prediction (A1).  The flower
        # arm predicts delta_CE ~ 0 because C- = C+; scoring it with the
        # positive-control criterion "delta > 0 else FAIL", as this used to,
        # reports the correct answer as a failure.
        delta = asym["delta"]
        if res["role"] == "null":
            verdict = ("PASS — no asymmetry, as predicted"
                       if abs(delta) < NULL_TOL else
                       f"FAIL — |delta| >= {NULL_TOL}, asymmetry where the "
                       "process has none")
        else:
            # The positive control needs the SAME noise band as the null.
            # It used to be a bare sign test, `delta > 0 else FAIL`, which
            # reports delta = -0.0006 -- six ten-thousandths of a bit, with
            # both arms sitting within 0.006 of H_inf -- as a failure.  That
            # is reading a direction off a number indistinguishable from zero,
            # the same over-interpretation the runner's summary table was
            # fixed for.
            #
            # A near-zero delta on the positive control is NOT a refutation.
            # delta_CE is a difference of residuals (IMPROVEMENT_PLAN 1.1), so
            # a converged model with capacity to spare gives ~0 whatever
            # C- - C+ is.  It is a null, and it stays ambiguous until the
            # d_model sweep (Phase 4.5) distinguishes "no asymmetry" from
            # "capacity absorbed it".
            if abs(delta) < NULL_TOL:
                verdict = (f"INCONCLUSIVE — |delta| < {NULL_TOL}, i.e. ~0. "
                           "Consistent with capacity absorbing the asymmetry; "
                           "needs the d_model sweep to interpret")
            elif delta > 0:
                verdict = "PASS — BW harder, as predicted"
            else:
                verdict = "FAIL — FW harder by more than the tolerance, "\
                          "opposite of C- > C+"
        print(f"    delta_CE        = {delta:+.4f}  ({verdict})")

        paired = res.get("paired")
        if paired:
            print(f"    paired delta_CE = {paired['mean']:+.4f} ± {paired['sem']:.4f} "
                  f"(sem over {paired['n']} folds, t={paired['t']:+.2f})")

        conv_fw = abs(asym['ce_fw'] - res['h_inf'])
        conv_bw = abs(asym['ce_bw'] - res['h_inf'])
        print(f"    FW convergence  = |CE_FW - H∞| = {conv_fw:.4f}  "
              f"({'OK' if conv_fw < 0.3 else 'NOT CONVERGED'})")
        print(f"    BW convergence  = |CE_BW - H∞| = {conv_bw:.4f}  "
              f"({'OK' if conv_bw < 0.5 else 'NOT CONVERGED'})")
    print(f"\n  Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Outputs in: {OUT_ROOT}/")
    print(sep)


if __name__ == "__main__":
    main()
