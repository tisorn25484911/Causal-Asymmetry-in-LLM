"""
run_experiments.py — the single training runner.

    python run_experiments.py --config QUICK      # ~1 hour
    python run_experiments.py --config LARGE      # ~13 hours
    python run_experiments.py --config SMOKE      # ~2 minutes, exercises paths
    python run_experiments.py --config QUICK --seed 1 --only exp1

Replaces Main_call.py and main_large.py, which were copies of the same file
differing only in their CFG block and two tag literals, and which both wrote to
results/ with a colliding tag (IMPROVEMENT_PLAN.md A4).  Configurations live in
configs.py, each with its own out_root; tags are derived from config values so
they cannot disagree with the parameters that produced them.

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
    Experiment 1.2: Train forward and backward models on the coin process data with different parameters. (see configs.py)
        - Repeat the same steps as in Experiment 1 for the new coin process data.
        - calculate the empirical and theoretical statistical complexity for the new coin process and compare with the previous one (plot a heat map).
        - plot a heat map of "the difference in statistical complexity" for the new coin process and compare with the previous one (plot a heat map)
        - plot a heat map of "the difference in perplexity" between forward and backward models across different (p, q) values on the same 2000 seq long test data.

    Experiment 2: Train forward and backward models on the flower process data, once per entry in cfg["flower_configs"].
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
import argparse
import json
import os
import pickle
import time

# ── third-party ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── project ────────────────────────────────────────────────────────────────
from configs import CONFIGS, QUICK
from Data_generation import CoinDataset, coin_generation
# C4: the flower Dataset AND generator both come from here now.  Data_generation
# used to carry a second, non-parametric copy of both, and every runner imported
# the generator from there and the Dataset from here.
from Flower_process_generation import FlowerDataset, flower_process_generation
from Model_analysis import (
    warm_up_umap,
    _sample_latents,
    _scatter_tokens,
    slim_results,
    savefig,
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
from utils import (
    cleanup, coin_tag, entropy_rate_coin, flower_tag, mkdir, save_pkl,
    save_weights, to_cpu_for_analysis,
)
from Training_model import (
    make_analysis_loader, make_chunked_loader, set_seed,
    train_test_val_pipeline,
)
from pq_experiment import heatmap_theory, plot_heatmap, pq_experiment_full

# D4: one lazy warm-up, called from main() -- this used to be a third copy of
# the same JIT compile (Model_analysis at import, here, and sanity_check).


# ══════════════════════════════════════════════════════════════════════════
# CONFIG — see configs.py.  Selected with --config; QUICK by default so that
# importing this module (notebooks, ad-hoc use) still gets a usable CFG.
# ══════════════════════════════════════════════════════════════════════════
CFG = dict(QUICK)


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# C4: cleanup, to_cpu_for_analysis, entropy_rate_coin, coin_tag, flower_tag,
# mkdir, save_pkl and save_weights are imported from utils.py -- they used to
# be defined here AND in Model_analysis.py AND in sanity_check.py.
# ══════════════════════════════════════════════════════════════════════════

def weight_meta(cfg, tag, num_token, max_len, res) -> dict:
    """
    Sidecar written next to every .pt — IMPROVEMENT_PLAN.md B11 / A4.

    Records the architecture a loader must rebuild (d_model, n_layers,
    token_size, max_len) and the process the weights were trained on (p/q or
    n/m).  Without it, LLM_asymmetry_testing.py and Test_data_eval.py had to
    *guess* both from their own hard-coded CFG -- and A4 is precisely the case
    where that guess is wrong and load_state_dict either raises confusingly or,
    once d_model happens to line up, silently scores a checkpoint against the
    wrong process.
    """
    return {
        "tag":        tag,
        "d_model":    cfg["d_model"],
        "n_layers":   cfg["n_layers"],
        "token_size": num_token,
        "max_len":    max_len,
        "embed_type": cfg["embed_type"],
        "seed":       cfg["seed"],
        "p":          res.get("p"),
        "q":          res.get("q"),
        "n":          res.get("n"),
        "m":          res.get("m"),
    }


def full_seq_len(dataset) -> int:
    """
    Length of the model's input sequence, i.e. len(seq) - 1 given that the
    Datasets emit (x[:-1], x[1:]).  Used for max_len so the positional
    encoding table covers the whole range at analysis time (B5).
    """
    inp, _ = dataset[0]
    return int(inp.shape[0])


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
            # D1: validation is recorded on its own, sparser cadence, so plot
            # it against the recorded global_step.  Plotting against the list
            # index would squeeze the whole run into the first few percent of
            # the axis and make it look as if training stopped early.
            xs = getattr(rec, "step_val_at", None) or range(len(rec.step_val_loss))
            ax2.plot(list(xs), rec.step_val_loss, color=col, lw=1.0,
                     alpha=0.75, marker=".", ms=3, label=lbl)
        if th == th:
            ax2.axhline(th, color=col, ls="--", lw=1.5, alpha=0.6, label=f"H∞ {lbl[:2]}={th:.4f}")
    ax2.set_xlabel("Gradient step"); ax2.set_ylabel("Val Loss (bits)")
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
                  umap_n_pts=1000, umap_n_neighbors=15,
                  umap_burn_in=32, umap_seed=0, max_batches=20):
    """
    `loader` must be the dedicated analysis loader from
    make_analysis_loader() — windows at the TRAINING chunk length, drawn from
    a different seed.  Not the full-length loader: the model only ever sees
    positional-encoding indices [0, chunk), so evaluating at full length
    measures extrapolation, and does so asymmetrically between the two arms.
    See make_analysis_loader for the measured numbers.

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
        latents, inp_arr, _ = latent_extraction(model, loader, max_batches=max_batches)
        # C6: pass the raw (N, T, d) latents, not a flattened prefix, and tell
        # plot_umap which direction this is so it samples the max-context
        # position and trims the burn-in from the correct end.
        _, coords = plot_umap(latents, inp_arr, num_token, title=tag,
                              save_path=os.path.join(out_dir, f"{tag}_umap.png"),
                              n_pts=umap_n_pts, n_neighbors=umap_n_neighbors,
                              use_t=use_t, burn_in=umap_burn_in, seed=umap_seed)
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
    n_nbr       = cfg.get("umap_n_neighbors", 15)
    n_pts_cfg   = cfg.get("umap_n_pts", 1000)
    umap_seed   = cfg.get("seed", 0)
    max_batches = cfg.get("max_batches", 20)
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
    # Both arms see the same forward-generated sequences; they differ only in
    # the attention mask and the batch convention.
    #
    # C6: each arm is sampled at ITS OWN max-context position — last for the
    # forward (tril) model, first for the backward (triu) one — one point per
    # sequence, across all sequences.  This used to take the first n_pts rows
    # of the flattened array, i.e. every position of the first few sequences,
    # so the two panels were comparing single trajectories.
    try:
        lfw, ifw, _ = latent_extraction(mfw, loader_ana, max_batches=max_batches)
        lbw, ibw, _ = latent_extraction(mbw, loader_ana, max_batches=max_batches)
        fl_fw, si_fw, lab_fw = _sample_latents(
            lfw, ifw, mode="per_sequence", use_t="last",
            n_pts=n_pts_cfg, seed=umap_seed)
        fl_bw, si_bw, lab_bw = _sample_latents(
            lbw, ibw, mode="per_sequence", use_t="first",
            n_pts=n_pts_cfg, seed=umap_seed)
        c_fw, mlbl = _project2d(fl_fw, n_neighbors=n_nbr)
        c_bw, _    = _project2d(fl_bw, n_neighbors=n_nbr)

        # Independent projections — no shared axis range
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, coords, raw_inp, lbl in zip(
            axes, [c_fw, c_bw], [si_fw, si_bw],
            [f"Forward model, max context ({mlbl})\n{lab_fw}",
             f"Backward model, max context ({mlbl})\n{lab_bw}"]
        ):
            _scatter_tokens(ax, coords, raw_inp, num_token, lbl)
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
    loader_fw_ana = make_analysis_loader(ds_fw, chunk, cfg["ana_batch"],
                                        seed=seed + 1000)
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
        n_layers=cfg["n_layers"], accelerator=cfg["accelerator"],
        val_every_n_steps=cfg["val_every_n_steps"],
    )
    cleanup()  # FIX-4

    print("\n  -- 1b Backward CV --")
    cv_bw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="backward",  # forward data → backward model
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"), seed=seed,
        n_layers=cfg["n_layers"], accelerator=cfg["accelerator"],
        val_every_n_steps=cfg["val_every_n_steps"],
    )
    cleanup()  # FIX-4

    print("\n  -- 1c Analysis --")
    ana_fw = analyse_model(f"{tag}_fw", cv_fw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, p, q, "forward",
                           k=2, use_t="last", attn_vis_len=cfg["attn_vis_len"],
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"],
                           umap_burn_in=cfg["umap_burn_in"], umap_seed=cfg["seed"],
                           max_batches=cfg["max_batches"])
    cleanup()  # GC between fw and bw analysis (model already moved to CPU)
    # A3: use_t="first" for the backward arm.  With a triu mask position T-1
    # attends to itself only, so reading complexity at "last" read latents that
    # encode a single token — which cannot express more than token identity.
    ana_bw = analyse_model(f"{tag}_bw", cv_bw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, p, q, "backward",
                           k=3, use_t="first", attn_vis_len=cfg["attn_vis_len"],
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"],
                           umap_burn_in=cfg["umap_burn_in"], umap_seed=cfg["seed"],
                           max_batches=cfg["max_batches"])
    cleanup()

    print("\n  -- 1d Comparison --")
    compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw,
                  loader_fw_ana, num_token, odir, sample_seq,
                  theory, theory, cfg["attn_vis_len"], p, q, cfg=cfg)
    paired = paired_delta_ce(cv_fw, cv_bw, label=tag, theory=theory)

    print(f"\n  Exp 1 done in {(time.time()-t0)/60:.1f} min")
    res = dict(tag=tag, p=p, q=q, theory=theory, seed=seed, paired=paired,
               C_plus=statistical_complexity(p, q, "forward"),
               C_minus=statistical_complexity(p, q, "backward"),
               cv_fw=cv_fw, cv_bw=cv_bw, ana_fw=ana_fw, ana_bw=ana_bw)
    save_pkl(slim_results(res), os.path.join(odir, "results.pkl"))
    _meta = weight_meta(cfg, tag, num_token, max_len, res)
    save_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw.pt"),
                 meta={**_meta, "mode": "forward"})
    save_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw.pt"),
                 meta={**_meta, "mode": "backward"})
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
    loader_fw_ana = make_analysis_loader(ds_fw, chunk, cfg["ana_batch"],
                                        seed=seed + 1000)
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
        n_layers=cfg["n_layers"], accelerator=cfg["accelerator"],
        val_every_n_steps=cfg["val_every_n_steps"],
    )
    cleanup()
    cv_bw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["coin_max_epochs"], lr=cfg["lr"], mode="backward",  # forward data → backward model
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"), seed=seed,
        n_layers=cfg["n_layers"], accelerator=cfg["accelerator"],
        val_every_n_steps=cfg["val_every_n_steps"],
    )
    cleanup()

    ana_fw = analyse_model(f"{tag}_fw", cv_fw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, p, q, "forward",
                           k=2, use_t="last", attn_vis_len=cfg["attn_vis_len"],
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"],
                           umap_burn_in=cfg["umap_burn_in"], umap_seed=cfg["seed"],
                           max_batches=cfg["max_batches"])
    cleanup()
    ana_bw = analyse_model(f"{tag}_bw", cv_bw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, p, q, "backward",
                           k=3, use_t="first", attn_vis_len=cfg["attn_vis_len"],  # A3
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"],
                           umap_burn_in=cfg["umap_burn_in"], umap_seed=cfg["seed"],
                           max_batches=cfg["max_batches"])
    cleanup()
    compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw,
                  loader_fw_ana, num_token, odir, sample_seq,
                  theory, theory, cfg["attn_vis_len"], p, q, cfg=cfg)
    paired = paired_delta_ce(cv_fw, cv_bw, label=tag, theory=theory)

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
    _meta = weight_meta(cfg, tag, num_token, max_len, res)
    save_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw.pt"),
                 meta={**_meta, "mode": "forward"})
    save_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw.pt"),
                 meta={**_meta, "mode": "backward"})
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
    loader_fw_ana = make_analysis_loader(ds_fw, chunk, cfg["ana_batch"],
                                        seed=seed + 1000)
    sample_seq    = next(iter(loader_fw))[0][0]

    max_len = full_seq_len(ds_fw)                        # B5

    print("\n  -- 2a Forward CV --")
    cv_fw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="forward",
        save_plot=os.path.join(odir, f"{tag}_fw_cv.png"), seed=seed,
        n_layers=cfg["n_layers"], accelerator=cfg["accelerator"],
        val_every_n_steps=cfg["val_every_n_steps"],
    )
    cleanup()

    print("\n  -- 2b Backward CV --")
    cv_bw = train_test_val_pipeline(
        loader_fw, test_ratio=(0.20, 0.80), n_folds=cfg["n_folds"],
        embed_type=cfg["embed_type"], num_token=num_token,
        d_model=cfg["d_model"], max_len=max_len,
        max_epochs=cfg["flower_max_epochs"], lr=cfg["lr"], mode="backward",  # forward data → backward model
        save_plot=os.path.join(odir, f"{tag}_bw_cv.png"), seed=seed,
        n_layers=cfg["n_layers"], accelerator=cfg["accelerator"],
        val_every_n_steps=cfg["val_every_n_steps"],
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
                           umap_n_neighbors=cfg["umap_n_neighbors"],
                           umap_burn_in=cfg["umap_burn_in"], umap_seed=cfg["seed"],
                           max_batches=cfg["max_batches"])
    ana_fw["S_theory"] = C_plus
    cleanup()
    ana_bw = analyse_model(f"{tag}_bw", cv_bw["best_model"], loader_fw_ana,
                           num_token, odir, sample_seq, None, None, "backward",
                           k=m+1, use_t="first", attn_vis_len=cfg["attn_vis_len"],  # A3
                           umap_n_pts=cfg["umap_n_pts"],
                           umap_n_neighbors=cfg["umap_n_neighbors"],
                           umap_burn_in=cfg["umap_burn_in"], umap_seed=cfg["seed"],
                           max_batches=cfg["max_batches"])
    ana_bw["S_theory"] = C_minus
    cleanup()

    print("\n  -- 2d Comparison --")
    compare_fw_bw(tag, cv_fw, cv_bw, ana_fw, ana_bw,
                  loader_fw_ana, num_token, odir, sample_seq,
                  theory, theory, cfg["attn_vis_len"], cfg=cfg)
    paired = paired_delta_ce(cv_fw, cv_bw, label=tag, theory=theory)
    if paired:
        mean, sem = paired["mean"], paired.get("sem", float("nan"))
        if np.isfinite(sem) and abs(mean) < 2 * sem:
            print(f"  delta_CE = {mean:+.4f} ± {sem:.4f} — not distinguishable "
                  f"from zero, so the sign carries no information  [{role}]")
        else:
            agrees = (mean > 0) == (C_minus > C_plus)
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
    _meta = weight_meta(cfg, tag, num_token, max_len, res)
    save_weights(cv_fw["best_model"], os.path.join(out_root, "models", f"{tag}_fw.pt"),
                 meta={**_meta, "mode": "forward"})
    save_weights(cv_bw["best_model"], os.path.join(out_root, "models", f"{tag}_bw.pt"),
                 meta={**_meta, "mode": "backward"})
    all_results[tag] = slim_results(res)   # keep only metrics; frees models+latents
    cleanup()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the causal-asymmetry training experiments.")
    ap.add_argument("--config", default="QUICK", choices=sorted(CONFIGS),
                    help="which configuration from configs.py to run")
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg['seed']; both arms always share it")
    ap.add_argument("--out-root", default=None,
                    help="override cfg['out_root']")
    ap.add_argument("--only", default="all",
                    choices=["all", "exp1", "exp1_2", "exp2"],
                    help="run a single experiment instead of the whole suite")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg  = dict(CONFIGS[args.config])
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.out_root is not None:
        cfg["out_root"] = args.out_root

    # A4: out_root comes from the config, so two configurations can no longer
    # overwrite each other's weights in a shared results/models/ directory.
    out_root = cfg["out_root"]
    # FIX-2: pre-compile numba kernels so a segfault, if any, happens now
    # rather than hours into training.
    warm_up_umap(cfg.get("umap_n_neighbors", 15))
    set_seed(cfg["seed"])                    # A2: reproducible end to end
    mkdir(out_root)
    mkdir(os.path.join(out_root, "models"))

    # Record exactly what produced this directory, next to the outputs.  Named
    # per config so a second config sharing an out_root cannot overwrite the
    # first one's provenance.
    with open(os.path.join(out_root, f"run_config_{args.config}.json"), "w") as f:
        json.dump({"config": args.config, "only": args.only,
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   **cfg}, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  config={args.config}  seed={cfg['seed']}  out_root={out_root}")
    print(f"  d_model={cfg['d_model']}  n_layers={cfg['n_layers']}  lr={cfg['lr']}")
    print(f"{'='*70}")

    all_results = {}
    t_start = time.time()

    if args.only in ("all", "exp1"):
        experiment_1(cfg, out_root, all_results)
    if args.only in ("all", "exp1_2"):
        experiment_1_2(cfg, out_root, all_results)
    if args.only in ("all", "exp2"):
        for n, m, role in cfg["flower_configs"]:
            experiment_2(cfg, out_root, all_results, n=n, m=m, role=role)

    # Merge into any existing bundle rather than replacing it.  With --only,
    # or when a second config writes to the same out_root, overwriting would
    # silently destroy the record of every experiment not run this time --
    # the .pkl and .pt files on disk would survive but all_results.pkl, which
    # is what the analysis notebooks read, would list only the latest subset.
    # Keyed by tag, and tags are derived from config values (A4), so a re-run
    # of the same experiment replaces its own entry and nothing else.
    combined_path = os.path.join(out_root, "all_results.pkl")
    combined = {}
    if os.path.exists(combined_path):
        try:
            with open(combined_path, "rb") as f:
                combined = pickle.load(f)
            kept = [k for k in combined if k not in all_results]
            if kept:
                print(f"\n  merging into existing all_results.pkl; keeping "
                      f"{len(kept)} earlier experiment(s): {', '.join(sorted(kept))}")
        except Exception as e:
            print(f"  ! could not read existing all_results.pkl ({e}) — writing fresh")
            combined = {}
    combined.update(all_results)
    save_pkl(combined, combined_path)                                 # already slim

    # ── Cross-experiment summary: does sign(delta_CE) track sign(C- - C+)? ──
    print(f"\n{'='*70}\n  ASYMMETRY SUMMARY  (every experiment in {out_root})\n{'='*70}")
    print(f"  {'experiment':<26} {'C+':>7} {'C-':>7} {'C- - C+':>9} "
          f"{'delta_CE':>10} {'sem':>8} {'verdict':>10}")
    for tag, r in sorted(combined.items()):
        pd = r.get("paired")
        cp, cm = r.get("C_plus"), r.get("C_minus")
        if pd is None or cp is None or cm is None:
            continue
        mean, sem = pd["mean"], pd.get("sem", float("nan"))
        # Only call a sign when it is distinguishable from zero.  delta_CE is a
        # difference of residuals (IMPROVEMENT_PLAN.md 1.1): a converged,
        # sufficient-capacity predictor gives 0 whatever C- - C+ is, so a
        # near-zero value is a null result, NOT evidence for or against the
        # sign of C- - C+.  Reading a sign off +0.0006 +/- 0.0011 is exactly
        # the over-interpretation this repo has been prone to.
        if np.isfinite(sem) and abs(mean) < 2 * sem:
            verdict = "n.s."
        elif (mean > 0) == (cm > cp):
            verdict = "match"
        else:
            verdict = "MISMATCH"
        print(f"  {tag:<26} {cp:>7.4f} {cm:>7.4f} {cm-cp:>+9.4f} "
              f"{mean:>+10.4f} {sem:>8.4f} {verdict:>10}")
    print(f"{'='*70}")
    print("  n.s. = |delta_CE| < 2 sem, i.e. not distinguishable from zero.")
    print("  delta_CE is a difference of RESIDUALS, so a converged predictor")
    print("  with enough capacity gives ~0 regardless of C- - C+.  A null here")
    print("  is ambiguous until the d_model sweep (plan Phase 4.5) is run.")
    print(f"{'='*70}")

    total = (time.time() - t_start) / 60
    print(f"\n{'='*70}\n  ALL COMPLETE — {total:.1f} min")
    for root, dirs, files in os.walk(out_root):
        lvl    = root.replace(out_root, "").count(os.sep)
        indent = "  " * (lvl + 1)
        print(f"{'  '*lvl}{os.path.basename(root)}/")
        for f in sorted(files):
            print(f"{indent}{f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()