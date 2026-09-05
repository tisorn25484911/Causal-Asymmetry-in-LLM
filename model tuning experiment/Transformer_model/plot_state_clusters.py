"""
plot_state_clusters.py — causal-state clusters, train vs held-out, per metric.

For every experiment in an output directory, and for each distance metric on
the probability simplex, writes one 2x2 figure:

        forward / TRAIN        backward / TRAIN
        forward / HELD-OUT     backward / HELD-OUT

Each panel is a UMAP of the model's PREDICTIVE DISTRIBUTION at the max-context
position of every sequence, coloured by the causal state recovered by
agglomerative clustering (Model_analysis.recover_causal_states).

    python plot_state_clusters.py                      # results_quick, all metrics
    python plot_state_clusters.py --out-root results_large --metrics js

WHY TRAIN vs HELD-OUT.  Every S_emp / S_hat / UMAP reported elsewhere in this
repo is computed on the FULL dataset -- roughly 80% of which the model trained
on.  That was flagged as a caveat but never quantified.  Splitting the panels
answers it directly: if the recovered state structure is real rather than
memorised, the held-out row must look like the train row and recover the same
k_hat.

The split is reproduced EXACTLY, not approximated.  Training used
    torch.utils.data.random_split(ds, [n_train, n_test], generator=seed)
which internally is randperm(N, generator)[:n_train] / [n_train:], and depends
only on N and the seed -- not on the dataset contents.  So the same permutation
reproduces the identical partition; verified against test_train_validation.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as tud

# ── repo path bootstrap ────────────────────────────────────────────────────
# REORGANISATION_FIX_PLAN.md 4.1.  This file lives in Transformer_model/ but
# imports `configs` from Experimental_setup/, so both directories have to be
# importable.  Python only ever puts the *script's own* directory on sys.path --
# which is why this is needed, and why cd-ing elsewhere does not help.  Anchored
# on __file__, so the script runs from any working directory.
#
# (That cross-directory import is also why this script arguably belongs in
# Experimental_setup/ -- see REORGANISATION_FIX_PLAN.md 7.1.)
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs import CONFIGS, QUICK
from Data_generation import CoinDataset, coin_generation
from Flower_process_generation import FlowerDataset, flower_process_generation
from Model_analysis import (DEFAULT_STATE_TOL, DISTANCE_MATRICES, _project2d,
                            recover_causal_states, savefig, warm_up_umap)
from OneHot_model import OneHotDecoder
from Training_model import ChunckDataset, set_seed
from utils import coin_tag, flower_tag, mkdir, repo_path


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild the exact data and the exact split a run used
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(kind, params, cfg):
    """Regenerate the dataset for one experiment, byte-identical to the run."""
    set_seed(cfg["seed"])
    if kind == "coin":
        p, q, n_samp, seq_len = params
        data, _ = coin_generation(num_samples=n_samp, seq_len=seq_len, p=p, q=q)
        return CoinDataset(data, seq_len=seq_len), 3
    n, m, n_samp, seq_len = params
    dice = np.random.default_rng(cfg["flower_dice_seed"]).dirichlet(np.ones(m), size=n)
    data, _ = flower_process_generation(num_samples=n_samp, seq_len=seq_len,
                                        n=n, m=m, dice_probs=dice)
    return FlowerDataset(data, seq_len=len(data[0])), n + m


def split_indices(n_total, seed, test_ratio=0.20):
    """
    The SAME partition train_test_val_pipeline used.

    torch.utils.data.random_split draws randperm(N, generator) and slices it,
    so the partition depends only on N and the seed.  Verified identical to
    test_train_validation's Subset.indices.
    """
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test
    return perm[:n_train], perm[n_train:]


def analysis_loaders(ds, chunk, batch, seed, test_ratio=0.20):
    """
    Two loaders over the SAME deterministic analysis windows (seed+1000, as in
    make_analysis_loader), one restricted to train-seen sequences and one to
    held-out.  Same windowing for both, so the only difference between the rows
    of the figure is whether the model was trained on those sequences.
    """
    ana = ChunckDataset(ds, chunk, seed=seed + 1000)
    tr_idx, te_idx = split_indices(len(ana), seed, test_ratio)
    mk = lambda idx: tud.DataLoader(tud.Subset(ana, idx), batch_size=batch,
                                    shuffle=False, num_workers=0)
    return mk(tr_idx), mk(te_idx), len(tr_idx), len(te_idx)


def load_arm(out_root, tag, arm, num_token, max_len, d_model, n_layers):
    path = os.path.join(out_root, "models", f"{tag}_{arm}.pt")
    if not os.path.exists(path):
        return None
    m = OneHotDecoder(token_size=num_token, d_model=d_model, max_len=max_len,
                      mode="forward" if arm == "fw" else "backward",
                      n_layers=n_layers)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# One panel
# ─────────────────────────────────────────────────────────────────────────────
# Marker per recovered state.  Colour is reserved for the input TOKEN so it
# stays consistent with every other UMAP in the repo, which means the two
# encodings are independent and the reader can see the mapping between them --
# e.g. the coin's forward arm shows tokens 0 and 2 in DIFFERENT colours but the
# SAME marker, which is precisely the claim "two tokens, one causal state".
_STATE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]


def draw_panel(ax, model, loader, use_t, metric, tol, k_theory, n_pts,
               n_neighbors, seed, split_name, arm_name):
    """
    UMAP of the predictive distribution.

    colour = input TOKEN            (same convention as the other UMAP figures)
    marker = RECOVERED causal state (from recover_causal_states)
    """
    try:
        r = recover_causal_states(model, loader, use_t=use_t, max_batches=None,
                                  metric=metric, state_tol=tol,
                                  n_pts=n_pts, seed=seed)
    except Exception as e:
        ax.set_title(f"{arm_name} / {split_name} — failed", fontsize=9)
        ax.text(0.5, 0.5, str(e)[:70], ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        return None

    P, lab, toks = r["probs"], r["labels"], r["tokens"]
    # Jitter only for the PICTURE, at a quarter of the state tolerance -- UMAP
    # is scale-free and would otherwise inflate sub-tolerance differences into
    # separate blobs.  k_hat itself is computed on the unjittered distances.
    rng = np.random.default_rng(seed)
    coords, mlbl = _project2d(P + rng.normal(0, tol / 4.0, P.shape),
                              n_neighbors=min(n_neighbors, max(len(P) - 1, 2)))

    from matplotlib.lines import Line2D
    cmap = plt.cm.tab10
    n_tok = int(toks.max()) + 1
    states = sorted(np.unique(lab))
    tok_colour = lambda t: cmap(t / max(n_tok - 1, 1))

    # One scatter per (state, token) pair actually present, so colour and
    # marker vary independently.
    for si, c in enumerate(states):
        for t in range(n_tok):
            sel = (lab == c) & (toks == t)
            if not sel.any():
                continue
            ax.scatter(coords[sel, 0], coords[sel, 1],
                       color=tok_colour(t), marker=_STATE_MARKERS[si % len(_STATE_MARKERS)],
                       s=22, alpha=0.75, linewidths=0.4, edgecolors="none")

    # Two legends: colour->token and marker->state, since they are independent.
    tok_handles = [Line2D([], [], marker="o", linestyle="", color=tok_colour(t),
                          markersize=6, label=f"Token {t}")
                   for t in range(n_tok) if (toks == t).any()]
    st_handles = []
    for si, c in enumerate(states):
        sel = lab == c
        comp = np.bincount(toks[sel], minlength=n_tok)
        members = ",".join(str(t) for t in np.argsort(comp)[::-1] if comp[t] > 0)
        st_handles.append(Line2D([], [], marker=_STATE_MARKERS[si % len(_STATE_MARKERS)],
                                 linestyle="", color="0.35", markersize=6,
                                 label=f"state {c}: tok {members}  (n={int(sel.sum())})"))
    leg1 = ax.legend(handles=tok_handles, title="colour = token", fontsize=6,
                     title_fontsize=6, loc="upper right", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=st_handles, title="marker = recovered state", fontsize=6,
              title_fontsize=6, loc="lower right", framealpha=0.9)

    ok = "OK" if r["k_hat"] == k_theory else "!="
    ax.set_title(f"{arm_name} — {split_name}   ({mlbl})\n"
                 f"k̂={r['k_hat']} {ok} theory {k_theory}   "
                 f"S={r['S_hat']:.4f}   n={len(P)}", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.2)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Experiment table
# ─────────────────────────────────────────────────────────────────────────────
def experiments(cfg):
    """(tag, kind, params, chunk, k_fw, k_bw) for every experiment in a run."""
    out = []
    ch = cfg["train_chunk_len"]
    p1, q1 = cfg["coin_p1"], cfg["coin_q1"]
    out.append((coin_tag("exp1", p1, q1), "coin",
                (p1, q1, cfg["coin_num_samples"], cfg["coin_seq_len"]), ch, 2, 3))
    p2, q2 = cfg["coin_p2"], cfg["coin_q2"]
    out.append((coin_tag("exp1_2", p2, q2), "coin",
                (p2, q2, cfg["coin_num_samples_12"], cfg["coin_seq_len_12"]),
                min(ch, cfg["coin_seq_len_12"] - 1), 2, 3))
    for n, m, _role in cfg["flower_configs"]:
        out.append((flower_tag("exp2", n, m), "flower",
                    (n, m, cfg["flower_num_samples"], cfg["flower_seq_len"]),
                    ch, n + 1, m + 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="QUICK", choices=sorted(CONFIGS))
    ap.add_argument("--also-config", default="QUICK_LARGE_HMM",
                    help="second config sharing the same out_root ('' to skip)")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--metrics", nargs="+", default=sorted(DISTANCE_MATRICES))
    args = ap.parse_args(argv)

    cfg = dict(CONFIGS[args.config])
    out_root = repo_path(args.out_root or cfg["out_root"])   # FIX_PLAN 4.2
    warm_up_umap(cfg["umap_n_neighbors"])

    todo = experiments(cfg)
    seen = {t[0] for t in todo}
    if args.also_config:
        for e in experiments(dict(CONFIGS[args.also_config])):
            if e[0] not in seen:
                todo.append(e); seen.add(e[0])

    summary = []
    for tag, kind, params, chunk, k_fw, k_bw in todo:
        odir = os.path.join(out_root, tag)
        if not os.path.isdir(odir):
            print(f"  skip {tag} — no output directory"); continue
        side = os.path.join(out_root, "models", f"{tag}_fw.json")
        if not os.path.exists(side):
            print(f"  skip {tag} — no sidecar"); continue
        meta = json.load(open(side))

        ds, V = build_dataset(kind, params, cfg)
        tr_loader, te_loader, n_tr, n_te = analysis_loaders(
            ds, chunk, cfg["ana_batch"], cfg["seed"])
        arms = {a: load_arm(out_root, tag, a, V, meta["max_len"],
                            meta["d_model"], meta["n_layers"]) for a in ("fw", "bw")}
        if any(v is None for v in arms.values()):
            print(f"  skip {tag} — missing weights"); continue

        print(f"\n{tag}   train={n_tr} seqs  held-out={n_te} seqs  "
              f"(theory k: fw={k_fw} bw={k_bw})")
        for metric in args.metrics:
            tol = DEFAULT_STATE_TOL[metric]
            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            for row, (split, loader) in enumerate((("TRAIN", tr_loader),
                                                   ("HELD-OUT", te_loader))):
                for col, (arm, use_t, kt) in enumerate((("fw", "last", k_fw),
                                                        ("bw", "first", k_bw))):
                    r = draw_panel(axes[row][col], arms[arm], loader, use_t,
                                   metric, tol, kt, cfg["umap_n_pts"],
                                   cfg["umap_n_neighbors"], cfg["seed"],
                                   split, "forward" if arm == "fw" else "backward")
                    if r:
                        summary.append((tag, metric, split, arm, r["k_hat"], kt,
                                        r["S_hat"]))
            fig.suptitle(
                f"{tag}   —   causal states from the predictive distribution\n"
                f"metric = {metric}   threshold = {tol}   "
                f"(jitter {tol/4:.3f} applied to the PICTURE only; k̂ is computed "
                f"on unjittered distances)\n"
                f"train = {n_tr} sequences the model saw   |   "
                f"held-out = {n_te} sequences it never saw",
                fontsize=11, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            savefig(fig, os.path.join(odir, f"{tag}_states_{metric}.png"))

    print(f"\n{'='*94}\n  SUMMARY — k̂ recovered, train vs held-out\n{'='*94}")
    print(f"  {'experiment':<24}{'metric':<11}{'arm':<4}{'theory':>7}"
          f"{'k̂ train':>9}{'k̂ held':>9}{'S train':>9}{'S held':>9}")
    by = {}
    for tag, metric, split, arm, kh, kt, sh in summary:
        by.setdefault((tag, metric, arm), {})[split] = (kh, sh)
        by[(tag, metric, arm)]["theory"] = kt
    for (tag, metric, arm), d in by.items():
        if "TRAIN" not in d or "HELD-OUT" not in d:
            continue
        (kt_, st), (kh_, sh) = d["TRAIN"], d["HELD-OUT"]
        flag = "" if kt_ == kh_ else "   <- train/held-out DISAGREE"
        print(f"  {tag:<24}{metric:<11}{arm:<4}{d['theory']:>7}"
              f"{kt_:>9}{kh_:>9}{st:>9.4f}{sh:>9.4f}{flag}")
    print("="*94)


if __name__ == "__main__":
    main()
