"""
The four figures, and the two aggregates.

    F1  causal states, DISCRETE     occupancy vs theory + emission table
    F2  causal states, CLUSTERED    predictive distributions, coloured by cluster
    F3  learned transition matrix   + the closed form, forward arms
    F4  complexity                  theory vs discrete vs clustered

    grid   F4_complexity_grid.png   theory vs empirical, every cell
    pooled F4_all_processes.png     theory vs empirical, every process

Labels are deliberately minimal and nothing here writes an explanation into the
image: the numbers are in the pickle and the reasoning is in the plan.

Everything is drawn FROM THE SAVED PICKLE, so `--plots-only` redraws without
retraining.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

_BLUE = "#3B6EA5"
_ORNG = "#C4703A"
_GREY = "#D8DEE6"
_INK  = "#3C4653"
ARMS  = (("fw", "forward"), ("bw", "backward"))


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _converged(rec, arch):
    """Indices of repeats converged for this architecture, else all of them."""
    idx = [i for i, r in enumerate(rec["runs"]) if r[arch]["converged"]]
    return (idx, False) if idx else (list(range(len(rec["runs"]))), True)


# ══════════════════════════════════════════════════════════════════════════
# F1 — discrete
# ══════════════════════════════════════════════════════════════════════════
def f1_causal_states_discrete(rec, out_dir):
    spec, best = rec["spec"], rec["best"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    for row, (arm, name) in enumerate(ARMS):
        d   = rec["runs"][best]["discrete"][arm]
        occ = np.asarray(d["occupancy"])
        th  = np.asarray(spec[f"occ_{arm}"])

        # Occupancy, visited states only, sorted descending so it can be laid
        # against the theoretical occupancy (which is identified only up to a
        # permutation of the state labels).
        ax = axes[row][0]
        vis = np.flatnonzero(occ > 0)
        v   = np.sort(occ[vis])[::-1]
        ax.bar(np.arange(len(v)), v, width=0.62, color=_BLUE, label="learned")
        t = np.sort(th)[::-1]
        ax.plot(np.arange(len(t)), t, "o", ms=6, color=_ORNG, label="theory")
        ax.set_xticks(range(max(len(v), len(t))))
        ax.set_xlabel("state (sorted)"); ax.set_ylabel("occupancy")
        ax.set_title(f"{name}   S={d['S_emp']:.3f} / C={d['C']:.3f}", fontsize=10)
        if row == 0:
            ax.legend(frameon=False, fontsize=8)

        # Emission table over the visited states.
        ax = axes[row][1]
        em = np.asarray(d["emissions"])[vis]
        im = ax.imshow(em, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(vis))); ax.set_yticklabels([str(k) for k in vis], fontsize=7)
        ax.set_xticks(range(em.shape[1]))
        ax.set_xlabel("prev token" if arm == "bw" else "next token")
        ax.set_ylabel("state")
        ax.set_title(f"{name}   k={d['n_states_used']}/{d['n_states']} (true {d['true_k']})",
                     fontsize=10)
        fig.colorbar(im, ax=ax, pad=0.02).outline.set_visible(False)

    fig.suptitle(spec["tag"])
    return _save(fig, os.path.join(out_dir, "F1_causal_states_discrete.png"))


# ══════════════════════════════════════════════════════════════════════════
# F2 — clustered
# ══════════════════════════════════════════════════════════════════════════
def f2_causal_states_clustered(rec, out_dir):
    """
    The standard model's predictive distributions, PCA-projected, coloured by
    agglomerative cluster at the PLATEAU tolerance.

    PCA rather than UMAP: it is deterministic, needs no extra dependency, and on
    a V-token simplex the first two components are an exact picture for V=3.
    """
    spec, best = rec["spec"], rec["best"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    for ax, (arm, name) in zip(axes, ARMS):
        o = rec["runs"][best]["onehot"][arm]
        probs = o.get("_probs"); labels = o.get("_labels")
        if probs is None:
            ax.set_axis_off(); continue
        probs = np.asarray(probs); labels = np.asarray(labels)
        xy = (PCA(n_components=2).fit_transform(probs) if probs.shape[1] > 2
              else probs[:, :2])
        for c in np.unique(labels):
            msk = labels == c
            ax.scatter(xy[msk, 0], xy[msk, 1], s=12, alpha=0.75,
                       label=str(c), edgecolors="none")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}   k={o['plateau']} (true {o['true_k']})  "
                     f"S={o['S_hat_plateau']:.3f} / C={o['C']:.3f}", fontsize=10)
        ax.legend(frameon=False, fontsize=7, markerscale=1.2, loc="best")

    fig.suptitle(spec["tag"])
    return _save(fig, os.path.join(out_dir, "F2_causal_states_clustered.png"))


# ══════════════════════════════════════════════════════════════════════════
# F3 — transition matrix
# ══════════════════════════════════════════════════════════════════════════
def _heat(ax, M, ticks, title):
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    K = M.shape[0]
    fs = 8 if K <= 8 else 5
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=fs, color="white" if M[i, j] > 0.55 else _INK)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
    ax.set_xlabel("to"); ax.set_ylabel("from")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-.5, K, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    return im


def f3_transition_matrix(rec, out_dir):
    """
    Learned T[i][j] = P(s_{t+1}=j | s_t=i), visited states only, both arms, with
    the closed form beside the FORWARD panel under the matched permutation.

    Only forward carries a theory panel.  The backward epsilon-machine
    transition structure is not derived anywhere in this project, and a wrong
    overlay is worse than none.
    """
    from extraction import match_permutation
    spec = rec["spec"]
    T_fw, T_bw = np.asarray(rec["transition"]["fw"]), np.asarray(rec["transition"]["bw"])
    theory = np.asarray(spec["T_theory_fw"])
    matched = match_permutation(T_fw, theory)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    if matched is not None:
        sub, order, err = matched
        _heat(axes[0], sub, order, f"forward (learned)")
        _heat(axes[1], theory, order, f"forward (theory)   max err {err:.3f}")
    else:
        vis = np.flatnonzero(T_fw.sum(1) > 0)
        _heat(axes[0], T_fw[np.ix_(vis, vis)], vis, "forward (learned)")
        _heat(axes[1], theory, range(theory.shape[0]),
              f"forward (theory)   {len(vis)} states found, {theory.shape[0]} expected")
    vis_b = np.flatnonzero(T_bw.sum(1) > 0)
    if vis_b.size == 0:
        vis_b = np.arange(T_bw.shape[0])
    im = _heat(axes[2], T_bw[np.ix_(vis_b, vis_b)], vis_b, "backward (learned)")
    fig.colorbar(im, ax=axes, pad=0.02).outline.set_visible(False)

    fig.suptitle(spec["tag"])
    return _save(fig, os.path.join(out_dir, "F3_transition_matrix.png"))


# ══════════════════════════════════════════════════════════════════════════
# F4 — complexity
# ══════════════════════════════════════════════════════════════════════════
def _estimates(rec, arm):
    """(theory, (discrete mean, sd, n), (clustered mean, sd, n), fell_back)."""
    di, fb_d = _converged(rec, "discrete")
    oi, fb_o = _converged(rec, "onehot")
    d = np.array([rec["runs"][i]["discrete"][arm]["S_emp"] for i in di])
    o = np.array([rec["runs"][i]["onehot"][arm]["S_hat_plateau"] for i in oi])
    C = rec["spec"]["C_plus"] if arm == "fw" else rec["spec"]["C_minus"]
    return C, (d.mean(), d.std(), len(d)), (o.mean(), o.std(), len(o)), (fb_d or fb_o)


def f4_complexity(rec, out_dir):
    """
    Theory vs the two extractors, per arm, over CONVERGED repeats.

    The figure the method comparison extends: a new extraction method is one
    more bar against the same theory.
    """
    spec = rec["spec"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for ax, (arm, name) in zip(axes, ARMS):
        C, (dm, ds, dn), (om, os_, on), fb = _estimates(rec, arm)
        ax.bar([0, 1, 2], [C, dm, om], width=0.6,
               color=[_GREY, _BLUE, _ORNG],
               yerr=[0, ds, os_], capsize=4, ecolor=_INK)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["theory", f"discrete\nn={dn}", f"clustered\nn={on}"],
                           fontsize=9)
        ax.set_ylabel("bits")
        ax.set_title(f"{name}{'  [no converged repeat]' if fb else ''}", fontsize=10)
    fig.suptitle(spec["tag"])
    return _save(fig, os.path.join(out_dir, "F4_complexity.png"))


def draw_all(rec, out_dir) -> list:
    return [f(rec, out_dir) for f in (f1_causal_states_discrete,
                                      f2_causal_states_clustered,
                                      f3_transition_matrix,
                                      f4_complexity)]


# ══════════════════════════════════════════════════════════════════════════
# AGGREGATES
# ══════════════════════════════════════════════════════════════════════════
def _scatter(ax, recs, arm, name):
    xs_t, ys_d, ys_o = [], [], []
    for rec in recs:
        C, (dm, _, _), (om, _, _), _ = _estimates(rec, arm)
        xs_t.append(C); ys_d.append(dm); ys_o.append(om)
    lo = min(xs_t + ys_d + ys_o + [0]); hi = max(xs_t + ys_d + ys_o) * 1.05
    ax.plot([lo, hi], [lo, hi], "-", lw=1, color=_GREY)
    ax.scatter(xs_t, ys_d, s=34, color=_BLUE, label="discrete", edgecolors="none")
    ax.scatter(xs_t, ys_o, s=34, color=_ORNG, label="clustered", marker="^",
               edgecolors="none")
    ax.set_xlabel("theory"); ax.set_ylabel("empirical")
    ax.set_title(name, fontsize=10)
    ax.legend(frameon=False, fontsize=8)


def draw_grid(grid, out_root) -> str:
    recs = list(grid["cells"].values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, (arm, name) in zip(axes, ARMS):
        _scatter(ax, recs, arm, name)
    fig.suptitle(f"{os.path.basename(out_root)}  ({len(recs)} cells)")
    return _save(fig, os.path.join(out_root, "F4_complexity_grid.png"))


def draw_pooled(recs, out_root, fname="F4_all_processes.png") -> str:
    """Every process on one theory-vs-empirical plot -- the cross-architecture view."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, (arm, name) in zip(axes, ARMS):
        _scatter(ax, recs, arm, name)
    fig.suptitle(f"{len(recs)} processes")
    return _save(fig, os.path.join(out_root, fname))
