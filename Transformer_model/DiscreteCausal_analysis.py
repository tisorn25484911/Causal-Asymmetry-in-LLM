"""
Reading causal states off a trained model, and the figures for them.

Every function here returns None (or draws nothing) for a model without
explicit states, so the runners call them unconditionally and the single branch
between the two architectures lives here -- MODULAR_MODELS_PLAN.md 3 and 7.1.

The figures deliberately keep the presentation of
Jupyter_notebooks/test_new training method.ipynb, which is the form the states
are legible in: occupancy with the count written on each bar, the state scatter
with the count written on each point, and the transition-probability heatmap
with the value printed in each cell beside the closed form.
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Flower_process_generation import FlowerDataset, flower_process_generation
from Data_generation import CoinDataset, coin_generation
# Notebook palette.  One sequential hue for magnitude (occupancy is a count, so
# sequential, not categorical), a grey for unused states, and text in ink
# rather than in the series colour.
_BLUE = "#3B6EA5"
_GREY = "#D8DEE6"
_EDGE = "#B4BCC7"
_INK  = "#3C4653"


def causal_state_report(model, data_loader, min_pos: int = 5, device=None) -> dict | None:
    """
    Assign every token position to a causal state and summarise.

    Returns None for a model whose states are not explicit, which is how the
    runners stay free of `if embed_type ==` branches.

    `min_pos` drops the short-context end of each sequence, and WHICH end that
    is depends on the arm: a forward model at position t has only t tokens of
    context, so its short end is the START; a backward model under a triu mask
    attends to [t, T-1], so its short end is the FINISH.  Trimming the wrong end
    compares a settled arm against an unsettled one.

    The loader yields (inputs, targets) in forward convention.  A backward model
    is TRAINED on the swap, so it must be scored on the swap too -- otherwise it
    is evaluated on a tensor it never saw in that alignment.
    """
    if getattr(model, "state_assignment", None) is None:
        return None
    if model.state_assignment() is None and not hasattr(model, "n_states"):
        return None

    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    K = int(getattr(model, "n_states", 0))
    if K <= 0:
        return None

    V = int(getattr(model, "token_size", 0))
    counts = torch.zeros(K, dtype=torch.long)
    # Joint (token, state) counts, for H(state | token) below.
    joint = torch.zeros(max(V, 1), K, dtype=torch.long)
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            if getattr(model, "mode", "forward") == "backward":
                _, inputs = batch                      # mirror training_step
            model(inputs.to(device))
            st = model.last_states
            T = st.shape[1]
            keep = (slice(min_pos, None) if model.mode == "forward"
                    else slice(0, max(T - min_pos, 1)))
            st = st[:, keep]
            counts += torch.bincount(st.reshape(-1).cpu(), minlength=K)
            if V:
                tk = inputs[:, keep].reshape(-1).cpu()
                joint += torch.bincount(tk * K + st.reshape(-1).cpu(),
                                        minlength=V * K).reshape(V, K)

        vectors = model.state_matrix.detach().cpu()
        emissions = model.emission_table().detach().cpu()

    if was_training:
        model.train()

    p = counts.numpy().astype(float)
    p = p / p.sum() if p.sum() > 0 else p
    nz = p[p > 0]
    s_emp = float(-(nz * np.log2(nz)).sum()) if nz.size else 0.0

    # H(state | token), in bits.
    #
    # For every process in this repository the true causal state is a
    # DETERMINISTIC function of the current token, in both directions -- coin
    # forward {0,2}->s0, {1}->s1; flower forward "which die", backward "which
    # outcome".  So the truth scores exactly 0, and any positive value means the
    # bottleneck is keying on context it does not need.
    #
    # Its value is that it needs NO ground truth: it is computed from the model's
    # own assignment and the input tokens.  Measured, it is 0.000 in every run
    # that recovered its state set and 0.11-0.48 in every run that merged states,
    # which makes it the one available unsupervised warning.
    j = joint.numpy().astype(float)
    tot = j.sum()
    h_cond = 0.0
    if tot > 0:
        for row in j:
            n_t = row.sum()
            if n_t <= 0:
                continue
            q = row[row > 0] / n_t
            h_cond += (n_t / tot) * float(-(q * np.log2(q)).sum())

    return {
        "counts":    counts.numpy(),
        "occupancy": p,
        "vectors":   vectors.numpy(),
        "emissions": emissions.numpy(),
        "occupied":  [k for k in range(K) if counts[k] > 0],
        "n_states":  K,
        "S_emp":     s_emp,
        "h_state_given_token": h_cond,
        "joint_token_state":   joint.numpy(),
        "mode":      getattr(model, "mode", "forward"),
        "usage_beta": float(getattr(model, "usage_beta", 0.0)),
    }


def s_emp_from_states(report) -> float | None:
    """S_emp = entropy of the state-occupancy distribution, in bits.

    Exact: no clustering, no state_tol, no silhouette.  Measured on the coin
    forward arm it lands within 0.001 bits of C+.

    Carries a usage_beta bias when beta > 0, and the bias is ARM-DEPENDENT --
    measured forward -0.001, backward +0.024.  Re-run with --usage-beta 0 to see
    the unbiased number.
    """
    return None if report is None else report["S_emp"]


def plot_causal_states(report, out_dir: str, tag: str, theory=None,
                       arm_label: str = "") -> str | None:
    """
    The three notebook-style panels, one figure per arm.

      1. occupancy       -- count written on each bar, unused states greyed
      2. state vectors   -- count written on each point, hollow if unused
      3. P(next | state) -- value printed in each cell, beside the closed form

    `theory` is an optional (rows, V) array of closed-form conditionals; when
    given, each learned row is matched to its nearest row and the total-variation
    distance is printed, exactly as the notebook reports learned-vs-theory.
    """
    if report is None:
        return None

    cnt = report["counts"].astype(float)
    K, V = report["emissions"].shape
    occ = report["occupied"]
    back = report["mode"] == "backward"
    tgt_word = "previous" if back else "next"

    # constrained_layout, not a hand-tuned wspace: panel 3's y-labels and
    # panel 2's colorbar both have data-dependent widths, so any fixed spacing
    # collides on some inputs.  Verified by rendering, not by reading.
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.25]))

    # ---- 1. occupancy -----------------------------------------------------
    ax = axes[0]
    ids = np.arange(K)
    ax.bar(ids, cnt, width=0.62,
           color=[_BLUE if c > 0 else _GREY for c in cnt])
    top = cnt.max() if cnt.max() > 0 else 1.0
    for k, c in zip(ids, cnt):
        ax.text(k, c + top * 0.02, f"{int(c):,}", ha="center", va="bottom",
                fontsize=8, color=_INK)
    ax.set_xlabel("Causal state")
    ax.set_ylabel("Token positions assigned")
    ax.set_title(f"Occupancy — {len(occ)} of {K} states used")
    ax.set_xticks(ids)
    ax.set_ylim(0, top * 1.14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    # ---- 2. state vectors -------------------------------------------------
    ax = axes[1]
    vec = report["vectors"]
    if min(vec.shape) >= 2:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=0).fit_transform(vec)
        lab = "PCA component"
    else:
        coords = np.column_stack([vec[:, 0], np.zeros(K)])
        lab = "state vector"
    live = cnt > 0
    ax.scatter(coords[~live, 0], coords[~live, 1], s=80, facecolor="none",
               edgecolor=_EDGE, linewidth=1.5, zorder=2)
    if live.any():
        sc = ax.scatter(coords[live, 0], coords[live, 1],
                        s=140 + 700 * cnt[live] / cnt.max(),
                        c=cnt[live], cmap="Blues", vmin=0,
                        edgecolor="white", linewidth=2.0, zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Token positions")
        cb.outline.set_visible(False)
    for k in range(K):
        txt = f"s{k}\n{int(cnt[k]):,}" if cnt[k] else f"s{k}\nunused"
        ax.annotate(txt, (coords[k, 0], coords[k, 1]), textcoords="offset points",
                    xytext=(0, -26), ha="center", fontsize=8, color=_INK)
    ax.set_xlabel(f"{lab} 1")
    ax.set_ylabel(f"{lab} 2")
    ax.set_title("State vectors")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(0.24)

    # ---- 3. transition probabilities --------------------------------------
    ax = axes[2]
    em = report["emissions"]
    rows = occ if occ else list(range(K))
    im = ax.imshow(em[rows], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for r, k in enumerate(rows):
        for t in range(V):
            v = em[k, t]
            ax.text(t, r, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 else _INK)
    ylab = []
    for k in rows:
        lab = f"s{k}  n={int(cnt[k]):,}"
        if theory is not None:
            th = np.atleast_2d(np.asarray(theory, dtype=float))
            tv = 0.5 * np.abs(th - em[k]).sum(axis=1)
            lab += f"\nTV={tv.min():.3f}"      # second line, not a wider label
        ylab.append(lab)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylab, fontsize=8)
    ax.set_xticks(range(V))
    ax.set_xlabel(f"{tgt_word.capitalize()} token")
    ax.set_title(f"P({tgt_word} token | state)")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Probability")
    cb.outline.set_visible(False)
    ax.set_xticks(np.arange(-.5, V, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)

    fig.text(0.5, -0.04,
             "state-vector geometry is identified only up to an invertible "
             "K x K map (plan 4.4); the emission table is the identified view",
             ha="center", fontsize=8, color=_INK)

    head = f"{tag} — {report['mode']}"
    if arm_label:
        head += f"  [{arm_label}]"
    head += f"   S_emp = {report['S_emp']:.4f} bits   (K={K}, beta={report['usage_beta']})"
    fig.suptitle(head, y=1.03)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_causal_states_{report['mode']}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved -> {path}")
    return path


def print_state_summary(report, theory=None, label: str = ""):
    """
    The notebook's learned-vs-theory readout, in the repo's summary style.
    """
    if report is None:
        return
    em = report["emissions"]
    cnt = report["counts"]
    print(f"    {label}{report['mode']}: {len(report['occupied'])}/{report['n_states']} "
          f"states used   S_emp = {report['S_emp']:.4f} bits")
    for k in report["occupied"]:
        line = f"      s{k}  n={int(cnt[k]):>9,d}  P({'prev' if report['mode']=='backward' else 'next'}) = " \
               f"{np.round(em[k], 4).tolist()}"
        if theory is not None:
            th = np.atleast_2d(np.asarray(theory, dtype=float))
            tv = 0.5 * np.abs(th - em[k]).sum(axis=1)
            j = int(tv.argmin())
            line += f"   ~ theory row {j}  TV={tv[j]:.4f}"
        print(line)


def plot_causal_states_pair(report_fw, report_bw, out_dir: str, tag: str,
                            theory_fw=None, theory_bw=None) -> list:
    """
    The PAIRED forward-vs-backward views, as the notebook draws them.

    Three wide figures, each one row of two panels:

        <tag>_causal_occupancy.png          occupancy, fw | bw
        <tag>_causal_state_vectors.png      state vectors, fw | bw
        <tag>_causal_transition_probs.png   P(next|state) | P(prev|state)

    The per-arm three-panel figure stays as the detail view; this is the
    comparison view, and the comparison is the point -- the coin's forward arm
    needs 2 states and its backward arm 3, which is why C- > C+.

    Returns nothing (an empty list) when either arm has no explicit states, so
    the runners call it unconditionally.
    """
    if report_fw is None or report_bw is None:
        return []

    arms = (("forward", report_fw, theory_fw), ("backward", report_bw, theory_bw))
    paths = []
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. occupancy, shared y so the two arms are directly comparable -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True,
                             constrained_layout=True)
    top = max(r["counts"].max() for _, r, _ in arms) or 1
    for ax, (name, rep, _) in zip(axes, arms):
        cnt = rep["counts"].astype(float)
        ids = np.arange(rep["n_states"])
        ax.bar(ids, cnt, width=0.62,
               color=[_BLUE if c > 0 else _GREY for c in cnt])
        for k, c in zip(ids, cnt):
            ax.text(k, c + top * 0.02, f"{int(c):,}", ha="center", va="bottom",
                    fontsize=8, color=_INK)
        ax.set_xlabel("Causal state")
        ax.set_title(f"{name} — {len(rep['occupied'])} of {rep['n_states']} used"
                     f"   S_emp={rep['S_emp']:.4f}")
        ax.set_xticks(ids)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Token positions assigned")
    axes[0].set_ylim(0, top * 1.14)
    fig.suptitle(f"{tag} — causal-state occupancy, forward vs backward")
    p = os.path.join(out_dir, f"{tag}_causal_occupancy.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # ---- 2. state vectors ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), constrained_layout=True)
    for ax, (name, rep, _) in zip(axes, arms):
        cnt = rep["counts"].astype(float)
        vec = rep["vectors"]
        if min(vec.shape) >= 2:
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2, random_state=0).fit_transform(vec)
            lab = "PCA component"
        else:
            coords = np.column_stack([vec[:, 0], np.zeros(len(cnt))])
            lab = "state vector"
        live = cnt > 0
        ax.scatter(coords[~live, 0], coords[~live, 1], s=90, facecolor="none",
                   edgecolor=_EDGE, linewidth=1.6, zorder=2)
        if live.any():
            sc = ax.scatter(coords[live, 0], coords[live, 1],
                            s=180 + 900 * cnt[live] / cnt.max(),
                            c=cnt[live], cmap="Blues", vmin=0,
                            edgecolor="white", linewidth=2.0, zorder=3)
            cb = fig.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label("Token positions"); cb.outline.set_visible(False)
        for k in range(len(cnt)):
            txt = f"state {k}\n{int(cnt[k]):,}" if cnt[k] else f"state {k}\nunused"
            ax.annotate(txt, (coords[k, 0], coords[k, 1]),
                        textcoords="offset points", xytext=(0, -32),
                        ha="center", fontsize=8, color=_INK)
        ax.set_xlabel(f"{lab} 1"); ax.set_ylabel(f"{lab} 2")
        ax.set_title(f"{name}  (hollow = unused)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True); ax.margins(0.24)
    fig.suptitle(f"{tag} — learned causal-state vectors, forward vs backward")
    fig.text(0.5, -0.03,
             "the two panels are NOT in a common frame: each arm's geometry is "
             "identified only up to its own invertible K x K map",
             ha="center", fontsize=8, color=_INK)
    p = os.path.join(out_dir, f"{tag}_causal_state_vectors.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # ---- 3. transition probabilities ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    for ax, (name, rep, th) in zip(axes, arms):
        em = rep["emissions"]; cnt = rep["counts"]
        rows = rep["occupied"] or list(range(rep["n_states"]))
        V = em.shape[1]
        im = ax.imshow(em[rows], cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for r, k in enumerate(rows):
            for t in range(V):
                v = em[k, t]
                ax.text(t, r, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 else _INK)
        ylab = []
        for k in rows:
            lb = f"state {k}  (n={int(cnt[k]):,})"
            if th is not None:
                tha = np.atleast_2d(np.asarray(th, dtype=float))
                tv = 0.5 * np.abs(tha - em[k]).sum(axis=1)
                lb += f"\nTV={tv.min():.3f}"
            ylab.append(lb)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(ylab, fontsize=8)
        ax.set_xticks(range(V))
        word = "prev" if name == "backward" else "next"
        ax.set_xlabel(f"{'Previous' if name=='backward' else 'Next'} token")
        ax.set_title(f"{name}:  P({word} | state)")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("Probability"); cb.outline.set_visible(False)
        ax.set_xticks(np.arange(-.5, V, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle(f"{tag} — transition probabilities, forward vs backward")
    p = os.path.join(out_dir, f"{tag}_causal_transition_probs.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    for p in paths:
        print(f"  figure saved -> {p}")
    return paths

def transition_matrix_extraction(model, process, n_p = None, m_q = None , burn_in = 100, total_run = 500, max_batches = None, window_size = None,
                                 dice_probs = None):
    """
    Empirical state-to-state transition matrix, read off the model's own
    free-running generation:  T[i][j] = P(s_{t+1} = j | s_t = i).

    The model is seeded with a burn-in drawn from the TRUE process, then rolls
    forward on its own samples.  At each step the state the bottleneck lands in
    at the newest position is recorded, and the s_i -> s_j counts are
    row-normalised.  This is the state-transition analogue of the emission
    heatmap above, and like `emission_table` it is only identified up to a
    PERMUTATION of the state labels -- which index the bottleneck assigns to
    which causal state is arbitrary, so compare against a closed form only after
    matching the labels up.

    `dice_probs` is the Flower dice realisation to seed from; pass the one in
    the run's spec.

    `window_size` is the context the model rolls forward on.  Pass the TRAINING
    chunk length: see the "analysis" note in configs.py -- the model only ever
    saw positional indices [0, chunk), so a longer window measures
    extrapolation rather than what was learned, and does so ASYMMETRICALLY
    between the two arms.  It falls back to max_len for a standalone model with
    no runner to ask.

    `n_p` / `m_q` are the process parameters -- (p, q) for Coin, (n, m) for
    Flower.  They are deliberately not named `np`/`mq`: a parameter called `np`
    shadows the numpy import for the whole body, which is what stopped this
    function from running at all.
    """
    # Data generation.  `dice_probs` MUST be the realisation the model was
    # trained on -- leaving it None re-draws a fresh random set of dice, so the
    # burn-in would come from a different process than the one being inspected.
    samples = 1
    pre_depth = 0
    if process == "Coin":
        data, states = coin_generation(samples, burn_in, n_p, m_q)
        NUM_TOKEN = 3
    elif process == "Flower":
        data, states = flower_process_generation(samples, burn_in, pre_depth, n_p, m_q, dice_probs)
        NUM_TOKEN = n_p + m_q
    else:
        raise ValueError(f"unknown process {process!r} -- expected 'Coin' or 'Flower'")

    # Recordings of transitioned states
    transition = np.zeros((model.n_states, model.n_states))

    # Model next token prediction
    model.eval()
    device = next(model.parameters()).device
    is_bw  = (getattr(model, "mode", "forward") == "backward")
    inputs_ini = torch.tensor(data, dtype=torch.long, device=device)

    pred_seq = []
    # d_model is the embedding WIDTH; the original `window_size = model.d_model`
    # silently cut the context to 32 tokens.
    window_size = model.max_len if window_size is None else int(window_size)
    window_size = min(window_size, model.max_len)
    # Against the seed LENGTH, not `burn_in`: Flower counts burn_in in cycles
    # and emits two tokens per cycle, so comparing burn_in warns wrongly.
    if inputs_ini.shape[1] < window_size: print("add more burn_in")
    pos = 0 if is_bw else -1
    inputs = inputs_ini[:, -window_size:]
    causal_idx_prev = None
    for i in range(total_run):
        if is_bw:
            inputs = inputs[:, :window_size]
        else:
            inputs = inputs[:, -window_size:]
        with torch.no_grad():
            logits = model(inputs)
        causal_state = model.last_states          # (B,T), already the argmax
        causal_idx = int(causal_state[0, pos])
        if causal_idx_prev is not None:
            transition[causal_idx_prev][causal_idx] += 1
        causal_idx_prev = causal_idx

        prob = torch.softmax(logits[0, pos], dim = -1).cpu().numpy().astype(np.float64)
        next_token = int(np.random.choice(NUM_TOKEN, p=prob / prob.sum()))
        nxt = torch.tensor([[next_token]], dtype=torch.long, device=device)
        if is_bw:
            inputs = torch.cat((nxt, inputs), dim=1)
            pred_seq = [next_token] + pred_seq
        else:
            inputs = torch.cat((inputs, nxt), dim=1)
            pred_seq = pred_seq + [next_token]

    # Row-normalise: T is P(s_j | s_i), so each row sums to 1.  Dividing the
    # whole table by one scalar count left it un-normalised.  Unvisited states
    # keep a zero row rather than raising.
    row = transition.sum(axis=1, keepdims=True)
    transition = np.divide(transition, row, out=np.zeros_like(transition), where=row > 0)
    return transition


def plot_transition_matrix_pair(T_fw, T_bw, out_dir: str, tag: str,
                                counts_fw=None, counts_bw=None) -> str | None:
    """
    The forward and backward state-transition matrices as one heatmap pair.

    Same presentation as the emission heatmap above -- Blues on [0, 1] with the
    value printed in each cell -- because both are row-stochastic tables over
    states and reading them side by side is the point.

    Only the states the model actually VISITED are drawn, as in the emission
    heatmap: K is the bottleneck's capacity, not its occupancy, and at K=15
    with 2 states used the full grid is 13 empty rows and 225 printed numbers.
    Ticks carry the real state index, so the picture stays traceable back to
    the occupancy plots, and the sub-matrix is still row-stochastic because a
    state that is never entered is never left either.

    Returns the path written, or None when either arm is missing.
    """
    if T_fw is None or T_bw is None:
        return None
    os.makedirs(out_dir, exist_ok=True)

    arms = (("forward", T_fw, counts_fw), ("backward", T_bw, counts_bw))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, (name, T, cnt) in zip(axes, arms):
        # Visited = entered or left.  Same index set on both axes, so what is
        # drawn is still a square transition matrix.
        seen = (T.sum(axis=1) > 0) | (T.sum(axis=0) > 0)
        keep = np.flatnonzero(seen)
        if keep.size == 0:
            keep = np.arange(T.shape[0])
        sub = T[np.ix_(keep, keep)]
        K = keep.size
        im = ax.imshow(sub, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        # The cell value, in white on the dark end so it stays readable.
        fs = 8 if K <= 8 else 5
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center",
                        fontsize=fs, color="white" if sub[i, j] > 0.55 else _INK)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([str(k) for k in keep])
        ax.set_yticklabels([str(k) for k in keep])
        ax.set_xlabel("to"); ax.set_ylabel("from")
        ax.set_title(name)
        # Thin white gridlines between cells, as in the emission heatmap.
        ax.set_xticks(np.arange(-.5, K, 1), minor=True)
        ax.set_yticks(np.arange(-.5, K, 1), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=axes, pad=0.02)
    cb.outline.set_visible(False)
    fig.suptitle(tag)

    p = os.path.join(out_dir, f"{tag}_state_transition.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p
