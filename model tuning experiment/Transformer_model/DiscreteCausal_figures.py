"""
Cross-experiment figures for the discrete decoder — DISCRETE_V2_PLAN.md section 5.

F0   calibration        S_emp vs C                       00_base
F1   K vs k_discovered  the headline                     01_ksweep
F1b  K vs S_emp         the calibration view of F1       01_ksweep
F2   state_dim          ARI-free: k deficit vs S         02_statedim
F3   sequence length    with beta printed per tick       03_seqlen
F4   beta vs collapse   two series, the control          03b_batch + 03_seqlen
F5   capacity           signed k deficit                 04_capacity
F6   budget             ARI/CE twin axis, early stop     05_budget
F7   beta ratio         x as multiples of 1/N            06_beta_verify
F8   seed spread        the noise floor                  07_seeds
F12  recovery rate      gates F9 panel 4                 08/09

F9 (four-panel scatter vs C- - C+), F10 (trajectory overlay) and F11 (per-process
loss curves) are NOT here: they already exist in run_sweep_experiment.py and
run_statistical_trj.py, and duplicating them would give two figures that drift.

House style, applied throughout
-------------------------------
  * each variable encoded ONCE -- no bar height plus printed number plus colorbar
  * no colorbar where values are printed or the scale is fixed
  * reference lines annotated IN PLACE, never in the legend
  * legend only at >= 5 series; below that the last point is labelled
  * axis labels are units, not sentences;  titles are identities, not summaries

Every function takes rows already loaded from a pickle and returns the saved
path, or None when there is nothing to draw.  None of them train anything.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Two hues, one per arm, plus grey for references and unused states.
FW, BW, REF, INK = "#3B6EA5", "#C1663A", "#9AA3AE", "#3C4653"
_ARM = {"fw": FW, "bw": BW, "forward": FW, "backward": BW}


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved -> {p}")
    return p


def _bare(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def _rule(ax, y, label, axis="h"):
    """A reference line annotated at its right edge -- never in a legend."""
    (ax.axhline if axis == "h" else ax.axvline)(y, color=REF, ls="--", lw=0.9)
    if axis == "h":
        ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7, color=REF)
    else:
        ax.annotate(label, xy=(y, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color=REF)


# ══════════════════════════════════════════════════════════════════════════
# F0 — calibration:  S_emp against the closed form
# ══════════════════════════════════════════════════════════════════════════
def f0_calibration(arms, out_dir, fname="F0_calibration.png"):
    """
    arms: [{tag, arm, C, S_emp, k_found, true_k}, ...] -- one per (process, arm).

    Hollow marker when the state set was NOT recovered.  That is the point of
    the figure: S_emp is only calibrated where k_found == true_k, and measured
    it is +0.005 there and -0.17..-1.04 where states merged.  With no hollow
    marker the reader cannot tell which regime a point is in.
    """
    if not arms:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo = min(min(a["C"] for a in arms), min(a["S_emp"] for a in arms))
    hi = max(max(a["C"] for a in arms), max(a["S_emp"] for a in arms))
    pad = 0.06 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=REF, lw=0.9, ls="--")
    ax.annotate("y = x", xy=(hi, hi), xytext=(-4, 6), textcoords="offset points",
                fontsize=7, color=REF, ha="right")
    for a in arms:
        ok = a.get("k_found") == a.get("true_k")
        ax.plot(a["C"], a["S_emp"], marker="o" if a["arm"] in ("fw", "forward") else "s",
                ms=7, mfc=_ARM[a["arm"]] if ok else "none",
                mec=_ARM[a["arm"]], mew=1.4, ls="none")
    # Label only the worst few, so the plot stays readable.
    worst = sorted(arms, key=lambda a: -abs(a["S_emp"] - a["C"]))[:4]
    for a in worst:
        ax.annotate(f"{a['tag']} {a['arm']}", (a["C"], a["S_emp"]),
                    xytext=(5, -3), textcoords="offset points",
                    fontsize=7, color=INK)
    ax.set_xlabel("C (bits)")
    ax.set_ylabel("S_emp (bits)")
    ax.set_title(f"S_emp vs closed form · {len(arms)} arms", fontsize=10)
    ax.annotate("filled = state set recovered", xy=(0.03, 0.95),
                xycoords="axes fraction", fontsize=7, color=INK)
    _bare(ax)
    return _save(fig, out_dir, fname)


# ══════════════════════════════════════════════════════════════════════════
# F1 / F1b — K vs k_discovered, and K vs S_emp
# ══════════════════════════════════════════════════════════════════════════
def f1_k_vs_kfound(series, out_dir, fname="F1_K_vs_k_discovered.png",
                   value="k_found"):
    """
    series: {(tag, arm): {"K": [...], "y": [[per-seed], ...],
                          "true_k": int, "C": float}}

    One panel per (process, arm).  y = x is the budget ceiling, y = true_k the
    target; both annotated in place so the panels carry no legend.

    Band = min..max over seeds, and it is not decoration: measured, k_found
    flips between 2 and 4 across neighbouring K at one seed, so a mean line
    alone would invent a trend.
    """
    if not series:
        return None
    keys = sorted(series)
    ncol = 2
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.0 * nrow),
                             squeeze=False, constrained_layout=True)
    is_k = value == "k_found"
    for i, key in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        d = series[key]
        K = np.asarray(d["K"], dtype=float)
        Y = [np.asarray(v, dtype=float) for v in d["y"]]
        med = np.array([np.median(v) for v in Y])
        lo = np.array([np.min(v) for v in Y])
        hi = np.array([np.max(v) for v in Y])
        col = _ARM[key[1]]
        ax.fill_between(K, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(K, med, color=col, lw=1.4, marker="o", ms=3)
        if is_k:
            ax.plot(K, K, color=REF, lw=0.9, ls=":")
            _rule(ax, d["true_k"], "true")
            # Clip y to the informative range.  y = x on a log-x axis climbs to
            # K_max (60 here) while k never exceeds a small multiple of true_k,
            # so an unclipped panel puts every data point in its bottom fifth.
            # The budget line then leaves the top of the panel, which is the
            # correct reading: above the crossing K stops being the constraint.
            top = max(float(np.max(hi)), float(d["true_k"])) * 1.35 + 0.5
            ax.set_ylim(0, top)
            cross = K[K <= top]
            if cross.size:
                ax.annotate("budget", xy=(cross[-1], min(cross[-1], top)),
                            xytext=(3, -9), textcoords="offset points",
                            fontsize=7, color=REF)
        else:
            _rule(ax, d["C"], "C")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{key[0]} · {key[1]}", fontsize=9)
        ax.set_xlabel("K")
        ax.set_ylabel("k" if is_k else "S_emp (bits)")
        _bare(ax)
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    return _save(fig, out_dir, fname)


def f1b_k_vs_semp(series, out_dir, fname="F1b_K_vs_S_emp.png"):
    return f1_k_vs_kfound(series, out_dir, fname, value="S_emp")


# ══════════════════════════════════════════════════════════════════════════
# F2 / F5 — an axis against the SIGNED state-count deficit
# ══════════════════════════════════════════════════════════════════════════
def _deficit_panel(ax, x, curves, xlabel, logx=None):
    for (tag, arm), y in curves.items():
        ax.plot(x, y, color=_ARM[arm], lw=1.3, marker="o", ms=3, alpha=0.85)
        ax.annotate(f"{tag} {arm}", (x[-1], y[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=6.5, color=INK,
                    va="center")
    _rule(ax, 0.0, "exact")
    if logx:
        ax.set_xscale("log", base=logx)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("k − true k")
    _bare(ax)


def f2_statedim(x, curves, out_dir, fname="F2_state_dim.png"):
    """x: state_dim / V.  curves: {(tag,arm): [k - true_k per x]}."""
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    _deficit_panel(ax, np.asarray(x, float), curves, "state_dim / V")
    ax.set_title("State-vector dimension", fontsize=10)
    return _save(fig, out_dir, fname)


def f5_capacity(x, curves, out_dir, fname="F5_capacity.png"):
    """
    x: d_model.  Plots the signed deficit rather than ARI deliberately -- the
    question is whether capacity lifts the state COUNT on the arms needing >= 5
    states, and the deficit answers that directly.
    """
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    _deficit_panel(ax, np.asarray(x, float), curves, "d_model", logx=2)
    ax.set_title("Capacity", fontsize=10)
    return _save(fig, out_dir, fname)


# ══════════════════════════════════════════════════════════════════════════
# F3 — sequence length, with beta printed under each tick
# ══════════════════════════════════════════════════════════════════════════
def f3_seqlen(chunks, betas, collapse, deficit, out_dir, fname="F3_seqlen.png"):
    """
    Printing beta under the x tick is what makes this figure honest: chunk_len
    and beta move together by construction (beta = 1/(B*T)), so the x axis is
    confounded and the figure must say so rather than imply a clean length
    effect.
    """
    if not chunks:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.2), sharex=True,
                             constrained_layout=True)
    x = np.arange(len(chunks), dtype=float)
    axes[0].plot(x, collapse, color=INK, lw=1.4, marker="o", ms=4)
    axes[0].set_ylabel("collapse rate")
    axes[0].set_title("Sequence length (beta moves with it)", fontsize=10)
    _bare(axes[0])
    for (tag, arm), y in deficit.items():
        axes[1].plot(x, y, color=_ARM[arm], lw=1.3, marker="o", ms=3)
        axes[1].annotate(f"{tag} {arm}", (x[-1], y[-1]), xytext=(4, 0),
                         textcoords="offset points", fontsize=6.5, color=INK,
                         va="center")
    _rule(axes[1], 0.0, "exact")
    axes[1].set_ylabel("k − true k")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{c}\n{b:.1e}" for c, b in zip(chunks, betas)],
                            fontsize=7)
    axes[1].set_xlabel("chunk_len  /  beta")
    _bare(axes[1])
    return _save(fig, out_dir, fname)


# ══════════════════════════════════════════════════════════════════════════
# F4 — beta vs collapse rate: the control
# ══════════════════════════════════════════════════════════════════════════
def f4_beta_collapse(series, out_dir, cliff=6e-4, fname="F4_beta_collapse.png"):
    """
    series: {"vary T": (betas, rates, labels), "vary B": (...)}

    The plan's whole beta justification rests on this figure.  If the two series
    superimpose, beta is the entire story and one cliff location covers both
    axes; if they separate, T and B act independently and the base config needs
    a minimum B as well as a minimum N.
    """
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for name, (b, r, labs) in series.items():
        b = np.asarray(b, float); r = np.asarray(r, float)
        o = np.argsort(b)
        ax.plot(b[o], r[o], lw=1.4, marker="o", ms=4,
                color=FW if "T" in name else BW)
        ax.annotate(name, (b[o][-1], r[o][-1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=7, color=INK,
                    va="center")
        for bi, ri, la in zip(b, r, labs):
            ax.annotate(la, (bi, ri), xytext=(0, -10),
                        textcoords="offset points", fontsize=6,
                        ha="center", color=INK)
    _rule(ax, cliff, "cliff", axis="v")
    ax.set_xscale("log")
    ax.set_xlabel("beta")
    ax.set_ylabel("collapse rate")
    ax.set_title("beta vs collapse — context varied two ways", fontsize=10)
    _bare(ax)
    return _save(fig, out_dir, fname)


# ══════════════════════════════════════════════════════════════════════════
# F6 — budget: the early-stopping figure
# ══════════════════════════════════════════════════════════════════════════
def f6_budget(panels, out_dir, conv_tol=0.1, fname="F6_budget.png"):
    """
    panels: [{tag, arm, steps, quality, ce_minus_hinf}, ...]

    The twin axis IS the argument.  Measured, state recovery peaks and decays
    while CE keeps falling through H_inf, so a CE-based stopping rule stops at
    the worst point.  Showing one without the other would not make the case.

    The marker is the FIRST entry into the conv_tol band -- the proposed
    stopping rule, which lost a mean 0.046 of the peak over 10 pilot runs.
    """
    if not panels:
        return None
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False,
                             constrained_layout=True)
    for i, p in enumerate(panels):
        ax = axes[0][i]
        s = np.asarray(p["steps"], float)
        q = np.asarray(p["quality"], float)
        d = np.asarray(p["ce_minus_hinf"], float)
        ax.plot(s, q, color=_ARM[p["arm"]], lw=1.5)
        ax.set_xscale("log")
        ax.set_ylabel("recovery")
        ax.set_xlabel("steps")
        ax.set_title(f"{p['tag']} · {p['arm']}", fontsize=9)
        _bare(ax)
        ax2 = ax.twinx()
        ax2.plot(s, d, color=REF, lw=1.2, ls="--")
        ax2.axhspan(-conv_tol, conv_tol, color=REF, alpha=0.12, lw=0)
        ax2.axhline(0.0, color=REF, lw=0.8)
        ax2.set_ylabel("CE − H∞ (bits)", color=REF, fontsize=8)
        ax2.tick_params(axis="y", labelcolor=REF, labelsize=7)
        ax2.spines[["top"]].set_visible(False)
        inside = np.where(np.abs(d) < conv_tol)[0]
        if inside.size:
            j = int(inside[0])
            ax.axvline(s[j], color=INK, lw=0.8, ls=":")
            ax.annotate("stop", xy=(s[j], q[j]), xytext=(4, 6),
                        textcoords="offset points", fontsize=7, color=INK)
        if i == 0:
            ax.annotate("solid: recovery   dashed: CE − H∞", xy=(0.03, 0.05),
                        xycoords="axes fraction", fontsize=7, color=INK)
    return _save(fig, out_dir, fname)


# ══════════════════════════════════════════════════════════════════════════
# F7 — beta as a RATIO to 1/N
# ══════════════════════════════════════════════════════════════════════════
def f7_beta_verify(ratios, curves, out_dir, fname="F7_beta_verify.png"):
    """
    x is beta / (1/N), so the figure reads the same at any geometry.  The
    absolute value belongs in the caption, not on the axis.
    """
    if not curves:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 5.0), sharex=True,
                             constrained_layout=True)
    x = np.asarray(ratios, float)
    for (tag, arm), (q, k) in curves.items():
        axes[0].plot(x, q, color=_ARM[arm], lw=1.3, marker="o", ms=3)
        axes[1].plot(x, k, color=_ARM[arm], lw=1.3, marker="o", ms=3)
        axes[1].annotate(f"{tag} {arm}", (x[-1], k[-1]), xytext=(4, 0),
                         textcoords="offset points", fontsize=6.5, color=INK,
                         va="center")
    for ax in axes:
        _rule(ax, 1.0, "1/N", axis="v")
        ax.set_xscale("log", base=2)
        _bare(ax)
    axes[0].set_ylabel("recovery")
    axes[1].set_ylabel("k − true k")
    axes[1].set_xlabel("beta / (1/N)")
    axes[0].set_title("beta verification around 1/N", fontsize=10)
    return _save(fig, out_dir, fname)


# ══════════════════════════════════════════════════════════════════════════
# F8 — the noise floor
# ══════════════════════════════════════════════════════════════════════════
def f8_seed_spread(per_arm, out_dir, fname="F8_seed_spread.png"):
    """
    per_arm: {(tag, arm): [value per seed]}.

    This figure exists to be QUOTED IN EVERY OTHER CAPTION.  Without it none of
    F0-F7 has an error bar, and the measured spread already exceeds most of the
    differences reported so far.
    """
    if not per_arm:
        return None
    keys = sorted(per_arm)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.5 * len(keys) + 2), 3.6))
    widest = 0.0
    for i, k in enumerate(keys):
        v = np.asarray(per_arm[k], float)
        widest = max(widest, float(np.ptp(v)) if v.size else 0.0)
        ax.plot(np.full_like(v, i, dtype=float), v, ls="none", marker="o",
                ms=4, alpha=0.65, color=_ARM[k[1]])
        ax.plot([i - 0.2, i + 0.2], [np.median(v)] * 2, color=INK, lw=1.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{t}\n{a}" for t, a in keys], fontsize=6.5, rotation=0)
    ax.set_ylabel("recovery")
    ax.set_title(f"Seed spread · widest {widest:.3f}", fontsize=10)
    _bare(ax)
    return _save(fig, out_dir, fname)


# ══════════════════════════════════════════════════════════════════════════
# F12 — recovery rate, which gates F9's S_emp panel
# ══════════════════════════════════════════════════════════════════════════
def f12_recovery(gaps, rate_fw, rate_bw, out_dir, fname="F12_recovery.png"):
    """
    Small, but F9's S_emp panel is uninterpretable without it: it says which
    part of the C- - C+ range the S_emp measurement is entitled to speak about.
    """
    if not len(gaps):
        return None
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    g = np.asarray(gaps, float)
    o = np.argsort(g)
    for r, arm, lab in ((rate_fw, "fw", "fw"), (rate_bw, "bw", "bw")):
        r = np.asarray(r, float)
        ax.plot(g[o], r[o], color=_ARM[arm], lw=1.3, marker="o", ms=3)
        ax.annotate(lab, (g[o][-1], r[o][-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7, color=INK,
                    va="center")
    _rule(ax, 1.0, "recovered")
    ax.set_xlabel("C− − C+ (bits)")
    ax.set_ylabel("fraction k = true k")
    ax.set_title("State-set recovery rate", fontsize=10)
    _bare(ax)
    return _save(fig, out_dir, fname)
