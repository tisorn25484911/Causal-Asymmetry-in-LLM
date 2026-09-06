#!/usr/bin/env python3
"""
Figures for the tau-schedule test.

    F1  the answer            paired change in states recovered, vs const_1.0
    F2  levels                states recovered and S_emp, against true k and C
    F3  the mechanism         distinct states claimed as training proceeds
    F4  robustness            collapse rate, convergence rate, CE

Drawn FROM THE SAVED JSON, so this redraws without retraining.  The one cell
whose job was stopped before it wrote its JSON is read from its log instead and
is labelled PARTIAL wherever it appears -- never silently averaged in.

Colour.  Three categorical slots for the three tau-variant arms, in fixed order,
validated all-pairs for colour-vision deficiency.  const_1.0 is the REFERENCE,
not a fourth series, so it is drawn in de-emphasis grey and carries a hatch or a
dashed stroke as well -- identity is never colour alone.  Every arm is direct-
labelled, which is also what relieves the aqua slot's sub-3:1 contrast.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

# ── palette ──────────────────────────────────────────────────────────────
_GREY = "#8a8f98"     # de-emphasis: the reference arm
_S1   = "#2a78d6"     # categorical slot 1
_S2   = "#eb6834"     # slot 2
_S3   = "#1baf7a"     # slot 3
_INK  = "#0b0b0b"
_INK2 = "#52514e"
_RULE = "#d7d7d4"
_SURF = "#fcfcfb"

BASE = "const_1.0"
ARMS = [BASE, "anneal_5_0.5", "const_5.0", "anneal_0.5_5"]
COLOR = {BASE: _GREY, "anneal_5_0.5": _S1, "const_5.0": _S2, "anneal_0.5_5": _S3}
HATCH = {BASE: "///", "anneal_5_0.5": None, "const_5.0": None, "anneal_0.5_5": None}
DASH  = {BASE: (3, 2), "anneal_5_0.5": None, "const_5.0": None, "anneal_0.5_5": None}
LABEL = {BASE: "const 1.0  (baseline)", "anneal_5_0.5": "anneal 5 -> 0.5",
         "const_5.0": "const 5.0", "anneal_0.5_5": "anneal 0.5 -> 5"}


def _style(ax, ygrid=True):
    ax.set_facecolor(_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_RULE)
    ax.tick_params(colors=_INK2, labelsize=8, length=3, width=0.8)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=_RULE, lw=0.7)
        ax.xaxis.grid(False)


def _save(fig, name):
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_SURF)
    plt.close(fig)
    print(f"  {path}")
    return path


def _legend(fig, ncol=4, arms=None):
    handles = [Patch(facecolor=COLOR[a], edgecolor="white", hatch=HATCH[a],
                     label=LABEL[a]) for a in (arms or ARMS)]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=8.5, labelcolor=_INK2, bbox_to_anchor=(0.5, -0.075))


# ══════════════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════════════
_RX = re.compile(
    r"seed (\d+)\s+(\S+)\s+[\d.]+s\s+CE ([\d.]+) \(([+-][\d.]+)\)\s+"
    r"k (\d+)/(\d+)\s+S_emp ([-\d.]+) \(([+-][\d.]+)\)\s+"
    r"H\(s\|tok\) ([\d.]+)\s+claimed \[([^\]]*)\]")


def _from_log(path, tag, mode, true_k, C):
    """A cell whose runner was stopped before it wrote JSON.  Complete seeds only:
    a seed missing an arm cannot enter a PAIRED comparison."""
    by_seed = {}
    for line in open(path):
        m = _RX.search(line)
        if not m:
            continue
        s, arm, ce, dce, k, _tk, se, _dse, h, cl = m.groups()
        by_seed.setdefault(int(s), {})[arm] = dict(
            seed=int(s), arm=arm, ce=float(ce), ce_minus_H=float(dce),
            k_occ=int(k), S_emp=float(se), h_state_given_token=float(h),
            within_tol=abs(float(dce)) <= 0.10,
            states_claimed_quarters=[int(v) for v in cl.split(",") if v.strip()])
    full = sorted(s for s, v in by_seed.items() if len(v) == len(ARMS))
    if not full:
        return None
    return dict(tag=tag, mode=mode, true_k=true_k, C=C, partial=True,
                seeds=full, runs=[by_seed[s][a] for s in full for a in ARMS])


def load_cells():
    cells = []
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        d = json.load(open(p))
        fw = d["mode"] == "forward"
        cells.append(dict(tag=d["spec"]["tag"], mode=d["mode"], partial=False,
                          true_k=d["spec"]["true_k_fw" if fw else "true_k_bw"],
                          C=d["spec"]["C_plus" if fw else "C_minus"],
                          seeds=sorted({r["seed"] for r in d["runs"]}),
                          runs=d["runs"]))
    log = os.path.join(HERE, "logs", "n6m4_fw.log")
    if os.path.exists(log):
        c = _from_log(log, "flower_n6_m4", "forward", 7, 2.2925)
        if c and not any(x["tag"] == c["tag"] and x["mode"] == c["mode"] for x in cells):
            cells.append(c)
    order = {("flower_n2_m6", "backward"): 0, ("flower_n6_m4", "backward"): 1,
             ("flower_n2_m6", "forward"): 2, ("flower_n6_m4", "forward"): 3}
    cells.sort(key=lambda c: order.get((c["tag"], c["mode"]), 9))
    return cells


def _title(c):
    n = c["tag"].replace("flower_", "").replace("_", " ")
    return (f"{n}  {c['mode']}   true k = {c['true_k']}"
            + ("   [PARTIAL]" if c["partial"] else ""))


def _get(c, arm, key):
    return np.array([r[key] for r in c["runs"] if r["arm"] == arm], dtype=float)


def _paired(c, arm, key):
    """(arm value, baseline value) per seed -- same data, same init."""
    b = {r["seed"]: r for r in c["runs"] if r["arm"] == BASE}
    out = [(r[key], b[r["seed"]][key]) for r in c["runs"] if r["arm"] == arm]
    return np.array(out, dtype=float)


# ══════════════════════════════════════════════════════════════════════════
# F1 — the answer
# ══════════════════════════════════════════════════════════════════════════
def f1_paired_delta(cells):
    """Change in states recovered against the baseline, seed by paired seed.

    The comparison is PAIRED -- each point is one seed's arm run minus that same
    seed's baseline run, on the same realisation from the same initialisation --
    so the spread between cells and seeds cancels and what is left is tau.
    """
    variants = ARMS[1:]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.patch.set_facecolor(_SURF)

    deltas = {a: np.concatenate([_paired(c, a, "k_occ")[:, 0]
                                 - _paired(c, a, "k_occ")[:, 1] for c in cells])
              for a in variants}
    lo = int(min(d.min() for d in deltas.values()))
    hi = int(max(d.max() for d in deltas.values()))
    bins = list(range(lo, hi + 1))

    # ── left: how the paired change is DISTRIBUTED ───────────────────────
    ax = axes[0]
    _style(ax)
    w = 0.26
    for i, arm in enumerate(variants):
        cnt = [int((deltas[arm] == b).sum()) for b in bins]
        xs = np.arange(len(bins)) + (i - 1) * w
        ax.bar(xs, cnt, width=w * 0.88, color=COLOR[arm], edgecolor="white",
               lw=1.2, zorder=2, label=LABEL[arm])
        for x, n in zip(xs, cnt):
            if n:
                ax.text(x, n + 0.22, str(n), ha="center", va="bottom",
                        fontsize=7.5, color=_INK2)
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels([f"{b:+d}" if b else "0" for b in bins], fontsize=9)
    ax.set_xlabel("change in states recovered vs. baseline, per paired seed",
                  fontsize=9, color=_INK2)
    ax.set_ylabel("number of seeds", fontsize=9, color=_INK2)
    ax.set_ylim(0, max(max(int((deltas[a] == b).sum()) for b in bins)
                       for a in variants) * 1.62)
    ax.set_title(f"Pooled over {sum(len(c['seeds']) for c in cells)} paired seeds, "
                 f"{len(cells)} cells   ·   negative = fewer states than baseline",
                 fontsize=10, color=_INK, loc="left")

    for i, arm in enumerate(variants):
        d = deltas[arm]
        ax.text(0.015, 0.975 - 0.075 * i,
                f"{LABEL[arm]}:   mean {d.mean():+.2f}    "
                f"{int((d > 0).sum())} better · {int((d == 0).sum())} tied · "
                f"{int((d < 0).sum())} worse",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                color=COLOR[arm], fontweight="bold")

    # ── right: per cell ──────────────────────────────────────────────────
    ax = axes[1]
    _style(ax)
    x = np.arange(len(cells))
    for i, arm in enumerate(variants):
        m = [(_paired(c, arm, "k_occ")[:, 0] - _paired(c, arm, "k_occ")[:, 1]).mean()
             for c in cells]
        ax.bar(x + (i - 1) * w, m, width=w * 0.88, color=COLOR[arm],
               edgecolor="white", lw=1.2, zorder=2)
    ax.axhline(0, color=_INK, lw=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c['tag'].replace('flower_','')}\n{c['mode'][:2]}"
                        + ("*" if c["partial"] else "") for c in cells], fontsize=8)
    ax.set_ylabel("mean change in states recovered", fontsize=9, color=_INK2)
    ax.set_title("By cell   (* partial)", fontsize=10, color=_INK, loc="left")

    fig.suptitle("F1   Annealing tau high -> low loses causal states on every seed tested",
                 fontsize=12.5, color=_INK, x=0.007, ha="left")
    _legend(fig, ncol=3, arms=variants)
    fig.text(0.5, -0.10, "const 1.0 (baseline) is the zero reference, not a bar",
             ha="center", fontsize=8, color=_INK2)
    return _save(fig, "F1_paired_delta.png")


# ══════════════════════════════════════════════════════════════════════════
# F2 — levels
# ══════════════════════════════════════════════════════════════════════════
def f2_levels(cells):
    """Where each arm actually lands, against the two closed-form targets.

    States recovered against true k, and S_emp against C.  Both matter: a state
    count that rises while S_emp stays flat is duplication, not recovery.
    """
    fig, axes = plt.subplots(2, len(cells), figsize=(3.35 * len(cells), 6.4),
                             constrained_layout=True)
    fig.patch.set_facecolor(_SURF)
    rng = np.random.default_rng(1)
    x = np.arange(len(ARMS))

    for j, c in enumerate(cells):
        for row, (key, target, name) in enumerate(
                [("k_occ", c["true_k"], "states recovered"),
                 ("S_emp", c["C"], "S_emp  (bits)")]):
            ax = axes[row][j]
            _style(ax)
            for i, arm in enumerate(ARMS):
                v = _get(c, arm, key)
                ax.bar(i, v.mean(), width=0.62, color=COLOR[arm], hatch=HATCH[arm],
                       edgecolor="white", lw=1.4, zorder=2)
                ax.scatter(np.full(v.size, i) + rng.uniform(0.20, 0.33, v.size), v,
                           s=20, color=_INK, alpha=0.5, lw=0, zorder=4)
                ax.text(i - 0.06, v.mean(),
                        f"{v.mean():.2f}" if row else f"{v.mean():.1f}",
                        ha="center", va="bottom", fontsize=8, color=_INK,
                        zorder=5, fontweight="bold")
            ax.axhline(target, color=_INK, lw=1.3, ls=(0, (5, 2)), zorder=3)
            ax.text(len(ARMS) - 0.42, target,
                    f" true k = {target}" if row == 0 else f" C = {target:.3f}",
                    ha="right", va="bottom", fontsize=8, color=_INK)
            ax.set_xticks(x)
            ax.set_xticklabels(["1.0", "5→0.5", "5.0", "0.5→5"], fontsize=8)
            ax.set_xlim(-0.62, len(ARMS) - 0.22)
            ax.set_ylim(0, max(target, max(_get(c, a, key).max() for a in ARMS)) * 1.26)
            if j == 0:
                ax.set_ylabel(name, fontsize=9, color=_INK2)
            if row == 0:
                ax.set_title(_title(c), fontsize=9.5, color=_INK, loc="left")
            else:
                ax.set_xlabel("tau schedule", fontsize=8.5, color=_INK2)

    fig.suptitle("F2   No schedule reaches the closed form; the ceiling is not a "
                 "temperature problem", fontsize=12.5, color=_INK, x=0.004, ha="left")
    _legend(fig)
    return _save(fig, "F2_levels.png")


# ══════════════════════════════════════════════════════════════════════════
# F3 — the mechanism
# ══════════════════════════════════════════════════════════════════════════
def f3_claim_trajectory(cells):
    """Distinct states the model assigned during one epoch, at four points.

    This is the mechanism.  Under a high tau the straight-through gradient is a
    near-uniform average over all K state vectors, nearly independent of which
    state the argmax picked, so the state head never learns to discriminate and
    the assignment stays where initialisation put it: the 5 -> 0.5 trace is flat
    from the start rather than climbing and then collapsing.
    """
    fig, axes = plt.subplots(1, len(cells), figsize=(3.3 * len(cells), 3.9),
                             constrained_layout=True, sharey=True)
    fig.patch.set_facecolor(_SURF)
    q = np.array([25, 50, 75, 100])

    for j, c in enumerate(cells):
        ax = axes[j]
        _style(ax)
        # Coincident integer traces: dodge each arm by a fixed factor in log
        # space so an arm hidden under another is still visible.  Values are
        # read from the end labels, which are NOT dodged.
        dodge = dict(zip(ARMS, [0.955, 0.985, 1.015, 1.046]))
        for arm in ARMS:
            tr = np.array([r["states_claimed_quarters"] for r in c["runs"]
                           if r["arm"] == arm], dtype=float)
            if tr.size == 0:
                continue
            f = dodge[arm]
            for row in tr:                       # every seed, faint
                ax.plot(q, row * f, color=COLOR[arm], lw=0.9, alpha=0.20, zorder=2)
            med = np.median(tr, axis=0)
            ax.plot(q, med * f, color=COLOR[arm], lw=2.0, zorder=3,
                    dashes=DASH[arm] if DASH[arm] else (None, None),
                    marker="o", ms=4.5, mec="white", mew=1.0)
            ax.text(102, med[-1] * f, f" {med[-1]:.0f}", fontsize=8,
                    color=COLOR[arm], va="center", ha="left", fontweight="bold")
        ax.axhline(c["true_k"], color=_INK, lw=1.2, ls=(0, (5, 2)), zorder=4)
        ax.text(26, c["true_k"], f" true k = {c['true_k']}", fontsize=8,
                color=_INK, va="bottom", ha="left")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 3, 5, 10, 20])
        ax.set_ylim(0.85, 26)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticks(q); ax.set_xlim(20, 116)
        ax.set_xlabel("% of training", fontsize=8.5, color=_INK2)
        ax.set_title(_title(c), fontsize=9.5, color=_INK, loc="left")
        if j == 0:
            ax.set_ylabel("distinct states claimed in the epoch\n(log scale; "
                          "median bold, seeds faint)", fontsize=8.5, color=_INK2)

    fig.suptitle("F3   High tau early freezes the assignment: the annealed arm never "
                 "claims states to begin with", fontsize=12.5, color=_INK,
                 x=0.004, ha="left")
    _legend(fig)
    return _save(fig, "F3_claim_trajectory.png")


# ══════════════════════════════════════════════════════════════════════════
# F4 — robustness
# ══════════════════════════════════════════════════════════════════════════
def f4_robustness(cells):
    """What the schedule costs in reliability, pooled over every seed.

    Collapse is the failure the pipeline does not currently catch: a bottleneck
    down to one state still passes a CE tolerance comfortably on most cells.
    """
    runs = [r for c in cells for r in c["runs"]]
    n = len(runs) // len(ARMS)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)
    fig.patch.set_facecolor(_SURF)
    rng = np.random.default_rng(2)

    panels = [
        ("collapsed to a single state", lambda r: r["k_occ"] == 1, True),
        ("converged  (|CE - H_inf| <= 0.10)", lambda r: bool(r["within_tol"]), True),
        ("final loss gap:  CE - H_inf   (mean ± sd over runs)", None, False),
    ]
    for ax, (name, pred, is_rate) in zip(axes, panels):
        _style(ax)
        for i, arm in enumerate(ARMS):
            sub = [r for r in runs if r["arm"] == arm]
            if is_rate:
                k = sum(pred(r) for r in sub)
                ax.bar(i, 100 * k / len(sub), width=0.62, color=COLOR[arm],
                       hatch=HATCH[arm], edgecolor="white", lw=1.4, zorder=2)
                ax.text(i, 100 * k / len(sub) + 1.5, f"{k}/{len(sub)}", ha="center",
                        va="bottom", fontsize=8.5, color=_INK, fontweight="bold")
                ax.set_ylim(0, 108); ax.set_ylabel("% of runs", fontsize=9, color=_INK2)
            else:
                v = np.array([r["ce_minus_H"] for r in sub])
                sd = v.std(ddof=1) if v.size > 1 else 0.0
                ax.bar(i, v.mean(), width=0.62, color=COLOR[arm], hatch=HATCH[arm],
                       edgecolor="white", lw=1.4, zorder=2,
                       yerr=sd, capsize=3.5, ecolor=_INK,
                       error_kw=dict(lw=1.0, alpha=0.75, zorder=6))
                ax.scatter(np.full(v.size, i) + rng.uniform(0.22, 0.35, v.size), v,
                           s=20, color=_INK, alpha=0.5, lw=0, zorder=4)
                ax.text(i - 0.02, v.mean() + sd + 0.02,
                        f"{v.mean():+.3f}\n±{sd:.3f}", ha="center", va="bottom",
                        fontsize=8, color=_INK, fontweight="bold", zorder=5,
                        linespacing=1.25)
                ax.axhline(0.10, color=_INK, lw=1.1, ls=(0, (5, 2)), zorder=3)
                ax.text(len(ARMS) - 0.42, 0.108, "conv_tol", ha="left",
                        va="bottom", fontsize=8, color=_INK)
                ax.set_xlim(-0.45, len(ARMS) + 0.30)
                ax.set_ylim(min(0, v.min()) - 0.05, None)
                ax.set_ylabel("CE - H_inf  (bits, mean ± sd)", fontsize=9,
                              color=_INK2)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(["1.0", "5→0.5", "5.0", "0.5→5"], fontsize=8.5)
        ax.set_xlabel("tau schedule", fontsize=8.5, color=_INK2)
        ax.set_title(name, fontsize=9.5, color=_INK, loc="left")
    axes[2].text(0.5, -0.30, "sd is wide because the runs are bimodal: converged "
                 "near 0, collapsed near +1 bit.\nRead the dots, not the mean "
                 "alone.", transform=axes[2].transAxes, ha="center", va="top",
                 fontsize=7.5, color=_INK2, linespacing=1.35)

    fig.suptitle(f"F4   Robustness, pooled over {n} runs per arm — including "
                 f"{sum(r['k_occ'] == 1 for r in runs if r['arm'] == BASE)} baseline "
                 f"collapses the pipeline does not flag",
                 fontsize=12.5, color=_INK, x=0.004, ha="left")
    _legend(fig)
    return _save(fig, "F4_robustness.png")


def main():
    cells = load_cells()
    if not cells:
        sys.exit("no results found -- run tau_anneal_test.py first")
    print(f"\n  {len(cells)} cells, "
          f"{sum(len(c['seeds']) for c in cells)} paired seeds\n")
    for c in cells:
        print(f"    {_title(c):<48} seeds {c['seeds']}")
    print()
    f1_paired_delta(cells)
    f2_levels(cells)
    f3_claim_trajectory(cells)
    f4_robustness(cells)
    print()


if __name__ == "__main__":
    main()
