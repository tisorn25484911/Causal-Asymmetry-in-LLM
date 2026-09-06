#!/usr/bin/env python3
"""
Study 1 tables and figures.  Reads the saved JSON, so it redraws without
retraining.

    python analyse.py

    F1  the answer        paired change in |S_emp - C| against the pipeline
                          default, pooled over every cell and seed
    F2  levels            the endpoint axis and the start axis, separately
    F3  shape             the four paths between the same endpoints
    F4  mechanism         distinct states claimed as training proceeds
    F5  robustness        the winner, cell by cell

THE STATISTIC.  Every arm is compared to `const:1` WITHIN a (cell, seed) pair and
the differences are pooled afterwards.  Pooling the levels instead would let a
hard cell (flower backward, true k = 7) dominate an easy one (coin, true k = 2)
purely through its larger |S_emp - C|, and the arms are paired by construction --
same realisation, same initialisation -- so the difference is the quantity the
design actually supports.

|S_emp - C| rather than k_occ, because it is symmetric about the truth: too few
states (merged) and too many (one causal state split across several bottleneck
states, 6.9) both move it up, whereas k_occ scores a duplicated state as a win.
H(state|token) is reported beside it as the independent duplication witness and
is never folded into the score.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt                                  # noqa: E402
from harness.report import (GREY, INK, INK2, RULE, S1, S2, S3, S4, SEQ,  # noqa: E402
                            load, parse_family, save, style, title)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
REF = "const:1"


# ══════════════════════════════════════════════════════════════════════════
def paired_pool(cells, key="abs_dS", ref=REF):
    """
    {spec: (mean, sd, n)} of the per-(cell, seed) difference against `ref`.

    n is the number of (cell, seed) pairs the arm actually appears in, so an arm
    that only ran on some cells is never silently averaged as if it had run on
    all of them.
    """
    def val(r):
        return {"abs_dS": abs(r["S_emp_minus_C"]),
                "dS": r["S_emp_minus_C"],
                "k_occ": r["k_occ"],
                "krat": r["k_occ"] / max(1, r["true_k"]),
                "ce": r["ce_minus_H"],
                "h": r["h_state_given_token"]}[key]

    acc = {}
    for c in cells:
        per = {}
        for r in c["runs"]:
            per.setdefault(r["seed"], {})[r["spec"]] = val(r)
        for seed, d in per.items():
            if ref not in d:
                continue
            for spec, v in d.items():
                acc.setdefault(spec, []).append(v - d[ref])
    out = {}
    for k, v in acc.items():
        a = np.asarray(v, dtype=float)
        sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
        sem = sd / np.sqrt(a.size) if a.size > 1 else 0.0
        # Win rate is a sign test: with n ~ 15 paired differences and a spread
        # this wide, "beat the reference on 13 of 15 cells" is a claim the data
        # supports, while a mean +- sd that straddles zero is not.
        out[k] = (float(a.mean()), sd, int(a.size), float(sem),
                  float((a < 0).mean()))
    return out


def levels_pool(cells, key="abs_dS"):
    def val(r):
        return {"abs_dS": abs(r["S_emp_minus_C"]),
                "krat": r["k_occ"] / max(1, r["true_k"]),
                "h": r["h_state_given_token"],
                "ce": r["ce_minus_H"]}[key]
    acc = {}
    for c in cells:
        for r in c["runs"]:
            acc.setdefault(r["spec"], []).append(val(r))
    return {k: (float(np.mean(v)),
                float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
            for k, v in acc.items()}


# ══════════════════════════════════════════════════════════════════════════
def table(cells):
    pool = paired_pool(cells, "abs_dS")
    lv = levels_pool(cells, "abs_dS")
    kr = levels_pool(cells, "krat")
    hh = levels_pool(cells, "h")
    ce = levels_pool(cells, "ce")

    print(f"\n{'='*108}")
    print("  STUDY 1  --  tau schedules, pooled over "
          f"{len(cells)} cells")
    print("  Ranked by the PAIRED change in |S_emp - C| against const:1.  "
          "Negative = better than the pipeline.")
    print(f"{'='*108}")
    print(f"  {'arm':<20}{'d|S-C| vs const:1':>21}{'SEM':>8}{'win':>7}"
          f"{'|S-C|':>9}{'k/true_k':>10}{'H(s|tok)':>10}{'CE-H':>9}{'n':>4}")
    print(f"  {'-'*105}")
    for spec, (m, sd, n, sem, win) in sorted(pool.items(), key=lambda kv: kv[1][0]):
        mark = "  <- reference" if spec == REF else ""
        star = " *" if (sem > 0 and abs(m) > 2 * sem) else "  "
        print(f"  {spec:<20}{m:>+12.4f}+-{sd:<6.4f}{sem:>8.4f}{win:>7.0%}"
              f"{lv[spec][0]:>9.4f}{kr[spec][0]:>10.2f}{hh[spec][0]:>10.4f}"
              f"{ce[spec][0]:>+9.4f}{n:>4}{star}{mark}")
    print(f"  {'-'*105}")
    print("  * = |mean| > 2 SEM on the paired differences.  `win` is the "
          "fraction of paired\n    comparisons the arm won outright -- a sign "
          "test, which small n supports better\n    than a mean whose sd "
          "straddles zero.\n")
    return pool, lv, kr, hh, ce


# ══════════════════════════════════════════════════════════════════════════
def f1_answer(cells, pool):
    order = sorted(pool.items(), key=lambda kv: kv[1][0])
    names = [k for k, _ in order]
    m = np.array([v[0] for _, v in order])
    sd = np.array([v[3] if len(v) > 3 else v[1] for _, v in order])   # SEM

    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(names) + 1.8))
    style(ax, ygrid=False)
    ax.grid(axis="x", color=RULE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    y = np.arange(len(names))[::-1]
    col = [GREY if n == REF else (S3 if v < 0 else S2) for n, v in zip(names, m)]
    ax.barh(y, m, xerr=sd, color=col, height=0.66,
            error_kw=dict(ecolor=INK2, lw=0.9, capsize=2.5))
    for n, yy, c in zip(names, y, col):
        if n == REF:
            ax.barh([yy], [m[list(names).index(n)]], color="none",
                    edgecolor=INK2, hatch="///", height=0.66, lw=0.8)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("paired change in |S_emp - C|  (bits, +-1 SEM)   <- better",
                  fontsize=9, color=INK2)
    ax.set_title("F1  tau schedule vs the pipeline default\n"
                 f"pooled over {len(cells)} cells; each arm paired to const:1 "
                 "on the same data and initialisation",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    return save(fig, FIGS, "F1_answer.png")


def f2_levels(cells):
    lv = levels_pool(cells, "abs_dS")
    kr = levels_pool(cells, "krat")
    fams = {k: parse_family(k) for k in lv}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for ax, axis, fixed, lab in (
            (axes[0], "end", 0.5, "tau at the END   (start held at 0.5)"),
            (axes[1], "start", 5.0, "tau at the START   (end held at 5)")):
        style(ax)
        pts = []
        for spec, f in fams.items():
            if f is None or f["kind"] not in ("geom", "const"):
                continue
            if f["kind"] == "const":
                continue
            if axis == "end" and abs(f["a"] - fixed) < 1e-9:
                pts.append((f["b"], spec))
            if axis == "start" and abs(f["b"] - fixed) < 1e-9:
                pts.append((f["a"], spec))
        pts.sort()
        if not pts:
            ax.set_visible(False)
            continue
        x = np.array([p for p, _ in pts])
        y = np.array([lv[s][0] for _, s in pts])
        e = np.array([lv[s][1] for _, s in pts])
        ax.errorbar(x, y, yerr=e, marker="o", ms=5.5, lw=1.6, color=S1,
                    ecolor=INK2, elinewidth=0.9, capsize=2.5, zorder=3)
        for xx, yy, (_, s) in zip(x, y, pts):
            ax.annotate(s.split(":", 1)[1], (xx, yy), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, color=INK2, ha="center")
        for cname, cc in (("const:1", GREY), ("const:5", S4)):
            if cname in lv:
                ax.axhline(lv[cname][0], color=cc, lw=1.2,
                           ls=(0, (3, 2)), zorder=2)
                ax.annotate(cname, (ax.get_xlim()[1], lv[cname][0]), fontsize=7.5,
                            color=cc, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xlabel(lab, fontsize=9, color=INK2)
        ax.set_ylabel("|S_emp - C|   (bits, lower is better)", fontsize=9,
                      color=INK2)
    fig.suptitle("F2  where the sweet spot is:  the endpoint and the start move "
                 "independently", fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, FIGS, "F2_levels.png")


def f3_shape(cells):
    from harness.schedules import parse_tau
    lv = levels_pool(cells, "abs_dS")
    shapes = [s for s in lv if parse_family(s) and
              parse_family(s)["kind"] in ("geom", "hold", "ramp", "cos", "lin")
              and parse_family(s).get("a") == 0.5
              and parse_family(s).get("b") == 5.0]
    shapes.sort()
    if not shapes:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7),
                             gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axes[0]
    style(ax)
    f = np.linspace(0, 1, 300)
    cols = [S1, S2, S3, S4, "#c2185b"]
    for s, c in zip(shapes, cols):
        _, fn = parse_tau(s)
        ax.plot(f, [fn(v) for v in f], lw=1.9, color=c, label=s)
        ax.annotate(s.split(":", 1)[0], (1.0, fn(1.0)), fontsize=7.5, color=c,
                    va="center", ha="left", xytext=(3, 0),
                    textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("training progress", fontsize=9, color=INK2)
    ax.set_ylabel("tau", fontsize=9, color=INK2)
    ax.set_title("the paths", fontsize=9.5, color=INK, loc="left")

    ax = axes[1]
    style(ax, ygrid=False)
    ax.grid(axis="x", color=RULE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    y = np.arange(len(shapes))[::-1]
    ax.barh(y, [lv[s][0] for s in shapes], xerr=[lv[s][1] for s in shapes],
            color=cols[:len(shapes)], height=0.6,
            error_kw=dict(ecolor=INK2, lw=0.9, capsize=2.5))
    if REF in lv:
        ax.axvline(lv[REF][0], color=GREY, lw=1.3, ls=(0, (3, 2)))
        ax.annotate("const:1", (lv[REF][0], y[0] + 0.5), fontsize=7.5,
                    color=GREY, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([s.split(":", 1)[0] for s in shapes], fontsize=8.5)
    ax.set_xlabel("|S_emp - C|  (bits)", fontsize=9, color=INK2)
    ax.set_title("the outcome", fontsize=9.5, color=INK, loc="left")
    fig.suptitle("F3  same endpoints, different paths", fontsize=10.5,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save(fig, FIGS, "F3_shape.png")


def f4_mechanism(cells, arms=None):
    cells = [c for c in cells if c["runs"] and
             c["runs"][0].get("states_claimed_per_epoch")]
    if not cells:
        return None
    specs = arms or [REF, "const:5", "geom:0.5:5", "geom:0.5:20", "geom:5:0.5"]
    cols = {REF: GREY, "const:5": S4, "geom:0.5:5": S3,
            "geom:0.5:20": S1, "geom:5:0.5": S2}
    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
    for ax, c in zip(axes[0], cells):
        style(ax)
        for s in specs:
            tr = [r["states_claimed_per_epoch"] for r in c["runs"]
                  if r["spec"] == s and r.get("states_claimed_per_epoch")]
            if not tr:
                continue
            L = min(len(t) for t in tr)
            y = np.mean([t[:L] for t in tr], axis=0)
            ax.plot(np.arange(1, L + 1), y, lw=1.7,
                    color=cols.get(s, S1),
                    ls=(0, (3, 2)) if s == REF else "-")
            ax.annotate(s, (L, y[-1]), fontsize=7, color=cols.get(s, S1),
                        va="center", xytext=(3, 0), textcoords="offset points")
        ax.axhline(c["true_k"], color=INK, lw=1.0, ls=(0, (1, 2)))
        ax.annotate(f"true k = {c['true_k']}", (1, c["true_k"]), fontsize=7,
                    color=INK, va="bottom")
        ax.set_xlabel("epoch", fontsize=9, color=INK2)
        ax.set_ylabel("distinct states claimed", fontsize=9, color=INK2)
        ax.set_title(title(c), fontsize=8.5, color=INK, loc="left")
    fig.suptitle("F4  the mechanism: WHEN states are claimed, not only how many",
                 fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save(fig, FIGS, "F4_mechanism.png")


def f5_robustness(cells, winners):
    rows = []
    for c in cells:
        per = {}
        for r in c["runs"]:
            per.setdefault(r["seed"], {})[r["spec"]] = abs(r["S_emp_minus_C"])
        for s in winners:
            d = [v[s] - v[REF] for v in per.values() if s in v and REF in v]
            if d:
                rows.append((title(c), s, float(np.mean(d))))
    if not rows:
        return None
    cellnames = sorted({r[0] for r in rows})
    fig, ax = plt.subplots(figsize=(8.8, 0.34 * len(cellnames) + 2.0))
    style(ax, ygrid=False)
    ax.grid(axis="x", color=RULE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    cols = {w: c for w, c in zip(winners, [S3, S1, S2, S4])}
    for i, cn in enumerate(cellnames):
        yy = len(cellnames) - 1 - i
        for w in winners:
            v = [r[2] for r in rows if r[0] == cn and r[1] == w]
            if v:
                ax.scatter(v[0], yy + 0.16 * (winners.index(w) - 0.5), s=48,
                           color=cols[w], zorder=3,
                           label=w if i == 0 else None)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(range(len(cellnames)))
    ax.set_yticklabels(cellnames[::-1], fontsize=7.5)
    ax.set_xlabel("paired change in |S_emp - C| vs const:1   <- better",
                  fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("F5  robustness: does the winner win on every cell?",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    fig.tight_layout()
    return save(fig, FIGS, "F5_robustness.png")


def f6_headline(cells_1b, s2_cells):
    """Pipeline configuration vs the recommended one, same cells, same seeds."""
    import numpy as np
    key = lambda c: (c["tag"], c["mode"])                        # noqa: E731
    s2 = {key(c): c for c in s2_cells}
    names, base, new_ = [], [], []
    for c in cells_1b:
        if key(c) not in s2:
            continue
        b = [abs(r["S_emp_minus_C"]) for r in c["runs"] if r["spec"] == REF]
        n = [abs(r["S_emp_minus_C"]) for r in s2[key(c)]["runs"]
             if r.get("opt") == "adam:0.9:0.999"]
        if not (b and n):
            continue
        names.append(title(c).split("   ")[0])
        base.append(np.mean(b))
        new_.append(np.mean(n))
    if not names:
        return None
    names.append("POOLED")
    base.append(np.mean(base))
    new_.append(np.mean(new_))

    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(names) + 1.8))
    style(ax, ygrid=False)
    ax.grid(axis="x", color=RULE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.barh(y + 0.19, base, height=0.34, color=GREY, hatch="///",
            edgecolor=INK2, lw=0.7, label="pipeline  tau=const:1")
    ax.barh(y - 0.19, new_, height=0.34, color=S3,
            label="recommended  tau=geom:0.05:20")
    for yy, b, n in zip(y, base, new_):
        ax.annotate(f"{b:.3f}", (b, yy + 0.19), fontsize=7.5, color=INK2,
                    va="center", xytext=(4, 0), textcoords="offset points")
        ax.annotate(f"{n:.3f}", (n, yy - 0.19), fontsize=7.5, color=S3,
                    va="center", xytext=(4, 0), textcoords="offset points")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("|S_emp - C|   (bits, lower is better)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("F6  what the schedule buys\n"
                 "same three cells, same seeds; optimiser unchanged",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    return save(fig, FIGS, "F6_headline.png")



def f7_where_it_helps(cells):
    """
    The schedule is a RESCUE, not a general improvement.

    x is how badly the pipeline does on a cell, y is how much the best rising
    schedule changes it.  If the schedule were a uniform improvement the cloud
    would be a flat band below zero; instead it slopes -- large gains exactly
    where const:1 fails, nothing where it already works.  Phase 1D exists in
    this plot as the four points nearest the origin, and they are the reason
    the headline is stated as a rescue rather than as a speed-up.
    """
    import numpy as np
    RISING = {"geom:0.5:5", "geom:0.05:20", "geom:0.1:5", "geom:0.1:20",
              "cos:0.5:5", "geom:0.5:2", "geom:0.5:10", "geom:0.5:20",
              "geom:0.25:5", "geom:1:5", "geom:2:5", "hold:0.5:5:0.3",
              "ramp:0.5:5:0.3", "geom:0.25:20"}
    pts = []
    for c in cells:
        per = {}
        for r in c["runs"]:
            per.setdefault(r["seed"], {})[r["spec"]] = abs(r["S_emp_minus_C"])
        base = [d[REF] for d in per.values() if REF in d]
        if not base:
            continue
        dl = {}
        for sp in RISING:
            v = [d[sp] - d[REF] for d in per.values() if sp in d and REF in d]
            if v:
                dl[sp] = float(np.mean(v))
        if dl:
            bs = min(dl.items(), key=lambda kv: kv[1])
            pts.append((float(np.mean(base)), bs[1], bs[0], c))
    if len(pts) < 3:
        return None

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    style(ax)
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    ax.axhline(0, color=INK, lw=1.1)
    ax.axvspan(0, 0.30, color=GREY, alpha=0.13, lw=0)
    kind = ["flower" if "flower" in p[3]["tag"] else "coin" for p in pts]
    # Label placement: the low-x cells pile up against the axis, so they are
    # labelled once as a group rather than five times on top of each other.
    order = np.argsort(x)
    for rank, i in enumerate(order):
        xi, yi, k, p = x[i], y[i], kind[i], pts[i]
        ax.scatter(xi, yi, s=95, zorder=3,
                   color=(S1 if k == "flower" else S2),
                   marker=("o" if k == "flower" else "^"),
                   edgecolor="white", linewidth=1.1)
        if xi < 0.30:
            continue
        dy = 9 if rank % 2 else -13
        ax.annotate(f"{p[3]['tag'].replace('_',' ')} {p[3]['mode'][:2]}",
                    (xi, yi), fontsize=7, color=INK2,
                    xytext=(8, dy), textcoords="offset points")
    ax.annotate("5 cells where the pipeline\nalready works  (no headroom)",
                (0.15, y.min() * 0.72), fontsize=8, color=INK2, ha="center",
                va="center", linespacing=1.5)
    m, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(0, x.max() * 1.05, 20)
    ax.plot(xs, m * xs + b0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    r = np.corrcoef(x, y)[0, 1]
    ax.annotate(f"r = {r:+.2f}   n = {len(x)} cells", (0.985, 0.06),
                xycoords="axes fraction", fontsize=9, color=INK2,
                ha="right", va="bottom")
    ax.set_xlabel("how badly the pipeline does on this cell   "
                  "|S_emp - C| at tau = const:1   (bits)", fontsize=9, color=INK2)
    ax.set_ylabel("change from the best rising schedule  (bits)\n<- schedule helps",
                  fontsize=9, color=INK2)
    ax.set_title("F7  the schedule is a rescue, not a general improvement\n"
                 "circles flower, triangles coin; each point one (process, arm) cell",
                 fontsize=10.5, color=INK, loc="left", pad=10)
    return save(fig, FIGS, "F7_where_it_helps.png")


def main():
    from harness.report import common_support, merge_cells
    cells = merge_cells(load(os.path.join(HERE, "results"),
                             os.path.join(HERE, "results_1A"),
                             os.path.join(os.path.dirname(HERE),
                                          "study0_pilot", "results")))
    if not cells:
        sys.exit("no results yet")
    print(f"\n  {len(cells)} cells:")
    for c in cells:
        print(f"    [{c['study']}] {title(c)}   seeds {c['seeds']}  "
              f"{len(c['runs'])} runs")

    # F1-F3 come from phase 1B, the only FULLY CROSSED design here: 12 arms on
    # every one of 3 cells.  Ranking across phases would compare arms that saw
    # different cells, which measures the cells rather than the arms.
    b = [c for c in cells if c["study"] == "1B"]
    sup = common_support(b)
    pool = {k: v for k, v in paired_pool(b, "abs_dS").items() if k in sup}
    print("\n  --- phase 1B, balanced (12 arms x 3 cells) ---")
    table(b)
    f1_answer(b, pool)
    f2_levels(b)
    f3_shape(b)
    f4_mechanism(cells)

    c1 = [c for c in cells if c["study"] == "1C"]
    if c1:
        print("\n  --- phase 1C, balanced (6 arms x 2 cells) ---")
        table(c1)

    d = [c for c in cells if c["study"] == "1D"]
    if d:
        print("\n  --- phase 1D, robustness ---")
        table(d)
        best = [k for k, _ in sorted(paired_pool(d, "abs_dS").items(),
                                     key=lambda kv: kv[1][0]) if k != REF][:2]
        f5_robustness(d, best)

    s2 = merge_cells(load(os.path.join(os.path.dirname(HERE),
                                       "study2_momentum", "results")))
    if s2 and b:
        f6_headline(b, s2)
    f7_where_it_helps(cells)
    print()


if __name__ == "__main__":
    main()
