#!/usr/bin/env python3
"""
Study 2 tables and figures.  Reads the saved JSON, so it redraws without
retraining.

    python analyse.py

    G1  the answer      paired change in |S_emp - C| against adam:0.9:0.999,
                        pooled over every cell and seed
    G2  the two axes    beta1 with beta2 fixed, beta2 with beta1 fixed
    G3  mechanism       distinct states claimed as training proceeds

Same statistic and same pairing as study 1, with the reference moved to the
optimiser default: every arm is compared to `adam:0.9:0.999` WITHIN a
(cell, seed) pair, tau held at study 1's winner in every arm, so the optimiser
is the only difference.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt                                  # noqa: E402
from harness.report import (GREY, INK, INK2, RULE, S1, S2, S3, S4,  # noqa: E402
                            load, save, style, title)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
REF = "adam:0.9:0.999"


def _spec(r):
    """Study 2 arms are named by the OPTIMISER spec, not the tau spec."""
    return r.get("opt", REF)


def paired_pool(cells, key="abs_dS", ref=REF):
    def val(r):
        return {"abs_dS": abs(r["S_emp_minus_C"]),
                "krat": r["k_occ"] / max(1, r["true_k"]),
                "ce": r["ce_minus_H"],
                "h": r["h_state_given_token"]}[key]
    acc = {}
    for c in cells:
        per = {}
        for r in c["runs"]:
            per.setdefault(r["seed"], {})[_spec(r)] = val(r)
        for d in per.values():
            if ref not in d:
                continue
            for spec, v in d.items():
                acc.setdefault(spec, []).append(v - d[ref])
    out = {}
    for k, v in acc.items():
        a = np.asarray(v, dtype=float)
        sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
        out[k] = (float(a.mean()), sd, int(a.size),
                  float(sd / np.sqrt(a.size)) if a.size > 1 else 0.0,
                  float((a < 0).mean()))
    return out


def levels_pool(cells, key="abs_dS"):
    def val(r):
        return {"abs_dS": abs(r["S_emp_minus_C"]),
                "krat": r["k_occ"] / max(1, r["true_k"]),
                "ce": r["ce_minus_H"],
                "h": r["h_state_given_token"]}[key]
    acc = {}
    for c in cells:
        for r in c["runs"]:
            acc.setdefault(_spec(r), []).append(val(r))
    return {k: (float(np.mean(v)),
                float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
            for k, v in acc.items()}


def table(cells):
    pool = paired_pool(cells)
    lv = levels_pool(cells)
    kr = levels_pool(cells, "krat")
    hh = levels_pool(cells, "h")
    ce = levels_pool(cells, "ce")
    tau = sorted({r.get("tau", "?") for c in cells for r in c["runs"]})

    print(f"\n{'='*108}")
    print(f"  STUDY 2  --  Adam momentum, tau held at {', '.join(tau)}, "
          f"pooled over {len(cells)} cells")
    print("  Ranked by the PAIRED change in |S_emp - C| against "
          f"{REF}.  Negative = better.")
    print(f"{'='*108}")
    print(f"  {'arm':<22}{'d|S-C| vs default':>21}{'SEM':>8}{'win':>7}"
          f"{'|S-C|':>9}{'k/true_k':>10}{'H(s|tok)':>10}{'CE-H':>9}{'n':>4}")
    print(f"  {'-'*105}")
    for spec, (m, sd, n, sem, win) in sorted(pool.items(), key=lambda kv: kv[1][0]):
        mark = "  <- reference" if spec == REF else ""
        star = " *" if (sem > 0 and abs(m) > 2 * sem) else "  "
        print(f"  {spec:<22}{m:>+12.4f}+-{sd:<6.4f}{sem:>8.4f}{win:>7.0%}"
              f"{lv[spec][0]:>9.4f}{kr[spec][0]:>10.2f}{hh[spec][0]:>10.4f}"
              f"{ce[spec][0]:>+9.4f}{n:>4}{star}{mark}")
    print(f"  {'-'*105}")
    print("  * = |mean| > 2 SEM on the paired differences.\n")
    return pool, lv


def g1_answer(cells, pool):
    order = sorted(pool.items(), key=lambda kv: kv[1][0])
    names = [k for k, _ in order]
    m = np.array([v[0] for _, v in order])
    sd = np.array([v[3] if len(v) > 3 else v[1] for _, v in order])   # SEM

    fig, ax = plt.subplots(figsize=(8.4, 0.44 * len(names) + 1.9))
    style(ax, ygrid=False)
    ax.grid(axis="x", color=RULE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    y = np.arange(len(names))[::-1]
    col = [GREY if n == REF else (S3 if v < 0 else S2) for n, v in zip(names, m)]
    ax.barh(y, m, xerr=sd, color=col, height=0.66,
            error_kw=dict(ecolor=INK2, lw=0.9, capsize=2.5))
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("paired change in |S_emp - C|  (bits)   <- better",
                  fontsize=9, color=INK2)
    ax.set_title("G1  Adam momentum vs the optimiser default\n"
                 f"tau held fixed; pooled over {len(cells)} cells, paired to "
                 f"{REF}", fontsize=10.5, color=INK, loc="left", pad=10)
    return save(fig, FIGS, "G1_answer.png")


def g2_axes(cells):
    lv = levels_pool(cells)
    b1, b2 = [], []
    for spec, (m, sd, n) in lv.items():
        p = spec.split(":")
        if p[0] != "adam" or len(p) != 3:
            continue
        a, b = float(p[1]), float(p[2])
        if abs(b - 0.999) < 1e-9:
            b1.append((a, m, sd, spec))
        if abs(a - 0.9) < 1e-9:
            b2.append((b, m, sd, spec))
    b1.sort()
    b2.sort()

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for ax, pts, lab in ((axes[0], b1, "beta1   (beta2 = 0.999)"),
                         (axes[1], b2, "beta2   (beta1 = 0.9)")):
        style(ax)
        if not pts:
            ax.set_visible(False)
            continue
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        e = np.array([p[2] for p in pts])
        ax.errorbar(x, y, yerr=e, marker="o", ms=5.5, lw=1.6, color=S1,
                    ecolor=INK2, elinewidth=0.9, capsize=2.5, zorder=3)
        for xx, yy, p in zip(x, y, pts):
            ax.annotate(f"{p[0]:g}", (xx, yy), textcoords="offset points",
                        xytext=(0, 8), fontsize=7.5, color=INK2, ha="center")
        if REF in lv:
            ax.axhline(lv[REF][0], color=GREY, lw=1.2, ls=(0, (3, 2)))
            ax.annotate("default", (x[-1], lv[REF][0]), fontsize=7.5,
                        color=GREY, va="bottom", ha="right")
        # 1/(1-beta) is the averaging window in steps -- the quantity that
        # actually varies, and it is wildly non-linear in beta.
        ax.set_xlabel(lab + "     [window ~ 1/(1-beta) steps]", fontsize=9,
                      color=INK2)
        ax.set_ylabel("|S_emp - C|   (bits, lower is better)", fontsize=9,
                      color=INK2)
    fig.suptitle("G2  the two momenta move independently", fontsize=10.5,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, FIGS, "G2_axes.png")


def g3_mechanism(cells):
    cells = [c for c in cells if c["runs"] and
             c["runs"][0].get("states_claimed_per_epoch")]
    if not cells:
        return None
    specs = [REF, "adam:0.99:0.999", "adam:0.9:0.99",
             "adam_b1:0.5:0.99", "adam_b1:0.99:0.5"]
    cols = {REF: GREY, "adam:0.99:0.999": S1, "adam:0.9:0.99": S3,
            "adam_b1:0.5:0.99": S4, "adam_b1:0.99:0.5": S2}
    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
    for ax, c in zip(axes[0], cells):
        style(ax)
        for s in specs:
            tr = [r["states_claimed_per_epoch"] for r in c["runs"]
                  if _spec(r) == s and r.get("states_claimed_per_epoch")]
            if not tr:
                continue
            L = min(len(t) for t in tr)
            y = np.mean([t[:L] for t in tr], axis=0)
            ax.plot(np.arange(1, L + 1), y, lw=1.7, color=cols.get(s, S1),
                    ls=(0, (3, 2)) if s == REF else "-")
            ax.annotate(s.replace("adam", ""), (L, y[-1]), fontsize=6.5,
                        color=cols.get(s, S1), va="center", xytext=(3, 0),
                        textcoords="offset points")
        ax.axhline(c["true_k"], color=INK, lw=1.0, ls=(0, (1, 2)))
        ax.set_xlabel("epoch", fontsize=9, color=INK2)
        ax.set_ylabel("distinct states claimed", fontsize=9, color=INK2)
        ax.set_title(title(c), fontsize=8.5, color=INK, loc="left")
    fig.suptitle("G3  does momentum change WHEN states are claimed?",
                 fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save(fig, FIGS, "G3_mechanism.png")


def main():
    cells = load(os.path.join(HERE, "results"))
    if not cells:
        sys.exit("no study 2 results yet")
    print(f"\n  {len(cells)} cells:")
    for c in cells:
        print(f"    {title(c)}   seeds {c['seeds']}  {len(c['runs'])} runs")
    pool, lv = table(cells)
    g1_answer(cells, pool)
    g2_axes(cells)
    g3_mechanism(cells)
    best = [k for k, _ in sorted(pool.items(), key=lambda kv: kv[1][0])
            if k != REF][:2]
    print(f"\n  best two optimiser arms: {best}\n")


if __name__ == "__main__":
    main()
