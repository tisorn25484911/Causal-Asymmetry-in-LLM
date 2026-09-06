#!/usr/bin/env python3
"""
The cross-study summary: what to set tau to, and what it buys over the pipeline.

    python summarise.py

Three tables, in the order the argument needs them:

    1  BALANCED, PER PHASE.  Every arm ranked against const:1 on the cells where
       ALL of that phase's arms ran.  The pooled ranking across phases is a trap
       -- an arm run only on the two easiest cells collects a bigger paired
       improvement than one run on all of them, so pooling would measure which
       cells an arm happened to see.  `common_support` prevents that.

    2  HEADLINE.  The pipeline's own configuration (tau const:1, adam:0.9:0.999)
       against the best configuration found, on the SAME three cells and the
       SAME seeds.  Study 2's arms all run at tau = the study 1 winner, so its
       adam:0.9:0.999 arm IS the recommended configuration and pairs directly
       against 1B's const:1 arm -- data, split and initialisation are pure
       functions of the seed, so the two are the same experiment differing only
       in the schedule.

    3  ROBUSTNESS.  The winner on the four cells of phase 1D, which it did not
       choose and which were fixed before it was known.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness.report import (common_support, load, merge_cells,   # noqa: E402
                            title)

HERE = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.join(HERE, "study1_tau_schedules")
S2 = os.path.join(HERE, "study2_momentum")
REF = "const:1"
OPT_REF = "adam:0.9:0.999"

METRICS = ("abs_dS", "krat", "ce", "h")


def _vals(r):
    return (abs(r["S_emp_minus_C"]), r["k_occ"] / max(1, r["true_k"]),
            r["ce_minus_H"], r["h_state_given_token"])


def paired(cells, arms=None, ref=REF, key=lambda r: r["spec"]):
    """{spec: (mean_d, sem, win, level, krat, ce, h, n)} against `ref`."""
    acc = {}
    for c in cells:
        per = {}
        for r in c["runs"]:
            per.setdefault(r["seed"], {})[key(r)] = _vals(r)
        for d in per.values():
            if ref not in d:
                continue
            for s, v in d.items():
                if arms and s not in arms:
                    continue
                acc.setdefault(s, []).append((v[0] - d[ref][0],) + v)
    out = {}
    for s, v in acc.items():
        a = np.array(v, dtype=float)
        sem = a[:, 0].std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        out[s] = (a[:, 0].mean(), sem, (a[:, 0] < 0).mean(), a[:, 1].mean(),
                  a[:, 2].mean(), a[:, 3].mean(), a[:, 4].mean(), len(a))
    return out


def show(res, ref=REF, note=""):
    print(f"  {'arm':<22}{'d|S-C|':>10}{'SEM':>8}{'win':>7}{'|S-C|':>9}"
          f"{'k/true':>9}{'CE-H':>9}{'H(s|tok)':>10}{'n':>4}")
    print("  " + "-" * 88)
    for s, (m, sem, win, lv, kr, ce, h, n) in sorted(res.items()):
        pass
    for s, (m, sem, win, lv, kr, ce, h, n) in sorted(res.items(),
                                                     key=lambda kv: kv[1][0]):
        star = " *" if sem > 0 and abs(m) > 2 * sem else "  "
        mark = f"  <- {note}" if s == ref else ""
        print(f"  {s:<22}{m:>+10.4f}{sem:>8.4f}{win:>7.0%}{lv:>9.4f}"
              f"{kr:>9.2f}{ce:>+9.4f}{h:>10.4f}{n:>4}{star}{mark}")
    print()


def main():
    cells = merge_cells(load(os.path.join(S1, "results"),
                             os.path.join(S1, "results_1A"),
                             os.path.join(HERE, "study0_pilot", "results")))
    s2 = merge_cells(load(os.path.join(S2, "results")))

    # ── 1. balanced, per phase ───────────────────────────────────────────
    for phase, label in (("1B", "1B  levels and shape"),
                         ("1C", "1C  how wide the traversal should be"),
                         ("1D", "1D  robustness on four cells it did not choose"),
                         ("0", "1A + pilot  direction, endpoint, range")):
        cs = [c for c in cells if c["study"] == phase]
        if not cs:
            continue
        sup = common_support(cs)
        print(f"\n{'='*98}\n  PHASE {label}   "
              f"{len(cs)} cells, {len(sup)} arms on common support")
        for c in cs:
            print(f"     {title(c)}")
        print()
        show(paired(cs, arms=sup), note="pipeline default")

    # ── 2. the headline ──────────────────────────────────────────────────
    # 1B's cells and study 2's cells are the same three, so const:1 (1B) and
    # the recommended configuration (study 2's adam default arm, run at the
    # winning tau) pair seed by seed.
    print(f"\n{'='*98}\n  HEADLINE  --  the pipeline's configuration vs the "
          f"recommended one, same cells, same seeds")
    tau_win = sorted({r.get("tau") for c in s2 for r in c["runs"]})
    print(f"  recommended:  tau = {', '.join(t for t in tau_win if t)}   "
          f"optimiser = {OPT_REF} (unchanged; study 2 found no better)")
    key = lambda c: (c["tag"], c["mode"])                        # noqa: E731
    s2_by = {key(c): c for c in s2}
    rows = []
    for c in [c for c in cells if c["study"] == "1B"]:
        if key(c) not in s2_by:
            continue
        base = {r["seed"]: _vals(r) for r in c["runs"] if r["spec"] == REF}
        new = {r["seed"]: _vals(r) for r in s2_by[key(c)]["runs"]
               if r.get("opt") == OPT_REF}
        for sd in sorted(set(base) & set(new)):
            rows.append((title(c), sd, base[sd], new[sd]))
    if rows:
        print(f"\n  {'cell':<44}{'|S-C| pipeline':>16}{'|S-C| new':>12}"
              f"{'k/true':>16}{'CE-H':>16}")
        print("  " + "-" * 92)
        seen = set()
        for t, sd, b, n in rows:
            if t in seen:
                continue
            seen.add(t)
            bs = [x for tt, _, x, _ in rows if tt == t]
            ns = [x for tt, _, _, x in rows if tt == t]
            bm = np.mean([x[0] for x in bs]); nm = np.mean([x[0] for x in ns])
            bk = np.mean([x[1] for x in bs]); nk = np.mean([x[1] for x in ns])
            bc = np.mean([x[2] for x in bs]); nc = np.mean([x[2] for x in ns])
            print(f"  {t:<44}{bm:>16.4f}{nm:>12.4f}"
                  f"{bk:>9.2f} ->{nk:>5.2f}{bc:>+10.4f} ->{nc:>+6.4f}")
        B = np.array([[x[0], x[1], x[2]] for _, _, x, _ in rows])
        N = np.array([[x[0], x[1], x[2]] for _, _, _, x in rows])
        d = N[:, 0] - B[:, 0]
        print("  " + "-" * 92)
        print(f"  {'POOLED  (n=' + str(len(rows)) + ' paired runs)':<44}"
              f"{B[:,0].mean():>16.4f}{N[:,0].mean():>12.4f}"
              f"{B[:,1].mean():>9.2f} ->{N[:,1].mean():>5.2f}"
              f"{B[:,2].mean():>+10.4f} ->{N[:,2].mean():>+6.4f}")
        sem = d.std(ddof=1) / np.sqrt(len(d))
        print(f"\n  paired change in |S_emp - C|: {d.mean():+.4f} +- {sem:.4f} "
              f"(SEM), improved on {(d < 0).mean():.0%} of runs")
        if B[:, 0].mean() > 0:
            print(f"  relative reduction in |S_emp - C|: "
                  f"{100*(1 - N[:,0].mean()/B[:,0].mean()):.0f}%")

    # ── 3. study 2 ───────────────────────────────────────────────────────
    print(f"\n{'='*98}\n  STUDY 2  --  Adam momentum at the winning tau, "
          f"{len(s2)} cells")
    print()
    show(paired(s2, ref=OPT_REF, key=lambda r: r.get("opt", OPT_REF)),
         ref=OPT_REF, note="optimiser default")


if __name__ == "__main__":
    main()
