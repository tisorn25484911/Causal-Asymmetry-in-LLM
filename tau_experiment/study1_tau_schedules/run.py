#!/usr/bin/env python3
"""
Study 1 -- how should tau vary during training?

    python run.py --phase 1B --cell 0
    python run.py --phase 1B --list

Study 0 (../study0_pilot) established, on flower only and with four arms, that
the DIRECTION matters and that the textbook direction is the wrong one: the arm
annealing 5 -> 0.5, which the pilot was built to confirm, was worst on every
cell, while its reversed control 0.5 -> 5 was best on two of three.  Study 1
asks the three questions that leaves open.

    1A  direction, endpoint and range, on a second process family.
        Does the ordering survive on the coin (V=3, K=15), which the pilot never
        ran, and does a wider range (20 <-> 0.1) change the answer?

    1B  levels and shape.  Given that ending high is better: HOW high, does the
        start matter independently of the end, and does the path between them
        matter?  Twelve arms, fully paired, on three cells chosen to span
        V = 3, 8, 10 and both arms.

    1D  robustness.  The two best schedules from 1B against the pipeline default
        across eight cells, to establish that the winner is not a property of
        the three cells that chose it.

Every arm within a cell is PAIRED: one realisation and one initialisation per
seed, shared by all arms, so the only difference between arms is the schedule.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import run_cell                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

_ADAM = "adam:0.9:0.999"          # the pipeline optimiser, held fixed in study 1


# ── phase 1B: levels and shape ───────────────────────────────────────────
# Arms 3-6 hold the START at 0.5 and move the END: the endpoint axis.
# Arms 4, 7-9 hold the END at 5 and move the START: the start axis.
# Arms 10-12 hold both endpoints and change the PATH: the shape axis.
# Two constants anchor it -- const:1 is the pipeline, const:5 separates "high
# tau helped" from "a schedule helped".
ARMS_1B = [
    ("const:1",          "const:1",          _ADAM),   # the pipeline, exactly
    ("const:5",          "const:5",          _ADAM),   # high, no schedule
    ("geom:0.5:2",       "geom:0.5:2",       _ADAM),   # ── endpoint axis
    ("geom:0.5:5",       "geom:0.5:5",       _ADAM),   #    (study 0's winner)
    ("geom:0.5:10",      "geom:0.5:10",      _ADAM),
    ("geom:0.5:20",      "geom:0.5:20",      _ADAM),
    ("geom:0.25:5",      "geom:0.25:5",      _ADAM),   # ── start axis
    ("geom:1:5",         "geom:1:5",         _ADAM),
    ("geom:2:5",         "geom:2:5",         _ADAM),
    ("hold:0.5:5:0.3",   "hold:0.5:5:0.3",   _ADAM),   # ── shape axis
    ("ramp:0.5:5:0.3",   "ramp:0.5:5:0.3",   _ADAM),
    ("cos:0.5:5",        "cos:0.5:5",        _ADAM),
]

CELLS_1B = [
    dict(process="flower", n=2, m=6, mode="forward"),    # V=8   true_k=3
    dict(process="flower", n=6, m=4, mode="backward"),   # V=10  true_k=5
    dict(process="coin", p=0.1, q=0.9, mode="backward"),  # V=3   true_k=3
]

# ── phase 1D: robustness of the winner ───────────────────────────────────
# Arms are filled in from 1B's result; the cells are fixed here so the choice of
# cells cannot be made after seeing which arms they favour.
CELLS_1D = [
    dict(process="coin", p=0.1, q=0.9, mode="forward"),
    dict(process="coin", p=0.3, q=0.4, mode="backward"),
    dict(process="coin", p=0.4, q=0.8, mode="forward"),
    dict(process="flower", n=2, m=6, mode="backward"),
    dict(process="flower", n=2, m=8, mode="forward"),
    dict(process="flower", n=4, m=2, mode="backward"),
    dict(process="flower", n=6, m=4, mode="forward"),
    dict(process="flower", n=2, m=8, mode="backward"),
]

# ── phase 1C: how wide should the traversal be? ──────────────────────────
# 1A measured, on flower(2,6) forward: geom:0.1:20 recovered the state set
# EXACTLY on 3 of 3 seeds (k=3/3, S_emp - C = -0.0017, H(state|token) = 0.0031)
# while geom:0.5:5 reached 2.67 states and -0.158.  Those two arms differ in the
# START and the END at once, and the corner (0.1, 20) lies outside 1B's grid,
# whose widest arm is 0.5 -> 20.  1C opens the range on both sides and holds one
# endpoint at a time, so "wider is better" can be told apart from "0.1 is a
# better start" and from "20 is a better end".
ARMS_1C = [
    ("const:1",       "const:1",       _ADAM),   # reference, for the pairing
    ("geom:0.1:20",   "geom:0.1:20",   _ADAM),   # 1A's exact recovery
    ("geom:0.1:5",    "geom:0.1:5",    _ADAM),   # same start, 1B's end
    ("geom:0.25:20",  "geom:0.25:20",  _ADAM),   # same end, higher start
    ("geom:0.05:20",  "geom:0.05:20",  _ADAM),   # wider still, on the low side
    ("geom:0.1:50",   "geom:0.1:50",   _ADAM),   # wider still, on the high side
]

CELLS_1C = [
    dict(process="flower", n=2, m=6, mode="forward"),     # V=8   true_k=3
    dict(process="flower", n=2, m=8, mode="backward"),    # V=10  the hardest cell
]

PHASES = {
    "1B": dict(arms=ARMS_1B, cells=CELLS_1B, seeds=2),
    "1C": dict(arms=ARMS_1C, cells=CELLS_1C, seeds=2),
    "1D": dict(arms=None,    cells=CELLS_1D, seeds=2),   # arms via --arms
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=sorted(PHASES), default="1B")
    ap.add_argument("--cell", type=int, default=None,
                    help="index into the phase's cell list; omit to run all")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="tau specs, for phase 1D (name == spec)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--accelerator", default="cpu",
                    help="cpu is exactly repeatable; MPS is not")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    ph = PHASES[a.phase]
    arms = ph["arms"] or [(s, s, _ADAM) for s in (a.arms or [])]
    if not arms:
        sys.exit("phase 1D needs --arms")
    seeds = a.seeds or ph["seeds"]

    if a.list:
        print(f"\n  phase {a.phase}: {len(arms)} arms x {len(ph['cells'])} cells "
              f"x {seeds} seeds = {len(arms)*len(ph['cells'])*seeds} runs")
        for i, c in enumerate(ph["cells"]):
            print(f"    --cell {i}   {c}")
        print("\n  arms: " + ", ".join(n for n, _, _ in arms) + "\n")
        return 0

    cells = ph["cells"] if a.cell is None else [ph["cells"][a.cell]]
    for c in cells:
        c = dict(c)
        mode = c.pop("mode")
        run_cell(a.phase, c.pop("process"), mode, arms, seeds, RESULTS,
                 epochs=a.epochs, accelerator=a.accelerator, **c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
