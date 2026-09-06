#!/usr/bin/env python3
"""
Study 2 -- does Adam's momentum help, once the tau schedule is fixed?

    python run.py --tau geom:0.5:5 --cell 0

WHY MOMENTUM, and why this is not a blind sweep.  Study 1's mechanism is
gradient starvation at the state head: the straight-through Jacobian is
(1/tau)(diag(p) - p p^T), whose off-diagonal mass decays like exp(-gap/tau), so
a bottleneck state that loses early receives exponentially little gradient and
is never claimed.  tau attacks that by keeping p spread.  Adam's two momenta
attack the same starvation from the other side, and in opposite ways:

    beta1  first-moment decay -- how long a gradient persists in the update.
           Higher beta1 means a state that receives a useful gradient only
           occasionally still accumulates one, because the old signal has not
           decayed away before the next arrives.  PREDICTION: higher helps.

    beta2  second-moment decay -- the window of the per-parameter normaliser.
           Adam divides by sqrt(v), so a starved parameter with tiny gradients
           also has a tiny v, and the ratio is what rescues it.  A SHORTER
           window (lower beta2) re-normalises faster, so a gradient that has
           just appeared is amplified sooner rather than being divided by a
           v still carrying the large gradients of early training.
           PREDICTION: lower helps.

The two are not redundant, and the arms are laid out to keep them separable: one
axis moves beta1 with beta2 fixed, one moves beta2 with beta1 fixed, and two
schedule beta1 in each direction.  The reversed schedule is a control in exactly
the sense study 0's reversed anneal turned out to be -- it is what separates "the
schedule helped" from "any departure from the default helped".

MODIFIES NOTHING in Experimental_pipeline.  Betas are injected by an
instance-level configure_optimizers override plus a callback that rewrites
optimizer.param_groups; see harness/runner.py and harness/schedules.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import run_cell                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

DEFAULT = "adam:0.9:0.999"        # the pipeline optimiser

OPT_ARMS = [
    DEFAULT,                # reference: AdamW's own defaults, what the pipeline uses
    "adam:0.5:0.999",       # ── beta1 axis, beta2 fixed
    "adam:0.95:0.999",
    "adam:0.99:0.999",
    "adam:0.9:0.99",        # ── beta2 axis, beta1 fixed
    "adam:0.9:0.9999",
    "adam_b1:0.5:0.99",     # ── beta1 scheduled low -> high
    "adam_b1:0.99:0.5",     #    control: the same range, reversed
]

CELLS = [
    dict(process="flower", n=2, m=6, mode="forward"),     # V=8   true_k=3
    dict(process="flower", n=6, m=4, mode="backward"),    # V=10  true_k=5
    dict(process="coin", p=0.1, q=0.9, mode="backward"),  # V=3   true_k=3
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau", default="geom:0.5:5",
                    help="the tau schedule to hold fixed (study 1's winner)")
    ap.add_argument("--cell", type=int, default=None)
    ap.add_argument("--opts", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    opts = a.opts or OPT_ARMS
    arms = [(o, a.tau, o) for o in opts]      # arm name == the optimiser spec

    if a.list:
        print(f"\n  study 2: {len(arms)} arms x {len(CELLS)} cells x "
              f"{a.seeds} seeds = {len(arms)*len(CELLS)*a.seeds} runs")
        print(f"  tau held at {a.tau}")
        for i, c in enumerate(CELLS):
            print(f"    --cell {i}   {c}")
        print("\n  arms: " + ", ".join(opts) + "\n")
        return 0

    for c in (CELLS if a.cell is None else [CELLS[a.cell]]):
        c = dict(c)
        mode = c.pop("mode")
        run_cell("2", c.pop("process"), mode, arms, a.seeds, RESULTS,
                 epochs=a.epochs, accelerator=a.accelerator, **c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
