#!/usr/bin/env python3
"""Print the best tau arms by the pooled paired statistic, best first.

    python pick_winner.py           -> "geom:0.5:5 geom:0.5:10"
    python pick_winner.py --n 1     -> "geom:0.5:5"

Used by run_all.sh to hand study 2 the schedule study 1 chose, so the choice is
made by the data rather than typed in by hand afterwards.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyse import REF, paired_pool                             # noqa: E402
from harness.report import load                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--study-only", action="store_true",
                help="rank on study 1's own cells, ignoring the pilot")
a = ap.parse_args()

dirs = [os.path.join(HERE, "results"), os.path.join(HERE, "results_1A")]
if not a.study_only:
    dirs.append(os.path.join(os.path.dirname(HERE), "study0_pilot", "results"))
cells = load(*dirs)
pool = paired_pool(cells, "abs_dS")
best = [k for k, _ in sorted(pool.items(), key=lambda kv: kv[1][0]) if k != REF]
print(" ".join(best[:a.n]) if best else "geom:0.5:5")
