#!/usr/bin/env python3
"""
The final loss gap, for every training in the tau test.

    CE - H_inf   the trained model's test cross-entropy minus the process
                 entropy rate -- the value CE converges to when the model has
                 learned the process.  Zero is perfect; conv_tol is 0.10.

Every cell here is a REPEATS run, so each arm is reported as mean +- sample
standard deviation (ddof=1) over its seeds, with the worst single seed beside
it -- a mean near tolerance can still hide a collapsed run, which is the whole
reason the worst-seed column is there.

Reads the saved JSON, so it re-reports without retraining.  A cell whose runner
was stopped before it wrote JSON is read from its log and marked PARTIAL.
"""
import sys

import numpy as np

from plot_tau_results import ARMS, LABEL, load_cells, _title


def _sd(v):
    return v.std(ddof=1) if len(v) > 1 else 0.0


def main():
    cells = load_cells()
    if not cells:
        sys.exit("no results found -- run tau_anneal_test.py first")

    print(f"\n{'='*104}")
    print("  FINAL LOSS GAP TO THE ENTROPY RATE     CE - H_inf, bits, mean +- sd "
          "over seeds")
    print("  Zero is perfect.  A repeat is converged when |CE - H_inf| <= 0.10 "
          "(conv_tol).")
    print(f"{'='*104}")

    pooled = {a: [] for a in ARMS}
    for c in cells:
        H = c["runs"][0]["ce"] - c["runs"][0]["ce_minus_H"]
        print(f"\n  {_title(c)}     H_inf = {H:.4f}     n = {len(c['seeds'])} seeds")
        print(f"    {'arm':<24}{'CE':>19}{'CE - H_inf':>21}"
              f"{'worst seed':>13}{'converged':>12}")
        print(f"    {'-'*87}")
        for a in ARMS:
            sub = [r for r in c["runs"] if r["arm"] == a]
            ce = np.array([r["ce"] for r in sub])
            d  = np.array([r["ce_minus_H"] for r in sub])
            pooled[a].extend(d.tolist())
            worst = d[np.argmax(np.abs(d))]
            print(f"    {LABEL[a]:<24}{ce.mean():>10.4f} +- {_sd(ce):<6.4f}"
                  f"{d.mean():>+13.4f} +- {_sd(d):<6.4f}{worst:>+13.4f}"
                  f"{sum(r['within_tol'] for r in sub):>8}/{len(sub)}")
        print(f"    {'-'*87}")

    print(f"\n{'='*104}")
    print(f"  POOLED over every cell and seed")
    print(f"    {'arm':<24}{'CE - H_inf':>21}{'worst run':>13}{'converged':>12}"
          f"{'collapsed':>12}")
    print(f"    {'-'*82}")
    allruns = [r for c in cells for r in c["runs"]]
    for a in ARMS:
        d = np.array(pooled[a])
        sub = [r for r in allruns if r["arm"] == a]
        print(f"    {LABEL[a]:<24}{d.mean():>+13.4f} +- {_sd(d):<6.4f}"
              f"{d[np.argmax(np.abs(d))]:>+13.4f}"
              f"{sum(r['within_tol'] for r in sub):>8}/{len(sub)}"
              f"{sum(r['k_occ'] == 1 for r in sub):>8}/{len(sub)}")
    print(f"    {'-'*82}")
    print("\n  The spread is the point.  An arm whose mean sits inside conv_tol can "
          "still\n  carry a collapsed seed a full bit away -- read the worst-seed "
          "and collapsed\n  columns beside the mean, never the mean alone.\n")


if __name__ == "__main__":
    main()
