"""
replot_complexity.py — redraw the complexity figures from stored records.

    python Experimental_setup/replot_complexity.py --dry-run
    python Experimental_setup/replot_complexity.py
    python Experimental_setup/replot_complexity.py All_Results/discrete_v2

Why this exists
---------------
`<tag>_complexity.png` and `summary_grid.png` were drawn from the k-means
`S_emp` for every architecture, including the discrete one, where it measures
the pre-bottleneck latent at a k taken from the ground truth rather than the
bottleneck itself.  At n_states=1 that reports ~0.71 bits where the true
complexity is exactly 0.

Both estimators are stored on every arm of every repeat, so the fix is a
REDRAW, not a rerun -- no model is loaded and no training happens here.  The
continuous records carry no `S_emp_states`, so they redraw identically; they
are included so that the whole tree is provably on one code path.

Trains nothing, and writes only files matching the two names above.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "Transformer_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                      # noqa: E402
matplotlib.use("Agg")

from Model_analysis import resolve_s_emp            # noqa: E402
from run_statistical_trj import (                   # noqa: E402
    _stats, plot_complexity, plot_grid_summary,
)
from utils import repo_path                         # noqa: E402


def _load(path):
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)


def find_records(root: str):
    """Every results.pkl under root that carries per-repeat runs, by folder."""
    out = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        if "results.pkl" not in filenames:
            continue
        rec = _load(os.path.join(dirpath, "results.pkl"))
        if not isinstance(rec, dict) or not rec.get("runs") or "spec" not in rec:
            continue
        out.setdefault(os.path.dirname(dirpath), {})[rec["spec"]["tag"]] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["All_Results"],
                    help="directories to walk (default: All_Results)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--no-grid", action="store_true",
                    help="skip summary_grid.png")
    a = ap.parse_args(argv)

    n_fig = n_moved = 0
    print(f"\n  {'folder':<44} {'process':<24} {'estimator':>14} {'fw':>8} {'bw':>8}")
    print("  " + "-" * 102)

    for root in a.roots:
        root = root if os.path.isabs(root) else repo_path(root)
        if not os.path.isdir(root):
            print(f"  skip (not a directory): {root}")
            continue

        for out_dir, recs in sorted(find_records(root).items()):
            shown = os.path.relpath(out_dir, _ROOT).replace("All_Results/", "")
            for tag, rec in sorted(recs.items()):
                key = resolve_s_emp(rec["runs"])
                fw = _stats([r["fw"][key] for r in rec["runs"]])["mean"]
                bw = _stats([r["bw"][key] for r in rec["runs"]])["mean"]
                # What the old code would have drawn, to show what moves.
                old_fw = _stats([r["fw"]["S_emp"] for r in rec["runs"]])["mean"]
                moved = abs(fw - old_fw) > 1e-9
                n_moved += moved
                print(f"  {shown[:44]:<44} {tag[:24]:<24} {key:>14} "
                      f"{fw:>8.4f} {bw:>8.4f}{'   <- moved' if moved else ''}")
                if not a.dry_run:
                    plot_complexity(rec, os.path.join(_ROOT, out_dir, tag))
                    n_fig += 1
            if not (a.dry_run or a.no_grid) and len(recs) > 1:
                plot_grid_summary(recs, os.path.join(_ROOT, out_dir))
                n_fig += 1

    verb = "would redraw" if a.dry_run else "redrew"
    print(f"\n  {verb} {n_fig} figure(s); {n_moved} process-arm set(s) change value.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
