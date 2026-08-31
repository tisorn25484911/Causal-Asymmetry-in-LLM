"""
plot_sweep_combined.py — the pq and nm sweeps on ONE pair of axes.

    python Experimental_setup/plot_sweep_combined.py
    python Experimental_setup/plot_sweep_combined.py --roots A B --out-root C

DISCRETE_V2_PLAN.md splits what used to be a single 50-process sweep into two
folders, 08_pqsweep (25 coin) and 09_nmsweep (25 flower), so that either can be
rerun without the other.  The hypothesis, though, is a statement about the SIGN
of delta_CE across the whole C- - C+ range, and only the flower family reaches
negative C- - C+ -- so the test is only legible with both families on one axis.
This merges the stored records and redraws.

Trains nothing.  Reads each folder's all_sweep.pkl, rebuilds the rows through
the same two-pass process_row the sweep runner uses, and writes:

    sweep_combined_delta_ce.png   the single panel -- delta_CE vs C- - C+
    sweep_scatter.png             the standard four panels, both families
    sweep_trajectories.png        the loss curves, both families

Why the two-pass row build is repeated here rather than shortcut: `area` is
integrated over a burn-in window chosen from the whole row set, so rows built
from one family alone are NOT comparable with rows built from both.  Pass one
discovers the window, pass two applies it.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "Transformer_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402

from Model_analysis import savefig                  # noqa: E402
from run_sweep_experiment import (                  # noqa: E402
    UNSTABLE_SEM_FACTOR, _spearman, global_burn_in, plot_sweep_scatter,
    plot_sweep_trajectories, process_row, unstable_mask,
)
from utils import mkdir, repo_path                  # noqa: E402

_DEFAULT_ROOTS = ["All_Results/discrete_v2/08_pqsweep",
                  "All_Results/discrete_v2/09_nmsweep"]

# One colour per family.  Colour carries the FAMILY, not the convergence state:
# the four-panel figure already encodes divergence in its colormap, and doing
# both at once in a single panel is what made it unreadable.
_STYLE = {
    "coin":   dict(colour="#4c72b0", marker="o", label="coin  (p,q) sweep"),
    "flower": dict(colour="#dd8452", marker="s", label="flower  (n,m) sweep"),
}


def load_rows(roots):
    """Merged rows from every sweep folder, built on one shared burn-in window."""
    import pickle
    combined, missing = {}, []
    for r in roots:
        path = os.path.join(r if os.path.isabs(r) else repo_path(r), "all_sweep.pkl")
        if not os.path.isfile(path):
            missing.append(path)
            continue
        with open(path, "rb") as fh:
            for tag, rec in pickle.load(fh).items():
                if rec.get("runs"):
                    combined[tag] = rec
    if missing:
        for p in missing:
            print(f"  ! not found, skipped: {p}")
    if not combined:
        return [], combined

    rows = [x for x in (process_row(t, r) for t, r in combined.items()) if x]
    if not rows:
        return [], combined
    B, capped = global_burn_in(rows)
    rows = [x for x in (process_row(t, r, global_burn=B)
                        for t, r in combined.items()) if x]
    print(f"  shared burn-in window: step {B}"
          + ("  ! CAPPED — some process never settled" if capped else ""))
    return rows, combined


def plot_combined_delta_ce(rows, out_root, fname="sweep_combined_delta_ce.png"):
    """
    delta_CE against C- - C+, both families on one axis.

    Hollow marker = at least one repeat was lost to divergence, so the mean is
    over a self-selected subset and the point is worth less than a solid one.
    A process where NO repeat converged has no finite dce_conv and is dropped
    entirely -- it is an optimisation outcome, not a measurement.  Error bars
    are +-1 sem over the converged repeats.
    """
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.axhline(0.0, color="k", ls="--", lw=0.9)
    ax.axvline(0.0, color="k", ls="--", lw=0.9)

    excluded = []
    for kind, st in _STYLE.items():
        fam = [r for r in rows if r["kind"] == kind
               and np.isfinite(r["gap"]) and np.isfinite(r["dce_conv"])]
        if not fam:
            continue
        # Same exclusion the sweep runner applies, computed PER FAMILY: the
        # threshold is relative to that family's own median sem, and the two
        # medians differ by about 2x, so pooling them would exclude a different
        # set from the one 08/09 each reported.
        mask = unstable_mask(fam, "dce_conv")
        sub = [r for r, bad in zip(fam, mask) if not bad]
        excluded += [r["tag"].replace("sweep_flower_", "").replace("sweep_coin_", "")
                     for r, bad in zip(fam, mask) if bad]
        rho, p, n = _spearman([r["gap"] for r in sub],
                              [r["dce_conv"] for r in sub])
        rho_a, p_a, n_a = _spearman([r["gap"] for r in fam],
                                    [r["dce_conv"] for r in fam])
        for r in sub:
            solid = r["n_conv"] == r["n_repeats"]
            ax.errorbar(r["gap"], r["dce_conv"],
                        yerr=(r["dce_conv_sem"] if np.isfinite(r["dce_conv_sem"])
                              else None),
                        marker=st["marker"], ms=6, lw=0, elinewidth=1.0,
                        capsize=2, ecolor=st["colour"],
                        mfc=st["colour"] if solid else "none",
                        mec=st["colour"], mew=1.3)
        # One proxy artist per family, so the legend has two entries not fifty.
        # Both rho are shown: dropping the noisy points is a defensible choice
        # but it is a choice, and it moves the number a lot here.
        ax.plot([], [], marker=st["marker"], ls="none", ms=6,
                mfc=st["colour"], mec=st["colour"],
                label=f"{st['label']}   n={n}, rho={rho:+.3f} (p={p:.3f})"
                      + (f"   |  all {n_a}: rho={rho_a:+.3f} (p={p_a:.3f})"
                         if n_a != n else ""))

    n_hollow = sum(r["n_conv"] != r["n_repeats"] for r in rows
                   if np.isfinite(r.get("dce_conv", np.nan)))
    n_dropped = sum(not np.isfinite(r.get("dce_conv", np.nan)) for r in rows)
    ax.plot([], [], marker="o", ls="none", ms=6, mfc="none", mec="0.35",
            label=f"hollow = some repeats lost to divergence ({n_hollow})"
                  + (f";  {n_dropped} process(es) dropped, none converged"
                     if n_dropped else ""))
    ax.set_xlabel(r"C^- - C^+ (bits)")
    ax.set_ylabel(r"\Delta CE = CE_BW - CE_FW (bits/symbols)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    savefig(fig, os.path.join(out_root, fname))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=_DEFAULT_ROOTS,
                    help="sweep folders to merge (each holding all_sweep.pkl)")
    ap.add_argument("--out-root", default="All_Results/discrete_v2/sweep_combined",
                    help="where the merged figures go")
    ap.add_argument("--panel-only", action="store_true",
                    help="only the single combined panel, not the four-panel set")
    a = ap.parse_args(argv)

    rows, combined = load_rows(a.roots)
    if not rows:
        print("  nothing to plot — no sweep records found yet.")
        return 1

    out_root = a.out_root if os.path.isabs(a.out_root) else repo_path(a.out_root)
    mkdir(out_root)
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        gaps = [r["gap"] for r in sub if np.isfinite(r["gap"])]
        print(f"  {kind:<7} {len(sub):>3} processes   "
              f"C− − C+ in [{min(gaps):+.3f}, {max(gaps):+.3f}]")

    plot_combined_delta_ce(rows, out_root)
    if not a.panel_only:
        plot_sweep_scatter(rows, out_root)
        plot_sweep_trajectories(rows, out_root)
    print(f"\n  {len(rows)} processes merged -> {out_root}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
