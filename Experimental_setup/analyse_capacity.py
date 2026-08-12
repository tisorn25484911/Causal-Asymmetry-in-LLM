"""
analyse_capacity.py — does |delta_CE| shrink as capacity grows?

    python Experimental_setup/analyse_capacity.py --root All_Results/results_dice_cap
    python Experimental_setup/analyse_capacity.py --root All_Results/results_sweep_cap

Implements REMAINING_WORK_PLAN.md S5.  Reads a directory of per-capacity folders
named d008/, d016/, d032/, d064/ (whatever `--root` contains), and tests the one
prediction the repo's own central argument makes and has never measured.

The prediction, and why it is the tie-breaker
--------------------------------------------
README 1.2 derives that delta_CE is a difference of RESIDUALS, not of entropy
rates.  H_inf is invariant under time reversal, so

    CE_fw -> H_inf + residual_fw       CE_bw -> H_inf + residual_bw
    delta_CE = residual_bw - residual_fw

A predictor with unbounded capacity trained to convergence has both residuals at
zero and therefore delta_CE = 0 REGARDLESS of C- - C+.  A non-zero delta_CE is a
symptom of bounded memory: the direction needing more causal states leaves the
larger residual at fixed d_model and fixed step budget.

That makes two things testable:

    |delta_CE| per process   must SHRINK toward 0 as d_model grows
    b = d(delta_CE)/d(gap)   must SHRINK toward 0 as d_model grows

and it is the only way to break the tie that hangs over every near-null result in
this repo:

    "this process pair is symmetric"            (a result)
    "d_model = 32 was already enough for both"  (no result)

Nothing else measured so far can distinguish those.  sanity_check.py's own
docstring flags the same gap for the positive control.

A flat profile would be the most interesting outcome
----------------------------------------------------
If |delta_CE| does NOT fall with capacity, the residual argument is wrong as
stated, and delta_CE is measuring something other than a memory-bound residual --
an optimisation asymmetry, say, which would not vanish with width.  That would be
a bigger finding than a confirmation, so the analysis reports the profile plainly
rather than testing for the expected sign only.

The design caution this cannot remove
-------------------------------------
QUICK uses lr=1e-2 at d_model=32 and LARGE uses 5e-3 at 64, so the "right"
learning rate scales with width.  Holding lr fixed across capacities -- which is
what makes this a clean capacity manipulation -- means training quality may vary
at the extremes.  So the divergence rate and the convergence step are reported per
capacity: if they drift, the reader can see that the comparison is between
differently-well-trained models, which is a caveat to state rather than a confound
to hide.
"""
# ── stdlib ─────────────────────────────────────────────────────────────────
import argparse
import contextlib
import io
import os
import pickle
import re

# ── third-party ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sstats

# ── repo path bootstrap ────────────────────────────────────────────────────
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── project ────────────────────────────────────────────────────────────────
from Model_analysis import savefig
from run_sweep_experiment import (
    _spearman, global_burn_in, process_row, weighted_slope,
)
from utils import mkdir, repo_path, save_pkl

BUNDLES = ("all_dice.pkl", "all_sweep.pkl", "all_trajectories.pkl")


def find_capacities(root: str) -> list:
    """[(d_model, folder)] for every d<NNN>/ under root that holds a bundle."""
    root = repo_path(root)
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        mt = re.fullmatch(r"d0*(\d+)", name)
        if not mt:
            continue
        folder = os.path.join(root, name)
        if any(os.path.exists(os.path.join(folder, b)) for b in BUNDLES):
            out.append((int(mt.group(1)), folder))
    return sorted(out)


def rows_for(folder: str) -> list:
    """Rows via the sweep's own aggregation, so figures here match the runners'."""
    path = next((os.path.join(folder, b) for b in BUNDLES
                 if os.path.exists(os.path.join(folder, b))), None)
    if path is None:
        return []
    with open(path, "rb") as f:
        combined = pickle.load(f)
    with contextlib.redirect_stdout(io.StringIO()):
        first = [r for r in (process_row(t, rec) for t, rec in combined.items()) if r]
        if not first:
            return []
        burn, _ = global_burn_in(first)
        rows = [r for r in (process_row(t, rec, global_burn=burn)
                            for t, rec in combined.items()) if r]
    return rows


def profile(by_cap: dict) -> list:
    """Per capacity: mean |delta_CE|, the gap slope, and training-health markers."""
    out = []
    for d, rows in sorted(by_cap.items()):
        use = [r for r in rows if np.isfinite(r["dce"])]
        if not use:
            continue
        absd = np.array([abs(r["dce"]) for r in use])
        w = weighted_slope(use, "dce")
        rho, p, n = _spearman([r["gap"] for r in use], [r["dce"] for r in use])
        div = sum(r["n_div"] for r in use)
        reps = sum(r["n_repeats"] for r in use)
        out.append(dict(
            d_model=d, n_proc=len(use),
            mean_abs_dce=float(absd.mean()),
            sem_abs_dce=float(absd.std(ddof=1) / np.sqrt(len(absd))) if len(absd) > 1 else np.nan,
            median_abs_dce=float(np.median(absd)),
            rho=rho, p=p,
            div_rate=div / reps if reps else np.nan,
            mean_settle=float(np.mean([r["burn_settled"] for r in use])),
            **{f"wls_{k}": v for k, v in (w or {}).items()}))
    return out


def trend(prof: list, key: str) -> dict:
    """Spearman of `key` against d_model — the shape of the profile, not a fit."""
    if len(prof) < 3:
        return {}
    x = [p["d_model"] for p in prof]
    y = [p.get(key, np.nan) for p in prof]
    ok = [(a, b) for a, b in zip(x, y) if np.isfinite(b)]
    if len(ok) < 3:
        return {}
    rho, pv = sstats.spearmanr([a for a, _ in ok], [b for _, b in ok])
    return dict(rho=float(rho), p=float(pv), n=len(ok))


def plot_capacity(by_cap: dict, prof: list, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    x = [p["d_model"] for p in prof]
    ax.errorbar(x, [p["mean_abs_dce"] for p in prof],
                yerr=[p["sem_abs_dce"] for p in prof],
                marker="o", capsize=3, lw=1.6, color="#dd8452", label="mean")
    ax.plot(x, [p["median_abs_dce"] for p in prof], marker="s", ls="--",
            color="#4c72b0", label="median")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(r"$d_{\mathrm{model}}$"); ax.set_ylabel(r"$|\Delta CE|$  (bits)")
    ax.set_title("Residual argument: this must fall\n(flat would falsify it)",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)

    ax = axes[1]
    ax.errorbar(x, [p.get("wls_slope", np.nan) for p in prof],
                yerr=[p.get("wls_slope_se", np.nan) for p in prof],
                marker="o", capsize=3, lw=1.6, color="#55a868")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$d_{\mathrm{model}}$")
    ax.set_ylabel(r"slope $b$ of $\Delta CE$ on $C^--C^+$")
    ax.set_title("Sensitivity vs capacity\nshould shrink toward 0", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(x, [p["div_rate"] for p in prof], marker="o", color="#c44e52",
            label="diverged repeat fraction")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$d_{\mathrm{model}}$"); ax.set_ylabel("divergence rate")
    ax2 = ax.twinx()
    ax2.plot(x, [p["mean_settle"] for p in prof], marker="s", ls="--",
             color="grey", label="mean settling step")
    ax2.set_ylabel("mean settling step")
    ax.set_title("Training health — is the comparison\nbetween equally-trained models?",
                 fontsize=10)
    ax.grid(alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")

    fig.suptitle(r"Does $|\Delta CE|$ shrink as capacity grows?  "
                 "The direct test of the residual argument "
                 "(REMAINING_WORK_PLAN S5)", fontsize=12, y=1.03)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "capacity_profile.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="All_Results/results_dice_cap",
                    help="directory holding d008/, d016/, ... subfolders")
    args = ap.parse_args(argv)

    caps = find_capacities(args.root)
    print(f"\n{'='*100}\n  CAPACITY PROFILE   ({repo_path(args.root)})\n{'='*100}")
    if len(caps) < 2:
        raise SystemExit(
            f"found {len(caps)} capacity folder(s) under {repo_path(args.root)}.\n"
            f"    Expected d008/, d016/, d032/, d064/ style subfolders, each with a\n"
            f"    bundle.  Produce them with e.g.\n"
            f"      for D in 8 16 32 64; do\n"
            f"        python Experimental_setup/run_dice_experiment.py --repeats 30 \\\n"
            f"          --d-model $D --out-root {args.root}/d$(printf %03d $D)\n"
            f"      done")

    by_cap = {}
    for d, folder in caps:
        rows = rows_for(folder)
        by_cap[d] = rows
        print(f"  d_model={d:<5} {folder:<52} {len(rows)} processes")

    prof = profile(by_cap)
    print(f"\n{'-'*100}")
    print(f"  {'d_model':>8}{'proc':>6}{'mean|dCE|':>12}{'sem':>10}{'median':>11}"
          f"{'slope b':>11}{'se':>9}{'t':>7}{'rho':>8}{'div rate':>10}{'settle':>8}")
    for p in prof:
        print(f"  {p['d_model']:>8}{p['n_proc']:>6}{p['mean_abs_dce']:>12.5f}"
              f"{p['sem_abs_dce']:>10.5f}{p['median_abs_dce']:>11.5f}"
              f"{p.get('wls_slope', np.nan):>+11.5f}{p.get('wls_slope_se', np.nan):>9.5f}"
              f"{p.get('wls_t', np.nan):>+7.1f}{p['rho']:>+8.3f}"
              f"{p['div_rate']:>10.3f}{p['mean_settle']:>8.1f}")

    t_abs = trend(prof, "mean_abs_dce")
    t_slope = trend(prof, "wls_slope")
    t_div = trend(prof, "div_rate")
    print(f"\n{'-'*100}\n  VERDICT\n{'-'*100}")
    if t_abs:
        print(f"  mean |delta_CE| vs d_model : rho = {t_abs['rho']:+.3f} "
              f"(p = {t_abs['p']:.3f}, {t_abs['n']} capacities)")
        if t_abs["rho"] < -0.5:
            print("     FALLS with capacity -- consistent with the residual argument.")
            print("     Every near-null delta_CE elsewhere in the repo can then be read")
            print("     as 'capacity was sufficient' rather than 'the process is")
            print("     symmetric', which is the ambiguity this experiment exists to fix.")
        elif t_abs["rho"] > 0.5:
            print("     RISES with capacity -- the opposite of the prediction.  Check the")
            print("     divergence column first: larger models at a fixed lr may simply be")
            print("     training worse here.")
        else:
            print("     FLAT.  This FALSIFIES the residual argument as stated: delta_CE is")
            print("     then not a memory-bound residual, and something that does not")
            print("     vanish with width -- an optimisation asymmetry, say -- is what is")
            print("     being measured.  That is a bigger claim than a confirmation would")
            print("     have been, so it needs the divergence column ruled out first.")
    if t_slope:
        print(f"\n  slope b vs d_model         : rho = {t_slope['rho']:+.3f} "
              f"(p = {t_slope['p']:.3f})")
    if t_div:
        print(f"  divergence rate vs d_model : rho = {t_div['rho']:+.3f} "
              f"(p = {t_div['p']:.3f})")
        if abs(t_div["rho"]) > 0.5:
            print("     Training health drifts with capacity, so the profile above is a")
            print("     comparison between differently-well-trained models.  State it as a")
            print("     caveat; lr was held fixed deliberately (see the module docstring).")
    print("\n  Caveat that does not go away: lr is held fixed across capacities, which")
    print("  is what makes this a clean capacity manipulation and also means the")
    print("  extremes may be trained sub-optimally.")
    print(f"{'='*100}")

    out_dir = mkdir(repo_path(args.root))
    plot_capacity(by_cap, prof, out_dir)
    save_pkl({"profile": prof, "trend_abs": t_abs, "trend_slope": t_slope,
              "trend_div": t_div}, os.path.join(out_dir, "capacity_profile.pkl"))


if __name__ == "__main__":
    main()
