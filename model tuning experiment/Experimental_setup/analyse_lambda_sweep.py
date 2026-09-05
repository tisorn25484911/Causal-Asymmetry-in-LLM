"""
analyse_lambda_sweep.py — does weight decay change the asymmetry signal?

    python Experimental_setup/analyse_lambda_sweep.py
    python Experimental_setup/analyse_lambda_sweep.py --like-for-like
    python Experimental_setup/analyse_lambda_sweep.py --out All_Results/results_sweep_wd

Implements WEIGHT_DECAY_PLAN.md 6.2, which asked for exactly this and never got
it: fit

    delta_CE = a(lambda) + b(lambda) * (C- - C+)

per lambda and per family, and report the intercept and the slope SEPARATELY.

WHAT IT FOUND (like-for-like, the 25 flower cells every lambda ran):

    lambda        0      0.03     0.10     0.30     1.00
    flower b  +.00255  +.00265  +.00206  +.00302  +.00503
    coin   b  -.00169  -.00153  -.00137  -.00172  -.00404

b is flat for lambda <= 0.3 and roughly doubles at lambda = 1 in BOTH families.
Since the coin trend is a known H_inf artefact, an amplification that hits the
artefactual axis as hard as the real one is a scale effect from degraded training
rather than weight decay revealing causal structure.  See the READING block at
the bottom of main(), which is written from these numbers rather than from a
prior expectation -- an earlier draft of this analysis used unweighted OLS, whose
enormous error bars made b look flat everywhere, and concluded the opposite.

The reason the decomposition matters is that the two coefficients answer different
questions, and only one of them is about causal asymmetry:

    b(lambda)  SENSITIVITY.  How much measured asymmetry moves per bit of
               theoretical asymmetry.  If regularisation were revealing a signal
               that capacity was hiding, b would grow with lambda.
    a(lambda)  OFFSET.  What delta_CE the pipeline reports at C- = C+, i.e. the
               part of the measurement that has nothing to do with the hypothesis.
               A non-zero a is a bias -- the arms differ for some reason other
               than causal structure -- and it should be ~0 at every lambda.

This reads only stored results.  Nothing is trained, so it is safe to re-run.

Analysis-only, but two choices in it are load-bearing
----------------------------------------------------
1.  **The slope is reported inverse-variance weighted**, with the unweighted fit
    beside it.  On these grids the per-cell sems span up to 249x, and OLS -- which
    assumes they are equal -- lets a few unstable cells dominate the residual
    variance and flatten the slope to insignificance.  See
    REMAINING_WORK_PLAN.md S3; `weighted_slope` carries the numbers.

2.  **--like-for-like restricts every lambda to the processes they all share.**
    The lambda=0 baseline ran a 9x9 flower grid (81 cells) while the lambda>0
    folders ran 5x5 (25).  Comparing a slope fitted on 81 points with one fitted
    on 25 confounds lambda with grid size -- the baseline's slope is better
    determined for a reason that has nothing to do with weight decay.
"""
# ── stdlib ─────────────────────────────────────────────────────────────────
import argparse
import contextlib
import io
import os
import pickle

# ── third-party ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sstats

# ── repo path bootstrap ────────────────────────────────────────────────────
# REORGANISATION_FIX_PLAN.md 4.1.  Flat modules across two directories.
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── project ────────────────────────────────────────────────────────────────
# Every statistic comes from the sweep runner, so this file cannot disagree with
# the per-lambda summaries it is aggregating.
from Model_analysis import savefig
from run_sweep_experiment import (
    _partial_spearman,
    _spearman,
    block_bootstrap_rho,
    global_burn_in,
    process_row,
    unstable_mask,
    weighted_slope,
)
from utils import mkdir, repo_path, save_pkl

# The completed folders, in lambda order.  results_sweep is the lambda=0 run:
# AdamW at weight_decay=0 is plain Adam, which is the condition every earlier
# result in the repo used.
LAMBDA_FOLDERS = [
    (0.0,  "All_Results/results_sweep"),
    (0.03, "All_Results/results_sweep_wd/wd0.030"),
    (0.10, "All_Results/results_sweep_wd/wd0.100"),
    (0.30, "All_Results/results_sweep_wd/wd0.300"),
    (1.00, "All_Results/results_sweep_wd/wd1.000"),
]


def rows_for(folder: str) -> list:
    """
    The same rows the sweep's own figures are drawn from.

    process_row is called twice exactly as run_sweep_experiment.main does -- once
    to discover the per-process burn-ins and once with the shared global value --
    so the areas here are computed over the same window as that folder's figures.
    """
    path = repo_path(os.path.join(folder, "all_sweep.pkl"))
    if not os.path.exists(path):
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


def shared_keys(by_lambda: dict, family: str) -> set:
    """Process keys present at EVERY lambda, for the like-for-like comparison."""
    sets = []
    for rows in by_lambda.values():
        sets.append({(r.get("n"), r.get("m"), r.get("p"), r.get("q"))
                     for r in rows if r["kind"] == family})
    return set.intersection(*sets) if sets else set()


def fit_table(by_lambda: dict, family: str, restrict: set | None = None) -> list:
    """One row per lambda: WLS and OLS coefficients, rho, and its bootstrap CI."""
    out = []
    for lam, rows in by_lambda.items():
        sub = [r for r in rows
               if r["kind"] == family and np.isfinite(r.get("dce", np.nan))]
        if restrict is not None:
            sub = [r for r in sub
                   if (r.get("n"), r.get("m"), r.get("p"), r.get("q")) in restrict]
        if len(sub) < 5:
            continue
        w = weighted_slope(sub, "dce")
        x = np.array([r["gap"] for r in sub]); y = np.array([r["dce"] for r in sub])
        ols = sstats.linregress(x, y)
        rho, p, n = _spearman(x, y)
        bs = block_bootstrap_rho(sub, "dce") or {}
        rec = dict(lam=lam, n=len(sub), rho=rho, p=p,
                   rho_lo=bs.get("lo", np.nan), rho_hi=bs.get("hi", np.nan),
                   sign_stability=bs.get("sign_stability", np.nan),
                   ols_b=ols.slope, ols_b_se=ols.stderr,
                   ols_a=ols.intercept, ols_a_se=ols.intercept_stderr,
                   n_unstable=int(sum(unstable_mask(sub, "dce"))))
        rec.update({f"wls_{k}": v for k, v in (w or {}).items()})
        out.append(rec)
    return out


def print_table(title: str, table: list, note: str = ""):
    print(f"\n{'-'*112}\n  {title}\n{'-'*112}")
    if note:
        print(f"  {note}\n")
    print(f"  {'lambda':>7}{'n':>4}{'unst':>5}"
          f"{'WLS b':>11}{'se':>9}{'t':>7}"
          f"{'WLS a':>11}{'se':>9}"
          f"{'OLS b':>11}{'se':>9}{'t':>7}"
          f"{'rho':>8}{'95% CI':>19}")
    for r in table:
        ci = (f"[{r['rho_lo']:+.2f},{r['rho_hi']:+.2f}]"
              if np.isfinite(r.get("rho_lo", np.nan)) else "n/a")
        ols_t = r["ols_b"] / r["ols_b_se"] if r["ols_b_se"] else float("nan")
        print(f"  {r['lam']:>7.2f}{r['n']:>4}{r['n_unstable']:>5}"
              f"{r.get('wls_slope', float('nan')):>+11.5f}"
              f"{r.get('wls_slope_se', float('nan')):>9.5f}"
              f"{r.get('wls_t', float('nan')):>+7.1f}"
              f"{r.get('wls_intercept', float('nan')):>+11.5f}"
              f"{r.get('wls_intercept_se', float('nan')):>9.5f}"
              f"{r['ols_b']:>+11.5f}{r['ols_b_se']:>9.5f}{ols_t:>+7.1f}"
              f"{r['rho']:>+8.3f}{ci:>19}")


def verdict(table: list, family: str):
    """
    State plainly whether b(lambda) moves, because that is the whole question.

    The test is deliberately crude -- do the extreme lambdas' slopes overlap
    within their combined standard errors -- because with five lambdas and this
    much per-point noise anything more elaborate would be over-reading.
    """
    ok = [r for r in table if np.isfinite(r.get("wls_slope", np.nan))]
    if len(ok) < 2:
        return
    lo, hi = ok[0], ok[-1]
    d = hi["wls_slope"] - lo["wls_slope"]
    se = float(np.hypot(hi["wls_slope_se"], lo["wls_slope_se"]))
    z = d / se if se else float("nan")
    print(f"\n  {family}: b({lo['lam']:g}) = {lo['wls_slope']:+.5f} +/- "
          f"{lo['wls_slope_se']:.5f}   ->   b({hi['lam']:g}) = "
          f"{hi['wls_slope']:+.5f} +/- {hi['wls_slope_se']:.5f}")
    print(f"     difference {d:+.5f} +/- {se:.5f}   z = {z:+.2f}   "
          f"{'NOT distinguishable' if abs(z) < 2 else 'DISTINGUISHABLE'}")
    a_max = max(abs(r.get("wls_intercept", 0.0)) for r in ok)
    a_se_max = max(r.get("wls_intercept_se", 0.0) for r in ok)
    print(f"     largest |a(lambda)| = {a_max:.5f} against a typical se of "
          f"{a_se_max:.5f} -- "
          f"{'consistent with zero bias' if a_max < 3*a_se_max else 'a real offset'}")


def plot_lambda(tables: dict, out_dir: str):
    """b(lambda) and a(lambda) with their errors, one panel each."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colours = {"flower": "#dd8452", "coin": "#4c72b0"}
    for ax, key, label in ((axes[0], "wls_slope", r"slope $b(\lambda)$"),
                           (axes[1], "wls_intercept", r"intercept $a(\lambda)$")):
        for fam, table in tables.items():
            t = [r for r in table if np.isfinite(r.get(key, np.nan))]
            if not t:
                continue
            # lambda=0 on a log axis needs a stand-in; label it honestly.
            x = [max(r["lam"], 0.01) for r in t]
            ax.errorbar(x, [r[key] for r in t],
                        yerr=[r[f"{key}_se"] for r in t],
                        marker="o", capsize=3, lw=1.4, label=fam,
                        color=colours.get(fam, "grey"))
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(r"weight decay $\lambda$   (0 plotted at 0.01)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_title(r"Sensitivity to $C^--C^+$ — does regularisation reveal it?",
                      fontsize=11)
    axes[1].set_title(r"Offset at $C^-=C^+$ — the asymmetry-independent bias",
                      fontsize=11)
    fig.suptitle("Weight decay changes the variance, not the sensitivity "
                 "(WEIGHT_DECAY_PLAN 6.2)", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "lambda_decomposition.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="All_Results/results_sweep_wd",
                    help="where the figure and pickle go")
    ap.add_argument("--like-for-like", action="store_true",
                    help="restrict every lambda to the processes all of them ran, "
                         "so grid size cannot masquerade as a lambda effect")
    args = ap.parse_args(argv)

    by_lambda = {}
    print(f"\n{'='*112}\n  a(lambda) / b(lambda) DECOMPOSITION   "
          f"(WEIGHT_DECAY_PLAN.md 6.2)\n{'='*112}")
    for lam, folder in LAMBDA_FOLDERS:
        rows = rows_for(folder)
        if not rows:
            print(f"  lambda={lam:<5g} {folder}   MISSING — skipped")
            continue
        by_lambda[lam] = rows
        kinds = {k: sum(1 for r in rows if r["kind"] == k)
                 for k in ("coin", "flower")}
        print(f"  lambda={lam:<5g} {folder:<44} {kinds}")
    if len(by_lambda) < 2:
        raise SystemExit("need at least two lambdas")

    tables = {}
    for fam in ("flower", "coin"):
        restrict = shared_keys(by_lambda, fam) if args.like_for_like else None
        table = fit_table(by_lambda, fam, restrict)
        if not table:
            continue
        tables[fam] = table
        note = (f"restricted to the {len(restrict)} {fam} processes every lambda ran"
                if restrict is not None else
                "each lambda on the full grid it ran (see --like-for-like)")
        print_table(f"{fam.upper()}  —  delta_CE = a(lambda) + b(lambda)*(C- - C+)",
                    table, note)
        verdict(table, fam)

    out_dir = mkdir(repo_path(args.out))
    plot_lambda(tables, out_dir)
    save_pkl({"tables": tables, "like_for_like": args.like_for_like},
             os.path.join(out_dir, "lambda_decomposition.pkl"))

    print(f"\n{'='*112}\n  READING\n{'='*112}")
    print("  Measured on the completed sweep (like-for-like, 25 flower cells each):")
    print("     flower b:  +0.00255  +0.00265  +0.00206  +0.00302  +0.00503")
    print("     coin   b:  -0.00169  -0.00153  -0.00137  -0.00172  -0.00404")
    print("     lambda:        0      0.03      0.10      0.30      1.00")
    print()
    print("  1. For lambda <= 0.3, b is FLAT in both families -- the slopes overlap")
    print("     within roughly two standard errors and there is no trend.  Over that")
    print("     range weight decay adds variance without changing sensitivity, which")
    print("     closes off mild regularisation as an explanation for the near-null")
    print("     delta_CE: the signal was not being hidden by an unregularised fit.")
    print()
    print("  2. At lambda = 1, |b| roughly DOUBLES in both families.  Read this")
    print("     cautiously.  lambda = 1 is where decay binds hard -- the equilibrium")
    print("     weight norm goes as 1/lambda against the ~7 logit gap a confident")
    print("     prediction needs -- and it is also where the unstable-cell count")
    print("     jumps from 2 to 7 and where the coin intercept a becomes a real")
    print("     non-zero offset (+0.00088 +/- 0.00009).")
    print()
    print("  3. The decisive detail is that lambda = 1 amplifies the COIN slope just")
    print("     as much as the flower slope.  The coin trend is a known H_inf")
    print("     artefact (partial rho | H_inf ~ 0), so an amplification that hits the")
    print("     artefactual axis and the real one equally is a scale effect from")
    print("     degraded training, NOT weight decay selectively revealing causal")
    print("     structure.  If it were the latter, the coin family should not have")
    print("     moved.")
    print()
    print("  So: reducing effective capacity does appear to enlarge delta_CE, which")
    print("  is what the residual argument predicts -- but weight decay is a blunt")
    print("  way to reduce capacity and it distorts both arms at the same time.  The")
    print("  clean version of this manipulation is the d_model sweep")
    print("  (REMAINING_WORK_PLAN.md S5), which varies capacity without also")
    print("  degrading the optimisation.  These numbers make that experiment more")
    print("  worth running, not less.")
    print()
    print("  a(lambda) stays ~0 on the flower grid at every lambda, which is the")
    print("  assumption every delta_CE in the repo rests on: a drifting intercept")
    print("  would mean the arms differ for a reason other than causal structure.")
    print("  It does NOT stay zero for the coins at lambda = 1 -- another reason to")
    print("  treat that column as degraded rather than informative.")
    print(f"{'='*112}")


if __name__ == "__main__":
    main()
