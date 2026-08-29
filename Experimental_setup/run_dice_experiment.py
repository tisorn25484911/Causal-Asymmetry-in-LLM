"""
run_dice_experiment.py — does delta_CE track C- - C+, or does it track m - n?

    python Experimental_setup/run_dice_experiment.py --dry-run
    python Experimental_setup/run_dice_experiment.py --repeats 30       # ~2.5 h
    python Experimental_setup/run_dice_experiment.py --plots-only

Implements REMAINING_WORK_PLAN.md S1.  Everything that trains, measures or
aggregates a single process is IMPORTED from run_statistical_trj.py and
run_sweep_experiment.py; this file only decides WHICH processes to run.

The problem it exists to solve
-----------------------------
The 81-cell flower sweep reports rho = +0.827 between measured delta_CE and the
theoretical gap C- - C+, with a weighted slope of +0.00294 +/- 0.00016 (t = 19).
That looks decisive and it is not, because on that grid

    corr(C- - C+,  m - n)  =  +0.977      (spearman +0.986)

so "delta_CE tracks causal asymmetry" and "delta_CE tracks the excess of faces
over dice" predict the same data.  Any nuisance mechanism that scales with m - n
-- token-frequency imbalance, how often each die is revisited, the ratio of
selection to outcome tokens -- reproduces the entire result.  No number of extra
repeats on that grid can separate the two, because the confound is in the DESIGN.

The way out
-----------
C- for the flower process is

    C- = 1 + H(pi_outcome) / 2,     pi_outcome_j = (1/n) sum_i dice_probs[i, j]

so the gap C- - C+ = [H(pi_outcome) - log2 n] / 2 depends on how much the n dice
OVERLAP in their support -- not only on (n, m).  Hold (n, m) fixed and vary the
dice, and every nuisance variable is pinned by construction:

    m - n            identical
    vocabulary n+m   identical
    C+ = 1+log2(n)/2 identical
    forward states   identical
    sequence length, token budget, capacity, step budget   identical

while C- - C+ still moves.  Measured with flower_complexity, at Dirichlet
alpha = 0.2:

    (n,m)=(2,6)   gap spans -0.034 .. +0.721      35% of the whole 5x5 grid range
    (n,m)=(2,8)   gap spans -0.051 .. +0.886      44%
    (n,m)=(4,6)   gap spans -0.331 .. +0.269      28%

Three of those four CROSS ZERO, so the sign of the prediction flips inside a
single (n, m) cell.  That is a controlled sign test -- the only thing the theory
actually predicts -- and the existing design cannot perform it anywhere.

H_inf is controlled too, which is the confound that ruined the coin family
------------------------------------------------------------------------------
H_inf for the flower is [log2 n + mean_i H(dice_i)] / 2, a different functional of
the dice from H(pi_outcome), so the two can be varied independently.  Over 4000
draws at (2,8), alpha = 0.35:

    H_inf overall                     0.7555 .. 1.8482
    inside the band 1.4484 .. 1.4662  gap spans +0.5109 .. +0.9041   (0.39 bits)
        corr(gap, H_inf) inside band  +0.061
        corr(gap, H_inf) over all     +0.772

So selecting within a narrow H_inf band keeps 0.39 bits of gap variation while
removing the entropy-rate confound.  `--hinf-band` does that selection.

What a result would mean
------------------------
  * within-cell slope ~ the between-cell slope (+0.0029)
        the confounds are exhausted; the trend is the theoretical gap.
  * within-cell slope ~ 0 while the 81-cell grid still shows rho = +0.83
        the flower trend was m - n all along.
  * sign of delta_CE follows sign of gap inside one (n, m)
        the strongest available outcome: a controlled sign test.

Two implementation notes that matter
------------------------------------
1.  **The tag carries the dice seed.**  run_sweep_experiment re-tags flower cells
    as flower_tag("sweep", n, m), which is a FUNCTION OF (n, m) ALONE -- twenty
    dice realisations at (2,8) would all collide on one key and, because the
    result store is keyed by tag, nineteen would be silently overwritten.  A
    distinct `dice_` prefix also keeps these cells from colliding with sweep or
    baseline processes in a shared pickle.

2.  **The merge rule is recorded, not assumed.**  At small alpha, near-zero
    outcome masses make C- sensitive to whether two vanishingly-rare outcomes
    count as one backward state: at (2,8) alpha=0.2 seed 74 the historical
    round-to-9-decimals rule gives C- = 2.1041 and a 1e-9 tolerance gives 2.0748.
    This experiment uses the historical default so its x-axis is comparable with
    the 81-cell grid, and writes `merge_tol` into the run config.  See
    Model_analysis.flower_complexity and REMAINING_WORK_PLAN.md C10.
"""
# ── stdlib ─────────────────────────────────────────────────────────────────
import argparse
import contextlib
import io
import os
import time

# ── third-party ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sstats

# ── repo path bootstrap ────────────────────────────────────────────────────
# REORGANISATION_FIX_PLAN.md 4.1.  Flat modules across two sibling directories.
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── project ────────────────────────────────────────────────────────────────
from configs import CONFIGS
from Model_analysis import flower_complexity, flower_entropy_rate, savefig
from run_statistical_trj import (
    _stats, flower_spec, load_combined, one_repeat, paired_stats, quiet,
)
from run_sweep_experiment import (
    _spearman, global_burn_in, process_row, unstable_mask, weighted_slope,
)
from utils import mkdir, repo_path, save_pkl, save_run_config

OUT_ROOT_DEFAULT = "All_Results/results_dice"
SAVE_EVERY_S = 60.0

# Spiky dice are what open up the gap range: alpha < 1 concentrates each die on a
# few faces, so how much the dice overlap -- and therefore H(pi_outcome) -- varies
# a lot between draws.
#
# Cells and alpha chosen by surveying (n,m) x alpha for the widest gap range that (a) fits inside a
# +/-1% H_inf band and (b) contains BOTH SIGNS, since a controlled sign test is the
# strongest outcome available.  Measured over 3000 draws per combination:
#
#   (n,m)=(3,5) alpha=0.2   gap [-0.206,+0.358]  span 0.564   corr(gap,H_inf) -0.002
#   (n,m)=(4,6) alpha=0.2   gap [-0.215,+0.270]  span 0.485   corr(gap,H_inf) +0.027
#
# against +0.772 unrestricted at (2,8) -- i.e. the band removes essentially all of
# the entropy-rate confound.  alpha=0.2 is spiky enough to reach negative gaps; it
# is also the regime where C- can become sensitive to the merge rule, so that was
# checked for these exact dice and the two rules agree EXACTLY (max difference
# 0.0e+00 over the selected realisations at both cells).
ALPHA_DEFAULT = 0.2
CELLS_DEFAULT = "3,5 4,6"


# ══════════════════════════════════════════════════════════════════════════
# SELECTING THE DICE
# ══════════════════════════════════════════════════════════════════════════
def survey(n: int, m: int, alpha: float, n_search: int) -> list:
    """(dice_seed, gap, H_inf) for n_search candidate realisations. No training."""
    out = []
    for seed in range(n_search):
        dice = np.random.default_rng(seed).dirichlet(np.full(m, alpha), size=n)
        cp, cm = flower_complexity(n, m, dice)
        out.append((seed, cm - cp, flower_entropy_rate(n, m, dice)))
    return out


def select_dice(n: int, m: int, alpha: float, n_dice: int, n_search: int,
                hinf_band: float, min_negative: int) -> tuple:
    """
    Choose `n_dice` realisations inside a narrow H_inf band, spanning the widest
    gap range that band allows.

    Returns (chosen, diagnostics).  The selection is deterministic given the
    arguments, so the experiment is reproducible from the run config alone.

    The band is centred on the MEDIAN H_inf rather than on a target value: the
    achievable H_inf range depends on (n, m) and alpha, and hard-coding a centre
    would silently return nothing for some cells.  Width is relative, so the same
    --hinf-band means the same fractional control at every cell.
    """
    cand = survey(n, m, alpha, n_search)
    H = np.array([c[2] for c in cand])
    centre = float(np.median(H))
    half = 0.5 * hinf_band * centre
    band = [c for c in cand if abs(c[2] - centre) <= half]
    if len(band) < n_dice:                    # widen rather than fail
        order = np.argsort(np.abs(H - centre))
        band = [cand[i] for i in order[: max(n_dice * 3, 30)]]

    band.sort(key=lambda c: c[1])             # by gap
    gaps = np.array([c[1] for c in band])

    # Spread the picks over gap QUANTILES, so the design has even leverage across
    # the x-range instead of clustering where draws happen to be dense.
    idx = np.unique(np.linspace(0, len(band) - 1, n_dice).round().astype(int))
    chosen = [band[i] for i in idx]

    # A sign test needs both signs.  If the band holds negative-gap draws that the
    # quantile spread missed, swap them in for the highest-gap picks -- the top of
    # the range is where the design has the least to learn, since every cell there
    # agrees on the predicted sign.
    negatives = sorted((c for c in band if c[1] < 0), key=lambda c: c[1])
    want = min(min_negative, len(negatives))
    have = [c for c in chosen if c[1] < 0]
    if len(have) < want:
        keep = [c for c in chosen if c[1] >= 0]
        keep.sort(key=lambda c: c[1])              # ascending gap
        add = [c for c in negatives if c[0] not in {z[0] for z in chosen}][
            : want - len(have)]
        # drop the largest-gap picks to make room, then re-sort
        keep = keep[: max(len(keep) - len(add), 0)]
        chosen = sorted(have + add + keep, key=lambda c: c[1])

    diag = dict(n_search=n_search, alpha=alpha, hinf_centre=centre,
                hinf_half_width=half, n_in_band=len(band),
                band_gap_range=[float(gaps.min()), float(gaps.max())],
                all_gap_range=[float(min(c[1] for c in cand)),
                               float(max(c[1] for c in cand))],
                corr_gap_hinf_in_band=float(
                    np.corrcoef([c[1] for c in band], [c[2] for c in band])[0, 1])
                if len(band) > 2 else float("nan"),
                corr_gap_hinf_all=float(np.corrcoef([c[1] for c in cand],
                                                    [c[2] for c in cand])[0, 1]))
    return chosen, diag


def dice_specs(cfg: dict, cells: list, alpha: float, n_dice: int, n_search: int,
               hinf_band: float, min_negative: int) -> tuple:
    """One spec per (cell, dice realisation), tagged with the seed (note 1)."""
    specs, diags = [], {}
    for (n, m) in cells:
        chosen, diag = select_dice(n, m, alpha, n_dice, n_search, hinf_band,
                                   min_negative)
        diags[f"{n},{m}"] = dict(diag, chosen=[(int(s), float(g), float(h))
                                               for s, g, h in chosen])
        for seed, _gap, _h in chosen:
            s = flower_spec(cfg, n, m, dice_seed=int(seed), dice_alpha=alpha)
            s["tag"] = f"dice_flower_n{n}_m{m}_s{int(seed):05d}"
            s["cell"] = (n, m)
            specs.append(s)
    return specs, diags


def coverage_table(specs: list, diags: dict):
    """What the design covers — printed before anything trains."""
    print(f"\n{'='*104}\n  DESIGN — {len(specs)} processes over "
          f"{len(diags)} (n,m) cell(s)\n{'='*104}")
    for key, d in diags.items():
        n, m = (int(v) for v in key.split(","))
        print(f"\n  cell (n,m)=({n},{m})   alpha={d['alpha']}   "
              f"searched {d['n_search']} realisations")
        print(f"    H_inf band       : {d['hinf_centre']:.4f} +/- "
              f"{d['hinf_half_width']:.4f}   ({d['n_in_band']} draws inside)")
        print(f"    gap in band      : [{d['band_gap_range'][0]:+.4f}, "
              f"{d['band_gap_range'][1]:+.4f}]   "
              f"(unrestricted: [{d['all_gap_range'][0]:+.4f}, "
              f"{d['all_gap_range'][1]:+.4f}])")
        print(f"    corr(gap, H_inf) : {d['corr_gap_hinf_in_band']:+.3f} in band, "
              f"{d['corr_gap_hinf_all']:+.3f} unrestricted"
              f"   <- the confound this design removes")
        sel = d["chosen"]
        neg = sum(1 for _s, g, _h in sel if g < 0)
        print(f"    selected {len(sel)}: gap "
              f"[{min(g for _s,g,_h in sel):+.4f}, {max(g for _s,g,_h in sel):+.4f}], "
              f"{neg} negative"
              + ("   <- SIGN TEST AVAILABLE" if neg else
                 "   <- no negative gap in band; magnitude test only"))
        for s, g, h in sel:
            print(f"       seed {s:>5}   gap {g:+.4f}   H_inf {h:.4f}")
    C = {s["num_token"] for s in specs}
    print(f"\n  held fixed by construction: vocabulary {sorted(C)}, "
          f"C+ per cell, m-n per cell, capacity, step budget")
    print(f"{'='*104}")


# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS — the whole point is the WITHIN-CELL slope
# ══════════════════════════════════════════════════════════════════════════
def within_cell_stats(rows: list) -> list:
    """
    Per cell: rho and the weighted slope of delta_CE on the gap.

    Within a cell every nuisance variable is constant, so a non-zero slope here
    cannot be m - n, vocabulary, C+ or (given the band) H_inf.  That is the entire
    argument of this experiment.
    """
    out = []
    cells = sorted({r["cell"] for r in rows})
    for cell in cells:
        sub = [r for r in rows if r["cell"] == cell and np.isfinite(r["dce"])]
        if len(sub) < 4:
            continue
        w = weighted_slope(sub, "dce")
        rho, p, n = _spearman([r["gap"] for r in sub], [r["dce"] for r in sub])
        gaps = [r["gap"] for r in sub]
        signs = sum(1 for r in sub
                    if np.isfinite(r["dce"]) and (r["dce"] > 0) == (r["gap"] > 0))
        out.append(dict(cell=cell, n=len(sub), rho=rho, p=p,
                        gap_lo=min(gaps), gap_hi=max(gaps),
                        n_negative=sum(1 for g in gaps if g < 0),
                        sign_agree=signs, hinf_spread=float(
                            np.ptp([r["theory"] for r in sub])),
                        **{f"wls_{k}": v for k, v in (w or {}).items()}))
    return out


def pooled_within(rows: list) -> dict:
    """
    One slope, fitted with a separate intercept per cell.

    Cell-centring the gap is what makes this a within-cell estimate: any
    between-cell difference (which is where m - n lives) is absorbed by the
    per-cell intercepts and cannot contribute to the slope.
    """
    use = [r for r in rows if np.isfinite(r["dce"]) and np.isfinite(r["dce_sem"])
           and r["dce_sem"] > 0]
    if len(use) < 6:
        return {}
    by_cell: dict = {}
    for r in use:
        by_cell.setdefault(r["cell"], []).append(r)
    x, y, s = [], [], []
    for cell, sub in by_cell.items():
        if len(sub) < 3:
            continue
        gbar = float(np.mean([r["gap"] for r in sub]))
        dbar = float(np.mean([r["dce"] for r in sub]))
        for r in sub:
            x.append(r["gap"] - gbar); y.append(r["dce"] - dbar); s.append(r["dce_sem"])
    if len(x) < 6:
        return {}
    x, y, w = np.array(x), np.array(y), 1.0 / np.array(s) ** 2
    b = float(np.sum(w * x * y) / np.sum(w * x * x))
    se = float(np.sqrt(1.0 / np.sum(w * x * x)))
    return dict(slope=b, slope_se=se, t=b / se if se else float("nan"),
                n=len(x), n_cells=len(by_cell))


def plot_dice(rows: list, within: list, out_root: str):
    """delta_CE against the gap, one panel per cell, plus the pooled fit."""
    cells = sorted({r["cell"] for r in rows})
    fig, axes = plt.subplots(1, len(cells) + 1,
                             figsize=(5.2 * (len(cells) + 1), 4.4))
    axes = np.atleast_1d(axes)
    for ax, cell in zip(axes, cells):
        sub = [r for r in rows if r["cell"] == cell and np.isfinite(r["dce"])]
        g = [r["gap"] for r in sub]; d = [r["dce"] for r in sub]
        e = [r["dce_sem"] for r in sub]
        ax.errorbar(g, d, yerr=e, fmt="o", capsize=3, color="#dd8452", ms=5)
        st = next((w for w in within if w["cell"] == cell), None)
        if st and np.isfinite(st.get("wls_slope", np.nan)):
            xs = np.linspace(min(g), max(g), 50)
            ax.plot(xs, st["wls_slope"] * xs + st["wls_intercept"],
                    color="#4c72b0", lw=1.6,
                    label=(f"WLS b={st['wls_slope']:+.5f}\n"
                           f"     ±{st['wls_slope_se']:.5f} (t={st['wls_t']:+.1f})\n"
                           f"ρ={st['rho']:+.3f} (p={st['p']:.3f}, n={st['n']})"))
            ax.legend(fontsize=8, loc="best")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.set_xlabel(r"$C^- - C^+$  (bits)   — dice varied, $(n,m)$ fixed")
        ax.set_ylabel(r"$\Delta CE = CE_{BW} - CE_{FW}$  (bits)")
        ax.set_title(f"(n,m) = {cell}\nm−n, vocabulary, $C^+$, "
                     r"$H_\infty$ all held fixed", fontsize=10)
        ax.grid(alpha=0.3)

    ax = axes[-1]
    for cell, colour in zip(cells, ("#dd8452", "#4c72b0", "#55a868", "#c44e52")):
        sub = [r for r in rows if r["cell"] == cell and np.isfinite(r["dce"])]
        if not sub:
            continue
        gbar = np.mean([r["gap"] for r in sub]); dbar = np.mean([r["dce"] for r in sub])
        ax.errorbar([r["gap"] - gbar for r in sub], [r["dce"] - dbar for r in sub],
                    yerr=[r["dce_sem"] for r in sub], fmt="o", ms=5, capsize=3,
                    color=colour, label=f"{cell}")
    pw = pooled_within(rows)
    if pw:
        xs = np.linspace(*ax.get_xlim(), 50)
        ax.plot(xs, pw["slope"] * xs, color="black", lw=1.8,
                label=(f"pooled within-cell\nb={pw['slope']:+.5f}±"
                       f"{pw['slope_se']:.5f} (t={pw['t']:+.1f})"))
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"$C^- - C^+$ , cell-centred")
    ax.set_ylabel(r"$\Delta CE$ , cell-centred")
    ax.set_title("Pooled within-cell\n(between-cell variation removed)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Does ΔCE track causal asymmetry, or m−n?  "
                 "Varying only the dice answers it (REMAINING_WORK_PLAN S1)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    savefig(fig, os.path.join(out_root, "dice_scatter.png"))


def print_summary(rows: list, within: list, out_root: str):
    print(f"\n{'='*104}\n  DICE-AXIS SUMMARY  ({out_root})\n{'='*104}")
    print(f"  {'process':<28}{'gap':>9}{'H_inf':>8}{'n':>4}{'div':>4}"
          f"{'dCE':>11}{'sem':>9}{'verdict':>10}")
    for r in sorted(rows, key=lambda r: (r["cell"], r["gap"])):
        print(f"  {r['tag'].replace('dice_',''):<28}{r['gap']:>+9.4f}"
              f"{r['theory']:>8.4f}{r['n_repeats']:>4}{r['n_div']:>4}"
              f"{r['dce']:>+11.5f}{r['dce_sem']:>9.5f}{r['verdict']:>10}")

    print(f"\n{'-'*104}\n  WITHIN-CELL SLOPE — the test this experiment exists for"
          f"\n{'-'*104}")
    print(f"  {'cell':<10}{'n':>4}{'gap range':>20}{'neg':>5}{'H_inf spread':>14}"
          f"{'WLS b':>11}{'se':>9}{'t':>7}{'rho':>8}{'sign ok':>9}")
    for w in within:
        rng = f"[{w['gap_lo']:+.3f},{w['gap_hi']:+.3f}]"
        sign = f"{w['sign_agree']}/{w['n']}"
        print(f"  {str(w['cell']):<10}{w['n']:>4}{rng:>20}"
              f"{w['n_negative']:>5}{w['hinf_spread']:>14.4f}"
              f"{w.get('wls_slope', float('nan')):>+11.5f}"
              f"{w.get('wls_slope_se', float('nan')):>9.5f}"
              f"{w.get('wls_t', float('nan')):>+7.1f}{w['rho']:>+8.3f}"
              f"{sign:>9}")
    pw = pooled_within(rows)
    if pw:
        print(f"\n  pooled within-cell (separate intercept per cell): "
              f"b = {pw['slope']:+.5f} +/- {pw['slope_se']:.5f}   "
              f"t = {pw['t']:+.2f}   n = {pw['n']} over {pw['n_cells']} cells")
    print(f"\n{'-'*104}\n  HOW TO READ IT\n{'-'*104}")
    print("  The 81-cell grid gives a between-cell weighted slope of +0.00294 +/-")
    print("  0.00016, but on that grid corr(gap, m-n) = +0.977, so the slope is")
    print("  equally consistent with tracking m - n.  Here (n,m) is FIXED, so:")
    print("    * a within-cell slope near +0.0029 -> the trend is the theoretical")
    print("      gap; m-n, vocabulary, C+ and H_inf are all excluded by design.")
    print("    * a within-cell slope near 0       -> the 81-cell trend was m - n.")
    print("    * sign of dCE following sign of gap inside one cell -> a controlled")
    print("      sign test, which is the only thing the theory actually predicts.")
    print("  Caveats that still apply: delta_CE is a difference of RESIDUALS, so a")
    print("  model with spare capacity gives ~0 whatever the gap is (the d_model")
    print("  sweep is what disambiguates that); and dice realisations at one (n,m)")
    print("  are independent draws, so unlike the lattice grid these points do not")
    print("  have the near-duplicate-neighbour problem.")
    print(f"{'='*104}")


# ══════════════════════════════════════════════════════════════════════════
def parse_cells(s: str) -> list:
    """'2,8 4,6' -> [(2, 8), (4, 6)].  Accepts semicolons too."""
    out = []
    for tok in s.replace(";", " ").split():
        n, m = tok.split(",")
        out.append((int(n), int(m)))
    if not out:
        raise SystemExit("--cells parsed to nothing; expected e.g. '2,8 4,6'")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="QUICK", choices=sorted(CONFIGS))
    ap.add_argument("--cells", default=CELLS_DEFAULT,
                    help='(n,m) cells, e.g. "2,8 4,6"')
    ap.add_argument("--n-dice", type=int, default=10,
                    help="dice realisations per cell")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT,
                    help="Dirichlet concentration; <1 is spiky and widens the "
                         "reachable gap range")
    ap.add_argument("--n-search", type=int, default=4000,
                    help="candidate realisations surveyed per cell (no training)")
    ap.add_argument("--hinf-band", type=float, default=0.02,
                    help="relative width of the H_inf band the dice are drawn "
                         "from (0.02 = +/-1%%), which is what removes the "
                         "entropy-rate confound")
    ap.add_argument("--min-negative", type=int, default=3,
                    help="insist on at least this many negative-gap cells when "
                         "the band contains them, so a sign test is possible")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--accelerator", default=None,
                    choices=["auto", "cpu", "mps", "gpu"])
    ap.add_argument("--d-model", type=int, default=None,
                    help="override cfg['d_model'].  The residual argument "
                         "(README 1.2) predicts |delta_CE| shrinks as capacity "
                         "grows, and that has never been measured -- see "
                         "REMAINING_WORK_PLAN.md S5.  Give each capacity its "
                         "OWN --out-root: tags do not encode d_model, so two "
                         "capacities in one folder overwrite each other.")
    ap.add_argument("--usage-beta", type=float, default=None,
                    help="override the computed beta = 1/(batch * chunk_len).  "
                         "Leave unset for every normal run: the penalty is "
                         "beta * S_emp, so beta is naturally per-token, and it "
                         "is bounded above by a collapse cliff (~6e-4 at "
                         "lr=1e-3) that a literal decouples it from.  This flag "
                         "exists for the beta sweep, which sweeps AROUND 1/N.")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = dict(CONFIGS[args.config])
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.accelerator is not None:
        cfg["accelerator"] = args.accelerator
    if args.d_model is not None:
        cfg["d_model"] = args.d_model
    # None here means "use the computed 1/N", which discrete_hparams resolves
    # per process from the geometry.  Set only by the beta sweep.
    cfg["usage_beta_fixed"] = args.usage_beta

    cells = parse_cells(args.cells)
    specs, diags = dice_specs(cfg, cells, args.alpha, args.n_dice,
                              args.n_search, args.hinf_band, args.min_negative)

    out_root = mkdir(repo_path(args.out_root))
    # REMAINING_WORK_PLAN.md S5.  Tags are a function of the process, not of
    # d_model, and the store is keyed by tag -- so mixing capacities in one
    # out_root silently overwrites.  Refuse rather than corrupt.
    if args.d_model is not None and args.d_model != CONFIGS[args.config]["d_model"]:
        import re as _re
        _tokens = {int(t) for t in _re.findall(r"(?:^|[^0-9a-zA-Z])d0*(\d+)", out_root)}
        if args.d_model not in _tokens:
            raise SystemExit(
                f"--d-model {args.d_model} differs from the {args.config} default "
                f"({CONFIGS[args.config]['d_model']}) but --out-root is "
                f"{out_root!r}, whose name does not mention d{args.d_model}.\n"
                f"    Tags do not encode d_model, so this would overwrite the "
                f"other capacity's results.\n"
                f"    Use e.g. --out-root .../d{args.d_model:03d}")
    combined_path = os.path.join(out_root, "all_dice.pkl")
    combined = load_combined(combined_path)

    print(f"\n{'='*104}")
    print(f"  config={args.config}  repeats={args.repeats}  base_seed={cfg['seed']}"
          f"  out_root={out_root}")
    print(f"  d_model={cfg['d_model']}  lr={cfg['lr']}  "
          f"weight_decay={cfg['weight_decay']}  accelerator={cfg['accelerator']}")
    print(f"{'='*104}")
    coverage_table(specs, diags)

    todo = [s for s in specs
            if args.redo
            or len((combined.get(s["tag"]) or {}).get("runs", [])) < args.repeats]
    print(f"\n  resume plan : {len(specs)-len(todo)} complete and skipped, "
          f"{len(todo)} to train")
    if not combined:
        print(f"  ! no existing all_dice.pkl at {combined_path} — this trains all "
              f"{len(specs)} process(es) from scratch.")

    if args.dry_run:
        print("\n  --dry-run: nothing trained.")
        return

    if not args.plots_only:
        save_run_config(os.path.join(out_root, f"run_config_{args.config}.json"),
                        {"config": args.config, "repeats": args.repeats,
                         "base_seed": cfg["seed"], "cells": [list(c) for c in cells],
                         "alpha": args.alpha, "n_dice": args.n_dice,
                         "hinf_band": args.hinf_band, "n_search": args.n_search,
                         "merge_tol": None,   # the historical rule; see docstring
                         "selection": diags,
                         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         **{k: v for k, v in cfg.items()}})
        t0, t_save = time.time(), [0.0]
        for pi, spec in enumerate(todo, 1):
            tag = spec["tag"]
            prev = (combined.get(tag) or {}).get("runs", []) if not args.redo else []
            rec = dict(spec=spec, config=args.config, base_seed=cfg["seed"],
                       repeats=args.repeats, khat=False, runs=list(prev))
            print(f"\n  [{pi}/{len(todo)}] {tag}   gap="
                  f"{spec['C_minus']-spec['C_plus']:+.4f}  H∞={spec['theory']:.4f}")
            tp = time.time()
            for i in range(len(rec["runs"]), args.repeats):
                with quiet(not args.verbose):
                    run = one_repeat(spec, cfg, cfg["seed"] + i, khat=False)
                rec["runs"].append(run)
                if run["fw"]["diverged"] or run["bw"]["diverged"]:
                    print(f"      ! repeat {i+1} DIVERGED")
                combined[tag] = rec
                if time.time() - t_save[0] > SAVE_EVERY_S:
                    with contextlib.redirect_stdout(io.StringIO()):
                        rec["paired"] = paired_stats(rec) if len(rec["runs"]) > 1 else None
                        save_pkl(combined, combined_path)
                    t_save[0] = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                rec["paired"] = paired_stats(rec) if len(rec["runs"]) > 1 else None
            combined[tag] = rec
            done = pi
            eta = (time.time() - t0) / done * (len(todo) - done) / 3600
            print(f"      dCE={rec['paired']['all_folds']['mean']:+.5f} "
                  f"±{rec['paired']['all_folds']['sem']:.5f}   "
                  f"({(time.time()-tp)/60:.1f} min, ETA {eta:.2f} h)")
            save_pkl(combined, combined_path)

    if not combined:
        raise SystemExit("nothing to analyse")
    with contextlib.redirect_stdout(io.StringIO()):
        first = [r for r in (process_row(t, rec) for t, rec in combined.items()) if r]
        burn, _ = global_burn_in(first)
        rows = [r for r in (process_row(t, rec, global_burn=burn)
                            for t, rec in combined.items()) if r]
    # process_row does not know about `cell`; re-attach it from the spec.
    for r in rows:
        s = combined[r["tag"]]["spec"]
        r["cell"] = tuple(s.get("cell", (s["n"], s["m"])))

    within = within_cell_stats(rows)
    save_pkl({"rows": rows, "within": within, "pooled": pooled_within(rows),
              "selection": diags}, os.path.join(out_root, "dice_rows.pkl"))
    plot_dice(rows, within, out_root)
    print_summary(rows, within, out_root)


if __name__ == "__main__":
    main()
