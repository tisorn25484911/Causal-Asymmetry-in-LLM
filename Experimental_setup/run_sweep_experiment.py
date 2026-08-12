"""
run_sweep_experiment.py — does measured asymmetry track theoretical asymmetry?

    python run_sweep_experiment.py --dry-run              # spec table + coverage, no training
    python run_sweep_experiment.py --repeats 2            # timing probe
    python run_sweep_experiment.py --repeats 30           # the real run, ~10 h
    python run_sweep_experiment.py --sweep-flower --repeats 30    # flower grid only
    python run_sweep_experiment.py --plots-only           # redraw from the saved pickle

Implements implementation_logbook/SWEEP_EXPERIMENT_PLAN.md.  Everything that
trains, measures or aggregates a single process is IMPORTED from
run_statistical_trj.py -- this file only decides *which* processes to run and how
to aggregate ACROSS them.  No other file in the repository is modified.

The question
------------
The seven-process harness answers "what is delta_CE for this process?" with high
precision at a few points.  This asks the stronger question:

    across many processes spanning a wide range of C- - C+, does the measured
    asymmetry track the theoretical one?

Two grids, 125 processes:

    coin    p,q in {0.05, 0.15, ..., 0.95}      100 processes
    flower  n,m in {2, 4, 6, 8, 10}              25 processes

Coverage, recomputed from the closed forms rather than assumed:

    coin    C- - C+ in [+0.014, +0.683]   ALL POSITIVE -- magnitude only
    flower  C- - C+ in [-1.164, +1.069]   14 neg / 1 ~zero / 10 pos
    pooled  -1.16 to +1.07 with points near zero

Response variables, in order of how much theory backs them
----------------------------------------------------------
1.  **final delta_CE** (PRIMARY).  The endpoint difference of held-out CE.  This
    is the only quantity the hypothesis makes a statement about: for a
    memory-bounded predictor, sign(delta_CE) should follow sign(C- - C+).

2.  **trajectory area** (EXPLORATORY).  A_r = mean over the settled part of
    training of D_r(s) = CE_BW,r(s) - CE_FW,r(s).  The plan's argument is that
    even when both endpoints converge to H_inf, the harder direction should sit
    above the other throughout optimisation.  That is an intuition, not a
    derivation: the integral of a loss difference along an Adam trajectory
    depends on the learning rate, the initialisation and the landscape's
    curvature, none of which appear anywhere in computational mechanics.  A
    positive result here generates a hypothesis; it does not confirm one.

3.  **convergence-step difference** (EXPLORATORY).  Steps for the backward arm to
    settle minus steps for the forward arm.  Included because it is the path
    statistic with the cleanest units -- it is immune to the magnitude of the
    transient, and to a diverged run's spike, both of which corrupt an integral.

Four departures from the plan, each fixing something that would have biased the
answer
-------------------------------------------------------------------------------
1.  **The area's burn-in is derived per process, not fixed at int(L/4.3).**
    Measured on the completed baseline run, step 30 is inside the transient for
    EVERY process, and differentially so: the coins settle at step 32-42 and the
    flowers at 60-67.  Integrating from a fixed step therefore folds ~8 transient
    steps into the coin areas and ~35 into the flower ones -- and since every
    negative C- - C+ point is a flower (see 2), that contamination is
    systematically different on the two halves of the x-axis.  It is a mechanism
    for manufacturing a trend with no involvement from causal asymmetry.
    `settled_from` instead finds, per process, the first step after which both
    arms' mean curves stay within CONV_STEP_TOL of their final value.  The
    fixed-fraction and whole-trajectory areas are recorded too, so the choice is
    visible rather than buried.

2.  **Correlations are reported WITHIN family as well as pooled.**  All 15
    negative-x points are flower processes and all 100 coin points are positive,
    so corr(family, C- - C+) = -0.46.  A pooled Spearman rho over 125 points is
    substantially a test of "do flower processes differ from coin processes"
    (vocabulary 3 vs 4-20, H_inf 0.29-0.99 vs 0.87-3.07).  The flower grid alone
    is the sign test; the coin grid alone is a magnitude test.

3.  **A partial correlation controlling for H_inf is reported for the coin
    grid**, where corr(C- - C+, H_inf) = +0.59.  Anything that depends on the
    entropy rate -- gradient scale, residual size at a fixed step budget,
    convergence speed -- would otherwise appear as a trend in C- - C+.  Within
    the flower grid C- - C+ is uncorrelated with both H_inf (-0.01) and
    vocabulary (+0.01), so no such correction is needed there; that decoupling is
    what makes the flower grid the better-designed axis.

4.  **The area's headline statistic is over CONVERGED repeats only.**  One
    diverged repeat spiking to 40 bits for ~10 steps contributes ~4 bits to that
    repeat's area and ~0.13 bits to a 30-repeat mean -- about 100x the entire
    effect being measured.  For the endpoint, reporting converged-only
    "alongside" is enough; for an integral it has to be the primary.

Everything else follows the plan: area computed at analysis time from stored
trajectories (so the burn-in stays tunable without retraining), normalised to a
mean so its units are bits, per-repeat then aggregated (which preserves the
pairing), its own out_root, its own tag prefix, no per-process figures, and an
incremental save after every repeat.

What this cannot show
---------------------
  * C- - C+ is a smooth deterministic function of (p, q) and of (n, m), so
    neighbouring grid cells are near-duplicate processes.  The effective degrees
    of freedom are far below 100 and 25, and the nominal Spearman p-values are
    correspondingly anti-conservative.  They are reported because they are the
    conventional statistic, not because n = 125 independent observations exist.
  * Each (n, m) uses ONE dice realisation from cfg["flower_dice_seed"], so
    C- - C+ carries realisation noise at fixed (n, m) and x is measured with
    error, which biases any fitted slope toward zero.  Deliberate: it keeps these
    processes identical to the ones run_experiments.py and run_statistical_trj.py
    train on, which is what makes the baseline cross-check in
    `baseline_crosscheck` possible.
  * Capacity relative to the process varies along the flower axis -- vocabulary 4
    to 20, up to 11 causal states, at a fixed d_model of 32.  The hypothesis is
    conditional on the predictor being memory-bounded, so a per-point delta_CE
    means slightly different things at different x.
  * At 30 repeats a per-point sem is ~2x wider than the baseline's at 100.  The
    correlation attenuation from that y-noise is about 11%, which is the price of
    5x the points; but most individual error bars will cross zero, and the
    scatter must not be read point by point.
"""
# ── stdlib ─────────────────────────────────────────────────────────────────
import argparse
import contextlib
import io
import json
import os
import time

# ── third-party ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as sstats

# ── repo path bootstrap ────────────────────────────────────────────────────
# REORGANISATION_FIX_PLAN.md 4.1.  The tree is split across Transformer_model/
# and Experimental_setup/ but the modules are still flat (`from utils import
# ...`), so both directories have to be importable.  Python only ever puts the
# *script's own* directory on sys.path -- which is why this is needed, and why
# cd-ing elsewhere does not help.  Anchored on __file__, so the script runs
# from any working directory.
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── project ────────────────────────────────────────────────────────────────
# Every measurement primitive comes from the harness: one_repeat trains and
# scores a paired repeat, paired_stats runs the same paired estimator the rest of
# the repo uses, _traj_pair/_stats/_verdict/CONV_TOL keep the definitions single.
from configs import CONFIGS
from Model_analysis import savefig
from run_statistical_trj import (
    CONV_TOL,
    _stats,
    _traj_pair,
    _verdict,
    coin_spec,
    flower_spec,
    load_combined,
    one_repeat,
    paired_stats,
    quiet,
)
from utils import (coin_tag, flower_tag, mkdir, repo_path, save_pkl,
                   save_run_config)


# ══════════════════════════════════════════════════════════════════════════
# GRIDS
# ══════════════════════════════════════════════════════════════════════════
# Verified coverage (see the module docstring).  Changing either grid changes
# what the sweep can test, so re-run --dry-run and re-read the coverage table
# before committing to a different one.
COIN_GRID_DEFAULT   = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
FLOWER_GRID_DEFAULT = [2, 4, 6, 8, 10]

OUT_ROOT_DEFAULT = "All_Results/results_sweep"

COIN_COLOUR, FLOWER_COLOUR = "#4c72b0", "#dd8452"

# A step counts as settled when the mean curve stays within this many bits of its
# final value for the rest of training.  0.01 bits is an order of magnitude below
# the smallest H_inf here (0.29) and an order above the effect being measured.
CONV_STEP_TOL = 0.01

# Kept only so the fixed-fraction area can be reported beside the settled one --
# this is the denominator _draw_paired_diff uses to scale its y-axis.
BURN_FRACTION_DENOM = 4.3

# Smoothing window for finding a SINGLE run's convergence step.  A single
# training curve is noisy step to step; the mean over repeats is not, so this
# applies to per-repeat convergence only.
SMOOTH_WIN = 5

# Minimum seconds between incremental saves of the combined pickle.
SAVE_EVERY_S = 60.0


def parse_grid(values, kind: str):
    """Validate a user-supplied grid and reject anything that cannot be run."""
    if not values:
        return COIN_GRID_DEFAULT if kind == "coin" else FLOWER_GRID_DEFAULT
    if kind == "coin":
        g = sorted({float(v) for v in values})
        if any(not (0.0 < v < 1.0) for v in g):
            raise SystemExit(f"--sweep-coin: probabilities must be in (0, 1), got {g}")
        # coin_tag rounds to integer percent, so two values closer than 0.01
        # would collide and one process would silently overwrite the other.
        tags = {coin_tag("sweep", p, q) for p in g for q in g}
        if len(tags) != len(g) ** 2:
            raise SystemExit(
                f"--sweep-coin: grid {g} produces colliding tags "
                f"({len(tags)} unique for {len(g)**2} cells).  coin_tag rounds to "
                f"integer percent, so values must differ by at least 0.01.")
        return g
    g = sorted({int(v) for v in values})
    if any(v < 1 for v in g):
        raise SystemExit(f"--sweep-flower: n and m must be >= 1, got {g}")
    return g


def sweep_specs(cfg: dict, coin_grid, flower_grid) -> list:
    """
    One spec per grid cell, built by the harness's own spec builders and then
    re-tagged with a `sweep_` prefix.

    Re-tagging rather than parameterising coin_spec/flower_spec keeps this
    experiment entirely inside one new file, and gives defence in depth against
    a sweep cell colliding with a baseline process in a shared pickle.
    """
    specs = []
    for p in (coin_grid or []):
        for q in coin_grid:
            s = coin_spec(cfg, p, q)
            s["tag"] = coin_tag("sweep", p, q)
            specs.append(s)
    for n in (flower_grid or []):
        for m in flower_grid:
            s = flower_spec(cfg, n, m)
            s["tag"] = flower_tag("sweep", n, m)
            specs.append(s)
    return specs


def coverage_table(specs: list):
    """What the grids actually cover — printed before any training happens."""
    print(f"\n{'='*78}\n  COVERAGE — {len(specs)} processes\n{'='*78}")
    for kind in ("coin", "flower"):
        sub = [s for s in specs if s["kind"] == kind]
        if not sub:
            continue
        gap = np.array([s["C_minus"] - s["C_plus"] for s in sub])
        H   = np.array([s["theory"] for s in sub])
        V   = np.array([s["num_token"] for s in sub])
        print(f"\n  {kind}: {len(sub)} processes")
        print(f"    C- - C+   [{gap.min():+.4f}, {gap.max():+.4f}]  median {np.median(gap):+.4f}")
        print(f"              {int((gap < -0.01).sum())} negative, "
              f"{int((np.abs(gap) <= 0.01).sum())} near zero, "
              f"{int((gap > 0.01).sum())} positive")
        print(f"    H_inf     [{H.min():.3f}, {H.max():.3f}]")
        print(f"    vocab     [{V.min()}, {V.max()}]")
        if len(sub) > 2:
            # corr(gap, H_inf) is the one that matters: on the coin grid it is
            # +0.59, so an uncorrected trend there cannot be told apart from a
            # dependence on the entropy rate.  vocab is constant on that grid, so
            # its correlation is undefined and omitted rather than printed as nan.
            line = f"    corr(C--C+, H_inf) = {np.corrcoef(gap, H)[0, 1]:+.3f}"
            if V.std() > 0:
                line += f"   corr(C--C+, vocab) = {np.corrcoef(gap, V)[0, 1]:+.3f}"
            print(line)
    gap = np.array([s["C_minus"] - s["C_plus"] for s in specs])
    fam = np.array([1.0 if s["kind"] == "flower" else 0.0 for s in specs])
    print(f"\n  pooled x-range [{gap.min():+.4f}, {gap.max():+.4f}]")
    if fam.std() > 0:
        neg = gap < -0.01
        print(f"  corr(family, C- - C+) = {np.corrcoef(fam, gap)[0, 1]:+.3f}"
              f"   negative-x points that are flower: "
              f"{int((neg & (fam == 1)).sum())}/{int(neg.sum())}")
        print("  -> the pooled correlation conflates family with the sign of x; "
              "read the within-family rows.")
    print(f"{'='*78}")


# ══════════════════════════════════════════════════════════════════════════
# DERIVED METRICS — pure functions of the stored trajectories
# ══════════════════════════════════════════════════════════════════════════
def settled_from(mu, tol: float = CONV_STEP_TOL) -> int:
    """
    First step after which the curve stays indistinguishable from its plateau.

    This is the burn-in the area needs.  A fixed fraction of the step count
    cannot serve, because convergence time varies systematically across these
    processes (32-42 steps for the coins, 60-67 for the flowers on the baseline
    run) and is therefore correlated with the x-axis of the sweep.

    Three details make it robust, and the first two were added after a 2-repeat
    smoke test returned step 117 of 130:

      * the plateau is the mean of the LAST TENTH of training, not the single
        final value, which is itself a noisy estimate;
      * the tolerance is `max(tol, 3 x sd of that plateau)`, so it adapts to how
        noisy the curve actually is.  A mean over 2 repeats wobbles by more than
        `tol` late in training and a fixed tolerance then declares nothing
        settled; a mean over 30 wobbles far less.  Without this the burn-in would
        depend on the repeat count and would not be comparable across processes
        with different `--repeats`;
      * the curve is smoothed first, so one spiky step does not reset the answer.

    "Settled" therefore reads as: indistinguishable from the plateau, given the
    plateau's own noise.
    """
    mu = _smooth(np.asarray(mu, dtype=float))
    if mu.size == 0:
        return 0
    tail = mu[max(int(0.9 * mu.size), mu.size - 1):]
    level = float(tail.mean())
    tol_eff = max(float(tol), 3.0 * float(tail.std()))
    bad = np.where(np.abs(mu - level) > tol_eff)[0]
    return int(bad[-1] + 1) if bad.size else 0


def _smooth(x, win: int = SMOOTH_WIN):
    """
    Centred moving average that PRESERVES length and step alignment.

    `mode="valid"` would shorten the array by win-1 and silently shift every
    index returned by settled_from off the step axis; `mode="same"` keeps the
    length but zero-pads, which drags the last points toward zero and corrupts
    exactly the plateau settled_from measures.  Replicating the edges avoids
    both.
    """
    x = np.asarray(x, dtype=float)
    if x.size < win or win < 2:
        return x
    pad = win // 2
    xp = np.pad(x, (pad, win - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(win) / win, mode="valid")


def trajectory_metrics(rec: dict, global_burn=None) -> dict:
    """
    Per-repeat areas and convergence-step differences for one process.

    A pure function of `rec["runs"][*][arm]["traj"]`, so every burn-in choice can
    be revisited without retraining, and re-running it on the same pickle gives
    identical numbers.

    Returns per-repeat lists plus the burn-in steps used, all of which go into
    the pickle so a later reader can audit the choice.
    """
    pair = _traj_pair(rec)
    if pair is None:
        return {}
    M_fw, M_bw, L, _ = pair
    D = M_bw - M_fw

    burn_settled = max(settled_from(M_fw.mean(axis=0)),
                       settled_from(M_bw.mean(axis=0)))
    burn_settled = int(min(burn_settled, max(L - 5, 0)))   # always keep a tail
    burn_fixed   = int(L / BURN_FRACTION_DENOM)

    # Per-repeat convergence step, from each run's own smoothed curve.
    d_step = []
    for i in range(D.shape[0]):
        f = settled_from(_smooth(M_fw[i]))
        b = settled_from(_smooth(M_bw[i]))
        d_step.append(float(b - f))

    # The window the PRIMARY area uses: the sweep-wide one when supplied,
    # otherwise this process's own.  Recorded so a reader can see which.
    burn_used = int(global_burn) if global_burn is not None else burn_settled
    burn_used = int(min(max(burn_used, 0), max(L - 5, 0)))

    return dict(
        L=int(L),
        burn_settled=burn_settled,
        burn_fixed=burn_fixed,
        burn_used=burn_used,
        area_full=[float(v) for v in D.mean(axis=1)],
        area_fixed=[float(v) for v in D[:, burn_fixed:].mean(axis=1)],
        area_settled=[float(v) for v in D[:, burn_settled:].mean(axis=1)],
        area_primary=[float(v) for v in D[:, burn_used:].mean(axis=1)],
        d_step=d_step,
        mean_D=[float(v) for v in D.mean(axis=0)],
    )


GLOBAL_BURN_CAP_FRAC = 0.6   # a burn-in past this fraction of L is pathological


def global_burn_in(rows: list) -> tuple:
    """
    One integration window for the whole sweep, and why the area needs it.

    The per-process burn-in of `settled_from` is the right cutoff for asking "is
    THIS process settled".  It is the wrong cutoff for comparing areas ACROSS
    processes: measured at 3 repeats the per-process value ranged from step 32 to
    123 of 130, so one cell's area would average 98 steps and another's 7.  A
    difference in averaging window is not a difference in asymmetry.

    So the primary area uses a single window [B, L] with B the SLOWEST process's
    settled step: every process is settled by then, and every area is a mean over
    the same steps.  B is capped at 60% of L so one pathological cell cannot
    shrink the window to nothing; the measured maximum on the baseline run is 67
    of 130 (52%), so the cap does not bind on well-behaved data.
    """
    burns = [r["burn_settled"] for r in rows if r.get("burn_settled") is not None]
    L = min(r["L"] for r in rows)
    if not burns:
        return 0, False
    cap = int(GLOBAL_BURN_CAP_FRAC * L)
    B = max(burns)
    return (min(B, cap), B > cap)


def process_row(tag: str, rec: dict, global_burn=None) -> dict:
    """
    Collapse one process to the scalars the cross-process figures need.

    `global_burn` sets the window the PRIMARY area is computed over; see
    global_burn_in.  Called once with None to discover the per-process burn-ins,
    then again with the global value.  Both passes are pure functions of the
    stored trajectories, so this costs nothing and changes no training.

    `*_conv` variants are computed over converged repeats only, using the mask
    paired_delta_ce already produces.  For the area that variant is the headline
    (see the module docstring); for delta_CE it is the secondary.
    """
    spec = rec["spec"]
    paired = rec.get("paired") or {}
    tm = trajectory_metrics(rec, global_burn=global_burn)
    if not tm:
        return {}

    conv = paired.get("converged")
    conv = (np.asarray(conv, dtype=bool) if conv is not None
            else np.ones(len(rec["runs"]), dtype=bool))
    n_div = sum(r["fw"]["diverged"] or r["bw"]["diverged"] for r in rec["runs"])

    def agg(key):
        v = np.asarray(tm[key], dtype=float)
        m = conv[:len(v)]
        return _stats(v), _stats(v[m] if m.any() else v)

    a_pri_all, a_pri_conv = agg("area_primary")     # the one plotted
    a_set_all, _          = agg("area_settled")
    a_fix_all, _          = agg("area_fixed")
    a_ful_all, _          = agg("area_full")
    _,         ds_conv    = agg("d_step")

    s_fw = _stats([r["fw"]["S_emp"] for r in rec["runs"]])
    s_bw = _stats([r["bw"]["S_emp"] for r in rec["runs"]])

    gap = spec["C_minus"] - spec["C_plus"]
    return dict(
        tag=tag, kind=spec["kind"], gap=gap,
        C_plus=spec["C_plus"], C_minus=spec["C_minus"],
        theory=spec["theory"], vocab=spec["num_token"],
        p=spec.get("p"), q=spec.get("q"), n=spec.get("n"), m=spec.get("m"),
        n_repeats=len(rec["runs"]), n_div=int(n_div), n_conv=int(conv.sum()),
        # primary
        dce=paired.get("all_folds", {}).get("mean", np.nan),
        dce_sem=paired.get("all_folds", {}).get("sem", np.nan),
        dce_conv=(paired.get("converged_only") or {}).get("mean", np.nan),
        dce_conv_sem=(paired.get("converged_only") or {}).get("sem", np.nan),
        verdict=_verdict(paired.get("all_folds", {}).get("mean", np.nan),
                         paired.get("all_folds", {}).get("sem", np.nan), gap),
        # exploratory.  `area` is over the sweep-wide window on converged
        # repeats; the other three are the same integral over other windows,
        # kept so the choice of window is auditable rather than buried.
        area=a_pri_conv["mean"], area_sem=a_pri_conv["sem"],
        area_all=a_pri_all["mean"], area_all_sem=a_pri_all["sem"],
        area_own_burn=a_set_all["mean"], area_fixed=a_fix_all["mean"],
        area_full=a_ful_all["mean"],
        d_step=ds_conv["mean"], d_step_sem=ds_conv["sem"],
        burn_settled=tm["burn_settled"], burn_fixed=tm["burn_fixed"],
        burn_used=tm["burn_used"], L=tm["L"],
        mean_D=tm["mean_D"],
        # estimator error, for the third scatter panel
        s_err_fw=s_fw["mean"] - spec["C_plus"],
        s_err_bw=s_bw["mean"] - spec["C_minus"],
    )


# ══════════════════════════════════════════════════════════════════════════
# CROSS-PROCESS STATISTICS
# ══════════════════════════════════════════════════════════════════════════
def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan"), float("nan"), int(ok.sum())
    rho, p = sstats.spearmanr(x[ok], y[ok])
    return float(rho), float(p), int(ok.sum())


def _partial_spearman(x, y, z):
    """
    Spearman rho between x and y with the linear effect of z removed from the
    RANKS of both.

    Used for the coin grid, where C- - C+ and H_inf correlate at +0.59, so an
    uncorrected rho cannot distinguish "tracks the asymmetry" from "tracks the
    entropy rate".
    """
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 5:
        return float("nan"), float("nan"), int(ok.sum())
    rx, ry, rz = (sstats.rankdata(v[ok]) for v in (x, y, z))
    if rz.std() == 0:
        # z is constant on this grid, so there is nothing to control for and
        # polyfit would return nan.  Reported as n/a rather than as a number.
        return float("nan"), float("nan"), int(ok.sum())
    resid = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)
    rho, p = sstats.pearsonr(resid(rx), resid(ry))
    return float(rho), float(p), int(ok.sum())


# A point whose standard error is this many times the family's median is not a
# measurement -- it is one or two diverged repeats.  Measured at lambda=0.03:
# n10_m8 carried 2 diverged repeats and a sem 67x the median, n8_m10 one and 51x,
# and their delta_CE of -0.069 and -0.033 (against a typical 0.002) set the whole
# y-axis by themselves.  Both are also the late-settling large-vocabulary cells.
UNSTABLE_SEM_FACTOR = 5.0


def unstable_mask(rows: list, key: str) -> np.ndarray:
    """
    Which points are too noisy to plot or correlate, judged per response.

    Instability is response-specific: a process can have a wild delta_CE and a
    perfectly tight area, so the mask is recomputed for each panel rather than
    shared.  The threshold is relative to the family's own median sem, so it
    transfers across lambda values and across response variables whose units
    differ by orders of magnitude.

    Excluded points are never hidden: the count and the tags are annotated on the
    panel, printed in the summary, and the correlation is reported both with and
    without them.
    """
    sem = np.array([r.get(key + "_sem", np.nan) for r in rows], dtype=float)
    ok = np.isfinite(sem) & (sem > 0)
    if ok.sum() < 3:
        return np.zeros(len(rows), dtype=bool)
    med = float(np.median(sem[ok]))
    if med <= 0:
        return np.zeros(len(rows), dtype=bool)
    return np.nan_to_num(sem, nan=np.inf) > UNSTABLE_SEM_FACTOR * med


def mixed_windows(rows: list) -> bool:
    """True when the families' areas were integrated over different windows."""
    return len({r.get("burn_used") for r in rows}) > 1


def weighted_slope(rows: list, key: str, sem_key: str | None = None) -> dict:
    """
    Inverse-variance weighted least squares of `key` on the theoretical gap.

    REMAINING_WORK_PLAN.md S3.  The unweighted fit this supplements is not
    trustworthy here, and the reason is measurable: on the 81-cell flower grid the
    per-cell sems span 249x (0.00038 to 0.09442).  OLS assumes they are equal, so
    a handful of unstable cells dominate the residual variance and flatten the
    slope --

        OLS, all 81 cells        b = +0.00344 +/- 0.00248   t = +1.38   (reads null)
        OLS, unstable dropped    b = +0.00317 +/- 0.00025   t = +12.7
        WLS, all 81 cells        b = +0.00294 +/- 0.00016   t = +18.9

    -- and the WLS estimate is identical with or without the unstable filter,
    because weighting down-weights those cells instead of discarding them.  That
    is the right estimator when the y-errors are known and heteroscedastic, which
    is exactly this situation: every point carries its own measured sem.

    `unstable_mask` is kept for the FIGURES, where clipping the y-range genuinely
    aids reading, but it no longer decides which cells enter the inference.
    """
    sem_key = sem_key or f"{key}_sem"
    x = np.asarray([r["gap"] for r in rows], float)
    y = np.asarray([r.get(key, np.nan) for r in rows], float)
    s = np.asarray([r.get(sem_key, np.nan) for r in rows], float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    if ok.sum() < 4:
        return {}
    x, y, s = x[ok], y[ok], s[ok]
    w = 1.0 / s ** 2
    X = np.vstack([x, np.ones_like(x)]).T
    XtW = X.T * w
    try:
        beta = np.linalg.solve(XtW @ X, XtW @ y)
        cov = np.linalg.inv(XtW @ X)
    except np.linalg.LinAlgError:
        return {}
    b, a = float(beta[0]), float(beta[1])
    b_se, a_se = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    # chi2/dof > 1 says the scatter exceeds the quoted sems, i.e. there is
    # process-to-process variation the error bars do not account for.
    resid = y - (b * x + a)
    chi2 = float(np.sum(w * resid ** 2))
    dof = max(len(x) - 2, 1)
    return dict(slope=b, slope_se=b_se, t=b / b_se if b_se > 0 else float("nan"),
                intercept=a, intercept_se=a_se, n=int(len(x)),
                chi2_dof=chi2 / dof, sem_ratio=float(s.max() / s.min()))


def block_bootstrap_rho(rows: list, key: str, n_boot: int = 2000,
                        seed: int = 0) -> dict:
    """
    Spearman rho with a confidence interval from resampling grid ROWS, not cells.

    REMAINING_WORK_PLAN.md S6.  The flower grid is a 9x9 lattice over (n,m) of a
    SMOOTH deterministic function of the process parameters, with one dice draw
    per cell, so neighbouring cells are near-duplicate processes.  The effective
    degrees of freedom are far below the cell count and the nominal Spearman
    p-value -- 1.9e-21 on 81 cells -- is correspondingly anti-conservative.  The
    module docstring has always said so; this reports a number that acts on it.

    Resampling whole rows (all cells sharing an `n`, or sharing a `p` for the
    coins) keeps within-row dependence intact instead of assuming it away.  It is
    a blunt instrument -- rows are not independent of each other either -- but it
    is honest in the right direction, which the nominal p-value is not.
    """
    rows = [r for r in rows if np.isfinite(r.get(key, np.nan))]
    if len(rows) < 8:
        return {}
    block_of = (lambda r: r["n"]) if rows[0]["kind"] == "flower" else (lambda r: r["p"])
    blocks: dict = {}
    for r in rows:
        blocks.setdefault(block_of(r), []).append(r)
    keys = list(blocks)
    if len(keys) < 3:
        return {}
    rng = np.random.default_rng(seed)
    obs = _spearman([r["gap"] for r in rows], [r[key] for r in rows])[0]
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sub = [r for i in pick for r in blocks[keys[i]]]
        rho = _spearman([r["gap"] for r in sub], [r[key] for r in sub])[0]
        if np.isfinite(rho):
            draws.append(rho)
    if len(draws) < 100:
        return {}
    draws = np.sort(np.asarray(draws))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # A sign test is what the hypothesis actually predicts, so report the
    # fraction of resamples that keep the observed sign.
    same_sign = float(np.mean(np.sign(draws) == np.sign(obs))) if obs else float("nan")
    return dict(rho=float(obs), lo=float(lo), hi=float(hi), n_blocks=len(keys),
                n_boot=len(draws), sign_stability=same_sign)


def correlations(rows: list, key: str, drop_pooled: bool = False) -> list:
    """
    (label, rho, p, n) for the pooled and the within-family tests.

    `drop_pooled` omits the pooled row entirely rather than printing a number
    that cannot be interpreted -- used for the area when the two families were
    integrated over different step windows, since the pooled rank correlation
    would then be mixing two different quantities.
    """
    out = []
    bad = unstable_mask(rows, key)
    stable = [r for r, b in zip(rows, bad) if not b]
    groups = (("coin", [r for r in stable if r["kind"] == "coin"]),
              ("flower", [r for r in stable if r["kind"] == "flower"]))
    if not drop_pooled:
        groups = (("pooled", stable),) + groups
    for label, sub in groups:
        if len(sub) < 4:
            continue
        rho, p, n = _spearman([r["gap"] for r in sub], [r[key] for r in sub])
        out.append((label, rho, p, n))
        if label == "coin":
            rho, p, n = _partial_spearman([r["gap"] for r in sub],
                                          [r[key] for r in sub],
                                          [r["theory"] for r in sub])
            out.append(("coin | H_inf", rho, p, n))
    # Nothing is hidden: when points were dropped, the all-points value is shown
    # beside the filtered one so the reader can see what the filter did.
    if bad.any():
        for label, sub in (("coin", [r for r in rows if r["kind"] == "coin"]),
                           ("flower", [r for r in rows if r["kind"] == "flower"])):
            if len(sub) >= 4 and any(b for r, b in zip(rows, bad)
                                     if r["kind"] == sub[0]["kind"]):
                rho, p, n = _spearman([r["gap"] for r in sub],
                                      [r[key] for r in sub])
                out.append((f"{label} + unstable", rho, p, n))
    return out


# ══════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════
def _corr_caption(corrs) -> str:
    """One correlation per line — on one line they run into the next panel."""
    return "\n".join(
        f"{lbl}: ρ={rho:+.3f} (p={p:.3f}, n={n})" if np.isfinite(rho)
        else f"{lbl}: n/a"
        for lbl, rho, p, n in corrs)


def _scatter_panel(ax, rows, key, sem_key, ylabel, title, corrs):
    """
    One response variable against C- - C+, coloured by family.

    Points flagged by unstable_mask are dropped from the drawing, from the
    y-limits and from the correlations.  Keeping them made the panel unreadable:
    at lambda=0.03 two cells with diverged repeats carried delta_CE of -0.069 and
    -0.033 against a typical 0.002, so the trend in the other 23 collapsed onto
    the zero line.  The exclusion is annotated on the panel, never silent.
    """
    bad = unstable_mask(rows, key)
    dropped = [r["tag"].replace("sweep_flower_", "").replace("sweep_coin_", "")
               for r, b in zip(rows, bad) if b]
    rows = [r for r, b in zip(rows, bad) if not b]
    for kind, colour, marker in (("coin", COIN_COLOUR, "o"),
                                 ("flower", FLOWER_COLOUR, "s")):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        x = np.array([r["gap"] for r in sub])
        y = np.array([r[key] for r in sub], dtype=float)
        e = np.array([r.get(sem_key, np.nan) for r in sub], dtype=float)
        ax.errorbar(x, y, yerr=np.nan_to_num(e), fmt=marker, ms=4.5,
                    color=colour, ecolor=colour, elinewidth=0.7, capsize=1.5,
                    alpha=0.75, linestyle="none",
                    label=f"{kind} ({len(sub)})")
    ax.axhline(0.0, color="k", ls="--", lw=0.9)
    ax.axvline(0.0, color="k", ls="--", lw=0.9)

    # Robust, symmetric y-limits.  A single cell whose repeats included a
    # divergence carries a sem two orders of magnitude above the rest (0.074 vs
    # 0.0005) and, left to autoscale, flattens the other 124 points onto the zero
    # line.  Symmetric about zero because the SIGN is what the panel is read for.
    y = np.array([r[key] for r in rows], dtype=float)
    e = np.nan_to_num(np.array([r.get(sem_key, np.nan) for r in rows], dtype=float))
    edge = np.concatenate([y - e, y + e])
    edge = edge[np.isfinite(edge)]
    if edge.size:
        span = float(np.nanpercentile(np.abs(edge), 97))
        span = max(span, float(np.nanpercentile(np.abs(y[np.isfinite(y)]), 100)) * 0.2, 1e-9)
        ax.set_ylim(-span * 1.35, span * 1.35)
        n_out = int((np.abs(y) > span * 1.35).sum())
        note = []
        if dropped:
            note.append(f"excluded, sem > {UNSTABLE_SEM_FACTOR:g}× median: "
                        + ", ".join(dropped))
        if n_out:
            note.append(f"{n_out} plotted point(s) outside the frame")
        if note:
            # Lower LEFT: the legend sits lower right and the two collided.
            ax.text(0.02, 0.02, "\n".join(note), transform=ax.transAxes,
                    fontsize=6.5, ha="left", va="bottom", color="0.35")

    ax.set_xlabel("C− − C+ (bits, closed form)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, fontweight="bold")
    # Correlations inside the axes rather than in the title: as a title they are
    # four lines tall and collide with the panel above.
    ax.text(0.02, 0.98, _corr_caption(corrs), transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="0.7", alpha=0.9))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_sweep_scatter(rows: list, out_root: str):
    """
    Four panels, all against C- - C+.

    The hypothesis predicts points confined to the lower-left and upper-right
    quadrants of panel 1 -- and only of panel 1, since it is the only response
    variable the theory speaks to.  Spearman rho per family is printed on each
    panel; the pooled value is there for completeness and should not be read as
    the test (see the module docstring).
    """
    if len(rows) < 4:
        print("  too few processes for the scatter")
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    _scatter_panel(axes[0][0], rows, "dce", "dce_sem",
                   "delta_CE = CE_BW − CE_FW (bits)",
                   "PRIMARY — final held-out delta_CE, ±1 sem over repeats",
                   correlations(rows, "dce"))
    mixed = mixed_windows(rows)
    win = ",  ".join(f"{k}: {w}–{rows[0]['L']}" for k, w in sorted(
        {r["kind"]: r["burn_used"] for r in rows}.items()))
    _scatter_panel(axes[0][1], rows, "area", "area_sem",
                   "mean (CE_BW − CE_FW) over the window (bits)",
                   "EXPLORATORY — trajectory area (normalised: mean over the "
                   f"window), converged repeats\nwindow {win}"
                   + ("   — families differ, so NO pooled ρ" if mixed else ""),
                   correlations(rows, "area", drop_pooled=mixed))
    _scatter_panel(axes[1][0], rows, "d_step", "d_step_sem",
                   "steps to settle: backward − forward",
                   "EXPLORATORY — convergence-step difference",
                   correlations(rows, "d_step"))

    # Panel 4: how far the fixed-k complexity estimator sits from the closed
    # form, per arm.  Not a test of the hypothesis -- a check on the instrument.
    ax = axes[1][1]
    for kind, marker in (("coin", "o"), ("flower", "s")):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        x = [r["gap"] for r in sub]
        ax.plot(x, [r["s_err_fw"] for r in sub], marker, ms=4, alpha=0.7,
                color="#4c72b0", linestyle="none",
                label=f"{kind} forward (S_emp − C+)")
        ax.plot(x, [r["s_err_bw"] for r in sub], marker, ms=4, alpha=0.7,
                color="#dd8452", linestyle="none",
                label=f"{kind} backward (S_emp − C−)")
    ax.axhline(0.0, color="k", ls="--", lw=0.9)
    ax.axvline(0.0, color="k", ls="--", lw=0.9)
    ax.set_xlabel("C− − C+ (bits, closed form)")
    ax.set_ylabel("S_emp − closed form (bits)")
    ax.set_title("INSTRUMENT CHECK — fixed-k complexity error per arm\n"
                 "S_emp is k-means at an assumed k, so S ≤ log2(k); it "
                 "overestimates on unbalanced occupancy", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    n_c = sum(r["kind"] == "coin" for r in rows)
    n_f = sum(r["kind"] == "flower" for r in rows)
    reps = sorted({r["n_repeats"] for r in rows})
    fig.suptitle(
        f"Sweep — does measured asymmetry track theoretical asymmetry?  "
        f"{len(rows)} processes ({n_c} coin, {n_f} flower), "
        f"repeats per point: {reps}\n"
        "Only panel 1 tests the hypothesis.  Every negative-x point is a flower "
        "process, so the POOLED ρ conflates family with the sign of x — the "
        "flower row is the sign test, the coin row a magnitude test.\n"
        "C− − C+ is a smooth function of (p,q) and (n,m), so neighbouring cells "
        "are near-duplicate processes and the nominal p-values are "
        "anti-conservative.",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, os.path.join(out_root, "sweep_scatter.png"))


def plot_sweep_trajectories(rows: list, out_root: str):
    """
    One mean D(s) curve per process, coloured by C- - C+ on a diverging map
    centred at zero (the TwoSlopeNorm convention plot_diff_heatmap already uses).

    If the hypothesis holds at the trajectory level the curves stratify by
    colour: warm above zero, cool below.  If not, they are a tangle.  Two panels
    because the two families must be separable -- pooled, the eye would read the
    family difference as the trend.
    """
    if not rows:
        return
    # Same exclusion as the scatter, judged on the primary response: a process
    # whose delta_CE is one diverged repeat also has a mean D(s) curve that would
    # set the y-limits for everyone else.
    bad = unstable_mask(rows, "dce")
    n_drop = int(bad.sum())
    rows = [r for r, b in zip(rows, bad) if not b]
    if not rows:
        return
    gaps = np.array([r["gap"] for r in rows])
    lo, hi = float(gaps.min()), float(gaps.max())
    norm = TwoSlopeNorm(vmin=min(lo, -1e-6), vcenter=0.0, vmax=max(hi, 1e-6))
    cmap = plt.cm.RdBu_r

    # Scale to the settled portions only -- the transient reaches whole bits and
    # would flatten everything the figure is about.  Computed BEFORE anything is
    # drawn: an annotation positioned against a y-limit that is later shrunk ends
    # up far outside the axes, and savefig(bbox_inches="tight") then grows the
    # canvas to include it.  That produced a 462-inch figure once.
    settled = [np.abs(np.asarray(r["mean_D"], dtype=float)[r["burn_settled"]:])
               for r in rows]
    settled = [s for s in settled if s.size]
    span = (float(np.nanpercentile(np.concatenate(settled), 99.5))
            if settled else 1e-3)
    ylim = (-span * 1.6, span * 1.6)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    for ax, kind in zip(axes, ("coin", "flower")):
        ax.set_ylim(*ylim)
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            ax.set_title(f"{kind} — nothing run", fontsize=10)
            continue
        # Order so the extreme colours are drawn last and stay visible.
        for r in sorted(sub, key=lambda r: abs(r["gap"])):
            D = np.asarray(r["mean_D"], dtype=float)
            ax.plot(np.arange(D.size), D, lw=1.0, alpha=0.75,
                    color=cmap(norm(r["gap"])))
        ax.axhline(0.0, color="k", ls="--", lw=1.0)
        burn = int(np.median([r["burn_settled"] for r in sub]))
        ax.axvline(burn, color="0.4", ls=":", lw=1.2)
        # Blended transform: x in data, y in axes fraction, so the label cannot
        # sit outside the visible range whatever the limits are.
        ax.annotate(" median settling step", xy=(burn, 0.97),
                    xycoords=("data", "axes fraction"), fontsize=7,
                    color="0.4", va="top")
        ax.set_xlabel("Gradient step")
        ax.set_title(f"{kind} — {len(sub)} processes", fontsize=10,
                     fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(*ylim)

    axes[0].set_ylabel("mean (CE_BW − CE_FW) (bits)")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, label="C− − C+ (bits)", fraction=0.03, pad=0.02)
    fig.suptitle(
        "Per-step paired difference, one mean curve per process, coloured by "
        "theoretical asymmetry\n"
        "stratification by colour would be trajectory-level evidence; y is "
        "clipped to the settled range so the transient is off-scale"
        + (f"\n{n_drop} process(es) excluded as unstable "
           f"(delta_CE sem > {UNSTABLE_SEM_FACTOR:g}× median)" if n_drop else ""),
        fontsize=11, fontweight="bold")
    savefig(fig, os.path.join(out_root, "sweep_trajectories.png"))


# ══════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════
def baseline_crosscheck(rows: list, baseline_path: str):
    """
    Free regression test on the whole pipeline.

    The default flower grid contains (2,6), (2,8), (4,2) and (6,4) -- every
    flower process the 100-repeat baseline ran, at the same dice seed.  If the
    sweep is measuring the same thing, those cells must reproduce the baseline
    delta_CE within their combined error.  A disagreement means something in the
    spec construction or the aggregation differs, and would be invisible in the
    scatter.
    """
    base = load_combined(baseline_path)
    if not base:
        # REORGANISATION_FIX_PLAN.md 5.5.  This used to `return` in silence, so a
        # stale --baseline path deleted the sweep's only end-to-end regression
        # check without leaving a trace in the log.  A cross-check that can
        # vanish unnoticed is not a check.
        print(f"\n  ! BASELINE CROSS-CHECK SKIPPED — no readable baseline at "
              f"{baseline_path}")
        print(f"  !   the four overlapping flower cells were NOT verified against "
              f"the 100-repeat run.")
        print(f"  !   pass --baseline <path to all_trajectories.pkl> to restore it.")
        return
    by_key = {}
    for tag, rec in base.items():
        s = rec.get("spec") or {}
        if not rec.get("runs") or not rec.get("paired"):
            continue
        key = (s["kind"], s.get("p"), s.get("q"), s.get("n"), s.get("m"))
        by_key[key] = (tag, rec["paired"])

    hits = []
    for r in rows:
        key = (r["kind"], r.get("p"), r.get("q"), r.get("n"), r.get("m"))
        if key in by_key:
            hits.append((r, *by_key[key]))
    if not hits:
        return

    print(f"\n{'='*100}\n  BASELINE CROSS-CHECK against {baseline_path}\n{'='*100}")
    print(f"  {'process':<22}{'sweep dCE':>12}{'sem':>9}{'n':>5}"
          f"{'baseline dCE':>14}{'sem':>9}{'n':>5}{'z':>8}  agree?")
    for r, btag, bp in hits:
        dm = r["dce"] - bp["mean"]
        se = float(np.sqrt(np.nan_to_num(r["dce_sem"]) ** 2
                           + np.nan_to_num(bp.get("sem", 0.0)) ** 2))
        z = dm / se if se > 0 else float("nan")
        ok = "yes" if (not np.isfinite(z) or abs(z) < 2) else "NO -- investigate"
        print(f"  {r['tag'].replace('sweep_',''):<22}{r['dce']:>+12.5f}"
              f"{r['dce_sem']:>9.5f}{r['n_repeats']:>5}"
              f"{bp['mean']:>+14.5f}{bp.get('sem', float('nan')):>9.5f}"
              f"{bp['n']:>5}{z:>+8.2f}  {ok}")
    print(f"{'='*100}")


def print_sweep_summary(rows: list, out_root: str):
    """Per-process table, then the correlations that answer the question."""
    print(f"\n{'='*118}\n  SWEEP SUMMARY  ({out_root})\n{'='*118}")
    print(f"  {'process':<22}{'C- - C+':>9}{'H_inf':>7}{'V':>4}{'n':>4}{'div':>4}"
          f"{'dCE':>10}{'sem':>8}{'verdict':>9}{'area':>10}{'sem':>8}"
          f"{'dstep':>7}{'settles':>8}")
    bad_dce = unstable_mask(rows, "dce")
    flagged = {r["tag"] for r, b in zip(rows, bad_dce) if b}
    for r in sorted(rows, key=lambda r: (r["kind"], -r["gap"])):
        mark = "  <-- UNSTABLE, excluded" if r["tag"] in flagged else ""
        print(f"  {r['tag'].replace('sweep_',''):<22}{r['gap']:>+9.4f}"
              f"{r['theory']:>7.3f}{r['vocab']:>4}{r['n_repeats']:>4}{r['n_div']:>4}"
              f"{r['dce']:>+10.5f}{r['dce_sem']:>8.5f}{r['verdict']:>9}"
              f"{r['area']:>+10.5f}{r['area_sem']:>8.5f}"
              f"{r['d_step']:>+7.1f}{r['burn_settled']:>8}{mark}")
    if flagged:
        sem = np.array([r["dce_sem"] for r in rows], float)
        med = float(np.median(sem[np.isfinite(sem)]))
        print(f"\n  {len(flagged)} process(es) excluded from the delta_CE figures "
              f"and correlations: their sem exceeds {UNSTABLE_SEM_FACTOR:g}x the "
              f"median of {med:.5f}.")
        for r in rows:
            if r["tag"] in flagged:
                print(f"      {r['tag'].replace('sweep_',''):<22}"
                      f"dCE={r['dce']:+.5f}  sem={r['dce_sem']:.5f} "
                      f"({r['dce_sem']/med:.0f}x median)  "
                      f"{r['n_div']} diverged repeat(s)  settles at "
                      f"{r['burn_settled']}/{r['L']}")
        print("      They are one or two diverged repeats, not measurements.  "
              "The rows above keep them so the\n      exclusion can be checked, "
              "and the correlations are reported both with and without.")
    print(f"\n  'settles' is that process's own convergence step; 'area' is the "
          f"mean difference over the SHARED window steps "
          f"{rows[0].get('burn_used', 0)}–{rows[0]['L']}.")

    print(f"\n{'-'*118}\n  DOES IT TRACK?  Spearman rho of each response against "
          f"C- - C+\n{'-'*118}")
    mixed = mixed_windows(rows)
    win = ",  ".join(f"{k} {w}–{rows[0]['L']}" for k, w in sorted(
        {r["kind"]: r["burn_used"] for r in rows}.items()))
    for key, label, note in (
        ("dce",    "final delta_CE     (PRIMARY)", "the only response the theory speaks to"),
        ("area",   "trajectory area    (exploratory)",
         f"converged repeats, window: {win}"
         + ("  -- families differ, pooled rho omitted" if mixed else "")),
        ("d_step", "convergence steps  (exploratory)", "backward minus forward"),
    ):
        print(f"\n  {label}   — {note}")
        for lbl, rho, p, n in correlations(
                rows, key, drop_pooled=(mixed and key == "area")):
            flag = ""
            if lbl == "pooled":
                flag = "   <- conflates family with sign(x); not the test"
            if not np.isfinite(rho):
                print(f"      {lbl:<14} n/a (nothing to control for on this grid)")
                continue
            print(f"      {lbl:<14} rho={rho:+.3f}  p={p:.4f}  n={n}{flag}")

    # ── the slope, properly weighted, and rho with an honest interval ───────
    # REMAINING_WORK_PLAN.md S3 and S6.  rho above answers "is it monotone?";
    # these answer "how big is it, and how sure are we given that neighbouring
    # cells are near-duplicates?"  Both are per family, because a pooled slope
    # would be fitting across two different vocabularies and entropy ranges.
    print(f"\n{'-'*118}\n  HOW BIG, AND HOW SURE?  inverse-variance weighted "
          f"slope of delta_CE on C- - C+\n{'-'*118}")
    print(f"  {'family':<10}{'n':>4}{'WLS slope':>13}{'se':>9}{'t':>8}"
          f"{'intercept':>12}{'se':>9}{'chi2/dof':>10}{'sem ratio':>11}")
    for fam in ("flower", "coin"):
        sub = [r for r in rows if r["kind"] == fam]
        w = weighted_slope(sub, "dce")
        if not w:
            continue
        print(f"  {fam:<10}{w['n']:>4}{w['slope']:>+13.5f}{w['slope_se']:>9.5f}"
              f"{w['t']:>+8.2f}{w['intercept']:>+12.5f}{w['intercept_se']:>9.5f}"
              f"{w['chi2_dof']:>10.2f}{w['sem_ratio']:>10.0f}x")
        ols = sstats.linregress([r["gap"] for r in sub if np.isfinite(r["dce"])],
                                [r["dce"] for r in sub if np.isfinite(r["dce"])])
        print(f"    {'':<8}unweighted OLS for comparison: "
              f"b={ols.slope:+.5f} +/- {ols.stderr:.5f}  t={ols.slope/ols.stderr:+.2f}"
              + ("   <- the sem spread makes this the wrong estimator"
                 if w["sem_ratio"] > 10 else ""))
        bs = block_bootstrap_rho(sub, "dce")
        if bs:
            print(f"    {'':<8}rho={bs['rho']:+.3f}, block-bootstrap 95% CI "
                  f"[{bs['lo']:+.3f}, {bs['hi']:+.3f}] over {bs['n_blocks']} "
                  f"grid rows; sign stable in {bs['sign_stability']*100:.0f}% "
                  f"of resamples")
    print("    chi2/dof > 1 means the scatter exceeds the quoted sems, i.e. there")
    print("    is process-to-process variation the error bars do not capture.")

    # ── what the coin grid is actually measuring ───────────────────────────
    # REMAINING_WORK_PLAN.md S2.  Reported here rather than left to the reader,
    # because the raw coin rho has the WRONG SIGN and vanishes under control.
    coin = [r for r in rows if r["kind"] == "coin" and np.isfinite(r["dce"])]
    if len(coin) >= 8:
        g = [r["gap"] for r in coin]; d = [r["dce"] for r in coin]
        h = [r["theory"] for r in coin]
        raw = _spearman(g, d); par = _partial_spearman(g, d, h); hce = _spearman(h, d)
        print(f"\n{'-'*118}\n  THE COIN GRID IS A CONFOUND CONTROL, NOT EVIDENCE"
              f"\n{'-'*118}")
        print(f"  corr(C- - C+, H_inf) = {np.corrcoef(g, h)[0,1]:+.3f}  on this grid")
        print(f"  raw      rho(gap, dCE) = {raw[0]:+.3f}  p={raw[1]:.2g}")
        print(f"  partial  rho | H_inf   = {par[0]:+.3f}  p={par[1]:.2f}")
        print(f"           rho(H_inf,dCE)= {hce[0]:+.3f}  p={hce[1]:.2g}")
        if abs(par[0]) < abs(raw[0]) / 2:
            print("  Reading: the raw coin trend is an H_inf artefact -- at a fixed step")
            print("  budget, higher-entropy processes sit further from convergence and")
            print("  their two arms' residuals differ differently.  Controlling for")
            print("  H_inf removes it.  This grid therefore contributes no evidence")
            print("  about causal asymmetry; it is a positive control showing the")
            print("  measurement DOES respond to entropy rate, which is precisely why")
            print("  the flower grid -- where corr(gap, H_inf) ~ 0 -- carries the claim.")
            print("  Never quote a coin rho without its partial.")

    n_neg = sum(r["gap"] < -0.01 for r in rows)
    n_match = sum(r["verdict"] == "match" for r in rows)
    n_mis = sum(r["verdict"] == "MISMATCH" for r in rows)
    n_ns = sum(r["verdict"] == "n.s." for r in rows)
    agree = sum((r["dce"] > 0) == (r["gap"] > 0) for r in rows if np.isfinite(r["dce"]))
    print(f"\n{'-'*118}")
    print(f"  verdicts: {n_match} match, {n_mis} MISMATCH, {n_ns} n.s.  "
          f"(converged = both arms within {CONV_TOL} bits of H∞)")
    print(f"  sign of delta_CE agrees with sign of C- - C+ in {agree}/{len(rows)} "
          f"processes ({n_neg} of which have C- < C+)")
    print("  Caveats that travel with every number above:")
    print("    * nominal p-values are anti-conservative -- neighbouring grid")
    print("      cells are near-duplicate processes, so the effective degrees of")
    print("      freedom are far below n.")
    print("    * no correction for testing many processes; an isolated verdict at")
    print("      the 2-sem threshold is expected by chance.")
    print("    * delta_CE is a difference of RESIDUALS, so a converged model with")
    print("      spare capacity gives ~0 whatever C- - C+ is.  Magnitudes near")
    print("      zero are the predicted outcome, not a refutation.")
    print(f"{'='*118}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Parameter sweep: does measured asymmetry track theory?")
    ap.add_argument("--config", default="QUICK", choices=sorted(CONFIGS),
                    help="configuration supplying model, data and optimiser "
                         "settings (default QUICK, as the baseline harness uses)")
    ap.add_argument("--repeats", type=int, default=30,
                    help="independent repeats per process (default 30)")
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed; repeat i uses base + i for BOTH arms")
    ap.add_argument("--accelerator", default=None,
                    choices=["auto", "cpu", "mps", "gpu"],
                    help="override cfg['accelerator'].  'cpu' is the only "
                         "bit-reproducible setting -- MPS reductions are "
                         "non-deterministic (README 1.3), so exact run-to-run "
                         "comparison needs this.  Default: the config's value, "
                         "i.e. unchanged behaviour.")
    ap.add_argument("--d-model", type=int, default=None,
                    help="override cfg['d_model'].  The residual argument "
                         "(README 1.2) predicts |delta_CE| shrinks as capacity "
                         "grows, and that has never been measured -- see "
                         "REMAINING_WORK_PLAN.md S5.  Give each capacity its "
                         "OWN --out-root: tags do not encode d_model, so two "
                         "capacities in one folder overwrite each other.")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--sweep-coin", nargs="*", default=None, metavar="P",
                    help="coin grid; bare flag uses the default 10-point grid")
    ap.add_argument("--sweep-flower", nargs="*", default=None, metavar="N",
                    help="flower grid; bare flag uses the default {2,4,6,8,10}")
    ap.add_argument("--only", nargs="+", default=None, metavar="TAG",
                    help="restrict to tags containing any of these substrings")
    ap.add_argument("--weight-decay", type=float, default=None, metavar="LAMBDA",
                    help="AdamW weight decay.  Omitted or 0.0 reproduces the "
                         "plain-Adam condition every earlier result used; a "
                         "nonzero value is a NEW condition, not a correction. "
                         "Recommended grid for this step budget: 0.03 0.1 0.3 1.0 "
                         "(see WEIGHT_DECAY_PLAN.md and the note in configs.py)")
    ap.add_argument("--khat", action="store_true",
                    help="also recover k-hat per arm per repeat.  Adds roughly "
                         "an hour per 2500 repeats; off by default")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the spec table and coverage, then stop")
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw the figures from the saved pickle")
    ap.add_argument("--redo", action="store_true",
                    help="retrain processes that are already complete instead of "
                         "skipping them")
    ap.add_argument("--area-burn-coin", type=int, default=None, metavar="STEP",
                    help="integrate the coin area from this gradient step "
                         "instead of the sweep-wide settled window")
    ap.add_argument("--area-burn-flower", type=int, default=None, metavar="STEP",
                    help="same, for the flower family.  Setting the two families "
                         "differently makes their areas non-comparable, so the "
                         "pooled correlation is then omitted rather than printed")
    ap.add_argument("--baseline",
                    default="All_Results/results_trajectories/all_trajectories.pkl",
                    help="pickle to cross-check overlapping processes against")
    ap.add_argument("--verbose", action="store_true",
                    help="do not suppress per-run training output")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = dict(CONFIGS[args.config])
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.d_model is not None:
        cfg["d_model"] = args.d_model
    if args.accelerator is not None:
        cfg["accelerator"] = args.accelerator

    # Neither flag given: run both grids, which is the plan's default sweep.
    both = args.sweep_coin is None and args.sweep_flower is None
    coin_grid = (parse_grid(args.sweep_coin, "coin")
                 if both or args.sweep_coin is not None else None)
    flower_grid = (parse_grid(args.sweep_flower, "flower")
                   if both or args.sweep_flower is not None else None)

    specs = sweep_specs(cfg, coin_grid, flower_grid)
    if args.only:
        specs = [s for s in specs if any(o in s["tag"] for o in args.only)]
        if not specs:
            raise SystemExit(f"--only {args.only} matched nothing")

    # REORGANISATION_FIX_PLAN.md 4.2.  Root-anchored: mkdir is exist_ok=True and
    # load_combined returns {} for a missing pickle, so a stale relative path
    # would silently discard every completed repeat and re-run the full grid.
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
    combined_path = os.path.join(out_root, "all_sweep.pkl")
    combined = load_combined(combined_path)

    print(f"\n{'='*78}")
    print(f"  config={args.config}  repeats={args.repeats}  "
          f"base_seed={cfg['seed']}  out_root={out_root}")
    wd = cfg["weight_decay"]
    print(f"  d_model={cfg['d_model']}  n_layers={cfg['n_layers']}  "
          f"lr={cfg['lr']}  chunk={cfg['train_chunk_len']}  khat={args.khat}")
    print(f"  optimiser=AdamW  weight_decay={wd}"
          + ("   (= plain Adam, the condition every earlier result used)" if wd == 0
             else f"   -> a weight shrinks by "
                  f"{(1-(1-cfg['lr']*wd)**130)*100:.1f}% over 130 steps from decay alone"))
    print(f"  coin grid   : {coin_grid}")
    print(f"  flower grid : {flower_grid}")
    print(f"  processes   : {len(specs)}")
    print(f"{'='*78}")
    coverage_table(specs)

    # REORGANISATION_FIX_PLAN.md 5.5.  The resume plan is computed HERE, above the
    # --dry-run return, so `--dry-run` reports what a real run would skip and what
    # it would train.  It used to be computed after that return, which meant the
    # one flag whose entire job is "tell me what you are about to do" could not
    # answer the most expensive question -- am I resuming, or silently starting a
    # 23 h grid over?  A stale out_root reads as "nothing done yet" (mkdir is
    # exist_ok=True, load_combined returns {} for a missing pickle), so this line
    # is the check that separates the two.
    todo = [s for s in specs
            if args.redo
            or len((combined.get(s["tag"], {}) or {}).get("runs", [])) < args.repeats]
    n_skip = len(specs) - len(todo)
    partial = [(s["tag"], len(combined[s["tag"]]["runs"])) for s in todo
               if s["tag"] in combined and combined[s["tag"]].get("runs")]
    print(f"  resume plan : {n_skip} complete and skipped, {len(todo)} to train"
          + ("  (--redo forces all)" if args.redo else ""))
    if not combined:
        print(f"  ! no existing all_sweep.pkl at {combined_path}")
        print(f"  ! nothing to resume from -- this trains all {len(specs)} "
              f"process(es) from scratch.")
        print(f"  ! if that is a surprise, check --out-root before committing "
              f"the compute.")
    if partial:
        print(f"  partial     : " + ", ".join(f"{t} ({n}/{args.repeats})"
                                              for t, n in partial))

    if args.dry_run:
        print("\n  --dry-run: nothing trained.")
        return

    if not args.plots_only:
        # REMAINING_WORK_PLAN.md C3: merge, do not overwrite.  --sweep-coin alone
        # must not erase the flower grid this folder also holds.
        save_run_config(
            os.path.join(out_root, f"run_config_{args.config}.json"),
            {"config": args.config, "repeats": args.repeats,
             "base_seed": cfg["seed"], "khat": args.khat,
             "coin_grid": coin_grid, "flower_grid": flower_grid,
             "n_processes": len(specs),
             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             **{k: v for k, v in cfg.items()}},
            keep_if_unset=("coin_grid", "flower_grid"))

        t0 = time.time()
        t_save = [0.0]          # mutable so the inner loop can update it
        done_repeats = 0
        # `todo` / `n_skip` were computed above, before the --dry-run return.
        total_repeats = len(todo) * args.repeats

        for pi, spec in enumerate(todo, 1):
            tag = spec["tag"]
            rec = dict(spec=spec, config=args.config, base_seed=cfg["seed"],
                       repeats=args.repeats, khat=args.khat, runs=[])
            print(f"\n  [{pi}/{len(todo)}] {tag}   "
                  f"C-−C+={spec['C_minus']-spec['C_plus']:+.4f}  "
                  f"H∞={spec['theory']:.4f}  V={spec['num_token']}")
            tp = time.time()
            for i in range(args.repeats):
                with quiet(not args.verbose):
                    run = one_repeat(spec, cfg, cfg["seed"] + i, khat=args.khat)
                rec["runs"].append(run)
                done_repeats += 1
                if run["fw"]["diverged"] or run["bw"]["diverged"]:
                    print(f"      ! repeat {i+1} DIVERGED")
                # Incremental, but throttled: a nine-hour run must not lose
                # everything to one crash, and rewriting a growing ~10 MB pickle
                # after each of 3750 repeats would be tens of gigabytes of writes
                # for no extra safety.  At most SAVE_EVERY_S of compute is at
                # risk, and the last repeat of each process always lands.
                last = i == args.repeats - 1
                if last or time.time() - t_save[0] > SAVE_EVERY_S:
                    with contextlib.redirect_stdout(io.StringIO()):
                        rec["paired"] = (paired_stats(rec)
                                         if len(rec["runs"]) > 1 else None)
                        combined[tag] = rec
                        save_pkl(combined, combined_path)
                    t_save[0] = time.time()
            el = time.time() - tp
            rate = (time.time() - t0) / max(done_repeats, 1)
            eta = rate * (total_repeats - done_repeats) / 3600
            p = rec["paired"] or {}
            print(f"      dCE={p.get('mean', float('nan')):+.5f} "
                  f"± {p.get('sem', float('nan')):.5f}   "
                  f"{el:.0f}s ({el/args.repeats:.1f}s/repeat)   "
                  f"ETA {eta:.1f} h")

    # ── analysis ─────────────────────────────────────────────────────────
    # Pass 1 discovers each process's own settled step; pass 2 recomputes the
    # primary area over one sweep-wide window so the areas are comparable.  Both
    # passes are pure functions of the stored trajectories.
    rows = [row for row in (process_row(t, r) for t, r in combined.items()
                            if r.get("runs")) if row]
    if not rows:
        print("  nothing to analyse")
        return
    B, capped = global_burn_in(rows)
    # A family may override the shared window.  Everything downstream reads
    # `burn_used` off each row, so nothing else needs to know.
    burn_by_kind = {"coin": args.area_burn_coin if args.area_burn_coin is not None else B,
                    "flower": args.area_burn_flower if args.area_burn_flower is not None else B}
    rows = [row for row in (
        process_row(t, r, global_burn=burn_by_kind.get(r["spec"]["kind"], B))
        for t, r in combined.items() if r.get("runs")) if row]
    own = [r["burn_settled"] for r in rows]
    print(f"\n  shared settled window would be steps {B}–{rows[0]['L']} "
          f"(slowest process settles at {max(own)}; per-process range "
          f"{min(own)}–{max(own)})"
          + ("  ! CAPPED — some process never settled" if capped else ""))
    for kind, w in sorted(burn_by_kind.items()):
        n_k = sum(r["kind"] == kind for r in rows)
        if not n_k:
            continue
        late = sum(r["kind"] == kind and r["burn_settled"] > w for r in rows)
        print(f"  area window for {kind:<7} steps {w}–{rows[0]['L']}"
              + (f"   ({late}/{n_k} of them are still unsettled at step {w}, "
                 f"so their area includes transient)" if late else
                 f"   (all {n_k} settled by then)"))
    if mixed_windows(rows):
        print("  ! the two families use DIFFERENT windows, so their areas are not "
              "comparable\n    with each other; the pooled area correlation is "
              "omitted rather than printed.")
    save_pkl({"rows": rows, "config": args.config, "global_burn": B,
              "burn_capped": capped, "burn_by_kind": burn_by_kind},
             os.path.join(out_root, "sweep_rows.pkl"))
    plot_sweep_scatter(rows, out_root)
    plot_sweep_trajectories(rows, out_root)
    baseline_crosscheck(rows, repo_path(args.baseline))
    print_sweep_summary(rows, out_root)


if __name__ == "__main__":
    main()
