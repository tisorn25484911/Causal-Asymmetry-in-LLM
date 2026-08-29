"""
run_discrete_v2.py — the DISCRETE_V2_PLAN.md experiment schedule.

    python Experimental_setup/run_discrete_v2.py --list
    python Experimental_setup/run_discrete_v2.py 07_seeds --dry-run
    python Experimental_setup/run_discrete_v2.py 07_seeds
    python Experimental_setup/run_discrete_v2.py all --dry-run

This file DECIDES WHAT TO RUN.  It trains nothing itself: every experiment is
an invocation of run_statistical_trj.py / run_sweep_experiment.py with one axis
moved, so the training, resume, provenance and analysis code is the same
already-tested path in every case.

Default is --dry-run=False but every experiment prints its full command line
before executing, and `--dry-run` prints without executing.  Use --dry-run
first: the schedule is ~61 h of compute in total and the per-experiment
estimates have NOT been measured at this configuration (see `--list`).

Why the base values are what they are
-------------------------------------
    n_states  = 5V   K is a BUDGET, not an estimate.  Measured, K at the exact
                     causal-state count FAILS -- flower(2,3) forward at its true
                     K=3 recovers 2 states at ARI 0.745 -- so slack is required.
    state_dim = V    carries no expressive power; setting it to K would make
                     state_matrix mostly reparameterisation freedom.
    beta = 1/(B*T)   the penalty is beta * S_emp and S_emp is a per-token mean,
                     so beta is per-token.  Bounded above by a collapse cliff
                     (~6e-4 at lr=1e-3); 1/N sits ~5x below it at this geometry.

All three resolve in Model_analysis.discrete_hparams and are never literals.
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PY   = sys.executable
_TRJ  = os.path.join(_HERE, "run_statistical_trj.py")
_SWP  = os.path.join(_HERE, "run_sweep_experiment.py")
_OUT  = "All_Results/discrete_v2"

# The plan's base geometry, needed to express beta multiples without hardcoding.
_BATCH, _CHUNK = 32, 256
_BETA_1N = 1.0 / (_BATCH * _CHUNK)          # 1.2207e-4


def _trj(folder, *extra, repeats=30):
    return [_PY, _TRJ, "--config", "DISCRETE", "--repeats", str(repeats),
            "--out-root", f"{_OUT}/{folder}", *map(str, extra)]


def _swp(folder, *extra, repeats=5):
    return [_PY, _SWP, "--config", "DISCRETE", "--repeats", str(repeats),
            "--out-root", f"{_OUT}/{folder}", *map(str, extra)]


def schedule() -> dict:
    """
    name -> (est_hours, why, [command, ...]).

    Order matters: 07_seeds first because it sets the error bar that decides
    whether anything else is a finding, and 06_beta_verify before 03/03b
    because those move beta mechanically and must not be confounded with it.
    """
    S = {}

    S["07_seeds"] = (4, "the noise floor -- run FIRST, nothing else is "
                        "interpretable without it",
                     [_trj("07_seeds/seed%d" % s, "--seed", s, repeats=30)
                      for s in range(5)])

    S["00_base"] = (4, "the reference run every other folder compares to",
                    [_trj("00_base", repeats=30)])

    # K grid is V-RELATIVE above the true count and absolute below it: an
    # absolute grid would be 20V for the coin and 6V for flower_n2_m8, i.e. a
    # different amount of slack per process.  V is not known here, so the
    # relative part is expressed as multiples and expanded per process by the
    # caller reading spec["num_token"].  These absolute values cover V=3..10.
    K_ABS = [1, 2, 3, 4, 6]
    K_REL = [1, 2, 3, 5, 8, 12]          # x V
    S["01_ksweep"] = (6, "K vs k_discovered -- the headline figure",
                      [_trj("01_ksweep/K%03d" % k, "--n-states", k, repeats=5)
                       for k in sorted(set(K_ABS + [m * v for m in K_REL
                                                    for v in (3, 5, 6, 8, 10)]))])

    S["02_statedim"] = (3, "is state_dim=V a real optimum or a lucky basin?",
                        [_trj("02_statedim/S%02d" % s, "--state-dim", s, repeats=5)
                         for s in (2, 3, 5, 8, 10, 16)])

    S["06_beta_verify"] = (3, "verify 1/N is inside the safe band -- NOT tuning",
                           [_trj("06_beta_verify/b%s" % lab,
                                 "--usage-beta", _BETA_1N * mult, repeats=5)
                            for lab, mult in (("0", 0.0), ("0p5N", 0.5),
                                              ("1N", 1.0), ("2N", 2.0),
                                              ("4N", 4.0))])

    # 03b before 03: it holds context fixed, so a collapse is attributable to
    # beta alone.  Each runs TWICE -- beta tracking 1/N, and beta pinned at the
    # base value -- because otherwise batch/chunk and beta are confounded.
    S["03b_batch"] = (4, "beta cliff with context HELD FIXED -- the clean control",
                      [_trj("03b_batch/B%03d_betaN" % b, "--batch", b, repeats=5)
                       for b in (2, 4, 8, 16, 32, 64)]
                      + [_trj("03b_batch/B%03d_betaFixed" % b, "--batch", b,
                              "--usage-beta", _BETA_1N, repeats=5)
                         for b in (2, 4, 8, 16, 32, 64)])

    S["03_seqlen"] = (5, "context length; beta moves with it by construction",
                      [_trj("03_seqlen/T%03d_betaN" % t, "--chunk-len", t, repeats=5)
                       for t in (128, 256, 512)]
                      + [_trj("03_seqlen/T%03d_betaFixed" % t, "--chunk-len", t,
                              "--usage-beta", _BETA_1N, repeats=5)
                         for t in (128, 256, 512)])

    S["05_budget"] = (4, "the budget is a CEILING: recovery peaks then decays "
                         "while CE keeps falling",
                      [_trj("05_budget/E%03d" % e, "--epochs", e, repeats=5)
                       for e in (50, 100, 150, 300)])

    S["04_capacity"] = (8, "the only completely untested axis -- run LAST",
                        [_trj("04_capacity/d%03d" % d, "--d-model", d, repeats=5)
                         for d in (16, 32, 64)])

    # ── the science ────────────────────────────────────────────────────
    S["08_pqsweep"] = (7, "coin (p,q) grid -> scatter vs C- - C+.  A CONFOUND "
                          "CONTROL, not evidence: its gap correlates +0.590 "
                          "with H_inf and the raw rho has the WRONG sign",
                       [_swp("08_pqsweep", "--sweep-coin",
                             0.15, 0.35, 0.55, 0.75, 0.95, repeats=5)])

    S["09_nmsweep"] = (7, "flower (n,m) grid -> the SIGN test.  Confounded with "
                          "m-n at corr +0.977; run_dice_experiment breaks that",
                       [_swp("09_nmsweep", "--sweep-flower",
                             2, 4, 6, 8, 10, repeats=5)])

    S["10_trajectories"] = (6, "loss curves, D(s), delta_CE.  Read delta_CE with "
                               "caution: measured, it swings -0.13..+0.26 bits "
                               "on (K,beta) alone against a 0.001-0.003 effect",
                            [_trj("10_trajectories", repeats=30)])
    return S


ORDER = ["07_seeds", "00_base", "01_ksweep", "02_statedim", "06_beta_verify",
         "03b_batch", "03_seqlen", "05_budget", "04_capacity",
         "08_pqsweep", "09_nmsweep", "10_trajectories"]


def main(argv=None):
    S = schedule()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", default=None,
                    choices=ORDER + ["all"], help="which folder to run")
    ap.add_argument("--list", action="store_true", help="show the schedule and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    a = ap.parse_args(argv)

    if a.list or a.experiment is None:
        total = sum(S[k][0] for k in ORDER)
        print(f"\n  DISCRETE_V2_PLAN.md schedule — {len(ORDER)} experiments, "
              f"~{total} h ESTIMATED\n")
        print(f"  base: n_states=5V  state_dim=V  beta=1/(B*T)={_BETA_1N:.4e} "
              f"(B={_BATCH}, T={_CHUNK})\n")
        print(f"  {'#':>3} {'experiment':<16} {'runs':>5} {'est h':>6}  why")
        print("  " + "-" * 100)
        for i, k in enumerate(ORDER):
            h, why, cmds = S[k]
            print(f"  {i:>3} {k:<16} {len(cmds):>5} {h:>6}  {why[:70]}")
        print(f"\n  estimates are EXTRAPOLATED from a smaller harness and are "
              f"not measurements;\n  time one repeat of 00_base and rescale "
              f"before committing to the schedule.\n")
        return 0

    names = ORDER if a.experiment == "all" else [a.experiment]
    for name in names:
        h, why, cmds = S[name]
        print(f"\n{'='*100}\n  {name}  ({len(cmds)} runs, ~{h} h est)\n  {why}\n{'='*100}")
        for c in cmds:
            print("  $ " + " ".join(str(x) for x in c))
            if not a.dry_run:
                r = subprocess.run(c, cwd=_ROOT)
                if r.returncode != 0:
                    print(f"  FAILED (exit {r.returncode}); stopping.")
                    return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
