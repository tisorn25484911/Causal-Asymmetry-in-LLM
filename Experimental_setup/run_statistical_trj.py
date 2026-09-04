"""
run_statistical_trj.py — repeat statistics for the causal-asymmetry test.

    python run_statistical_trj.py                       # 100 repeats, all 7 processes
    python run_statistical_trj.py --repeats 3           # shakedown
    python run_statistical_trj.py --only traj_coin_p030_q040 traj_flower_n2_m6
    python run_statistical_trj.py --khat                # also recover k-hat per run
    python run_statistical_trj.py --plots-only          # redraw from the saved pickle

Experimental plan
-----------------
Seven processes, all at the QUICK configuration's scale, capacity and learning
rate:

    coin    p=0.1 q=0.9  |  p=0.3 q=0.4  |  p=0.4 q=0.8
    flower  n=2 m=6  |  n=2 m=8  |  n=4 m=2  |  n=6 m=4

For each process a forward (tril) and a backward (triu) model are trained on the
SAME forward data, `--repeats` times.  Every run records its loss trajectory,
its final held-out cross-entropy and the empirical statistical complexity of
each arm.  Per process, three figures go to results_trajectories/<tag>/:

    <tag>_complexity.png     S_emp for both arms with error bars over the runs,
                             against the closed-form C+ / C-
    <tag>_final_loss.png     three bars -- forward, backward, backward - forward
                             -- with their statistical error
    <tag>_trajectories.png   every run at alpha 0.3, the two mean curves in bold,
                             and a shaded interval

and two cross-process figures at the root:

    summary_delta_ce.png     delta_CE with its sem per process, and delta_CE
                             against C- - C+
    summary_grid.png         all three quantities for all processes in one
                             3 x N grid -- the whole experiment on one sheet

Why this exists
---------------
Every delta_CE this repo has reported carries a `sem` computed over the five CV
folds of ONE seed, and those folds share a training set: it measures
fold-to-fold variability, not sampling variability over datasets, so it is a
lower bound on the real uncertainty and cannot support a claim about the sign of
delta_CE.  Here each repeat regenerates the process from a fresh seed, so the
spread across repeats IS sampling variability and the paired statistic is
interpretable.

What is held fixed, and why
---------------------------
  * The two arms of one repeat are PAIRED.  Repeat i uses seed = base + i for
    both arms, so they get identical sequences, identical chunk windows, an
    identical train/held-out split, identical batch order and identical weight
    initialisation.  Only the attention mask and the batch convention differ.
    An unpaired difference would be contaminated by split noise of the same
    order as the effect being measured.
  * `dice_probs` is drawn from cfg["flower_dice_seed"], NOT from the repeat
    seed.  The dice define the process -- they set C+ and C- -- so resampling
    them per repeat would average over different processes.  They are the same
    dice run_experiments.py trains on, which keeps these numbers comparable to
    that run's.
  * All three coin processes use the exp1 settings (num_samples, seq_len,
    epochs, batch), not exp1's for one and exp1_2's for another.  Uniform
    settings are what make the seven processes comparable to each other.

Two deliberate departures from the scaffolding this file started as
-------------------------------------------------------------------
  * ONE training run per repeat, not a 5-fold cross-validation.  The repeats
    replace the folds as the source of statistics.  Running a full CV per repeat
    would be 5x the compute (7000 trainings at 100 repeats) *and* would bias the
    per-repeat number, because train_test_val_pipeline reports the best fold by
    validation loss -- a minimum over five draws is not a sample of the same
    distribution as a single draw.
  * No weights are saved.  2 x repeats x 7 checkpoints is a lot of disk for
    models nothing downstream reads; the trajectories and scalars are the
    product here.

Caveats carried over from the main runs
---------------------------------------
  * Divergence.  Both processes contain deterministic transitions, so CE has no
    finite minimiser and training eventually blows up after reaching H_inf.
    Every run is screened with diagnose_divergence(), and the paired statistic
    is reported twice -- over all repeats and over converged repeats only.  At
    these settings a run is ~130 gradient steps, which is the low end of the
    observed onset window, so expect a nonzero rate.
  * S_emp is k-means at a hand-specified k, bounded by log2(k), and
    overestimates when the true state occupancy is unbalanced.  It is what the
    plan asks for and what the rest of the repo reports, so it is the headline
    here.  `--khat` additionally recovers k-hat and S_hat from the model's
    predictive distribution, which is the better estimator but costs an
    agglomerative clustering pass per arm per repeat.
  * The complexity is read over the whole dataset, ~80% of which that repeat
    trained on -- the same convention as run_experiments.py, kept so the numbers
    are comparable.  plot_state_clusters.py is the train/held-out split of that
    question.
  * The held-out set is passed as train_model's val_loader so a held-out
    trajectory is recorded alongside the training one.  Nothing selects on it --
    there is no model selection anywhere in this file -- so it stays a clean
    held-out estimate.
"""
# ── stdlib ─────────────────────────────────────────────────────────────────
import argparse
import contextlib
import io
import json
import os
import pickle
import time

# ── third-party ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── project ────────────────────────────────────────────────────────────────
from configs import CONFIGS
from Data_generation import CoinDataset, coin_generation
from Flower_process_generation import FlowerDataset, flower_process_generation
from DiscreteCausal_analysis import (
    causal_state_report, plot_causal_states, plot_causal_states_pair,
    print_state_summary,
)
from Model_analysis import (
    causal_state_count,
    causal_state_occupancy,
    discrete_hparams,
    coin_true_conditional,
    flower_complexity,
    flower_entropy_rate,
    paired_delta_ce,
    recover_causal_states,
    resolve_s_emp,
    savefig,
    statistical_complexity,
    statistical_complexity_empirical,
    S_EMP_NOTE,
)
from utils import (
    cleanup, coin_tag, entropy_rate_coin, flower_tag, mkdir, repo_path,
    save_pkl, save_run_config, to_cpu_for_analysis,
)
from Training_model import (
    _eval_loss_on_loader, diagnose_divergence, make_analysis_loader,
    make_chunked_loader, set_seed, test_train_validation, train_model,
)
from run_experiments import full_seq_len


# ══════════════════════════════════════════════════════════════════════════
# The seven processes named in the plan
# ══════════════════════════════════════════════════════════════════════════
COIN_PQ   = [(0.1, 0.9), (0.3, 0.4), (0.4, 0.8)]
FLOWER_NM = [(2, 6), (2, 8), (4, 2), (6, 4)]

# Historical location, kept only so an explicit --out-root can still reach the
# onehot baseline.  NEW runs go to All_Results/<model>/trajectories -- see
# _default_out_root -- so the two architectures cannot overwrite one another.
OUT_ROOT_LEGACY  = "All_Results/results_trajectories"
OUT_ROOT_DEFAULT = OUT_ROOT_LEGACY


def _default_out_root(model: str) -> str:
    """
    All_Results/<model>/trajectories.

    The existing results_* trees stay exactly where they are -- they are the
    onehot baseline and their provenance records repo-relative paths.  Replot or
    resume THOSE with an explicit --out-root All_Results/results_trajectories;
    without it a run against the new default finds an empty directory, and
    load_combined reads a missing pickle as "nothing done yet" and silently
    retrains everything.  That failure cost a 23-hour sweep once already.
    """
    return f"All_Results/{model}/trajectories"

FW_COLOUR, BW_COLOUR, D_COLOUR = "#4c72b0", "#dd8452", "crimson"

# Convergence band used to split "measurement" from "optimisation failure",
# matching paired_delta_ce's default: a run counts as converged when BOTH arms
# land within this many bits of the process entropy rate.
CONV_TOL = 0.10


def coin_spec(cfg: dict, p: float, q: float) -> dict:
    """Everything about one coin process that does not depend on the repeat."""
    return dict(
        tag         = coin_tag("traj", p, q),
        kind        = "coin",
        p           = p,
        q           = q,
        num_token   = cfg["coin_num_token"],
        num_samples = cfg["coin_num_samples"],
        seq_len     = cfg["coin_seq_len"],
        max_epochs  = cfg["coin_max_epochs"],
        batch       = cfg["coin_batch"],
        k_fw        = 2,                 # 2 forward causal states, any (p, q)
        k_bw        = 3,                 # 3 backward causal states
        # The TRUE counts, for comparison against what the bottleneck finds.
        # Not the model's K -- see n_states below.
        true_k_fw   = causal_state_count("coin", "forward"),
        true_k_bw   = causal_state_count("coin", "backward"),
        # Stationary probability of each causal state; its entropy IS C+/C-.
        # Compared against the measured occupancy; NOT a training target.
        occupancy_fw = causal_state_occupancy("coin", "forward", p=p, q=q),
        occupancy_bw = causal_state_occupancy("coin", "backward", p=p, q=q),
        # K, state_dim and beta, resolved from V and the geometry.  ONE K for
        # both arms: it is a budget, not an estimate, and unequal budgets would
        # be exactly the arm-dependent asymmetry the paired design excludes.
        **discrete_hparams(cfg, cfg["coin_num_token"], cfg["coin_batch"]),
        C_plus      = statistical_complexity(p, q, "forward"),
        C_minus     = statistical_complexity(p, q, "backward"),
        theory      = entropy_rate_coin(p, q),      # H_inf, both directions
    )


def flower_spec(cfg: dict, n: int, m: int,
                dice_seed: int | None = None, dice_alpha: float = 1.0) -> dict:
    """
    Everything about one flower process that does not depend on the repeat.

    The dice come from cfg["flower_dice_seed"], exactly as in
    run_experiments.experiment_2, so this is the same process that run trained
    on rather than a new one with the same (n, m).

    `dice_seed` / `dice_alpha` override that draw, for the dice-axis experiment
    (REMAINING_WORK_PLAN.md S1 / run_dice_experiment.py).  They exist because
    C- - C+ = [H(pi_outcome) - log2 n] / 2 depends on how much the n dice OVERLAP,
    not only on (n, m) -- so the theoretical asymmetry can be varied with (n, m)
    held completely fixed, which is the only way to separate it from m - n.

    The defaults reproduce the original draw exactly: dice_seed=None uses
    cfg["flower_dice_seed"], and np.full(m, 1.0) is elementwise identical to
    np.ones(m).  Verified against the behaviour fingerprint, so every existing
    result is unaffected.
    """
    seed = cfg["flower_dice_seed"] if dice_seed is None else dice_seed
    dice_probs = np.random.default_rng(seed).dirichlet(
        np.full(m, float(dice_alpha)), size=n)
    C_plus, C_minus = flower_complexity(n, m, dice_probs)
    return dict(
        tag         = flower_tag("traj", n, m),
        kind        = "flower",
        n           = n,
        m           = m,
        dice_probs  = dice_probs,
        num_token   = n + m,
        num_samples = cfg["flower_num_samples"],
        seq_len     = cfg["flower_seq_len"],     # CYCLES; tokens = 2 x this
        max_epochs  = cfg["flower_max_epochs"],
        batch       = cfg["flower_batch"],
        # The TRUE counts, for comparison against what the bottleneck finds.
        # Note true_k_bw is not m+1 in general -- outcomes whose posteriors
        # coincide are ONE backward state, and at n=1 they all are -- so it is
        # computed from the dice actually drawn, under the same merge rule as C-.
        true_k_fw   = causal_state_count("flower", "forward", n=n, m=m),
        true_k_bw   = causal_state_count("flower", "backward", n=n, m=m,
                                         dice_probs=dice_probs),
        occupancy_fw = causal_state_occupancy("flower", "forward", n=n, m=m),
        occupancy_bw = causal_state_occupancy("flower", "backward", n=n, m=m,
                                              dice_probs=dice_probs),
        # K = 5V, state_dim = V, beta = 1/(batch*chunk).  V = n+m here, so K
        # ranges 20..100 across the (n,m) grid rather than being a constant.
        **discrete_hparams(cfg, n + m, cfg["flower_batch"]),
        k_fw        = n + 1,
        k_bw        = m + 1,                     # an upper bound: outcomes that
                                                 # induce the same posterior over
                                                 # dice are ONE backward state
        C_plus      = C_plus,
        C_minus     = C_minus,
        theory      = flower_entropy_rate(n, m, dice_probs),
        # Provenance for the dice-axis experiment.  `None` means "the config's
        # flower_dice_seed", i.e. the historical process, so an old record and a
        # new one are distinguishable without guessing.
        dice_seed   = dice_seed,
        dice_alpha  = float(dice_alpha),
    )


def process_specs(cfg: dict) -> list:
    return ([coin_spec(cfg, p, q) for p, q in COIN_PQ]
            + [flower_spec(cfg, n, m) for n, m in FLOWER_NM])


# ══════════════════════════════════════════════════════════════════════════
# ONE REPEAT
# ══════════════════════════════════════════════════════════════════════════
def build_dataset(spec: dict, seed: int):
    """
    Regenerate the process from `seed`.

    set_seed covers numpy, which is what coin_generation and
    flower_process_generation draw from, so the sequences are a pure function of
    the repeat seed -- and identical for the two arms of that repeat.
    """
    set_seed(seed)
    if spec["kind"] == "coin":
        data, _ = coin_generation(num_samples=spec["num_samples"],
                                  seq_len=spec["seq_len"],
                                  p=spec["p"], q=spec["q"])
        return CoinDataset(data, seq_len=spec["seq_len"])
    data, _ = flower_process_generation(num_samples=spec["num_samples"],
                                        seq_len=spec["seq_len"],
                                        n=spec["n"], m=spec["m"],
                                        dice_probs=spec["dice_probs"])
    return FlowerDataset(data, seq_len=len(data[0]))


def one_repeat(spec: dict, cfg: dict, seed: int, khat: bool = False,
               keep_models: bool = False) -> dict:
    """
    Train both arms once on one realisation of the process and measure them.

    Returns {"seed", "fw": {...}, "bw": {...}, "delta_ce"}, where each arm dict
    carries final_ce / final_ppl / S_emp / traj / val_traj / val_at / diverged
    (+ k_hat / S_hat when `khat`).

    `keep_models` additionally returns the two live models under "_models" and
    the analysis chunk under "_chunk", for transition_matrix_extraction.  It is
    off by default because holding 30 repeats x 2 arms of trained models is
    exactly what the `del rec` below exists to prevent; only the single best
    repeat is ever re-trained with it on.
    """
    ds      = build_dataset(spec, seed)
    max_len = full_seq_len(ds)                       # PE table covers the input
    chunk   = min(cfg["train_chunk_len"], max_len)
    loader  = make_chunked_loader(ds, chunk, spec["batch"], seed=seed)
    # Windows at the TRAINING chunk length, from a different seed -- evaluating
    # at full length would ask the model to extrapolate to positional-encoding
    # indices it never saw, and would do so asymmetrically between the arms.
    loader_ana = make_analysis_loader(ds, chunk, cfg["ana_batch"], seed=seed + 1000)

    out = {"seed": seed}
    for arm, mode, use_t, k in (("fw", "forward",  "last",  spec["k_fw"]),
                                ("bw", "backward", "first", spec["k_bw"])):
        # K, state_dim and beta come from the spec, where discrete_hparams
        # resolved them once from V and the geometry.  All three are the SAME
        # for both arms; only embed_type="discrete" reads them.
        n_states  = spec.get("n_states")
        state_dim = spec.get("state_dim")
        # --usage-beta overrides the computed 1/N.  That is the beta sweep's
        # only entry point; every other run leaves it None and gets 1/N.
        beta_arm  = (cfg["usage_beta_fixed"]
                     if cfg.get("usage_beta_fixed") is not None
                     else spec.get("usage_beta"))
        # Rebuilt per arm rather than shared: the split is a pure function of
        # (N, seed) so both arms get the same one, while the training loader's
        # shuffle generator is freshly seeded instead of carrying the state the
        # first arm left it in.
        train_loader, test_loader = test_train_validation(
            loader, test_ratio=0.20, train_ratio=0.80, seed=seed)

        # Identical initialisation for the two arms of this repeat.
        torch.manual_seed(seed * 1000)
        rec = train_model(
            train_loader,
            num_token=spec["num_token"], d_model=cfg["d_model"],
            max_len=max_len, max_epochs=spec["max_epochs"], lr=cfg["lr"],
            mode=mode, embed_type=cfg["embed_type"],
            val_loader=test_loader, n_layers=cfg["n_layers"],
            accelerator=cfg["accelerator"],
            val_every_n_steps=cfg["val_every_n_steps"],
            weight_decay=cfg.get("weight_decay", 0.0),
            n_states=n_states,
            state_dim=state_dim,
            tau=cfg.get("tau", 1.0),
            usage_beta=beta_arm,
        )
        to_cpu_for_analysis(rec.model)

        ce, ppl = _eval_loss_on_loader(rec.model, test_loader)
        dv      = diagnose_divergence(rec.step_loss)
        arm_res = dict(
            final_ce=ce, final_ppl=ppl,
            traj=[float(v) for v in rec.step_loss],
            traj_at=[int(v) for v in rec.step_at],
            val_traj=[float(v) for v in rec.step_val_loss],
            val_at=[int(v) for v in rec.step_val_at],
            diverged=bool(dv["diverged"]), divergence=dv,
        )

        try:
            arm_res["S_emp"] = statistical_complexity_empirical(
                rec.model, loader_ana, max_batches=cfg["max_batches"],
                use_t=use_t, k=k)
        except Exception as e:
            print(f"  complexity failed ({arm}, seed {seed}): {e}")
            arm_res["S_emp"] = float("nan")

        if khat:
            # The better estimator: cluster the model's PREDICTIVE distribution
            # with a distance threshold, so k falls out of the data instead of
            # being assumed.  Off by default because it is a clustering pass per
            # arm per repeat.
            try:
                rcs = recover_causal_states(
                    rec.model, loader_ana, use_t=use_t,
                    max_batches=cfg["max_batches"],
                    state_tol=cfg["state_tol"], n_pts=cfg["umap_n_pts"],
                    seed=seed)
                arm_res.update(k_hat=rcs["k_hat"], S_hat=rcs["S_hat"],
                               k_plateau=rcs["plateau"])
            except Exception as e:
                print(f"  k-hat failed ({arm}, seed {seed}): {e}")
                arm_res.update(k_hat=None, S_hat=float("nan"), k_plateau=None)

        # Explicit causal states, when the architecture has them.  Returns None
        # for OneHotDecoder, so this is unconditional rather than an
        # embed_type branch -- MODULAR_MODELS_PLAN.md 3.
        csr = causal_state_report(rec.model, loader_ana, min_pos=cfg.get("state_min_pos", 5))
        if csr is not None:
            # Stored as plain lists rather than the live report: they are only
            # K x V and K x S, they make the figure reproducible from the pickle
            # alone (so --plots-only works), and they keep no tensors alive.
            arm_res["S_emp_states"]  = csr["S_emp"]
            arm_res["n_states"]      = csr["n_states"]
            arm_res["n_states_used"] = len(csr["occupied"])
            arm_res["state_counts"]  = csr["counts"].tolist()
            arm_res["emission_table"] = csr["emissions"].tolist()
            arm_res["state_vectors"]  = csr["vectors"].tolist()
            arm_res["usage_beta"]     = csr["usage_beta"]
            # H(state | current token), in bits.  0 exactly when the assignment
            # is a function of the token -- which the TRUE causal state is, in
            # both arms of both processes -- so any positive value means the
            # bottleneck is keying on context it does not need.  The only
            # diagnostic here that needs no ground truth.
            arm_res["h_state_given_token"] = csr["h_state_given_token"]
            # True count for this arm, so K vs k_discovered can be plotted from
            # the pickle alone.  spec carries it; the record must too, because
            # --plots-only reads records, not specs.
            arm_res["true_k"] = spec.get(f"true_k_{arm}")

        out[arm] = arm_res
        if keep_models:
            out.setdefault("_models", {})[arm] = rec.model
            out["_chunk"] = chunk
        del rec
        cleanup()

    out["delta_ce"] = out["bw"]["final_ce"] - out["fw"]["final_ce"]
    return out


@contextlib.contextmanager
def quiet(enabled: bool = True):
    """
    Swallow the per-run training chatter, and re-emit it if the block raises.

    train_test_val_pipeline is not used here, but test_train_validation and
    Lightning still print several lines per run; at 100 repeats x 7 processes
    x 2 arms that is the whole log.  Nothing is lost on failure.
    """
    if not enabled:
        yield None
        return
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield buf
    except Exception:
        print(buf.getvalue())
        raise


# ══════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════
def _stats(vals) -> dict:
    """mean / sd / sem / n over the finite entries of `vals`."""
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return dict(mean=float("nan"), sd=float("nan"), sem=float("nan"), n=0)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    return dict(mean=float(a.mean()), sd=sd,
                sem=(sd / np.sqrt(n) if n > 1 else float("nan")), n=n)


def _traj_matrix(trajs) -> np.ndarray:
    """
    (n_runs, L) matrix of trajectories truncated to the shortest one.

    Every run here has the same number of gradient steps, but a truncation
    guard costs nothing and keeps the mean well defined if one run is short.
    """
    kept = [t for t in trajs if t]
    if not kept:
        return np.zeros((0, 0))
    L = min(len(t) for t in kept)
    return np.asarray([t[:L] for t in kept], dtype=float)


def paired_stats(rec: dict) -> dict:
    """
    Paired delta-CE over repeats, reusing the runner's estimator.

    paired_delta_ce takes the two arms' per-unit held-out CE and returns the
    mean paired difference, its sem, a paired t and the same statistic over
    converged units only.  Its unit is a CV fold in run_experiments.py and a
    REPEAT here; the arithmetic is identical, and the repeat version is the
    stronger one because repeats do not share a training set.
    """
    fw = [r["fw"]["final_ce"] for r in rec["runs"]]
    bw = [r["bw"]["final_ce"] for r in rec["runs"]]
    seed = rec["runs"][0]["seed"] if rec["runs"] else None
    return paired_delta_ce(
        {"fold_test_loss": fw, "seed": seed},
        {"fold_test_loss": bw, "seed": seed},
        label=f"{rec['spec']['tag']}  (one line per REPEAT, not per fold)",
        theory=rec["spec"]["theory"], conv_tol=CONV_TOL,
    )


def converged_mask(rec: dict, paired: dict) -> np.ndarray:
    """Boolean mask over repeats: both arms within CONV_TOL of H_inf."""
    n = len(rec["runs"])
    if paired and paired.get("converged") is not None:
        return np.asarray(paired["converged"], dtype=bool)
    return np.ones(n, dtype=bool)


def converged_view(rec: dict, paired: dict):
    """
    `rec` restricted to repeats where BOTH arms converged, plus a label saying
    how many survived.

    Every per-process figure is drawn from this rather than from all repeats.  A
    diverged repeat is an optimisation failure, not a measurement: its |delta|
    is large and arbitrarily signed, and one of them can dominate a mean.  Mixing
    the two is how a figure ends up disagreeing with the statistic printed beside
    it -- `paired["mean"]` has always been the converged-only number while the
    plots showed all repeats.

    The count is returned so it can go ON the figure.  It is not a footnote: at
    the discrete architecture's budget only 8 of 30 repeats converged on some
    coin processes, and a reader who cannot see that will over-read the bar.

    Falls back to all repeats when NOTHING converged, and says so in the label,
    because an empty figure hides the problem instead of showing it.
    """
    conv = converged_mask(rec, paired)
    runs = [r for r, ok in zip(rec["runs"], conv) if ok]
    n_c, n_t = int(conv.sum()), int(conv.size)
    if not runs:
        return rec, paired, (f"NO repeat converged (0/{n_t}); showing ALL repeats "
                             f"— these are optimisation failures, not measurements")
    rec_c = dict(rec)
    rec_c["runs"] = runs
    p_c = dict(paired or {})
    p_c["converged"] = [True] * len(runs)
    return rec_c, p_c, (f"converged repeats only: {n_c} of {n_t}"
                        f"   (|CE−H∞| ≤ {CONV_TOL} in BOTH arms)")


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════
def _errbars(ax, xs, means, sds, sems):
    """
    Thin whisker = +/- 1 sd (spread over repeats), thick bar = +/- 1 sem
    (precision of the mean).  Drawing both removes the usual ambiguity about
    which one an error bar is.
    """
    nz = lambda v: np.nan_to_num(np.asarray(v, dtype=float), nan=0.0)
    ax.errorbar(xs, means, yerr=nz(sds), fmt="none", ecolor="0.30",
                elinewidth=1.0, capsize=6, zorder=5)
    ax.errorbar(xs, means, yerr=nz(sems), fmt="none", ecolor="k",
                elinewidth=3.0, capsize=0, zorder=6)


# Each panel is drawn by a _draw_* helper that takes an Axes, so the same code
# serves both the one-figure-per-process outputs and the grid summary.  Anything
# duplicated between the two would drift apart the first time one was fixed.
# `compact` shrinks the annotation for a grid cell; `legend` lets the grid draw
# the key in one column only.

def _draw_complexity(ax, rec: dict, compact: bool = False, legend: bool = True):
    """
    S_emp for both arms with error bars over the repeats, against C+ / C-.

    The theoretical bars are the closed forms -- statistical_complexity for the
    coin, flower_complexity for the flower -- so a systematic gap is visible
    rather than inferred.

    The estimator is resolved, not assumed: a discrete run is plotted from its
    state occupancy and a continuous one from k-means, and the caption says
    which.  Plotting k-means for a discrete run reports the wrong instrument --
    it clusters the pre-bottleneck latent at a k taken from the ground truth,
    so at n_states=1 it returns ~0.7 bits where the true complexity is 0.
    """
    spec = rec["spec"]
    key  = resolve_s_emp(rec["runs"])
    s_fw = _stats([r["fw"][key] for r in rec["runs"]])
    s_bw = _stats([r["bw"][key] for r in rec["runs"]])
    emp  = [s_fw["mean"], s_bw["mean"]]
    th   = [spec["C_plus"], spec["C_minus"]]
    x    = np.arange(2)
    fs, fsl = (6, 6) if compact else (9, 9)

    b1 = ax.bar(x - 0.2, emp, 0.35, color=[FW_COLOUR, BW_COLOUR],
                alpha=0.85, edgecolor="k",
                label=f"Empirical {key} (n={s_fw['n']} runs)")
    b2 = ax.bar(x + 0.2, th, 0.35, color=[FW_COLOUR, BW_COLOUR],
                alpha=0.45, edgecolor="k", hatch="//", label="Theoretical (closed form)")
    _errbars(ax, x - 0.2, emp, [s_fw["sd"], s_bw["sd"]], [s_fw["sem"], s_bw["sem"]])
    ax.bar_label(b1, fmt="%.4f", padding=(8 if compact else 14), fontsize=fs)
    ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=fs)

    # The tick tells the reader what the bar was measured with.  For k-means
    # that is the assumed k; for the state estimator k is not used at all, so
    # quoting it there would be the same confusion in the axis instead of the
    # bar -- the budget K and the states actually occupied are what matter.
    if key == "S_emp_states":
        def _lab(arm):
            u = [r[arm].get("n_states_used") for r in rec["runs"]
                 if r[arm].get("n_states_used")]
            K = spec.get("n_states", "?")
            return f"K={K}, used {int(np.median(u))}" if u else f"K={K}"
    else:
        def _lab(arm):
            return f"k={spec['k_fw' if arm == 'fw' else 'k_bw']}"

    ax.set_xticks(x)
    if compact:
        ax.set_xticklabels([f"FW  {_lab('fw')}", f"BW  {_lab('bw')}"], fontsize=7)
    else:
        ax.set_xticklabels([f"Forward  (C+, {_lab('fw')})",
                            f"Backward  (C-, {_lab('bw')})"])
    ax.set_ylabel("Statistical complexity (bits)")
    ax.set_ylim(0, max(emp + th) * 1.20)          # headroom for the bar labels
    if not compact:
        ax.set_title(
            f"{spec['tag']} — empirical vs theoretical statistical complexity\n"
            f"mean over {s_fw['n']} runs;  thick bar = ±1 sem, thin = ±1 sd  "
            f"(sem: {s_fw['sem']:.4f} / {s_bw['sem']:.4f})",
            fontsize=11, fontweight="bold")
    if legend:
        ax.legend(fontsize=fsl, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    if compact:
        return
    note = f"{key}: {S_EMP_NOTE[key]}."
    if key == "S_emp":
        note += ("  It overestimates when the true state occupancy is "
                 "unbalanced.")
    k_fw = [r["fw"].get("k_hat") for r in rec["runs"] if r["fw"].get("k_hat")]
    k_bw = [r["bw"].get("k_hat") for r in rec["runs"] if r["bw"].get("k_hat")]
    if k_fw and k_bw:
        h_fw = _stats([r["fw"]["S_hat"] for r in rec["runs"]])
        h_bw = _stats([r["bw"]["S_hat"] for r in rec["runs"]])
        note += (f"\nRecovered from the predictive distribution: "
                 f"k̂ median {int(np.median(k_fw))} / {int(np.median(k_bw))}, "
                 f"S_hat {h_fw['mean']:.4f} / {h_bw['mean']:.4f}")
    ax.text(0.5, -0.16, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="0.25")


def plot_complexity(rec: dict, out_dir: str, note: str = ""):
    """One complexity figure for one process."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _draw_complexity(ax, rec)
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=9, color="#3C4653")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{rec['spec']['tag']}_complexity.png"))


def _draw_final_loss(ax, rec: dict, paired: dict, compact: bool = False,
                     legend: bool = True, right_label: bool = True):
    """
    Three bars — forward, backward, backward - forward — with their error.

    The third bar is on its own axis on the right.  delta_CE is two to three
    orders of magnitude smaller than the CE values beside it, so on a shared
    axis it is a flat line at zero and the figure says nothing.  Its error is
    the PAIRED sd/sem, i.e. the spread of the per-repeat differences, which is
    the whole point of pairing the arms and is much smaller than what
    propagating the two independent errors would give.
    """
    spec = rec["spec"]
    fw = np.asarray([r["fw"]["final_ce"] for r in rec["runs"]], dtype=float)
    bw = np.asarray([r["bw"]["final_ce"] for r in rec["runs"]], dtype=float)
    d  = bw - fw
    conv = converged_mask(rec, paired)
    fs, fsl = (6, 6) if compact else (9, 8)

    groups = [("repeats shown", np.ones(len(fw), dtype=bool), 0.85, None)]
    if conv.sum() and conv.sum() < len(fw):
        groups.append((f"converged only (|CE−H∞| ≤ {CONV_TOL} both arms)",
                       conv, 0.45, "//"))

    ax2 = ax.twinx()
    w = 0.34 if len(groups) > 1 else 0.5

    for gi, (glabel, mask, alpha, hatch) in enumerate(groups):
        off = (gi - (len(groups) - 1) / 2) * w
        s_fw, s_bw, s_d = _stats(fw[mask]), _stats(bw[mask]), _stats(d[mask])
        b = ax.bar([0 + off, 1 + off], [s_fw["mean"], s_bw["mean"]], w,
                   color=[FW_COLOUR, BW_COLOUR], alpha=alpha, edgecolor="k",
                   hatch=hatch, label=f"{glabel}  (n={s_fw['n']})")
        ax.bar_label(b, padding=(6 if compact else 12), fontsize=fs,
                     labels=([f"{s['mean']:.4f}" for s in (s_fw, s_bw)] if compact
                             else [f"{s['mean']:.4f}\n±{s['sd']:.4f} sd"
                                   for s in (s_fw, s_bw)]))
        _errbars(ax, [0 + off, 1 + off], [s_fw["mean"], s_bw["mean"]],
                 [s_fw["sd"], s_bw["sd"]], [s_fw["sem"], s_bw["sem"]])

        bd = ax2.bar([2 + off], [s_d["mean"]], w, color=D_COLOUR, alpha=alpha,
                     edgecolor="k", hatch=hatch)
        ax2.bar_label(bd, fmt="%+.4f", padding=(6 if compact else 14),
                      fontsize=fs, color=D_COLOUR)
        _errbars(ax2, [2 + off], [s_d["mean"]], [s_d["sd"]], [s_d["sem"]])

    ax.axhline(spec["theory"], color="crimson", ls="--", lw=1.5,
               label=f"H∞ = {spec['theory']:.4f}")
    # Zero line only under the difference bar: drawn across the whole axes it
    # cuts through the two CE bars at a height that means nothing for them.
    ax2.hlines(0.0, 1.55, 2.55, color="k", ls=":", lw=1.0)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Forward", "Backward", "BW − FW\n(right axis)"]
                       if compact else
                       ["Forward", "Backward", "BW − FW  (right axis)"],
                       fontsize=(7 if compact else None))
    ax.set_ylabel("Final held-out CE (bits/token)")
    ax.set_ylim(0, max(np.nanmax(fw), np.nanmax(bw)) * 1.30)   # label headroom
    if right_label:
        ax2.set_ylabel("delta_CE = CE_BW − CE_FW (bits)", color=D_COLOUR)
    ax2.tick_params(axis="y", colors=D_COLOUR,
                    labelsize=(6 if compact else None))
    # Keep zero visible and the difference bar centred on it.
    lim = max(abs(np.nanmin(d)), abs(np.nanmax(d)), 1e-6) * 1.6
    ax2.set_ylim(-lim, lim)

    all_s = _stats(d)
    if not compact:
        ax.set_title(
            f"{spec['tag']} — final held-out loss and the paired difference\n"
            f"delta_CE = {all_s['mean']:+.4f} ± {all_s['sem']:.4f} (sem over "
            f"{all_s['n']} repeats);  C− − C+ = {spec['C_minus']-spec['C_plus']:+.4f}\n"
            f"thick bar = ±1 sem, thin = ±1 sd;  note the two y-scales",
            fontsize=11, fontweight="bold")
    if legend:
        ax.legend(fontsize=fsl, loc="lower left")
    ax.grid(True, alpha=0.3, axis="y")


def plot_final_loss(rec: dict, paired: dict, out_dir: str, note: str = ""):
    """One final-loss figure for one process."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _draw_final_loss(ax, rec, paired)
    fig.tight_layout()
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=9, color="#3C4653")
    savefig(fig, os.path.join(out_dir, f"{rec['spec']['tag']}_final_loss.png"))


def plot_trajectories(rec: dict, out_dir: str, note: str = ""):
    """
    Every run's training-loss trajectory at alpha 0.3, both mean curves in bold,
    with a shaded ±1 sd interval — and, beside it, the same treatment of the
    paired difference.

    The second panel is the trajectory form of the headline number: at each step
    it is (CE_BW − CE_FW) within the SAME repeat, so it inherits the pairing
    instead of differencing two independent means.  Every run's difference is
    drawn, with the mean and its bands over the top.
    """
    spec = rec["spec"]
    if _traj_pair(rec) is None:
        print(f"  no trajectories to plot for {spec['tag']}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    _draw_loss_trajectories(axes[0], rec)
    _draw_paired_diff(axes[1], rec)
    fig.tight_layout()
    if note:
        fig.text(0.5, -0.02, note, ha="center", fontsize=9, color="#3C4653")
    savefig(fig, os.path.join(out_dir, f"{spec['tag']}_trajectories.png"))


def _traj_pair(rec: dict):
    """
    (M_fw, M_bw, L, steps) with both arms truncated to a common length, or None
    when there is nothing recorded.  Shared by the two trajectory panels so they
    cannot end up drawn over different step ranges.
    """
    M_fw = _traj_matrix([r["fw"]["traj"] for r in rec["runs"]])
    M_bw = _traj_matrix([r["bw"]["traj"] for r in rec["runs"]])
    if M_fw.size == 0 or M_bw.size == 0:
        return None
    L = min(M_fw.shape[1], M_bw.shape[1])
    return M_fw[:, :L], M_bw[:, :L], L, np.arange(L)


def _draw_loss_trajectories(ax, rec: dict, compact: bool = False,
                            legend: bool = True):
    """Every run's training loss, both arms, with their means and ±1 sd bands."""
    spec = rec["spec"]
    M_fw, M_bw, L, steps = _traj_pair(rec)
    fsl = 6 if compact else 8

    for M, colour, label in ((M_fw, FW_COLOUR, "Forward"),
                             (M_bw, BW_COLOUR, "Backward")):
        for row in M:
            ax.plot(steps, row, color=colour, alpha=0.3, lw=0.6, zorder=1)
        mu = M.mean(axis=0)
        sd = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(L)
        ax.fill_between(steps, mu - sd, mu + sd, color=colour, alpha=0.28,
                        zorder=2, linewidth=0)
        ax.plot(steps, mu, color=colour, lw=2.6, zorder=3,
                label=f"{label} mean of {M.shape[0]}  (band ±1 sd)")
    ax.axhline(spec["theory"], color="crimson", ls="--", lw=1.6,
               label=f"H∞ = {spec['theory']:.4f}")

    # A diverged run reaches CE 15-58 bits, which would squash the region the
    # figure is about.  Clip the view rather than the data, and say so.
    allv = np.concatenate([M_fw.ravel(), M_bw.ravel()])
    top  = max(float(np.nanpercentile(allv, 99.5)), spec["theory"] * 1.6)
    n_above = int((allv > top).sum())
    ax.set_ylim(0.0, top * 1.05)
    n_div = sum(r["fw"]["diverged"] or r["bw"]["diverged"] for r in rec["runs"])
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("Training CE loss (bits)")
    if not compact:
        ax.set_title(
            f"{spec['tag']} — {M_fw.shape[0]} runs per direction\n"
            f"{n_div} run(s) diverged"
            + (f";  y clipped at {top:.2f}, {n_above} point(s) above" if n_above else ""),
            fontsize=11, fontweight="bold")
    if legend:
        ax.legend(fontsize=fsl)
    ax.grid(True, alpha=0.3)


def _draw_paired_diff(ax, rec: dict, compact: bool = False,
                      legend: bool = True):
    """The per-step paired difference: every run, plus the mean and its bands."""
    spec = rec["spec"]
    M_fw, M_bw, L, steps = _traj_pair(rec)
    fsl = 6 if compact else 8
    D  = M_bw - M_fw                      # paired, step by step
    mu = D.mean(axis=0)
    sd = D.std(axis=0, ddof=1) if D.shape[0] > 1 else np.zeros(L)
    sem = sd / np.sqrt(D.shape[0]) if D.shape[0] > 1 else np.zeros(L)
    # Every run's own difference, same convention as the loss panel.  Drawn under
    # the aggregate, so the bands and the mean stay readable on top.  A grid cell
    # is roughly a quarter of the standalone panel's area, where alpha 0.03 is
    # invisible rather than faint, so the compact form is a little stronger.
    for row in D:
        ax.plot(steps, row, color=D_COLOUR, alpha=(0.01 if compact else 0.03),
                lw=0.6, zorder=1)
    ax.fill_between(steps, mu - sd, mu + sd, color=D_COLOUR, alpha=0.18,
                    linewidth=0, zorder=2, label="±1 sd over runs")
    ax.fill_between(steps, mu - sem, mu + sem, color=D_COLOUR, alpha=0.40,
                    linewidth=0, zorder=2, label="±1 sem")
    ax.plot(steps, mu, color=D_COLOUR, lw=2.2, zorder=3,
            label="mean (CE_BW − CE_FW)")
    ax.axhline(0.0, color="k", ls="--", lw=0.9, zorder=4)
    # Scale to the SETTLED portion of training, from the aggregate band.
    #
    # There are two regimes here and one axis cannot serve both.  Through the
    # first few dozen steps both arms are still descending and their difference
    # swings by whole bits -- measured, 30 to 90 times the settled spread, and up
    # to 8.8 bits on the flower configs.  That swing is optimisation transient,
    # not the quantity under study.  Later the difference IS the residual
    # difference the hypothesis is about, and it lives at 0.005 to 0.03 bits.
    # Scaling to the transient flattens the settled region onto zero and the
    # panel then says nothing; scaling to the settled region costs only that the
    # transient leaves the frame, and the loss panel beside it already shows the
    # descent of both arms.
    #
    # The limit comes from the mean and its band rather than from the individual
    # traces, because the mean and band are what this panel asserts and they
    # must never be clipped.  Over the tail a max is safe -- there is no
    # transient left to guard against -- and 1.5x keeps the band clear of the
    # frame and the legend.
    # `int(...)`: L // 4.3 is a FLOAT (130 // 4.3 == 30.0), and numpy raises
    # "slice indices must be integers" on a float slice, so the divisor has to be
    # made an index explicitly.  The title reads the same value, so the stated
    # window cannot drift from the one actually used.
    tail_from = int(L / 4.3)
    tail = slice(tail_from, L)
    span = max(float(np.nanmax(np.abs(np.concatenate([(mu - sd)[tail],
                                                      (mu + sd)[tail]])))), 1e-6)
    ax.set_ylim(-span * 1.5, span * 1.5)
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("CE_BW − CE_FW (bits)")
    if not compact:
        ax.set_title(f"Paired difference per step — every one of {D.shape[0]} runs\n"
                     f"y scaled to steps {tail_from}–{L}; the earlier transient "
                     f"runs off-scale;  prediction: "
                     f"{'delta_CE > 0' if spec['C_minus'] > spec['C_plus'] else 'delta_CE < 0'}"
                     f"  (C− − C+ = {spec['C_minus']-spec['C_plus']:+.4f})",
                     fontsize=11, fontweight="bold")
    if legend:
        ax.legend(fontsize=fsl)
    ax.grid(True, alpha=0.3)


def plot_summary(records: dict, out_root: str):
    """
    Cross-process summary: delta_CE with its sem per process, and delta_CE
    against C- - C+.

    The sign of delta_CE tracking the sign of (C- - C+) ACROSS processes is the
    strongest thing this design can show; a single process cannot show it.
    """
    items = [(tag, r) for tag, r in sorted(records.items())
             if r.get("runs") and r.get("paired")]
    if not items:
        return
    tags   = [t for t, _ in items]
    means  = np.array([r["paired"]["mean"] for _, r in items], dtype=float)
    sems   = np.array([r["paired"].get("sem", np.nan) for _, r in items], dtype=float)
    gaps   = np.array([r["spec"]["C_minus"] - r["spec"]["C_plus"] for _, r in items])
    ns     = [r["paired"]["n"] for _, r in items]

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    ax = axes[0]
    y = np.arange(len(tags))
    # Same three-way rule as the printed table: grey out anything that is not
    # distinguishable from zero, so the figure cannot suggest a verdict the
    # numbers do not support.
    def _colour(m, s, g):
        if np.isfinite(s) and abs(m) < 2 * s:
            return "0.65"
        return FW_COLOUR if (m > 0) == (g > 0) else D_COLOUR
    colours = [_colour(m, s, g) for m, s, g in zip(means, sems, gaps)]
    ax.barh(y, means, xerr=np.nan_to_num(2 * sems), color=colours, alpha=0.85,
            edgecolor="k", error_kw=dict(ecolor="k", elinewidth=1.4, capsize=4))
    ax.axvline(0.0, color="k", ls="--", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}\n(n={n}, C−−C+={g:+.3f})"
                        for t, n, g in zip(tags, ns, gaps)], fontsize=8)
    ax.set_xlabel("delta_CE = CE_BW − CE_FW (bits),  error bars = ±2 sem")
    ax.set_title("Paired delta_CE per process\n"
                 "grey = not distinguishable from zero (|delta_CE| < 2 sem)\n"
                 "blue = sign agrees with C− − C+,  red = disagrees",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[1]
    ax.errorbar(gaps, means, yerr=np.nan_to_num(sems), fmt="o", ms=7,
                color="#4c72b0", ecolor="0.3", capsize=4)
    for g, m, t in zip(gaps, means, tags):
        ax.annotate(t.replace("traj_", ""), (g, m), textcoords="offset points",
                    xytext=(6, 5), fontsize=7)
    ax.axhline(0.0, color="k", ls="--", lw=0.9)
    ax.axvline(0.0, color="k", ls="--", lw=0.9)
    ax.set_xlabel("C− − C+ (bits, closed form)")
    ax.set_ylabel("delta_CE (bits, mean over repeats)")
    ax.set_title("Does delta_CE track the theoretical asymmetry?\n"
                 "the hypothesis predicts points in the\n"
                 "lower-left and upper-right quadrants only",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    # Stated once, at the figure level.  Both panels plot the CONVERGED-only
    # paired statistic, which is what `paired["mean"]` has always been; without
    # this line a reader compares the bar against the per-repeat delta printed
    # during the run and finds them disagreeing, including in sign on a process
    # whose delta is n.s. anyway.
    fig.text(0.5, -0.015,
             f"Both panels: CONVERGED repeats only  —  n is how many of the "
             f"attempted repeats that was  (|CE−H∞| ≤ {CONV_TOL} in BOTH arms). "
             f"A diverged repeat is an optimisation failure, not a measurement.",
             ha="center", fontsize=9, color="#3C4653")
    savefig(fig, os.path.join(out_root, "summary_delta_ce.png"))


def _verdict(mean: float, sem: float, gap: float) -> str:
    """
    The three-way call used by both the table and the grid.

    A sign is only claimed when it is distinguishable from zero: delta_CE is a
    difference of residuals, so a converged predictor with spare capacity gives
    zero whatever C- - C+ is, and reading a direction off noise is exactly the
    over-interpretation this repo has been prone to.
    """
    if not np.isfinite(mean):
        return "—"
    if np.isfinite(sem) and abs(mean) < 2 * sem:
        return "n.s."
    return "match" if (mean > 0) == (gap > 0) else "MISMATCH"


def _grid_order(records: dict) -> list:
    """
    Column order for the grid: coin processes then flower ones, each by
    DESCENDING C- - C+.

    Families stay together so a column is easy to find, and within a family the
    theoretical asymmetry decreases left to right -- which is the order that makes
    the question the grid exists to answer readable along a row: does anything in
    these panels vary with C- - C+?
    """
    rank = {"coin": 0, "flower": 1}
    return sorted(((t, r) for t, r in records.items() if r.get("runs")),
                  key=lambda kv: (rank.get(kv[1]["spec"]["kind"], 9),
                                  -(kv[1]["spec"]["C_minus"]
                                    - kv[1]["spec"]["C_plus"])))


def plot_grid_summary(records: dict, out_root: str,
                      fname: str = "summary_grid.png"):
    """
    Every process's three figures in one 3 x N grid — the whole experiment on a
    single sheet.

    Rows are the three quantities, columns the processes, so each row is a
    like-for-like comparison across processes:

        row 1   statistical complexity, empirical against the closed form
        row 2   final held-out loss, and the paired difference on the right axis
        row 3   the per-step paired difference, every run plus mean and bands

    Panels are drawn by the same _draw_* helpers as the per-process figures, in
    `compact` mode, so the grid can never disagree with them.  Row 3 is the
    difference panel rather than the loss panel: the loss curves are nearly
    identical across the two arms by construction and differ between processes
    only in where H∞ sits, whereas the difference is the quantity under study.
    The absolute loss trajectories remain in <tag>_trajectories.png.

    Y-scales are per column, NOT shared.  H∞ ranges from 0.47 to 2.10 bits across
    these processes, so a shared axis would compress every panel in order to
    accommodate the largest; each column is a statement about its own process
    converging to its own H∞.  The consequence is that bar heights must not be
    compared by eye between columns -- read the printed numbers.
    """
    items = _grid_order(records)
    if not items:
        print("  nothing to put in the grid summary")
        return
    n = len(items)
    fig, axes = plt.subplots(3, n, figsize=(4.4 * n, 12.8), squeeze=False)

    for col, (tag, rec) in enumerate(items):
        spec   = rec["spec"]
        paired = rec.get("paired") or {}
        first, last = col == 0, col == n - 1

        _draw_complexity(axes[0][col], rec, compact=True, legend=first)
        _draw_final_loss(axes[1][col], rec, rec.get("paired"), compact=True,
                         legend=first, right_label=last)
        _draw_paired_diff(axes[2][col], rec, compact=True, legend=first)

        gap  = spec["C_minus"] - spec["C_plus"]
        mean = paired.get("mean", float("nan"))
        sem  = paired.get("sem", float("nan"))
        n_div = sum(r["fw"]["diverged"] or r["bw"]["diverged"]
                    for r in rec["runs"])
        axes[0][col].set_title(
            f"{tag.replace('traj_', '')}   ({len(rec['runs'])} repeats"
            + (f", {n_div} diverged" if n_div else "") + ")\n"
            f"C+ = {spec['C_plus']:.3f}   C− = {spec['C_minus']:.3f}   "
            f"C− − C+ = {gap:+.3f}\n"
            f"ΔCE = {mean:+.4f} ± {sem:.4f}   [{_verdict(mean, sem, gap)}]",
            fontsize=9, fontweight="bold")

        # Row identity comes from the leftmost column's y-label; repeating it in
        # every column is noise.
        if not first:
            for row in range(3):
                axes[row][col].set_ylabel("")

    fig.suptitle(
        "Repeat-statistics summary — every process, every quantity\n"
        "columns ordered coin then flower, each by descending C− − C+;  "
        "rows: statistical complexity  |  final held-out loss + paired ΔCE  |  "
        "per-step paired difference\n"
        "y-scales are PER COLUMN (H∞ differs by process), so compare numbers "
        "rather than bar heights across columns;  "
        "thick error bar = ±1 sem, thin = ±1 sd",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    savefig(fig, os.path.join(out_root, fname))


def _report_from_run(rec: dict, arm: str, mode: str):
    """
    Rebuild a causal-state report from the LAST repeat's stored arrays.

    Returns None when the architecture has no explicit states, which is what
    makes the plotting call site free of embed_type branches.  Reading it back
    from the pickle rather than holding the live report is what lets
    --plots-only redraw these figures without retraining.
    """
    runs = rec.get("runs") or []
    # Prefer a converged repeat: the states of a diverged run describe a model
    # that never settled.  Falls back to any repeat with data if none converged,
    # so the figure still appears rather than silently vanishing.
    conv = converged_mask(rec, rec.get("paired") or {})
    ordered = [r for r, ok in zip(runs, conv) if ok] or list(runs)
    for r in reversed(ordered):
        a = (r.get(arm) or {})
        if "emission_table" not in a:
            continue
        counts = np.asarray(a["state_counts"], dtype=float)
        return {
            "counts":    counts.astype(int),
            "occupancy": counts / counts.sum() if counts.sum() else counts,
            "vectors":   np.asarray(a["state_vectors"], dtype=float),
            "emissions": np.asarray(a["emission_table"], dtype=float),
            "occupied":  [k for k in range(len(counts)) if counts[k] > 0],
            "n_states":  int(a.get("n_states", len(counts))),
            "S_emp":     float(a.get("S_emp_states", float("nan"))),
            "mode":      mode,
            "usage_beta": float(a.get("usage_beta", 0.0)),
        }
    return None


def print_summary(records: dict, out_root: str):
    """The runner's asymmetry table, over repeats instead of folds."""
    print(f"\n{'='*104}\n  ASYMMETRY SUMMARY over repeats  ({out_root})\n{'='*104}")
    print(f"  {'process':<26} {'C+':>7} {'C-':>7} {'C- - C+':>9} "
          f"{'delta_CE':>10} {'sem':>8} {'t':>7} {'n':>4} {'div':>5} {'verdict':>9}")
    for tag, r in sorted(records.items()):
        pd_ = r.get("paired")
        if not pd_ or not r.get("runs"):
            continue
        spec = r["spec"]
        mean, sem = pd_["mean"], pd_.get("sem", float("nan"))
        cp, cm = spec["C_plus"], spec["C_minus"]
        verdict = _verdict(mean, sem, cm - cp)      # shared with the grid
        n_div = sum(x["fw"]["diverged"] or x["bw"]["diverged"] for x in r["runs"])
        print(f"  {tag:<26} {cp:>7.4f} {cm:>7.4f} {cm-cp:>+9.4f} "
              f"{mean:>+10.4f} {sem:>8.4f} {pd_['t']:>+7.2f} {pd_['n']:>4} "
              f"{n_div:>5} {verdict:>9}")
    print(f"{'='*104}")
    print("  n.s. = |delta_CE| < 2 sem.  The sem is now over REPEATS, each with")
    print("  its own dataset, so unlike the per-fold sem it is sampling")
    print("  variability.  delta_CE is still a difference of residuals, so a")
    print("  null remains ambiguous until the d_model sweep is run.")
    print("  div = repeats where either arm diverged (reached H∞, then blew up).")
    print(f"{'='*104}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Repeat-statistics harness for the causal-asymmetry test.")
    ap.add_argument("--config", default="QUICK", choices=sorted(CONFIGS),
                    help="configuration from configs.py supplying the model, "
                         "data and optimiser settings (default QUICK)")
    ap.add_argument("--repeats", type=int, default=100,
                    help="independent repeats per process (default 100)")
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed; repeat i uses base + i for BOTH arms "
                         "(default cfg['seed'])")
    ap.add_argument("--out-root", default=None,
                    help="default: All_Results/<model>/trajectories")
    ap.add_argument("--model", default=None, choices=("onehot", "discrete"),
                    help="architecture; default from the config (onehot). "
                         "'discrete' forces the prediction through a hard "
                         "one-hot over the process's theoretical number of "
                         "causal states.")
    ap.add_argument("--usage-beta", type=float, default=None,
                    help="override the computed beta = 1/(batch*chunk_len).  "
                         "Leave unset for normal runs; this is 06_beta_verify's "
                         "axis and the control arm of 03/03b.  Pass 0 to "
                         "measure S_emp with no penalty at all.")
    # ── DISCRETE_V2_PLAN.md axes ───────────────────────────────────────
    # Each overrides one resolved hyperparameter so the plan's sweeps run
    # through this runner rather than needing their own.  All default to None,
    # i.e. "use the value discrete_hparams computes", so no existing invocation
    # changes.  Every one of these moves a quantity the results depend on, so
    # each is written into run_config for provenance.
    ap.add_argument("--n-states", type=int, default=None,
                    help="override K (default 5V).  01_ksweep's axis.  K is a "
                         "BUDGET: measured, K at the exact causal-state count "
                         "fails and slack is required.")
    ap.add_argument("--state-dim", type=int, default=None,
                    help="override the state-vector dimension (default V).  "
                         "02_statedim's axis.  Carries no expressive power -- "
                         "state_matrix then emission compose to one (K,V) map.")
    ap.add_argument("--d-model", type=int, default=None,
                    help="override cfg['d_model'] (default 32).  04_capacity's "
                         "axis, and the only completely untested one.")
    ap.add_argument("--chunk-len", type=int, default=None,
                    help="override cfg['train_chunk_len'] (default 256).  "
                         "03_seqlen's axis.  NOTE this also moves beta, since "
                         "beta = 1/(batch * chunk_len); pass --usage-beta to "
                         "hold beta fixed and separate the two effects.")
    ap.add_argument("--batch", type=int, default=None,
                    help="override coin_batch and flower_batch (default 32).  "
                         "03b_batch's axis.  Also moves beta -- same caveat.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override coin/flower_max_epochs (default 150).  "
                         "05_budget's axis.  Measured, the budget is a CEILING: "
                         "state recovery peaks and then decays while CE keeps "
                         "falling, so more is not better.")
    ap.add_argument("--only", nargs="+", default=None, metavar="TAG",
                    help="run a subset; matches a full tag or any substring of "
                         "one, e.g. --only coin_p030 flower_n2_m6")
    ap.add_argument("--khat", action="store_true",
                    help="also recover k-hat / S_hat from the predictive "
                         "distribution (slower, better estimator)")
    ap.add_argument("--accelerator", default=None,
                    choices=["auto", "cpu", "mps", "gpu"],
                    help="override cfg['accelerator'].  'cpu' is the only "
                         "bit-reproducible setting -- MPS reductions are "
                         "non-deterministic (README 1.3), so exact run-to-run "
                         "comparison needs this.  Default: the config's value, "
                         "i.e. unchanged behaviour.")
    ap.add_argument("--redo", action="store_true",
                    help="retrain from scratch, discarding stored repeats "
                         "(default is to resume; REMAINING_WORK_PLAN.md C2)")
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw every figure from the saved pickle, no training")
    ap.add_argument("--verbose", action="store_true",
                    help="do not suppress the per-run training output")
    # The matrices cannot be read back off a finished folder: this runner keeps
    # no weights, so the best repeat has to be retrained.  run_transition_matrix
    # does that, and can also be pointed at an already-finished folder.
    ap.add_argument("--transition", action="store_true",
                    help="after the run, retrain the best repeat of each "
                         "process and write its state-transition matrices")
    ap.add_argument("--transition-total-run", type=int, default=5000,
                    help="generation steps per arm for --transition "
                         "(default 5000)")

    return ap.parse_args(argv)


def select_specs(specs: list, only) -> list:
    if not only:
        return specs
    keep = []
    for s in specs:
        if any(o == s["tag"] or o in s["tag"] for o in only):
            keep.append(s)
    if not keep:
        raise SystemExit(f"--only {only} matched none of: "
                         + ", ".join(s['tag'] for s in specs))
    return keep


def resumable(prev: dict | None, spec: dict, cfg: dict, args) -> tuple[bool, str]:
    """
    May the repeats stored in `prev` be extended by this invocation?

    REMAINING_WORK_PLAN.md C2.  This harness used to retrain every selected
    process from scratch, discarding completed repeats.  Resuming is a large
    saving (~20 min per process at 100 repeats) but it is only *sound* when the
    stored repeats measure the same thing the new ones will: repeats are pooled
    into one paired statistic, so silently mixing two different processes, two
    base seeds, or two configs would corrupt delta_CE rather than merely waste
    time.  A wrong resume is worse than no resume, so every identifying field is
    checked and anything unrecognised refuses.

    The seed check is the subtle one.  Repeat i uses seed = base_seed + i, so a
    resume must CONTINUE that sequence.  If the stored seeds are not exactly
    base..base+k-1 then the run was produced some other way and appending to it
    would duplicate or skip seeds -- which breaks the pairing argument that makes
    the whole comparison valid.
    """
    if not prev or not prev.get("runs"):
        return False, "nothing stored"
    if prev.get("config") != args.config:
        return False, f"config {prev.get('config')!r} != {args.config!r}"
    if prev.get("base_seed") != cfg["seed"]:
        return False, f"base_seed {prev.get('base_seed')} != {cfg['seed']}"
    if bool(prev.get("khat")) != bool(args.khat):
        return False, "--khat differs"

    old, new = prev.get("spec") or {}, spec
    for k in ("kind", "p", "q", "n", "m", "num_token", "num_samples",
              "seq_len", "max_epochs", "batch"):
        if old.get(k) != new.get(k):
            return False, f"spec differs at {k}: {old.get(k)!r} != {new.get(k)!r}"
    od, nd = old.get("dice_probs"), new.get("dice_probs")
    if (od is None) != (nd is None):
        return False, "one spec has dice_probs and the other does not"
    if od is not None and not np.array_equal(np.asarray(od), np.asarray(nd)):
        return False, "different dice_probs (a different process)"

    seeds = [r.get("seed") for r in prev["runs"]]
    want = [cfg["seed"] + i for i in range(len(seeds))]
    if seeds != want:
        return False, "stored seeds are not a prefix of the seed sequence"
    return True, ""


def load_combined(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  ! could not read {path} ({e}) — starting fresh")
        return {}


def main(argv=None):
    args = parse_args(argv)
    cfg  = dict(CONFIGS[args.config])
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.accelerator is not None:
        cfg["accelerator"] = args.accelerator
    if args.model is not None:
        cfg["embed_type"] = args.model
    if args.d_model is not None:
        cfg["d_model"] = args.d_model
    if args.chunk_len is not None:
        cfg["train_chunk_len"] = args.chunk_len
    if args.batch is not None:
        cfg["coin_batch"] = cfg["flower_batch"] = args.batch
    if args.epochs is not None:
        cfg["coin_max_epochs"] = cfg["flower_max_epochs"] = args.epochs
    if args.n_states is not None:
        cfg["n_states"] = args.n_states
    if args.state_dim is not None:
        cfg["state_dim"] = args.state_dim
    if args.usage_beta is not None:
        # A FIXED beta for every process, overriding the per-process rule.
        # This is how the two fixed-beta arms of the three-way sweep are run.
        cfg["usage_beta"] = args.usage_beta
        cfg["usage_beta_fixed"] = args.usage_beta
    # REORGANISATION_FIX_PLAN.md 4.2.  repo_path so a relative --out-root is
    # resolved against the repo root, not the cwd.  mkdir below is
    # exist_ok=True and load_combined treats a missing pickle as 'nothing done
    # yet', so a stale path silently re-runs the whole experiment instead of
    # failing.
    # Resolved AFTER cfg["embed_type"] is set from --model, so the default tree
    # follows the architecture.  Printed below before any training starts.
    out_root = repo_path(args.out_root or _default_out_root(cfg["embed_type"]))
    mkdir(out_root)

    specs = select_specs(process_specs(cfg), args.only)
    combined_path = os.path.join(out_root, "all_trajectories.pkl")
    combined = load_combined(combined_path)

    # Provenance next to the outputs, named per config so two configs sharing an
    # out_root cannot overwrite each other's record.
    if not args.plots_only:
        # REMAINING_WORK_PLAN.md C3: merge, do not overwrite.  A run with --only
        # must not shrink the record to the one process it touched.
        save_run_config(
            os.path.join(out_root, f"run_config_{args.config}.json"),
            {"config": args.config, "repeats": args.repeats,
             "base_seed": cfg["seed"], "khat": args.khat,
             "processes": [s["tag"] for s in specs],
             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             **{k: v for k, v in cfg.items()}},
            union_lists=("processes",))

    print(f"\n{'='*70}")
    print(f"  config={args.config}  repeats={args.repeats}  "
          f"base_seed={cfg['seed']}  out_root={out_root}")
    print(f"  d_model={cfg['d_model']}  n_layers={cfg['n_layers']}  "
          f"lr={cfg['lr']}  chunk={cfg['train_chunk_len']}")
    print(f"  processes: {', '.join(s['tag'] for s in specs)}")
    if args.plots_only:
        print("  --plots-only: no training, redrawing from the saved pickle")
    print(f"{'='*70}")

    t_start = time.time()
    for spec in specs:
        tag  = spec["tag"]
        odir = os.path.join(out_root, tag)      # created once there is output
        print(f"\n{'='*70}\n  {tag}   [{spec['kind']}]\n{'='*70}")
        if spec["kind"] == "flower":
            print(f"  dice_probs (seed {cfg['flower_dice_seed']}):\n{spec['dice_probs']}")
            print(f"  seq_len={spec['seq_len']} cycles = "
                  f"{2*spec['seq_len']} tokens/sequence")
        print(f"  H∞ = {spec['theory']:.4f} bits   "
              f"C+ = {spec['C_plus']:.4f}   C- = {spec['C_minus']:.4f}   "
              f"C- - C+ = {spec['C_minus']-spec['C_plus']:+.4f}")
        print(f"  prediction: "
              f"{'delta_CE > 0' if spec['C_minus'] > spec['C_plus'] else 'delta_CE < 0'}")

        if args.plots_only:
            rec = combined.get(tag)
            if not rec or not rec.get("runs"):
                print("  nothing saved for this process — skipping")
                continue
            mkdir(odir)
        else:
            mkdir(odir)
            # ── resume (REMAINING_WORK_PLAN.md C2) ─────────────────────────
            prev = combined.get(tag)
            ok, why = (False, "--redo") if args.redo else resumable(prev, spec, cfg, args)
            if ok:
                rec = dict(prev)
                rec["runs"] = list(prev["runs"])
                rec["repeats"] = args.repeats
                start = len(rec["runs"])
            else:
                rec = dict(spec=spec, config=args.config, base_seed=cfg["seed"],
                           repeats=args.repeats, khat=args.khat, runs=[])
                start = 0
                if prev and prev.get("runs"):
                    print(f"  ! NOT resuming ({why}) — retraining all "
                          f"{args.repeats} repeats from scratch")
            if start >= args.repeats:
                print(f"  already has {start} >= {args.repeats} repeats — "
                      f"skipping training (--redo to force)")
            elif start:
                print(f"  resuming: {start} repeat(s) kept, training "
                      f"{args.repeats - start} more (seeds "
                      f"{cfg['seed']+start}..{cfg['seed']+args.repeats-1})")

            t_proc = time.time()
            for i in range(start, args.repeats):
                seed = cfg["seed"] + i
                t0 = time.time()
                with quiet(not args.verbose):
                    run = one_repeat(spec, cfg, seed, khat=args.khat)
                rec["runs"].append(run)

                flag = ""
                if run["fw"]["diverged"] or run["bw"]["diverged"]:
                    flag = ("   ! DIVERGED "
                            f"({'fw' if run['fw']['diverged'] else ''}"
                            f"{'bw' if run['bw']['diverged'] else ''})")
                done = i + 1
                # Rate over what THIS session trained, not over `done`, which
                # includes resumed repeats that cost no time here.
                trained = i - start + 1
                eta = (time.time() - t_proc) / trained * (args.repeats - done) / 60
                extra = ""
                if args.khat:
                    extra = (f"  k̂ {run['fw'].get('k_hat')}/"
                             f"{run['bw'].get('k_hat')}")
                # Same resolution as the figure, on this repeat alone, so the
                # console and the plot cannot quote different instruments.
                _k = resolve_s_emp([run])
                print(f"  [{tag} {done:>3}/{args.repeats}] seed={seed}  "
                      f"CE_FW={run['fw']['final_ce']:.4f}  "
                      f"CE_BW={run['bw']['final_ce']:.4f}  "
                      f"delta={run['delta_ce']:+.4f}  "
                      f"{_k} {run['fw'][_k]:.3f}/{run['bw'][_k]:.3f}"
                      f"{extra}  ({time.time()-t0:.1f}s, ETA {eta:.1f} min)"
                      f"{flag}")

                # Written every repeat: this loop is hours long and a crash
                # should cost one repeat, not the whole process.  Quiet, because
                # paired_delta_ce prints one line per repeat and save_pkl prints
                # a size -- both are wanted once at the end, not n times.
                with contextlib.redirect_stdout(io.StringIO()):
                    rec["paired"] = (paired_stats(rec)
                                     if len(rec["runs"]) > 1 else None)
                    save_pkl(rec, os.path.join(odir, "results.pkl"))
                    combined[tag] = rec
                    save_pkl(combined, combined_path)
            print(f"\n  {tag} done in {(time.time()-t_proc)/60:.1f} min")

        rec["paired"] = paired_stats(rec) if len(rec["runs"]) > 1 else None
        combined[tag] = rec
        save_pkl(rec, os.path.join(odir, "results.pkl"))

        # Every per-process figure is drawn from the CONVERGED repeats only,
        # and says how many that was.  A diverged repeat is an optimisation
        # failure, not a measurement -- and `paired["mean"]` was already the
        # converged-only number, so drawing all repeats made the figures
        # disagree with the statistic beside them.
        rec_c, paired_c, conv_note = converged_view(rec, rec["paired"])
        plot_complexity(rec_c, odir, note=conv_note)
        plot_final_loss(rec_c, paired_c, odir, note=conv_note)
        plot_trajectories(rec_c, odir, note=conv_note)
        print(f"  {conv_note}")

        # The causal-state figure, one per arm.  Drawn from the LAST repeat's
        # report; draws nothing when the architecture has no explicit states.
        reports, theories = {}, {}
        for arm, mode in (("fw", "forward"), ("bw", "backward")):
            csr = _report_from_run(rec, arm, mode)
            if csr is None:
                continue
            th = None
            if spec["kind"] == "coin":
                th = coin_true_conditional(spec["p"], spec["q"])[0 if arm == "fw" else 1]
            reports[arm], theories[arm] = csr, th
            plot_causal_states(csr, odir, tag, theory=th)          # per-arm detail
            print_state_summary(csr, theory=th, label="  ")
        # The paired view -- occupancy, state vectors and transition
        # probabilities with the two arms side by side, as the notebook draws
        # them.  The comparison is the point: the coin needs 2 forward states
        # and 3 backward ones, which is why C- > C+.
        plot_causal_states_pair(reports.get("fw"), reports.get("bw"), odir, tag,
                                theory_fw=theories.get("fw"),
                                theory_bw=theories.get("bw"))
        cleanup()

    save_pkl(combined, combined_path)
    plot_summary(combined, out_root)
    plot_grid_summary(combined, out_root)
    print_summary(combined, out_root)
    if args.transition:
        from run_transition_matrix import one_folder
        one_folder(os.path.basename(out_root.rstrip("/")),
                   os.path.dirname(out_root.rstrip("/")),
                   args.transition_total_run)

    print(f"\n{'='*70}\n  ALL COMPLETE — {(time.time()-t_start)/60:.1f} min")
    for root, _dirs, files in os.walk(out_root):
        lvl = root.replace(out_root, "").count(os.sep)
        print(f"{'  '*lvl}{os.path.basename(root)}/")
        for f in sorted(files):
            print(f"{'  '*(lvl+1)}{f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
