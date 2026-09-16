"""
The master config, and the per-process specs derived from it.

ONE config.  The exploratory tree carried six (SMOKE / QUICK / LARGE /
DISCRETE / ...) plus a dozen command-line overrides, because it was searching
parameter axes.  This pipeline is not: it takes a process and returns the
extracted causal structure at fixed, stated settings.

Nothing here is a magic literal that could drift out of step with something
else.  K, state_dim and beta are RESOLVED from the vocabulary and the training
geometry (`resolve_hparams`); every theoretical quantity is computed by
`processes.py` from the same parameters the sampler is given.
"""
import numpy as np

from processes import (causal_state_count, causal_state_occupancy, coin_complexity,
                       coin_rev_transition_matrix, coin_tag, coin_transition_matrix,
                       entropy_rate_coin, flower_complexity, flower_entropy_rate,
                       flower_rev_transition_matrix, flower_tag,
                       flower_transition_matrix, make_dice)

CONFIG = dict(
    # ── reproducibility ────────────────────────────────────────────────
    seed              = 0,
    flower_dice_seed  = 42,

    # ── sequence geometry ──────────────────────────────────────────────
    # seq_len is in TOKENS for BOTH processes, and every generated token is
    # trained on.
    seq_len           = 300,
    burn_in           = 250,    # tokens drawn and discarded before the window
    num_samples       = 500,    # sequences per repeat
    batch             = 32,
    test_ratio        = 0.20,

    # ── model ──────────────────────────────────────────────────────────
    d_model           = 32,
    n_layers          = 2,

    # ── discrete bottleneck ────────────────────────────────────────────
    n_states_mult     = 5,      # K = 5V.  A state BUDGET, not an estimate.
    state_dim_mult    = 1,      # S = V.

    tau               = 1,      #"geom:0.5:5",
    usage_beta        = None,   # None -> 1/(batch*seq_len); see resolve_hparams

    # ── optimiser ──────────────────────────────────────────────────────
    lr                = 1e-3,
    lr_schedule       = "const",  # opt-in: "cos:MIN" or "cos:MIN:HOLD"; see schedules.parse_lr
    restore_best      = False,    # opt-in: hand back the best-validation weights, not the last step's
    weight_decay      = 0.01,
    max_epochs        = 150,
    val_every_n_steps = 25,
    accelerator       = "auto",

    # ── analysis ───────────────────────────────────────────────────────
    ana_batch         = 32,
    max_batches       = 20,
    state_min_pos     = 5,
    state_tol         = 0.10,   
    cluster_metric    = "euclidean",
    n_pts             = 1000,
    conv_tol          = 0.10,   # |CE - H_inf| above this = not converged
    trans_total_run   = 5000,
)

# The seven baseline processes -- what `--all` runs, and the direct counterpart
# of the archived 00_base.
COIN_PQ   = [(0.1, 0.9), (0.3, 0.4), (0.4, 0.8)]
FLOWER_NM = [(2, 6), (2, 8), (4, 2), (6, 4)]


def resolve_hparams(cfg: dict, num_token: int) -> dict:
    """
    Set beta value to that of set by the config
    """
    beta = (1.0 / (cfg["batch"] * cfg["seq_len"]) if cfg["usage_beta"] is None
            else float(cfg["usage_beta"]))
    return dict(n_states=int(cfg["n_states_mult"] * num_token),
                state_dim=int(cfg["state_dim_mult"] * num_token),
                usage_beta=beta, tau=cfg["tau"])


def _finish(spec: dict, cfg: dict) -> dict:
    spec.update(resolve_hparams(cfg, spec["num_token"]))
    return spec


def coin_spec(cfg: dict, p: float, q: float) -> dict:
    """Everything about one coin process that does not depend on the repeat."""
    C_plus, C_minus = coin_complexity(p, q)
    return _finish(dict(
        tag        = coin_tag(p, q),
        kind       = "coin",
        params     = dict(p=p, q=q),
        num_token  = 3,
        theory     = entropy_rate_coin(p, q),          # H_inf, both arms
        C_plus     = C_plus, C_minus = C_minus,
        true_k_fw  = causal_state_count("coin", "forward"),
        true_k_bw  = causal_state_count("coin", "backward"),
        occ_fw     = causal_state_occupancy("coin", "forward",  p=p, q=q),
        occ_bw     = causal_state_occupancy("coin", "backward", p=p, q=q),
        T_theory_fw = coin_transition_matrix(p, q),
        T_theory_bw = coin_rev_transition_matrix(p, q),
    ), cfg)


def flower_spec(cfg: dict, n: int, m: int) -> dict:
    """Everything about one flower process that does not depend on the repeat."""
    dice = make_dice(n, m, cfg["flower_dice_seed"])
    C_plus, C_minus = flower_complexity(n, m, dice)
    return _finish(dict(
        tag        = flower_tag(n, m),
        kind       = "flower",
        params     = dict(n=n, m=m, dice_probs=dice),
        num_token  = n + m,
        theory     = flower_entropy_rate(n, m, dice),
        C_plus     = C_plus, C_minus = C_minus,
        true_k_fw  = causal_state_count("flower", "forward", n=n, m=m),
        true_k_bw  = causal_state_count("flower", "backward", n=n, m=m,
                                        dice_probs=dice),
        occ_fw     = causal_state_occupancy("flower", "forward", n=n),
        occ_bw     = causal_state_occupancy("flower", "backward", n=n, m=m,
                                            dice_probs=dice),
        T_theory_fw = flower_transition_matrix(n),
        T_theory_bw = flower_rev_transition_matrix(n, m, dice),
    ), cfg)


def baseline_specs(cfg: dict) -> list:
    """The seven processes -- three coin, four flower."""
    return ([coin_spec(cfg, p, q) for p, q in COIN_PQ]
            + [flower_spec(cfg, n, m) for n, m in FLOWER_NM])


def coin_grid_specs(cfg: dict, values) -> list:
    """
    The (p, q) grid, k values give k**2 cells.
    """
    v = [float(x) for x in values]
    return [coin_spec(cfg, p, q) for p in v for q in v]


def flower_grid_specs(cfg: dict, values) -> list:
    """The (n, m) grid"""
    v = [int(x) for x in values]
    return [flower_spec(cfg, n, m) for n in v for m in v]
