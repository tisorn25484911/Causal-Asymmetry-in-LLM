"""
configs.py — every experiment configuration in one place.

That is not untidiness, it is a correctness bug: LLM_asymmetry_testing.py and
Test_data_eval.py hard-code main_large's tags *and* its CFG, so running either
after Main_call.py would load weights trained at p=0.4,q=0.8 and score them
against p=0.3,q=0.4 data and theory.  Today the d_model mismatch (32 vs 64)
makes load_state_dict raise, which is the only reason this has not silently
produced nonsense.  Aligning d_model would make it go quiet.

Two changes remove the whole class of error:

  * `out_root` is part of the config, so each config writes to its own
    directory and cannot overwrite another's weights.
  * tags are DERIVED from the config values (see Main_call.coin_tag), so a tag
    can never disagree with the parameters that produced it.

Select one with `python run_experiments.py --config QUICK`.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Shared defaults.  Anything a config does not override comes from here.
# ─────────────────────────────────────────────────────────────────────────────
BASE = dict(
    # ── reproducibility (A2) ───────────────────────────────────────────
    seed             = 0,
    # ── model ──────────────────────────────────────────────────────────
    # Which architecture.  "onehot" is the continuous decoder every existing
    # result used; "discrete" forces the prediction through a hard one-hot over
    # a bounded causal-state set (MODULAR_MODELS_PLAN.md).  The four settings
    # below apply only to "discrete" and are ignored by "onehot".
    embed_type       = "onehot",
    # ── discrete bottleneck ────────────────────────────────────────────
    # All four resolve through Model_analysis.discrete_hparams(cfg, V, batch),
    # which the runners call.  DISCRETE_V2_PLAN.md sections 2 and 6.
    #
    # K -- the state BUDGET, not an estimate.  None -> n_states_mult * V.
    # An explicit int overrides, which is how the K sweep works.
    #
    # Measured: K at the exact theoretical count FAILS.  flower(2,3) forward at
    # K=3, its true count, recovers 2 states at ARI 0.745; K=4 the same.  Slack
    # is required.  At K=5V both pilot processes recovered their full state set
    # with S_emp within 0.006 bits of the closed form (flower(2,3) 5V=25, ARI
    # 1.000, S_emp-C+ +0.0050; flower(3,5) 5V=40, ARI 0.982, +0.0064).
    #
    # Single-seed evidence from processes outside the baseline seven: a starting
    # value, not an established optimum.  01_ksweep tests it.
    n_states         = None,
    n_states_mult    = 5,
    # Dimension of a state VECTOR.  None -> state_dim_mult * V.
    #
    # It carries NO expressive power: state_matrix followed by emission composes
    # to one (K, V) map, so state_matrix @ Q with inv(Q) . emission is the same
    # model for any invertible Q.  Setting it to K would make state_matrix mostly
    # reparameterisation freedom -- 1600 parameters at K=40, of which only the
    # K x V emission table is identified.
    #
    # Measured over 28 cells: S=V gave 6 exact state-count recoveries, S=K gave
    # 3, S=round(1.3V) gave 5.
    state_dim        = None,
    state_dim_mult   = 1,
    # Straight-through surrogate temperature.  Never changes the forward value.
    # Measured: tau <= 0.2 collapses the bottleneck.  1.0 is plain softmax.
    tau              = 1.0,
    # Strength of the complexity penalty.  None -> 1 / (batch * train_chunk_len),
    # computed by discrete_hparams from the geometry actually in use.
    #
    # The penalty is beta * H(p_bar), and p_bar is the state occupancy averaged
    # over every scored token, so beta is naturally per-token: 1/N with N the
    # count the cross-entropy itself averages over.
    #
    # NEVER write this as a literal.  Measured, beta is bounded above by an
    # optimisation cliff -- above ~6e-4 at lr=1e-3 the bottleneck collapses to
    # one state -- and the cliff scales with 1/lr, not with N.  At this geometry
    # N = 32*256 = 8192 so 1/N = 1.22e-4, about 5x below it.  A literal decouples
    # beta from N: at N = 32*49 the same rule gives 6.4e-4, already past the
    # cliff, and 1 of 28 pilot cells collapsed there against 0 at every larger N.
    #
    # An explicit float overrides, which is how 06_beta_verify sweeps around 1/N.
    usage_beta       = None,
    n_folds          = 5,
    n_layers         = 2,
    # ── optimiser ──────────────────────────────────────────────────────
    # AdamW's decay coefficient.  0.0 is bit-identical to the plain Adam every
    # result in this repository was produced with, so this default changes
    # nothing; a nonzero value is a NEW experimental condition, not a
    # correction of the old ones.
    #
    # Two uses.  It restores a finite optimum -- cross-entropy on a
    # deterministic transition has none, which is why long runs diverge
    # (measured on the flower null control, 6 seeds: Adam lr=5e-3 diverged 4/6,
    # AdamW wd=0.01 1/6) -- and it is a continuous capacity axis, which is what
    # a null delta_CE needs before it can be distinguished from "capacity
    # absorbed the asymmetry".
    #
    # Its total effect scales with lr * weight_decay * steps, so a grid must be
    # chosen for the step budget it runs at: at lr=1e-2 over the 130 steps the
    # repeat harness uses, lambda=0.01 shrinks a weight by only 1.3%.
    weight_decay     = 0.0,
    # D1: the train-loss curve stays at every step (free -- the loss is
    # already computed); only the expensive validation pass is sub-sampled.
    # Measured overhead of validating every step, vs not validating at all:
    # 1.63x at QUICK sizes and 2.58x at LARGE sizes.  Note the pq sweep trains
    # without a val_loader, so it never paid this and D1 does not speed it up.
    val_every_n_steps = 25,
    # "auto" uses Apple MPS here, which is ~6x faster than CPU at these sizes
    # but is NOT bit-reproducible across runs (see Training_model.train_model).
    # Pairing of the two arms is unaffected.  Set "cpu" for an exactly
    # reproducible run at ~6x the wall-clock.
    accelerator      = "auto",
    # ── analysis ───────────────────────────────────────────────────────
    # Analysis runs at the TRAINING chunk length, not full sequence length --
    # see Training_model.make_analysis_loader.  The model only ever sees
    # positional-encoding indices [0, chunk), so evaluating at full length
    # measures extrapolation rather than what was learned, and does so
    # asymmetrically between the two arms.  Because T is back to chunk size,
    # the attention matrix is the same size as in training and the analysis
    # batch does not need to be shrunk.
    ana_batch        = 32,
    max_batches      = 20,
    # B7: 15, not 200.  200 neighbours on 1000 points smears exactly the local
    # cluster structure the plot exists to show.  (This value was unreachable
    # until B7 was fixed -- every plot silently used the hard-coded 200.)
    umap_n_neighbors = 15,
    umap_n_pts       = 1000,
    # C6: positions with too little context are dropped from the "random"
    # UMAP panel.  Direction-aware -- trimmed from the start for a forward
    # model and from the end for a backward one.
    umap_burn_in     = 32,
    # C1: distance threshold on the model's PREDICTIVE DISTRIBUTION, used to
    # DISCOVER the number of causal states instead of assuming it.  This is a
    # genuine free parameter and no single value is right everywhere: the
    # smallest true separation between backward states ranges from 0.135 at
    # p=0.1,q=0.9 to 0.612 at p=q=0.5.  recover_causal_states therefore always
    # reports k̂ across a grid of thresholds as well as at this one, and the
    # plateau -- the k that survives the widest span -- is the defensible claim.
    state_tol        = 0.10,
    coin_num_token   = 3,
    flower_dice_seed = 42,
    # ── flower configurations (A1) ─────────────────────────────────────
    # C+ = 1 + (1/2)log2(n) and C- <= 1 + (1/2)log2(m), so C- > C+ requires
    # m > n.  Every flower config originally in the repo had n > m, i.e. all
    # of them tested the negation of the hypothesis.  Both signs are run now,
    # because delta_CE tracking the sign of (C- - C+) across configurations is
    # much stronger evidence than one positive case.
    flower_configs   = [
        (2, 6, "positive"),   # C+=1.500  C-~2.111  -> predict delta_CE > 0
        (4, 2, "reversed"),   # C+=2.000  C-~1.495  -> predict delta_CE < 0
    ],
)


def _cfg(**overrides) -> dict:
    cfg = dict(BASE)
    cfg.update(overrides)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE — tiny; exercises every code path in a couple of minutes.
# Not for results: 3 epochs on 40 sequences measures nothing.
# ─────────────────────────────────────────────────────────────────────────────
SMOKE = _cfg(
    out_root            = "All_Results/results_smoke",
    d_model             = 16,
    n_folds             = 2,
    lr                  = 1e-2,
    train_chunk_len     = 64,
    attn_vis_len        = 32,
    ana_batch           = 4,
    max_batches         = 2,
    val_every_n_steps   = 5,     # SMOKE folds are only ~15 steps long
    umap_n_pts          = 200,
    coin_p1             = 0.4,  coin_q1             = 0.8,
    coin_num_samples    = 40,   coin_seq_len        = 200,
    coin_max_epochs     = 3,    coin_batch          = 8,
    coin_p2             = 0.1,  coin_q2             = 0.9,
    coin_num_samples_12 = 40,   coin_seq_len_12     = 200,
    flower_configs      = [(2, 3, "positive"), (3, 2, "reversed")],
    flower_num_samples  = 40,   flower_seq_len      = 100,
    flower_max_epochs   = 3,    flower_batch        = 8,
    pq_grid             = [0.25, 0.75],
    pq_epochs           = 2,    pq_samples          = 40,   pq_len = 100,
)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK — the old Main_call.py.  ~1 hour.
# ─────────────────────────────────────────────────────────────────────────────
QUICK = _cfg(
    out_root            = "All_Results/results_quick",
    d_model             = 32,
    lr                  = 1e-2,
    train_chunk_len     = 256,   # tokens seen per training step
    attn_vis_len        = 64,    # tokens used for attention heatmap plots
    # ── coin exp 1 ─────────────────────────────────────────────────────
    coin_p1             = 0.4,  coin_q1             = 0.8,
    coin_num_samples    = 500,  coin_seq_len        = 2000,
    coin_max_epochs     = 10,   coin_batch          = 32,
    # ── coin exp 1.2 ───────────────────────────────────────────────────
    coin_p2             = 0.1,  coin_q2             = 0.9,
    coin_num_samples_12 = 500,  coin_seq_len_12     = 500,
    # ── flower exp 2 ───────────────────────────────────────────────────
    flower_num_samples  = 500,  flower_seq_len      = 2000,
    flower_max_epochs   = 10,   flower_batch        = 32,
    # ── pq heatmap ─────────────────────────────────────────────────────
    pq_grid   = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    pq_epochs = 5,  pq_samples = 500,  pq_len = 200,
)


# ─────────────────────────────────────────────────────────────────────────────
# LARGE — the old main_large.py.  ~13 hours, most of it the pq sweep.
# ─────────────────────────────────────────────────────────────────────────────
LARGE = _cfg(
    out_root            = "All_Results/results_large",
    d_model             = 64,
    lr                  = 5e-3,
    train_chunk_len     = 512,
    attn_vis_len        = 128,
    # ── coin exp 1 ─────────────────────────────────────────────────────
    # 2000 samples x 80 epochs x 5 folds x 2 directions ~ 2.5 hr
    coin_p1             = 0.3,  coin_q1             = 0.4,
    coin_num_samples    = 2000, coin_seq_len        = 2000,
    coin_max_epochs     = 80,   coin_batch          = 64,
    # ── coin exp 1.2 ───────────────────────────────────────────────────
    # CV (~2 hr) + pq grid (~8 hr)
    coin_p2             = 0.4,  coin_q2             = 0.8,
    coin_num_samples_12 = 2000, coin_seq_len_12     = 500,
    # ── flower exp 2 ───────────────────────────────────────────────────
    # 2000 samples x 80 epochs x 5 folds x 2 directions ~ 3.5 hr per config.
    # The old (6,4) had C+=2.2925 > C-=1.9899, i.e. reversed; (2,8) is the
    # positive case (C+=1.500, C- up to 2.5).
    flower_configs      = [(2, 8, "positive"), (6, 4, "reversed")],
    flower_num_samples  = 2000, flower_seq_len      = 2000,
    flower_max_epochs   = 80,   flower_batch        = 64,
    # ── pq heatmap ─────────────────────────────────────────────────────
    # 2 x 16 x 16 = 512 models x 25 epochs ~ 8 hr
    pq_grid   = [0.05, 0.10, 0.17, 0.23, 0.30, 0.37, 0.43, 0.50,
                 0.57, 0.63, 0.70, 0.77, 0.83, 0.90, 0.95, 0.99],
    pq_epochs  = 25,
    pq_samples = 1000,
    pq_len     = 400,
)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK_LARGE_HMM — QUICK's scale, capacity and learning rate, run on LARGE's
# PROCESSES.  Writes into the same out_root as QUICK.
#
# Why this exists.  QUICK and LARGE differ along four axes at once: data volume
# (8x the training tokens), capacity (d_model 32 vs 64), optimisation (lr 1e-2
# vs 5e-3, 100 vs 1600 steps per fold) AND the processes themselves -- they are
# trained on different HMMs with different entropy rates and different
# theoretical asymmetries.  That last one makes the two runs non-comparable:
# any difference between them confounds scale, capacity and process.
#
# Running LARGE's processes at QUICK's settings holds the process fixed, so
# QUICK vs LARGE becomes a controlled comparison of scale+capacity+lr.
#
# Only the processes MISSING from QUICK are listed.  LARGE's exp1.2 coin
# (p=0.4, q=0.8) is already QUICK's exp1, so it is not repeated -- and because
# tags are derived from parameter values (A4), re-running it would have landed
# on the same tag rather than creating a duplicate.
#
#     python run_experiments.py --config QUICK_LARGE_HMM --only exp1
#     python run_experiments.py --config QUICK_LARGE_HMM --only exp2
#
# exp1_2 is deliberately not run: it would repeat a process QUICK already has
# and re-run the pq sweep for no new information.
# ─────────────────────────────────────────────────────────────────────────────
QUICK_LARGE_HMM = dict(QUICK)
QUICK_LARGE_HMM.update(
    coin_p1        = LARGE["coin_p1"],   coin_q1 = LARGE["coin_q1"],   # 0.3, 0.4
    flower_configs = list(LARGE["flower_configs"]),                    # (2,8), (6,4)
)


# ─────────────────────────────────────────────────────────────────────────────
# DISCRETE — QUICK's processes and data, at the settings the discrete
# causal-state bottleneck actually converges under.
#
# QUICK is lr=1e-2 over ~130 gradient steps, which the continuous decoder
# reaches H_inf under.  The discrete decoder does NOT: measured on the coin at
# d_model=32, lr=1e-2 leaves the forward arm collapsed onto 1 of its 2 states
# even after 1500 steps, sitting 0.57 bits above H_inf.
#
#     lr      steps    forward CE-H_inf  states    backward CE-H_inf  states
#     1e-2     1500          +0.573       1/2           +0.120         2/3
#     3e-3     3000          +0.197       2/2           +0.084         3/3
#     1e-3     3000          +0.087       2/2           +0.088         3/3
#
# At lr=1e-3 the two arms converge TOGETHER -- +0.087 against +0.088, an arm
# difference of ~0.001 bits, which is the order of the effect under study.  So
# the two architectures are each run at a working point, rather than at a
# shared one where one of them fails.  That difference is a real caveat on any
# model-to-model comparison and must be stated with the results, not buried.
#
# The step budget is raised by increasing epochs, not by shrinking the data:
# shrinking it would change the process being learned as well as the budget.
DISCRETE = _cfg(
    **{k: v for k, v in QUICK.items()
       if k not in ("out_root", "lr", "embed_type", "weight_decay",
                    "coin_max_epochs", "flower_max_epochs", "pq_epochs")},
    out_root            = "All_Results/discrete/quick",
    embed_type          = "discrete",
    lr                  = 1e-3,
    # 150 epochs, and weight_decay ON -- both measured, and the epoch count is a
    # CEILING rather than a floor.  The first version of this config used 230
    # with no decay, on the reasoning that the backward arm had not converged at
    # +0.13 above H_inf and therefore needed longer.  That was wrong: it had
    # converged and then DIVERGED past it.  Held-out CE for the coin backward
    # arm, lr=1e-3:
    #
    #   epochs      10      25      50      75     100     150     200     300
    #   wd=0.0   +.2884  +.1719  +.0999  +.0231  +.0222  +.0621  +.1345  +.2006
    #   wd=0.01  +.2889  +.1844  +.1208  +.0215  +.0238  +.0155  +.1213  +.1854
    #
    # Cross-entropy on a deterministic transition has no finite minimiser, and
    # the backward coin conditional has a point-mass row, P(prev|2) = [0,1,0].
    # So the backward arm overshoots and the held-out loss climbs again.  At 230
    # it sat outside conv_tol, which is why the first discrete run converged on
    # only 8-9 of 30 repeats for two coin processes.
    #
    # At 150 with wd=0.01 EVERY arm of both process families is inside
    # conv_tol=0.1 -- coin fw +0.0087 bw +0.0223, flower(2,8) fw +0.0082
    # bw +0.0856.  The forward arm would still improve at 230, but both arms must
    # share a budget or delta_CE stops being paired, so the budget is set by the
    # arm that fails first.
    #
    # weight_decay is 0.01 here and 0.0 everywhere else: AdamW restores a finite
    # optimum on exactly the deterministic transitions that cause this, which is
    # what the weight-decay work already established.
    coin_max_epochs     = 150,
    flower_max_epochs   = 150,
    weight_decay        = 0.01,
    pq_epochs           = 75,
)


CONFIGS = {"SMOKE": SMOKE, "QUICK": QUICK, "LARGE": LARGE,
           "QUICK_LARGE_HMM": QUICK_LARGE_HMM,
           "DISCRETE": DISCRETE}
