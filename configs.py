"""
configs.py — every experiment configuration in one place.

IMPROVEMENT_PLAN.md A4.  `Main_call.py` and `main_large.py` were two copies of
the same 700-line runner differing only in their CFG block and two tag
literals, and *both* hard-coded `OUT_ROOT = "results"`.  Both also used the tag
"exp1_coin_p03_q04" for experiment 1 -- at different parameters (0.4/0.8 vs
0.3/0.4) and different d_model (32 vs 64).  There was exactly one
results/models/exp1_coin_p03_q04_{fw,bw}.pt pair, so whichever runner finished
last owned it, and results/models/ ended up holding a mixture from at least two
runs.

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
    embed_type       = "onehot",
    n_folds          = 5,
    n_layers         = 2,
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
    out_root            = "results_smoke",
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
    out_root            = "results_quick",
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
    out_root            = "results_large",
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


CONFIGS = {"SMOKE": SMOKE, "QUICK": QUICK, "LARGE": LARGE}
