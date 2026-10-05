"""
Train and score the discrete transformer and GRUs on the SAME data: one draw of
each process, both arms, one seeded split.

The backward arm is the same draw read the other way (generator.reverse(), which
also switches the theory to the reverse machine M-); every model runs at
mode="forward".  So all models of one process see byte-identical sequences and
the same hold-out rows.

    python run_model.py --name default                                transformer + gru_feedback x coin, flower x both arms
    python run_model.py --name default --arch gru_feedback --process coin --arm forward   one run
    python run_model.py --name even --process even --params p=0.3     any process in HMM_processes.PROCESS
    python run_model.py --name gumbel --gumbel-gru --gumbel-transformer   a named variant of the config
    python run_model.py --name default --plots-only                   redraw every figure from the pickles
    scripts/launch_grid.sh default                                    the whole grid in parallel, one log per run
    python run_model.py --smoke                                       2 epochs on 64 sequences -> checks/results/smoke
    python run_model.py --name cont --data-mode continuous --process renewal --num-states-mult 8
                                                                      the continuous data mode (below)

Two data modes (SHARED_TRAJECTORY_EMBEDDING_WARMUP_PLAN.md):
  legacy      1000 independent 300-token draws, a seeded 80/20 split by sequence, every output
              scored, the argmax head and the modular optimiser -- the runs before 2026-10-03,
              unchanged bit for bit.
  continuous  ONE 300,000-token trajectory per process (saved once under <out>/<name>/data/),
              split 80/10/10 by token range before any sample is cut; the transformer reads
              301-token blocks scoring outputs >= 20, both GRUs 30-token windows (20 context +
              10 scored) at stride 1, validation/test at stride 10; training rows re-cut at a
              random offset every epoch; checkpoints selected on validation, scored on test; one
              regime for all three (Gumbel-ST, AdamW 5e-4, cosine LR, clip 1, tau 2.25 -> 0.1) on
              schedules indexed to twice the 30,000-update budget; every model also scored on the
              common profile (test blocks of 301, outputs >= 20) so architectures compare.

README.md explains the models, the theory comparison and how to add a process.

Where things go -- results/ holds real runs ONLY; smoke runs, cross-checks and
diagnostics write under checks/results/ instead:

    results/<name>/runs/<arch>_<process>_<arm>_s<seed>.pkl     everything a run produced (incl. weights)
    results/<name>/runs/<arch>_<process>_<arm>_s<seed>.json    its config + headline metrics, human-readable
    results/<name>/logs/<arch>_<process>_<arm>_s<seed>.log     (scripts/launch_grid.sh)
    results/<name>/figures/scorecard_s<seed>.png               every run of one seed on one page
    results/<name>/figures/<process>/<arch>/<arm>_s<seed>_{training,states,machine}.png
    results/<name>/figures/<process>/<arch>/s<seed>_{per_position,transitions,complexity}.png
                                                               (+ _transitions_exact.png for gru_feedback)

From a notebook:  res = run_arm(RunConfig(), "gru", "coin", "backward").
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Optional

import numpy as np
import torch
import torch.utils.data as tud

from data_views import WindowSpec, build_views, canonical_trajectory, make_loaders, sha1, split_spans
from extraction import analyse_arm, exact_per_position_ce
from HMM_processes import PROCESS
from process_generator import SequenceDataset, generator, split_loader
from scoring import score_mask
from training import (cleanup, load_pkl, mkdir, quiet, save_pkl, set_seed,
                      to_cpu_for_analysis, train_model)

HERE = os.path.dirname(os.path.abspath(__file__))

# Parameters used when a process is chosen without --params: every one checked to
# build its forward and reverse epsilon-machines (tests/test_models.py).  `edges` and
# `tilt` have none -- they are defined BY their parameters (see HMM_processes.py).
# coin and flower take theirs from the RunConfig fields below instead.
DEFAULT_PARAMS = {
    "pcoin": {"p": 0.3, "q": 0.4},
    "iid": {"probs": [0.5, 0.3, 0.2]},
    "rip": {"p": 0.4, "q": 0.3},
    "rnc": {"p": 0.5, "q": 0.3},
    "butterfly": {},
    "explosive": {},
    "t3": {},
    "even": {"p": 0.5},
    "gm": {"p": 0.5},
    "nemo": {"p": 0.5, "q": 0.5},
    "rgm": {"k": 2},
    "sgm": {"k": 2, "p": 0.5},
    "ising": {},
    "sns": {"p": 0.5, "q": 0.5},
    "renewal": {"F": [0.1, 0.2, 0.3, 0.25, 0.15]},     # 5 causal states, C+ = C- = 2.111 bits
}
DEFAULT_BURN_IN = 100                  # generator's default; coin and flower have their own fields


@dataclass
class RunConfig:
    # --- processes ---------------------------------------------------------
    coin_p: float = 0.70
    coin_q: float = 0.80
    coin_burn_in: int = 200
    flower_n: int = 3
    flower_m: int = 4
    flower_dice_seed: int = 0          # the dice DEFINE the flower; 0 = the notebooks' / study5's
    flower_burn_in: int = 4
    # overrides for any process: {"even": {"p": 0.3}, "coin": {"p": 0.6}, ...}
    process_params: dict = field(default_factory=dict)
    data_seed: int = 1                 # rng for the one draw every model trains on

    # --- data: tau_experiment/study5's shape, so 800 epochs = 20k updates ---
    num_samples: int = 1000
    seq_len: int = 300
    batch_size: int = 32
    test_ratio: float = 0.20

    # --- models --------------------------------------------------------------
    d_model: int = 16
    n_layers_transformer: int = 4
    n_layers_gru: int = 1
    num_states_mult: int = 4           # K = 4V, the notebook GRU's budget (user, 2026-09-28);
                                       # >= k for coin (3/V=3), flower (5/V=7) and renewal (5/V=2)
    state_dim_mult: int = 1            # S = V
    tau: str = "geom:5:0.5"            # the GRUs (tau_schedue.py)
    # The transformer's schedule (user's standard, 2026-09-25): descend 5 -> 0.5 over the
    # first 80% of the run, then hold 0.5 (study5's `geomhold`; the 4th number is the
    # DESCENT fraction).  At 1200 epochs: 960 descending, 240 held.
    tau_transformer: str = "geomhold:5:0.5:0.8"
    usage_beta: Optional[float] = None # None -> 1 / (batch_size * seq_len)

    # --- training ------------------------------------------------------------
    # 25 batches x 1200 epochs = 30,000 updates for every model: the notebook GRU's 29,850
    # (user, 2026-09-28: match its UPDATES; the data stay 1000 x 301 tokens, ~9x its 30,000).
    max_epochs: int = 1200             # the GRUs
    max_epochs_transformer: int = 1200
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    accelerator_transformer: str = "auto"   # MPS on a Mac, as study5
    accelerator_gru: str = "cpu"            # both GRUs; measured ~4x faster than MPS for nn.GRU here
    val_every_n_steps: int = 25
    restore_best: bool = True
    gradient_clip_val: Optional[float] = None

    gumbel_transformer: bool = False
    gumbel_gru: bool = False                  # gru_feedback and gru
    progress_every: int = 2500               # a progress line every this many steps (0 = none)

    # --- extraction ----------------------------------------------------------
    fresh_samples: int = 100
    ana_batch: int = 32
    state_min_pos: int = 5
    conv_tol: float = 0.10
    full_tol: float = 0.05
    trans_total_run: int = 5000

    random_seed: int = 0               # split + initialisation + generation rng

    # --- the continuous data mode (SHARED_TRAJECTORY_EMBEDDING_WARMUP_PLAN.md section 12.8) ---
    data_mode: str = "legacy"          # "legacy" | "continuous"
    total_tokens: int = 300_000
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    transformer_raw_window: int = 301
    transformer_stride: int = 301
    transformer_batch_size: int = 32
    transformer_score_start: int = 20
    gru_raw_window: int = 30
    gru_stride: int = 1           # user, 3 Oct 2026 (was 10)
    gru_eval_stride: int = 10     # validation/test windows: disjoint scored targets
    gru_batch_size: int = 128     # user, 3 Oct 2026 (was 32)
    gru_score_start: int = 19
    gru_d_model: int = 32         # continuous mode; legacy keeps d_model for every architecture
    transformer_d_model: int = 16
    random_shift: bool = True     # continuous mode: re-cut training rows every epoch (user, 3 Oct 2026)
    learn_input_embedding: bool = False
    usage_scope: Optional[str] = None
    max_steps: Optional[int] = None
    generation_context: int = 300
    # training regime; continuous mode takes CONTINUOUS_REGIME below unless a flag overrides it
    lr_schedule: str = "constant"       # "constant" | "cosine"
    lr_min: float = 1e-5
    schedule_horizon_mult: float = 1.0  # horizon = mult x update budget
    # validate the final step when the cadence misses it; False reproduces a historical cadence exactly
    final_validation: bool = True
    # the RunConfig fields set explicitly (command-line flags): resolve_regime leaves them alone
    explicit_overrides: tuple = ()


# Continuous-mode defaults, applied by resolve_regime to every field not set explicitly.
CONTINUOUS_REGIME = dict(          # section 0, decisions 1-2 (study6; user, 3 Oct 2026)
    gumbel_transformer=True, gumbel_gru=True,
    tau="geom:2.25:0.1", tau_transformer="geom:2.25:0.1",
    weight_decay=5e-4, lr_schedule="cosine", lr_min=1e-5,
    gradient_clip_val=1.0, schedule_horizon_mult=2.0,
)
CONTINUOUS_MAX_STEPS = 30_000      # the update budget unless --max-steps or an explicit epoch budget
CONTINUOUS_USAGE_BETA = 1.0 / 9600  # fixed, not recomputed from the new batch geometry (plan section 5)
# The common evaluation profile (user, 2026-10-03: study6's): the TEST split cut into fixed
# 301-token blocks, outputs >= 20 scored, for every architecture -- identical targets and context,
# so cross-architecture CE compares.
COMMON_EVAL = {"raw_window": 301, "stride": 301, "score_start": 20}


def resolve_regime(cfg, explicit):
    """Continuous-mode training defaults; fields the user set on the command line
    (`explicit`) win.  Legacy mode is returned unchanged."""
    if cfg.data_mode != "continuous":
        return cfg
    return replace(cfg, **{k: v for k, v in CONTINUOUS_REGIME.items()
                           if k not in explicit})


def window_settings(cfg, arch):
    prefix = "transformer" if arch == "transformer" else "gru"
    return WindowSpec(
        raw_window=getattr(cfg, f"{prefix}_raw_window"),
        stride=getattr(cfg, f"{prefix}_stride"),
        batch_size=getattr(cfg, f"{prefix}_batch_size"),
        score_start=getattr(cfg, f"{prefix}_score_start"),
        eval_stride=getattr(cfg, f"{prefix}_eval_stride", None),
        shift=cfg.random_shift,
    )


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    return str(v)


def process_params(cfg: RunConfig, process: str):
    """
    (generator params, burn-in, tag) for any process registered in
    HMM_processes.PROCESS: its defaults (the RunConfig fields for coin and flower,
    DEFAULT_PARAMS otherwise) updated by cfg.process_params[process].
    """
    if process not in PROCESS:
        raise ValueError(f"unknown process {process!r}; registered in HMM_processes.PROCESS: {sorted(PROCESS)}")
    over = dict(cfg.process_params.get(process, {}))
    if process == "coin":
        params, burn_in = {"p": cfg.coin_p, "q": cfg.coin_q}, cfg.coin_burn_in
    elif process == "flower":
        params = {"n": cfg.flower_n, "m": cfg.flower_m, "dice_seed": cfg.flower_dice_seed}
        burn_in = cfg.flower_burn_in
    else:
        if process not in DEFAULT_PARAMS and not over:
            raise ValueError(f"process {process!r} has no default parameters; pass them with --params "
                             f"(see its functions in HMM_processes.py)")
        params, burn_in = dict(DEFAULT_PARAMS.get(process, {})), DEFAULT_BURN_IN
    burn_in = int(over.pop("burn_in", burn_in))
    params.update(over)
    if process == "coin":
        tag = f"coin({params['p']:g},{params['q']:g})"
    elif process == "flower":
        tag = f"flower({params['n']},{params['m']})"
    else:
        tag = process + (("(" + ",".join(f"{k}={_fmt(v)}" for k, v in sorted(params.items())) + ")")
                         if params else "")
    return params, burn_in, tag


def arch_settings(cfg: RunConfig, arch: str, V: int) -> dict:
    """
    Everything that differs between architectures, in one place (run_arm and the
    diagnostics both read it): depth, the state vector's width, device, sampler,
    schedule and length.
    """
    if arch not in ("transformer", "gru", "gru_feedback"):
        raise ValueError(f"unknown arch {arch!r}")
    transformer = arch == "transformer"
    # continuous mode: per-architecture widths (GRUs 32, transformer 16); legacy: one d_model
    d_model = ((cfg.transformer_d_model if transformer else cfg.gru_d_model)
               if cfg.data_mode == "continuous" else cfg.d_model)
    return {
        "n_layers": {"transformer": cfg.n_layers_transformer, "gru": cfg.n_layers_gru, "gru_feedback": 1}[arch],
        "d_model": d_model,
        # the feedback GRU's state vector IS its hidden state, so S = d_model there
        "state_dim": d_model if arch == "gru_feedback" else cfg.state_dim_mult * V,
        "accelerator": cfg.accelerator_transformer if transformer else cfg.accelerator_gru,
        "gumbel": cfg.gumbel_transformer if transformer else cfg.gumbel_gru,
        "tau": cfg.tau_transformer if transformer else cfg.tau,
        "max_epochs": cfg.max_epochs_transformer if transformer else cfg.max_epochs,
    }


def draw(cfg: RunConfig, process: str, arm: str, n: int, seq_len: int, seed: int):
    """
    (generator in the arm's view, its (n, seq_len+1) sequences in that view's
    reading order).  The draw is always made forwards from the process;
    reverse() flips it and switches the theory, so the backward arm of a seed is
    exactly the forward draw read right to left.
    """
    params, burn_in, _ = process_params(cfg, process)
    gen = generator(process, n, seq_len, params, burn_in=burn_in)
    gen.generate(np.random.default_rng(seed))
    if arm == "backward":
        gen.reverse()
    return gen, np.array(gen.all_sequence, dtype=np.int64, copy=True)


def canonical_draw(cfg: RunConfig, process: str, data_dir=None):
    """
    The continuous mode's ONE forward trajectory of cfg.total_tokens tokens for this process,
    parameters and data seed: (raw, sha1, path, meta).  Saved once as
    <data_dir>/<process>_<hash of meta>.npz with its generation parameters and verified on
    every reuse (data_views.canonical_trajectory), so every architecture and arm reads the
    same tokens.  data_dir=None draws it in memory only.
    """
    params, burn_in, _ = process_params(cfg, process)
    meta = {"process": process, "params": params, "burn_in": burn_in, "data_seed": cfg.data_seed,
            "total_tokens": cfg.total_tokens}
    key = hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:10]
    path = None if data_dir is None else os.path.join(data_dir, f"{process}_{key}.npz")

    def make():
        _, seqs = draw(cfg, process, "forward", 1, cfg.total_tokens - 1, cfg.data_seed)
        return seqs[0]

    raw, digest = canonical_trajectory(path, meta, make)
    return raw, digest, path, meta


def common_rows(raw, spans, arm):
    """The common evaluation profile's rows (COMMON_EVAL): the test span cut into fixed blocks,
    each reversed for the backward arm; and the profile's id, the same for every architecture
    and arm of one draw (a comparison is valid only between runs whose ids match)."""
    lo, hi = spans["test"]
    L, s = COMMON_EVAL["raw_window"], COMMON_EVAL["stride"]
    starts = np.arange(lo, hi - L + 1, s)
    rows = raw[starts[:, None] + np.arange(L)]
    if arm == "backward":
        rows = rows[:, ::-1]
    manifest = {"raw_sha1": sha1(raw), "test_span": [int(lo), int(hi)], **COMMON_EVAL}
    mid = hashlib.sha1(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    return np.ascontiguousarray(rows), mid


def run_arm(cfg: RunConfig, arch: str, process: str, arm: str, verbose: bool = True,
            data_dir=None) -> dict:
    """Train one architecture on one arm of one process and score it -> result dict.
    cfg.data_mode picks the data (module docstring); `data_dir` is where the continuous
    mode keeps its canonical trajectory (None: in memory)."""
    if arm not in ("forward", "backward"):
        raise ValueError(f"arm must be 'forward' or 'backward', got {arm!r}")
    if cfg.data_mode == "continuous":
        return _run_arm_continuous(cfg, arch, process, arm, verbose, data_dir)
    if cfg.data_mode != "legacy":
        raise ValueError(f"data_mode must be 'legacy' or 'continuous', got {cfg.data_mode!r}")
    params, burn_in, tag = process_params(cfg, process)
    say = print if verbose else (lambda *a, **k: None)

    # --- data: the one draw, in this arm's reading order ----------------------
    gen, seqs = draw(cfg, process, arm, cfg.num_samples, cfg.seq_len, cfg.data_seed)
    fw = seqs if arm == "forward" else seqs[:, ::-1]
    data_sha1 = hashlib.sha1(np.ascontiguousarray(fw).tobytes()).hexdigest()[:12]
    ds = SequenceDataset(seqs)
    train_loader, test_loader = split_loader(ds, cfg.batch_size, cfg.test_ratio, seed=cfg.random_seed)
    test_indices = np.asarray(test_loader.dataset.indices)
    ana_loader = tud.DataLoader(ds, batch_size=cfg.ana_batch, shuffle=False)

    V = int(gen.vocab_size)
    K = cfg.num_states_mult * V
    a = arch_settings(cfg, arch, V)
    S, n_layers, accelerator = a["state_dim"], a["n_layers"], a["accelerator"]
    beta = 1.0 / (cfg.batch_size * cfg.seq_len) if cfg.usage_beta is None else float(cfg.usage_beta)
    say(f"[{arch} | {tag} | {arm}] V {V}, K {K}, S {S}, d_model {cfg.d_model}, {n_layers} layer(s), "
        f"tau {a['tau']}, {a['max_epochs']} epochs, gumbel {a['gumbel']}, beta {beta:.2e}, {accelerator} | "
        f"data {data_sha1} | theory: k {gen.n_causal}, C {gen.statistical_c:.4f}, H_inf {gen.entropy_rate:.4f}")
    if K < gen.n_causal:
        # K = num_states_mult * V is set from the vocabulary alone; a binary process
        # with many causal states can exceed it, and then no run can be FULL.
        say(f"  ! state budget K = {K} is below the {gen.n_causal} causal states of this arm: the "
            f"model cannot represent them all -- raise --num-states-mult")

    # --- train ------------------------------------------------------------------
    set_seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed * 1000)
    t0 = time.time()
    with quiet(True):
        rec = train_model(
            train_loader, arch, "discrete", val_loader=test_loader,
            token_size=V, d_model=a["d_model"], max_len=ds.seq_len, max_epochs=a["max_epochs"],
            lr=cfg.learning_rate, n_layers=n_layers, weight_decay=cfg.weight_decay, mode="forward",
            accelerator=accelerator, val_every_n_steps=cfg.val_every_n_steps,
            n_states=K, state_dim=S, tau=a["tau"], usage_beta=beta,
            restore_best=cfg.restore_best, gradient_clip_val=cfg.gradient_clip_val,
            progress_every=cfg.progress_every if verbose else 0, gumbel=a["gumbel"],
            usage_scope=cfg.usage_scope or "all", learn_input_embedding=cfg.learn_input_embedding,
            max_steps=cfg.max_steps, final_validation=cfg.final_validation)
    train_seconds = time.time() - t0
    model = to_cpu_for_analysis(rec.model)

    # --- score ------------------------------------------------------------------
    _, fresh = draw(cfg, process, arm, cfg.fresh_samples, cfg.seq_len, cfg.data_seed + 1000)
    fresh_loader = tud.DataLoader(SequenceDataset(fresh), batch_size=cfg.ana_batch, shuffle=False)
    _, seed_tokens = draw(cfg, process, arm, 1, cfg.seq_len + 50, cfg.random_seed + 2000)
    res = analyse_arm(model, rec, gen, ds, test_indices, test_loader, ana_loader,
                      fresh_loader=fresh_loader, seed_tokens=seed_tokens[0],
                      min_pos=cfg.state_min_pos, total_run=cfg.trans_total_run,
                      conv_tol=cfg.conv_tol, full_tol=cfg.full_tol,
                      rng=np.random.default_rng(cfg.random_seed))

    res.update({
        "arch": arch, "process": process, "tag": tag, "arm": arm,
        "process_params": params, "burn_in": burn_in, "budget_below_k": bool(K < gen.n_causal),
        # `tau` and `max_epochs` are what THIS run used (the per-architecture value),
        # so every figure and table reads the right one
        "cfg": {**asdict(cfg), "token_size": V, "n_states": K, "state_dim": S,
                "usage_beta_resolved": beta, "n_layers": n_layers, "accelerator": accelerator,
                "tau": a["tau"], "max_epochs": a["max_epochs"], "gumbel": a["gumbel"]},
        "model_class": type(model).__name__, "hparams": dict(model.hparams),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": train_seconds, "data_sha1": data_sha1,
        "test_indices": test_indices,
        "run_config": asdict(cfg),          # the RunConfig as given (config_collision)
    })
    m = res["metrics"]
    # FULL is judged on the machine after the notebook's merge; the raw verdict is kept beside it
    say(f"[{arch} | {tag} | {arm}] {'FULL' if m['full'] else 'not full'} (raw: {'FULL' if m['full_raw'] else 'not full'}) | "
        f"merged {m['k_used']} -> {m['merged_k']} states, S-C {m['merged_S_minus_C']:+.3f} | discovered "
        f"{m['discovered']}/{m['true_k']} (emission TV {m['emission_tv']:.3f}), used {m['k_used']}/{m['K']} | "
        f"S-C {m['S_minus_C']:+.3f} | "
        f"T_err {m['transition_max_err']:.3f} | gap {m['gap']:+.4f} (exact {m['gap_exact']:+.4f}, "
        f"fresh {m['fresh_gap']:+.4f}) | det {m['determinism']:.3f} | best {m['best_step']}/{m['n_steps']} | "
        f"max|g| {m['max_grad_norm']:.2f} | {train_seconds / 60:.1f} min")
    return res


def _jsonable(x):
    """Plain JSON types; a long integer array (window starts) becomes a summary, never opaque text."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and x.size > 64 and np.issubdtype(x.dtype, np.integer):
            return {"n": int(x.size), "first": int(x[0]), "last": int(x[-1]),
                    "sha1": hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()[:12]}
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def continuous_budget(cfg, arch, steps_per_epoch):
    """
    (max_steps, total updates, schedule horizon) of a continuous run: cfg.max_steps when set;
    else an epoch budget when one was given explicitly (--max-epochs for every architecture,
    --max-epochs-transformer for the transformer); else CONTINUOUS_MAX_STEPS.  The horizon is
    schedule_horizon_mult x the budget (plan section 12.8).
    """
    explicit = set(cfg.explicit_overrides)
    epochs_explicit = "max_epochs" in explicit or (arch == "transformer" and "max_epochs_transformer" in explicit)
    max_steps = (int(cfg.max_steps) if cfg.max_steps is not None
                 else None if epochs_explicit else CONTINUOUS_MAX_STEPS)
    epochs = cfg.max_epochs_transformer if arch == "transformer" else cfg.max_epochs
    total_steps = max_steps if max_steps is not None else steps_per_epoch * epochs
    return max_steps, total_steps, int(round(cfg.schedule_horizon_mult * total_steps))


def _run_arm_continuous(cfg, arch, process, arm, verbose, data_dir):
    """run_arm for cfg.data_mode == "continuous" (plan section 12.8)."""
    cfg = resolve_regime(cfg, set(cfg.explicit_overrides))      # idempotent; main() applied it already
    params, burn_in, tag = process_params(cfg, process)
    say = print if verbose else (lambda *a, **k: None)

    # --- data: the one trajectory, cut into this architecture's view --------------
    raw, raw_sha1, raw_path, raw_meta = canonical_draw(cfg, process, data_dir)
    gen = generator(process, 1, cfg.total_tokens - 1, params, burn_in=burn_in)   # the theory
    spec = window_settings(cfg, arch)
    spans = split_spans(len(raw), cfg.train_fraction, cfg.val_fraction)
    ds, split_indices, data_manifest = build_views(raw, spans, spec, arm)
    train_loader, val_loader, test_loader, ana_loader = make_loaders(
        ds, split_indices, spec, cfg.random_seed, cfg.ana_batch,
        raw=raw, train_span=spans["train"], arm=arm,
    )
    test_indices = split_indices["test"]
    if arm == "backward":
        gen.reverse()  # switches theory; sample rows are already reversed

    V = int(gen.vocab_size)
    K = cfg.num_states_mult * V
    a = arch_settings(cfg, arch, V)
    S, n_layers, accelerator, d_model = a["state_dim"], a["n_layers"], a["accelerator"], a["d_model"]
    beta = CONTINUOUS_USAGE_BETA if cfg.usage_beta is None else float(cfg.usage_beta)
    usage_scope = cfg.usage_scope or "scored"
    explicit = set(cfg.explicit_overrides)
    max_steps, total_steps, horizon = continuous_budget(cfg, arch, len(train_loader))
    # capacity for the longest context the model is run on: training, the common profile, generation
    max_len = max(spec.raw_window - 1, COMMON_EVAL["raw_window"] - 1, cfg.generation_context)
    say(f"[{arch} | {tag} | {arm} | continuous] V {V}, K {K}, S {S}, d_model {d_model}, {n_layers} layer(s) | "
        f"rows {spec.raw_window} stride {spec.stride} (eval {spec.stride_for('val')}), batch {spec.batch_size}, "
        f"score from {spec.score_start}, shift {spec.shift} | {total_steps} updates on a {horizon}-update horizon, "
        f"tau {a['tau']}, gumbel {a['gumbel']}, wd {cfg.weight_decay}, lr {cfg.lr_schedule}, clip "
        f"{cfg.gradient_clip_val}, beta {beta:.2e} ({usage_scope}), learned input {cfg.learn_input_embedding}, "
        f"{accelerator} | data {raw_sha1[:12]} | theory: k {gen.n_causal}, C {gen.statistical_c:.4f}, "
        f"H_inf {gen.entropy_rate:.4f}")
    if K < gen.n_causal:
        say(f"  ! state budget K = {K} is below the {gen.n_causal} causal states of this arm: the "
            f"model cannot represent them all -- raise --num-states-mult")

    # the overfitting monitor: a fixed subset of TRAINING rows, about as many as the validation rows
    train_rows = ds.seqs[split_indices["train"]]
    probe_rows = train_rows[::max(1, len(train_rows) // len(split_indices["val"]))]
    probe_loader = tud.DataLoader(SequenceDataset(probe_rows), batch_size=spec.batch_size, shuffle=False)
    probe_mask = score_mask(torch.from_numpy(probe_rows[:, 1:]), spec.score_start).numpy()
    probe_exact = float(exact_per_position_ce(gen, probe_rows)[probe_mask].mean())

    # --- train ------------------------------------------------------------------
    set_seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed * 1000)
    t0 = time.time()
    with quiet(True):
        rec = train_model(
            train_loader, arch, "discrete", val_loader=val_loader,
            token_size=V, d_model=d_model, max_len=max_len, max_epochs=a["max_epochs"],
            lr=cfg.learning_rate, n_layers=n_layers, weight_decay=cfg.weight_decay, mode="forward",
            accelerator=accelerator, val_every_n_steps=cfg.val_every_n_steps,
            n_states=K, state_dim=S, tau=a["tau"], usage_beta=beta,
            restore_best=cfg.restore_best, gradient_clip_val=cfg.gradient_clip_val,
            progress_every=cfg.progress_every if verbose else 0, gumbel=a["gumbel"],
            score_start=spec.score_start, usage_scope=usage_scope,
            learn_input_embedding=cfg.learn_input_embedding, max_steps=max_steps,
            lr_schedule=cfg.lr_schedule, lr_min=cfg.lr_min, schedule_horizon=horizon,
            final_validation=cfg.final_validation, train_probe_loader=probe_loader,
            probe_exact_ce=probe_exact)
    train_seconds = time.time() - t0
    model = to_cpu_for_analysis(rec.model)

    # --- score ------------------------------------------------------------------
    # fresh windows of the native length, as many as the test split has rows (user, 2026-10-03)
    _, fresh = draw(cfg, process, arm, len(test_indices), spec.raw_window - 1, cfg.data_seed + 1000)
    fresh_loader = tud.DataLoader(SequenceDataset(fresh), batch_size=spec.batch_size, shuffle=False)
    _, seed_tokens = draw(cfg, process, arm, 1, cfg.generation_context + 50, cfg.random_seed + 2000)
    cseqs, common_id = common_rows(raw, spans, arm)
    common = {"loader": tud.DataLoader(SequenceDataset(cseqs), batch_size=32, shuffle=False),
              "seqs": cseqs, "score_start": COMMON_EVAL["score_start"], "manifest_id": common_id}
    res = analyse_arm(model, rec, gen, ds, test_indices, test_loader, ana_loader,
                      fresh_loader=fresh_loader, seed_tokens=seed_tokens[0],
                      min_pos=cfg.state_min_pos, total_run=cfg.trans_total_run,
                      conv_tol=cfg.conv_tol, full_tol=cfg.full_tol,
                      rng=np.random.default_rng(cfg.random_seed),
                      analysis_start=spec.score_start, generation_context=cfg.generation_context,
                      fresh_seqs=fresh, common=common)

    m = res["metrics"]
    shift_offsets = list(train_loader.sampler.offsets) if spec.shift else []
    data_manifest.update({
        "raw_sha1": raw_sha1, "raw_path": raw_path, "raw_meta": raw_meta, "total_tokens": int(len(raw)),
        "splits": {k: [int(lo), int(hi)] for k, (lo, hi) in spans.items()},
        "view_sha1": sha1(ds.seqs)[:12], "common_manifest_id": common_id,
        "train_offsets": shift_offsets,
        "train_rows_per_epoch": len(train_loader.sampler) if spec.shift else int(len(split_indices["train"])),
    })
    res.update({
        "arch": arch, "process": process, "tag": tag, "arm": arm,
        "process_params": params, "burn_in": burn_in, "budget_below_k": bool(K < gen.n_causal),
        "cfg": {**asdict(cfg), "token_size": V, "n_states": K, "state_dim": S,
                "usage_beta_resolved": beta, "n_layers": n_layers, "accelerator": accelerator,
                "tau": a["tau"], "max_epochs": a["max_epochs"], "gumbel": a["gumbel"], "d_model": d_model,
                "usage_scope_resolved": usage_scope, "max_steps_resolved": max_steps},
        "model_class": type(model).__name__, "hparams": dict(model.hparams),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": train_seconds, "data_sha1": raw_sha1[:12],
        "test_indices": test_indices,
        "run_config": asdict(cfg),          # the RunConfig as given (config_collision)
        "schema_version": 2, "data_mode": "continuous",
        "view": {
            "raw_window": spec.raw_window,
            "input_len": spec.raw_window - 1,
            "stride": spec.stride,
            "eval_stride": spec.stride_for("val"),
            "batch_size": spec.batch_size,
            "score_start": spec.score_start,
            "generation_context": cfg.generation_context,
            "random_shift": spec.shift,
            "target_multiplicity": max(1, (spec.raw_window - 1 - spec.score_start) // spec.stride),
            "max_len": max_len,
        },
        "data_manifest": data_manifest,
        "scoring": {"score_start": spec.score_start, "usage_scope": usage_scope, "usage_beta": beta,
                    "analysis_start": spec.score_start, "common": dict(COMMON_EVAL, manifest_id=common_id)},
        "regime": {
            "gumbel": bool(a["gumbel"]),         # this architecture's resolved sampler
            "tau": a["tau"],
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "lr_schedule": cfg.lr_schedule,
            "lr_min": cfg.lr_min,
            "gradient_clip_val": cfg.gradient_clip_val,
            "d_model": d_model,
            "input_map": "learned" if cfg.learn_input_embedding else "frozen",
            "update_budget": int(rec.update_budget),
            "actual_updates": int(rec.actual_steps),
            "schedule_horizon": int(rec.schedule_horizon),
            "tau_at_final_update": float(rec.tau_at_final_update),
            "lr_at_final_update": float(rec.lr_last_used),
            "scored_token_exposure": int(np.sum(rec.step_scored_tokens)),
            "selected_on_separate_validation": True,
            "final_validation": bool(cfg.final_validation),
            "explicit_overrides": sorted(explicit),
        },
        "collapse": {                     # section 8, test 15: <= min(2, k-1) states at the selected checkpoint
            "states_at_first_val": m.get("states_at_first_val"),
            "states_at_selected": m.get("states_at_selected"),
            "flagged": m.get("collapse_flagged"),
            "threshold": m.get("collapse_threshold"),
            "first_val_above_collapse": m.get("first_val_above_collapse"),
        },
        "overfitting": {                  # section 8, test 17
            "probe_rows": int(len(probe_rows)), "probe_exact_ce": probe_exact,
            "min_excess": m.get("train_probe_min_excess"), "final_excess": m.get("train_probe_final_excess"),
            "flagged": m.get("overfit_flagged"),
        },
    })
    say(f"[{arch} | {tag} | {arm} | continuous] {'FULL' if m['full'] else 'not full'} "
        f"(raw: {'FULL' if m['full_raw'] else 'not full'}) | merged {m['k_used']} -> {m['merged_k']} states, "
        f"S-C {m['merged_S_minus_C']:+.3f} | discovered {m['discovered']}/{m['true_k']} | "
        f"CE-exact test {m['gap_exact']:+.4f}, common {m['common_gap_exact']:+.4f}, fresh {m['fresh_gap_exact']:+.4f} | "
        f"det {m['determinism']:.3f} | best {m['best_step']}/{rec.actual_steps} | states at best "
        f"{m.get('states_at_selected')}{' COLLAPSED' if m.get('collapse_flagged') else ''} | train-probe excess "
        f"min {m.get('train_probe_min_excess', float('nan')):+.4f}{' OVERFIT' if m.get('overfit_flagged') else ''} | "
        f"{train_seconds / 60:.1f} min")
    return res


CHECKS_RESULTS = os.path.join(HERE, "checks", "results")     # smoke runs, cross-checks, diagnostics


def run_stem(cfg, arch, process, arm):
    return f"{arch}_{process}_{arm}_s{cfg.random_seed}"


def save_run(res, run_dir, stem):
    """<stem>.pkl (everything) and <stem>.json (config + headline metrics; for a continuous
    run also its view, data manifest summary, scoring, regime, collapse and overfitting
    records -- window starts as summaries, the arrays stay in the pickle)."""
    save_pkl(res, os.path.join(run_dir, f"{stem}.pkl"))
    known = {f.name for f in fields(RunConfig)}
    head = {"arch": res["arch"], "process": res["process"], "arm": res["arm"], "tag": res["tag"],
            "model_class": res["model_class"], "n_params": res["n_params"], "data_sha1": res["data_sha1"],
            "train_minutes": res["train_seconds"] / 60, "cfg": res["cfg"],
            # the RunConfig as given, for the collision check (config_collision)
            "run_config": res.get("run_config", {k: v for k, v in res["cfg"].items()
                                                 if k in known and k not in PER_RUN_REWRITES}),
            "metrics": {k: (v if not isinstance(v, (np.floating, np.integer, np.bool_)) else v.item())
                        for k, v in res["metrics"].items()}}
    for key in ("schema_version", "data_mode", "view", "data_manifest", "scoring", "regime", "collapse",
                "overfitting"):
        if key in res:
            head[key] = _jsonable(res[key])
    with open(os.path.join(run_dir, f"{stem}.json"), "w") as f:
        json.dump(head, f, indent=2, default=str)


def theory_inputs(r):
    """(process params, burn-in, horizon) of a saved run -- rebuilt from its config
    for runs saved before the params were stored.  The horizon is the length the model
    read: the view's input length for a continuous run, seq_len for a legacy one."""
    horizon = r.get("view", {}).get("input_len", r["cfg"]["seq_len"])
    if "process_params" in r:
        return r["process_params"], r["burn_in"], horizon
    known = {f.name for f in fields(RunConfig)}
    cfg = RunConfig(**{k: v for k, v in r["cfg"].items() if k in known})
    params, burn_in, _ = process_params(cfg, r["process"])
    return params, burn_in, cfg.seq_len


VOLATILE_FIELDS = {"progress_every", "explicit_overrides"}     # change no result, only logs/provenance
PER_RUN_REWRITES = {"tau", "max_epochs", "d_model"}             # res["cfg"] holds the per-architecture value


def config_collision(run_dir, stem, cfg):
    """
    {field: (saved, new)} where the run saved as <stem> in run_dir used a different
    configuration, else None.  Only fields both know are compared, so a directory written
    before a field existed is still matched on what it did record; the per-architecture
    rewrites stored in an old result's "cfg" (tau, max_epochs) are skipped.
    """
    path = os.path.join(run_dir, f"{stem}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        head = json.load(f)
    mine = json.loads(json.dumps(asdict(cfg), default=str))
    saved = head.get("run_config")
    if saved is None:
        saved = {k: v for k, v in head.get("cfg", {}).items() if k not in PER_RUN_REWRITES}
    diff = {k: (saved[k], mine[k]) for k in saved
            if k in mine and k not in VOLATILE_FIELDS and saved[k] != mine[k]}
    return diff or None


def experiment_key(r):
    """What makes runs one comparison: the process and its parameters, the data and split,
    the seed, and the evaluation profile.  Training lengths are NOT in it, so a transformer
    and a GRU of one experiment are never split apart by their different views."""
    params, burn_in, _ = theory_inputs(r)
    if r.get("view") is not None:
        dm = r.get("data_manifest", {})
        data = (dm.get("raw_sha1", r["data_sha1"]), json.dumps(dm.get("splits"), sort_keys=True))
        manifest = r["metrics"].get("common_manifest_id", "native")
    else:
        data = (r["data_sha1"], r["cfg"].get("num_samples"), r["cfg"].get("seq_len"), r["cfg"].get("test_ratio"))
        manifest = "legacy"
    return (r["process"], json.dumps(params, sort_keys=True, default=str), burn_in, data,
            r["cfg"]["random_seed"], manifest)


def redraw(run_dir: str, fig_dir: str):
    """Every figure from the pickles in run_dir -> fig_dir (layout in the module docstring),
    each compared with the process's theory from HMM_processes (theory.process_theory)."""
    import matplotlib
    matplotlib.use("Agg")
    import plotting
    from theory import process_theory

    results = [load_pkl(p) for p in sorted(glob.glob(os.path.join(run_dir, "*.pkl")))]
    if not results:
        print(f"no results in {run_dir}")
        return []
    from extraction import add_merged_metrics
    for r in results:                  # runs saved before the merge was scored get it here (idempotent)
        add_merged_metrics(r, r["cfg"].get("full_tol", 0.05))
    # Theory is cached per (process, parameters, burn-in, horizon), and looked up at each
    # run's own horizon; the comparison groups come from experiment_key, which has no horizon.
    theories = {}

    def theory(r, horizon=None):
        params, burn_in, h = theory_inputs(r)
        key = (r["process"], json.dumps(params, sort_keys=True, default=str), burn_in, horizon or h)
        if key not in theories:
            theories[key] = process_theory(r["process"], params, burn_in, horizon or h)
        return theories[key]

    paths = []
    for r in results:
        paths += plotting.save_run_figures(r, fig_dir)
    groups, by_process = {}, {}
    for r in results:
        key = experiment_key(r)
        groups.setdefault((r["arch"], key), {})[r["arm"]] = r
        by_process.setdefault(key, {}).setdefault(r["arch"], {})[r["arm"]] = r
    for (arch, key), runs in groups.items():
        paths += plotting.save_group_figures(runs, fig_dir, theory(next(iter(runs.values()))))
    for key, runs_by_arch in by_process.items():
        everyone = [r for runs in runs_by_arch.values() for r in runs.values()]
        horizon = max(theory_inputs(r)[2] for r in everyone)       # the longest view: the full floor
        seed = everyone[0]["cfg"]["random_seed"]
        paths += plotting.save_process_figures(runs_by_arch, theory(everyone[0], horizon),
                                               everyone[0]["tag"], fig_dir, seed)
    for seed in sorted({r["cfg"]["random_seed"] for r in results}):
        paths.append(plotting.save_scorecard([r for r in results if r["cfg"]["random_seed"] == seed],
                                             fig_dir, name=f"scorecard_s{seed}"))
    print(f"{len(paths)} figures -> {fig_dir}")
    return paths


def parse_params(items, processes):
    """
    ['p=0.3', 'even.q=0.1', 'probs=[0.5,0.5]'] -> {process: {key: value}}.  A bare
    key applies to every process of the run, 'process.key' to that one only; values
    are read as JSON when they parse (numbers, lists), as strings otherwise.
    """
    out = {p: {} for p in processes}
    for item in items or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"--params expects key=value, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target, _, key = key.rpartition(".")
        for p in ([target] if target else processes):
            out.setdefault(p, {})[key] = value
    return {p: kv for p, kv in out.items() if kv}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", nargs="+", default=["transformer", "gru_feedback"],
                    choices=["transformer", "gru_feedback", "gru"],
                    help="default: the two main models; 'gru' is the read-out GRU kept for comparison")
    ap.add_argument("--process", nargs="+", default=["coin", "flower"], choices=sorted(PROCESS),
                    help="any process registered in HMM_processes.PROCESS")
    ap.add_argument("--params", nargs="*", default=[], metavar="[PROCESS.]KEY=VALUE",
                    help="process parameters, e.g. p=0.3  even.p=0.3  'probs=[0.5,0.3,0.2]'  burn_in=100 "
                         "(defaults: RunConfig for coin/flower, DEFAULT_PARAMS otherwise). "
                         "Use one --name per parameter setting: runs are named by process only.")
    ap.add_argument("--arm", nargs="+", default=["forward", "backward"], choices=["forward", "backward"])
    ap.add_argument("--tau", default=None, metavar="SPEC",
                    help="temperature schedule for EVERY architecture: a float, 'const:X', 'geom:A:B', or "
                         "'geomhold:A:B[:DESCENT]' (descend A -> B over the first DESCENT fraction of the "
                         "run, default 0.5, then hold B for the rest) -- see tau_schedue.py.  Defaults: "
                         "GRUs geom:5:0.5, transformer geomhold:5:0.5:0.8")
    ap.add_argument("--tau-transformer", default=None, metavar="SPEC", help="the transformer's schedule only")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="training length for EVERY architecture (default 1200 each = 30,000 updates; "
                         "the schedule rescales with it)")
    ap.add_argument("--max-epochs-transformer", type=int, default=None, help="the transformer's length only")
    ap.add_argument("--num-states-mult", type=int, default=None,
                    help="state budget K = this x V (RunConfig default 4); raise it when a process has "
                         "more causal states than 4V, e.g. a binary process with k > 8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="default",
                    help="experiment name: everything goes to results/<name>/ (smoke: checks/results/smoke/<name>/)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"), help="root of real results")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 64 sequences (continuous: 40 updates on 20,000 tokens), into "
                         "checks/results/smoke/<name>/ -- never into results/")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    # --- the continuous data mode (SHARED_TRAJECTORY_EMBEDDING_WARMUP_PLAN.md) ---------------
    # Every flag below defaults to None = "not given": the data mode's default applies, and a
    # flag that is given is recorded in cfg.explicit_overrides and wins over CONTINUOUS_REGIME.
    ap.add_argument("--data-mode", choices=["legacy", "continuous"], default=None,
                    help="legacy (default): independent draws, every output scored; continuous: one "
                         "trajectory split 80/10/10, short views, suffix scoring, one regime (module doc)")
    ap.add_argument("--total-tokens", type=int, default=None, help="continuous: trajectory length (300,000)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="exact optimiser-update budget (continuous default 30,000); conflicts with --max-epochs*")
    ap.add_argument("--learn-input-embedding", action="store_true", default=None,
                    help="learn the input map, for all three architectures together (an ablation)")
    ap.add_argument("--usage-scope", choices=["all", "scored"], default=None,
                    help="positions the usage penalty averages over (legacy all, continuous scored)")
    ap.add_argument("--usage-beta", type=float, default=None,
                    help="usage-penalty coefficient (legacy 1/(batch x seq_len), continuous 1/9600 fixed)")
    # training regime: default None means "not given", so the data mode's default applies
    ap.add_argument("--gumbel-transformer", action=argparse.BooleanOptionalAction, default=None,
                    help="Gumbel-ST sampling for the transformer's head (legacy default off, continuous on)")
    ap.add_argument("--gumbel-gru", action=argparse.BooleanOptionalAction, default=None,
                    help="Gumbel-ST sampling for both GRUs' head (legacy default off, continuous on)")
    ap.add_argument("--weight-decay", type=float, default=None, help="AdamW decay (legacy 0.01, continuous 5e-4)")
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default=None,
                    help="learning rate (legacy constant, continuous cosine over the schedule horizon)")
    ap.add_argument("--lr-min", type=float, default=None, help="the cosine schedule's floor (1e-5)")
    ap.add_argument("--gradient-clip-val", type=float, default=None,
                    help="gradient-norm clipping (legacy off, continuous 1.0)")
    ap.add_argument("--schedule-horizon-mult", type=float, default=None,
                    help="tau and learning-rate schedules indexed to this x the budget (legacy 1, continuous 2)")
    ap.add_argument("--gru-d-model", type=int, default=None, help="continuous: both GRUs' width (32)")
    ap.add_argument("--transformer-d-model", type=int, default=None, help="continuous: the transformer's width (16)")
    ap.add_argument("--random-shift", action=argparse.BooleanOptionalAction, default=None,
                    help="continuous: re-cut the training rows at a random offset every epoch (default on)")
    ap.add_argument("--final-validation", action=argparse.BooleanOptionalAction, default=None,
                    help="validate the final step when the cadence misses it (default on; off reproduces a "
                         "historical cadence exactly)")
    for prefix, d in (("transformer", (301, 301, 32, 20)), ("gru", (30, 1, 128, 19))):
        for field_, dflt in zip(("raw-window", "stride", "batch-size", "score-start"), d):
            ap.add_argument(f"--{prefix}-{field_}", type=int, default=None,
                            help=f"continuous: the {prefix}'s {field_.replace('-', ' ')} ({dflt})")
    ap.add_argument("--gru-eval-stride", type=int, default=None,
                    help="continuous: the GRUs' validation/test stride (10)")
    ap.add_argument("--train-fraction", type=float, default=None, help="continuous: training split (0.8)")
    ap.add_argument("--val-fraction", type=float, default=None, help="continuous: validation split (0.1)")
    ap.add_argument("--generation-context", type=int, default=None,
                    help="continuous: context of the free-running transition extraction (300)")
    args = ap.parse_args(argv)

    root = os.path.join(CHECKS_RESULTS, "smoke") if args.smoke else args.out
    out = os.path.join(root, args.name)
    run_dir, fig_dir = mkdir(os.path.join(out, "runs")), mkdir(os.path.join(out, "figures"))
    if args.plots_only:
        redraw(run_dir, fig_dir)
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    # Lightning reports through `logging`, whose handlers were bound to stderr at
    # import time, so training.quiet()'s redirect does not reach it.
    import logging
    for name in ("lightning", "lightning.pytorch", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)

    cfg = RunConfig(random_seed=args.seed, process_params=parse_params(args.params, args.process))
    explicit = set()
    if args.num_states_mult is not None:
        cfg = replace(cfg, num_states_mult=args.num_states_mult)
        explicit.add("num_states_mult")
    from tau_schedue import parse_tau
    for spec in (args.tau, args.tau_transformer):
        if spec is not None:
            parse_tau(spec)                       # reject a bad spec now, not a minute into the first fit
    if args.tau is not None:
        cfg = replace(cfg, tau=args.tau, tau_transformer=args.tau)
        explicit |= {"tau", "tau_transformer"}    # --tau sets both schedules
    if args.tau_transformer is not None:
        cfg = replace(cfg, tau_transformer=args.tau_transformer)
        explicit.add("tau_transformer")
    if args.max_epochs is not None:
        cfg = replace(cfg, max_epochs=args.max_epochs, max_epochs_transformer=args.max_epochs)
        explicit |= {"max_epochs", "max_epochs_transformer"}
    if args.max_epochs_transformer is not None:
        cfg = replace(cfg, max_epochs_transformer=args.max_epochs_transformer)
        explicit.add("max_epochs_transformer")
    FIELD_FLAGS = ("data_mode", "total_tokens", "max_steps", "learn_input_embedding", "usage_scope",
                   "usage_beta", "final_validation", "gru_eval_stride", "train_fraction", "val_fraction",
                   "generation_context",
                   *(f"{p}_{f}" for p in ("transformer", "gru")
                     for f in ("raw_window", "stride", "batch_size", "score_start")))
    REGIME_FLAGS = ("gumbel_transformer", "gumbel_gru", "weight_decay", "lr_schedule",
                    "lr_min", "gradient_clip_val", "schedule_horizon_mult",
                    "gru_d_model", "transformer_d_model", "random_shift")
    given = {k: getattr(args, k) for k in FIELD_FLAGS + REGIME_FLAGS if getattr(args, k) is not None}
    cfg = replace(cfg, **given)
    explicit |= set(given)
    if "max_steps" in explicit and explicit & {"max_epochs", "max_epochs_transformer"}:
        ap.error("--max-steps conflicts with --max-epochs / --max-epochs-transformer: give one budget")
    cfg = replace(cfg, explicit_overrides=tuple(sorted(explicit)))
    cfg = resolve_regime(cfg, explicit)      # after the --tau replacements above
    if args.smoke:
        if cfg.data_mode == "continuous":
            # a trajectory long enough to populate every split, and a small verified budget
            cfg = replace(cfg, total_tokens=20_000, max_steps=40, trans_total_run=200)
        else:
            cfg = replace(cfg, max_epochs=2, max_epochs_transformer=2, num_samples=64, fresh_samples=16,
                          trans_total_run=200)
    # never silently overwrite a different experimental configuration
    clashes = {stem: diff for stem in (run_stem(cfg, a_, p, arm) for p in args.process for a_ in args.arch
                                       for arm in args.arm)
               if (diff := config_collision(run_dir, stem, cfg))}
    if clashes:
        lines = [f"  {stem}: " + ", ".join(f"{k} {v[0]!r} -> {v[1]!r}" for k, v in diff.items())
                 for stem, diff in clashes.items()]
        ap.error(f"{run_dir} already holds runs with a different configuration:\n" + "\n".join(lines)
                 + "\nuse another --name (or remove those runs)")
    for process in args.process:
        for arch in args.arch:
            for arm in args.arm:
                res = run_arm(cfg, arch, process, arm, data_dir=os.path.join(out, "data"))
                save_run(res, run_dir, run_stem(cfg, arch, process, arm))
                cleanup()
    if not args.no_plots:
        redraw(run_dir, fig_dir)


if __name__ == "__main__":
    sys.exit(main())
