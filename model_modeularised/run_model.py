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
    python run_model.py --name det --no-gumbel-gru                    a named variant of the config
    python run_model.py --name default --plots-only                   redraw every figure from the pickles
    scripts/launch_grid.sh default                                    the whole grid in parallel, one log per run
    python run_model.py --smoke                                       2 epochs on 64 sequences -> checks/results/smoke

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

from extraction import analyse_arm
from HMM_processes import PROCESS
from process_generator import SequenceDataset, generator, split_loader
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
    num_states_mult: int = 2           # K = 2V (study5 round 3)
    state_dim_mult: int = 1            # S = V
    tau: str = "geom:5:0.5"            # the GRUs (tau_schedue.py)
    # The transformer's schedule and length (user's standard, 2026-09-25): descend
    # 5 -> 0.5 over the first 80% of 1000 epochs (800), then hold 0.5 for the last 200
    # (study5's `geomhold`; the 4th number is the DESCENT fraction).
    tau_transformer: str = "geomhold:5:0.5:0.8"
    usage_beta: Optional[float] = None # None -> 1 / (batch_size * seq_len)

    # --- training ------------------------------------------------------------
    max_epochs: int = 800              # the GRUs
    max_epochs_transformer: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    accelerator_transformer: str = "auto"   # MPS on a Mac, as study5
    accelerator_gru: str = "cpu"            # both GRUs; measured ~4x faster than MPS for nn.GRU here
    val_every_n_steps: int = 25
    restore_best: bool = True
    gradient_clip_val: Optional[float] = None
    # Gumbel-ST while training (evaluation is always the deterministic argmax).  The
    # GRUs need it: with the deterministic head a state that loses early is never
    # chosen again, so merged states stay merged (checks/results/crosscheck).  The
    # transformer keeps the pipeline's deterministic head.
    gumbel_transformer: bool = False
    gumbel_gru: bool = True                  # gru_feedback and gru
    progress_every: int = 2500               # a progress line every this many steps (0 = none)

    # --- extraction ----------------------------------------------------------
    fresh_samples: int = 100
    ana_batch: int = 32
    state_min_pos: int = 5
    conv_tol: float = 0.10
    full_tol: float = 0.05
    trans_total_run: int = 5000

    random_seed: int = 0               # split + initialisation + generation rng


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
    return {
        "n_layers": {"transformer": cfg.n_layers_transformer, "gru": cfg.n_layers_gru, "gru_feedback": 1}[arch],
        # the feedback GRU's state vector IS its hidden state, so S = d_model there
        "state_dim": cfg.d_model if arch == "gru_feedback" else cfg.state_dim_mult * V,
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


def run_arm(cfg: RunConfig, arch: str, process: str, arm: str, verbose: bool = True) -> dict:
    """Train one architecture on one arm of one process and score it -> result dict."""
    if arm not in ("forward", "backward"):
        raise ValueError(f"arm must be 'forward' or 'backward', got {arm!r}")
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
            token_size=V, d_model=cfg.d_model, max_len=ds.seq_len, max_epochs=a["max_epochs"],
            lr=cfg.learning_rate, n_layers=n_layers, weight_decay=cfg.weight_decay, mode="forward",
            accelerator=accelerator, val_every_n_steps=cfg.val_every_n_steps,
            n_states=K, state_dim=S, tau=a["tau"], usage_beta=beta,
            restore_best=cfg.restore_best, gradient_clip_val=cfg.gradient_clip_val,
            progress_every=cfg.progress_every if verbose else 0, gumbel=a["gumbel"])
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
    })
    m = res["metrics"]
    say(f"[{arch} | {tag} | {arm}] {'FULL' if m['full'] else 'not full'} | discovered "
        f"{m['discovered']}/{m['true_k']} (emission TV {m['emission_tv']:.3f}), used {m['k_used']}/{m['K']} | "
        f"S-C {m['S_minus_C']:+.3f} | "
        f"T_err {m['transition_max_err']:.3f} | gap {m['gap']:+.4f} (exact {m['gap_exact']:+.4f}, "
        f"fresh {m['fresh_gap']:+.4f}) | det {m['determinism']:.3f} | best {m['best_step']}/{m['n_steps']} | "
        f"max|g| {m['max_grad_norm']:.2f} | {train_seconds / 60:.1f} min")
    return res


CHECKS_RESULTS = os.path.join(HERE, "checks", "results")     # smoke runs, cross-checks, diagnostics


def run_stem(cfg, arch, process, arm):
    return f"{arch}_{process}_{arm}_s{cfg.random_seed}"


def save_run(res, run_dir, stem):
    """<stem>.pkl (everything) and <stem>.json (config + headline metrics)."""
    save_pkl(res, os.path.join(run_dir, f"{stem}.pkl"))
    head = {"arch": res["arch"], "process": res["process"], "arm": res["arm"], "tag": res["tag"],
            "model_class": res["model_class"], "n_params": res["n_params"], "data_sha1": res["data_sha1"],
            "train_minutes": res["train_seconds"] / 60, "cfg": res["cfg"],
            "metrics": {k: (v if not isinstance(v, (np.floating, np.integer, np.bool_)) else v.item())
                        for k, v in res["metrics"].items()}}
    with open(os.path.join(run_dir, f"{stem}.json"), "w") as f:
        json.dump(head, f, indent=2, default=str)


def theory_inputs(r):
    """(process params, burn-in, seq_len) of a saved run -- rebuilt from its config
    for runs saved before the params were stored."""
    if "process_params" in r:
        return r["process_params"], r["burn_in"], r["cfg"]["seq_len"]
    known = {f.name for f in fields(RunConfig)}
    cfg = RunConfig(**{k: v for k, v in r["cfg"].items() if k in known})
    params, burn_in, _ = process_params(cfg, r["process"])
    return params, burn_in, cfg.seq_len


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
    theories, key_of = {}, {}
    for i, r in enumerate(results):
        params, burn_in, horizon = theory_inputs(r)
        key = (r["process"], json.dumps(params, sort_keys=True, default=str), burn_in, horizon)
        if key not in theories:
            theories[key] = process_theory(r["process"], params, burn_in, horizon)
        key_of[i] = key
    paths = []
    for r in results:
        paths += plotting.save_run_figures(r, fig_dir)
    groups, by_process = {}, {}
    for i, r in enumerate(results):
        seed = r["cfg"]["random_seed"]
        groups.setdefault((r["arch"], key_of[i], seed), {})[r["arm"]] = r
        by_process.setdefault((key_of[i], seed), {}).setdefault(r["arch"], {})[r["arm"]] = r
    for (arch, key, seed), runs in groups.items():
        paths += plotting.save_group_figures(runs, fig_dir, theories[key])
    for (key, seed), runs_by_arch in by_process.items():
        tag = next(iter(next(iter(runs_by_arch.values())).values()))["tag"]
        paths += plotting.save_process_figures(runs_by_arch, theories[key], tag, fig_dir, seed)
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
                    help="training length for EVERY architecture (defaults: GRUs 800, transformer 1000; "
                         "the schedule rescales with it)")
    ap.add_argument("--max-epochs-transformer", type=int, default=None, help="the transformer's length only")
    ap.add_argument("--num-states-mult", type=int, default=None,
                    help="state budget K = this x V (RunConfig default 2); raise it when a process has "
                         "more causal states than 2V, e.g. a binary process with k > 4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="default",
                    help="experiment name: everything goes to results/<name>/ (smoke: checks/results/smoke/<name>/)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"), help="root of real results")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 64 sequences, into checks/results/smoke/<name>/ -- never into results/")
    ap.add_argument("--gumbel-transformer", action="store_true",
                    help="train the transformer's head with Gumbel-ST too (off by default)")
    ap.add_argument("--no-gumbel-gru", action="store_true",
                    help="train the GRUs with the deterministic head (the ablation)")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
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

    cfg = RunConfig(random_seed=args.seed, gumbel_transformer=args.gumbel_transformer,
                    gumbel_gru=not args.no_gumbel_gru,
                    process_params=parse_params(args.params, args.process))
    if args.num_states_mult is not None:
        cfg = replace(cfg, num_states_mult=args.num_states_mult)
    from tau_schedue import parse_tau
    for spec in (args.tau, args.tau_transformer):
        if spec is not None:
            parse_tau(spec)                       # reject a bad spec now, not a minute into the first fit
    if args.tau is not None:
        cfg = replace(cfg, tau=args.tau, tau_transformer=args.tau)
    if args.tau_transformer is not None:
        cfg = replace(cfg, tau_transformer=args.tau_transformer)
    if args.max_epochs is not None:
        cfg = replace(cfg, max_epochs=args.max_epochs, max_epochs_transformer=args.max_epochs)
    if args.max_epochs_transformer is not None:
        cfg = replace(cfg, max_epochs_transformer=args.max_epochs_transformer)
    if args.smoke:
        cfg = replace(cfg, max_epochs=2, max_epochs_transformer=2, num_samples=64, fresh_samples=16,
                      trans_total_run=200)
    for process in args.process:
        for arch in args.arch:
            for arm in args.arm:
                res = run_arm(cfg, arch, process, arm)
                save_run(res, run_dir, run_stem(cfg, arch, process, arm))
                cleanup()
    if not args.no_plots:
        redraw(run_dir, fig_dir)


if __name__ == "__main__":
    sys.exit(main())
