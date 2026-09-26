"""
Diagnosis: which setting decides whether an arm recovers its causal states?

Trains one architecture on one arm of one process -- the same data and split as
run_model.py -- under the run's baseline settings and under one-knob variants of
them, and reports states found / S_emp - C / CE - exact for each.  Light scoring
(no free-running generation), so a variant costs only its training time.

    python checks/diagnose_one_knob.py --arch gru --process coin --arm backward \
        --epochs 150 --variants det det+tau_const1 det+beta0 gumbel gumbel+beta0 \
        > checks/results/diagnostics/<name>.log

'base' is RunConfig as it stands (Gumbel-ST for the GRUs since 2026-09-25); the logs
already in checks/results/diagnostics were made when 'base' meant the deterministic
head for every architecture -- 'det' reproduces that.

Knobs combine with '+'.  It prints one line per variant and writes nothing else;
keep the logs in checks/results/diagnostics/, never in results/.

A variant that recovers the states where the baseline does not isolates that
knob; if none does, look for a bug instead.  Each variant is a separate run, so
single-seed differences are indicative only -- repeat with --seeds.
"""
import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch
import torch.utils.data as tud

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from extraction import causal_state_report, compare_transition_matrix, theory_from_generator  # noqa: E402
from process_generator import SequenceDataset, split_loader                                   # noqa: E402
from run_model import RunConfig, arch_settings, draw                                           # noqa: E402
from training import eval_ce, per_position_ce, quiet, set_seed, train_model                   # noqa: E402
from extraction import exact_per_position_ce                                                    # noqa: E402

# name -> (RunConfig overrides, train_model overrides)
VARIANTS = {
    "base":       ({}, {}),
    "tau_const1": ({"tau": "const:1", "tau_transformer": "const:1"}, {}),
    "tau_rise":   ({"tau": "geom:0.5:5", "tau_transformer": "geom:0.5:5"}, {}),
    "tau_geom":   ({"tau": "geom:5:0.5", "tau_transformer": "geom:5:0.5"}, {}),
    "beta0":      ({"usage_beta": 0.0}, {}),
    "K4V":        ({"num_states_mult": 4}, {}),
    "wd0":        ({"weight_decay": 0.0}, {}),
    "d32":        ({"d_model": 32}, {}),
    "clip1":      ({"gradient_clip_val": 1.0}, {}),
    # the head's sampler, for whichever architecture is run (RunConfig's defaults:
    # Gumbel-ST for the GRUs, the deterministic argmax for the transformer)
    "gumbel":     ({"gumbel_transformer": True, "gumbel_gru": True}, {}),
    "det":        ({"gumbel_transformer": False, "gumbel_gru": False}, {}),
    "tau_old":    ({"tau": "geom:2.25:0.1", "tau_transformer": "geom:2.25:0.1"}, {}),   # the notebook GRU's endpoints
}


def overrides(variant):
    """'gumbel+K4V' -> the two knobs' RunConfig overrides together."""
    cfg_over = {}
    for part in variant.split("+"):
        if part not in VARIANTS:
            raise ValueError(f"unknown variant {part!r}; known: {sorted(VARIANTS)}")
        cfg_over.update(VARIANTS[part][0])
    return cfg_over


def one(arch, process, arm, variant, epochs, seed):
    cfg_over = overrides(variant)
    cfg = replace(RunConfig(random_seed=seed), max_epochs=epochs, max_epochs_transformer=epochs, **cfg_over)
    gen, seqs = draw(cfg, process, arm, cfg.num_samples, cfg.seq_len, cfg.data_seed)
    ds = SequenceDataset(seqs)
    tr, te = split_loader(ds, cfg.batch_size, cfg.test_ratio, seed=cfg.random_seed)
    test_idx = np.asarray(te.dataset.indices)
    V = int(gen.vocab_size)
    K = cfg.num_states_mult * V
    a = arch_settings(cfg, arch, V)
    S, n_layers = a["state_dim"], a["n_layers"]
    beta = 1.0 / (cfg.batch_size * cfg.seq_len) if cfg.usage_beta is None else float(cfg.usage_beta)
    set_seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed * 1000)
    t0 = time.time()
    with quiet(True):
        rec = train_model(tr, arch, "discrete", val_loader=te, token_size=V, d_model=cfg.d_model,
                          max_len=ds.seq_len, max_epochs=a["max_epochs"], lr=cfg.learning_rate,
                          n_layers=n_layers, weight_decay=cfg.weight_decay, accelerator="cpu",
                          val_every_n_steps=cfg.val_every_n_steps, n_states=K, state_dim=S,
                          tau=a["tau"], usage_beta=beta, restore_best=cfg.restore_best,
                          gradient_clip_val=cfg.gradient_clip_val, gumbel=a["gumbel"])
    m = rec.model.cpu()
    th = theory_from_generator(gen)
    ana = tud.DataLoader(ds, batch_size=64, shuffle=False)
    rep = causal_state_report(m, ana, min_pos=cfg.state_min_pos)
    # emission-row matching over the states the data visits (no generation)
    occ = rep["occupancy"]
    visited = np.zeros(K); visited[rep["occupied"]] = 1.0
    cmp = compare_transition_matrix(np.diag(visited), np.eye(th["true_k"]), rep["emissions"],
                                    th["true_machine"]["emission_probs"], weights=occ)
    discovered = th["true_k"] - len(cmp["missing"])
    tv = cmp["emission_tv"]; seen = np.flatnonzero(np.isfinite(tv))
    etv = float(np.sum(occ[seen] * tv[seen]) / max(occ[seen].sum(), 1e-12))
    ce = float(per_position_ce(m, te).mean())
    exact = float(exact_per_position_ce(gen, ds.seqs[test_idx]).mean())
    S_minus_C = float(rep["S_emp"]) - th["C"]
    full = discovered == th["true_k"] and abs(S_minus_C) < 0.05
    print(f"{arch:12s} {process:6s} {arm:8s} {variant:10s} seed {seed} {epochs:4d} ep | "
          f"{'FULL' if full else '    '} discovered {discovered}/{th['true_k']} (emission TV {etv:.3f}) "
          f"used {rep['n_states_used']}/{K} | S-C {S_minus_C:+.3f} | CE - exact {ce - exact:+.4f} | "
          f"best {rec.best_step}/{len(rec.step_loss)} | {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="gru", choices=["transformer", "gru", "gru_feedback"])
    ap.add_argument("--process", default="coin", choices=["coin", "flower"])
    ap.add_argument("--arm", default="backward", choices=["forward", "backward"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    help=f"knobs, combinable with '+': {', '.join(VARIANTS)}")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    import logging
    for name in ("lightning", "lightning.pytorch", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)
    for seed in a.seeds:
        for v in a.variants:
            one(a.arch, a.process, a.arm, v, a.epochs, seed)


if __name__ == "__main__":
    main()
