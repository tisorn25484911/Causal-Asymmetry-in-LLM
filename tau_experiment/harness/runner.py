"""
Train one discrete model under one (tau schedule, optimiser schedule) and score
it; run a whole cell of arms x seeds and write the JSON.

STANDALONE.  Imports Experimental_pipeline read-only and MODIFIES NOTHING in it.
The two things that vary are injected from outside the model class: tau through
a callback (it is a plain attribute read fresh on every forward, models.py:296)
and Adam's betas through an instance-level override of `configure_optimizers`,
which shadows the class method without touching it.

Construction order is copied from study0_pilot/tau_anneal_test.py, which in turn
copies pipeline.one_repeat, so the arm `const:1` + `adam:0.9:0.999` reproduces
the pipeline's own discrete run and doubles as a check that this harness is
faithful.

WHAT IS SCORED, and why it is not CE.  The documented failure is a STATE-COUNT
CEILING (DISCRETE_STATE_EXPERIMENTS.md 6.10): no configuration recovered more
than four distinct causal states, so cells whose true count is 5, 7 or 9 come
back merged.  The headline is therefore `k_occ` against `true_k`, with
`S_emp - C` beside it -- signed, so it penalises BOTH under-recovery (too few
states, S below C) and duplication (one causal state split across several
bottleneck states, S above C).  CE is carried only as the convergence gate the
pipeline uses it as.
"""
import json
import os
import sys
import time
from datetime import datetime

import lightning as L
import numpy as np
import torch
import torch.utils.data as tud
from torch.optim import AdamW

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "Experimental_pipeline"))

from config import CONFIG, coin_spec, flower_spec            # noqa: E402
from extraction import causal_state_report                   # noqa: E402
from models import build_model                               # noqa: E402
from processes import SequenceDataset, generate              # noqa: E402
from training import (Recorder, cleanup, diagnose_divergence, eval_ce,  # noqa: E402
                      quiet, set_seed, split_loader, to_cpu_for_analysis)

from .schedules import (BetaSchedule, StateTrace, TauSchedule,  # noqa: E402
                        parse_opt, parse_tau)

OCC_FLOOR = 0.01          # occupancy a state must hold to count as recovered


def make_spec(cfg, process, **kw):
    if process == "coin":
        return coin_spec(cfg, kw["p"], kw["q"])
    return flower_spec(cfg, kw["n"], kw["m"])


def build_loaders(spec, cfg, seed):
    """One realisation and one split per seed, shared by every arm in the cell."""
    ds = SequenceDataset(generate(spec["kind"], spec["params"],
                                  cfg["num_samples"], cfg["seq_len"],
                                  cfg["burn_in"], np.random.default_rng(seed)))
    train_loader, test_loader = split_loader(ds, cfg["batch"],
                                             cfg["test_ratio"], seed=seed)
    ana_loader = tud.DataLoader(ds, batch_size=cfg["ana_batch"], shuffle=False)
    return ds, train_loader, test_loader, ana_loader


def run_one(spec, cfg, ds, train_loader, test_loader, ana_loader,
            seed, tau_spec, opt_spec, mode):
    """Train one discrete model under one arm and score it."""
    tau_label, tau_fn = parse_tau(tau_spec)
    opt_label, opt_fn = parse_opt(opt_spec)

    set_seed(seed)
    torch.manual_seed(seed * 1000)

    model = build_model(
        "discrete", token_size=spec["num_token"], d_model=cfg["d_model"],
        max_len=ds.seq_len, lr=cfg["lr"], mode=mode, n_layers=cfg["n_layers"],
        weight_decay=cfg["weight_decay"], n_states=spec["n_states"],
        state_dim=spec["state_dim"], tau=tau_fn(0.0),
        usage_beta=spec["usage_beta"])

    # Instance-level override: shadows DiscreteCausalDecoder.configure_optimizers
    # for THIS object only.  Lightning calls it on the instance, so the betas
    # land in param_groups where BetaSchedule can keep moving them.
    b1_0, b2_0 = opt_fn(0.0)
    model.configure_optimizers = (
        lambda m=model, b=(b1_0, b2_0): AdamW(m.parameters(), lr=m.lr,
                                              weight_decay=m.weight_decay,
                                              betas=b))

    total_steps = len(train_loader) * cfg["max_epochs"]
    tsched = TauSchedule(tau_fn, total_steps)
    bsched = BetaSchedule(opt_fn, total_steps)
    strace = StateTrace()
    rec = Recorder(val_loader=test_loader,
                   val_every_n_steps=cfg["val_every_n_steps"])

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"], accelerator=cfg["accelerator"],
        devices="auto", log_every_n_steps=5,
        callbacks=[rec, tsched, bsched, strace],
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train_loader)

    to_cpu_for_analysis(model)
    ce, _ = eval_ce(model, test_loader)
    dv = diagnose_divergence(rec.step_loss)
    r = causal_state_report(model, ana_loader, min_pos=cfg["state_min_pos"])

    occ = np.asarray(r["occupancy"], dtype=float)
    true_k = spec["true_k_fw"] if mode == "forward" else spec["true_k_bw"]
    C = spec["C_plus"] if mode == "forward" else spec["C_minus"]
    ep = strace.per_epoch
    quarters = [ep[min(len(ep) - 1, int(len(ep) * f) - 1)]
                for f in (.25, .5, .75, 1.)] if ep else []

    out = dict(
        seed=seed, mode=mode, tau=tau_label, opt=opt_label,
        ce=float(ce), ce_minus_H=float(ce - spec["theory"]),
        within_tol=bool(abs(ce - spec["theory"]) <= cfg["conv_tol"]),
        diverged=bool(dv["diverged"]),
        k_used=int(r["n_states_used"]),
        k_occ=int((occ >= OCC_FLOOR).sum()),
        true_k=int(true_k),
        S_emp=float(r["S_emp"]), C=float(C),
        S_emp_minus_C=float(r["S_emp"] - C),
        h_state_given_token=float(r["h_state_given_token"]),
        occupancy_top=[round(float(v), 4) for v in np.sort(occ)[::-1][:12]],
        states_claimed_per_epoch=[int(v) for v in ep],
        states_claimed_quarters=quarters,
        tau_trace=tsched.trace[:4] + tsched.trace[-2:],
        final_tau=round(tsched.tau_at(total_steps - 1), 5),
        beta_trace=bsched.trace[:2] + bsched.trace[-1:],
    )
    del model, trainer, rec
    cleanup()
    return out


def run_cell(study, process, mode, arms, seeds, out_dir,
             epochs=None, accelerator="cpu", **pkw):
    """
    One (process, mode) cell: every arm on every seed, paired.

    `arms` is [(name, tau_spec, opt_spec), ...].  Paired means one realisation
    and one initialisation per seed, shared by every arm -- the arms differ in
    the schedule and in nothing else.
    """
    cfg = dict(CONFIG)
    if epochs is not None:
        cfg["max_epochs"] = int(epochs)
    cfg["accelerator"] = accelerator

    spec = make_spec(cfg, process, **pkw)
    true_k = spec["true_k_fw"] if mode == "forward" else spec["true_k_bw"]
    C = spec["C_plus"] if mode == "forward" else spec["C_minus"]

    print(f"\n{'='*104}")
    print(f"  STUDY {study}   {spec['tag']}  {mode}  discrete")
    print(f"  V={spec['num_token']}  K={spec['n_states']}  true_k={true_k}  "
          f"H_inf={spec['theory']:.4f}  C={C:.4f}  beta={spec['usage_beta']:.3e}")
    print(f"  {cfg['max_epochs']} epochs, lr={cfg['lr']}, wd={cfg['weight_decay']}, "
          f"{seeds} seeds, accelerator={cfg['accelerator']}")
    print(f"  {len(arms)} arms: {', '.join(a[0] for a in arms)}")
    print(f"{'='*104}\n", flush=True)

    runs, t0 = [], time.time()
    for si in range(seeds):
        seed = cfg["seed"] + si
        ds, train_loader, test_loader, ana_loader = build_loaders(spec, cfg, seed)
        for name, tau_spec, opt_spec in arms:
            ti = time.time()
            with quiet(True):
                r = run_one(spec, cfg, ds, train_loader, test_loader, ana_loader,
                            seed, tau_spec, opt_spec, mode)
            r["arm"] = name
            runs.append(r)
            print(f"  seed {seed}  {name:<22} {time.time()-ti:6.1f}s   "
                  f"CE {r['ce']:.4f} ({r['ce_minus_H']:+.4f})  "
                  f"k {r['k_occ']}/{true_k}  S_emp {r['S_emp']:.4f} "
                  f"({r['S_emp_minus_C']:+.4f})  H(s|tok) "
                  f"{r['h_state_given_token']:.4f}"
                  f"  claimed {r['states_claimed_quarters']}"
                  f"{'  [DIV]' if r['diverged'] else ''}", flush=True)
        print(flush=True)

    _summary(runs, arms, true_k, C, spec, seeds)

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{study}_{spec['tag']}_{mode}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(dict(study=study,
                       spec={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in spec.items()},
                       cfg=dict(cfg), mode=mode, seeds=seeds,
                       occ_floor=OCC_FLOOR,
                       arms=[dict(name=n, tau=t, opt=o) for n, t, o in arms],
                       runs=runs, minutes=(time.time() - t0) / 60),
                  f, indent=2, default=str)
    print(f"  {(time.time()-t0)/60:.1f} min -> {path}\n", flush=True)
    return path


def _summary(runs, arms, true_k, C, spec, seeds):
    def agg(name, key):
        v = np.array([r[key] for r in runs if r["arm"] == name], dtype=float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

    print(f"{'='*104}")
    print(f"  SUMMARY  ({seeds} seeds, paired: same data, same init; the arm is "
          f"the only difference)")
    print(f"{'='*104}")
    print(f"  {'arm':<22}{'k_occ':>12}{'S_emp':>15}{'S_emp-C':>11}"
          f"{'CE-H_inf':>16}{'H(s|tok)':>10}{'conv':>7}")
    print(f"  {'':<22}{'(true '+str(true_k)+')':>12}{'(C='+f'{C:.3f}'+')':>15}")
    print(f"  {'-'*101}")
    for name, _, _ in arms:
        sub = [r for r in runs if r["arm"] == name]
        if not sub:
            continue
        ko, kos = agg(name, "k_occ")
        se, ses = agg(name, "S_emp")
        sc, _ = agg(name, "S_emp_minus_C")
        ce, ces = agg(name, "ce_minus_H")
        hc, _ = agg(name, "h_state_given_token")
        print(f"  {name:<22}{ko:>6.2f}+-{kos:<5.2f}{se:>8.4f}+-{ses:<6.4f}"
              f"{sc:>+11.4f}{ce:>+8.4f}+-{ces:<6.4f}{hc:>10.4f}"
              f"{sum(r['within_tol'] for r in sub):>4}/{len(sub)}")
    print(f"  {'-'*101}\n", flush=True)
