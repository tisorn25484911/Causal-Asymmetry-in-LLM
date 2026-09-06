#!/usr/bin/env python3
"""
Does annealing the straight-through temperature high -> low rescue the backward
flower arm?

STANDALONE.  This imports Experimental_pipeline and MODIFIES NOTHING in it.  tau
is a plain float attribute read fresh on every forward (models.py:296), so a
Lightning callback can rewrite it per step without touching the model class.

WHAT IS BEING TESTED, and why the metric is not CE.  The documented backward
failure is a STATE-COUNT CEILING: DISCRETE_STATE_EXPERIMENTS.md 6.10 records
that no configuration anywhere recovered more than four distinct causal states,
so flower cells whose true backward count is 5, 7 or 9 come back merged.  The
mechanism that would explain it is gradient starvation at the state head: with
p = softmax(logits/tau), the straight-through Jacobian is
(1/tau)(diag(p) - p p^T), whose off-diagonal mass decays like exp(-gap/tau).  A
state that loses early receives exponentially little gradient and never gets
claimed.  Raising tau early spreads that gradient over every state; lowering it
later restores the sharp assignment.  That is the hypothesis.

So the headline number is DISTINCT STATES RECOVERED against true_k, with S_emp
against C and H(state|token) beside it.  CE is carried only as the convergence
check the pipeline uses it as.

ARMS.  const_1.0 is the current pipeline exactly -- same seeding, same split,
same construction order -- so at accelerator=cpu it reproduces the pipeline's
own discrete/backward result and doubles as a check that this harness is
faithful.  The two constant arms and the reversed anneal are controls: they
separate "the schedule helped" from "any tau != 1 helped" and from "the range of
tau traversed helped".
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import lightning as L
import numpy as np
import torch
import torch.utils.data as tud

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Experimental_pipeline"))

from config import CONFIG, coin_spec, flower_spec                # noqa: E402
from extraction import causal_state_report                       # noqa: E402
from models import build_model                                   # noqa: E402
from processes import SequenceDataset, generate                  # noqa: E402
from training import (Recorder, cleanup, diagnose_divergence, eval_ce,   # noqa: E402
                      quiet, set_seed, split_loader, to_cpu_for_analysis)


# ══════════════════════════════════════════════════════════════════════════
# THE ONE THING THAT IS NEW: a tau schedule, as a callback
# ══════════════════════════════════════════════════════════════════════════
class TauSchedule(L.Callback):
    """
    Rewrite `pl_module.tau` before every training batch.

    Geometric interpolation, i.e. exactly the exp(-r t) schedule of Jang et al.
    with the endpoint pinned instead of the rate guessed:

        tau(s) = tau0 * (tau1 / tau0) ** (s / (S - 1))

    tau0 == tau1 makes it a constant arm, so every arm runs through the same
    code path and the baseline is not a different branch.
    """

    def __init__(self, tau0: float, tau1: float, total_steps: int):
        super().__init__()
        self.tau0, self.tau1 = float(tau0), float(tau1)
        self.total_steps = max(2, int(total_steps))
        self.trace = []          # (step, tau) every 100 steps, for the record

    def tau_at(self, step: int) -> float:
        f = min(1.0, max(0.0, step / (self.total_steps - 1)))
        return self.tau0 * (self.tau1 / self.tau0) ** f

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        tau = self.tau_at(trainer.global_step)
        pl_module.tau = tau
        if trainer.global_step % 100 == 0:
            self.trace.append((int(trainer.global_step), round(tau, 5)))


class StateTrace(L.Callback):
    """
    Distinct states the model actually assigned during each epoch.

    This is the mechanistic read-out: if high tau de-starves the state head, the
    states are claimed EARLIER, not only more of them at the end.
    """

    def __init__(self):
        super().__init__()
        self.per_epoch = []
        self._seen = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        st = getattr(pl_module, "last_states", None)
        if st is not None:
            self._seen.update(st.detach().reshape(-1).cpu().tolist())

    def on_train_epoch_end(self, trainer, pl_module):
        self.per_epoch.append(len(self._seen))
        self._seen = set()


# ══════════════════════════════════════════════════════════════════════════
# ONE TRAINED MODEL
# ══════════════════════════════════════════════════════════════════════════
def run_one(spec, cfg, ds, train_loader, test_loader, ana_loader,
            seed, tau0, tau1, mode, occ_floor):
    """Train one discrete model under one tau schedule and score it."""
    # Identical construction and initialisation to pipeline.one_repeat, so the
    # const_1.0 arm is the pipeline's own run and not merely a lookalike.
    set_seed(seed)
    torch.manual_seed(seed * 1000)

    model = build_model(
        "discrete", token_size=spec["num_token"], d_model=cfg["d_model"],
        max_len=ds.seq_len, lr=cfg["lr"], mode=mode, n_layers=cfg["n_layers"],
        weight_decay=cfg["weight_decay"], n_states=spec["n_states"],
        state_dim=spec["state_dim"], tau=tau0, usage_beta=spec["usage_beta"])

    total_steps = len(train_loader) * cfg["max_epochs"]
    sched = TauSchedule(tau0, tau1, total_steps)
    strace = StateTrace()
    rec = Recorder(val_loader=test_loader, val_every_n_steps=cfg["val_every_n_steps"])

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"], accelerator=cfg["accelerator"],
        devices="auto", log_every_n_steps=5,
        callbacks=[rec, sched, strace],
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train_loader)

    to_cpu_for_analysis(model)
    ce, ppl = eval_ce(model, test_loader)
    dv = diagnose_divergence(rec.step_loss)
    r = causal_state_report(model, ana_loader, min_pos=cfg["state_min_pos"])

    occ = np.asarray(r["occupancy"], dtype=float)
    true_k = spec["true_k_fw"] if mode == "forward" else spec["true_k_bw"]
    C = spec["C_plus"] if mode == "forward" else spec["C_minus"]
    ep = strace.per_epoch
    quarters = [ep[min(len(ep) - 1, int(len(ep) * f) - 1)] for f in (.25, .5, .75, 1.)] if ep else []

    out = dict(
        seed=seed, tau0=tau0, tau1=tau1, mode=mode,
        ce=float(ce), ce_minus_H=float(ce - spec["theory"]),
        within_tol=bool(abs(ce - spec["theory"]) <= cfg["conv_tol"]),
        diverged=bool(dv["diverged"]),
        k_used=int(r["n_states_used"]),
        k_occ=int((occ >= occ_floor).sum()),
        true_k=int(true_k),
        S_emp=float(r["S_emp"]), C=float(C), S_emp_minus_C=float(r["S_emp"] - C),
        h_state_given_token=float(r["h_state_given_token"]),
        occupancy_top=[round(float(v), 4) for v in np.sort(occ)[::-1][:12]],
        states_claimed_quarters=quarters,
        tau_trace=sched.trace[:6] + sched.trace[-2:],
        final_tau=round(sched.tau_at(total_steps - 1), 5),
    )
    del model, trainer, rec
    cleanup()
    return out


# ══════════════════════════════════════════════════════════════════════════
# THE EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════
SCHEDULES = {
    "const_1.0":    (1.0, 1.0),     # the current pipeline, exactly
    "anneal_5_0.5": (5.0, 0.5),     # the hypothesis: high -> low
    "const_5.0":    (5.0, 5.0),     # control: high tau, no schedule
    "anneal_0.5_5": (0.5, 5.0),     # control: same range, reversed direction
}


def parse_arm(name):
    if name in SCHEDULES:
        return name, SCHEDULES[name]
    a, _, b = name.partition(":")
    if not b:
        raise argparse.ArgumentTypeError(
            f"unknown arm {name!r}; use one of {list(SCHEDULES)} or 'tau0:tau1'")
    return f"anneal_{a}_{b}", (float(a), float(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", choices=("flower", "coin"), default="flower")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--q", type=float, default=0.9)
    ap.add_argument("--mode", choices=("backward", "forward"), default="backward")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override CONFIG['max_epochs'] (default 150)")
    ap.add_argument("--arms", nargs="+", default=list(SCHEDULES),
                    help="arm names, or 'tau0:tau1' for a custom schedule")
    ap.add_argument("--accelerator", default="cpu",
                    help="cpu is exactly repeatable; MPS is not (training.py:284)")
    ap.add_argument("--occ-floor", type=float, default=0.01,
                    help="occupancy a state must hold to count as recovered")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.epochs is not None:
        cfg["max_epochs"] = args.epochs
    cfg["accelerator"] = args.accelerator

    spec = (flower_spec(cfg, args.n, args.m) if args.process == "flower"
            else coin_spec(cfg, args.p, args.q))
    arms = [parse_arm(a) for a in args.arms]
    true_k = spec["true_k_fw"] if args.mode == "forward" else spec["true_k_bw"]
    C = spec["C_plus"] if args.mode == "forward" else spec["C_minus"]

    print(f"\n{'='*100}")
    print(f"  TAU SCHEDULE TEST   {spec['tag']}  {args.mode}  discrete")
    print(f"  V={spec['num_token']}  K={spec['n_states']}  true_k={true_k}  "
          f"H_inf={spec['theory']:.4f}  C={C:.4f}  beta={spec['usage_beta']:.3e}")
    print(f"  {cfg['max_epochs']} epochs, lr={cfg['lr']}, wd={cfg['weight_decay']}, "
          f"{args.seeds} seeds, accelerator={cfg['accelerator']}")
    print(f"  arms: {', '.join(n for n, _ in arms)}")
    print(f"{'='*100}\n")

    runs, t0 = [], time.time()
    for si in range(args.seeds):
        seed = cfg["seed"] + si
        # ONE realisation and ONE split per seed, shared by every arm: the arms
        # differ in tau and in nothing else.
        ds = SequenceDataset(generate(spec["kind"], spec["params"],
                                      cfg["num_samples"], cfg["seq_len"],
                                      cfg["burn_in"], np.random.default_rng(seed)))
        train_loader, test_loader = split_loader(ds, cfg["batch"],
                                                 cfg["test_ratio"], seed=seed)
        ana_loader = tud.DataLoader(ds, batch_size=cfg["ana_batch"], shuffle=False)

        for name, (tau0, tau1) in arms:
            ti = time.time()
            with quiet(True):
                r = run_one(spec, cfg, ds, train_loader, test_loader, ana_loader,
                            seed, tau0, tau1, args.mode, args.occ_floor)
            r["arm"] = name
            runs.append(r)
            print(f"  seed {seed}  {name:<14} {time.time()-ti:6.1f}s   "
                  f"CE {r['ce']:.4f} ({r['ce_minus_H']:+.4f})  "
                  f"k {r['k_occ']}/{true_k}  S_emp {r['S_emp']:.4f} "
                  f"({r['S_emp_minus_C']:+.4f})  H(s|tok) {r['h_state_given_token']:.4f}"
                  f"  claimed {r['states_claimed_quarters']}"
                  f"{'  [DIV]' if r['diverged'] else ''}")
        print()

    # ── summary ──────────────────────────────────────────────────────────
    def agg(name, key):
        v = np.array([r[key] for r in runs if r["arm"] == name], dtype=float)
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

    print(f"{'='*100}")
    print(f"  SUMMARY  ({args.seeds} seeds, paired: same data, same init, tau is "
          f"the only difference)")
    print(f"{'='*100}")
    print(f"  {'arm':<15}{'k_occ':>12}{'k_used':>12}{'S_emp':>15}"
          f"{'S_emp-C':>11}{'CE-H_inf':>16}{'H(s|tok)':>9}{'conv':>7}")
    print(f"  {'':<15}{'(true '+str(true_k)+')':>12}{'':>12}{'(C='+f'{C:.3f}'+')':>15}"
          f"{'':>11}{'mean +- sd':>16}{'':>9}{'':>7}")
    print(f"  {'-'*97}")
    for name, _ in arms:
        sub = [r for r in runs if r["arm"] == name]
        ko, kos = agg(name, "k_occ"); ku, kus = agg(name, "k_used")
        se, ses = agg(name, "S_emp"); sc, _ = agg(name, "S_emp_minus_C")
        ce, ces = agg(name, "ce_minus_H"); hc, _ = agg(name, "h_state_given_token")
        nconv = sum(r["within_tol"] for r in sub)
        print(f"  {name:<15}{ko:>6.2f}+-{kos:<5.2f}{ku:>6.2f}+-{kus:<5.2f}"
              f"{se:>8.4f}+-{ses:<6.4f}{sc:>+11.4f}{ce:>+8.4f}+-{ces:<6.4f}"
              f"{hc:>9.4f}{nconv:>4}/{len(sub)}")
    print(f"  {'-'*97}")

    # ── the final loss gap, stated on its own ────────────────────────────
    print(f"\n  FINAL LOSS GAP TO THE ENTROPY RATE   (CE - H_inf, bits; "
          f"H_inf = {spec['theory']:.4f})")
    print(f"  {'arm':<15}{'CE':>18}{'CE - H_inf':>20}{'worst seed':>13}{'converged':>12}")
    print(f"  {'-'*78}")
    for name, _ in arms:
        sub = [r for r in runs if r["arm"] == name]
        c = np.array([r["ce"] for r in sub]); d = np.array([r["ce_minus_H"] for r in sub])
        sd = (lambda v: v.std(ddof=1) if len(v) > 1 else 0.0)
        worst = d[np.argmax(np.abs(d))]
        print(f"  {name:<15}{c.mean():>10.4f} +- {sd(c):<6.4f}"
              f"{d.mean():>+12.4f} +- {sd(d):<6.4f}{worst:>+13.4f}"
              f"{sum(r['within_tol'] for r in sub):>8}/{len(sub)}")
    print(f"  {'-'*78}")

    base = [r for r in runs if r["arm"] == arms[0][0]]
    print(f"\n  Read it this way.  k_occ is the number of states holding at least "
          f"{args.occ_floor:.0%} of\n  positions -- the quantity the ceiling in "
          f"DISCRETE_STATE_EXPERIMENTS.md 6.10 caps.  The\n  schedule helped only "
          f"if k_occ rose toward {true_k} AND S_emp moved toward C={C:.4f}\n"
          f"  while H(state|token) stayed near 0; a k_occ that rises with "
          f"H(state|token) is\n  state duplication, not state recovery (6.9).\n")

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.out, f"tau_{spec['tag']}_{args.mode}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(dict(spec={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in spec.items()},
                       cfg={k: v for k, v in cfg.items()},
                       mode=args.mode, seeds=args.seeds, occ_floor=args.occ_floor,
                       arms=dict(arms), runs=runs,
                       minutes=(time.time() - t0) / 60), f, indent=2, default=str)
    print(f"  {(time.time()-t0)/60:.1f} min -> {path}\n")


if __name__ == "__main__":
    main()
