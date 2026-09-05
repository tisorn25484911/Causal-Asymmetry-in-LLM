"""
The experiment itself.

    one_repeat        one realisation -> 4 trained models -> extracted states
    run_process       one process x N repeats -> repeats.pkl + F1-F4
    run_sweep         a grid of processes -> per-cell F1-F4 + the grid figure

Four models per repeat: {discrete, onehot} x {forward, backward}, all four on
the SAME realisation with the same hold-out split.  That is not incidental --
F4 puts the two estimators on one axis, and they are only comparable if they saw
the same data.

The backward arm is kept because it is where the state count differs (coin 3
against 2 forward; flower's backward count depends on how the dice merge), so it
is where an extraction method is actually stressed.  What was dropped is the
delta-CE STATISTIC, not the backward model.

CE is recorded only as a CONVERGENCE CHECK -- is this model trained enough for
its extracted states to mean anything.  A repeat outside conv_tol is excluded
from the figures and counted in the caption, never silently averaged in.
"""
import os
import time

import numpy as np
import torch
import torch.utils.data as tud

import figures as FIG
from extraction import (causal_state_report, recover_causal_states,
                        transition_matrix_extraction)
from processes import SequenceDataset, generate
from training import (cleanup, mkdir, quiet, save_pkl, save_run_config, set_seed,
                      diagnose_divergence, eval_ce, split_loader, to_cpu_for_analysis,
                      train_model)

ARMS = (("fw", "forward", "last"), ("bw", "backward", "first"))
ARCHS = ("discrete", "onehot")


# ══════════════════════════════════════════════════════════════════════════
# ONE REPEAT
# ══════════════════════════════════════════════════════════════════════════
def one_repeat(spec: dict, cfg: dict, seed: int, keep_discrete: bool = False) -> dict:
    """
    Train all four models on one realisation and extract from each.

    `keep_discrete` additionally returns the two discrete models under
    "_models", for the transition matrix.  They are ~24k parameters each, so
    holding them across every repeat costs a few MB and saves retraining the
    best repeat later.
    """
    ds = SequenceDataset(generate(spec["kind"], spec["params"],
                                  cfg["num_samples"], cfg["seq_len"],
                                  cfg["burn_in"], np.random.default_rng(seed)))
    # One split, shared by all four models: the seed makes it a pure function of
    # (N, seed) rather than of RNG position, so every model sees the same
    # hold-out set.
    train_loader, test_loader = split_loader(ds, cfg["batch"], cfg["test_ratio"],
                                             seed=seed)
    ana_loader = tud.DataLoader(ds, batch_size=cfg["ana_batch"], shuffle=False)

    out = {"seed": seed}
    models = {}
    for arch in ARCHS:
        out[arch] = {}
        for arm, mode, use_t in ARMS:
            kw = (dict(n_states=spec["n_states"], state_dim=spec["state_dim"],
                       tau=spec["tau"], usage_beta=spec["usage_beta"])
                  if arch == "discrete" else {})
            # Identical initialisation across arch/arm within a repeat.
            set_seed(seed)
            torch.manual_seed(seed * 1000)
            rec = train_model(
                train_loader, arch, val_loader=test_loader,
                num_token=spec["num_token"], d_model=cfg["d_model"],
                max_len=ds.seq_len, max_epochs=cfg["max_epochs"], lr=cfg["lr"],
                mode=mode, n_layers=cfg["n_layers"],
                weight_decay=cfg["weight_decay"], accelerator=cfg["accelerator"],
                val_every_n_steps=cfg["val_every_n_steps"], **kw)
            to_cpu_for_analysis(rec.model)

            ce, ppl = eval_ce(rec.model, test_loader)
            dv = diagnose_divergence(rec.step_loss)
            res = dict(final_ce=ce, final_ppl=ppl, diverged=bool(dv["diverged"]),
                       within_tol=bool(abs(ce - spec["theory"]) <= cfg["conv_tol"]),
                       step_loss=[float(v) for v in rec.step_loss],
                       step_at=[int(v) for v in rec.step_at],
                       val_loss=[float(v) for v in rec.step_val_loss],
                       val_at=[int(v) for v in rec.step_val_at],
                       true_k=spec[f"true_k_{arm}"],
                       C=spec["C_plus"] if arm == "fw" else spec["C_minus"])

            if arch == "discrete":
                r = causal_state_report(rec.model, ana_loader,
                                        min_pos=cfg["state_min_pos"])
                res.update(S_emp=r["S_emp"], n_states=r["n_states"],
                           n_states_used=r["n_states_used"],
                           counts=r["counts"].tolist(),
                           occupancy=r["occupancy"].tolist(),
                           emissions=r["emissions"].tolist(),
                           h_state_given_token=r["h_state_given_token"])
                if keep_discrete:
                    models[arm] = rec.model
            else:
                r = recover_causal_states(
                    rec.model, ana_loader, use_t=use_t,
                    max_batches=cfg["max_batches"], state_tol=cfg["state_tol"],
                    n_pts=cfg["n_pts"], seed=seed, metric=cfg["cluster_metric"])
                res.update(k_hat=r["k_hat"], S_hat=r["S_hat"],
                           plateau=r["plateau"], tol_plateau=r["tol_plateau"],
                           S_hat_plateau=r["S_hat_plateau"],
                           stability=r["stability"],
                           # kept for F2, dropped for all but the best repeat
                           _probs=r["probs"], _labels=r["labels_plateau"],
                           _tokens=r["tokens"])

            out[arch][arm] = res
            if not (arch == "discrete" and keep_discrete):
                del rec
            cleanup()

    # A repeat is converged FOR AN ARCHITECTURE when both of its arms are inside
    # conv_tol.  Tracked per architecture because the two can disagree, and
    # averaging a converged discrete arm with a non-converged clustered one
    # would put two different things in one figure.
    for arch in ARCHS:
        out[arch]["converged"] = bool(out[arch]["fw"]["within_tol"]
                                      and out[arch]["bw"]["within_tol"])
    if keep_discrete:
        out["_models"] = models
    return out


def best_repeat_index(runs: list, spec: dict, arch: str = "discrete") -> tuple:
    """
    (index, fell_back) of the repeat to draw F1-F3 from.

    Among repeats converged for `arch`, the one minimising
    max(|CE_fw - H|, |CE_bw - H|) -- the same two-arm distance conv_tol
    thresholds, so "best" means what "converged" means rather than introducing a
    second, competing notion.  With no converged repeat, the closest overall,
    flagged.
    """
    def err(r):
        return max(abs(r[arch]["fw"]["final_ce"] - spec["theory"]),
                   abs(r[arch]["bw"]["final_ce"] - spec["theory"]))
    idx = [i for i, r in enumerate(runs) if r[arch]["converged"]]
    fell_back = not idx
    if fell_back:
        idx = list(range(len(runs)))
    return min(idx, key=lambda i: err(runs[i])), fell_back


# ══════════════════════════════════════════════════════════════════════════
# ONE PROCESS
# ══════════════════════════════════════════════════════════════════════════
def run_process(spec: dict, cfg: dict, out_root: str, repeats: int,
                draw: bool = True, verbose: bool = True) -> dict:
    """One process x `repeats` repeats -> repeats.pkl and F1-F4."""
    out_dir = mkdir(os.path.join(out_root, spec["tag"]))
    t0 = time.time()
    if verbose:
        print(f"\n{'='*78}\n  {spec['tag']}   V={spec['num_token']} K={spec['n_states']}  "
              f"H_inf={spec['theory']:.4f}  C+={spec['C_plus']:.4f} C-={spec['C_minus']:.4f}  "
              f"true k {spec['true_k_fw']}/{spec['true_k_bw']}\n{'='*78}")

    runs = []
    for i in range(repeats):
        ti = time.time()
        with quiet(not verbose or True):
            r = one_repeat(spec, cfg, cfg["seed"] + i, keep_discrete=True)
        runs.append(r)
        if verbose:
            d, o = r["discrete"], r["onehot"]
            print(f"  [{i+1:>3}/{repeats}] seed={r['seed']:<3} {time.time()-ti:5.1f}s  "
                  f"disc CE {d['fw']['final_ce']:.4f}/{d['bw']['final_ce']:.4f} "
                  f"S_emp {d['fw']['S_emp']:.3f}/{d['bw']['S_emp']:.3f}  |  "
                  f"1hot k {o['fw']['plateau']}/{o['bw']['plateau']} "
                  f"S_hat {o['fw']['S_hat_plateau']:.3f}/{o['bw']['S_hat_plateau']:.3f}"
                  f"{'' if d['converged'] else '  [disc not conv]'}")

    best, fell_back = best_repeat_index(runs, spec)
    if verbose:
        n_conv = sum(r["discrete"]["converged"] for r in runs)
        print(f"  converged (discrete): {n_conv}/{repeats}   best repeat {best}"
              f"{'  [NO converged repeat -- closest overall]' if fell_back else ''}")

    # F3 from the best repeat's discrete models, which were held rather than
    # retrained.
    trans = {}
    for arm, _, _ in ARMS:
        model = runs[best]["_models"][arm]
        trans[arm] = transition_matrix_extraction(
            model, spec["kind"], spec["params"],
            burn_in=cfg["seq_len"] + 50, total_run=cfg["trans_total_run"],
            window_size=cfg["seq_len"], rng=np.random.default_rng(cfg["seed"]))
    if verbose:
        print(f"  transition matrices: visited "
              f"{int((trans['fw'].sum(1) > 0).sum())} fw / "
              f"{int((trans['bw'].sum(1) > 0).sum())} bw of {spec['n_states']}")

    # Drop the held models and every non-best repeat's F2 payload before saving.
    for i, r in enumerate(runs):
        r.pop("_models", None)
        if i != best:
            for arm, _, _ in ARMS:
                for k in ("_probs", "_labels", "_tokens"):
                    r["onehot"][arm].pop(k, None)

    rec = dict(spec={k: v for k, v in spec.items()},
               cfg=dict(cfg), runs=runs, repeats=repeats,
               best=best, best_fell_back=fell_back,
               transition={k: v for k, v in trans.items()},
               minutes=(time.time() - t0) / 60)
    save_pkl(rec, os.path.join(out_dir, "repeats.pkl"))
    if draw:
        FIG.draw_all(rec, out_dir)
    if verbose:
        print(f"  {rec['minutes']:.1f} min -> {out_dir}")
    return rec


# ══════════════════════════════════════════════════════════════════════════
# A GRID
# ══════════════════════════════════════════════════════════════════════════
def run_sweep(specs: list, cfg: dict, out_root: str, repeats: int,
              draw: bool = True) -> dict:
    """A grid of processes -> per-cell F1-F4 plus the grid figure."""
    mkdir(out_root)
    save_run_config(cfg, os.path.join(out_root, "run_config.json"))
    cells, t0 = {}, time.time()
    for j, spec in enumerate(specs):
        print(f"\n[cell {j+1}/{len(specs)}]", end="")
        cells[spec["tag"]] = run_process(spec, cfg, out_root, repeats, draw=draw)
        done = time.time() - t0
        print(f"  elapsed {done/60:.1f} min, ETA {done/(j+1)*(len(specs)-j-1)/60:.1f} min")
    grid = dict(cells=cells, cfg=dict(cfg), repeats=repeats)
    save_pkl(grid, os.path.join(out_root, "grid.pkl"))
    if draw:
        FIG.draw_grid(grid, out_root)
    return grid
