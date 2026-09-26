"""
Cross-check 2: the GRU before modularisation, on the SAME data.

Trains the notebook's discrete-memory GRU (Experimental_pipeline/
updated_pipeline_asymmetric_process.ipynb, ported verbatim -- model, Gumbel-ST,
windows, AdamW + cosine + clip, best-val -- in
tau_experiment/study4_shared_config/gru_model.py) at the NOTEBOOK's settings on
exactly the sequences and split run_model.py trained on, then scores it with the
same functions the modularised models are scored with.

    python checks/old_gru_same_data.py [--process coin flower] [--arm forward backward]
        -> checks/results/crosscheck/old_gru_<process>_<arm>_s<seed>.pkl

Same data, checked: the draw is regenerated from a saved run's config and its
SHA-1 must equal the one that run recorded; the held-out rows are that run's.
Windows are cut inside each sequence at stride 10 (study4's `nb_s10`: 150 epochs
~ 26k updates, the notebook's own budget was 29,850); the backward arm is the same
sequences reversed.  Needs results/<name>/runs/*.pkl from run_model.py, and
tau_experiment/study4_shared_config (git-ignored, on disk).
"""
import argparse
import glob
import hashlib
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODULAR = os.path.dirname(HERE)
STUDY4 = os.path.join(os.path.dirname(MODULAR), "tau_experiment", "study4_shared_config")

# Modular modules first: Experimental_pipeline (which study4 puts on the path) has
# its own `extraction` and `training`.  extraction imports `training` only inside
# analyse_arm, which is not used here.
sys.path.insert(0, MODULAR)
from extraction import (compare_transition_matrix, machine_state_transition,   # noqa: E402
                        theory_from_generator)
from HMM_processes import entropy_bits, make_dice                              # noqa: E402
from process_generator import generator                                        # noqa: E402

sys.path.insert(0, STUDY4)
import gru_model as nb                                                          # noqa: E402
import processes as pipeline_processes                                          # noqa: E402  (study4 put it on the path)
import torch                                                                    # noqa: E402
from torch.utils.data import DataLoader                                         # noqa: E402


def saved_run(process, arm, seed, runs_dir):
    """Any modular run of this process/arm: its config, data hash and held-out rows."""
    paths = sorted(glob.glob(os.path.join(runs_dir, f"*_{process}_{arm}_s{seed}.pkl")))
    if not paths:
        raise FileNotFoundError(f"no {runs_dir}/*_{process}_{arm}_s{seed}.pkl -- run run_model.py first")
    with open(paths[0], "rb") as f:
        r = pickle.load(f)
    return r["cfg"], r["data_sha1"], np.asarray(r["test_indices"])


def params_of(cfg, process):
    if process == "coin":
        return {"p": cfg["coin_p"], "q": cfg["coin_q"]}, cfg["coin_burn_in"]
    return ({"n": cfg["flower_n"], "m": cfg["flower_m"], "dice_seed": cfg["flower_dice_seed"]},
            cfg["flower_burn_in"])


def run(process, arm, seed=0, nb_seed=7, threads=1, runs_dir=None, max_epochs=None):
    cfg, sha1, test_idx = saved_run(process, arm, seed,
                                    runs_dir or os.path.join(MODULAR, "results", "default", "runs"))
    params, burn_in = params_of(cfg, process)

    # --- the same forward draw, verified -----------------------------------------
    gen = generator(process, cfg["num_samples"], cfg["seq_len"], params, burn_in=burn_in)
    seqs = gen.generate(np.random.default_rng(cfg["data_seed"])).copy()
    got = hashlib.sha1(np.ascontiguousarray(seqs).tobytes()).hexdigest()[:12]
    assert got == sha1, f"data differs from the modular run: {got} != {sha1}"
    train_idx = np.setdiff1d(np.arange(len(seqs)), test_idx)

    # --- the notebook's settings -------------------------------------------------
    V = int(gen.vocab_size)
    K = 4 * V                                         # notebook: num_states_mult = 4
    width = max(32, K)                                # notebook resolve_cfg: memory = encoder = max(32, K)
    g = nb.GRUConfig(num_states=K, symbol_vocab_size=V, memory_dim=width, encoder_hidden_dim=width,
                     stride=10, seed=nb_seed, threads=threads,
                     **({"max_epochs": max_epochs} if max_epochs else {}))
    spec = {"kind": process, "params": dict(params)}
    if process == "flower":
        dice = make_dice(params["n"], params["m"], params["dice_seed"])
        assert np.array_equal(dice, pipeline_processes.make_dice(params["n"], params["m"], params["dice_seed"]))
        spec["params"]["dice_probs"] = dice

    # --- nb.run_arm, unrolled so the model and its raw machine are kept -----------
    torch.set_num_threads(g.threads)
    torch.manual_seed(g.seed); np.random.seed(g.seed)
    x = seqs[:, ::-1].copy() if arm == "backward" else seqs
    tr_p, tr_f = nb.make_windows(x[train_idx], g.past_len, g.future_len, g.stride)
    te_p, te_f = nb.make_windows(x[test_idx], g.past_len, g.future_len, g.stride)
    train_loader = DataLoader(nb.WindowDataset(tr_p, tr_f), batch_size=g.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(g.seed))
    val_loader = DataLoader(nb.WindowDataset(te_p, te_f), batch_size=512, shuffle=False)
    model = nb.EmbeddedDiscreteMemoryGRU(g)
    t0 = time.time()
    model, info = nb.train_model(model, train_loader, val_loader, g)
    minutes = (time.time() - t0) / 60
    seqm = nb.sequence_metrics(model, x[test_idx], g, spec, "fw" if arm == "forward" else "bw")
    raw = nb.extract_raw_machine(model, g)
    occ = seqm["occupancy"]
    merged = nb.merged_machine_size(raw, (occ > 0).astype(float))

    # --- scored like the modular models -------------------------------------------
    view = generator(process, 1, 10, params, burn_in=burn_in)
    if arm == "backward":
        view.reverse()
    th = theory_from_generator(view)
    T_exact = machine_state_transition(raw)
    T_exact[occ == 0] = 0.0
    cmp = compare_transition_matrix(T_exact, th["T_theory"], raw["emission_probs"],
                                    th["true_machine"]["emission_probs"], weights=occ)
    discovered = th["true_k"] - len(cmp["missing"])
    tv = cmp["emission_tv"]
    seen = np.flatnonzero(np.isfinite(tv))
    S = float(seqm["S_emp"])
    res = {
        "process": process, "arm": arm, "tag": f"{process}", "nb_seed": nb_seed, "K": K, "width": width,
        "steps": info["steps"], "minutes": minutes, "best_epoch": info["best_epoch"],
        "ce": seqm["ce"], "exact": seqm["exact"], "gap": seqm["gap"],        # positions >= past_len
        "S_emp": S, "C": th["C"], "S_minus_C": S - th["C"],
        "k_used": seqm["n_states_used"], "true_k": th["true_k"], "discovered": int(discovered),
        "full": bool(discovered == th["true_k"] and abs(S - th["C"]) < 0.05),
        "emission_tv": float(np.sum(occ[seen] * tv[seen]) / max(occ[seen].sum(), 1e-12)),
        "T_err": cmp["max_error"], "h_state_given_token": seqm["h_state_given_token"],
        "live_machine_size": int((occ > 0).sum()), "merged_machine_size": int(merged),
        "raw_machine": raw, "occupancy": occ, "comparison": cmp, "data_sha1": sha1,
    }
    print(f"[old notebook GRU | {process} | {arm}] {'FULL' if res['full'] else 'not full'} | discovered "
          f"{discovered}/{th['true_k']} (emission TV {res['emission_tv']:.3f}), used {res['k_used']}/{K}, "
          f"merged machine {merged} | S-C {res['S_minus_C']:+.3f} | T_err {res['T_err']:.3f} | "
          f"CE - exact {res['gap']:+.4f} (positions >= {g.past_len}) | {info['steps']} steps, {minutes:.1f} min",
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", nargs="+", default=["coin", "flower"], choices=["coin", "flower"])
    ap.add_argument("--arm", nargs="+", default=["forward", "backward"], choices=["forward", "backward"])
    ap.add_argument("--seed", type=int, default=0, help="which modular run's data/split to reuse")
    ap.add_argument("--nb-seed", type=int, default=7, help="the old GRU's own seed (notebook: 7)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs-dir", default=None,
                    help="where the modular run pickles are (default results/default/runs)")
    ap.add_argument("--max-epochs", type=int, default=None, help="override (plumbing checks only; not saved)")
    a = ap.parse_args()
    out = os.path.join(HERE, "results", "crosscheck")
    os.makedirs(out, exist_ok=True)
    for process in a.process:
        for arm in a.arm:
            res = run(process, arm, a.seed, a.nb_seed, a.threads, a.runs_dir, a.max_epochs)
            if a.max_epochs is None:
                with open(os.path.join(out, f"old_gru_{process}_{arm}_s{a.seed}.pkl"), "wb") as f:
                    pickle.dump(res, f)


if __name__ == "__main__":
    main()
