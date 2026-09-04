#!/usr/bin/env python
"""
State-to-state transition matrices for a finished experiment folder.

    T[i][j] = P(s_{t+1} = j | s_t = i)

read off the trained bottleneck by letting the model free-run on its own
samples -- DiscreteCausal_analysis.transition_matrix_extraction.

WHY THIS IS A SEPARATE RUNNER.  run_statistical_trj.py and
run_sweep_experiment.py keep no weights: every repeat's model is dropped at the
end of `one_repeat`, and only the metrics reach the pickle.  So the matrices
cannot be recovered from a finished folder by reading it -- the single best
repeat has to be TRAINED AGAIN.  That is what this does, and it is cheap
(one repeat per process, not the full 20-30) next to re-running the experiment.

  best repeat = among the repeats the pickle marks converged (|CE - H_inf| <=
  conv_tol in BOTH arms), the one minimising max(|CE_fw - H_inf|,
  |CE_bw - H_inf|).  Processes with no converged repeat fall back to the
  closest overall and are flagged in the output.

The retrain is seeded from the repeat's own recorded seed, so it is the same
draw -- but cfg["accelerator"]="auto" selects MPS, which configs.py notes is
not bit-reproducible, so it is a re-draw from that seed rather than the
identical model.  The achieved CE is printed next to the recorded one.

Usage
-----
    python Experimental_setup/run_transition_matrix.py 00_base 08_pqsweep 09_nmsweep
    python Experimental_setup/run_transition_matrix.py 00_base --tags traj_coin_p010_q090
    python Experimental_setup/run_transition_matrix.py 00_base --total-run 500   # smoke
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

# Both Transformer_model/ and Experimental_setup/ have to be importable; python
# only puts this script's own directory on sys.path.  Anchored on __file__ so
# the script runs from any working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs import CONFIGS
from DiscreteCausal_analysis import (plot_transition_matrix_pair,
                                     transition_matrix_extraction)
from run_statistical_trj import one_repeat, quiet
from utils import mkdir, save_pkl

_OUT_DEFAULT = os.path.join(_ROOT, "All_Results", "discrete_v2")
# Written under <folder>/, one subfolder for every experiment, so 25 sweep
# points do not land loose next to the sweep's own figures.
_SUBDIR = "state_transition"


def load_folder(path: str):
    """
    The records dict for a finished folder.

    run_statistical_trj writes all_trajectories.pkl and run_sweep_experiment
    writes all_sweep.pkl, but the two have the SAME shape --
    {tag: {spec, config, base_seed, repeats, runs, paired}} -- so one loader
    covers both and the caller never branches on which runner produced it.
    """
    for name in ("all_trajectories.pkl", "all_sweep.pkl"):
        p = os.path.join(path, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f), name
    raise FileNotFoundError(
        f"{path} holds neither all_trajectories.pkl nor all_sweep.pkl")


def load_cfg(path: str, config_name: str) -> dict:
    """
    The config the folder was actually run at.

    Starts from CONFIGS[name] and overlays run_config_<NAME>.json, which the
    runners write at start-up -- that is where command-line overrides such as
    --chunk-len live, and reading them back is the only way a re-train matches
    the original.  Keys the base config does not define are ignored.
    """
    cfg = dict(CONFIGS[config_name])
    j = os.path.join(path, f"run_config_{config_name}.json")
    if os.path.exists(j):
        with open(j) as f:
            rec = json.load(f)
        if isinstance(rec, list):
            rec = rec[0]
        for k, v in rec.items():
            if k in cfg:
                cfg[k] = v
    return cfg


def best_repeat(rec: dict):
    """
    (index, seed, err, fell_back) for the repeat to inspect.

    `err` is max(|CE_fw - H_inf|, |CE_bw - H_inf|) -- the same two-arm distance
    conv_tol thresholds, so "best" here means the same thing "converged" means
    everywhere else in the repo rather than a second, competing notion.
    """
    runs   = rec["runs"]
    paired = rec.get("paired") or {}
    theory = paired.get("theory", rec["spec"].get("theory"))
    conv   = paired.get("converged")

    def err(r):
        return max(abs(r["fw"]["final_ce"] - theory),
                   abs(r["bw"]["final_ce"] - theory))

    idx = [i for i in range(len(runs)) if conv is not None and conv[i]]
    fell_back = not idx
    if fell_back:
        idx = list(range(len(runs)))
    i = min(idx, key=lambda i: err(runs[i]))
    return i, runs[i]["seed"], err(runs[i]), fell_back


def process_args(spec: dict):
    """(process, n_p, m_q, dice_probs) for transition_matrix_extraction."""
    if spec["kind"] == "coin":
        return "Coin", spec["p"], spec["q"], None
    return "Flower", spec["n"], spec["m"], spec.get("dice_probs")


def one_folder(folder: str, out_root: str, total_run: int, tags=None) -> dict:
    path = os.path.join(out_root, folder)
    records, src = load_folder(path)
    out_dir = mkdir(os.path.join(path, _SUBDIR))
    print(f"\n{'='*78}\n  {folder}   ({src}, {len(records)} processes)\n{'='*78}")

    results = {}
    for tag, rec in records.items():
        if tags and tag not in tags:
            continue
        spec = rec["spec"]
        cfg  = load_cfg(path, rec["config"])
        i, seed, err, fell_back = best_repeat(rec)
        note = "  [NO converged repeat -- closest overall]" if fell_back else ""
        print(f"\n  {tag}   best repeat {i} (seed {seed})  "
              f"max|CE-H|={err:.4f}{note}")

        t0 = time.time()
        with quiet(True):
            run = one_repeat(spec, cfg, seed, keep_models=True)
        chunk = run["_chunk"]
        rec_ce = (rec["runs"][i]["fw"]["final_ce"], rec["runs"][i]["bw"]["final_ce"])
        got_ce = (run["fw"]["final_ce"], run["bw"]["final_ce"])
        print(f"    retrained in {time.time()-t0:6.1f}s   "
              f"CE fw/bw recorded {rec_ce[0]:.4f}/{rec_ce[1]:.4f}  "
              f"-> got {got_ce[0]:.4f}/{got_ce[1]:.4f}")

        proc, n_p, m_q, dice = process_args(spec)
        mats = {}
        for arm in ("fw", "bw"):
            model = run["_models"][arm]
            t1 = time.time()
            mats[arm] = transition_matrix_extraction(
                model, proc, n_p, m_q,
                # Enough burn-in to fill the window in both processes: Flower
                # emits two tokens per cycle, Coin one.
                burn_in=chunk + 50, total_run=total_run,
                window_size=chunk, dice_probs=dice)
            used = int((mats[arm].sum(axis=1) > 0).sum())
            print(f"    {arm}: K={mats[arm].shape[0]}  states visited={used}  "
                  f"({time.time()-t1:.1f}s)")

        p = plot_transition_matrix_pair(
            mats["fw"], mats["bw"], out_dir, tag,
            counts_fw=run["fw"].get("state_counts"),
            counts_bw=run["bw"].get("state_counts"))
        print(f"    figure -> {p}")

        results[tag] = {
            "tag": tag, "kind": spec["kind"],
            "T_fw": mats["fw"], "T_bw": mats["bw"],
            "seed": seed, "repeat_index": i, "fell_back": fell_back,
            "err_recorded": err, "ce_recorded": rec_ce, "ce_retrained": got_ce,
            "theory": rec.get("paired", {}).get("theory", spec.get("theory")),
            "chunk": chunk, "total_run": total_run,
            "n_states": int(mats["fw"].shape[0]),
        }
        # Dropped explicitly: one_repeat holds both arms alive when
        # keep_models is on, and the loop would otherwise carry every
        # process's pair to the end of the folder.
        run["_models"].clear()

    if results:
        save_pkl(results, os.path.join(out_dir, "transition_matrices.pkl"))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+",
                    help="folder names under --out-root, e.g. 00_base 08_pqsweep")
    ap.add_argument("--out-root", default=_OUT_DEFAULT)
    ap.add_argument("--total-run", type=int, default=5000,
                    help="generation steps per arm (default 5000)")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="restrict to these process tags (default: all)")
    args = ap.parse_args()

    t0 = time.time()
    for folder in args.folders:
        one_folder(folder, args.out_root, args.total_run, args.tags)
    print(f"\n  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
