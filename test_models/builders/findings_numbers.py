"""Every number the section-7 findings quote, from the cached reports and machines of the finished runs."""
import sys
from itertools import combinations
sys.path.insert(0, "/Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version/test_models")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import analysis as A
import training as TR

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
PROCESSES = {"renewal-60": dict(process="renewal", params={"F": [1 / 60] * 60}, history_len=120),
             "flower-9": dict(process="flower", params={"n": 8, "m": 12, "dice_seed": 0}),
             "flower-13-backward": dict(process="flower", params={"n": 8, "m": 12, "dice_seed": 0, "direction": "backward"})}
MODELS = ["5b", "B", "C", "feedback_GRU", "discrete_Transformer"]
BELOW = {"renewal-60": (0.01, 0.005, 0.002, 0.001), "flower-9": (0.1, 0.01, 0.002), "flower-13-backward": (0.1, 0.01, 0.002)}
for label, crit in PROCESSES.items():
    runs = {f"{s.model} s{s.seed}": f for f, s in sorted(TR.find_runs(steps=50000, **crit).items(),
                                                          key=lambda kv: (MODELS.index(kv[1].model), kv[1].seed))}
    s0 = TR.read_settings(next(iter(runs.values())))
    levels = A.reference_levels(s0, "test")
    n_true = len(s0.process_object().T)
    mt = A.machine_table(runs).set_index("run")
    fb = A.first_below(runs, BELOW[label]).set_index("run")
    rows = []
    for name, f in runs.items():
        rep, h = A.report(f), TR.history(f)
        main = TR.load_model_module(rep["model"]).MAIN_PREDICTOR
        used = h[f"{main}_states_used"]
        rows.append({"run": name, "test excess": rep["metrics"][f"excess_{main}_future"],
                     "gap closed %": 100 * (1 - rep["metrics"][f"excess_{main}_future"] / levels["remembers_nothing"]),
                     "states": mt.loc[name, "states"], f"covered of {n_true}": mt.loc[name, "covered"],
                     "merge cost": mt.loc[name, "merge_cost"], "mean err": mt.loc[name, "mean_error"],
                     "states used (final val)": int(used.iloc[-1]), "max states used": int(used.max()),
                     "final val excess": h[TR.main_metric(rep["model"])].iloc[-1],
                     **{f"below {l:g}": fb.loc[name, f"below {l:g}"] for l in BELOW[label]},
                     "handoff gap final": h["handoff_margin_median"].iloc[-1] if "handoff_margin_median" in h else np.nan})
    print(f"\n===== {label}: levels {', '.join(f'{k} {v:.4f}' for k, v in levels.items())}")
    print(pd.DataFrame(rows).set_index("run").round(4).to_string())
    t = pd.DataFrame(rows)
    t["model"] = t["run"].str.rsplit(" ", n=1).str[0]
    print(t.groupby("model", sort=False)[["test excess", f"covered of {n_true}", "merge cost"]].agg(["mean", "min", "max"]).round(4).to_string())
    if label.startswith("flower"):
        names = s0.process_object().state_names
        costs = {(i, j): A.merge_cost(s0, [[i, j]]) for i, j in combinations(range(n_true), 2)}
        rank = {k: r + 1 for r, k in enumerate(sorted(costs, key=costs.get))}
        for name, f in runs.items():
            m = A.machine(f)
            merged = [g for g in m["sent_to"] if len(g) > 1]
            print(f"  {name:24s} merges: " + ("; ".join("{" + ",".join(names[c] for c in g) + "}"
                  + (f" rank {rank[tuple(g)]}/{len(costs)} cost {costs[tuple(g)]:.4f}" if len(g) == 2
                     else f" cost {A.merge_cost(s0, [g]):.4f}") for g in merged) or "none"))
