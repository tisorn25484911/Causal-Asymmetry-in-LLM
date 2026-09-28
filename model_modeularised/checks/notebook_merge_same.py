"""
The notebook's merge, ported: extraction.merge_equivalent_states must do exactly what
Experimental_pipeline/updated_pipeline_asymmetric_process.ipynb's merge_states_by_future_distributions
does (run_pipeline steps 6-7), with the notebook's own thresholds.

Runs both on the empirical machine of every saved run it finds (results/, checks/results/) and
compares the merges (pairs, in order), the state grouping, the successors and the emission rows.

    python checks/notebook_merge_same.py            -> one line per run, then SAME / DIFFERENT
"""
import contextlib
import glob
import io
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.dirname(HERE)
EP = os.path.join(os.path.dirname(MOD), "Experimental_pipeline")
sys.path.insert(0, MOD)
import extraction  # noqa: E402

NOTEBOOK = os.path.join(EP, "updated_pipeline_asymmetric_process.ipynb")


def notebook_namespace():
    """exec the notebook's imports, config and merge cells, found by what they define."""
    nb = json.load(open(NOTEBOOK))

    def cell(name):
        hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and any(
            l.startswith((f"def {name}(", f"class {name}:", f"class {name}(")) for l in "".join(c["source"]).splitlines())]
        assert len(hits) == 1, (name, hits)
        return "".join(nb["cells"][hits[0]]["source"])

    ns, cwd = {}, os.getcwd()
    os.chdir(EP)                                        # the imports cell reads processes.py
    sys.path.append(EP)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for name in ("set_seed", "ExperimentConfig", "merge_states_by_future_distributions"):
                exec(compile(cell(name), name, "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


def main():
    ns = notebook_namespace()
    cfg = ns["cfg"]
    assert (cfg.merge_future_horizon, cfg.merge_js_threshold, cfg.min_state_count_for_merge) == \
        (extraction.MERGE_HORIZON, extraction.MERGE_JS, extraction.MERGE_MIN_COUNT), "thresholds differ"
    paths = sorted(set(glob.glob(os.path.join(MOD, "results", "*", "runs", "*.pkl")) +
                       glob.glob(os.path.join(HERE, "results", "**", "runs", "*.pkl"), recursive=True) +
                       glob.glob(os.path.join(HERE, "results", "diagnostics", "*", "*.pkl"))))
    same = True
    for p in paths:
        r = pickle.load(open(p, "rb"))
        raw = r.get("machine")
        if raw is None or "counts" not in raw:
            continue
        n = len(raw["counts"])
        nb_in = {"next_state": np.asarray(raw["next_state"]), "emission_probs": np.asarray(raw["emission_probs"]),
                 "init_probs": np.full(n, 1.0 / n), "original_to_current": np.arange(n)}
        nb_out, nb_log = ns["merge_states_by_future_distributions"](
            nb_in, state_weights=np.asarray(raw["counts"], float), horizon=cfg.merge_future_horizon,
            js_threshold=cfg.merge_js_threshold, min_state_count_for_merge=cfg.min_state_count_for_merge,
            verbose=False)
        ours, groups, log = extraction.merge_equivalent_states(raw, raw["counts"])
        o2c = np.empty(n, dtype=np.int64)
        for g, members in enumerate(groups):
            o2c[members] = g
        checks = {
            "pairs": [tuple(e["merged_pair_old_labels"]) for e in nb_log] == [(i, j) for i, j, *_ in log],
            "js": np.allclose([e["js_divergence"] for e in nb_log], [js for _, _, js, *_ in log], rtol=1e-9, atol=0),
            "groups": np.array_equal(o2c, np.asarray(nb_out["original_to_current"])),
            "next_state": np.array_equal(ours["next_state"], nb_out["next_state"]),
            "emission": np.allclose(ours["emission_probs"], nb_out["emission_probs"], atol=1e-12),
        }
        ok = all(checks.values())
        same &= ok
        print(f"{'same' if ok else 'DIFFERENT':9s} {os.path.relpath(p, MOD):80s} {n} states -> {len(groups)} "
              f"({len(log)} merges)" + ("" if ok else f"   {checks}"))
    print("SAME" if same else "DIFFERENT")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
