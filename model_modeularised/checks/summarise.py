"""
One table for every model on every arm, all on the same data draw per process:

    results/default                      the main models: transformer, feedback GRU + Gumbel-ST
    results/renewal_K8                   all three architectures on the renewal control (K = 8)
    results/renewal                      the same at K = 4, below the process's 5 causal states
    checks/results/ablations/*           the runs that led there (deterministic head; Gumbel GRUs;
                                         the transformer at the earlier standard, geom:5:0.5 / 800 ep)
    checks/results/crosscheck/old_gru_*  the notebook GRU before modularisation

    python checks/summarise.py     -> prints markdown, writes checks/results/crosscheck/summary.{md,csv,png}

CE is compared on the SAME positions for every model: the old GRU scores only
positions >= past_len (20) -- its encoder eats the first 20 tokens -- so the
modular models' CE - exact is also given over positions >= 20, from their saved
per-position curves (same 200 held-out sequences).
"""
import csv
import glob
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")       # real runs
ABLATIONS = os.path.join(HERE, "results", "ablations")
CROSS = os.path.join(HERE, "results", "crosscheck")            # the notebook GRU, and where this writes
PAST_LEN = 20
# (experiment folder, arch) -> label, in the order the table lists them
SOURCES = [
    (os.path.join(RESULTS, "default"), {"transformer": "Transformer (main)",
                                        "gru_feedback": "Feedback GRU + Gumbel-ST (main)"}),
    (os.path.join(RESULTS, "renewal_K8"), {"transformer": "Transformer (main)",
                                           "gru_feedback": "Feedback GRU + Gumbel-ST (main)",
                                           "gru": "Read-out GRU + Gumbel-ST"}),
    (os.path.join(RESULTS, "renewal"), {"transformer": "Transformer, K = 4 < k",
                                        "gru_feedback": "Feedback GRU + Gumbel-ST, K = 4 < k",
                                        "gru": "Read-out GRU + Gumbel-ST, K = 4 < k"}),
    (os.path.join(ABLATIONS, "transformer_geom800", "default"), {"transformer": "Transformer, earlier standard (run 2)"}),
    (os.path.join(ABLATIONS, "transformer_geom800", "renewal_K8"), {"transformer": "Transformer, earlier standard (run 2)"}),
    (os.path.join(ABLATIONS, "deterministic_head"), {"transformer": "Transformer, earlier standard (run 1)",
                                                     "gru_feedback": "Feedback GRU, deterministic head",
                                                     "gru": "Read-out GRU, deterministic head"}),
    (os.path.join(ABLATIONS, "gumbel_gru"), {"gru_feedback": "Feedback GRU + Gumbel-ST (earlier run)",
                                             "gru": "Read-out GRU + Gumbel-ST"}),
]
OLD_GRU = "Notebook GRU (before modularisation)"
LABEL = {lab: lab for _, labels in SOURCES for lab in labels.values()} | {OLD_GRU: OLD_GRU}
ORDER = list(dict.fromkeys([lab for _, labels in SOURCES for lab in labels.values()] + [OLD_GRU]))


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def rows():
    out = []
    for folder, labels in SOURCES:
        for p in sorted(glob.glob(os.path.join(folder, "runs", "*.pkl"))):
            r = load(p)
            if r["arch"] not in labels:
                continue
            m, pp = r["metrics"], r["per_position"]
            out.append(dict(
                model=labels[r["arch"]], process=r["process"], arm=r["arm"],
                schedule=f"{r['cfg']['tau']}, {r['cfg']['max_epochs']} ep",
                full=m["full"], discovered=m["discovered"], true_k=m["true_k"], used=m["k_used"],
                S_minus_C=m["S_minus_C"], emission_tv=m["emission_tv"],
                T_err=m.get("exact_T_err", m["transition_max_err"]),
                gap_all=m["gap_exact"],
                gap_20=float(np.mean(pp["neural"][PAST_LEN - 1:]) - np.mean(pp["exact"][PAST_LEN - 1:])),
                determinism=m["determinism"], data=r["data_sha1"]))
    for p in sorted(glob.glob(os.path.join(CROSS, "old_gru_*.pkl"))):
        r = load(p)
        out.append(dict(
            model=OLD_GRU, process=r["process"], arm=r["arm"], schedule="notebook", full=r["full"],
            discovered=r["discovered"], true_k=r["true_k"], used=r["k_used"],
            S_minus_C=r["S_minus_C"], emission_tv=r["emission_tv"], T_err=r["T_err"],
            gap_all=float("nan"), gap_20=r["gap"], determinism=1.0, data=r["data_sha1"]))
    out.sort(key=lambda d: (d["process"], ("forward", "backward").index(d["arm"]), ORDER.index(d["model"])))
    return out


def markdown(rs):
    head = ("| process | arm | model | τ, epochs | FULL | discovered | states used | S_emp − C | emission TV | "
            "T err | CE − exact (all) | CE − exact (pos ≥ 20) | data |")
    lines = [head, "|" + "---|" * 13]
    for d in rs:
        lines.append(
            f"| {d['process']} | {d['arm']} | {LABEL[d['model']]} | {d['schedule']} | {'**FULL**' if d['full'] else '–'} | "
            f"{d['discovered']}/{d['true_k']} | {d['used']} | {d['S_minus_C']:+.3f} | {d['emission_tv']:.3f} | "
            f"{d['T_err']:.3f} | " + ("–" if np.isnan(d['gap_all']) else f"{d['gap_all']:+.4f}") +
            f" | {d['gap_20']:+.4f} | `{d['data']}` |")
    return "\n".join(lines)


def main():
    rs = rows()
    if not rs:
        print("no results yet"); return 1
    md = markdown(rs)
    print(md)
    os.makedirs(CROSS, exist_ok=True)
    with open(os.path.join(CROSS, "summary.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(CROSS, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader(); w.writerows(rs)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["process", "arm", "model", "τ, epochs", "FULL", "discovered", "used", "S_emp − C", "emission TV",
            "T err", "CE − exact (pos ≥ 20)"]
    body = [[d["process"], d["arm"], LABEL[d["model"]], d["schedule"], "FULL" if d["full"] else "–",
             f"{d['discovered']}/{d['true_k']}", str(d["used"]), f"{d['S_minus_C']:+.3f}",
             f"{d['emission_tv']:.3f}", f"{d['T_err']:.3f}", f"{d['gap_20']:+.4f}"] for d in rs]
    fig, ax = plt.subplots(figsize=(17, 0.27 * len(rs) + 0.9))
    ax.axis("off")
    widths = [0.06, 0.07, 0.25, 0.14, 0.05, 0.07, 0.05, 0.08, 0.08, 0.06, 0.11]   # the model names need room
    tbl = ax.table(cellText=body, colLabels=cols, colWidths=widths, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.25)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8DEE6")
        if i == 0 or (i > 0 and body[i - 1][4] == "FULL" and cols[j] == "FULL"):
            cell.set_text_props(weight="bold", color="#3C4653")
    ax.set_title("Cross-check: every model on every arm, same data draw per process "
                 "(τ schedule and epochs per row, from each run's saved config)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(CROSS, "summary.png"), dpi=110, bbox_inches="tight")
    print(f"\nwritten: {os.path.join(CROSS, 'summary.{md,csv,png}')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
