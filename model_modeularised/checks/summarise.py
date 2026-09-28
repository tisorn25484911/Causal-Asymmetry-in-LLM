"""
One table for every model on every arm, all on the same data draw per process:

    results/default                          the main runs at the current defaults (since 2026-09-28:
                                             K = 4V, 1200 epochs, the argmax head, MPS workaround on)
    checks/results/ablations/gumbel_2026-09-28   the same with Gumbel-ST for every model
    checks/results/ablations/pre_2026-09-28      the main runs at the defaults before 2026-09-28
                                             (K = 2V; renewal also at K = 8 and K = 4 < k)
    checks/results/ablations/*               the runs that led there (deterministic head; Gumbel GRUs;
                                             the transformer at geom:5:0.5 / 800 ep)
    checks/results/crosscheck/old_gru_*      the notebook GRU before modularisation

    python checks/summarise.py     -> prints markdown, writes checks/results/crosscheck/summary.{md,csv,png}

Every row says what it ran: K, the head's sampler, the tau schedule and length, from the run's
saved config.  FULL is judged after the notebook's merge (extraction.add_merged_metrics); the
raw verdict and state count are beside it.

CE is compared on the SAME positions for every model: the old GRU scores only
positions >= past_len (20) -- its encoder eats the first 20 tokens -- so the
modular models' CE - exact is given over positions >= 20, from their saved
per-position curves (same 200 held-out sequences).
"""
import csv
import glob
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from extraction import add_merged_metrics  # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results")       # real runs
ABLATIONS = os.path.join(HERE, "results", "ablations")
CROSS = os.path.join(HERE, "results", "crosscheck")            # the notebook GRU, and where this writes
PAST_LEN = 20
PRE = os.path.join(ABLATIONS, "pre_2026-09-28")
# (experiment folder, {arch: label}), in the order the table lists them
SOURCES = [
    (os.path.join(RESULTS, "default"), {"transformer": "Transformer (main)", "gru_feedback": "Feedback GRU (main)",
                                        "gru": "Read-out GRU"}),
    (os.path.join(ABLATIONS, "gumbel_2026-09-28", "default"), {"transformer": "Transformer, Gumbel 09-28",
                                                               "gru_feedback": "Feedback GRU, Gumbel 09-28"}),
    (os.path.join(PRE, "default"), {"transformer": "Transformer, before 09-28",
                                    "gru_feedback": "Feedback GRU, before 09-28"}),
    (os.path.join(PRE, "renewal_K8"), {"transformer": "Transformer, before 09-28",
                                       "gru_feedback": "Feedback GRU, before 09-28",
                                       "gru": "Read-out GRU, before 09-28"}),
    (os.path.join(PRE, "renewal_K4"), {"transformer": "Transformer, K = 4 < k",
                                       "gru_feedback": "Feedback GRU, K = 4 < k",
                                       "gru": "Read-out GRU, K = 4 < k"}),
    (os.path.join(ABLATIONS, "transformer_geom800", "default"), {"transformer": "Transformer, geom 800 ep (run 2)"}),
    (os.path.join(ABLATIONS, "transformer_geom800", "renewal_K8"), {"transformer": "Transformer, geom 800 ep (run 2)"}),
    (os.path.join(ABLATIONS, "deterministic_head"), {"transformer": "Transformer, geom 800 ep (run 1)",
                                                     "gru_feedback": "Feedback GRU, argmax 09-25",
                                                     "gru": "Read-out GRU, argmax 09-25"}),
    (os.path.join(ABLATIONS, "gumbel_gru"), {"gru_feedback": "Feedback GRU, Gumbel 09-25",
                                             "gru": "Read-out GRU, Gumbel 09-25"}),
]
OLD_GRU = "Notebook GRU (before modularisation)"
ORDER = list(dict.fromkeys([lab for _, labels in SOURCES for lab in labels.values()] + [OLD_GRU]))


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sampler(r) -> str:
    """The head's sampler a run trained with, from its saved config (runs before the flag: argmax)."""
    c = r["cfg"]
    g = c.get("gumbel")
    if g is None:
        g = c.get("gumbel_transformer" if r["arch"] == "transformer" else "gumbel_gru")
    return "Gumbel" if g else "argmax"


def rows():
    out = []
    for folder, labels in SOURCES:
        for p in sorted(glob.glob(os.path.join(folder, "runs", "*.pkl"))):
            r = load(p)
            if r["arch"] not in labels:
                continue
            add_merged_metrics(r, r["cfg"].get("full_tol", 0.05))     # idempotent; older runs get it here
            m, pp = r["metrics"], r["per_position"]
            V = r["hparams"]["token_size"]
            out.append(dict(
                model=labels[r["arch"]], process=r["process"], arm=r["arm"],
                K=f"{m['K']} ({m['K'] // V}V)", sampler=sampler(r),
                schedule=f"{r['cfg']['tau']}, {r['cfg']['max_epochs']} ep",
                full=m["full"], full_raw=m["full_raw"], discovered=m["discovered"], true_k=m["true_k"],
                states=f"{m['k_used']}→{m['merged_k']}",
                S_minus_C=m["merged_S_minus_C"], S_minus_C_raw=m["S_minus_C"], emission_tv=m["emission_tv"],
                T_err=m.get("exact_T_err", m["transition_max_err"]),
                gap_20=float(np.mean(pp["neural"][PAST_LEN - 1:]) - np.mean(pp["exact"][PAST_LEN - 1:])),
                determinism=m["determinism"], data=r["data_sha1"]))
    for p in sorted(glob.glob(os.path.join(CROSS, "old_gru_*.pkl"))):
        r = load(p)
        out.append(dict(
            model=OLD_GRU, process=r["process"], arm=r["arm"], K=f"{r['K']} (4V)", sampler="Gumbel",
            schedule="notebook", full=r["full"], full_raw=r["full"], discovered=r["discovered"],
            true_k=r["true_k"], states=f"{r['k_used']}→{r['merged_machine_size']}",
            S_minus_C=float("nan"), S_minus_C_raw=r["S_minus_C"], emission_tv=r["emission_tv"], T_err=r["T_err"],
            gap_20=r["gap"], determinism=1.0, data=r["data_sha1"]))
    out.sort(key=lambda d: (d["process"], ("forward", "backward").index(d["arm"]), ORDER.index(d["model"])))
    return out


def _f(v, fmt):
    return "–" if isinstance(v, float) and not np.isfinite(v) else fmt.format(v)


COLS = [("process", "process", "{}"), ("arm", "arm", "{}"), ("model", "model", "{}"), ("K", "K", "{}"),
        ("sampler", "sampler", "{}"), ("τ, epochs", "schedule", "{}"), ("FULL", "full", None),
        ("FULL raw", "full_raw", None), ("discovered", None, None), ("states raw→merged", "states", "{}"),
        ("S_emp − C", "S_minus_C", "{:+.3f}"), ("S_emp − C raw", "S_minus_C_raw", "{:+.3f}"),
        ("emission TV", "emission_tv", "{:.3f}"), ("T err", "T_err", "{:.3f}"),
        ("CE − exact (pos ≥ 20)", "gap_20", "{:+.4f}")]


def cells(d, bold=True):
    out = []
    for title, key, fmt in COLS:
        if title == "discovered":
            out.append(f"{d['discovered']}/{d['true_k']}")
        elif fmt is None:
            out.append(("**FULL**" if bold else "FULL") if d[key] else "–")
        else:
            out.append(_f(d[key], fmt))
    return out


def markdown(rs):
    lines = ["| " + " | ".join(t for t, _, _ in COLS) + " | data |", "|" + "---|" * (len(COLS) + 1)]
    for d in rs:
        lines.append("| " + " | ".join(cells(d)) + f" | `{d['data']}` |")
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
    cols = [t for t, _, _ in COLS]
    body = [cells(d, bold=False) for d in rs]
    chars = [max(5, len(c), *(len(b[j]) for b in body)) for j, c in enumerate(cols)]   # widths by content
    fig, ax = plt.subplots(figsize=(22, 0.27 * len(rs) + 0.9))
    ax.axis("off")
    tbl = ax.table(cellText=body, colLabels=cols, colWidths=[n / sum(chars) for n in chars],
                   loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.25)
    full_col = cols.index("FULL")
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D8DEE6")
        if i == 0 or (i > 0 and body[i - 1][full_col] == "FULL" and j == full_col):
            cell.set_text_props(weight="bold", color="#3C4653")
    ax.set_title("Every model on every arm, same data draw per process "
                 "(K, sampler, τ and epochs per row, from each run's saved config; FULL after the notebook's merge)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(CROSS, "summary.png"), dpi=110, bbox_inches="tight")
    print(f"\nwritten: {os.path.join(CROSS, 'summary.{md,csv,png}')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
