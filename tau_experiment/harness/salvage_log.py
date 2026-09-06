#!/usr/bin/env python3
"""
Rebuild a cell's JSON from its console log, for a job stopped before it wrote.

    python -m harness.salvage_log study1_tau_schedules/results/1B_cell2_*.log

`run_cell` writes its JSON only at the end, so a job killed at run 20 of 24
loses all 20.  Every one of those runs is nevertheless printed in full -- seed,
arm, CE, k_occ, S_emp and H(state|token) are all on the line -- so the record is
recoverable exactly, not approximated.

The output carries `"partial": true` and only the runs the log actually shows.
Nothing is imputed: an arm that ran on one seed instead of two ends up with n=1,
which the paired statistics already report per arm, so an incomplete cell is
visible in the tables rather than silently averaged as if it were complete.

WHAT IS NOT RECOVERABLE.  `states_claimed_per_epoch` (the log prints only the
four quarter values) and `occupancy_top`.  The mechanism figure therefore skips a
salvaged cell; every table and every other figure is unaffected.
"""
import json
import os
import re
import sys
from datetime import datetime

HEAD = re.compile(r"STUDY\s+(\S+)\s+(\S+)\s+(forward|backward)")
GEOM = re.compile(r"V=(\d+)\s+K=(\d+)\s+true_k=(\d+)\s+H_inf=([\d.-]+)\s+C=([\d.-]+)")
RUN = re.compile(
    r"^\s*seed\s+(\d+)\s+(\S+)\s+([\d.]+)s\s+"
    r"CE\s+([\d.-]+)\s+\(([+-][\d.]+)\)\s+"
    r"k\s+(\d+)/(\d+)\s+"
    r"S_emp\s+([\d.-]+)\s+\(([+-][\d.]+)\)\s+"
    r"H\(s\|tok\)\s+([\d.]+)\s+"
    r"claimed\s+\[([\d,\s]*)\]")


def salvage(log_path: str, out_dir: str | None = None, conv_tol: float = 0.10):
    txt = open(log_path).read()
    h, g = HEAD.search(txt), GEOM.search(txt)
    if not (h and g):
        raise SystemExit(f"{log_path}: no header -- nothing to salvage")
    study, tag, mode = h.group(1), h.group(2), h.group(3)
    V, K, true_k = int(g.group(1)), int(g.group(2)), int(g.group(3))
    H_inf, C = float(g.group(4)), float(g.group(5))
    fw = mode == "forward"

    runs = []
    for line in txt.splitlines():
        m = RUN.match(line)
        if not m:
            continue
        q = [int(x) for x in m.group(11).split(",") if x.strip()]
        ce, dce = float(m.group(4)), float(m.group(5))
        runs.append(dict(
            seed=int(m.group(1)), arm=m.group(2), tau=m.group(2),
            opt="adam:0.9:0.999", mode=mode,
            ce=ce, ce_minus_H=dce, within_tol=bool(abs(dce) <= conv_tol),
            diverged="[DIV]" in line,
            k_occ=int(m.group(6)), k_used=int(m.group(6)),
            true_k=int(m.group(7)),
            S_emp=float(m.group(8)), C=C, S_emp_minus_C=float(m.group(9)),
            h_state_given_token=float(m.group(10)),
            states_claimed_quarters=q,
            states_claimed_per_epoch=[],      # not in the log; see the docstring
            salvaged=True))
    if not runs:
        raise SystemExit(f"{log_path}: header found but no run lines")

    out = dict(
        study=study, partial=True, salvaged_from=os.path.basename(log_path),
        spec=dict(tag=tag, num_token=V, n_states=K,
                  true_k_fw=true_k if fw else true_k,
                  true_k_bw=true_k, C_plus=C, C_minus=C, theory=H_inf),
        cfg=dict(conv_tol=conv_tol), mode=mode,
        seeds=len(sorted({r["seed"] for r in runs})),
        arms=[dict(name=a, tau=a, opt="adam:0.9:0.999")
              for a in dict.fromkeys(r["arm"] for r in runs)],
        runs=runs, minutes=None)

    out_dir = out_dir or os.path.dirname(log_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join(out_dir, f"{study}_{tag}_{mode}_{stamp}_PARTIAL.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    seeds = sorted({r["seed"] for r in runs})
    print(f"  salvaged {len(runs)} runs, seeds {seeds}, "
          f"{len(out['arms'])} arms -> {p}")
    return p


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        salvage(a)
