#!/usr/bin/env python3
"""
Build REPORT.html from report_template.html, inlining every figure as a data URI.

The Artifact CSP admits no external images, so the figures have to travel with
the page.  Inlining also keeps the report readable straight off disk, with no
dependency on the figures/ directories still being where they were.
"""
import base64
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = {
    "F1": "study1_tau_schedules/figures/F1_answer.png",
    "F2": "study1_tau_schedules/figures/F2_levels.png",
    "F3": "study1_tau_schedules/figures/F3_shape.png",
    "F4": "study1_tau_schedules/figures/F4_mechanism.png",
    "F5": "study1_tau_schedules/figures/F5_robustness.png",
    "F6": "study1_tau_schedules/figures/F6_headline.png",
    "F7": "study1_tau_schedules/figures/F7_where_it_helps.png",
    "G1": "study2_momentum/figures/G1_answer.png",
    "G2": "study2_momentum/figures/G2_axes.png",
    "G3": "study2_momentum/figures/G3_mechanism.png",
}


def main():
    html = open(os.path.join(HERE, "report_template.html")).read()
    total = 0
    for key, rel in FIGS.items():
        p = os.path.join(HERE, rel)
        token = "{{" + key + "}}"
        if not os.path.exists(p):
            html = html.replace(token, "")
            print(f"  MISSING {rel}")
            continue
        b = open(p, "rb").read()
        total += len(b)
        html = html.replace(token,
                            "data:image/png;base64," + base64.b64encode(b).decode())
    left = re.findall(r"\{\{[A-Z0-9]+\}\}", html)
    if left:
        print(f"  unresolved placeholders: {sorted(set(left))}")
    out = os.path.join(HERE, "REPORT.html")
    open(out, "w").write(html)
    print(f"  {len(FIGS)} figures, {total/1e3:.0f} kB raw -> "
          f"{os.path.getsize(out)/1e6:.2f} MB  {out}")


if __name__ == "__main__":
    main()
