"""
Loading, aggregation and shared plot style for the tau / optimiser studies.

Reads the saved JSON, so every table and figure redraws without retraining.

NORMALISATION.  study0 wrote arms as (tau0, tau1) pairs and study1/2 write them
as spec strings.  `load` converts the former to the latter -- (0.5, 5) becomes
"geom:0.5:5", (1, 1) becomes "const:1" -- so the pilot's four arms sit in the
same namespace as everything since and one loader serves all of them.

COLOUR.  A sequential ramp for the endpoint axis (tau_end is ordered, so its
series must not be categorical), one de-emphasised grey for the reference arm,
and two categorical slots for everything else.  The reference also carries a
dashed stroke: identity is never colour alone.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────
GREY = "#8a8f98"      # de-emphasis: the reference arm (const:1)
INK = "#0b0b0b"
INK2 = "#52514e"
RULE = "#d7d7d4"
SURF = "#fcfcfb"
S1 = "#2a78d6"
S2 = "#eb6834"
S3 = "#1baf7a"
S4 = "#a855c7"
# sequential, low tau_end -> high tau_end
SEQ = ["#cfe3f7", "#8fc0ee", "#4f97dd", "#2a78d6", "#17508f"]

REFERENCE = "const:1"


def style(ax, ygrid=True):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    if ygrid:
        ax.grid(axis="y", color=RULE, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> {p}")
    return p


def _arm_spec(run, cell):
    """Spec string for a run, whatever schema wrote it."""
    if run.get("tau"):
        return run["tau"]
    t0, t1 = run.get("tau0"), run.get("tau1")
    if t0 is None:
        return run.get("arm", "?")
    return (f"const:{t0:g}" if t0 == t1 else f"geom:{t0:g}:{t1:g}")


def load(*dirs, study=None):
    """Every cell found under `dirs`, arms normalised to spec strings."""
    cells = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            j = json.load(open(p))
            if study and str(j.get("study", "0")) != str(study):
                continue
            fw = j["mode"] == "forward"
            runs = []
            for r in j["runs"]:
                r = dict(r)
                r["spec"] = _arm_spec(r, j)
                r.setdefault("opt", "adam:0.9:0.999")
                runs.append(r)
            cells.append(dict(
                path=p, study=str(j.get("study", "0")),
                tag=j["spec"]["tag"], mode=j["mode"],
                V=int(j["spec"]["num_token"]), K=int(j["spec"]["n_states"]),
                true_k=int(j["spec"]["true_k_fw" if fw else "true_k_bw"]),
                C=float(j["spec"]["C_plus" if fw else "C_minus"]),
                H=float(j["spec"]["theory"]),
                conv_tol=float(j["cfg"]["conv_tol"]),
                seeds=sorted({r["seed"] for r in runs}),
                runs=runs))
    return cells


def title(c):
    t = c["tag"].replace("traj_", "").replace("_", " ")
    return f"{t}  {c['mode']}   V={c['V']}  true k={c['true_k']}  C={c['C']:.3f}"


def by_arm(cell, key, aggregate="mean"):
    """{spec: value} over the cell's runs."""
    out = {}
    for r in cell["runs"]:
        out.setdefault(r["spec"], []).append(r[key])
    if aggregate == "mean":
        return {k: float(np.mean(v)) for k, v in out.items()}
    if aggregate == "sd":
        return {k: (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                for k, v in out.items()}
    return out


def paired_delta(cell, key, ref=REFERENCE):
    """
    Per-seed change against the reference arm, then averaged.

    Paired: arms within a cell share the realisation and the initialisation, so
    the per-seed difference removes the seed variance that a difference of means
    would leave in.  Returns {spec: (mean_delta, sd_delta, n)}.
    """
    per = {}
    for r in cell["runs"]:
        per.setdefault(r["seed"], {})[r["spec"]] = r[key]
    out = {}
    for seed, d in per.items():
        if ref not in d:
            continue
        for spec, v in d.items():
            out.setdefault(spec, []).append(v - d[ref])
    return {k: (float(np.mean(v)),
                float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                len(v))
            for k, v in out.items()}


def recovery_score(cell):
    """
    One number per arm, lower is better:  mean |S_emp - C| over the cell's seeds.

    Why this and not k_occ.  |S_emp - C| is SIGNED-symmetric about the truth: it
    punishes under-recovery (states merged, S below C) and duplication (one
    causal state split across several bottleneck states, S above C) alike,
    whereas k_occ counts a duplicated state as a success.  6.9 of
    DISCRETE_STATE_EXPERIMENTS.md is exactly that failure, and h_state_given_token
    is its independent witness -- reported beside this, never folded into it.
    """
    out = {}
    for r in cell["runs"]:
        out.setdefault(r["spec"], []).append(abs(r["S_emp_minus_C"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def pooled_table(cells, keys=("k_occ", "S_emp_minus_C", "ce_minus_H",
                              "h_state_given_token")):
    """{spec: {key: (mean, sd)}} pooled over every run of every cell."""
    acc = {}
    for c in cells:
        for r in c["runs"]:
            d = acc.setdefault(r["spec"], {k: [] for k in keys})
            for k in keys:
                d[k].append(r[k])
            d.setdefault("_abs_dS", []).append(abs(r["S_emp_minus_C"]))
            d.setdefault("_krat", []).append(r["k_occ"] / max(1, r["true_k"]))
            d.setdefault("_conv", []).append(1.0 if r["within_tol"] else 0.0)
    out = {}
    for spec, d in acc.items():
        out[spec] = {k: (float(np.mean(v)),
                         float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                     for k, v in d.items()}
        out[spec]["n"] = (len(d[keys[0]]), 0)
    return out


def parse_family(spec: str):
    """
    Structural view of a tau spec, for grouping series on a figure.

    Returns {kind, a, b, extra} -- `a` the start tau, `b` the end tau -- or None
    for anything that is not a tau spec (an optimiser arm, say).  The endpoint
    and start AXES are read off this rather than off the label text, so a new
    arm joins the right series by construction.
    """
    parts = str(spec).split(":")
    kind = parts[0]
    try:
        v = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if kind == "const" and len(v) == 1:
        return dict(kind="const", a=v[0], b=v[0], extra=None)
    if kind in ("geom", "lin", "cos") and len(v) == 2:
        return dict(kind=kind, a=v[0], b=v[1], extra=None)
    if kind in ("hold", "ramp") and len(v) == 3:
        return dict(kind=kind, a=v[0], b=v[1], extra=v[2])
    return None


def merge_cells(cells):
    """
    Fold JSONs describing the SAME (study, tag, mode) into one cell.

    Needed whenever a cell's arms were produced by more than one invocation --
    a phase whose reference arm was run separately, or a job salvaged from its
    log.  Pairing happens WITHIN a cell, so leaving them apart would put the
    reference in one cell and the arms in another and silently drop every pair.

    Merging is sound here because a run is a pure function of its seed: data
    (`np.random.default_rng(seed)`), split (`split_loader(..., seed=seed)`) and
    initialisation (`torch.manual_seed(seed*1000)`) are all seeded, and CPU is
    exactly repeatable.  A (seed, spec) present in both files therefore carries
    the same numbers, and the later one simply replaces it.
    """
    out = {}
    for c in cells:
        k = (c["study"], c["tag"], c["mode"])
        if k not in out:
            out[k] = dict(c)
            out[k]["runs"] = list(c["runs"])
            out[k]["paths"] = [c["path"]]
        else:
            out[k]["runs"] += c["runs"]
            out[k]["paths"].append(c["path"])
            out[k]["seeds"] = sorted({r["seed"] for r in out[k]["runs"]})
    return list(out.values())


def common_support(cells, ref=REFERENCE):
    """
    Arms that appear on EVERY cell in `cells`, so a pooled mean compares like
    with like.

    Pooling without this is the trap: an arm run only on the two easiest cells
    collects a larger paired improvement than one run on all of them, and the
    ranking then measures which cells an arm happened to see rather than how
    good it is.
    """
    sets = [set(r["spec"] for r in c["runs"]) for c in cells]
    return (set.intersection(*sets) if sets else set()) | {ref}
