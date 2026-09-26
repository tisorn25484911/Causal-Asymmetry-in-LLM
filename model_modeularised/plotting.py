"""
Figures for the modularised runs, after the transformer notebook's
(Experimental_pipeline/updated_pipeline_transformer.ipynb, section 5).

Every function takes the result dict(s) that run_model.run_arm saves and RETURNS
the figure.  Nothing here selects a backend or calls show(), so it works from a
notebook (display the figure) and from a script (which sets the Agg backend and
calls the save_* helpers).

    plot_training                      CE against H_inf, tau, states used, gradient norm
    plot_states                        occupancy vs theory, learned vs true emission rows, determinism
    plot_machine_graph                 the machine the model became beside the theoretical one
    plot_per_position                  model vs exact-model CE along the sequence, and the
                                       theory floor h(t) = H[X_t | X_<t]
    plot_transition_comparison_arms    learned vs theoretical state-to-state matrices, both arms
    plot_complexity                    C against S_emp, both arms
    plot_process_theory                one process's closed forms: M+, M-, entropy convergence
    plot_theory_comparison             every architecture on one process against those closed forms
    plot_scorecard, scorecard_table    every run of a sweep on one page
    save_run_figures, save_group_figures, save_process_figures, save_scorecard
                                       write them under <fig_root>/<process>/[<arch>/]
                                       (run_model.redraw calls these)

Every theoretical quantity comes from HMM_processes.py (theory.process_theory, and
extraction.theory_from_generator for the per-arm machine stored in each result).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from theory import DRAWABLE_K

_BLUE, _ORNG, _GREY, _INK = "#3B6EA5", "#C4703A", "#D8DEE6", "#3C4653"     # figures.py palette
_TEAL = "#12A094"
# Fixed order, never cycled; validated all-pairs for CVD (worst deutan ΔE 12.3).
ARCH_COLOR = {"transformer": _BLUE, "gru": _ORNG, "gru_feedback": _TEAL}
ARCH_LABEL = {"transformer": "Transformer", "gru": "GRU", "gru_feedback": "Feedback GRU"}
ARMS = ("forward", "backward")


def _title(r) -> str:
    return f"{ARCH_LABEL[r['arch']]} · {r['tag']} · {r['arm']}"


def _what(arm: str) -> str:
    return "next token" if arm == "forward" else "previous token"


def _heat(ax, M, xticks, yticks, title, cmap="Blues", vmax=1.0):
    M = np.asarray(M, dtype=float)
    ax.imshow(np.nan_to_num(M, nan=0.0), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    K, V = M.shape
    fs = 8 if max(K, V) <= 8 else 6
    for i in range(K):
        for j in range(V):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=fs,
                        color="white" if M[i, j] > 0.6 * vmax else _INK)
    ax.set_xticks(range(V)); ax.set_xticklabels([str(t) for t in xticks], fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels([str(t) for t in yticks], fontsize=7)
    ax.set_title(title, fontsize=10)


def _state_labels(r):
    """'s<id> → <theoretical state>' for each visited state, matched by its emission row."""
    machine, th = r["machine"], r["theory"]
    true = np.asarray(th["true_machine"]["emission_probs"])
    out = []
    for sid, row in zip(machine["state_ids"], machine["emission_probs"]):
        d = 0.5 * np.abs(np.asarray(row)[None, :] - true).sum(axis=1)
        out.append(f"s{sid} → {th['names'][int(np.argmin(d))]}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# one run
# ══════════════════════════════════════════════════════════════════════════
def plot_training(r):
    """Cross-entropy against H_inf over the run and the restored checkpoint; the tau
    schedule; states used by the training batch; the gradient norm."""
    h, cfg, m = r["history"], r["cfg"], r["metrics"]
    H, tol = m["entropy_rate"], cfg["conv_tol"]
    fig, axes = plt.subplots(1, 4, figsize=(21, 3.9), gridspec_kw=dict(width_ratios=[2.2, 1, 1, 1]))

    ax = axes[0]
    ax.axhspan(H - tol, H + tol, color=_GREY, alpha=0.55, lw=0, label="H_inf ± conv_tol")
    ax.axhline(H, color=_INK, lw=1.2, ls=(0, (5, 2)))
    ax.plot(h["step_at"], h["step_loss"], color=_BLUE, lw=0.7, alpha=0.5, label="train objective")
    if h["val_at"]:
        ax.plot(h["val_at"], h["val_loss"], color=_ORNG, lw=2.0, label="validation CE")
    if h.get("best_step", -1) >= 0:
        ax.axvline(h["best_step"], color=_INK, lw=1.0, ls="-.", alpha=0.7)
        ax.plot([h["best_step"]], [h["best_val"]], "o", ms=8, color=_INK,
                label=f"restored checkpoint (step {h['best_step']})")
    top = max(float(np.percentile(h["step_loss"], 99)), H + 0.5)
    ax.set_ylim(H - 0.15, top)
    ax.set_xlabel("gradient step"); ax.set_ylabel("bits / token")
    ax.set_title(f"{_title(r)}:  test {m['test_ce']:.3f}, fresh {m['fresh_ce']:.3f}, "
                 f"exact {m['exact_ce']:.3f}, H_inf {H:.3f}", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(h["step_at"], h["step_tau"], color=_BLUE, lw=2.0)
    ax.set_yscale("log"); ax.set_xlabel("gradient step"); ax.set_ylabel("tau")
    ax.set_title(f"tau  ({cfg['tau']})", fontsize=10); ax.grid(alpha=0.25)

    ax = axes[2]
    n = min(len(h["val_at"]), len(h["states_used"]))
    ax.plot(h["val_at"][:n], h["states_used"][:n], color=_BLUE, lw=1.2, drawstyle="steps-post")
    ax.axhline(m["true_k"], color=_INK, lw=1.2, ls=(0, (5, 2)), label=f"true k = {m['true_k']}")
    ax.set_ylim(0, m["K"] + 0.5)
    ax.set_xlabel("gradient step"); ax.set_ylabel("states in the training batch")
    ax.set_title(f"states used  (budget K = {m['K']})", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    ax = axes[3]
    g = np.asarray(h["step_grad_norm"], dtype=float)
    ax.plot(np.arange(len(g)), g, color=_BLUE, lw=0.5, alpha=0.7)
    ax.set_yscale("log"); ax.set_xlabel("optimiser step"); ax.set_ylabel("‖grad‖₂")
    ax.set_title(f"gradient norm  (max {m['max_grad_norm']:.2f})", fontsize=10); ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_states(r):
    """The restored model's causal states: occupancy against theory, its emission rows
    beside the true machine's, and how deterministic its (state, token) → state map is
    (1 = unifilar)."""
    rep, th, machine, m = r["report"], r["theory"], r["machine"], r["metrics"]
    occ = np.asarray(rep["occupancy"])
    learned, theory = np.sort(occ[occ > 0])[::-1], np.sort(np.asarray(th["occupancy"]))[::-1]
    tokens = range(r["cfg"]["token_size"])
    what = _what(r["arm"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1, 1, 1]))

    ax = axes[0]
    ax.bar(np.arange(len(learned)), learned, width=0.62, color=_BLUE, label="learned")
    ax.plot(np.arange(len(theory)), theory, "o", ms=8, color=_ORNG, label="theory")
    ax.set_xticks(range(max(len(learned), len(theory))))
    ax.set_xlabel("state (sorted by occupancy)"); ax.set_ylabel("occupancy")
    ax.set_title(f"{_title(r)}\nS_emp {m['S_emp']:.3f} / C {th['C']:.3f} bits   k = {m['k_used']}/{m['K']} "
                 f"(true {th['true_k']}, discovered {m['discovered']})", fontsize=10)
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8)

    _heat(axes[1], machine["emission_probs"], tokens, _state_labels(r), f"learned P({what} | state)")
    if th["true_k"] > DRAWABLE_K:
        _too_big(axes[2], th["true_k"], "theory")
    else:
        _heat(axes[2], th["true_machine"]["emission_probs"], tokens, th["names"], f"true P({what} | state)")
    det = np.asarray(machine["determinism"], dtype=float)
    _heat(axes[3], det, tokens, [f"s{s}" for s in machine["state_ids"]],
          f"determinism of (state, token) → state\nvisit-weighted {m['determinism']:.3f}")
    for ax in axes[1:]:
        ax.set_xlabel("token"); ax.set_ylabel("state")
    fig.tight_layout()
    return fig


def _too_big(ax, k, what):
    """A panel that says why a machine is not drawn."""
    ax.axis("off")
    ax.text(0.5, 0.5, f"{what}: {k} states\n(not drawn above {DRAWABLE_K})", ha="center", va="center",
            fontsize=11, color=_INK, transform=ax.transAxes)


def _draw_machine(ax, next_state, emission_probs, stationary, labels, title):
    """An emission-labelled unifilar machine as a graph: edge 'x|p' = emit token x with
    probability p and move along the edge; self-loops are listed on the node."""
    import networkx as nx
    next_state, emission_probs = np.asarray(next_state), np.asarray(emission_probs)
    num_states, alphabet_size = next_state.shape
    if num_states > DRAWABLE_K:
        _too_big(ax, num_states, title)
        return
    G = nx.DiGraph()
    edge_labels, self_loops = {}, {s: [] for s in range(num_states)}
    for s in range(num_states):
        for x in range(alphabet_size):
            p = float(emission_probs[s, x])
            if p <= 0.01:
                continue
            ns = int(next_state[s, x])
            lbl = f"{x}|{p:.2f}"
            if ns == s:
                self_loops[s].append(lbl)
            if (s, ns) in edge_labels:
                edge_labels[(s, ns)] += f"\n{lbl}"
            else:
                G.add_edge(s, ns); edge_labels[(s, ns)] = lbl
    for s in range(num_states):
        label = f"{labels[s]}\nπ={stationary[s]:.3f}"
        if self_loops[s]:
            label += "\n↺ " + ", ".join(self_loops[s])
        G.add_node(s, label=label)
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=7, k=2.5 / max(1, num_states ** 0.5), iterations=200)
    node_size = max(1200, 3000 - 100 * num_states)
    font_size = max(6, 9 - num_states // 6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color="lightsteelblue",
                           edgecolors="steelblue", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=nx.get_node_attributes(G, "label"),
                            font_size=font_size, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=14, width=1.2,
                           edge_color="gray", connectionstyle="arc3,rad=0.1",
                           min_source_margin=18, min_target_margin=18)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=max(5, font_size - 1), label_pos=0.35,
                                 edge_labels={k: v for k, v in edge_labels.items() if k[0] != k[1]},
                                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _draw_hmm(ax, M, stationary, title):
    """Any edge-emitting HMM, M[s, s', x] = P(x, s' | s) -- also non-unifilar ones,
    where one (state, token) can lead to several states.  Edge 'x|p'."""
    import networkx as nx
    M = np.asarray(M)
    S, _, V = M.shape
    if S > DRAWABLE_K:
        _too_big(ax, S, title)
        return
    G = nx.DiGraph()
    edge_labels, self_loops = {}, {s: [] for s in range(S)}
    for s in range(S):
        for s2 in range(S):
            for x in range(V):
                p = float(M[s, s2, x])
                if p <= 1e-3:
                    continue
                lbl = f"{x}|{p:.2f}"
                if s2 == s:
                    self_loops[s].append(lbl)
                    continue
                edge_labels[(s, s2)] = edge_labels[(s, s2)] + f"\n{lbl}" if (s, s2) in edge_labels else lbl
                G.add_edge(s, s2)
    for s in range(S):
        label = f"q{s}\nπ={stationary[s]:.3f}" + (("\n↺ " + ", ".join(self_loops[s])) if self_loops[s] else "")
        G.add_node(s, label=label)
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=7, iterations=200)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=max(1200, 3000 - 100 * S), node_color="#E4EAF1",
                           edgecolors=_INK, linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=nx.get_node_attributes(G, "label"),
                            font_size=max(6, 9 - S // 6), font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=14, width=1.2,
                           edge_color="gray", connectionstyle="arc3,rad=0.1",
                           min_source_margin=18, min_target_margin=18)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=max(5, 8 - S // 6),
                                 label_pos=0.35,
                                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def plot_machine_graph(r):
    """The machine the model became beside the theoretical epsilon-machine of its arm
    (HMM_processes via extraction.theory_from_generator).  Learned: the majority
    successor of each (state, token) along the data, exact for the feedback GRU.
    For the backward arm the edges read one step back in original time."""
    machine, th, m = r["machine"], r["theory"], r["metrics"]
    k = max(machine["next_state"].shape[0], th["true_k"])
    scale = max(1.0, min(k, DRAWABLE_K) / 10.0)
    fig, axes = plt.subplots(1, 2, figsize=(max(18, int(14 * scale)), max(6.5, int(5.5 * scale))))
    _draw_machine(axes[0], machine["next_state"], machine["emission_probs"], r["minimal_stationary"],
                  _state_labels(r),
                  f"learned: {machine['next_state'].shape[0]} states, determinism {m['determinism']:.3f}\n"
                  f"symbolic CE {m['symbolic_ce']:.3f} vs neural {m['neural_ce_span']:.3f} "
                  f"vs exact {m['exact_ce_span']:.3f}")
    _draw_machine(axes[1], th["true_machine"]["next_state"], th["true_machine"]["emission_probs"],
                  th["occupancy"], th["names"],
                  f"theory ({r['arm']} epsilon-machine): {th['true_k']} states, C = {th['C']:.3f} bits")
    fig.suptitle(f"{_title(r)}: the machine the model became vs the theoretical one", fontsize=12)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# both arms of one architecture on one process
# ══════════════════════════════════════════════════════════════════════════
def _arms(runs):
    return [a for a in ARMS if a in runs]


def plot_per_position(runs, theory=None):
    """Mean test CE at each position, the model against the exact model on the same
    tokens -- how fast each synchronises, and the gap left once it has -- and, with
    `theory` (theory.process_theory), the expected floor h(t) = H[X_t | X_<t] from
    HMM_processes.conditional_entropies."""
    arms = _arms(runs)
    fig, axes = plt.subplots(1, len(arms), figsize=(6.5 * len(arms), 3.8), squeeze=False)
    for ax, arm in zip(axes.ravel(), arms):
        r = runs[arm]
        pp, H = r["per_position"], r["metrics"]["entropy_rate"]
        t = np.arange(1, len(pp["neural"]) + 1)
        ax.plot(t, pp["neural"], color=ARCH_COLOR[r["arch"]], lw=2.0, label=ARCH_LABEL[r["arch"]])
        ax.plot(t, pp["exact"], color=_INK, lw=1.2, label="exact model, these sequences")
        if theory is not None:
            n = min(len(t), len(theory["floor"]))
            ax.plot(t[:n], theory["floor"][:n], color=_INK, lw=1.2, ls=":",
                    label="theory floor h(t) (expected)")
        ax.axhline(H, color=_INK, lw=1.0, ls=(0, (5, 2)), label="H_inf")
        ax.set_xscale("log"); ax.set_xlabel("tokens of context (log)"); ax.set_ylabel("bits / token")
        ax.set_title(f"{r['tag']} · {arm}:  gap to exact {r['metrics']['gap_exact']:+.4f} bits", fontsize=10)
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle(f"{ARCH_LABEL[runs[arms[0]]['arch']]}: cross-entropy along the sequence (test split)")
    fig.tight_layout()
    return fig


def _draw_transition_comparison(axes, r, which="free"):
    cmp = r["transition_comparison"] if which == "free" else r["exact_machine"]["comparison"]
    th = r["theory"]
    names = th["names"]
    if len(names) > DRAWABLE_K:
        for ax, what in zip(axes, ("learned (aggregated)", "theory", "|learned − theory|")):
            _too_big(ax, len(names), f"{r['arm']}: {what}")
        axes[2].set_title(f"max |err| {cmp['max_error']:.3f}, missing {len(cmp['missing'])}", fontsize=10)
        return
    how = "free-running" if which == "free" else "exact, enumerated"
    yt = [f"{names[t]}\n← {cmp['groups'][t] if cmp['groups'][t] else 'none'}" for t in range(len(names))]
    _heat(axes[0], cmp["aggregated"], names, yt,
          f"{r['arm']}: learned, {how} ({cmp['n_visited']} states → {len(names)})")
    _heat(axes[1], cmp["theory"], names, names, f"{r['arm']}: theory")
    err = cmp["abs_error"]
    missing = f"\nmissing: {[names[t] for t in cmp['missing']]}" if cmp["missing"] else ""
    _heat(axes[2], err, names, names,
          f"|learned − theory|   max {cmp['max_error']:.3f}, mean {cmp['mean_error']:.3f}{missing}",
          cmap="Reds", vmax=max(0.05, float(np.nanmax(err)) if np.isfinite(err).any() else 0.05))
    for ax in axes:
        ax.set_xlabel("to"); ax.set_ylabel("from")


def plot_transition_comparison_arms(runs, which="free"):
    """Both arms: each learned state-to-state matrix, mapped onto the theory states by
    emission row, against the closed form of its own arm.  which="free" is the
    free-running estimate every model has; which="exact" is the enumerated machine of
    a model that has one (the feedback GRU)."""
    arms = _arms(runs)
    fig, axes = plt.subplots(len(arms), 3, figsize=(16, 4.6 * len(arms)), squeeze=False)
    for row, arm in enumerate(arms):
        _draw_transition_comparison(axes[row], runs[arm], which)
    r0 = runs[arms[0]]
    fig.suptitle(f"{ARCH_LABEL[r0['arch']]} · {r0['tag']}: learned vs theoretical state-to-state transition matrices"
                 + ("  (exact machine)" if which == "exact" else ""))
    fig.tight_layout()
    return fig


def plot_complexity(runs):
    """Statistical complexity per arm: C from the closed form against S_emp of the restored model."""
    arms = _arms(runs)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 3.8), squeeze=False)
    for ax, arm in zip(axes.ravel(), arms):
        r = runs[arm]; m = r["metrics"]
        ax.bar([0, 1], [m["C"], m["S_emp"]], width=0.6, color=[_GREY, ARCH_COLOR[r["arch"]]])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"theory\nk = {m['true_k']}",
                            f"learned\nk = {m['k_used']}" + ("" if m["within_tol"] else "\n[not converged]")],
                           fontsize=9)
        ax.set_ylabel("bits"); ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"{arm}:  C {m['C']:.3f}  /  S_emp {m['S_emp']:.3f}", fontsize=10)
    r0 = runs[arms[0]]
    fig.suptitle(f"{ARCH_LABEL[r0['arch']]} · {r0['tag']}: statistical complexity")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# one process: its theory, and every model against it
# ══════════════════════════════════════════════════════════════════════════
def _theory_line(theory):
    return (f"h {theory['entropy_rate']:.4f}  E {theory['excess_entropy']:.4f}  "
            f"C+ {theory['C_plus']:.4f}  C− {theory['C_minus']:.4f}  "
            f"χ+ {theory['crypticity_plus']:.4f}  χ− {theory['crypticity_minus']:.4f}  bits")


def plot_process_theory(theory, tag):
    """The closed forms of one process, all from HMM_processes (theory.process_theory):
    the generating HMM exactly as HMM_processes defines it, its forward and reverse
    epsilon-machines, and the entropy-convergence floor h(L), whose excess over h
    sums to E."""
    fig, axes = plt.subplots(1, 4, figsize=(31, 7), gridspec_kw=dict(width_ratios=[1.1, 1.2, 1.2, 1]))
    gh = theory["generator_hmm"]
    _draw_hmm(axes[0], gh["M"], gh["stationary"],
              f"generating HMM (HMM_processes.{theory['process']}_*): {gh['M'].shape[0]} states"
              + ("" if gh["unifilar"] else ", non-unifilar"))
    for ax, arm, sign in ((axes[1], "forward", "+"), (axes[2], "backward", "−")):
        th = theory[arm]
        flag = " (infinite: truncated)" if th["infinite"] else (" (truncated)" if th["truncated"] else "")
        _draw_machine(ax, th["true_machine"]["next_state"], th["true_machine"]["emission_probs"],
                      th["occupancy"], th["names"],
                      f"M{sign} ({arm}): {th['true_k']} states{flag}, C{sign} = {th['C']:.4f} bits")
    ax = axes[3]
    L = np.arange(1, len(theory["h_L"]) + 1)
    ax.plot(L, theory["h_L"], "o-", color=_BLUE, lw=2.0, ms=4, label="h(L) = H[X_L | L − 1 tokens of context]")
    ax.axhline(theory["entropy_rate"], color=_INK, lw=1.0, ls=(0, (5, 2)), label=f"h = {theory['entropy_rate']:.4f}")
    ax.set_xscale("log"); ax.set_xlabel("block length L = tokens of context + 1 (log)"); ax.set_ylabel("bits / token")
    ax.set_title(f"entropy convergence: Σ(h(t) − h) = {theory['floor_excess']:.4f} = E {theory['excess_entropy']:.4f}",
                 fontsize=11)
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    note = ("\n" + "\n".join("⚠ " + w for w in theory["warnings"])) if theory["warnings"] else ""
    fig.suptitle(f"{tag}: theory (HMM_processes)   {_theory_line(theory)}{note}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_theory_comparison(runs_by_arch, theory, tag):
    """
    Every architecture on one process against the closed forms (one seed):
      S_emp vs C+ / C-   the memory the bottleneck uses vs the statistical complexity
      causal asymmetry   S_emp(backward) - S_emp(forward) vs C- - C+
      states             theoretical states discovered vs k, and states used
      CE - exact         cross-entropy above the exact model on the same tokens
    `runs_by_arch` is {arch: {arm: result}}.
    """
    archs = [a for a in ARCH_COLOR if a in runs_by_arch]
    arms = [a for a in ARMS if any(a in runs_by_arch[x] for x in archs)]
    x = np.arange(len(arms))
    w = 0.8 / max(1, len(archs))
    off = lambda i: (i - (len(archs) - 1) / 2) * w                                  # noqa: E731
    get = lambda arch, arm, key: (runs_by_arch[arch][arm]["metrics"][key]           # noqa: E731
                                  if arm in runs_by_arch[arch] else np.nan)
    theory_proxy = Line2D([], [], color=_INK, ls=(0, (5, 2)), lw=1.5, label="theory")
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6), gridspec_kw=dict(width_ratios=[1.1, 0.9, 1.1, 1.1]))

    ax = axes[0]
    for i, arch in enumerate(archs):
        ax.bar(x + off(i), [get(arch, a, "S_emp") for a in arms], width=w * 0.92,
               color=ARCH_COLOR[arch], label=ARCH_LABEL[arch])
    for j, arm in enumerate(arms):
        C = theory[arm]["C"]
        ax.hlines(C, j - 0.45, j + 0.45, colors=_INK, linestyles=(0, (5, 2)), lw=1.5)
        ax.text(j, C, f"C{'+' if arm == 'forward' else '−'} {C:.3f}", ha="center", va="bottom", fontsize=8, color=_INK)
    ax.set_xticks(x); ax.set_xticklabels(arms); ax.set_ylabel("bits")
    ax.set_title("S_emp vs statistical complexity", fontsize=10); ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [theory_proxy], fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(archs) + 1, frameon=False)

    ax = axes[1]
    if len(arms) == 2:
        d = [get(a, "backward", "S_emp") - get(a, "forward", "S_emp") for a in archs]
        ax.bar(np.arange(len(archs)), d, width=0.6, color=[ARCH_COLOR[a] for a in archs])
        ax.axhline(theory["C_minus"] - theory["C_plus"], color=_INK, ls=(0, (5, 2)), lw=1.5,
                   label=f"theory C− − C+ = {theory['C_minus'] - theory['C_plus']:+.3f}")
        ax.axhline(0, color=_INK, lw=0.8)
        ax.set_xticks(np.arange(len(archs))); ax.set_xticklabels([ARCH_LABEL[a] for a in archs], fontsize=8)
        ax.legend(fontsize=8)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "needs both arms", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("bits"); ax.set_title("causal asymmetry: S_emp(bw) − S_emp(fw)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    for i, arch in enumerate(archs):
        bars = ax.bar(x + off(i), [get(arch, a, "discovered") for a in arms], width=w * 0.92,
                      color=ARCH_COLOR[arch], label=ARCH_LABEL[arch])
        for b, a in zip(bars, arms):
            if a in runs_by_arch[arch]:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"used {int(get(arch, a, 'k_used'))}", ha="center", va="bottom", fontsize=7, color=_INK)
    for j, arm in enumerate(arms):
        k = theory[arm]["true_k"]
        ax.hlines(k, j - 0.45, j + 0.45, colors=_INK, linestyles=(0, (5, 2)), lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels(arms); ax.set_ylabel("theoretical states discovered")
    ax.set_title("states: discovered vs k (dashed)", fontsize=10); ax.grid(axis="y", alpha=0.25)

    ax = axes[3]
    for i, arch in enumerate(archs):
        ax.bar(x + off(i), [get(arch, a, "gap_exact") for a in arms], width=w * 0.92,
               color=ARCH_COLOR[arch], label=ARCH_LABEL[arch])
    ax.axhline(0, color=_INK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(arms); ax.set_ylabel("bits / token")
    ax.set_title("test CE − exact model CE (same tokens)", fontsize=10); ax.grid(axis="y", alpha=0.25)

    note = ("\n" + "   ".join("⚠ " + w_ for w_ in theory["warnings"])) if theory["warnings"] else ""
    fig.suptitle(f"{tag}: learned vs theory (HMM_processes)   {_theory_line(theory)}{note}", fontsize=11)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# every run
# ══════════════════════════════════════════════════════════════════════════
def _cells(results):
    """(process tag, arm) cells in a stable order, and a lookup (cell, arch) -> result."""
    cells, by = [], {}
    for r in results:
        cell = (r["tag"], r["arm"])
        if cell not in cells:
            cells.append(cell)
        by[(cell, r["arch"])] = r
    cells.sort(key=lambda c: (c[0], ARMS.index(c[1])))
    return cells, by


def scorecard_table(results):
    """One row per run: the numbers the scorecard draws."""
    rows = []
    cells, by = _cells(results)
    for cell in cells:
        for arch in ARCH_COLOR:
            r = by.get((cell, arch))
            if r is None:
                continue
            m = r["metrics"]
            rows.append({
                "process": cell[0], "arm": cell[1], "arch": ARCH_LABEL[arch],
                "FULL": m["full"], "discovered": f"{m['discovered']}/{m['true_k']}",
                "emission TV": m.get("emission_tv", float("nan")),
                "states used": m["k_used"], "S_emp - C": m["S_minus_C"],
                "T max err": m["transition_max_err"],
                "T err exact": m.get("exact_T_err", float("nan")), "CE - H_inf": m["gap"],
                "CE - exact": m["gap_exact"], "fresh CE - H_inf": m["fresh_gap"],
                "determinism": m["determinism"], "best step": f"{m['best_step']}/{m['n_steps']}",
                "params": r.get("n_params", float("nan")),
                "minutes": r.get("train_seconds", float("nan")) / 60.0,
            })
    return rows


def plot_scorecard(results):
    """Every run on one page: theory states discovered, S_emp - C, and the CE gap to
    the exact model, the two architectures side by side in each (process, arm) cell;
    below, the same runs as a table."""
    cells, by = _cells(results)
    archs = [a for a in ARCH_COLOR if any((c, a) in by for c in cells)]
    x = np.arange(len(cells))
    w = 0.8 / max(1, len(archs))
    fig = plt.figure(figsize=(19, 8.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95])
    panels = [
        ("discovered / true k", lambda m: m["discovered"] / m["true_k"], (0, 1.3), None),
        ("S_emp − C  (bits)", lambda m: m["S_minus_C"], None, 0.05),
        ("test CE − exact CE  (bits)", lambda m: m["gap_exact"], None, None),
    ]
    for col, (label, fn, ylim, band) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        if band is not None:
            ax.axhspan(-band, band, color=_GREY, alpha=0.55, lw=0, label=f"±{band} (FULL)")
        for i, arch in enumerate(archs):
            vals = [fn(by[(c, arch)]["metrics"]) if (c, arch) in by else np.nan for c in cells]
            bars = ax.bar(x + (i - (len(archs) - 1) / 2) * w, vals, width=w * 0.92,
                          color=ARCH_COLOR[arch], label=ARCH_LABEL[arch])
            if col == 0:
                for b, c in zip(bars, cells):
                    if (c, arch) in by:
                        mm = by[(c, arch)]["metrics"]
                        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                                f"{mm['discovered']}/{mm['true_k']}" + (" ✓" if mm["full"] else ""),
                                ha="center", va="bottom", fontsize=8, color=_INK)
        ax.axhline(0, color=_INK, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f"{c[0]}\n{c[1]}" for c in cells], fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(label, fontsize=10); ax.grid(axis="y", alpha=0.25)
        if col == 0:                          # one row above the "k/k" labels, never over a bar
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.legend(fontsize=8, loc="upper center", ncol=len(archs), frameon=False)
        else:
            ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    rows = scorecard_table(results)
    if rows:
        cols = ["process", "arm", "arch", "FULL", "discovered", "emission TV", "states used",
                "S_emp - C", "T max err", "T err exact", "CE - H_inf", "CE - exact", "fresh CE - H_inf",
                "determinism", "best step", "params", "minutes"]
        fmts = {"emission TV": "{:.3f}", "S_emp - C": "{:+.3f}", "T max err": "{:.3f}",
                "T err exact": "{:.3f}", "CE - H_inf": "{:+.4f}",
                "CE - exact": "{:+.4f}", "fresh CE - H_inf": "{:+.4f}", "determinism": "{:.3f}",
                "params": "{:,}", "minutes": "{:.1f}"}
        def cell(c, v):
            if c == "FULL":
                return "FULL" if v else "–"
            if isinstance(v, float) and not np.isfinite(v):
                return "–"                                   # e.g. no exact machine
            return fmts.get(c, "{}").format(v)
        body = [[cell(c, row[c]) for c in cols] for row in rows]
        # each column as wide as its longest entry, so a long process tag fits
        chars = [max(6, len(c), *(len(b[j]) for b in body)) for j, c in enumerate(cols)]
        tbl = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center",
                       colWidths=[n / sum(chars) for n in chars])
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.35)
        for (i, j), cell in tbl.get_celld().items():
            cell.set_edgecolor(_GREY)
            if i == 0:
                cell.set_text_props(weight="bold", color=_INK)
            elif cols[j] == "FULL":
                cell.set_text_props(weight="bold", color=_INK)
    r0 = results[0]
    per_arch = {}
    for r in results:                         # schedule and length can differ by architecture
        per_arch.setdefault(ARCH_LABEL[r["arch"]], f"tau {r['cfg']['tau']}, {r['cfg']['max_epochs']} ep")
    fig.suptitle("Discrete models on one data draw per process — "
                 + "; ".join(f"{k}: {v}" for k, v in per_arch.items())
                 + f" | K = {r0['cfg']['num_states_mult']}V, {r0['cfg']['num_samples']} x {r0['cfg']['seq_len']} "
                   f"tokens, seed {r0['cfg']['random_seed']}", fontsize=11)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# writing them out
# ══════════════════════════════════════════════════════════════════════════
def _save(fig, path):
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_dir(fig_root, process, arch):
    """Where one architecture's figures on one process go: <fig_root>/<process>/<arch>/."""
    d = os.path.join(fig_root, process, arch)
    os.makedirs(d, exist_ok=True)
    return d


def save_run_figures(r, fig_root):
    """training / states / machine graph for one run ->
    <fig_root>/<process>/<arch>/<arm>_s<seed>_*.png; returns the paths."""
    stem = os.path.join(figure_dir(fig_root, r["process"], r["arch"]), f"{r['arm']}_s{r['cfg']['random_seed']}")
    return [_save(plot_training(r), f"{stem}_training.png"),
            _save(plot_states(r), f"{stem}_states.png"),
            _save(plot_machine_graph(r), f"{stem}_machine.png")]


def save_group_figures(runs, fig_root, theory=None):
    """per-position CE (with the theory floor when `theory` is given), transition
    matrices and complexity for both arms of one architecture on one process ->
    <fig_root>/<process>/<arch>/s<seed>_*.png."""
    r0 = runs[_arms(runs)[0]]
    stem = os.path.join(figure_dir(fig_root, r0["process"], r0["arch"]), f"s{r0['cfg']['random_seed']}")
    paths = [_save(plot_per_position(runs, theory), f"{stem}_per_position.png"),
             _save(plot_transition_comparison_arms(runs), f"{stem}_transitions.png"),
             _save(plot_complexity(runs), f"{stem}_complexity.png")]
    if all(runs[a].get("exact_machine") is not None for a in _arms(runs)):
        paths.append(_save(plot_transition_comparison_arms(runs, "exact"), f"{stem}_transitions_exact.png"))
    return paths


def save_process_figures(runs_by_arch, theory, tag, fig_root, seed):
    """<fig_root>/<process>/theory.png and theory_comparison_s<seed>.png."""
    d = os.path.join(fig_root, theory["process"])
    os.makedirs(d, exist_ok=True)
    return [_save(plot_process_theory(theory, tag), os.path.join(d, "theory.png")),
            _save(plot_theory_comparison(runs_by_arch, theory, tag), os.path.join(d, f"theory_comparison_s{seed}.png"))]


def save_scorecard(results, fig_root, name="scorecard"):
    """Every run of one seed on one page -> <fig_root>/<name>.png."""
    os.makedirs(fig_root, exist_ok=True)
    return _save(plot_scorecard(results), os.path.join(fig_root, f"{name}.png"))
