"""
Everything after training, for every model of this folder: held-out results, the machine a model has learned, how its
learned states relate to the true ones, and all the figures.  Used by walkthrough.ipynb and `python main.py report`.

WHICH ANALYSIS WORKS FOR WHICH MODEL  (applicability() prints this table for the model files present)
    report, test_table          every model.  Excess CE of every predictor on 4,096 test windows; over 200 future
                                tokens for every step rule (GRU / next-state network from the handoff, the feedback GRU
                                from the start).
    reference_levels            every process: the CE of predicting without memory ("remembers nothing") and from the
                                last token only ("last symbol only"), above the true process's.
    plot_learning_curves        every model (validation history of each predictor).
    plot_states_over_training   every model: states in use by each predictor, and the gap between the two highest state
                                scores at the last history position (the Transformer's; the feedback GRU's own).
    plot_extra_terms            models with an extra loss term: 3, 4, A (next-state network), 5a, 5b, C (inferred next
                                state), B (copy of the GRU's state), 1_L2, 2_L2 (distance D).
    state tables, plot_state_tables, plot_token_probs_by_true_state, true states with their own learned state,
    excess_by_true_state, plot_excess_by_true_state (the excess CE at each true state)
                                every model, on a process whose generating machine is its own epsilon-machine
                                (process.Process.is_epsilon_machine; then the sampler's states are the causal states).
    entry_tokens                every process: the tokens that lead into each state of its generating machine.
    machine (step rule)         models with a fixed step rule (state, token) -> next state: the GRU of 1, 2, 3, 5a, A, B,
                                1_L2, 2_L2; the next-state network of 3, 4, A; the feedback GRU.  Exact: every
                                (state, token) pair is tried.
    machine (state sequence)    models whose own predictions pick a state at every position: the Transformer of every
                                hybrid, the discrete Transformer, the feedback GRU.  Read off the states picked on data,
                                keeping how often each next state follows; need not be deterministic.
    transition_comparison, plot_machines, plot_machine_grid, plot_machine_summary
                                any machine above, on a process that is its own epsilon-machine.
    distance_D, state_agreement hybrids with a GRU (1, 2, 3, 5a, A, B, 1_L2, 2_L2): the distance between the
                                Transformer's and the GRU's state probabilities after the same tokens, and how often the
                                two pick the same state.
    inferred_entropy_on_true_machine   5a, 5b (one token), C (two tokens): the penalty's value on the true machine.
    copy_target_stats           B: what the copy target (the GRU's own state after the history) looks like.
    lock_in_trace               hybrids (a Transformer with a scoring layer): the first updates, measured after each.

Predictor names (training.py): transformer, gru_from_handoff, next_state_from_handoff, gru_from_start.
report() runs a model's measures() on 512 windows at a time (measures_in_batches): model C's three-step head on a
20-token process would otherwise need about 10 GB for the 4,096 test windows.
Excess CE = the predictor's CE minus the true process's CE on the same tokens, bits per token; 0 is perfect.
"""
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.patches import Rectangle
from scipy.sparse.csgraph import connected_components

import process as process_module
import training as TR
from causal_state_matrix import StateSelection

EVAL_SEEDS = dict(alignment=3303, test=4404, long=5505)
EVAL_ROWS = dict(alignment=4096, test=4096, long=1024)
LONG = 200                       # future tokens of the long windows
MIN_SHARE = 0.01                 # a state of a state-sequence machine is drawn if a true state is mostly in it or it
                                 # holds at least this share of the visits
REPORT_VERSION = 1

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, PURPLE, NEUTRAL, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#8c5a9e", "#898781", "#b8860b"
PREDICTOR_STYLE = {"transformer": ("Transformer: reads all tokens so far", BLUE),
                   "gru_from_handoff": ("GRU from the handoff", ORANGE),
                   "next_state_from_handoff": ("next-state network from the handoff", AQUA),
                   "gru_from_start": ("feedback GRU: reads every token from the start", PURPLE),
                   "node_table": ("node table: reads the last R tokens (de Bruijn)", GOLD)}
SEED_STYLE = {0: "-", 1: "--", 2: ":", 3: "-.", 4: (0, (1, 3))}
EXTRA_TERMS = {"next_entropy": "entropy of the next-state network's output (bits)",
               "next_predicts_transformer": "share: network's next state = Transformer's",
               "inferred_entropy": "entropy of the inferred next state (bits)",
               "inferred_predicts_transformer": "share: inferred next state = Transformer's",
               "copy_agree": "share: starting state = GRU's own state",
               "l2": "distance D between the two parts",
               "U_bits": "commutation entropy U of the reader's states (bits)",
               "head_agreement": "share: G's next state = the reader's"}
FLOOR = 1e-4


# ================================================================================================ which model, which analysis
def capabilities(model_name):
    """What a model has, read from a freshly built model of that file."""
    module = TR.load_model_module(model_name)
    model = TR.make_model(TR.settings_for(model_name))
    rules = list(model.step_rules())
    own = [p for p in ("transformer", "gru_from_start", "node_table")
           if p == module.MAIN_PREDICTOR or (p == "transformer" and hasattr(model, "transformer_selector"))
           or (p == "node_table" and hasattr(model, "reader"))]
    return dict(model=model_name, description=module.DESCRIPTION, main_predictor=module.MAIN_PREDICTOR,
                step_rules=rules, own_predictors=own, gru_hybrid=hasattr(model, "gru_selector"),
                transformer_scoring_layer=hasattr(model, "transformer_selector"),
                extra_measures=hasattr(model, "measures"), weight=TR.settings_for(model_name).weight)


def applicability():
    """Which analysis of this file applies to which model file present (a table, models x analyses)."""
    rows = []
    for name in TR.available_models():
        c = capabilities(name)
        rows.append({"model": name, "description": c["description"], "main predictor": c["main_predictor"],
                     "test results, learning curves, states over training": "yes",
                     "state tables (needs an epsilon-machine process)": "yes",
                     "machine from a step rule": ", ".join(c["step_rules"]) or "-",
                     "machine from the state sequence": ", ".join(c["own_predictors"]) or "-",
                     "extra-term curves": "yes" if c["extra_measures"] else "-",
                     "distance D": "yes" if c["gru_hybrid"] else "-",
                     "lock-in trace": "yes" if c["transformer_scoring_layer"] else "-"})
    return pd.DataFrame(rows).set_index("model")


# ================================================================================================ data
@lru_cache(maxsize=32)
def _windows(process_name, params_json, total_len, history_len, kind, rows, seed):
    proc = TR.get_process(process_name, params_json)
    x, st = proc.sampler(seed).draw(rows, total_len, states=True)
    return x, st, TR.exact_ce_bits(proc, x)


def windows(settings, kind):
    """(tokens, true states, exact CE) of the fixed held-out windows of a run's process and window size:
    'validation' (the run's own), 'alignment' (to match learned with true states), 'test', 'long' (H + 200 tokens)."""
    params = json.dumps(settings.params, sort_keys=True)
    if kind == "validation":
        return _windows(settings.process, params, settings.total_len, settings.history_len, kind,
                        settings.validation_rows, settings.validation_seed)
    length = settings.history_len + LONG if kind == "long" else settings.total_len
    return _windows(settings.process, params, length, settings.history_len, kind, EVAL_ROWS[kind], EVAL_SEEDS[kind])


def reference_levels(settings, kind="test"):
    """Excess CE over the future tokens of two predictors that need no training: 'remembers nothing' gives each token
    its long-run frequency; 'last symbol only' predicts from the previous token alone with the true pair frequencies.
    A learned predictor near these levels does not use the process's memory."""
    proc = settings.process_object()
    M, pi, H = proc.M, proc.pi, settings.history_len
    p_x = np.einsum("s,stx->x", pi, M)
    p_xy = np.einsum("s,stx,tuy->xy", pi, M, M)
    q = p_xy / np.maximum(p_xy.sum(1, keepdims=True), 1e-300)
    x, _, exact = windows(settings, kind)
    x = x.numpy()
    exact_future = exact[:, H:].mean()
    nothing = -np.log2(p_x[x[:, H:]]).mean()
    last = -np.log2(q[x[:, H - 1:-1], x[:, H:]]).mean()
    return dict(remembers_nothing=float(nothing - exact_future), last_symbol_only=float(last - exact_future))


# ================================================================================================ machines
def by_true_state(states, true_states, n_states, n_true):
    """(learned state x true state) table of counts."""
    table = np.zeros((n_states, n_true), dtype=int)
    np.add.at(table, (np.asarray(states).ravel(), np.asarray(true_states).ravel()), 1)
    return table


def own_state_count(table):
    """True states with a learned state of their own: each true state goes to its most common learned state; the number
    of different learned states reached (= the number of true states when every one has its own)."""
    seen = table.sum(0) > 0
    return int(np.unique(table.argmax(0)[seen]).size)


def sent_to(table):
    """For every learned state (row), the true states sent to it (each true state to its most common learned state)."""
    major = table.argmax(0)
    return [[c for c in range(table.shape[1]) if table[:, c].sum() > 0 and major[c] == s] for s in range(len(table))]


@torch.no_grad()
def codebook_machine(model, step):
    """The deterministic machine a step rule defines on all K rows: delta[i, x] = next state, emission[i] = P(token | i),
    M[i, j, x] = P(token x, next state j | i)."""
    K, V = model.cfg.n_states, model.cfg.vocab_size
    emission = model.causal_head.token_probs().numpy().astype(float)
    delta = np.stack([step(torch.arange(K), torch.full((K,), x, dtype=torch.long)).numpy() for x in range(V)], 1)
    M = np.zeros((K, K, V))
    for i in range(K):
        for x in range(V):
            M[i, delta[i, x], x] = emission[i, x]
    return dict(delta=delta, emission=emission, M=M)


def recurrent_minimal_machine(raw, initial_states):
    """The part of a deterministic machine reached from `initial_states` that it keeps returning to (its closed classes),
    with states whose probabilities of every future are identical merged (process.minimise, tolerance 1e-10).  Copied
    from section 13 of hybrid_model_test.ipynb."""
    reached = set(map(int, np.unique(initial_states)))
    todo = list(reached)
    while todo:
        s = todo.pop()
        for x in np.flatnonzero(raw["emission"][s] > 0):
            successor = int(raw["delta"][s, x])
            if successor not in reached:
                reached.add(successor)
                todo.append(successor)
    reachable = np.array(sorted(reached), dtype=int)
    adjacency = raw["M"].sum(axis=-1)[np.ix_(reachable, reachable)] > 0
    n_components, labels = connected_components(adjacency, directed=True, connection="strong")
    closed = [reachable[labels == k] for k in range(n_components)
              if not adjacency[np.ix_(labels == k, labels != k)].any()]
    closed.sort(key=lambda ids: int(ids.min()))
    quotients, groups, component_ids = [], [], []
    raw_to_minimal = np.full(len(raw["delta"]), -1, dtype=int)
    offset = 0
    for c, ids in enumerate(closed):
        quotient, local_labels = process_module.minimise(raw["M"][np.ix_(ids, ids)], tol=1e-10)
        quotients.append(quotient)
        for k in range(len(quotient)):
            members = ids[local_labels == k]
            groups.append(members.tolist())
            component_ids.append(c)
            raw_to_minimal[members] = offset + k
        offset += len(quotient)
    minimal = np.zeros((offset, offset, raw["M"].shape[-1]))
    start = 0
    for quotient in quotients:
        stop = start + len(quotient)
        minimal[start:stop, start:stop] = quotient
        start = stop
    assert np.allclose(minimal.sum(axis=(1, 2)), 1)
    return dict(M=minimal, groups=groups, components=component_ids, raw_to_minimal=raw_to_minimal,
                reachable=reachable, n_closed_classes=len(closed), recurrent_raw=np.concatenate(closed))


def rollout(model, start_ids, future, step):
    """Step through `future` from `start_ids` with `step`: states (N, n) predicting each token and their logits."""
    E, W = model.causal_head.state_matrix, model.causal_head.emission
    ids, states, logits = start_ids, [], []
    for k in range(future.shape[1]):
        states.append(ids)
        logits.append(W(E[ids]))
        ids = step(ids, future[:, k])
    return torch.stack(states, 1), torch.stack(logits, 1)


@torch.no_grad()
def step_machine(model, settings, name):
    """The machine of step rule `name`, read off exactly: tried on every row, reduced to its recurrent part reached from
    the model's states after the history on the alignment windows, identical states merged.  pi = its stationary
    distribution (closed classes weighted by the share of rollouts that end in them).  Each state is named by the true
    states sent to it while the rule steps through the alignment windows' future tokens."""
    step = model.step_rules()[name]
    x, st, _ = windows(settings, "alignment")
    H, T = settings.history_len, settings.total_len
    start = model.handoff_states(x[:, :H], settings.tau_end)
    rec = recurrent_minimal_machine(codebook_machine(model, step), start.numpy())
    n = len(rec["M"])
    states, _ = rollout(model, start, x[:, H:], step)
    minimal = rec["raw_to_minimal"][states.numpy()]                                  # -1: a transient state
    on = minimal >= 0
    component = np.array(rec["components"])
    ends = component[minimal[:, -1][minimal[:, -1] >= 0]]
    P = rec["M"].sum(-1)
    pi = np.zeros(n)
    for c in np.unique(component):
        ids = np.flatnonzero(component == c)
        share = (ends == c).mean() if len(ends) else 1.0 / len(np.unique(component))
        pi[ids] = share * process_module.stationary_dist(P[np.ix_(ids, ids)])
    table = by_true_state(minimal[on], st[:, H:T][on], n, len(settings.process_object().T))
    return dict(kind="step rule", predictor=name, M=rec["M"], pi=pi, by_true_state=table, sent_to=sent_to(table),
                n_states=n, n_closed_classes=rec["n_closed_classes"])


@torch.no_grad()
def sequence_machine(model, settings, name, min_share=MIN_SHARE):
    """The machine read off the states predictor `name` picks at every position of the alignment windows, keeping how
    often each successor follows: M[s, s', x] = P(x | s) (the output layer) x the share of (s, x) visits followed by s'.
    `determinism` = share of steps that go to the most common successor (1 for a deterministic machine).  Kept: every
    state a true state is mostly in, and any other with at least min_share of the visits."""
    x, st, _ = windows(settings, "alignment")
    p = model.predict(x, settings.tau_end)["predictors"][name]
    first = p["first"]
    ids = p["states"].numpy()
    tokens = x.numpy()[:, first - 1:first - 1 + ids.shape[1]]            # the token read at each state's position
    K, V = model.cfg.n_states, model.cfg.vocab_size
    true = st[:, first:first + ids.shape[1]]
    emission = model.causal_head.token_probs().numpy().astype(float)
    visits = np.bincount(ids.ravel(), minlength=K)
    table = by_true_state(ids, true, K, len(settings.process_object().T))
    sent = sent_to(table)
    keep = np.flatnonzero((visits >= min_share * visits.sum()) | np.array([len(c) > 0 for c in sent]))
    triples = np.zeros((K, V, K))
    np.add.at(triples, (ids[:, :-1].ravel(), tokens[:, 1:].ravel(), ids[:, 1:].ravel()), 1)
    follow = triples / np.maximum(triples.sum(-1, keepdims=True), 1)
    M = np.einsum("sx,sxt->stx", emission, follow)[np.ix_(keep, keep)]
    return dict(kind="state sequence", predictor=name, M=M, pi=visits[keep] / visits.sum(), by_true_state=table[keep],
                sent_to=[sent[s] for s in keep], n_states=len(keep), codebook_ids=keep.tolist(),
                states_used=int((visits > 0).sum()), determinism=TR.determinism(ids, tokens[:, 1:]))


def machine(folder, which="latest", predictor=None, force=False):
    """The machine of one predictor of a saved run (default: the predictor its loss trains on the future tokens), from a
    step rule when the predictor has one, otherwise from its state sequence.  Cached in the run folder."""
    folder = Path(folder)
    model, settings, update = TR.load_run(folder, which)
    predictor = predictor or TR.load_model_module(settings.model).MAIN_PREDICTOR
    path = folder / f"machine_{which}_{predictor}.json"
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        if cached["update"] == update and cached.get("version") == REPORT_VERSION:
            return cached
    m = (step_machine(model, settings, predictor) if predictor in model.step_rules()
         else sequence_machine(model, settings, predictor))
    out = dict(version=REPORT_VERSION, update=update, which=which,
               **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in m.items()})
    path.write_text(json.dumps(out))
    return out


def transition_comparison(machine_, settings):
    """A machine's state-to-state matrix laid out on the true states, against the true one (the process must be its own
    epsilon-machine).  Each learned state is put in the row of the first true state sent to it, or (a state no true
    state is mostly in) of the true state it is most often at; rows of the states in one row are mixed by pi and their
    columns added.  A true state no learned state is assigned to is 'missing': its row is NaN and is left out of the
    errors; covered = the true states that are not missing.  As extraction.compare_transition_matrix of
    model_modeularised, but matched by data rather than by the nearest token probabilities."""
    proc = settings.process_object()
    if not proc.is_epsilon_machine:
        raise ValueError(f"{proc.label}'s generating machine is not its own epsilon-machine: no true states to compare")
    true = proc.M.sum(-1)
    n_true = len(true)
    T = np.asarray(machine_["M"]).sum(-1)
    pi = np.asarray(machine_["pi"], dtype=float)
    table = np.asarray(machine_["by_true_state"])
    assignment = np.array([c[0] if c else (int(row.argmax()) if row.sum() > 0 else -1)
                           for c, row in zip(machine_["sent_to"], table)], dtype=int)
    aggregated = np.full((n_true, n_true), np.nan)
    for c in range(n_true):
        members = np.flatnonzero(assignment == c)
        if len(members) == 0:
            continue
        w = pi[members] / pi[members].sum() if pi[members].sum() > 0 else np.full(len(members), 1.0 / len(members))
        row = w @ T[members]
        aggregated[c] = [row[assignment == u].sum() for u in range(n_true)]
    abs_error = np.abs(aggregated - true)
    present = ~np.isnan(abs_error[:, 0])
    return dict(aggregated=aggregated, true=true, abs_error=abs_error,
                missing=[c for c in range(n_true) if not present[c]], covered=int(present.sum()),
                max_error=float(abs_error[present].max()) if present.any() else float("nan"),
                mean_error=float(abs_error[present].mean()) if present.any() else float("nan"))


def entry_tokens(settings):
    """For every state of the process's generating machine, the tokens on which some state moves into it (for the
    flower: the hub is entered on a face going forward, on a die going backward)."""
    M = settings.process_object().M
    return [np.flatnonzero(M[:, s, :].sum(0) > 0).tolist() for s in range(M.shape[1])]


@torch.no_grad()
def excess_by_true_state(folder, which="latest"):
    """Excess CE of every predictor at each true state: over the future tokens of the test windows, the mean of (the
    predictor's CE - the true process's CE) at the positions where the generating machine is in that state just
    before the token.  Rows: true states; columns: predictors, and `visits` (positions per state)."""
    model, settings, _ = TR.load_run(folder, which)
    x, st, exact = windows(settings, "test")
    H, T, n_true = settings.history_len, settings.total_len, len(settings.process_object().T)
    true = st[:, H:T]
    out = {"visits": np.bincount(true.ravel(), minlength=n_true)}
    for name, p in model.predict(x, settings.tau_end)["predictors"].items():
        first, n = p["first"], p["logits"].shape[1]
        excess = TR.ce_bits(p["logits"], x[:, first:first + n]).numpy()[:, H - first:] - exact[:, H:T]
        sums = np.bincount(true.ravel(), weights=excess.ravel(), minlength=n_true)
        out[name] = np.where(out["visits"] > 0, sums / np.maximum(out["visits"], 1), np.nan)
    return pd.DataFrame(out, index=settings.process_object().state_names)


def merge_cost(settings, groups):
    """The excess CE (bits per token, averaged over all positions) of predicting, at every true state of a group, the
    long-run mixture of the group's next-token distributions instead of the state's own: the sum over groups of
    pi_s KL(p_s || mixture) over the group's states s.  With a learned machine's groups (its sent_to), this is the
    excess its merges cause by themselves.  On the flower, whose state is fixed by the last token, a machine that merges
    these groups and otherwise predicts exactly has exactly this excess."""
    proc = settings.process_object()
    p, pi = proc.emission(), proc.pi
    total = 0.0
    for group in groups:
        g = list(group)
        if len(g) < 2:
            continue
        mixture = (pi[g, None] * p[g]).sum(0) / pi[g].sum()
        for state in g:
            q = p[state]
            on = q > 0
            total += pi[state] * float(np.sum(q[on] * np.log2(q[on] / mixture[on])))
    return total


# ================================================================================================ reports
MEASURE_ROWS = 512               # windows per call of a model's measures() (measures_in_batches)


def _rows(value, part, n):
    """`value` with every tensor whose first axis has length n cut to the rows `part` (dicts are followed)."""
    if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == n:
        return value[part]
    if isinstance(value, dict):
        return {k: _rows(v, part, n) for k, v in value.items()}
    return value


def measures_in_batches(model, rows=MEASURE_ROWS):
    """Make the model's measures() run on `rows` windows at a time.  Every measure is a mean over windows (and
    positions), so the mean of the batches weighted by their size is the same number up to rounding.  Without this,
    model C on the 20-token flower process needs windows x positions x 20^3 numbers at once (about 10 GB for the
    4,096 test windows)."""
    if not hasattr(model, "measures") or getattr(model, "measures_in_batches", False):
        return model
    if getattr(sys.modules.get(type(model).__module__), "MEASURES_ONE_PASS", False):
        return model                                   # (its measures are not means over windows)
    whole = model.measures

    def measures(tokens, tau, prediction):
        n = len(tokens)
        if n <= rows:
            return whole(tokens, tau, prediction)
        sums = {}
        for start in range(0, n, rows):
            part = slice(start, min(start + rows, n))
            for k, v in whole(tokens[part], tau, _rows(prediction, part, n)).items():
                sums[k] = sums.get(k, 0.0) + v * (part.stop - part.start)
        return {k: v / n for k, v in sums.items()}

    model.measures = measures
    model.measures_in_batches = True
    return model


@torch.no_grad()
def evaluate_run(model, settings):
    """All held-out results of one model (see report)."""
    H, T = settings.history_len, settings.total_len
    tau = settings.tau_end
    proc = settings.process_object()
    n_true = len(proc.T)
    x, st, exact = windows(settings, "test")
    metrics = TR.evaluate(measures_in_batches(model), settings, x, st, tau, exact)
    tables, curves = {"token_probs_by_state": model.causal_head.token_probs().tolist()}, {}
    prediction = model.predict(x, tau)
    for name, p in prediction["predictors"].items():
        first, n = p["first"], p["logits"].shape[1]
        states = p["states"].numpy()
        table = by_true_state(states, st[:, first:first + states.shape[1]], model.cfg.n_states, n_true)
        tables[f"{name}_by_true_state"] = table.tolist()
        metrics[f"{name}_true_states_with_own_state"] = own_state_count(table)
        probs = p["logits"].softmax(-1).numpy()
        true_pred = st[:, first:first + n]
        curves[f"{name}_token_probs_by_true_state"] = [
            (probs[true_pred == a].mean(0).tolist() if (true_pred == a).any() else None) for a in range(n_true)]
    xl, sl, exact_l = windows(settings, "long")
    xa, _, _ = windows(settings, "alignment")
    for name, step in model.step_rules().items():
        if name == "gru_from_start":                      # reads every token from the start of the window
            metrics[f"long_excess_{name}"] = TR.evaluate(model, settings, xl, sl, tau, exact_l)[f"excess_{name}_future"]
        else:
            start = model.handoff_states(xl[:, :H], tau)
            _, logits = rollout(model, start, xl[:, H:], step)
            ce = TR.ce_bits(logits, xl[:, H:]).numpy()
            metrics[f"long_excess_{name}"] = float(ce.mean() - exact_l[:, H:].mean())
        raw = codebook_machine(model, step)
        rec = recurrent_minimal_machine(raw, model.handoff_states(xa[:, :H], tau).numpy())
        metrics[f"{name}_long_run_states"] = int(len(rec["M"]))
        metrics[f"{name}_closed_classes"] = int(rec["n_closed_classes"])
        tables[f"{name}_next_state"] = raw["delta"].tolist()
    if hasattr(model, "gru_selector"):
        metrics["distance_D"] = distance_D(model, settings, x, tau)
    return metrics, curves, tables


def report(folder, which="latest", force=False):
    """Held-out results of a saved run's parameters, cached in <folder>/report_<which>.json:
        metrics   training.evaluate on the 4,096 test windows (excess CE of every predictor, states in use, ...),
                  <predictor>_true_states_with_own_state, long_excess_<step rule> over 200 future tokens,
                  <step rule>_long_run_states (its machine's size), distance_D (hybrids with a GRU)
        curves    <predictor>_token_probs_by_true_state: its mean predicted P(next token) at each true state
        tables    <predictor>_by_true_state, token_probs_by_state (every row), <step rule>_next_state"""
    folder = Path(folder)
    model, settings, update = TR.load_run(folder, which)
    path = folder / f"report_{which}.json"
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        if cached["update"] == update and cached.get("version") == REPORT_VERSION:
            return cached
    metrics, curves, tables = evaluate_run(model, settings)
    out = dict(version=REPORT_VERSION, which=which, update=update, model=settings.model, seed=settings.seed,
               weight=settings.weight, metrics=metrics, curves=curves, tables=tables)
    path.write_text(json.dumps(out, indent=2))
    return out


def test_table(folders, which="latest"):
    """One row per run: the model, seed, weight, update, and every number of its report."""
    rows = []
    for label, folder in _items(folders):
        rep = report(folder, which)
        rows.append(dict(run=label, group=group_label(TR.read_settings(folder)), model=rep["model"], seed=rep["seed"],
                         weight=rep["weight"], update=rep["update"], **rep["metrics"]))
    return pd.DataFrame(rows)


def machine_table(folders, which="latest"):
    """One row per run: the machine of its main predictor, its size, the true states it covers, its difference from the
    true transition matrix, the excess its merges cause by themselves (merge_cost of its sent_to groups), and the test
    excess CE of that predictor."""
    rows = []
    for label, folder in _items(folders):
        _, settings, _ = TR.load_run(folder, which)
        m, rep = machine(folder, which), report(folder, which)
        cmp = transition_comparison(m, settings)
        rows.append(dict(run=label, group=group_label(settings), model=settings.model, seed=settings.seed,
                         weight=settings.weight, update=m["update"],
                         machine=f"{m['predictor']} ({m['kind']})", states=m["n_states"], covered=cmp["covered"],
                         mean_error=cmp["mean_error"], max_error=cmp["max_error"],
                         merge_cost=merge_cost(settings, m["sent_to"]),
                         test_excess=rep["metrics"][f"excess_{m['predictor']}_future"]))
    return pd.DataFrame(rows)


def first_below(folders, levels=(0.04, 0.02, 0.01)):
    """For every run, the first validation update at which its main predictor's excess fell below each level."""
    rows = []
    for label, folder in _items(folders):
        settings = TR.read_settings(folder)
        h, metric = TR.history(folder), TR.main_metric(settings.model)
        row = dict(run=label, model=settings.model, seed=settings.seed, final_update=int(h["update"].max()),
                   best=float(h[metric].min()), best_update=int(h.loc[h[metric].idxmin(), "update"]))
        for level in levels:
            below = h[h[metric] < level]
            row[f"below {level:g}"] = int(below["update"].iloc[0]) if len(below) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _items(folders):
    """[(name, folder)] from a dict {name: folder} or a list of folders; a None folder stays None."""
    if isinstance(folders, dict):
        return [(str(k), Path(v) if v is not None else None) for k, v in folders.items()]
    return [(Path(f).name, Path(f)) for f in folders]


# ================================================================================================ model-specific
@torch.no_grad()
def distance_D(model, settings, tokens, tau, steps=None):
    """The distance D of hybrid_model_test.ipynb's section 8: after each future token, the Euclidean distance between the
    Transformer's state probabilities (it reads all tokens so far) and the GRU's (stepped from the Transformer's state
    after the history), without noise, averaged over windows and the first `steps` future tokens (default: all L)."""
    was = model.training
    model.eval()
    try:
        H, steps = settings.history_len, steps or settings.rollout_len
        values = []
        for batch in tokens.split(256):
            _, sel = model.transformer_pass(batch, tau)
            state = StateSelection(sel.probs[:, H - 1], sel.onehot[:, H - 1], sel.ids[:, H - 1],
                                   sel.representation[:, H - 1])
            q = []
            for k in range(steps):
                state = model.transition(state, batch[:, H + k], tau)
                q.append(state.probs)
            values.append(torch.linalg.vector_norm(sel.probs[:, H:H + steps] - torch.stack(q, 1), ord=2, dim=-1))
        return float(torch.cat(values).mean())
    finally:
        model.train(was)


@torch.no_grad()
def state_agreement(folder, which="latest", rows=1024):
    """Hybrids with a GRU: after each of the first L - 1 future tokens of `rows` test windows, without noise, compare the
    GRU's state (stepped from the Transformer's state after the history) with the Transformer's state at the same
    position: the share of positions where they pick the same state, the mean distance D between their probability
    vectors, and the mean largest probability of each (near 1: the choice is nearly certain)."""
    model, settings, _ = TR.load_run(folder, which)
    x, _, _ = windows(settings, "test")
    x = x[:rows]
    H, L, tau = settings.history_len, settings.rollout_len, settings.tau_end
    _, sel = model.transformer_pass(x, tau)
    state = StateSelection(sel.probs[:, H - 1], sel.onehot[:, H - 1], sel.ids[:, H - 1], sel.representation[:, H - 1])
    q_gru, ids_gru = [], []
    for k in range(L - 1):
        state = model.transition(state, x[:, H + k], tau)
        q_gru.append(state.probs)
        ids_gru.append(state.ids)
    q_gru, ids_gru = torch.stack(q_gru, 1), torch.stack(ids_gru, 1)
    q_t, ids_t = sel.probs[:, H:H + L - 1], sel.ids[:, H:H + L - 1]
    return dict(same_state=float((ids_t == ids_gru).float().mean()),
                distance_D=float(torch.linalg.vector_norm(q_t - q_gru, dim=-1).mean()),
                largest_probability_transformer=float(q_t.max(-1).values.mean()),
                largest_probability_gru=float(q_gru.max(-1).values.mean()))


def inferred_entropy_on_true_machine(settings, tokens_ahead):
    """The penalty of models 5a, 5b (tokens_ahead = 1) and C (2) evaluated on the true machine, with every head
    predicting exactly: the entropy of the inferred next state, averaged over the long-run state and the next token.
    Also returns the crypticity C_mu - E, below which it cannot fall for the true machine however many tokens are
    used (a mixture of distributions has at least their average entropy; Mahoney, Ellison & Crutchfield 2009)."""
    proc = settings.process_object()
    T, E, pi, V = proc.T, proc.E, proc.pi, proc.vocab_size
    n = len(T)

    def nxt(s, x):
        p = T[s] * E[s, :, x]
        return int(np.argmax(p)) if p.sum() > 0 else None

    def words(s, L):
        out = np.zeros(V ** L)
        for idx in range(V ** L):
            digits = [(idx // V ** (L - 1 - j)) % V for j in range(L)]
            p, state = 1.0, s
            for d in digits:
                pd_ = (T[state] * E[state, :, d]).sum()
                p *= pd_
                if p == 0:
                    break
                state = nxt(state, d)
            out[idx] = p
        return out

    like = np.array([words(k, tokens_ahead) for k in range(n)])
    post = like * pi[:, None]
    post = post / np.maximum(post.sum(0, keepdims=True), 1e-300)
    total = 0.0
    for s in range(n):
        for x in range(V):
            p_x = (T[s] * E[s, :, x]).sum()
            if p_x == 0:
                continue
            q = words(nxt(s, x), tokens_ahead)
            r = post @ q
            r = r[r > 0]
            total += pi[s] * p_x * float(-(r * np.log2(r)).sum())
    a = proc.analytics()
    return dict(true_machine_entropy=total, crypticity=a["crypticity"], statistical_complexity=a["statistical_c"],
                excess_entropy=a["excess_entropy"])


@torch.no_grad()
def copy_target_stats(folder, which="latest"):
    """Model B: on the test windows, how many different values the copy target (the GRU's own state after the history,
    from row 0) takes, how many true states have their own target value, and the share of windows whose starting state
    equals it; at the saved parameters and at the starting weights."""
    model, settings, update = TR.load_run(folder, which)
    x, st, _ = windows(settings, "test")
    H, n_true = settings.history_len, len(settings.process_object().T)
    rows = []
    for label, m in (("starting weights", TR.make_model(settings).eval()), (f"update {update:,}", model)):
        target = m.gru_state_after_history(x[:, :H]).numpy()
        start = m.handoff_states(x[:, :H], settings.tau_end).numpy()
        table = by_true_state(target, st[:, H], m.cfg.n_states, n_true)
        rows.append(dict(parameters=label, target_values=len(np.unique(target)),
                         true_states_with_own_target=own_state_count(table), start_equals_target=float((start == target).mean())))
    return pd.DataFrame(rows)


def lock_in_trace(settings, updates=500):
    """The first `updates` updates of a hybrid run with these settings, measured after every update on the validation
    windows: the median gap between the two highest state scores at the last history position, the mean length of the
    encoder's output vectors (before any LayerNorm), the number of states the Transformer picks, and the mean P(token 1)
    of the picked states.  (Section 16.1 of the notebook.)"""
    model = TR.make_model(settings)
    optimizer = TR.make_optimizer(model, settings)
    proc = settings.process_object()
    source = proc.sampler(settings.train_seed)
    x, _, _ = windows(settings, "validation")
    H, C = settings.history_len, settings.context

    @torch.no_grad()
    def probe():
        was = model.training
        model.eval()
        values = model.encoder(x)
        length = float(values.norm(dim=-1)[:, C:].mean())
        if model.encoder_norm is not None:
            values = model.encoder_norm(values)
        scores = model.transformer_selector(values)
        top2 = scores[:, H - 1].topk(2, dim=-1).values
        ids = scores.argmax(-1)[:, C:]
        p1 = model.causal_head.emission(model.causal_head.state_matrix[ids]).softmax(-1)[..., 1]
        model.train(was)
        return dict(handoff_gap=float((top2[:, 0] - top2[:, 1]).median()), encoder_length=length,
                    states_used=int(ids.unique().numel()), picked_p1=float(p1.mean()))

    rows = [dict(update=0, tau=settings.tau_start, **probe())]
    schedule = settings.schedule_steps or settings.steps
    for step in range(updates):
        tau = TR.temperature(step, schedule, settings.tau_start, settings.tau_end)
        out = TR.train_step(model, source.draw(settings.batch_size, settings.total_len), optimizer, tau, settings.weight,
                            settings.grad_clip, record_grads=True)
        rows.append(dict(update=step + 1, tau=tau, loss=out["loss"],
                         grad_transformer_selector=out.get("grad_transformer_selector"), **probe()))
    return pd.DataFrame(rows)


# ================================================================================================ figures
def style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.4,
                         "axes.titlesize": 9.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.5, "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False})


def ramp(colour):
    rgb = np.array(to_rgb(colour))
    return LinearSegmentedColormap.from_list("ramp", [to_rgb(SURFACE), rgb, rgb * 0.55])


def group_label(settings):
    """A name for the runs that differ only in their seed: the model, its weight, and any setting that differs from the
    model's defaults that matters for the comparison (training setup, window sizes)."""
    s, base = settings, TR.settings_for(settings.model)
    parts = [f"model {s.model}"]
    if s.weight:
        parts.append(f"weight {s.weight:g}")
    if (s.encoder_norm, s.weight_decay, s.grad_clip) != (base.encoder_norm, base.weight_decay, base.grad_clip):
        original = not s.encoder_norm and s.weight_decay == 0 and s.grad_clip == 0
        parts.append("original setup" if original else "other training settings")
    if (s.history_len, s.rollout_len) != (base.history_len, base.rollout_len):
        parts.append(f"H {s.history_len}, L {s.rollout_len}")
    if s.process_object().label != base.process_object().label:
        parts.append(s.process_object().label)
    return ", ".join(parts)


def _group(folders):
    """{group label: [(seed, folder), ...]} in the given order."""
    groups = {}
    for _, folder in _items(folders):
        s = TR.read_settings(folder)
        groups.setdefault(group_label(s), []).append((s.seed, folder))
    return groups


def draw_curves(ax, runs, levels=None):
    """Validation excess CE of every predictor of a model's runs [(seed, folder)], one line style per seed."""
    for seed, folder in runs:
        h = TR.history(folder)
        u = h["update"] / 1000
        for name, (label, colour) in PREDICTOR_STYLE.items():
            col = f"excess_{name}_future"
            if col in h:
                ax.plot(u, h[col].clip(lower=FLOOR), color=colour, ls=SEED_STYLE.get(seed, "-"), lw=1.1,
                        label=label if seed == runs[0][0] else None)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 0.7)
    for value in (levels or {}).values():
        if value > 0:                                    # (a level of 0, e.g. 'last symbol only' on the flower)
            ax.axhline(value, color=NEUTRAL, lw=0.9, ls=(0, (4, 2)))
    ax.set_xlabel("update (thousands)")


def plot_learning_curves(folders, title="Validation excess CE of every predictor (seeds: solid, dashed, dotted)"):
    """One panel per model: the validation excess CE of each predictor over training; the grey dashed lines are the
    'remembers nothing' and 'last symbol only' levels on the validation windows."""
    style()
    groups = _group(folders)
    n = len(groups)
    ncols = min(4, n)
    fig, axes = plt.subplots(math.ceil(n / ncols), ncols, figsize=(3.1 * ncols, 2.4 * math.ceil(n / ncols)),
                             squeeze=False, layout="constrained")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, (m, runs) in zip(axes.flat, groups.items()):
        draw_curves(ax, runs, reference_levels(TR.read_settings(runs[0][1]), "validation"))
        ax.set_title(m, fontsize=8.5)
    axes.flat[0].set_ylabel("validation excess CE (bits)")
    handles = {}
    for ax in axes.flat:
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lb, hd)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=min(4, len(handles)))
    fig.suptitle(title, fontsize=10.5)
    return fig


def plot_states_over_training(folders):
    """Top: states in use by each predictor on the validation windows.  Bottom: the gap between the two highest state
    scores at the last history position; during training another state is picked there with probability about
    exp(-gap), so above a gap of about 10 the choice no longer changes."""
    style()
    groups = _group(folders)
    fig, axes = plt.subplots(2, len(groups), figsize=(2.6 * len(groups), 4.6), sharex="col", squeeze=False,
                             layout="constrained")
    for j, (m, runs) in enumerate(groups.items()):
        for seed, folder in runs:
            h = TR.history(folder)
            u = h["update"] / 1000
            for name, (label, colour) in PREDICTOR_STYLE.items():
                if f"{name}_states_used" in h:
                    axes[0, j].plot(u, h[f"{name}_states_used"], color=colour, ls=SEED_STYLE.get(seed, "-"), lw=1.0,
                                    label=label if seed == runs[0][0] else None)
            if "handoff_margin_median" in h:
                axes[1, j].plot(u, h["handoff_margin_median"].clip(lower=0.01), color=INK2, ls=SEED_STYLE.get(seed, "-"),
                                lw=1.0)
        axes[0, j].set_title(m, fontsize=7.5)
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("update (thousands)")
    axes[0, 0].set_ylabel("states in use")
    axes[1, 0].set_ylabel("score gap at the last\nhistory position")
    axes[0, 0].legend(loc="upper left", fontsize=6.5)
    fig.suptitle("States in use and the score gap over training", fontsize=10.5)
    return fig


def plot_extra_terms(folders):
    """The model-specific measures on the validation windows over training (EXTRA_TERMS), one panel per measure."""
    style()
    groups = _group(folders)
    present = [c for c in EXTRA_TERMS if any(c in TR.history(f) for runs in groups.values() for _, f in runs)]
    if not present:
        raise ValueError("these runs have no extra-term measures")
    fig, axes = plt.subplots(1, len(present), figsize=(3.4 * len(present), 2.8), squeeze=False, layout="constrained")
    colours = dict(zip(groups, [ORANGE, BLUE, AQUA, PURPLE, INK2, NEUTRAL] * 3))
    for ax, col in zip(axes.flat, present):
        for m, runs in groups.items():
            for seed, folder in runs:
                h = TR.history(folder)
                if col in h:
                    ax.plot(h["update"] / 1000, h[col], color=colours[m], ls=SEED_STYLE.get(seed, "-"), lw=1.0,
                            label=m if seed == runs[0][0] else None)
        ax.set_title(EXTRA_TERMS[col], fontsize=8)
        ax.set_xlabel("update (thousands)")
        ax.legend(fontsize=6.5)
    fig.suptitle("Extra-term measures on the validation windows", fontsize=10.5)
    return fig


def plot_test_summary(table, levels, title="Test excess CE of every predictor (final parameters)"):
    """table: test_table(...).  One row per model; a marker per seed and a bar at the mean, for every predictor."""
    style()
    models = list(dict.fromkeys(table["group"]))[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(models) + 1.3), layout="constrained")
    markers = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v"}
    offsets = {"transformer": 0.24, "node_table": 0.24, "gru_from_handoff": 0.08, "next_state_from_handoff": -0.08,
               "gru_from_start": -0.24}
    for name, (label, colour) in PREDICTOR_STYLE.items():
        col = f"excess_{name}_future"
        if col not in table:
            continue
        for i, m in enumerate(models):
            rows = table[(table.group == m) & table[col].notna()]
            if rows.empty:
                continue
            y = i + offsets[name]
            for _, r in rows.iterrows():
                ax.scatter(max(r[col], FLOOR), y, marker=markers.get(int(r.seed), "o"), s=24, color=colour, zorder=3)
            mean = max(rows[col].mean(), FLOOR)
            ax.plot([mean, mean], [y - 0.07, y + 0.07], color=colour, lw=2.2)
        ax.scatter([], [], color=colour, label=label)
    for value in levels.values():
        if value > 0:
            ax.axvline(value, color=NEUTRAL, lw=0.9, ls=(0, (4, 2)))
    ax.set_xscale("log")
    largest = max([0.5] + [2 * v for v in levels.values()]
                  + [1.5 * float(table[c].max()) for c in table if c.startswith("excess_") and c.endswith("_future")])
    ax.set_xlim(FLOOR * 0.8, largest)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.set_xlabel("test excess CE (bits per token; below 1e-4 drawn at 1e-4); dashed: 'remembers nothing' / "
                  "'last symbol only'")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    ax.set_title(title, fontsize=10)
    return fig


def plot_token_probs_by_true_state(folders, which="latest", token=1):
    """Each predictor's mean predicted P(next token = token) at each true state (test windows), against the true value
    (grey).  One panel per run."""
    style()
    items = _items(folders)
    fig, axes = plt.subplots(math.ceil(len(items) / 3), min(3, len(items)), figsize=(3.6 * min(3, len(items)),
                             2.6 * math.ceil(len(items) / 3)), squeeze=False, layout="constrained")
    for ax in axes.flat[len(items):]:
        ax.set_visible(False)
    for ax, (label, folder) in zip(axes.flat, items):
        rep = report(folder, which)
        proc = TR.read_settings(folder).process_object()
        truth = proc.emission()[:, token]
        ax.plot(range(len(truth)), truth, color=NEUTRAL, lw=3, alpha=0.5, label="true")
        for name, (plabel, colour) in PREDICTOR_STYLE.items():
            curve = rep["curves"].get(f"{name}_token_probs_by_true_state")
            if curve:
                ax.plot(range(len(curve)), [c[token] if c else np.nan for c in curve], "o-", ms=3, color=colour,
                        lw=1, label=plabel)
        ax.set_title(f"model {rep['model']}, seed {rep['seed']}", fontsize=8.5)
        ticks = range(0, len(truth), 5 if len(truth) > 30 else 1)
        ax.set_xticks(ticks, [proc.state_names[t] for t in ticks], fontsize=6, rotation=90)
        ax.set_ylabel(f"P(next token = {token})")
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("Predicted next-token probability at each true state (test windows)", fontsize=10.5)
    return fig


def plot_excess_by_true_state(folders, which="latest", title="Excess CE at each true state (test windows, future tokens)",
                              main_only=False):
    """One panel per run: excess_by_true_state for every predictor (main_only: the model's main predictor only), and
    (grey bars, right axis) the share of positions at each true state."""
    style()
    items = _items(folders)
    ncols = min(3, len(items))
    fig, axes = plt.subplots(math.ceil(len(items) / ncols), ncols, figsize=(3.9 * ncols, 2.7 * math.ceil(len(items) / ncols)),
                             squeeze=False, layout="constrained")
    for ax in axes.flat[len(items):]:
        ax.set_visible(False)
    for ax, (label, folder) in zip(axes.flat, items):
        table, s = excess_by_true_state(folder, which), TR.read_settings(folder)
        x = np.arange(len(table))
        share = ax.twinx()
        share.bar(x, table["visits"] / table["visits"].sum(), color=GRID, width=0.8, zorder=0)
        share.set_ylabel("share of positions", fontsize=7)
        share.grid(False)
        ax.set_zorder(share.get_zorder() + 1)
        ax.patch.set_visible(False)
        main = TR.load_model_module(s.model).MAIN_PREDICTOR
        for name, (plabel, colour) in PREDICTOR_STYLE.items():
            if name in table and (name == main or not main_only):
                ax.plot(x, table[name], "o-", ms=2.5, lw=1, color=colour, label=plabel)
        ax.axhline(0, color=NEUTRAL, lw=0.8)
        ticks = range(0, len(table), 5 if len(table) > 30 else 1)
        ax.set_xticks(ticks, [table.index[t] for t in ticks], fontsize=6, rotation=90)
        ax.set_ylabel("excess CE (bits)")
        ax.set_title(f"model {s.model}, seed {s.seed}, {s.process_object().label}", fontsize=8.5)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle(title, fontsize=10.5)
    return fig


def plot_state_tables(folders, which="latest", predictors=None, min_share=0.01, main_only=False):
    """For each run and predictor: which learned states each true state is sent to (share of the true state's test
    positions in each learned state; rows ordered by the mean true state they are used at).  One learned state per true
    state gives a diagonal; one state covering several gives a horizontal run.  main_only: each model's main predictor."""
    style()
    items = _items(folders)
    reps = [(label, report(folder, which), TR.read_settings(folder)) for label, folder in items]
    pairs = [(label, rep, s, name) for label, rep, s in reps for name in PREDICTOR_STYLE
             if f"{name}_by_true_state" in rep["tables"] and (predictors is None or name in predictors)
             and (not main_only or name == TR.load_model_module(s.model).MAIN_PREDICTOR)]
    ncols = min(3, len(pairs))
    fig, axes = plt.subplots(math.ceil(len(pairs) / ncols), ncols, figsize=(3.9 * ncols, 3.2 * math.ceil(len(pairs) / ncols)),
                             squeeze=False, layout="constrained")
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    for ax, (label, rep, s, name) in zip(axes.flat, pairs):
        table = np.array(rep["tables"][f"{name}_by_true_state"], dtype=float)
        share = table / np.maximum(table.sum(0, keepdims=True), 1)
        keep = np.where(share.max(1) >= min_share)[0]
        mean_state = (table[keep] * np.arange(table.shape[1])).sum(1) / np.maximum(table[keep].sum(1), 1)
        rows = keep[np.argsort(mean_state)]
        ax.imshow(share[rows], cmap=ramp(PREDICTOR_STYLE[name][1]), vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        names = s.process_object().state_names
        ticks = range(0, len(names), 5 if len(names) > 30 else 1)
        ax.set_xticks(ticks, [names[t] for t in ticks], fontsize=6, rotation=90)
        ax.set_yticks(range(len(rows)), [str(r) for r in rows], fontsize=6 if len(rows) <= 40 else 4)
        ax.grid(False)
        ax.set_title(f"model {rep['model']} s{rep['seed']}: {PREDICTOR_STYLE[name][0]}\n"
                     f"{own_state_count(table.astype(int))} of {table.shape[1]} true states have their own state",
                     fontsize=7.5)
        ax.set_xlabel("true state")
        ax.set_ylabel("learned state")
    fig.suptitle("Which learned state each true state is sent to (share of its test positions)", fontsize=10.5)
    return fig


def _names(machine_, settings):
    names = settings.process_object().state_names
    out = []
    for c, row in zip(machine_["sent_to"], machine_["by_true_state"]):
        if c:
            parts, start = [], None
            for i, v in enumerate(c):
                start = v if start is None else start
                if i + 1 == len(c) or c[i + 1] != v + 1:
                    parts.append(names[start] if start == v else f"{names[start]}–{names[v]}")
                    start = None
            out.append(", ".join(parts))
        else:
            out.append(f"extra ({names[int(np.argmax(row))]})" if np.sum(row) else "extra")
    return out


def draw_machine(ax, machine_, settings, colour, min_p=0.01, edge_labels=True):
    """A machine as a graph: a node per state, named by the true states sent to it, with its long-run share pi and its
    moves back to itself; an arrow 'token|p' per move with probability p = P(token, next state | state) >= min_p
    (edge_labels=False: arrows only, for processes with many tokens)."""
    import networkx as nx
    M, pi = np.asarray(machine_["M"]), np.asarray(machine_["pi"])
    n = len(M)
    names = _names(machine_, settings)
    table = np.asarray(machine_["by_true_state"])
    mean_state = [np.dot(r, np.arange(len(r))) / max(np.sum(r), 1) for r in table]
    G = nx.DiGraph()
    G.add_nodes_from(sorted(range(n), key=lambda s: mean_state[s]))
    labels, loops = {}, {s: [] for s in range(n)}
    for s in range(n):
        for t in range(n):
            for x in range(M.shape[-1]):
                if M[s, t, x] < min_p:
                    continue
                label = f"{x}|{M[s, t, x]:.2f}"
                if s == t:
                    loops[s].append(label)
                else:
                    labels[(s, t)] = f"{labels[(s, t)]}\n{label}" if (s, t) in labels else label
                    G.add_edge(s, t)
    pos = nx.kamada_kawai_layout(G) if n > 1 else {0: np.zeros(2)}
    size = 2300 if n <= 4 else max(1000, 2600 - 110 * n)
    font = 7 if n <= 9 else 6
    tint = to_hex(0.22 * np.array(to_rgb(colour)) + 0.78 * np.array(to_rgb(SURFACE)))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=size, node_color=tint, edgecolors=colour, linewidths=1.2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font, font_color=INK,
                            labels={s: f"{names[s]}\nπ={pi[s]:.3f}" + ("\n↺ " + ", ".join(loops[s])
                                                                         if loops[s] and edge_labels else "")
                                    for s in range(n)})
    nx.draw_networkx_edges(G, pos, ax=ax, node_size=size, arrows=True, arrowstyle="-|>", arrowsize=10, width=0.9,
                           edge_color=INK2, connectionstyle="arc3,rad=0.12")
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels, font_size=font - 1, label_pos=0.35,
                                     connectionstyle="arc3,rad=0.12", font_color=INK2,
                                     bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.85))
    ax.set_axis_off()
    if n == 1:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    else:
        ax.margins(0.16)


def heat(ax, matrix, names, cmap, vmax, title):
    """A matrix on the true states, values >= 0.005 written in; a NaN row (a missing true state) is hatched."""
    A = np.asarray(matrix, dtype=float)
    ax.imshow(np.nan_to_num(A, nan=0.0), cmap=cmap, vmin=0, vmax=vmax, aspect="equal", interpolation="nearest")
    if len(names) <= 20:                                 # numbers only while they can be read
        for i, j in zip(*np.nonzero(np.nan_to_num(A) >= 0.005)):
            ax.text(j, i, f"{A[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=4.8,
                    color="white" if A[i, j] > 0.6 * vmax else INK)
    for i in np.flatnonzero(np.isnan(A[:, 0])):
        ax.add_patch(Rectangle((-0.5, i - 0.5), A.shape[1], 1, facecolor=GRID, edgecolor=NEUTRAL, hatch="////", lw=0))
    ticks = range(0, len(names), 5 if len(names) > 30 else 1)
    ax.set_xticks(ticks, [names[t].lstrip("as") if len(names) > 10 else names[t] for t in ticks], fontsize=6)
    ax.set_yticks(ticks, [names[t] for t in ticks], fontsize=6)
    ax.set_xlabel("to (true state)", fontsize=7)
    ax.set_ylabel("from", fontsize=7)
    ax.grid(False)
    ax.set_title(title, fontsize=8)


def true_machine(settings):
    """The true machine (the process's generating machine, when it is its own epsilon-machine) as a machine dict."""
    proc = settings.process_object()
    n = len(proc.T)
    return dict(M=proc.M, pi=proc.pi, sent_to=[[c] for c in range(n)], by_true_state=np.eye(n, dtype=int), n_states=n)


def plot_true_machine(settings):
    """The process's epsilon-machine drawn like the learned ones, and its state-to-state matrix."""
    style()
    proc = settings.process_object()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw=dict(width_ratios=[1.45, 1.0]), layout="constrained")
    labels = proc.vocab_size <= 20                          # (more tokens: too many labels to read)
    draw_machine(axes[0], true_machine(settings), settings, NEUTRAL, edge_labels=labels)
    axes[0].set_title(f"{len(proc.T)} states; " + ("arrow token|p = emit the token with probability p and move" if labels
                      else f"{proc.vocab_size} tokens, arrows without labels"), fontsize=8.5)
    heat(axes[1], proc.M.sum(-1), proc.state_names, ramp(NEUTRAL), 1.0, "its state-to-state transition matrix")
    fig.suptitle(f"The true ε-machine of {proc.label}", fontsize=10.5)
    return fig


def plot_machines(folders, which="latest", predictor=None, max_drawn=24):
    """One model's runs (one per seed): its learning curves, and for each run the machine of its main predictor (or
    `predictor`), that machine's state-to-state matrix laid out on the true states, and |that - the true matrix|
    (hatched rows: true states no learned state is assigned to).  The process must be its own epsilon-machine.  A
    machine with more than `max_drawn` states is not drawn as a graph (it could not be read)."""
    style()
    items = _items(folders)
    settings0 = TR.read_settings(items[0][1])
    names = settings0.process_object().state_names
    n_rows = max(3, len(items))
    fig = plt.figure(figsize=(22, 5.7 * n_rows), layout="constrained")
    outer = fig.add_gridspec(n_rows, 4, width_ratios=[1.05, 1.55, 1.0, 1.0])
    ax = fig.add_subplot(outer[0, 0])
    runs = [(TR.read_settings(f).seed, f) for _, f in items]
    draw_curves(ax, runs, reference_levels(settings0, "validation"))
    ax.set_ylabel("validation excess CE (bits)")
    ax.set_title("learning curves (seeds 0, 1, 2: solid, dashed, dotted)", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=6.5)
    cells, cols = [], None
    for row, (label, folder) in enumerate(items):
        settings = TR.read_settings(folder)
        m, rep = machine(folder, which, predictor), report(folder, which)
        cmp = transition_comparison(m, settings)
        colour = PREDICTOR_STYLE[m["predictor"]][1]
        extra = (f"\n{m['states_used']} states used; drawn: a true state's own state or ≥1% of visits"
                 f"\n{m['determinism']:.0%} of steps go to the most common next state" if m["kind"] == "state sequence"
                 else "")
        excess = rep["metrics"]["excess_" + m["predictor"] + "_future"]
        plural = "" if m["n_states"] == 1 else "s"
        axm = fig.add_subplot(outer[row, 1])
        if m["n_states"] <= max_drawn:
            draw_machine(axm, m, settings, colour)
        else:
            axm.set_axis_off()
            axm.text(0.5, 0.5, f"{m['n_states']} states: too many to draw as a graph\n(its matrix is on the right)",
                     ha="center", va="center", fontsize=9, color=INK2, transform=axm.transAxes)
        axm.set_title(f"seed {settings.seed}, update {m['update']:,}: {PREDICTOR_STYLE[m['predictor']][0]}\n"
                      f"machine ({m['kind']}): {m['n_states']} state{plural}, covers {cmp['covered']} of the "
                      f"{len(names)} true states\ntest excess CE {excess:.4f} bits" + extra, fontsize=8.5)
        heat(fig.add_subplot(outer[row, 2]), cmp["aggregated"], names, ramp(colour), 1.0,
             f"seed {settings.seed}: its transition matrix on the true states")
        err = cmp["abs_error"]
        missing = (", ".join(names[c] for c in cmp["missing"]) if len(cmp["missing"]) <= 8
                   else f"{len(cmp['missing'])} of the {len(names)} true states")
        heat(fig.add_subplot(outer[row, 3]), err, names, "Reds", max(0.05, float(np.nanmax(err))),
             f"seed {settings.seed}: |learned − true|, max {cmp['max_error']:.2f}, mean {cmp['mean_error']:.3f}"
             + (f"\nmissing: {missing}" if missing else ""))
        predictors = [k[len("excess_"):-len("_future")] for k in rep["metrics"]
                      if k.startswith("excess_") and k.endswith("_future")]
        cols = ["seed", "update", "states", "true states\ncovered"] + [p.replace("_from_", "\nfrom ") for p in predictors]
        cells.append([str(settings.seed), f"{m['update']:,}", str(m["n_states"]), f"{cmp['covered']} of {len(names)}"]
                     + [f"{rep['metrics'][f'excess_{p}_future']:.4f}" for p in predictors])
    axt = fig.add_subplot(outer[1, 0])
    axt.set_axis_off()
    table = axt.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 2.4)
    axt.set_title("test excess CE (bits) of every predictor", fontsize=8.5)
    axr = fig.add_subplot(outer[2, 0])
    if len(names) <= max_drawn:
        draw_machine(axr, true_machine(settings0), settings0, NEUTRAL)
        axr.set_title(f"for comparison: the true ε-machine, {len(names)} states", fontsize=8.5)
    else:
        axr.set_axis_off()
    fig.suptitle(f"{group_label(settings0)}: learning curves and the machine of each run", fontsize=11)
    return fig


def plot_machine_grid(folders, which="latest", predictor=None, kind="auto", ncols=3, max_drawn=24):
    """One panel per run (e.g. a row per model, a column per seed; a None folder leaves its panel empty): the machine of
    its main predictor (or `predictor`),
    as a graph when kind='graph', or kind='auto' and it has at most `max_drawn` states; otherwise its state-to-state
    matrix laid out on the true states (as in plot_machines).  Arrows carry 'token|p' labels only for processes with at
    most 4 tokens.  Titles: the machine's size, the true states it covers, the mean |learned - true| over the covered
    rows, and the predictor's test excess CE."""
    style()
    items = _items(folders)
    nrows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False, layout="constrained")
    for ax in axes.flat[len(items):]:
        ax.set_visible(False)
    for ax, (label, folder) in zip(axes.flat, items):
        if folder is None:                                   # (a run that has not finished yet)
            ax.set_axis_off()
            ax.set_title(f"{label}: not finished yet", fontsize=7.5, color=INK2)
            continue
        settings = TR.read_settings(folder)
        m, rep = machine(folder, which, predictor), report(folder, which)
        cmp = transition_comparison(m, settings)
        colour = PREDICTOR_STYLE[m["predictor"]][1]
        names = settings.process_object().state_names
        if kind == "graph" or (kind == "auto" and m["n_states"] <= max_drawn):
            draw_machine(ax, m, settings, colour, edge_labels=settings.vocab_size <= 4)
        else:
            heat(ax, cmp["aggregated"], names, ramp(colour), 1.0, "")
        excess = rep["metrics"][f"excess_{m['predictor']}_future"]
        ax.set_title(f"model {settings.model}, {settings.process_object().label}, seed {settings.seed} ({m['kind']}):\n"
                     f"{m['n_states']} states, covers {cmp['covered']} of {len(names)}, mean |learned − true| "
                     f"{cmp['mean_error']:.3f}, test excess {excess:.4f} bits", fontsize=7.5)
    return fig


def plot_machine_summary(table, levels):
    """table: machine_table(...).  Per run: machine states (hollow) and true states covered (filled), mean |learned -
    true| over the covered rows, and the test excess CE of the machine's predictor."""
    style()
    models = list(dict.fromkeys(table["group"]))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.5 * len(models) + 1.4), sharey=True, layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.2, 1.0, 1.0]))
    markers = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v"}
    for i, m in enumerate(models):
        rows = table[table.group == m]
        for j, (_, r) in enumerate(rows.iterrows()):
            y = i + (j - (len(rows) - 1) / 2) * 0.18
            colour = PREDICTOR_STYLE[r.machine.split(" ")[0]][1]
            mk = markers.get(int(r.seed), "o")
            axes[0].plot([r.covered, r.states], [y, y], color=colour, lw=0.8, alpha=0.6)
            axes[0].scatter(r.states, y, marker=mk, s=28, facecolor=SURFACE, edgecolor=colour, linewidth=1.1, zorder=3)
            axes[0].scatter(r.covered, y, marker=mk, s=28, color=colour, zorder=3)
            axes[1].scatter(r.mean_error, y, marker=mk, s=28, color=colour, zorder=3)
            axes[2].scatter(max(r.test_excess, FLOOR), y, marker=mk, s=28, color=colour, zorder=3)
    axes[0].set_yticks(range(len(models)), models, fontsize=8)
    axes[0].set_xlabel("machine states (hollow) and true states covered (filled)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("mean |learned − true| over the covered rows")
    for value in levels.values():
        if value > 0:
            axes[2].axvline(value, color=NEUTRAL, lw=0.9, ls=(0, (4, 2)))
    axes[2].set_xscale("log")
    axes[2].set_xlim(FLOOR * 0.8, max([0.5] + [2 * v for v in levels.values()]))
    axes[2].set_xlabel("test excess CE of that predictor (bits)")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    for name, (label, colour) in PREDICTOR_STYLE.items():
        axes[2].scatter([], [], color=colour, label=label)
    axes[2].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    fig.suptitle("The machine of every run: size, true states covered, difference from the true transitions, test excess",
                 fontsize=10.5)
    return fig


def save_run_figures(folder, which="latest"):
    """The main figures of one run, saved in <folder>/figures/: learning curves, machine, state tables."""
    folder = Path(folder)
    out = folder / "figures"
    out.mkdir(exist_ok=True)
    paths = []
    for name, fig in (("learning_curves", plot_learning_curves([folder])),
                      ("machine", plot_machines([folder], which)),
                      ("state_tables", plot_state_tables([folder], which))):
        path = out / f"{name}_{which}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
