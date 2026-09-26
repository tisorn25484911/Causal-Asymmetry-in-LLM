"""
Reading a trained DISCRETE model against the theory of the arm it was trained on.
Ported from Experimental_pipeline/extraction.py and the transformer notebook;
the theory now comes from process_generator.generator, in whichever view
(forward or reversed) the generator is in.

    theory_from_generator         closed forms of the generator's current view
    causal_state_report           states per position, occupancy, emissions, S_emp
    transition_matrix_extraction  T[i][j] = P(s_j | s_i), from free-running generation
    compare_transition_matrix     that matrix against the closed form, learned states
                                  mapped onto theoretical ones by their emission rows
    empirical_symbolic_machine    the (state, token) -> state machine the model became
    analyse_arm                   all of the above for one trained model -> one dict

Arms.  The runs in this folder take the backward arm from the DATA:
generator.reverse() flips the sequences and switches the theory to the reverse
machine M-, and the model is trained at mode="forward" on the flipped sequences.
The model's "next token" is then the previous token of the original process, its
transition matrix is one step back in original time, and both are compared with
the reverse machine -- which is also read left to right over the flipped data --
so no transposition is needed anywhere.
"""
import math

import numpy as np
import torch

from HMM_processes import entropy_bits, joint_map


# ══════════════════════════════════════════════════════════════════════════
# 0. THEORY
# ══════════════════════════════════════════════════════════════════════════
def theory_from_generator(gen) -> dict:
    """
    The closed forms a model trained on `gen`'s current view is compared with.

    true_machine is the view's epsilon-machine in (next_state, emission_probs)
    form, the format the empirical machine is read into:
        emission_probs[s, x] = P(x | s)              = sum_s' M[s, s', x]
        next_state[s, x]     = the unique s' with M[s, s', x] > 0
    (unifilar, so unique; a pair (s, x) that is never emitted gets a self-loop,
    the empirical machine's convention).  `min_emission_tv` is the smallest total
    variation between two theoretical emission rows: matching learned states by
    emission row (compare_transition_matrix) can only tell apart states whose
    rows differ, so it is checked, not assumed.
    """
    T = np.asarray(gen.transition, dtype=float)
    M = joint_map(T, gen.state_map)                  # (k, k, V)  P(x, s' | s)
    k, _, V = M.shape
    emission = M.sum(axis=1)                         # (k, V)
    next_state = np.tile(np.arange(k)[:, None], (1, V))
    for s in range(k):
        for x in range(V):
            succ = np.flatnonzero(M[s, :, x] > 0)
            if succ.size > 1:
                raise ValueError(f"not unifilar: state {s} token {x} -> {succ.tolist()}")
            if succ.size == 1:
                next_state[s, x] = int(succ[0])
    tv = [0.5 * np.abs(emission[a] - emission[b]).sum()
          for a in range(k) for b in range(a + 1, k)]
    return {
        "view": gen.direct, "C": float(gen.statistical_c), "true_k": int(gen.n_causal),
        "occupancy": np.asarray(gen.station_dist, dtype=float), "T_theory": T,
        "true_machine": {"next_state": next_state, "emission_probs": emission},
        "entropy_rate": float(gen.entropy_rate),
        "names": [f"σ{i}" for i in range(k)],
        "min_emission_tv": float(min(tv)) if tv else float("nan"),
    }


# ══════════════════════════════════════════════════════════════════════════
# 1. States, occupancy, emissions
# ══════════════════════════════════════════════════════════════════════════
def kept_positions(mode: str, T: int, min_pos: int) -> slice:
    """
    The positions with enough context to read a state from.  A forward model at
    position t has only t tokens of context, so its short end is the START; a
    mode="backward" model reads right to left, so its short end is the FINISH.
    """
    return slice(min_pos, None) if mode == "forward" else slice(0, max(T - min_pos, 1))


def causal_state_report(model, data_loader, min_pos: int = 5, device=None):
    """
    Assign every token position to a causal state and summarise.  Returns None
    for a model with no explicit states (the onehot decoders).
    """
    K = int(getattr(model, "n_states", 0) or 0)
    if K <= 0:
        return None

    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    V      = int(model.token_size)
    counts = torch.zeros(K, dtype=torch.long)
    joint  = torch.zeros(V, K, dtype=torch.long)       # for H(state | token)

    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = model._split(batch, model.mode)
            model(inputs.to(device))
            st = model.last_states
            keep = kept_positions(model.mode, st.shape[1], min_pos)
            st = st[:, keep].reshape(-1).cpu()
            counts += torch.bincount(st, minlength=K)
            tk = inputs[:, keep].reshape(-1).cpu()
            joint += torch.bincount(tk * K + st, minlength=V * K).reshape(V, K)
        vectors   = model.state_matrix.detach().cpu()
        emissions = model.emission_table().detach().cpu()

    if was_training:
        model.train()

    p = counts.numpy().astype(float)
    p = p / p.sum() if p.sum() > 0 else p

    # H(state | current token), in bits -- 0 for a machine synchronised by the last token.
    j   = joint.numpy().astype(float)
    tot = j.sum()
    h_cond = 0.0
    for row in j:
        n_t = row.sum()
        if n_t > 0:
            h_cond += (n_t / tot) * entropy_bits(row[row > 0] / n_t)

    return {
        "counts": counts.numpy(), "occupancy": p,
        "vectors": vectors.numpy(), "emissions": emissions.numpy(),
        "occupied": [k for k in range(K) if counts[k] > 0],
        "n_states": K, "n_states_used": int((counts > 0).sum()),
        "S_emp": entropy_bits(p), "h_state_given_token": float(h_cond),
        "joint_token_state": j,
    }


# ══════════════════════════════════════════════════════════════════════════
# 2. The learned state-to-state transition matrix
# ══════════════════════════════════════════════════════════════════════════
def transition_matrix_extraction(model, seed_tokens, *, total_run: int,
                                 window_size: int, rng=None) -> np.ndarray:
    """
    T[i][j] = P(s_{t+1} = j | s_t = i), read off the model's own generation.

    The model is seeded with `seed_tokens` -- one sequence drawn from the TRUE
    process, in the arm's reading order -- then rolls forward on its own
    samples; at each step the state at the newest position is recorded, and the
    counts are row-normalised.  Free-running, not teacher-forced: the matrix
    describes the process the model has BECOME.

    Identified only up to a PERMUTATION of the state labels, so compare it with
    a closed form through compare_transition_matrix.  `window_size` is the
    context it rolls forward on; pass the training seq_len.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    device = next(model.parameters()).device
    is_bw  = (model.mode == "backward")
    V      = int(model.token_size)
    inputs = torch.as_tensor(np.asarray(seed_tokens).reshape(1, -1), dtype=torch.long, device=device)

    window_size = min(int(window_size), int(model.max_len))
    if inputs.shape[1] < window_size:
        print(f"  ! transition seed is {inputs.shape[1]} tokens, "
              f"shorter than the {window_size}-token window")
    # The newest token, which is also the one with full context: LAST for a
    # forward model, FIRST for a mode="backward" one.
    pos = 0 if is_bw else -1

    was_training = model.training
    model.eval()
    transition = np.zeros((model.n_states, model.n_states))
    prev = None
    with torch.no_grad():
        for _ in range(total_run):
            inputs = inputs[:, :window_size] if is_bw else inputs[:, -window_size:]
            logits = model(inputs)
            cur = int(model.last_states[0, pos])
            if prev is not None:
                transition[prev][cur] += 1
            prev = cur

            p = torch.softmax(logits[0, pos], dim=-1).cpu().numpy().astype(np.float64)
            nxt = int(rng.choice(V, p=p / p.sum()))
            nxt = torch.tensor([[nxt]], dtype=torch.long, device=device)
            inputs = (torch.cat((nxt, inputs), dim=1) if is_bw
                      else torch.cat((inputs, nxt), dim=1))
    if was_training:
        model.train()

    row = transition.sum(axis=1, keepdims=True)
    # Unvisited states keep a zero row rather than raising.
    return np.divide(transition, row, out=np.zeros_like(transition), where=row > 0)


def match_permutation(learned: np.ndarray, theory: np.ndarray, occupied=None):
    """
    Best label permutation of `learned`'s visited states onto `theory` (brute
    force, exact at these sizes).  Returns (permuted_submatrix, chosen_indices,
    max_abs_error), or None when the number of visited states does not match
    the theoretical count.
    """
    import itertools
    if occupied is None:
        occupied = np.flatnonzero(learned.sum(axis=1) > 0)
    occupied = np.asarray(occupied)
    k = theory.shape[0]
    if occupied.size != k:
        return None
    sub = learned[np.ix_(occupied, occupied)]
    best = None
    for pm in itertools.permutations(range(k)):
        pm = list(pm)
        err = float(np.abs(sub[np.ix_(pm, pm)] - theory).max())
        if best is None or err < best[2]:
            best = (sub[np.ix_(pm, pm)], occupied[pm], err)
    return best


def assign_states_by_emission(learned_emissions, true_emissions, states=None):
    """
    Map each learned state to the theoretical state whose emission row is
    nearest in total variation.  Many-to-one is allowed (the bottleneck may split
    one causal state into several); a theoretical state no learned state maps to
    is left missing rather than forced.  `states` restricts the assignment; the
    rest get -1.  Returns (assignment (K,) int, tv (K,) distance to the match).
    """
    E = np.asarray(learned_emissions, dtype=float)
    F = np.asarray(true_emissions, dtype=float)
    K = E.shape[0]
    if states is None:
        states = np.arange(K)
    assignment = np.full(K, -1, dtype=np.int64)
    tv = np.full(K, np.nan)
    for s in np.asarray(states, dtype=np.int64):
        d = 0.5 * np.abs(E[s][None, :] - F).sum(axis=1)
        assignment[s] = int(np.argmin(d))
        tv[s] = float(d[assignment[s]])
    return assignment, tv


def compare_transition_matrix(learned, theory, learned_emissions, true_emissions,
                              weights=None):
    """
    Aggregate a learned state-to-state matrix onto the theoretical states and
    compare it with the closed form cell by cell.

    Each visited learned state is assigned to a theoretical state by
    assign_states_by_emission.  The rows of one group are mixed by `weights`
    (occupancy) and its columns summed, giving a (k,k) matrix on the theoretical
    states; a group with no learned state is a zero row and is reported in
    `missing`.  Returns assignment, groups, emission_tv, aggregated, theory,
    abs_error, max_error, mean_error, missing, row_mass, n_visited, permutation.
    """
    L = np.asarray(learned, dtype=float)
    T = np.asarray(theory, dtype=float)
    K, k = L.shape[0], T.shape[0]
    visited = np.flatnonzero(L.sum(axis=1) > 0)
    assignment, tv = assign_states_by_emission(learned_emissions, true_emissions,
                                               states=visited)
    w = (np.ones(K) if weights is None else np.asarray(weights, dtype=float)).copy()

    groups = {t: [int(s) for s in visited if assignment[s] == t] for t in range(k)}
    mixed = np.zeros((k, K))
    for t, members in groups.items():
        if not members:
            continue
        wm = w[members]
        wm = wm / wm.sum() if wm.sum() > 0 else np.full(len(members), 1.0 / len(members))
        mixed[t] = wm @ L[members]
    aggregated = np.zeros((k, k))
    for u, members in groups.items():
        if members:
            aggregated[:, u] = mixed[:, members].sum(axis=1)

    missing = [t for t, m in groups.items() if not m]
    present = [t for t in range(k) if t not in missing]
    abs_error = np.full((k, k), np.nan)
    abs_error[present] = np.abs(aggregated[present] - T[present])
    errs = abs_error[present]
    return dict(
        assignment=assignment, groups=groups, emission_tv=tv,
        aggregated=aggregated, theory=T, abs_error=abs_error,
        max_error=float(errs.max()) if errs.size else float("nan"),
        mean_error=float(errs.mean()) if errs.size else float("nan"),
        missing=missing, row_mass=aggregated.sum(axis=1), n_visited=int(visited.size),
        permutation=(match_permutation(L, T, occupied=visited) if visited.size == k else None),
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. The empirical symbolic machine
# ══════════════════════════════════════════════════════════════════════════
def arm_inputs(seqs: np.ndarray, mode: str) -> np.ndarray:
    """The tokens a model sees, in its own alignment (_Decoder._split):
    x[:-1] for mode="forward", x[1:] for mode="backward"."""
    seqs = np.asarray(seqs, dtype=np.int64)
    return seqs[:, :-1] if mode == "forward" else seqs[:, 1:]


@torch.no_grad()
def hard_state_trajectories(model, loader):
    """(states, tokens), (N, T) each: the bottleneck state and the model's input
    token at every position, in the model's own alignment."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    states, tokens = [], []
    for batch in loader:
        inputs, _ = model._split(batch, model.mode)
        model(inputs.to(device))
        states.append(model.last_states.detach().cpu().numpy())
        tokens.append(inputs.detach().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(states, axis=0), np.concatenate(tokens, axis=0)


def empirical_symbolic_machine(model, loader, min_pos: int) -> dict:
    """
    The unifilar machine the discrete model has BECOME, read off its state
    trajectories.

    The state is a function of the whole context, so this is an estimate:
    next_state[s, x] is the majority successor of (state s, token x) and
    determinism[s, x] the fraction of visits that agree with it -- 1 everywhere
    for a genuinely unifilar machine.  The step is (s_t, x_{t+1}) -> s_{t+1} in
    the model's reading order (mode="backward": one step back, (s_t, x_{t-1}) ->
    s_{t-1}).  Visited states are relabelled 0..k-1; `state_ids` maps back to the
    bottleneck's indices and `relabel` the other way.
    """
    states, tokens = hard_state_trajectories(model, loader)
    K, V = int(model.n_states), int(model.token_size)
    keep = kept_positions(model.mode, states.shape[1], min_pos)
    states, tokens = states[:, keep], tokens[:, keep]

    if model.mode == "forward":
        s_from, x, s_to = states[:, :-1], tokens[:, 1:], states[:, 1:]
    else:
        s_from, x, s_to = states[:, 1:], tokens[:, :-1], states[:, :-1]
    triples = np.zeros((K, V, K), dtype=np.float64)
    np.add.at(triples, (s_from.ravel(), x.ravel(), s_to.ravel()), 1.0)

    counts = np.bincount(states.ravel(), minlength=K).astype(np.float64)
    visited = np.flatnonzero(counts > 0)
    relabel = np.full(K, -1, dtype=np.int64)
    relabel[visited] = np.arange(len(visited))

    k = len(visited)
    next_state = np.zeros((k, V), dtype=np.int64)
    determinism = np.full((k, V), np.nan)
    for i, s in enumerate(visited):
        for symbol in range(V):
            row = triples[s, symbol]
            if row.sum() > 0:
                j = int(row.argmax())
                next_state[i, symbol] = relabel[j]
                determinism[i, symbol] = row[j] / row.sum()
            else:
                next_state[i, symbol] = i          # (s, x) never observed: a self-loop under a ~0 emission
    emission_probs = model.emission_table().detach().cpu().numpy()[visited]

    return {
        "next_state": next_state, "emission_probs": emission_probs,
        "determinism": determinism, "state_ids": visited, "relabel": relabel,
        "counts": counts[visited], "pair_counts": triples[visited].sum(axis=-1),
    }


def machine_state_transition(machine: dict) -> np.ndarray:
    """(k, k) state-to-state matrix induced by an emission-labelled machine."""
    next_state = np.asarray(machine["next_state"], dtype=np.int64)
    emission_probs = np.asarray(machine["emission_probs"], dtype=np.float64)
    k = next_state.shape[0]
    T = np.zeros((k, k), dtype=np.float64)
    for s in range(k):
        for x in range(emission_probs.shape[1]):
            T[s, int(next_state[s, x])] += float(emission_probs[s, x])
    return T


def stationary_distribution_from_machine(machine: dict) -> np.ndarray:
    T = machine_state_transition(machine)
    k = T.shape[0]
    A = T.T - np.eye(k)
    A[-1, :] = 1.0
    b = np.zeros(k, dtype=np.float64)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:              # a reducible empirical machine
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    return pi / max(pi.sum(), 1e-12)


def symbolic_cross_entropy(machine: dict, tokens: np.ndarray, start_states, mode: str, t0: int) -> float:
    """
    Bits/token of a unifilar machine run along the model's reading order over its
    input tokens, started from start_states at position t0: mode="forward"
    scores positions t0+1.., mode="backward" positions t0-1..0.
    """
    next_state = np.asarray(machine["next_state"], dtype=np.int64)
    emission_probs = np.asarray(machine["emission_probs"], dtype=np.float64)
    total_bits, total_steps = 0.0, 0
    for row, s0 in zip(tokens, start_states):
        s = int(s0)
        path = row[t0 + 1:] if mode == "forward" else row[:t0][::-1]
        for x in path:
            total_bits -= math.log2(max(float(emission_probs[s, int(x)]), 1e-12))
            s = int(next_state[s, int(x)])
            total_steps += 1
    return total_bits / max(total_steps, 1)


def exact_machine_report(model, report, empirical, th) -> dict:
    """
    For a model whose machine is exact by construction (GRU.DiscreteFeedbackGRU):
    its enumerated machine, whether the trajectories actually follow it, and its
    transition matrix against the closed form -- no generation, no estimation.

        follows     fraction of the (state, token) pairs seen on the data whose
                    observed successor is the enumerated one (1.0 by construction;
                    checked, not assumed)
        comparison  compare_transition_matrix on the enumerated matrix, over the
                    states visited on the data (the rest are zeroed out)
    """
    ex = model.machine()
    K = ex["next_state"].shape[0]
    ids, relabel = empirical["state_ids"], empirical["relabel"]
    seen = np.isfinite(empirical["determinism"])
    agree = [relabel[ex["next_state"][ids[i], x]] == empirical["next_state"][i, x]
             for i, x in zip(*np.nonzero(seen))]
    T_exact = machine_state_transition(ex)
    visited = np.zeros(K, dtype=bool)
    visited[report["occupied"]] = True
    T_exact[~visited] = 0.0
    comparison = compare_transition_matrix(T_exact, th["T_theory"], ex["emission_probs"],
                                           th["true_machine"]["emission_probs"],
                                           weights=report["occupancy"])
    return {"machine": ex, "transition": T_exact, "comparison": comparison,
            "follows": float(np.mean(agree)) if agree else float("nan"),
            "n_pairs_seen": int(seen.sum())}


# ══════════════════════════════════════════════════════════════════════════
# 4. Everything for one trained model
# ══════════════════════════════════════════════════════════════════════════
def exact_per_position_ce(gen, seqs) -> np.ndarray:
    """(N, T) bits of the EXACT model on every scored token of `seqs`, which are
    in the generator's current reading order (generator.ideal_probs)."""
    seqs = np.asarray(seqs, dtype=np.int64)
    P = gen.ideal_probs(seqs)[:, 1:]
    p_tgt = np.take_along_axis(P, seqs[:, 1:, None], axis=2)[..., 0]
    return -np.log2(np.clip(p_tgt, 1e-300, None))


def analyse_arm(model, rec, gen, ds, test_indices, test_loader, ana_loader, *,
                fresh_loader, seed_tokens, min_pos: int, total_run: int,
                conv_tol: float, full_tol: float = 0.05, rng=None) -> dict:
    """
    Score one trained discrete model against the theory of `gen`'s current view.

    `ds` holds the sequences the model was trained and tested on, in the view's
    reading order; `test_indices` are the held-out rows; `fresh_loader` is a
    fresh draw of the process (the checkpoint was selected on the test split, so
    only the fresh CE is unbiased); `seed_tokens` seeds the free-running
    generation.  FULL follows tau_experiment/study5: every theoretical state
    discovered and |S_emp - C| < full_tol.
    """
    # Imported here, not at the top: Experimental_pipeline has a `training` module
    # too, and a cross-check that loads both must still be able to import the
    # numpy-only functions above.
    from training import diagnose_divergence, eval_ce, per_position_ce

    th = theory_from_generator(gen)
    H = th["entropy_rate"]
    mode = model.mode

    # --- cross-entropy -------------------------------------------------------
    test_ce, test_ppl = eval_ce(model, test_loader)
    fresh_ce, _ = eval_ce(model, fresh_loader)
    test_seqs = ds.seqs[test_indices]
    neural_pp = per_position_ce(model, test_loader)                 # (N_test, T)
    exact_pp = exact_per_position_ce(gen, test_seqs)                # same tokens
    exact_ce = float(exact_pp.mean())
    divergence = diagnose_divergence(rec.step_loss)

    # --- states --------------------------------------------------------------
    report = causal_state_report(model, ana_loader, min_pos=min_pos)
    report["S_emp"] = float(report["S_emp"]) + 0.0                  # a single state gives -0.0

    # --- transition matrix against the closed form ----------------------------
    T_learned = transition_matrix_extraction(
        model, seed_tokens, total_run=total_run, window_size=ds.seq_len, rng=rng)
    comparison = compare_transition_matrix(
        T_learned, th["T_theory"], report["emissions"], th["true_machine"]["emission_probs"],
        weights=report["occupancy"])
    discovered = th["true_k"] - len(comparison["missing"])
    full = bool(discovered == th["true_k"] and abs(report["S_emp"] - th["C"]) < full_tol)
    # How GOOD the emission-row matches are: any visited state is matched to its
    # nearest theoretical row, so `discovered` alone credits an untrained model;
    # the occupancy-weighted TV to the matched row says whether the rows agree.
    tv = comparison["emission_tv"]
    seen = np.flatnonzero(np.isfinite(tv))
    w_seen = report["occupancy"][seen]
    emission_tv = (float(np.sum(w_seen * tv[seen]) / w_seen.sum()) if w_seen.sum() > 0
                   else float("nan"))

    # --- the empirical machine, scored on the test sequences ------------------
    machine = empirical_symbolic_machine(model, ana_loader, min_pos)
    det = machine["determinism"]
    weighted = machine["pair_counts"] / machine["pair_counts"].sum()
    det_weighted = float(np.nansum(np.where(np.isfinite(det), det * weighted, 0.0)))
    minimal_stationary = stationary_distribution_from_machine(machine)
    states_te, tokens_te = hard_state_trajectories(model, test_loader)
    Tn = tokens_te.shape[1]
    t0 = min_pos if mode == "forward" else Tn - 1 - min_pos
    start = machine["relabel"][states_te[:, t0]]
    start = np.where(start >= 0, start, 0)
    symbolic_ce = symbolic_cross_entropy(machine, tokens_te, start, mode, t0)
    # the neural and exact models on exactly the tokens the symbolic scorer read
    span = slice(t0, Tn - 1) if mode == "forward" else slice(0, t0)
    neural_ce_span, exact_ce_span = float(neural_pp[:, span].mean()), float(exact_pp[:, span].mean())
    # a model with an exact machine (the feedback GRU): read it off, no estimation
    exact = (exact_machine_report(model, report, machine, th)
             if callable(getattr(model, "machine", None)) else None)

    # --- when did validation first reach the entropy-rate band ----------------
    val_at, val_loss = np.asarray(rec.step_val_at), np.asarray(rec.step_val_loss)
    inside = np.flatnonzero(np.abs(val_loss - H) <= conv_tol)
    n_steps = len(rec.step_loss)
    found_at = int(val_at[inside[0]]) if inside.size else -1

    history = {
        "step_at": [int(v) for v in rec.step_at], "step_loss": [float(v) for v in rec.step_loss],
        "val_at": [int(v) for v in rec.step_val_at], "val_loss": [float(v) for v in rec.step_val_loss],
        "step_tau": [float(v) for v in rec.step_tau],
        "step_grad_norm": [float(v) for v in rec.step_grad_norm],
        "states_used": [int(v) for v in rec.step_states_used],
        "epoch_loss": [float(v) for v in rec.epoch_loss],
        "best_step": int(rec.best_step) if rec.restored_best else -1,
        "best_val": float(rec.best_val) if rec.restored_best else float("nan"),
    }
    metrics = {
        "test_ce": test_ce, "test_ppl": test_ppl, "fresh_ce": fresh_ce, "exact_ce": exact_ce,
        "entropy_rate": H, "gap": test_ce - H, "gap_exact": test_ce - exact_ce,
        "fresh_gap": fresh_ce - H, "within_tol": bool(abs(test_ce - H) <= conv_tol),
        "diverged": bool(divergence["diverged"]), "best_step": history["best_step"],
        "n_steps": n_steps, "found_at_step": found_at,
        "S_emp": report["S_emp"], "C": th["C"], "S_minus_C": report["S_emp"] - th["C"],
        "k_used": report["n_states_used"], "K": report["n_states"], "true_k": th["true_k"],
        "discovered": int(discovered), "full": full,
        "emission_tv": emission_tv,
        "emission_tv_max": float(np.nanmax(tv)) if seen.size else float("nan"),
        "h_state_given_token": report["h_state_given_token"],
        "transition_max_err": comparison["max_error"], "transition_mean_err": comparison["mean_error"],
        "transition_missing": len(comparison["missing"]), "n_visited_generation": comparison["n_visited"],
        "machine_size": int(machine["next_state"].shape[0]), "determinism": det_weighted,
        "symbolic_ce": symbolic_ce, "neural_ce_span": neural_ce_span, "exact_ce_span": exact_ce_span,
        "max_grad_norm": float(max(rec.step_grad_norm)) if rec.step_grad_norm else float("nan"),
    }
    if exact is not None:
        metrics.update({
            "exact_follows": exact["follows"],
            "exact_T_err": exact["comparison"]["max_error"],
            "exact_missing": len(exact["comparison"]["missing"]),
        })
    return {
        "theory": th, "history": history, "metrics": metrics, "report": report,
        "transition": T_learned, "transition_comparison": comparison,
        "machine": machine, "minimal_stationary": minimal_stationary,
        "exact_machine": exact,
        "per_position": {"neural": neural_pp.mean(axis=0), "exact": exact_pp.mean(axis=0)},
    }
