"""
The three extractors.

    causal_state_report          DISCRETE model -> states, occupancy, emissions,
                                 S_emp_states.  Exact: a position's state is
                                 argmax(state_logits).  No free parameter.
    recover_causal_states        STANDARD model -> k_hat, S_hat.  Clusters the
                                 model's PREDICTIVE DISTRIBUTIONS.  Has one
                                 genuinely free parameter, `state_tol`.
    transition_matrix_extraction DISCRETE model -> T[i][j] = P(s_j | s_i), read
                                 off the model's own free-running generation.

What is NOT here: `statistical_complexity_empirical`.  It runs k-means at an
ASSUMED k, so S <= log2(k) by construction -- it cannot report a state count it
was not told, which makes it useless as an extraction method.  The two above are
strictly better and are exactly the pipeline's two methods.

A NOTE ON THE BACKWARD ARM.  The loader always yields the forward convention
(x[:-1], x[1:]).  A backward model is TRAINED on the swap, so every function
here that feeds it must swap too, via `model._split`.  In the old tree
`causal_state_report` did this and `latent_extraction` did not, so the two
extractors were reading a one-token-shifted view of the same data; harmless in
expectation, but the whole point of F4 is that both estimators see the same
realisation.
"""
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from processes import entropy_bits, generate


# ══════════════════════════════════════════════════════════════════════════
# 1. DISCRETE — exact states
# ══════════════════════════════════════════════════════════════════════════
def causal_state_report(model, data_loader, min_pos: int = 5, device=None):
    """
    Assign every token position to a causal state and summarise.

    Returns None for a model with no explicit states (OneHotDecoder), which is
    how callers stay free of `if embed_type ==` branches.

    `min_pos` drops the short-context end of each sequence, and WHICH end that
    is depends on the arm: a forward model at position t has only t tokens of
    context, so its short end is the START; a backward model under a triu mask
    attends to [t, T-1], so its short end is the FINISH.  Trimming the wrong end
    compares a settled arm against an unsettled one.
    """
    if getattr(model, "state_assignment", None) is None:
        return None
    K = int(getattr(model, "n_states", 0))
    if K <= 0:
        return None

    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    V      = int(getattr(model, "token_size", 0))
    counts = torch.zeros(K, dtype=torch.long)
    joint  = torch.zeros(max(V, 1), K, dtype=torch.long)   # for H(state | token)

    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = model._split(batch, getattr(model, "mode", "forward"))
            model(inputs.to(device))
            st = model.last_states
            T  = st.shape[1]
            keep = (slice(min_pos, None) if model.mode == "forward"
                    else slice(0, max(T - min_pos, 1)))
            st = st[:, keep]
            counts += torch.bincount(st.reshape(-1).cpu(), minlength=K)
            if V:
                tk = inputs[:, keep].reshape(-1).cpu()
                joint += torch.bincount(tk * K + st.reshape(-1).cpu(),
                                        minlength=V * K).reshape(V, K)
        vectors   = model.state_matrix.detach().cpu()
        emissions = model.emission_table().detach().cpu()

    if was_training:
        model.train()

    p = counts.numpy().astype(float)
    p = p / p.sum() if p.sum() > 0 else p
    s_emp = entropy_bits(p)

    # H(state | current token), in bits.  For every process here the true causal
    # state is a DETERMINISTIC function of the current token in BOTH arms, so
    # the truth scores exactly 0 and any positive value means the bottleneck is
    # keying on context it does not need.  It needs no ground truth, which makes
    # it the one available unsupervised warning that states were merged --
    # measured 0.000 on every run that recovered its state set, 0.11-0.48 on
    # every run that did not.
    j   = joint.numpy().astype(float)
    tot = j.sum()
    h_cond = 0.0
    if tot > 0:
        for row in j:
            n_t = row.sum()
            if n_t > 0:
                h_cond += (n_t / tot) * entropy_bits(row[row > 0] / n_t)

    return {
        "counts": counts.numpy(), "occupancy": p,
        "vectors": vectors.numpy(), "emissions": emissions.numpy(),
        "occupied": [k for k in range(K) if counts[k] > 0],
        "n_states": K, "n_states_used": int((counts > 0).sum()),
        "S_emp": s_emp, "h_state_given_token": h_cond,
        "joint_token_state": joint.numpy(),
        "mode": getattr(model, "mode", "forward"),
    }


def s_emp_from_states(report):
    return None if report is None else report["S_emp"]


# ══════════════════════════════════════════════════════════════════════════
# 2. STANDARD — states by clustering predictive distributions
# ══════════════════════════════════════════════════════════════════════════
# Points live on the probability simplex, so a distance THRESHOLD has to be
# recalibrated per metric -- the ranges differ:
#     euclidean  [0, sqrt(2)]      tv  [0, 1]      js  [0, 1] bits

def euclidean_distance_matrix(P) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    return np.sqrt(np.clip(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1), 0.0, None))


def tv_distance_matrix(P) -> np.ndarray:
    """Total variation, (1/2) L1 -- the largest disagreement about any event."""
    P = np.asarray(P, dtype=float)
    return 0.5 * np.abs(P[:, None, :] - P[None, :, :]).sum(-1)


def js_distance_matrix(P, eps: float = 1e-12) -> np.ndarray:
    """
    sqrt(Jensen-Shannon divergence), in bits.

    KL is unusable as a clustering metric here -- asymmetric, unbounded, and
    INFINITE wherever q=0<p, which a trained model reaches.  JS compares both to
    their mixture, so it is symmetric, bounded by 1 bit, always finite, and its
    square root satisfies the triangle inequality.

    The clip before the sqrt is load-bearing: near-identical rows compute a JSD
    of about -1e-17, whose sqrt is NaN.
    """
    P = np.clip(np.asarray(P, dtype=float), eps, None)
    P = P / P.sum(1, keepdims=True)
    _H = lambda x: -(x * np.log2(np.clip(x, eps, None))).sum(-1)
    M = 0.5 * (P[:, None, :] + P[None, :, :])
    return np.sqrt(np.clip(_H(M) - 0.5 * (_H(P)[:, None] + _H(P)[None, :]), 0.0, 1.0))


DISTANCE_MATRICES = {"euclidean": euclidean_distance_matrix,
                     "tv": tv_distance_matrix, "js": js_distance_matrix}

# NOT interchangeable -- see the range note above.  Each is the value that
# maximised agreement with the closed forms over the 14 arms of the archived
# quick run: euclidean 13/14, tv 13/14, js 12/14.  Worth stating plainly: JS has
# the better theory but is very slightly WORSE than plain L2 on this data.
DEFAULT_STATE_TOL = {"euclidean": 0.10, "tv": 0.075, "js": 0.075}


def _predictive_distributions(model, data_loader, use_t, n_pts, seed,
                              max_batches=None):
    """
    (points, tokens) -- one predictive distribution per sequence, at the
    max-context position for the arm.

    What gets clustered is the model's PREDICTIVE DISTRIBUTION, not its latent.
    A causal state is an equivalence class of histories inducing the same
    distribution over the future; it is not a region of latent space, and
    nothing forces the network to give two histories in one state the same
    latent.  Measured on the coin forward arm, where tokens 0 and 2 are ONE
    causal state: their latents sit 1.59 apart against a true state gap of
    19.10, while their predictive distributions sit 0.041 apart against a state
    gap of 1.065.  Both carry the structure, but the latent also carries token
    identity the state does not depend on.
    """
    model.eval()
    device = next(model.parameters()).device
    lats, toks = [], []
    with torch.no_grad():
        for bi, batch in enumerate(data_loader):
            if max_batches is not None and bi >= max_batches:
                break
            inputs, _ = model._split(batch, getattr(model, "mode", "forward"))
            inputs = inputs.to(device)
            model(inputs)
            lats.append(model.last_encodings.detach().cpu().numpy())
            toks.append(inputs.detach().cpu().numpy())
    lat = np.concatenate(lats, axis=0)          # (N, T, D)
    inp = np.concatenate(toks, axis=0)          # (N, T)

    T = lat.shape[1]
    t = T - 1 if use_t == "last" else (0 if use_t == "first" else int(use_t) % T)
    pts, tk = lat[:, t, :], inp[:, t]
    if len(pts) > n_pts:                        # sample ACROSS sequences
        keep = np.random.default_rng(seed).choice(len(pts), n_pts, replace=False)
        pts, tk = pts[keep], tk[keep]

    with torch.no_grad():
        probs = torch.softmax(
            model.output_prj(torch.from_numpy(pts).float().to(device)),
            dim=-1).cpu().numpy()
    return probs, tk


def recover_causal_states(model, data_loader, use_t="last", max_batches=20,
                          state_tol=None, n_pts=1000, seed=0, metric="euclidean",
                          tol_grid=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60)):
    """
    DISCOVER the number of causal states and their occupancy entropy, rather
    than assuming it.

    THE THRESHOLD IS A REAL FREE PARAMETER AND NO SINGLE VALUE IS CORRECT.  The
    smallest true separation between backward states varies by an order of
    magnitude across the processes here -- 0.135 at p=0.1,q=0.9 against 0.612 at
    p=q=0.5 -- so a threshold that resolves one merges another.  That is why
    `stability` is always returned and always reported: "k_hat = 2" is not
    defensible alone, whereas "k_hat = 2, stable across tol in [0.10, 0.60]" is.

    Returns k_hat, S_hat, labels, tokens, probs, state_tol, metric, stability,
    plateau.
    """
    if metric not in DISTANCE_MATRICES:
        raise ValueError(f"metric must be one of {sorted(DISTANCE_MATRICES)}")
    if getattr(model, "output_prj", None) is None:
        return None                       # discrete model: use causal_state_report
    if state_tol is None:
        state_tol = DEFAULT_STATE_TOL[metric]

    probs, toks = _predictive_distributions(model, data_loader, use_t, n_pts,
                                            seed, max_batches)
    D = DISTANCE_MATRICES[metric](probs)          # one matrix, reused per threshold

    def _fit(tol):
        # COMPLETE linkage: two clusters merge only if every cross-pair is within
        # tol, so a cluster is a set whose members are all mutually within tol --
        # the literal reading of "same predictive distribution to within
        # tolerance".  Single linkage merges on the closest pair and chains
        # distinct states together.
        return AgglomerativeClustering(n_clusters=None, distance_threshold=tol,
                                       metric="precomputed",
                                       linkage="complete").fit_predict(D)

    labels = _fit(state_tol)
    k_hat  = int(len(np.unique(labels)))
    counts = np.bincount(labels, minlength=k_hat).astype(float)
    S_hat  = entropy_bits(counts / counts.sum())

    stability = [(float(t), int(len(np.unique(_fit(t))))) for t in tol_grid]
    spans: dict = {}
    for _, k in stability:
        spans[k] = spans.get(k, 0) + 1
    plateau = max(spans, key=spans.get) if spans else k_hat

    # S_hat AT THE PLATEAU, alongside S_hat at the requested tol.
    #
    # This is not a second knob, it is the mitigation for the first one.  A
    # single threshold over-splits: measured on coin p=0.3,q=0.4 at tol=0.10,
    # k_hat is 4 forward against a true 2 and 7 backward against a true 3, which
    # biases S_hat by +0.68 and +0.96 bits.  The plateau -- the k that survives
    # the widest span of thresholds -- recovers 2 and 3 exactly, and it is
    # chosen WITHOUT reference to the truth, so using it is not tuning on the
    # answer.  The plateau tolerance is the smallest one that reaches it, i.e.
    # the least-merged member of the winning span.
    tol_plateau = next((t for t, k in stability if k == plateau), float(state_tol))
    lab_p = _fit(tol_plateau)
    cnt_p = np.bincount(lab_p, minlength=int(len(np.unique(lab_p)))).astype(float)
    S_hat_plateau = entropy_bits(cnt_p / cnt_p.sum())

    return dict(k_hat=k_hat, S_hat=float(S_hat), labels=labels, tokens=toks,
                probs=probs, state_tol=float(state_tol), metric=metric,
                stability=stability, plateau=int(plateau),
                tol_plateau=float(tol_plateau),
                S_hat_plateau=float(S_hat_plateau),
                labels_plateau=lab_p)


# ══════════════════════════════════════════════════════════════════════════
# 3. DISCRETE — the learned state-to-state transition matrix
# ══════════════════════════════════════════════════════════════════════════
def transition_matrix_extraction(model, kind: str, params: dict, *,
                                 burn_in: int, total_run: int,
                                 window_size: int, rng=None) -> np.ndarray:
    """
    T[i][j] = P(s_{t+1} = j | s_t = i), read off the model's own generation.

    The model is seeded with a burn-in drawn from the TRUE process, then rolls
    forward on its own samples; at each step the state the bottleneck lands in
    at the newest position is recorded, and the counts are row-normalised.

    Free-running, not teacher-forced: the matrix describes the process the model
    has BECOME.  Where that differs from the truth, the difference is the
    measurement.

    Identified only up to a PERMUTATION of the state labels -- which index the
    bottleneck assigns to which causal state is arbitrary -- so match labels
    before comparing to a closed form.

    `window_size` is the context it rolls forward on; pass `seq_len`, which is
    what the model trained at.  A longer window measures extrapolation, and does
    so asymmetrically between the arms.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    device = next(model.parameters()).device
    is_bw  = (getattr(model, "mode", "forward") == "backward")
    V      = int(model.token_size)

    # The seed sequence comes from processes.generate, so there is exactly one
    # sampler in the pipeline -- and a flower burn-in uses the run's own dice
    # rather than a freshly drawn set.
    seed_tokens = generate(kind, params, num_samples=1, seq_len=burn_in,
                           burn_in=0, rng=rng)
    inputs = torch.as_tensor(seed_tokens, dtype=torch.long, device=device)

    window_size = min(int(window_size), model.max_len)
    if inputs.shape[1] < window_size:
        print(f"  ! transition seed is {inputs.shape[1]} tokens, "
              f"shorter than the {window_size}-token window")
    # The newest token, which is also the one with full context: LAST for a
    # forward model (tril), FIRST for a backward one (triu).
    pos = 0 if is_bw else -1

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

    row = transition.sum(axis=1, keepdims=True)
    # Unvisited states keep a zero row rather than raising.
    return np.divide(transition, row, out=np.zeros_like(transition), where=row > 0)


def match_permutation(learned: np.ndarray, theory: np.ndarray, occupied=None):
    """
    Best label permutation of `learned`'s visited states onto `theory`.

    The bottleneck's state indices are arbitrary, so a learned matrix cannot be
    compared cell-by-cell to a closed form until the labels are matched.  Brute
    force over permutations, which is fine at the sizes here (theory has 2-11
    states) and exact, unlike a greedy assignment.

    Returns (permuted_submatrix, chosen_indices, max_abs_error), or None when the
    number of visited states does not match the theoretical count -- in which
    case there is no permutation to find and the honest answer is "no overlay".
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
