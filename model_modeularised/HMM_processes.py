"""
this file contains all generator of all the processes of interest,
together with their theoretical quantities and structures

Every process is described by three functions of `params`, and nothing else:

    transition(params)  -> T[s, s']    = P(s_{t+1} = s' | s_t = s)
    state_map(params)   -> E[s, s', x] = P(x_t = x | s_t = s, s_{t+1} = s')
    vocab_size(params)  -> |X|

The joint map M[s, s', x] = P(x_t = x, s_{t+1} = s' | s_t = s) is the product
T[:, :, None] * E (the symbol-labelled transition matrices T^(x) of the
literature, stored as M[:, :, x]); every other quantity -- emission P(x|s),
entropy rate, statistical complexity, the machines in both time directions -- is
computed from it, and `tokens_from_map` samples from it without knowing which
process it is holding.  Register a new process in PROCESS at the bottom and
nothing else in the codebase has to learn its name.

The two factors carry the randomness in opposite places, which is why they are
kept separate: the coin has stochastic transitions and a deterministic token on
each edge, the flower has deterministic transitions out of each petal and a
stochastic token on the way back.

Equation, proposition and footnote numbers refer to Ellison, Mahoney &
Crutchfield, arXiv:0905.3587v1 (J. Stat. Phys. 136, 1005 (2009)).
"""
import numpy as np

##############################################################################
# GENERIC MACHINERY -- process agnostic
##############################################################################
def stationary_dist(T: np.ndarray) -> np.ndarray:
    """
    left eigenvector of T for eigenvalue 1, normalised to a distribution.
    Unique when the chain has a single closed class (Perron-Frobenius).  Chains
    with more than 300 states (the truncated machines of infinite processes) use a
    sparse linear solve, which is much faster than a dense eigendecomposition.
    """
    T = np.asarray(T, dtype=float)
    S = T.shape[0]
    if S <= 300:
        eigvals, eigvecs = np.linalg.eig(T.T)
        pi = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
    else:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        A = sparse.lil_matrix(T.T - np.eye(S))
        A[S - 1, :] = 1.0                        # one balance equation -> normalisation
        b = np.zeros(S)
        b[-1] = 1.0
        pi = spsolve(A.tocsr(), b)
    pi = np.clip(pi / pi.sum(), 0.0, None)       # sign fixed, -1e-17 on transients -> 0
    return pi / pi.sum()

def tokens_from_map(n_tokens: int, T: np.ndarray, E: np.ndarray,
                    pi: np.ndarray, rng) -> np.ndarray:
    """
    `n_tokens` tokens of any edge-emitting machine (T, E), started from `pi`.

    One uniform per token: the pair (s_{t+1}, x_t) is drawn jointly from row
    s_t of the flattened joint map, so a single searchsorted does both the
    state update and the emission and there is no per-process branching.
    Both cumulative sums end at exactly 1 and the search is side="right", so a
    pair (or start state) of probability zero is never drawn.
    """
    S, _, V = E.shape
    C = (np.asarray(T, dtype=float)[:, :, None] * E).reshape(S, S * V).cumsum(axis=1)
    C /= C[:, -1:]                              # renormalise away fp drift
    c0 = np.asarray(pi, dtype=float).cumsum()
    c0 /= c0[-1]
    u = rng.random(n_tokens + 1)                # every draw from the passed rng

    s = int(np.searchsorted(c0, u[0], side="right"))
    out = np.empty(n_tokens, dtype=np.int64)
    for t in range(n_tokens):
        s, out[t] = divmod(int(np.searchsorted(C[s], u[t + 1], side="right")), V)
    return out

def emission_dist(T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    P(x | s), the ideal next-token prediction from each causal state. (S, V)
    """
    return (np.asarray(T, dtype=float)[:, :, None] * E).sum(axis=1)

def entropy_rate(T: np.ndarray, E: np.ndarray, pi: np.ndarray) -> float:
    """
    h_mu = sum_s pi_s H[P(x|s)] in bits/token; exact for a unifilar machine
    (for a non-unifilar presentation it is H[X_t | S_t], a lower bound).  It is
    the cross-entropy floor only in the limit of infinite context: at position t
    the floor is H[X_t | x_0 .. x_{t-1}] >= h_mu (generator.ideal_ce()).
    """
    P = emission_dist(T, E)
    logP = np.zeros_like(P)
    np.log2(P, out=logP, where=P > 0)
    return float(-(np.asarray(pi, dtype=float) * (P * logP).sum(axis=1)).sum()) + 0.0
    # + 0.0 turns -0.0 into 0.0, as statistical_complexity and entropy_bits both do

def statistical_complexity(pi: np.ndarray) -> float:
    """
    C_mu = H[pi] over the causal states, in bits.  Only the true statistical
    complexity when (T, E) is the epsilon-machine: unifilar, minimal and
    recurrent (see flower_state_map, pcoin_transition and sns_transition).
    """
    pi = np.asarray(pi, dtype=float)
    logpi = np.zeros_like(pi)
    np.log2(pi, out=logpi, where=pi > 0)
    return float(-(pi * logpi).sum()) + 0.0        # + 0.0 turns -0.0 into 0.0

def live_map(T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    E with its dead edges zeroed: the canonical form, the one split_map returns and
    the one that makes split_map(joint_map(T, E)) give (T, E) back unchanged.

    The state maps write E on every edge the machine structurally has, so a parameter
    that drives T[s, s'] to zero -- p = 0 or p = 1 at the ends of a sweep -- leaves
    stale emission mass behind.  That mass is inert: every consumer goes through
    joint_map, and T * E multiplies it away.  So this normalises, it never corrects.
    """
    E = np.array(E, dtype=float, copy=True)
    E[np.asarray(T, dtype=float) == 0.0] = 0.0
    return E

def check_map(T: np.ndarray, E: np.ndarray, V: int = None) -> None:
    """
    (T, E) is a well-formed machine: T stochastic, E stochastic on every live edge,
    no negative entries, E.shape[2] == V.

    E on a dead edge is deliberately not checked.  The state maps write those entries
    whenever a parameter reaches 0 or 1, joint_map multiplies them away, and requiring
    them to be zero only refused well-formed machines at the ends of a parameter sweep.
    Use live_map for the canonical form.
    """
    T = np.asarray(T, dtype=float)
    assert T.ndim == 2 and T.shape[0] == T.shape[1], f"T must be square, got {T.shape}"
    assert E.shape[:2] == T.shape, f"E {E.shape} does not match T {T.shape}"
    assert V is None or E.shape[2] == V, f"E has {E.shape[2]} tokens, vocab_size is {V}"
    assert (T >= 0).all() and (E >= 0).all(), "negative probabilities"
    assert np.allclose(T.sum(axis=1), 1.0), "rows of T must sum to 1"
    assert np.allclose(E.sum(axis=2)[T > 0], 1.0), "E must sum to 1 on every edge with T > 0"

##############################################################################
# REVERSAL MACHINERY -- process agnostic
#   epsilon_machine(T, E)  -> M+, the forward (predictive) epsilon-machine
#   reverse_machine(M+)    -> M- = U(T(M+)), diagram (26): time reversal T
#                             (Eq. 20), then the mixed-state presentation U
#                             (Eqs. 27-32), its recurrent part, minimisation
##############################################################################
def joint_map(T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    M[s, s', x] = T[s, s'] E[s, s', x] = P(x_t = x, s_{t+1} = s' | s_t = s)   (Eq. 19)
    """
    return np.asarray(T, dtype=float)[:, :, None] * E

def split_map(M: np.ndarray):
    """
    inverse of joint_map: T = sum_x M, and E = M / T on every edge with T > 0.
    """
    T = M.sum(axis=2)
    E = np.zeros_like(M)
    live = T > 0
    E[live] = M[live] / T[live][:, None]
    return T, E

def entropy_bits(p) -> float:
    """
    H[p] in bits of any distribution (a mixed state, a row of a switching map).
    """
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 1e-300]
    return float(-(p * np.log2(p)).sum()) + 0.0          # + 0.0 turns -0.0 into 0.0

def is_unifilar(T: np.ndarray, E: np.ndarray) -> bool:
    """
    at most one successor per (state, token), i.e. H[S' | S, X] = 0 (footnote [35]).
    """
    return bool(((joint_map(T, E) > 0).sum(axis=1) <= 1).all())

def closed_class(T: np.ndarray) -> np.ndarray:
    """
    indices of the single closed communicating class of the chain T: its recurrent states.
    """
    from scipy.sparse.csgraph import connected_components
    T = np.asarray(T)
    n_comp, lab = connected_components(T > 0, directed=True, connection="strong")
    closed = [c for c in range(n_comp) if not (T[np.ix_(lab == c, lab != c)] > 0).any()]
    assert len(closed) == 1, f"expected one recurrent class, found {len(closed)}"
    return np.flatnonzero(lab == closed[0])

def time_reverse(M: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    R[s', s, x] = pi[s] M[s, s', x] / pi[s'] = P(x_t = x, s_t = s | s_{t+1} = s')   (Eq. 20)

    Bayes' rule on the stationary joint.  R generates the same process read right
    to left, has the same stationary distribution (Prop. 10) and is in general not
    unifilar.  Applied twice with the same pi it returns M.
    """
    pi = np.asarray(pi, dtype=float)
    assert (pi > 0).all(), "time_reverse needs every state recurrent (pi > 0)"
    return np.transpose(M, (1, 0, 2)) * pi[None, :, None] / pi[:, None, None]

def mixed_states(M: np.ndarray, eta0: np.ndarray, tol: float = 1e-10, max_states: int = 2000):
    """
    mixed-state presentation U(M), grown breadth-first from eta0 one token at a time:
    eta -> eta M[:, :, x] / (eta M[:, :, x] 1), taken with probability eta M[:, :, x] 1
    (Eqs. 21-25; Eqs. 27-32 when M is a time-reversed presentation).  A new vector
    within `tol` (max-abs) of a known state is that state.  Returns the states, the
    edges {(k, x): (j, prob)} and whether the search stopped at max_states.
    """
    Mx = np.ascontiguousarray(np.moveaxis(M, 2, 0))       # (V, S, S): one matrix per token
    states, edges, queue, head = [np.asarray(eta0, dtype=float)], {}, [0], 0
    arr = states[0][None, :]
    while head < len(queue) and len(states) < max_states:
        k = queue[head]
        head += 1
        for x in range(Mx.shape[0]):
            v = states[k] @ Mx[x]
            p = float(v.sum())
            if p <= 1e-14:
                continue
            v = v / p
            d = np.max(np.abs(arr - v), axis=1)
            j = int(np.argmin(d))
            if d[j] >= tol:
                states.append(v)
                arr = np.vstack([arr, v])
                j = len(states) - 1
                queue.append(j)
            edges[(k, x)] = (j, p)
    return states, edges, head < len(queue)

def recurrent_part(states, edges, V: int):
    """
    drop the states a truncated search left unexpanded, then the transient states
    (Sec. X: the mixed-state construction also produces transient causal states, "but
    we eventually remove them"); return the joint map on the recurrent class and its
    mixed states, one row per recurrent state.
    """
    alive = set(range(len(states)))
    while True:
        dead = alive - {k for (k, _) in edges}
        if not dead:
            break
        alive -= dead
        edges = {(k, x): (j, p) for (k, x), (j, p) in edges.items() if k in alive and j in alive}
    idx = sorted(alive)
    pos = {k: i for i, k in enumerate(idx)}
    M = np.zeros((len(idx), len(idx), V))
    for (k, x), (j, p) in edges.items():
        M[pos[k], pos[j], x] = p
    M /= M.sum(axis=(1, 2))[:, None, None]                  # rows cut by truncation
    keep = closed_class(M.sum(axis=2))
    M = M[np.ix_(keep, keep)]
    M /= M.sum(axis=(1, 2))[:, None, None]
    return M, np.array([states[idx[i]] for i in keep])

def minimise(M: np.ndarray, tol: float = 1e-10):
    """
    merge the states of a unifilar machine that have the same future (partition
    refinement).  Start from classes of equal next-token distributions; split a class
    whenever two of its states move on some token into different classes; stop when
    nothing splits.  Returns the quotient machine and the class of every input state.
    """
    S, _, V = M.shape
    p = M.sum(axis=1)                                        # (S, V)  P(x | s)
    succ = np.where(p > 0, M.argmax(axis=1), -1)             # (S, V)  unique successor, -1 if none
    reps, labels = [], np.empty(S, dtype=int)
    for s in range(S):
        j = next((c for c, r in enumerate(reps) if np.max(np.abs(p[s] - p[r])) < tol), None)
        if j is None:
            reps.append(s)
            j = len(reps) - 1
        labels[s] = j
    while True:
        sig = [(labels[s],) + tuple(labels[succ[s, x]] if succ[s, x] >= 0 else -1 for x in range(V))
               for s in range(S)]
        order = {k: i for i, k in enumerate(dict.fromkeys(sig))}
        new = np.array([order[k] for k in sig])
        if new.max() == labels.max():
            break
        labels = new
    n = labels.max() + 1
    Q = np.zeros((n, n, V))
    for c in range(n):
        r = int(np.flatnonzero(labels == c)[0])
        for x in range(V):
            if succ[r, x] >= 0:
                Q[c, labels[succ[r, x]], x] = p[r, x]
    return Q, labels

def merge_mixed(mixed: np.ndarray, labels: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """
    Appendix, Props. 11-13: mixed states that belong to one causal state combine as
    P(S+ | S-) = sum over the mixed states R in S- of P(S+ | R) P(R) / P(S-).
    """
    n = labels.max() + 1
    return np.stack([(pr[labels == c, None] * mixed[labels == c]).sum(axis=0) / pr[labels == c].sum()
                     for c in range(n)])

def _prune(M, mixed, pi_floor):
    """
    drop recurrent states whose stationary probability is below pi_floor (the far
    tail of a truncated infinite machine, where pi underflows), renormalise, and
    keep the closed class.
    """
    while True:
        pi = stationary_dist(M.sum(axis=2))
        keep = np.flatnonzero(pi > pi_floor)
        if len(keep) == M.shape[0]:
            return M, mixed
        M = M[np.ix_(keep, keep)]
        mixed = None if mixed is None else mixed[keep]
        live = np.flatnonzero(M.sum(axis=(1, 2)) > 0)
        M, mixed = M[np.ix_(live, live)], (None if mixed is None else mixed[live])
        M /= M.sum(axis=(1, 2))[:, None, None]
        cc = closed_class(M.sum(axis=2))
        M, mixed = M[np.ix_(cc, cc)], (None if mixed is None else mixed[cc])
        M /= M.sum(axis=(1, 2))[:, None, None]

def _unifilarise(M, eta0, tol, max_states, pi_floor=1e-15):
    """
    mixed-state presentation from eta0 and its recurrent part.  The search is run
    again at tol * 1e-3: an infinite machine gains recurrent states when the merge
    tolerance tightens, a finite one does not.  Returns (M, mixed, flags).
    """
    V = M.shape[2]
    st, ed, truncated = mixed_states(M, eta0, tol, max_states)
    Mr, mixed = recurrent_part(st, ed, V)
    st2, ed2, _ = mixed_states(M, eta0, tol * 1e-3, max_states)
    Mr2, _ = recurrent_part(st2, ed2, V)
    infinite = bool(truncated or Mr2.shape[0] > Mr.shape[0])
    Mr, mixed = _prune(Mr, mixed, pi_floor)
    Mr2, _ = _prune(Mr2, None, pi_floor)
    C_1 = statistical_complexity(stationary_dist(Mr.sum(axis=2)))
    C_2 = statistical_complexity(stationary_dist(Mr2.sum(axis=2)))
    flags = dict(n_raw=len(st), truncated=truncated, infinite=infinite, C_shift=abs(C_2 - C_1))
    return Mr, mixed, flags

def epsilon_machine(T: np.ndarray, E: np.ndarray, tol: float = 1e-10, max_states: int = 2000) -> dict:
    """
    forward epsilon-machine M+ of the process that (T, E) generates.
    Unifilar (T, E): keep the recurrent states, then minimise.
    Non-unifilar (T, E), e.g. the SNS: forward mixed states from pi (Eqs. 21-25),
    their recurrent part, then minimise.
    """
    M = joint_map(T, E)
    if is_unifilar(T, E):
        keep = closed_class(T)
        Mr = M[np.ix_(keep, keep)]
        flags = dict(n_raw=T.shape[0], truncated=False, infinite=False, C_shift=0.0)
    else:
        Mr, _, flags = _unifilarise(M, stationary_dist(T), tol, max_states)
    Q, labels = minimise(Mr, tol)
    Tm, Em = split_map(Q)
    return dict(T=Tm, E=Em, n_states=Tm.shape[0], merged=bool(Q.shape[0] < Mr.shape[0]), **flags)

def reverse_machine(T: np.ndarray, E: np.ndarray, tol: float = 1e-10, max_states: int = 2000) -> dict:
    """
    reverse epsilon-machine M- = U(T(M+)) (diagram (26)) of a forward epsilon-machine
    (T, E).  Also returns mixed[j] = P(S+ | S- = j), one row per reverse causal state,
    and the flags truncated, infinite, C_shift, merged.
    """
    M, pi = joint_map(T, E), stationary_dist(T)
    R = time_reverse(M, pi)                                    # T(M+), Eq. 20
    Mr, mixed, flags = _unifilarise(R, pi, tol, max_states)    # U(T(M+)), Eqs. 27-32
    Q, labels = minimise(Mr, tol)                              # a no-op if U(T(M+)) is minimal
    if Q.shape[0] < Mr.shape[0]:
        mixed = merge_mixed(mixed, labels, stationary_dist(Mr.sum(axis=2)))
    Tm, Em = split_map(Q)
    return dict(T=Tm, E=Em, mixed=mixed, n_states=Tm.shape[0],
                merged=bool(Q.shape[0] < Mr.shape[0]), **flags)

def isomorphic(T1, E1, T2, E2, tol: float = 1e-8):
    """
    state map between two unifilar, irreducible machines with the same labelled
    transitions, or None: fix the image of state 0 and follow the unique successors.
    """
    A, B = joint_map(T1, E1), joint_map(T2, E2)
    if A.shape != B.shape:
        return None
    S, _, V = A.shape
    for start in range(S):
        m, stack, ok = {0: start}, [0], True
        while stack and ok:
            s = stack.pop()
            for x in range(V):
                pa, pb = A[s, :, x].sum(), B[m[s], :, x].sum()
                if abs(pa - pb) > tol:
                    ok = False
                    break
                if pa > tol:
                    sa, sb = int(A[s, :, x].argmax()), int(B[m[s], :, x].argmax())
                    if sa not in m:
                        m[sa] = sb
                        stack.append(sa)
                    elif m[sa] != sb:
                        ok = False
                        break
        if ok and len(m) == S and len(set(m.values())) == S:
            return m
    return None

def word_probs(T: np.ndarray, E: np.ndarray, L: int) -> np.ndarray:
    """
    P(w) for every word of length L, index = sum_t x_t V^(L-1-t).
    """
    M = joint_map(T, E)
    v = stationary_dist(T)[None, :]
    for _ in range(L):
        v = np.einsum("ws,stx->wxt", v, M).reshape(-1, M.shape[0])
    return v.sum(axis=1)

def conditional_entropies(T: np.ndarray, E: np.ndarray, K: int) -> np.ndarray:
    """
    h(L) = H(L) - H(L-1) = H[X_{L-1} | X_0 .. X_{L-2}] for L = 1..K, to rounding
    error: the expected ideal cross-entropy of a token that has L - 1 tokens of
    context.  It decreases to h_mu and is the same in both reading directions.
    Carries the distribution over mixed states (equal ones pooled) instead of
    enumerating words.
    """
    Mx = np.ascontiguousarray(np.moveaxis(joint_map(T, E), 2, 0))     # (V, S, S)
    pi = stationary_dist(T)
    states = {tuple(np.round(pi, 14)): (pi, 1.0)}
    out = np.empty(K)
    for L in range(K):
        new, h = {}, 0.0
        for eta, w in states.values():
            nxt = eta @ Mx                                  # (V, S): P(x, s' | context)
            p = nxt.sum(axis=1)
            h += w * entropy_bits(p)
            for x in np.flatnonzero(p > 1e-14):
                v = nxt[x] / p[x]
                key = tuple(np.round(v, 14))
                new[key] = (v, new[key][1] + w * p[x]) if key in new else (v, w * p[x])
        out[L], states = h, new
    return out

def filter_probs(M: np.ndarray, eta0: np.ndarray, seqs: np.ndarray) -> np.ndarray:
    """
    HMM forward algorithm, normalised at every step: out[n, t] = P(X_t = . | x_0 .. x_{t-1})
    for a batch of sequences (N, L), started from eta0.  Exact for any presentation,
    unifilar or not, finite or not.
    """
    seqs = np.atleast_2d(np.asarray(seqs))
    N, L = seqs.shape
    eta = np.tile(np.asarray(eta0, dtype=float), (N, 1))
    out = np.empty((N, L, M.shape[2]))
    rows = np.arange(N)
    for t in range(L):
        nxt = np.einsum("ni,ijx->nxj", eta, M)                # (N, V, S)
        out[:, t] = nxt.sum(axis=2)
        sel = nxt[rows, seqs[:, t]]
        eta = sel / sel.sum(axis=1, keepdims=True)
    return out

def beta_tilt(T: np.ndarray, E: np.ndarray, beta: float):
    """
    beta-map (Aghamohammadi, Loomis, Mahoney & Crutchfield 2018): raise every
    non-zero M[s, s', x] to the power beta and renormalise with the Perron eigenvalue
    and right eigenvector.  Same support, new probabilities.
    """
    M = joint_map(T, E)
    Mb = np.zeros_like(M)
    live = M > 0
    Mb[live] = M[live] ** beta
    w, vr = np.linalg.eig(Mb.sum(axis=2))
    k = np.argmax(w.real)
    lam, r = w[k].real, np.abs(vr[:, k].real)
    return split_map(Mb * r[None, :, None] / (lam * r[:, None, None]))

##############################################################################
# COIN PROCESS
##############################################################################
def coin_transition(params: dict) -> np.ndarray:
    """
    state s is the coin's value at the previous step.
    """
    p, q = float(params["p"]), float(params["q"])
    return np.array([[1.0 - p, p],
                     [q,       1.0 - q]], dtype=float)

def coin_state_map(params: dict) -> np.ndarray:
    """
    E[s, s', x].  The token is carried by the edge s -> s' and is deterministic
    given that edge, so every live entry is one-hot.
    """
    E = np.zeros((2, 2, coin_vocab_size(params)))
    E[0, 0] = [1, 0, 0]          # 0 -> 0  emits 0
    E[0, 1] = [0, 1, 0]          # 0 -> 1  emits 1
    E[1, 0] = [0, 0, 1]          # 1 -> 0  emits 2
    E[1, 1] = [0, 1, 0]          # 1 -> 1  emits 1
    return E

def coin_vocab_size(params: dict) -> int:
    return 3

##############################################################################
# PERTURBED COIN (the coin's matched null)
##############################################################################
def pcoin_transition(params: dict) -> np.ndarray:
    """
    the coin's hidden chain, emitting the new hidden state with no herald.  On the
    line p + q = 1 both states emit the same distribution, the output is i.i.d.
    and the epsilon-machine has one state, while T.shape[0] stays 2.
    """
    return coin_transition(params)

def pcoin_state_map(params: dict) -> np.ndarray:
    """
    E[s, s', x] = 1 for x = s': arriving in state s' emits token s'.
    """
    E = np.zeros((2, 2, pcoin_vocab_size(params)))
    E[:, 0] = [1, 0]             # -> 0  emits 0
    E[:, 1] = [0, 1]             # -> 1  emits 1
    return E

def pcoin_vocab_size(params: dict) -> int:
    return 2

##############################################################################
# I.I.D. CONTROL
##############################################################################
def iid_transition(params: dict) -> np.ndarray:
    """
    one state: the zero-memory control.
    """
    return np.ones((1, 1))

def iid_state_map(params: dict) -> np.ndarray:
    """
    every token rides the self-loop, token x with probability params["probs"][x].
    """
    E = np.zeros((1, 1, iid_vocab_size(params)))
    E[0, 0] = np.asarray(params["probs"], dtype=float)
    return E

def iid_vocab_size(params: dict) -> int:
    return len(params["probs"])

##############################################################################
# FLOWER PROCESS
##############################################################################
def make_dice(n: int, m: int, seed: int = 42) -> np.ndarray:
    """
    the repository's dice (make_dice in Experimental_pipeline/processes.py): n rows
    of m face probabilities drawn from Dirichlet(1, ..., 1).  The dice are part of
    the process, so the same seed must be used wherever the flower is rebuilt.
    """
    return np.random.default_rng(seed).dirichlet(np.ones(m), size=n)

def flower_transition(params: dict) -> np.ndarray:
    """
    state 0 is the hub, state k+1 is "die k has been selected and is rolled on
    the way back to the hub".  Period 2: the chain alternates hub, petal, hub, ...
    so pi = [1/2, 1/2n, ..., 1/2n] is stationary but not a limit -- seeding from
    pi randomises the phase.
    """
    n = int(params["n"])
    T = np.zeros((n + 1, n + 1), dtype=float)
    T[0, 1:] = 1.0 / n
    T[1:, 0] = 1.0
    return T

def flower_state_map(params: dict) -> np.ndarray:
    """
    E[s, s', x].  Mirror of the coin: the transition out of a petal is certain
    but the token on it is not, which is exactly what the third axis buys and
    what a single integer label per edge could not express.  Tokens 0..n-1 name
    the selected die, tokens n..n+m-1 are the faces; `dice_probs` defaults to
    make_dice(n, m, params.get("dice_seed", 42)).

    The n+1 states are the minimal causal states only if the rows of `dice_probs`
    are pairwise distinct: identical dice give those petals the same future, so they
    are one causal state.  epsilon_machine minimises, so generator.n_causal and
    statistical_c already account for that -- what identical dice inflate is H[pi] of
    this raw presentation, not the reported C_mu.
    """
    n, m = int(params["n"]), int(params["m"])
    dp = params.get("dice_probs")
    dp = make_dice(n, m, int(params.get("dice_seed", 42))) if dp is None else np.asarray(dp, dtype=float)
    E = np.zeros((n + 1, n + 1, flower_vocab_size(params)))
    for k in range(n):
        E[0, k + 1, k]  = 1.0        # hub      --select die k--> petal k+1, token k
        E[k + 1, 0, n:] = dp[k]      # petal k+1 --roll die k--> hub, token n+j w.p. dp[k, j]
    return E

def flower_vocab_size(params: dict) -> int:
    return int(params["n"]) + int(params["m"])

##############################################################################
# RANDOM INSERTION PROCESS (RIP)
##############################################################################
def rip_transition(params: dict) -> np.ndarray:
    """
    states A, B, C = 0, 1, 2.  A emits a bit with P(0) = p: a 0 goes to B, a 1
    to C.  B inserts a random bit and goes to C.  C emits 1 and returns to A.
    """
    p = float(params["p"])
    return np.array([[0.0, p,   1.0 - p],
                     [0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0]], dtype=float)

def rip_state_map(params: dict) -> np.ndarray:
    """
    E[s, s', x].  The edge B -> C carries both tokens: the inserted bit is 0
    with probability q.
    """
    q = float(params["q"])
    E = np.zeros((3, 3, rip_vocab_size(params)))
    E[0, 1] = [1, 0]             # A -> B  emits 0
    E[0, 2] = [0, 1]             # A -> C  emits 1
    E[1, 2] = [q, 1.0 - q]       # B -> C  emits the inserted bit
    E[2, 0] = [0, 1]             # C -> A  emits 1
    return E

def rip_vocab_size(params: dict) -> int:
    return 2

##############################################################################
# RANDOM NOISY COPY (RnC)
##############################################################################
def rnc_transition(params: dict) -> np.ndarray:
    """
    states A, B, C = 0, 1, 2.  A emits a bit with P(0) = p: a 0 goes to B, a 1
    to C.  B copies the 0; C copies the 1, flipped to 0 with probability q.  Both
    return to A.
    """
    p = float(params["p"])
    return np.array([[0.0, p,   1.0 - p],
                     [1.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0]], dtype=float)

def rnc_state_map(params: dict) -> np.ndarray:
    """
    E[s, s', x].  The edge C -> A carries both tokens: the copy of the 1 is
    flipped to 0 with probability q.
    """
    q = float(params["q"])
    E = np.zeros((3, 3, rnc_vocab_size(params)))
    E[0, 1] = [1, 0]             # A -> B  emits 0
    E[0, 2] = [0, 1]             # A -> C  emits 1
    E[1, 0] = [1, 0]             # B -> A  copies the 0
    E[2, 0] = [q, 1.0 - q]       # C -> A  copies the 1, flipped with probability q
    return E

def rnc_vocab_size(params: dict) -> int:
    return 2

def rnc_zero_locus_q(p: float) -> float:
    """
    q*(p) = (1 - 2p) / (1 - p): C+ = C- exactly on this curve, for 0 < p < 1/2.
    """
    return (1.0 - 2.0 * p) / (1.0 - p)

##############################################################################
# BUTTERFLY
##############################################################################
BUTTERFLY_EDGES = [(0, 2, 1), (0, 3, 3),       # A -2-> B,  A -3-> D
                   (1, 0, 2), (1, 4, 1),       # B -0-> C,  B -4-> B
                   (2, 1, 0), (2, 6, 2),       # C -1-> A,  C -6-> C
                   (3, 0, 4), (3, 5, 3),       # D -0-> E,  D -5-> D
                   (4, 1, 0), (4, 7, 4)]       # E -1-> A,  E -7-> E    (s, x, s'), each w.p. 1/2

def butterfly_transition(params: dict) -> np.ndarray:
    """
    states A..E = 0..4: two wings A -> B -> C -> A and A -> D -> E -> A, every
    edge of BUTTERFLY_EDGES taken with probability 1/2.
    """
    T = np.zeros((5, 5))
    for s, x, s_next in BUTTERFLY_EDGES:
        T[s, s_next] += 0.5
    return T

def butterfly_state_map(params: dict) -> np.ndarray:
    """
    one token per edge.  The wings share tokens 0 and 1; each wing state has a
    private self-loop token.
    """
    E = np.zeros((5, 5, butterfly_vocab_size(params)))
    for s, x, s_next in BUTTERFLY_EDGES:
        E[s, s_next, x] = 1.0
    return E

def butterfly_vocab_size(params: dict) -> int:
    return 8

##############################################################################
# EXPLOSIVE IRREVERSIBILITY
##############################################################################
def explosive_transition(params: dict) -> np.ndarray:
    """
    states A, B = 0, 1: A if the last token other than 1 was a 0, B if it was a
    2.  After a 2, another 2 is forbidden until a 0 intervenes, however many 1s
    come between.  Finite forwards, countably infinite backwards.
    """
    return np.array([[2 / 3, 1 / 3],
                     [1 / 2, 1 / 2]], dtype=float)

def explosive_state_map(params: dict) -> np.ndarray:
    """
    E[s, s', x].  The self-loop on A carries tokens 0 and 1 with equal probability.
    """
    E = np.zeros((2, 2, explosive_vocab_size(params)))
    E[0, 0] = [0.5, 0.5, 0.0]    # A -> A  emits 0 or 1
    E[0, 1] = [0, 0, 1]          # A -> B  emits 2
    E[1, 0] = [1, 0, 0]          # B -> A  emits 0
    E[1, 1] = [0, 1, 0]          # B -> B  emits 1
    return E

def explosive_vocab_size(params: dict) -> int:
    return 3

##############################################################################
# T3 (largest Delta C among three-state binary topological machines)
##############################################################################
def t3_transition(params: dict) -> np.ndarray:
    """
    states A, B, C = 0, 1, 2, found by exhaustive enumeration in the survey (not
    from a paper).  Finite forwards, countably infinite backwards.
    """
    return np.array([[0.0, 1.0, 0.0],
                     [0.5, 0.0, 0.5],
                     [0.0, 0.5, 0.5]], dtype=float)

def t3_state_map(params: dict) -> np.ndarray:
    """
    one token per edge.
    """
    E = np.zeros((3, 3, t3_vocab_size(params)))
    E[0, 1] = [1, 0]             # A -> B  emits 0
    E[1, 0] = [0, 1]             # B -> A  emits 1
    E[1, 2] = [1, 0]             # B -> C  emits 0
    E[2, 1] = [0, 1]             # C -> B  emits 1
    E[2, 2] = [1, 0]             # C -> C  emits 0
    return E

def t3_vocab_size(params: dict) -> int:
    return 2

##############################################################################
# EVEN PROCESS
##############################################################################
def even_transition(params: dict) -> np.ndarray:
    """
    states A, B = 0, 1: every block of 1s has even length.  A: a 0 stays with
    probability p, a 1 goes to B; B: a 1 returns to A.
    """
    p = float(params["p"])
    return np.array([[p,   1.0 - p],
                     [1.0, 0.0]], dtype=float)

def even_state_map(params: dict) -> np.ndarray:
    """
    one token per edge.
    """
    E = np.zeros((2, 2, even_vocab_size(params)))
    E[0, 0] = [1, 0]             # A -> A  emits 0
    E[0, 1] = [0, 1]             # A -> B  emits 1
    E[1, 0] = [0, 1]             # B -> A  emits 1
    return E

def even_vocab_size(params: dict) -> int:
    return 2

##############################################################################
# GOLDEN MEAN PROCESS
##############################################################################
def gm_transition(params: dict) -> np.ndarray:
    """
    states A, B = 0, 1: no two consecutive 0s.  A: a 1 stays with probability p,
    a 0 goes to B; B: a 1 returns to A.
    """
    p = float(params["p"])
    return np.array([[p,   1.0 - p],
                     [1.0, 0.0]], dtype=float)

def gm_state_map(params: dict) -> np.ndarray:
    """
    one token per edge.
    """
    E = np.zeros((2, 2, gm_vocab_size(params)))
    E[0, 0] = [0, 1]             # A -> A  emits 1
    E[0, 1] = [1, 0]             # A -> B  emits 0
    E[1, 0] = [0, 1]             # B -> A  emits 1
    return E

def gm_vocab_size(params: dict) -> int:
    return 2

##############################################################################
# NEMO PROCESS
##############################################################################
def nemo_transition(params: dict) -> np.ndarray:
    """
    states A, B, C = 0, 1, 2.  A: a 1 stays with probability p, a 0 goes to B;
    B: a 0 goes to C; C returns to A emitting 1 with probability q, else 0.
    """
    p = float(params["p"])
    return np.array([[p,   1.0 - p, 0.0],
                     [0.0, 0.0,     1.0],
                     [1.0, 0.0,     0.0]], dtype=float)

def nemo_state_map(params: dict) -> np.ndarray:
    """
    E[s, s', x].  The edge C -> A carries both tokens.
    """
    q = float(params["q"])
    E = np.zeros((3, 3, nemo_vocab_size(params)))
    E[0, 0] = [0, 1]             # A -> A  emits 1
    E[0, 1] = [1, 0]             # A -> B  emits 0
    E[1, 2] = [1, 0]             # B -> C  emits 0
    E[2, 0] = [1.0 - q, q]       # C -> A  emits 1 with probability q
    return E

def nemo_vocab_size(params: dict) -> int:
    return 2

##############################################################################
# RESTRICTED GOLDEN MEAN (parameter k)
##############################################################################
def rgm_transition(params: dict) -> np.ndarray:
    """
    states 0..k: consecutive 0s are separated by at least k 1s.  State 0: a 1
    stays, a 0 starts the forced run (probability 1/2 each); states 1..k emit the
    k forced 1s.  k = 1 is the Golden Mean at p = 1/2.
    """
    k = int(params["k"])
    T = np.zeros((k + 1, k + 1))
    T[0, 0] = T[0, 1] = 0.5
    for i in range(1, k):
        T[i, i + 1] = 1.0
    T[k, 0] = 1.0
    return T

def rgm_state_map(params: dict) -> np.ndarray:
    """
    one token per edge.
    """
    k = int(params["k"])
    E = np.zeros((k + 1, k + 1, rgm_vocab_size(params)))
    E[0, 0] = [0, 1]             # 0 -> 0  emits 1
    E[0, 1] = [1, 0]             # 0 -> 1  emits 0
    for i in range(1, k):
        E[i, i + 1] = [0, 1]     # forced 1s
    E[k, 0] = [0, 1]             # the last forced 1
    return E

def rgm_vocab_size(params: dict) -> int:
    return 2

##############################################################################
# STRETCHED GOLDEN MEAN (parameter k)
##############################################################################
def sgm_transition(params: dict) -> np.ndarray:
    """
    states 0..k: blocks of 0s have length exactly k.  State 0: a 1 stays with
    probability p (default 1/2), a 0 starts a block; states 1..k-1 emit the
    remaining forced 0s and state k the 1 that closes the block.
    """
    k, p = int(params["k"]), float(params.get("p", 0.5))
    T = np.zeros((k + 1, k + 1))
    T[0, 0], T[0, 1] = p, 1.0 - p
    for i in range(1, k):
        T[i, i + 1] = 1.0
    T[k, 0] = 1.0
    return T

def sgm_state_map(params: dict) -> np.ndarray:
    """
    one token per edge.
    """
    k = int(params["k"])
    E = np.zeros((k + 1, k + 1, sgm_vocab_size(params)))
    E[0, 0] = [0, 1]             # 0 -> 0  emits 1
    E[0, 1] = [1, 0]             # 0 -> 1  emits the first 0 of the block
    for i in range(1, k):
        E[i, i + 1] = [1, 0]     # forced 0s
    E[k, 0] = [0, 1]             # the 1 that closes the block
    return E

def sgm_vocab_size(params: dict) -> int:
    return 2

##############################################################################
# NEXT-NEAREST-NEIGHBOUR ISING CHAIN
##############################################################################
def ising_transition(params: dict) -> np.ndarray:
    """
    H = -J1 sum s_i s_{i+1} - J2 sum s_i s_{i+2} - B sum s_i at temperature
    `temp` (named so that T stays the transition matrix); defaults J1 = 1,
    J2 = 0.25, B = 0, temp = 1.  State = the last two spins, index
    2 * older + newer with bit 1 = up; T is the transfer matrix made stochastic
    with its Perron vector.
    """
    J1, J2 = float(params.get("J1", 1.0)), float(params.get("J2", 0.25))
    B, temp = float(params.get("B", 0.0)), float(params.get("temp", 1.0))
    spin = lambda bit: 1 if bit else -1
    W = np.zeros((4, 4))
    for i in range(4):
        s1, s2 = spin(i >> 1 & 1), spin(i & 1)
        for j in range(4):
            t1, t2 = spin(j >> 1 & 1), spin(j & 1)
            if t1 == s2:                       # the two blocks overlap on the shared spin
                W[i, j] = np.exp((J1 * s2 * t2 + J2 * s1 * t2 + B * t2) / temp)
    w, vr = np.linalg.eig(W)
    k = np.argmax(w.real)
    lam, uR = w[k].real, np.abs(vr[:, k].real)
    return W * uR[None, :] / (lam * uR[:, None])

def ising_state_map(params: dict) -> np.ndarray:
    """
    the token is the newest spin (1 = up).
    """
    E = np.zeros((4, 4, ising_vocab_size(params)))
    for i in range(4):
        for j in range(4):
            if (j >> 1 & 1) == (i & 1):
                E[i, j, j & 1] = 1.0
    return E

def ising_vocab_size(params: dict) -> int:
    return 2

##############################################################################
# SIMPLE NONUNIFILAR SOURCE (SNS)
##############################################################################
def sns_transition(params: dict) -> np.ndarray:
    """
    states A, B = 0, 1.  A emits 0 and moves to B with probability 1 - p, or
    stays; B emits 0 and stays with probability q, or emits 1 and returns to A.
    A emits 0 either way, so the tokens do not show whether it moved: the one
    non-unifilar presentation here.  Its causal states ("n 0s since the last 1")
    are countably infinite, so T.shape[0] = 2 is not n_causal.
    """
    p, q = float(params["p"]), float(params["q"])
    return np.array([[p,       1.0 - p],
                     [1.0 - q, q]], dtype=float)

def sns_state_map(params: dict) -> np.ndarray:
    """
    one token per edge; A -> A and A -> B both emit 0.
    """
    E = np.zeros((2, 2, sns_vocab_size(params)))
    E[0, 0] = E[0, 1] = [1, 0]   # A -> A, A -> B  emit 0
    E[1, 1] = [1, 0]             # B -> B  emits 0
    E[1, 0] = [0, 1]             # B -> A  emits 1
    return E

def sns_vocab_size(params: dict) -> int:
    return 2

##############################################################################
# ANY MACHINE AS AN EDGE LIST
##############################################################################
def edges_transition(params: dict) -> np.ndarray:
    """
    params = dict(n_states=S, vocab=V, edges=[(s, x, s', prob), ...]): an
    enumerated topological machine, or any machine copied from a paper.
    """
    S = int(params["n_states"])
    T = np.zeros((S, S))
    for s, x, s_next, prob in params["edges"]:
        T[s, s_next] += prob
    return T

def edges_state_map(params: dict) -> np.ndarray:
    """
    E from the edge list; an edge (s, s') listed with several tokens shares them
    in proportion to their probabilities.
    """
    S, V = int(params["n_states"]), edges_vocab_size(params)
    M = np.zeros((S, S, V))
    for s, x, s_next, prob in params["edges"]:
        M[s, s_next, x] += prob
    return split_map(M)[1]

def edges_vocab_size(params: dict) -> int:
    return int(params["vocab"])

##############################################################################
# BETA-MAP OF ANY REGISTERED PROCESS
##############################################################################
def tilt_transition(params: dict) -> np.ndarray:
    """
    params = dict(base=<key of PROCESS>, base_params=..., beta=...).  The typical
    sequences of the result are a rare class of the base process; support and
    vocabulary are unchanged.
    """
    return _tilt(params)[0]

def tilt_state_map(params: dict) -> np.ndarray:
    return _tilt(params)[1]

def tilt_vocab_size(params: dict) -> int:
    return PROCESS[params["base"]]["vocab_size"](params["base_params"])

def _tilt(params: dict):
    f = PROCESS[params["base"]]
    base = params["base_params"]
    return beta_tilt(f["transition"](base), f["state_map"](base), float(params["beta"]))

##############################################################################
# REGISTRY
##############################################################################
PROCESS = {
    "coin":      {"transition": coin_transition,
                  "state_map":  coin_state_map,
                  "vocab_size": coin_vocab_size},
    "pcoin":     {"transition": pcoin_transition,
                  "state_map":  pcoin_state_map,
                  "vocab_size": pcoin_vocab_size},
    "iid":       {"transition": iid_transition,
                  "state_map":  iid_state_map,
                  "vocab_size": iid_vocab_size},
    "flower":    {"transition": flower_transition,
                  "state_map":  flower_state_map,
                  "vocab_size": flower_vocab_size},
    "rip":       {"transition": rip_transition,
                  "state_map":  rip_state_map,
                  "vocab_size": rip_vocab_size},
    "rnc":       {"transition": rnc_transition,
                  "state_map":  rnc_state_map,
                  "vocab_size": rnc_vocab_size},
    "butterfly": {"transition": butterfly_transition,
                  "state_map":  butterfly_state_map,
                  "vocab_size": butterfly_vocab_size},
    "explosive": {"transition": explosive_transition,
                  "state_map":  explosive_state_map,
                  "vocab_size": explosive_vocab_size},
    "t3":        {"transition": t3_transition,
                  "state_map":  t3_state_map,
                  "vocab_size": t3_vocab_size},
    "even":      {"transition": even_transition,
                  "state_map":  even_state_map,
                  "vocab_size": even_vocab_size},
    "gm":        {"transition": gm_transition,
                  "state_map":  gm_state_map,
                  "vocab_size": gm_vocab_size},
    "nemo":      {"transition": nemo_transition,
                  "state_map":  nemo_state_map,
                  "vocab_size": nemo_vocab_size},
    "rgm":       {"transition": rgm_transition,
                  "state_map":  rgm_state_map,
                  "vocab_size": rgm_vocab_size},
    "sgm":       {"transition": sgm_transition,
                  "state_map":  sgm_state_map,
                  "vocab_size": sgm_vocab_size},
    "ising":     {"transition": ising_transition,
                  "state_map":  ising_state_map,
                  "vocab_size": ising_vocab_size},
    "sns":       {"transition": sns_transition,
                  "state_map":  sns_state_map,
                  "vocab_size": sns_vocab_size},
    "edges":     {"transition": edges_transition,
                  "state_map":  edges_state_map,
                  "vocab_size": edges_vocab_size},
    "tilt":      {"transition": tilt_transition,
                  "state_map":  tilt_state_map,
                  "vocab_size": tilt_vocab_size},
}