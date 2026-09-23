import numpy as np
"""
this file contains all generator of all the processes of interest,
together with their theoretical quantities and structures

Every process is described by three functions of `params`, and nothing else:

    transition(params)  -> T[s, s']    = P(s_{t+1} = s' | s_t = s)
    state_map(params)   -> E[s, s', x] = P(x_t = x | s_t = s, s_{t+1} = s')
    vocab_size(params)  -> |X|

The causal map M(x, s' | s) is the product T[:, :, None] * E; every other
quantity -- emission P(x|s), entropy rate, statistical complexity -- is a
marginal of it, and `tokens_from_map` samples from it without knowing which
process it is holding.  Register a new process in PROCESS at the bottom and
nothing else in the codebase has to learn its name.

The two factors carry the randomness in opposite places, which is why they are
kept separate: the coin has stochastic transitions and a deterministic token on
each edge, the flower has deterministic transitions out of each petal and a
stochastic token on the way back.
"""

##############################################################################
# GENERIC MACHINERY -- process agnostic
##############################################################################
def stationary_dist(T: np.ndarray) -> np.ndarray:
    """
    left eigenvector of T for eigenvalue 1, normalised to a distribution.
    """
    eigvals, eigvecs = np.linalg.eig(np.asarray(T, dtype=float).T)
    pi = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
    return pi / pi.sum()


def tokens_from_map(n_tokens: int, T: np.ndarray, E: np.ndarray,
                    pi: np.ndarray, rng) -> np.ndarray:
    """
    `n_tokens` tokens of any edge-emitting machine (T, E), started from `pi`.

    One uniform per token: the pair (s_{t+1}, x_t) is drawn jointly from row
    s_t of the flattened causal map, so a single searchsorted does both the
    state update and the emission and there is no per-process branching.
    """
    S, _, V = E.shape
    C = (np.asarray(T, dtype=float)[:, :, None] * E).reshape(S, S * V).cumsum(axis=1)
    C /= C[:, -1:]                              # renormalise away fp drift
    u = rng.random(n_tokens + 1)                # every draw from the passed rng

    s = int(np.searchsorted(np.asarray(pi, dtype=float).cumsum(), u[0]))
    out = np.empty(n_tokens, dtype=np.int64)
    for t in range(n_tokens):
        s, out[t] = divmod(int(np.searchsorted(C[s], u[t + 1])), V)
    return out


def emission_dist(T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    P(x | s), the ideal next-token prediction from each causal state. (S, V)
    """
    return (np.asarray(T, dtype=float)[:, :, None] * E).sum(axis=1)


def entropy_rate(T: np.ndarray, E: np.ndarray, pi: np.ndarray) -> float:
    """
    h_mu = sum_s pi_s H[P(x|s)] in bits/token; exact for a unifilar machine,
    and the cross-entropy floor any model trained on these tokens can reach.
    """
    P = emission_dist(T, E)
    logP = np.zeros_like(P)
    np.log2(P, out=logP, where=P > 0)
    return float(-(np.asarray(pi, dtype=float) * (P * logP).sum(axis=1)).sum())


def statistical_complexity(pi: np.ndarray) -> float:
    """
    C_mu = H[pi] over the causal states, in bits.  Only the true statistical
    complexity when the machine is minimal (see flower_state_map).
    """
    pi = np.asarray(pi, dtype=float)
    logpi = np.zeros_like(pi)
    np.log2(pi, out=logpi, where=pi > 0)
    return float(-(pi * logpi).sum())


def check_map(T: np.ndarray, E: np.ndarray) -> None:
    """
    (T, E) is a well-formed machine: T stochastic, E stochastic on every live
    edge and zero on every dead one.
    """
    T = np.asarray(T, dtype=float)
    assert T.ndim == 2 and T.shape[0] == T.shape[1], f"T must be square, got {T.shape}"
    assert E.shape[:2] == T.shape, f"E {E.shape} does not match T {T.shape}"
    assert np.allclose(T.sum(axis=1), 1.0), "rows of T must sum to 1"
    assert np.allclose(E.sum(axis=2)[T > 0], 1.0), "E must sum to 1 on every edge with T > 0"
    assert np.allclose(E[T == 0], 0.0), "E must be 0 wherever T == 0"


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
# FLOWER PROCESS
##############################################################################
def flower_transition(params: dict) -> np.ndarray:
    """
    state 0 is the hub, state k+1 is "face k was just rolled".  Period 2: the
    chain alternates hub, petal, hub, ... so pi = [1/2, 1/2n, ..., 1/2n] is
    stationary but not a limit -- seeding from pi randomises the phase.
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
    what a single integer label per edge could not express.

    The n+1 states are the minimal causal states only if the rows of
    `dice_probs` are pairwise distinct -- identical dice make those petals
    statistically indistinguishable and inflate C_mu above its true value.
    """
    n, m = int(params["n"]), int(params["m"])
    dp = np.asarray(params["dice_probs"], dtype=float)      # (n, m)
    E = np.zeros((n + 1, n + 1, flower_vocab_size(params)))
    for k in range(n):
        E[0, k + 1, k]  = 1.0        # hub      --face k--> petal k+1, token k
        E[k + 1, 0, n:] = dp[k]      # petal k+1 --> hub,  token n+j w.p. dp[k, j]
    return E


def flower_vocab_size(params: dict) -> int:
    return int(params["n"]) + int(params["m"])


##############################################################################
# REGISTRY 
##############################################################################
PROCESS = {
    "coin":   {"transition": coin_transition,
               "state_map":  coin_state_map,
               "vocab_size": coin_vocab_size},
    "flower": {"transition": flower_transition,
               "state_map":  flower_state_map,
               "vocab_size": flower_vocab_size},
}
