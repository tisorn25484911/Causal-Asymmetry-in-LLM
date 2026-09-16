"""
The two processes: how to draw them, and what their causal structure provably is.

Sampler and theory live in one file on purpose.  "How to generate it" and "what
its epsilon-machine is" are the same subject, and separating them is how a
generator change silently invalidates a closed form.

Both processes emit TOKENS, and `seq_len` counts tokens for both.  The old tree
counted flower `seq_len` in *cycles* (two tokens each), which is why a flower
"seq_len=2000" produced 4000 tokens; that trap is gone here.

Vocabulary
    coin    V = 3     0, 1, 2
    flower  V = n+m   0..n-1 die selections, n..n+m-1 roll outcomes
"""
import numpy as np
import torch
from torch.utils.data import Dataset

# Rounding used when deciding whether two backward outcome-posteriors are the
# same state.  Nine decimal places: exact-arithmetic dice merge, floating noise
# does not.
MERGE_ROUND_DP = 9


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════
class SequenceDataset(Dataset):
    """
    (num_samples, seq_len+1) tokens -> (input, target) = (x[:-1], x[1:]).
    """

    def __init__(self, seqs):
        self.seqs = np.asarray(seqs, dtype=np.int64)
        if self.seqs.ndim != 2:
            raise ValueError(f"expected (num_samples, T), got {self.seqs.shape}")
        self.seq_len = self.seqs.shape[1] - 1      # scored positions

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.seqs[idx])
        return x[:-1], x[1:]


# ══════════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════════
def _coin_tokens(n_tokens: int, p: float, q: float, rng) -> np.ndarray:
    """
    `n_tokens` tokens of the coin process, started from the stationary state.

    Hidden chain on {0,1}:  0 -> 1 w.p. p,  1 -> 0 w.p. q.
    Emission:  state 1 -> token 1;  state 0 after 1 -> token 2;  0 after 0 -> 0.

    The emission depends on the PREVIOUS state as well as the current one, so
    the initial draw needs both.  Stationary  pi = [q, p]/(p+q), and
    P(prev=1 | cur=0) = pi1*T[1][0]/pi0 = p.

    (The old sampler used q there rather than p -- wrong conditional, though it
    biased only the very first token of each sequence.  With a 250-token burn-in
    the initialisation is unobservable either way; it is written correctly here
    rather than reproduced.)
    """
    if rng.random() < p / (p + q):
        cur, prev = 1, 1
    else:
        cur, prev = 0, (1 if rng.random() < p else 0)

    out = np.empty(n_tokens, dtype=np.int64)
    for t in range(n_tokens):
        if cur == 1:
            out[t] = 1
        elif prev == 1:
            out[t] = 2
        else:
            out[t] = 0
        prev = cur
        if cur == 0:
            cur = 1 if rng.random() < p else 0
        else:
            cur = 0 if rng.random() < q else 1
    return out


def _flower_tokens(n_tokens: int, n: int, m: int, dice_probs, rng) -> np.ndarray:
    """
    `n_tokens` tokens of the n-m flower process.

    One cycle emits two tokens -- a uniform die selection, then that die's roll
    -- so ceil(n_tokens/2) cycles are drawn and the tail is trimmed.  The
    process is iid across cycles, so it is at stationarity from the first cycle
    and the burn-in is belt-and-braces.
    """
    dp = np.asarray(dice_probs, dtype=float)
    n_cycles = -(-n_tokens // 2)                       # ceil
    out = np.empty(2 * n_cycles, dtype=np.int64)
    for c in range(n_cycles):
        die = int(rng.integers(0, n))
        out[2 * c]     = die
        out[2 * c + 1] = n + int(rng.choice(m, p=dp[die]))
    return out[:n_tokens]


def generate(kind: str, params: dict, num_samples: int, seq_len: int,
             burn_in: int, rng) -> np.ndarray:
    """
    (num_samples, seq_len + 1) tokens, every one of them trained on.

    `seq_len` is in TOKENS for both processes.  The +1 is the shift: the Dataset
    emits (x[:-1], x[1:]), so seq_len+1 tokens give seq_len scored positions.

    `burn_in` tokens are drawn and discarded first, so every kept token is at
    stationarity regardless of how the chain was initialised.
    """
    total = burn_in + seq_len + 1
    rows = []
    for _ in range(num_samples):
        if kind == "coin":
            tok = _coin_tokens(total, params["p"], params["q"], rng)
        elif kind == "flower":
            tok = _flower_tokens(total, params["n"], params["m"],
                                 params["dice_probs"], rng)
        else:
            raise ValueError(f"unknown process {kind!r}; expected 'coin' or 'flower'")
        rows.append(tok[burn_in:])
    out = np.stack(rows)
    assert out.shape == (num_samples, seq_len + 1), out.shape
    return out


def make_dice(n: int, m: int, seed: int) -> np.ndarray:
    """
    The (n, m) dice that DEFINE a flower process.

    Drawn from their own seed, never the repeat seed: the dice set C+ and C-, so
    resampling them per repeat would average over different processes.
    """
    return np.random.default_rng(seed).dirichlet(np.ones(m), size=n)


# ══════════════════════════════════════════════════════════════════════════
# CLOSED FORMS
# ══════════════════════════════════════════════════════════════════════════
def entropy_bits(probs) -> float:
    """Shannon entropy in bits, ignoring zero-probability atoms."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def entropy_rate_coin(p: float, q: float) -> float:
    """
    Entropy rate H_inf (bits/token) of the coin process.

        H_inf = pi0*H2(p) + pi1*H2(q),   pi = [q, p]/(p+q)

    Time-reversal invariant, so the same value is the reference for both arms.
    """
    def _h2(a):
        b = 1.0 - a
        a, b = max(a, 1e-12), max(b, 1e-12)
        return -a * np.log2(a) - b * np.log2(b)
    return float((q / (p + q)) * _h2(p) + (p / (p + q)) * _h2(q))


def flower_entropy_rate(n: int, m: int, dice_probs) -> float:
    """
    Entropy rate H_inf (bits/token) of the n-m flower process.

    Half the tokens are die selections, uniform over n, contributing log2(n)
    bits each; the other half are rolls of a uniformly chosen die.

        H_inf = (1/2) log2(n) + (1/2) mean_i H(dice_probs[i])
    """
    dp = np.asarray(dice_probs, dtype=float)
    return float(0.5 * np.log2(n)
                 + 0.5 * np.mean([entropy_bits(dp[i]) for i in range(n)]))


def _face_marginal(n: int, dice_probs) -> np.ndarray:
    """
    (m,) stationary probability of each outcome, from the dice alone: a
    uniform 1/n choice of die, then that die's bias.

        P(face j) = (1/n) sum_i p^i_j
    """
    dp = np.asarray(dice_probs, dtype=float)
    return dp.sum(axis=0) / n


def _bayes_rev_prob(n: int, dice_probs) -> np.ndarray:
    """
    (n, m) posterior P(die = i | face = j), by Bayes from the dice alone:

        P(die i | face j) = P(die i, face j) / P(face j) = (p^i_j / n) / P(face j)

    Column j is everything a backward observer can retrodict about the die
    from outcome j, so two faces with the same column are the same backward
    causal state.  A face that never occurs gets a zero column.
    """
    dp     = np.asarray(dice_probs, dtype=float)
    joint  = dp / n
    p_face = _face_marginal(n, dp)
    return np.divide(joint, p_face, out=np.zeros_like(joint), where=p_face > 0)


def _outcome_classes(dp, merge_tol: float | None = None) -> tuple[np.ndarray, list]:
    """
    (class of each face, mass of each class): the backward outcome states.

    Two outcomes j and j' are the same backward state exactly when they induce
    the same posterior over dice (_bayes_rev_prob) -- i.e. when columns j and
    j' of dice_probs are proportional.  A state's mass is the outcome marginal
    (_face_marginal) summed over its faces.  Classes are numbered in order of
    first appearance; a face that never occurs gets class -1 and no mass.
    Closed form throughout: nothing here is sampled.

    The ONE merge rule.  _merged_outcome_mass (and through it flower_complexity,
    causal_state_count, causal_state_occupancy, flower_rev_transition_matrix)
    and the backward true_machine all read it from here.
    """
    dp = np.asarray(dp, dtype=float)
    n, m = dp.shape
    pi_outcome = _face_marginal(n, dp)
    posterior  = _bayes_rev_prob(n, dp)

    cls  = np.full(m, -1, dtype=np.int64)
    mass: list = []
    if merge_tol is None:
        keys: dict[tuple, int] = {}
        for j in range(m):
            if pi_outcome[j] <= 0:                     # outcome never occurs
                continue
            key = tuple(np.round(posterior[:, j], MERGE_ROUND_DP))
            if key not in keys:
                keys[key] = len(mass)
                mass.append(0.0)
            cls[j] = keys[key]
            mass[cls[j]] += float(pi_outcome[j])
        return cls, mass

    reps = []
    for j in range(m):
        if pi_outcome[j] <= 0:
            continue
        for k, r in enumerate(reps):
            if np.max(np.abs(posterior[:, j] - r)) <= merge_tol:
                cls[j] = k
                mass[k] += float(pi_outcome[j])
                break
        else:
            cls[j] = len(reps)
            reps.append(posterior[:, j])
            mass.append(float(pi_outcome[j]))
    return cls, mass


def _merged_outcome_mass(dp, merge_tol: float | None = None) -> list:
    """
    Mass of each *distinguishable* backward outcome state -- the masses half of
    _outcome_classes, kept as the name the closed forms call.
    """
    return _outcome_classes(dp, merge_tol)[1]


def coin_complexity(p: float, q: float) -> tuple[float, float]:
    """
    Closed-form (C+, C-) for the coin process, in bits.

        forward   2 states, pi = [q, p]/(p+q)
        backward  3 states, pi = [q-pq, p, pq]/(p+q)
    """
    fwd = np.array([q, p], dtype=float)
    bwd = np.array([q - p * q, p, p * q], dtype=float)
    return (entropy_bits(fwd / fwd.sum()), entropy_bits(bwd / bwd.sum()))


def flower_complexity(n: int, m: int, dice_probs,
                      merge_tol: float | None = None) -> tuple[float, float]:
    """
    Closed-form (C+, C-) for the n-m flower process, in bits.

    FORWARD states: one "a roll just happened" state (the next token is a
    uniform die selection, independent of everything before) plus one per die.
    Masses 1/2 and 1/(2n), so  C+ = 1 + (1/2) log2(n).

    BACKWARD states: one "the current token is a selection" state plus one per
    *distinguishable* outcome, so  C- = 1 + (1/2) H(merged mass).

    Two consequences the older m+1 formula missed:
      * n=1 degenerates -- every outcome collapses to one state and C- = C+ = 1.
      * C- <= 1 + (1/2)log2(m), so C- > C+ requires m > n.
    """
    dp = np.asarray(dice_probs, dtype=float)
    if dp.shape != (n, m):
        raise ValueError(f"dice_probs must have shape ({n}, {m}), got {dp.shape}")
    if not np.allclose(dp.sum(axis=1), 1.0):
        raise ValueError(f"dice_probs rows must sum to 1, got {dp.sum(axis=1)}")

    C_plus  = 1.0 + 0.5 * np.log2(n)
    C_minus = 1.0 + 0.5 * entropy_bits(_merged_outcome_mass(dp, merge_tol))
    return float(C_plus), float(C_minus)


def causal_state_count(kind: str, mode: str, n=None, m=None, dice_probs=None,
                       merge_tol: float | None = None) -> int:
    """
    Number of causal states in this direction -- the true k an extractor should
    find, and the K a discrete bottleneck is budgeted against.

        coin    forward   2        backward  3
        flower  forward   n + 1    backward  1 + #distinguishable outcomes

    The flower backward count is NOT m+1 in general: generic Dirichlet dice do
    give m+1, but dice whose posterior columns are proportional merge, and at
    n=1 every outcome merges.
    """
    if mode not in ("forward", "backward"):
        raise ValueError(f"mode must be 'forward' or 'backward', got {mode!r}")
    if kind == "coin":
        return 2 if mode == "forward" else 3
    if kind == "flower":
        if n is None:
            raise ValueError("flower needs n")
        if mode == "forward":
            return int(n) + 1
        if dice_probs is None:
            raise ValueError("flower backward count needs dice_probs")
        return 1 + len(_merged_outcome_mass(dice_probs, merge_tol))
    raise ValueError(f"unknown process {kind!r}; expected 'coin' or 'flower'")


def causal_state_occupancy(kind: str, mode: str, p=None, q=None, n=None, m=None,
                           dice_probs=None, merge_tol: float | None = None):
    """
    Stationary probability of each causal state -- the distribution whose entropy
    IS the statistical complexity.

        coin   forward   [q, p]/(p+q)                        H = C+
               backward  [q-pq, p, pq]/(p+q)                 H = C-
        flower forward   [1/2] + [1/(2n)] x n                H = C+
               backward  [1/2] + [mass_j/2] per merged j     H = C-

    entropy(occupancy) == C by construction; the checks assert it rather than
    trusting it.  Used as the overlay on F1's occupancy bars.
    """
    if mode not in ("forward", "backward"):
        raise ValueError(f"mode must be 'forward' or 'backward', got {mode!r}")

    if kind == "coin":
        if p is None or q is None:
            raise ValueError("coin occupancy needs p and q")
        occ = (np.array([q, p], dtype=float) if mode == "forward"
               else np.array([q - p * q, p, p * q], dtype=float))
        return occ / occ.sum()

    if kind == "flower":
        if n is None:
            raise ValueError("flower needs n")
        if mode == "forward":
            return np.concatenate([[0.5], np.full(int(n), 0.5 / int(n))])
        if dice_probs is None:
            raise ValueError("flower backward occupancy needs dice_probs")
        mass = np.asarray(_merged_outcome_mass(dice_probs, merge_tol), dtype=float)
        return np.concatenate([[0.5], 0.5 * mass / mass.sum()])

    raise ValueError(f"unknown process {kind!r}; expected 'coin' or 'flower'")


# ── theoretical STATE-TO-STATE transition matrices ────────────────────────
# Forward:   T[i][j] = P(s_{t+1} = j | s_t = i).
# Backward:  T[i][j] = P(s_{t-1} = j | s_t = i) -- the same layout one step in
# REVERSE time, which is what extraction.transition_matrix_extraction records
# for a backward model: it prepends each generated token and reads the state at
# position 0, so consecutive readings are s_t then s_{t-1}.  Every state order
# matches causal_state_occupancy for the same (kind, mode).

def coin_transition_matrix(p: float, q: float) -> np.ndarray:
    """
    (2, 2) forward state transition matrix, state order [0, 1] matching
    causal_state_occupancy("coin", "forward").

        0 -> 1 w.p. p        1 -> 0 w.p. q

    Validated against extraction: median max-cell error 0.009 (worst 0.025)
    across the 24 two-state cells of the archived pq sweep.
    """
    return np.array([[1.0 - p, p],
                     [q,       1.0 - q]], dtype=float)


def coin_rev_transition_matrix(p: float, q: float) -> np.ndarray:
    """
    (3, 3) backward state transition matrix, state order [0, 1, 2] matching
    causal_state_occupancy("coin", "backward").

    Backward the causal state is the current TOKEN (Thompson et al., Fig. 2b),
    and the token before it is fixed by the hidden coin one and two steps back:

        0 -> 0 w.p. 1-p      0 -> 2 w.p. p
        1 -> 0 w.p. q(1-p)   1 -> 1 w.p. 1-q      1 -> 2 w.p. pq
        2 -> 1 w.p. 1

    A 2 heralds the first tail after heads, so it is always preceded by a 1.
    A 0 is preceded by another tail, written 2 if the coin was heads before
    that -- w.p. p, the two-state chain being reversible -- and 0 otherwise.
    A 1 is preceded by heads w.p. 1-q, else by a tail that splits the same
    way.  Stationary under [q-pq, p, pq]/(p+q), the C- occupancy.
    """
    return np.array([[1.0 - p,       0.0,     p    ],
                     [q * (1.0 - p), 1.0 - q, p * q],
                     [0.0,           1.0,     0.0  ]], dtype=float)


def flower_transition_matrix(n: int) -> np.ndarray:
    """
    (n+1, n+1) forward state transition matrix, state order
    [R, die 0, ..., die n-1] matching causal_state_occupancy("flower", "forward").

    R is "a roll just happened": the next token is a uniform die selection, so
    R -> die i with probability 1/n.  A die state is always followed by its roll,
    which returns to R, so die i -> R with probability 1.

    Deterministic in one direction and uniform in the other, which is exactly why
    C+ = 1 + log2(n)/2 rather than log2(n+1).
    """
    n = int(n)
    T = np.zeros((n + 1, n + 1), dtype=float)
    T[0, 1:] = 1.0 / n
    T[1:, 0] = 1.0
    return T


def flower_rev_transition_matrix(n: int, m: int, dice_probs,
                                 merge_tol: float | None = None) -> np.ndarray:
    """
    (k+1, k+1) backward state transition matrix, k the number of
    distinguishable outcomes, state order [S, class 0, ..., class k-1] matching
    causal_state_occupancy("flower", "backward").  Closed form: every entry
    comes from the dice and the uniform 1/n die choice, nothing is sampled.

    S is "the current token is a selection".  Read backwards the token before
    it is the PREVIOUS cycle's outcome, independent of everything seen, so
    S -> class c with the outcome marginal (_face_marginal) summed over the
    faces in the class.  An outcome is always preceded by the selection that
    produced it, so class c -> S w.p. 1.

    Where Bayes enters: faces are grouped by their posterior over dice
    (_bayes_rev_prob), since that posterior is all a backward observer can
    retrodict from an outcome, so equal posteriors are one causal state.  The
    posterior itself is the reverse machine's EMISSION -- which die follows the
    outcome -- not a transition: every selection token is the one state S, so
    the die identity is a symbol.  The grouping is _merged_outcome_mass, so
    this matrix, C-, the state count and the occupancy merge under one rule.

    The mirror of flower_transition_matrix -- one hub, deterministic return --
    with the hub branching over merged outcomes instead of uniformly over
    dice, which is the whole of the asymmetry: C- = 1 + H(mass)/2 against
    C+ = 1 + log2(n)/2.
    """
    dp = np.asarray(dice_probs, dtype=float)
    if dp.shape != (n, m):
        raise ValueError(f"dice_probs must have shape ({n}, {m}), got {dp.shape}")

    # outcome marginal per distinguishable posterior, from the dice alone
    face_dist = np.asarray(_merged_outcome_mass(dp, merge_tol), dtype=float)
    face_dist = face_dist / face_dist.sum()
    rev_dim   = len(face_dist) + 1

    T = np.zeros((rev_dim, rev_dim), dtype=float)
    T[0, 1:] = face_dist        # S -> outcome class: the previous cycle's outcome
    T[1:, 0] = 1.0              # outcome class -> S: its selection always precedes it
    return T


# ── theoretical epsilon-machines WITH emissions ───────────────────────────
def true_machine(kind: str, params: dict, mode: str = "forward",
                 merge_tol: float | None = None) -> dict:
    """
    The epsilon-machine of the process in this direction, as a unifilar HMM:

        next_state      (K, V) int    state after emitting token x from state s
        emission_probs  (K, V) float  P(x | s)

    the format an extracted symbolic machine is compared against.  Forward the
    emission is P(next token | state); backward it is P(previous token | state)
    and next_state is the state one step back in time -- the convention of the
    *_rev_transition_matrix functions.  The state order matches
    causal_state_occupancy for the same (kind, mode), and the induced
    state-to-state matrix is the matching *_transition_matrix; checks.py holds
    all three against each other and against the samplers.

    Every machine here is synchronised by its LAST token -- the state after
    emitting x is a function of x alone -- so next_state has identical rows and
    the state after any history is read off its final token.  Zero-probability
    entries of next_state carry that same synchronising state; they are never
    reached under exact generation.

        coin forward     [tails, heads]     tails --0|1-p--> tails,  tails --1|p--> heads
                                            heads --1|1-q--> heads,  heads --2|q--> tails
        coin backward    [0, 1, 2]          state = current token; rows of
                                            coin_rev_transition_matrix
        flower forward   [R, die 0..n-1]    R --i|1/n--> die i,  die i --n+j|p^i_j--> R
        flower backward  [S, class 0..k-1]  S --n+j|pi_j--> class(j),
                                            class c --i|P(die i | face in c)--> S
    """
    if mode not in ("forward", "backward"):
        raise ValueError(f"mode must be 'forward' or 'backward', got {mode!r}")

    if kind == "coin":
        p, q = float(params["p"]), float(params["q"])
        if mode == "forward":
            emission = np.array([[1.0 - p, p,       0.0],
                                 [0.0,     1.0 - q, q  ]], dtype=float)
            sync = np.array([0, 1, 0], dtype=np.int64)
        else:
            emission = coin_rev_transition_matrix(p, q)
            sync = np.array([0, 1, 2], dtype=np.int64)

    elif kind == "flower":
        n, m = int(params["n"]), int(params["m"])
        dp = np.asarray(params["dice_probs"], dtype=float)
        if dp.shape != (n, m):
            raise ValueError(f"dice_probs must have shape ({n}, {m}), got {dp.shape}")
        if mode == "forward":
            emission = np.zeros((n + 1, n + m), dtype=float)
            emission[0, :n]  = 1.0 / n                 # R: uniform die selection
            emission[1:, n:] = dp                      # die i: its roll
            sync = np.concatenate([np.arange(1, n + 1), np.zeros(m, dtype=np.int64)])
        else:
            cls, mass = _outcome_classes(dp, merge_tol)
            posterior = _bayes_rev_prob(n, dp)
            emission = np.zeros((len(mass) + 1, n + m), dtype=float)
            emission[0, n:] = _face_marginal(n, dp)    # S: the previous cycle's outcome
            for j in range(m):
                if cls[j] >= 0:                        # class c: which die produced it
                    emission[1 + cls[j], :n] = posterior[:, j]
            # a face that never occurs has no class; it is sent to S and never reached
            sync = np.concatenate([np.zeros(n, dtype=np.int64),
                                   np.where(cls >= 0, 1 + cls, 0)])
    else:
        raise ValueError(f"unknown process {kind!r}; expected 'coin' or 'flower'")

    return {"next_state": np.tile(sync, (emission.shape[0], 1)),
            "emission_probs": emission}


# ══════════════════════════════════════════════════════════════════════════
# TAGS
# ══════════════════════════════════════════════════════════════════════════
def coin_tag(p: float, q: float) -> str:
    """coin_tag(0.4, 0.8) -> 'coin_p040_q080'.  Derived, never a literal, so a
    tag cannot disagree with the run that produced it."""
    return f"coin_p{round(p*100):03d}_q{round(q*100):03d}"


def flower_tag(n: int, m: int) -> str:
    return f"flower_n{n}_m{m}"
