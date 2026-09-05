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

    One class for both processes.  The old tree had CoinDataset and
    FlowerDataset, which differed only in an unused `num_token` attribute.
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

    This replaces generate-2000-then-keep-a-256-window.  Nothing is discarded
    beyond the burn-in, training and analysis run at the same length by
    construction, and `max_len = seq_len` -- so the model is never asked for a
    positional index it did not train on.

    FLOWER PHASE.  A flower sequence alternates select/roll, so token type is
    fixed by index parity.  With an even `burn_in` every kept sequence starts on
    a selection token -- exactly what the old whole-cycle `pre_depth` discard
    did.  Preserved rather than randomised, because randomising it would change
    the process the archived results were measured on.
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


def _merged_outcome_mass(dp, merge_tol: float | None = None) -> list:
    """
    Mass of each *distinguishable* backward outcome state.

    Two outcomes j and j' are the same backward state exactly when they induce
    the same posterior P(die=i | outcome=j) over dice -- i.e. when columns j and
    j' of dice_probs are proportional.

    Shared by flower_complexity, causal_state_count and causal_state_occupancy so
    all three count states under one rule.  Two definitions of "the same backward
    state" in one repository would drift, and a K that disagreed with its own
    theory is worse than no K.
    """
    dp = np.asarray(dp, dtype=float)
    pi_outcome = dp.mean(axis=0)
    col_mass   = dp.sum(axis=0)
    m = dp.shape[1]

    if merge_tol is None:
        merged: dict[tuple, float] = {}
        for j in range(m):
            if col_mass[j] <= 0:                       # outcome never occurs
                continue
            key = tuple(np.round(dp[:, j] / col_mass[j], MERGE_ROUND_DP))
            merged[key] = merged.get(key, 0.0) + pi_outcome[j]
        return list(merged.values())

    reps, mass = [], []
    for j in range(m):
        if col_mass[j] <= 0:
            continue
        posterior = dp[:, j] / col_mass[j]
        for k, r in enumerate(reps):
            if np.max(np.abs(posterior - r)) <= merge_tol:
                mass[k] += float(pi_outcome[j])
                break
        else:
            reps.append(posterior)
            mass.append(float(pi_outcome[j]))
    return mass


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
# T[i][j] = P(s_{t+1} = j | s_t = i).  Forward arms only.  The backward
# epsilon-machine transition structure -- coin's 3 states, flower's merged
# outcomes -- is real derivation work with room for error, and a wrong theory
# panel is worse than none, so F3's backward panels carry no overlay.
#
# Nothing in the old tree computed these.  `coin_true_conditional` there is
# P(next TOKEN | current token), which is a different object.

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


# ══════════════════════════════════════════════════════════════════════════
# TAGS
# ══════════════════════════════════════════════════════════════════════════
def coin_tag(p: float, q: float) -> str:
    """coin_tag(0.4, 0.8) -> 'coin_p040_q080'.  Derived, never a literal, so a
    tag cannot disagree with the run that produced it."""
    return f"coin_p{round(p*100):03d}_q{round(q*100):03d}"


def flower_tag(n: int, m: int) -> str:
    return f"flower_n{n}_m{m}"
