"""
An independent reference implementation, deliberately sharing no code with
HMM_processes: a lazy power iteration instead of an eigensolve, explicit path
enumeration instead of the transfer-matrix recursion, and a hand-rolled sampler.

Two traps this had to avoid, both found the hard way:
  * a plain power iteration v <- vT does NOT converge on a periodic chain (the
    flower, and rnc at p = 0, both have period 2); the lazy chain does.
  * clustering finite-length histories by their finite-length future morphs
    over-refines a sofic process -- the never-seen-a-0 histories of the Even
    process form a spurious extra class whose weight only vanishes as the
    history grows -- so it is NOT a usable oracle for C_mu.  The literature
    closed forms in test_processes.py play that role instead.
"""
import numpy as np


def ref_stationary(T, iters=100000, tol=1e-16):
    """pi T = pi by lazy iteration v <- (v + vT)/2: same fixed point, periodic-safe."""
    T = np.asarray(T, dtype=float)
    v = np.ones(T.shape[0]) / T.shape[0]
    for _ in range(iters):
        w = 0.5 * (v + v @ T)
        w = w / w.sum()
        if np.max(np.abs(w - v)) < tol:
            return w
        v = w
    return v


def ref_words(M, pi, L):
    """P(w) for every word of length L by explicit enumeration of state paths."""
    S, _, V = M.shape
    out = np.zeros(V ** L)

    def walk(s, depth, idx, pr):
        if depth == L:
            out[idx] += pr
            return
        for x in range(V):
            for t in range(S):
                if M[s, t, x] > 0:
                    walk(t, depth + 1, idx * V + x, pr * M[s, t, x])

    for s in range(S):
        if pi[s] > 0:
            walk(s, 0, 0, pi[s])
    return out


def ref_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ref_sample(M, pi, n, rng):
    """draw n tokens from the joint map, independently of tokens_from_map."""
    S, _, V = M.shape
    c = M.reshape(S, S * V).cumsum(axis=1)
    c = c / c[:, -1:]
    s = int(np.searchsorted(pi.cumsum(), rng.random()))
    out = np.empty(n, dtype=np.int64)
    for t in range(n):
        s, out[t] = divmod(int(np.searchsorted(c[s], rng.random())), V)
    return out
