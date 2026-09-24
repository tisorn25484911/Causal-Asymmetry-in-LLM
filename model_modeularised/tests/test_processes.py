"""
Regression tests for HMM_processes.py and process_generator.py.

Run:  pytest tests/ -q          (from model_modeularised/)

Three kinds of test live here.

1. LITERATURE.  Closed forms and published values from Ellison, Mahoney &
   Crutchfield, "Prediction, Retrodiction, and the Amount of Information Stored
   in the Present", J. Stat. Phys. 136, 1005 (2009) / arXiv:0905.3587.  These
   are the authoritative oracle: they pin C_mu+, C_mu-, E, h_mu and the
   crypticity independently of anything in this repository.

2. IDENTITIES.  Facts that must hold for every process -- E recovered from block
   entropies without touching the reverse machine, the reversal round trip, the
   sampler against exact word probabilities, and an independent reference
   implementation in reference.py.

3. BUGS.  Cases that were broken and are now fixed; these fail on the code as it
   stood before 2026-09-24.  See the BUG-n markers.
"""
import numpy as np
import pytest

import HMM_processes as P
from process_generator import SequenceDataset, generator
from reference import ref_entropy, ref_sample, ref_stationary, ref_words


def H(r):
    """binary entropy in bits, with the boundary cases."""
    return 0.0 if r in (0.0, 1.0) else float(-(r * np.log2(r) + (1 - r) * np.log2(1 - r)))


# every registered process at a representative, non-degenerate parameter point
FINITE = {
    "coin":      {"p": 0.3, "q": 0.6},
    "pcoin":     {"p": 0.3, "q": 0.6},
    "iid":       {"probs": [0.2, 0.3, 0.5]},
    "flower":    {"n": 3, "m": 2},
    "rip":       {"p": 0.4, "q": 0.3},
    "rnc":       {"p": 0.3, "q": 0.4},
    "butterfly": {},
    "explosive": {},
    "t3":        {},
    "even":      {"p": 0.5},
    "gm":        {"p": 0.5},
    "nemo":      {"p": 0.5, "q": 0.5},
    "rgm":       {"k": 3},
    "sgm":       {"k": 3, "p": 0.5},
    "ising":     {},
    "edges":     {"n_states": 2, "vocab": 2,
                  "edges": [(0, 0, 1, 0.5), (0, 1, 0, 0.5), (1, 1, 0, 1.0)]},
    "tilt":      {"base": "gm", "base_params": {"p": 0.5}, "beta": 2.0},
}
ALL = dict(FINITE, sns={"p": 0.5, "q": 0.5})


def raw(name, par):
    """the generating presentation straight from the registry, as (T, joint map)."""
    T = P.PROCESS[name]["transition"](par)
    E = P.PROCESS[name]["state_map"](par)
    return T, T[:, :, None] * E


def bidirectional(g):
    """(C+, C-, E, h, chi+, C+-, Xi) -- C+- = H[S+, S-] is computed in __init__ but not exposed."""
    Cp, chi_p, E, h = g.statistical_c, g.crypticity, g.excess_entropy, g.entropy_rate
    g.reverse()
    Cm, pi_b, nu = g.statistical_c, g.station_dist, g.switching_map
    g.reverse()
    return dict(Cp=Cp, Cm=Cm, E=E, h=h, chi_p=chi_p,
                Cpm=P.entropy_bits(pi_b[:, None] * nu), Xi=Cm - Cp)


# ══════════════════════════════════════════════════════════════════════════
# 1. LITERATURE
# ══════════════════════════════════════════════════════════════════════════
def test_rip_published_values():
    """Sec. XI.3, p = q = 1/2.  Paper prints 4 decimals, so compare at 1e-4."""
    r = bidirectional(generator("rip", 1, 1, {"p": 0.5, "q": 0.5}))
    published = dict(Cp=1.5219, Cm=1.8464, Cpm=2.1219, E=1.2464, h=0.6, Xi=0.3245)
    for key, want in published.items():
        assert r[key] == pytest.approx(want, abs=1e-4), f"{key}: {r[key]} vs paper {want}"
    assert r["Cpm"] - r["E"] == pytest.approx(0.8755, abs=1e-4)     # chi_+- in the paper


@pytest.mark.parametrize("p", [0.1, 0.25, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("q", [0.15, 0.4, 0.85])
def test_rnc_closed_forms(p, q):
    """Sec. X:  C_mu+ = 1 + H(p)/2   and   chi+ = [p+q(1-p)]/2 * H(p/[p+q(1-p)])."""
    r = bidirectional(generator("rnc", 1, 1, {"p": p, "q": q}))
    d = p + q * (1 - p)
    assert r["Cp"] == pytest.approx(1 + H(p) / 2, abs=1e-12)
    assert r["chi_p"] == pytest.approx(d / 2 * H(p / d), abs=1e-12)


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_even_closed_forms(p):
    """Sec. XI.1:  C_mu = H(1/(2-p)),  chi = 0 for every p,  E = C_mu,  h_mu = H(p)/(2-p)."""
    r = bidirectional(generator("even", 1, 1, {"p": p}))
    assert r["Cp"] == pytest.approx(H(1 / (2 - p)), abs=1e-12)
    assert r["chi_p"] == pytest.approx(0.0, abs=1e-12)
    assert r["E"] == pytest.approx(H(1 / (2 - p)), abs=1e-12)
    assert r["h"] == pytest.approx(H(p) / (2 - p), abs=1e-12)


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_gm_closed_forms(p):
    """Sec. XI.2:  E = H(1/(2-p)) - H(p)/(2-p)."""
    r = bidirectional(generator("gm", 1, 1, {"p": p}))
    assert r["E"] == pytest.approx(H(1 / (2 - p)) - H(p) / (2 - p), abs=1e-12)


def test_gm_bidirectional_complexity_at_one_half():
    """Sec. XI.2 at p = 1/2:  C_mu+- = log2 3, and chi_+- = C_+- - E = 4/3."""
    r = bidirectional(generator("gm", 1, 1, {"p": 0.5}))
    assert r["Cpm"] == pytest.approx(np.log2(3), abs=1e-12)
    assert r["Cpm"] - r["E"] == pytest.approx(4 / 3, abs=1e-12)


@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49])
def test_rnc_zero_locus(p):
    """rnc_zero_locus_q: causal irreversibility vanishes on q*(p) = (1-2p)/(1-p)."""
    r = bidirectional(generator("rnc", 1, 1, {"p": p, "q": P.rnc_zero_locus_q(p)}))
    assert r["Cp"] == pytest.approx(r["Cm"], abs=1e-9)


@pytest.mark.parametrize("p,q", [(0.3, 0.2), (0.3, 0.6), (0.1, 0.5)])
def test_rnc_off_locus_is_irreversible(p, q):
    """off that curve the two directions must actually differ, or the test above is vacuous."""
    r = bidirectional(generator("rnc", 1, 1, {"p": p, "q": q}))
    assert abs(r["Cp"] - r["Cm"]) > 1e-3


# ══════════════════════════════════════════════════════════════════════════
# 2. IDENTITIES
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("name", sorted(ALL))
def test_excess_entropy_from_block_entropies(name):
    """
    E = sum_{L>=1} (h(L) - h_mu).  This route goes through word statistics only,
    so it never touches reverse_machine, mixed_states or merge_mixed -- it is the
    independent check on Theorem 1 (E = I[S+; S-]).
    """
    g = generator(name, 1, 1, ALL[name])
    f = g._views["forward"]
    h = P.conditional_entropies(f["gen_transition"], f["gen_state_map"], 200)
    assert h[-1] == pytest.approx(g.entropy_rate, abs=1e-9), "h(L) has not converged to h_mu"
    assert float((h - g.entropy_rate).sum()) == pytest.approx(g.excess_entropy, abs=1e-9)


@pytest.mark.parametrize("name", sorted(FINITE))
def test_bidirectional_identity(name):
    """H[S+, S-] = H[S+] + H[S-] - I[S+; S-], i.e. C_+- = C+ + C- - E."""
    r = bidirectional(generator(name, 1, 1, FINITE[name]))
    assert r["Cpm"] == pytest.approx(r["Cp"] + r["Cm"] - r["E"], abs=1e-9)


@pytest.mark.parametrize("name", sorted(ALL))
def test_reversal_round_trip(name):
    """diagram (26) read backwards: U(T(M-)) is M+ again.  Asserts inside for finite machines."""
    rep = generator(name, 1, 1, ALL[name]).check_involution()
    assert rep["presentation_twice"] < 1e-12
    if rep["finite"]:
        assert rep["isomorphic"] and rep["n_causal_again"] == rep["n_causal"]
        assert max(rep["word_prob_gap"], rep["h_gap"], rep["E_gap"]) < 1e-7


@pytest.mark.parametrize("name,par,L", [
    ("gm", {"p": 0.5}, 10), ("even", {"p": 0.5}, 10), ("coin", {"p": 0.3, "q": 0.6}, 6),
    ("rip", {"p": 0.5, "q": 0.5}, 10), ("rnc", {"p": 0.3, "q": 0.4}, 10),
    ("nemo", {"p": 0.5, "q": 0.5}, 10), ("flower", {"n": 3, "m": 2}, 4),
    ("butterfly", {}, 4), ("sns", {"p": 0.5, "q": 0.5}, 10),
])
def test_word_probs_against_independent_enumeration(name, par, L):
    """word_probs uses a transfer-matrix recursion; reference.ref_words enumerates paths."""
    T, M = raw(name, par)
    keep = P.closed_class(T)
    Tk = T[np.ix_(keep, keep)]
    Ek = P.PROCESS[name]["state_map"](par)[np.ix_(keep, keep)]
    ref = ref_words(M, ref_stationary(T), L)
    assert ref.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.abs(ref - P.word_probs(Tk, Ek, L)).max() < 1e-12


@pytest.mark.parametrize("name", sorted(ALL))
def test_stationary_against_lazy_power_iteration(name):
    """stationary_dist eigensolves; the reference iterates.  Covers the periodic chains."""
    T, _ = raw(name, ALL[name])
    pi = P.stationary_dist(T)
    assert np.abs(pi @ T - pi).max() < 1e-10
    assert pi.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.abs(pi - ref_stationary(T)).max() < 1e-8


@pytest.mark.parametrize("name,par,L", [
    ("coin", {"p": 0.3, "q": 0.6}, 7), ("flower", {"n": 3, "m": 2}, 5),
    ("rip", {"p": 0.4, "q": 0.3}, 12), ("even", {"p": 0.5}, 12), ("sns", {"p": 0.5, "q": 0.5}, 12),
])
def test_sampler_matches_word_probs(name, par, L):
    """tokens_from_map against the exact word distribution, 400k tokens."""
    g = generator(name, 1, 1, par)
    f = g._views["forward"]
    T, E, V = f["gen_transition"], f["gen_state_map"], g.vocab_size
    seq = P.tokens_from_map(400_000, T, E, P.stationary_dist(T), np.random.default_rng(0))
    idx = sum(seq[i:len(seq) - L + 1 + i] * V ** (L - 1 - i) for i in range(L))
    emp = np.bincount(idx, minlength=V ** L) / idx.size
    assert np.abs(emp - P.word_probs(T, E, L)).max() < 0.01      # ~6 sigma at 4e5 draws


def test_filter_probs_matches_brute_force():
    """the forward algorithm against conditionals read straight off word_probs."""
    g = generator("rip", 1, 1, {"p": 0.4, "q": 0.3})
    f = g._views["forward"]
    T, E = f["gen_transition"], f["gen_state_map"]
    seqs = np.array([[0, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0]])
    pr = P.filter_probs(P.joint_map(T, E), P.stationary_dist(T), seqs)
    assert np.allclose(pr.sum(axis=2), 1.0)
    W = P.word_probs(T, E, 6).reshape((2,) * 6)
    for n, s in enumerate(seqs):
        for t in range(6):
            prior = W[tuple(s[:t])].sum() if t else 1.0
            assert pr[n, t, s[t]] == pytest.approx(W[tuple(s[:t]) + (s[t],)].sum() / prior, abs=1e-12)


@pytest.mark.parametrize("name", sorted(ALL))
def test_ideal_floor_is_monotone_and_above_hmu(name):
    """h(L) = H[X_L | X_0..X_{L-1}] is non-increasing in L and bounded below by h_mu."""
    g = generator(name, 1, 48, ALL[name])
    fl = g.ideal_floor()
    assert len(fl) == g.seq_len
    assert np.all(np.diff(fl) <= 1e-12)
    assert np.all(fl >= g.entropy_rate - 1e-12)


@pytest.mark.parametrize("name", sorted(FINITE))
def test_ideal_ce_tracks_ideal_floor(name):
    """the empirical floor on drawn data must sit on the exact one."""
    g = generator(name, 3000, 32, FINITE[name])
    g.generate(np.random.default_rng(1))
    assert np.abs(g.ideal_ce() - g.ideal_floor()).mean() < 0.05


@pytest.mark.parametrize("name", sorted(ALL))
def test_reverse_is_an_involution(name):
    """twice is the identity, on the machine and on the stored sequences."""
    g = generator(name, 16, 32, ALL[name])
    before, summary = g.generate(np.random.default_rng(2)).copy(), g.summary()
    g.reverse()
    assert np.array_equal(g.all_sequence, before[:, ::-1])
    assert g.direct == "backward"
    g.reverse()
    assert np.array_equal(g.all_sequence, before)
    assert g.summary() == summary


@pytest.mark.parametrize("name", sorted(ALL))
def test_backward_presentation_generates_the_reversed_process(name):
    """gen_* in the backward view must reproduce the forward word probabilities, reversed."""
    g = generator(name, 1, 1, ALL[name])
    f, b, V = g._views["forward"], g._views["backward"], g.vocab_size
    L = max(1, int(np.log(4096) / np.log(max(V, 2))))
    pf = P.word_probs(f["gen_transition"], f["gen_state_map"], L)
    pb = P.word_probs(b["gen_transition"], b["gen_state_map"], L)
    pb_rev = pb.reshape((V,) * L).transpose(tuple(range(L))[::-1]).ravel()
    assert np.abs(pf - pb_rev).max() < 1e-12


@pytest.mark.parametrize("name", sorted(ALL))
def test_beta_tilt_identity_at_one(name):
    """beta = 1 must leave the measure alone."""
    T, _ = raw(name, ALL[name])
    E = P.PROCESS[name]["state_map"](ALL[name])
    T1, E1 = P.beta_tilt(T, E, 1.0)
    assert np.abs(P.joint_map(T1, E1) - P.joint_map(T, E)).max() < 1e-12


def test_minimise_merges_duplicated_states():
    """split every coin state in two; the epsilon-machine must put them back together."""
    par = {"p": 0.3, "q": 0.6}
    T, E = P.coin_transition(par), P.coin_state_map(par)
    M = P.joint_map(T, E)
    S, _, V = M.shape
    D = np.zeros((2 * S, 2 * S, V))
    for s in range(S):
        for t in range(S):
            for c in (0, 1):
                D[2 * s + c, 2 * t] = D[2 * s + c, 2 * t + 1] = M[s, t] / 2
    em = P.epsilon_machine(*P.split_map(D))
    assert em["n_states"] == 2
    assert P.isomorphic(em["T"], em["E"], T, E) is not None


def test_isomorphic_rejects_different_machines():
    """otherwise the positive case above proves nothing."""
    par = {"p": 0.3, "q": 0.6}
    assert P.isomorphic(P.coin_transition(par), P.coin_state_map(par),
                        P.gm_transition({"p": 0.5}), P.gm_state_map({"p": 0.5})) is None


@pytest.mark.parametrize("name", sorted(ALL))
def test_split_joint_round_trip(name):
    T, _ = raw(name, ALL[name])
    E = P.PROCESS[name]["state_map"](ALL[name])
    T2, E2 = P.split_map(P.joint_map(T, E))
    assert np.abs(T2 - T).max() < 1e-12
    assert np.abs(P.joint_map(T2, E2) - P.joint_map(T, E)).max() < 1e-12


def test_check_map_rejects_malformed_machines():
    par = {"p": 0.3, "q": 0.6}
    T, E = P.coin_transition(par), P.coin_state_map(par)
    with pytest.raises(AssertionError, match="rows of T"):
        P.check_map(T * 0.9, E, 3)
    with pytest.raises(AssertionError, match="E must sum to 1"):
        P.check_map(T, E * 0.5, 3)
    with pytest.raises(AssertionError, match="vocab_size"):
        P.check_map(T, E, 5)
    Tneg = T.copy()
    Tneg[0, 0], Tneg[0, 1] = 1.2, -0.2               # row still sums to 1
    with pytest.raises(AssertionError, match="negative"):
        P.check_map(Tneg, E, 3)
    Eneg = E.copy()
    Eneg[0, 0] = [1.5, -0.5, 0.0]                    # row still sums to 1
    with pytest.raises(AssertionError, match="negative"):
        P.check_map(T, Eneg, 3)


@pytest.mark.parametrize("k", [200, 400, 800])
def test_stationary_dist_sparse_path(k):
    """above 300 states stationary_dist switches to a sparse solve."""
    T = P.rgm_transition({"k": k})
    pi = P.stationary_dist(T)
    assert np.abs(pi @ T - pi).max() < 1e-10
    assert pi.sum() == pytest.approx(1.0, abs=1e-12)
    assert (pi > 0).all()


def test_sequence_dataset_and_reproducibility():
    g = generator("coin", 16, 32, {"p": 0.3, "q": 0.6})
    a = g.generate(np.random.default_rng(7)).copy()
    b = generator("coin", 16, 32, {"p": 0.3, "q": 0.6}).generate(np.random.default_rng(7))
    assert np.array_equal(a, b)
    ds = SequenceDataset(a)
    xi, yi = ds[0]
    assert ds.seq_len == g.seq_len == 32
    assert np.array_equal(xi[1:].numpy(), yi[:-1].numpy())
    assert np.array_equal(g.input[0], xi.numpy()) and np.array_equal(g.target[0], yi.numpy())


def test_infinite_backward_machines_are_flagged():
    """explosive and t3 are finite forwards and countably infinite backwards."""
    for name in ("explosive", "t3"):
        g = generator(name, 1, 1, {})
        assert g._views["forward"]["infinite"] is False
        assert g._views["backward"]["infinite"] is True, name
    assert generator("sns", 1, 1, {"p": 0.5, "q": 0.5})._views["forward"]["infinite"] is True


def test_nonergodic_parameters_raise_a_clear_error():
    """coin at p = q = 0 has two absorbing states; there is no unique stationary law."""
    with pytest.raises(AssertionError, match="recurrent class"):
        generator("coin", 1, 1, {"p": 0.0, "q": 0.0})


# ══════════════════════════════════════════════════════════════════════════
# 3. BUGS
# ══════════════════════════════════════════════════════════════════════════
# BUG-1.  p = 0 and p = 1 raised "E must be 0 wherever T == 0" from check_map.
# The state maps write E on every structurally present edge, so a parameter that
# kills an edge leaves stale mass behind; the joint map T * E is unaffected, so
# the machine was well formed and was being rejected anyway.  Any parameter sweep
# that includes its endpoints hit this.
#
# n_causal, C+ and h_mu below are derived by hand from the degenerate machine and
# confirmed against reference.ref_sample in the test that follows.
BOUNDARY = [
    ("gm",   {"p": 0.0},             2, 1.0,        0.0,   1.0),
    ("gm",   {"p": 1.0},             1, 0.0,        0.0,   0.0),
    ("even", {"p": 0.0},             1, 0.0,        0.0,   0.0),
    ("even", {"p": 1.0},             1, 0.0,        0.0,   0.0),
    ("rnc",  {"p": 0.0, "q": 0.5},   2, 1.0,        0.5,   1.0),
    ("rnc",  {"p": 1.0, "q": 0.5},   1, 0.0,        0.0,   0.0),
    ("coin", {"p": 0.0, "q": 0.5},   1, 0.0,        0.0,   0.0),
    ("coin", {"p": 1.0, "q": 0.5},   2, H(1 / 3),   2 / 3, H(1 / 3) - 2 / 3),
]


@pytest.mark.parametrize("name,par,n_causal,Cp,hmu,E", BOUNDARY)
def test_boundary_parameters_build(name, par, n_causal, Cp, hmu, E):
    """BUG-1: these raised AssertionError instead of returning the degenerate machine."""
    g = generator(name, 4, 16, par)
    assert g.n_causal == n_causal
    assert g.statistical_c == pytest.approx(Cp, abs=1e-12)
    assert g.entropy_rate == pytest.approx(hmu, abs=1e-12)
    assert g.excess_entropy == pytest.approx(E, abs=1e-9)
    g.generate(np.random.default_rng(0))
    assert g.all_sequence.shape == (4, 17)


@pytest.mark.parametrize("name,par,n_causal,Cp,hmu,E", BOUNDARY)
def test_boundary_parameters_match_block_entropies(name, par, n_causal, Cp, hmu, E):
    """the same degenerate machines must still satisfy E = sum_L (h(L) - h_mu)."""
    g = generator(name, 1, 1, par)
    f = g._views["forward"]
    h = P.conditional_entropies(f["gen_transition"], f["gen_state_map"], 200)
    assert float((h - g.entropy_rate).sum()) == pytest.approx(g.excess_entropy, abs=1e-9)


@pytest.mark.parametrize("name,par,expected", [
    ("gm",   {"p": 0.0},           "alternating"),
    ("gm",   {"p": 1.0},           "constant"),
    ("even", {"p": 0.0},           "constant"),
    ("even", {"p": 1.0},           "constant"),
    ("rnc",  {"p": 1.0, "q": 0.5}, "constant"),
    ("coin", {"p": 0.0, "q": 0.5}, "constant"),
])
def test_boundary_sequences_have_the_derived_structure(name, par, expected):
    """
    BUG-1, independently: the degenerate processes really are what the expected
    C+ and h_mu above assume, checked with the reference sampler, not this repo's.
    """
    T, M = raw(name, par)
    s = ref_sample(M, ref_stationary(T), 4000, np.random.default_rng(0))
    if expected == "constant":
        assert len(set(s.tolist())) == 1
    else:
        assert bool(np.all(s[1:] != s[:-1])) and len(set(s.tolist())) == 2


@pytest.mark.parametrize("name,par,formula", [
    ("even", {"p": 0.0},           1.0),     # H(1/(2-p)) = H(1/2) = 1, true C+ = 0
    ("rnc",  {"p": 1.0, "q": 0.5}, 1.0),     # 1 + H(p)/2  = 1,        true C+ = 0
])
def test_closed_forms_are_discontinuous_at_the_boundary(name, par, formula):
    """
    Not a bug: at these endpoints the presentation stops being minimal (both states
    emit the same thing forever), so C_mu drops to 0 while the published closed form,
    which assumes minimality, keeps its interior value.  The code reports the true
    minimal machine.  This test exists so that the disagreement stays deliberate.
    """
    g = generator(name, 1, 1, par)
    assert g.n_causal == 1
    assert g.statistical_c == pytest.approx(0.0, abs=1e-12)
    assert abs(formula - g.statistical_c) > 0.9


# BUG-2.  flower_state_map's docstring claimed identical dice "inflate C_mu above
# its true value".  epsilon_machine now minimises, so the petals merge and C_mu is
# correct; the docstring described a failure mode the code handles.
def test_flower_identical_dice_merge():
    dice = np.array([[0.8, 0.2], [0.8, 0.2], [0.1, 0.9]])     # dice 0 and 1 identical
    g = generator("flower", 1, 1, {"n": 3, "m": 2, "dice_probs": dice})
    assert g.n_causal == 3                                     # hub + merged petal + petal
    assert g.statistical_c == pytest.approx(P.entropy_bits([0.5, 1 / 3, 1 / 6]), abs=1e-12)
    raw_pi = P.stationary_dist(P.flower_transition({"n": 3}))
    assert P.statistical_complexity(raw_pi) > g.statistical_c  # the un-minimised value
    distinct = generator("flower", 1, 1, {"n": 3, "m": 2,
                                          "dice_probs": np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])})
    assert distinct.n_causal == 4


# BUG-4.  entropy_rate returned -0.0 for a deterministic process (it lacked the
# "+ 0.0" that statistical_complexity and entropy_bits both carry), so tables of
# results printed "-0.000000".
@pytest.mark.parametrize("name,par", [
    ("iid", {"probs": [1.0, 0.0]}), ("gm", {"p": 1.0}), ("even", {"p": 0.0}),
    ("gm", {"p": 0.0}), ("rnc", {"p": 1.0, "q": 0.5}),
])
def test_no_negative_zero(name, par):
    g = generator(name, 1, 1, par)
    for field in ("entropy_rate", "statistical_c", "excess_entropy", "crypticity"):
        v = getattr(g, field)
        assert not (v == 0.0 and np.signbit(v)), f"{name} {field} is -0.0"
