"""
model_de_bruijn.py against DEBRUIJN_HYBRID_IMPLEMENTATION_PLAN.md ("the plan"):
    * the counting merge gives exactly the plan's result (its dense code, copied below as the reference);
    * the order rules: the Markov order, and the plan's information criterion with the estimates it printed;
    * the plan's contract tests (its section 10), on this folder's model;
    * model_de_bruijn_bic.py is model_de_bruijn.py with only the order rule changed.

Run from test_models/:  python -m pytest tests/test_de_bruijn.py -q
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import model_de_bruijn as D                                                # noqa: E402
import process as P                                                        # noqa: E402
import training as TR                                                      # noqa: E402

F5 = [0.1, 0.2, 0.3, 0.25, 0.15]
F9 = [.04, .08, .12, .16, .20, .16, .12, .08, .04]
F15 = [1.0 / 15] * 15


# ------------------------------------------------------------------------------------------------ the plan's code
def plan_successor_ids(cfg):
    n = cfg.n_nodes
    return (np.arange(n)[:, None] * cfg.vocab_size) % n + np.arange(cfg.vocab_size)[None, :]


def plan_suffix_representative(frequent, visits, cfg):
    V, R = cfg.vocab_size, cfg.order
    rep = np.full(cfg.n_nodes, -1)
    rep[frequent] = frequent
    order = frequent[np.argsort(-visits[frequent], kind="stable")]
    for keep in range(R - 1, -1, -1):
        m = V ** keep
        best = {}
        for u in order:
            best.setdefault(int(u % m), int(u))
        for u in np.flatnonzero(rep < 0):
            r = best.get(int(u % m))
            if r is not None:
                rep[u] = r
    return rep


def plan_merge_downwards(counts, cfg, alpha=1e-3, min_visits=100, max_rounds=None):
    """The plan's merge_downwards (section 4), unchanged except for the returned tuple."""
    V = cfg.vocab_size
    visits = counts.sum(axis=1)
    frequent = np.flatnonzero(visits >= min_visits)
    rep = plan_suffix_representative(frequent, visits, cfg)
    succ_node = rep[plan_successor_ids(cfg)]
    cls = np.full(cfg.n_nodes, -1)
    cls[frequent] = np.arange(len(frequent))
    sizes = [len(frequent)]

    def pooled_and_successors():
        k = cls.max() + 1
        pooled = np.zeros((k, V))
        np.add.at(pooled, cls[frequent], counts[frequent])
        succ = np.full((k, V), -1)
        for u in frequent:
            for x in np.flatnonzero(counts[u] > 0):
                succ[cls[u], x] = cls[succ_node[u, x]]
        return pooled, succ

    for _ in range(10 * cfg.order + 10 if max_rounds is None else max_rounds):
        pooled, succ = pooled_and_successors()
        k = len(pooled)
        order = np.argsort(-pooled.sum(axis=1), kind="stable")
        reps, new_label = [], np.empty(k, dtype=int)
        for c in order:
            target = None
            for r in reps:
                both = (succ[c] >= 0) & (succ[r] >= 0)
                if np.all(succ[c][both] == succ[r][both]) and D.compatible(
                        pooled[c], pooled[c].sum(), pooled[r], pooled[r].sum(), alpha):
                    target = r
                    break
            if target is None:
                reps.append(c)
                new_label[c] = len(reps) - 1
            else:
                new_label[c] = new_label[target]
        if len(reps) == k:
            break
        cls[frequent] = new_label[cls[frequent]]
        sizes.append(len(reps))
    pooled, succ = pooled_and_successors()
    return cls[rep], pooled / pooled.sum(axis=1, keepdims=True), succ, pooled.sum(axis=1), sizes


def process(name, params):
    return TR.get_process(name, __import__("json").dumps(params, sort_keys=True))


def exact_counts(proc, R, scale=10 ** 7):
    """Counts proportional to the true (R+1)-word probabilities (an infinite sample), as sparse (nodes, counts)."""
    V = proc.vocab_size
    words = P.word_probs(proc.T, proc.E, R + 1).reshape(V ** R, V)
    counts = np.rint(scale * words).astype(np.int64)
    nodes = np.flatnonzero(counts.sum(axis=1) > 0)
    return nodes, counts[nodes]


# ------------------------------------------------------------------------------------------------ the counting merge
CASES = [("coin", dict(p=0.7, q=0.8), 1), ("coin", dict(p=0.7, q=0.8), 2), ("rgm", dict(k=2), 2), ("ising", {}, 2),
         ("flower", dict(n=3, m=4, dice_seed=0), 1), ("butterfly", {}, 2), ("renewal", dict(F=F5), 4),
         ("renewal", dict(F=F9), 8), ("renewal", dict(F=F15), 14)]


@pytest.mark.parametrize("name,params,R", CASES)
def test_the_merge_gives_the_plans_result(name, params, R):
    """Sampled counts (20,000 windows of 60 tokens): the same states for every seen node, the same rows, successors,
    visits and classes per round as the plan's dense merge_downwards."""
    proc = process(name, params)
    V = proc.vocab_size
    windows = proc.sampler(2).draw(20_000, 60).numpy()
    nodes, counts = D.count_nodes(windows, R, V)
    q = D.merge_downwards(nodes, counts, R, V)
    dense = np.zeros((V ** R, V), dtype=np.int64)
    dense[nodes] = counts
    cfg = SimpleNamespace(vocab_size=V, order=R, n_nodes=V ** R)
    node_state, rows, succ, visits, sizes = plan_merge_downwards(dense, cfg)
    assert q.sizes == sizes
    assert np.array_equal(q.node_state, node_state[nodes])
    assert np.allclose(q.rows, rows) and np.array_equal(q.successor, succ) and np.allclose(q.visits, visits)


def test_exact_counts_reproduce_the_plans_merge_rounds():
    """The plan's section 10: renewal-5 merges 16 -> 9 -> 6 -> 5; the coin at order 2 merges 6 -> 3 -> 2."""
    proc = process("renewal", dict(F=F5))
    q = D.merge_downwards(*exact_counts(proc, 4), 4, 2)
    assert q.k == 5 and q.sizes == [16, 9, 6, 5]
    coin = process("coin", dict(p=0.7, q=0.8))
    q = D.merge_downwards(*exact_counts(coin, 2), 2, 3)
    assert q.k == 2 and q.sizes == [6, 3, 2]


def test_renewal15_counting_gives_its_15_states():
    """This model's own counting windows for renewal-15 (section 16's windows of 40 + 20 tokens): 1,840,000 positions,
    470 frequent nodes and the merge rounds the plan printed for Experiment C, ending at the 15 causal states."""
    R, q, info = D.starting_point(TR.settings_for("de_bruijn"), "markov")
    assert R == 14 and info["counted_positions"] == 1_840_000 and q.k == 15
    assert q.sizes == [470, 379, 301, 235, 180, 135, 99, 71, 50, 35, 25, 19, 16, 15]


def test_the_back_off_takes_the_longest_common_ending():
    """For any node: itself if frequent, else the most visited frequent node with the longest common ending."""
    frequent, visits = np.array([0b0011, 0b0101, 0b1101, 0b0000]), np.array([5, 9, 7, 1])
    order = np.argsort(frequent)
    backoff = D.SuffixBackoff(frequent[order], visits[order], 4, 2)
    assert backoff([0b0101])[0] == 0b0101           # frequent: itself
    assert backoff([0b1001])[0] == 0b0101           # no frequent node ends in 001; 0101 and 1101 end in 01, 0101 is
                                                    # visited more
    assert backoff([0b1011])[0] == 0b0011           # 0011 ends in 011
    assert backoff([0b1000])[0] == 0b0000           # 0000 ends in 000


# ------------------------------------------------------------------------------------------------ the order rules
@pytest.mark.parametrize("name,params,R", [("renewal", dict(F=F5), 4), ("renewal", dict(F=F15), 14),
                                           ("renewal", dict(F=[1 / 60] * 60), 59), ("coin", dict(p=0.7, q=0.8), 1),
                                           ("rgm", dict(k=2), 2), ("flower", dict(n=8, m=12, dice_seed=0), 1),
                                           ("flower", dict(n=8, m=12, dice_seed=0, direction="backward"), 1)])
def test_the_markov_order(name, params, R):
    assert D.markov_order(process(name, params), 200) == R


def test_the_information_criterion_gives_the_plans_estimates():
    """The plan's section 3, on the same tokens (its sampler with seed 1 is this folder's WindowSampler): coin 1,
    rgm 2, ising 2, renewal-9 8."""
    for name, params, expected in (("coin", dict(p=0.7, q=0.8), 1), ("rgm", dict(k=2), 2), ("ising", {}, 2),
                                   ("renewal", dict(F=F9), 8)):
        proc = process(name, params)
        tokens = proc.sampler(1).draw(1024, 1024).numpy()
        assert D.estimate_order(tokens, proc.vocab_size, 10 if proc.vocab_size == 2 else 6)[0] == expected, name


# ------------------------------------------------------------------------------------------------ the plan's contract tests
TINY = dict(history_len=20, rollout_len=10, context=9, batch_size=16, steps=4, eval_every=2, validation_rows=32)


@pytest.fixture(scope="module")
def tiny_model():
    """renewal-5 (order 4), a small window."""
    s = TR.settings_for("de_bruijn", params={"F": F5}, **TINY)
    return TR.make_model(s), s


def test_node_ids_and_the_scored_positions(tiny_model):
    model, s = tiny_model
    x = torch.randint(2, (16, s.total_len))
    ids = D.node_ids(x, 4, 2)
    assert (ids[:, :3] == -1).all()
    for t in range(3, s.total_len):
        assert torch.equal(ids[:, t], 8 * x[:, t - 3] + 4 * x[:, t - 2] + 2 * x[:, t - 1] + x[:, t])
    assert model.order == 4 and len(range(model.order - 1, s.total_len - 1)) == s.total_len - model.order


def test_the_reader_sees_the_window_only_through_the_node(tiny_model):
    """Condition C1: changing tokens more than R positions back changes no score."""
    model, s = tiny_model
    x = s.process_object().sampler(5).draw(16, s.total_len)
    y = x.clone()
    y[:, :2] = 1 - y[:, :2]
    with torch.no_grad():
        assert torch.equal(model.reader(x)[:, model.order + 1:], model.reader(y)[:, model.order + 1:])


def test_a_rollout_prediction_never_sees_its_own_or_later_tokens(tiny_model):
    model, s = tiny_model
    x = s.process_object().sampler(6).draw(64, s.total_len)
    y = x.clone()
    H = s.history_len
    y[:, H + 2:] = 1 - y[:, H + 2:]
    a = model.eval().predict(x, s.tau_end)["predictors"]["next_state_from_handoff"]["logits"]
    b = model.predict(y, s.tau_end)["predictors"]["next_state_from_handoff"]["logits"]
    torch.testing.assert_close(a[:, :3], b[:, :3])


def test_the_parallel_terms_equal_the_position_by_position_terms(tiny_model):
    model, s = tiny_model
    model.eval()
    x = s.process_object().sampler(7).draw(16, s.total_len)
    with torch.no_grad():
        _, terms, _ = model.loss(x, 1.0, 1.0)
        sel, _ = model.choose(model.reader(x), 1.0)
        manual = [F.cross_entropy(model.transition_logits(sel.onehot[:, t], x[:, t + 1]), sel.ids[:, t + 1],
                                  reduction="none") for t in range(model.order - 1, s.total_len - 1)]
    torch.testing.assert_close(terms["fit"], torch.stack(manual).mean() / math.log(2))


def test_each_term_reaches_the_parameters_it_trains():
    s = TR.settings_for("de_bruijn", params={"F": F5}, **TINY)
    model = TR.make_model(s).train()
    with torch.no_grad():
        model.reader.table.normal_()
        model.transition_table.normal_()
    x = s.process_object().sampler(8).draw(16, s.total_len)
    named = dict(model.named_parameters())
    for part, names in (("pred", ("reader.table", "causal_head.state_matrix", "causal_head.emission.weight")),
                        ("use", ("reader.table",)), ("fit", ("transition_table", "reader.table")),
                        ("commit", ("reader.table",))):
        model.zero_grad(set_to_none=True)
        model.loss(x, 1.5, 1.0)[1][part].backward()
        for name in names:
            grad = named[name].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, (part, name)


def test_the_warm_start_commutes(tiny_model):
    """At the quotient: U = 0, no defect, G agrees with the reader, and the rollout states are the reader's states."""
    model, s = tiny_model
    model.eval()
    x, _ = s.process_object().sampler(9).draw(512, s.total_len, states=True)
    pred = model.predict(x, s.tau_end)
    m = model.measures(x, s.tau_end, pred)
    assert m["U_bits"] < 1e-9 and m["defect"] == 0.0 and m["head_agreement"] == 1.0
    H, R = s.history_len, model.order
    reader = pred["predictors"]["node_table"]["states"][:, H - R + 1:]               # the reader's states from t = H
    assert torch.equal(pred["predictors"]["next_state_from_handoff"]["states"][:, 1:], reader[:, :-1])


def test_evaluation_draws_no_random_number_and_changes_no_parameter(tiny_model):
    model, s = tiny_model
    x, st = TR.validation_windows(s)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    TR.evaluate(model, s, x, st, s.tau_end)
    assert torch.equal(torch.get_rng_state(), rng)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_the_gru_transition_head_fits_the_successor_map(monkeypatch):
    monkeypatch.setattr(D, "TRANSITION", "gru")
    s = TR.settings_for("de_bruijn", params={"F": F5}, state_dim=16, **TINY)
    model = TR.make_model(s)
    k = model.quotient.k
    states = torch.arange(k).repeat_interleave(2)
    tokens = torch.arange(2).repeat(k)
    assert torch.equal(model.step(states, tokens), torch.as_tensor(model.successor.reshape(-1)))


# ------------------------------------------------------------------------------------------------ the second file
def test_the_bic_file_is_the_markov_file_with_its_order_rule_changed():
    a = (HERE / "model_de_bruijn.py").read_text().splitlines()
    b = (HERE / "model_de_bruijn_bic.py").read_text().splitlines()
    assert len(a) == len(b)
    differ = [i for i, (u, v) in enumerate(zip(a, b)) if u != v]
    assert all(a[i].startswith(("NAME =", "DESCRIPTION =", "ORDER_RULE =")) for i in differ), [a[i] for i in differ]
    assert 'ORDER_RULE = "bic"' in "\n".join(b)
