"""
The modularised models, and the conventions the runs in run_model.py rely on.

Run:  pytest tests/ -q          (from model_modeularised/)
"""
import numpy as np
import pytest
import torch
import torch.utils.data as tud

import GRU
import Transformer
from decoder_base import cross_ent_onehot
from extraction import compare_transition_matrix, machine_state_transition, theory_from_generator
from process_generator import SequenceDataset, generator, split_loader
from run_model import RunConfig, draw
from training import eval_ce, train_model

BUILDERS = {"transformer": Transformer.build_model, "gru": GRU.build_model,
            "gru_feedback": GRU.build_feedback_model}
ARCHS = list(BUILDERS)                       # every architecture with a discrete head
V, K, S, D = 5, 10, 5, 8


def make(arch, embed_type="discrete", mode="forward", seed=0, **over):
    torch.manual_seed(seed)
    kw = dict(token_size=V, d_model=D, max_len=40, lr=1e-3, mode=mode,
              n_layers=2 if arch == "transformer" else 1, weight_decay=0.0)
    if embed_type == "discrete":
        # the feedback GRU's state vector is its hidden state: state_dim = d_model
        kw.update(n_states=K, state_dim=None if arch == "gru_feedback" else S, tau=1.0, usage_beta=1e-3)
    kw.update(over)
    return BUILDERS[arch](embed_type, **kw)


# ══════════════════════════════════════════════════════════════════════════
# the models
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("embed_type", ["onehot", "discrete"])
def test_shapes_and_analysis_surface(arch, embed_type):
    if arch == "gru_feedback" and embed_type == "onehot":
        with pytest.raises(ValueError):
            make(arch, embed_type)
        return
    m = make(arch, embed_type)
    out = m(torch.randint(0, V, (3, 17)))
    assert out.shape == (3, 17, V)
    assert m.last_encodings.shape == (3, 17, D)
    if embed_type == "discrete":
        assert m.state_assignment().shape == (3, 17)
        E = m.emission_table()
        assert E.shape == (K, V)
        assert torch.allclose(E.sum(-1), torch.ones(K))
    else:
        assert m.state_assignment() is None and m.emission_table() is None


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("mode", ["forward", "backward"])
def test_causality(arch, mode):
    """forward: position t never sees a token after t; backward: never one before t."""
    m = make(arch, mode=mode).eval()
    x = torch.randint(0, V, (2, 20))
    s = 10
    y = x.clone()
    y[:, s] = (y[:, s] + 1) % V
    with torch.no_grad():
        a, ea = m(x), m.last_encodings.clone()
        b, eb = m(y), m.last_encodings.clone()
    blind = slice(0, s) if mode == "forward" else slice(s + 1, None)
    assert torch.allclose(a[:, blind], b[:, blind], atol=1e-6)
    assert torch.allclose(ea[:, blind], eb[:, blind], atol=1e-6)
    sees = slice(s, None) if mode == "forward" else slice(0, s + 1)
    assert not torch.allclose(ea[:, sees], eb[:, sees])


@pytest.mark.parametrize("arch", ARCHS)
def test_rows_are_independent_and_calls_stateless(arch):
    m = make(arch).eval()
    x = torch.randint(0, V, (4, 20))
    y = x.clone()
    y[2] = torch.randint(0, V, (20,))
    with torch.no_grad():
        a, b, again = m(x), m(y), m(x)
    keep = [0, 1, 3]
    assert torch.allclose(a[keep], b[keep], atol=1e-6)
    assert torch.equal(a, again)


@pytest.mark.parametrize("arch", ARCHS)
def test_head_is_registered_and_trained(arch):
    """The head's weights are the model's (built once, in __init__) and one step moves them."""
    m = make(arch)
    head = dict(m.causal_head.named_parameters())
    assert {f"causal_head.{n}" for n in head} <= {n for n, _ in m.named_parameters()}
    before = {n: p.detach().clone() for n, p in head.items()}
    opt = m.configure_optimizers()
    x = torch.randint(0, V, (4, 16))
    loss, _ = cross_ent_onehot(m(x[:, :-1]), x[:, 1:])
    (loss + m.usage_penalty()[1]).backward()
    opt.step()
    for n, p in head.items():
        assert not torch.equal(before[n], p.detach()), n


@pytest.mark.parametrize("arch", ARCHS)
def test_tau_changes_the_gradient_not_the_value(arch):
    """model.tau reaches the head: same forward value at any tau, different gradient."""
    x = torch.randint(0, V, (4, 16))
    outs, grads = [], []
    for tau in (0.3, 5.0):
        m = make(arch, seed=0)
        m.tau = tau
        out = m(x[:, :-1])
        cross_ent_onehot(out, x[:, 1:])[0].backward()
        outs.append(out.detach())
        grads.append(m.causal_head.state_head.weight.grad.clone())
    assert torch.equal(outs[0], outs[1])
    assert not torch.allclose(grads[0], grads[1])


@pytest.mark.parametrize("arch", ARCHS)
def test_gumbel_switch(arch):
    """Off: no random number drawn, the deterministic head.  On: a sample while
    training, and still the deterministic argmax in eval mode."""
    x = torch.randint(0, V, (4, 30), generator=torch.Generator().manual_seed(3))
    ref = make(arch, seed=0).eval()
    with torch.no_grad():
        want = ref(x)
    m = make(arch, seed=0)
    m.gumbel = True
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(x), want)                      # eval: no noise
    m.train()
    torch.manual_seed(1)
    with torch.no_grad():
        a = m(x).clone(); sa = m.last_states.clone()
        b = m(x).clone(); sb = m.last_states.clone()
    assert not torch.equal(sa, sb)                         # training: sampled
    off = make(arch, seed=0).train()
    torch.manual_seed(1)
    r0 = torch.rand(1)
    torch.manual_seed(1)
    with torch.no_grad():
        off(x)
    assert torch.equal(torch.rand(1), r0)                  # off: the RNG is untouched


@pytest.mark.parametrize("arch", ["gru", "gru_feedback"])
def test_gru_backward_mode_mirrors_forward(arch):
    """Same weights: a mode='backward' GRU on x is a forward GRU on x reversed, reversed back."""
    fw, bw = make(arch, mode="forward"), make(arch, mode="backward")
    bw.load_state_dict(fw.state_dict())
    x = torch.randint(0, V, (3, 25))
    with torch.no_grad():
        assert torch.allclose(bw(x), fw(x.flip(1)).flip(1), atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_feedback_gru_is_unifilar_by_construction(seed):
    """Every state on every trajectory is what the enumerated machine says:
    s_0 = start_state[x_0], s_t = next_state[s_{t-1}, x_t] -- for any weights."""
    m = make("gru_feedback", seed=seed, tau=0.7).eval()
    x = torch.randint(0, V, (6, 60), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        m(x)
    s = m.last_states.numpy()
    mach = m.machine()
    xn = x.numpy()
    assert np.array_equal(s[:, 0], mach["start_state"][xn[:, 0]])
    assert np.array_equal(s[:, 1:], mach["next_state"][s[:, :-1], xn[:, 1:]])
    # and the emission of a state is the machine's emission row
    with torch.no_grad():
        probs = torch.softmax(m(x), dim=-1).numpy()
    np.testing.assert_allclose(probs, mach["emission_probs"][s], atol=1e-6)


def test_feedback_gru_rejects_a_state_dim_it_cannot_feed_back():
    with pytest.raises(ValueError):
        make("gru_feedback", state_dim=S)          # S != d_model
    with pytest.raises(ValueError):
        make("gru_feedback", n_layers=2)


# ══════════════════════════════════════════════════════════════════════════
# the data conventions
# ══════════════════════════════════════════════════════════════════════════
def test_reversed_data_equals_backward_mode():
    """generator.reverse() + mode='forward' scores exactly what mode='backward' scores
    on the original data -- the convention the runs use.  Exact for the GRU."""
    gen = generator("coin", 8, 30, {"p": 0.7, "q": 0.8}, burn_in=10)
    seqs = gen.generate(np.random.default_rng(0)).copy()
    gen.reverse()
    assert np.array_equal(gen.all_sequence, seqs[:, ::-1])
    over = dict(token_size=3, n_states=6, state_dim=3)
    fw, bw = make("gru", mode="forward", **over), make("gru", mode="backward", **over)
    bw.load_state_dict(fw.state_dict())
    loader = lambda s: tud.DataLoader(SequenceDataset(s), batch_size=4)    # noqa: E731
    assert eval_ce(fw, loader(gen.all_sequence))[0] == pytest.approx(eval_ce(bw, loader(seqs))[0], abs=1e-6)


def test_both_arms_share_one_draw():
    cfg = RunConfig(num_samples=10, seq_len=20)
    _, f = draw(cfg, "flower", "forward", 10, 20, 1)
    g, b = draw(cfg, "flower", "backward", 10, 20, 1)
    assert np.array_equal(f[:, ::-1], b) and g.direct == "backward"


def test_split_loader_is_seeded_and_disjoint():
    ds = SequenceDataset(np.random.default_rng(0).integers(0, 3, (50, 11)))
    tr1, te1 = split_loader(ds, 8, 0.2, seed=3)
    tr2, te2 = split_loader(ds, 8, 0.2, seed=3)
    assert list(te1.dataset.indices) == list(te2.dataset.indices)
    assert set(tr1.dataset.indices).isdisjoint(te1.dataset.indices)
    assert len(tr1.dataset) + len(te1.dataset) == 50


# ══════════════════════════════════════════════════════════════════════════
# the theory the runs are scored against
# ══════════════════════════════════════════════════════════════════════════
CASES = [("coin", {"p": 0.7, "q": 0.8}, "forward", 2), ("coin", {"p": 0.7, "q": 0.8}, "backward", 3),
         ("flower", {"n": 3, "m": 4, "dice_seed": 0}, "forward", 4),
         ("flower", {"n": 3, "m": 4, "dice_seed": 0}, "backward", 5)]


@pytest.mark.parametrize("process,params,arm,k", CASES)
def test_theory_from_generator(process, params, arm, k):
    gen = generator(process, 1, 10, params)
    if arm == "backward":
        gen.reverse()
    th = theory_from_generator(gen)
    assert th["true_k"] == k and th["view"] == arm
    np.testing.assert_allclose(th["true_machine"]["emission_probs"].sum(1), 1.0, atol=1e-12)
    # the (next_state, emission) machine reproduces the state-to-state matrix: unifilar
    np.testing.assert_allclose(machine_state_transition(th["true_machine"]), th["T_theory"], atol=1e-12)
    # emission-row matching can separate every pair of theoretical states
    assert th["min_emission_tv"] > 0.05


@pytest.mark.parametrize("process,params,arm,k", CASES)
def test_compare_transition_matrix_recovers_the_theory_itself(process, params, arm, k):
    gen = generator(process, 1, 10, params)
    if arm == "backward":
        gen.reverse()
    th = theory_from_generator(gen)
    E = th["true_machine"]["emission_probs"]
    cmp = compare_transition_matrix(th["T_theory"], th["T_theory"], E, E, weights=th["occupancy"])
    assert cmp["missing"] == [] and cmp["max_error"] < 1e-12 and cmp["n_visited"] == k


# ══════════════════════════════════════════════════════════════════════════
# the training loop
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("arch", ARCHS)
def test_train_model_schedules_tau_and_restores_best(arch):
    gen = generator("coin", 32, 40, {"p": 0.7, "q": 0.8}, burn_in=10)
    ds = SequenceDataset(gen.generate(np.random.default_rng(0)))
    tr, te = split_loader(ds, 8, 0.25, seed=0)
    rec = train_model(tr, arch, "discrete", val_loader=te, token_size=3, d_model=8, max_len=40,
                      max_epochs=2, lr=1e-3, n_layers=1, weight_decay=0.01, accelerator="cpu",
                      val_every_n_steps=2, n_states=6,
                      state_dim=None if arch == "gru_feedback" else 3, tau="geom:5:0.5",
                      usage_beta=1e-3, restore_best=True)
    steps = len(rec.step_loss)
    assert steps == 2 * len(tr) == len(rec.step_grad_norm) == len(rec.step_tau)
    assert rec.step_tau[0] == pytest.approx(5.0) and rec.step_tau[-1] == pytest.approx(0.5)
    assert rec.restored_best and 0 < rec.best_step <= steps
    assert eval_ce(rec.model, te)[0] == pytest.approx(rec.best_val, abs=1e-5)
