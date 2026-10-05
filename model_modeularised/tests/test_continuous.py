"""
The continuous data mode's model, training and analysis changes (SHARED_TRAJECTORY_
EMBEDDING_WARMUP_PLAN.md section 8, gates 5-17): masked objectives, learned input maps,
exact budgets and schedule horizons, the regime resolution and its flags, the collapse
gate and the overfitting monitor, coverage-aware symbolic scoring, checkpoints, collisions.

Run:  pytest tests/ -q          (from model_modeularised/)
"""
import copy
import glob
import json
import math
import os

import numpy as np
import pytest
import torch
import torch.utils.data as tud

import GRU
import Transformer
from extraction import (collapse_record, empirical_symbolic_machine, overfit_record, symbolic_trace)
from process_generator import SequenceDataset, generator, split_loader
from scoring import masked_ce_bits
from tau_schedue import parse_tau
from training import eval_ce, train_model

BUILDERS = {"transformer": Transformer.build_model, "gru": GRU.build_model,
            "gru_feedback": GRU.build_feedback_model}
ARCHS = list(BUILDERS)
V, K, S, D = 3, 6, 3, 8


def make(arch, mode="forward", seed=0, **over):
    torch.manual_seed(seed)
    kw = dict(token_size=V, d_model=D, max_len=40, lr=1e-3, mode=mode,
              n_layers=2 if arch == "transformer" else 1, weight_decay=0.0,
              n_states=K, state_dim=None if arch == "gru_feedback" else S, tau=1.0, usage_beta=1e-3)
    kw.update(over)
    return BUILDERS[arch]("discrete", **kw)


def _input_capture(m, arch):
    """Retain the per-position INPUT representation each architecture reads."""
    got = []

    def keep(t):
        t.retain_grad()
        got.append(t)

    if arch == "transformer":                   # projected tokens + positional encoding
        m.pe.register_forward_hook(lambda mod, inp, out: keep(out))
    elif arch == "gru":                         # the mapped tokens entering nn.GRU
        m.gru.register_forward_pre_hook(lambda mod, inp: keep(inp[0]))
    else:                                       # each step's mapped token entering the cell
        m.cell.register_forward_pre_hook(lambda mod, inp: keep(inp[0]))
    return got


# ══════════════════════════════════════════════════════════════════════════
# gate 5: the masked objective trains the prefix without scoring it
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("with_usage", [False, True])
def test_masked_loss_trains_the_prefix_without_scoring_it(arch, with_usage):
    # the learned map makes every input representation differentiable, so its gradient can be read
    m = make(arch, score_start=20, usage_scope="scored", learn_input_embedding=True).train()
    got = _input_capture(m, arch)
    torch.manual_seed(1)
    x, y = torch.randint(0, V, (4, 30)), torch.randint(0, V, (4, 30))
    logits = m(x)
    logits.retain_grad()
    loss, mask = masked_ce_bits(logits, y, 20)
    if with_usage:
        loss = loss + m.usage_penalty(mask)[1]
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :20]).item() == 0          # no direct CE on the prefix
    assert torch.count_nonzero(logits.grad[:, 20:]).item() > 0
    prefix = (torch.stack([g.grad for g in got[:20]], dim=1) if arch == "gru_feedback"
              else got[0].grad[:, :20])                                 # suffix losses reach the prefix inputs
    assert torch.isfinite(prefix).all() and torch.count_nonzero(prefix).item() > 0


# ══════════════════════════════════════════════════════════════════════════
# gates 6-8: causality, embedding registration, the feedback machine
# ══════════════════════════════════════════════════════════════════════════
def _one_update(m, steps=3):
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    for _ in range(steps):
        x = torch.randint(0, V, (4, 20))
        loss, _ = masked_ce_bits(m(x[:, :-1]), x[:, 1:], 5)
        opt.zero_grad()
        loss.backward()
        opt.step()


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("mode", ["forward", "backward"])
def test_causality_with_a_learned_input_map(arch, mode):
    m = make(arch, mode=mode, learn_input_embedding=True, score_start=3)
    _one_update(m)                               # the map is no longer the identity
    m.eval()
    x = torch.randint(0, V, (2, 25), generator=torch.Generator().manual_seed(3))
    t = 12
    x2 = x.clone()
    if mode == "forward":
        x2[:, t + 1:] = (x2[:, t + 1:] + 1) % V   # change only the future
        keep = slice(0, t + 1)
    else:
        x2[:, :t] = (x2[:, :t] + 1) % V
        keep = slice(t, None)
    with torch.no_grad():
        assert torch.allclose(m(x)[:, keep], m(x2)[:, keep], atol=1e-6)


@pytest.mark.parametrize("arch", ARCHS)
def test_input_maps_register_and_start_equivalent(arch):
    frozen, learned = make(arch, seed=4), make(arch, seed=4, learn_input_embedding=True)
    x = torch.randint(0, V, (3, 20))
    with torch.no_grad():
        assert torch.equal(frozen.eval()(x), learned.eval()(x))       # identical start, bit for bit
    name = "rand_prj" if arch == "transformer" else "symbol_embedding"
    assert name in dict(learned.named_parameters())
    assert name not in dict(frozen.named_parameters())
    opt = learned.configure_optimizers()
    assert any(p is dict(learned.named_parameters())[name] for g in opt.param_groups for p in g["params"])
    before_l, before_f = learned.get_parameter(name).detach().clone(), getattr(frozen, name)
    before_f = None if before_f is None else before_f.detach().clone()
    _one_update(learned.train())
    _one_update(frozen.train())
    assert not torch.equal(learned.get_parameter(name), before_l)    # learned: it moves
    if before_f is not None:
        assert torch.equal(getattr(frozen, name), before_f)          # frozen buffer: it does not


def test_feedback_machine_follows_the_learned_map_after_an_update():
    m = make("gru_feedback", learn_input_embedding=True, tau=0.7)
    _one_update(m.train(), steps=5)
    assert not torch.allclose(m.symbol_embedding, torch.eye(V))
    m.eval()
    x = torch.randint(0, V, (6, 50), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        m(x)
    s, mach, xn = m.last_states.numpy(), m.machine(), x.numpy()
    assert np.array_equal(s[:, 0], mach["start_state"][xn[:, 0]])
    assert np.array_equal(s[:, 1:], mach["next_state"][s[:, :-1], xn[:, 1:]])


# ══════════════════════════════════════════════════════════════════════════
# gates 11 and 14: budgets, horizons, the learning-rate schedule, final validation
# ══════════════════════════════════════════════════════════════════════════
def _loaders(n=40, T=30, batch=8):
    gen = generator("coin", n, T, {"p": 0.7, "q": 0.8}, burn_in=10)
    ds = SequenceDataset(gen.generate(np.random.default_rng(0)))
    return split_loader(ds, batch, 0.25, seed=0)


def _train(**over):
    tr, te = _loaders()
    kw = dict(token_size=3, d_model=8, max_len=30, max_epochs=50, lr=1e-3, n_layers=1, weight_decay=5e-4,
              accelerator="cpu", val_every_n_steps=4, n_states=6, state_dim=None, tau="geom:2.25:0.1",
              usage_beta=1e-4, restore_best=True, gumbel=True, score_start=5, usage_scope="scored",
              max_steps=10, lr_schedule="cosine", lr_min=1e-5, schedule_horizon=20)
    kw.update(over)
    return train_model(tr, "gru_feedback", "discrete", val_loader=te, **kw)


def _cosine(k, T, lr0=1e-3, lr_min=1e-5):
    return lr_min + (lr0 - lr_min) * (1 + math.cos(math.pi * k / T)) / 2


def test_exact_budget_horizon_and_learning_rate_per_update():
    rec = _train()
    assert rec.actual_steps == 10 and rec.update_budget == 10 and rec.schedule_horizon == 20
    # step_lr: the rate after update k (k = 4, 8, then the final 10), on a 20-update cosine
    assert rec.step_val_at == [4, 8, 10]
    for k, lr in zip(rec.step_val_at, rec.step_lr):
        assert lr == pytest.approx(_cosine(k, 20), rel=1e-9)
    assert rec.lr_last_used == pytest.approx(_cosine(9, 20), rel=1e-9)      # the rate update 10 used
    _, f = parse_tau("geom:2.25:0.1")
    assert rec.tau_at_final_update == pytest.approx(f(9 / 19))
    assert rec.step_tau[-1] == pytest.approx(rec.tau_at_final_update)        # what the schedule set
    assert rec.final_validation_done


def test_the_plans_two_times_horizon_values():
    _, f = parse_tau("geom:2.25:0.1")
    assert f(29_999 / 59_999) == pytest.approx(0.4744, abs=1e-4)
    assert _cosine(29_999, 60_000) == pytest.approx(5.05e-4, rel=1e-3)


def test_a_horizon_shorter_than_the_budget_is_rejected():
    with pytest.raises(ValueError):
        _train(schedule_horizon=5)
    with pytest.raises(ValueError):
        _train(max_steps=0)


def test_final_validation_can_be_switched_off():
    rec = _train(final_validation=False)
    assert rec.step_val_at == [4, 8] and not rec.final_validation_done


def test_the_overfitting_probe_is_recorded_at_every_validation():
    tr, te = _loaders()
    rec = _train(train_probe_loader=te, probe_exact_ce=0.5)
    assert len(rec.step_train_probe_ce) == len(rec.step_val_at) == len(rec.val_states_in_use)


# ══════════════════════════════════════════════════════════════════════════
# gates 15 and 17: the collapse gate and the overfitting monitor
# ══════════════════════════════════════════════════════════════════════════
def test_a_forced_one_state_model_is_counted_as_one_state():
    m = make("transformer")
    with torch.no_grad():
        m.causal_head.state_head.weight.zero_()
        m.causal_head.state_head.bias.copy_(torch.tensor([10.0] + [0.0] * (K - 1)))
    gen = generator("coin", 12, 30, {"p": 0.7, "q": 0.8}, burn_in=10)
    loader = tud.DataLoader(SequenceDataset(gen.generate(np.random.default_rng(0))), batch_size=4)
    ce, _, used = eval_ce(m, loader, count_states=True)
    assert used == 1 and np.isfinite(ce)
    assert collapse_record([25, 50], [used, used], 50)["collapse_flagged"]


def test_collapse_is_judged_on_the_selected_checkpoint_only():
    """User, 2026-10-03: the early plateau is recorded, not flagged."""
    rec = collapse_record([25, 50, 75, 100], [1, 2, 4, 9], best_step=100, true_k=9)
    assert not rec["collapse_flagged"] and rec["states_at_first_val"] == 1 and rec["states_at_selected"] == 9
    assert rec["first_val_above_collapse"] == 75 and rec["n_val_collapsed"] == 2 and rec["collapse_threshold"] == 2
    assert collapse_record([25, 50, 75, 100], [1, 2, 4, 9], best_step=50, true_k=9)["collapse_flagged"]
    assert collapse_record([], [], -1) == {}


def test_collapse_means_fewer_states_than_needed():
    """User, 2026-10-03: at most min(2, k - 1) states.  A 2-state process using its 2 states is
    not collapsed (coin forward); at 1 state it is.  k >= 3 keeps the '<= 2 states' rule."""
    assert not collapse_record([25, 50], [2, 2], 50, true_k=2)["collapse_flagged"]
    assert collapse_record([25, 50], [2, 1], 50, true_k=2)["collapse_flagged"]
    assert collapse_record([25, 50], [3, 2], 50, true_k=3)["collapse_flagged"]
    assert not collapse_record([25, 50], [3, 3], 50, true_k=3)["collapse_flagged"]
    assert collapse_record([25, 50], [2, 2], 50, true_k=2)["collapse_threshold"] == 1


def test_training_ce_below_the_exact_model_flags_overfitting():
    """Flagged when some validation falls more than OVERFIT_TOL (0.001 bits; user, 2026-10-03)
    below the exact model: noise around a near-exact model stays within it."""
    from extraction import OVERFIT_TOL
    assert OVERFIT_TOL == 1e-3
    assert overfit_record([0.80, 0.71, 0.69], 0.70)["overfit_flagged"]          # 0.01 below: memorising
    assert overfit_record([0.80, 0.7015, 0.6985], 0.70)["overfit_flagged"]      # 0.0015 below
    rec = overfit_record([0.80, 0.7002, 0.6997], 0.70)                          # 0.0003 below: noise
    assert not rec["overfit_flagged"] and rec["train_probe_min_excess"] == pytest.approx(-0.0003)
    rec = overfit_record([0.80, 0.71, 0.705], 0.70)
    assert not rec["overfit_flagged"] and rec["train_probe_min_excess"] == pytest.approx(0.005)
    assert overfit_record([], 0.7) == {}


# ══════════════════════════════════════════════════════════════════════════
# gate 9: analysis alignment -- adjacency and coverage
# ══════════════════════════════════════════════════════════════════════════
class _Stub(torch.nn.Module):
    """A 'model' whose states are fixed numbers, to read the empirical machine off exactly."""

    def __init__(self, states):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.states, self.n_states, self.token_size, self.mode = torch.as_tensor(states), 4, 2, "forward"
        self.last_states = None

    @staticmethod
    def _split(batch, mode):
        return batch

    def forward(self, x):
        self.last_states = self.states[: x.shape[0]]
        return torch.zeros(x.shape[0], x.shape[1], 2)

    def emission_table(self):
        return torch.full((4, 2), 0.5)


def test_transitions_pair_only_adjacent_scored_positions():
    states = [[0, 1, 2, 3, 0, 1]]
    tokens = torch.tensor([[0, 1, 0, 1, 1, 0]])
    loader = [(tokens, tokens)]
    mach = empirical_symbolic_machine(_Stub(states), loader, min_pos=0, analysis_start=2)
    # positions 2..5 are scored: pairs (2,3), (3,4), (4,5) only -> 2->3 on token 1, 3->0 on 1, 0->1 on 0
    assert list(mach["state_ids"]) == [0, 1, 2, 3]
    assert mach["pair_counts"].sum() == 3
    assert mach["next_state"][2, 1] == 3 and mach["next_state"][3, 1] == 0 and mach["next_state"][0, 0] == 1
    assert np.isnan(mach["determinism"][1]).all()          # state 1 is only ever a successor
    assert mach["counts"].tolist() == [1, 1, 1, 1]          # visits on positions 2..5


def test_symbolic_trace_never_invents_coverage():
    mach = {"next_state": np.array([[0, 1], [0, 1]]), "emission_probs": np.array([[0.5, 0.5], [0.25, 0.75]]),
            "determinism": np.array([[1.0, 1.0], [1.0, np.nan]])}   # (1, token 1) never observed
    tokens = np.array([[0, 0, 1, 1, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1]])
    tr = symbolic_trace(mach, tokens, np.array([0, 0, -1]), t0=1)
    # row 0: from state 0 read 1 -> state 1, then token 1 in state 1 is unseen: scored there, then stops
    assert tr["valid"][0].tolist() == [False, True, True, False, False]
    assert tr["losses"][0, 1] == pytest.approx(1.0) and tr["losses"][0, 2] == pytest.approx(-math.log2(0.75))
    assert tr["valid"][1].tolist() == [False, True, True, True, False]
    assert not tr["valid"][2].any()                           # unknown start: nothing substituted
    assert tr["unknown_start"] == 1 and tr["unseen_transition"] == 1


# ══════════════════════════════════════════════════════════════════════════
# gate 13: regime resolution and the command-line flags
# ══════════════════════════════════════════════════════════════════════════
def _parsed(argv):
    import shlex
    import unittest.mock as um
    import run_model

    class Parsed(Exception):
        pass

    def fake(cfg, arch, process, arm, verbose=True, data_dir=None):
        raise Parsed(cfg)

    with um.patch.object(run_model, "run_arm", fake), pytest.raises(Parsed) as e:
        run_model.main(shlex.split(argv) + ["--no-plots", "--name", "flagtest", "--out", "/tmp/claude-flagtest"])
    return e.value.args[0]


def test_legacy_resolves_exactly_the_current_values():
    from run_model import RunConfig, arch_settings
    cfg = _parsed("")
    assert cfg == RunConfig()                                  # nothing given, nothing changed
    t, f = arch_settings(cfg, "transformer", 2), arch_settings(cfg, "gru_feedback", 2)
    assert (t["gumbel"], f["gumbel"], t["tau"], f["tau"]) == (False, False, "geomhold:5:0.5:0.8", "geom:5:0.5")
    assert (cfg.weight_decay, cfg.lr_schedule, cfg.gradient_clip_val, cfg.schedule_horizon_mult) == \
           (0.01, "constant", None, 1.0)
    assert t["d_model"] == f["d_model"] == 16


def test_continuous_resolves_the_studied_regime_for_every_architecture():
    from run_model import arch_settings
    cfg = _parsed("--data-mode continuous")
    for arch in ("transformer", "gru", "gru_feedback"):
        a = arch_settings(cfg, arch, 2)
        assert a["gumbel"] is True and a["tau"] == "geom:2.25:0.1"
    assert (cfg.weight_decay, cfg.lr_schedule, cfg.lr_min, cfg.gradient_clip_val, cfg.schedule_horizon_mult) == \
           (5e-4, "cosine", 1e-5, 1.0, 2.0)
    assert arch_settings(cfg, "transformer", 2)["d_model"] == 16
    assert arch_settings(cfg, "gru", 2)["d_model"] == 32
    assert arch_settings(cfg, "gru_feedback", 2)["state_dim"] == 32       # S = d_model
    assert (cfg.gru_stride, cfg.gru_eval_stride, cfg.gru_batch_size, cfg.random_shift) == (1, 10, 128, True)


def test_explicit_flags_override_the_continuous_regime():
    from run_model import arch_settings
    cfg = _parsed("--data-mode continuous --no-gumbel-gru --weight-decay 0.01 --tau const:1 --no-random-shift")
    assert arch_settings(cfg, "gru", 2)["gumbel"] is False and arch_settings(cfg, "transformer", 2)["gumbel"] is True
    assert cfg.weight_decay == 0.01 and cfg.tau == cfg.tau_transformer == "const:1" and cfg.random_shift is False
    assert {"gumbel_gru", "weight_decay", "tau", "tau_transformer", "random_shift"} <= set(cfg.explicit_overrides)
    assert cfg.lr_schedule == "cosine"                          # not given: still the regime's


def test_conflicting_budgets_are_rejected():
    import run_model
    with pytest.raises(SystemExit):
        run_model.main(["--max-steps", "10", "--max-epochs", "5", "--no-plots", "--name", "flagtest",
                        "--out", "/tmp/claude-flagtest"])


# ══════════════════════════════════════════════════════════════════════════
# collisions and checkpoints
# ══════════════════════════════════════════════════════════════════════════
def test_a_different_configuration_is_never_silently_overwritten(tmp_path):
    from dataclasses import asdict, replace
    from run_model import RunConfig, config_collision
    cfg = RunConfig(data_mode="continuous")
    with open(tmp_path / "gru_coin_forward_s0.json", "w") as fh:
        json.dump({"run_config": json.loads(json.dumps(asdict(cfg), default=str))}, fh)
    assert config_collision(str(tmp_path), "gru_coin_forward_s0", cfg) is None
    assert config_collision(str(tmp_path), "gru_coin_forward_s0", replace(cfg, progress_every=1)) is None
    diff = config_collision(str(tmp_path), "gru_coin_forward_s0", replace(cfg, gru_stride=10))
    assert diff == {"gru_stride": (1, 10)}
    assert config_collision(str(tmp_path), "absent", cfg) is None


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("learned", [False, True])
def test_checkpoints_rebuild_from_their_hyperparameters(arch, learned):
    m = make(arch, learn_input_embedding=learned, score_start=4, lr_schedule="cosine", schedule_horizon=50)
    _one_update(m.train())
    again = type(m)(**dict(m.hparams))
    again.load_state_dict({k: v.clone() for k, v in m.state_dict().items()})
    assert again.learn_input_embedding == learned and again.score_start == 4 and again.schedule_horizon == 50
    x = torch.randint(0, V, (2, 15))
    with torch.no_grad():
        assert torch.equal(m.eval()(x), again.eval()(x))


def test_an_old_saved_run_loads_with_the_new_code():
    from training import load_pkl
    paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "default", "runs", "*.pkl")))
    if not paths:
        pytest.skip("no saved legacy runs in results/default")
    import importlib
    for p in paths[:3]:
        r = load_pkl(p)
        cls = {"DiscreteCausalDecoder": Transformer.DiscreteCausalDecoder, "DiscreteCausalGRU": GRU.DiscreteCausalGRU,
               "DiscreteFeedbackGRU": GRU.DiscreteFeedbackGRU}[r["model_class"]]
        m = cls(**r["hparams"])
        m.load_state_dict(r["state_dict"])                     # strict: no key added or missing
        assert m.score_start == 0 and not m.learn_input_embedding and m.lr_schedule == "constant"
    importlib.invalidate_caches()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
@pytest.mark.parametrize("learned", [False, True])
def test_masked_transformer_gradients_on_mps_match_cpu_in_both_input_modes(learned):
    from training import ARCHITECTURES
    torch.manual_seed(0)
    model = ARCHITECTURES["transformer"]("discrete", token_size=3, d_model=16, max_len=40, lr=1e-3,
                                         mode="forward", n_layers=2, weight_decay=0.0, n_states=6, state_dim=3,
                                         tau=1.0, usage_beta=0.0, score_start=20, learn_input_embedding=learned)
    x = torch.randint(0, 3, (8, 41))

    def grads(dev):
        m = copy.deepcopy(model).to(dev).train()
        masked_ce_bits(m(x[:, :-1].to(dev)), x[:, 1:].to(dev), 20)[0].backward()
        return {n: p.grad.detach().cpu() for n, p in m.named_parameters() if p.grad is not None}

    cpu, mps = grads("cpu"), grads("mps")
    assert ("rand_prj" in cpu) == learned
    err = {n: float((mps[n] - cpu[n]).abs().max() / (cpu[n].abs().max() + 1e-12))
           for n in cpu if not n.endswith("wk.bias")}
    assert max(err.values()) < 1e-3, sorted(err.items(), key=lambda kv: -kv[1])[:3]


def test_the_continuous_budget_rules():
    """Plan section 12.8: 30,000 updates unless --max-steps, or an explicit epoch budget; horizon 2x."""
    from run_model import continuous_budget
    cfg = _parsed("--data-mode continuous")
    assert continuous_budget(cfg, "gru", 1875) == (30_000, 30_000, 60_000)
    assert continuous_budget(_parsed("--data-mode continuous --max-steps 100"), "transformer", 25) == (100, 100, 200)
    cfg = _parsed("--data-mode continuous --max-epochs 3")
    assert continuous_budget(cfg, "gru", 1875) == (None, 5625, 11_250)
    assert continuous_budget(cfg, "transformer", 25) == (None, 75, 150)
    cfg = _parsed("--data-mode continuous --max-epochs-transformer 4")      # the transformer's only
    assert continuous_budget(cfg, "transformer", 25) == (None, 100, 200)
    assert continuous_budget(cfg, "gru", 1875) == (30_000, 30_000, 60_000)
    cfg = _parsed("--data-mode continuous --schedule-horizon-mult 1")
    assert continuous_budget(cfg, "gru", 1875) == (30_000, 30_000, 30_000)
