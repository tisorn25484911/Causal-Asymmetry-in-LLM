"""
test_models against the code that produced sections 13-17 of hybrid_model_test.ipynb (model_modeularised/).

    process.py            the same machines and analytical results as HMM_processes.py; the same training windows
    models 1 ... C        a section-16 checkpoint gives the same test results; the same loss and gradients on a batch;
                          20 updates from the same starting parameters, windows and noise give the same parameters;
                          models 2, 5a and B: the whole training run (training.train) is bit-identical
    model 2_L2            the same loss as section 17's hybrid_loss; a section-17 checkpoint gives the same results
    model 1_L2            the same loss as the notebook's train_step (section 7)
    feedback GRU,         the same starting weights for the same seed; the same loss; a section-17 checkpoint gives the
    discrete Transformer  same test results

Run from test_models/:  python -m pytest tests -q        (skipped when model_modeularised/ is not next to test_models/)
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
OLD = HERE.parent / "model_modeularised"
sys.path.insert(0, str(HERE))
import process as P                                                        # noqa: E402
import training as TR                                                      # noqa: E402

pytestmark = pytest.mark.skipif(not OLD.exists(), reason="model_modeularised/ not found next to test_models/")
if OLD.exists() and str(OLD) not in sys.path:
    sys.path.append(str(OLD))           # after test_models, so that `training`, `process` are this folder's

HYBRID16 = ["1", "2", "3", "4", "5a", "5b", "A", "B", "C"]
MODULAR = ["feedback_GRU", "discrete_Transformer"]


def s16():
    from checks import section16_experiments as S, section16_report as R
    return S, R


def s17():
    from checks import section17_experiments as S17
    return S17


def section16_trial(key, seed=0):
    S, R = s16()
    return R.full_trials()[(TR.load_model_module(key).SECTION16_NAME, seed)]


def settings16(key, trial, **changes):
    return TR.settings_for(key, weight=trial["weight"], **changes)


def same_parameters(new, old_state):
    for k, v in new.state_dict().items():
        old_key = k if k in old_state else k.replace("state_head.", "causal_head.state_head.")
        assert torch.equal(v, old_state[old_key]), k


# ------------------------------------------------------------------------------------------------ process
@pytest.mark.parametrize("name, params", [("renewal", {"F": [1 / 15] * 15}),
                                          ("renewal", {"F": [.04, .08, .12, .16, .20, .16, .12, .08, .04]}),
                                          ("even", {"p": 0.5}), ("gm", {"p": 0.3}), ("coin", {"p": 0.4, "q": 0.8})])
def test_process_copy_gives_the_same_machines_and_results(name, params):
    import HMM_processes as H
    p = P.Process(name, params)
    T, E = H.PROCESS[name]["transition"](params), H.PROCESS[name]["state_map"](params)
    assert np.array_equal(p.T, T) and np.array_equal(p.E, E)
    assert np.allclose(p.pi, H.stationary_dist(T))
    assert P.entropy_rate(p.T, p.E, p.pi) == pytest.approx(H.entropy_rate(T, E, H.stationary_dist(T)))
    assert P.epsilon_machine(p.T, p.E)["n_states"] == H.epsilon_machine(T, E)["n_states"]


def test_the_window_sampler_draws_section16s_windows():
    S, _ = s16()
    a, sa = S.Sampler(4404).draw(64, 60, states=True)
    b, sb = P.Process("renewal", {"F": [1 / 15] * 15}).sampler(4404).draw(64, 60, states=True)
    assert torch.equal(a, b) and np.array_equal(sa, sb)


def test_renewal15_analytical_results():
    a = P.Process("renewal", {"F": [1 / 15] * 15}).analytics()
    assert a["n_causal"] == 15
    assert a["statistical_c"] == pytest.approx(3.6706, abs=1e-4)
    assert a["excess_entropy"] == pytest.approx(0.4342, abs=1e-4)
    assert a["crypticity"] == pytest.approx(3.2363, abs=1e-4)


# ------------------------------------------------------------------------------------------------ section 16
@pytest.mark.parametrize("key", HYBRID16)
def test_a_section16_checkpoint_gives_the_same_test_results(key):
    S, R = s16()
    trial = section16_trial(key)
    file, _ = R.checkpoint(trial, "latest")
    settings = settings16(key, trial)
    model = TR.adopt_state_dict(TR.make_model(settings), torch.load(file, map_location="cpu", weights_only=False)["model_state"])
    x, st = settings.process_object().sampler(4404).draw(4096, settings.total_len, states=True)
    out = TR.evaluate(model, settings, x, st, trial["tau_end"])
    rep = json.loads((S.trial_dir(trial) / "report_latest.json").read_text())["metrics"]
    assert out["excess_transformer_future"] == pytest.approx(rep["test_excess_transformer_future"], abs=1e-9)
    assert out["transformer_states_used"] == rep["transformer_states_used"]
    assert out["transformer_information"] == pytest.approx(rep["transformer_information"], abs=1e-9)
    assert out["transformer_determinism"] == pytest.approx(rep["transformer_determinism"], abs=1e-9)
    for old, new in (("test_excess_gru_rollout", "excess_gru_from_handoff_future"),
                     ("test_excess_next_state_net_rollout", "excess_next_state_from_handoff_future")):
        assert (old in rep) == (new in out)
        if old in rep:
            assert out[new] == pytest.approx(rep[old], abs=1e-9)


@pytest.mark.parametrize("key", HYBRID16)
def test_the_loss_and_its_gradients_match_section16(key):
    S, R = s16()
    trial = section16_trial(key)
    file, _ = R.checkpoint(trial, "latest")
    state = torch.load(file, map_location="cpu", weights_only=False)["model_state"]
    old = S.make_model(trial)
    old.load_state_dict(state)
    new = TR.adopt_state_dict(TR.make_model(settings16(key, trial)), state)
    old.train(), new.train()
    w = P.Process("renewal", {"F": [1 / 15] * 15}).sampler(99).draw(16, 60)
    torch.manual_seed(5)
    lo, to, _ = S.loss_terms(old, w, 1.3, trial["weight"])
    lo.backward()
    torch.manual_seed(5)
    ln, tn, _ = new.loss(w, 1.3, trial["weight"])
    ln.backward()
    assert float(lo) == float(ln)
    assert set(to) == set(tn) and all(float(to[k]) == float(tn[k]) for k in to)
    old_params = dict(old.named_parameters())
    for k, p in new.named_parameters():
        assert torch.equal(p.grad, old_params[k].grad), k


@pytest.mark.parametrize("key", HYBRID16)
def test_twenty_updates_match_section16(key):
    """Same starting parameters (section 16's for this seed), windows and noise: identical parameters after 20 updates,
    including the running state counts of 5a, 5b and C."""
    S, R = s16()
    trial = section16_trial(key)
    settings = settings16(key, trial)
    old = S.make_model(trial)
    new = TR.adopt_state_dict(TR.make_model(settings), old.state_dict())
    o_opt, n_opt = S.make_optimizer(old, trial), TR.make_optimizer(new, settings)
    sampler = settings.process_object().sampler(1)
    for step in range(20):
        w = sampler.draw(8, settings.total_len)
        tau = TR.temperature(step, settings.steps, settings.tau_start, settings.tau_end)
        torch.manual_seed(100 + step)
        S.train_step(old, w, o_opt, tau, trial["weight"], clip=trial["grad_clip"])
        torch.manual_seed(100 + step)
        TR.train_step(new, w, n_opt, tau, settings.weight, settings.grad_clip)
    same_parameters(new, old.state_dict())


@pytest.mark.parametrize("key", ["2", "5a", "B"])
def test_a_whole_training_run_is_bit_identical_to_section16(key, tmp_path, monkeypatch):
    S, R = s16()
    monkeypatch.setattr(S, "RESULTS", tmp_path / "s16")
    name = TR.load_model_module(key).SECTION16_NAME
    weight = TR.settings_for(key).weight
    trial = S.trial_settings(name, 0, weight if name in S.REGULARISED else None, 20,
                             dict(eval_every=10, encoder_norm=True, weight_decay=5e-4, grad_clip=1.0))
    old, _ = S.run_trial(trial, verbose=False)
    new, folder = TR.train(TR.settings_for(key, steps=20, eval_every=10), folder=tmp_path / "new", verbose=False)
    same_parameters(new, old.state_dict())
    import pandas as pd
    h_old, h_new = pd.read_csv(S.trial_dir(trial) / "validation_history.csv"), TR.history(folder)
    metric_old = S.main_metric(name)
    metric_new = TR.main_metric(key)
    assert np.allclose(h_old[metric_old], h_new[metric_new], atol=1e-12)


# ------------------------------------------------------------------------------------------------ section 17
def section17_trial(model, seed=0, weight=None):
    return s17().trial_settings(model, seed, weight)


def test_2_L2_loss_matches_section17():
    S17 = s17()
    for weight in (0.0, 0.03):
        trial = section17_trial("l2_both_losses", 0, weight)
        old = S17.make_model(trial)
        new = TR.adopt_state_dict(TR.make_model(TR.settings_for("2_L2", weight=weight)), old.state_dict())
        old.train(), new.train()
        w = P.Process("renewal", {"F": [1 / 15] * 15}).sampler(7).draw(16, 60)
        torch.manual_seed(2)
        lo, to, _ = S17.hybrid_loss(old, w, 1.7, weight)
        torch.manual_seed(2)
        ln, tn, _ = new.loss(w, 1.7, weight)
        assert float(lo) == float(ln) and all(float(to[k]) == float(tn[k]) for k in to)


def test_a_section17_2_L2_checkpoint_gives_the_same_test_results():
    S17 = s17()
    trial = section17_trial("l2_both_losses", 0, 0.03)
    file, _ = S17.latest(trial)
    settings = TR.settings_for("2_L2", weight=0.03)
    model = TR.adopt_state_dict(TR.make_model(settings), torch.load(file, map_location="cpu", weights_only=False)["model_state"])
    x, st = settings.process_object().sampler(4404).draw(4096, 60, states=True)
    out = TR.evaluate(model, settings, x, st, 0.5)
    rep = json.loads((S17.trial_dir(trial) / "report_latest.json").read_text())["metrics"]
    assert out["excess_gru_from_handoff_future"] == pytest.approx(rep["test_excess_gru_rollout"], abs=1e-9)
    assert out["excess_transformer_future"] == pytest.approx(rep["test_excess_transformer_future"], abs=1e-9)
    assert out["l2"] == pytest.approx(rep["test_l2"], abs=1e-6)


@pytest.mark.parametrize("key", MODULAR)
def test_modular_models_start_from_section17s_weights(key):
    S17 = s17()
    old = S17.make_model(section17_trial(TR.load_model_module(key).SECTION17_NAME))
    new = TR.make_model(TR.settings_for(key))
    same_parameters(new, old.state_dict())


@pytest.mark.parametrize("key", MODULAR)
def test_modular_loss_and_gradients_match_section17(key):
    S17 = s17()
    trial = section17_trial(TR.load_model_module(key).SECTION17_NAME)
    old = S17.make_model(trial)
    new = TR.adopt_state_dict(TR.make_model(TR.settings_for(key)), old.state_dict())
    old.train(), new.train()
    w = P.Process("renewal", {"F": [1 / 15] * 15}).sampler(3).draw(16, 60)
    torch.manual_seed(9)
    lo = S17.modular_loss(old, w, 1.9)
    lo.backward()
    torch.manual_seed(9)
    ln, _, _ = new.loss(w, 1.9, 0.0)
    ln.backward()
    assert float(lo) == float(ln)
    old_params = {k.replace("causal_head.state_head.", "state_head."): p for k, p in old.named_parameters()}
    for k, p in new.named_parameters():
        assert torch.equal(p.grad, old_params[k].grad), k


@pytest.mark.parametrize("key", MODULAR)
def test_a_section17_modular_checkpoint_gives_the_same_test_results(key):
    S17 = s17()
    trial = section17_trial(TR.load_model_module(key).SECTION17_NAME)
    file, _ = S17.latest(trial)
    settings = TR.settings_for(key)
    model = TR.adopt_state_dict(TR.make_model(settings), torch.load(file, map_location="cpu", weights_only=False)["model_state"])
    x, st = settings.process_object().sampler(4404).draw(4096, 60, states=True)
    out = TR.evaluate(model, settings, x, st, 0.5)
    rep = json.loads((S17.trial_dir(trial) / "report_latest.json").read_text())["metrics"]
    p = TR.load_model_module(key).MAIN_PREDICTOR
    assert out[f"excess_{p}_future"] == pytest.approx(rep["test_excess_future"], abs=1e-9)
    assert out[f"excess_{p}_all"] == pytest.approx(rep["test_excess_all"], abs=1e-9)
    assert out[f"{p}_states_used"] == rep["states_used"]
    assert out[f"{p}_determinism"] == pytest.approx(rep["determinism"], abs=1e-9)
    assert out["handoff_margin_median"] == pytest.approx(rep["state_gap_median"], abs=1e-6)


# ------------------------------------------------------------------------------------------------ the notebook's L2
def test_1_L2_loss_is_the_notebooks_train_step():
    S, _ = s16()
    nb = json.loads((OLD / "hybrid_model_test.ipynb").read_text())
    ns = dict(vars(S))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.FunctionDef) and node.name in ("continuation_terms", "probability_l2"):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook", "exec"), ns)
    cfg = S.HybridConfig(vocab_size=2, n_states=32, state_dim=32, d_model=16, n_layers=4, history_len=40, rollout_len=20)
    torch.manual_seed(17)
    old = S.HybridTransformerGRU(cfg).train()
    new = TR.adopt_state_dict(TR.make_model(TR.settings_for("1_L2", encoder_norm=False)), old.state_dict()).train()
    w = P.Process("renewal", {"F": [1 / 15] * 15}).sampler(11).draw(16, 60)
    torch.manual_seed(4)
    logits, future, q_t, q_g = ns["continuation_terms"](old, w, 1.1)
    lo = S.prediction_loss(logits, future) + 0.03 * ns["probability_l2"](q_t, q_g)
    torch.manual_seed(4)
    ln, _, _ = new.loss(w, 1.1, 0.03)
    assert float(lo) == float(ln)
