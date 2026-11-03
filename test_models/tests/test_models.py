"""
Every model file of test_models/ works end to end on a tiny run: it trains, its run reloads with the same parameters,
longer training continues it, its predictors are measured, and analysis.py reads its machine.  A new model_<name>.py
file is found without changing any other file.

Run from test_models/:  python -m pytest tests -q
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import analysis as A                                                       # noqa: E402
import training as TR                                                      # noqa: E402

TINY = dict(steps=6, eval_every=3, history_len=24, rollout_len=8, context=11, batch_size=4, validation_rows=16)


def tiny(name, **changes):
    extra = {} if name == "feedback_GRU" else dict(n_layers=1)
    return TR.settings_for(name, **dict(TINY, **extra, **changes))


@pytest.mark.parametrize("name", TR.available_models())
def test_every_model_trains_reloads_and_continues(name, tmp_path):
    settings = tiny(name)
    model, folder = TR.train(settings, folder=tmp_path / name, verbose=False)
    for file in ("settings.json", "completed.pt", "best.pt", "validation_history.csv", "training_history.csv"):
        assert (folder / file).exists(), file
    again, s2, update = TR.load_run(folder)
    assert update == settings.steps and s2 == settings
    for k, v in model.state_dict().items():
        assert torch.equal(v, again.state_dict()[k]), k
    h = TR.history(folder)
    assert list(h["update"]) == [0, 3, 6] and np.isfinite(h[TR.main_metric(name)]).all()
    TR.extend(folder, 3, verbose=False)
    assert TR.checkpoint(folder, "latest")[1] == 9 and TR.history(folder)["update"].max() == 9


@pytest.mark.parametrize("name", TR.available_models())
def test_every_model_is_measured_and_its_machine_is_read_off(name, tmp_path):
    settings = tiny(name)
    _, folder = TR.train(settings, folder=tmp_path / name, verbose=False)
    model, _, _ = TR.load_run(folder)
    x, st, exact = A.windows(settings, "validation")
    out = TR.evaluate(model, settings, x, st, settings.tau_end, exact)
    module = TR.load_model_module(name)
    assert f"excess_{module.MAIN_PREDICTOR}_future" in out
    for rule in model.step_rules():
        assert f"excess_{rule}_future" in out
    m = A.machine(folder)
    assert m["predictor"] == module.MAIN_PREDICTOR and m["n_states"] >= 1
    assert np.allclose(np.asarray(m["pi"]).sum(), 1.0, atol=1e-6) or m["kind"] == "state sequence"
    cmp = A.transition_comparison(m, settings)
    assert 1 <= cmp["covered"] <= len(settings.process_object().T)


def test_a_new_model_file_is_found(tmp_path, monkeypatch):
    new = HERE / "model_zz_copy_of_2.py"
    try:
        shutil.copy(HERE / "model_2.py", new)
        assert "zz_copy_of_2" in TR.available_models()
        s = tiny("zz_copy_of_2")
        model, folder = TR.train(s, folder=tmp_path / "zz", verbose=False)
        assert TR.history(folder)["update"].max() == s.steps
    finally:
        new.unlink(missing_ok=True)
        sys.modules.pop("model_zz_copy_of_2", None)


def test_settings_for_applies_the_model_defaults_then_the_changes():
    assert TR.settings_for("5b").weight == 0.1 and TR.settings_for("5b", weight=0.3).weight == 0.3
    assert TR.settings_for("feedback_GRU").d_model == 32 and TR.settings_for("2").d_model == 16
    with pytest.raises(ValueError):
        TR.settings_for("2", no_such_setting=1)


def test_a_finished_run_reloads_instead_of_training_again(tmp_path):
    s = tiny("1")
    _, folder = TR.train(s, folder=tmp_path / "r", verbose=False)
    stamp = (folder / "completed.pt").stat().st_mtime_ns
    TR.train(s, folder=folder, verbose=False)
    assert (folder / "completed.pt").stat().st_mtime_ns == stamp


def test_the_true_machine_scores_the_known_inferred_entropy():
    """Models 5a/5b and C on the true renewal-15 machine: 3.66 bits from one or two tokens; the crypticity 3.2363 is the
    floor for any number of tokens (section 16.2 of hybrid_model_test.ipynb)."""
    s = TR.settings_for("5b")
    one, two = A.inferred_entropy_on_true_machine(s, 1), A.inferred_entropy_on_true_machine(s, 2)
    assert one["true_machine_entropy"] == pytest.approx(3.6645, abs=1e-3)
    assert two["true_machine_entropy"] == pytest.approx(3.6571, abs=1e-3)
    assert one["crypticity"] == pytest.approx(3.2363, abs=1e-4) and two["true_machine_entropy"] > one["crypticity"]


def test_the_reference_levels_of_renewal15():
    """Section 16's levels on its test windows: 'remembers nothing' 0.0549, 'last symbol only' 0.0513."""
    levels = A.reference_levels(TR.settings_for("2"), "test")
    assert levels["remembers_nothing"] == pytest.approx(0.0549, abs=1e-4)
    assert levels["last_symbol_only"] == pytest.approx(0.0513, abs=1e-4)


# ------------------------------------------------------------------------------------------------ new processes (section 7)
FLOWER = dict(process="flower", params={"n": 8, "m": 12, "dice_seed": 0})
FLOWER_BACKWARD = dict(process="flower", params={"n": 8, "m": 12, "dice_seed": 0, "direction": "backward"})


def word_probs(proc, length):
    """P(w) of every word w of `length` tokens (index: the tokens as digits base V), from the process's machine."""
    V = proc.vocab_size
    probs = proc.pi[None, :]
    for _ in range(length):
        probs = np.einsum("ws,stx->wxt", probs, proc.M).reshape(-1, len(proc.pi))
    return probs.sum(-1)


def test_the_new_processes_have_their_known_numbers():
    """renewal-60: 60 states, C_mu 5.6398; flower (8 dice, 12 faces): 9 states and C_mu 2.5 = 1 + log2(8)/2 forward
    (Thompson et al. 2018, Appendix D), 13 states and C_mu 2.7535 backward; each machine is its own epsilon-machine."""
    for changes, n, c_mu in ((dict(process="renewal", params={"F": [1 / 60] * 60}), 60, 5.6398), (FLOWER, 9, 2.5),
                             (FLOWER_BACKWARD, 13, 2.7535)):
        proc = TR.settings_for("2", **changes).process_object()
        a = proc.analytics()
        assert len(proc.T) == n and a["n_causal"] == n and proc.is_epsilon_machine
        assert a["statistical_c"] == pytest.approx(c_mu, abs=1e-4)


def test_the_backward_flower_is_the_forward_one_read_from_right_to_left():
    """Every 3-token word has the probability of the reversed word under the forward machine."""
    fwd = TR.settings_for("2", **FLOWER).process_object()
    bwd = TR.settings_for("2", **FLOWER_BACKWARD).process_object()
    V = fwd.vocab_size
    p_fwd, p_bwd = word_probs(fwd, 3), word_probs(bwd, 3)
    reverse = np.array([(w % V) * V * V + (w // V % V) * V + w // (V * V) for w in range(V ** 3)])
    assert np.abs(p_bwd - p_fwd[reverse]).max() < 1e-15 and p_fwd.sum() == pytest.approx(1.0)


def test_the_tokens_into_each_flower_state():
    """Forward: the hub (s0) is entered on a face (8-19) and state k + 1 on die k; backward: s0 on a die."""
    forward = A.entry_tokens(TR.settings_for("2", **FLOWER))
    assert forward[0] == list(range(8, 20)) and forward[1:] == [[k] for k in range(8)]
    assert A.entry_tokens(TR.settings_for("2", **FLOWER_BACKWARD))[0] == list(range(8))


def test_measures_in_batches_match_one_pass():
    """analysis.measures_in_batches gives a model's measures as one pass over all windows does (they are means over
    windows), with an uneven last batch."""
    for name, changes in (("5b", FLOWER), ("C", {}), ("B", {})):
        s = TR.settings_for(name, **changes)
        model = TR.make_model(s).eval()
        x = s.process_object().sampler(7).draw(1100, s.total_len)
        prediction = model.predict(x, s.tau_end)
        whole = model.measures(x, s.tau_end, prediction)
        A.measures_in_batches(model, rows=256)
        parts = model.measures(x, s.tau_end, prediction)
        assert whole.keys() == parts.keys()
        for k in whole:
            assert parts[k] == pytest.approx(whole[k], abs=1e-6), (name, k)


def test_find_runs_reads_the_settings_of_saved_runs(tmp_path):
    for seed in (0, 1):
        s = tiny("2", seed=seed)
        TR.train(s, folder=tmp_path / "model_2" / f"run_s{seed}", verbose=False)
    found = TR.find_runs(tmp_path, model="2", seed=1)
    assert [f.name for f in found] == ["run_s1"] and next(iter(found.values())).seed == 1
    assert len(TR.find_runs(tmp_path, model="2")) == 2 and not TR.find_runs(tmp_path, model="5b")


def test_the_excess_at_each_true_state_averages_to_the_test_excess(tmp_path):
    """Weighted by how often each true state occurs, the excess CE at each true state is the test excess CE."""
    s = tiny("B")
    _, folder = TR.train(s, folder=tmp_path / "B", verbose=False)
    table, metrics = A.excess_by_true_state(folder), A.report(folder)["metrics"]
    for name in ("transformer", "gru_from_handoff"):
        mean = np.nansum(table[name] * table["visits"]) / table["visits"].sum()
        assert mean == pytest.approx(metrics[f"excess_{name}_future"], abs=1e-6), name


def test_merge_cost_is_zero_without_merges_and_remembers_nothing_with_all():
    """No merges cost nothing; merging all states of renewal-15 is predicting without memory: 'remembers nothing'
    (0.0549 on the test windows, section 16); merging two flower states costs pi_i KL(p_i||m) + pi_j KL(p_j||m)."""
    s = TR.settings_for("2")
    assert A.merge_cost(s, [[c] for c in range(15)]) == 0.0
    assert A.merge_cost(s, [list(range(15))]) == pytest.approx(0.0549, abs=2e-3)
    f = TR.settings_for("2", **FLOWER)
    proc = f.process_object()
    p = proc.emission()
    m = (p[3] + p[4]) / 2                                         # two dice states, each pi = 1/16
    kl = lambda q: float(np.sum(q[q > 0] * np.log2(q[q > 0] / m[q > 0])))
    assert A.merge_cost(f, [[3, 4]]) == pytest.approx((kl(p[3]) + kl(p[4])) / 16, abs=1e-12)
