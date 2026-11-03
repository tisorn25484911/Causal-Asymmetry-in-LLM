"""
long_windows.py trains exactly as training.train does, records every update's losses and gradient lengths correctly,
resumes an interrupted run to exactly the same end, and plans the runs the user chose.

Run from test_models/:  python -m pytest tests/test_long_windows.py -q
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import long_windows as LW                                                  # noqa: E402
import training as TR                                                      # noqa: E402

TINY = dict(process="renewal", params={"F": [0.2] * 5}, steps=12, eval_every=3, history_len=24, rollout_len=8,
            context=11, batch_size=4, validation_rows=16, n_layers=1)


def tiny(name, **changes):
    return TR.settings_for(name, **dict(TINY, **changes))


def same_state(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def saved(folder, name):
    return torch.load(Path(folder) / name, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("name", LW.MODELS)
def test_training_is_exactly_training_train_s(name, tmp_path):
    s = tiny(name)
    TR.train(s, folder=tmp_path / "train", verbose=False)
    LW.train(s, folder=tmp_path / "long", save_every=0, verbose=False)
    a, b = saved(tmp_path / "train", "completed.pt"), saved(tmp_path / "long", "completed.pt")
    assert same_state(a["model_state"], b["model_state"])
    assert a["rng_state"] == b["rng_state"] and torch.equal(a["torch_rng_state"], b["torch_rng_state"])
    assert (a["best_update"], a["best_value"]) == (b["best_update"], b["best_value"])
    ha, hb = (pd.read_csv(tmp_path / k / "training_history.csv") for k in ("train", "long"))
    assert hb[ha.columns].equals(ha)                                    # every loss term, every update
    va, vb = (pd.read_csv(tmp_path / k / "validation_history.csv").drop(columns="seconds") for k in ("train", "long"))
    assert va.equals(vb[va.columns])
    assert same_state(saved(tmp_path / "train", "best.pt")["model_state"],
                      saved(tmp_path / "long", "best.pt")["model_state"])


@pytest.mark.parametrize("name", ["2", "5b"])
def test_a_stopped_run_resumes_to_exactly_the_same_end(name, tmp_path, monkeypatch):
    s = tiny(name)
    LW.train(s, folder=tmp_path / "straight", save_every=5, verbose=False)
    original, calls = TR.train_step, {"n": 0}

    def stop_at_eight(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 8:
            raise KeyboardInterrupt("stopped")
        return original(*args, **kwargs)

    monkeypatch.setattr(TR, "train_step", stop_at_eight)
    with pytest.raises(KeyboardInterrupt):
        LW.train(s, folder=tmp_path / "stopped", save_every=5, verbose=False)
    monkeypatch.setattr(TR, "train_step", original)
    assert saved(tmp_path / "stopped", "resume.pt")["update"] == 5 and not (tmp_path / "stopped/completed.pt").exists()
    LW.train(s, folder=tmp_path / "stopped", save_every=5, verbose=False)
    assert not (tmp_path / "stopped/resume.pt").exists()
    a, b = saved(tmp_path / "straight", "completed.pt"), saved(tmp_path / "stopped", "completed.pt")
    assert same_state(a["model_state"], b["model_state"]) and a["rng_state"] == b["rng_state"]
    assert same_state(a["optimizer_state"]["state"][0], b["optimizer_state"]["state"][0])
    ha, hb = (pd.read_csv(tmp_path / k / "training_history.csv") for k in ("straight", "stopped"))
    assert ha.equals(hb)
    va, vb = (pd.read_csv(tmp_path / k / "validation_history.csv").drop(columns="seconds")
              for k in ("straight", "stopped"))
    assert va.equals(vb)
    assert same_state(saved(tmp_path / "straight", "best.pt")["model_state"],
                      saved(tmp_path / "stopped", "best.pt")["model_state"])


def test_a_run_does_not_resume_with_other_settings(tmp_path, monkeypatch):
    s = tiny("2")
    original, calls = TR.train_step, {"n": 0}

    def stop_at_seven(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 7:
            raise KeyboardInterrupt("stopped")
        return original(*args, **kwargs)

    monkeypatch.setattr(TR, "train_step", stop_at_seven)
    with pytest.raises(KeyboardInterrupt):
        LW.train(s, folder=tmp_path / "run", save_every=5, verbose=False)
    monkeypatch.setattr(TR, "train_step", original)
    with pytest.raises(ValueError):
        LW.train(tiny("2", lr=2e-3), folder=tmp_path / "run", save_every=5, verbose=False)


def test_every_update_records_the_loss_terms_and_gradient_lengths(tmp_path):
    s = tiny("1")
    LW.train(s, folder=tmp_path / "run", save_every=0, verbose=False)
    h = pd.read_csv(tmp_path / "run" / "training_history.csv")
    assert len(h) == s.steps and h["update"].tolist() == list(range(1, s.steps + 1))
    for col in ("ce_gru", "loss", "ce_t1_measured", "grad_encoder", "grad_gru_cell", "grad_transformer_part",
                "grad_gru_part", "grad_shared_part", "grad_total"):
        assert col in h and h[col].notna().all() and (h[col] > 0).all(), col
    parts = np.sqrt(h["grad_encoder"] ** 2 + h["grad_encoder_norm"] ** 2 + h["grad_transformer_selector"] ** 2)
    assert np.allclose(parts, h["grad_transformer_part"], rtol=1e-12)
    every = np.sqrt(sum(h[c] ** 2 for c in h if c.startswith("grad_") and not c.endswith(("_part", "_total"))))
    assert np.allclose(every, h["grad_total"], rtol=1e-12)


def test_the_measured_transformer_ce_is_the_one_a_trained_term_would_be():
    """Model 2's loss has the Transformer's own CE (ce_t1); measured the long_windows way it is the same number."""
    s = tiny("2")
    model = LW.instrument(TR.make_model(s), always=True)
    optimizer = TR.make_optimizer(model, s)
    source = s.process_object().sampler(3)
    for _ in range(4):
        out = TR.train_step(model, source.draw(s.batch_size, s.total_len), optimizer, 1.0, s.weight, s.grad_clip)
        assert out["ce_t1_measured"] == out["ce_t1"]
    for name, has in (("1", True), ("B", True), ("5b", False), ("C", False), ("5a", False)):
        m = LW.instrument(TR.make_model(tiny(name)))
        _, terms, _ = m.loss(source.draw(4, s.total_len), 1.0, tiny(name).weight)
        assert ("ce_t1_measured" in terms) == has, name


@pytest.mark.parametrize("name", LW.MODELS)
def test_the_parts_hold_every_parameter_and_grad_total_is_the_clipped_length(name):
    s = tiny(name)
    model = TR.make_model(s)
    children = dict(model.named_children())
    grouped = {n for names in LW.PARTS.values() for n in names}
    assert set(children) <= grouped, set(children) - grouped
    assert sum(p.numel() for c in children.values() for p in c.parameters()) == sum(p.numel() for p in model.parameters())
    loss, _, _ = model.loss(s.process_object().sampler(4).draw(s.batch_size, s.total_len), 1.0, s.weight)
    loss.backward()
    grads = {f"grad_{n}": TR.grad_norm(c.parameters()) for n, c in children.items()}
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    assert math.isclose(LW.part_lengths(grads)["grad_total"], float(total), rel_tol=1e-5)


def test_the_plan_is_the_user_s_choice(tmp_path):
    assert [(k, v["n_states"], v["context"], v["history_len"], v["rollout_len"]) for k, v in LW.PROCESSES.items()] == \
        [("renewal-30", 64, 29, 300, 180), ("renewal-60", 128, 59, 300, 180), ("renewal-90", 192, 89, 300, 180)]
    for k, v in LW.PROCESSES.items():
        n = int(k.split("-")[1])
        assert v["params"]["F"] == [1.0 / n] * n
    runs = LW.jobs()
    assert len(runs) == 54 and len({(j["model"], j["process"], j["seed"]) for j in runs}) == 54
    assert [j["seed"] for j in runs[:18]] == [0] * 18
    for j in runs:
        s = LW.settings_of(j["model"], j["process"], j["seed"])
        assert (s.total_len, s.steps, s.batch_size, s.seed) == (480, LW.STEPS, 128, j["seed"])
        assert s.weight == TR.load_model_module(j["model"]).DEFAULTS.get("weight", 0.0)
    plan = LW.write_plan(tmp_path / "plan.json")
    again = json.loads((tmp_path / "plan.json").read_text())
    assert plan == again and len(again) == 54
    for entry in again:
        assert TR.settings_for(entry["model"], **entry["settings"]) == LW.settings_of(entry["model"], entry["process"],
                                                                                     entry["seed"])
    assert len(LW.jobs(models=["1"], seeds=[0])) == 3 and len(LW.jobs(processes=["renewal-90"])) == 18


def test_status_and_commands_of_a_fresh_runs_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(TR, "RUNS", tmp_path)
    table = LW.status_table()
    assert len(table) == 54 and set(table["state"]) == {"not started"}
    lines = LW.commands()
    assert len(lines) == 54 and all("long_windows.py train --model" in line for line in lines)
    assert len(LW.commands(seeds=[0], processes=["renewal-30"])) == 6


def test_the_figure_of_the_parts(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    runs = {}
    for name in ("1", "5b"):
        for seed in (0, 1):
            folder = tmp_path / f"{name}_{seed}"
            LW.train(tiny(name, seed=seed), folder=folder, save_every=0, verbose=False)
            runs[f"{name} s{seed}"] = folder
    fig = LW.plot_parts(runs, title="tiny", smooth=3)
    assert len(fig.axes) == 4
    table = LW.term_table(runs, last=5)
    assert set(table.index) == set(runs) and "grad_transformer_part" in table


def test_the_reports_of_a_finished_run_are_cached(tmp_path):
    folder = tmp_path / "run"
    LW.train(tiny("2"), folder=folder, save_every=0, verbose=False)
    LW.report_one(folder)
    assert (folder / "report_latest.json").exists() and list(folder.glob("machine_latest_*.json"))
    rep = json.loads((folder / "report_latest.json").read_text())
    assert rep["update"] == TINY["steps"] and "excess_gru_from_handoff_future" in rep["metrics"]
