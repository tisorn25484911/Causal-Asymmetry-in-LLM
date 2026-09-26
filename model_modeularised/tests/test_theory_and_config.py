"""
The run configuration and the theory layer: any registered process can be chosen,
its parameters parsed, and its closed forms computed from HMM_processes.

Run:  pytest tests/ -q          (from model_modeularised/)
"""
import numpy as np
import pytest
from torch.utils.data import DataLoader

from extraction import causal_state_report, theory_from_generator
from HMM_processes import PROCESS, isomorphic
from process_generator import SequenceDataset, generator, split_loader
from run_model import DEFAULT_PARAMS, RunConfig, draw, parse_params, process_params
from theory import process_theory
from training import train_model


def test_every_registered_process_is_covered():
    """Every process has defaults, except the two defined BY their parameters."""
    assert set(PROCESS) - set(DEFAULT_PARAMS) - {"coin", "flower"} == {"edges", "tilt"}
    with pytest.raises(ValueError):
        process_params(RunConfig(), "edges")
    with pytest.raises(ValueError):
        process_params(RunConfig(), "not_a_process")


@pytest.mark.parametrize("process", ["coin", "flower"] + sorted(DEFAULT_PARAMS))
def test_defaults_build_and_draw(process):
    cfg = RunConfig(num_samples=3, seq_len=20)
    params, burn_in, tag = process_params(cfg, process)
    assert tag.startswith(process)
    gen, seqs = draw(cfg, process, "forward", 3, 20, 0)
    assert seqs.shape == (3, 21) and seqs.max() < gen.vocab_size
    _, back = draw(cfg, process, "backward", 3, 20, 0)
    assert np.array_equal(back, seqs[:, ::-1])


def test_params_override_and_parse():
    over = parse_params(["p=0.3", "even.p=0.25", "probs=[0.5,0.5]", "burn_in=7"], ["even", "gm"])
    assert over == {"even": {"p": 0.25, "probs": [0.5, 0.5], "burn_in": 7},
                    "gm": {"p": 0.3, "probs": [0.5, 0.5], "burn_in": 7}}
    cfg = RunConfig(process_params={"coin": {"p": 0.6}, "even": {"p": 0.25, "burn_in": 7}})
    assert process_params(cfg, "coin")[0] == {"p": 0.6, "q": 0.8}
    params, burn_in, tag = process_params(cfg, "even")
    assert params == {"p": 0.25} and burn_in == 7 and tag == "even(p=0.25)"


def test_geomhold_schedule():
    """study5's geomhold: geometric A -> B over the first DESCENT fraction of the run, then held at B."""
    from tau_schedue import is_scheduled, parse_tau
    _, f = parse_tau("geomhold:5:0.5")                       # DESCENT defaults to 0.5
    assert f(0.0) == pytest.approx(5.0)
    assert f(0.25) == pytest.approx(5 * 0.1 ** 0.5)          # halfway through the descent
    assert f(0.5) == pytest.approx(0.5) and f(0.9) == pytest.approx(0.5) and f(1.0) == pytest.approx(0.5)
    _, g = parse_tau("geomhold:5:0.5:0.25")
    assert g(0.125) == pytest.approx(5 * 0.1 ** 0.5) and g(0.25) == pytest.approx(0.5)
    _, full = parse_tau("geomhold:5:0.5:1")                  # DESCENT = 1 is the plain geom schedule
    _, geom = parse_tau("geom:5:0.5")
    assert all(full(x) == pytest.approx(geom(x)) for x in np.linspace(0, 1, 11))
    assert is_scheduled("geomhold:5:0.5") and not is_scheduled("geomhold:1:1")
    for bad in ("geomhold:5:0.5:0", "geomhold:5:0.5:1.5", "geomhold:0:0.5"):
        with pytest.raises(ValueError):
            parse_tau(bad)


def test_train_model_applies_geomhold():
    gen = generator("coin", 32, 40, {"p": 0.7, "q": 0.8}, burn_in=10)
    ds = SequenceDataset(gen.generate(np.random.default_rng(0)))
    tr, te = split_loader(ds, 8, 0.25, seed=0)                  # 3 batches x 4 epochs = 12 steps
    rec = train_model(tr, "gru_feedback", "discrete", val_loader=te, token_size=3, d_model=8, max_len=40,
                      max_epochs=4, lr=1e-3, n_layers=1, weight_decay=0.01, accelerator="cpu",
                      val_every_n_steps=3, n_states=6, state_dim=None, tau="geomhold:5:0.5:0.5",
                      usage_beta=1e-3, restore_best=True, gumbel=True)
    tau = np.asarray(rec.step_tau)
    assert tau[0] == pytest.approx(5.0)
    assert np.all(np.diff(tau[: len(tau) // 2]) < 0)         # descending in the first half
    assert np.allclose(tau[len(tau) // 2 + 1:], 0.5)          # held at 0.5 in the second


def test_the_gru_trains_with_gumbel_and_the_transformer_does_not():
    cfg = RunConfig()
    assert cfg.gumbel_gru is True and cfg.gumbel_transformer is False


def test_per_architecture_settings():
    """The standard (2026-09-25): transformer geomhold:5:0.5:0.8 for 1000 epochs, the GRUs
    geom:5:0.5 for 800; everything a run uses comes from run_model.arch_settings."""
    from run_model import arch_settings
    cfg = RunConfig()
    t, g, f = (arch_settings(cfg, a, 3) for a in ("transformer", "gru", "gru_feedback"))
    assert (t["tau"], t["max_epochs"], t["gumbel"], t["n_layers"]) == ("geomhold:5:0.5:0.8", 1000, False, 4)
    assert (g["tau"], g["max_epochs"], g["gumbel"], g["n_layers"]) == ("geom:5:0.5", 800, True, 1)
    assert (f["tau"], f["max_epochs"], f["state_dim"]) == ("geom:5:0.5", 800, cfg.d_model)
    assert t["state_dim"] == 3 and t["accelerator"] == "auto" and g["accelerator"] == "cpu"


@pytest.mark.parametrize("argv, want", [
    ("", ("geomhold:5:0.5:0.8", 1000, "geom:5:0.5", 800)),
    ("--tau const:1 --max-epochs 50", ("const:1", 50, "const:1", 50)),
    ("--tau-transformer geom:5:0.5 --max-epochs-transformer 700", ("geom:5:0.5", 700, "geom:5:0.5", 800)),
])
def test_command_line_schedule_flags(argv, want):
    import shlex
    import unittest.mock as um
    import run_model

    class Parsed(Exception):
        pass

    def fake(cfg, arch, process, arm):
        t, g = run_model.arch_settings(cfg, "transformer", 3), run_model.arch_settings(cfg, "gru", 3)
        raise Parsed((t["tau"], t["max_epochs"], g["tau"], g["max_epochs"]))

    with um.patch.object(run_model, "run_arm", fake), pytest.raises(Parsed) as e:
        run_model.main(shlex.split(argv) + ["--no-plots", "--name", "flagtest", "--out", "/tmp/claude-flagtest"])
    assert e.value.args[0] == want


def test_process_theory_coin_closed_forms():
    t = process_theory("coin", {"p": 0.7, "q": 0.8}, 200, 50)
    assert t["C_plus"] == pytest.approx(0.99679, abs=1e-4)
    assert t["C_minus"] == pytest.approx(1.46681, abs=1e-4)
    assert t["entropy_rate"] == pytest.approx(0.80692, abs=1e-4)
    assert (t["forward"]["true_k"], t["backward"]["true_k"]) == (2, 3)
    assert t["warnings"] == []


@pytest.mark.parametrize("process", ["coin", "flower", "rip", "even", "explosive"])
def test_entropy_convergence_sums_to_the_excess_entropy(process):
    """sum_L (h(L) - h) = E (Crutchfield & Feldman 2003): the floor and E are consistent."""
    params, burn_in, _ = process_params(RunConfig(), process)
    t = process_theory(process, params, burn_in, 300)
    assert t["floor_excess"] == pytest.approx(t["excess_entropy"], abs=1e-3)
    assert np.all(np.diff(t["h_L"]) <= 1e-9)                    # h(L) never increases
    assert t["h_L"][-1] == pytest.approx(t["entropy_rate"], abs=1e-3)


# ══════════════════════════════════════════════════════════════════════════
# the renewal process (Marzen & Crutchfield 2015): a time-symmetric control
# ══════════════════════════════════════════════════════════════════════════
RENEWAL_FS = [[0.1, 0.2, 0.3, 0.25, 0.15], [0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25], [0.5, 0.0, 0.5]]


def _renewal_as_edges(F):
    """The same process built independently, as an edge list for `edges`."""
    F = np.asarray(F, float) / np.sum(F)
    edges = []
    for s in range(len(F)):
        hazard = F[s] / F[s:].sum()
        edges.append([s, 1, 0, float(hazard)])
        if s < len(F) - 1:
            edges.append([s, 0, s + 1, float(1 - hazard)])
    return {"n_states": len(F), "vocab": 2, "edges": edges}


@pytest.mark.parametrize("F", RENEWAL_FS)
def test_renewal_is_time_symmetric_and_matches_its_edge_list(F):
    gen = generator("renewal", 1, 10, {"F": F})
    forward = (gen.transition, gen.state_map)
    gen.reverse()
    assert isomorphic(*forward, gen.transition, gen.state_map) is not None       # M- is M+
    t = process_theory("renewal", {"F": F}, 100, 50)
    assert t["C_plus"] == pytest.approx(t["C_minus"], abs=1e-9)
    ref = process_theory("edges", _renewal_as_edges(F), 100, 50)
    for key in ("C_plus", "C_minus", "entropy_rate", "excess_entropy"):
        assert t[key] == pytest.approx(ref[key], abs=1e-9), key


def test_renewal_inter_event_counts_follow_F():
    F = DEFAULT_PARAMS["renewal"]["F"]
    x = generator("renewal", 1, 100_000, {"F": F}).generate(np.random.default_rng(0))[0]
    gaps = np.diff(np.flatnonzero(x == 1)) - 1
    np.testing.assert_allclose(np.bincount(gaps, minlength=len(F)) / len(gaps), F, atol=0.01)


@pytest.mark.parametrize("arch", ["transformer", "gru_feedback", "gru"])
def test_every_model_trains_on_the_renewal_process(arch):
    gen = generator("renewal", 24, 60, DEFAULT_PARAMS["renewal"])
    ds = SequenceDataset(gen.generate(np.random.default_rng(0)))
    tr, te = split_loader(ds, 8, 0.25, seed=0)
    rec = train_model(tr, arch, "discrete", val_loader=te, token_size=2, d_model=8, max_len=60,
                      max_epochs=2, lr=1e-3, n_layers=1, weight_decay=0.01, accelerator="cpu",
                      val_every_n_steps=2, n_states=4, state_dim=None if arch == "gru_feedback" else 2,
                      tau="geom:5:0.5", usage_beta=1e-3, restore_best=True,
                      gumbel=RunConfig().gumbel_transformer if arch == "transformer" else RunConfig().gumbel_gru)
    assert np.all(np.isfinite(rec.step_loss)) and rec.restored_best
    rep = causal_state_report(rec.model, DataLoader(ds, batch_size=8), min_pos=5)
    assert rep["emissions"].shape == (4, 2) and rep["n_states_used"] >= 1
    assert theory_from_generator(gen)["true_k"] == 5


def test_warnings_flag_what_the_comparison_cannot_see():
    nemo = process_theory("nemo", DEFAULT_PARAMS["nemo"], 100, 20)
    assert any("share a next-token distribution" in w for w in nemo["warnings"])
    explosive = process_theory("explosive", {}, 100, 20)
    assert any("infinite" in w for w in explosive["warnings"])
