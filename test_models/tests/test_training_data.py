"""
training_data.py finds the training windows of any update again, exactly, and its statistics count what they claim.

Run from test_models/:  python -m pytest tests/test_training_data.py -q
"""
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import process as P                                                        # noqa: E402
import training as TR                                                      # noqa: E402
import training_data as TD                                                 # noqa: E402

TINY = dict(steps=6, eval_every=3, history_len=24, rollout_len=8, context=11, batch_size=4, validation_rows=16)
SMALL = [("renewal", {"F": [0.2] * 5}), ("flower", {"n": 3, "m": 5, "dice_seed": 0}),
         ("flower", {"n": 3, "m": 5, "dice_seed": 0, "direction": "backward"})]


def tiny(name, **changes):
    extra = {} if name == "feedback_GRU" else dict(n_layers=1)
    return TR.settings_for(name, **dict(TINY, **extra, **changes))


@pytest.mark.parametrize("process,params", SMALL)
def test_the_skip_gives_the_windows_of_drawing_in_order(process, params):
    s = TR.settings_for("2", process=process, params=params, batch_size=8, history_len=10, rollout_len=4, context=5)
    sampler = s.process_object().sampler(s.train_seed)
    for update in range(1, 16):
        x, st = sampler.draw(s.batch_size, s.total_len, states=True)
        y, sy = TD.windows_at(s, update, states=True)
        assert torch.equal(x, y) and np.array_equal(st, sy), update
    for update, x in TD.windows(s, 4, 9):
        assert torch.equal(x, TD.windows_at(s, update))


def test_updates_are_numbered_from_one_and_a_run_has_no_windows_after_its_last_update():
    s = TR.settings_for("2", process="renewal", params={"F": [0.2] * 5})
    with pytest.raises(ValueError):
        TD.windows_at(s, 0)
    folder = next((f for f in TD.run_folders() if TR.read_settings(f).steps == 300), None)
    if folder is None:
        pytest.skip("no saved 300-update run")
    TD.windows_at(folder, 300)
    with pytest.raises(ValueError):
        TD.windows_at(folder, 301)


def test_saved_runs_kept_the_sampler_state_of_the_skip():
    folders = TD.run_folders()
    if not folders:
        pytest.skip("no saved runs")
    chosen = [folders[0]] + [f for f in folders if (f / "extensions.json").exists()][:2]
    for folder in chosen:
        result = TD.check_run(folder)
        assert result and all(result.values()), (folder, result)


def test_training_again_on_the_found_windows_gives_the_saved_losses(tmp_path, monkeypatch):
    """The 300-update model-2 run made with the current code: 20 updates trained again on windows_at()'s windows give
    its saved training losses exactly; the windows of the next update instead do not."""
    folder = next((f for f in TD.run_folders() if TR.read_settings(f).steps == 300
                   and TR.read_settings(f).model == "2" and TR.run_folder(TR.read_settings(f)).name == f.name), None)
    if folder is None:
        pytest.skip("no 300-update model-2 run made with the current code")
    s = TR.read_settings(folder)
    saved = pd.read_csv(folder / "training_history.csv").iloc[:20].reset_index(drop=True)
    found = [TD.windows_at(s, k) for k in range(1, 23)]
    original = P.Process.sampler

    class Replay:
        def __init__(self, shift):
            self.k, self.shift, self.rng = 0, shift, np.random.default_rng(0)

        def draw(self, batch_size, total_len, states=False):
            self.k += 1
            return found[self.k - 1 + self.shift]

    for shift, same in ((0, True), (1, False)):
        monkeypatch.setattr(P.Process, "sampler",
                            lambda self, seed, shift=shift: Replay(shift) if seed == s.train_seed else original(self, seed))
        again = TR.settings_from_dict(dict(s.to_dict(), steps=20, schedule_steps=300))
        TR.train(again, folder=tmp_path / f"shift{shift}", force=True, verbose=False)
        h = pd.read_csv(tmp_path / f"shift{shift}" / "training_history.csv")
        equal = (h[saved.columns].to_numpy() == saved.to_numpy()).all(axis=1)
        assert equal.all() if same else not equal.any(), (shift, equal)
        monkeypatch.setattr(P.Process, "sampler", original)


@pytest.mark.parametrize("process,params", SMALL + [("renewal", {"F": [1 / 15] * 15}),
                                                    ("flower", {"n": 8, "m": 12, "dice_seed": 0, "direction": "backward"})])
def test_the_fast_exact_ce_equals_training_s(process, params):
    proc = TR.get_process(process, json.dumps(params, sort_keys=True))
    x = proc.sampler(5).draw(64, 40).numpy()
    assert np.abs(TD.ExactCE(proc)(x) - TR.exact_ce_bits(proc, x)).max() < 1e-9


def test_a_renewal_state_is_known_after_a_one_or_after_the_longest_run_of_zeros():
    N = 5
    proc = TR.get_process("renewal", json.dumps({"F": [1 / N] * N}))
    x = proc.sampler(3).draw(500, 20).numpy()
    _, known = TD.ExactCE(proc)(x, known=True)
    ones = np.cumsum(x == 1, axis=1) > 0
    zeros_from_start = np.cumsum(np.cumsum(x == 1, axis=1) == 0, axis=1)
    assert np.array_equal(known, ones | (zeros_from_start >= N - 1))


@pytest.mark.parametrize("process,params", SMALL[:2])
def test_the_chance_two_windows_are_equal_matches_counting_every_word(process, params):
    proc = TR.get_process(process, json.dumps(params, sort_keys=True))
    L, V = 6, proc.vocab_size
    words = np.array(list(itertools.product(range(V), repeat=L)))
    v = np.tile(proc.pi, (len(words), 1))                       # P(word so far, state), impossible words included
    for j in range(L):
        v = np.einsum("ns,nst->nt", v, proc.M[:, :, words[:, j]].transpose(2, 0, 1))
    p = v.sum(axis=1)
    assert abs(p.sum() - 1) < 1e-9
    assert abs(TD.collision_probability(proc, L) - (p ** 2).sum()) < 1e-12


@pytest.mark.parametrize("V", [2, 3, 8, 20, 50, 256])
def test_rows_are_packed_and_counted_exactly(V):
    rng = np.random.default_rng(V)
    base = rng.integers(0, V, size=(30, 17))
    x = base[rng.integers(0, 30, size=400)]
    rows = TD.pack_rows(x, V)
    assert np.array_equal(TD.unpack_rows(rows, V, 17), x)
    counts, first = TD.row_counts(rows)
    expected = collections.Counter(map(tuple, x.tolist()))
    assert sorted(counts.tolist()) == sorted(expected.values())
    for c, i in zip(counts, first):
        assert expected[tuple(x[i].tolist())] == c


@pytest.mark.parametrize("direct_cells", [2 ** 24, 10])
def test_word_counts_match_a_simple_count(direct_cells, monkeypatch):
    monkeypatch.setattr(TD, "DIRECT_CELLS", direct_cells)            # 10: every length is counted by sorting
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, size=(60, 12))
    words = TD.WordCounts(3, [1, 2, 3, 5, 8])
    words.add(x[:25])
    words.add(x[25:])
    for L in (1, 2, 3, 5, 8):
        expected = collections.Counter(tuple(r[j:j + L]) for r in x.tolist() for j in range(12 - L + 1))
        assert words.distinct(L) == len(expected) and words.positions[L] == sum(expected.values())
        ids = TD.word_ids(x, 3, [L])[L].ravel()
        seen = words.counts_of(L, ids)
        as_tuples = [tuple(r[j:j + L]) for r in x.tolist() for j in range(12 - L + 1)]
        assert seen.tolist() == [expected[t] for t in as_tuples]
        assert words.counts_of(L, np.array([3 ** L - 1 + 0]))[0] == expected.get((2,) * L, 0)
    monkeypatch.setattr(TD, "DIRECT_CELLS", 10)                      # only lengths counted by sorting are capped
    capped = TD.WordCounts(3, [1, 5], cap=20)
    capped.add(x)
    assert capped.distinct(1) == 3 and capped.distinct(5) is None and capped.counts_of(5, np.array([0])) is None


KNOWN = sorted(TD.GRU_HYBRIDS | TD.OWN_ONE_STEP | TD.ALONE | TD.DE_BRUIJN | {"1", "4"})


def test_every_model_file_has_its_positions():
    """Every model_<name>.py that defines a Model is known to term_positions (other files named model_*.py, such as
    analysis scripts, are not models)."""
    models = [m for m in TR.available_models() if hasattr(TR.load_model_module(m), "Model")]
    assert set(models) <= set(KNOWN), set(models) - set(KNOWN)


@pytest.mark.parametrize("name", KNOWN)
def test_the_loss_terms_score_the_positions_term_positions_gives(name):
    """Each next-token CE term of a model's loss (without noise) equals the CE of the predictor that predicts the
    tokens loss_positions names: the same first position, the same number of tokens."""
    settings = tiny(name, process="renewal", params={"F": [0.2] * 5})
    torch.manual_seed(0)
    model = TR.make_model(settings)
    model.eval()
    x = settings.process_object().sampler(9).draw(8, settings.total_len)
    with torch.no_grad():
        _, terms, _ = model.loss(x, 0.5, settings.weight)
        predictors = model.predict(x, 0.5)["predictors"]
    weights = TD.loss_positions(settings)
    by_term = dict(ce_gru="gru_from_handoff", ce_t1="transformer", pred="node_table",
                   ce=next(iter(predictors)) if len(predictors) == 1 else None)
    checked = 0
    for term, w in weights.items():
        if term not in by_term:
            continue
        p = predictors[by_term[term]]
        positions = np.flatnonzero(w)
        assert p["first"] == positions[0] and p["logits"].shape[1] == len(positions) and positions[-1] == x.shape[1] - 1
        ce = TR.ce_bits(p["logits"], x[:, p["first"]:p["first"] + p["logits"].shape[1]]).mean()
        assert abs(float(ce) - float(terms[term])) < 1e-4, (term, float(ce), float(terms[term]))
        checked += 1
    assert checked >= 1


def test_the_statistics_of_a_small_stream(tmp_path):
    """Two tiny runs on the same windows: every count of stream_statistics agrees with counting the windows directly."""
    folders = []
    for name in ("2", "feedback_GRU"):
        s = tiny(name, process="renewal", params={"F": [0.2] * 5})
        TR.train(s, folder=tmp_path / name, verbose=False)
        folders.append(tmp_path / name)
    key = TD.stream_key(TR.read_settings(folders[0]))
    assert key == TD.stream_key(TR.read_settings(folders[1]))
    summary = TD.stream_statistics(key, folders, out=tmp_path / "stats", verbose=False)
    info, arrays = TD.load_statistics(summary["label"], tmp_path / "stats")
    s = TR.read_settings(folders[0])
    proc = s.process_object()
    x, st = zip(*[TD.windows_at(s, k, states=True) for k in range(1, s.steps + 1)])
    x, st = torch.cat(x).numpy(), np.concatenate(st)
    # true states at every position
    counts = arrays["state_counts"].sum(axis=0)
    for j in range(s.total_len + 1):
        assert np.array_equal(counts[j], np.bincount(st[:, j], minlength=len(proc.pi)))
    # the exact CE of each term's tokens, update by update
    ce = TR.exact_ce_bits(proc, x).reshape(s.steps, s.batch_size, -1).mean(axis=1)
    for i, name in enumerate(info["exact_terms"]):
        assert np.allclose(arrays["exact_terms"][:, i], ce @ arrays[f"weights_{name}"], atol=1e-6)
    assert set(info["exact_terms"]) == {f"ce_gru_from{s.history_len}", f"ce_t1_from{s.context + 1}",
                                        f"ce_from{s.context + 1}"}
    # repeats of whole windows
    c = collections.Counter(map(tuple, x.tolist()))
    assert info["windows"]["different"] == len(c) and info["windows"]["max_count"] == max(c.values())
    assert info["windows"]["equal_pairs"] == sum(v * (v - 1) / 2 for v in c.values())
    # rare states at the handoff
    rare = np.flatnonzero(np.isclose(proc.pi, proc.pi.min()))
    handoff = np.isin(st[:, s.history_len], rare).reshape(s.steps, s.batch_size).sum(axis=1)
    assert np.array_equal(arrays[f"rare_handoff_H{s.history_len}"], handoff)
    # coverage of the validation windows' words of 3 tokens
    held = TD._eval_windows(s, "validation")
    train3 = collections.Counter(tuple(r[j:j + 3]) for r in x.tolist() for j in range(s.total_len - 2))
    held3 = [tuple(r[j:j + 3]) for r in held.tolist() for j in range(s.total_len - 2)]
    row = next(r for r in info["coverage"] if r["kind"] == "validation" and r["L"] == 3)
    assert abs(row["seen"] - np.mean([train3[w] > 0 for w in held3])) < 1e-12
