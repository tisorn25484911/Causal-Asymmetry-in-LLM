"""
The notebook's post-hoc merge (extraction.merge_equivalent_states), on machines built by hand.
checks/notebook_merge_same.py holds the port to the notebook's own code on saved runs.

Run:  pytest tests/ -q          (from model_modeularised/)
"""
import numpy as np
import pytest

from extraction import add_merged_metrics, merge_equivalent_states, theory_from_generator
from process_generator import generator

F = (0.1, 0.2, 0.3, 0.25, 0.15)


def renewal_machine(F=F):
    """The renewal epsilon-machine as {next_state, emission_probs}, states in count order."""
    F = np.asarray(F) / np.sum(F)
    k = len(F)
    hazard = F / np.cumsum(F[::-1])[::-1]
    return {"next_state": np.array([[min(c + 1, k - 1), 0] for c in range(k)]),
            "emission_probs": np.stack([1 - hazard, hazard], axis=1)}


def with_copy_of_state_2(count_of_copy):
    """The renewal machine plus a sixth state that duplicates count 2 (same row, same successors)."""
    m = renewal_machine()
    nxt = np.vstack([m["next_state"], m["next_state"][2]])
    E = np.vstack([m["emission_probs"], m["emission_probs"][2]])
    nxt[1, 0] = 5                                    # count 1 now continues into the copy
    counts = np.array([100.0, 90.0, 40.0, 30.0, 10.0, count_of_copy])
    return {"next_state": nxt, "emission_probs": E}, counts


def test_an_epsilon_machine_is_left_alone():
    m = renewal_machine()
    merged, groups, log = merge_equivalent_states(m, np.full(5, 100.0))
    assert log == [] and groups == [[0], [1], [2], [3], [4]]
    assert np.array_equal(merged["next_state"], m["next_state"])


def test_a_duplicated_state_is_merged_back():
    machine, counts = with_copy_of_state_2(50.0)
    merged, groups, log = merge_equivalent_states(machine, counts)
    assert len(log) == 1 and log[0][:2] == (2, 5) and log[0][2] == pytest.approx(0.0, abs=1e-12)
    assert groups[2] == [2, 5] and merged["next_state"].shape[0] == 5
    np.testing.assert_allclose(merged["emission_probs"], renewal_machine()["emission_probs"], atol=1e-12)
    assert merged["next_state"][1, 0] == 2           # count 1 -> the merged count-2 state
    assert merged["counts"][2] == 90.0


def test_states_seen_fewer_than_min_count_times_are_not_merged():
    machine, counts = with_copy_of_state_2(10.0)     # below the notebook's 35
    _, groups, log = merge_equivalent_states(machine, counts)
    assert log == [] and len(groups) == 6


@pytest.mark.parametrize("dp, merges", [(0.01, 1), (0.1, 0)])
def test_the_threshold_is_a_jensen_shannon_divergence_in_nats(dp, merges):
    """JS(Bern(0.5), Bern(0.5 + dp)) ~ dp^2 / 2 nats: 5e-5 merges, 5e-3 does not (threshold 1e-3)."""
    machine = {"next_state": np.array([[0, 1], [0, 1]]),
               "emission_probs": np.array([[0.5, 0.5], [0.5 + dp, 0.5 - dp]])}
    _, _, log = merge_equivalent_states(machine, np.array([100.0, 100.0]))
    assert len(log) == merges


def test_full_is_judged_on_the_merged_machine_and_the_raw_verdict_is_kept():
    th = theory_from_generator(generator("renewal", 1, 10, {"F": list(F)}))
    machine, counts = with_copy_of_state_2(50.0)
    occ = counts / counts.sum()
    S_raw = float(-(occ * np.log2(occ)).sum())
    res = {"theory": th, "metrics": {"full": False},
           "machine": {**machine, "counts": counts, "state_ids": np.arange(6)}}
    add_merged_metrics(res)
    m = res["metrics"]
    assert m["full_raw"] is False and m["n_merges"] == 1 and m["merged_k"] == 5
    assert m["merged_S_emp"] < S_raw and m["merged_discovered"] == 5
    assert res["merged_machine"]["state_ids"][2] == [2, 5]
    add_merged_metrics(res)                          # idempotent: a second call changes nothing
    assert m["n_merges"] == 1 and m["full_raw"] is False
