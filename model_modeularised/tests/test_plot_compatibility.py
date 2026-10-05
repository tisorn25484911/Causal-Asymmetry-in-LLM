"""
Plots for both saved-result schemas (SHARED_TRAJECTORY_EMBEDDING_WARMUP_PLAN.md section 12.10-12.11):
a legacy fixture and a continuous (schema-2) fixture whose transformer and GRU read different
lengths, both made by real, tiny runs and redrawn from their pickles alone.

Run:  pytest tests/ -q          (from model_modeularised/)
"""
import copy
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402
import pytest                                                          # noqa: E402

import plotting                                                        # noqa: E402
import run_model                                                       # noqa: E402
from training import load_pkl                                          # noqa: E402

TINY = dict(trans_total_run=200, progress_every=0)
LEGACY = replace(run_model.RunConfig(), max_epochs=2, max_epochs_transformer=2, num_samples=64, fresh_samples=16,
                 **TINY)
CONTINUOUS = replace(run_model.RunConfig(), data_mode="continuous", total_tokens=20_000, max_steps=40, **TINY)


def _make(root, name, cfg, process, archs, arms):
    out = os.path.join(root, name)
    run_dir, fig_dir = os.path.join(out, "runs"), os.path.join(out, "figures")
    os.makedirs(run_dir)
    for arch in archs:
        for arm in arms:
            res = run_model.run_arm(cfg, arch, process, arm, verbose=False, data_dir=os.path.join(out, "data"))
            run_model.save_run(res, run_dir, run_model.run_stem(cfg, arch, process, arm))
    paths = run_model.redraw(run_dir, fig_dir)
    results = {(r["arch"], r["arm"]): r for r in
               (load_pkl(os.path.join(run_dir, f)) for f in sorted(os.listdir(run_dir)) if f.endswith(".pkl"))}
    return {"paths": paths, "fig_dir": fig_dir, "results": results, "process": process}


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("plots"))
    return {"legacy": _make(root, "legacy", LEGACY, "coin", ["transformer", "gru_feedback"], ["forward"]),
            "continuous": _make(root, "continuous", CONTINUOUS, "renewal", ["transformer", "gru"],
                                ["forward", "backward"])}


@pytest.mark.parametrize("mode", ["legacy", "continuous"])
def test_the_expected_figure_set_without_collisions(saved, mode):
    s = saved[mode]
    paths = s["paths"]
    assert len(paths) == len(set(paths)) and all(os.path.exists(p) for p in paths)
    rel = {os.path.relpath(p, s["fig_dir"]) for p in paths}
    want = {"scorecard_s0.png", f"{s['process']}/theory.png", f"{s['process']}/theory_comparison_s0.png"}
    for arch, arm in s["results"]:
        want |= {f"{s['process']}/{arch}/{arm}_s0_{k}.png" for k in ("training", "states", "machine")}
        want |= {f"{s['process']}/{arch}/s0_{k}.png" for k in ("per_position", "transitions", "complexity")}
        if arch == "gru_feedback":
            want.add(f"{s['process']}/{arch}/s0_transitions_exact.png")
    assert rel == want


def test_architectures_with_different_views_stay_one_comparison(saved):
    res = saved["continuous"]["results"]
    t, g = res[("transformer", "forward")], res[("gru", "forward")]
    assert t["view"]["input_len"] == 300 and g["view"]["input_len"] == 29
    assert run_model.experiment_key(t) == run_model.experiment_key(g)          # never split by horizon
    assert t["metrics"]["common_manifest_id"] == g["metrics"]["common_manifest_id"]
    runs_by_arch = {"transformer": {a: res[("transformer", a)] for a in ("forward", "backward")},
                    "gru": {a: res[("gru", a)] for a in ("forward", "backward")}}
    from theory import process_theory
    params, burn_in, _ = run_model.theory_inputs(t)
    fig = plotting.plot_theory_comparison(runs_by_arch, process_theory("renewal", params, burn_in, 300), t["tag"])
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert {"Transformer", "GRU"} <= set(labels)                              # no architecture omitted
    assert "common profile" in fig.axes[3].get_title()                         # matched comparison, labelled
    plt.close(fig)


def _line(ax, label):
    return next(ln for ln in ax.get_lines() if ln.get_label() == label)


@pytest.mark.parametrize("arch, n, start", [("transformer", 300, 20), ("gru", 29, 19)])
def test_per_position_lengths_and_the_shaded_prefix(saved, arch, n, start):
    r = saved["continuous"]["results"][(arch, "forward")]
    fig = plotting.plot_per_position({"forward": r})
    ax = fig.axes[0]
    line = _line(ax, plotting.ARCH_LABEL[arch])
    assert len(line.get_xdata()) == n and np.allclose(line.get_ydata(), r["per_position"]["neural"])
    mask = np.asarray(r["per_position"]["score_mask"])
    assert mask.sum() == n - start and not mask[:start].any()
    handles, labels = ax.get_legend_handles_labels()
    span = handles[labels.index("context only (excluded from primary CE)")]
    x = span.get_xy()[:, 0] if hasattr(span, "get_xy") and np.ndim(span.get_xy()) == 2 else \
        np.array([span.get_x(), span.get_x() + span.get_width()])
    assert x.min() == pytest.approx(1.0) and x.max() == pytest.approx(start + 0.5)   # context 1..start shaded
    plt.close(fig)


def test_legacy_per_position_has_no_prefix(saved):
    r = saved["legacy"]["results"][("transformer", "forward")]
    fig = plotting.plot_per_position({"forward": r})
    assert "context only (excluded from primary CE)" not in fig.axes[0].get_legend_handles_labels()[1]
    assert "score_mask" not in r["per_position"]
    plt.close(fig)


def test_captions_say_which_comparison_they_draw(saved):
    cont = list(saved["continuous"]["results"].values())
    leg = list(saved["legacy"]["results"].values())
    fig = plotting.plot_scorecard(cont)
    title = fig._suptitle.get_text()
    assert "views per row" in title and "input map frozen" in title and "usage scope scored" in title
    assert "common profile" in fig.axes[2].get_title()
    plt.close(fig)
    fig = plotting.plot_scorecard(leg)
    assert "64 x 300 tokens" in fig._suptitle.get_text() and "same tokens" in fig.axes[2].get_title()
    plt.close(fig)
    assert plotting.view_label(cont[0]).startswith("input ") and plotting.view_label(leg[0]).endswith("(legacy)")


@pytest.mark.parametrize("mode", ["legacy", "continuous"])
def test_titles_and_table_headers_fit(saved, mode):
    """Inspected, not just written: every title inside the figure, every header inside its cell."""
    results = list(saved[mode]["results"].values())
    for fig in (plotting.plot_scorecard(results), plotting.plot_training(results[0])):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width = fig.bbox.width
        texts = [fig._suptitle] if fig._suptitle is not None else []
        texts += [ax.title for ax in fig.axes if ax.get_title()]
        for t in texts:
            box = t.get_window_extent(renderer)
            assert box.x0 >= -1 and box.x1 <= width + 1, t.get_text()[:60]
        for ax in fig.axes:
            for tbl in ax.tables:
                for (row, col), cell in tbl.get_celld().items():
                    if row == 0:
                        assert cell.get_text().get_window_extent(renderer).width <= \
                               cell.get_window_extent(renderer).width + 1, cell.get_text().get_text()
        plt.close(fig)


def test_unavailable_symbolic_ce_is_never_drawn_as_zero(saved):
    r = copy.deepcopy(saved["continuous"]["results"][("gru", "forward")])
    r["metrics"].update(symbolic_ce=float("nan"), symbolic_n_span=10, symbolic_n_covered=4)
    assert plotting._symbolic(r["metrics"]) == "unavailable (coverage 4/10)"
    fig = plotting.plot_machine_graph(r)
    assert "unavailable (coverage 4/10)" in fig.axes[0].get_title()
    plt.close(fig)


@pytest.mark.parametrize("mode", ["legacy", "continuous"])
def test_a_replot_reproduces_the_saved_numbers(saved, mode):
    r = next(iter(saved[mode]["results"].values()))
    figs = [plotting.plot_training(r), plotting.plot_training(copy.deepcopy(r))]
    a, b = (_line(f.axes[0], "validation CE") for f in figs)
    np.testing.assert_array_equal(a.get_xdata(), b.get_xdata())
    np.testing.assert_array_equal(a.get_ydata(), b.get_ydata())
    np.testing.assert_allclose(a.get_ydata(), r["history"]["val_loss"])
    for f in figs:
        plt.close(f)


def test_each_arm_is_scored_against_its_own_theory_and_rows(saved):
    """Gate 2: the backward arm reads the reversed rows and is compared with the reverse machine."""
    res = saved["continuous"]["results"]
    for arch in ("transformer", "gru"):
        fw, bw = res[(arch, "forward")], res[(arch, "backward")]
        assert fw["theory"]["view"] == "forward" and bw["theory"]["view"] == "backward"
        assert fw["data_manifest"]["raw_sha1"] == bw["data_manifest"]["raw_sha1"]   # one draw, both arms
        np.testing.assert_array_equal(fw["data_manifest"]["test"]["window_starts"],
                                      bw["data_manifest"]["test"]["window_starts"])  # the same spans


def test_a_multi_process_scorecard_still_compares_each_cell_on_the_common_profile(saved):
    """Each process has its own data, hence its own profile id; within a (process, arm) cell the
    architectures share one, so the scorecard compares them on it.  A cell whose ids differ does not."""
    runs = list(saved["continuous"]["results"].values())
    other = copy.deepcopy(runs)
    for r in other:                                   # a second process: other tag, other data
        r["tag"], r["metrics"]["common_manifest_id"] = "other(1)", "a-different-id"
    assert plotting._ce_gap_panel(runs + other)[0] == "common_gap_exact"
    other[0]["metrics"]["common_manifest_id"] = "mismatch"
    assert plotting._ce_gap_panel(runs + other)[0] == "gap_exact"
