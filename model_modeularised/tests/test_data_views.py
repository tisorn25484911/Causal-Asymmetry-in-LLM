"""
Samples cut from one trajectory (data_views.py; plan section 8, gates 1, 2 and 16):
the split, the fixed views of both arms, the random shift, and the canonical artifact.
Most tests use a NUMBERED trajectory (token i = i), so every row's start, offset and
direction can be read straight off its tokens.

Run:  pytest tests/ -q          (from model_modeularised/)
"""
import lightning as L
import numpy as np
import pytest
import torch

from data_views import (ShiftSampler, ShiftedTrainRows, WindowSpec, build_views, canonical_trajectory,
                        make_loaders, sha1, split_spans)

N = 300_000
SPANS = split_spans(N)
TRANSFORMER = WindowSpec(raw_window=301, stride=301, batch_size=32, score_start=20, shift=True)
GRU = WindowSpec(raw_window=30, stride=1, batch_size=128, score_start=19, eval_stride=10, shift=True)


def test_numbered_windows_reverse_without_changing_spans():
    raw = np.arange(120)
    spans = {"train": (0, 60), "val": (60, 90), "test": (90, 120)}
    spec = WindowSpec(raw_window=30, stride=10, batch_size=2, score_start=19)
    fw, fi, fm = build_views(raw, spans, spec, "forward")
    bw, bi, bm = build_views(raw, spans, spec, "backward")
    np.testing.assert_array_equal(bw.seqs, fw.seqs[:, ::-1])
    for name in spans:
        np.testing.assert_array_equal(fi[name], bi[name])
        np.testing.assert_array_equal(
            fm[name]["window_starts"], bm[name]["window_starts"]
        )


def test_the_confirmed_split_and_counts():
    """80/10/10 of 300,000 tokens; the counts the plan's section 6 budgets rest on."""
    assert SPANS == {"train": (0, 240_000), "val": (240_000, 270_000), "test": (270_000, 300_000)}
    raw = np.arange(N)
    _, idx, man = build_views(raw, SPANS, TRANSFORMER, "forward")
    assert [man[s]["n_windows"] for s in ("train", "val", "test")] == [797, 99, 99]
    assert man["train"]["tail_tokens_dropped"] == 240_000 - 797 * 301
    _, idx, man = build_views(raw, SPANS, GRU, "forward")
    assert [man[s]["n_windows"] for s in ("train", "val", "test")] == [239_971, 2_998, 2_998]
    stride10 = WindowSpec(raw_window=30, stride=10, batch_size=32, score_start=19)
    assert build_views(raw, SPANS, stride10, "forward")[2]["train"]["n_windows"] == 23_998


def test_no_sample_crosses_a_split_and_transformer_blocks_are_disjoint():
    raw = np.arange(N)
    for spec in (TRANSFORMER, GRU):
        ds, idx, _ = build_views(raw, SPANS, spec, "forward")
        for name, (lo, hi) in SPANS.items():
            rows = ds.seqs[idx[name]]
            assert rows.min() >= lo and rows.max() < hi, (spec, name)
            assert (np.diff(rows, axis=1) == 1).all()            # contiguous raw tokens
    ds, idx, _ = build_views(raw, SPANS, TRANSFORMER, "forward")
    assert np.unique(ds.seqs).size == ds.seqs.size              # no raw token used twice


def test_backward_rows_are_the_forward_rows_reversed():
    raw = np.random.default_rng(0).integers(0, 3, N)
    fw, fi, _ = build_views(raw, SPANS, GRU, "forward")
    bw, bi, _ = build_views(raw, SPANS, GRU, "backward")
    np.testing.assert_array_equal(bw.seqs, fw.seqs[:, ::-1])
    x, y = bw[5]
    np.testing.assert_array_equal(np.r_[x.numpy(), y.numpy()[-1]], bw.seqs[5])   # (x[:-1], x[1:])


def _epoch_starts(loader, arm="forward"):
    starts = []
    for x, y in loader:
        row = torch.cat([x, y[:, -1:]], dim=1).numpy()
        starts.extend((row[:, 0] if arm == "forward" else row[:, -1]).tolist())
        step = 1 if arm == "forward" else -1
        assert (np.diff(row, axis=1) == step).all()
    return np.sort(starts)


@pytest.mark.parametrize("arm", ["forward", "backward"])
def test_random_shift_recuts_the_training_rows_every_epoch(arm):
    raw = np.arange(N)
    ds, idx, _ = build_views(raw, SPANS, TRANSFORMER, arm)
    train, val, test, ana = make_loaders(ds, idx, TRANSFORMER, 0, 32, raw=raw, train_span=SPANS["train"], arm=arm)
    assert len(train) == 25 and train.sampler.n_rows == 796
    for _ in range(3):
        st = _epoch_starts(train, arm)
        r = train.sampler.offsets[-1]
        assert st.size == 796
        np.testing.assert_array_equal(st, r + 301 * np.arange(796))     # lo + r + k * stride
        assert st.max() + 300 < SPANS["train"][1]
    assert len(set(train.sampler.offsets)) == 3                       # a new cut each epoch
    rng = [int(np.random.default_rng([0, e]).integers(0, 301)) for e in range(3)]
    assert train.sampler.offsets == rng                               # the seeded generator's draws
    assert len(ana.dataset) == 797 and len(val.dataset) == 99 and len(test.dataset) == 99


def test_the_same_seed_reproduces_the_offsets_and_another_does_not():
    a, b, c = ShiftSampler(796, 301, 0), ShiftSampler(796, 301, 0), ShiftSampler(796, 301, 1)
    for s in (a, b, c):
        for _ in range(4):
            list(s)
    assert a.offsets == b.offsets and a.offsets != c.offsets


def test_shift_at_stride_one_is_the_fixed_windows():
    raw = np.arange(N)
    ds, idx, _ = build_views(raw, SPANS, GRU, "forward")
    train, *_ = make_loaders(ds, idx, GRU, 0, 128, raw=raw, train_span=SPANS["train"])
    assert len(train) == 1875
    np.testing.assert_array_equal(_epoch_starts(train), np.sort(ds.seqs[idx["train"], 0]))
    assert set(train.sampler.offsets) == {0}


def test_without_shift_only_the_order_changes():
    raw = np.arange(N)
    spec = WindowSpec(raw_window=301, stride=301, batch_size=32, score_start=20, shift=False)
    ds, idx, _ = build_views(raw, SPANS, spec, "forward")
    train, *_ = make_loaders(ds, idx, spec, 0, 32)
    a, b = _epoch_starts(train), _epoch_starts(train)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, np.sort(ds.seqs[idx["train"], 0]))


def test_shifted_rows_reject_a_span_too_short_for_some_offset():
    with pytest.raises(ValueError):
        ShiftedTrainRows(np.arange(400), (0, 400), WindowSpec(301, 301, 1, 20), "forward")


class _Probe(L.LightningModule):
    """Ignores the token values (they are raw positions, not a vocabulary); records each batch's starts."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.starts = []

    def training_step(self, batch, batch_idx):
        x, _ = batch
        self.starts.append(x[:, 0].cpu().numpy())
        return (self.w ** 2).sum() + 0.0 * x.float().sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_random_shift_under_lightning():
    """Gate 16: under a real two-epoch Lightning fit the offset changes, and the recorded
    offsets are the seeded generator's (Lightning draws exactly one per epoch)."""
    raw = np.arange(30_000)
    spans = split_spans(len(raw))
    spec = WindowSpec(raw_window=301, stride=301, batch_size=8, score_start=20, shift=True)
    ds, idx, _ = build_views(raw, spans, spec, "forward")
    train, *_ = make_loaders(ds, idx, spec, 3, 8, raw=raw, train_span=spans["train"])
    model = _Probe()
    trainer = L.Trainer(max_epochs=2, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train)
    want = [int(np.random.default_rng([3, e]).integers(0, 301)) for e in range(2)]
    assert train.sampler.offsets == want and want[0] != want[1]
    per_epoch = len(train)
    for e, r in enumerate(want):
        st = np.sort(np.concatenate(model.starts[e * per_epoch:(e + 1) * per_epoch]))
        np.testing.assert_array_equal(st, r + 301 * np.arange(train.sampler.n_rows))


def test_canonical_trajectory_is_saved_once_and_verified(tmp_path):
    path = str(tmp_path / "data" / "renewal.npz")
    meta = {"process": "renewal", "params": {"F": [0.5, 0.5]}, "burn_in": 100, "data_seed": 1, "n": 50}
    calls = []

    def make():
        calls.append(1)
        return np.random.default_rng(1).integers(0, 2, 50)

    raw, digest = canonical_trajectory(path, meta, make)
    again, digest2 = canonical_trajectory(path, meta, make)
    assert len(calls) == 1 and digest == digest2 == sha1(raw)
    np.testing.assert_array_equal(raw, again)
    with pytest.raises(ValueError):                         # different metadata: never reused
        canonical_trajectory(path, {**meta, "data_seed": 2}, make)
    in_memory, _ = canonical_trajectory(None, meta, make)
    np.testing.assert_array_equal(in_memory, raw)
