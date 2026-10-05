"""
Samples cut from ONE long trajectory (SHARED_TRAJECTORY_EMBEDDING_WARMUP_PLAN.md
sections 4 and 12.1): the continuous data mode.

    canonical_trajectory   the one forward draw per process/parameters/data seed, saved
                           once with its hash and generation parameters and verified on
                           every reuse, so every architecture and arm reads the same tokens
    WindowSpec             an architecture's view: raw window, stride, batch, score start,
                           evaluation stride, and whether training rows are re-cut
    split_spans            train/validation/test by raw token range, fixed BEFORE any
                           sample is cut (80/10/10)
    build_views            fixed rows cut inside each split, never across a boundary;
                           the backward arm is each row reversed
    ShiftedTrainRows,      the random shift: every epoch the training span is re-cut at
    ShiftSampler           an offset drawn from a generator seeded by (seed, epoch), so a
                           transformer never sees an identical block twice (study6: with
                           fixed cuts it memorised its 797 blocks)
    make_loaders           train / validation / test / analysis loaders

Only the training loader is shifted.  Validation, test and the analysis rows keep the
fixed offset-0 cuts, so every evaluation is deterministic.  At stride 1 every offset is
0 and the shift is a no-op.
"""
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from process_generator import SequenceDataset


def sha1(raw) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(raw, dtype=np.int64)).tobytes()).hexdigest()


def canonical_trajectory(path, meta: dict, make):
    """
    (raw, sha1): the trajectory saved at `path` when its metadata equals `meta`, else
    make() it and save it there.  A stored file whose metadata differs, or whose tokens
    no longer hash to the stored value, is an error, never silently replaced.  The file
    is written to a sibling temp and os.replace()d, so parallel runs of one experiment
    cannot read a half-written draw.  path=None keeps the draw in memory only.
    """
    meta = json.loads(json.dumps(meta, sort_keys=True, default=str))
    if path is not None and os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            saved, raw, digest = json.loads(str(z["meta"])), z["raw"].astype(np.int64), str(z["sha1"])
        if saved != meta:
            raise ValueError(f"{path} holds a different trajectory:\n  saved {saved}\n  wanted {meta}")
        if sha1(raw) != digest:
            raise ValueError(f"{path}: the stored tokens do not match their hash")
        return raw, digest
    raw = np.ascontiguousarray(np.asarray(make(), dtype=np.int64))
    if raw.ndim != 1:
        raise ValueError(f"expected a 1D trajectory, got {raw.shape}")
    digest = sha1(raw)
    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}.npz"
        np.savez_compressed(tmp, raw=raw, sha1=digest, meta=json.dumps(meta, sort_keys=True))
        os.replace(tmp, path)
    return raw, digest


@dataclass(frozen=True)
class WindowSpec:
    raw_window: int
    stride: int
    batch_size: int
    score_start: int
    eval_stride: int | None = None  # validation/test windows; None -> stride
    shift: bool = False             # training rows re-cut at a random offset every epoch

    def __post_init__(self):
        if self.raw_window < 2:
            raise ValueError("a window needs an input and a target")
        if not 1 <= self.stride <= self.raw_window:
            raise ValueError("invalid stride")
        if self.eval_stride is not None and not 1 <= self.eval_stride <= self.raw_window:
            raise ValueError("invalid eval_stride")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.score_start < self.raw_window - 1:
            raise ValueError("score_start leaves no scored targets")

    def stride_for(self, split):
        if split == "train" or self.eval_stride is None:
            return self.stride
        return self.eval_stride


def split_spans(n_tokens, train_fraction=0.8, val_fraction=0.1):
    if not (0 < train_fraction < 1 and
            0 < val_fraction < 1 - train_fraction):
        raise ValueError("train/validation/test fractions must be positive")
    tr = int(n_tokens * train_fraction)
    va = tr + int(n_tokens * val_fraction)
    return {"train": (0, tr), "val": (tr, va), "test": (va, n_tokens)}


def build_views(raw, spans, spec, arm):
    raw = np.asarray(raw, dtype=np.int64)
    if raw.ndim != 1 or arm not in ("forward", "backward"):
        raise ValueError("expected a 1D trajectory and a valid arm")
    expected = 0
    for name in ("train", "val", "test"):
        lo, hi = spans[name]
        if lo != expected or not lo < hi <= len(raw):
            raise ValueError("splits must be ordered, disjoint and exhaustive")
        expected = hi
    if expected != len(raw):
        raise ValueError("splits must cover the trajectory")

    blocks, indices, manifest = [], {}, {}
    offset = 0
    for name in ("train", "val", "test"):
        lo, hi = spans[name]
        starts = np.arange(lo, hi - spec.raw_window + 1,
                           spec.stride_for(name), dtype=np.int64)
        if not starts.size:
            raise ValueError(f"{name} contains no complete window")
        rows = raw[starts[:, None] + np.arange(spec.raw_window)]
        if arm == "backward":
            rows = rows[:, ::-1]
        rows = np.ascontiguousarray(rows)  # positive strides for torch
        blocks.append(rows)
        indices[name] = np.arange(offset, offset + len(rows))
        offset += len(rows)
        manifest[name] = {
            "raw_span": [int(lo), int(hi)],
            "window_starts": starts,
            "n_windows": len(rows),
            "tail_tokens_dropped": int(hi - starts[-1] - spec.raw_window),
        }
    return SequenceDataset(np.concatenate(blocks)), indices, manifest


class ShiftedTrainRows(Dataset):
    """Training rows for the key (r, k): raw[lo + r + k*stride : ... + raw_window].
    ShiftSampler chooses r once per epoch.  Returns SequenceDataset's (x[:-1], x[1:])."""

    def __init__(self, raw, span, spec, arm):
        self.raw = np.asarray(raw, dtype=np.int64)
        self.lo, self.hi = span
        self.spec, self.arm = spec, arm
        L, s = spec.raw_window, spec.stride
        # the same row count at every offset, so every epoch has the same number of updates
        self.n_rows = min(len(range(self.lo + r, self.hi - L + 1, s)) for r in range(s))
        if self.n_rows < 1:
            raise ValueError("the training span holds no complete row at some offset")

    def __len__(self):
        return self.n_rows

    def __getitem__(self, key):
        r, k = key
        start = self.lo + r + k * self.spec.stride
        row = self.raw[start:start + self.spec.raw_window]
        if self.arm == "backward":
            row = row[::-1]
        x = torch.from_numpy(np.ascontiguousarray(row))
        return x[:-1], x[1:]


class ShiftSampler(Sampler):
    """One random offset in [0, stride) per pass over the loader, then a shuffled row order.
    __iter__ runs once per epoch, so this needs no Lightning hook; the offsets are recorded."""

    def __init__(self, n_rows, stride, seed):
        self.n_rows, self.stride, self.seed = n_rows, stride, int(seed)
        self.epoch, self.offsets = 0, []

    def __len__(self):
        return self.n_rows

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        r = int(rng.integers(0, self.stride))
        self.offsets.append(r)
        for k in rng.permutation(self.n_rows):
            yield (r, int(k))


def make_loaders(ds, indices, spec, shuffle_seed, analysis_batch_size,
                 raw=None, train_span=None, arm="forward"):
    subsets = {k: Subset(ds, ids.tolist()) for k, ids in indices.items()}
    if spec.shift:
        rows = ShiftedTrainRows(raw, train_span, spec, arm)
        train = DataLoader(
            rows, batch_size=spec.batch_size, drop_last=False,
            sampler=ShiftSampler(rows.n_rows, spec.stride, shuffle_seed),
        )
    else:
        train = DataLoader(
            subsets["train"], batch_size=spec.batch_size, shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(shuffle_seed),
        )
    val = DataLoader(subsets["val"], batch_size=spec.batch_size, shuffle=False)
    test = DataLoader(subsets["test"], batch_size=spec.batch_size, shuffle=False)
    analysis = DataLoader(
        subsets["train"], batch_size=analysis_batch_size, shuffle=False
    )
    return train, val, test, analysis
