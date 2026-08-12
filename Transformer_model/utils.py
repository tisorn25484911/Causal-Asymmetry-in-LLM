"""
utils.py — small shared helpers, defined once.

IMPROVEMENT_PLAN.md C4.  These had drifted into several copies across the
repo: `mkdir`, `save_pkl`, `save_weights`, `savefig` and `cleanup` existed in
both Model_analysis.py and sanity_check.py, and `entropy_rate_coin` existed
three times (Main_call, sanity_check, LLM_asymmetry_testing).  Duplicated
definitions are fine until one of them is fixed and the others are not.

Tag derivation lives here too, because the runner and both post-hoc consumers
have to agree on it (A4/B8): the runner writes results/models/<tag>_{fw,bw}.pt
and the consumers load it, so a tag computed differently in the two places
would silently load the wrong weights.
"""

import contextlib
import gc
import json
import os
import pickle

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Experiment tags — derived from config values, never written as literals
# ─────────────────────────────────────────────────────────────────────────────
def coin_tag(prefix: str, p: float, q: float) -> str:
    """
    e.g. coin_tag("exp1", 0.4, 0.8) -> "exp1_coin_p040_q080".

    Tags used to be string literals, so Main_call.py wrote
    "exp1_coin_p03_q04" while its CFG said p=0.4, q=0.8 -- main_large's
    parameters (B8).  Deriving the tag makes it impossible for a tag to
    disagree with the run that produced it.
    """
    return f"{prefix}_coin_p{round(p*100):03d}_q{round(q*100):03d}"


def flower_tag(prefix: str, n: int, m: int) -> str:
    return f"{prefix}_flower_n{n}_m{m}"


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem
# ─────────────────────────────────────────────────────────────────────────────
# REORGANISATION_FIX_PLAN.md 4.2.  This file lives in Transformer_model/, so the
# repo root is two levels up from __file__.  Anchoring on __file__ rather than
# the cwd is the whole point: every output directory in this repo is named
# repo-relative ("All_Results/results_quick"), which was unambiguous while the
# scripts sat at the root and is not any more.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(p: str) -> str:
    """
    Resolve a repo-relative path against the repo root rather than the cwd.

    Absolute paths pass through untouched, so `--out-root /tmp/scratch` still
    means /tmp/scratch.  A *relative* argument resolves against the repo root,
    which makes the documented invariant ("run from the repo root") true by
    construction instead of a thing the caller has to remember.

    This matters more than it looks.  `mkdir` below is exist_ok=True and
    `load_combined` treats a missing pickle as "nothing done yet", so a stale
    relative path does not raise -- it silently creates an empty directory and
    re-runs the whole experiment.  That failure mode cost a 23 h sweep once
    already; resolving paths from a fixed anchor is what stops it recurring.
    """
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_pkl(obj, path: str):
    """
    Atomically replace `path` with a pickle of `obj`.

    REMAINING_WORK_PLAN.md C1.  This used to be a bare open(path, "wb") +
    pickle.dump, and open(..., "wb") TRUNCATES before a single byte is written.
    A 12 MB sweep bundle takes real time to serialise, and for that whole window
    the only copy on disk was a partial file -- which load_combined then reads as
    "no repeats completed yet", silently discarding the run and restarting it.

    That is not hypothetical: the lambda=1.0 coin run died inside this function.
    It survived only because the failure was on open() with the results tree
    already moved away, so there was nothing to truncate.  A kill one second
    later would have destroyed 122 completed cells.

    Writing to a sibling temp file and os.replace()-ing it is atomic within a
    filesystem, so a reader sees either the whole old file or the whole new one.
    The pid in the temp name matters because two sweeps have been run
    concurrently before (different out_roots, same code path) -- a shared temp
    name would let them corrupt each other.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
            f.flush()
            os.fsync(f.fileno())        # survive a power loss, not just a crash
        os.replace(tmp, path)           # atomic
    except BaseException:
        # Never leave the temp file behind on Ctrl-C or an exception; the real
        # bundle at `path` is untouched either way, which is the point.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    mb = os.path.getsize(path) / 1024**2
    print(f"  pickle saved -> {path}  ({mb:.1f} MB)")


def save_run_config(path: str, record: dict,
                    keep_if_unset: tuple = (), union_lists: tuple = ()):
    """
    Merge a run's provenance into `path` instead of overwriting it.

    REMAINING_WORK_PLAN.md C3.  Both runners used to write run_config_<CONFIG>.json
    fresh on every invocation, recording only the grids THAT invocation set --
    while the result pickle beside it MERGES across invocations.  The two then
    disagree, and the JSON is the one a reader trusts.  It has already destroyed
    information: All_Results/results_sweep holds 100 coin cells and 81 flower
    cells, and its JSON says `coin_grid: null`, because the flower invocation ran
    last.  The weight-decay folders escaped only because the launcher copied the
    earlier record aside by hand.

    `keep_if_unset` names keys whose previous value must survive when this
    invocation leaves them falsy (the grids: `--sweep-coin` alone must not erase
    the flower grid).  `union_lists` names keys that accumulate rather than
    replace (the process list, so `--only` does not shrink the record).
    Everything else is overwritten, which is right for scalars like `repeats`.

    An `invocations` list preserves the per-run history, so the merge never hides
    the fact that a folder was built up over several commands.
    """
    prev = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                prev = json.load(f)
        except Exception as e:                 # corrupt or hand-edited: keep going
            print(f"  ! could not read {path} ({e}) — starting a fresh record")

    merged = dict(prev)
    for k, v in record.items():
        if k in keep_if_unset and not v:
            continue                            # leave the earlier value in place
        if k in union_lists:
            old = prev.get(k) or []
            new = v or []
            seen, out = set(), []
            for item in list(old) + list(new):
                key = json.dumps(item, sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key); out.append(item)
            merged[k] = out
            continue
        merged[k] = v

    merged["invocations"] = list(prev.get("invocations") or []) + [
        {k: record.get(k) for k in ("utc", "repeats", *keep_if_unset, *union_lists)
         if k in record}
    ]
    with open(path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    return merged


def save_weights(model, path: str, meta: dict | None = None):
    """
    Save weights, and next to them a JSON sidecar describing the architecture
    and the process they were trained on -- IMPROVEMENT_PLAN.md B11.

    Without it a loader has to *guess* d_model / n_layers / p / q, and guessing
    wrong surfaces as a confusing shape error from load_state_dict rather than
    a clear message.  It also lets a consumer verify it is scoring a checkpoint
    against the data it was actually trained on, which is the failure A4
    describes.
    """
    torch.save(model.state_dict(), path)
    print(f"  weights -> {path}")
    if meta is not None:
        import json
        side = os.path.splitext(path)[0] + ".json"
        with open(side, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  sidecar -> {side}")


def load_weight_meta(weights_path: str) -> dict | None:
    """Read the JSON sidecar written by save_weights, if present."""
    import json
    side = os.path.splitext(weights_path)[0] + ".json"
    if not os.path.exists(side):
        return None
    with open(side) as f:
        return json.load(f)


def check_weight_meta(weights_path: str, expected: dict) -> bool:
    """
    Warn loudly when a checkpoint's sidecar disagrees with what the caller
    believes it is loading (A4).  Returns True when everything checked matches
    or no sidecar exists.
    """
    meta = load_weight_meta(weights_path)
    if meta is None:
        print(f"  ! no sidecar for {weights_path} — cannot verify architecture")
        return True
    bad = {k: (meta.get(k), v) for k, v in expected.items()
           if k in meta and meta.get(k) != v}
    if bad:
        print(f"  ! MISMATCH loading {weights_path}:")
        for k, (got, want) in bad.items():
            print(f"      {k}: checkpoint={got!r}  caller expects={want!r}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Accelerator hygiene
# ─────────────────────────────────────────────────────────────────────────────
def cleanup():
    """
    GC + accelerator flush.
    NOTE: torch.mps.empty_cache() segfaults on PyTorch <= 2.1 (known Apple
    bug).  We intentionally do NOT call it.  Models are moved to CPU before
    analysis (via to_cpu_for_analysis) so the Metal heap is freed that way.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def to_cpu_for_analysis(model):
    """
    Move a trained model to CPU in-place and return it.
    """
    model.cpu()
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Information theory
# ─────────────────────────────────────────────────────────────────────────────
def entropy_bits(probs) -> float:
    """Shannon entropy in bits, ignoring zero-probability atoms."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def entropy_rate_coin(p: float, q: float) -> float:
    """
    Entropy rate H_inf (bits/token) of the coin-process HMM.

    H_inf is time-reversal invariant, so the same value applies to both
    directions -- which is why the runners pass one `theory` to both arms, and
    why delta_CE is a difference of *residuals* rather than of entropy rates.

    H_inf = pi0*H2(p) + pi1*H2(q),  pi0 = q/(p+q),  pi1 = p/(p+q).
    """
    def _h2(a):
        b = 1.0 - a
        a, b = max(a, 1e-12), max(b, 1e-12)
        return -a * np.log2(a) - b * np.log2(b)
    return (q / (p + q)) * _h2(p) + (p / (p + q)) * _h2(q)
