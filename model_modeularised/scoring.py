"""
One definition of which predictions are scored (SHARED_TRAJECTORY_EMBEDDING_WARMUP_PLAN.md
section 5), used by training, validation, held-out and fresh scoring, the exact-model
aggregates and the plots, so none of them can drift from the others.

    score_mask      (B,T) bool: True where an output is scored
    position_mask   the same for one row, in numpy, for the analysis code
    token_ce_bits   per-token cross-entropy in bits, full length
    masked_ce_bits  its mean over the scored outputs, and the mask

Indexing (plan section 3).  For a raw row x[0:R] the model reads inputs x[:-1] and
output t predicts x[t+1] after t+1 tokens of context, so score_start=20 scores the
targets x21 onward and score_start=19 scores x20 onward.  The prefix is excluded from
the OBJECTIVE only: it stays in the forward pass and in the autograd graph, so suffix
losses still train the prefix representation.

mode="backward" (a model reading right to left, decoder_base._Decoder._split) has its
short context at the END, so the mask is flipped.  The runner itself keeps every model
at mode="forward" and takes the backward arm from reversed data.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def score_mask(targets, score_start=0, mode="forward"):
    if targets.ndim != 2:
        raise ValueError("expected (batch, time)")
    T = targets.shape[1]
    if not 0 <= score_start < T:
        raise ValueError("score_start leaves no scored targets")
    if mode not in ("forward", "backward"):
        raise ValueError("invalid model mode")
    keep = torch.arange(T, device=targets.device) >= score_start
    if mode == "backward":
        keep = keep.flip(0)
    return keep.unsqueeze(0).expand_as(targets)


def position_mask(T, score_start=0, mode="forward") -> np.ndarray:
    """(T,) bool: score_mask for one row of T outputs, in numpy."""
    return score_mask(torch.zeros(1, int(T)), score_start, mode)[0].numpy().copy()


def token_ce_bits(logits, targets):
    if logits.shape[:2] != targets.shape:
        raise ValueError("logits and targets must share batch/time dimensions")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1), reduction="none",
    ).reshape_as(targets) / math.log(2.0)


def masked_ce_bits(logits, targets, score_start=0, mode="forward"):
    """(mean CE in bits over the scored outputs, the (B,T) mask).  The denominator is the
    number of scored targets, so a partial batch is weighted by what it actually scores."""
    mask = score_mask(targets, score_start, mode)
    values = token_ce_bits(logits, targets)
    return values[mask].mean(), mask
