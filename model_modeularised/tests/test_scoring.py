"""
The one scoring policy (scoring.py; plan section 8, gates 3-4): which outputs count,
and the masked cross-entropy arithmetic.

Run:  pytest tests/ -q          (from model_modeularised/)
"""
import numpy as np
import pytest
import torch

from decoder_base import cross_ent_onehot
from scoring import masked_ce_bits, position_mask, score_mask, token_ce_bits


def test_mask_is_per_row_and_preserves_gradient_boundary():
    torch.manual_seed(2)
    logits = torch.randn(2, 30, 3, requires_grad=True)
    targets = torch.randint(0, 3, (2, 30))
    loss, mask = masked_ce_bits(logits, targets, score_start=20)
    expected = token_ce_bits(logits, targets)[:, 20:].mean()
    torch.testing.assert_close(loss, expected)
    assert mask.sum().item() == 20
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :20]).item() == 0
    assert torch.count_nonzero(logits.grad[:, 20:]).item() > 0


@pytest.mark.parametrize("start, first_target", [(20, 21), (19, 20)])
def test_numbered_tokens_show_which_targets_are_scored(start, first_target):
    """Output t predicts raw token x[t+1]: start 20 scores x21 onward, start 19 x20 onward."""
    raw = torch.arange(301)
    targets = raw[1:].unsqueeze(0)                     # (1, 300): target t is token t+1
    mask = score_mask(targets, start)
    scored = targets[mask]
    assert scored[0].item() == first_target and scored[-1].item() == 300
    assert scored.numel() == 300 - start


def test_the_20_plus_10_gru_window():
    """30 raw tokens = 20 of context + 10 scored: inputs x0..x28, scored targets x20..x29."""
    raw = torch.arange(30)
    targets = raw[1:].unsqueeze(0)
    scored = targets[score_mask(targets, 19)]
    assert scored.tolist() == list(range(20, 30))


def test_empty_or_invalid_score_regions_are_rejected():
    targets = torch.zeros(2, 10, dtype=torch.long)
    for bad in (-1, 10, 11):
        with pytest.raises(ValueError):
            score_mask(targets, bad)
    with pytest.raises(ValueError):
        score_mask(targets[0], 0)                       # not (batch, time)
    with pytest.raises(ValueError):
        score_mask(targets, 0, mode="sideways")


def test_backward_mode_flips_the_mask():
    """A mode='backward' model has its short context at the END."""
    targets = torch.zeros(1, 10, dtype=torch.long)
    fw, bw = score_mask(targets, 3)[0], score_mask(targets, 3, "backward")[0]
    assert fw.tolist() == [False] * 3 + [True] * 7
    assert bw.tolist() == fw.flip(0).tolist()
    assert position_mask(10, 3, "backward").tolist() == bw.tolist()


def test_skip_zero_reproduces_the_legacy_ce():
    torch.manual_seed(0)
    logits, targets = torch.randn(4, 25, 3), torch.randint(0, 3, (4, 25))
    new, mask = masked_ce_bits(logits, targets, 0)
    old, _ = cross_ent_onehot(logits, targets)
    torch.testing.assert_close(new, old)
    assert bool(mask.all())


def test_partial_batches_aggregate_to_the_concatenated_mean():
    """Token-weighted over batches of unequal size, exactly as one concatenated batch."""
    torch.manual_seed(1)
    logits, targets = torch.randn(7, 30, 3), torch.randint(0, 3, (7, 30))
    whole, _ = masked_ce_bits(logits, targets, 19)
    total = n = 0.0
    for sl in (slice(0, 4), slice(4, 7)):               # a full batch and a partial one
        ce, mask = masked_ce_bits(logits[sl], targets[sl], 19)
        total += ce.item() * mask.sum().item()
        n += mask.sum().item()
    assert total / n == pytest.approx(whole.item(), rel=1e-6)
    np.testing.assert_allclose(n, 7 * 11)
