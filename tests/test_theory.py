"""
Regression tests for the closed forms and the loss path.

Run:  pytest tests/ -q      (from the repo root)

These lock in the Phase 1 fixes so a future edit cannot quietly undo them.
IMPROVEMENT_PLAN.md Phase 4.6 asks for most of these; the B3 and B2 cases are
added because those are the two bugs that were silent in production.
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_generation import coin_generation                    # noqa: E402
from Flower_process_generation import flower_process_generation  # noqa: E402
from Model_analysis import (                                   # noqa: E402
    coin_true_conditional,
    flower_complexity,
    flower_entropy_rate,
    statistical_complexity,
)
from OneHot_model import cross_ent_onehot                      # noqa: E402
from Training_model import ChunckDataset, _eval_loss_on_loader, set_seed  # noqa: E402
from utils import entropy_rate_coin                            # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Coin process
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("p,q", [(0.5, 0.5), (0.4, 0.8), (0.1, 0.9), (0.3, 0.4)])
def test_coin_conditionals_are_stochastic(p, q):
    fw, bw = coin_true_conditional(p, q)
    assert np.allclose(fw.sum(axis=1), 1.0)
    assert np.allclose(bw.sum(axis=1), 1.0)
    assert (fw >= 0).all() and (bw >= 0).all()


@pytest.mark.parametrize("p,q", [(0.5, 0.5), (0.4, 0.8), (0.1, 0.9)])
def test_forward_conditional_matches_monte_carlo(p, q):
    """The analytic P(next | current) must match the generator that produced it."""
    set_seed(0)
    data, _ = coin_generation(num_samples=40, seq_len=4000, p=p, q=q)
    counts = np.zeros((3, 3))
    for seq in data:
        a = np.asarray(seq)
        np.add.at(counts, (a[:-1], a[1:]), 1)
    emp = counts / counts.sum(axis=1, keepdims=True)
    fw, _ = coin_true_conditional(p, q)
    assert np.allclose(emp, fw, atol=0.02), f"\nempirical:\n{emp}\nanalytic:\n{fw}"


def test_entropy_rate_symmetric_coin_is_one_bit():
    assert entropy_rate_coin(0.5, 0.5) == pytest.approx(1.0)


def test_coin_complexity_reference_values():
    # IMPROVEMENT_PLAN.md section 4
    assert statistical_complexity(0.5, 0.5, "forward")  == pytest.approx(1.0,    abs=1e-9)
    assert statistical_complexity(0.5, 0.5, "backward") == pytest.approx(1.5,    abs=1e-9)
    assert statistical_complexity(0.4, 0.8, "forward")  == pytest.approx(0.9183, abs=1e-4)
    assert statistical_complexity(0.4, 0.8, "backward") == pytest.approx(1.5656, abs=1e-4)


@pytest.mark.parametrize("p,q", [(0.1, 0.9), (0.3, 0.4), (0.4, 0.8), (0.5, 0.5),
                                 (0.05, 0.95), (0.7, 0.2)])
def test_coin_is_asymmetric_for_every_pq(p, q):
    """C- >= C+ identically for the coin — it is a positive control everywhere."""
    assert (statistical_complexity(p, q, "backward")
            >= statistical_complexity(p, q, "forward") - 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Flower process — A1
# ─────────────────────────────────────────────────────────────────────────────
def test_flower_generator_alternates_select_and_roll():
    set_seed(0)
    n, m = 3, 4
    dice = np.full((n, m), 1.0 / m)
    data, _ = flower_process_generation(num_samples=5, seq_len=50, n=n, m=m,
                                        dice_probs=dice)
    for seq in data:
        a = np.asarray(seq)
        assert (a[0::2] < n).all(),  "even positions must be die selections"
        assert (a[1::2] >= n).all(), "odd positions must be roll outcomes"
        assert (a[1::2] < n + m).all()


def test_flower_n1_m2_is_a_null_control():
    """The central A1 correction: n=1 gives C- == C+, not m+1 states."""
    C_plus, C_minus = flower_complexity(1, 2, [[0.5, 0.5]])
    assert C_plus == pytest.approx(1.0)
    assert C_minus == pytest.approx(1.0)
    assert C_minus == pytest.approx(C_plus)


def test_flower_reference_values_match_plan():
    # IMPROVEMENT_PLAN.md section 4
    dice42_2 = np.random.default_rng(42).dirichlet(np.ones(2), size=4)
    cp, cm = flower_complexity(4, 2, dice42_2)
    assert (cp, cm) == (pytest.approx(2.0000, abs=1e-4),
                        pytest.approx(1.4952, abs=1e-4))

    dice42_4 = np.random.default_rng(42).dirichlet(np.ones(4), size=6)
    cp, cm = flower_complexity(6, 4, dice42_4)
    assert (cp, cm) == (pytest.approx(2.2925, abs=1e-4),
                        pytest.approx(1.9899, abs=1e-4))


def test_flower_asymmetry_requires_m_greater_than_n():
    """C- > C+ iff m > n — the property every original config violated."""
    rng = np.random.default_rng(0)
    for n, m in [(2, 6), (2, 8), (4, 2), (6, 4), (3, 3)]:
        dice = rng.dirichlet(np.ones(m), size=n)
        cp, cm = flower_complexity(n, m, dice)
        if m > n:
            assert cm > cp, f"n={n}, m={m}: expected C- > C+"
        elif n > m:
            assert cp > cm, f"n={n}, m={m}: expected C+ > C-"


def test_flower_indistinguishable_outcomes_merge():
    """Proportional dice_probs columns are ONE backward causal state."""
    # Both dice have identical distributions -> every outcome has the same
    # posterior over dice -> all outcomes collapse to a single state.
    _, cm = flower_complexity(2, 3, [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
    assert cm == pytest.approx(1.0)


def test_flower_entropy_rate_matches_hand_derivation():
    # n=1, fair 2-sided die: 0.5*log2(1) + 0.5*H(0.5,0.5) = 0.5
    assert flower_entropy_rate(1, 2, [[0.5, 0.5]]) == pytest.approx(0.5)
    # n=4 uniform dice: 0.5*log2(4) + 0.5*1 = 1.5
    assert flower_entropy_rate(4, 2, np.full((4, 2), 0.5)) == pytest.approx(1.5)


def test_flower_complexity_rejects_bad_dice():
    with pytest.raises(ValueError):
        flower_complexity(2, 3, [[0.5, 0.5, 0.5], [0.2, 0.3, 0.5]])   # rows != 1
    with pytest.raises(ValueError):
        flower_complexity(2, 3, [[0.5, 0.5]])                          # wrong shape


# ─────────────────────────────────────────────────────────────────────────────
# Loss path — B3, B1, B2
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("vocab", [2, 3, 6, 10])
def test_uniform_logits_give_log2_vocab(vocab):
    logits = torch.zeros(1, 4, vocab)
    targets = torch.zeros(1, 4, dtype=torch.long)
    loss, ppl = cross_ent_onehot(logits, targets)
    assert loss.item() == pytest.approx(math.log2(vocab), abs=1e-5)
    assert ppl.item() == pytest.approx(vocab, abs=1e-4)


@pytest.mark.parametrize("scale", [10, 50, 100, 200, 1000])
def test_cross_entropy_finite_at_saturated_logits(scale):
    """B3 regression: the old implementation returned NaN at scale 200."""
    logits = torch.tensor([[[float(scale), 0.0, 0.0]]])
    targets = torch.tensor([[0]])
    loss, ppl = cross_ent_onehot(logits, targets)
    assert torch.isfinite(loss), f"NaN/inf at logit scale {scale}"
    assert torch.isfinite(ppl)
    assert loss.item() >= 0.0


def test_cross_entropy_matches_reference_implementation():
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 5) * 3
    targets = torch.randint(0, 5, (4, 7))
    loss, _ = cross_ent_onehot(logits, targets)
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 5), targets.reshape(-1)) / math.log(2)
    assert loss.item() == pytest.approx(ref.item(), abs=1e-6)


def test_perplexity_is_two_to_the_ce():
    """B1: the loss is in bits, so PPL must be 2**CE and never exp(CE)."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 4)
    targets = torch.randint(0, 4, (2, 5))
    loss, ppl = cross_ent_onehot(logits, targets)
    assert ppl.item() == pytest.approx(2 ** loss.item(), rel=1e-6)
    assert ppl.item() != pytest.approx(math.exp(loss.item()), rel=1e-3)


def test_eval_loss_is_token_weighted_and_unbiased():
    """B2: token-weighted 2**mean(CE), not mean(2**CE) over batches."""
    spec = [(4, 3, 0.9), (4, 3, 1.4), (4, 3, 1.1), (1, 3, 0.6)]   # ragged last

    def logits_for(ce_bits, n_seq, T):
        pt = 2 ** (-ce_bits)
        a = math.log(pt / (1 - pt))
        return torch.tensor([[[a, 0.0]] * T] * n_seq)

    lg = [logits_for(ce, n, T) for n, T, ce in spec]

    class Model(torch.nn.Module):
        mode = "forward"

        def __init__(self):
            super().__init__()
            self.i = 0
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            out = lg[self.i]
            self.i += 1
            return out

    batches = [(torch.zeros(n, T, dtype=torch.long),
                torch.zeros(n, T, dtype=torch.long)) for n, T, _ in spec]
    ce, ppl = _eval_loss_on_loader(Model(), batches)

    tok = [n * T for n, T, _ in spec]
    ces = [c for _, _, c in spec]
    expected = sum(c * t for c, t in zip(ces, tok)) / sum(tok)

    assert ce == pytest.approx(expected, abs=1e-6)
    assert ppl == pytest.approx(2 ** ce, rel=1e-12)     # identity holds
    # and it is NOT the old unweighted mean-of-batch-means
    assert ce != pytest.approx(float(np.mean(ces)), abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Pairing — A2
# ─────────────────────────────────────────────────────────────────────────────
def _toy_base(n_seq=12, T=40):
    class Base(torch.utils.data.Dataset):
        def __len__(self):
            return n_seq

        def __getitem__(self, i):
            x = torch.arange(i * 1000, i * 1000 + T + 1)
            return x[:-1], x[1:]
    return Base()


def test_chunk_dataset_is_a_pure_function_of_index():
    """A2: the two arms must see identical windows."""
    base = _toy_base()
    a = ChunckDataset(base, 16, seed=0)
    b = ChunckDataset(base, 16, seed=0)
    for i in range(len(base)):
        assert torch.equal(a[i][0], b[i][0])
    # repeated and out-of-order access does not change anything
    c = ChunckDataset(base, 16, seed=0)
    _ = [c[i] for i in reversed(range(len(base)))]
    for i in range(len(base)):
        assert torch.equal(a[i][0], c[i][0])
        assert torch.equal(a[i][0], a[i][0])


def test_chunk_dataset_seed_actually_varies_windows():
    base = _toy_base()
    a = ChunckDataset(base, 16, seed=0)
    b = ChunckDataset(base, 16, seed=1)
    assert any(not torch.equal(a[i][0], b[i][0]) for i in range(len(base)))


def test_chunk_dataset_returns_full_sequence_when_shorter_than_chunk():
    base = _toy_base(T=8)
    ds = ChunckDataset(base, 64, seed=0)
    inp, tgt = ds[0]
    assert inp.shape[0] == 8 and tgt.shape[0] == 8


def test_analysis_loader_matches_training_length():
    """
    The analysis loader must emit sequences of the TRAINING chunk length.

    A chunk is fed to the model as a standalone sequence, so the model only
    ever sees positional-encoding indices [0, chunk).  Evaluating at full
    length measures extrapolation to unseen positions -- measured at 1.556
    bits beyond position 512 versus 1.025 within, against H_inf = 1.0 -- and
    does so asymmetrically, since the forward arm's complexity is read at
    use_t="last" (untrained at full length) and the backward arm's at
    use_t="first" (always trained).
    """
    from Training_model import make_analysis_loader

    base = _toy_base(n_seq=8, T=100)
    chunk = 32
    loader = make_analysis_loader(base, chunk, batch_size=4, seed=1000)
    for inputs, targets in loader:
        assert inputs.shape[1] == chunk, (
            f"analysis loader emitted T={inputs.shape[1]}, expected the "
            f"training chunk length {chunk}")
        assert targets.shape[1] == chunk

    # and it must not simply hand back the training windows
    train = ChunckDataset(base, chunk, seed=0)
    ana   = ChunckDataset(base, chunk, seed=1000)
    assert any(not torch.equal(train[i][0], ana[i][0]) for i in range(len(base)))


# ─────────────────────────────────────────────────────────────────────────────
# Latent sampling for the UMAPs — C6
# ─────────────────────────────────────────────────────────────────────────────
def _tagged_latents(N=50, T=64, d=4):
    """latent[..., 0] = sequence index, latent[..., 1] = position index."""
    lat = np.zeros((N, T, d), dtype=np.float32)
    lat[..., 0] = np.arange(N)[:, None]
    lat[..., 1] = np.arange(T)[None, :]
    inp = np.random.default_rng(0).integers(0, 3, size=(N, T))
    return lat, inp


def test_prefix_sampling_is_the_c6_bug():
    """Documents the old behaviour: n_pts rows = ceil(n_pts/T) sequences."""
    from Model_analysis import _sample_latents
    lat, inp = _tagged_latents(N=50, T=64)
    pts, _, _ = _sample_latents(lat, inp, mode="prefix", n_pts=128)
    assert len(np.unique(pts[:, 0])) == 2      # 128/64 = 2 sequences only


def test_per_sequence_sampling_covers_every_sequence():
    from Model_analysis import _sample_latents
    N, T = 50, 64
    lat, inp = _tagged_latents(N=N, T=T)
    pts, toks, _ = _sample_latents(lat, inp, mode="per_sequence",
                                   use_t="last", n_pts=1000)
    assert len(pts) == N
    assert len(np.unique(pts[:, 0])) == N          # every sequence, once
    assert (pts[:, 1] == T - 1).all()              # all at max-context position
    # backward reads position 0 instead
    pts_b, _, _ = _sample_latents(lat, inp, mode="per_sequence",
                                  use_t="first", n_pts=1000)
    assert (pts_b[:, 1] == 0).all()


def test_random_sampling_spreads_over_sequences_and_respects_burn_in():
    from Model_analysis import _sample_latents
    N, T, b = 50, 64, 16
    lat, inp = _tagged_latents(N=N, T=T)

    # forward: context grows left-to-right, so the FIRST b positions are dropped
    pts, _, _ = _sample_latents(lat, inp, mode="random", use_t="last",
                                n_pts=400, burn_in=b, seed=0)
    assert len(pts) == 400
    assert pts[:, 1].min() >= b
    assert len(np.unique(pts[:, 0])) > N // 2      # spread over many sequences

    # backward: context grows right-to-left, so the LAST b positions are dropped
    pts_b, _, _ = _sample_latents(lat, inp, mode="random", use_t="first",
                                  n_pts=400, burn_in=b, seed=0)
    assert pts_b[:, 1].max() < T - b


def test_random_sampling_is_without_replacement_and_seeded():
    from Model_analysis import _sample_latents
    lat, inp = _tagged_latents(N=20, T=32)
    a, _, _ = _sample_latents(lat, inp, mode="random", n_pts=200, seed=3)
    b, _, _ = _sample_latents(lat, inp, mode="random", n_pts=200, seed=3)
    c, _, _ = _sample_latents(lat, inp, mode="random", n_pts=200, seed=4)
    assert np.array_equal(a, b)                     # reproducible
    assert not np.array_equal(a, c)                 # seed matters
    pairs = {(int(r[0]), int(r[1])) for r in a}
    assert len(pairs) == len(a)                     # no duplicates


# ─────────────────────────────────────────────────────────────────────────────
# Recording cadence — D1 + B12
# ─────────────────────────────────────────────────────────────────────────────
class _FakeTrainer:
    def __init__(self):
        self.global_step = 0
        self.callback_metrics = {}


def _drive_recorder(rec, n_steps, epochs=1):
    """Feed a Record_training synthetic steps without running Lightning."""
    tr = _FakeTrainer()
    per_epoch = n_steps // epochs
    for e in range(epochs):
        for i in range(per_epoch):
            loss = torch.tensor(1.0 + 0.001 * tr.global_step)
            rec.on_train_batch_end(tr, None, loss, None, i)
            tr.global_step += 1
        rec.on_train_epoch_end(tr, None)
    return rec


def test_train_curve_stays_full_resolution_while_val_is_subsampled():
    """
    D1.  The training loss is already computed, so recording it is free; only
    the validation pass is expensive.  Gating them together (the plan's literal
    record_every_n_steps=25) would leave QUICK with 4 points per curve.
    """
    from Training_model import Record_training
    rec = Record_training(record_every_n_steps=1, val_every_n_steps=25,
                          val_loader=None)
    _drive_recorder(rec, n_steps=100)
    assert len(rec.step_loss) == 100          # free series: every step
    assert rec.step_at == list(range(100))


def test_epoch_loss_is_a_true_epoch_mean_at_any_stride():
    """
    B12 regression, and the reason B12 had to land before D1: epoch_loss must
    average EVERY step, not only recorded ones.
    """
    from Training_model import Record_training
    for stride in (1, 7, 25):
        rec = Record_training(record_every_n_steps=stride, val_loader=None)
        _drive_recorder(rec, n_steps=60, epochs=3)
        assert len(rec.epoch_loss) == 3, "one entry per epoch regardless of stride"
        # every step contributed: mean of 1.0 + 0.001*step over each 20-step epoch
        expected = [np.mean([1.0 + 0.001 * (e * 20 + i) for i in range(20)])
                    for e in range(3)]
        assert np.allclose(rec.epoch_loss, expected), (
            f"stride={stride}: epoch_loss became the mean over recorded steps only")


def test_val_cadence_records_its_own_x_axis():
    """Without step_val_at, a sparse val curve plotted against its list index
    would compress the run into the first few percent of the axis."""
    from Training_model import Record_training

    class _Loader:                      # minimal 1-batch loader
        def __iter__(self):
            yield (torch.zeros(2, 4, dtype=torch.long),
                   torch.zeros(2, 4, dtype=torch.long))

    class _M(torch.nn.Module):
        mode = "forward"
        def __init__(s):
            super().__init__(); s.p = torch.nn.Parameter(torch.zeros(1))
        def forward(s, x):
            return torch.zeros(x.shape[0], x.shape[1], 3)
        def train(s, mode=True):
            return s

    rec = Record_training(record_every_n_steps=1, val_every_n_steps=10,
                          val_loader=_Loader())
    tr = _FakeTrainer()
    m = _M()
    for i in range(50):
        rec.on_train_batch_end(tr, m, torch.tensor(1.0), None, i)
        tr.global_step += 1
    assert rec.step_val_at == [0, 10, 20, 30, 40]
    assert len(rec.step_val_loss) == len(rec.step_val_at)
    assert len(rec.step_loss) == 50            # train curve untouched
