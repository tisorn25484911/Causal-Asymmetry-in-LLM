"""
Regression tests for the closed forms and the loss path.

Run:  pytest tests/ -q      (from the repo root)

These lock in the Phase 1 fixes so a future edit cannot quietly undo them.
IMPROVEMENT_PLAN.md Phase 4.6 asks for most of these; the B3 and B2 cases are
added because those are the two bugs that were silent in production.
"""

import math
import os
import pickle
import sys

import numpy as np
import pytest
import torch

# REORGANISATION_FIX_PLAN.md 5.6.  The modules are flat but live in two sibling
# directories, so both go on the path.  conftest.py at the repo root does this
# too, for the whole pytest session; keeping it here as well means
# `python tests/test_theory.py` and single-file runs outside pytest still work.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
from utils import entropy_bits, entropy_rate_coin              # noqa: E402


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


# ─────────────────────────────────────────────────────────────────────────────
# Vectorisation must not change the numbers — D2
# ─────────────────────────────────────────────────────────────────────────────
def test_heatmap_theory_matches_the_scalar_closed_form():
    """D2 vectorised heatmap_theory against statistical_complexity per cell."""
    from pq_experiment import heatmap_theory
    p = np.linspace(0.05, 0.95, 11)
    q = np.linspace(0.05, 0.95, 13)
    FW, BW, pv, qv = heatmap_theory(p, q)
    assert FW.shape == (len(p), len(q)) and BW.shape == (len(p), len(q))
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            assert FW[i, j] == pytest.approx(
                statistical_complexity(pi, qj, "forward"), abs=1e-12)
            assert BW[i, j] == pytest.approx(
                statistical_complexity(pi, qj, "backward"), abs=1e-12)


def test_vectorised_soft_label_ce_matches_explicit_loop():
    """The (B,T,V) fancy-index form must equal the original double loop."""
    rng = np.random.default_rng(0)
    B, T, V = 4, 60, 3
    p_model = rng.dirichlet(np.ones(V), size=(B, T))
    inp = rng.integers(0, V, size=(B, T))
    tp = rng.dirichlet(np.ones(V), size=V)

    total = 0.0
    for b in range(B):
        for t in range(T):
            total += -np.sum(tp[inp[b, t]] * np.log2(p_model[b, t] + 1e-12))
    vec = float((-(tp[inp] * np.log2(p_model + 1e-12)).sum(-1)).sum())
    assert vec == pytest.approx(total, rel=1e-9)


def test_vectorised_per_token_kl_matches_explicit_loop():
    rng = np.random.default_rng(1)
    B, T, V = 4, 60, 3
    p_model = rng.dirichlet(np.ones(V), size=(B, T))
    inp = rng.integers(0, V, size=(B, T))
    tc = rng.dirichlet(np.ones(V), size=V)

    kl_loop, cnt_loop = np.zeros(V), np.zeros(V)
    for b in range(B):
        for t in range(T):
            c = inp[b, t]
            kl_loop[c] += float(np.sum(
                tc[c] * np.log2(tc[c] / (p_model[b, t] + 1e-12) + 1e-12)))
            cnt_loop[c] += 1

    kl_bt = (tc[inp] * np.log2(tc[inp] / (p_model + 1e-12) + 1e-12)).sum(-1)
    kl_vec = np.bincount(inp.reshape(-1), weights=kl_bt.reshape(-1), minlength=V)
    cnt_vec = np.bincount(inp.reshape(-1), minlength=V)

    assert np.allclose(kl_vec, kl_loop, rtol=1e-9)
    assert np.array_equal(cnt_vec, cnt_loop)


# ─────────────────────────────────────────────────────────────────────────────
# Cheap wins must not change behaviour — D4
# ─────────────────────────────────────────────────────────────────────────────
def test_causal_mask_is_cached_correct_and_not_serialised():
    from OneHot_model import OneHotDecoder
    dev = torch.device("cpu")
    fw = OneHotDecoder(token_size=3, d_model=8, max_len=32, mode="forward", n_layers=1)
    bw = OneHotDecoder(token_size=3, d_model=8, max_len=32, mode="backward", n_layers=1)

    a, b = fw._causal_mask(8, dev), fw._causal_mask(8, dev)
    assert a is b, "mask should be built once per (T, device)"
    assert torch.equal(a, torch.tril(torch.ones(8, 8, dtype=torch.bool)).unsqueeze(0))
    assert torch.equal(bw._causal_mask(8, dev),
                       torch.triu(torch.ones(8, 8, dtype=torch.bool)).unsqueeze(0))
    # different T must not reuse the wrong cache entry
    assert fw._causal_mask(5, dev).shape == (1, 5, 5)
    # the cache must never reach a checkpoint
    assert not any("mask" in k for k in fw.state_dict())

    # tril and triu are exact mirror images -- the mask is NOT the fw/bw
    # confound (C2 says the positional encoding is)
    T = 9
    f = fw._causal_mask(T, dev)[0]
    r = bw._causal_mask(T, dev)[0]
    assert torch.equal(f, torch.flip(r, dims=[0, 1]))
    assert f.sum() == r.sum()


def test_perplexity_calculation_matches_concatenate_first():
    """D4 made this incremental; it must equal the old concatenate-then-CE."""
    from Model_analysis import perplexity_calculation
    from OneHot_model import cross_ent_onehot

    class M(torch.nn.Module):
        mode = "forward"
        def __init__(s):
            super().__init__(); s.p = torch.nn.Parameter(torch.zeros(1)); s.i = 0
        def forward(s, x):
            g = torch.Generator().manual_seed(s.i); s.i += 1
            return torch.randn(x.shape[0], x.shape[1], 3, generator=g)

    batches = [(torch.randint(0, 3, (8, 40)), torch.randint(0, 3, (8, 40)))
               for _ in range(4)]
    m = M()
    logits, targs = [], []
    with torch.no_grad():
        for inp, tg in batches:
            logits.append(m(inp)); targs.append(tg)
    ref = cross_ent_onehot(torch.cat(logits), torch.cat(targs))[1].item()

    m.i = 0
    got = perplexity_calculation(m, batches)
    assert got == pytest.approx(ref, rel=1e-5)


def test_umap_warm_up_is_idempotent():
    from Model_analysis import warm_up_umap
    assert warm_up_umap(15) == warm_up_umap(15)     # second call is a no-op


def test_diagnose_divergence_distinguishes_blow_up_from_a_bad_optimum():
    """
    A fold that reaches the entropy rate and then blows up cannot be spotted
    from its final loss alone -- it often partially recovers.  Measured on the
    sanity_check flower config, 4/6 MPS seeds do exactly this.
    """
    from Training_model import diagnose_divergence

    converged = [2.0, 1.2, 0.6, 0.50, 0.499, 0.499]
    assert not diagnose_divergence(converged)["diverged"]

    # reaches 0.499 then spikes to 40 and partially recovers to 1.07 --
    # this is the real shape observed in sanity_check fold 3
    blew_up = [1.9, 0.8, 0.499, 40.6, 12.0, 1.07]
    d = diagnose_divergence(blew_up)
    assert d["diverged"]
    assert d["min"] == pytest.approx(0.499)
    assert d["peak_after_min"] == pytest.approx(40.6)
    assert d["final"] == pytest.approx(1.07)

    # a fold merely stuck high never dips low, so it is NOT flagged as
    # divergence -- the convergence filter is what catches that case
    stuck = [1.9, 1.6, 1.5, 1.49, 1.49]
    assert not diagnose_divergence(stuck)["diverged"]

    assert not diagnose_divergence([])["diverged"]


# ─────────────────────────────────────────────────────────────────────────────
# Causal-state discovery — C1
# ─────────────────────────────────────────────────────────────────────────────
class _StubDecoder(torch.nn.Module):
    """
    A model whose predictive distribution is a chosen function of the token,
    so the true causal-state partition is known exactly.
    """
    def __init__(self, rows):
        super().__init__()
        self.mode = "forward"
        self.rows = np.asarray(rows, float)          # (V, V) P(next | token)
        V = self.rows.shape[0]
        # latent = one-hot(token); output_prj maps it to log P(next | token),
        # so softmax(output_prj(latent)) reproduces `rows` exactly.
        self.emb = torch.nn.Parameter(torch.eye(V), requires_grad=False)
        self.output_prj = torch.nn.Linear(V, V, bias=False)
        with torch.no_grad():
            self.output_prj.weight.copy_(
                torch.log(torch.tensor(self.rows.T, dtype=torch.float32) + 1e-9))
        self.last_encodings = None

    def forward(self, tokens):
        x = self.emb[tokens.long()]
        self.last_encodings = x.detach()
        return self.output_prj(x)

    def parameters(self, *a, **k):
        return super().parameters(*a, **k)


def _stub_loader(V, n_seq=120, T=8, seed=0):
    rng = np.random.default_rng(seed)
    toks = torch.from_numpy(rng.integers(0, V, size=(n_seq, T)))
    return [(toks[i:i+30], toks[i:i+30]) for i in range(0, n_seq, 30)]


def test_recover_causal_states_merges_tokens_with_equal_futures():
    """
    C1.  Tokens 0 and 2 share a predictive distribution, so they are ONE causal
    state even though they are two tokens -- exactly the coin's forward case.
    A fixed-k estimator cannot discover this; a threshold on the predictive
    distribution can.
    """
    from Model_analysis import recover_causal_states
    rows = [[0.6, 0.4, 0.0],      # token 0  \  same future
            [0.0, 0.2, 0.8],      # token 1   > distinct
            [0.6, 0.4, 0.0]]      # token 2  /  same future  -> 2 states
    m = _StubDecoder(rows)
    r = recover_causal_states(m, _stub_loader(3), use_t="last",
                              max_batches=None, state_tol=0.10, n_pts=500)
    assert r["k_hat"] == 2, f"expected 2 causal states, got {r['k_hat']}"
    # and tokens 0 and 2 must land in the SAME cluster
    lab, tok = r["labels"], r["tokens"]
    assert set(lab[tok == 0]) == set(lab[tok == 2])
    assert set(lab[tok == 1]).isdisjoint(set(lab[tok == 0]))


def test_recover_causal_states_keeps_genuinely_distinct_futures_apart():
    from Model_analysis import recover_causal_states
    rows = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
    m = _StubDecoder(rows)
    r = recover_causal_states(m, _stub_loader(3), use_t="last",
                              max_batches=None, state_tol=0.10, n_pts=500)
    assert r["k_hat"] == 3


def test_recover_causal_states_reports_a_stability_profile():
    """
    The threshold is a real free parameter -- the smallest true state gap
    ranges from 0.135 to 0.612 across this repo's configs -- so k_hat at one
    threshold is not defensible alone.  The profile must always come back.
    """
    from Model_analysis import recover_causal_states
    rows = [[0.6, 0.4, 0.0], [0.0, 0.2, 0.8], [0.6, 0.4, 0.0]]
    r = recover_causal_states(_StubDecoder(rows), _stub_loader(3), use_t="last",
                              max_batches=None, state_tol=0.10, n_pts=500)
    assert len(r["stability"]) >= 5
    assert all(isinstance(t, float) and isinstance(k, int) for t, k in r["stability"])
    assert r["plateau"] == 2                     # 2 survives the widest span
    # k is non-increasing as the threshold grows (agglomerative merging)
    ks = [k for _, k in r["stability"]]
    assert all(a >= b for a, b in zip(ks, ks[1:])), ks


def test_S_hat_is_entropy_over_recovered_clusters():
    from Model_analysis import recover_causal_states
    rows = [[0.6, 0.4, 0.0], [0.0, 0.2, 0.8], [0.6, 0.4, 0.0]]
    r = recover_causal_states(_StubDecoder(rows), _stub_loader(3), use_t="last",
                              max_batches=None, state_tol=0.10, n_pts=500)
    counts = np.bincount(r["labels"], minlength=r["k_hat"]).astype(float)
    assert r["S_hat"] == pytest.approx(entropy_bits(counts / counts.sum()))
    assert r["S_hat"] <= math.log2(r["k_hat"]) + 1e-9      # S <= log2(k)


# ─────────────────────────────────────────────────────────────────────────────
# Distances on the probability simplex
# ─────────────────────────────────────────────────────────────────────────────
def test_distance_matrices_match_reference_implementations():
    from scipy.spatial.distance import pdist, squareform
    from Model_analysis import (euclidean_distance_matrix, tv_distance_matrix,
                                js_distance_matrix)
    P = np.random.default_rng(0).dirichlet(np.ones(4), size=40)
    assert np.allclose(euclidean_distance_matrix(P), squareform(pdist(P, "euclidean")))
    assert np.allclose(tv_distance_matrix(P), squareform(0.5 * pdist(P, "cityblock")))
    J = js_distance_matrix(P)
    assert np.isfinite(J).all(), "scipy's jensenshannon returns nan here; ours must not"
    assert np.allclose(J, J.T)
    assert np.allclose(np.diag(J), 0.0)


def test_js_is_a_bounded_metric():
    """
    JS distance = sqrt(JSD) is bounded by 1 bit and satisfies the triangle
    inequality (Endres & Schindelin 2003) -- the property a distance-threshold
    clusterer needs and that raw KL does not have.
    """
    from Model_analysis import js_distance_matrix
    P = np.random.default_rng(1).dirichlet(np.ones(5), size=30)
    J = js_distance_matrix(P)
    assert J.max() <= 1.0 + 1e-9
    for i, j, k in [(0,1,2), (3,7,11), (5,9,20)]:
        assert J[i, k] <= J[i, j] + J[j, k] + 1e-9


def test_js_survives_zero_probabilities_where_kl_would_diverge():
    """A trained model drives probabilities to ~1e-6; KL would be infinite."""
    from Model_analysis import js_distance_matrix
    P = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.5, 0.5, 0.0]])
    J = js_distance_matrix(P)
    assert np.isfinite(J).all()
    assert J[0, 1] == pytest.approx(1.0, abs=1e-6)   # disjoint support -> 1 bit


def test_metrics_agree_on_a_clean_two_state_case():
    from Model_analysis import recover_causal_states
    rows = [[0.6, 0.4, 0.0], [0.0, 0.2, 0.8], [0.6, 0.4, 0.0]]
    for metric in ("euclidean", "tv", "js"):
        r = recover_causal_states(_StubDecoder(rows), _stub_loader(3),
                                  use_t="last", max_batches=None,
                                  metric=metric, n_pts=400)
        assert r["k_hat"] == 2, f"{metric} gave {r['k_hat']}"
        assert r["metric"] == metric


def test_split_indices_reproduce_the_pipeline_split_exactly():
    """
    The train/held-out panels are only meaningful if the split is the SAME one
    training used.  random_split draws randperm(N, generator) and slices it, so
    it depends only on N and the seed -- not on dataset contents.
    """
    import io, contextlib
    from plot_state_clusters import split_indices
    from Training_model import test_train_validation, make_chunked_loader
    from Data_generation import CoinDataset, coin_generation

    set_seed(0)
    d, _ = coin_generation(num_samples=200, seq_len=300, p=0.4, q=0.8)
    loader = make_chunked_loader(CoinDataset(d, seq_len=300), 64, 16, seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        tv_loader, te_loader = test_train_validation(loader, 0.20, 0.80, seed=0)

    tr_idx, te_idx = split_indices(200, seed=0, test_ratio=0.20)
    assert list(tv_loader.dataset.indices) == tr_idx
    assert list(te_loader.dataset.indices) == te_idx
    assert set(tr_idx).isdisjoint(te_idx)
    assert len(te_idx) == 40


# ─────────────────────────────────────────────────────────────────────────────
# Weight decay — WEIGHT_DECAY_PLAN.md §4
# ─────────────────────────────────────────────────────────────────────────────
def test_adamw_zero_decay_equals_adam():
    """
    The guarantee the whole change rests on: at weight_decay=0.0 AdamW takes the
    same steps as Adam, so every number produced before this change stays valid.
    """
    def run(opt_cls, **kw):
        torch.manual_seed(0)
        m = torch.nn.Linear(6, 4)
        opt = opt_cls(m.parameters(), lr=1e-2, **kw)
        g = torch.Generator().manual_seed(1)
        for _ in range(20):
            x = torch.randn(8, 6, generator=g)
            opt.zero_grad()
            m(x).pow(2).mean().backward()
            opt.step()
        return [p.detach().clone() for p in m.parameters()]

    a = run(torch.optim.Adam)
    w = run(torch.optim.AdamW, weight_decay=0.0)
    for pa, pw in zip(a, w):
        assert torch.allclose(pa, pw, atol=0, rtol=0), \
            f"max|delta| = {(pa - pw).abs().max().item():.3e}"


def test_weight_decay_defaults_to_zero():
    """
    Guards against adopting PyTorch's AdamW default of 0.01.  AdamW is not a
    drop-in for Adam, and a silent 0.01 would change every existing result.
    """
    from configs import BASE, CONFIGS
    from OneHot_model import OneHotDecoder
    assert BASE["weight_decay"] == 0.0

    # DISCRETE is the one deliberate exception, and it is exempt by NAME so that
    # a nonzero value appearing anywhere else still fails.  It sets 0.01 because
    # the discrete bottleneck diverges without it: cross-entropy on the backward
    # coin's point-mass row has no finite minimiser, and held-out CE climbs again
    # past ~150 epochs.  It has no historical results to invalidate -- the guard
    # exists to stop a SILENT 0.01 changing numbers that already exist.
    EXEMPT = {"DISCRETE"}
    for name, cfg in CONFIGS.items():
        if name in EXEMPT:
            assert cfg["weight_decay"] != 0.0, (
                f"{name} is exempt because its nonzero decay is deliberate; "
                "if it is now 0.0, remove it from EXEMPT")
            continue
        assert cfg["weight_decay"] == 0.0, f"{name} ships a nonzero default"
    assert OneHotDecoder(token_size=3, d_model=8, max_len=16).weight_decay == 0.0


def test_weight_decay_reaches_the_optimiser():
    """
    The threading bug this change risks is a value accepted and then dropped --
    exactly what happened to n_layers (B11), unnoticed because every config
    happened to agree.  Assert it arrives.
    """
    from OneHot_model import OneHotDecoder
    m = OneHotDecoder(token_size=3, d_model=8, max_len=16, weight_decay=0.03)
    opt = m.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["weight_decay"] == 0.03


def test_weight_decay_threads_through_train_model():
    """The same, one level up: train_model must forward it to the model."""
    from Training_model import train_model

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            x = torch.randint(0, 3, (8,), generator=torch.Generator().manual_seed(i))
            return x, x

    loader = torch.utils.data.DataLoader(_DS(), batch_size=2)
    rec = train_model(loader, num_token=3, d_model=8, max_len=8, max_epochs=1,
                      mode="forward", accelerator="cpu", weight_decay=0.03)
    assert rec.model.weight_decay == 0.03
    assert rec.model.configure_optimizers().param_groups[0]["weight_decay"] == 0.03


def test_old_checkpoints_still_load():
    """
    Adding a keyword argument adds no parameters and no buffers, so every .pt
    written before this change must still load.
    """
    import glob
    import json as _json
    from OneHot_model import OneHotDecoder
    from utils import repo_path
    # REORGANISATION_FIX_PLAN.md 5.6.  Root-anchored, and via repo_path so the
    # test does not depend on the cwd.  The bare relative glob this replaced
    # matched nothing after the results tree moved under All_Results/, and the
    # skip below turned that into a silently passing test -- the worst outcome
    # for a backward-compatibility check.
    paths = sorted(glob.glob(repo_path("All_Results/results_quick/models/*_fw.pt")))
    if not paths:
        pytest.skip("no saved checkpoints in this clone")
    side = os.path.splitext(paths[0])[0] + ".json"
    if not os.path.exists(side):
        pytest.skip("checkpoint has no sidecar")
    meta = _json.load(open(side))
    m = OneHotDecoder(token_size=meta["token_size"], d_model=meta["d_model"],
                      max_len=meta["max_len"], n_layers=meta["n_layers"],
                      mode="forward")
    m.load_state_dict(torch.load(paths[0], map_location="cpu"))


# ═══════════════════════════════════════════════════════════════════════════
# REMAINING_WORK_PLAN.md Phase 0 — the repair work these lock in
# ═══════════════════════════════════════════════════════════════════════════
def test_repo_path_anchors_relative_and_passes_absolute():
    """REORGANISATION_FIX_PLAN 4.2.  Output paths must not depend on the cwd."""
    from utils import REPO_ROOT, repo_path
    assert repo_path("/tmp/scratch") == "/tmp/scratch"
    assert repo_path("All_Results/x") == os.path.join(REPO_ROOT, "All_Results/x")
    assert os.path.isdir(os.path.join(REPO_ROOT, "Transformer_model"))


def test_every_default_out_root_resolves_to_existing_data():
    """
    A default out_root that resolves to a NEW EMPTY directory is the silent
    failure that cost a 23 h sweep: mkdir is exist_ok=True and a missing pickle
    reads as "nothing done yet", so the run restarts instead of resuming.
    """
    from configs import CONFIGS
    from utils import repo_path
    import run_statistical_trj as trj
    import run_sweep_experiment as swp
    roots = [CONFIGS[c]["out_root"] for c in ("SMOKE", "QUICK", "LARGE")]
    roots += [trj.OUT_ROOT_DEFAULT, swp.OUT_ROOT_DEFAULT]
    for rel in roots:
        assert rel.startswith("All_Results/"), f"{rel} is not under All_Results/"
        p = repo_path(rel)
        assert os.path.isdir(p), f"default out_root does not exist: {p}"
        assert os.listdir(p), f"default out_root is empty: {p}"


def test_every_entry_point_imports_without_pythonpath():
    """
    The test that would have caught the reorganisation breakage.  PYTHONPATH is
    scrubbed so the shell environment cannot mask a missing bootstrap.
    """
    import glob
    import subprocess
    from utils import REPO_ROOT
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    targets = []
    for d in ("Experimental_setup", "Transformer_model"):
        for f in sorted(glob.glob(os.path.join(REPO_ROOT, d, "*.py"))):
            if "__main__" in open(f).read():
                targets.append((d, os.path.basename(f)[:-3]))
    assert len(targets) >= 8, f"expected the known entry points, found {targets}"
    for d, mod in targets:
        r = subprocess.run([sys.executable, "-c",
                            f"import sys; sys.path.insert(0, {os.path.join(REPO_ROOT, d)!r}); "
                            f"import {mod}"],
                           capture_output=True, text=True, env=env, cwd="/")
        assert r.returncode == 0, f"{d}/{mod}.py fails to import:\n{r.stderr[-600:]}"


def test_save_pkl_is_atomic(tmp_path):
    """
    C1.  A crash mid-write must leave the previous bundle intact.  The old
    implementation truncated on open(), so this is the regression that matters.
    """
    from utils import save_pkl
    p = str(tmp_path / "bundle.pkl")
    save_pkl({"generation": 1}, p)
    original = open(p, "rb").read()

    class Boom:
        def __reduce__(self):
            raise KeyboardInterrupt("simulated interrupt mid-dump")

    with pytest.raises(KeyboardInterrupt):
        save_pkl({"generation": 2, "bad": Boom()}, p)
    assert open(p, "rb").read() == original, "the old bundle was damaged"
    assert pickle.load(open(p, "rb"))["generation"] == 1
    assert not list(tmp_path.glob("*.tmp.*")), "a temp file was left behind"


def test_save_run_config_merges_instead_of_overwriting(tmp_path):
    """
    C3.  A coin-only invocation must not erase the flower grid recorded by an
    earlier run into the same folder -- that is how results_sweep lost its coin
    provenance while its pickle held 100 coin cells.
    """
    import json as _json
    from utils import save_run_config
    p = str(tmp_path / "run_config_QUICK.json")
    save_run_config(p, {"coin_grid": [0.25], "flower_grid": None, "repeats": 1,
                        "processes": ["a"]},
                    keep_if_unset=("coin_grid", "flower_grid"),
                    union_lists=("processes",))
    save_run_config(p, {"coin_grid": None, "flower_grid": [2, 4], "repeats": 1,
                        "processes": ["b"]},
                    keep_if_unset=("coin_grid", "flower_grid"),
                    union_lists=("processes",))
    d = _json.load(open(p))
    assert d["coin_grid"] == [0.25], "the earlier grid was erased"
    assert d["flower_grid"] == [2, 4]
    assert d["processes"] == ["a", "b"], "the process list did not accumulate"
    assert len(d["invocations"]) == 2


def test_trajectory_resume_refuses_incompatible_records():
    """
    C2.  Repeats are pooled into one paired statistic, so resuming across a
    different process, base seed or config would corrupt delta_CE rather than
    merely waste time.  A wrong resume is worse than no resume.
    """
    from configs import CONFIGS
    import run_statistical_trj as trj

    class Args:
        config, khat, redo = "SMOKE", False, False

    cfg = dict(CONFIGS["SMOKE"]); cfg["seed"] = 0
    spec = trj.flower_spec(cfg, 2, 6)
    good = dict(spec=spec, config="SMOKE", base_seed=0, khat=False,
                runs=[{"seed": 0}, {"seed": 1}])
    assert trj.resumable(good, spec, cfg, Args)[0] is True

    for label, prev, c in [
            ("base_seed",  good,                              {**cfg, "seed": 7}),
            ("config",     {**good, "config": "QUICK"},       cfg),
            ("khat",       {**good, "khat": True},            cfg),
            ("dice",       {**good, "spec": trj.flower_spec({**cfg, "flower_dice_seed": 999}, 2, 6)}, cfg),
            ("n,m",        {**good, "spec": trj.flower_spec(cfg, 2, 8)}, cfg),
            ("seed gap",   {**good, "runs": [{"seed": 0}, {"seed": 5}]}, cfg),
            ("empty",      {**good, "runs": []},              cfg)]:
        ok, why = trj.resumable(prev, spec, c, Args)
        assert ok is False, f"resume wrongly allowed across differing {label}"
        assert why, "a refusal must say why"


def test_flower_merge_default_is_the_historical_rule():
    """
    C10.  `merge_tol=None` must reproduce the round-to-9-decimals rule exactly,
    including for spiky dice where the two rules genuinely disagree -- every C-
    the repo has reported was computed that way.
    """
    from Model_analysis import flower_complexity
    from utils import entropy_bits

    def historical(n, m, dp):
        dp = np.asarray(dp, dtype=float)
        pi, col, merged = dp.mean(axis=0), dp.sum(axis=0), {}
        for j in range(m):
            if col[j] <= 0:
                continue
            key = tuple(np.round(dp[:, j] / col[j], 9))
            merged[key] = merged.get(key, 0.0) + pi[j]
        return 1.0 + 0.5 * entropy_bits(list(merged.values()))

    for alpha in (0.2, 1.0, 5.0):
        for n in (1, 2, 4, 6):
            for m in (2, 4, 8):
                for seed in (0, 7, 74):
                    dp = np.random.default_rng(seed).dirichlet(np.full(m, alpha), size=n)
                    assert flower_complexity(n, m, dice_probs=dp)[1] == historical(n, m, dp)


def test_flower_null_control_holds_under_both_merge_rules():
    """The n=1 process is exactly time-reversible; C- must equal C+ either way."""
    from Model_analysis import flower_complexity
    for m in (2, 4, 8):
        dp = np.full((1, m), 1.0 / m)
        for tol in (None, 1e-9, 1e-6):
            cp, cm = flower_complexity(1, m, dice_probs=dp, merge_tol=tol)
            assert abs(cm - cp) < 1e-12, f"null control broken at m={m}, tol={tol}"


# ══════════════════════════════════════════════════════════════════════════
# DISCRETE_V2_PLAN.md — the resolved hyperparameters and the new diagnostic
# ══════════════════════════════════════════════════════════════════════════

def _discrete_cfg(**over):
    from configs import CONFIGS
    cfg = dict(CONFIGS["DISCRETE"])
    cfg.update(over)
    return cfg


def test_discrete_hparams_gives_5V_and_V():
    """K = 5V and state_dim = V, for every vocabulary in the baseline set."""
    from Model_analysis import discrete_hparams
    cfg = _discrete_cfg()
    for V in (3, 5, 6, 8, 10, 20):
        h = discrete_hparams(cfg, V, 32)
        assert h["n_states"] == 5 * V, f"K != 5V at V={V}"
        assert h["state_dim"] == V, f"state_dim != V at V={V}"


def test_usage_beta_is_computed_from_geometry_not_hardcoded():
    """
    beta must equal 1/(batch * chunk_len) and MOVE when either factor moves.

    The whole point of computing it: a literal decouples beta from N, and the
    penalty collapses the bottleneck above ~6e-4 at lr=1e-3.
    """
    from Model_analysis import discrete_hparams
    for batch, chunk in ((32, 256), (32, 128), (64, 256), (8, 64)):
        cfg = _discrete_cfg(train_chunk_len=chunk)
        got = discrete_hparams(cfg, 3, batch)["usage_beta"]
        assert got == pytest.approx(1.0 / (batch * chunk)), (batch, chunk)
    # and the DISCRETE geometry specifically
    assert discrete_hparams(_discrete_cfg(), 3, 32)["usage_beta"] == pytest.approx(1 / 8192)


def test_usage_beta_explicit_value_overrides_the_rule():
    """The beta sweep works by setting cfg['usage_beta']; nothing else should."""
    from Model_analysis import discrete_hparams
    h = discrete_hparams(_discrete_cfg(usage_beta=0.2), 3, 32)
    assert h["usage_beta"] == 0.2


def test_dead_beta_machinery_is_gone():
    """
    usage_target / usage_beta_gap|high|low tuned beta from log2(K) - H(p_bar),
    the OLD penalty's uniformity gap.  The penalty is now beta * H(p_bar), so
    those keys select nothing and must not linger to be read by mistake.
    """
    import Model_analysis
    cfg = _discrete_cfg()
    for key in ("usage_target", "usage_beta_gap", "usage_beta_high", "usage_beta_low"):
        assert key not in cfg, f"{key} still in the config"
    for fn in ("usage_beta_shared", "usage_beta_for"):
        assert not hasattr(Model_analysis, fn), f"{fn} still defined"


def test_specs_carry_resolved_hparams_and_true_counts():
    """
    Every spec gets K=5V, state_dim=V, one beta -- and keeps the TRUE causal
    state counts separately, which is what K vs k_discovered plots against.
    """
    import run_statistical_trj as R
    cfg = _discrete_cfg(usage_beta_fixed=None)
    specs = R.process_specs(cfg)
    assert len(specs) == 7
    betas = {s["usage_beta"] for s in specs}
    assert len(betas) == 1 and betas.pop() == pytest.approx(1 / 8192)
    for s in specs:
        V = s["num_token"]
        assert s["n_states"] == 5 * V, s["tag"]
        assert s["state_dim"] == V, s["tag"]
        assert s["true_k_fw"] >= 1 and s["true_k_bw"] >= 1, s["tag"]
        # K must leave slack over BOTH true counts, or the count cannot be found
        assert s["n_states"] > max(s["true_k_fw"], s["true_k_bw"]), s["tag"]


def test_h_state_given_token_is_zero_for_a_function_of_the_token():
    """
    The true causal state IS a deterministic function of the current token in
    both arms of both processes, so the diagnostic must read exactly 0 there --
    and rise to log2(k) for an assignment independent of the token.
    """
    from torch.utils.data import DataLoader, Dataset
    from DiscreteCausal_analysis import causal_state_report

    class _Stub:
        def __init__(self, K, V, rule):
            self.n_states, self.token_size, self.mode = K, V, "forward"
            self.rule, self.training, self.usage_beta = rule, False, 0.0
            self.last_states = torch.zeros(1, 1, dtype=torch.long)
        def state_assignment(self, *a): return self.last_states
        def eval(self): pass
        def train(self): pass
        def parameters(self): yield torch.zeros(1)
        def __call__(self, x): self.last_states = self.rule(x)
        @property
        def state_matrix(self): return torch.zeros(self.n_states, 2)
        def emission_table(self): return torch.zeros(self.n_states, self.token_size)

    class _DS(Dataset):
        def __init__(self, x): self.x = x
        def __len__(self): return len(self.x)
        def __getitem__(self, i):
            t = torch.tensor(self.x[i]); return t[:-1], t[1:]

    X = np.random.default_rng(0).integers(0, 3, size=(64, 60))
    ld = DataLoader(_DS(X), batch_size=32)

    r = causal_state_report(_Stub(3, 3, lambda x: x.clone()), ld, min_pos=5)
    assert r["h_state_given_token"] == pytest.approx(0.0, abs=1e-9)

    r = causal_state_report(_Stub(3, 3, lambda x: (x == 1).long()), ld, min_pos=5)
    assert r["h_state_given_token"] == pytest.approx(0.0, abs=1e-9)

    g = torch.Generator().manual_seed(1)
    r = causal_state_report(
        _Stub(4, 3, lambda x: torch.randint(0, 4, x.shape, generator=g)), ld, min_pos=5)
    assert r["h_state_given_token"] == pytest.approx(2.0, abs=0.02)
