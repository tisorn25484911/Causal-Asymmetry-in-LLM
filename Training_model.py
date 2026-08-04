import random

import lightning as L
from OneHot_model import OneHotDecoder, cross_ent_onehot
import torch
import torch.utils.data as tud
import numpy as np

"""
# dataloader: number of samples (5000), each loader includes (input, target) pair
# input:  (B, T) token indices in [0, V-1]
# target: (B, T) token indices in [0, V-1]
"""
# ─────────────────────────────────────────────────────────────────────────────
# class and functions for large training sequence
# ────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 0) -> int:
    """
    Seed every global RNG this repo draws from, and return the seed.

    Covers `random`, `numpy.random` (which `coin_generation` and
    `flower_process_generation` use via np.random.rand / randint / choice)
    and `torch` (model init, DataLoader shuffling, random_split).  Call this
    at the top of every runner so a run is reproducible end to end.

    IMPROVEMENT_PLAN.md A2.  Note this is necessary but not sufficient on its
    own: the pairing of the forward and backward arms also needs the seeded
    `random_split` and the deterministic `ChunckDataset` below, because a
    global seed does not stop the forward run from *advancing* a shared RNG
    before the backward run starts.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


class ChunckDataset(tud.Dataset):
    """
    Fixed-length windows into a dataset of longer sequences.

    Each sequence gets ONE window, chosen once at construction from `seed`.
    `__getitem__` is therefore a pure function of `idx` -- IMPROVEMENT_PLAN.md
    A2.

    Why this changed.  The offsets used to come from a single stateful
    `np.random.default_rng(seed)` consumed at access time, which had three
    consequences, none of them intended:

      * The forward and backward pipelines were handed the *same* loader
        object.  The forward run advanced the generator, so the backward run
        trained on different windows of the same sequences -- and the headline
        number is a difference between the two arms, expected to be a few
        hundredths of a bit.
      * `sample_seq = next(iter(loader_fw))[0][0]` in the runners advanced it
        once more before either arm started.
      * Validation re-drew fresh windows at *every* recorded step, so the
        step_val_loss curve carried window noise on top of learning signal,
        and "best fold" was selected partly on which fold drew easy windows.

    The cost is that a sequence now only ever contributes one window instead
    of a new one each epoch.  That is a deliberate trade: exact pairing of the
    two arms is what the whole study rests on, whereas the lost window
    diversity is recoverable by generating more sequences.
    """

    def __init__(self, base: tud.Dataset, chunck_len: int, seed: int = 0):
        self.base = base
        self.chunck_len = chunck_len
        self.seed = seed

        # One offset per sequence, drawn once, in index order so the table is
        # a pure function of (seed, sequence lengths) and independent of the
        # order in which items are later requested.
        rng = np.random.default_rng(seed)
        self.offsets = []
        for i in range(len(base)):
            inp, _ = base[i]
            T = inp.shape[0]
            span = T - chunck_len + 1
            self.offsets.append(0 if span <= 1 else int(rng.integers(0, span)))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        input, target = self.base[idx]          # (T,)
        T = input.shape[0]
        if T <= self.chunck_len:
            return input, target
        start = self.offsets[idx]
        end   = start + self.chunck_len
        return input[start:end], target[start:end]


def make_chunked_loader(
    dataset: tud.Dataset, chunk_len: int, batch_size: int,
    shuffle: bool = True, seed: int = 0,
) -> tud.DataLoader:
    return tud.DataLoader(
        ChunckDataset(dataset, chunk_len, seed=seed),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=0,
        persistent_workers=False,
    )
def _loader(dataset: tud.Dataset, batch_size: int, shuffle: bool = False) -> tud.DataLoader:
    """
    Plain full-length loader (no chunking).

    WARNING: do NOT use this for analysis when the model was trained on chunks.
    See make_analysis_loader below for why.  It remains here because the
    post-hoc evaluators load models whose max_len covers the full sequence.
    """
    return tud.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=0, persistent_workers=False,
    )


def make_analysis_loader(
    dataset: tud.Dataset, chunk_len: int, batch_size: int, seed: int = 0,
) -> tud.DataLoader:
    """
    Loader for post-training analysis, at the SAME sequence length the model
    was trained on.

    Why not the full-length loader.  IMPROVEMENT_PLAN.md B4 says analysis
    should run on full sequences, quoting README:233 -- "analysis on full
    sequences remains in-distribution".  **That claim is false whenever
    chunk_len < full sequence length.**  A chunk is fed to the model as a
    standalone sequence, so the positional encoding index is the position
    *within the chunk*: the model only ever sees PE indices [0, chunk_len).
    Evaluating at full length asks it to extrapolate to positions it has never
    been trained on.

    Measured on the sanity_check coin checkpoint (trained at chunk 512,
    evaluated at T=999, H_inf = 1.0):

        positions    0-511  (trained)     CE = 1.0249   <- converged
        positions  512-998  (never seen)  CE = 1.5563
        whole sequence                    CE = 1.2840   <- what was reported

    So the "unconverged" CE was extrapolation error, not a failure to learn.

    Worse, it biases the FW/BW comparison *asymmetrically*, which is fatal for
    a study measuring a difference between the two arms.
    statistical_complexity_empirical reads the forward arm at use_t="last" and
    the backward arm at use_t="first" (correctly -- those are the max-context
    positions, A3).  At full length that is position T-1 for forward, which is
    untrained, versus position 0 for backward, which is trained.  The two arms
    are then measured under different amounts of extrapolation, and the
    difference between them is contaminated by an artefact of chunked
    training.  At chunk length both positions are in-distribution.

    Windows are drawn deterministically (as in ChunckDataset) from a seed
    offset from the training one, so analysis sees fresh windows of the same
    length rather than exactly the windows that were trained on.
    """
    return tud.DataLoader(
        ChunckDataset(dataset, chunk_len, seed=seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helper: evaluate a (possibly mid-training) model on a DataLoader
# Temporarily switches to eval mode, then restores train mode.
# ─────────────────────────────────────────────────────────────────────────────
def _eval_loss_on_loader(model, loader) -> tuple[float, float]:
    """
    Return (dataset-level cross-entropy in bits, matching perplexity 2**CE).

    Two bugs fixed here (IMPROVEMENT_PLAN.md B2).

    1. This used to return ``mean(2**CE_batch)`` rather than ``2**mean(CE)``.
       By Jensen's inequality mean(2^x) >= 2^mean(x), so the reported
       perplexity was biased *high* by an amount that grows with the variance
       of the per-batch CE.  On a realistic spread (CE in {0.9, 1.4, 1.1, 0.6})
       that is 2.0411 vs 2.0000 -- a +0.041 bias, the same order as the
       delta-CE the whole study is trying to measure.  The perplexity is now
       derived from the aggregated CE, so the identity PPL = 2**CE holds by
       construction and log2(PPL) is exactly the CE.

    2. Both accumulators were divided by ``n_batches``, so a ragged final batch
       (DataLoader defaults to drop_last=False) carried the same weight as a
       full one.  cross_ent_onehot returns a per-token mean, so the aggregate
       has to be token-weighted: accumulate sum(CE * n_tokens) / sum(n_tokens).

    This now matches perplexity_calculation in Model_analysis.py, which was
    already correct -- the repo contained both a right and a wrong estimator.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_ce_tokens, total_tokens = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            # mirror training_step convention for backward mode
            if getattr(model, "mode", "forward") == "backward":
                targets, inputs = batch
            inputs  = inputs.to(device)
            targets = targets.to(device)
            logits  = model(inputs)                     # (B, T, V)
            loss, _ = cross_ent_onehot(logits, targets)
            n_tok    = targets.numel()
            total_ce_tokens += loss.item() * n_tok
            total_tokens    += n_tok

    if was_training:
        model.train()                                   # restore training mode

    if total_tokens == 0:
        return float("nan"), float("nan")
    ce = total_ce_tokens / total_tokens
    return ce, 2.0 ** ce


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Callback
# ─────────────────────────────────────────────────────────────────────────────
class Record_training(L.Callback):
    """
    Records per-step and per-epoch metrics during training.

    New parameters
    --------------
    val_loader : DataLoader | None
        When supplied, the model is evaluated on this loader at every
        recorded gradient step, producing step_val_loss and step_val_ppl.
    """

    def __init__(
        self,
        record_every_n_steps: int = 1,
        record_latents: bool = False,
        record_probs: bool = False,
        max_batches_per_epoch: int | None = None,
        val_loader=None,
        val_every_n_steps: int = 25,
    ):
        super().__init__()
        self.record_every_n_steps   = record_every_n_steps
        self.record_latents         = record_latents
        self.record_probs           = record_probs
        self.max_batches_per_epoch  = max_batches_per_epoch
        self.val_loader             = val_loader
        # D1: the train curve and the validation pass are gated SEPARATELY.
        # Recording the training loss is free (it is already computed and
        # returned by training_step), whereas each validation point costs a
        # full pass over the val set.  The plan's single record_every_n_steps
        # =25 would have thrown away 96% of the free series to save the
        # expensive one -- QUICK has only 100 steps per fold, so it would have
        # produced FOUR points on every curve.
        self.val_every_n_steps      = max(1, int(val_every_n_steps))

        # ── per-step series ──────────────────────────────────────────────
        self.step_loss     = []     # training loss at each recorded step
        self.step_ppl      = []     # training perplexity at each recorded step
        self.step_at       = []     # global_step for each step_loss entry
        self.step_val_loss = []     # validation loss, on its own cadence
        self.step_val_ppl  = []     # validation perplexity, on its own cadence
        # D1: val points are sparser than train points, so they need their own
        # x-axis.  Plotting step_val_loss against its list index would compress
        # the whole run into the first few percent of the axis.
        self.step_val_at   = []     # global_step for each step_val_loss entry

        # ── per-epoch series ─────────────────────────────────────────────
        self.epoch_loss = []
        self.val_loss   = []        # optional (legacy Lightning metric)

        # ── optional heavy data ──────────────────────────────────────────
        self.latents = []
        self.probs   = []

        self._epoch_loss_sum   = 0.0
        self._epoch_loss_count = 0

    # ------------------------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Optional cap
        if self.max_batches_per_epoch is not None and batch_idx >= self.max_batches_per_epoch:
            return

        # ── Training loss ────────────────────────────────────────────────
        if isinstance(outputs, dict):
            loss_tensor = outputs.get("loss", None)
            if loss_tensor is None:
                return
            loss = loss_tensor.detach().float().cpu().item()
        else:
            loss = outputs.detach().float().cpu().item()

        # ── Epoch accumulation — ALWAYS, before the sub-sampling gate ────
        # IMPROVEMENT_PLAN.md B12.  This used to sit *below* the
        # `global_step % record_every_n_steps` early return, so `epoch_loss`
        # was the mean over recorded steps only.  At record_every_n_steps=1
        # that is every step and the two agree, which is why it was never
        # noticed -- but the moment D1 raises it to 25, `epoch_loss` silently
        # becomes the mean over every 25th step: a different estimator, no
        # error raised.  Accumulating here keeps it a true epoch mean at any
        # recording stride.
        self._epoch_loss_sum   += loss
        self._epoch_loss_count += 1

        step = trainer.global_step

        # ── Train series — FREE, so keep it at full resolution ───────────
        if step % self.record_every_n_steps == 0:
            self.step_loss.append(loss)
            self.step_at.append(step)
            # B1: the loss is in BITS (cross_ent_onehot divides by ln 2), so
            # the matching perplexity is 2**loss.  This was math.exp(loss),
            # which put panel 3 (train PPL) on a different scale from panel 4
            # (val PPL, base 2) while plotting them side by side.
            self.step_ppl.append(2.0 ** loss)

        # ── Validation — EXPENSIVE, so gate it separately (D1) ───────────
        # This is the whole of the D1 speedup: a full pass over the val set was
        # running at every gradient step.  Measured overhead of that, against
        # training with no validation recording at all: 1.63x at QUICK sizes
        # (10 train / 3 val batches) and 2.58x at LARGE sizes (20 train / 5 val
        # batches).  The plan estimated 2-4x; the upper half of that is right
        # for LARGE, the lower for QUICK.
        #
        # Note the pq sweep calls train_model WITHOUT a val_loader, so it never
        # paid this cost and D1 does not speed it up.
        if self.val_loader is not None:
            if step % self.val_every_n_steps == 0:
                val_loss, val_ppl = _eval_loss_on_loader(pl_module, self.val_loader)
                self.step_val_loss.append(val_loss)
                self.step_val_ppl.append(val_ppl)
                self.step_val_at.append(step)
                pl_module.train()                       # ensure still in train mode
        else:
            # Fall back to whatever Lightning logged, on the train cadence.
            if step % self.record_every_n_steps == 0:
                val_loss_metric = trainer.callback_metrics.get("val_loss", None)
                if val_loss_metric is not None:
                    self.val_loss.append(val_loss_metric.detach().float().cpu().item())

        if step % self.record_every_n_steps != 0:
            return

        # ── Optional heavy data ──────────────────────────────────────────
        if self.record_latents or self.record_probs:
            inputs, targets = batch
            if getattr(pl_module, "mode", "forward") == "backward":
                targets, inputs = batch
            device = pl_module.device
            inputs = inputs.to(device)
            with torch.no_grad():
                logits = pl_module(inputs)
            if self.record_probs:
                p = torch.softmax(logits, dim=-1).detach().cpu()
                self.probs.append(p)
            if self.record_latents:
                z = pl_module.last_encodings.detach().cpu()
                self.latents.append(z)

    # ------------------------------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_loss_count > 0:
            self.epoch_loss.append(self._epoch_loss_sum / self._epoch_loss_count)
        self._epoch_loss_sum   = 0.0
        self._epoch_loss_count = 0


# ─────────────────────────────────────────────────────────────────────────────
# train_model
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    train_loader,
    num_token: int   = 3,
    d_model:   int   = 20,
    max_len:   int   = 15,
    max_epochs: int  = 5,
    lr:        float = 1e-2,
    mode:      str   = "forward",
    embed_type: str  = "onehot",
    val_loader       = None,
    n_layers:  int   = 2,
    accelerator: str = "auto",
    val_every_n_steps: int = 25,
):
    """
    Trains a OneHotDecoder and returns a Record_training object.

    `n_layers` is now a real parameter — IMPROVEMENT_PLAN.md B11.  It used to
    be absent here and never forwarded, so OneHotDecoder's default of 2 applied
    to every trained model regardless of intent, while
    LLM_asymmetry_testing.py and Test_data_eval.py both instantiated with
    n_layers=cfg["n_layers"].  Both configs happened to say 2, so it worked —
    silently.  Changing either config would have made load_state_dict fail with
    a shape error rather than a clear message.

    recorder.model          - trained model
    recorder.step_loss      - training loss per gradient step
    recorder.step_ppl       - training perplexity per gradient step
    recorder.step_val_loss  - validation loss per gradient step (if val_loader given)
    recorder.step_val_ppl   - validation perplexity per gradient step (if val_loader given)
    recorder.epoch_loss     - epoch-averaged training loss
    """
    # C5: WordEmbDecoder is gone — no runner ever set embed_type="wordemb".
    if embed_type != "onehot":
        raise ValueError(f"Invalid embed_type: {embed_type!r}. Only 'onehot' "
                         "remains; WordEmbDecoder was deleted (C5).")
    model = OneHotDecoder(
        token_size=num_token, d_model=d_model, max_len=max_len, lr=lr, mode=mode,
        n_layers=n_layers,
    )

    recorder = Record_training(
        record_every_n_steps=1,
        record_latents=False,
        record_probs=False,
        max_batches_per_epoch=None,
        val_loader=val_loader,
        val_every_n_steps=val_every_n_steps,   # D1
    )

    # A note on reproducibility (A2).  Seeding makes the forward and backward
    # arms *paired* -- within one run they share splits, folds, batch order and
    # init -- and that is what delta_CE needs.  It does NOT make a run
    # bit-reproducible across invocations on Apple MPS: measured here, model
    # init is identical and the first loss matches to 8 decimals, then the
    # trajectories diverge from the first backward pass (~1e-3 by step 2).
    # That is the MPS backend, not this code -- the same test on CPU is exactly
    # repeatable.  CPU is ~6x slower at these sizes (LARGE: 304 vs 49 ms/step,
    # i.e. ~13 hr -> ~78 hr), so MPS stays the default and `accelerator="cpu"`
    # is available when an exactly reproducible run is worth the wall-clock.
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices="auto",
        log_every_n_steps=5,
        callbacks=[recorder],
        # Nothing in this repo ever reads lightning_logs/version_N/*.ckpt --
        # the weights that get used are written by save_weights(), and every
        # curve that gets plotted comes off `recorder`.  Left on, one
        # Main_call.py run drops ~230 checkpoint dirs on disk for nothing.
        # See IMPROVEMENT_PLAN.md D5.
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_loader)

    recorder.model = model
    return recorder


# ─────────────────────────────────────────────────────────────────────────────
# Utility: initial train/test split
# ─────────────────────────────────────────────────────────────────────────────
def test_train_validation(
    train_loader, test_ratio, train_ratio, seed: int = 0
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Hold-out split.  `seed` is required for the forward and backward arms to
    receive the SAME hold-out test set -- IMPROVEMENT_PLAN.md A2.

    `random_split` used to be called with no `generator=`, so it consumed the
    global torch RNG.  This function is called once per direction, so the two
    arms drew different test sets and different train/val pools.  Seeding the
    fold *permutation* (which was already done) does not help, because the
    pool being permuted was itself different.
    """
    if test_ratio + train_ratio != 1.0:
        s = test_ratio + train_ratio
        test_ratio  /= s
        train_ratio /= s
        print(f"Ratios normalised → test={test_ratio:.4f}, train={train_ratio:.4f}")

    total_samples = len(train_loader.dataset)
    num_test      = int(total_samples * test_ratio)
    num_train     = total_samples - num_test

    print(f"Total samples: {total_samples}")
    print(f"Splitting into: {num_train} train, {num_test} test samples")

    train_set, test_set = torch.utils.data.random_split(
        train_loader.dataset, [num_train, num_test],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Actual split sizes: Train={len(train_set)}, Test={len(test_set)}")

    train_loader_out = torch.utils.data.DataLoader(
        train_set, batch_size=train_loader.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader_out = torch.utils.data.DataLoader(
        test_set, batch_size=train_loader.batch_size, shuffle=False
    )
    return train_loader_out, test_loader_out


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def train_test_val_pipeline(
    train_loader,
    test_ratio:  tuple = (0.20, 0.80),  # (test_ratio, train_ratio)
    n_folds:     int   = 5,
    embed_type:  str   = "onehot",
    num_token:   int   = 3,
    d_model:     int   = 20,
    max_len:     int   = 1000,
    max_epochs:  int   = 5,
    lr:          float = 1e-2,
    mode:        str   = "forward",
    save_plot:   str   = "cv_results.png",
    seed:        int   = 0,
    n_layers:    int   = 2,
    accelerator: str   = "auto",
    val_every_n_steps: int = 25,
):
    """
    Full cross-validation pipeline with step-level training + validation curves.

    `seed` makes the forward and backward arms *paired* -- IMPROVEMENT_PLAN.md
    A2.  Called twice with the same seed and the same dataset, the two arms get
    an identical hold-out test set, identical fold membership, identical batch
    order, and identical weight initialisation.  The only remaining difference
    is the one under study: the attention mask and the batch convention.  That
    turns delta-CE from an unpaired comparison contaminated by split noise into
    a paired one where per-fold differences are meaningful.

    Returns
    -------
    dict with keys:
        best_fold      - 0-based index of the winning fold
        best_recorder  - Record_training from the best fold
        best_model     - trained model from the best fold
        fold_val_loss  - list[float]: val CE (bits) per fold
        fold_val_ppl   - list[float]: val perplexity per fold
        fold_test_loss - list[float]: test CE (bits) for *every* fold model
        fold_test_ppl  - list[float]: test PPL for *every* fold model
        test_loss      - float: test CE loss for the best-fold model
        test_ppl       - float: test perplexity for the best-fold model
        all_recorders  - list of all fold recorders
        seed           - the seed used, so a paired comparison can assert on it
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    batch_size   = train_loader.batch_size
    full_dataset = train_loader.dataset
    N            = len(full_dataset)
    test, train_val = test_ratio

    # ── 1. Hold-out test split ───────────────────────────────────────────
    print(f"Initial split ratios: test={test}, train_val={train_val}")
    trainval_loader, test_loader = test_train_validation(
        train_loader, test_ratio=test, train_ratio=train_val, seed=seed
    )
    trainval_set = trainval_loader.dataset
    test_set     = test_loader.dataset
    n_test       = len(test_set)
    n_trainval   = N - n_test

    print(f"\n{'='*65}")
    print(f"  {n_folds}-Fold Cross-Validation Pipeline")
    print(f"{'='*65}")
    print(f"  Total samples    : {N}")
    print(f"  Held-out test    : {n_test}  ({test*100:.0f}%)")
    print(f"  Train+Val pool   : {n_trainval}")
    print(f"  Folds            : {n_folds}")
    print(f"  Mode             : {mode}  |  embed: {embed_type}")
    print(f"  d_model={d_model}  n_layers={n_layers}  max_epochs={max_epochs}"
          f"  lr={lr}  seed={seed}")
    print(f"{'='*65}\n")

    # ── 2. n-fold cross-validation ───────────────────────────────────────
    fold_size     = n_trainval // n_folds
    indices       = torch.randperm(
        n_trainval, generator=torch.Generator().manual_seed(seed)
    ).tolist()

    fold_val_loss  = []
    fold_val_ppl   = []
    all_recorders  = []

    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        val_idx   = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = indices[: fold * fold_size] + indices[(fold + 1) * fold_size :]

        fold_train = torch.utils.data.Subset(trainval_set, train_idx)
        fold_val   = torch.utils.data.Subset(trainval_set, val_idx)

        fold_train_loader = torch.utils.data.DataLoader(
            fold_train, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed + fold),
        )
        fold_val_loader = torch.utils.data.DataLoader(
            fold_val, batch_size=batch_size, shuffle=False
        )

        print(f"  Train samples: {len(fold_train)}  |  Val samples: {len(fold_val)}")

        # Identical weight init across directions for this fold, so the paired
        # difference is not partly an initialisation difference (A2).
        torch.manual_seed(seed * 1000 + fold)

        # ── Train – pass the val loader so step-level val is recorded ──
        recorder = train_model(
            fold_train_loader,
            num_token=num_token,
            d_model=d_model,
            max_len=max_len,
            max_epochs=max_epochs,
            lr=lr,
            mode=mode,
            embed_type=embed_type,
            val_loader=fold_val_loader,
            n_layers=n_layers,            # B11
            accelerator=accelerator,
            val_every_n_steps=val_every_n_steps,   # D1
        )
        all_recorders.append(recorder)

        # Final val metrics (whole val set, one pass)
        val_loss, val_ppl = _eval_loss_on_loader(recorder.model, fold_val_loader)
        fold_val_loss.append(val_loss)
        fold_val_ppl.append(val_ppl)

        print(f"  ✓ Fold {fold+1}  val loss: {val_loss:.4f}  |  val ppl: {val_ppl:.4f}")

    # ── 3. Select best fold ──────────────────────────────────────────────
    best_fold     = int(np.argmin(fold_val_loss))
    best_recorder = all_recorders[best_fold]
    best_model    = best_recorder.model

    print(f"\n{'='*65}")
    print(f"  Best fold: {best_fold + 1}  "
          f"(val loss={fold_val_loss[best_fold]:.4f}, "
          f"val ppl={fold_val_ppl[best_fold]:.4f})")
    print(f"{'='*65}")

    # ── 4. Evaluate ALL fold models on the held-out test set ─────────────
    print("\n  Evaluating ALL fold models on the held-out test set …")
    fold_test_loss = []
    fold_test_ppl  = []
    for fold, rec in enumerate(all_recorders):
        t_loss, t_ppl = _eval_loss_on_loader(rec.model, test_loader)
        fold_test_loss.append(t_loss)
        fold_test_ppl.append(t_ppl)
        marker = "  ★ best" if fold == best_fold else ""
        print(f"    Fold {fold+1}: test loss={t_loss:.4f}  test ppl={t_ppl:.4f}{marker}")

    test_loss = fold_test_loss[best_fold]
    test_ppl  = fold_test_ppl[best_fold]

    # ── 5. Visualisation (6 subplots, 3×2) ──────────────────────────────
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]
    line_colors = plt.cm.tab10(np.linspace(0, 0.9, n_folds))
    bar_color   = "#3a86ff"          # single colour for all bar charts

    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.38, wspace=0.22
    )

    # ── (1) Training loss / step ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i, rec in enumerate(all_recorders):
        is_best = (i == best_fold)
        ax1.plot(
            rec.step_loss,
            color=line_colors[i],
            linewidth=2.5 if is_best else 1.0,
            alpha=1.0 if is_best else 0.40,
            label=fold_labels[i] + (" ★" if is_best else ""),
        )
    ax1.set_title("Training Loss / Step", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Gradient Step")
    ax1.set_ylabel("Loss (bits)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── (2) Validation loss / step ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, rec in enumerate(all_recorders):
        is_best = (i == best_fold)
        if rec.step_val_loss:
            # D1: val runs on its own cadence -> use the recorded step index.
            xs = getattr(rec, "step_val_at", None) or range(len(rec.step_val_loss))
            ax2.plot(
                list(xs), rec.step_val_loss,
                color=line_colors[i],
                linewidth=2.5 if is_best else 1.0,
                alpha=1.0 if is_best else 0.40,
                marker="." if len(rec.step_val_loss) < 120 else None, ms=3,
                label=fold_labels[i] + (" ★" if is_best else ""),
            )
    ax2.set_title(f"Validation Loss / Step  (every {val_every_n_steps} steps)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Gradient Step")
    ax2.set_ylabel("Val Loss (bits)")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── (3) Training perplexity / step ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for i, rec in enumerate(all_recorders):
        is_best = (i == best_fold)
        ax3.plot(
            rec.step_ppl,
            color=line_colors[i],
            linewidth=2.5 if is_best else 1.0,
            alpha=1.0 if is_best else 0.40,
            label=fold_labels[i] + (" ★" if is_best else ""),
        )
    ax3.set_title("Training Perplexity / Step", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Gradient Step")
    ax3.set_ylabel("Perplexity")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # ── (4) Validation perplexity / step ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for i, rec in enumerate(all_recorders):
        is_best = (i == best_fold)
        if rec.step_val_ppl:
            xs = getattr(rec, "step_val_at", None) or range(len(rec.step_val_ppl))
            ax4.plot(
                list(xs), rec.step_val_ppl,
                color=line_colors[i],
                linewidth=2.5 if is_best else 1.0,
                alpha=1.0 if is_best else 0.40,
                marker="." if len(rec.step_val_ppl) < 120 else None, ms=3,
                label=fold_labels[i] + (" ★" if is_best else ""),
            )
    ax4.set_title(f"Validation Perplexity / Step  (every {val_every_n_steps} steps)",
                  fontsize=12, fontweight="bold")
    ax4.set_xlabel("Gradient Step")
    ax4.set_ylabel("Val Perplexity")
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    # ── (5) Bar: final validation PPL per fold ───────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    alphas = [1.0 if i == best_fold else 0.3 for i in range(n_folds)]
    for i, (label, val) in enumerate(zip(fold_labels, fold_val_ppl)):
        ax5.bar(label, val, color=bar_color, alpha=alphas[i], edgecolor="k", linewidth=0.8)
    ax5.bar_label(
        ax5.containers[0], fmt="%.4f", padding=3, fontsize=9
    )
    ax5.axhline(
        np.mean(fold_val_ppl), color="crimson", linestyle="--", linewidth=1.4,
        label=f"Mean = {np.mean(fold_val_ppl):.4f}",
    )
    ax5.set_title("Final Validation PPL per Fold", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Val Perplexity")
    ax5.set_ylim(0, max(fold_val_ppl) * 1.22)
    ax5.legend(fontsize=9)
    ax5.grid(True, axis="y", alpha=0.3)

    # ── (6) Bar: test PPL for every fold model ───────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    for i, (label, val) in enumerate(zip(fold_labels, fold_test_ppl)):
        ax6.bar(label, val, color=bar_color, alpha=alphas[i], edgecolor="k", linewidth=0.8)
    ax6.bar_label(
        ax6.containers[0], fmt="%.4f", padding=3, fontsize=9
    )
    ax6.axhline(
        np.mean(fold_test_ppl), color="crimson", linestyle="--", linewidth=1.4,
        label=f"Mean = {np.mean(fold_test_ppl):.4f}",
    )
    ax6.set_title("Test PPL per Fold Model (held-out set)", fontsize=12, fontweight="bold")
    ax6.set_ylabel("Test Perplexity")
    ax6.set_ylim(0, max(fold_test_ppl) * 1.22)
    ax6.legend(fontsize=9)
    ax6.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"{n_folds}-Fold CV  |  mode={mode}  embed={embed_type}  "
        f"d_model={d_model}  lr={lr}  epochs={max_epochs}\n"
        f"Best: Fold {best_fold+1}  "
        f"(val ppl={fold_val_ppl[best_fold]:.4f},  "
        f"test ppl={test_ppl:.4f})",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    plt.savefig(save_plot, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ CV visualisation saved → {save_plot}")
    plt.close(fig)

    # ── 6. Console summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  HELD-OUT TEST SET  (best fold = {best_fold + 1})")
    print(f"{'='*65}")
    print(f"  Test loss (best fold) : {test_loss:.4f} bits")
    print(f"  Test ppl  (best fold) : {test_ppl:.4f}")
    print(f"\n  Cross-Validation Summary")
    print(f"  {'Fold':<8} {'Val Loss':>10} {'Val PPL':>10} {'Test PPL':>10}")
    print(f"  {'-'*44}")
    for i in range(n_folds):
        marker = "  ★" if i == best_fold else ""
        print(
            f"  {fold_labels[i]:<8} "
            f"{fold_val_loss[i]:>10.4f} "
            f"{fold_val_ppl[i]:>10.4f} "
            f"{fold_test_ppl[i]:>10.4f}{marker}"
        )
    print(f"  {'-'*44}")
    print(
        f"  {'Mean':<8} "
        f"{float(np.mean(fold_val_loss)):>10.4f} "
        f"{float(np.mean(fold_val_ppl)):>10.4f} "
        f"{float(np.mean(fold_test_ppl)):>10.4f}"
    )
    print(
        f"  {'Std':<8} "
        f"{float(np.std(fold_val_loss)):>10.4f} "
        f"{float(np.std(fold_val_ppl)):>10.4f} "
        f"{float(np.std(fold_test_ppl)):>10.4f}"
    )
    print(f"{'='*65}\n")

    return {
        "best_fold"     : best_fold,
        "best_recorder" : best_recorder,
        "best_model"    : best_model,
        "fold_val_loss" : fold_val_loss,
        "fold_val_ppl"  : fold_val_ppl,
        "fold_test_loss": fold_test_loss,     # per-fold CE in bits — needed for
        "fold_test_ppl" : fold_test_ppl,      # the paired delta-CE (A2)
        "test_loss"     : test_loss,
        "test_ppl"      : test_ppl,
        "all_recorders" : all_recorders,
        "seed"          : seed,
        "n_layers"      : n_layers,
        "val_every_n_steps": val_every_n_steps,
    }