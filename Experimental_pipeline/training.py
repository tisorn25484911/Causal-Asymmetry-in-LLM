"""
Training one model, and the plumbing around it.

Dropped from the old tree on the way over:

  ChunckDataset / make_chunked_loader / make_analysis_loader
      All three existed to carve a short training window out of a long
      generated sequence and then hand analysis a matching window.  With
      `processes.generate` producing exactly `seq_len` tokens there is nothing
      to carve: training and analysis run at the same length by construction, so
      the extrapolation hazard those functions dodged cannot arise.

  train_test_val_pipeline
      5-fold cross-validation.  The folds were the source of statistics; repeats
      are, and a minimum-over-five-folds is not a sample from the same
      distribution as a single draw.

  save_weights / load_weight_meta / check_weight_meta
      Nothing downstream reads a checkpoint; the pickles are the product.
"""
import contextlib
import gc
import json
import os
import pickle
import random

import lightning as L
import numpy as np
import torch
import torch.utils.data as tud

from models import build_model, cross_ent_onehot


# ══════════════════════════════════════════════════════════════════════════
# FILESYSTEM / HOUSEKEEPING
# ══════════════════════════════════════════════════════════════════════════
def mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_pkl(obj, path: str):
    """
    Atomically replace `path` with a pickle of `obj`.

    `open(path, "wb")` TRUNCATES before a byte is written, so a crash mid-dump
    leaves a partial file that a resume then reads as "nothing completed yet"
    and silently discards the run.  Writing to a sibling temp and os.replace()-ing
    is atomic within a filesystem: a reader sees the whole old file or the whole
    new one.  The pid in the temp name keeps two concurrent runs from corrupting
    each other.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_run_config(cfg: dict, path: str):
    """The config a run actually executed at, beside its results."""
    with open(path, "w") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in cfg.items()}, f, indent=2, default=str)


def cleanup():
    """
    GC + accelerator flush between models.

    `torch.mps.empty_cache()` is deliberately NOT called -- it segfaults on
    PyTorch <= 2.1.  Models are moved to CPU before analysis instead, which
    frees the Metal heap that way.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        with contextlib.suppress(Exception):
            torch.mps.synchronize()


def to_cpu_for_analysis(model):
    """Move a trained model to CPU in place and return it."""
    model.cpu()
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        with contextlib.suppress(Exception):
            torch.mps.synchronize()
    return model


def set_seed(seed: int = 0) -> int:
    """
    Seed every global RNG, and return the seed.

    Necessary but not sufficient on its own: the four models of one repeat also
    need the seeded split below, because a global seed does not stop the first
    model from ADVANCING a shared RNG before the second starts.  Sequence
    generation does not rely on this at all -- `processes.generate` takes an
    explicit Generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@contextlib.contextmanager
def quiet(enabled: bool = True):
    """
    Swallow the per-model training chatter, re-emitting it if the block raises.

    Lightning prints a table per model; at 30 repeats x 4 models that is the
    whole log.  Nothing is lost on failure.
    """
    if not enabled:
        yield None
        return
    import io
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield buf
    except BaseException:
        print(buf.getvalue())
        raise


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION / DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════
def eval_ce(model, loader) -> tuple[float, float]:
    """
    (dataset-level cross-entropy in bits, matching perplexity 2**CE).

    Two things this gets right that the obvious version does not:

    1. Perplexity is derived from the AGGREGATED CE, so PPL = 2**CE holds by
       construction.  `mean(2**CE_batch)` is biased high by Jensen, by an amount
       that grows with the per-batch spread.
    2. The accumulator is TOKEN-weighted, not batch-weighted.  DataLoader
       defaults to drop_last=False, so a ragged final batch would otherwise
       carry the same weight as a full one.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_ce, total_tok = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = model._split(batch, getattr(model, "mode", "forward"))
            inputs, targets = inputs.to(device), targets.to(device)
            loss, _ = cross_ent_onehot(model(inputs), targets)
            n = targets.numel()
            total_ce  += loss.item() * n
            total_tok += n

    if was_training:
        model.train()
    if total_tok == 0:
        return float("nan"), float("nan")
    ce = total_ce / total_tok
    return ce, 2.0 ** ce


def diagnose_divergence(step_loss, rise_factor: float = 2.0) -> dict:
    """
    Detect a run that reached a good solution and then blew up.

    Returns {min, min_at, peak_after_min, final, diverged}.

    This cannot be diagnosed from the final loss alone: the failure is not a bad
    local optimum -- the loss reaches the entropy rate FIRST and then rises.  The
    cause is that both processes contain deterministic transitions, and
    cross-entropy on a deterministic transition has no finite optimum, so
    training drives the logit gap toward infinity until a step overshoots.
    """
    sl = np.asarray(step_loss, dtype=float)
    if sl.size == 0:
        return dict(min=float("nan"), min_at=-1, peak_after_min=float("nan"),
                    final=float("nan"), diverged=False)
    i    = int(np.argmin(sl))
    lo   = float(sl[i])
    peak = float(sl[i:].max())
    return dict(min=lo, min_at=i, peak_after_min=peak, final=float(sl[-1]),
                diverged=bool(peak > max(rise_factor * lo, lo + 0.5)))


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════
class Recorder(L.Callback):
    """
    Per-step training curve, and a validation curve on its own cadence.

    The two cadences are gated SEPARATELY on purpose.  Recording the training
    loss is free -- it is already returned by training_step -- whereas each
    validation point costs a full pass over the val set.  One shared stride
    would either make validation ruinous or throw away 96% of the free series.

    The epoch mean accumulates before the recording gate, so it stays a true
    epoch mean at any stride.
    """

    def __init__(self, val_loader=None, val_every_n_steps: int = 25):
        super().__init__()
        self.val_loader        = val_loader
        self.val_every_n_steps = max(1, int(val_every_n_steps))
        self.step_loss, self.step_at         = [], []
        self.step_val_loss, self.step_val_at = [], []
        self.epoch_loss = []
        self.model = None
        self._sum, self._count = 0.0, 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = (outputs["loss"] if isinstance(outputs, dict) else outputs)
        if loss is None:
            return
        loss = float(loss.detach().float().cpu().item())
        self._sum += loss
        self._count += 1

        step = trainer.global_step
        self.step_loss.append(loss)
        self.step_at.append(step)

        if self.val_loader is not None and step % self.val_every_n_steps == 0:
            val_ce, _ = eval_ce(pl_module, self.val_loader)
            self.step_val_loss.append(val_ce)
            self.step_val_at.append(step)
            pl_module.train()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._count:
            self.epoch_loss.append(self._sum / self._count)
        self._sum, self._count = 0.0, 0


def split_loader(dataset, batch: int, test_ratio: float = 0.20, seed: int = 0):
    """
    (train_loader, test_loader) from one dataset, with a SEEDED split.

    The seed is not decoration: every model of one repeat must get the same
    hold-out set, or the four are not comparable.  `random_split` without an
    explicit generator consumes the global torch RNG, so each call would draw a
    different split.
    """
    n_test  = int(len(dataset) * test_ratio)
    n_train = len(dataset) - n_test
    tr, te = tud.random_split(dataset, [n_train, n_test],
                              generator=torch.Generator().manual_seed(seed))
    return (tud.DataLoader(tr, batch_size=batch, shuffle=True,
                           generator=torch.Generator().manual_seed(seed)),
            tud.DataLoader(te, batch_size=batch, shuffle=False))


def train_model(train_loader, embed_type: str, val_loader=None, *,
                num_token: int, d_model: int, max_len: int, max_epochs: int,
                lr: float, mode: str, n_layers: int, weight_decay: float,
                accelerator: str = "auto", val_every_n_steps: int = 25,
                n_states=None, state_dim=None, tau: float = 1.0,
                usage_beta: float = 0.0) -> Recorder:
    """
    Train one model and return the Recorder holding its curves and the model.

    A note on reproducibility: seeding makes the models of one repeat *paired* --
    they share the split and the initialisation -- but it does NOT make a run
    bit-reproducible across invocations on Apple MPS, where trajectories diverge
    from the first backward pass.  `accelerator="cpu"` is exactly repeatable at
    roughly 6x the wall clock.
    """
    model = build_model(
        embed_type, token_size=num_token, d_model=d_model, max_len=max_len,
        lr=lr, mode=mode, n_layers=n_layers, weight_decay=weight_decay,
        n_states=n_states, state_dim=state_dim, tau=tau, usage_beta=usage_beta)

    rec = Recorder(val_loader=val_loader, val_every_n_steps=val_every_n_steps)
    trainer = L.Trainer(
        max_epochs=max_epochs, accelerator=accelerator, devices="auto",
        log_every_n_steps=5, callbacks=[rec],
        # Nothing here reads lightning_logs/*.ckpt, and left on it drops a
        # checkpoint directory per model for nothing.
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, train_loader)
    rec.model = model
    return rec
