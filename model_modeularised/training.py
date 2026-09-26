"""
Training one model, and the plumbing around it (ported from
Experimental_pipeline/training.py; the learning-rate schedules were left there).

    train_model   the one entry point: builds the model for `arch` ("transformer"
                  or "gru"), fits it, and returns the Recorder holding the
                  curves and the model
    Recorder      training / validation curves, the tau and gradient-norm traces,
                  and the best-validation snapshot
    eval_ce, per_position_ce   cross-entropy in bits
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

import GRU
import Transformer
from decoder_base import cross_ent_onehot
from tau_schedue import TauSchedule, is_scheduled, parse_tau

# The ONLY place that maps an architecture name to its builder.
ARCHITECTURES = {"transformer": Transformer.build_model, "gru": GRU.build_model,
                 "gru_feedback": GRU.build_feedback_model}


# ══════════════════════════════════════════════════════════════════════════
# FILESYSTEM / HOUSEKEEPING
# ══════════════════════════════════════════════════════════════════════════
def mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_pkl(obj, path: str):
    """
    Atomically replace `path` with a pickle of `obj`: write a sibling temp and
    os.replace() it, so a crash mid-dump never leaves a truncated file.
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
    GC + accelerator flush between models.  `torch.mps.empty_cache()` is
    deliberately NOT called -- it segfaults on PyTorch <= 2.1.
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
    Seed every global RNG, and return the seed.  Sequence generation does not
    rely on this -- generator.generate takes an explicit numpy Generator -- and
    the split does not either (process_generator.split_loader is seeded).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@contextlib.contextmanager
def quiet(enabled: bool = True):
    """Swallow the per-model training chatter, re-emitting it if the block raises."""
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

    Token-weighted, and the perplexity is taken from the AGGREGATED CE, so
    PPL = 2**CE holds by construction (mean(2**CE_batch) is biased high).
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


def per_position_ce(model, loader) -> np.ndarray:
    """(N, T) cross-entropy in bits of every scored token, in loader order."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets = model._split(batch, getattr(model, "mode", "forward"))
            logp = torch.log_softmax(model(inputs.to(device)), dim=-1)
            nll = -logp.gather(-1, targets.to(device).unsqueeze(-1)).squeeze(-1)
            out.append((nll / np.log(2)).cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(out, axis=0)


def diagnose_divergence(step_loss, rise_factor: float = 2.0) -> dict:
    """
    Detect a run that reached a good solution and then blew up:
    {min, min_at, peak_after_min, final, diverged}.  Deterministic transitions
    give cross-entropy no finite optimum, so the failure shows as a loss that
    reaches the entropy rate FIRST and then rises.
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

    Per step (free): the training objective, tau, and the global gradient norm
    (before any clipping -- the number to look at if an RNN's gradients blow up).
    Every `val_every_n_steps`: the validation CE and, for a discrete model, the
    number of distinct states the training batch used.

    With `keep_best` the weights are snapshotted at every validation point that
    improves on the best so far, so train_model can hand back the model at its
    validation minimum.  The snapshot is selected on `val_loader`, so a CE
    reported on that same loader is a selected quantity; score a fresh draw of
    the process for an unbiased one.
    """

    def __init__(self, val_loader=None, val_every_n_steps: int = 25, keep_best: bool = False,
                 progress_every: int = 0):
        super().__init__()
        self.val_loader        = val_loader
        self.val_every_n_steps = max(1, int(val_every_n_steps))
        self.keep_best         = bool(keep_best)
        self.progress_every    = int(progress_every)     # 0 = silent
        self._t0 = None
        self.step_loss, self.step_at         = [], []
        self.step_val_loss, self.step_val_at = [], []
        self.step_tau, self.step_grad_norm   = [], []
        self.step_states_used                = []    # on the validation cadence
        self.epoch_loss = []
        self.model = None
        self.best_val, self.best_step, self.best_state = float("inf"), -1, None
        self.restored_best = False
        self._sum, self._count = 0.0, 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        sq = [p.grad.detach().float().pow(2).sum() for p in pl_module.parameters()
              if p.grad is not None]
        self.step_grad_norm.append(float(torch.stack(sq).sum().sqrt()) if sq else 0.0)

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
        self.step_tau.append(float(getattr(pl_module, "tau", float("nan"))))

        if self.val_loader is not None and step % self.val_every_n_steps == 0:
            states = getattr(pl_module, "last_states", None)     # the training batch's
            if states is not None:
                self.step_states_used.append(int(torch.unique(states).numel()))
            val_ce, _ = eval_ce(pl_module, self.val_loader)
            self.step_val_loss.append(val_ce)
            self.step_val_at.append(step)
            if self.keep_best and val_ce < self.best_val:
                self.best_val, self.best_step = float(val_ce), int(step)
                self.best_state = {k: v.detach().clone()
                                   for k, v in pl_module.state_dict().items()}
            pl_module.train()

        if self.progress_every and step % self.progress_every == 0:
            # sys.__stderr__, not sys.stderr: this has to get past quiet()'s redirect.
            import sys
            import time
            self._t0 = self._t0 or time.time()
            val = self.step_val_loss[-1] if self.step_val_loss else float("nan")
            used = self.step_states_used[-1] if self.step_states_used else -1
            print(f"    step {step:6d}/{trainer.estimated_stepping_batches} | val CE {val:.4f} | "
                  f"best {self.best_val:.4f} @ {self.best_step} | tau {self.step_tau[-1]:.3f} | "
                  f"states {used} | {(time.time() - self._t0) / 60:.1f} min",
                  file=sys.__stderr__, flush=True)

    def on_train_start(self, trainer, pl_module):
        import time
        self._t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._count:
            self.epoch_loss.append(self._sum / self._count)
        self._sum, self._count = 0.0, 0


def train_model(train_loader, arch: str, embed_type: str = "discrete", val_loader=None, *,
                token_size: int, d_model: int, max_len: int, max_epochs: int,
                lr: float, n_layers: int, weight_decay: float, mode: str = "forward",
                accelerator: str = "auto", val_every_n_steps: int = 25,
                n_states=None, state_dim=None, tau: float | str = 1.0,
                usage_beta: float = 0.0, restore_best: bool = False,
                gradient_clip_val=None, progress_every: int = 0,
                gumbel: bool = False) -> Recorder:
    """
    Train one model and return the Recorder holding its curves and the model.

    `arch` is "transformer", "gru" or "gru_feedback"; everything else is shared,
    so they are trained identically on the same loaders.

    `tau` is a float (constant) or a schedule spec such as "geom:5:0.5"; see
    tau_schedue.py.  The callback is attached only for a genuinely varying
    schedule and only for the discrete head -- build_model discards tau for
    "onehot", so attaching it there would write an attribute nothing reads.

    `restore_best` returns the model at its best validation point instead of at
    the last step; `rec.best_step` / `rec.best_val` say which.  It needs a
    `val_loader`.  Restoring is clean for the discrete model: tau shapes only the
    gradient, never the forward value.

    `gradient_clip_val` is off by default: tau_experiment/study5 found clipping
    does not help the transformer, and both architectures train the same way
    unless a run shows it is needed (Recorder.step_grad_norm).

    `gumbel` (discrete head only, off by default) trains with Gumbel-softmax
    straight-through instead of the deterministic argmax; evaluation is always
    the deterministic argmax.  See causal_matrix._learn_causal_state.forward.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {sorted(ARCHITECTURES)}, got {arch!r}")
    if restore_best and val_loader is None:
        raise ValueError("restore_best needs a val_loader to select the checkpoint on")
    _, tau_fn = parse_tau(tau)
    # build_model needs a FLOAT: the head divides the state logits by it.
    model = ARCHITECTURES[arch](
        embed_type, token_size=token_size, d_model=d_model, max_len=max_len,
        lr=lr, mode=mode, n_layers=n_layers, weight_decay=weight_decay,
        n_states=n_states, state_dim=state_dim, tau=tau_fn(0.0),
        usage_beta=usage_beta)
    if gumbel:
        if embed_type != "discrete":
            raise ValueError("gumbel applies to the discrete head only")
        model.gumbel = True

    rec = Recorder(val_loader=val_loader, val_every_n_steps=val_every_n_steps,
                   keep_best=restore_best, progress_every=progress_every)
    callbacks = [rec]
    if embed_type == "discrete" and is_scheduled(tau):
        callbacks.append(TauSchedule(tau_fn, len(train_loader) * max_epochs))
    trainer = L.Trainer(
        max_epochs=max_epochs, accelerator=accelerator, devices="auto",
        log_every_n_steps=5, callbacks=callbacks, gradient_clip_val=gradient_clip_val,
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, train_loader)
    if restore_best and rec.best_state is not None:
        model.load_state_dict(rec.best_state)
        rec.restored_best = True
    rec.model = model
    return rec
