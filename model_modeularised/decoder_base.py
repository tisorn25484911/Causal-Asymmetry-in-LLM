"""
What every decoder shares, whatever turns tokens into per-position vectors.

    cross_ent_onehot   the loss, in bits
    _Decoder           the LightningModule both architectures subclass: the stored
                       hyperparameters, the training step, the optimiser and the
                       analysis surface.  A subclass supplies `_encode`, the
                       (B,T) tokens -> (B,T,D) map -- the attention stack in
                       Transformer.py, the GRU in GRU.py.

The discrete causal head that can sit on top of `_encode` is in causal_matrix.py.  Which
outputs the loss counts is scoring.py's (score_start): the continuous data mode scores
only the suffix of each row, the legacy mode every output.
"""
import math

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from scoring import masked_ce_bits


def cross_ent_onehot(logits, targets):
    """
    Mean cross-entropy in BITS, and the matching perplexity 2**CE.

    F.cross_entropy rather than a hand-rolled `-(target * softmax.log2()).sum()`:
    the latter returns NaN once any non-target probability underflows to exactly
    0 (that term is 0 * -inf), which is reachable here rather than hypothetical.
    Both processes contain deterministic transitions -- coin token 2 -> token 1
    w.p. 1, a flower roll is always followed by a selection -- and training on a
    deterministic transition drives the logit gap toward infinity.  F.cross_entropy
    uses log-sum-exp internally, so it is exact at any logit scale.  It returns
    nats, hence / ln 2.
    """
    C = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, C), targets.reshape(-1)) / math.log(2)
    return loss, 2 ** loss


class _Decoder(L.LightningModule):
    """
    Everything that does not depend on the encoder.

    Subclasses implement `_encode` ((B,T) -> (B,T,D)) and add a head in
    `forward`.  The constructor takes the same names for both architectures, so a
    config written for one builds the other:

        token_size    V, the vocabulary
        d_model       D, the width of the per-position vector the head reads
        max_len       the sequence length trained on (the transformer sizes its
                      positional table from it; the GRU only records it)
        lr, weight_decay   AdamW
        mode          "forward" reads left to right, "backward" right to left
        n_layers      depth of the encoder
        score_start   the first output the loss counts (scoring.score_mask); 0 scores
                      every output, the legacy objective
        usage_scope   "all" or "scored": the positions the usage penalty averages over
        learn_input_embedding   learn the token input map (transformer rand_prj as a
                      Parameter; an identity-initialised V x V map in front of the GRUs)
        lr_schedule, lr_min, schedule_horizon   "constant" (the legacy optimiser), or
                      "cosine" to lr_min over schedule_horizon optimiser updates
    """

    def __init__(self, token_size=3, d_model=32, max_len=300, lr=1e-3,
                 mode="forward", n_layers=2, weight_decay=0.0,
                 score_start=0, usage_scope="all", learn_input_embedding=False,
                 lr_schedule="constant", lr_min=1e-5, schedule_horizon=None):
        super().__init__()
        if mode not in ("forward", "backward"):
            raise ValueError(f"mode must be 'forward' or 'backward', got {mode!r}")
        if score_start < 0:
            raise ValueError("score_start must be nonnegative")
        if usage_scope not in ("all", "scored"):
            raise ValueError("invalid usage scope")
        if lr_schedule not in ("constant", "cosine"):
            raise ValueError("invalid learning-rate schedule")
        self.mode         = mode
        self.token_size   = token_size
        self.d_model      = d_model
        self.max_len      = max_len
        self.lr           = lr
        self.n_layers     = n_layers
        self.weight_decay = weight_decay
        self.score_start = int(score_start)
        self.usage_scope = usage_scope
        self.learn_input_embedding = bool(learn_input_embedding)
        self.lr_schedule = lr_schedule
        self.lr_min = float(lr_min)
        self.schedule_horizon = None if schedule_horizon is None else int(schedule_horizon)
        self.last_encodings = None

    def _encode(self, tokens) -> torch.Tensor:
        """(B,T) tokens -> (B,T,D) encodings."""
        raise NotImplementedError

    def _tokens(self, tokens) -> torch.Tensor:
        """Anything array-like -> a long tensor on this module's device."""
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.as_tensor(tokens, device=self.device)
        if tokens.dtype != torch.long:
            tokens = tokens.long()
        return tokens

    @staticmethod
    def _split(batch, mode):
        """
        (inputs, targets) for this arm.

        The loader always yields forward convention (x[:-1], x[1:]); a backward
        model is trained on the SWAP, so it predicts the previous token.  Every
        place that scores a model has to mirror this or it evaluates on an
        alignment the model never saw.

        The runs in this folder take the backward arm from the DATA instead
        (process_generator.generator.reverse flips the sequences) and keep every
        model at mode="forward"; the swap is here so a mode="backward" model is
        still scored on the alignment it trained on.  Never combine the two: the
        flips cancel and the forward arm comes back.
        """
        inputs, targets = batch
        return (inputs, targets) if mode == "forward" else (targets, inputs)

    def training_step(self, batch, batch_idx):
        inputs, targets = self._split(batch, self.mode)
        logits = self(inputs)  # includes every prefix token
        if self.score_start == 0:
            # Every output scored: the legacy reduction, kept so a legacy run stays bit-identical
            # (the masked mean agrees numerically but sums in a different order).
            ce, _ = cross_ent_onehot(logits, targets)
            mask = None
        else:
            ce, mask = masked_ce_bits(logits, targets, self.score_start, self.mode)
        extra = self._extra_loss(mask)
        self.log("train_loss", ce, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_perplexity", 2 ** ce, prog_bar=True, on_step=True, on_epoch=True)
        # scored targets per row are T - score_start in either mode; arithmetic, so no device sync
        n = targets.shape[0] * (targets.shape[1] - self.score_start)
        return {
            "loss": ce + extra,
            "ce_sum": ce.detach() * n,
            "scored_tokens": n,
        }

    def _extra_loss(self, score_mask=None):
        """Architecture-specific loss term; 0 for a plain decoder."""
        return 0.0

    # ── analysis surface ─────────────────────────────────────────────────
    def state_assignment(self, tokens=None):
        """(B,T) state indices, or None when the architecture has no explicit
        states -- in which case the analysis layer clusters instead."""
        return None

    def emission_table(self):
        """(K,V) P(token | state), or None when there are no explicit states."""
        return None

    def configure_optimizers(self):
        """
        AdamW.  At weight_decay=0.0 it is bit-identical to Adam.

        TRAP, and why the value is always passed explicitly: PyTorch's AdamW
        defaults to weight_decay=0.01 while Adam defaults to 0.0.

        The decay is not cosmetic here.  Cross-entropy on a deterministic
        transition has no finite optimum, which is why training eventually blows
        up (see training.diagnose_divergence); decoupled decay restores one.
        Its total effect scales with lr * weight_decay * steps, so a given lambda
        bites harder in a long run than a short one.

        lr_schedule="cosine" (the continuous mode's regime, from the notebook GRU) anneals
        the rate to lr_min over schedule_horizon UPDATES, stepped once per optimiser update.
        The horizon may exceed the run (continuous mode: twice the budget, so the run
        stops halfway down); training.train_model rejects one shorter than the run,
        because CosineAnnealingLR rises again after T_max.  "constant" is the legacy optimiser.
        """
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_schedule == "constant":
            return opt
        if self.schedule_horizon is None or self.schedule_horizon < 2:
            raise ValueError("a cosine learning rate needs a schedule horizon >= 2")
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.schedule_horizon, eta_min=self.lr_min)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}
