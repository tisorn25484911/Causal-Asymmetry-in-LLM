"""
What every decoder shares, whatever turns tokens into per-position vectors.

    cross_ent_onehot   the loss, in bits
    _Decoder           the LightningModule both architectures subclass: the stored
                       hyperparameters, the training step, the optimiser and the
                       analysis surface.  A subclass supplies `_encode`, the
                       (B,T) tokens -> (B,T,D) map -- the attention stack in
                       Transformer.py, the GRU in GRU.py.

The discrete causal head that can sit on top of `_encode` is in causal_matrix.py.
"""
import math

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim import AdamW


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
    """

    def __init__(self, token_size=3, d_model=32, max_len=300, lr=1e-3,
                 mode="forward", n_layers=2, weight_decay=0.0):
        super().__init__()
        if mode not in ("forward", "backward"):
            raise ValueError(f"mode must be 'forward' or 'backward', got {mode!r}")
        self.mode         = mode
        self.token_size   = token_size
        self.d_model      = d_model
        self.max_len      = max_len
        self.lr           = lr
        self.n_layers     = n_layers
        self.weight_decay = weight_decay
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
        loss, ppl = cross_ent_onehot(self(inputs), targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_perplexity", ppl, prog_bar=True, on_step=True, on_epoch=True)
        return loss + self._extra_loss()

    def _extra_loss(self):
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
        """
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
