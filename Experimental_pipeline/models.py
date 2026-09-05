"""
The two architectures the pipeline extracts causal states from.

    OneHotDecoder           the STANDARD model.  No explicit states; they have
                            to be clustered out of its predictive distributions
                            (extraction.recover_causal_states).
    DiscreteCausalDecoder   the DISCRETE model.  Prediction is forced through a
                            K-way bottleneck, so a position's state is just
                            argmax(state_logits) and needs no clustering.

Both share one transformer stack, in `_Decoder`.  In the old tree the stack --
embedding projection, positional encoding, masks, attention/FFN blocks,
optimiser -- was written out twice, once per class, and the discrete file
imported three pieces from the onehot file to paper over the rest.

ANALYSIS SURFACE.  Both expose `state_assignment()` and `emission_table()`;
OneHotDecoder returns None from both.  That is what lets `extraction.py` hold
the single branch between the two architectures instead of every caller testing
`embed_type`.

Dropped on the way over: `capture_attention` / `store_attention` /
`last_attention` (only the attention heatmaps read them, and those are out of
scope) and `reverse_pos_for_backward` (never set True anywhere in the tree).
Backward is the triu mask plus the batch swap, and nothing else.
"""
import math

import lightning as L
import torch
import torch.nn as nn
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


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, grown on demand if T exceeds max_len."""

    def __init__(self, d_model=32, max_len=300):
        super().__init__()
        self.d_model = d_model
        self._build_pe(max_len)

    def _build_pe(self, max_len, device=None, dtype=torch.float32):
        d_model = self.d_model
        pe  = torch.zeros(max_len, d_model, device=device, dtype=dtype)
        pos = torch.arange(0, max_len, device=device, dtype=dtype).unsqueeze(1)
        ith = torch.arange(0, d_model, 2, device=device, dtype=dtype)
        div = 10000 ** (ith / d_model)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        T = x.shape[1]
        if T > self.pe.size(0):
            self._build_pe(max(T, self.pe.size(0) * 2), device=x.device, dtype=x.dtype)
        return x + self.pe[:T].unsqueeze(0)


class AttentionModel(nn.Module):
    """Single-head scaled dot-product attention.  mask True = allowed."""

    def __init__(self, d_model=32):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, q, k, v, mask=None):
        Q, K, V = self.wq(q), self.wk(k), self.wv(v)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_model ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        return torch.softmax(scores, dim=-1) @ V


class _Decoder(L.LightningModule):
    """
    The stack both architectures share: fixed random input projection, PE,
    n_layers of pre-norm attention + FFN, AdamW.

    Subclasses add a head and implement `forward`; `_encode` gives them (B,T,D).
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

        # A BUFFER, not a Parameter: `one_hot @ rand_prj` as a Parameter is a
        # learned embedding table, which is what this class is named for NOT
        # being.  As a buffer, d_model varies representational capacity alone.
        rand_prj = F.normalize(torch.randn(token_size, d_model), dim=1)
        self.register_buffer("rand_prj", rand_prj)

        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.attn_layers = nn.ModuleList(
            [AttentionModel(d_model=d_model) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(),
                          nn.Linear(4 * d_model, d_model))
            for _ in range(n_layers)])
        self.ln_attn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_ffn  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.last_encodings = None
        self._mask_cache: dict = {}          # plain dict: stays out of state_dict

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        """
        (1,T,T) mask, cached per (T, device).  True = allowed.

        Forward (tril): position t attends to [0, t].
        Backward (triu): position t attends to [t, T-1].

        Exact mirror images -- summed over positions the context budget is
        identical, which is why the mask is not itself a forward/backward
        confound.  The mode is fixed for the life of the model, so it is
        deliberately not part of the cache key.
        """
        key = (T, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]
        ones = torch.ones((T, T), device=device, dtype=torch.bool)
        mask = (torch.tril(ones) if self.mode == "forward"
                else torch.triu(ones)).unsqueeze(0)
        self._mask_cache[key] = mask
        return mask

    def _encode(self, tokens) -> torch.Tensor:
        """(B,T) tokens -> (B,T,D) encodings."""
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.as_tensor(tokens, device=self.rand_prj.device)
        if tokens.dtype != torch.long:
            tokens = tokens.long()

        x = F.one_hot(tokens, num_classes=self.token_size).float() @ self.rand_prj
        x = self.pe(x)
        mask = self._causal_mask(x.shape[1], x.device)
        for attn, ffn, ln1, ln2 in zip(self.attn_layers, self.ffn_layers,
                                       self.ln_attn, self.ln_ffn):
            x = x + attn(ln1(x), ln1(x), ln1(x), mask=mask)
            x = x + ffn(ln2(x))
        self.last_encodings = x
        return x

    @staticmethod
    def _split(batch, mode):
        """
        (inputs, targets) for this arm.

        The loader always yields forward convention (x[:-1], x[1:]); a backward
        model is trained on the SWAP, so it predicts the previous token.  Every
        place that scores a model has to mirror this or it evaluates on an
        alignment the model never saw.
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


class OneHotDecoder(_Decoder):
    """
    The standard decoder: encodings -> logits, no bottleneck.

    Has no explicit causal states.  `state_assignment` and `emission_table`
    return None, and the states are recovered by clustering its PREDICTIVE
    DISTRIBUTIONS -- see extraction.recover_causal_states.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.output_prj = nn.Linear(self.d_model, self.token_size)
        self.save_hyperparameters()

    def forward(self, tokens):
        return self.output_prj(self._encode(tokens))


class DiscreteCausalDecoder(_Decoder):
    """
    Prediction forced through a DISCRETE causal state.

        x          (B,T,D)   transformer stack
        state_lgt  (B,T,K)   state_head   D -> K
        onehot     (B,T,K)   argmax of state_lgt, straight-through
        rep        (B,T,S)   onehot @ state_matrix
        out        (B,T,V)   emission     S -> V     <- the cross-entropy is here

    The point is that the causal states become readable: a position's state is
    argmax(state_lgt), so S_emp is the entropy of the occupancy distribution and
    needs no clustering hyperparameters at all.

    `n_states` (K) is a state BUDGET, not an estimate, and it is REQUIRED -- the
    config resolves it as 5V.  A model does not know which process it is being
    trained on, so it cannot compute the theoretical count for itself, and
    defaulting to the vocabulary size would pin the bottleneck to V, which is
    the artefact this architecture exists to avoid.

    `state_dim` (S) carries no expressive power -- state_matrix followed by
    emission composes to a single (K,V) map -- so it is set to V rather than K,
    which would make state_matrix mostly reparameterisation freedom.
    """

    def __init__(self, n_states=None, state_dim=None, tau=1.0,
                 usage_beta=0.0, **kw):
        super().__init__(**kw)
        if n_states is None:
            raise ValueError(
                "DiscreteCausalDecoder requires n_states.  Resolve it from the "
                "config (K = n_states_mult * V); defaulting to the vocabulary "
                "reintroduces the bottleneck this architecture removes.")
        self.n_states   = int(n_states)
        self.state_dim  = self.n_states if state_dim is None else int(state_dim)
        self.tau        = tau            # temperature of the SURROGATE only
        self.usage_beta = usage_beta

        self.state_head   = nn.Linear(self.d_model, self.n_states)
        self.state_matrix = nn.Parameter(
            torch.randn(self.n_states, self.state_dim) / (self.state_dim ** 0.5))
        self.emission     = nn.Linear(self.state_dim, self.token_size)
        self.save_hyperparameters()

        # Filled by forward(); overwritten every call.
        self.last_state_probs  = None    # (B,T,K)
        self.last_state_onehot = None    # (B,T,K)
        self.last_states       = None    # (B,T)
        self.last_causal_reps  = None    # (B,T,S)

    def forward(self, tokens):
        x = self._encode(tokens)

        state_logits = self.state_head(x)
        probs  = F.softmax(state_logits / self.tau, dim=-1)
        states = state_logits.argmax(dim=-1)
        hard   = F.one_hot(states, num_classes=self.n_states).float()

        # Straight-through: forward value `hard`, gradient through `probs`.
        state_onehot = (hard - probs).detach() + probs

        causal_rep = state_onehot @ self.state_matrix
        self.last_state_probs  = probs
        self.last_state_onehot = state_onehot
        self.last_states       = states
        self.last_causal_reps  = causal_rep
        return self.emission(causal_rep)

    def usage_penalty(self, occ):
        """
        (H(p_bar), penalty).  H is reported rather than the KL because it is the
        quantity directly comparable to C.

        beta is 1/(batch * seq_len) -- one over the count the cross-entropy
        itself averages over -- and is COMPUTED, never a literal: it is bounded
        above by an optimisation cliff (~6e-4 at lr=1e-3) above which the
        bottleneck collapses to a single state, and the cliff scales with 1/lr
        rather than with N, so a literal drifts relative to it silently when
        batch or seq_len moves.
        """
        p_bar = occ.reshape(-1, self.n_states).mean(dim=0)
        H = -(p_bar * torch.log2(p_bar + 1e-12)).sum()
        if self.usage_beta == 0.0:
            return H, p_bar.sum() * 0.0            # keeps the graph
        return H, self.usage_beta * H

    def _extra_loss(self):
        H, penalty = self.usage_penalty(self.last_state_onehot)
        self.log("state_usage_entropy", H, prog_bar=True, on_step=True, on_epoch=True)
        return penalty

    # ── analysis surface ─────────────────────────────────────────────────
    def state_assignment(self, tokens=None):
        """(B,T) integer state index per position -- the real thing, no clustering."""
        if tokens is not None:
            with torch.no_grad():
                self(tokens)
        return self.last_states

    def emission_table(self):
        """
        (K,V) P(next token | state) -- P(previous token | state) backward.

        This is the IDENTIFIED object: unlike the raw state vectors it is
        invariant to the reparameterisation state_matrix admits, so it is the one
        to compare against a closed form.
        """
        with torch.no_grad():
            return self.emission(self.state_matrix).softmax(dim=-1)


def build_model(embed_type: str, **kw):
    """The ONLY place that decides which architecture is built."""
    if embed_type == "onehot":
        kw.pop("n_states", None); kw.pop("state_dim", None)
        kw.pop("tau", None);      kw.pop("usage_beta", None)
        return OneHotDecoder(**kw)
    if embed_type == "discrete":
        return DiscreteCausalDecoder(**kw)
    raise ValueError(f"embed_type must be 'onehot' or 'discrete', got {embed_type!r}")
