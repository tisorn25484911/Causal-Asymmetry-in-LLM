"""
The two transformer architectures the pipeline extracts causal states from.

    OneHotDecoder           the STANDARD model.  No explicit states; they have
                            to be clustered out of its predictive distributions
                            (extraction.recover_causal_states).
    DiscreteCausalDecoder   the DISCRETE model.  Prediction is forced through a
                            K-way bottleneck, so a position's state is just
                            argmax(state_logits) and needs no clustering.

GRU.py builds the same two with a GRU in place of the attention stack; the loss,
the training step and the discrete head are shared (decoder_base.py,
causal_matrix.py), so the two files differ only in `_encode`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_matrix import _DiscreteHead
from decoder_base import _Decoder, cross_ent_onehot  # noqa: F401  (re-exported)


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


class _TransformerStack(_Decoder):
    """
    The encoder both transformer heads share: fixed random input projection, PE,
    n_layers of pre-norm attention + FFN.  `_encode` gives (B,T,D).
    """

    def __init__(self, token_size=3, d_model=32, max_len=300, lr=1e-3,
                 mode="forward", n_layers=2, weight_decay=0.0):
        super().__init__(token_size=token_size, d_model=d_model, max_len=max_len,
                         lr=lr, mode=mode, n_layers=n_layers, weight_decay=weight_decay)

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
        tokens = self._tokens(tokens)
        x = F.one_hot(tokens, num_classes=self.token_size).float() @ self.rand_prj
        x = self.pe(x)
        mask = self._causal_mask(x.shape[1], x.device)
        for attn, ffn, ln1, ln2 in zip(self.attn_layers, self.ffn_layers,
                                       self.ln_attn, self.ln_ffn):
            # ln1(x) three times, exactly as Experimental_pipeline/models.py: once
            # is the same value, but it reorders the backward sums, and the key
            # bias (whose true gradient is 0 -- softmax ignores a shift shared by
            # every key) then drifts on the rounding noise, so training stops
            # matching the pipeline bit for bit.
            x = x + attn(ln1(x), ln1(x), ln1(x), mask=mask)
            x = x + ffn(ln2(x))
        self.last_encodings = x
        return x


class OneHotDecoder(_TransformerStack):
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


class DiscreteCausalDecoder(_DiscreteHead, _TransformerStack):
    """
    Prediction forced through a DISCRETE causal state.
        x          (B,T,D)   transformer stack
        logits     (B,T,V)   causal_matrix._learn_causal_state   <- the cross-entropy is here
    """

    def __init__(self, n_states=None, state_dim=None, tau=1.0,
                 usage_beta=0.0, **kw):
        super().__init__(**kw)
        self._init_causal_head(n_states, state_dim, tau, usage_beta)
        self.save_hyperparameters()

    def forward(self, tokens):
        return self.causal_head(self._encode(tokens), self.tau, self._sampling())


def build_model(embed_type: str, **kw):
    """The ONLY place that decides which transformer is built."""
    if embed_type == "onehot":
        kw.pop("n_states", None); kw.pop("state_dim", None)
        kw.pop("tau", None);      kw.pop("usage_beta", None)
        return OneHotDecoder(**kw)
    if embed_type == "discrete":
        return DiscreteCausalDecoder(**kw)
    raise ValueError(f"embed_type must be 'onehot' or 'discrete', got {embed_type!r}")
