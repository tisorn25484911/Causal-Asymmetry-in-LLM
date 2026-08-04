import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with optional REVERSE position assignment.

    reverse_pos=False:
        position t uses PE[t]

    reverse_pos=True:
        position t uses PE[T-1-t]   (mirror positions within current sequence length T)
    """
    def __init__(self, d_model=20, max_len=150):
        super().__init__()
        self.d_model = d_model
        self._build_pe(max_len)

    def _build_pe(self, max_len, device=None, dtype=None):
        # Ensure float dtype (sin/cos needs float)
        if dtype is None:
            dtype = torch.float32

        d_model = self.d_model
        pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)

        pos = torch.arange(0, max_len, device=device, dtype=dtype).unsqueeze(1)  # (max_len, 1)
        ith = torch.arange(0, d_model, 2, device=device, dtype=dtype)            # (d_model/2,)
        div = 10000 ** (ith / d_model)                                           # (d_model/2,)

        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, reverse_pos: bool = False):
        B, T, D = x.shape

        # Grow PE if needed (no info loss)
        if T > self.pe.size(0):
            new_len = max(T, self.pe.size(0) * 2)
            self._build_pe(new_len, device=x.device, dtype=x.dtype)

        pe_T = self.pe[:T]  # (T, D)
        if reverse_pos:
            pe_T = torch.flip(pe_T, dims=[0])  # (T, D) reversed along position axis

        return x + pe_T.unsqueeze(0)  # (B, T, D)


class AttentionModel(nn.Module):
    def __init__(self, d_model=20):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, q, k, v, mask=None, return_attn=False):
        Q = self.wq(q)  # (B, T, D)
        K = self.wk(k)  # (B, T, D)
        V = self.wv(v)  # (B, T, D)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_model ** 0.5)  # (B, T, T)

        if mask is not None:
            # mask True = allowed, False = blocked
            scores = scores.masked_fill(~mask, -1e9)

        prob = torch.softmax(scores, dim=-1)  # (B, T, T)
        out = prob @ V                        # (B, T, D)

        if return_attn:
            return out, prob
        return out
    
def cross_ent_onehot(logits, targets):
    """
    Mean cross-entropy in BITS, and the matching perplexity 2**CE.

    Implementation note (IMPROVEMENT_PLAN.md B3).  The previous version built
    the one-hot target explicitly and evaluated

        -(target_prob * logits.softmax(-1).log2()).sum(dim=1)

    which returns NaN once any *non-target* class probability underflows to
    exactly 0: that term is 0 * -inf, and the NaN then propagates through the
    .sum() and poisons the whole batch even though the target class was fine.
    That is reachable here rather than hypothetical -- both processes contain
    deterministic transitions (coin token 2 -> token 1 w.p. 1; a flower roll is
    always followed by a selection), and cross-entropy training on a
    deterministic transition drives the logit gap to infinity, so a gap of ~200
    after 60-80 epochs at lr=1e-2 is realistic.  Measured: at logit scale 200
    the old expression returns nan where the true value is 0.0.

    F.cross_entropy uses the log-sum-exp trick internally, so it is exact at
    any logit scale, never touches a 0 * -inf product, and is faster (no
    separate softmax + log over (B*T, V)).  It returns nats, hence / ln 2.
    """
    C = logits.shape[-1]
    flat_input  = logits.reshape(-1, C)   # (B*T, V)
    flat_target = targets.reshape(-1)     # (B*T,)

    loss       = F.cross_entropy(flat_input, flat_target) / math.log(2)
    perplexity = 2 ** loss

    return loss, perplexity



class OneHotDecoder(L.LightningModule):
    """
    Forward mode:
      - causal mask (tril)
      - normal PE

    Backward mode:
      - anti-causal mask (triu) to allow attending to future
      - optional reversed PE via reverse_pos_for_backward flag
    """
    def __init__(
        self,
        token_size=3,
        d_model=20,
        max_len=150,
        lr=1e-2,
        mode="forward",
        reverse_pos_for_backward: bool = False,
        n_layers=2,
    ):
        super().__init__()
        self.mode = mode
        self.reverse_pos_for_backward = reverse_pos_for_backward
        self.n_layers = n_layers

        self.token_size = token_size
        self.d_model = d_model
        self.max_len = max_len
        self.lr = lr

        # IMPROVEMENT_PLAN.md C5.  This was an nn.Parameter, which makes
        # `one_hot @ self.rand_prj` a LEARNED embedding table -- mathematically
        # identical to nn.Embedding -- despite the class name and the "fixed
        # random projection" comment.  It is now a buffer, so "onehot" mode
        # really is a fixed random projection.
        #
        # This is what makes the Phase 4 d_model sweep interpretable: varying
        # d_model then varies representational capacity alone, rather than
        # capacity plus the size of a learned input code.  Note it invalidates
        # every checkpoint written before this commit -- rand_prj moves from
        # the parameter list to the buffer list.
        rand_prj = torch.randn(token_size, d_model)
        rand_prj = F.normalize(rand_prj, dim=1)
        self.register_buffer("rand_prj", rand_prj)

        # `self.we = nn.Embedding(...)` used to live here.  It was never used
        # in forward(): dead parameters that received no gradient and were
        # serialised into every .pt.  Deleted (C5).

        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        self.attn_layers = nn.ModuleList([
            AttentionModel(d_model=d_model) for _ in range(n_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Linear(4 * d_model, d_model),
            ) for _ in range(n_layers)
        ])
        
        self.ln_attn = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.ln_ffn = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        self.output_prj = nn.Linear(d_model, token_size)

        # REMOVED: self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

        self.last_encodings = None
        self.last_attention = None  # (B, T, T) — only when store_attention

        # IMPROVEMENT_PLAN.md D3.  The (B, T, T) attention map used to be
        # retained on every single forward pass.  At training time T = chunk,
        # so that is 8-34 MB and nobody noticed.  It only bites at full
        # sequence length -- and B4 (running analysis on the full-length loader,
        # as the README says it should) is exactly what takes T from 256 to
        # ~2000.  At T=1999, B=32 that is 32 * 1999^2 * 4 B = 511 MB retained
        # per pass.  So B4 and D3 have to land together: fixing B4 alone
        # creates the memory problem, gating alone leaves B4 unfixed.
        #
        # Only the two attention *plotting* helpers ever read this, and they
        # run on a 64-128 token prefix.  Default off; turn it on with the
        # capture_attention() context manager.
        self.store_attention = False

        # D4: (T, device) -> (1, T, T) bool mask.  A plain dict, deliberately
        # not a buffer, so it stays out of state_dict.
        self._mask_cache: dict = {}

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        """
        (1, T, T) attention mask, built once per (T, device) — D4.

        The tril/triu mask was rebuilt on EVERY forward pass, allocating a
        T x T bool tensor each time.  It depends only on (T, device, mode) and
        the mode is fixed for the life of the model, so it is cached.  Held in
        a plain dict rather than a buffer so it never lands in state_dict and
        cannot affect checkpoint compatibility.

        True = allowed.  Forward (tril): position t attends to [0, t].
        Backward (triu): position t attends to [t, T-1].  The two are exact
        mirror images -- summed over positions the context budget is identical,
        which is why the mask is NOT the forward/backward confound (the
        positional encoding is; see IMPROVEMENT_PLAN.md C2).
        """
        key = (T, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        ones = torch.ones((T, T), device=device, dtype=torch.bool)
        if self.mode == "forward":
            mask = torch.tril(ones).unsqueeze(0)
        elif self.mode == "backward":
            mask = torch.triu(ones).unsqueeze(0)
        else:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'forward' or 'backward'")
        self._mask_cache[key] = mask
        return mask

    @contextmanager
    def capture_attention(self):
        """Temporarily retain last_attention (see store_attention, D3)."""
        prev = self.store_attention
        self.store_attention = True
        try:
            yield self
        finally:
            self.store_attention = prev

    def forward(self, tokens):
        # --- sanitize tokens dtype/device ---
        if isinstance(tokens, torch.Tensor):
            if tokens.dtype in (torch.float32, torch.float64):
                tokens = tokens.long()
            elif tokens.dtype not in (torch.long, torch.int64):
                tokens = tokens.to(torch.long)
        else:
            tokens = torch.LongTensor(tokens).to(self.rand_prj.device)

        # tokens: (B, T)
        one_hot = F.one_hot(tokens, num_classes=self.token_size).float()  # (B, T, V)
        x = one_hot @ self.rand_prj
        
        # --- positional encoding (optionally reversed for backward mode) ---
        reverse_pos = (self.mode == "backward" and self.reverse_pos_for_backward)
        x = self.pe(x, reverse_pos=reverse_pos)  # (B, T, D)

        # --- build mask consistent with goal (cached — D4) ---
        B, T, _ = x.shape
        mask = self._causal_mask(T, x.device)

        want_attn = self.store_attention
        attn_maps = []
        for attn, ffn, ln1, ln2 in zip(self.attn_layers, self.ffn_layers,
                                         self.ln_attn, self.ln_ffn):
            # attention with pre-norm
            normed = ln1(x)
            if want_attn:
                attn_out, attn_prob = attn(normed, normed, normed, mask=mask,
                                           return_attn=True)
                attn_maps.append(attn_prob.detach())
            else:
                attn_out = attn(normed, normed, normed, mask=mask,
                                return_attn=False)
            x = x + attn_out

            # feedforward with pre-norm
            normed = ln2(x)
            x = x + ffn(normed)

        self.last_encodings = x.detach()
        # C8: keep every layer, not just the last.  `last_attention` stays the
        # final layer for backwards compatibility, but callers that want to
        # label a heatmap with its layer index can read last_attention_layers.
        self.last_attention_layers = attn_maps if want_attn else None
        self.last_attention = attn_maps[-1] if attn_maps else None

        logits = self.output_prj(x)  # (B, T, V)
        return logits


    def training_step(self, batch, batch_idx):
        if self.mode == "forward":
            inputs, targets = batch
        elif self.mode == "backward":
            targets, inputs = batch

        #print(f"Batch {batch_idx}: input ={inputs[:5]}, targets ={targets[:5]}")
        logits = self.forward(inputs)                      # (B, T, V)

        loss, perplexity = cross_ent_onehot(logits, targets)

        self.log("train_loss",       loss,       prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_perplexity", perplexity, prog_bar=True, on_step=True, on_epoch=True)
        #print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Perplexity={perplexity.item():.4f}")

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

# WordEmbDecoder was deleted here -- IMPROVEMENT_PLAN.md C5.
#
# No runner ever set embed_type="wordemb", so it was unreachable.  It was also
# not, as v1 of the plan supposed, a near-twin of OneHotDecoder: it was a
# SINGLE attention layer with no FFN, no LayerNorm and no n_layers argument,
# while OneHotDecoder is n-layer with pre-norm LayerNorm, FFN and residuals.
# Unifying them would have meant inventing a shared abstraction for two
# genuinely different architectures; deleting the unused one is the honest fix.
