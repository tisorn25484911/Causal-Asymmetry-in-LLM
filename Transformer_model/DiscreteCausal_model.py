"""
Decoder whose prediction is forced through a DISCRETE causal state.

Same transformer stack as OneHotDecoder -- same fixed random projection, same
positional encoding, same masks, same attention and FFN blocks, which are
imported from OneHot_model rather than copied.  Only the head differs:

    x          (B,T,D)   transformer stack
    state_lgt  (B,T,K)   state_head   D -> K     the state code
    onehot     (B,T,K)   argmax of state_lgt, straight-through
    rep        (B,T,S)   onehot @ state_matrix   the state VECTOR
    out        (B,T,V)   emission     S -> V     <- the cross-entropy is here

The point is that the causal states become readable.  OneHotDecoder's states
have to be clustered out of the latents (Model_analysis.recover_causal_states,
with a silhouette-selected k-hat and a state_tol); here the state of a position
is just `argmax(state_lgt)`, so S_emp is the entropy of the occupancy
distribution and needs no clustering hyperparameters at all.

MODULAR_MODELS_PLAN.md sections 4 and 5.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import AdamW

# Shared with OneHotDecoder on purpose.  IMPROVEMENT_PLAN C4 deleted a set of
# duplicated definitions that had already drifted apart; re-importing keeps the
# two architectures differing ONLY in the head, which is what makes a
# model-to-model comparison mean anything.
from OneHot_model import (
    PositionalEncoding,
    AttentionModel,
    cross_ent_onehot,
)


class DiscreteCausalDecoder(L.LightningModule):
    """
    Forward mode:
      - causal mask (tril)
      - normal PE

    Backward mode:
      - anti-causal mask (triu) to allow attending to future
      - optional reversed PE via reverse_pos_for_backward flag

    `n_states` (K) is the number of causal states the bottleneck may use, and it
    is a REQUIRED experimental quantity, not a convenience default.  Pass the
    theoretical count for the process and arm being trained:

        coin     forward  2          backward  3
        flower   forward  n + 1      backward  1 + #distinguishable outcomes

    The runner computes it -- `Model_analysis.causal_state_count()` -- because
    the backward flower count depends on the dice realisation and cannot be
    assumed to be m + 1: dice whose posterior columns are proportional merge
    into one backward state, and at n = 1 every outcome merges.  A model does
    not know which process it is being trained on, so it cannot compute this
    for itself; `n_states=None` falls back to `token_size` only so the class is
    usable standalone.
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
        weight_decay: float = 0.0,
        n_states: int | None = None,
        state_dim: int | None = None,
        tau: float = 1.0,
        usage_beta: float = 0.0,
        target_occupancy=None,
    ):
        super().__init__()
        self.mode = mode
        self.reverse_pos_for_backward = reverse_pos_for_backward
        self.n_layers = n_layers

        self.token_size = token_size
        self.d_model = d_model
        self.max_len = max_len
        self.lr = lr
        self.weight_decay = weight_decay

        # K -- how many causal states exist.  See the class docstring: this is
        # an experimental setting and the runner is expected to supply it.
        self.n_states = token_size if n_states is None else int(n_states)

        # S -- the DIMENSION of a state vector.  Defaults to K, which makes
        # state_matrix square.  It carries no expressive power (see below), so
        # there is rarely a reason to change it.
        self.state_dim = self.n_states if state_dim is None else int(state_dim)

        # Temperature of the straight-through SURROGATE only.  argmax is
        # invariant to positive scaling, so tau never changes the forward
        # value.  tau < 1 sharpens `probs` toward the hard one-hot, shrinking
        # the estimator's bias at the cost of a flatter gradient; measured, tau
        # <= 0.2 sharpens early enough to lock in whichever state happens to be
        # winning and collapses the bottleneck.  1.0 is plain softmax.
        self.tau = tau

        # Strength of the anti-collapse penalty; see usage_penalty().  0.0 is
        # the default because the penalty pushes usage toward uniform over K,
        # and K is the state BUDGET, not necessarily the count the process
        # needs.
        self.usage_beta = usage_beta

        # The TARGET of the anti-collapse penalty: the stationary probability of
        # each causal state, from the closed form (causal_state_occupancy).  Its
        # entropy is exactly C+ / C-.
        #
        # None falls back to penalising distance from UNIFORM, which is what the
        # first version did and which is biased by construction: uniform has
        # entropy log2(K), the truth has entropy C, and log2(K) - C is exactly
        # the error a large beta drives S_emp toward.  Measured, that gap ranges
        # from 0.019 to 0.694 bits across the seven baseline processes, so the
        # bias is not small and not constant.
        if target_occupancy is None:
            self.register_buffer("target_occupancy", None)
        else:
            t = torch.as_tensor(target_occupancy, dtype=torch.float32).flatten()
            if t.numel() != self.n_states:
                raise ValueError(
                    f"target_occupancy has {t.numel()} entries but n_states is "
                    f"{self.n_states}; they describe the same state set")
            self.register_buffer("target_occupancy", t / t.sum())

        # Same fixed random projection as OneHotDecoder -- a buffer, not a
        # parameter, so "onehot" input coding really is fixed and a d_model
        # sweep varies representational capacity alone.
        rand_prj = torch.randn(token_size, d_model)
        rand_prj = F.normalize(rand_prj, dim=1)
        self.register_buffer("rand_prj", rand_prj)

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

        self.ln_attn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ln_ffn  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # There is deliberately no `output_prj: D -> V` here.  OneHotDecoder has
        # one, and the notebook prototype took its argmax over it -- but once the
        # argmax moves to a dedicated state head the alphabet logits have no job:
        # the loss flows x -> state_head -> state_matrix -> emission and never
        # touches them.  Measured, output_prj received exactly ZERO gradient, so
        # `last_logits` would be an untrained random projection and meaningless
        # to plot.  The causal-state figures replace it.
        #
        # D -> K.  The state code.  The argmax is taken HERE and not over
        # the alphabet logits, which is the whole reason n_states can be chosen: taking
        # it over the V-dimensional logits would pin the state budget to the
        # vocabulary, and V is a property of the process that cannot be set per
        # arm.  Measured consequence of doing that on the coin, where H_inf is
        # time-reversal invariant and therefore identical for both arms:
        # forward reached H_inf (+0.0016) while backward sat +0.2371 above it,
        # an artefact ~100x the effect being measured, pointing the same way as
        # the hypothesis.  MODULAR_MODELS_PLAN.md 4.1.
        self.state_head = nn.Linear(d_model, self.n_states)

        # K -> S.  The learned state VECTORS, one row per state.  Held as a
        # bare Parameter so `onehot @ state_matrix` is literally a row lookup
        # and row k reads out as "the vector of state k".
        #
        # It carries no expressive power: state_matrix followed by emission
        # composes into a single (K, V) linear map, so state_matrix @ Q with
        # inv(Q) . emission is an identical model for any invertible Q
        # (verified numerically, max output deviation 3.6e-07).  It exists so
        # there is something to plot; the GEOMETRY of that plot is not
        # identified by training and the figure must say so.  The identified
        # object is emission_table().
        self.state_matrix = nn.Parameter(
            torch.randn(self.n_states, self.state_dim) / (self.state_dim ** 0.5)
        )

        # S -> V.  Decodes a state vector into a next-token distribution.
        #
        # This layer is why the loss stays a genuine cross-entropy.  F.cross_entropy
        # softmaxes whatever it is handed, and softmax of a one-hot cannot exceed
        # 1/(1 + (V-1)/e) confidence -- 0.576 at V=3, 0.198 at V=12 -- so returning
        # the bottleneck directly would put a floor under the achievable CE that
        # GROWS WITH THE VOCABULARY.  The flower grid varies V, so that floor would
        # be an artefact along the sweep's own axis.
        self.emission = nn.Linear(self.state_dim, token_size)

        self.save_hyperparameters()

        # Filled by forward().
        self.last_encodings   = None    # (B,T,D)
        self.last_state_probs = None    # (B,T,K)  softmax of the state code
        self.last_states      = None    # (B,T)    the state index actually taken
        self.last_causal_reps = None    # (B,T,S)  the state vector received
        self.last_attention   = None
        self.last_attention_layers = None

        self.store_attention = False
        self._mask_cache = {}

        # Per-epoch diagnostics.
        self.train_loss_history       = []
        self.train_perplexity_history = []
        self.usage_entropy_history    = []
        self.states_occupied_history  = []

        # Union of the states seen across an epoch.  Logging
        # last_states.unique().numel() per batch with on_epoch=True would let
        # Lightning AVERAGE it -- a fractional "number of states", and an
        # undercount, since a state used in only some batches is still used.
        self._epoch_states = set()

    # ------------------------------------------------------------------ mask

    def _causal_mask(self, T: int, device):
        """
        True = allowed.  Forward (tril): position t attends to [0, t].
        Backward (triu): position t attends to [t, T-1].  The two are exact
        mirrors with equal total context, so the MASK is not the fw/bw
        confound.

        Cached per (T, device); the mode is fixed for the life of the model, so
        it is deliberately not part of the key.  Mutating `.mode` after a
        forward pass therefore returns a stale mask -- do not do it.
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
                f"Invalid mode: {self.mode}. Must be 'forward' or 'backward'."
            )

        self._mask_cache[key] = mask
        return mask

    # --------------------------------------------------------------- forward

    def forward(self, tokens):
        if isinstance(tokens, torch.Tensor):
            if tokens.dtype not in (torch.long, torch.int64):
                tokens = tokens.long()
        else:
            tokens = torch.LongTensor(tokens).to(self.rand_prj.device)

        one_hot = F.one_hot(tokens, num_classes=self.token_size).float()
        x = one_hot @ self.rand_prj

        reverse_pos = (self.mode == "backward" and self.reverse_pos_for_backward)
        x = self.pe(x, reverse_pos=reverse_pos)

        mask = self._causal_mask(x.shape[1], x.device)
        want_attn = self.store_attention
        attn_maps = []

        for attn, ffn, ln1, ln2 in zip(
            self.attn_layers, self.ffn_layers, self.ln_attn, self.ln_ffn
        ):
            normed = ln1(x)
            if want_attn:
                attn_out, attn_prob = attn(normed, normed, normed,
                                           mask=mask, return_attn=True)
                attn_maps.append(attn_prob.detach())
            else:
                attn_out = attn(normed, normed, normed, mask=mask, return_attn=False)
            x = x + attn_out
            x = x + ffn(ln2(x))

        self.last_encodings = x
        self.last_attention_layers = attn_maps if want_attn else None
        self.last_attention = attn_maps[-1] if attn_maps else None

        # The state code, and the discretisation.
        state_logits = self.state_head(x)
        probs  = F.softmax(state_logits / self.tau, dim=-1)
        states = state_logits.argmax(dim=-1)
        hard   = F.one_hot(states, num_classes=self.n_states).float()

        # Straight-through estimator.  argmax has no derivative and F.one_hot
        # returns a leaf, so this is mandatory rather than a refinement:
        # without it the graph is severed here and every layer beneath stays at
        # its initialisation while the loss still appears to fall.  Measured on
        # the notebook prototype, 2 of 33 parameter tensors trained.
        #
        # Forward value:     hard        Backward gradient: through probs
        state_onehot = (hard - probs).detach() + probs

        # Row lookup: causal_rep[b,t] = state_matrix[states[b,t]].
        causal_rep = state_onehot @ self.state_matrix
        out_logits = self.emission(causal_rep)

        self.last_state_probs = probs
        self.last_states      = states
        self.last_causal_reps = causal_rep

        return out_logits

    # ------------------------------------------------- analysis surface (§3)

    def state_assignment(self, tokens=None):
        """
        (B,T) integer state index per position, or None for a model whose
        states are not explicit.  OneHotDecoder returns None and the analysis
        layer falls back to clustering; this class returns the real thing.
        """
        if tokens is not None:
            with torch.no_grad():
                self(tokens)
        return self.last_states

    def emission_table(self):
        """
        (K, V) table of P(next token | state) -- P(previous token | state) in
        backward mode.  This is the IDENTIFIED object: unlike the raw state
        vectors it is invariant to the reparameterisation noted on
        `state_matrix`, so it is the one to compare against a closed form.
        """
        with torch.no_grad():
            return self.emission(self.state_matrix).softmax(dim=-1)

    # ---------------------------------------------------------------- losses

    def usage_penalty(self, probs):
        """
        Penalise the bottleneck for not using its states as the process does.

        With p_bar[k] the state occupancy marginalised over batch and positions:

          * target_occupancy given -- beta * KL(pi_theory || p_bar), which is 0
            exactly when the model's occupancy equals the process's stationary
            distribution.  UNBIASED by construction: the minimiser of the
            penalty is the truth, so adding it does not move the optimum.
          * target_occupancy None -- beta * (log2(K) - H(p_bar)), the older
            uniform version, whose minimiser is uniform and therefore NOT the
            truth unless the process happens to be uniform.

        Both are compared on SORTED distributions.  The state labels are
        arbitrary -- nothing ties the model's state 0 to the theory's state 0 --
        so an unsorted KL would additionally penalise the model for choosing a
        different permutation, which is not a defect.  Sorting is a permutation,
        so gradients still reach every entry.

        Returns (H(p_bar), penalty).  H is reported rather than the KL because
        it is the quantity comparable to C.
        """
        p_bar = probs.reshape(-1, self.n_states).mean(dim=0)
        H_usage = -(p_bar * torch.log2(p_bar + 1e-12)).sum()

        if self.usage_beta == 0.0:
            return H_usage, p_bar.sum() * 0.0          # keeps the graph, costs nothing

        if self.target_occupancy is None:
            H_max = math.log2(self.n_states) if self.n_states > 1 else 0.0
            return H_usage, self.usage_beta * (H_max - H_usage)

        pi = torch.sort(self.target_occupancy, descending=True).values
        pb = torch.sort(p_bar, descending=True).values
        kl = (pi * (torch.log2(pi + 1e-12) - torch.log2(pb + 1e-12))).sum()
        return H_usage, self.usage_beta * kl

    def training_step(self, batch, batch_idx):
        # Backward mode swaps the batch, matching OneHot_model.training_step.
        # This is not cosmetic: fed unswapped, a triu-masked model can see its
        # own target -- target[t] = x[t+1] sits at input position t+1, which is
        # inside the window [t, T-1] -- and the CE drops BELOW H_inf, which is
        # impossible for a real predictor.
        if self.mode == "forward":
            inputs, targets = batch
        else:
            targets, inputs = batch

        out_logits = self(inputs)
        loss, perplexity = cross_ent_onehot(out_logits, targets)

        # The CE is logged separately from the penalised objective.  Only the
        # CE is comparable to H_inf; the total is not a cross-entropy and must
        # never be quoted as one.
        H_usage, penalty = self.usage_penalty(self.last_state_probs)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_perplexity", perplexity,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("state_usage_entropy", H_usage,
                 on_step=False, on_epoch=True, prog_bar=True)

        self._epoch_states.update(self.last_states.unique().tolist())

        return loss + penalty

    def on_train_epoch_end(self):
        loss  = self.trainer.callback_metrics.get("train_loss")
        ppl   = self.trainer.callback_metrics.get("train_perplexity")
        usage = self.trainer.callback_metrics.get("state_usage_entropy")

        if loss is not None:
            self.train_loss_history.append(loss.detach().cpu().item())
        if ppl is not None:
            self.train_perplexity_history.append(ppl.detach().cpu().item())
        if usage is not None:
            self.usage_entropy_history.append(usage.detach().cpu().item())

        self.states_occupied_history.append(len(self._epoch_states))
        self._epoch_states = set()

    def configure_optimizers(self):
        # AdamW so weight_decay is decoupled.  At weight_decay=0.0 this is
        # bit-identical to plain Adam, which is what every existing result used.
        #
        # Note decay reaches state_matrix like any other parameter and shrinks
        # the state vectors toward the origin -- measured ~18% at lambda=0.1.
        # emission can rescale to compensate, so the model need not change, but
        # the SCALE of the state-vector scatter does.  Hold lambda fixed when
        # comparing state geometry across runs.
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
