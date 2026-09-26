"""
The discrete causal-state head, shared by every architecture.

    value_vec  (B,T,D)  per-position vector from the encoder (transformer stack or GRU)
    state_lgt  (B,T,K)  state_head   D -> K
    onehot     (B,T,K)  argmax of state_lgt, straight-through
    rep        (B,T,S)  onehot @ state_matrix
    out        (B,T,V)  emission     S -> V     <- token LOGITS; the softmax over
                                                   tokens happens inside the loss

    _learn_causal_state   the head itself, an nn.Module built ONCE in the model's
                          __init__, so its weights are registered and trained
    _DiscreteHead         what a decoder with this head adds on top of the plain
                          decoder: the usage penalty, and the analysis surface
                          (state_assignment, emission_table, n_states, ...).
                          Mixed into DiscreteCausalDecoder (Transformer.py) and
                          DiscreteCausalGRU (GRU.py), so the two cannot drift.

tau is an ARGUMENT of forward, not an attribute of the head.  The schedule
(tau_schedue.TauSchedule) rewrites `pl_module.tau` before every batch; a copy
cached in here would silently freeze the schedule at its first value.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _learn_causal_state(nn.Module):
    """value vector -> discrete causal state -> token logits."""

    def __init__(self, d_model, n_states, state_dim, token_size):
        super().__init__()
        self.n_states  = int(n_states)
        self.state_dim = int(state_dim)

        self.state_head   = nn.Linear(d_model, self.n_states)
        self.state_matrix = nn.Parameter(
            torch.randn(self.n_states, self.state_dim) / (self.state_dim ** 0.5))
        self.emission     = nn.Linear(self.state_dim, token_size)

        # Filled by forward(); overwritten every call.
        self.last_state_probs  = None    # (B,T,K)
        self.last_state_onehot = None    # (B,T,K)
        self.last_states       = None    # (B,T)
        self.last_causal_reps  = None    # (B,T,S)

    def forward(self, value_vec, tau, gumbel=False):
        """
        gumbel=True (training only; off by default) perturbs the state logits with
        Gumbel(0, 1) noise before the softmax and the argmax -- the Gumbel-softmax
        straight-through of Jang et al. (2017) / Maddison et al. (2017), which the
        old notebook GRU trained with.  It makes the hard state a SAMPLE, so states
        that are losing still get chosen, and trained, now and then.  Off, no
        random number is drawn and the head is exactly the deterministic one.
        """
        state_logits = self.state_head(value_vec)
        if gumbel:
            u = torch.rand_like(state_logits).clamp_(1e-10, 1.0 - 1e-10)
            state_logits = state_logits - torch.log(-torch.log(u))
        probs  = F.softmax(state_logits / tau, dim=-1)
        states = probs.argmax(dim=-1)
        hard   = F.one_hot(states, num_classes=self.n_states).float()

        # Straight-through: forward value `hard`, gradient through `probs`.
        state_onehot = (hard - probs).detach() + probs
        causal_rep = state_onehot @ self.state_matrix

        self.last_state_probs  = probs
        self.last_state_onehot = state_onehot
        self.last_states       = states
        self.last_causal_reps  = causal_rep
        return self.emission(causal_rep)

    def usage_entropy(self):
        """
        H(p_bar) in bits, p_bar the state occupancy of the last forward.
        Differentiable: p_bar is a mean of the straight-through one-hots.
        """
        p_bar = self.last_state_onehot.reshape(-1, self.n_states).mean(dim=0)
        return -(p_bar * torch.log2(p_bar + 1e-12)).sum()

    def emission_table(self):
        """
        (K,V) P(next token | state).

        This is the IDENTIFIED object: unlike the raw state vectors it is
        invariant to the reparameterisation state_matrix admits, so it is the one
        to compare against a closed form.
        """
        with torch.no_grad():
            return self.emission(self.state_matrix).softmax(dim=-1)


class _DiscreteHead:
    """
    Mixin for a decoder whose prediction goes through `self.causal_head`.

    List it BEFORE the encoder base, e.g.
        class DiscreteCausalGRU(_DiscreteHead, _GRUStack)
    so these methods override the plain decoder's defaults.  The subclass calls
    `_init_causal_head` in its __init__ and writes its own `forward`.
    """

    def _init_causal_head(self, n_states, state_dim, tau, usage_beta):
        if n_states is None:
            raise ValueError(
                f"{type(self).__name__} requires n_states.  Resolve it from the "
                "config (K = n_states_mult * V); defaulting to the vocabulary "
                "reintroduces the bottleneck this architecture removes.")
        self.tau        = tau            # temperature of the SURROGATE only
        self.usage_beta = usage_beta
        self.gumbel     = False          # Gumbel-ST while training (_learn_causal_state.forward)
        self.causal_head = _learn_causal_state(
            d_model=self.d_model, n_states=n_states,
            state_dim=n_states if state_dim is None else state_dim,
            token_size=self.token_size)

    def _sampling(self) -> bool:
        """Gumbel noise only while training, and only when switched on."""
        return bool(self.gumbel and self.training)

    def usage_penalty(self):
        """
        (H(p_bar), penalty).  H is reported rather than the KL because it is the
        quantity directly comparable to C.

        beta is 1/(batch * seq_len) -- one over the count the cross-entropy
        itself averages over -- and is COMPUTED, never a literal: it is bounded
        above by an optimisation cliff (~6e-4 at lr=1e-3) above which the
        bottleneck collapses to a single state.
        """
        H = self.causal_head.usage_entropy()
        if self.usage_beta == 0.0:
            return H, H * 0.0                      # keeps the graph
        return H, self.usage_beta * H

    def _extra_loss(self):
        H, penalty = self.usage_penalty()
        self.log("state_usage_entropy", H, prog_bar=True, on_step=True, on_epoch=True)
        return penalty

    # ── analysis surface ─────────────────────────────────────────────────
    @property
    def n_states(self):
        return self.causal_head.n_states

    @property
    def state_matrix(self):
        return self.causal_head.state_matrix

    @property
    def last_states(self):
        return self.causal_head.last_states

    @property
    def last_state_probs(self):
        return self.causal_head.last_state_probs

    def state_assignment(self, tokens=None):
        """(B,T) integer state index per position -- the real thing, no clustering."""
        if tokens is not None:
            with torch.no_grad():
                self(tokens)
        return self.causal_head.last_states

    def emission_table(self):
        """(K,V) P(next token | state) -- see _learn_causal_state.emission_table."""
        return self.causal_head.emission_table()
