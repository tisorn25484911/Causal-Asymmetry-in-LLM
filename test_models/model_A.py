"""
Model A: model 3 with the Transformer's actual next state as a FIXED target.  (Section 16 `next_state_fixed_target`.)

The parts and the loss are model 3's (model_3.py), except that the state the Transformer picks at t + 1 is detached
in the next-state CE: no gradient of the next-state terms reaches the Transformer through its next state.  The network
learns to follow the Transformer, and the Transformer is not pulled toward states that are easy for the network to
predict.  Holding the prediction target fixed is the known remedy against the collapse to a constant state in models
that predict their own next internal state (Tang et al. 2023; Chen & He 2021).

Predictors: transformer, gru_from_handoff, next_state_from_handoff.  Default weight 0.01 (section 16's trial runs).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from causal_state_matrix import CausalStateMatrix, StateSelection
from training import prediction_loss, entropy_bits, LN2

NAME = "A"
DESCRIPTION = "model 3 with the Transformer's next state as a fixed target"
SECTION16_NAME = "next_state_fixed_target"            # the name of this model's saved runs in section 16
DEFAULTS = {"weight": 0.01}
MAIN_PREDICTOR = "gru_from_handoff"                     # the predictor the loss trains on the future tokens (best.pt uses it)


# ------------------------------------------------------------------------------------------- the Transformer
class CausalAttentionBlock(nn.Module):
    """One Transformer layer: LayerNorm -> single-head attention over the current and earlier positions -> add to the
    input; then LayerNorm -> two-layer ReLU network (width 4 x) -> add to the input."""

    def __init__(self, width):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.ReLU(), nn.Linear(4 * width, width))

    def forward(self, x, allowed):
        z = self.norm_attention(x)
        scores = self.query(z) @ self.key(z).transpose(-2, -1)
        scores = scores / math.sqrt(x.shape[-1])
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
        x = x + weights @ self.value(z)
        return x + self.ffn(self.norm_ffn(x))


class HistoryEncoder(nn.Module):
    """Tokens -> one vector per position that depends only on the tokens up to it.  Each token becomes a fixed random
    vector of length 1 (not trained), a fixed sine/cosine pattern of the position is added, then n_layers layers."""

    def __init__(self, cfg):
        super().__init__()
        projection = F.normalize(torch.randn(cfg.vocab_size, cfg.d_model), dim=-1)
        self.register_buffer("token_vectors", projection)
        positions = torch.arange(cfg.total_len).float().unsqueeze(1)
        divisors = 10000 ** (torch.arange(0, cfg.d_model, 2).float() / cfg.d_model)
        pe = torch.zeros(cfg.total_len, cfg.d_model)
        pe[:, 0::2] = torch.sin(positions / divisors)
        pe[:, 1::2] = torch.cos(positions / divisors)[:, :cfg.d_model // 2]
        self.register_buffer("positions", pe)
        self.blocks = nn.ModuleList([CausalAttentionBlock(cfg.d_model) for _ in range(cfg.n_layers)])

    def forward(self, tokens):
        length = tokens.shape[1]
        if length < 1 or length > len(self.positions):
            raise ValueError("history length is outside the configured window")
        x = self.token_vectors[tokens] + self.positions[:length]
        if x.device.type == "mps" and torch.is_grad_enabled() and not x.requires_grad:
            x.requires_grad_(True)                     # the project's MPS workaround for LayerNorm on buffer-only inputs
        allowed = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
        for block in self.blocks:
            x = block(x, allowed)
        return x


# ------------------------------------------------------------------------------------------- the model
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        K, S, V = cfg.n_states, cfg.state_dim, cfg.vocab_size
        # the order in which the layers are built sets their starting random weights; it is section 16's order
        self.encoder = HistoryEncoder(cfg)
        self.transformer_selector = nn.Linear(cfg.d_model, K)                # the Transformer's scoring layer
        self.causal_head = CausalStateMatrix(K, S, V)
        self.gru_cell = nn.GRUCell(V, S)
        self.gru_selector = nn.Linear(S, K)                                   # the GRU's scoring layer
        self.encoder_norm = nn.LayerNorm(cfg.d_model) if cfg.encoder_norm else None
        # the next-state network: (state vector, next token) -> scores of the K possible next states
        self.next_state_net = nn.Sequential(nn.Linear(S + V, cfg.next_hidden), nn.ReLU(), nn.Linear(cfg.next_hidden, K))

    def _noise(self):
        return self.training and self.cfg.gumbel

    def transformer_pass(self, tokens, tau):
        """The Transformer's scores (N, T, K) and its chosen state at every position."""
        values = self.encoder(tokens)
        if self.encoder_norm is not None:
            values = self.encoder_norm(values)
        scores = self.transformer_selector(values)
        return scores, self.causal_head.select(scores, tau, self._noise())

    def transition(self, state, token, tau):
        """One GRU step during training: (current state, real token) -> the next state."""
        symbol = F.one_hot(token, self.cfg.vocab_size).to(state.representation.dtype)
        scores = self.gru_selector(self.gru_cell(symbol, state.representation))
        return self.causal_head.select(scores, tau, self._noise())

    def next_state_logits(self, representation, next_tokens):
        """The next-state network: scores of the K next states from (state vectors, next tokens)."""
        symbol = F.one_hot(next_tokens, self.cfg.vocab_size).to(representation.dtype)
        return self.next_state_net(torch.cat([representation, symbol], dim=-1))

    # ---- training
    def loss(self, window, tau, weight):
        cfg = self.cfg
        H, L, C, V = cfg.history_len, cfg.rollout_len, cfg.context, cfg.vocab_size
        T = H + L
        scores, sel = self.transformer_pass(window, tau)
        emission = self.causal_head.emission
        terms, loss = {}, 0.0
        state = StateSelection(sel.probs[:, H - 1], sel.onehot[:, H - 1], sel.ids[:, H - 1],
                               sel.representation[:, H - 1])
        token_logits = []
        for k in range(L):
            token_logits.append(emission(state.representation))              # predicts x_{H+k}
            if k + 1 < L:
                state = self.transition(state, window[:, H + k], tau)          # the state after reading x_{H+k}
        terms["ce_gru"] = prediction_loss(torch.stack(token_logits, 1), window[:, H:])
        loss = loss + terms["ce_gru"]
        terms["ce_t1"] = prediction_loss(emission(sel.representation[:, C:T - 1]), window[:, C + 1:T])
        loss = loss + terms["ce_t1"]
        z = self.next_state_logits(sel.representation[:, C:T - 1], window[:, C + 1:T])
        logp = z.log_softmax(-1)
        target = sel.onehot[:, C + 1:T]                                       # the state the Transformer picks next
        target = target.detach()                                              # a fixed target: no gradient through it
        terms["next_entropy"] = entropy_bits(logp.exp()).mean()
        terms["next_nll"] = -(target * logp).sum(-1).mean() / LN2
        loss = loss + weight * (terms["next_entropy"] + terms["next_nll"])
        return loss, terms, sel.ids

    # ---- evaluation (no noise)
    def gru_step(self, ids, tokens):
        """The GRU's fixed step rule at evaluation: (state ids, tokens) -> next state ids (highest score)."""
        E = self.causal_head.state_matrix
        return self.gru_selector(self.gru_cell(F.one_hot(tokens, self.cfg.vocab_size).to(E.dtype), E[ids])).argmax(-1)

    def next_state_step(self, ids, tokens):
        """The next-state network's step at evaluation: (state ids, tokens) -> the most likely next state."""
        return self.next_state_logits(self.causal_head.state_matrix[ids], tokens).argmax(-1)

    def rollout(self, start_ids, future, step):
        """Step through `future` from `start_ids` with `step`: the states (N, n) that predict each future token and
        their token logits (N, n, V)."""
        E, W = self.causal_head.state_matrix, self.causal_head.emission
        ids, states, logits = start_ids, [], []
        for k in range(future.shape[1]):
            states.append(ids)
            logits.append(W(E[ids]))
            ids = step(ids, future[:, k])
        return torch.stack(states, 1), torch.stack(logits, 1)

    @torch.no_grad()
    def transformer_states(self, tokens, tau):
        """The Transformer's scores (N, T, K) and chosen states (N, T) without noise, in batches of 256."""
        scores, ids = [], []
        for batch in tokens.split(256):
            s, sel = self.transformer_pass(batch, tau)
            scores.append(s)
            ids.append(sel.ids)
        return torch.cat(scores), torch.cat(ids)

    @torch.no_grad()
    def handoff_states(self, history, tau):
        """The Transformer's state after reading the history: the starting state of the step rules."""
        return self.transformer_states(history, tau)[1][:, -1]

    def step_rules(self):
        """The model's fixed step rules (state, token) -> next state, by predictor name (analysis.py reads the machine
        each one defines)."""
        return {"gru_from_handoff": self.gru_step, "next_state_from_handoff": self.next_state_step}

    @torch.no_grad()
    def predict(self, tokens, tau):
        H, C = self.cfg.history_len, self.cfg.context
        T = tokens.shape[1]
        scores, ids = self.transformer_states(tokens, tau)
        E, W = self.causal_head.state_matrix, self.causal_head.emission
        predictors = {"transformer": dict(first=C + 1, logits=W(E[ids[:, C:T - 1]]), states=ids[:, C:T])}
        for name, step in self.step_rules().items():
            states, logits = self.rollout(ids[:, H - 1], tokens[:, H:], step)
            predictors[name] = dict(first=H, logits=logits, states=states)
        return dict(predictors=predictors, handoff_scores=scores[:, H - 1], transformer_ids=ids)

    @torch.no_grad()
    def measures(self, tokens, tau, prediction):
        """The next-state terms on fixed windows: the entropy of the network's next-state probabilities, and how often
        its most likely next state is the state the Transformer actually picks next (positions t >= context)."""
        C, T = self.cfg.context, tokens.shape[1]
        ids = prediction["transformer_ids"]
        logp = self.next_state_logits(self.causal_head.state_matrix[ids[:, C:T - 1]], tokens[:, C + 1:T]).log_softmax(-1)
        return {"next_entropy": float(entropy_bits(logp.exp()).mean()),
                "next_predicts_transformer": float((logp.argmax(-1) == ids[:, C + 1:T]).float().mean())}
