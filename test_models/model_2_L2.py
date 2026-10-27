"""
Model 2 + L2: model 2 + weight x D over the L - 1 GRU steps that the GRU's loss already makes.  (Section 17's
`l2_both_losses`.)

The parts are model 2's.  D compares the GRU's state probabilities after x_H .. x_{H+L-2} with the Transformer's at the
same positions (Euclidean distance, averaged); during training both include the random noise of the state choice.  At
weight 0 D is only measured (no gradient), so weight 0 is exactly model 2.

The loss, in bits
    loss = CE of the GRU's predictions of the L future tokens + CE of the Transformer's own next-token predictions
         + weight x D

Predictors: transformer, gru_from_handoff.  Default weight 0.03 (section 17 tried 0.01, 0.03 and 0.1).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from causal_state_matrix import CausalStateMatrix, StateSelection
from training import prediction_loss

NAME = "2_L2"
DESCRIPTION = "2 + weight x D over the L - 1 GRU steps (section 17's l2_both_losses)"
SECTION16_NAME = None
DEFAULTS = {"weight": 0.03}
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

    @staticmethod
    def probability_l2(q_transformer, q_gru):
        """The distance D (the notebook's L2): the Euclidean distance between two state-probability vectors, averaged
        over windows and positions; between 0 (equal) and sqrt(2) (two different one-hot vectors)."""
        return torch.linalg.vector_norm(q_transformer - q_gru, ord=2, dim=-1).mean()

    # ---- training
    def loss(self, window, tau, weight):
        """Model 2's loss + weight x D over the L - 1 GRU steps the GRU's loss already makes (section 17's
        hybrid_loss): the GRU's state probabilities after x_H .. x_{H+L-2} against the Transformer's at the same
        positions.  At weight 0 D is measured without gradient, so weight 0 is exactly model 2."""
        cfg = self.cfg
        H, L, C = cfg.history_len, cfg.rollout_len, cfg.context
        T = H + L
        scores, sel = self.transformer_pass(window, tau)
        emission = self.causal_head.emission
        terms, loss = {}, 0.0
        state = StateSelection(sel.probs[:, H - 1], sel.onehot[:, H - 1], sel.ids[:, H - 1],
                               sel.representation[:, H - 1])
        token_logits, q_gru = [], []
        for k in range(L):
            token_logits.append(emission(state.representation))              # predicts x_{H+k}
            if k + 1 < L:
                state = self.transition(state, window[:, H + k], tau)          # the state after reading x_{H+k}
                q_gru.append(state.probs)
        terms["ce_gru"] = prediction_loss(torch.stack(token_logits, 1), window[:, H:])
        loss = loss + terms["ce_gru"]
        terms["ce_t1"] = prediction_loss(emission(sel.representation[:, C:T - 1]), window[:, C + 1:T])
        loss = loss + terms["ce_t1"]
        q_gru, q_transformer = torch.stack(q_gru, 1), sel.probs[:, H:H + L - 1]
        if weight > 0:
            terms["l2"] = self.probability_l2(q_transformer, q_gru)
            loss = loss + weight * terms["l2"]
        else:
            with torch.no_grad():
                terms["l2"] = self.probability_l2(q_transformer, q_gru)
        return loss, terms, sel.ids

    # ---- evaluation (no noise)
    def gru_step(self, ids, tokens):
        """The GRU's fixed step rule at evaluation: (state ids, tokens) -> next state ids (highest score)."""
        E = self.causal_head.state_matrix
        return self.gru_selector(self.gru_cell(F.one_hot(tokens, self.cfg.vocab_size).to(E.dtype), E[ids])).argmax(-1)

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
        return {"gru_from_handoff": self.gru_step}

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
        """D without noise on fixed windows, over the L - 1 GRU steps of the loss (section 17's mean_l2)."""
        return {"l2": self.mean_l2(tokens, tau, self.cfg.rollout_len - 1)}

    @torch.no_grad()
    def mean_l2(self, tokens, tau, steps):
        H = self.cfg.history_len
        values = []
        for batch in tokens.split(256):
            _, sel = self.transformer_pass(batch, tau)
            state = StateSelection(sel.probs[:, H - 1], sel.onehot[:, H - 1], sel.ids[:, H - 1],
                                   sel.representation[:, H - 1])
            q = []
            for k in range(steps):
                state = self.transition(state, batch[:, H + k], tau)
                q.append(state.probs)
            values.append(torch.linalg.vector_norm(sel.probs[:, H:H + steps] - torch.stack(q, 1), ord=2, dim=-1))
        return float(torch.cat(values).mean())
