"""
Model 5b: model 5a without the GRU.  (Section 16 `two_step_no_gru`.)

The parts: the Transformer, the shared state table and output layer, the two-step head and the running count pi.

The loss, in bits: model 5a's without the GRU's CE,
    loss = CE of the Transformer's one-step predictions + CE of its two-step predictions
         + weight x entropy of the inferred next state (model_5a.py gives the three steps of the calculation)

Predictors: transformer (the one the loss trains).  Default weight 0.1 (section 16's trial runs).  Note: section 16
built this model with a GRU and then removed it, so with the same seed its starting weights differ from section 16's
runs; the trained model is otherwise the same.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from causal_state_matrix import CausalStateMatrix, StateSelection
from training import prediction_loss, entropy_bits, LN2, FREQ_DECAY

NAME = "5b"
DESCRIPTION = "model 5a without the GRU"
SECTION16_NAME = "two_step_no_gru"            # the name of this model's saved runs in section 16
DEFAULTS = {"weight": 0.1}
MAIN_PREDICTOR = "transformer"                     # the predictor the loss trains on the future tokens (best.pt uses it)
AHEAD = 2                                    # tokens ahead the Transformer predicts


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
        self.encoder_norm = nn.LayerNorm(cfg.d_model) if cfg.encoder_norm else None
        self.pair_head = nn.Linear(S, V ** 2)          # two-step head: P(x_{t+1}, x_{t+2} | state), V * V outcomes
        # how often the Transformer picks each state (a running count, updated after every update): the pi of Bayes' rule
        self.register_buffer("state_freq", torch.full((K,), 1.0 / K))

    def _noise(self):
        return self.training and self.cfg.gumbel

    def transformer_pass(self, tokens, tau):
        """The Transformer's scores (N, T, K) and its chosen state at every position."""
        values = self.encoder(tokens)
        if self.encoder_norm is not None:
            values = self.encoder_norm(values)
        scores = self.transformer_selector(values)
        return scores, self.causal_head.select(scores, tau, self._noise())

    def inferred_next_state(self, representation, future):
        """The distribution of the next state worked out from the predictions (models 5a and 5b), at P positions:
        representation (B, P, S) are the vectors of the states s_t; future (B, P, 2) holds x_{t+1}, x_{t+2} (only the
        real x_{t+1} is used).  Returns r (B, P, K) and the size of the rescaling |sum - 1| (logged, not trained).
            1. q(x)     = P(x_{t+1}, x | s_t) / P(x_{t+1} | s_t), rescaled to sum to 1: the token after x_{t+1}
                          that s_t predicts (two-step head over output layer)
            2. P(k | x) = P(x | k) pi_k / sum_j P(x | j) pi_j: which state would predict x (output layer, Bayes)
            3. r(k)     = sum_x q(x) P(k | x)"""
        V = self.cfg.vocab_size
        W, E = self.causal_head.emission, self.causal_head.state_matrix
        p1 = W(representation).softmax(-1)                                               # P(x_{t+1} | s_t)
        x1 = future[..., 0]
        den = p1.gather(-1, x1.unsqueeze(-1)).clamp_min(1e-12)
        pi = self.state_freq / self.state_freq.sum()
        joint = self.pair_head(representation).softmax(-1).view(*representation.shape[:-1], V, V)
        num = joint.gather(-2, x1[..., None, None].expand(*x1.shape, 1, V)).squeeze(-2)    # P(x_{t+1}, x | s_t)
        like = W(E).softmax(-1)                                                          # P(x | k)    (K, V)
        cond = num / den
        norm_error = (cond.sum(-1) - 1).abs()
        cond = cond / cond.sum(-1, keepdim=True).clamp_min(1e-12)                        # q
        post = like * pi[:, None]
        post = post / post.sum(0, keepdim=True)                                          # P(k | x)    (K, V)
        return torch.einsum("bpw,kw->bpk", cond, post), norm_error

    # ---- training
    def loss(self, window, tau, weight):
        cfg = self.cfg
        H, L, C, V = cfg.history_len, cfg.rollout_len, cfg.context, cfg.vocab_size
        T = H + L
        scores, sel = self.transformer_pass(window, tau)
        emission = self.causal_head.emission
        terms, loss = {}, 0.0
        terms["ce_t1"] = prediction_loss(emission(sel.representation[:, C:T - 1]), window[:, C + 1:T])
        loss = loss + terms["ce_t1"]
        r2 = sel.representation[:, C:T - 2]
        pair = window[:, C + 1:T - 1] * V + window[:, C + 2:T]
        terms["ce_t2"] = F.cross_entropy(self.pair_head(r2).reshape(-1, V ** 2), pair.reshape(-1)) / LN2
        loss = loss + terms["ce_t2"]
        n = 1                                                                 # predicted tokens that identify a state
        P = T - 1 - n - C
        r = sel.representation[:, C:C + P]
        fut = torch.stack([window[:, C + 1 + j:C + 1 + j + P] for j in range(n + 1)], dim=-1)
        inferred, norm_error = self.inferred_next_state(r, fut)
        terms["inferred_entropy"] = entropy_bits(inferred).mean()
        terms["ratio_correction"] = norm_error.mean()
        loss = loss + weight * terms["inferred_entropy"]
        return loss, terms, sel.ids

    def after_update(self, ids):
        """After every update: move the running count of state choices (pi) toward this batch's counts."""
        with torch.no_grad():
            used = ids[:, self.cfg.context:].reshape(-1)
            freq = torch.bincount(used, minlength=self.cfg.n_states).to(self.state_freq) / used.numel()
            self.state_freq.mul_(FREQ_DECAY).add_((1 - FREQ_DECAY) * freq)

    # ---- evaluation (no noise)
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
        return {}

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
        """The inferred-next-state terms on fixed windows: its entropy, the rescaling size, and how often its most
        likely state is the state the Transformer actually picks next."""
        C, T = self.cfg.context, tokens.shape[1]
        ids = prediction["transformer_ids"]
        n = AHEAD - 1
        P = T - 1 - n - C
        fut = torch.stack([tokens[:, C + 1 + j:C + 1 + j + P] for j in range(n + 1)], -1)
        inferred, err = self.inferred_next_state(self.causal_head.state_matrix[ids[:, C:C + P]], fut)
        return {"inferred_entropy": float(entropy_bits(inferred).mean()), "ratio_correction": float(err.mean()),
                "inferred_predicts_transformer": float((inferred.argmax(-1) == ids[:, C + 1:C + 1 + P]).float().mean())}
