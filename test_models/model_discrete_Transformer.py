"""
Discrete Transformer: a Transformer alone whose prediction at every position goes through one state of the state table.
(Section 17's `discrete_transformer`: the modular DiscreteCausalDecoder of model_modeularised/Transformer.py, rewritten
here self-contained.)

The model
    Each token becomes a fixed random vector of length 1 (a buffer, not trained) and a sine/cosine pattern of the
    position is added.  n_layers layers follow; each: LayerNorm -> single-head attention over the current and earlier
    positions -> add to the input; LayerNorm -> two-layer ReLU network (width 4 x) -> add to the input.  At every
    position the scoring layer gives a score to each of the K states, one state is chosen (random noise added during
    training, the highest score at evaluation), and the shared output layer predicts the next token from its vector.

The loss, in bits
    loss = CE of its own next-token predictions at every position t >= context (19): it reads x_0 .. x_{T-2} and the
    output at t predicts x_{t+1}.

Compared with the hybrid's Transformer: no GRU; no LayerNorm on the output; masked attention scores are set to -1e9
(not -inf); the state table starts from random values and the noise clamp is 1e-10 (causal_state_matrix.py).
Its states need not follow a fixed rule (state + token -> next state); analysis.py reads a machine off the sequence of
states it picks, keeping how often each next state follows.

Predictor: transformer (its own predictions; at every position it reads all real tokens so far).
"""
import torch
from torch import nn
from torch.nn import functional as F

from causal_state_matrix import CausalStateMatrix
from training import LN2

NAME = "discrete_Transformer"
DESCRIPTION = "a Transformer alone with a state choice at every position (section 17)"
SECTION17_NAME = "discrete_transformer"     # the name of this model's saved runs in section 17
DEFAULTS = {"d_model": 16, "n_layers": 4, "encoder_norm": False}
MAIN_PREDICTOR = "transformer"
NOISE_CLAMP = 1e-10


class PositionalEncoding(nn.Module):
    """Sine/cosine pattern of the position, grown if a window is longer than max_len."""

    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self._build(max_len)

    def _build(self, max_len, device=None, dtype=torch.float32):
        pe = torch.zeros(max_len, self.d_model, device=device, dtype=dtype)
        pos = torch.arange(0, max_len, device=device, dtype=dtype).unsqueeze(1)
        div = 10000 ** (torch.arange(0, self.d_model, 2, device=device, dtype=dtype) / self.d_model)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)[:, : self.d_model // 2]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        T = x.shape[1]
        if T > self.pe.size(0):
            self._build(max(T, self.pe.size(0) * 2), device=x.device, dtype=x.dtype)
        return x + self.pe[:T].unsqueeze(0)


class AttentionModel(nn.Module):
    """Single-head scaled dot-product attention; mask True = allowed."""

    def __init__(self, d_model):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, q, k, v, mask):
        Q, K, V = self.wq(q), self.wk(k), self.wv(v)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_model ** 0.5)
        scores = scores.masked_fill(~mask, -1e9)
        return torch.softmax(scores, dim=-1) @ V


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        K, D, S, V, L = cfg.n_states, cfg.d_model, cfg.state_dim, cfg.vocab_size, cfg.n_layers
        # the build order sets the starting random weights; it is Transformer.py's order
        self.register_buffer("rand_prj", F.normalize(torch.randn(V, D), dim=1))
        self.pe = PositionalEncoding(D, cfg.total_len)
        self.attn_layers = nn.ModuleList([AttentionModel(D) for _ in range(L)])
        self.ffn_layers = nn.ModuleList([nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))
                                         for _ in range(L)])
        self.ln_attn = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.ln_ffn = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.state_head = nn.Linear(D, K)                                    # the scoring layer
        self.causal_head = CausalStateMatrix(K, S, V, start="random")

    def encode(self, tokens):
        x = F.one_hot(tokens, num_classes=self.cfg.vocab_size).float() @ self.rand_prj
        x = self.pe(x)
        mask = torch.ones((x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool).tril().unsqueeze(0)
        for attn, ffn, ln1, ln2 in zip(self.attn_layers, self.ffn_layers, self.ln_attn, self.ln_ffn):
            # ln1(x) is computed three times, as in Transformer.py: the same value, but the same order of the
            # gradient sums, so that training matches the original bit for bit
            x = x + attn(ln1(x), ln1(x), ln1(x), mask=mask)
            x = x + ffn(ln2(x))
        return x

    def run(self, tokens, tau):
        """Token logits (N, T, V) (logits[:, t] predicts x_{t+1}), chosen states (N, T) and scores (N, T, K)."""
        scores = self.state_head(self.encode(tokens))
        sel = self.causal_head.select(scores, tau, self.training and self.cfg.gumbel, NOISE_CLAMP)
        return self.causal_head.emission(sel.representation), sel.ids, scores

    # ---- training
    def loss(self, window, tau, weight):
        C, V = self.cfg.context, self.cfg.vocab_size
        logits, ids, _ = self.run(window[:, :-1], tau)
        ce = F.cross_entropy(logits[:, C:].reshape(-1, V), window[:, C + 1:].reshape(-1)) / LN2
        return ce, {"ce": ce}, ids

    # ---- evaluation (no noise)
    def step_rules(self):
        return {}                       # no fixed step rule: the machine is read off its sequence of states

    @torch.no_grad()
    def own_states(self, tokens, tau):
        out = [self.run(batch, tau) for batch in tokens.split(256)]
        return tuple(torch.cat(part) for part in zip(*out))

    @torch.no_grad()
    def handoff_states(self, history, tau):
        return self.own_states(history, tau)[1][:, -1]

    @torch.no_grad()
    def predict(self, tokens, tau):
        C, H = self.cfg.context, self.cfg.history_len
        T = tokens.shape[1]
        logits, ids, scores = self.own_states(tokens, tau)
        return dict(predictors={"transformer": dict(first=C + 1, logits=logits[:, C:T - 1], states=ids[:, C:T])},
                    handoff_scores=scores[:, H - 1], transformer_ids=ids)
