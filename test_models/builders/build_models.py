"""Write the hybrid model files of test_models/ (model_1, 2, 3, 4, 5a, 5b, A, B, C, 1_L2, 2_L2) from shared text blocks,
so that the code every file copies (the Transformer, the GRU step, evaluation) is identical in all of them.  Each
written file is self-contained: it imports only causal_state_matrix and training."""
from pathlib import Path

OUT = Path("/Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version/test_models")

COMMON_MODEL_TEXT = "the model is model 2's (see model_2.py for the full description of its parts)"

TRANSFORMER_BLOCK = '''

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
'''


def model_class(spec):
    gru, nxt, ahead, copy, l2 = spec["gru"], spec["next"], spec["ahead"], spec["copy"], spec["l2"]
    lines = ['''

# ------------------------------------------------------------------------------------------- the model
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        K, S, V = cfg.n_states, cfg.state_dim, cfg.vocab_size
        # the order in which the layers are built sets their starting random weights; it is section 16's order
        self.encoder = HistoryEncoder(cfg)
        self.transformer_selector = nn.Linear(cfg.d_model, K)                # the Transformer's scoring layer
        self.causal_head = CausalStateMatrix(K, S, V)''']
    if gru:
        lines.append('''
        self.gru_cell = nn.GRUCell(V, S)
        self.gru_selector = nn.Linear(S, K)                                   # the GRU's scoring layer''')
    lines.append('''
        self.encoder_norm = nn.LayerNorm(cfg.d_model) if cfg.encoder_norm else None''')
    if nxt:
        lines.append('''
        # the next-state network: (state vector, next token) -> scores of the K possible next states
        self.next_state_net = nn.Sequential(nn.Linear(S + V, cfg.next_hidden), nn.ReLU(), nn.Linear(cfg.next_hidden, K))''')
    if ahead >= 2:
        lines.append('''
        self.pair_head = nn.Linear(S, V ** 2)          # two-step head: P(x_{t+1}, x_{t+2} | state), V * V outcomes''')
    if ahead >= 3:
        lines.append('''
        self.triple_head = nn.Linear(S, V ** 3)        # three-step head: P(x_{t+1}, x_{t+2}, x_{t+3} | state)''')
    if ahead >= 2:
        lines.append('''
        # how often the Transformer picks each state (a running count, updated after every update): the pi of Bayes' rule
        self.register_buffer("state_freq", torch.full((K,), 1.0 / K))''')
    lines.append('''

    def _noise(self):
        return self.training and self.cfg.gumbel

    def transformer_pass(self, tokens, tau):
        """The Transformer's scores (N, T, K) and its chosen state at every position."""
        values = self.encoder(tokens)
        if self.encoder_norm is not None:
            values = self.encoder_norm(values)
        scores = self.transformer_selector(values)
        return scores, self.causal_head.select(scores, tau, self._noise())''')
    if gru:
        lines.append('''

    def transition(self, state, token, tau):
        """One GRU step during training: (current state, real token) -> the next state."""
        symbol = F.one_hot(token, self.cfg.vocab_size).to(state.representation.dtype)
        scores = self.gru_selector(self.gru_cell(symbol, state.representation))
        return self.causal_head.select(scores, tau, self._noise())''')
    if nxt:
        lines.append('''

    def next_state_logits(self, representation, next_tokens):
        """The next-state network: scores of the K next states from (state vectors, next tokens)."""
        symbol = F.one_hot(next_tokens, self.cfg.vocab_size).to(representation.dtype)
        return self.next_state_net(torch.cat([representation, symbol], dim=-1))''')
    if ahead >= 2:
        lines.append(INFERRED[ahead])
    if copy:
        lines.append('''

    @torch.no_grad()
    def gru_state_after_history(self, history):
        """The copy target: the GRU's own state after reading the history from row 0 of the table, taking the
        highest-scoring state at every step (no noise, no gradient)."""
        ids = torch.zeros(history.shape[0], dtype=torch.long, device=history.device)
        E = self.causal_head.state_matrix
        for j in range(history.shape[1]):
            symbol = F.one_hot(history[:, j], self.cfg.vocab_size).to(E.dtype)
            ids = self.gru_selector(self.gru_cell(symbol, E[ids])).argmax(-1)
        return ids''')
    if l2:
        lines.append('''

    @staticmethod
    def probability_l2(q_transformer, q_gru):
        """The distance D (the notebook's L2): the Euclidean distance between two state-probability vectors, averaged
        over windows and positions; between 0 (equal) and sqrt(2) (two different one-hot vectors)."""
        return torch.linalg.vector_norm(q_transformer - q_gru, ord=2, dim=-1).mean()''')
    lines.append(LOSS[spec["key"]])
    if ahead >= 2:
        lines.append('''

    def after_update(self, ids):
        """After every update: move the running count of state choices (pi) toward this batch's counts."""
        with torch.no_grad():
            used = ids[:, self.cfg.context:].reshape(-1)
            freq = torch.bincount(used, minlength=self.cfg.n_states).to(self.state_freq) / used.numel()
            self.state_freq.mul_(FREQ_DECAY).add_((1 - FREQ_DECAY) * freq)''')
    lines.append('''

    # ---- evaluation (no noise)''')
    if gru:
        lines.append('''
    def gru_step(self, ids, tokens):
        """The GRU's fixed step rule at evaluation: (state ids, tokens) -> next state ids (highest score)."""
        E = self.causal_head.state_matrix
        return self.gru_selector(self.gru_cell(F.one_hot(tokens, self.cfg.vocab_size).to(E.dtype), E[ids])).argmax(-1)
''')
    if nxt:
        lines.append('''
    def next_state_step(self, ids, tokens):
        """The next-state network's step at evaluation: (state ids, tokens) -> the most likely next state."""
        return self.next_state_logits(self.causal_head.state_matrix[ids], tokens).argmax(-1)
''')
    rules = []
    if gru:
        rules.append('"gru_from_handoff": self.gru_step')
    if nxt:
        rules.append('"next_state_from_handoff": self.next_state_step')
    lines.append('''
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
        return {''' + ", ".join(rules) + '''}

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
        return dict(predictors=predictors, handoff_scores=scores[:, H - 1], transformer_ids=ids)''')
    if spec["measures"]:
        lines.append(MEASURES[spec["measures"]])
    return "".join(lines) + "\n"


INFERRED = {2: '''

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
        return torch.einsum("bpw,kw->bpk", cond, post), norm_error''',
            3: '''

    def inferred_next_state(self, representation, future):
        """The distribution of the next state worked out from the predictions of the next TWO tokens (model C), at P
        positions: representation (B, P, S) are the vectors of the states s_t; future (B, P, 3) holds x_{t+1..t+3}
        (only the real x_{t+1} is used).  Returns r (B, P, K) and the size of the rescaling |sum - 1| (logged).
            1. q(w)     = P(x_{t+1}, w | s_t) / P(x_{t+1} | s_t), rescaled: the pair of tokens w after x_{t+1}
                          (three-step head over output layer)
            2. P(k | w) = P(w | k) pi_k / sum_j P(w | j) pi_j: which state would predict w (two-step head, Bayes)
            3. r(k)     = sum_w q(w) P(k | w)"""
        V = self.cfg.vocab_size
        W, E = self.causal_head.emission, self.causal_head.state_matrix
        p1 = W(representation).softmax(-1)                                               # P(x_{t+1} | s_t)
        x1 = future[..., 0]
        den = p1.gather(-1, x1.unsqueeze(-1)).clamp_min(1e-12)
        pi = self.state_freq / self.state_freq.sum()
        joint = self.triple_head(representation).softmax(-1).view(*representation.shape[:-1], V, V * V)
        num = joint.gather(-2, x1[..., None, None].expand(*x1.shape, 1, V * V)).squeeze(-2)
        like = self.pair_head(E).softmax(-1)                                             # P(x, x' | k)   (K, V*V)
        cond = num / den
        norm_error = (cond.sum(-1) - 1).abs()
        cond = cond / cond.sum(-1, keepdim=True).clamp_min(1e-12)                        # q
        post = like * pi[:, None]
        post = post / post.sum(0, keepdim=True)                                          # P(k | w)
        return torch.einsum("bpw,kw->bpk", cond, post), norm_error'''}

LOSS_HEAD = '''

    # ---- training
    def loss(self, window, tau, weight):
        cfg = self.cfg
        H, L, C, V = cfg.history_len, cfg.rollout_len, cfg.context, cfg.vocab_size
        T = H + L
        scores, sel = self.transformer_pass(window, tau)
        emission = self.causal_head.emission
        terms, loss = {}, 0.0'''
LOSS_GRU = '''
        state = StateSelection(sel.probs[:, H - 1], sel.onehot[:, H - 1], sel.ids[:, H - 1],
                               sel.representation[:, H - 1])
        token_logits = []
        for k in range(L):
            token_logits.append(emission(state.representation))              # predicts x_{H+k}
            if k + 1 < L:
                state = self.transition(state, window[:, H + k], tau)          # the state after reading x_{H+k}
        terms["ce_gru"] = prediction_loss(torch.stack(token_logits, 1), window[:, H:])
        loss = loss + terms["ce_gru"]'''
LOSS_T1 = '''
        terms["ce_t1"] = prediction_loss(emission(sel.representation[:, C:T - 1]), window[:, C + 1:T])
        loss = loss + terms["ce_t1"]'''
LOSS_NEXT = '''
        z = self.next_state_logits(sel.representation[:, C:T - 1], window[:, C + 1:T])
        logp = z.log_softmax(-1)
        target = sel.onehot[:, C + 1:T]                                       # the state the Transformer picks next{detach_comment}
        terms["next_entropy"] = entropy_bits(logp.exp()).mean()
        terms["next_nll"] = -(target * logp).sum(-1).mean() / LN2
        loss = loss + weight * (terms["next_entropy"] + terms["next_nll"])'''
LOSS_AHEAD = '''
        r2 = sel.representation[:, C:T - 2]
        pair = window[:, C + 1:T - 1] * V + window[:, C + 2:T]
        terms["ce_t2"] = F.cross_entropy(self.pair_head(r2).reshape(-1, V ** 2), pair.reshape(-1)) / LN2
        loss = loss + terms["ce_t2"]'''
LOSS_AHEAD3 = '''
        r3 = sel.representation[:, C:T - 3]
        triple = window[:, C + 1:T - 2] * V * V + window[:, C + 2:T - 1] * V + window[:, C + 3:T]
        terms["ce_t3"] = F.cross_entropy(self.triple_head(r3).reshape(-1, V ** 3), triple.reshape(-1)) / LN2
        loss = loss + terms["ce_t3"]'''
LOSS_INFER = '''
        n = {n}                                                                 # predicted tokens that identify a state
        P = T - 1 - n - C
        r = sel.representation[:, C:C + P]
        fut = torch.stack([window[:, C + 1 + j:C + 1 + j + P] for j in range(n + 1)], dim=-1)
        inferred, norm_error = self.inferred_next_state(r, fut)
        terms["inferred_entropy"] = entropy_bits(inferred).mean()
        terms["ratio_correction"] = norm_error.mean()
        loss = loss + weight * terms["inferred_entropy"]'''
LOSS_COPY = '''
        target = self.gru_state_after_history(window[:, :H])
        terms["copy_ce"] = F.cross_entropy(scores[:, H - 1], target) / LN2
        terms["copy_agree"] = (scores[:, H - 1].argmax(-1) == target).float().mean()
        loss = loss + weight * terms["copy_ce"]'''
LOSS_END = '''
        return loss, terms, sel.ids'''

LOSS_1_L2 = '''

    # ---- training
    def loss(self, window, tau, weight):
        """The notebook's train_step (section 7): the GRU's future-token CE + weight x D, where D compares the GRU's
        state probabilities after EACH of the L future tokens (the last step included) with the Transformer's at the
        same positions."""
        cfg = self.cfg
        H, L = cfg.history_len, cfg.rollout_len
        scores, sel = self.transformer_pass(window, tau)
        state = StateSelection(sel.probs[:, H - 1], sel.onehot[:, H - 1], sel.ids[:, H - 1],
                               sel.representation[:, H - 1])
        token_logits, q_gru = [], []
        for k in range(L):
            token_logits.append(self.causal_head.emission(state.representation))   # predicts x_{H+k}
            state = self.transition(state, window[:, H + k], tau)                   # the state after x_{H+k}
            q_gru.append(state.probs)
        terms = {"ce_gru": prediction_loss(torch.stack(token_logits, 1), window[:, H:])}
        terms["l2"] = self.probability_l2(sel.probs[:, H:H + L], torch.stack(q_gru, 1))
        loss = terms["ce_gru"] + weight * terms["l2"]
        return loss, terms, sel.ids'''

LOSS_2_L2 = '''

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
        return loss, terms, sel.ids'''


def build_loss(key, spec):
    if key == "1_L2":
        return LOSS_1_L2
    if key == "2_L2":
        return LOSS_2_L2
    parts = [LOSS_HEAD]
    if spec["gru"]:
        parts.append(LOSS_GRU)
    if spec["t1"]:
        parts.append(LOSS_T1)
    if spec["next"]:
        parts.append(LOSS_NEXT.format(detach_comment=("\n        target = target.detach()                              "
                                                      "                # a fixed target: no gradient through it"
                                                      if spec["next"] == "fixed" else "")))
    if spec["ahead"] >= 2:
        parts.append(LOSS_AHEAD)
        if spec["ahead"] == 3:
            parts.append(LOSS_AHEAD3)
        parts.append(LOSS_INFER.format(n=spec["ahead"] - 1))
    if spec["copy"]:
        parts.append(LOSS_COPY)
    parts.append(LOSS_END)
    return "".join(parts)


MEASURES = {
    "next": '''

    @torch.no_grad()
    def measures(self, tokens, tau, prediction):
        """The next-state terms on fixed windows: the entropy of the network's next-state probabilities, and how often
        its most likely next state is the state the Transformer actually picks next (positions t >= context)."""
        C, T = self.cfg.context, tokens.shape[1]
        ids = prediction["transformer_ids"]
        logp = self.next_state_logits(self.causal_head.state_matrix[ids[:, C:T - 1]], tokens[:, C + 1:T]).log_softmax(-1)
        return {"next_entropy": float(entropy_bits(logp.exp()).mean()),
                "next_predicts_transformer": float((logp.argmax(-1) == ids[:, C + 1:T]).float().mean())}''',
    "inferred": '''

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
                "inferred_predicts_transformer": float((inferred.argmax(-1) == ids[:, C + 1:C + 1 + P]).float().mean())}''',
    "copy": '''

    @torch.no_grad()
    def measures(self, tokens, tau, prediction):
        """The share of windows whose starting state for the GRU (the Transformer's state after the history) is the
        GRU's own state after reading the history (the copy target)."""
        H = self.cfg.history_len
        target = self.gru_state_after_history(tokens[:, :H])
        return {"copy_agree": float((prediction["transformer_ids"][:, H - 1] == target).float().mean())}''',
    "l2_all": '''

    @torch.no_grad()
    def measures(self, tokens, tau, prediction):
        """D without noise on fixed windows, over all L future tokens (the notebook's final_diagnostics mean_l2)."""
        return {"l2": self.mean_l2(tokens, tau, self.cfg.rollout_len)}

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
        return float(torch.cat(values).mean())''',
    "l2_steps": '''

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
        return float(torch.cat(values).mean())'''}

LOSS = {}

SPECS = {
    "1": dict(gru=True, t1=False, next=None, ahead=1, copy=False, l2=None, measures=None, main="gru_from_handoff",
              s16="gru_loss", weight=None, description="the GRU's future-token CE only"),
    "2": dict(gru=True, t1=True, next=None, ahead=1, copy=False, l2=None, measures=None, main="gru_from_handoff",
              s16="both_losses", weight=None, description="GRU's future-token CE + the Transformer's own next-token CE"),
    "3": dict(gru=True, t1=True, next="free", ahead=1, copy=False, l2=None, measures="next", main="gru_from_handoff",
              s16="next_state_net", weight=0.001, description="model 2 + a next-state network following the Transformer"),
    "4": dict(gru=False, t1=True, next="free", ahead=1, copy=False, l2=None, measures="next", main="transformer",
              s16="next_state_net_no_gru", weight=0.001, description="model 3 without the GRU"),
    "A": dict(gru=True, t1=True, next="fixed", ahead=1, copy=False, l2=None, measures="next", main="gru_from_handoff",
              s16="next_state_fixed_target", weight=0.01,
              description="model 3 with the Transformer's next state as a fixed target"),
    "5a": dict(gru=True, t1=True, next=None, ahead=2, copy=False, l2=None, measures="inferred", main="gru_from_handoff",
               s16="two_step", weight=0.01,
               description="1 + one- and two-step CE + entropy of the next state worked out from them"),
    "5b": dict(gru=False, t1=True, next=None, ahead=2, copy=False, l2=None, measures="inferred", main="transformer",
               s16="two_step_no_gru", weight=0.1, description="model 5a without the GRU"),
    "C": dict(gru=False, t1=True, next=None, ahead=3, copy=False, l2=None, measures="inferred", main="transformer",
              s16="three_step_no_gru", weight=0.01, description="model 5b with the next state worked out from two tokens"),
    "B": dict(gru=True, t1=False, next=None, ahead=1, copy=True, l2=None, measures="copy", main="gru_from_handoff",
              s16="copy_gru_handoff", weight=3.0,
              description="1 + the Transformer's starting state trained to copy the GRU's own state"),
    "1_L2": dict(gru=True, t1=False, next=None, ahead=1, copy=False, l2="all", measures="l2_all", main="gru_from_handoff",
                 s16=None, weight=0.03, description="1 + weight x D over all L steps (the notebook's L2, sections 13, 15)"),
    "2_L2": dict(gru=True, t1=True, next=None, ahead=1, copy=False, l2="steps", measures="l2_steps",
                 main="gru_from_handoff", s16=None, weight=0.03,
                 description="2 + weight x D over the L - 1 GRU steps (section 17's l2_both_losses)"),
}
for key, spec in SPECS.items():
    spec["key"] = key
    LOSS[key] = build_loss(key, spec)

DOCS = {}
DOCS["1"] = '''"""
Model 1: the hybrid trained only on the GRU's predictions of the future tokens.  (Section 16 `gru_loss`; the loss of
sections 1-13 of hybrid_model_test.ipynb.)

The model is model 2's (model_2.py describes its parts): a Transformer picks a state at every position; a GRU starts
from the Transformer's state at the last history position ("the handoff") and steps through the real future tokens;
every chosen state predicts the next token through the shared output layer (causal_state_matrix.py).

The loss, in bits
    loss = CE of the GRU's predictions of the L future tokens

The Transformer is trained only through the GRU: the gradient reaches it through the starting state it gives the GRU.
Its predictions at the other positions are not trained, but they are still measured (predictor "transformer").

"1 (original setup)" of section 16 is this model with training.ORIGINAL_SETUP: no LayerNorm on the Transformer's
output, no weight decay and no gradient clipping.

Predictors: transformer (not trained by this loss), gru_from_handoff (the one the loss trains).
"""'''
DOCS["2"] = None   # model_2.py keeps its hand-written docstring (read from the file)
DOCS["3"] = '''"""
Model 3: model 2 + a next-state network trained to follow the Transformer's states.  (Section 16 `next_state_net`.)

The parts are model 2's (model_2.py), plus a next-state network: two layers (hidden width next_hidden = 64) that read a
state vector and the next token and give a score to each of the K possible next states.

The loss, in bits (at every position t >= context, the network reads the Transformer's state at t and token t + 1)
    loss = CE of the GRU's predictions of the L future tokens
         + CE of the Transformer's own next-token predictions
         + weight x ( entropy of the network's next-state probabilities
                      + CE of those probabilities against the state the Transformer actually picks at t + 1 )

Both next-state terms can change the network, the Transformer's state at t and its state at t + 1 (model A holds the
state at t + 1 fixed).  The idea: if the Transformer's states follow a fixed rule (state + token -> next state), the
network can learn it and then step through the future like the GRU; the entropy term asks for one definite next state.
Known risk (Tang et al. 2023; Chen & He 2021): when the target state also receives the gradient, a constant state is
the cheapest way to make the prediction easy.

Predictors: transformer, gru_from_handoff, next_state_from_handoff (the network stepping from the Transformer's state
after the history).  Default weight 0.001, chosen by section 16's 10,000-update trial runs.
"""'''
DOCS["4"] = '''"""
Model 4: model 3 without the GRU.  (Section 16 `next_state_net_no_gru`.)  Can the next-state network replace the GRU?

The parts: the Transformer, the shared state table and output layer (causal_state_matrix.py), and the next-state
network of model 3 (two layers, hidden width 64: (state vector, next token) -> scores of the K next states).

The loss, in bits (positions t >= context)
    loss = CE of the Transformer's own next-token predictions
         + weight x ( entropy of the network's next-state probabilities
                      + CE of those probabilities against the state the Transformer actually picks at t + 1 )

Predictors: transformer (the one the loss trains on the future tokens), next_state_from_handoff.
Default weight 0.001 (section 16's trial runs).  Note: section 16 built this model with a GRU and then removed it, so
with the same seed its starting weights differ from section 16's runs; the trained model is otherwise the same.
"""'''
DOCS["A"] = '''"""
Model A: model 3 with the Transformer's actual next state as a FIXED target.  (Section 16 `next_state_fixed_target`.)

The parts and the loss are model 3's (model_3.py), except that the state the Transformer picks at t + 1 is detached
in the next-state CE: no gradient of the next-state terms reaches the Transformer through its next state.  The network
learns to follow the Transformer, and the Transformer is not pulled toward states that are easy for the network to
predict.  Holding the prediction target fixed is the known remedy against the collapse to a constant state in models
that predict their own next internal state (Tang et al. 2023; Chen & He 2021).

Predictors: transformer, gru_from_handoff, next_state_from_handoff.  Default weight 0.01 (section 16's trial runs).
"""'''
DOCS["5a"] = '''"""
Model 5a: model 1 + the Transformer's one- and two-step predictions + a penalty on the spread of the next state worked
out from those predictions.  (Section 16 `two_step`.)

The parts are model 2's plus a two-step head (one linear layer, state vector -> V * V scores: P(x_{t+1}, x_{t+2} | s_t))
and a running count pi of how often the Transformer picks each state.

The loss, in bits (positions t >= context)
    loss = CE of the GRU's predictions of the L future tokens
         + CE of the Transformer's one-step predictions (output layer)  + CE of its two-step predictions (two-step head)
         + weight x entropy of r, the inferred distribution of the next state:
             1. q(x)     = P(x_{t+1}, x | s_t) / P(x_{t+1} | s_t), rescaled to sum to 1   (what the next state should
                           predict for the token after the real x_{t+1})
             2. P(k | x) = P(x | k) pi_k / sum_j P(x | j) pi_j                          (Bayes: which state predicts x)
             3. r(k)     = sum_x q(x) P(k | x)
The idea: in a causal state the next token fixes the next state, so the predictions themselves can say which state
comes next.  Known limit (section 16.2 of the notebook): even the true 15-state renewal machine scores 3.66 bits on
this entropy, a single state scores 0, and no number of predicted tokens brings the true machine below its crypticity
(3.2363 bits for renewal-15; Mahoney, Ellison & Crutchfield 2009), so the penalty favours fewer states.

Predictors: transformer, gru_from_handoff.  Default weight 0.01 (section 16's trial runs).
"""'''
DOCS["5b"] = '''"""
Model 5b: model 5a without the GRU.  (Section 16 `two_step_no_gru`.)

The parts: the Transformer, the shared state table and output layer, the two-step head and the running count pi.

The loss, in bits: model 5a's without the GRU's CE,
    loss = CE of the Transformer's one-step predictions + CE of its two-step predictions
         + weight x entropy of the inferred next state (model_5a.py gives the three steps of the calculation)

Predictors: transformer (the one the loss trains).  Default weight 0.1 (section 16's trial runs).  Note: section 16
built this model with a GRU and then removed it, so with the same seed its starting weights differ from section 16's
runs; the trained model is otherwise the same.
"""'''
DOCS["C"] = '''"""
Model C: model 5b with the next state worked out from the next TWO predicted tokens.  (Section 16 `three_step_no_gru`.)

The parts: model 5b's plus a three-step head (one linear layer, state vector -> V^3 scores:
P(x_{t+1}, x_{t+2}, x_{t+3} | s_t)).

The loss, in bits (positions t >= context)
    loss = CE of the Transformer's one-, two- and three-step predictions
         + weight x entropy of r, the inferred next state:
             1. q(w)     = P(x_{t+1}, w | s_t) / P(x_{t+1} | s_t), rescaled: the predicted PAIR of tokens w after the
                           real x_{t+1} (three-step head over output layer)
             2. P(k | w) = P(w | k) pi_k / sum_j P(w | j) pi_j (Bayes, with the two-step head describing each state k)
             3. r(k)     = sum_w q(w) P(k | w)
How it differs from 5b: the three-step head and its CE are in the loss at any weight; each candidate state is described
by the two-step head instead of the output layer, so the entropy term never changes the output layer; the default weight
is 0.01 (5b: 0.1).  The rescaling in step 1 cancels the division, which only feeds the logged ratio_correction.

Predictors: transformer (the one the loss trains).  Default weight 0.01 (section 16's trial runs).  Note: section 16
built this model with a GRU and then removed it, so with the same seed its starting weights differ from section 16's.
"""'''
DOCS["B"] = '''"""
Model B: model 1 + the Transformer's starting state for the GRU trained to copy the GRU's own state.  (Section 16
`copy_gru_handoff`.)

The parts are model 2's.  The copy target, for each window: put the GRU in row 0 of the table and let it read the H
history tokens with its own step rule, taking the highest-scoring state at every step (no noise, no gradient); the state
it ends in is the target.  A GRU that has learned the process ends in the right state wherever it starts, because the
history determines the state, and that state is in the GRU's own numbering of the states.

The loss, in bits
    loss = CE of the GRU's predictions of the L future tokens
         + weight x CE of the Transformer's scores at the last history position against the copy target

The copy term trains only the Transformer (encoder and scoring layer); the GRU learns only from its future-token CE.
The target moves while the GRU learns, and before the GRU has learned anything it carries no information about the
state (section 16 of the notebook).  Related: distillation (Hinton, Vinyals & Dean 2015).

Predictors: transformer (not trained at positions other than the last history position), gru_from_handoff.
Default weight 3 (section 16's trial runs).
"""'''
DOCS["1_L2"] = '''"""
Model 1 + L2: model 1 + weight x D, the distance between the Transformer's and the GRU's state probabilities after each
future token.  (The notebook's train_step, sections 7, 13 and 15 of hybrid_model_test.ipynb.)

The parts are model 2's.  D after future token x_{H+k}: the Transformer reads x_0..x_{H+k} and gives probabilities over
the K states; the GRU, having stepped from the Transformer's state after the history through x_H..x_{H+k}, gives its own;
D is the Euclidean distance between the two vectors, averaged over windows and the L future tokens (the last GRU step,
after x_{H+L-1}, included).  During training both vectors include the random noise of the state choice.

The loss, in bits
    loss = CE of the GRU's predictions of the L future tokens + weight x D

What was found (sections 13 and 15): the cheapest way to make D small is for both parts to use very few states; on
renewal-15 with H = 300, L = 40 and weight 0.03, both parts ended on the same 2 states, which track the last token only.
Section 13's own run used training.ORIGINAL_SETUP, H = 300, L = 40, 50,000 updates and weight 0.03.

Predictors: transformer (not trained), gru_from_handoff.  Default weight 0.03.
"""'''
DOCS["2_L2"] = '''"""
Model 2 + L2: model 2 + weight x D over the L - 1 GRU steps that the GRU's loss already makes.  (Section 17's
`l2_both_losses`.)

The parts are model 2's.  D compares the GRU's state probabilities after x_H .. x_{H+L-2} with the Transformer's at the
same positions (Euclidean distance, averaged); during training both include the random noise of the state choice.  At
weight 0 D is only measured (no gradient), so weight 0 is exactly model 2.

The loss, in bits
    loss = CE of the GRU's predictions of the L future tokens + CE of the Transformer's own next-token predictions
         + weight x D

Predictors: transformer, gru_from_handoff.  Default weight 0.03 (section 17 tried 0.01, 0.03 and 0.1).
"""'''

HEADER = '''
import math

import torch
from torch import nn
from torch.nn import functional as F

from causal_state_matrix import CausalStateMatrix, StateSelection
from training import {imports}

NAME = "{name}"
DESCRIPTION = "{description}"
{origin}
DEFAULTS = {defaults}
MAIN_PREDICTOR = "{main}"                     # the predictor the loss trains on the future tokens (best.pt uses it)
{extra_constants}'''


def write(key):
    spec = SPECS[key]
    imports = ["prediction_loss"]
    if spec["next"] or spec["ahead"] >= 2:
        imports.append("entropy_bits")
    if spec["next"] or spec["ahead"] >= 2 or spec["copy"]:
        imports.append("LN2")
    if spec["ahead"] >= 2:
        imports.append("FREQ_DECAY")
    origin = (f'SECTION16_NAME = "{spec["s16"]}"            # the name of this model\'s saved runs in section 16'
              if spec["s16"] else 'SECTION16_NAME = None')
    defaults = "{}" if spec["weight"] is None else '{"weight": ' + repr(spec["weight"]) + '}'
    extra = f'AHEAD = {spec["ahead"]}                                    # tokens ahead the Transformer predicts\n' \
        if spec["ahead"] >= 2 else ""
    doc = DOCS[key]
    if doc is None:
        doc = '"""' + (OUT / "model_2.py").read_text().split('"""')[1] + '"""'
    text = doc + HEADER.format(imports=", ".join(imports), name=key, description=spec["description"], origin=origin,
                               defaults=defaults, main=spec["main"], extra_constants=extra)
    text += TRANSFORMER_BLOCK + model_class(spec)
    (OUT / f"model_{key}.py").write_text(text)
    print("wrote", f"model_{key}.py", len(text.splitlines()), "lines")


for key in SPECS:
    write(key)
