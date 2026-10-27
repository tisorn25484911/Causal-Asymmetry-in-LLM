"""
Feedback GRU: a GRU alone whose state IS a row of the state table.  (Section 17's `feedback_gru`: the modular
DiscreteFeedbackGRU of model_modeularised/GRU.py, rewritten here self-contained.)

The model
    At every position t the GRU computes a proposal h_t = GRUCell(onehot(x_t), m_{t-1}) from the token x_t and the
    previous state's vector m_{t-1}.  Its scoring layer gives a score to each of the K states from h_t, and one state
    s_t is chosen (random noise added during training, the highest score at evaluation).  The chosen row m_t = E[s_t]
    is fed back as the next step's state vector, and the shared output layer predicts x_{t+1} from it.  Every window
    starts blank (m_{-1} = 0), so s_0 is read from x_0 alone.
    The next state depends only on (state, token), so the model is a finite machine by construction; step_rules() gives
    that rule, and analysis.py reads the machine off exactly by trying every (state, token) pair.

The loss, in bits
    loss = CE of its own next-token predictions at every position t >= context (19): it reads x_0 .. x_{T-2} and the
    output at t predicts x_{t+1}.

Compared with the hybrid's parts: there is no Transformer and so no starting state from one (no handoff); the state
vector is the GRU's hidden state, so state_dim must equal d_model (32); the state table starts from random values and
the noise clamp is 1e-10 (causal_state_matrix.py explains both); no LayerNorm.

Predictor: gru_from_start (its own predictions, reading every token from the start of the window).
"""
import torch
from torch import nn
from torch.nn import functional as F

from causal_state_matrix import CausalStateMatrix
from training import LN2

NAME = "feedback_GRU"
DESCRIPTION = "a GRU alone whose state is a row of the state table (section 17)"
SECTION17_NAME = "feedback_gru"        # the name of this model's saved runs in section 17
DEFAULTS = {"d_model": 32, "n_layers": 1, "state_dim": 32, "encoder_norm": False}
MAIN_PREDICTOR = "gru_from_start"
NOISE_CLAMP = 1e-10                    # the modular head's clamp of the uniform draws behind the noise


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.state_dim != cfg.d_model:
            raise ValueError("the feedback GRU's state vector is its hidden state: state_dim must equal d_model")
        if cfg.n_layers != 1:
            raise ValueError("the feedback GRU has one cell: n_layers must be 1")
        self.cfg = cfg
        K, D, V = cfg.n_states, cfg.d_model, cfg.vocab_size
        # the build order sets the starting random weights; it is GRU.py's order
        self.cell = nn.GRUCell(V, D)
        self.state_head = nn.Linear(D, K)                                    # the scoring layer
        self.causal_head = CausalStateMatrix(K, D, V, start="random")

    def run(self, tokens, tau):
        """Read the windows from a blank start.  Returns token logits (N, T, V) (logits[:, t] predicts x_{t+1}), the
        chosen states (N, T) (the state after reading x_t) and the scores (N, T, K)."""
        x = F.one_hot(tokens, self.cfg.vocab_size).float()
        m = x.new_zeros(x.shape[0], self.cfg.d_model)
        noise = self.training and self.cfg.gumbel
        logits, ids, scores = [], [], []
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], m)
            s = self.state_head(h)
            sel = self.causal_head.select(s, tau, noise, NOISE_CLAMP)
            logits.append(self.causal_head.emission(sel.representation))
            m = sel.representation                                           # fed back as the next state vector
            ids.append(sel.ids)
            scores.append(s)
        return torch.stack(logits, 1), torch.stack(ids, 1), torch.stack(scores, 1)

    # ---- training
    def loss(self, window, tau, weight):
        C, V = self.cfg.context, self.cfg.vocab_size
        logits, ids, _ = self.run(window[:, :-1], tau)
        ce = F.cross_entropy(logits[:, C:].reshape(-1, V), window[:, C + 1:].reshape(-1)) / LN2
        return ce, {"ce": ce}, ids

    # ---- evaluation (no noise)
    def step(self, ids, tokens):
        """The fixed step rule: (state ids, tokens) -> next state ids (highest score)."""
        E = self.causal_head.state_matrix
        return self.state_head(self.cell(F.one_hot(tokens, self.cfg.vocab_size).float(), E[ids])).argmax(-1)

    def step_rules(self):
        return {"gru_from_start": self.step}

    @torch.no_grad()
    def own_states(self, tokens, tau):
        """Token logits, states and scores over whole windows, in batches of 256."""
        out = [self.run(batch, tau) for batch in tokens.split(256)]
        return tuple(torch.cat(part) for part in zip(*out))

    @torch.no_grad()
    def handoff_states(self, history, tau):
        """The model's state after reading the history (the starting states for reading off its machine)."""
        return self.own_states(history, tau)[1][:, -1]

    @torch.no_grad()
    def predict(self, tokens, tau):
        C, H = self.cfg.context, self.cfg.history_len
        T = tokens.shape[1]
        logits, ids, scores = self.own_states(tokens, tau)
        return dict(predictors={"gru_from_start": dict(first=C + 1, logits=logits[:, C:T - 1], states=ids[:, C:T])},
                    handoff_scores=scores[:, H - 1])
