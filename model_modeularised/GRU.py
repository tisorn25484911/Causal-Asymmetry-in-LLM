"""
The GRU counterpart of Transformer.py: the same two architectures with a GRU in
place of the attention stack, plus one the transformer cannot have.

    OneHotGRU            h_t -> Linear -> logits.  No explicit states.
    DiscreteCausalGRU    h_t -> causal_matrix._learn_causal_state -> logits, so
                         prediction is forced through a K-way bottleneck.  The
                         state is READ OFF the continuous h_t.
    DiscreteFeedbackGRU  the state IS the recurrence: the head sits inside the
                         loop and its state vector is the next step's hidden
                         state, so the machine is unifilar by construction and
                         is read off exactly (`machine()`).

Same constructor names, same (B,T) batches, same loss, same head as the
transformer, so a run swaps architectures by swapping the import:

    from GRU import build_model          # or: from Transformer import build_model

How the encoder reads a batch.  Each row is an independent sequence; nn.GRU
steps through its T tokens in order (sequential in time, parallel over the
batch), starting from h_0 = 0 on every call, so nothing is carried between
sequences or batches.  h[:, t] summarises tokens 0..t and is what the head reads
to predict token t+1 -- the same context a tril-masked transformer has at t.

What it deliberately does not have:
  * a learned embedding.  The one-hot token goes straight in; the GRU's input
    weights already are a learned linear map of it, so an Embedding in front
    would be a second linear map (and at E < V a rank bottleneck on the tokens).
  * a positional encoding.  Position is the order the recurrence reads in.
  * LayerNorm.  h_t is a convex blend of tanh outputs and h_{t-1}, so it stays
    in (-1, 1) by construction.
  * (DiscreteCausalGRU) feedback of the state.  The state is READ OFF h_t and
    never fed back, exactly as in the transformer, so those two differ only in
    the encoder.  DiscreteFeedbackGRU is the other choice: the old notebook
    GRU's transition cell (EmbeddedDiscreteMemoryGRU, Experimental_pipeline/
    models.py), run over every position from a blank start instead of from a
    continuous encoder's summary of a 20-token window.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_matrix import _DiscreteHead
from decoder_base import _Decoder


class _GRUStack(_Decoder):
    """
    The encoder both GRU heads share: one-hot tokens -> nn.GRU -> (B,T,D).

    max_len is accepted and stored so a config (and training.train_model) builds
    either architecture with the same keywords; a GRU runs at any length, so
    nothing here reads it.
    """

    def __init__(self, token_size=3, d_model=32, max_len=300, lr=1e-3,
                 mode="forward", n_layers=1, weight_decay=0.0):
        super().__init__(token_size=token_size, d_model=d_model, max_len=max_len,
                         lr=lr, mode=mode, n_layers=n_layers, weight_decay=weight_decay)
        self.gru = nn.GRU(input_size=token_size, hidden_size=d_model,
                          num_layers=n_layers, batch_first=True)

    def _encode(self, tokens) -> torch.Tensor:
        """
        (B,T) tokens -> (B,T,D) encodings.

        mode="backward" reads right to left: flip, run, flip back, so h[:, t]
        summarises tokens t..T-1 -- the exact mirror of the transformer's triu
        mask.  (The runs here take the backward arm from reversed data instead,
        with mode="forward"; see decoder_base._Decoder._split.)
        """
        tokens = self._tokens(tokens)
        x = F.one_hot(tokens, num_classes=self.token_size).float()
        if self.mode == "backward":
            x = x.flip(1)
        h, _ = self.gru(x)
        if self.mode == "backward":
            h = h.flip(1)
        self.last_encodings = h
        return h


class OneHotGRU(_GRUStack):
    """
    The standard GRU decoder: encodings -> logits, no bottleneck.  No explicit
    causal states; `state_assignment` and `emission_table` return None.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.output_prj = nn.Linear(self.d_model, self.token_size)
        self.save_hyperparameters()

    def forward(self, tokens):
        return self.output_prj(self._encode(tokens))


class DiscreteCausalGRU(_DiscreteHead, _GRUStack):
    """
    Prediction forced through a DISCRETE causal state.
        h          (B,T,D)   GRU hidden state at every position
        logits     (B,T,V)   causal_matrix._learn_causal_state   <- the cross-entropy is here
    """

    def __init__(self, n_states=None, state_dim=None, tau=1.0,
                 usage_beta=0.0, **kw):
        super().__init__(**kw)
        self._init_causal_head(n_states, state_dim, tau, usage_beta)
        self.save_hyperparameters()

    def forward(self, tokens):
        return self.causal_head(self._encode(tokens), self.tau, self._sampling())


class DiscreteFeedbackGRU(_DiscreteHead, _Decoder):
    """
    The discrete state IS the recurrence.  At every position

        h_t      = GRUCell(onehot(x_t), m_{t-1})   proposal from (previous state vector, token)
        s_t      = argmax state_head(h_t)          straight-through: the shared head
        m_t      = state_matrix[s_t]               fed back as the next step's hidden state
        logits_t = emission(m_t)                   predicts x_{t+1}

    m_{-1} = 0 -- no information yet, and not one of the K states -- so s_0 is read
    from x_0 alone.  After that s_t depends on (s_{t-1}, x_t) and nothing else: the
    machine is unifilar by construction, and `machine()` reads it off exactly by
    pushing every (state, token) pair through the cell and the head.

    What it costs: the loop runs in Python over T; and the gradient reaches an
    early step only through one straight-through estimate per later step, so its
    bias compounds along the chain (the old notebook GRU kept that chain to 10
    steps behind a continuous encoder).

    state_dim is d_model here: the state vector is the cell's hidden state.  One
    cell, so n_layers must be 1.
    """

    def __init__(self, token_size=3, d_model=32, max_len=300, lr=1e-3,
                 mode="forward", n_layers=1, weight_decay=0.0,
                 n_states=None, state_dim=None, tau=1.0, usage_beta=0.0):
        super().__init__(token_size=token_size, d_model=d_model, max_len=max_len,
                         lr=lr, mode=mode, n_layers=n_layers, weight_decay=weight_decay)
        if n_layers != 1:
            raise ValueError(f"DiscreteFeedbackGRU has one cell; n_layers must be 1, got {n_layers}")
        if state_dim not in (None, d_model):
            raise ValueError(f"DiscreteFeedbackGRU feeds the state vector back as the hidden "
                             f"state, so state_dim must be d_model ({d_model}), got {state_dim}")
        self.cell = nn.GRUCell(input_size=token_size, hidden_size=d_model)
        self._init_causal_head(n_states, d_model, tau, usage_beta)
        self.save_hyperparameters()

    def forward(self, tokens):
        tokens = self._tokens(tokens)
        x = F.one_hot(tokens, num_classes=self.token_size).float()
        if self.mode == "backward":
            x = x.flip(1)
        head, sampling = self.causal_head, self._sampling()
        m = x.new_zeros(x.shape[0], self.d_model)
        steps = {k: [] for k in ("logits", "h", "probs", "onehot", "states", "reps")}
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], m)
            steps["logits"].append(head(h, self.tau, sampling))  # (B, V)
            m = head.last_causal_reps                            # (B, D): fed back
            steps["h"].append(h)
            steps["probs"].append(head.last_state_probs)
            steps["onehot"].append(head.last_state_onehot)
            steps["states"].append(head.last_states)
            steps["reps"].append(m)
        out = {k: torch.stack(v, dim=1) for k, v in steps.items()}
        if self.mode == "backward":
            out = {k: v.flip(1) for k, v in out.items()}
        # The head saw one step at a time; give the analysis surface and the usage
        # penalty the whole sequence.
        head.last_state_probs, head.last_state_onehot = out["probs"], out["onehot"]
        head.last_states, head.last_causal_reps = out["states"], out["reps"]
        self.last_encodings = out["h"]
        return out["logits"]

    @torch.no_grad()
    def machine(self) -> dict:
        """
        The exact machine, by enumeration over all K states and V tokens:
            next_state[s, x]    the state after reading x in state s
            emission_probs[s]   P(next token | s)                 (= emission_table)
            start_state[x]      the state after reading x from the blank start
        In the model's own reading order (mode="backward": right to left).
        """
        K, V = self.n_states, self.token_size
        M = self.state_matrix
        eye = torch.eye(V, device=M.device)
        next_state = torch.empty(K, V, dtype=torch.long)
        for x in range(V):
            h = self.cell(eye[x].expand(K, V), M)
            next_state[:, x] = self.causal_head.state_head(h).argmax(dim=-1).cpu()
        h0 = self.cell(eye, torch.zeros(V, self.d_model, device=M.device))
        start = self.causal_head.state_head(h0).argmax(dim=-1).cpu()
        return {"next_state": next_state.numpy(), "start_state": start.numpy(),
                "emission_probs": self.emission_table().cpu().numpy()}


def build_feedback_model(embed_type: str, **kw):
    """The feedback GRU exists only with the discrete head."""
    if embed_type != "discrete":
        raise ValueError(f"DiscreteFeedbackGRU has a discrete head only; got embed_type {embed_type!r}")
    return DiscreteFeedbackGRU(**kw)


def build_model(embed_type: str, **kw):
    """The ONLY place that decides which GRU is built."""
    if embed_type == "onehot":
        kw.pop("n_states", None); kw.pop("state_dim", None)
        kw.pop("tau", None);      kw.pop("usage_beta", None)
        return OneHotGRU(**kw)
    if embed_type == "discrete":
        return DiscreteCausalGRU(**kw)
    raise ValueError(f"embed_type must be 'onehot' or 'discrete', got {embed_type!r}")
