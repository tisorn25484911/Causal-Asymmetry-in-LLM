"""
The part every model in this folder shares: a table of K learned state vectors, the output layer that turns a state
vector into next-token probabilities, and the code that chooses one state from a list of scores.

    E        the state table, K rows of length S (state_matrix; "the codebook" in the notebook)
    W        the output layer, S -> V token scores (emission)
    select   scores (..., K) -> the chosen state

Each model gives scores to the K states with its own scoring layer (the Transformer's, the GRU's, ...) and then calls
`select`.  Because every part chooses rows of the same E and predicts through the same W, a state means the same
thing to every part of a model.

How a state is chosen (section 4 of hybrid_model_test.ipynb)
    training    random Gumbel noise g is added to the scores a, and the state with the highest a + g is taken: this
                picks state k with probability softmax(a)_k (Jang, Gu & Poole 2017).  The chosen row of E is used
                exactly (a one-hot vector times E).
    gradient    the backward pass goes through the smooth stand-in q = softmax((a + g) / tau) ("straight-through":
                y = stopgrad(onehot - q) + q is the one-hot vector forward and q backward).  The temperature tau only
                shapes q and its gradient; it never changes which state is picked.
    evaluation  no noise; the state with the highest score is taken.

This is the notebook's SharedCausalHead.  The modular models' head (model_modeularised/causal_matrix.py) is the same
computation with the scoring layer kept inside it, its table starting from random values, and a noise clamp of 1e-10;
`start="random"` and `noise_clamp=1e-10` reproduce it.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class StateSelection:
    probs: torch.Tensor           # [..., K] the soft probabilities q = softmax((scores + noise) / tau)
    onehot: torch.Tensor          # [..., K] the chosen state: one-hot in the forward pass, q in the backward pass
    ids: torch.Tensor             # [...]    the chosen state's row number
    representation: torch.Tensor  # [..., S] the chosen row of the state table (onehot @ E)


class CausalStateMatrix(nn.Module):
    """
    n_states K, state_dim S, vocab_size V.

    start="equal"   every row of E starts at 1/sqrt(S) (the hybrid): every state predicts the same at the start
    start="random"  rows start at randn / sqrt(S) (the modular feedback GRU and Transformer)
    """

    def __init__(self, n_states, state_dim, vocab_size, start="equal"):
        super().__init__()
        if start == "equal":
            table = torch.ones(n_states, state_dim) / math.sqrt(state_dim)
        elif start == "random":
            table = torch.randn(n_states, state_dim) / (state_dim ** 0.5)
        else:
            raise ValueError(f"start must be 'equal' or 'random', got {start!r}")
        self.state_matrix = nn.Parameter(table)
        self.emission = nn.Linear(state_dim, vocab_size)

    def select(self, scores, tau, noise, noise_clamp=None):
        """Choose a state from `scores` (..., K).  noise=True adds Gumbel noise (training); the uniform draws are
        clamped to [c, 1 - c], c = noise_clamp (default: the smallest float step of the scores' dtype)."""
        if not math.isfinite(float(tau)) or tau <= 0:
            raise ValueError("temperature must be finite and positive")
        perturbed = scores
        if noise:
            c = torch.finfo(scores.dtype).eps if noise_clamp is None else noise_clamp
            u = torch.rand_like(scores).clamp(min=c, max=1 - c)
            perturbed = scores - torch.log(-torch.log(u))
        probs = (perturbed / tau).softmax(dim=-1)
        ids = probs.argmax(dim=-1)
        hard = F.one_hot(ids, num_classes=scores.shape[-1]).to(scores.dtype)
        onehot = (hard - probs).detach() + probs
        return StateSelection(probs, onehot, ids, onehot @ self.state_matrix)

    def token_logits(self, representation):
        """Token scores (..., V) of state vectors (..., S); softmax gives P(next token | state)."""
        return self.emission(representation)

    @torch.no_grad()
    def token_probs(self):
        """P(next token | state) for every row of the table, (K, V)."""
        return self.emission(self.state_matrix).softmax(-1)
