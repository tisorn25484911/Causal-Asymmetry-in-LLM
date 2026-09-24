import math
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from Transformer import OneHotDecoder, _Decoder


class _learn_causal_state:
    def __init__(self, cfg, tau):
        d_model, n_states, state_dim, token_size = cfg["d_model"], cfg["n_states"], cfg["state_dim"], cfg["token_size"]
        self.tau = tau # the schedue is called from tau_schedue.py

        self.val_vec_to_state = nn.Linear(d_model, n_states)
        self.state_matrix = nn.Parameter(
                    torch.randn(n_states, state_dim) / (state_dim ** 0.5))
        self.emission = nn.Linear(state_dim, token_size)
        self.save_hyperparameters()

        self.last_state_probs  = None    # (B,T,K)
        self.last_state_onehot = None    # (B,T,K)
        self.last_states       = None    # (B,T)
        self.last_causal_reps  = None    # (B,T,S)

    def forward(self, value_vec):  # the function recieves the value vector from the transformer/ GRU
                                        # the function outputs one of the causal representation
        state_logits = self.state_head(value_vec)
        probs = F.softmax(state_logits/self.tau, dim = -1)
        states = probs.argmax(dim = -1)
        hard = F.onehot(states, num_classes = self.n_states).float()

        state_onehot = (hard - probs).detach() + probs
        causal_rep = state_onehot @ self.state_matrix

        committed_state = self.emission(causal_rep)
        return committed_state # post-softmax state
