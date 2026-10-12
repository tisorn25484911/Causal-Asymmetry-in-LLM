"""
the hybrid architecture:
for H length history compacted with transformer and L rollout using GRU.

The current testing: 
1. found that the model can't extracts correctly up to 15 renewal states
2. currently testing if a regurisation being the L2 difference between the update rule given
   by GRU and history mapped by transformer would helps: seems not to improve much
Advantage of the model: the model has long window length and would be interesting to test 
an extraction from a natural language, or at least, take some approximation of the natural language
so that one could take the structure and observe a causal asymmetry.

An interesting idea in the architecture is that the model could use the transformer's long length
window to explore the markov order of the natural language and map them to different history
and let the GRU architecture rollouts the simulation to see the transition rule. 
    Here there are some ideas to go by: (hypothesis)
    1. transformer might not be efficient for a discrete task and must allow it to express
       continuously and explore how far one state is different to another. Since the architecture
       updates the value vector by fining some correction from the previous observable representation
       perhaps that continuous correction can be fed into the GRU as another input along with
       the observable to search for the update rule. So the GRU will have the full control of the local update rule
       but the transformer could provide information from distant past.
    2. the idea of a causal state count update as the model learns. the transformer could help take
       in the acount of the markov order and expland the set of causal states with respect to
       the nescessary transition, of which can look up with the De Bruijn graph. 
       The graph would provide the minimal condition where the process stills remains unifilar.
       each of the transition may only transits to the adjacent node and not any further.
       If instead of having a shared causal head, we construct a shared De Bruijn graph.
       the notion of how close each vectors are together in the continuous transformer 
       could indicate adjacent or approximated same node base on if there exist a transition that give them unifilarity. 
       However, one must also consider that in the training, it might not be appearent at first 
       it might be nescessary to train a continuous transformer as a first approximation. Then
       we could add the transformer's representation as an inputs to the rollingout GRU. In the hope 
       that GRU would trace this huge De Bruijn graph out for us and map the same causal state together.
       we could let the GRU starts with a uniformly place states among the continuous transformer space,
       and let it adjusts itself to construct an approximated graph.
       The downside I can think of is that you would have to start with a good approximation of the number of causal states, or enforcing unifilarity would become very hard. So let GRU have a discrete causal state, so it can introduce a new one when needed, but then we would need a criterion for when it needs more. So the hypothesis here is actually that the GRU is limiting the expressivity of the transformer when they are forced to be in a single training.
       
    Future problems and decision forseen:

"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.manual_seed(7)
torch.set_num_threads(2)
DEVICE = torch.device("cpu")  # Small, repeatable teaching run; no accelerator required.
LN2 = math.log(2.0)

@dataclass(frozen=True)
class HybridConfig:
    vocab_size: int = 2
    n_states: int = 16
    state_dim: int = 32 #16
    d_model: int = 16
    n_layers: int = 4
    history_len: int = 320 # can expand to test transformer's capability
    rollout_len: int =40

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def total_len(self):
        return self.history_len + self.rollout_len


def split_window(window, cfg):
    if window.ndim != 2 or window.shape[1] != cfg.total_len:
        raise ValueError(f"expected [batch, {cfg.total_len}], got {tuple(window.shape)}")
    if window.dtype != torch.long:
        raise TypeError("tokens must have dtype torch.long")
    if window.shape[0] == 0 or (window < 0).any() or (window >= cfg.vocab_size).any():
        raise ValueError("empty batch or token outside the vocabulary")
    return window[:, :cfg.history_len], window[:, cfg.history_len:]