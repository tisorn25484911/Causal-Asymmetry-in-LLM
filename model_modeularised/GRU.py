import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class EmbeddedDiscreteMemoryGRU(nn.Module):
    # v2 architecture:
    #   encoder_cell:    continuous GRU that processes the past (no discretization).
    #   encoder_head:    projects encoder hidden state to discrete-state logits at boundary.
    #   transition_cell: discrete GRU that processes future symbols (Gumbel-Softmax each step).
    #   transition_head: projects transition proposal to discrete-state logits.
    #   emission_head:   linear projection from state embedding to symbol logits.
    #   state_emb:       learned embedding table for discrete states.
    #
    # The encoder_cell is used ONLY during training (sequence_objective) and
    # window-end-state inference. Machine extraction and rollout use only
    # transition_cell + state_emb.

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.symbol_emb = nn.Embedding(cfg.symbol_vocab_size, cfg.symbol_embed_dim)
        self.state_emb = nn.Embedding(cfg.num_states, cfg.memory_dim)

        # Continuous encoder for past processing (no discretization per step).
        self.encoder_cell = nn.GRUCell(cfg.symbol_embed_dim, cfg.encoder_hidden_dim)
        self.encoder_head = nn.Linear(cfg.encoder_hidden_dim, cfg.num_states)

        # Discrete transition cell for future processing.
        self.transition_cell = nn.GRUCell(cfg.symbol_embed_dim, cfg.memory_dim)
        self.transition_head = nn.Linear(cfg.memory_dim, cfg.num_states)

        # Linear emission head: discrete state -> symbol logits.
        # No hidden layer: forces all predictive information through the discrete bottleneck.
        self.emission_head = nn.Linear(cfg.memory_dim, cfg.symbol_vocab_size)

        self.init_logits = nn.Parameter(torch.zeros(cfg.num_states))

    # --- Encoder (past processing) ---

    def encode_past(self, past, tau, training_mode):
        batch_size = past.shape[0]
        h = torch.zeros(batch_size, self.cfg.encoder_hidden_dim, device=past.device)

        for t in range(past.shape[1]):
            symbol_vec = self.symbol_emb(past[:, t])
            h = self.encoder_cell(symbol_vec, h)

        logits = self.encoder_head(h)
        if training_mode:
            onehot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            idx = logits.argmax(dim=-1)
            onehot = F.one_hot(idx, num_classes=self.cfg.num_states).float()

        idx = onehot.argmax(dim=-1)
        memory = onehot @ self.state_emb.weight
        return {"idx": idx, "onehot": onehot, "memory": memory, "logits": logits}

    # --- Discrete transition (future processing and extraction) ---

    def next_state_from_memory_and_symbol(self, memory, symbols, tau, training_mode):
        symbol_vec = self.symbol_emb(symbols)
        proposal = self.transition_cell(symbol_vec, memory)
        logits = self.transition_head(proposal)
        if training_mode:
            onehot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            idx = logits.argmax(dim=-1)
            onehot = F.one_hot(idx, num_classes=self.cfg.num_states).float()
        idx = onehot.argmax(dim=-1)
        next_memory = onehot @ self.state_emb.weight
        return {"proposal": proposal, "logits": logits, "idx": idx, "onehot": onehot, "memory": next_memory}

    # --- Emission ---

    def emission_logits_from_memory(self, memory):
        return self.emission_head(memory)
