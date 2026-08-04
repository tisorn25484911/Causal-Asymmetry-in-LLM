import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CoinDataset(Dataset):
    def __init__(self, seqs, seq_len = 20):
        self.num_token = 3  # Assuming 3 tokens for coin flip: 0, 1, 2
        self.seqs = seqs
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype = torch.long)
        input = x[:-1]
        target = x[1:]
        return input, target
    
# IMPROVEMENT_PLAN.md C4.  `FlowerDataset` and `flower_process_generation`
# used to be defined HERE as well as in Flower_process_generation.py -- this
# copy with n=4/m=2 hard-coded as attributes and no `pre_depth`, that one
# parametric and with it.  Every runner imported the generator from this file
# and the Dataset from the other, so this file's FlowerDataset was dead in all
# of them.  Both copies are deleted; Flower_process_generation.py is the single
# definition.  Duplicated definitions are fine right up until one is fixed and
# the other is not.


def coin_generation(num_samples = 1000, seq_len = 20, p = 0.6, q = 0.4):
    data = []   #num_sample x seq_len
    states = [] #num_sample x seq_len
    T = seq_len

    for _ in range(num_samples):
        seq = []
        state_seq = []
        if np.random.rand() < p/(p+q):
            cur_state = 1
        else:
            if np.random.rand() < q:
                cur_state, prev_state = 0, 1
            else:
                cur_state, prev_state = 0, 0
        for t in range(T):
            state_seq.append(cur_state)
            if cur_state == 1:
                obs = 1
            elif cur_state == 0 and prev_state == 1:
                obs = 2
            else:
                obs = 0
            seq.append(obs)
            prev_state = cur_state

            if cur_state == 0:
                cur_state = 1 if np.random.rand() < p else 0
            else:
                cur_state = 0 if np.random.rand() < q else 1
        data.append(seq)
        states.append(state_seq)
    return data, states

"""
DataLoader creation function
"""

def Rev_HMM_generation(data, states):
    rev_data = [list(reversed(seq)) for seq in data]
    rev_states = [list(reversed(state_seq)) for state_seq in states]
    return rev_data, rev_states

def make_loader(data, batch_size, shuffle=True):
    """
    Plain DataLoader over `data`.

    IMPROVEMENT_PLAN.md C7.  This used to take a `states` argument that was a
    no-op: the body read `seqs, _ = data, states`, a tuple unpack that assigns
    seqs = data and discards states, written in a form that made `states` look
    load-bearing.  No caller ever used the state sequences.

    It also had a mode="backward" branch that reversed the *data* — a third,
    unused notion of "backward" alongside the triu mask and the batch swap,
    and NOT the mechanism any experiment uses.  Only pq_experiment.py calls
    this, and always with mode="forward", so the branch is removed rather than
    left as a trap.
    """
    seqs = data
    ds = CoinDataset(seqs, seq_len=len(seqs[0]))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
