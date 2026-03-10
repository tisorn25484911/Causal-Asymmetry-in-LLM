import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CoinDataset(Dataset):
    def __init__(self, seqs, seq_len = 20):
        self.seqs = seqs
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype = torch.long)
        input = x[:-1]
        target = x[1:]
        return input, target
    
def HMM_generation(num_samples = 1000, seq_len = 20, pre_depth = 10, p = 0.6, q = 0.4):
    data = []
    states = []
    T = seq_len + pre_depth

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

def Rev_HMM_generation(num_samples = 1000, seq_len = 20, pre_depth = 10, p = 0.6, q = 0.4):
    data, states = HMM_generation(num_samples, seq_len, pre_depth, p, q)
    rev_data = [list(reversed(seq)) for seq in data]
    rev_states = [list(reversed(state_seq)) for state_seq in states]
    return rev_data, rev_states

def make_loader(pp = 0.6, qq = 0.4, batch_size = 32, seq_len = 100, num_samples = 2000, shuffle = True, mode = "forward"):
    if mode == "backward":
        seqs, _ = Rev_HMM_generation(q = qq, p = pp, num_samples = num_samples, seq_len = seq_len)
    else:
        seqs, _ = HMM_generation(q = qq, p = pp, num_samples = num_samples, seq_len = seq_len)
    ds = CoinDataset(seqs, seq_len = seq_len)
    dl = DataLoader(ds, batch_size = batch_size, shuffle = shuffle)
    return dl