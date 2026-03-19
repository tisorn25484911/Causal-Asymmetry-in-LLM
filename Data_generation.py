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
    
class FlowerDataset(Dataset):
    def __init__(self, seqs, seq_len=20):
        self.n = 4  # Number of dice
        self.m = 2  # Number of outcomes per die
        self.num_token = self.n + self.m  # Total number of tokens
        self.seqs = seqs
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype=torch.long)
        input = x[:-1]
        target = x[1:]
        return input, target

    
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
n_m flower process generation
"""

def flower_process_generation(num_samples=500, seq_len=200, n=4, m=2, dice_probs=None):
    data = []
    states = []
    
    if dice_probs is None:
        # Each die gets a random bias using Dirichlet distribution
        dice_probs = np.random.dirichlet(np.ones(m), size=n)
    
    T = seq_len  # Total number of cycles
    
    for _ in range(num_samples):
        seq = []
        state_seq = []
        
        for t in range(T):
            # Step 1: Randomly select a die (uniform distribution)
            die_idx = np.random.randint(0, n)
            obs_die_choice = die_idx  # Observation in {0, ..., n-1}
            seq.append(obs_die_choice)
            state_seq.append(('select', die_idx))
            
            # Step 2: Roll the selected die
            die_outcome = np.random.choice(m, p=dice_probs[die_idx])
            obs_die_outcome = n + die_outcome  # Observation in {n, ..., n+m-1}
            seq.append(obs_die_outcome)
            state_seq.append(('roll', die_outcome))
        
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

def make_loader(data, states, batch_size, shuffle = True, mode = "forward"):
    if mode == "backward":
        seqs, _ = Rev_HMM_generation(data, states)
        seq_len = len(seqs[0])
    elif mode == "forward":
        seqs, _ = data, states
        seq_len = len(seqs[0])
    else:
        raise ValueError("Invalid mode. Choose 'forward' or 'backward'.")
    ds = CoinDataset(seqs, seq_len = seq_len)
    dl = DataLoader(ds, batch_size = batch_size, shuffle = shuffle)
    return dl
