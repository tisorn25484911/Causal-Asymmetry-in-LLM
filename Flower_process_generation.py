import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FlowerDataset(Dataset):
    def __init__(self, seqs, seq_len=20):
        self.seqs = seqs
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype=torch.long)
        input = x[:-1]
        target = x[1:]
        return input, target


def flower_process_generation(num_samples=1000, seq_len=20, pre_depth=10, n=4, m=2, dice_probs=None):
    """
    Generate sequences from the n-m flower process.
    
    The process alternates between:
    1. Randomly selecting a die i ∈ {0, ..., n-1}, recording x_t = i
    2. Rolling that die to get outcome j ∈ {0, ..., m-1}, recording x_{t+1} = j + n
    
    Total vocabulary size: n + m tokens
    - Tokens 0 to n-1: die selection (which die was chosen)
    - Tokens n to n+m-1: die outcome (result of rolling the selected die)
    
    Args:
        num_samples: number of sequences to generate
        seq_len: number of complete cycles (each cycle = select die + roll die = 2 observations)
        pre_depth: number of initial cycles to discard (for equilibrium, though less relevant here)
        n: number of dice
        m: number of sides on each die
        dice_probs: n x m array where dice_probs[i,j] = P(die i lands on side j)
                   If None, random biased dice are generated
    
    Returns:
        data: list of sequences (each sequence has length 2*(seq_len + pre_depth))
        states: list of (die_choice, outcome) tuples for each timestep
    """
    data = []
    states = []
    
    # Generate random biased dice if not provided
    if dice_probs is None:
        # Each die gets a random bias using Dirichlet distribution
        dice_probs = np.random.dirichlet(np.ones(m), size=n)
    
    T = seq_len + pre_depth  # Total number of cycles
    
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
        
        # Discard pre_depth cycles (2*pre_depth observations)
        seq = seq[2*pre_depth:]
        state_seq = state_seq[2*pre_depth:]
        
        data.append(seq)
        states.append(state_seq)
    
    return data, states


def Rev_flower_process_generation(num_samples=1000, seq_len=20, pre_depth=10, n=4, m=2, dice_probs=None):
    """
    Generate reversed flower process sequences.
    """
    data, states = flower_process_generation(num_samples, seq_len, pre_depth, n, m, dice_probs)
    rev_data = [list(reversed(seq)) for seq in data]
    rev_states = [list(reversed(state_seq)) for state_seq in states]
    return rev_data, rev_states


def make_flower_loader(n=4, m=2, dice_probs=None, batch_size=32, seq_len=100, 
                       num_samples=2000, shuffle=True, mode="forward"):
    """
    Create a DataLoader for the flower process.
    
    Args:
        n: number of dice
        m: number of sides per die
        dice_probs: optional n x m array of die biases
        batch_size: batch size
        seq_len: number of complete cycles (total observations = 2*seq_len)
        num_samples: number of sequences
        shuffle: whether to shuffle data
        mode: "forward" or "backward" (reversed sequences)
    
    Returns:
        DataLoader object
    """
    if mode == "backward":
        seqs, _ = Rev_flower_process_generation(
            num_samples=num_samples, 
            seq_len=seq_len, 
            n=n, 
            m=m, 
            dice_probs=dice_probs
        )
    else:
        seqs, _ = flower_process_generation(
            num_samples=num_samples, 
            seq_len=seq_len, 
            n=n, 
            m=m, 
            dice_probs=dice_probs
        )
    
    ds = FlowerDataset(seqs, seq_len=2*seq_len)  # 2*seq_len observations total
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl


# Example usage
if __name__ == "__main__":
    # Generate some example sequences
    n, m = 4, 2  # 4 dice, 2 sides each
    
    # Create specific biased dice (optional)
    # Example: first die favors side 0, second die is fair, etc.
    dice_probs = np.array([
        [0.7, 0.3],  # die 0: 70% chance of side 0
        [0.5, 0.5],  # die 1: fair
        [0.3, 0.7],  # die 2: 70% chance of side 1
        [0.6, 0.4],  # die 3: 60% chance of side 0
    ])
    
    # Generate sequences
    data, states = flower_process_generation(
        num_samples=5, 
        seq_len=10, 
        n=n, 
        m=m, 
        dice_probs=dice_probs
    )
    
    print("Example sequences (observations):")
    print(f"Vocabulary: 0-{n-1} = die selection, {n}-{n+m-1} = die outcomes")
    for i, seq in enumerate(data[:3]):
        print(f"Sequence {i}: {seq[:20]}...")  # First 20 observations
    
    # Create a DataLoader
    loader = make_flower_loader(
        n=n, 
        m=m, 
        dice_probs=dice_probs,
        batch_size=32,
        seq_len=50,
        num_samples=1000
    )
    
    print(f"\nDataLoader created with {len(loader.dataset)} sequences")
    print(f"Token vocabulary size: {n + m} (tokens 0 to {n+m-1})")
