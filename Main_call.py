import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Training_model import train_model
from Data_generation import make_loader
from OneHot_model import OneHotDecoder
from Model_analysis import (
    statistical_complexity,
    statistical_complexity_empirical,
    statistical_complexity_compare,
    FW_BW_attention_comparison,
    FW_BW_loss_comparison,
    compare_FW_BW_latents,
    plot_attention_heatmap,
    plot_perplexity
)

def main():
    # Hyperparameters
    num_token = 3
    d_model = 20
    max_len = 1000
    batch_size = 32
    num_samples = 5000
    max_epochs = 50
    lr = 1e-2
    p, q = 0.3, 0.4

    # Data Loader
    train_loader = make_loader(pp=p, qq=q, batch_size=batch_size, seq_len=max_len, num_samples=num_samples)

    # Train Forward Model
    print("Training Forward Model...")
    recorder_fw = train_model(
        train_loader, 
        num_token=num_token, 
        d_model=d_model, 
        max_len=max_len, 
        max_epochs=max_epochs, 
        lr=lr, 
        mode='forward'
    )
    
    # Get sample data for visualization (FIX 17: Use actual data, not torch.arange!)
    data_iter = iter(train_loader)
    sample_inputs, _ = next(data_iter)
    sample_seq = sample_inputs[0]  # Get first sequence from batch (contains tokens in [0,1,2])
    
    print("Plotting forward attention heatmap...")
    #plot_attention_heatmap(recorder_fw.model, sample_seq)

    # Train Backward Model
    print("Training Backward Model...")
    recorder_bw = train_model(
        train_loader, 
        num_token=num_token, 
        d_model=d_model, 
        max_len=max_len, 
        max_epochs=max_epochs, 
        lr=lr, 
        mode='backward'
    )
    
    print("Plotting backward attention heatmap...")
    #plot_attention_heatmap(recorder_bw.model, sample_seq)

    # Model Analysis
    print("Analyzing Models...")

    # Loss comparison needs recorders (has .epoch_loss)
    FW_BW_loss_comparison(recorder_fw, recorder_bw)

    # Get sample data
    data_iter = iter(train_loader)
    sample_inputs, _ = next(data_iter)
    sample_seq = sample_inputs[0]

    # All these need models (have .eval(), .forward(), etc.)
    #FW_BW_attention_comparison(recorder_fw.model, recorder_bw.model, sample_seq)
    compare_FW_BW_latents(recorder_fw.model, recorder_bw.model, data_loader=train_loader, max_batches=10)
    statistical_complexity_compare(recorder_fw.model, recorder_bw.model, data_loader=train_loader, p=p, q=q, max_batches=10)
    plot_perplexity(recorder_fw.model, recorder_bw.model, data_loader=train_loader, max_batches=10)  # ← Fixed!

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()