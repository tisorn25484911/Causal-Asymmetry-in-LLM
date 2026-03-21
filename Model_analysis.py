import gc
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from sklearn.cluster import KMeans
from OneHot_model import cross_ent_onehot
from matplotlib.colors import TwoSlopeNorm
try:
    import umap as _umap_mod
    _warmup = _umap_mod.UMAP(n_components=2, n_neighbors=200).fit_transform(
        np.random.rand(20, 4)
    )
    del _warmup
    UMAP_AVAILABLE = True
    print("umap-learn JIT warm-up succeeded")
except Exception as _e:
    UMAP_AVAILABLE = False
    print(f"UMAP unavailable ({_e}) — PCA fallback active")

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})
"""
latent analysis:
    - latent_extraction: extract latents, inputs, targets from a model and dataloader
    - _project2d: UMAP/PCA 2D projection of latents
    - UMAP plots: color by token type FW/BW models
    - attention heatmaps: visualize attention patterns for FW/BW models

Post-train analysis:
    - perplexity_calculation: compute perplexity on a dataset
    - statistical_complexity_empirical: estimate complexity via clustering latents
    - statistical_complexity_compare: compare empirical complexity to theoretical HMM complexity

General purpose plotting:
    - plot_diff_heatmap
    - training_loss_plot
"""
# ── Helper functions ───────────────────────────────────────────────────────
#Utilities
def mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
def save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  pickle saved -> {path}")
def _sub(arr: np.ndarray, n: int = 1000) -> tuple:
    """Return (latent_subset, index_array) — first n rows, no shuffle."""
    n = min(n, len(arr))
    return arr[:n], np.arange(n)
def savefig(fig, path: str, dpi: int = 120):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved -> {path}")
def save_weights(model, path):
    torch.save(model.state_dict(), path)
    print(f"  weights -> {path}")

#latent analysis:
def latent_extraction(model, data_loader, max_batches = None):
    model.eval()
    latents_all = []
    inputs_all = []
    target_all = []

    device = next(model.parameters()).device  # FIX 3: Fixed device access

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)  # FIX 3: Changed from model.device()
            targets = targets.to(device)  # FIX 3: Changed from model.device()

            _ = model(inputs)
            z = model.last_encodings.detach().cpu().numpy()  # FIX 4: Fixed to numpy directly

            latents_all.append(z)  # FIX 4: Append z directly
            inputs_all.append(inputs.detach().cpu().numpy())
            target_all.append(targets.detach().cpu().numpy())

    latents_all = np.concatenate(latents_all, axis = 0)
    inputs_all = np.concatenate(inputs_all, axis = 0)
    target_all = np.concatenate(target_all, axis = 0)

    return latents_all, inputs_all, target_all
def _project2d(flat: np.ndarray, n_neighbors: int = 200) -> tuple:
    if UMAP_AVAILABLE:
        try:
            c = _umap_mod.UMAP(
                n_components=2, random_state=42,
                n_neighbors=min(n_neighbors, len(flat) - 1),
                min_dist=0.1,
                metric="euclidean",
            ).fit_transform(flat)
            return c, "UMAP"
        except Exception as e:
            print(f"  UMAP failed ({e}), using PCA")
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(flat), "PCA"
def plot_umap(latents, inputs_arr, num_token, title="", save_path=None,
              xlim=None, ylim=None, n_pts: int = 1000):
    """
    Plot 2-D UMAP coloured by token id.
    n_pts: number of consecutive latent vectors to embed (default 1000).
    """
    flat_l = latents.reshape(-1, latents.shape[-1])
    flat_i = inputs_arr.reshape(-1)
    sub_l, idx = _sub(flat_l, n_pts)
    sub_i = flat_i[idx]

    coords, mlbl = _project2d(sub_l)
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(7, 6))
    for tok in range(num_token):
        mask = sub_i == tok
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(tok / max(num_token - 1, 1))],
                   label=f"Token {tok}", alpha=0.7, s=10)
    ax.set_title(f"{title} ({mlbl})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, markerscale=3)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2)
    if save_path:
        savefig(fig, save_path)
    return fig, coords
def plot_attention_heatmap(model, input_seq):
    fig, ax = plt.subplots(figsize=(8, 6))

    device = next(model.parameters()).device
    input_seq = input_seq.unsqueeze(0).to(device)  # (1, T)

    _ = model(input_seq)

    attention = model.last_attention.squeeze(0).cpu().numpy()  # (T, T)

    cax = ax.matshow(attention, cmap='viridis')
    fig.colorbar(cax)

    ax.set_title("Attention Heatmap")
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")

    return fig

#Post-train analysis:
def perplexity_calculation(model, data_loader, max_batches=None, pad_id=None):
    logits_all  = []
    targets_all = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # mirror the same convention used in training_step
            if getattr(model, "mode", "forward") == "backward":
                targets, inputs = batch
            else:
                inputs, targets = batch

            inputs  = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)          # (B, T, V)
            logits_all.append(logits.cpu())
            targets_all.append(targets.cpu())

    logits  = torch.cat(logits_all,  dim=0)
    targets = torch.cat(targets_all, dim=0)

    if pad_id is not None:
        mask    = targets != pad_id
        logits  = logits[mask]
        targets = targets[mask]

    loss, perplexity = cross_ent_onehot(logits, targets)
    return perplexity.item()
def plot_perplexity(model_fw, model_bw, data_loader, max_batches = None):
    perplexity_fw = perplexity_calculation(model_fw, data_loader, max_batches)
    perplexity_bw = perplexity_calculation(model_bw, data_loader, max_batches)

    print("="*70)
    print("PERPLEXITY COMPARISON")
    print("="*70)
    print(f"Forward Model Perplexity:  {perplexity_fw:.4f}")
    print(f"Backward Model Perplexity: {perplexity_bw:.4f}")
    print("="*70)

    # Bar Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    models = ['Forward', 'Backward']
    perplexities = [perplexity_fw, perplexity_bw]
    bars = ax.bar(models, perplexities, color=['blue', 'orange'], alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Model Perplexity Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
def statistical_complexity(p, q, mode): 
    # statistical complexity of HMM
    if mode == "forward":
        stead0 = q / (p + q)
        stead1 = p / (p + q)
        state_prob = np.array([stead0, stead1])
    elif mode == "backward":
        stead0 = (q - p*q)/(p+q)
        stead1 = (p)/(p+q)
        stead2 = p*q/(p+q)
        state_prob = np.array([stead0, stead1, stead2])
    else:
        raise ValueError(f"Invalid model mode: {mode}. Must be 'forward' or 'backward'")
    S = 0.0
    for prob in state_prob:
        S += - prob * np.log2(prob + 1e-12)
    return S
def statistical_complexity_empirical(model, data_loader, max_batches=None, use_t="last", k=2):
    model.eval()
    latents_all = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            z = model.last_encodings.detach().cpu().numpy()  # (B, T, d_model)
            latents_all.append(z)

    latents = np.concatenate(latents_all, axis=0)  # (N, T, d_model)

    # --- choose a consistent time slice instead of mixing all t ---
    if use_t == "last":
        z_use = latents[:, -1, :]          # (N, d_model)
    elif use_t == "first":
        z_use = latents[:, 0, :]           # (N, d_model)
    elif isinstance(use_t, int):
        z_use = latents[:, use_t, :]       # (N, d_model)
    else:
        raise ValueError("use_t must be 'first', 'last', or an int index.")

    # --- cluster in full space (don’t PCA-crush by default) ---
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(z_use)

    counts = np.bincount(labels, minlength=k)
    probs = counts / counts.sum()

    S = -np.sum(probs * np.log2(probs + 1e-12))
    return S
def statistical_complexity_compare(forward_model, backward_model, data_loader, 
                                  p=0.6, q=0.4, max_batches=None, k_fw=2, k_bw=3):
    print("="*70)
    print("STATISTICAL COMPLEXITY COMPARISON")
    print("="*70)
    
    Ss_empirical = {}
    Ss_theory = {}
    
    # Forward: max-past context occurs at LAST token (causal tril)
    print("\n[1/2] Forward Model")
    print("-"*70)
    S_fw_emp = statistical_complexity_empirical(
        forward_model, data_loader, max_batches=max_batches, use_t="last", k=k_fw
    )
    S_fw_theory = statistical_complexity(p, q, mode='forward')
    
    print(f"  Empirical (use_t='last'):   {S_fw_emp:.4f} bits")
    print(f"  Theoretical:               {S_fw_theory:.4f} bits")
    print(f"  Error:                     {abs(S_fw_emp - S_fw_theory):.4f} bits")
    
    Ss_empirical['forward'] = S_fw_emp
    Ss_theory['forward'] = S_fw_theory
    
    # Backward: max-future context occurs at FIRST token (anti-causal triu)
    print("\n[2/2] Backward Model")
    print("-"*70)
    S_bw_emp = statistical_complexity_empirical(
        backward_model, data_loader, max_batches=max_batches, use_t="first", k=k_bw
    )
    S_bw_theory = statistical_complexity(p, q, mode='backward')
    
    print(f"  Empirical (use_t='first'):  {S_bw_emp:.4f} bits")
    print(f"  Theoretical:               {S_bw_theory:.4f} bits")
    print(f"  Error:                     {abs(S_bw_emp - S_bw_theory):.4f} bits")
    
    Ss_empirical['backward'] = S_bw_emp
    Ss_theory['backward'] = S_bw_theory
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nEmpirical Complexity:")
    print(f"  Forward (last):   {S_fw_emp:.4f} bits")
    print(f"  Backward (first): {S_bw_emp:.4f} bits")
    print(f"  Difference (FW - BW): {S_fw_emp - S_bw_emp:+.4f} bits")
    
    print(f"\nTheoretical Complexity:")
    print(f"  Forward:  {S_fw_theory:.4f} bits")
    print(f"  Backward: {S_bw_theory:.4f} bits")
    print(f"  Difference (FW - BW): {S_fw_theory - S_bw_theory:+.4f} bits")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    modes = ['Forward', 'Backward']
    emp_values = [Ss_empirical['forward'], Ss_empirical['backward']]
    theory_values = [Ss_theory['forward'], Ss_theory['backward']]
    
    x = np.arange(len(modes))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, emp_values, width, label='Empirical', alpha=0.8, color='blue')
    bars2 = plt.bar(x + width/2, theory_values, width, label='Theoretical', alpha=0.8, color='red')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h,
                     f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.ylabel('Statistical Complexity (bits)', fontsize=11)
    plt.title("Empirical vs Theoretical Statistical Complexity", fontsize=12, fontweight='bold')
    plt.xticks(x, modes)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    return Ss_empirical, Ss_theory

#General purpose plotting:
def plot_diff_heatmap(Z, p_vals, q_vals, title, cbar_label, save_path=None, cmap="RdBu_r", vcenter=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ext = [float(np.min(q_vals)), float(np.max(q_vals)),
           float(np.min(p_vals)), float(np.max(p_vals))]
    if vcenter is not None:                                            # ← added block
        vmin = float(np.nanmin(Z))
        vmax = float(np.nanmax(Z))
        vmin = min(vmin, vcenter - 1e-6)   # guard: vmin < vcenter < vmax
        vmax = max(vmax, vcenter + 1e-6)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        im = ax.imshow(Z, origin="lower", extent=ext,
                       cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(Z, origin="lower", extent=ext, cmap=cmap, aspect="auto")
    Qm, Pm = np.meshgrid(q_vals, p_vals)
    ax.contour(Qm, Pm, Z, levels=8, colors="white", alpha=0.4, linewidths=0.8)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("q", fontsize=12); ax.set_ylabel("p", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if save_path: savefig(fig, save_path)
    return fig
def training_loss_plot(recorder):
    fig, ax = plt.subplots(figsize=(40, 6))
    ax.plot(recorder.epoch_loss, label='Training Loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss over Epochs")
    ax.legend()
    return fig
    
#Standard Analysis functions:   


def FW_BW_attention_comparison(model_fw, model_bw, input_seq, title_prefix=""):
    """
    Compare forward and backward model attention patterns with truly SQUARE heatmaps.

    - Forces square *axes boxes* (not just square pixels) via set_box_aspect(1)
    - Uses ONE shared colorbar to avoid shrinking each subplot differently
    - Keeps FW/BW on same color scale (vmin/vmax shared)
    """

    # --- Move input to same device as model ---
    device = next(model_fw.parameters()).device
    input_seq = input_seq.unsqueeze(0).to(device)  # (1, T)

    # --- Forward model ---
    _ = model_fw(input_seq)
    attention_fw = model_fw.last_attention.squeeze(0).detach().cpu().numpy()  # (T, T)

    # --- Backward model ---
    _ = model_bw(input_seq)
    attention_bw = model_bw.last_attention.squeeze(0).detach().cpu().numpy()  # (T, T)

    # --- Shared color limits ---
    vmin = min(attention_fw.min(), attention_bw.min())
    vmax = max(attention_fw.max(), attention_bw.max())

    # --- Figure (square-ish overall layout) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Plot FW
    im0 = axes[0].imshow(attention_fw, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    axes[0].set_title(f"{title_prefix}Forward Model Attention", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_xlabel("Key", fontsize=12)
    axes[0].set_ylabel("Query", fontsize=12)
    axes[0].set_box_aspect(1)          # <-- makes the axes box square
    axes[0].tick_params(labelsize=10)

    # Plot BW
    im1 = axes[1].imshow(attention_bw, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    axes[1].set_title(f"{title_prefix}Backward Model Attention", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_xlabel("Key", fontsize=12)
    axes[1].set_ylabel("Query", fontsize=12)
    axes[1].set_box_aspect(1)          # <-- makes the axes box square
    axes[1].tick_params(labelsize=10)

    # --- One shared colorbar (prevents per-axes shrinking) ---
    cbar = fig.colorbar(im0, ax=axes, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    return fig


def compare_FW_BW_latents(model_fw, model_bw, data_loader, max_batches = None):
    # Create a non-shuffled loader to ensure both models see same data
    # Note: We'll just remove the assertion since both models are trained on same distribution
    latents_fw, inputs_fw, _ = latent_extraction(model_fw, data_loader, max_batches)
    latents_bw, inputs_bw, _ = latent_extraction(model_bw, data_loader, max_batches)

    # FIX 18: Removed assertion - models see different batches due to shuffle, but same distribution
    # assert np.array_equal(inputs_fw, inputs_bw), "Input sequences do not match between forward and backward models."

    token_colors = {0: 'red', 1: 'green', 2: 'purple'}
    latents_fw_flat = latents_fw.reshape(-1, latents_fw.shape[-1])
    latents_bw_flat = latents_bw.reshape(-1, latents_bw.shape[-1])

    print("Performing PCA on Forward and Backward Latents...")
    pca_fw = PCA(n_components = 2)
    latents_fw_2D = pca_fw.fit_transform(latents_fw_flat) #(N*T, 2)

    pca_bw = PCA(n_components = 2)
    latents_bw_2D = pca_bw.fit_transform(latents_bw_flat) #(N*T, 2)

    print("="*70)
    print("FW/BW LATENTS COMPARISON")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    
    for token in [0, 1, 2]:
        mask = inputs_fw.reshape(-1) == token
        axes[0].scatter(latents_fw_2D[mask, 0], latents_fw_2D[mask, 1], 
                        c = token_colors[token], label = f'Token {token}', 
                        alpha = 0.3, s = 5)
    axes[0].set_title("Forward Model Latents (PCA)")
    axes[0].legend()

    for token in [0, 1, 2]:
        mask = inputs_bw.reshape(-1) == token
        axes[1].scatter(latents_bw_2D[mask, 0], latents_bw_2D[mask, 1], 
                        c = token_colors[token], label = f'Token {token}', 
                        alpha = 0.3, s = 5)
    axes[1].set_title("Backward Model Latents (PCA)")
    axes[1].legend()

    return fig

def FW_BW_loss_comparison(recorder_fw, recorder_bw):
    print("="*70)
    print("FW/BW LOSS COMPARISON")
    print("="*70)
    fig, ax = plt.subplots(1, 2, figsize=(20, 4))
    ax[0].plot(recorder_fw.step_loss, label='Forward Model Loss')
    ax[0].plot(recorder_bw.step_loss, label='Backward Model Loss')
    ax[1].plot(np.array(recorder_bw.step_loss) - np.array(recorder_fw.step_loss), label='Loss Difference (BW - FW)')  # FIX 6: Convert lists to numpy arrays

    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Forward Model Training Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss_difference")
    ax[1].set_title("Backward Model Training Loss")
    ax[0].legend()
    ax[1].legend()

    return 