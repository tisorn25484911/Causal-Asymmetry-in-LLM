from Data_generation import HMM_generation
from Data_generation import make_loader
from Training_model import train_model
from Model_analysis import (
    statistical_complexity,
    statistical_complexity_empirical,
)
import numpy as np
import matplotlib.pyplot as plt


def heatmap_theory(
    p_values=None,
    q_values=None,
):
    """
    Compute theoretical statistical complexity on a p-q grid.
    Returns:
        Ss_theory_FW: (len(p), len(q))
        Ss_theory_BW: (len(p), len(q))
        p_values, q_values (so plotting is consistent)
    """
    if p_values is None:
        p_values = np.linspace(0.01, 0.99, 100)  # avoid exact 1.0 edge cases
    if q_values is None:
        q_values = np.linspace(0.01, 0.99, 100)

    p_values = np.array(p_values, dtype=float)
    q_values = np.array(q_values, dtype=float)

    Ss_theory_FW = np.zeros((len(p_values), len(q_values)), dtype=float)
    Ss_theory_BW = np.zeros((len(p_values), len(q_values)), dtype=float)

    print(f"Computing theoretical complexity for {len(p_values)} x {len(q_values)} grid...")

    for i, p in enumerate(p_values):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(p_values)}")
        for j, q in enumerate(q_values):
            Ss_theory_FW[i, j] = statistical_complexity(p, q, mode="forward")
            Ss_theory_BW[i, j] = statistical_complexity(p, q, mode="backward")

    print("✓ Theoretical complexity computation complete")
    return Ss_theory_FW, Ss_theory_BW, p_values, q_values


def pq_experiment(
    num_token=3,
    d_model=20,
    max_len=100,
    batch_size=32,
    num_samples=2000,
    max_epochs=20,
    lr=1e-2,
    p_values=None,
    q_values=None,
    max_batches_for_empirical=10,
):
    """
    Runs empirical experiments on a p-q grid.
    Returns:
        Ss_emp: shape (2, len(p), len(q)) where Ss_emp[0]=FW, Ss_emp[1]=BW
        p_values, q_values (so plotting is consistent)
    """
    if p_values is None:
        p_values = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    if q_values is None:
        q_values = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

    p_values = np.array(p_values, dtype=float)
    q_values = np.array(q_values, dtype=float)

    Ss_emp_FW = np.zeros((len(p_values), len(q_values)), dtype=float)
    Ss_emp_BW = np.zeros((len(p_values), len(q_values)), dtype=float)

    total = len(p_values) * len(q_values)

    for i, p in enumerate(p_values):
        for j, q in enumerate(q_values):
            idx = i * len(q_values) + j + 1
            print(f"\n{'='*70}")
            print(f"Experiment {idx}/{total}: p={p}, q={q}")
            print(f"{'='*70}")

            train_loader = make_loader(
                pp=float(p),
                qq=float(q),
                batch_size=batch_size,
                seq_len=max_len,
                num_samples=num_samples,
            )

            print("\n[1/2] Training Forward Model...")
            recorder_fw = train_model(
                train_loader,
                num_token=num_token,
                d_model=d_model,
                max_len=max_len,
                max_epochs=max_epochs,
                lr=lr,
                mode="forward",
            )

            print("\n[2/2] Training Backward Model...")
            recorder_bw = train_model(
                train_loader,
                num_token=num_token,
                d_model=d_model,
                max_len=max_len,
                max_epochs=max_epochs,
                lr=lr,
                mode="backward",
            )

            print("\n[Analysis] Computing empirical complexity...")

            # Forward: max-past context at last position (causal tril)
            S_emp_fw = statistical_complexity_empirical(
                recorder_fw.model,
                train_loader,
                max_batches=max_batches_for_empirical,
                use_t="last",
                k=2,
            )

            # Backward (anti-causal triu): max-future context at first position
            S_emp_bw = statistical_complexity_empirical(
                recorder_bw.model,
                train_loader,
                max_batches=max_batches_for_empirical,
                use_t="first",
                k=3,
            )

            Ss_emp_FW[i, j] = S_emp_fw
            Ss_emp_BW[i, j] = S_emp_bw

            print(f"  Forward:  {S_emp_fw:.4f} bits")
            print(f"  Backward: {S_emp_bw:.4f} bits")

    Ss_emp = np.stack([Ss_emp_FW, Ss_emp_BW], axis=0)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Forward complexity range:  [{np.min(Ss_emp_FW):.3f}, {np.max(Ss_emp_FW):.3f}] bits")
    print(f"Backward complexity range: [{np.min(Ss_emp_BW):.3f}, {np.max(Ss_emp_BW):.3f}] bits")

    return Ss_emp, p_values, q_values


def plot_heatmap(
    Ss_emp,
    Ss_theory_FW,
    Ss_theory_BW,
    p_emp,
    q_emp,
    p_theory,
    q_theory,
    save_path="complexity_heatmap.png",
):
    """
    Keeps your plotting style (2x2, magma, colorbars, white contours, aspect='auto'),
    but fixes:
      - tick placement/labels (so they correspond to actual p/q)
      - contour X/Y grids matching Z dimensions
      - consistent axis meaning: x=q, y=p
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))

    vmin = min(np.min(Ss_emp), np.min(Ss_theory_FW), np.min(Ss_theory_BW))
    vmax = max(np.max(Ss_emp), np.max(Ss_theory_FW), np.max(Ss_theory_BW))

    # --- helper: apply ticks properly while keeping your style ---
    def _set_ticks(ax, q_vals, p_vals, max_ticks=11):
        # If too many ticks, subsample to avoid clutter while preserving style
        q_vals = np.array(q_vals, dtype=float)
        p_vals = np.array(p_vals, dtype=float)

        if len(q_vals) > max_ticks:
            idx = np.linspace(0, len(q_vals) - 1, max_ticks).round().astype(int)
            q_show = q_vals[idx]
        else:
            q_show = q_vals

        if len(p_vals) > max_ticks:
            idx = np.linspace(0, len(p_vals) - 1, max_ticks).round().astype(int)
            p_show = p_vals[idx]
        else:
            p_show = p_vals

        ax.set_xticks(q_show)
        ax.set_yticks(p_show)

    # --- coordinate grids for contours ---
    Qe, Pe = np.meshgrid(q_emp, p_emp)               # empirical: (len(p_emp), len(q_emp))
    Qt, Pt = np.meshgrid(q_theory, p_theory)         # theory: (len(p_theory), len(q_theory))

    # --- extents so axes are truly q (x) and p (y) ---
    emp_extent = [float(np.min(q_emp)), float(np.max(q_emp)), float(np.min(p_emp)), float(np.max(p_emp))]
    th_extent  = [float(np.min(q_theory)), float(np.max(q_theory)), float(np.min(p_theory)), float(np.max(p_theory))]

    # Plot 1: Empirical Forward
    im1 = axes[0, 0].imshow(
        Ss_emp[0],
        origin="lower",
        extent=emp_extent,
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
        aspect="auto",
    )
    axes[0, 0].set_title("Empirical Statistical Complexity (Forward)", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("q", fontsize=12)
    axes[0, 0].set_ylabel("p", fontsize=12)
    _set_ticks(axes[0, 0], q_emp, p_emp)
    fig.colorbar(im1, ax=axes[0, 0], label="Complexity (bits)")
    axes[0, 0].contour(
        Qe, Pe, Ss_emp[0],
        levels=5,
        colors="white",
        alpha=0.5,
        linewidths=1,
    )

    # Plot 2: Empirical Backward
    im2 = axes[0, 1].imshow(
        Ss_emp[1],
        origin="lower",
        extent=emp_extent,
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
        aspect="auto",
    )
    axes[0, 1].set_title("Empirical Statistical Complexity (Backward)", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("q", fontsize=12)
    axes[0, 1].set_ylabel("p", fontsize=12)
    _set_ticks(axes[0, 1], q_emp, p_emp)
    fig.colorbar(im2, ax=axes[0, 1], label="Complexity (bits)")
    axes[0, 1].contour(
        Qe, Pe, Ss_emp[1],
        levels=5,
        colors="white",
        alpha=0.5,
        linewidths=1,
    )

    # Plot 3: Theoretical Forward
    im3 = axes[1, 0].imshow(
        Ss_theory_FW,
        origin="lower",
        extent=th_extent,
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
        aspect="auto",
    )
    axes[1, 0].set_title("Theoretical Statistical Complexity (Forward)", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("q", fontsize=12)
    axes[1, 0].set_ylabel("p", fontsize=12)
    _set_ticks(axes[1, 0], q_theory, p_theory)
    fig.colorbar(im3, ax=axes[1, 0], label="Complexity (bits)")
    axes[1, 0].contour(
        Qt, Pt, Ss_theory_FW,
        levels=30,
        colors="white",
        alpha=0.3,
        linewidths=0.5,
    )

    # Plot 4: Theoretical Backward
    im4 = axes[1, 1].imshow(
        Ss_theory_BW,
        origin="lower",
        extent=th_extent,
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
        aspect="auto",
    )
    axes[1, 1].set_title("Theoretical Statistical Complexity (Backward)", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("q", fontsize=12)
    axes[1, 1].set_ylabel("p", fontsize=12)
    _set_ticks(axes[1, 1], q_theory, p_theory)
    fig.colorbar(im4, ax=axes[1, 1], label="Complexity (bits)")
    axes[1, 1].contour(
        Qt, Pt, Ss_theory_BW,
        levels=30,
        colors="white",
        alpha=0.3,
        linewidths=0.5,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved heatmap to {save_path}")
    plt.show()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    print(
        """
    ╔══════════════════════════════════════════════════════════════════╗
    ║        Statistical Complexity Experiment Suite                   ║
    ╚══════════════════════════════════════════════════════════════════╝

    This script runs experiments to compare empirical and theoretical
    statistical complexity across different HMM parameters.
    """
    )

    print("\n" + "=" * 70)
    print("STEP 1: Computing Theoretical Complexity Heatmap")
    print("=" * 70)

    Ss_theory_FW, Ss_theory_BW, p_th, q_th = heatmap_theory(
        p_values=np.linspace(0.01, 0.99, 100),
        q_values=np.linspace(0.01, 0.99, 100),
    )

    print("\n" + "=" * 70)
    print("STEP 2: Running Empirical Experiments")
    print("=" * 70)
    print("WARNING: This will train 200 models (100 forward + 100 backward) for the 10x10 grid.")
    print("         (This can take a long time.)")

    response = input("\nProceed? (y/n): ")
    if response.lower() == "y":
        Ss_emp, p_emp, q_emp = pq_experiment(
            num_token=3,
            d_model=20,
            max_len=200,
            batch_size=32,
            num_samples=2000,
            max_epochs=20,
            lr=1e-2,
        )

        print("\n" + "=" * 70)
        print("STEP 3: Plotting Results")
        print("=" * 70)

        plot_heatmap(
            Ss_emp,
            Ss_theory_FW,
            Ss_theory_BW,
            p_emp=p_emp,
            q_emp=q_emp,
            p_theory=p_th,
            q_theory=q_th,
            save_path="complexity_heatmap.png",
        )

        print("\n✓ Experiment complete!")
        print("  Results saved to complexity_heatmap.png")
    else:
        print("\nExperiment cancelled.")
