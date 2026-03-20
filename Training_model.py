import math
import lightning as L
from OneHot_model import OneHotDecoder, WordEmbDecoder, cross_ent_onehot
import torch
import torch.utils.data as tud
import numpy as np

"""
# dataloader: number of samples (5000), each loader includes (input, target) pair
# input:  (B, T) token indices in [0, V-1]
# target: (B, T) token indices in [0, V-1]
"""
# ─────────────────────────────────────────────────────────────────────────────
# class and functions for large training sequence
# ────────────────────────────────────────────────────────────────────────────
class ChunckDataset(tud.Dataset):
    def __init__(self, base: tud.Dataset, chunck_len: int, seed: int = 0):
        self.base = base
        self.chunck_len = chunck_len
        self.rng = np.random.default_rng(seed)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        input, target = self.base[idx] #(T, )
        T = input.shape[0] # length T of the sequence
        if T <= self.chunck_len:
            return input, target
        start = self.rng.integers(0, T - self.chunck_len +1)
        end = start + self.chunck_len
        return input[start:end], target[start:end]
    
def make_chunked_loader(
    dataset: tud.Dataset, chunk_len: int, batch_size: int, shuffle: bool = True
) -> tud.DataLoader:
    return tud.DataLoader(
        ChunckDataset(dataset, chunk_len),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,             
        persistent_workers=False,  
    )
def _loader(dataset: tud.Dataset, batch_size: int, shuffle: bool = False) -> tud.DataLoader:
    """Plain loader — FIX-3 only (no chunking; used for analysis)."""
    return tud.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=0, persistent_workers=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helper: evaluate a (possibly mid-training) model on a DataLoader
# Temporarily switches to eval mode, then restores train mode.
# ─────────────────────────────────────────────────────────────────────────────
def _eval_loss_on_loader(model, loader) -> tuple[float, float]:
    """Return (mean cross-entropy loss, mean perplexity) over *loader*."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_ppl, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            # mirror training_step convention for backward mode
            if getattr(model, "mode", "forward") == "backward":
                targets, inputs = batch
            inputs  = inputs.to(device)
            targets = targets.to(device)
            logits  = model(inputs)                     # (B, T, V)
            loss, ppl = cross_ent_onehot(logits, targets)
            total_loss += loss.item()
            total_ppl  += ppl.item()
            n_batches  += 1

    if was_training:
        model.train()                                   # restore training mode

    if n_batches == 0:
        return float("nan"), float("nan")
    return total_loss / n_batches, total_ppl / n_batches


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Callback
# ─────────────────────────────────────────────────────────────────────────────
class Record_training(L.Callback):
    """
    Records per-step and per-epoch metrics during training.

    New parameters
    --------------
    val_loader : DataLoader | None
        When supplied, the model is evaluated on this loader at every
        recorded gradient step, producing step_val_loss and step_val_ppl.
    """

    def __init__(
        self,
        record_every_n_steps: int = 1,
        record_latents: bool = False,
        record_probs: bool = False,
        max_batches_per_epoch: int | None = None,
        record_val_loss: bool = False,
        val_loader=None,                                # ← NEW
    ):
        super().__init__()
        self.record_every_n_steps   = record_every_n_steps
        self.record_latents         = record_latents
        self.record_probs           = record_probs
        self.max_batches_per_epoch  = max_batches_per_epoch
        self.val_loader             = val_loader        # ← NEW

        # ── per-step series ──────────────────────────────────────────────
        self.step_loss     = []     # training loss at each recorded step
        self.step_ppl      = []     # training perplexity at each recorded step (NEW)
        self.step_val_loss = []     # validation loss at each recorded step (NEW)
        self.step_val_ppl  = []     # validation perplexity at each recorded step (NEW)

        # ── per-epoch series ─────────────────────────────────────────────
        self.epoch_loss = []
        self.val_loss   = []        # optional (legacy Lightning metric)

        # ── optional heavy data ──────────────────────────────────────────
        self.latents = []
        self.probs   = []

        self._epoch_loss_sum   = 0.0
        self._epoch_loss_count = 0

    # ------------------------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Optional cap
        if self.max_batches_per_epoch is not None and batch_idx >= self.max_batches_per_epoch:
            return
        if trainer.global_step % self.record_every_n_steps != 0:
            return

        # ── Training loss ────────────────────────────────────────────────
        if isinstance(outputs, dict):
            loss_tensor = outputs.get("loss", None)
            if loss_tensor is None:
                return
            loss = loss_tensor.detach().float().cpu().item()
        else:
            loss = outputs.detach().float().cpu().item()

        self.step_loss.append(loss)
        self.step_ppl.append(math.exp(loss))            # PPL consistent with model's exp(CE)

        self._epoch_loss_sum   += loss
        self._epoch_loss_count += 1

        # ── Validation loss at this step ─────────────────────────────────
        if self.val_loader is not None:
            val_loss, val_ppl = _eval_loss_on_loader(pl_module, self.val_loader)
            self.step_val_loss.append(val_loss)
            self.step_val_ppl.append(val_ppl)
            pl_module.train()                           # ensure still in train mode

        # ── Optional legacy Lightning val metric ─────────────────────────
        if self.val_loader is None:                     # fall back to logged metric
            val_loss_metric = trainer.callback_metrics.get("val_loss", None)
            if val_loss_metric is not None:
                self.val_loss.append(val_loss_metric.detach().float().cpu().item())

        # ── Optional heavy data ──────────────────────────────────────────
        if self.record_latents or self.record_probs:
            inputs, targets = batch
            if getattr(pl_module, "mode", "forward") == "backward":
                targets, inputs = batch
            device = pl_module.device
            inputs = inputs.to(device)
            with torch.no_grad():
                logits = pl_module(inputs)
            if self.record_probs:
                p = torch.softmax(logits, dim=-1).detach().cpu()
                self.probs.append(p)
            if self.record_latents:
                z = pl_module.last_encodings.detach().cpu()
                self.latents.append(z)

    # ------------------------------------------------------------------
    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_loss_count > 0:
            self.epoch_loss.append(self._epoch_loss_sum / self._epoch_loss_count)
        self._epoch_loss_sum   = 0.0
        self._epoch_loss_count = 0


# ─────────────────────────────────────────────────────────────────────────────
# train_model
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    train_loader,
    num_token: int   = 3,
    d_model:   int   = 20,
    max_len:   int   = 15,
    max_epochs: int  = 5,
    lr:        float = 1e-2,
    mode:      str   = "forward",
    embed_type: str  = "onehot",
    val_loader       = None, 
):
    """
    Trains a OneHotDecoder / WordEmbDecoder and returns a Record_training object.

    recorder.model          - trained model
    recorder.step_loss      - training loss per gradient step
    recorder.step_ppl       - training perplexity per gradient step
    recorder.step_val_loss  - validation loss per gradient step (if val_loader given)
    recorder.step_val_ppl   - validation perplexity per gradient step (if val_loader given)
    recorder.epoch_loss     - epoch-averaged training loss
    """
    if embed_type == "onehot":
        model = OneHotDecoder(
            token_size=num_token, d_model=d_model, max_len=max_len, lr=lr, mode=mode,
        )
    elif embed_type == "wordemb":
        model = WordEmbDecoder(
            token_size=num_token, d_model=d_model, max_len=max_len, lr=lr, mode=mode,
        )
    else:
        raise ValueError(f"Invalid embed_type: {embed_type!r}. Must be 'onehot' or 'wordemb'.")

    recorder = Record_training(
        record_every_n_steps=1,
        record_latents=False,
        record_probs=False,
        record_val_loss=True,
        max_batches_per_epoch=None,
        val_loader=val_loader,      # ← NEW
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=5,
        callbacks=[recorder],
    )
    trainer.fit(model, train_loader)

    recorder.model = model
    return recorder


# ─────────────────────────────────────────────────────────────────────────────
# Utility: initial train/test split
# ─────────────────────────────────────────────────────────────────────────────
def test_train_validation(
    train_loader, test_ratio, train_ratio
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if test_ratio + train_ratio != 1.0:
        s = test_ratio + train_ratio
        test_ratio  /= s
        train_ratio /= s
        print(f"Ratios normalised → test={test_ratio:.4f}, train={train_ratio:.4f}")

    total_samples = len(train_loader.dataset)
    num_test      = int(total_samples * test_ratio)
    num_train     = total_samples - num_test

    print(f"Total samples: {total_samples}")
    print(f"Splitting into: {num_train} train, {num_test} test samples")

    train_set, test_set = torch.utils.data.random_split(
        train_loader.dataset, [num_train, num_test]
    )
    print(f"Actual split sizes: Train={len(train_set)}, Test={len(test_set)}")

    train_loader_out = torch.utils.data.DataLoader(
        train_set, batch_size=train_loader.batch_size, shuffle=True
    )
    test_loader_out = torch.utils.data.DataLoader(
        test_set, batch_size=train_loader.batch_size, shuffle=False
    )
    return train_loader_out, test_loader_out


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def train_test_val_pipeline(
    train_loader,
    test_ratio:  tuple = (0.20, 0.80),  # (test_ratio, train_ratio)
    n_folds:     int   = 5,
    embed_type:  str   = "onehot",
    num_token:   int   = 3,
    d_model:     int   = 20,
    max_len:     int   = 1000,
    max_epochs:  int   = 5,
    lr:          float = 1e-2,
    mode:        str   = "forward",
    save_plot:   str   = "cv_results.png",
):
    """
    Full cross-validation pipeline with step-level training + validation curves.

    Returns
    -------
    dict with keys:
        best_fold      - 0-based index of the winning fold
        best_recorder  - Record_training from the best fold
        best_model     - trained model from the best fold
        fold_val_loss  - list[float]: mean val loss per fold
        fold_val_ppl   - list[float]: mean val perplexity per fold
        fold_test_ppl  - list[float]: test PPL for *every* fold model (NEW)
        test_loss      - float: test CE loss for the best-fold model
        test_ppl       - float: test perplexity for the best-fold model
        all_recorders  - list of all fold recorders
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    batch_size   = train_loader.batch_size
    full_dataset = train_loader.dataset
    N            = len(full_dataset)
    test, train_val = test_ratio

    # ── 1. Hold-out test split ───────────────────────────────────────────
    print(f"Initial split ratios: test={test}, train_val={train_val}")
    trainval_loader, test_loader = test_train_validation(
        train_loader, test_ratio=test, train_ratio=train_val
    )
    trainval_set = trainval_loader.dataset
    test_set     = test_loader.dataset
    n_test       = len(test_set)
    n_trainval   = N - n_test

    print(f"\n{'='*65}")
    print(f"  {n_folds}-Fold Cross-Validation Pipeline")
    print(f"{'='*65}")
    print(f"  Total samples    : {N}")
    print(f"  Held-out test    : {n_test}  ({test*100:.0f}%)")
    print(f"  Train+Val pool   : {n_trainval}")
    print(f"  Folds            : {n_folds}")
    print(f"  Mode             : {mode}  |  embed: {embed_type}")
    print(f"  d_model={d_model}  max_epochs={max_epochs}  lr={lr}")
    print(f"{'='*65}\n")

    # ── 2. n-fold cross-validation ───────────────────────────────────────
    fold_size     = n_trainval // n_folds
    indices       = torch.randperm(
        n_trainval, generator=torch.Generator().manual_seed(0)
    ).tolist()

    fold_val_loss  = []
    fold_val_ppl   = []
    all_recorders  = []

    for fold in range(n_folds):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        val_idx   = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = indices[: fold * fold_size] + indices[(fold + 1) * fold_size :]

        fold_train = torch.utils.data.Subset(trainval_set, train_idx)
        fold_val   = torch.utils.data.Subset(trainval_set, val_idx)

        fold_train_loader = torch.utils.data.DataLoader(
            fold_train, batch_size=batch_size, shuffle=True
        )
        fold_val_loader = torch.utils.data.DataLoader(
            fold_val, batch_size=batch_size, shuffle=False
        )

        print(f"  Train samples: {len(fold_train)}  |  Val samples: {len(fold_val)}")

        # ── Train – pass the val loader so step-level val is recorded ──
        recorder = train_model(
            fold_train_loader,
            num_token=num_token,
            d_model=d_model,
            max_len=max_len,
            max_epochs=max_epochs,
            lr=lr,
            mode=mode,
            embed_type=embed_type,
            val_loader=fold_val_loader,   # ← NEW
        )
        all_recorders.append(recorder)

        # Final val metrics (whole val set, one pass)
        val_loss, val_ppl = _eval_loss_on_loader(recorder.model, fold_val_loader)
        fold_val_loss.append(val_loss)
        fold_val_ppl.append(val_ppl)

        print(f"  ✓ Fold {fold+1}  val loss: {val_loss:.4f}  |  val ppl: {val_ppl:.4f}")

    # ── 3. Select best fold ──────────────────────────────────────────────
    best_fold     = int(np.argmin(fold_val_loss))
    best_recorder = all_recorders[best_fold]
    best_model    = best_recorder.model

    print(f"\n{'='*65}")
    print(f"  Best fold: {best_fold + 1}  "
          f"(val loss={fold_val_loss[best_fold]:.4f}, "
          f"val ppl={fold_val_ppl[best_fold]:.4f})")
    print(f"{'='*65}")

    # ── 4. Evaluate ALL fold models on the held-out test set ─────────────
    print("\n  Evaluating ALL fold models on the held-out test set …")
    fold_test_loss = []
    fold_test_ppl  = []
    for fold, rec in enumerate(all_recorders):
        t_loss, t_ppl = _eval_loss_on_loader(rec.model, test_loader)
        fold_test_loss.append(t_loss)
        fold_test_ppl.append(t_ppl)
        marker = "  ★ best" if fold == best_fold else ""
        print(f"    Fold {fold+1}: test loss={t_loss:.4f}  test ppl={t_ppl:.4f}{marker}")

    test_loss = fold_test_loss[best_fold]
    test_ppl  = fold_test_ppl[best_fold]

    # ── 5. Visualisation (6 subplots, 3×2) ──────────────────────────────
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]
    line_colors = plt.cm.tab10(np.linspace(0, 0.9, n_folds))
    bar_color   = "#3a86ff"          # single colour for all bar charts

    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.38, wspace=0.22
    )

    # ── (1) Training loss / step ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i, rec in enumerate(all_recorders):
        is_best = (i == best_fold)
        ax1.plot(
            rec.step_loss,
            color=line_colors[i],
            linewidth=2.5 if is_best else 1.0,
            alpha=1.0 if is_best else 0.40,
            label=fold_labels[i] + (" ★" if is_best else ""),
        )
    ax1.set_title("Training Loss / Step", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Gradient Step")
    ax1.set_ylabel("Loss (bits)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── (2) Validation loss / step ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, rec in enumerate(all_recorders):
        is_best = (i == best_fold)
        if rec.step_val_loss:
            ax2.plot(
                rec.step_val_loss,
                color=line_colors[i],
                linewidth=2.5 if is_best else 1.0,
                alpha=1.0 if is_best else 0.40,
                label=fold_labels[i] + (" ★" if is_best else ""),
            )
    ax2.set_title("Validation Loss / Step", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Gradient Step")
    ax2.set_ylabel("Val Loss (bits)")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── (3) Training perplexity / step ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for i, rec in enumerate(all_recorders):
        is_best = (i == best_fold)
        ax3.plot(
            rec.step_ppl,
            color=line_colors[i],
            linewidth=2.5 if is_best else 1.0,
            alpha=1.0 if is_best else 0.40,
            label=fold_labels[i] + (" ★" if is_best else ""),
        )
    ax3.set_title("Training Perplexity / Step", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Gradient Step")
    ax3.set_ylabel("Perplexity")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # ── (4) Validation perplexity / step ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for i, rec in enumerate(all_recorders):
        is_best = (i == best_fold)
        if rec.step_val_ppl:
            ax4.plot(
                rec.step_val_ppl,
                color=line_colors[i],
                linewidth=2.5 if is_best else 1.0,
                alpha=1.0 if is_best else 0.40,
                label=fold_labels[i] + (" ★" if is_best else ""),
            )
    ax4.set_title("Validation Perplexity / Step", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Gradient Step")
    ax4.set_ylabel("Val Perplexity")
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    # ── (5) Bar: final validation PPL per fold ───────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    alphas = [1.0 if i == best_fold else 0.3 for i in range(n_folds)]
    for i, (label, val) in enumerate(zip(fold_labels, fold_val_ppl)):
        ax5.bar(label, val, color=bar_color, alpha=alphas[i], edgecolor="k", linewidth=0.8)
    ax5.bar_label(
        ax5.containers[0], fmt="%.4f", padding=3, fontsize=9
    )
    ax5.axhline(
        np.mean(fold_val_ppl), color="crimson", linestyle="--", linewidth=1.4,
        label=f"Mean = {np.mean(fold_val_ppl):.4f}",
    )
    ax5.set_title("Final Validation PPL per Fold", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Val Perplexity")
    ax5.set_ylim(0, max(fold_val_ppl) * 1.22)
    ax5.legend(fontsize=9)
    ax5.grid(True, axis="y", alpha=0.3)

    # ── (6) Bar: test PPL for every fold model ───────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    for i, (label, val) in enumerate(zip(fold_labels, fold_test_ppl)):
        ax6.bar(label, val, color=bar_color, alpha=alphas[i], edgecolor="k", linewidth=0.8)
    ax6.bar_label(
        ax6.containers[0], fmt="%.4f", padding=3, fontsize=9
    )
    ax6.axhline(
        np.mean(fold_test_ppl), color="crimson", linestyle="--", linewidth=1.4,
        label=f"Mean = {np.mean(fold_test_ppl):.4f}",
    )
    ax6.set_title("Test PPL per Fold Model (held-out set)", fontsize=12, fontweight="bold")
    ax6.set_ylabel("Test Perplexity")
    ax6.set_ylim(0, max(fold_test_ppl) * 1.22)
    ax6.legend(fontsize=9)
    ax6.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"{n_folds}-Fold CV  |  mode={mode}  embed={embed_type}  "
        f"d_model={d_model}  lr={lr}  epochs={max_epochs}\n"
        f"Best: Fold {best_fold+1}  "
        f"(val ppl={fold_val_ppl[best_fold]:.4f},  "
        f"test ppl={test_ppl:.4f})",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )

    plt.savefig(save_plot, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ CV visualisation saved → {save_plot}")
    plt.close(fig)

    # ── 6. Console summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  HELD-OUT TEST SET  (best fold = {best_fold + 1})")
    print(f"{'='*65}")
    print(f"  Test loss (best fold) : {test_loss:.4f} bits")
    print(f"  Test ppl  (best fold) : {test_ppl:.4f}")
    print(f"\n  Cross-Validation Summary")
    print(f"  {'Fold':<8} {'Val Loss':>10} {'Val PPL':>10} {'Test PPL':>10}")
    print(f"  {'-'*44}")
    for i in range(n_folds):
        marker = "  ★" if i == best_fold else ""
        print(
            f"  {fold_labels[i]:<8} "
            f"{fold_val_loss[i]:>10.4f} "
            f"{fold_val_ppl[i]:>10.4f} "
            f"{fold_test_ppl[i]:>10.4f}{marker}"
        )
    print(f"  {'-'*44}")
    print(
        f"  {'Mean':<8} "
        f"{float(np.mean(fold_val_loss)):>10.4f} "
        f"{float(np.mean(fold_val_ppl)):>10.4f} "
        f"{float(np.mean(fold_test_ppl)):>10.4f}"
    )
    print(
        f"  {'Std':<8} "
        f"{float(np.std(fold_val_loss)):>10.4f} "
        f"{float(np.std(fold_val_ppl)):>10.4f} "
        f"{float(np.std(fold_test_ppl)):>10.4f}"
    )
    print(f"{'='*65}\n")

    return {
        "best_fold"     : best_fold,
        "best_recorder" : best_recorder,
        "best_model"    : best_model,
        "fold_val_loss" : fold_val_loss,
        "fold_val_ppl"  : fold_val_ppl,
        "fold_test_ppl" : fold_test_ppl,      
        "test_loss"     : test_loss,
        "test_ppl"      : test_ppl,
        "all_recorders" : all_recorders,
    }