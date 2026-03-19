import lightning as L
from OneHot_model import OneHotDecoder, WordEmbDecoder
import torch
"""
# dataloader: numper of sample (5000), each loader includes (input, target) pair
# input: (B, T) with token indices in [0, V-1] with batches and each betch having sequence length T
# target: (B, T) with token indices in [0, V-1] with batches and each betch having sequence length T
""" 
class Record_training(L.Callback):
    def __init__(
        self,
        record_every_n_steps=1,
        record_latents=False,
        record_probs=False,
        max_batches_per_epoch=None,
    ):
        super().__init__()
        self.record_every_n_steps = record_every_n_steps
        self.record_latents = record_latents
        self.record_probs = record_probs
        self.max_batches_per_epoch = max_batches_per_epoch

        self.step_loss = []
        self.epoch_loss = []

        self.latents = []
        self.probs = []

        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Optional cap (useful to reduce memory/time) 
        if self.max_batches_per_epoch is not None and batch_idx >= self.max_batches_per_epoch:
            return

        if trainer.global_step % self.record_every_n_steps != 0:
            return

        # Lightning "outputs" could be a tensor loss or a dict containing "loss"
        if isinstance(outputs, dict):
            loss_tensor = outputs.get("loss", None)
            if loss_tensor is None:
                return
            loss = loss_tensor.detach().float().cpu().item()
        else:
            loss = outputs.detach().float().cpu().item()

        self.step_loss.append(loss)

        self._epoch_loss_sum += loss
        self._epoch_loss_count += 1

        # Record optional heavy data consistently with the model's mode
        if self.record_latents or self.record_probs:
            # batch format expected from your loader: (inputs, targets)
            inputs, targets = batch

            # IMPORTANT:
            # Your OneHotDecoder.training_step swaps for backward:
            #   backward: targets, inputs = batch
            # So we must feed the SAME "inputs" that training uses.
            if getattr(pl_module, "mode", "forward") == "backward":
                targets, inputs = batch  # match model's backward convention

            device = pl_module.device
            inputs = inputs.to(device)

            with torch.no_grad():
                logits = pl_module(inputs)  # (B, T, V)

                if self.record_probs:
                    # Token probs from logits (NOT attention probs)
                    p = torch.softmax(logits, dim=-1).detach().cpu()
                    self.probs.append(p)

                if self.record_latents:
                    # Latents saved by your model forward pass
                    z = pl_module.last_encodings.detach().cpu()
                    self.latents.append(z)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_loss_count > 0:
            epoch_loss = self._epoch_loss_sum / self._epoch_loss_count
            self.epoch_loss.append(epoch_loss)

        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0


def train_model(
    train_loader,
    num_token=3,
    d_model=20,
    max_len=15,
    max_epochs=5,
    lr=1e-2,
    mode="forward",
    embed_type="onehot",
):
    """
    Trains a OneHotDecoder in either forward or backward mode and returns a recorder object.
    recorder.model holds the trained model; recorder.step_loss / recorder.epoch_loss store losses.
    """
    if embed_type == "onehot":
        model = OneHotDecoder(
            token_size=num_token,
            d_model=d_model,
            max_len=max_len,
            lr=lr,
            mode=mode,
        )
    elif embed_type == "wordemb":
        model = WordEmbDecoder(
            token_size=num_token,
            d_model=d_model,
            max_len=max_len,
            lr=lr,
            mode=mode,
        )
    else:
        raise ValueError(f"Invalid embed_type: {embed_type}. Must be 'onehot' or 'wordemb'.")

    recorder = Record_training(
        record_every_n_steps=1,
        record_latents=False,
        record_probs=False,
        max_batches_per_epoch=None,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=5,
        callbacks=[recorder],
    )

    trainer.fit(model, train_loader)

    # attach model for convenient access outside
    recorder.model = model
    return recorder

def test_train_validation(train_loader, test_ratio, train_ratio) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    if test_ratio + train_ratio != 1.0:
        sum = test_ratio + train_ratio
        test_ratio /= sum
        train_ratio /= sum
        print(f"Ratios normalized to sum to 1.0: test_ratio={test_ratio}, train_ratio={train_ratio}")
    total_samples = len(train_loader.dataset)
    train_features, train_labels = next(iter(train_loader))
    total_batches = len(train_features) # total number of batches

    num_test = int(total_samples * test_ratio)
    num_train = int(total_samples - num_test)
    print(f"Total samples: {total_samples}, Total batches: {total_batches}")
    print(f"Splitting into: {num_train} train samples, {num_test} test samples") 
    print(f"Total data length {len(train_loader.dataset)}, Number of data in each separation ={[(num_train), (num_test)]}, total separation: {num_train + num_test}")

    train_set, test_set = torch.utils.data.random_split(
        train_loader.dataset, [num_train, num_test]
    )
    print(f"\n Actual split sizes: Train={len(train_set)}, Test={len(test_set)}")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_loader.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=train_loader.batch_size, shuffle=False)
    return train_loader, test_loader


def _eval_loss_on_loader(model, loader) -> tuple[float, float]: # evaluation of a trained model
    from OneHot_model import cross_ent_onehot
    model.eval() # define model to eval mode and setup device
    device = next(model.parameters()).device
    total_loss, total_ppl, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            # mirror training_step convention for backward mode
            if getattr(model, "mode", "forward") == "backward":
                targets, inputs = batch

            inputs  = inputs.to(device)
            targets = targets.to(device) # (B, T) with token indices in [0, V-1]

            logits = model(inputs)                        # (B, T, V)
            loss, ppl = cross_ent_onehot(logits, targets)
            total_loss += loss.item()
            total_ppl  += ppl.item()
            n_batches  += 1

    if n_batches == 0:
        return float("nan"), float("nan")
    return total_loss / n_batches, total_ppl / n_batches


def train_test_val_pipeline(
    train_loader,
    test_ratio: tuple = (0.20, 0.8),  # (test_ratio, train_ratio)
    n_folds: int = 5,
    embed_type: str = "onehot",
    num_token: int = 3,
    d_model: int = 20,
    max_len: int = 15,
    max_epochs: int = 5,
    lr: float = 1e-2,
    mode: str = "forward",
    save_plot: str = "cv_results.png",
):
    """
    Returns
    -------
    dict with keys
        ``best_fold``      - 0-based index of the winning fold
        ``best_recorder``  - Record_training from the best fold
        ``best_model``     - trained model from the best fold
        ``fold_val_loss``  - list[float], mean val loss per fold
        ``fold_val_ppl``   - list[float], mean val perplexity per fold
        ``test_loss``      - float, cross-entropy loss on held-out test set
        ``test_ppl``       - float, perplexity on held-out test set
        ``all_recorders``  - list of all fold recorders (for detailed inspection)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    batch_size   = train_loader.batch_size
    full_dataset = train_loader.dataset
    N            = len(full_dataset)

    train_val, test = test_ratio
    print(f"Initial split ratios: test={test}, train={train_val}")
    trainval_loader, test_loader = test_train_validation(train_loader, test_ratio=test, train_ratio=train_val)  

    trainval_set = trainval_loader.dataset  # Subset of the original dataset for train+val
    test_set     = test_loader.dataset      # Subset of the original dataset for test
    # ------------------------------------------------------------------ #
    # 1.  Hold-out test split                                              #
    # ------------------------------------------------------------------ #
    n_test     = test_loader.dataset.__len__()  # number of samples in test set
    n_trainval = N - n_test



    print(f"\n{'='*65}")
    print(f"  {n_folds}-Fold Cross-Validation Pipeline")
    print(f"{'='*65}")
    print(f"  Total samples    : {N}")
    print(f"  Held-out test    : {n_test}  ({train_val*100:.0f}%, {test*100:.0f}%)")
    print(f"  Train+Val pool   : {n_trainval}")
    print(f"  Folds            : {n_folds}")
    print(f"  Mode             : {mode}  |  embed: {embed_type}")
    print(f"  d_model={d_model}  max_epochs={max_epochs}  lr={lr}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------ #
    # 2.  n-fold cross-validation                                          #
    # ------------------------------------------------------------------ #
    fold_size     = n_trainval // n_folds
    indices       = torch.randperm(n_trainval, generator=torch.Generator().manual_seed(0)).tolist()

    fold_val_loss = []
    fold_val_ppl  = []
    all_recorders = []

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

        recorder = train_model(
            fold_train_loader,
            num_token=num_token,
            d_model=d_model,
            max_len=max_len,
            max_epochs=max_epochs,
            lr=lr,
            mode=mode,
            embed_type=embed_type,
        )
        all_recorders.append(recorder)

        val_loss, val_ppl = _eval_loss_on_loader(recorder.model, fold_val_loader)
        fold_val_loss.append(val_loss)
        fold_val_ppl.append(val_ppl)

        print(f"  ✓ Fold {fold+1}  val loss: {val_loss:.4f}  |  val perplexity: {val_ppl:.4f}")

    # ------------------------------------------------------------------ #
    # 3.  Select best fold (lowest validation loss)                        #
    # ------------------------------------------------------------------ #
    best_fold     = int(np.argmin(fold_val_loss))
    best_recorder = all_recorders[best_fold]
    best_model    = best_recorder.model

    print(f"\n{'='*65}")
    print(f"  Best fold: {best_fold + 1}  "
          f"(val loss = {fold_val_loss[best_fold]:.4f}, "
          f"val ppl = {fold_val_ppl[best_fold]:.4f})")
    print(f"{'='*65}")

    # ------------------------------------------------------------------ #
    # 4.  Visualisation                                                    #
    # ------------------------------------------------------------------ #
    colors      = plt.cm.tab10(np.linspace(0, 0.9, n_folds))
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── (A) Step-level training loss curves ────────────────────────────
    ax_step = fig.add_subplot(gs[0, 0])
    for i, rec in enumerate(all_recorders):
        is_best = (i == best_fold)
        ax_step.plot(
            rec.step_loss,
            color=colors[i],
            linewidth=2.5 if is_best else 1.0,
            alpha=1.0  if is_best else 0.45,
            label=fold_labels[i] + (" ★" if is_best else ""),
        )
    ax_step.set_title("Training Loss – Step Level", fontsize=12, fontweight="bold")
    ax_step.set_xlabel("Gradient Step")
    ax_step.set_ylabel("Loss (bits)")
    ax_step.legend(fontsize=8, ncol=2)
    ax_step.grid(True, alpha=0.3)

    # ── (B) Epoch-averaged training loss curves ─────────────────────────
    ax_epoch = fig.add_subplot(gs[0, 1])
    for i, rec in enumerate(all_recorders):
        is_best = (i == best_fold)
        ax_epoch.plot(
            range(1, len(rec.epoch_loss) + 1),
            rec.epoch_loss,
            marker="o",
            markersize=4,
            color=colors[i],
            linewidth=2.5 if is_best else 1.0,
            alpha=1.0  if is_best else 0.45,
            label=fold_labels[i] + (" ★" if is_best else ""),
        )
    ax_epoch.set_title("Training Loss – Epoch Average", fontsize=12, fontweight="bold")
    ax_epoch.set_xlabel("Epoch")
    ax_epoch.set_ylabel("Loss (bits)")
    ax_epoch.set_xticks(range(1, max_epochs + 1))
    ax_epoch.legend(fontsize=8, ncol=2)
    ax_epoch.grid(True, alpha=0.3)

    # ── (C) Val loss per fold bar chart ─────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 0])
    bar_cols           = list(colors)
    bar_cols[best_fold] = (1.0, 0.84, 0.0, 1.0)   # gold for best
    bars = ax_bar.bar(fold_labels, fold_val_loss, color=bar_cols, edgecolor="k", linewidth=0.8)
    ax_bar.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax_bar.axhline(
        np.mean(fold_val_loss), color="crimson", linestyle="--", linewidth=1.4,
        label=f"Mean = {np.mean(fold_val_loss):.4f}",
    )
    ax_bar.set_title("Validation Loss per Fold", fontsize=12, fontweight="bold")
    ax_bar.set_ylabel("Val Loss (bits)")
    ax_bar.set_ylim(0, max(fold_val_loss) * 1.22)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.3)

    # ── (D) Val perplexity per fold bar chart ────────────────────────────
    ax_ppl = fig.add_subplot(gs[1, 1])
    ppl_cols           = list(colors)
    ppl_cols[best_fold] = (1.0, 0.84, 0.0, 1.0)
    bars2 = ax_ppl.bar(fold_labels, fold_val_ppl, color=ppl_cols, edgecolor="k", linewidth=0.8)
    ax_ppl.bar_label(bars2, fmt="%.4f", padding=3, fontsize=9)
    ax_ppl.axhline(
        np.mean(fold_val_ppl), color="crimson", linestyle="--", linewidth=1.4,
        label=f"Mean = {np.mean(fold_val_ppl):.4f}",
    )
    ax_ppl.set_title("Validation Perplexity per Fold", fontsize=12, fontweight="bold")
    ax_ppl.set_ylabel("Val Perplexity")
    ax_ppl.set_ylim(0, max(fold_val_ppl) * 1.22)
    ax_ppl.legend(fontsize=9)
    ax_ppl.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"{n_folds}-Fold CV  |  mode={mode}  embed={embed_type}  "
        f"d_model={d_model}  lr={lr}  epochs={max_epochs}\n"
        f"Best: Fold {best_fold+1}  "
        f"(val loss={fold_val_loss[best_fold]:.4f}, "
        f"val ppl={fold_val_ppl[best_fold]:.4f})",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    plt.savefig(save_plot, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ CV visualisation saved → {save_plot}")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 5.  Held-out test evaluation (best-fold model)                       #
    # ------------------------------------------------------------------ #
    test_loss, test_ppl = _eval_loss_on_loader(best_model, test_loader)

    print(f"\n{'='*65}")
    print(f"  HELD-OUT TEST SET  (best fold = {best_fold + 1})")
    print(f"{'='*65}")
    print(f"  Test loss        : {test_loss:.4f} bits")
    print(f"  Test perplexity  : {test_ppl:.4f}")
    print(f"\n  Cross-Validation Summary")
    print(f"  {'Fold':<8} {'Val Loss':>10} {'Val PPL':>10}")
    print(f"  {'-'*32}")
    for i in range(n_folds):
        marker = "  ★ best" if i == best_fold else ""
        print(f"  {fold_labels[i]:<8} {fold_val_loss[i]:>10.4f} {fold_val_ppl[i]:>10.4f}{marker}")
    print(f"  {'-'*32}")
    print(f"  {'Mean':<8} {float(np.mean(fold_val_loss)):>10.4f} {float(np.mean(fold_val_ppl)):>10.4f}")
    print(f"  {'Std':<8} {float(np.std(fold_val_loss)):>10.4f} {float(np.std(fold_val_ppl)):>10.4f}")
    print(f"{'='*65}\n")

    return {
        "best_fold"     : best_fold,
        "best_recorder" : best_recorder,
        "best_model"    : best_model,
        "fold_val_loss" : fold_val_loss,
        "fold_val_ppl"  : fold_val_ppl,
        "test_loss"     : test_loss,
        "test_ppl"      : test_ppl,
        "all_recorders" : all_recorders,
    }

