import lightning as L
from OneHot_model import OneHotDecoder, WordEmbDecoder
import torch


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

def test_train_validation(train_loader, test_ratio, train_ratio, val_ratio):
    if test_ratio + train_ratio + val_ratio != 1.0:
        sum = test_ratio + train_ratio + val_ratio
        test_ratio /= sum
        train_ratio /= sum
        val_ratio /= sum
        print(f"Ratios normalized to sum to 1.0: test_ratio={test_ratio}, train_ratio={train_ratio}, val_ratio={val_ratio}")
    total_samples = len(train_loader.dataset)
    train_features, train_labels = next(iter(train_loader))
    total_batches = len(train_features)

    num_test = int(total_batches * test_ratio)
    num_train = int(total_batches * train_ratio)
    num_val = int(total_batches * val_ratio)

    print(f"Total samples: {total_samples}, Total batches: {total_batches}")
    print(f"Splitting into: {num_train} train batches, {num_val} val batches, {num_test} test batches") 

    train_set, val_set, test_set = torch.utils.data.random_split(
        train_loader.dataset, [num_train, num_val, num_test]
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_loader.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=train_loader.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=train_loader.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

