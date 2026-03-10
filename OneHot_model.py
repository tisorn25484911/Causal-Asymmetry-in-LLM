import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with optional REVERSE position assignment.

    reverse_pos=False:
        position t uses PE[t]

    reverse_pos=True:
        position t uses PE[T-1-t]   (mirror positions within current sequence length T)
    """
    def __init__(self, d_model=20, max_len=150):
        super().__init__()
        self.d_model = d_model
        self._build_pe(max_len)

    def _build_pe(self, max_len, device=None, dtype=None):
        # Ensure float dtype (sin/cos needs float)
        if dtype is None:
            dtype = torch.float32

        d_model = self.d_model
        pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)

        pos = torch.arange(0, max_len, device=device, dtype=dtype).unsqueeze(1)  # (max_len, 1)
        ith = torch.arange(0, d_model, 2, device=device, dtype=dtype)            # (d_model/2,)
        div = 10000 ** (ith / d_model)                                           # (d_model/2,)

        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, reverse_pos: bool = False):
        B, T, D = x.shape

        # Grow PE if needed (no info loss)
        if T > self.pe.size(0):
            new_len = max(T, self.pe.size(0) * 2)
            self._build_pe(new_len, device=x.device, dtype=x.dtype)

        pe_T = self.pe[:T]  # (T, D)
        if reverse_pos:
            pe_T = torch.flip(pe_T, dims=[0])  # (T, D) reversed along position axis

        return x + pe_T.unsqueeze(0)  # (B, T, D)


class AttentionModel(nn.Module):
    def __init__(self, d_model=20):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, q, k, v, mask=None, return_attn=False):
        Q = self.wq(q)  # (B, T, D)
        K = self.wk(k)  # (B, T, D)
        V = self.wv(v)  # (B, T, D)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_model ** 0.5)  # (B, T, T)

        if mask is not None:
            # mask True = allowed, False = blocked
            scores = scores.masked_fill(~mask, -1e9)

        prob = torch.softmax(scores, dim=-1)  # (B, T, T)
        out = prob @ V                        # (B, T, D)

        if return_attn:
            return out, prob
        return out
    
def cross_ent_onehot(logits, targets):
    B, T, C = logits.shape
    flat_input  = logits.reshape(-1, C)   # (B*T, V)
    flat_target = targets.reshape(-1)     # (B*T,)

    # replaces the entire for-loop — same result, fully vectorized
    target_prob = F.one_hot(flat_target, num_classes=C).float()   # (B*T, V)

    logits_prob = flat_input.softmax(dim=-1)
    cond_ent    = -(target_prob * logits_prob.log2()).sum(dim=1)
    loss        = cond_ent.mean()
    perplexity  = torch.exp(loss)

    return loss, perplexity



class OneHotDecoder(L.LightningModule):
    """
    Forward mode:
      - causal mask (tril)
      - normal PE

    Backward mode:
      - anti-causal mask (triu) to allow attending to future
      - optional reversed PE via reverse_pos_for_backward flag
    """
    def __init__(
        self,
        token_size=3,
        d_model=20,
        max_len=150,
        lr=1e-2,
        mode="forward",
        reverse_pos_for_backward: bool = False,
        n_layers=2,
    ):
        super().__init__()
        self.mode = mode
        self.reverse_pos_for_backward = reverse_pos_for_backward
        self.n_layers = n_layers

        self.token_size = token_size
        self.we = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.d_model = d_model
        self.max_len = max_len
        self.lr = lr

        rand_prj = torch.randn(token_size, d_model)
        rand_prj = F.normalize(rand_prj, dim=1)
        self.rand_prj = nn.Parameter(rand_prj)

        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        
        self.attn_layers = nn.ModuleList([
            AttentionModel(d_model=d_model) for _ in range(n_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Linear(4 * d_model, d_model),
            ) for _ in range(n_layers)
        ])
        
        self.ln_attn = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.ln_ffn = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        self.output_prj = nn.Linear(d_model, token_size)

        # REMOVED: self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

        self.last_encodings = None
        self.last_attention = None  # (B, T, T)

    def forward(self, tokens):
        # --- sanitize tokens dtype/device ---
        if isinstance(tokens, torch.Tensor):
            if tokens.dtype in (torch.float32, torch.float64):
                tokens = tokens.long()
            elif tokens.dtype not in (torch.long, torch.int64):
                tokens = tokens.to(torch.long)
        else:
            tokens = torch.LongTensor(tokens).to(self.rand_prj.device)

        # tokens: (B, T)
        one_hot = F.one_hot(tokens, num_classes=self.token_size).float()  # (B, T, V)
        x = one_hot @ self.rand_prj
        
        # --- positional encoding (optionally reversed for backward mode) ---
        reverse_pos = (self.mode == "backward" and self.reverse_pos_for_backward)
        x = self.pe(x, reverse_pos=reverse_pos)  # (B, T, D)

        # --- build mask consistent with goal ---
        B, T, _ = x.shape
        if self.mode == "forward":
            mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool)).unsqueeze(0)  # (1,T,T)
        elif self.mode == "backward":
            mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool)).unsqueeze(0)  # (1,T,T)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'forward' or 'backward'")

        for attn, ffn, ln1, ln2 in zip(self.attn_layers, self.ffn_layers, 
                                         self.ln_attn, self.ln_ffn):
            # attention with pre-norm
            normed = ln1(x)
            attn_out, attn_prob = attn(normed, normed, normed, mask=mask, return_attn=True)
            x = x + attn_out
            
            # feedforward with pre-norm
            normed = ln2(x)
            x = x + ffn(normed)

        self.last_encodings = x.detach()
        self.last_attention = attn_prob.detach()

        logits = self.output_prj(x)  # (B, T, V)
        return logits


    def training_step(self, batch, batch_idx):
        if self.mode == "forward":
            inputs, targets = batch
        elif self.mode == "backward":
            targets, inputs = batch

        #print(f"Batch {batch_idx}: input ={inputs[:5]}, targets ={targets[:5]}")
        logits = self.forward(inputs)                      # (B, T, V)

        loss, perplexity = cross_ent_onehot(logits, targets)

        self.log("train_loss",       loss,       prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_perplexity", perplexity, prog_bar=True, on_step=True, on_epoch=True)
        #print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Perplexity={perplexity.item():.4f}")

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

class WordEmbDecoder(L.LightningModule):

    def __init__(
        self,
        token_size=3,
        d_model=20,
        max_len=150,
        lr=1e-2,
        mode="forward",
        reverse_pos_for_backward: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.reverse_pos_for_backward = reverse_pos_for_backward

        self.token_size = token_size
        self.we = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.d_model = d_model
        self.max_len = max_len
        self.lr = lr

        # Fixed random projection acts like embedding table (V, D)
        rand_prj = torch.randn(token_size, d_model)
        rand_prj = F.normalize(rand_prj, dim=1)
        self.register_buffer("rand_prj", rand_prj)

        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.attn = AttentionModel(d_model=d_model)
        self.output_prj = nn.Linear(d_model, token_size)

        # REMOVED: self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

        self.last_encodings = None
        self.last_attention = None  # (B, T, T)

    def forward(self, tokens):
        # --- sanitize tokens dtype/device ---
        if isinstance(tokens, torch.Tensor):
            if tokens.dtype in (torch.float32, torch.float64):
                tokens = tokens.long()
            elif tokens.dtype not in (torch.long, torch.int64):
                tokens = tokens.to(torch.long)
        else:
            tokens = torch.LongTensor(tokens).to(self.rand_prj.device)

        # tokens: (B, T)
        x = self.we(tokens)
        
        # --- positional encoding (optionally reversed for backward mode) ---
        reverse_pos = (self.mode == "backward" and self.reverse_pos_for_backward)
        x = self.pe(x, reverse_pos=reverse_pos)  # (B, T, D)

        # --- build mask consistent with goal ---
        B, T, _ = x.shape
        if self.mode == "forward":
            mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool)).unsqueeze(0)  # (1,T,T)
        elif self.mode == "backward":
            mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool)).unsqueeze(0)  # (1,T,T)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'forward' or 'backward'")

        # --- attention ---
        attn_out, attn_prob = self.attn(x, x, x, mask=mask, return_attn=True)  # (B,T,D), (B,T,T)
        values = x + attn_out

        self.last_encodings = values.detach()
        self.last_attention = attn_prob.detach()

        logits = self.output_prj(values)  # (B, T, V)
        return logits



    def training_step(self, batch, batch_idx):
        if self.mode == "forward":
            input, targets = batch
        elif self.mode == "backward":
            targets, input = batch
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'forward' or 'backward'")

        logits = self.forward(input)  # (B, T, V)

        # CHANGED: use cross_ent_onehot instead of self.loss
        loss, perplexity = cross_ent_onehot(logits, targets)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # ADDED: also log perplexity for consistency
        self.log("train_perplexity", perplexity, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)