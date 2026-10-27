"""
The hybrid transformer-GRU (HYBRID_TRANSFORMER_GRU_PLAN.md, approved 2026-10-06).

The transformer reads each history with its long causal window and CHOOSES that history's causal
state from a learned codebook.  The GRU starts from the transformer's choice and SIMULATES the state
forward on the observed tokens -- the deterministic, unifilar update rule s_{t+1} = f(s_t, x_{t+1}) --
predicting every next token from the state it reaches.  At every rollout step the GRU's state is
compared with the transformer's state for the same history.  One codebook and one emission serve
both parts, so state k means the same vector and the same next-token distribution to each.

    HybridTransformerGRU   the model: the discrete transformer (_TransformerStack + the shared causal
                           head) and the GRU simulator (GRUCell + its own state selector) on that head
    HybridRecorder         curves, validation (CE_T + CE_G selects the checkpoint) and the records the
                           standard analysis reads
    train_hybrid           one run: continuous data, the continuous regime, the rollouts, the analysis
    main                   the command line

Precedents: MuZero (Schrittwieser et al. 2020; an encoder gives the initial state, a dynamics model is
unrolled K = 5 steps), DreamerV2 (Hafner et al. 2021; discrete straight-through latents, posterior vs
prior by KL balancing with alpha = 0.8), the notebook GRU (a 20-token encoder, then a 10-step discrete
rollout).  The losses, in bits, per block of 300 inputs:

    CE_T    the transformer's next-token CE on outputs >= score_start (20)
    CE_G    the GRU's next-token CE along rollouts from t0 = 19, 29, ... (L = 10 steps each)
    KL      alpha * KL(sg(q) || p) + (1 - alpha) * KL(q || sg(p)) at every rollout step, q the
            transformer's state distribution and p the GRU's (alpha = 0.8: the GRU follows 4x faster)
    usage   beta * (H_T + H_G), the usage entropy of both selectors on their scored positions

    total = CE_T + w_gru * CE_G + w_kl * KL + usage

Run (from model_modeularised/):

    python hybrid_transformer_GRU.py --name hybrid --process coin flower --seed 0
    python hybrid_transformer_GRU.py --name hybrid9 --process renewal \\
        --params 'F=[0.04,0.08,0.12,0.16,0.2,0.16,0.12,0.08,0.04]' --num-states-mult 8
    python hybrid_transformer_GRU.py --smoke                    plumbing check -> checks/results/smoke/hybrid
    python hybrid_transformer_GRU.py --name hybrid --plots-only redraw every figure from the pickles

Reused from the pipeline: the model's building blocks (decoder_base, causal_matrix, Transformer, GRU),
data views (data_views), the tau schedule (tau_schedue), scoring masks (scoring), the process settings,
the continuous configuration and the results layout (run_model), and the analysis (extraction, plotting).
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

import plotting
import run_model
from causal_matrix import _DiscreteHead
from data_views import build_views, make_loaders, split_spans
from extraction import analyse_arm, exact_per_position_ce
from GRU import symbol_vectors
from process_generator import SequenceDataset, generator
from scoring import masked_ce_bits, score_mask, token_ce_bits
from tau_schedue import TauSchedule, parse_tau
from Transformer import _TransformerStack

LN2 = math.log(2.0)
HYBRID_COLOR, HYBRID_LABEL = "#8C5A9E", "Hybrid T+GRU"
# plotting draws architectures from these tables; registering here keeps the hybrid in its own file
plotting.ARCH_COLOR.setdefault("hybrid", HYBRID_COLOR)
plotting.ARCH_LABEL.setdefault("hybrid", HYBRID_LABEL)


# ══════════════════════════════════════════════════════════════════════════
# the model
# ══════════════════════════════════════════════════════════════════════════
class HybridTransformerGRU(_DiscreteHead, _TransformerStack):
    """
    The transformer path is exactly the discrete transformer (Transformer.DiscreteCausalDecoder):
    encoder -> selector A (causal_head.state_head, d_model -> K) -> Gumbel-ST -> codebook E
    (causal_head.state_matrix, K x S) -> emission W (causal_head.emission, S -> V).  `forward` is that
    path, so the pipeline's analysis reads the hybrid like any other model.

    The GRU simulator adds a GRUCell(V -> S) whose hidden state IS the state vector (as in
    GRU.DiscreteFeedbackGRU) and selector B (Linear(S -> K)); it reuses E and W.  `rollout` runs it
    from the transformer's choices; `machine` enumerates its exact update rule.

    The runner always reads forwards (the backward arm comes from reversed data), so mode must be
    "forward".  State dimension S = state_dim (32, the GRU width).
    """

    def __init__(self, n_states=None, state_dim=32, tau=1.0, usage_beta=0.0,
                 rollout_len=10, rollout_stride=10, rollout_first=19,
                 kl_alpha=0.8, w_gru=1.0, w_kl=1.0, detach_start=False, separate_gru_emission=False,
                 transformer_gumbel=True, **kw):
        super().__init__(**kw)
        if self.mode != "forward":
            raise ValueError("the hybrid reads forwards; take the backward arm from reversed data")
        if not 0.0 <= kl_alpha <= 1.0:
            raise ValueError("kl_alpha must lie in [0, 1]")
        self._init_causal_head(n_states, state_dim, tau, usage_beta)     # selector A, codebook E, emission W
        S = self.causal_head.state_dim
        self.cell = nn.GRUCell(input_size=self.token_size, hidden_size=S)
        self.gru_state_head = nn.Linear(S, self.n_states)                 # selector B
        # None when off: the GRU's input map, identity at the start (learn_input_embedding)
        self.register_parameter(
            "symbol_embedding",
            nn.Parameter(torch.eye(self.token_size)) if self.learn_input_embedding else None,
        )
        self.rollout_len, self.rollout_stride, self.rollout_first = int(rollout_len), int(rollout_stride), \
            int(rollout_first)
        self.kl_alpha, self.w_gru, self.w_kl = float(kl_alpha), float(w_gru), float(w_kl)
        # inspection switches (HYBRID_TRANSFORMER_GRU_PLAN.md section 11); the defaults are the design:
        #   detach_start           the GRU's losses do not reach the transformer through its start choice
        #   separate_gru_emission  the GRU predicts through its own emission (the codebook stays shared)
        #   transformer_gumbel     False: the transformer's selector trains with argmax, the GRU's with Gumbel
        self.detach_start, self.transformer_gumbel = bool(detach_start), bool(transformer_gumbel)
        self.gru_emission = nn.Linear(S, self.token_size) if separate_gru_emission else None
        self.save_hyperparameters()

    # ── the transformer path ─────────────────────────────────────────────
    def forward(self, tokens):
        return self.causal_head(self._encode(tokens), self.tau, self._sampling() and self.transformer_gumbel)

    def transformer_logits(self):
        """(B,T,K) the transformer's raw state logits of the last forward (no noise, no tau)."""
        return self.causal_head.state_head(self.last_encodings)

    # ── the GRU simulator ────────────────────────────────────────────────
    def _select(self, logits):
        """Gumbel-ST selection, exactly as causal_matrix._learn_causal_state.forward: noise only
        while training with Gumbel on.  Returns (state index, straight-through one-hot)."""
        z = logits
        if self._sampling():
            u = torch.rand_like(z).clamp_(1e-10, 1.0 - 1e-10)
            z = z - torch.log(-torch.log(u))
        probs = F.softmax(z / self.tau, dim=-1)
        states = probs.argmax(dim=-1)
        hard = F.one_hot(states, num_classes=self.n_states).float()
        return states, (hard - probs).detach() + probs

    def rollout_starts(self, T, L=None):
        """t0 = first, first + stride, ... while the rollout's last prediction stays inside the row:
        t0 + L <= T - 1 (T outputs).  For T = 300, L = 10: t0 = 19, 29, ..., 289 (28 rollouts)."""
        L = self.rollout_len if L is None else int(L)
        starts = list(range(self.rollout_first, T - L, self.rollout_stride))
        if not starts:
            raise ValueError(f"no rollout of {L} steps fits {T} outputs from {self.rollout_first}")
        return starts

    def rollout(self, inputs, start_onehot, starts=None, L=None):
        """
        The GRU simulation from the transformer's choices.  `start_onehot` (B,T,K) is the transformer's
        straight-through one-hot of the same forward, so the GRU's losses reach the transformer through
        its choice of start.  From each start t0 the GRU reads x_{t0+1}, ..., x_{t0+L}; step k gives the
        state at output position t0+k (the same history as the transformer's state there) and predicts
        x_{t0+k+1}.  Returns logits (B,R,L,V), raw state logits z (B,R,L,K), states (B,R,L), one-hots
        (B,R,L,K) and the output positions (R,L).
        """
        B, T = inputs.shape
        L = self.rollout_len if L is None else int(L)
        starts = self.rollout_starts(T, L) if starts is None else list(starts)
        dev = inputs.device
        t0 = torch.as_tensor(starts, device=dev)
        R = len(starts)
        E = self.causal_head.state_matrix
        x = symbol_vectors(self._tokens(inputs), self.token_size, self.symbol_embedding)    # (B,T,V)
        start = start_onehot.detach() if self.detach_start else start_onehot
        m = (start[:, t0] @ E).reshape(B * R, -1)                                          # (B*R, S)
        emit = self.causal_head.emission if self.gru_emission is None else self.gru_emission
        out = {k: [] for k in ("logits", "z", "states", "onehot")}
        for k in range(1, L + 1):
            h = self.cell(x[:, t0 + k].reshape(B * R, -1), m)
            z = self.gru_state_head(h)
            st, oh = self._select(z)
            m = oh @ E                                                                     # fed back
            out["logits"].append(emit(m))
            out["z"].append(z)
            out["states"].append(st)
            out["onehot"].append(oh)
        res = {k: torch.stack(v, dim=1).reshape(B, R, L, *v[0].shape[1:]) for k, v in out.items()}
        res["positions"] = t0[:, None] + torch.arange(1, L + 1, device=dev)[None, :]
        return res

    @torch.no_grad()
    def machine(self) -> dict:
        """
        The GRU's exact update rule, by enumeration over all K states and V tokens (as
        GRU.DiscreteFeedbackGRU.machine): next_state[s, x] = the state after reading x in state s;
        emission_probs = the shared emission.  The hybrid has no blank start -- the transformer
        chooses every start -- so start_state is -1 throughout.
        """
        K, V = self.n_states, self.token_size
        E = self.state_matrix
        vectors = symbol_vectors(torch.arange(V, device=E.device), V, self.symbol_embedding)
        next_state = torch.empty(K, V, dtype=torch.long)
        for x in range(V):
            h = self.cell(vectors[x].expand(K, V), E)
            next_state[:, x] = self.gru_state_head(h).argmax(dim=-1).cpu()
        return {"next_state": next_state.numpy(), "start_state": np.full(V, -1),
                "emission_probs": (self.emission_table() if self.gru_emission is None else
                                   self.gru_emission(E).softmax(dim=-1)).cpu().numpy()}

    # ── the objective ────────────────────────────────────────────────────
    @staticmethod
    def kl_balance(zq, zp, alpha):
        """DreamerV2's KL balancing between categorical distributions, in bits:
        alpha * KL(sg(q) || p) trains p (the GRU), (1 - alpha) * KL(q || sg(p)) trains q (the
        transformer).  The value is KL(q || p) whatever alpha is; alpha splits the gradient."""
        lq, lp = F.log_softmax(zq, dim=-1), F.log_softmax(zp, dim=-1)
        q = lq.exp()
        to_p = (q.detach() * (lq.detach() - lp)).sum(-1)
        to_q = (q * (lq - lp.detach())).sum(-1)
        return (alpha * to_p + (1.0 - alpha) * to_q).mean() / LN2

    @staticmethod
    def _usage(onehot):
        p_bar = onehot.reshape(-1, onehot.shape[-1]).mean(dim=0)
        return -(p_bar * torch.log2(p_bar + 1e-12)).sum()

    def losses(self, inputs, targets):
        """Every term of the objective for one batch, and what validation reads."""
        logits_T = self(inputs)
        ce_T, mask = masked_ce_bits(logits_T, targets, self.score_start, self.mode)
        zT = self.transformer_logits()
        states_T, onehot_T = self.causal_head.last_states, self.causal_head.last_state_onehot
        ro = self.rollout(inputs, onehot_T)
        pos = ro["positions"]
        B, R, Lr, V = ro["logits"].shape
        ce_G = token_ce_bits(ro["logits"].reshape(B, R * Lr, V), targets[:, pos].reshape(B, R * Lr)).mean()
        kl = self.kl_balance(zT[:, pos], ro["z"], self.kl_alpha)
        H_T = self.causal_head.usage_entropy(mask)
        H_G = self._usage(ro["onehot"])
        return {"ce_T": ce_T, "ce_G": ce_G, "kl": kl, "H_T": H_T, "H_G": H_G, "mask": mask,
                "states_T": states_T, "states_G": ro["states"], "positions": pos,
                "agree": (ro["states"] == states_T[:, pos]).float().mean()}

    def training_step(self, batch, batch_idx):
        inputs, targets = self._split(batch, self.mode)
        t = self.losses(inputs, targets)
        loss = t["ce_T"] + self.w_gru * t["ce_G"] + self.w_kl * t["kl"] + self.usage_beta * (t["H_T"] + t["H_G"])
        for k in ("ce_T", "ce_G", "kl", "H_T", "H_G", "agree"):
            self.log(f"train_{k}", t[k], on_step=True, on_epoch=True)
        n = targets.shape[0] * (targets.shape[1] - self.score_start)
        return {"loss": loss, "ce_sum": t["ce_T"].detach() * n, "scored_tokens": n,
                "ce_G": t["ce_G"].detach(), "kl": t["kl"].detach(), "agree": t["agree"].detach()}


def build_TF_GRU_model(**kw):
    return HybridTransformerGRU(**kw)


# ══════════════════════════════════════════════════════════════════════════
# validation and the records
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_split(model, loader):
    """Token-weighted CE_T and CE_G, the KL, the agreement rate and both selectors' states in use,
    at evaluation (argmax) on one split."""
    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    s = {"ce_T": 0.0, "n_T": 0, "ce_G": 0.0, "n_G": 0, "kl": 0.0, "agree": 0.0, "n_kl": 0}
    used_T = torch.zeros(model.n_states, dtype=torch.bool, device=dev)
    used_G = torch.zeros_like(used_T)
    for batch in loader:
        inputs, targets = model._split(batch, model.mode)
        inputs, targets = inputs.to(dev), targets.to(dev)
        t = model.losses(inputs, targets)
        n_T, n_G = int(t["mask"].sum()), int(t["states_G"].numel())
        s["ce_T"] += t["ce_T"].item() * n_T; s["n_T"] += n_T
        s["ce_G"] += t["ce_G"].item() * n_G; s["n_G"] += n_G
        s["kl"] += t["kl"].item() * n_G; s["agree"] += t["agree"].item() * n_G; s["n_kl"] += n_G
        used_T[t["states_T"][t["mask"]]] = True
        used_G[t["states_G"].reshape(-1)] = True
    if was:
        model.train()
    return {"ce_T": s["ce_T"] / s["n_T"], "ce_G": s["ce_G"] / s["n_G"], "kl": s["kl"] / s["n_kl"],
            "agree": s["agree"] / s["n_kl"], "states_T": int(used_T.sum()), "states_G": int(used_G.sum())}


class HybridRecorder(L.Callback):
    """
    The hybrid's curves and checkpoint selection.  Every `val_every_n_steps` (and at the final step):
    evaluate_split on the validation loader; the checkpoint with the lowest CE_T + CE_G is kept
    (user, 2026-10-06; the KL is left out, being small for a trivial solution too).

    It carries the attributes extraction.analyse_arm reads from training.Recorder, under the same
    names (step_loss, step_val_loss = the transformer's validation CE, best_val = its CE at the
    selected step, val_states_in_use = the transformer's states, step_train_probe_ce, ...), plus the
    hybrid's own: step_ce_gru, step_kl, step_agree (per update) and val_ce_gru, val_kl, val_agree,
    val_states_gru, val_sum (per validation).
    """

    def __init__(self, val_loader, val_every_n_steps=25, keep_best=True, progress_every=0,
                 final_validation=True, train_probe_loader=None, probe_exact_ce=None):
        super().__init__()
        self.val_loader, self.val_every_n_steps = val_loader, max(1, int(val_every_n_steps))
        self.keep_best, self.progress_every, self.final_validation = bool(keep_best), int(progress_every), \
            bool(final_validation)
        self.train_probe_loader, self.probe_exact_ce = train_probe_loader, probe_exact_ce
        self.step_loss, self.step_at, self.step_tau, self.step_grad_norm = [], [], [], []
        self.step_val_loss, self.step_val_at, self.step_states_used, self.states_at = [], [], [], []
        self.step_ce, self.step_scored_tokens, self.epoch_loss, self.epoch_ce = [], [], [], []
        self.val_states_in_use, self.step_lr, self.step_train_probe_ce = [], [], []
        self.step_ce_gru, self.step_kl, self.step_agree = [], [], []
        self.val_ce_gru, self.val_kl, self.val_agree, self.val_states_gru, self.val_sum = [], [], [], [], []
        self.lr_last_used, self.final_validation_done = float("nan"), False
        self.best_val, self.best_sum, self.best_step, self.best_state = float("inf"), float("inf"), -1, None
        self.restored_best, self.model = False, None
        self._sum = self._count = 0.0
        self._ce_sum, self._ce_count = 0.0, 0
        self._t0 = None

    def on_train_start(self, trainer, pl_module):
        self._t0 = time.time()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        sq = [p.grad.detach().float().pow(2).sum() for p in pl_module.parameters() if p.grad is not None]
        self.step_grad_norm.append(float(torch.stack(sq).sum().sqrt()) if sq else 0.0)
        self.lr_last_used = float(optimizer.param_groups[0]["lr"])

    def _validate(self, trainer, pl_module, step):
        v = evaluate_split(pl_module, self.val_loader)
        self.step_val_loss.append(v["ce_T"]); self.step_val_at.append(int(step))
        self.val_ce_gru.append(v["ce_G"]); self.val_kl.append(v["kl"]); self.val_agree.append(v["agree"])
        self.val_states_in_use.append(v["states_T"]); self.val_states_gru.append(v["states_G"])
        self.val_sum.append(v["ce_T"] + v["ce_G"])
        if trainer.optimizers:
            self.step_lr.append(float(trainer.optimizers[0].param_groups[0]["lr"]))
        if self.train_probe_loader is not None:
            self.step_train_probe_ce.append(evaluate_split(pl_module, self.train_probe_loader)["ce_T"])
        if self.keep_best and self.val_sum[-1] < self.best_sum:
            self.best_sum, self.best_val, self.best_step = self.val_sum[-1], v["ce_T"], int(step)
            self.best_state = {k: t.detach().clone() for k, t in pl_module.state_dict().items()}
        pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = float(outputs["loss"].detach().float().cpu().item())
        self._sum += loss; self._count += 1
        n = int(outputs["scored_tokens"])
        ce_sum = float(outputs["ce_sum"].detach().float().cpu().item())
        self.step_scored_tokens.append(n); self.step_ce.append(ce_sum / n)
        self._ce_sum += ce_sum; self._ce_count += n
        self.step_ce_gru.append(float(outputs["ce_G"].float().cpu().item()))
        self.step_kl.append(float(outputs["kl"].float().cpu().item()))
        self.step_agree.append(float(outputs["agree"].float().cpu().item()))
        step = trainer.global_step
        self.step_loss.append(loss); self.step_at.append(step)
        self.step_tau.append(float(getattr(pl_module, "tau", float("nan"))))
        if step % self.val_every_n_steps == 0:
            st = pl_module.causal_head.last_states            # the training batch's, before validation
            self.step_states_used.append(int(torch.unique(st[score_mask(st, pl_module.score_start)]).numel()))
            self.states_at.append(int(step))
            self._validate(trainer, pl_module, step)
        if self.progress_every and step % self.progress_every == 0:
            print(f"    step {step:6d}/{trainer.estimated_stepping_batches} | val CE_T {self.step_val_loss[-1]:.4f} "
                  f"CE_G {self.val_ce_gru[-1]:.4f} | agree {self.val_agree[-1]:.3f} | states T {self.val_states_in_use[-1]} "
                  f"G {self.val_states_gru[-1]} | best {self.best_sum:.4f} @ {self.best_step} | tau {self.step_tau[-1]:.3f} | "
                  f"{(time.time() - self._t0) / 60:.1f} min", file=sys.__stderr__, flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._count:
            self.epoch_loss.append(self._sum / self._count)
        if self._ce_count:
            self.epoch_ce.append(self._ce_sum / self._ce_count)
        self._sum = self._count = 0.0
        self._ce_sum, self._ce_count = 0.0, 0

    def on_train_end(self, trainer, pl_module):
        step = int(trainer.global_step)
        if self.final_validation and (not self.step_val_at or self.step_val_at[-1] != step):
            self._validate(trainer, pl_module, step)
            self.final_validation_done = True


# ══════════════════════════════════════════════════════════════════════════
# one run
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class HybridSettings:
    state_dim: int = 32           # S: the GRU width, the shared codebook's vector size (user, 2026-10-06)
    rollout_len: int = 10         # L (user, 2026-10-06; the notebook's 10)
    rollout_stride: int = 10
    rollout_first: int = 19       # every start has >= 20 tokens of context
    kl_alpha: float = 0.8         # DreamerV2's KL balancing (user, 2026-10-06)
    w_gru: float = 1.0
    w_kl: float = 1.0
    detach_start: bool = False            # inspection switches (plan section 11); defaults = the design
    separate_gru_emission: bool = False
    transformer_gumbel: bool = True


@torch.no_grad()
def hybrid_measures(model, loader, seqs, gen, hs: HybridSettings) -> dict:
    """The hybrid's own test measures: the GRU's CE along the L-step rollouts and along one long
    rollout from t0 = rollout_first to the row's end, each against the exact model on the same
    targets; the agreement rate; the KL; both selectors' states in use."""
    model.eval()
    v = evaluate_split(model, loader)
    exact = exact_per_position_ce(gen, seqs)                      # (N, T) bits, the same rows
    T = exact.shape[1]
    pos = np.asarray(model.rollout_starts(T))[:, None] + np.arange(1, model.rollout_len + 1)[None, :]
    long_L = T - 1 - model.rollout_first
    ce_long, n_long = 0.0, 0
    dev = next(model.parameters()).device
    for batch in loader:
        inputs, targets = model._split(batch, model.mode)
        inputs, targets = inputs.to(dev), targets.to(dev)
        model(inputs)
        ro = model.rollout(inputs, model.causal_head.last_state_onehot, starts=[model.rollout_first], L=long_L)
        p = ro["positions"]
        ce = token_ce_bits(ro["logits"][:, 0], targets[:, p[0]])
        ce_long += ce.sum().item(); n_long += ce.numel()
    long_pos = model.rollout_first + np.arange(1, long_L + 1)
    return {"hybrid_ce_T": v["ce_T"], "hybrid_ce_G": v["ce_G"], "hybrid_kl": v["kl"],
            "hybrid_agreement": v["agree"], "hybrid_states_T": v["states_T"], "hybrid_states_G": v["states_G"],
            "hybrid_rollout_gap_exact": v["ce_G"] - float(exact[:, pos].mean()),
            "hybrid_long_rollout_ce": ce_long / n_long,
            "hybrid_long_rollout_gap_exact": ce_long / n_long - float(exact[:, long_pos].mean())}


def train_hybrid(cfg, hs: HybridSettings, process: str, arm: str, data_dir=None, verbose=True) -> dict:
    """Train and score the hybrid on one arm of one process, in the continuous data mode with the
    continuous regime (run_model's settings, so it compares with results/cont_*)."""
    cfg = run_model.resolve_regime(replace(cfg, data_mode="continuous"), set(cfg.explicit_overrides))
    params, burn_in, tag = run_model.process_params(cfg, process)
    say = print if verbose else (lambda *a, **k: None)

    # --- data: the transformer's continuous view of the one trajectory ------------
    raw, raw_sha1, raw_path, raw_meta = run_model.canonical_draw(cfg, process, data_dir)
    gen = generator(process, 1, cfg.total_tokens - 1, params, burn_in=burn_in)
    spec = run_model.window_settings(cfg, "transformer")
    spans = split_spans(len(raw), cfg.train_fraction, cfg.val_fraction)
    ds, idx, manifest = build_views(raw, spans, spec, arm)
    train_loader, val_loader, test_loader, ana_loader = make_loaders(
        ds, idx, spec, cfg.random_seed, cfg.ana_batch, raw=raw, train_span=spans["train"], arm=arm)
    if arm == "backward":
        gen.reverse()
    V = int(gen.vocab_size)
    K = cfg.num_states_mult * V
    beta = run_model.CONTINUOUS_USAGE_BETA if cfg.usage_beta is None else float(cfg.usage_beta)
    max_steps, total_steps, horizon = run_model.continuous_budget(cfg, "transformer", len(train_loader))
    max_len = max(spec.raw_window - 1, run_model.COMMON_EVAL["raw_window"] - 1, cfg.generation_context)
    _, tau_fn = parse_tau(cfg.tau_transformer)
    train_rows = ds.seqs[idx["train"]]
    probe_rows = train_rows[::max(1, len(train_rows) // len(idx["val"]))]
    probe_loader = tud.DataLoader(SequenceDataset(probe_rows), batch_size=spec.batch_size, shuffle=False)
    probe_mask = score_mask(torch.from_numpy(probe_rows[:, 1:]), spec.score_start).numpy()
    probe_exact = float(exact_per_position_ce(gen, probe_rows)[probe_mask].mean())
    say(f"[hybrid | {tag} | {arm}] V {V}, K {K}, S {hs.state_dim}, transformer d {cfg.transformer_d_model} x "
        f"{cfg.n_layers_transformer} | rollouts L {hs.rollout_len} every {hs.rollout_stride} from {hs.rollout_first}, "
        f"KL alpha {hs.kl_alpha}, w_G {hs.w_gru}, w_KL {hs.w_kl} | {total_steps} updates on a {horizon}-update horizon, "
        f"tau {cfg.tau_transformer}, gumbel {cfg.gumbel_transformer}, beta {beta:.2e} | data {raw_sha1[:12]} | "
        f"theory: k {gen.n_causal}, C {gen.statistical_c:.4f}, H_inf {gen.entropy_rate:.4f}")

    # --- train ------------------------------------------------------------------
    run_model.set_seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed * 1000)
    model = build_TF_GRU_model(
        n_states=K, state_dim=hs.state_dim, tau=tau_fn(0.0), usage_beta=beta,
        rollout_len=hs.rollout_len, rollout_stride=hs.rollout_stride, rollout_first=hs.rollout_first,
        kl_alpha=hs.kl_alpha, w_gru=hs.w_gru, w_kl=hs.w_kl, detach_start=hs.detach_start,
        separate_gru_emission=hs.separate_gru_emission, transformer_gumbel=hs.transformer_gumbel,
        token_size=V, d_model=cfg.transformer_d_model, max_len=max_len, lr=cfg.learning_rate, mode="forward",
        n_layers=cfg.n_layers_transformer, weight_decay=cfg.weight_decay, score_start=spec.score_start,
        usage_scope="scored", learn_input_embedding=cfg.learn_input_embedding, lr_schedule=cfg.lr_schedule,
        lr_min=cfg.lr_min, schedule_horizon=horizon)
    model.gumbel = bool(cfg.gumbel_transformer)
    rec = HybridRecorder(val_loader, cfg.val_every_n_steps, keep_best=cfg.restore_best,
                         progress_every=cfg.progress_every if verbose else 0, final_validation=cfg.final_validation,
                         train_probe_loader=probe_loader, probe_exact_ce=probe_exact)
    trainer = L.Trainer(max_epochs=math.ceil(total_steps / len(train_loader)), max_steps=total_steps,
                        accelerator=cfg.accelerator_transformer, devices="auto", log_every_n_steps=5,
                        callbacks=[rec, TauSchedule(tau_fn, horizon)], gradient_clip_val=cfg.gradient_clip_val,
                        logger=False, enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False)
    t0 = time.time()
    with run_model.quiet(True):
        trainer.fit(model, train_loader)
    train_seconds = time.time() - t0
    if cfg.restore_best and rec.best_state is not None:
        model.load_state_dict(rec.best_state)
        rec.restored_best = True
    rec.model = model
    model.cpu().eval()

    # --- score: the standard analysis (the transformer's states, the GRU's exact machine) ---------
    _, fresh = run_model.draw(cfg, process, arm, len(idx["test"]), spec.raw_window - 1, cfg.data_seed + 1000)
    fresh_loader = tud.DataLoader(SequenceDataset(fresh), batch_size=spec.batch_size, shuffle=False)
    _, seed_tokens = run_model.draw(cfg, process, arm, 1, cfg.generation_context + 50, cfg.random_seed + 2000)
    cseqs, common_id = run_model.common_rows(raw, spans, arm)
    common = {"loader": tud.DataLoader(SequenceDataset(cseqs), batch_size=32, shuffle=False), "seqs": cseqs,
              "score_start": run_model.COMMON_EVAL["score_start"], "manifest_id": common_id}
    res = analyse_arm(model, rec, gen, ds, idx["test"], test_loader, ana_loader,
                      fresh_loader=fresh_loader, seed_tokens=seed_tokens[0], min_pos=cfg.state_min_pos,
                      total_run=cfg.trans_total_run, conv_tol=cfg.conv_tol, full_tol=cfg.full_tol,
                      rng=np.random.default_rng(cfg.random_seed), analysis_start=spec.score_start,
                      generation_context=cfg.generation_context, fresh_seqs=fresh, common=common)
    m = res["metrics"]
    m.update(hybrid_measures(model, test_loader, ds.seqs[idx["test"]], gen, hs))
    res["history"].update({"step_ce_gru": rec.step_ce_gru, "step_kl": rec.step_kl, "step_agree": rec.step_agree,
                           "val_ce_gru": rec.val_ce_gru, "val_kl": rec.val_kl, "val_agree": rec.val_agree,
                           "val_states_gru": rec.val_states_gru, "val_sum": rec.val_sum,
                           "best_sum": float(rec.best_sum)})
    manifest.update({"raw_sha1": raw_sha1, "raw_path": raw_path, "raw_meta": raw_meta,
                     "splits": {k: [int(lo), int(hi)] for k, (lo, hi) in spans.items()},
                     "common_manifest_id": common_id,
                     "train_offsets": list(train_loader.sampler.offsets) if spec.shift else []})
    res.update({
        "arch": "hybrid", "process": process, "tag": tag, "arm": arm, "process_params": params, "burn_in": burn_in,
        "budget_below_k": bool(K < gen.n_causal),
        "cfg": {**asdict(cfg), "token_size": V, "n_states": K, "state_dim": hs.state_dim, "usage_beta_resolved": beta,
                "n_layers": cfg.n_layers_transformer, "accelerator": cfg.accelerator_transformer,
                "tau": cfg.tau_transformer, "max_epochs": cfg.max_epochs_transformer, "gumbel": cfg.gumbel_transformer,
                "d_model": cfg.transformer_d_model, "max_steps_resolved": max_steps, **asdict(hs)},
        "run_config": {**asdict(cfg), **asdict(hs)},
        "hybrid": asdict(hs),
        "model_class": type(model).__name__, "hparams": dict(model.hparams),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": train_seconds, "data_sha1": raw_sha1[:12], "test_indices": idx["test"],
        "schema_version": 2, "data_mode": "continuous",
        "view": {"raw_window": spec.raw_window, "input_len": spec.raw_window - 1, "stride": spec.stride,
                 "eval_stride": spec.stride_for("val"), "batch_size": spec.batch_size, "score_start": spec.score_start,
                 "generation_context": cfg.generation_context, "random_shift": spec.shift, "target_multiplicity": 1,
                 "max_len": max_len},
        "data_manifest": manifest,
        "scoring": {"score_start": spec.score_start, "usage_scope": "scored", "usage_beta": beta,
                    "analysis_start": spec.score_start, "common": dict(run_model.COMMON_EVAL, manifest_id=common_id),
                    "checkpoint_selection": "validation CE_T + CE_G"},
        "regime": {"gumbel": bool(cfg.gumbel_transformer), "tau": cfg.tau_transformer, "lr": cfg.learning_rate,
                   "weight_decay": cfg.weight_decay, "lr_schedule": cfg.lr_schedule, "lr_min": cfg.lr_min,
                   "gradient_clip_val": cfg.gradient_clip_val, "d_model": cfg.transformer_d_model,
                   "update_budget": int(total_steps), "actual_updates": int(trainer.global_step),
                   "schedule_horizon": int(horizon),
                   "tau_at_final_update": float(tau_fn((total_steps - 1) / max(1, horizon - 1))),
                   "lr_at_final_update": float(rec.lr_last_used),
                   "explicit_overrides": sorted(cfg.explicit_overrides)},
        "collapse": {"states_at_first_val": m.get("states_at_first_val"),
                     "states_at_selected": m.get("states_at_selected"), "flagged": m.get("collapse_flagged"),
                     "threshold": m.get("collapse_threshold")},
        "overfitting": {"probe_rows": int(len(probe_rows)), "probe_exact_ce": probe_exact,
                        "min_excess": m.get("train_probe_min_excess"), "flagged": m.get("overfit_flagged")},
    })
    say(f"[hybrid | {tag} | {arm}] {'FULL' if m['full'] else 'not full'} | merged {m['k_used']} -> {m['merged_k']} "
        f"states, discovered {m['discovered']}/{m['true_k']} | CE-exact test {m['gap_exact']:+.4f}, common "
        f"{m['common_gap_exact']:+.4f} | GRU rollout {m['hybrid_rollout_gap_exact']:+.4f}, long "
        f"{m['hybrid_long_rollout_gap_exact']:+.4f} | agree {m['hybrid_agreement']:.3f} | states T "
        f"{m['hybrid_states_T']} G {m['hybrid_states_G']} | exact machine T_err {m.get('exact_T_err', float('nan')):.3f} "
        f"follows {m.get('exact_follows', float('nan')):.3f} | best {m['best_step']}/{trainer.global_step} | "
        f"{train_seconds / 60:.1f} min")
    return res


# ══════════════════════════════════════════════════════════════════════════
# the command line
# ══════════════════════════════════════════════════════════════════════════
def hybrid_collision(run_dir, stem, cfg, hs):
    """{field: (saved, new)} when <stem> in run_dir was run with a different configuration, else None."""
    path = os.path.join(run_dir, f"{stem}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        saved = json.load(f).get("run_config", {})
    mine = json.loads(json.dumps({**asdict(cfg), **asdict(hs)}, default=str))
    diff = {k: (saved[k], mine[k]) for k in saved
            if k in mine and k not in run_model.VOLATILE_FIELDS and saved[k] != mine[k]}
    return diff or None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="hybrid", help="results/<name>/ (smoke: checks/results/smoke/<name>/)")
    ap.add_argument("--process", nargs="+", default=["coin", "flower"], choices=sorted(run_model.PROCESS))
    ap.add_argument("--params", nargs="*", default=[], metavar="[PROCESS.]KEY=VALUE")
    ap.add_argument("--arm", nargs="+", default=["forward", "backward"], choices=["forward", "backward"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-states-mult", type=int, default=None, help="K = this x V (default 4; 8 for the 9-state renewal)")
    ap.add_argument("--max-steps", type=int, default=None, help="update budget (default 30,000)")
    ap.add_argument("--learn-input-embedding", action="store_true", default=None)
    ap.add_argument("--state-dim", type=int, default=None, help="S, the shared codebook's vector size (32)")
    ap.add_argument("--rollout-len", type=int, default=None, help="L, steps per rollout (10)")
    ap.add_argument("--rollout-stride", type=int, default=None, help="positions between rollout starts (10)")
    ap.add_argument("--rollout-first", type=int, default=None,
                    help="the first rollout start (19: every start has >= 20 tokens of context)")
    ap.add_argument("--transformer-score-start", type=int, default=None,
                    help="the transformer's first scored output (20)")
    ap.add_argument("--transformer-d-model", type=int, default=None, help="the transformer's width (16)")
    ap.add_argument("--usage-beta", type=float, default=None,
                    help="usage-penalty coefficient on H_T + H_G (continuous default 1/9600)")
    ap.add_argument("--kl-alpha", type=float, default=None, help="KL balancing alpha (0.8)")
    ap.add_argument("--w-gru", type=float, default=None, help="weight of the GRU's CE (1)")
    ap.add_argument("--w-kl", type=float, default=None, help="weight of the state KL (1)")
    ap.add_argument("--detach-start", action="store_true", default=None,
                    help="inspection: no gradient from the GRU into the transformer through the start state")
    ap.add_argument("--separate-gru-emission", action="store_true", default=None,
                    help="inspection: the GRU predicts through its own emission (the codebook stays shared)")
    ap.add_argument("--no-transformer-gumbel", dest="transformer_gumbel", action="store_false", default=None,
                    help="inspection: the transformer's selector trains with argmax")
    ap.add_argument("--accelerator", default=None, choices=["auto", "cpu", "mps"],
                    help="device for the run (default auto: MPS on a Mac, as the transformer)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(run_model.HERE, "results"))
    ap.add_argument("--smoke", action="store_true", help="40 updates on 20,000 tokens -> checks/results/smoke/<name>/")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    root = os.path.join(run_model.CHECKS_RESULTS, "smoke") if args.smoke else args.out
    out = os.path.join(root, args.name)
    run_dir, fig_dir = run_model.mkdir(os.path.join(out, "runs")), run_model.mkdir(os.path.join(out, "figures"))
    if args.plots_only:
        run_model.redraw(run_dir, fig_dir)
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    import logging
    for name in ("lightning", "lightning.pytorch", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)

    cfg = replace(run_model.RunConfig(), random_seed=args.seed, data_mode="continuous",
                  process_params=run_model.parse_params(args.params, args.process))
    explicit = {"data_mode"}
    for flag, field_ in (("num_states_mult", "num_states_mult"), ("max_steps", "max_steps"),
                         ("learn_input_embedding", "learn_input_embedding"),
                         ("transformer_score_start", "transformer_score_start"),
                         ("transformer_d_model", "transformer_d_model"), ("usage_beta", "usage_beta")):
        if getattr(args, flag) is not None:
            cfg = replace(cfg, **{field_: getattr(args, flag)})
            explicit.add(field_)
    if args.accelerator is not None:
        cfg = replace(cfg, accelerator_transformer=args.accelerator)
    cfg = run_model.resolve_regime(replace(cfg, explicit_overrides=tuple(sorted(explicit))), explicit)
    hs = HybridSettings(**{k: getattr(args, k) for k in ("state_dim", "rollout_len", "rollout_stride", "rollout_first",
                                                          "kl_alpha",
                                                          "w_gru", "w_kl", "detach_start", "separate_gru_emission",
                                                          "transformer_gumbel") if getattr(args, k) is not None})
    if args.smoke:
        cfg = replace(cfg, total_tokens=20_000, max_steps=40, trans_total_run=200)
    stems = [f"hybrid_{p}_{a}_s{cfg.random_seed}" for p in args.process for a in args.arm]
    clashes = {s: d for s in stems if (d := hybrid_collision(run_dir, s, cfg, hs))}
    if clashes:
        ap.error(f"{run_dir} already holds runs with a different configuration: {clashes}; use another --name")
    for process in args.process:
        for arm in args.arm:
            res = train_hybrid(cfg, hs, process, arm, data_dir=os.path.join(out, "data"))
            run_model.save_run(res, run_dir, f"hybrid_{process}_{arm}_s{cfg.random_seed}")
            run_model.cleanup()
    if not args.no_plots:
        run_model.redraw(run_dir, fig_dir)


if __name__ == "__main__":
    sys.exit(main())
