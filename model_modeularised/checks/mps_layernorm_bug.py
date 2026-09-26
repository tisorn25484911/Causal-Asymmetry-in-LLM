"""
The PyTorch-MPS LayerNorm bug, and what it does to the transformer.

PyTorch 2.12.1 on MPS returns WRONG weight/bias gradients from nn.LayerNorm's
backward when the LayerNorm's INPUT does not require grad.  In the transformer
that is exactly one module, ln_attn.0: its input is the token projection plus
the positional encoding, both buffers.  Every later LayerNorm's input depends
on parameters, so its gradients are right.

    python checks/mps_layernorm_bug.py repro
        seconds.  1. a bare nn.LayerNorm, MPS vs CPU, input with / without requires_grad;
                  2. the transformer as run_model builds it: one batch's gradients,
                     MPS vs CPU, as-is and with the workaround.

    python checks/mps_layernorm_bug.py train --process renewal --arm forward \\
        --num-states-mult 4 --variants fix cpu > checks/results/diagnostics/<name>.log
        a full run_model.run_arm per variant (the same training and analysis as a
        real run), saved with its end-of-training weights to
        checks/results/diagnostics/mps_layernorm_runs/; nothing is written to results/.

    python checks/mps_layernorm_bug.py saved results/renewal_K8/runs/transformer_*.pkl
        the same drift summary for runs already saved.

The workaround (verified by `repro`): make the positional encoding's output -- the
one LayerNorm input that does not require grad -- require it.  Forward values are
unchanged; only which MPS backward kernel runs changes.  It is applied here by
patching PositionalEncoding.forward in THIS process; Transformer.py is untouched.

Variants for `train`:   asis   MPS, as run_model runs it
                        fix    MPS + the workaround
                        cpu    CPU (no MPS kernels; bit-identical to the pipeline on CPU)
`--gumbel` trains any of them with the Gumbel-ST head (RunConfig.gumbel_transformer).
"""
import argparse
import copy
import glob
import os
import pickle
import sys
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import run_model                                                     # noqa: E402
import training                                                      # noqa: E402
import Transformer                                                   # noqa: E402
from decoder_base import cross_ent_onehot                            # noqa: E402
from process_generator import SequenceDataset, split_loader          # noqa: E402

_PE_FORWARD = Transformer.PositionalEncoding.forward
RUNS = os.path.join(HERE, "results", "diagnostics", "mps_layernorm_runs")      # `train` saves its runs here


def workaround(on: bool):
    """Patch (or unpatch) PositionalEncoding.forward so its output requires grad."""
    def forward(self, x):
        out = _PE_FORWARD(self, x)
        if torch.is_grad_enabled() and not out.requires_grad:
            out.requires_grad_(True)          # a leaf: built from buffers only
        return out
    Transformer.PositionalEncoding.forward = forward if on else _PE_FORWARD


# ══════════════════════════════════════════════════════════════════════════
# repro
# ══════════════════════════════════════════════════════════════════════════
def bare_layernorm():
    torch.manual_seed(0)
    x, r = torch.randn(8, 50, 16), torch.randn(8, 50, 16)
    ln = nn.LayerNorm(16)
    with torch.no_grad():
        ln.weight.normal_(1.0, 0.1)
        ln.bias.normal_(0.0, 0.1)
    print("1. bare nn.LayerNorm(16), loss = sum(LN(x) * r), gradients MPS - CPU:")
    for req in (False, True):
        g = {}
        for dev in ("cpu", "mps"):
            m = copy.deepcopy(ln).to(dev)
            xi = x.to(dev).clone().requires_grad_(req)
            (m(xi) * r.to(dev)).sum().backward()
            g[dev] = (m.weight.grad.cpu(), m.bias.grad.cpu())
        dw = float((g["mps"][0] - g["cpu"][0]).abs().max())
        db = float((g["mps"][1] - g["cpu"][1]).abs().max())
        print(f"   input requires_grad={req!s:5s}: max|dW| {dw:.2e}   max|dB| {db:.2e}"
              f"   (|W grad| ~ {float(g['cpu'][0].abs().max()):.1f})")


def one_batch_grads(model, batch, dev):
    m = copy.deepcopy(model).to(dev).train()
    m.tau = 1.0
    inputs, targets = m._split(tuple(t.to(dev) for t in batch), m.mode)
    loss, _ = cross_ent_onehot(m(inputs), targets)
    (loss + m._extra_loss()).backward()
    return {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters() if p.grad is not None}


def model_grads(process="coin", arm="forward"):
    cfg = run_model.RunConfig()
    gen, seqs = run_model.draw(cfg, process, arm, 64, cfg.seq_len, cfg.data_seed)
    V = int(gen.vocab_size)
    a = run_model.arch_settings(cfg, "transformer", V)
    ds = SequenceDataset(seqs)
    tr, _ = split_loader(ds, cfg.batch_size, cfg.test_ratio, seed=0)
    batch = next(iter(tr))
    torch.manual_seed(0)
    model = training.ARCHITECTURES["transformer"](
        "discrete", token_size=V, d_model=cfg.d_model, max_len=ds.seq_len, lr=cfg.learning_rate,
        mode="forward", n_layers=a["n_layers"], weight_decay=cfg.weight_decay,
        n_states=cfg.num_states_mult * V, state_dim=a["state_dim"], tau=1.0, usage_beta=1e-4)
    print(f"\n2. the transformer as run_model builds it ({process} {arm}, one batch), gradients MPS - CPU,"
          f"\n   relative to the CPU gradient's max|.|:")
    cpu = one_batch_grads(model, batch, "cpu")
    # the key biases' TRUE gradient is 0 (softmax ignores a shift shared by every key):
    # both devices return rounding noise there, so a relative error means nothing.
    zero = [n for n in cpu if n.endswith("wk.bias")]
    print(f"   (key biases, true gradient 0: CPU max|grad| {max(float(cpu[n].abs().max()) for n in zero):.1e}"
          f" -- rounding noise, left out below)")
    for on in (False, True):
        workaround(on)
        mps = one_batch_grads(model, batch, "mps")
        err = {n: float((mps[n] - cpu[n]).abs().max() / (cpu[n].abs().max() + 1e-12)) for n in cpu if n not in zero}
        worst = sorted(err, key=err.get, reverse=True)[:3]
        rest = max(v for n, v in err.items() if not n.startswith("ln_attn.0"))
        print(f"   {'with the workaround' if on else 'as run_model runs it':21s}: "
              + "  ".join(f"{n} {err[n]:.1e}" for n in worst) + f"   | every parameter but ln_attn.0 <= {rest:.1e}")
    workaround(False)


# ══════════════════════════════════════════════════════════════════════════
# the drift, from a result dict (+ the end-of-training weights when we have them)
# ══════════════════════════════════════════════════════════════════════════
def drift_summary(res, end_state=None, label=""):
    h, m = res["history"], res["metrics"]
    g = np.asarray(h["step_grad_norm"])
    sd = res["state_dict"]

    def ln0(state):
        w, b = np.asarray(state["ln_attn.0.weight"], float), np.asarray(state["ln_attn.0.bias"], float)
        return f"gain {w.mean():+7.2f} (max|.| {np.abs(w).max():6.2f}), bias max|.| {np.abs(b).max():6.2f}"

    others = [k for k in sd if k.endswith(".weight") and k.startswith(("ln_attn", "ln_ffn")) and k != "ln_attn.0.weight"]
    other_gain = max(float(np.abs(np.asarray(sd[k], float) - 1).max()) for k in others)
    print(f"   {label}val CE best {h['best_val']:.4f} @ {h['best_step']} / end {h['val_loss'][-1]:.4f}"
          f" | states at end {h['states_used'][-1] if h['states_used'] else '?'}"
          f" | |g| median {np.median(g):.3g}, max {g.max():.3g}")
    print(f"   {' ' * len(label)}ln_attn.0 at best: {ln0(sd)}"
          + (f"\n   {' ' * len(label)}ln_attn.0 at end:  {ln0(end_state)}" if end_state is not None else "")
          + f"\n   {' ' * len(label)}the other 7 LayerNorms at best: max|gain - 1| {other_gain:.2f}")
    print(f"   {' ' * len(label)}result: {'FULL' if m['full'] else 'not full'}, discovered {m['discovered']}/{m['true_k']}, "
          f"used {m['k_used']}, S-C {m['S_minus_C']:+.3f}, T_err {m['transition_max_err']:.3f}, "
          f"gap {m['gap_exact']:+.4f}, det {m['determinism']:.3f}", flush=True)


def train(args):
    # Recorder restores the best checkpoint AFTER fit; keep the end-of-training weights too.
    def on_train_end(self, trainer, pl_module):
        self.end_state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}
    training.Recorder.on_train_end = on_train_end
    captured = {}

    def spy(*a, **k):
        captured["rec"] = rec = training.train_model(*a, **k)
        return rec
    run_model.train_model = spy

    base = run_model.RunConfig(random_seed=args.seed, num_states_mult=args.num_states_mult,
                               progress_every=2500, gumbel_transformer=args.gumbel)
    tag = "+gumbel" if args.gumbel else ""
    for v in args.variants:
        workaround(v == "fix")
        cfg = replace(base, accelerator_transformer="cpu") if v == "cpu" else base
        print(f"\n######## {v}{tag}: transformer {args.process} {args.arm}, K = {args.num_states_mult}V, "
              f"{cfg.tau_transformer}, {cfg.max_epochs_transformer} epochs, seed {args.seed}", flush=True)
        t0 = time.time()
        res = run_model.run_arm(cfg, "transformer", args.process, args.arm)
        res["end_state"] = getattr(captured["rec"], "end_state", None)
        drift_summary(res, res["end_state"], label=f"[{v}{tag}] ")
        os.makedirs(RUNS, exist_ok=True)
        out = os.path.join(RUNS, f"{v}{tag}_{args.process}_{args.arm}_s{args.seed}.pkl")
        with open(out, "wb") as f:
            pickle.dump(res, f)
        print(f"   saved {os.path.relpath(out)}", flush=True)
        print(f"   {(time.time() - t0) / 60:.1f} min", flush=True)
    workaround(False)


def saved(paths):
    for p in sorted(sum((glob.glob(x) for x in paths), [])):
        with open(p, "rb") as f:
            res = pickle.load(f)
        print(f"\n{p}  ({res['cfg'].get('tau_transformer', res['cfg']['tau'])}, "
              f"{res['cfg'].get('max_epochs_transformer', res['cfg']['max_epochs'])} epochs)")
        drift_summary(res, res.get("end_state"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("repro")
    t = sub.add_parser("train")
    t.add_argument("--process", default="renewal", choices=sorted(run_model.PROCESS))
    t.add_argument("--arm", default="forward", choices=["forward", "backward"])
    t.add_argument("--num-states-mult", type=int, default=2)
    t.add_argument("--variants", nargs="+", default=["fix", "cpu"], choices=["asis", "fix", "cpu"])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--gumbel", action="store_true",
                   help="train the transformer's head with Gumbel-ST (as the GRUs); evaluation stays argmax")
    s = sub.add_parser("saved")
    s.add_argument("paths", nargs="+")
    a = ap.parse_args()
    import logging
    for name in ("lightning", "lightning.pytorch", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)
    if a.cmd == "repro":
        if not torch.backends.mps.is_available():
            print("MPS is not available here; nothing to reproduce"); return 0
        print(f"torch {torch.__version__}")
        bare_layernorm()
        model_grads()
    elif a.cmd == "train":
        torch.set_num_threads(a.threads)
        train(a)
    else:
        saved(a.paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
