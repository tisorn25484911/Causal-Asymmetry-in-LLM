"""
Cross-check 1: the modularised transformer IS the pipeline's.

Trains Experimental_pipeline's DiscreteCausalDecoder (models.py + training.py +
schedules.py) and model_modeularised's (Transformer.py + causal_matrix.py +
training.py + tau_schedue.py) on the same sequences, same split, same seeds, on
CPU, and compares every training loss, every validation CE, every final weight,
the logits and the state assignments.  Both modes (tril / triu).

    python checks/transformer_bit_identical.py            (from model_modeularised/; writes nothing)

The two sides run in separate processes: both codebases have a module called
`training`, and one Python process cannot import both under that name.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODULAR = os.path.dirname(HERE)
PIPELINE = os.path.join(os.path.dirname(MODULAR), "Experimental_pipeline")

SETTINGS = dict(d_model=16, n_layers=4, max_epochs=5, lr=1e-3, weight_decay=0.01, accelerator="cpu",
                val_every_n_steps=5, n_states=6, state_dim=3, tau="geom:5:0.5", usage_beta=1 / (32 * 100),
                restore_best=True)
# the pipeline's head parameters, and where the modular head keeps them
RENAME = {"state_head.weight": "causal_head.state_head.weight", "state_head.bias": "causal_head.state_head.bias",
          "state_matrix": "causal_head.state_matrix", "emission.weight": "causal_head.emission.weight",
          "emission.bias": "causal_head.emission.bias"}


def one_side(side, mode, data, out):
    """Train one side and save what the comparison needs."""
    import logging
    import torch
    for name in ("lightning", "lightning.pytorch", "lightning.fabric"):
        logging.getLogger(name).setLevel(logging.ERROR)
    sys.path.insert(0, PIPELINE if side == "pipeline" else MODULAR)
    import training
    if side == "pipeline":
        from processes import SequenceDataset
        from training import split_loader
    else:
        from process_generator import SequenceDataset, split_loader
    seqs = np.load(data)
    ds = SequenceDataset(seqs)
    tr, te = split_loader(ds, 32, 0.2, seed=0)
    training.set_seed(0)
    torch.manual_seed(0)
    kw = dict(SETTINGS, max_len=ds.seq_len, mode=mode)
    with training.quiet(True):
        if side == "pipeline":
            rec = training.train_model(tr, "discrete", val_loader=te, num_token=3, **kw)
        else:
            rec = training.train_model(tr, "transformer", "discrete", val_loader=te, token_size=3, **kw)
    m = rec.model.eval()
    sd = {(RENAME.get(k, k) if side == "pipeline" else k): v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        logits = m(torch.as_tensor(seqs[:8, :-1]))
    torch.save(dict(step_loss=rec.step_loss, val=rec.step_val_loss, best_step=rec.best_step, sd=sd,
                    logits=logits, states=m.last_states.clone(), test_ce=training.eval_ce(m, te)[0]), out)


def main():
    import torch
    sys.path.insert(0, MODULAR)
    from process_generator import generator
    tmp = tempfile.mkdtemp(prefix="bit_identical_")
    data = os.path.join(tmp, "seqs.npy")
    np.save(data, generator("coin", 160, 100, {"p": 0.7, "q": 0.8}, burn_in=200)
            .generate(np.random.default_rng(1)))
    ok = True
    for mode in ("forward", "backward"):
        outs = {}
        for side in ("pipeline", "modular"):
            outs[side] = os.path.join(tmp, f"{side}_{mode}.pt")
            subprocess.run([sys.executable, __file__, "--side", side, mode, data, outs[side]], check=True)
        a, b = (torch.load(outs[s]) for s in ("pipeline", "modular"))
        checks = {
            "training losses": a["step_loss"] == b["step_loss"],
            "validation CEs": a["val"] == b["val"],
            "restored step": a["best_step"] == b["best_step"],
            "parameter names": set(a["sd"]) == set(b["sd"]),
            "every weight": set(a["sd"]) == set(b["sd"]) and all(torch.equal(a["sd"][k], b["sd"][k]) for k in a["sd"]),
            "logits": torch.equal(a["logits"], b["logits"]),
            "state assignments": torch.equal(a["states"], b["states"]),
            "test CE": a["test_ce"] == b["test_ce"],
        }
        ok &= all(checks.values())
        print(f"{mode:8s} ({len(a['step_loss'])} steps): " +
              ", ".join(f"{k} {'identical' if v else 'DIFFER'}" for k, v in checks.items()))
    print("BIT-IDENTICAL" if ok else "NOT IDENTICAL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--side":
        one_side(*sys.argv[2:6])
    else:
        sys.exit(main())
