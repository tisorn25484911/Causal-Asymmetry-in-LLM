"""Controlled renewal-9 experiments for hybrid_model_test.ipynb.

Read the notebook's reference class definitions without executing its training cells.
Keep the user's hybrid_transformer_GRU_v2.py untouched. CLI: --trial <name>.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse
import ast
import hashlib
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import HMM_processes as hmm

LN2 = math.log(2)
REFERENCE_NAMES = {"HybridConfig", "StateSelection", "CausalAttentionBlock", "HistoryEncoder",
                   "SharedCausalHead", "HybridTransformerGRU", "temperature", "prediction_loss"}
_notebook = json.loads((ROOT / "hybrid_model_test.ipynb").read_text())
_definitions = []
for _cell in _notebook["cells"]:
    if _cell["cell_type"] == "code":
        for _node in ast.parse("".join(_cell["source"])).body:
            if isinstance(_node, (ast.ClassDef, ast.FunctionDef)) and _node.name in REFERENCE_NAMES:
                _definitions.append(_node)
assert len(_definitions) == len(REFERENCE_NAMES)
_module = ast.Module(body=_definitions, type_ignores=[])
exec(compile(_module, str(ROOT / "hybrid_model_test.ipynb"), "exec"), globals())
REFERENCE_HASH = hashlib.sha256(ast.dump(_module).encode()).hexdigest()

PARAMS = {"F": [.04, .08, .12, .16, .20, .16, .12, .08, .04]}
SPEC = hmm.PROCESS["renewal"]
T = SPEC["transition"](PARAMS)
E = SPEC["state_map"](PARAMS)
M = hmm.joint_map(T, E)
PI = hmm.stationary_dist(T)
EMISSION = hmm.emission_dist(T, E)
assert hmm.epsilon_machine(T, E)["n_states"] == 9

TRIALS = {
    "baseline": dict(label="Hybrid baseline", kind="hybrid", rollout_len=10, batch_size=128,
                     learned_input=False, tau_end=.1),
    "long_rollout": dict(label="Hybrid: rollout 40", kind="hybrid", rollout_len=40, batch_size=32,
                         learned_input=False, tau_end=.1),
    "learned_input": dict(label="Hybrid: learned input", kind="hybrid", rollout_len=10, batch_size=128,
                          learned_input=True, tau_end=.1),
    "warmer_tau": dict(label="Hybrid: final tau 0.5", kind="hybrid", rollout_len=10, batch_size=128,
                       learned_input=False, tau_end=.5),
    "feedback_gru": dict(label="Existing feedback GRU", kind="gru", rollout_len=10, batch_size=128,
                         learned_input=False, tau_end=.1),
}


class Sampler:
    """Batch version of tokens_from_map, with optional true states for analysis only."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.cdf = M.reshape(9, -1).cumsum(-1)
        self.cdf /= self.cdf[:, -1:]

    def draw(self, batch_size, length, states=False):
        current = self.rng.choice(9, batch_size, p=PI)
        out = np.empty((batch_size, length), dtype=np.int64)
        st = np.empty((batch_size, length + 1), dtype=np.int64)
        st[:, 0] = current
        for j in range(length):
            u = self.rng.random(batch_size)
            flat = (self.cdf[current] <= u[:, None]).sum(-1)
            current = flat // 2
            out[:, j] = flat % 2
            st[:, j + 1] = current
        tensor = torch.from_numpy(out)
        return (tensor, st) if states else tensor


def settings(name, seed=17, steps=30000):
    return dict(TRIALS[name], name=name, seed=int(seed), steps=int(steps), history_len=20,
                n_states=16, state_dim=32, d_model=16, n_layers=4, lr=.001, tau_start=2.25,
                eval_every=500, params=PARAMS, reference_hash=REFERENCE_HASH,
                train_seed=5101, validation_seed=6202, alignment_seed=7303, test_seed=8404,
                long_test_seed=9505, version=1)


def result_dir(config):
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    return ROOT / "notebook_results" / "renewal9_trials" / f"{config['name']}_s{config['seed']}_{key}"


def make_model(config):
    torch.manual_seed(config["seed"])
    if config["kind"] == "gru":
        from GRU import DiscreteFeedbackGRU
        model = DiscreteFeedbackGRU(token_size=2, d_model=32, n_states=16, state_dim=32,
                                    n_layers=1, usage_beta=0., weight_decay=0., tau=2.25)
        model.gumbel = True
        return model
    cfg = HybridConfig(vocab_size=2, n_states=16, state_dim=32, d_model=16, n_layers=4,
                       history_len=20, rollout_len=config["rollout_len"])
    model = HybridTransformerGRU(cfg)
    if config["learned_input"]:
        initial = model.encoder.token_vectors.detach().clone()
        del model.encoder.token_vectors
        model.encoder.register_parameter("token_vectors", nn.Parameter(initial))
    return model


def raw_machine(model, kind):
    if kind == "gru":
        machine = model.machine()
        delta, probs = machine["next_state"], machine["emission_probs"]
        logits = model.causal_head.emission(model.state_matrix)
    else:
        book = model.causal_head.state_matrix
        probs = model.causal_head.emission(book).softmax(-1).detach().numpy()
        rows = []
        for x in range(2):
            token = F.one_hot(torch.full((len(book),), x, dtype=torch.long), 2).float()
            rows.append(model.gru_selector(model.gru_cell(token, book)).argmax(-1).detach().numpy())
        delta = np.stack(rows, axis=-1)
        logits = model.causal_head.emission(book)
    joint = np.zeros((len(delta), len(delta), 2))
    for s in range(len(delta)):
        for x in range(2):
            joint[s, delta[s, x], x] = probs[s, x]
    return dict(delta=delta, emission=probs, M=joint,
                log_probs=logits.log_softmax(-1).detach().numpy())


@torch.no_grad()
def initial_ids(model, tokens, kind, tau):
    if kind == "gru":
        model.tau = tau
        model(tokens)
        return model.causal_head.last_states[:, -1].cpu().numpy()
    return model.initial_state(tokens, tau).ids.cpu().numpy()


@torch.no_grad()
def evaluate(model, config, tokens, true_states, l2=True, keep_arrays=False):
    """Same target rows for all trials, with any continuation length; no resets within a row."""
    was = model.training
    model.eval()
    H = config["history_len"]
    length = tokens.shape[1] - H
    raw = raw_machine(model, config["kind"])
    losses, state_rows, starts, distances = [], [], [], []
    for batch_start in range(0, len(tokens), 256):
        batch = tokens[batch_start:batch_start+256]
        state = initial_ids(model, batch[:, :H], config["kind"], config["tau_eval"])
        starts.append(state.copy())
        y = batch[:, H:].numpy()
        ids, nll, l2_rows = [], [], []
        if l2 and config["kind"] == "hybrid":
            qT = model.transformer_states(batch, config["tau_eval"]).probs[:, H:]
            # Probe at a common reference temperature too; argmax choices do not depend on tau.
            raw_tf = model.transformer_selector(model.encoder(batch))[:, H:]
            qT_fixed = raw_tf.softmax(-1)
        for k in range(length):
            ids.append(state.copy())
            nll.append(-raw["log_probs"][state, y[:, k]] / LN2)
            if l2 and config["kind"] == "hybrid":
                rep = model.causal_head.state_matrix[torch.from_numpy(state)]
                symbol = F.one_hot(torch.from_numpy(y[:, k]), 2).float()
                z = model.gru_selector(model.gru_cell(symbol, rep))
                qG = (z / config["tau_eval"]).softmax(-1)
                d = torch.linalg.vector_norm(qT[:, k] - qG, dim=-1)
                fixed = torch.linalg.vector_norm(qT_fixed[:, k] - z.softmax(-1), dim=-1)
                l2_rows.append(torch.stack([d, fixed], dim=-1).numpy())
            state = raw["delta"][state, y[:, k]]
        losses.append(np.stack(nll, axis=1))
        state_rows.append(np.stack(ids, axis=1))
        if l2_rows:
            distances.append(np.stack(l2_rows, axis=1))
    state_rows = np.concatenate(state_rows)
    losses = np.concatenate(losses)
    y = tokens[:, H:].numpy()
    ages = true_states[:, H:H+length]
    oracle = -np.log2(EMISSION[ages, y])
    result = dict(ce_bits=float(losses.mean()), exact_ce_bits=float(oracle.mean()),
                  excess_ce_bits=float((losses-oracle).mean()),
                  used_codes=int(np.unique(state_rows).size))
    if distances:
        d = np.concatenate(distances)
        result.update(mean_l2=float(d[:, :, 0].mean()), fixed_tau1_l2=float(d[:, :, 1].mean()))
    else:
        result.update(mean_l2=None, fixed_tau1_l2=None)
    if keep_arrays:
        result.update(states=state_rows, starts=np.concatenate(starts), losses=losses,
                      exact_losses=oracle, ages=ages)
    model.train(was)
    return result


def train_trial(config, force=False):
    folder = result_dir(config)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "completed.pt"
    if checkpoint.exists() and not force:
        return load_trial(config)
    torch.set_num_threads(1)
    model = make_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.)
    sampler = Sampler(config["train_seed"])
    # Always evaluate common 20-history + 10-future windows, including the long-rollout trial.
    validation, val_states = Sampler(config["validation_seed"]).draw(512, 30, states=True)
    training, records = [], []
    start = time.perf_counter()
    for step in range(config["steps"]):
        model.train()
        tau = temperature(step, config["steps"], config["tau_start"], config["tau_end"])
        x = sampler.draw(config["batch_size"], 20 + config["rollout_len"])
        optimizer.zero_grad(set_to_none=True)
        if config["kind"] == "hybrid":
            logits = model(x[:, :20], x[:, 20:], tau)
        else:
            model.tau = tau
            logits = model(x[:, :-1])[:, 19:]
        loss = prediction_loss(logits, x[:, 20:])
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at {step+1}")
        loss.backward()
        optimizer.step()
        training.append(dict(update=step+1, ce_bits=float(loss.detach()), tau=tau))
        if (step+1) % config["eval_every"] == 0 or step+1 == config["steps"]:
            ev = evaluate(model, dict(config, tau_eval=tau), validation, val_states)
            ev.update(update=step+1, tau=tau, seconds=time.perf_counter()-start)
            records.append(ev)
            (folder / "progress.json").write_text(json.dumps(ev, indent=2))
            pd.DataFrame(records).to_csv(folder / "validation.csv", index=False)
            if (step+1) % 5000 == 0 or step+1 == config["steps"]:
                print(config["name"], step+1, f"val gap {ev['excess_ce_bits']:.5f}", flush=True)
    saved = dict(config=config, model_state=model.state_dict(), training=training, validation=records,
                 seconds=time.perf_counter()-start, updates=config["steps"])
    torch.save(saved, checkpoint)
    pd.DataFrame(training).to_csv(folder / "training.csv", index=False)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    return model, saved


def load_trial(config):
    saved = torch.load(result_dir(config) / "completed.pt", map_location="cpu", weights_only=True)
    assert saved["config"] == config
    model = make_model(config)
    model.load_state_dict(saved["model_state"])
    return model, saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", choices=TRIALS, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--steps", type=int, default=30000)
    args = ap.parse_args()
    train_trial(settings(args.trial, args.seed, args.steps))
