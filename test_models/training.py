"""
Shared training code for every model in this folder.

    Settings            every setting of a run (defaults: section 16 of hybrid_model_test.ipynb, with its 16.1 changes)
    settings_for        a model's settings: Settings defaults <- the model file's DEFAULTS <- your changes
    available_models    the model files in this folder (model_<name>.py); a new file is found automatically
    make_model          build a model with its starting random weights (seeded)
    temperature         the temperature schedule: multiplied by the same factor at every update
    make_optimizer      AdamW; the Transformer's scoring layer in its own group (for selector_lr_mult)
    evaluate            the held-out measures recorded during training (excess CE of every predictor, states in use, ...)
    train               train a model and save the run; a finished run with the same settings and code is reloaded
    extend              train a finished run for more updates at the final temperature
    train_until_settled section 16's rule: extend by 25,000 updates while the main measure still improves, up to 100,000
    load_run, history   read a saved run back

What a model file provides (model_2.py is the reference layout)
    NAME, DESCRIPTION, DEFAULTS, MAIN_PREDICTOR   see model_2.py
    class Model(nn.Module), built from a Settings object, with
        loss(window, tau, weight)  -> (loss tensor, dict of loss terms (tensors), state ids for after_update or None)
        predict(tokens, tau)       -> {"predictors": {name: {"first", "logits", "states"}}, ...}  (no noise)
            first   the index of the first token predicted
            logits  (N, n, V): logits[:, j] predicts token first + j
            states  (N, n) or (N, n + 1): states[:, j] is the state that makes prediction j; one extra column is the
                    state after the last token (a model's own states are kept at every position)
          optional entries: "handoff_scores" (N, K), the Transformer's scores at the last history position;
                            "transformer_ids" (N, T), the Transformer's state at every position
        optional: after_update(ids), measures(tokens, tau, prediction), step_rules(), handoff_states(history, tau)

Predictor names used by every model (the notebook's walkthrough explains them):
    transformer               at every position the Transformer reads all real tokens so far and picks a state afresh
    gru_from_handoff          the GRU steps from the Transformer's state after the history, reading the real future tokens
    next_state_from_handoff   the same with the next-state network making each step (models 3, 4, A)
    gru_from_start            the feedback GRU alone: it reads every token of the window from a blank start
"""
import hashlib
import importlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import process as process_module                                                   # noqa: E402

LN2 = math.log(2)
RUNS = Path(os.environ.get("TEST_MODELS_RUNS") or HERE / "runs")   # where runs are saved
FREQ_DECAY = 0.99                     # running count of state choices (models 5a, 5b, C)
EXTEND_STEP, EXTEND_MAX, EXTEND_TOL = 25_000, 100_000, 5e-4
ORIGINAL_SETUP = dict(encoder_norm=False, weight_decay=0.0, grad_clip=0.0)   # section 13's settings ("1 (original setup)")


# ------------------------------------------------------------------------------------------------ settings
@dataclass
class Settings:
    """Every setting of a run.  Defaults: section 16's setup, including the three changes of 16.1."""
    model: str = "2"
    process: str = "renewal"
    params: dict = field(default_factory=lambda: {"F": [1.0 / 15] * 15})
    vocab_size: int = 0                  # filled from the process
    history_len: int = 40                # H: tokens read before the GRU starts
    rollout_len: int = 20                # L: future tokens predicted after the history
    context: int = 19                    # the Transformer's own predictions are scored at every position t >= context
    n_states: int = 32                   # K: rows of the state table
    state_dim: int = 32                  # S: length of a state vector
    d_model: int = 16                    # width of the Transformer (of the GRU, for the GRU alone)
    n_layers: int = 4                    # Transformer layers
    next_hidden: int = 64                # hidden width of the next-state network (models 3, 4, A)
    encoder_norm: bool = True            # LayerNorm on the Transformer's output (16.1)
    gumbel: bool = True                  # random noise in every state choice during training
    weight: float = 0.0                  # weight of the model's extra loss term (models that have one)
    steps: int = 50_000
    batch_size: int = 128
    lr: float = 1e-3
    selector_lr_mult: float = 1.0        # learning-rate multiplier for the Transformer's scoring layer
    lr_warmup: int = 0                   # raise the learning rate linearly over this many updates
    weight_decay: float = 5e-4           # AdamW (0 = Adam)
    grad_clip: float = 1.0               # clip the gradient's length to this value (0 = off)
    tau_start: float = 2.5
    tau_end: float = 0.5
    schedule_steps: int | None = None    # length of the temperature schedule (None: steps)
    emission_start: str = "default"      # "data_frequency": the output layer starts at the data's token frequency
    seed: int = 0
    model_seed_base: int = 17            # starting weights: torch seed model_seed_base + seed
    train_seed_base: int = 1101          # training windows: seed train_seed_base + seed
    eval_every: int = 500
    validation_rows: int = 512
    validation_seed: int = 2202
    threads: int = 1

    @property
    def total_len(self):
        return self.history_len + self.rollout_len

    @property
    def model_seed(self):
        return self.model_seed_base + self.seed

    @property
    def train_seed(self):
        return self.train_seed_base + self.seed

    def to_dict(self):
        return asdict(self)

    def process_object(self):
        return get_process(self.process, json.dumps(self.params, sort_keys=True))


@lru_cache(maxsize=16)
def get_process(name, params_json):
    return process_module.Process(name, json.loads(params_json))


def settings_from_dict(d):
    known = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in d.items() if k in known})


def settings_for(model, **changes):
    """Settings for model `model` ('1', '2', '5a', 'B', 'GRU_alone', ...): the Settings defaults, then the model
    file's DEFAULTS, then `changes`."""
    module = load_model_module(model)
    values = dict(Settings().to_dict(), model=str(model))
    values.update(getattr(module, "DEFAULTS", {}))
    unknown = set(changes) - set(values)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}")
    values.update(changes)
    s = Settings(**values)
    s.vocab_size = s.process_object().vocab_size
    return s


# ------------------------------------------------------------------------------------------------ models
def available_models():
    """The model files in this folder: model_<name>.py -> name."""
    return sorted(p.stem[len("model_"):] for p in HERE.glob("model_*.py"))


def load_model_module(name):
    name = str(name)
    if name not in available_models():
        raise ValueError(f"no file model_{name}.py in {HERE}; available: {available_models()}")
    return importlib.import_module(f"model_{name}")


def make_model(settings):
    """The model with its starting weights (torch seed settings.model_seed)."""
    module = load_model_module(settings.model)
    torch.manual_seed(settings.model_seed)
    model = module.Model(settings)
    if settings.emission_start == "data_frequency":
        # the output layer starts by predicting the training data's token frequency (a separate draw, not the
        # training stream): weights scaled to 1% of their random start, bias = log frequency
        sample = settings.process_object().sampler(settings.train_seed + 1_000_000).draw(4096, settings.total_len)
        freq = torch.bincount(sample.reshape(-1), minlength=settings.vocab_size).float()
        with torch.no_grad():
            model.causal_head.emission.weight.mul_(0.01)
            model.causal_head.emission.bias.copy_(torch.log(freq / freq.sum()))
    return model


def code_fingerprint(model):
    """A hash of the code a run depends on: its model file, causal_state_matrix.py, training.py and process.py."""
    files = [HERE / f"model_{model}.py", HERE / "causal_state_matrix.py", HERE / "training.py", HERE / "process.py"]
    return hashlib.sha256("".join(p.read_text() for p in files).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ schedule
def temperature(step, total_steps, start=2.25, end=0.1):
    """The temperature at update `step` (0-based): start * (end / start) ** (step / (total_steps - 1))."""
    if total_steps < 1 or not 0 <= step < total_steps or min(start, end) <= 0:
        raise ValueError("invalid temperature schedule")
    fraction = step / max(1, total_steps - 1)
    return start * (end / start) ** fraction


def make_optimizer(model, settings):
    """AdamW (weight decay 0 is Adam).  The Transformer's scoring layer, when the model has one
    (`transformer_selector`), is in its own group so that selector_lr_mult can change its learning rate."""
    selector = list(model.transformer_selector.parameters()) if hasattr(model, "transformer_selector") else []
    chosen = {id(p) for p in selector}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in chosen], base_lr=settings.lr)]
    if selector:
        groups.append(dict(params=selector, base_lr=settings.lr * settings.selector_lr_mult))
    for g in groups:
        g["lr"] = g["base_lr"]
    return torch.optim.AdamW(groups, lr=settings.lr, weight_decay=settings.weight_decay)


# ------------------------------------------------------------------------------------------------ measures
def prediction_loss(token_logits, targets):
    """Mean cross-entropy in bits of token logits (..., V) against integer targets (...)."""
    return F.cross_entropy(token_logits.reshape(-1, token_logits.shape[-1]), targets.reshape(-1)) / LN2


def ce_bits(logits, targets):
    """Cross-entropy in bits of every prediction (no mean)."""
    return -logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1) / LN2


def entropy_bits(p):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1) / LN2


def exact_ce_bits(process, tokens):
    """-log2 of the true process's probability of every token given all tokens before it, (N, T)."""
    x = np.asarray(tokens)
    p = process.exact_probs(x)
    return -np.log2(np.take_along_axis(p, x[..., None], -1)[..., 0])


def information_fraction(states, true_states):
    """I(learned state; true state) / H(true state), from their joint counts: 1 when the learned state tells the
    true state exactly, 0 when it tells nothing about it."""
    s, a = np.asarray(states).ravel(), np.asarray(true_states).ravel()
    joint = np.zeros((s.max() + 1, a.max() + 1))
    np.add.at(joint, (s, a), 1)
    joint /= joint.sum()
    ps, pa = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    info = float((joint[nz] * np.log2(joint[nz] / (ps @ pa)[nz])).sum())
    p = pa[pa > 0]
    return max(0.0, info / float(-(p * np.log2(p)).sum()))


def determinism(states, next_tokens):
    """The share of steps (state, token read next) -> next state that go to the most common next state of that
    (state, token) pair: 1 for a machine whose next state is fixed by its state and the token.
    states (N, n); next_tokens[:, j] is the token read between states[:, j] and states[:, j + 1]."""
    s0, x1, s1 = states[:, :-1].ravel(), next_tokens[:, :states.shape[1] - 1].ravel(), states[:, 1:].ravel()
    key = s0 * (int(x1.max()) + 1 if x1.size else 1) + x1
    counts = pd.DataFrame(dict(key=key, nxt=s1)).groupby(["key", "nxt"]).size()
    return float(counts.groupby(level=0).max().sum() / len(s0))


@torch.no_grad()
def evaluate(model, settings, tokens, true_states, tau, exact_ce=None):
    """
    Held-out measures on fixed windows, without noise or gradients (recorded during training; analysis.py reuses it).

    For every predictor p the model has (see the module docstring):
        excess_<p>_future   CE of its predictions of the L future tokens minus the true process's CE on the same tokens,
                            in bits per token (0 is perfect)
        excess_<p>_all      the same over every token it predicts from position context + 1 on (own predictors only)
        <p>_states_used     different states it uses
        <p>_information     I(its state; true state) / H(true state)
        <p>_determinism     share of its steps that go to the most common next state of (state, token)
    and, for models with a Transformer, handoff_margin_median (gap between the two highest state scores at the last
    history position; above about 10 the choice can no longer change in training) and handoff_codes (different states
    the Transformer picks there).  The model's own measures (measures()) are added.
    """
    was = model.training
    model.eval()
    try:
        H = settings.history_len
        if exact_ce is None:
            exact_ce = exact_ce_bits(settings.process_object(), tokens)
        prediction = model.predict(tokens, tau)
        x = tokens.numpy()
        out = {}
        for name, p in prediction["predictors"].items():
            first, logits, states = p["first"], p["logits"], p["states"]
            n = logits.shape[1]
            ce = ce_bits(logits, tokens[:, first:first + n]).numpy()
            out[f"excess_{name}_future"] = float(ce[:, H - first:].mean() - exact_ce[:, H:first + n].mean())
            if first < H:
                out[f"excess_{name}_all"] = float(ce.mean() - exact_ce[:, first:first + n].mean())
            s = states.numpy()
            out[f"{name}_states_used"] = int(np.unique(s).size)
            out[f"{name}_information"] = information_fraction(s, true_states[:, first:first + s.shape[1]])
            if s.shape[1] > 1:
                out[f"{name}_determinism"] = determinism(s, x[:, first:first + s.shape[1] - 1])
        if "handoff_scores" in prediction:
            scores = prediction["handoff_scores"]
            top2 = scores.topk(2, dim=-1).values
            out["handoff_margin_median"] = float((top2[:, 0] - top2[:, 1]).median())
            out["handoff_codes"] = int(scores.argmax(-1).unique().numel())
        if hasattr(model, "measures"):
            out.update(model.measures(tokens, tau, prediction))
        return out
    finally:
        model.train(was)


def main_metric(model_name):
    return f"excess_{load_model_module(model_name).MAIN_PREDICTOR}_future"


# ------------------------------------------------------------------------------------------------ training
def grad_norm(params):
    g = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return float(torch.cat(g).norm()) if g else 0.0


def train_step(model, window, optimizer, tau, weight, clip=0.0, record_grads=False):
    """One update: loss -> gradients -> (clip) -> optimizer step -> the model's after_update."""
    model.train()
    loss, terms, ids = model.loss(window, tau, weight)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    out = {k: float(v.detach()) for k, v in terms.items()}
    out["loss"] = float(loss.detach())
    if record_grads:
        for name, child in model.named_children():
            out[f"grad_{name}"] = grad_norm(child.parameters())
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    if hasattr(model, "after_update"):
        model.after_update(ids)
    return out


def run_folder(settings):
    """runs/model_<name>/<process>_w<weight>_n<steps>_s<seed>_<key>: key = hash of the settings and of the code."""
    key = hashlib.sha256((json.dumps(settings.to_dict(), sort_keys=True)
                          + code_fingerprint(settings.model)).encode()).hexdigest()[:10]
    label = settings.process_object().label
    return RUNS / f"model_{settings.model}" / f"{label}_w{settings.weight:g}_n{settings.steps}_s{settings.seed}_{key}"


def validation_windows(settings):
    return settings.process_object().sampler(settings.validation_seed).draw(
        settings.validation_rows, settings.total_len, states=True)


def _progress(settings, update, values, metric, tau, prefix=""):
    used = values.get("transformer_states_used", values.get("gru_from_start_states_used"))
    print(f"{prefix}model {settings.model} s{settings.seed} w{settings.weight:g} | {update:6d} | {metric} "
          f"{values[metric]:+.5f} | states {used} | tau {tau:.3f} | {values['seconds'] / 60:.1f} min", flush=True)


def train(settings, folder=None, force=False, verbose=True):
    """Train a model with `settings` and save the run in `folder` (default run_folder(settings)).  A finished run in
    that folder is loaded instead, unless force=True.  Returns (model, folder)."""
    folder = Path(folder) if folder is not None else run_folder(settings)
    if (folder / "completed.pt").exists() and not force:
        return load_run(folder)[0], folder
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(json.dumps(dict(settings.to_dict(), code=code_fingerprint(settings.model)),
                                                     indent=2))
    torch.set_num_threads(settings.threads)
    model = make_model(settings)
    optimizer = make_optimizer(model, settings)
    process = settings.process_object()
    schedule_steps = settings.schedule_steps or settings.steps
    train_source = process.sampler(settings.train_seed)
    validation, validation_states = validation_windows(settings)
    validation_exact = exact_ce_bits(process, validation)
    T, metric = settings.total_len, main_metric(settings.model)
    training, hist, best = [], [], None
    start = time.perf_counter()

    def record(update, tau, grads):
        nonlocal best
        values = evaluate(model, settings, validation, validation_states, tau, validation_exact)
        values.update(update=update, tau=tau, seconds=time.perf_counter() - start, **grads)
        hist.append(values)
        pd.DataFrame(hist).to_csv(folder / "validation_history.csv", index=False)
        (folder / "progress.json").write_text(json.dumps(values, indent=2))
        if best is None or values[metric] < best[0]:
            best = (values[metric], update)
            torch.save(dict(settings=settings.to_dict(), model_state=model.state_dict(), update=update, metric=metric,
                            value=values[metric]), folder / "best.pt")
        if verbose and (update % 5000 == 0 or update == settings.steps):
            _progress(settings, update, values, metric, tau)

    record(0, settings.tau_start, {})
    for step in range(settings.steps):
        tau = temperature(step, schedule_steps, settings.tau_start, settings.tau_end)
        if settings.lr_warmup:
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * min(1.0, (step + 1) / settings.lr_warmup)
        window = train_source.draw(settings.batch_size, T)
        is_eval = (step + 1) % settings.eval_every == 0 or step + 1 == settings.steps
        out = train_step(model, window, optimizer, tau, settings.weight, settings.grad_clip, record_grads=is_eval)
        if not math.isfinite(out["loss"]):
            raise RuntimeError(f"non-finite loss at update {step + 1}")
        training.append(dict(update=step + 1, tau=tau, **{k: v for k, v in out.items() if not k.startswith("grad_")}))
        if is_eval:
            record(step + 1, tau, {k: v for k, v in out.items() if k.startswith("grad_")})
    torch.save(dict(settings=settings.to_dict(), model_state=model.state_dict(), updates=settings.steps,
                    seconds=time.perf_counter() - start, best_update=best[1], best_value=best[0], metric=metric,
                    optimizer_state=optimizer.state_dict(), rng_state=train_source.rng.bit_generator.state,
                    torch_rng_state=torch.get_rng_state(), torch_version=str(torch.__version__)),
               folder / "completed.pt")
    pd.DataFrame(training).to_csv(folder / "training_history.csv", index=False)
    return model, folder


def extensions(folder):
    path = Path(folder) / "extensions.json"
    return json.loads(path.read_text()) if path.exists() else []


def checkpoint(folder, which="latest"):
    """(file, update) of a run's parameters: 'completed' = after the planned updates, 'latest' = after any longer
    training, 'best' = the best value of the main measure on the validation windows."""
    folder = Path(folder)
    if which == "completed":
        return folder / "completed.pt", read_settings(folder).steps
    if which == "latest":
        ext = extensions(folder)
        return (folder / ext[-1]["file"], ext[-1]["updates"]) if ext else (folder / "completed.pt",
                                                                            read_settings(folder).steps)
    if which == "best":
        return folder / "best.pt", int(torch.load(folder / "best.pt", map_location="cpu", weights_only=False)["update"])
    raise ValueError("which must be 'completed', 'latest' or 'best'")


def read_settings(folder):
    return settings_from_dict(json.loads((Path(folder) / "settings.json").read_text()))


def load_run(folder, which="latest"):
    """(model with the saved parameters, settings, update) of a saved run."""
    settings = read_settings(folder)
    file, update = checkpoint(folder, which)
    model = make_model(settings)
    model.load_state_dict(torch.load(file, map_location="cpu", weights_only=False)["model_state"])
    model.eval()
    return model, settings, update


def adopt_state_dict(model, state):
    """Load parameters saved by the code of sections 13-17 (model_modeularised) into a model of this folder.  The modular
    models kept their scoring layer inside the head ('causal_head.state_head.*'; here 'state_head.*'); saved entries the
    model does not have (section 16 kept the state counts 'state_freq' in every model) are left out.  Every parameter
    and buffer of the model must be in `state`."""
    state = {(k.replace("causal_head.state_head.", "state_head.") if k.startswith("causal_head.state_head.") else k): v
             for k, v in state.items()}
    own = model.state_dict()
    missing = sorted(set(own) - set(state))
    if missing:
        raise ValueError(f"the saved parameters lack {missing}")
    model.load_state_dict({k: v for k, v in state.items() if k in own})
    return model


def find_runs(root=None, finished=True, **criteria):
    """Saved runs under runs/model_*/ whose settings match every given setting, e.g.
    find_runs(model="5b", process="flower", params={...}, seed=0).  Returns {folder: Settings}.  It reads each run's
    settings.json, so it finds runs made before the code changed (the folder names include a code fingerprint)."""
    out = {}
    for path in sorted(Path(root or RUNS).glob("model_*/*/settings.json")):
        d = json.loads(path.read_text())
        if all(d.get(k) == v for k, v in criteria.items()) and (not finished or (path.parent / "completed.pt").exists()):
            out[path.parent] = settings_from_dict(d)
    return out


def history(folder):
    """The validation history of a run, including any longer training."""
    folder = Path(folder)
    h = pd.read_csv(folder / "validation_history.csv")
    if (folder / "validation_history_extended.csv").exists():
        h = pd.concat([h, pd.read_csv(folder / "validation_history_extended.csv")], ignore_index=True)
    return h


def extend(folder, extra_steps, verbose=True):
    """Train a finished run `extra_steps` more updates at its final temperature, continuing its optimizer state, its
    training windows and its noise exactly; best.pt is replaced when the main measure improves."""
    folder = Path(folder)
    settings = read_settings(folder)
    src, _ = checkpoint(folder, "latest")
    saved = torch.load(src, map_location="cpu", weights_only=False)
    torch.set_num_threads(settings.threads)
    model = make_model(settings)
    model.load_state_dict(saved["model_state"])
    optimizer = make_optimizer(model, settings)
    optimizer.load_state_dict(saved["optimizer_state"])
    process = settings.process_object()
    train_source = process.sampler(settings.train_seed)
    train_source.rng.bit_generator.state = saved["rng_state"]
    validation, validation_states = validation_windows(settings)
    validation_exact = exact_ce_bits(process, validation)
    torch.set_rng_state(saved["torch_rng_state"])
    T, metric = settings.total_len, main_metric(settings.model)
    best_value = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)["value"]
    first, tau, start = saved["updates"], settings.tau_end, time.perf_counter()
    hist_path = folder / "validation_history_extended.csv"
    hist = pd.read_csv(hist_path).to_dict("records") if hist_path.exists() else []
    training = []
    for step in range(first, first + extra_steps):
        window = train_source.draw(settings.batch_size, T)
        is_eval = (step + 1) % settings.eval_every == 0 or step + 1 == first + extra_steps
        out = train_step(model, window, optimizer, tau, settings.weight, settings.grad_clip, record_grads=is_eval)
        if not math.isfinite(out["loss"]):
            raise RuntimeError(f"non-finite loss at update {step + 1}")
        training.append(dict(update=step + 1, tau=tau, **{k: v for k, v in out.items() if not k.startswith("grad_")}))
        if is_eval:
            values = evaluate(model, settings, validation, validation_states, tau, validation_exact)
            values.update(update=step + 1, tau=tau, seconds=time.perf_counter() - start,
                          **{k: v for k, v in out.items() if k.startswith("grad_")})
            hist.append(values)
            pd.DataFrame(hist).to_csv(hist_path, index=False)
            if values[metric] < best_value:
                best_value = values[metric]
                torch.save(dict(settings=settings.to_dict(), model_state=model.state_dict(), update=step + 1,
                                metric=metric, value=best_value), folder / "best.pt")
            if verbose and (step + 1) % 5000 == 0:
                _progress(settings, step + 1, values, metric, tau, prefix="extend ")
    total = first + extra_steps
    name = f"extended_{total}.pt"
    torch.save(dict(settings=settings.to_dict(), model_state=model.state_dict(), updates=total,
                    optimizer_state=optimizer.state_dict(), rng_state=train_source.rng.bit_generator.state,
                    torch_rng_state=torch.get_rng_state(), seconds=time.perf_counter() - start), folder / name)
    pd.DataFrame(training).to_csv(folder / f"training_history_extended_{total}.csv", index=False)
    ext = extensions(folder) + [dict(file=name, updates=total, seconds=time.perf_counter() - start)]
    (folder / "extensions.json").write_text(json.dumps(ext, indent=2))
    return model


def still_improving(folder, tol=EXTEND_TOL, window=10_000):
    """Section 16's rule: True when the main validation measure over the last `window` updates beat everything before
    by more than `tol` bits.  Returns (improving, last update)."""
    h = history(folder)
    metric, last = main_metric(read_settings(folder).model), int(h["update"].max())
    recent, before = h[h["update"] > last - window][metric], h[h["update"] <= last - window][metric]
    return bool(len(before) and recent.min() < before.min() - tol), last


def train_until_settled(settings, step=EXTEND_STEP, max_updates=EXTEND_MAX, tol=EXTEND_TOL, verbose=True):
    """train(), then extend by `step` updates while still_improving, up to `max_updates` (section 16's schedule)."""
    model, folder = train(settings, verbose=verbose)
    while True:
        improving, last = still_improving(folder, tol)
        if not improving or last >= max_updates:
            return folder
        if verbose:
            print(f"still improving at {last:,} updates: {step:,} more", flush=True)
        extend(folder, step, verbose=verbose)
