"""
Section 11 of walkthrough.ipynb: can model 1 learn the coin, the smallest process with memory?

The coin (process.py, p = 0.4, q = 0.8) has two causal states and three tokens. From state 0 the next token is 0 (stay)
or 1 (go to state 1); from state 1 it is 1 (stay) or 2 (go to state 0). The state is fixed by the last token: 0 or 2
-> state 0, 1 -> state 1. A model only has to remember the last token.

Every run is model 1 (or a reference model) with section 16's settings except for the changes its configuration names.
Some changes are not settings of training.py (Adam's beta1 and beta2, SGD, a random start of the state table): this file
makes them by giving training.train its own optimizer and starting model, and changes nothing else. The default
configuration gives exactly the parameters of training.train (checked in the notebook).

    python coin_check.py run --stage pilot --workers 6      the pilot: model 1 and the reference models, 3 seeds
    python coin_check.py run --stage search --workers 6     the one-at-a-time changes and the lr x beta1 grid
    python coin_check.py run --stage even --workers 6       the even process: model 1 and the feedback GRU
    python coin_check.py run --stage long --workers 6       50,000 updates: section 16's settings and section 13's
    python coin_check.py queue --workers 8                  long, even and search in one queue (longest first)
    python coin_check.py status

Runs are saved in runs/coin_check/<configuration>_n<updates>_s<seed>/ (the same files as training.train), with
coin_check.json (the configuration) and diagnostics_<which>.json (diagnose()).
"""
import argparse
import contextlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import training as TR                                                           # noqa: E402

ROOT = TR.RUNS / "coin_check"
COIN = dict(process="coin", params={"p": 0.4, "q": 0.8})
SEEDS = (0, 1, 2)
SEARCH_STEPS = 10_000
LONG_STEPS = 50_000
TEST_ROWS, TEST_SEED, FUTURE = 4096, 4404, 20                    # the test windows of analysis.py: H + 20 tokens

# ------------------------------------------------------------------------------------------- configurations
# Each: the model, the settings that differ from section 16's, the optimizer, and the state table's start.
ADAM = dict(kind="adamw", beta1=0.9, beta2=0.999)               # training.make_optimizer's AdamW
TRAINING_OPTIMIZER, TRAINING_MODEL = TR.make_optimizer, TR.make_model   # the originals (patched() replaces them)


def config(model="1", optimizer=None, table_start="equal", group="", search_steps=None, **settings):
    return dict(model=model, settings=settings, optimizer=dict(ADAM, **(optimizer or {})), table_start=table_start,
                group=group, search_steps=search_steps)


CONFIGS = {
    # the pilot: model 1 at section 16's settings, and three references
    "default": config(group="section 16's settings"),
    "model_2": config(model="2", group="reference models"),
    "feedback_GRU": config(model="feedback_GRU", group="reference models"),
    "discrete_Transformer": config(model="discrete_Transformer", group="reference models"),
    # one change at a time (model 1)
    "lr_1e-4": config(lr=1e-4, group="learning rate"),
    "lr_3e-4": config(lr=3e-4, group="learning rate"),
    "lr_3e-3": config(lr=3e-3, group="learning rate"),
    "lr_1e-2": config(lr=1e-2, group="learning rate"),
    "beta1_0": config(optimizer=dict(beta1=0.0), group="Adam beta1 (momentum)"),
    "beta1_0.5": config(optimizer=dict(beta1=0.5), group="Adam beta1 (momentum)"),
    "beta1_0.99": config(optimizer=dict(beta1=0.99), group="Adam beta1 (momentum)"),
    "beta2_0.99": config(optimizer=dict(beta2=0.99), group="Adam beta2"),
    "beta2_0.9999": config(optimizer=dict(beta2=0.9999), group="Adam beta2"),
    "sgd_m0_lr0.3": config(lr=0.3, optimizer=dict(kind="sgd", momentum=0.0), group="SGD"),
    "sgd_m0.9_lr0.03": config(lr=0.03, optimizer=dict(kind="sgd", momentum=0.9), group="SGD"),
    "sgd_m0.9_lr0.3": config(lr=0.3, optimizer=dict(kind="sgd", momentum=0.9), group="SGD"),
    # SGD adds the weight decay to the gradient: at lr 0.3 it shrinks the weights by lr x 5e-4 = 1.5e-4 per update,
    # 300 times AdamW's 1e-3 x 5e-4.  These two remove it, and then also the equal state-table rows.
    "sgd_m0_lr0.3_wd0": config(lr=0.3, weight_decay=0.0, optimizer=dict(kind="sgd", momentum=0.0), group="SGD"),
    "sgd_m0_lr0.3_wd0_table_random": config(lr=0.3, weight_decay=0.0, optimizer=dict(kind="sgd", momentum=0.0),
                                            table_start="random", group="SGD"),
    "wd_0": config(weight_decay=0.0, group="weight decay"),
    "wd_5e-3": config(weight_decay=5e-3, group="weight decay"),
    "clip_off": config(grad_clip=0.0, group="gradient clipping"),
    "tau_const1": config(tau_start=1.0, tau_end=1.0, group="temperature"),
    "tau_2.5_0.1": config(tau_start=2.5, tau_end=0.1, group="temperature"),
    "tau_5_1": config(tau_start=5.0, tau_end=1.0, group="temperature"),
    "no_noise": config(gumbel=False, group="noise in the choice"),
    "no_layernorm": config(encoder_norm=False, group="LayerNorm on the Transformer's output"),
    "table_random": config(table_start="random", group="state table start"),
    "original_setup": config(**TR.ORIGINAL_SETUP, group="section 13's setup (no LayerNorm, decay, clipping)"),
    "batch_32": config(batch_size=32, group="batch size"),
    "batch_512": config(batch_size=512, search_steps=2_500, group="batch size"),     # the same windows as 10,000 x 128
    "K_4": config(n_states=4, group="table rows K"),
    "L_2": config(rollout_len=2, group="future tokens L"),
    "L_5": config(rollout_len=5, group="future tokens L"),
}
# an extra: the even process, also 2 states, but its state is not fixed by the last token (a model must carry it)
EVEN = dict(process="even", params={"p": 0.5})
CONFIGS["even_model_1"] = config(group="the even process", **EVEN)
CONFIGS["even_feedback_GRU"] = config(model="feedback_GRU", group="the even process", **EVEN)
# the learning rate x beta1 grid (lr 1e-3 and beta1 0.9 are in the one-at-a-time list)
for _lr in (3e-4, 3e-3):
    for _b1 in (0.0, 0.5, 0.99):
        CONFIGS[f"lr_{_lr:g}_beta1_{_b1:g}"] = config(lr=_lr, optimizer=dict(beta1=_b1), group="lr x beta1 grid")

STAGES = {"pilot": (["default", "model_2", "feedback_GRU", "discrete_Transformer"], SEARCH_STEPS),
          "search": ([n for n in CONFIGS if n not in ("default", "model_2", "feedback_GRU", "discrete_Transformer")
                      and not n.startswith("even_")], SEARCH_STEPS),
          "even": (["even_model_1", "even_feedback_GRU"], SEARCH_STEPS),
          "long": (["default", "original_setup"], LONG_STEPS)}


# ------------------------------------------------------------------------------------------- training
def steps_for(name, steps):
    """A configuration's own number of updates in the search (batch 512: 2,500, the same windows as the default)."""
    own = CONFIGS[name].get("search_steps")
    return own if (own and steps == SEARCH_STEPS) else steps


def settings_of(name, seed, steps):
    c = CONFIGS[name]
    return TR.settings_for(c["model"], **dict(COIN, **c["settings"]), seed=seed, steps=steps)


def folder_of(name, seed, steps):
    return ROOT / f"{name}_n{steps}_s{seed}"


def make_optimizer(model, settings, spec):
    """training.make_optimizer with Adam's betas, or SGD with momentum, in the same parameter groups."""
    if spec["kind"] == "adamw" and (spec["beta1"], spec["beta2"]) == (0.9, 0.999):
        return TRAINING_OPTIMIZER(model, settings)
    selector = list(model.transformer_selector.parameters()) if hasattr(model, "transformer_selector") else []
    chosen = {id(p) for p in selector}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in chosen], base_lr=settings.lr)]
    if selector:
        groups.append(dict(params=selector, base_lr=settings.lr * settings.selector_lr_mult))
    for g in groups:
        g["lr"] = g["base_lr"]
    if spec["kind"] == "adamw":
        return torch.optim.AdamW(groups, lr=settings.lr, betas=(spec["beta1"], spec["beta2"]),
                                 weight_decay=settings.weight_decay)
    if spec["kind"] == "sgd":                   # (weight decay here is added to the gradient, unlike AdamW's)
        return torch.optim.SGD(groups, lr=settings.lr, momentum=spec["momentum"], weight_decay=settings.weight_decay)
    raise ValueError(spec["kind"])


@contextlib.contextmanager
def patched(spec, table_start):
    """training.train with this optimizer and state-table start (restored afterwards)."""
    def model_maker(settings):
        model = TRAINING_MODEL(settings)
        if table_start == "random":            # as CausalStateMatrix(start="random"): randn / sqrt(S)
            with torch.no_grad():
                E = model.causal_head.state_matrix
                E.copy_(torch.randn_like(E) / E.shape[1] ** 0.5)
        return model

    TR.make_optimizer = lambda model, settings: make_optimizer(model, settings, spec)
    TR.make_model = model_maker
    try:
        yield
    finally:
        TR.make_optimizer, TR.make_model = TRAINING_OPTIMIZER, TRAINING_MODEL


def train_one(name, seed, steps, verbose=True):
    """Train (or load) one run and diagnose it; returns its folder."""
    c = CONFIGS[name]
    settings, folder = settings_of(name, seed, steps), folder_of(name, seed, steps)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "coin_check.json").write_text(json.dumps(dict(name=name, **c), indent=2))
    with patched(c["optimizer"], c["table_start"]):
        TR.train(settings, folder=folder, verbose=verbose)
    for which in ("latest", "best"):
        diagnose(folder, which)
    return folder


def _worker(name, seed, steps):
    log = ROOT / "logs" / f"{name}_n{steps}_s{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with open(log, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        train_one(name, seed, steps)
    return name, seed, steps, time.perf_counter() - start


def run(names, steps, workers, seeds=SEEDS, jobs=None):
    jobs = jobs or [(n, s, steps_for(n, steps)) for n in names for s in seeds]
    print(f"{len(jobs)} runs, {workers} at a time", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, *j) for j in jobs]
        for f in as_completed(futures):
            name, seed, n, seconds = f.result()
            print(f"done {name} s{seed} n{n}: {seconds / 60:.1f} min", flush=True)


# ------------------------------------------------------------------------------------------- diagnostics
def test_windows(settings):
    """The 4,096 test windows of analysis.py with H + 20 tokens, whatever the run's own L: (tokens, states, exact CE)."""
    proc = settings.process_object()
    x, st = proc.sampler(TEST_SEED).draw(TEST_ROWS, settings.history_len + FUTURE, states=True)
    return x, st, TR.exact_ce_bits(proc, x)


def linear_classifier(xtr, ytr, xte, yte, steps=1500):
    """Accuracy of a linear classifier (softmax regression on standardised inputs) fitted on (xtr, ytr), on (xte, yte)."""
    torch.manual_seed(0)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    lin = torch.nn.Linear(xtr.shape[1], int(ytr.max()) + 1)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(xte).argmax(-1) == yte).float().mean())


def encoder_output(model, tokens):
    with torch.no_grad():
        v = model.encoder(tokens)
        if getattr(model, "encoder_norm", None) is not None:
            v = model.encoder_norm(v)
    return v[:, model.cfg.history_len - 1]


def gradient_probe(model, settings, batches=8):
    """Training-mode gradients of the run's own loss at its final temperature, on fresh windows: the size of the
    gradient at the handoff vector, after the choice (at the Transformer's scores there), and reaching each part; and
    how often the soft choice q at the handoff is above 0.999 (then no gradient passes the choice)."""
    H, tau = settings.history_len, settings.tau_end
    sampler = settings.process_object().sampler(4242)
    parts = {"encoder": list(model.encoder.parameters()), "selector": list(model.transformer_selector.parameters()),
             "gru": list(model.gru_cell.parameters()) + list(model.gru_selector.parameters()),
             "state_table": [model.causal_head.state_matrix]}
    sums = {}
    own = model.transformer_pass
    for _ in range(batches):
        store = {}

        def wrapped(tokens, t):
            scores, sel = own(tokens, t)
            scores.retain_grad()
            sel.representation.retain_grad()
            store.update(scores=scores, sel=sel)
            return scores, sel

        model.train()
        model.zero_grad(set_to_none=True)
        model.transformer_pass = wrapped
        try:
            loss, _, _ = model.loss(sampler.draw(settings.batch_size, settings.total_len), tau, settings.weight)
            loss.backward()
        finally:
            model.transformer_pass = own
        row = {f"grad_{k}": TR.grad_norm(v) for k, v in parts.items()}
        row["grad_at_handoff_vector"] = float(store["sel"].representation.grad[:, H - 1].norm())
        row["grad_after_choice"] = float(store["scores"].grad[:, H - 1].norm())
        q = store["sel"].probs[:, H - 1].detach().max(-1).values
        row["choice_locked_share"] = float((q > 0.999).float().mean())
        for k, v in row.items():
            sums[k] = sums.get(k, 0.0) + v / batches
    model.eval()
    return sums


def diagnose(folder, which="latest", force=False):
    """Everything section 11 shows about one run, cached in <folder>/diagnostics_<which>.json."""
    folder = Path(folder)
    path = folder / f"diagnostics_{which}.json"
    model, settings, update = TR.load_run(folder, which)
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        if cached["update"] == update:
            return cached
    torch.set_num_threads(1)
    H, tau = settings.history_len, settings.tau_end
    x, st, exact = test_windows(settings)
    out = dict(update=update, which=which, model=settings.model, seed=settings.seed)
    with torch.no_grad():
        if hasattr(model, "gru_step") and hasattr(model, "transformer_selector"):        # hybrids: 1, 2
            start = model.handoff_states(x[:, :H], tau)
            states, logits = model.rollout(start, x[:, H:], model.gru_step)
            ce = TR.ce_bits(logits, x[:, H:]).numpy() - exact[:, H:]
            out.update(excess_future=float(ce.mean()), excess_first=float(ce[:, 0].mean()),
                       excess_rest=float(ce[:, 1:].mean()), excess_by_position=ce.mean(0).tolist(),
                       handoff_states=int(start.unique().numel()),
                       handoff_information=TR.information_fraction(start.numpy(), st[:, H]),
                       gru_states_used=int(states[:, 1:].unique().numel()),
                       gru_information=TR.information_fraction(states[:, 1:].numpy(), st[:, H + 1:H + FUTURE]))
            pred = model.predict(x, tau) if settings.rollout_len == FUTURE else None
            if pred is not None:                                      # the Transformer's own states at every position
                p = pred["predictors"]["transformer"]
                out["transformer_information"] = TR.information_fraction(p["states"].numpy(),
                                                                         st[:, p["first"]:p["first"] + p["states"].shape[1]])
        else:                                                         # the reference models alone
            pred = model.predict(x, tau)
            name = TR.load_model_module(settings.model).MAIN_PREDICTOR
            p = pred["predictors"][name]
            first, n = p["first"], p["logits"].shape[1]
            ce = TR.ce_bits(p["logits"], x[:, first:first + n]).numpy() - exact[:, first:first + n]
            fut = ce[:, H - first:]
            out.update(excess_future=float(fut.mean()), excess_first=float(fut[:, 0].mean()),
                       excess_rest=float(fut[:, 1:].mean()), excess_by_position=fut.mean(0).tolist(),
                       information=TR.information_fraction(p["states"].numpy(),
                                                           st[:, first:first + p["states"].shape[1]]))
    if hasattr(model, "encoder"):
        tr_x, tr_st = settings.process_object().sampler(6161).draw(4096, settings.history_len, states=True)
        untrained = TRAINING_MODEL(settings)
        y_tr, y_te = torch.from_numpy(tr_st[:, H]), torch.from_numpy(st[:, H])
        out["encoder_classifier"] = linear_classifier(encoder_output(model, tr_x), y_tr,
                                                      encoder_output(model, x[:, :H]), y_te)
        out["encoder_classifier_untrained"] = linear_classifier(encoder_output(untrained, tr_x), y_tr,
                                                                encoder_output(untrained, x[:, :H]), y_te)
    if hasattr(model, "transformer_selector") and hasattr(model, "gru_step"):
        out.update(gradient_probe(model, settings))
    path.write_text(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------------------------------- reading the results
def finished(names=None, steps=SEARCH_STEPS, seeds=SEEDS):
    """{(name, seed): folder} of the finished runs."""
    names = names or list(CONFIGS)
    return {(n, s): folder_of(n, s, steps_for(n, steps)) for n in names for s in seeds
            if (folder_of(n, s, steps_for(n, steps)) / f"diagnostics_best.json").exists()}


def summary(names=None, steps=SEARCH_STEPS, seeds=SEEDS):
    """One row per finished run: its configuration, the diagnostics at the final and the best parameters, and when its
    validation excess first fell below 0.05 and 0.005 bits."""
    import pandas as pd
    rows = []
    for (name, seed), folder in finished(names, steps, seeds).items():
        h = TR.history(folder)
        metric = TR.main_metric(CONFIGS[name]["model"])
        row = dict(config=name, group=CONFIGS[name]["group"], seed=seed,
                   code=json.loads((folder / "settings.json").read_text())["code"][:10],
                   minutes=float(h["seconds"].max()) / 60)
        for which in ("latest", "best"):
            d = json.loads((folder / f"diagnostics_{which}.json").read_text())
            for k, v in d.items():
                if not isinstance(v, list) and k not in ("which", "model", "seed"):
                    row[f"{k}" if which == "latest" else f"best_{k}"] = v
        for level in (0.05, 0.005):
            below = h[h[metric] < level]
            row[f"first_below_{level:g}"] = int(below["update"].iloc[0]) if len(below) else None
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- figures
# Every figure uses plain labels: LABELS names each configuration by what it changes.
LABELS = {
    "default": "section 16's settings", "model_2": "model 2 (+ the Transformer's own predictions)",
    "feedback_GRU": "feedback GRU alone", "discrete_Transformer": "discrete Transformer alone",
    "lr_1e-4": "learning rate 0.0001", "lr_3e-4": "learning rate 0.0003", "lr_3e-3": "learning rate 0.003",
    "lr_1e-2": "learning rate 0.01", "beta1_0": "Adam beta1 0 (no momentum)", "beta1_0.5": "Adam beta1 0.5",
    "beta1_0.99": "Adam beta1 0.99", "beta2_0.99": "Adam beta2 0.99", "beta2_0.9999": "Adam beta2 0.9999",
    "sgd_m0_lr0.3": "SGD, no momentum, learning rate 0.3", "sgd_m0.9_lr0.03": "SGD, momentum 0.9, learning rate 0.03",
    "sgd_m0.9_lr0.3": "SGD, momentum 0.9, learning rate 0.3",
    "sgd_m0_lr0.3_wd0": "SGD, no momentum, learning rate 0.3, no weight decay",
    "sgd_m0_lr0.3_wd0_table_random": "SGD as above, no weight decay, state table rows start at random", "wd_0": "weight decay 0", "wd_5e-3": "weight decay 0.005",
    "clip_off": "no gradient clipping", "tau_const1": "temperature 1 throughout", "tau_2.5_0.1": "temperature 2.5 to 0.1",
    "tau_5_1": "temperature 5 to 1", "no_noise": "no random noise in the choice of state",
    "no_layernorm": "no LayerNorm on the Transformer's output", "table_random": "state table rows start at random",
    "original_setup": "section 13's setup (no LayerNorm, weight decay or clipping)", "batch_32": "batch of 32 windows",
    "batch_512": "batch of 512 windows (2,500 updates)", "K_4": "4 rows in the state table (not 32)",
    "L_2": "2 future tokens per window (not 20)", "L_5": "5 future tokens per window (not 20)",
    "even_model_1": "model 1, even process", "even_feedback_GRU": "feedback GRU alone, even process"}
for _lr in (3e-4, 3e-3):
    for _b1 in (0.0, 0.5, 0.99):
        LABELS[f"lr_{_lr:g}_beta1_{_b1:g}"] = f"learning rate {_lr:g}, Adam beta1 {_b1:g}"
EXCESS = "CE above the true process (bits per token)"
REFERENCE_LABELS = {"remembers_nothing": "predicting each token's long-run frequency",
                    "handoff_useless": "best possible when the GRU's starting state tells nothing"}


def levels(name="default"):
    """The reference levels on the test windows of a configuration's process (analysis.reference_levels) and the best
    possible excess over the 20 future tokens when the GRU's starting state tells nothing ('handoff_useless': the true
    process's CE of the future tokens read without their history, minus its CE with the history).  On the coin the
    first future token fixes the state, so this is 'remembers nothing' on that token only (0.679 / 20); the even
    process needs a 0 to fix it, so it is larger there."""
    import analysis as A
    settings = settings_of(name, 0, SEARCH_STEPS)
    lv = A.reference_levels(settings, "test")
    tokens, _, exact = test_windows(settings)
    H = settings.history_len
    alone = TR.exact_ce_bits(settings.process_object(), np.asarray(tokens)[:, H:])
    return dict(lv, handoff_useless=float(alone.mean() - exact[:, H:].mean()))


def _style():
    import model_1_checks as C1
    C1.style()
    return C1


def _main_curve(folder):
    h = TR.history(folder)
    return h["update"] / 1000, h[TR.main_metric(TR.read_settings(folder).model)]


def plot_curves(names, steps=SEARCH_STEPS, title=None):
    """One panel per configuration: the validation CE above the true process of the model's main predictor (model 1:
    the GRU started from the Transformer's state), one line per seed; dashed and dotted: the reference levels."""
    C1 = _style()
    names = [n for n in names if finished([n], steps)]
    if not names:
        return print("no finished runs yet")
    fig, axes = C1._grid(len(names), width=6.0, height=3.4)
    for ax, name in zip(axes, names):
        lv = levels(name)
        for (n, s), f in finished([name], steps).items():
            u, y = _main_curve(f)
            ax.plot(u, y.clip(lower=1e-4), color=["#eb6834", "#2a78d6", "#1baf7a"][s], label=f"seed {s}")
        ax.axhline(lv["remembers_nothing"], color="#898781", ls="--", lw=1.3, label=REFERENCE_LABELS["remembers_nothing"])
        if CONFIGS[name]["model"] in ("1", "2"):          # (the models whose GRU starts from the Transformer's state)
            ax.axhline(lv["handoff_useless"], color="#898781", ls=":", lw=1.3,
                       label=REFERENCE_LABELS["handoff_useless"])
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 2)
        ax.set_title(LABELS.get(name, name))
        ax.set_xlabel("update (thousands)")
        ax.set_ylabel("CE above true (bits/token)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3)
    fig.suptitle(title or "Validation " + EXCESS + " over training (values below 0.0001 drawn at 0.0001)")
    return fig


def plot_search(table, title=None):
    """table: summary().  One row per configuration, one marker per seed: the test CE above the true process over the
    20 future tokens.  Filled: final parameters; hollow: the parameters with the best validation value."""
    C1 = _style()
    import matplotlib.pyplot as plt
    order = [n for n in CONFIGS if n in set(table.config)][::-1]
    fig, ax = plt.subplots(figsize=(11, 0.36 * len(order) + 1.8), layout="constrained")
    for i, name in enumerate(order):
        for _, r in table[table.config == name].iterrows():
            y = i + (int(r.seed) - 1) * 0.18
            ax.scatter(max(r["excess_future"], 1e-5), y, s=40, color="#eb6834", zorder=3)
            ax.scatter(max(r["best_excess_future"], 1e-5), y, s=40, facecolor="none", edgecolor="#52514e", zorder=2)
    lv = levels()
    ax.axvline(lv["remembers_nothing"], color="#898781", ls="--", lw=1.3, label=REFERENCE_LABELS["remembers_nothing"])
    ax.axvline(lv["handoff_useless"], color="#898781", ls=":", lw=1.3, label=REFERENCE_LABELS["handoff_useless"])
    ax.scatter([], [], s=40, color="#eb6834", label="final parameters (one dot per seed)")
    ax.scatter([], [], s=40, facecolor="none", edgecolor="#52514e", label="parameters with the best validation value")
    ax.set_xscale("log")
    ax.set_xlim(7e-6, 2)
    ax.set_yticks(range(len(order)), [LABELS.get(n, n) for n in order])
    ax.set_xlabel("test CE above the true process (bits/token), 20 future tokens")
    ax.grid(axis="y", visible=False)
    groups = [CONFIGS[n]["group"] for n in order]
    for i in range(1, len(order)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="#e6e5e1", lw=1)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1)
    ax.set_title(title or "The coin: every configuration\n(model 1 unless named; values below 0.00001 drawn at 0.00001)")
    return fig


def plot_grid(table, column="excess_future"):
    """The learning rate x Adam beta1 grid: colour = the median over seeds of `column` at the final parameters; text =
    every seed's value."""
    C1 = _style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import analysis as A
    lrs, betas = [3e-4, 1e-3, 3e-3], [0.0, 0.5, 0.9, 0.99]

    def name(lr, b1):
        if lr == 1e-3:
            return "default" if b1 == 0.9 else f"beta1_{b1:g}"
        return {3e-4: "lr_3e-4", 3e-3: "lr_3e-3"}[lr] if b1 == 0.9 else f"lr_{lr:g}_beta1_{b1:g}"

    med = np.full((len(lrs), len(betas)), np.nan)
    text = [["" for _ in betas] for _ in lrs]
    for i, lr in enumerate(lrs):
        for j, b1 in enumerate(betas):
            v = table[table.config == name(lr, b1)].sort_values("seed")[column].to_numpy()
            if len(v):
                med[i, j] = np.median(v)
                text[i][j] = "\n".join(f"{x:.5f}" for x in v)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    im = ax.imshow(np.clip(med, 1e-5, None), cmap=A.ramp("#eb6834"), norm=LogNorm(1e-5, 1))
    for i in range(len(lrs)):
        for j in range(len(betas)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=11,
                    color="white" if med[i, j] > 0.01 else "#0b0b0b")
    ax.set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
    ax.set_yticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
    ax.set_xlabel("Adam beta1: share of the previous average gradient kept at each update (momentum)")
    ax.set_ylabel("learning rate")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="median over 3 seeds")
    ax.set_title("Learning rate and Adam beta1: test " + EXCESS + "\n(text: seeds 0, 1, 2)")
    return fig


def plot_by_position(names, steps=SEARCH_STEPS, which="latest"):
    """The test CE above the true process at each of the 20 future tokens (token 1 is predicted by the state the
    Transformer gives the GRU), per configuration: the mean over seeds for a name, one seed for a (name, seed) pair."""
    C1 = _style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    colours = ["#eb6834", "#2a78d6", "#1baf7a", "#8c5a9e", "#b8860b", "#52514e"]
    for k, item in enumerate(names):
        name, seeds = (item, SEEDS) if isinstance(item, str) else (item[0], (item[1],))
        runs = finished([name], steps, seeds)
        if not runs:
            continue
        rows = [json.loads((f / f"diagnostics_{which}.json").read_text())["excess_by_position"] for f in runs.values()]
        label = LABELS.get(name, name) + (" (mean over seeds)" if len(seeds) > 1 else f", seed {seeds[0]}")
        ax.plot(range(1, FUTURE + 1), np.clip(np.mean(rows, 0), 1e-5, None), "o-", ms=4, color=colours[k % 6],
                label=label)
    ax.set_yscale("log")
    ax.set_xticks(range(1, FUTURE + 1))
    ax.set_xlabel("future token (1 = predicted by the state the Transformer gives the GRU)")
    ax.set_ylabel("CE above true (bits/token)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1)
    ax.set_title("Test " + EXCESS + " at each future token\n(values below 0.00001 drawn at 0.00001)")
    return fig


def plot_over_training(names, steps=SEARCH_STEPS):
    """One panel per configuration, every 500 updates: the gap between the two highest of the Transformer's state scores
    at position 39 (median over the validation windows; log, left) and the number of different states the Transformer
    gives the GRU (right); one line style per seed."""
    C1 = _style()
    names = [n for n in names if finished([n], steps)]
    if not names:
        return print("no finished runs yet")
    fig, axes = C1._grid(len(names), width=6.0, height=3.4)
    styles = {0: "-", 1: "--", 2: ":"}
    for ax, name in zip(axes, names):
        twin = ax.twinx()
        most = 1
        for (n, s), f in finished([name], steps).items():
            h = TR.history(f)
            u = h["update"] / 1000
            ax.plot(u, h["handoff_margin_median"].clip(lower=0.01), color="#0b0b0b", ls=styles[s], lw=1.4)
            twin.plot(u, h["handoff_codes"], color="#2a78d6", ls=styles[s], lw=1.4)
            most = max(most, int(h["handoff_codes"].max()))
        ax.axhline(10, color="#898781", lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 100)
        ax.set_ylabel("gap (black)")
        twin.set_ylim(0, max(4, most + 1))
        twin.set_ylabel("different states (blue)", color="#2a78d6")
        twin.grid(False)
        ax.set_xlabel("update (thousands)")
        ax.set_title(LABELS.get(name, name))
    fig.suptitle("Black: gap between the two highest of the Transformer's state scores at position 39.\n"
                 "Blue: different states the Transformer gives the GRU.  Seeds 0, 1, 2: solid, dashed, dotted.")
    return fig


def plot_machines(runs, which="latest"):
    """The machine the GRU defines in each run {label: folder} (analysis.machine: read off exactly from its step
    rule), next to the true machine."""
    C1 = _style()
    import matplotlib.pyplot as plt
    import analysis as A
    items = list(runs.items())
    if not items:
        return print("no finished runs yet")
    fig, axes = plt.subplots(1, len(items) + 1, figsize=(4.6 * (len(items) + 1), 4.6), squeeze=False,
                             layout="constrained")
    s0 = TR.read_settings(items[0][1])
    C1.draw_machine(axes[0, 0], A.true_machine(s0), s0, "#898781", title="the true machine")
    for ax, (label, folder) in zip(axes[0, 1:], items):
        settings = TR.read_settings(folder)
        m = A.machine(folder, which)
        cmp = A.transition_comparison(m, settings)
        C1.draw_machine(ax, m, settings, "#eb6834",
                        title=f"{label}: {m['n_states']} states\nlargest |learned − true| probability "
                              f"{cmp['max_error']:.3f}")
    fig.suptitle("Circles: states (named by the true state they stand for; π = share of time spent in it).\n"
                 "Arrow 'x|p': on token x, with probability p, move along the arrow; ↺: stay in the same state.\n"
                 "'s1 (2nd)': a second learned state that also stands for true state s1.")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "queue", "status"])
    ap.add_argument("--stage", choices=sorted(STAGES))
    ap.add_argument("--configs", nargs="*", default=None, help="configurations to run instead of / added to a stage")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.action == "status":
        for steps in (SEARCH_STEPS, LONG_STEPS):
            done = finished(steps=steps)
            print(f"{steps} updates: {len(done)} finished runs")
        sys.exit()
    if a.action == "queue":
        jobs = [(n, seed, steps_for(n, st)) for stage in ("long", "even", "search")
                for n in STAGES[stage][0] for seed in SEEDS for st in [STAGES[stage][1]]]
        run([], None, a.workers, jobs=jobs)
        sys.exit()
    names, steps = STAGES[a.stage] if a.stage else ([], SEARCH_STEPS)
    names = list(names) + [n for n in (a.configs or []) if n not in names]
    run(names, a.steps or steps, a.workers)
