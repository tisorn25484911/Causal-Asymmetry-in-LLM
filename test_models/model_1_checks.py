"""
Checks of model 1 (model_1.py) used by model_1_study.ipynb: does the gradient of the GRU's loss reach and train the
Transformer?  Run on the imported renewal-15 runs of model 1 (runs/imported/model_1/), at their best-validation and
final parameters, and on an untrained model.  Results are cached in runs/model_1_checks/.

    gradient_at_start()     gradients of an untrained model 1 (equal state-table rows), and after adding noise to
                            the table
    first_updates()         the gradients over the first 300 updates of real training
    gradient_check(folder)  a trained run: gradient sizes at the handoff vector, after the choice, reaching each part;
                            how often the choice is locked; what the handoff state knows about the true state; Adam's
                            step sizes (final parameters)
    encoder_check(folder)   a linear classifier on the Transformer's output at the last history position, trained
                            against untrained, and how far each part's weights moved
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import training as TR                                                           # noqa: E402

IMPORTED = TR.RUNS / "imported" / "model_1"
CACHE = TR.RUNS / "model_1_checks"
RUN_TAGS = ["s0_section16", "s1_section16", "s2_section16", "s0_original_setup"]


def imported_run(tag):
    return IMPORTED / f"renewal-15_w0_{tag}"


def parts(model):
    """The parameters of each part of a hybrid."""
    enc = list(model.encoder.parameters()) + (list(model.encoder_norm.parameters())
                                              if getattr(model, "encoder_norm", None) is not None else [])
    return {"encoder": enc, "selector": list(model.transformer_selector.parameters()),
            "state_table": [model.causal_head.state_matrix],
            "emission": list(model.causal_head.emission.parameters()),
            "gru": list(model.gru_cell.parameters()) + list(model.gru_selector.parameters())}


def probe(model, window, tau, weight=0.0):
    """One training-mode loss and backward pass.  Returns the gradient length reaching each part; at the handoff
    vector (the state vector handed to the GRU); after the choice (at the Transformer's scores at the last history
    position); and the soft choice q at the handoff (its median largest entry and the share above 0.999)."""
    model.train()
    model.zero_grad(set_to_none=True)
    H = model.cfg.history_len
    store, own = {}, model.transformer_pass

    def wrapped(tokens, t):
        scores, sel = own(tokens, t)
        scores.retain_grad()
        sel.representation.retain_grad()
        store.update(scores=scores, sel=sel)
        return scores, sel

    model.transformer_pass = wrapped
    try:
        loss, _, _ = model.loss(window, tau, weight)
        loss.backward()
    finally:
        model.transformer_pass = own
    out = {f"grad_{k}": TR.grad_norm(v) for k, v in parts(model).items()}
    out["grad_at_handoff_vector"] = float(store["sel"].representation.grad[:, H - 1].norm())
    out["grad_after_choice"] = float(store["scores"].grad[:, H - 1].norm())
    q = store["sel"].probs[:, H - 1].detach().max(-1).values
    out["q_max_median"] = float(q.median())
    out["choice_locked_share"] = float((q > 0.999).float().mean())
    return out


def _cached(name, compute, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    out = compute()
    path.write_text(json.dumps(out, indent=2))
    return out


def gradient_at_start(seed=0, force=False):
    """An untrained model 1 (renewal-15, section 16's settings), one batch at the starting temperature; then the same
    after adding noise of size 0.01 to the state table."""
    def compute():
        torch.manual_seed(1234)
        s = TR.settings_for("1", seed=seed)
        model = TR.make_model(s)
        window = s.process_object().sampler(999).draw(s.batch_size, s.total_len)
        E = model.causal_head.state_matrix
        out = dict(rows_all_equal=bool((E - E[0]).abs().max() == 0), as_built=probe(model, window, s.tau_start))
        with torch.no_grad():
            E.add_(0.01 * torch.randn_like(E))
        out["table_noise_0.01"] = probe(model, window, s.tau_start)
        return out
    return _cached(f"gradient_at_start_s{seed}", compute, force)


def first_updates(seed=0, updates=(0, 1, 2, 5, 10, 50, 100, 300), force=False):
    """Real training of model 1 (renewal-15, section 16's settings and data stream), measured before chosen updates:
    the gradients, and how far the state table's rows have moved apart (largest distance of a row from the mean row)."""
    def compute():
        s = TR.settings_for("1", seed=seed)
        torch.set_num_threads(1)
        model = TR.make_model(s)
        opt = TR.make_optimizer(model, s)
        src = s.process_object().sampler(s.train_seed)
        rows = []
        for step in range(max(updates) + 1):
            tau = TR.temperature(step, s.steps, s.tau_start, s.tau_end)
            window = src.draw(s.batch_size, s.total_len)
            if step in updates:
                E = model.causal_head.state_matrix.detach()
                rows.append(dict(update=step, row_spread=float((E - E.mean(0)).norm(dim=1).max()),
                                 **probe(model, window, tau)))
            TR.train_step(model, window, opt, tau, 0.0, s.grad_clip)
        return rows
    return _cached(f"first_updates_s{seed}", compute, force)


def gradient_check(folder, which, batches=8, force=False):
    """A trained run at its final temperature: probe() averaged over `batches` batches of fresh windows; what the
    handoff state tells about the true state (information fraction, 4,096 windows), also for the same model untrained;
    and for the final parameters, Adam's step sizes per part from the saved optimizer state."""
    folder = Path(folder)

    def compute():
        torch.set_num_threads(1)
        model, s, update = TR.load_run(folder, which)
        H = s.history_len
        sampler = s.process_object().sampler(4242)
        rows = [probe(model, sampler.draw(128, s.total_len), s.tau_end) for _ in range(batches)]
        out = dict(update=update, **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
        tok, true = s.process_object().sampler(5151).draw(4096, s.total_len, states=True)
        ids = model.handoff_states(tok[:, :H], s.tau_end)
        untrained = TR.make_model(s)
        untrained.eval()
        ids0 = untrained.handoff_states(tok[:, :H], s.tau_end)
        out.update(handoff_states=int(ids.unique().numel()),
                   handoff_information=TR.information_fraction(ids.numpy(), true[:, H]),
                   handoff_information_untrained=TR.information_fraction(ids0.numpy(), true[:, H]))
        if which == "completed":
            saved = torch.load(folder / "completed.pt", map_location="cpu", weights_only=False)
            opt = TR.make_optimizer(model, s)
            opt.load_state_dict(saved["optimizer_state"])
            for part, params in parts(model).items():
                steps = [(s.lr * st["exp_avg"].abs() / (st["exp_avg_sq"].sqrt() + 1e-8)).flatten()
                         for p in params if (st := opt.state.get(p))]
                out[f"adam_step_median_{part}"] = float(torch.cat(steps).median())
        return out
    return _cached(f"gradient_{folder.name}_{which}", compute, force)


def linear_classifier(xtr, ytr, xte, yte, steps=3000):
    """Test accuracy of a linear classifier (softmax regression on standardised inputs) fitted on (xtr, ytr)."""
    torch.manual_seed(0)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    lin = torch.nn.Linear(xtr.shape[1], int(max(ytr.max(), yte.max())) + 1)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(xte).argmax(-1) == yte).float().mean())


def encoder_output(model, tokens):
    """The vector the Transformer's scoring layer reads at the last history position (after the LayerNorm, if any)."""
    model.eval()
    with torch.no_grad():
        v = model.encoder(tokens)
        if getattr(model, "encoder_norm", None) is not None:
            v = model.encoder_norm(v)
    return v[:, model.cfg.history_len - 1]


def encoder_check(folder, which, force=False):
    """Can a linear classifier read the true state after the history from the Transformer's output there?  Trained
    model against the same model untrained (same starting weights); 8,192 training and 4,096 test windows.  Also how
    far each part's weights moved from their start, as a multiple of their starting length."""
    folder = Path(folder)

    def compute():
        torch.set_num_threads(2)
        model, s, update = TR.load_run(folder, which)
        H = s.history_len
        proc = s.process_object()
        tr_x, tr_st = proc.sampler(6161).draw(8192, H, states=True)
        te_x, te_st = proc.sampler(7171).draw(4096, H, states=True)
        ytr, yte = torch.from_numpy(tr_st[:, H]), torch.from_numpy(te_st[:, H])
        untrained = TR.make_model(s)
        out = dict(update=update,
                   classifier=linear_classifier(encoder_output(model, tr_x), ytr, encoder_output(model, te_x), yte),
                   classifier_untrained=linear_classifier(encoder_output(untrained, tr_x), ytr,
                                                          encoder_output(untrained, te_x), yte),
                   most_common_share=float(np.bincount(te_st[:, H]).max() / len(te_st)))
        for name in ("encoder", "transformer_selector", "gru_cell"):
            a = torch.cat([p.detach().flatten() for p in getattr(model, name).parameters()])
            b = torch.cat([p.detach().flatten() for p in getattr(untrained, name).parameters()])
            out[f"moved_{name}"] = float((a - b).norm() / b.norm())
        return out
    return _cached(f"encoder_{folder.name}_{which}", compute, force)


# ------------------------------------------------------------------------------------------- the whole training
def recorded_history(tag):
    """The validation history the run recorded during training (every 500 updates): the gradient lengths of one
    training batch reaching each part (before clipping), the handoff's states and the score gap there."""
    return TR.history(imported_run(tag))


def traced_run(tag, every=250, force=False, train=True):
    """Train the run `tag` again from its start with its own settings, data stream and noise, recording at every update
    the training loss and the gradient length (before clipping) reaching the whole Transformer (encoder, LayerNorm,
    state-choice layer), the whole GRU (cell and its state-choice layer) and the state table with the output layer;
    and every `every` updates probe() on a separate stream of windows, with the random-number state restored
    afterwards.  Training is unchanged: the final parameters are compared with the saved run's completed.pt.
    Saves runs/model_1_checks/steps_<tag>.csv (every update) and trace_<tag>.csv (every `every`); returns
    (steps, trace, final parameters identical to the saved run's?); with train=False, None when not yet saved."""
    import pandas as pd
    steps_path, trace_path = CACHE / f"steps_{tag}.csv", CACHE / f"trace_{tag}.csv"
    folder = CACHE / f"trace_{tag}"
    if steps_path.exists() and trace_path.exists() and (folder / "same_as_saved.json").exists() and not force:
        return (pd.read_csv(steps_path), pd.read_csv(trace_path),
                json.loads((folder / "same_as_saved.json").read_text())["same"])
    if not train:
        return None
    settings = TR.read_settings(imported_run(tag))
    sampler = settings.process_object().sampler(4242)
    rows, per_step, count = [], [], {"n": 0}
    own_step = TR.train_step

    def length(out, *names):
        return float(np.sqrt(sum(out.get(f"grad_{n}", 0.0) ** 2 for n in names)))

    def traced_step(model, window, optimizer, tau, weight, clip=0.0, record_grads=False):
        if count["n"] % every == 0:
            rng = torch.get_rng_state()
            E = model.causal_head.state_matrix.detach()
            rows.append(dict(update=count["n"], tau=tau, row_spread=float((E - E.mean(0)).norm(dim=1).max()),
                             **probe(model, sampler.draw(settings.batch_size, settings.total_len), tau, weight)))
            torch.set_rng_state(rng)
        out = own_step(model, window, optimizer, tau, weight, clip, record_grads=True)     # (norms are only read)
        per_step.append(dict(update=count["n"] + 1, loss=out["loss"],
                             grad_transformer=length(out, "encoder", "encoder_norm", "transformer_selector"),
                             grad_gru=length(out, "gru_cell", "gru_selector"),
                             grad_table_output=length(out, "causal_head")))
        count["n"] += 1
        if not record_grads:                     # training.train keeps the gradient lengths only when it asked
            out = {k: v for k, v in out.items() if not k.startswith("grad_")}
        return out

    TR.train_step = traced_step
    try:
        model, _ = TR.train(settings, folder=folder, force=True, verbose=False)
    finally:
        TR.train_step = own_step
    saved = torch.load(imported_run(tag) / "completed.pt", map_location="cpu", weights_only=False)["model_state"]
    same = all(torch.equal(v, saved[k]) for k, v in model.state_dict().items())
    (folder / "same_as_saved.json").write_text(json.dumps(dict(same=same)))
    steps, trace = pd.DataFrame(per_step), pd.DataFrame(rows)
    steps.to_csv(steps_path, index=False)
    trace.to_csv(trace_path, index=False)
    return steps, trace, same


# ------------------------------------------------------------------------------------------- the starting point
def start_gradients(n=200, rows=512, force=False):
    """The gradient length at n different starting points (model seeds 0 .. n-1; renewal-15, section 16's settings, the
    starting temperature), reaching the whole Transformer, the whole GRU, and the state table with the output layer:
    with the state table as built (all rows equal) and with its rows drawn at random (randn / sqrt(S)).  Every
    starting point sees the same `rows` windows and the same noise in the choices."""
    def compute():
        torch.set_num_threads(2)
        s = TR.settings_for("1")
        window = s.process_object().sampler(999).draw(rows, s.total_len)
        out = []
        for seed in range(n):
            for start in ("equal", "random"):
                model = TR.make_model(TR.settings_for("1", seed=seed))
                if start == "random":
                    with torch.no_grad():
                        E = model.causal_head.state_matrix
                        E.copy_(torch.randn_like(E) / E.shape[1] ** 0.5)
                torch.manual_seed(12345)
                g = probe(model, window, s.tau_start)
                out.append(dict(seed=seed, start=start,
                                transformer=float(np.hypot(g["grad_encoder"], g["grad_selector"])),
                                gru=g["grad_gru"],
                                table_output=float(np.hypot(g["grad_state_table"], g["grad_emission"]))))
        return out
    import pandas as pd
    return pd.DataFrame(_cached(f"start_gradients_n{n}", compute, force))


def flatness_at_start(seeds=range(20), sizes=(1e-3, 1e-2, 1e-1), force=False):
    """Is the starting point flat?  Move the Transformer's and the GRU's parameters (not the state table or the output
    layer) by a random step of each size (relative to their own length) and record the change of the loss, with the
    same windows and noise before and after the move: largest |change| over the seeds, per start of the state table."""
    def compute():
        torch.set_num_threads(2)
        s = TR.settings_for("1")
        window = s.process_object().sampler(999).draw(256, s.total_len)
        out = []
        for start in ("equal", "random"):
            for size in sizes:
                worst = 0.0
                for seed in seeds:
                    model = TR.make_model(TR.settings_for("1", seed=seed))
                    model.train()
                    if start == "random":
                        with torch.no_grad():
                            E = model.causal_head.state_matrix
                            E.copy_(torch.randn_like(E) / E.shape[1] ** 0.5)
                    moved = [p for k, ps in parts(model).items() if k in ("encoder", "selector", "gru") for p in ps]
                    torch.manual_seed(777)
                    with torch.no_grad():
                        before = float(model.loss(window, s.tau_start, 0.0)[0])
                        gen = torch.Generator().manual_seed(seed)
                        for p in moved:
                            d = torch.randn(p.shape, generator=gen)
                            p.add_(size * d * p.norm() / d.norm().clamp_min(1e-12))
                        torch.manual_seed(777)
                        after = float(model.loss(window, s.tau_start, 0.0)[0])
                    worst = max(worst, abs(after - before))
                out.append(dict(state_table_start=start, step_size=size, largest_loss_change=worst))
        return out
    import pandas as pd
    return pd.DataFrame(_cached("flatness_at_start", compute, force))


# ------------------------------------------------------------------------------------------- figures
TAG_TITLES = {"s0_section16": "section 16, seed 0", "s1_section16": "section 16, seed 1",
              "s2_section16": "section 16, seed 2", "s0_original_setup": "original setup, seed 0"}
BLUE, ORANGE, INK, GREY, PURPLE = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#8c5a9e"


def style():
    """analysis.style() with larger text and lines (every figure of model_1_study.ipynb)."""
    import matplotlib.pyplot as plt
    import analysis as A
    A.style()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 11, "lines.linewidth": 1.8,
                         "figure.titlesize": 14})


def _grid(n, width=6.2, height=3.6):
    import matplotlib.pyplot as plt
    ncols = 2 if n > 1 else 1
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False,
                             layout="constrained")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat)


def plot_whole_training(steps_by_tag, smooth=250, levels=None):
    """One panel per run: the gradient length reaching the whole Transformer and the whole GRU at every update (median
    over `smooth` updates; log scale, left) and the training loss (mean over `smooth` updates; right).  `levels`:
    {label: loss} drawn as grey lines on the loss axis (e.g. the loss of the true probabilities)."""
    style()
    fig, axes = _grid(len(steps_by_tag))
    twins = []
    for ax, (tag, st) in zip(axes, steps_by_tag.items()):
        u = st["update"] / 1000
        ax.plot(u, st["grad_transformer"].clip(lower=1e-14).rolling(smooth, min_periods=1).median(), color=BLUE,
                label="gradient length: whole Transformer")
        ax.plot(u, st["grad_gru"].clip(lower=1e-14).rolling(smooth, min_periods=1).median(), color=ORANGE,
                label="gradient length: whole GRU")
        ax.set_yscale("log")
        ax.set_ylim(1e-14, 1)
        ax.set_xlabel("update (thousands)")
        ax.set_ylabel("gradient length")
        ax.set_title(TAG_TITLES.get(tag, tag))
        twin = ax.twinx()
        twins.append(twin)
        twin.plot(u, st["loss"].rolling(smooth, min_periods=1).mean(), color=INK, lw=1.4,
                  label="training loss (right axis)")
        for (name, value), ls in zip((levels or {}).items(), ("--", ":")):
            twin.axhline(value, color=GREY, lw=1.4, ls=ls, label=name)
        twin.set_ylim(0.46, 0.64)
        twin.set_ylabel("training loss (bits per token)")
        twin.grid(False)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = twins[0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=2)
    fig.suptitle(f"Gradient length and training loss over training\n(each point: median gradient and mean loss of the "
                 f"{smooth} updates up to it; a gradient of 0 is drawn at 1e-14)")
    return fig


def plot_choice_over_training(traces):
    """One panel per run: the gradient arriving at the handoff vector and left after the handoff choice (log, left),
    and the share of windows whose handoff choice is locked (right), every 250 updates."""
    style()
    fig, axes = _grid(len(traces))
    for ax, (tag, t) in zip(axes, traces.items()):
        u = t["update"] / 1000
        ax.plot(u, t["grad_at_handoff_vector"].clip(lower=1e-14), color=INK,
                label="at the state vector the Transformer gives the GRU")
        ax.plot(u, t["grad_after_choice"].clip(lower=1e-14), color=PURPLE,
                label="at the Transformer's 32 state scores (after the choice)")
        ax.set_yscale("log")
        ax.set_ylim(1e-14, 1)
        ax.set_xlabel("update (thousands)")
        ax.set_ylabel("gradient length")
        ax.set_title(TAG_TITLES.get(tag, tag))
        twin = ax.twinx()
        twin.plot(u, t["choice_locked_share"], color=GREY, lw=1.4, ls="--",
                  label="share of windows where the picked state has softmax probability > 0.999")
        twin.set_ylim(-0.02, 1.02)
        twin.set_ylabel("share of windows")
        twin.grid(False)
    h1, l1 = axes[0].get_legend_handles_labels()
    twins = [a for a in fig.axes if a not in list(axes)]
    h2, l2 = twins[0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=1)
    fig.suptitle("Gradient length on both sides of the Transformer's choice of state at position 39 (0 drawn at 1e-14)")
    return fig


def plot_gap_over_training(tags=RUN_TAGS):
    """One panel per run, from the run's own record (every 500 updates): the gap between the two highest scores at the
    handoff (median over the validation windows; log, left) and the number of states the handoff uses (right)."""
    style()
    fig, axes = _grid(len(tags))
    for ax, tag in zip(axes, tags):
        h = recorded_history(tag)
        u = h["update"] / 1000
        ax.plot(u, h["handoff_margin_median"].clip(lower=0.01), color=INK,
                label="gap between the two highest state scores (median over windows)")
        ax.axhline(10, color=GREY, lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.02, 100)
        ax.set_xlabel("update (thousands)")
        ax.set_ylabel("gap")
        ax.set_title(TAG_TITLES.get(tag, tag))
        twin = ax.twinx()
        twin.plot(u, h["handoff_codes"], color=BLUE, lw=1.4, label="different states the Transformer gives the GRU")
        twin.set_ylim(0, 33)
        twin.set_ylabel("different states", color=BLUE)
        twin.grid(False)
    twins = [a for a in fig.axes if a not in list(axes)]
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = twins[0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=1)
    fig.suptitle("The Transformer's choice of state at position 39, every 500 updates (dotted: a gap of 10)")
    return fig


def plot_first_updates(first):
    """The first 300 updates of seed 0: the gradient reaching the whole Transformer and the whole GRU, and how far the
    state table's rows are apart."""
    style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    x = first["update"] + 0.5
    ax.plot(x, np.hypot(first["grad_encoder"], first["grad_selector"]), "o-", color=BLUE, label="gradient: whole Transformer")
    ax.plot(x, first["grad_gru"], "o-", color=ORANGE, label="gradient: whole GRU")
    ax.plot(x, first["row_spread"], "o:", color=GREY, label="largest distance of a state-table row from the mean row")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("update number + 0.5 (so that update 0 fits on a log scale)")
    ax.set_ylabel("length")
    ax.legend(loc="lower right")
    ax.set_title("The first 300 updates (renewal-15, seed 0): gradient length")
    return fig


def plot_start_gradients(data):
    """Histograms of the gradient length at the starting points of start_gradients(), one panel per state-table
    start."""
    style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True, layout="constrained")
    bins = np.logspace(-14, 1, 61)
    for ax, (start, title) in zip(axes, [("equal", "state table as built: all 32 rows equal"),
                                         ("random", "state table rows drawn at random")]):
        d = data[data.start == start]
        for col, label, colour in [("transformer", "whole Transformer", BLUE), ("gru", "whole GRU", ORANGE),
                                   ("table_output", "state table + output layer", GREY)]:
            ax.hist(d[col].clip(lower=1e-14), bins=bins, color=colour, alpha=0.75, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("gradient length at the starting point")
        ax.set_title(title)
    axes[0].set_ylabel(f"number of seeds (of {data.seed.nunique()})")
    axes[1].legend(loc="upper left")
    fig.suptitle(f"Gradient length before the first update, for {data.seed.nunique()} seeds "
                 "(each seed = one set of starting weights; renewal-15, model 1)")
    return fig


def draw_machine(ax, machine_, settings, colour, title=None):
    """A small machine drawn with large text (analysis.draw_machine's layout)."""
    import networkx as nx
    import analysis as A
    from matplotlib.colors import to_hex, to_rgb
    M, pi = np.asarray(machine_["M"]), np.asarray(machine_["pi"])
    n = len(M)
    names = [n[len("extra ("):-1] + " (2nd)" if n.startswith("extra (") else n for n in A._names(machine_, settings)]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    labels, loops = {}, {s: [] for s in range(n)}
    for s in range(n):
        for t in range(n):
            for x in range(M.shape[-1]):
                if M[s, t, x] < 0.01:
                    continue
                lab = f"{x}|{M[s, t, x]:.2f}"
                if s == t:
                    loops[s].append(lab)
                else:
                    labels[(s, t)] = f"{labels[(s, t)]}\n{lab}" if (s, t) in labels else lab
                    G.add_edge(s, t)
    pos = nx.circular_layout(G) if n > 1 else {0: np.zeros(2)}
    tint = to_hex(0.22 * np.array(to_rgb(colour)) + 0.78 * np.array(to_rgb(A.SURFACE)))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=4200, node_color=tint, edgecolors=colour, linewidths=1.6)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, labels={
        s: f"{names[s]}\nπ={pi[s]:.2f}" + ("\n↺ " + ", ".join(loops[s]) if loops[s] else "") for s in range(n)})
    nx.draw_networkx_edges(G, pos, ax=ax, node_size=4200, arrows=True, arrowstyle="-|>", arrowsize=16, width=1.3,
                           edge_color=A.INK2, connectionstyle="arc3,rad=0.15")
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=labels, font_size=11, label_pos=0.35,
                                 connectionstyle="arc3,rad=0.15", font_color=A.INK2,
                                 bbox=dict(boxstyle="round,pad=0.15", fc=A.SURFACE, ec="none", alpha=0.9))
    ax.set_axis_off()
    ax.margins(0.25)
    if title:
        ax.set_title(title)


def plot_checkpoints(grad):
    """grad: one row per run and parameters (gradient_check).  The gradient at the state vector the Transformer gives
    the GRU (orange) and at the Transformer's 32 state scores (purple), joined by a line; 0 is drawn at 1e-14."""
    style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
    for i, r in grad.reset_index(drop=True).iterrows():
        before, after = r["grad_at_handoff_vector"], max(r["grad_after_choice"], 1e-14)
        ax.plot([i, i], [before, after], color="#e6e5e1", lw=4, zorder=1)
        ax.scatter(i, before, s=60, color=ORANGE, zorder=3,
                   label="at the state vector the Transformer gives the GRU" if i == 0 else None)
        ax.scatter(i, after, s=60, color=PURPLE, zorder=3,
                   label="at the Transformer's 32 state scores (after the choice)" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_ylim(1e-14, 1)
    ax.set_xticks(range(len(grad)), [f"{TAG_TITLES.get(r['run'], r['run']).replace(', ', chr(10))}\n{r['parameters']}"
                                     for _, r in grad.iterrows()], fontsize=10)
    ax.set_ylabel("gradient length")
    ax.legend(loc="lower left")
    ax.set_title("Gradient on both sides of the Transformer's choice of state (mean over 8 batches; 0 drawn at 1e-14)")
    return fig


def plot_renewal_curves(groups):
    """groups: {title: {label: run folder}}.  One panel per group: the GRU's validation CE above the true process over
    training, one line per seed; dashed: predicting each token's long-run frequency."""
    style()
    import analysis as A
    fig, axes = _grid(len(groups), width=6.2, height=3.8)
    colours = ["#eb6834", "#2a78d6", "#1baf7a"]
    for ax, (title, runs) in zip(axes, groups.items()):
        for k, (label, folder) in enumerate(runs.items()):
            h = TR.history(folder)
            ax.plot(h["update"] / 1000, h["excess_gru_from_handoff_future"].clip(lower=1e-3), color=colours[k % 3],
                    label=label)
        level = A.reference_levels(TR.read_settings(next(iter(runs.values()))), "validation")["remembers_nothing"]
        ax.axhline(level, color=GREY, ls="--", lw=1.3, label="predicting each token's long-run frequency")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1)
        ax.set_xlabel("update (thousands)")
        ax.set_ylabel("CE above true (bits/token)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=10)
    fig.suptitle("Renewal-15: validation CE above the true process of the GRU started from the Transformer's state")
    return fig
