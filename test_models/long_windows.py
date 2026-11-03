"""
Six models on renewal processes with 30, 60 and 90 states, with windows of 480 tokens (300 history + 180 future): every
update's losses and gradient lengths are recorded, and a run can be stopped and resumed.  (Section 12 of
walkthrough.ipynb; the user's request of 2026-10-30; reports/Long-window runs schedule.md is the full plan.)

The runs (54)
    models     1, 2, 5a, 5b, B, C, each with its model file's DEFAULTS (extra-term weight 0, 0, 0.01, 0.1, 3, 0.01)
    processes  renewal-30, renewal-60, renewal-90: between two 1s there are 0 to N - 1 zeros, each equally likely
               (F = 1/N, as renewal-15, -30 and -60 before)
    windows    history_len 300 + rollout_len 180 = 480 tokens.  context N - 1 (29, 59, 89): the Transformer's own
               predictions are scored from token N on, from where every window's true state is known.  n_states 64,
               128, 192 (about 2N, as sections 10 and 7 used for renewal-30 and renewal-60)
    training   section 16's settings otherwise: 50,000 updates of 128 windows, AdamW (learning rate 0.001, weight decay
               0.0005), gradient clipping at 1, temperature 2.5 -> 0.5, one thread; seeds 0, 1, 2

What is recorded at every update (training_history.csv, one row per update; training.train keeps only the loss terms)
    every term of the model's loss (ce_gru, ce_t1, ce_t2, ce_t3, inferred_entropy, copy_ce, ..., loss)
    ce_t1_measured      models 1 and B: the Transformer's own next-token CE (tokens context + 1 .. 479), from the same
                        noisy state choices as the loss, without gradient (their loss does not train it)
    grad_<name>         the gradient length of each named part of the model before clipping (training.train_step)
    grad_transformer_part, grad_gru_part, grad_shared_part, grad_heads_part, grad_total
                        the Transformer (encoder, its LayerNorm, its scoring layer), the GRU (cell, scoring layer), the
                        shared state table and output layer, the extra heads (pair_head, triple_head), and all parameters
                        together (the length the clipping compares with grad_clip)
Everything else is training.train's: the same windows, starting weights, noise and updates (tests/test_long_windows.py
checks that the parameters and losses are identical), validation_history.csv every 500 updates, best.pt, completed.pt.
The runs are saved where training.train saves them, runs/model_<name>/<process>_w<weight>_n50000_s<seed>_<key>/, so the
walkthrough's helpers of sections 7-9 find them.

Resuming: every SAVE_EVERY updates the whole state of a run (parameters, optimizer, window sampler, noise generator and
the histories so far) is written to resume.pt in its folder.  `train` on an unfinished run goes on from there and ends
exactly as an uninterrupted run would (tested); resume.pt is deleted when the run finishes.

    python long_windows.py plan                      write plans/long_windows.json: the 54 runs, seed 0 first
    python long_windows.py bench [--updates 3]       time one update and one evaluation of each model and process on
                                                     this machine, and estimate the whole plan for --workers
    python long_windows.py queue --workers 8 --memory-gb 12
                                                     run every unfinished run, at most --workers at a time; interrupted
                                                     runs resume; a log for each run in runs/logs/
    python long_windows.py status                    each run: finished, resumable at update k, or not started
    python long_windows.py commands                  one `train` command per unfinished run (for a cluster job array)
    python long_windows.py reports --workers 8       after training: compute and cache every finished run's test report
                                                     and machine (analysis.py), so section 12 of the notebook only reads
    python long_windows.py train --model 2 --process renewal-60 --seed 0
                                                     one run (what queue starts)
Do not change training.py, process.py, causal_state_matrix.py or a model file while runs are going: the run folders'
names contain a hash of them, and a resumed run refuses to go on with changed code.
"""
import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import training as TR                                                       # noqa: E402

MODELS = ["1", "2", "5a", "5b", "B", "C"]
SEEDS = (0, 1, 2)
STEPS = int(os.environ.get("LONG_WINDOWS_STEPS", 50_000))           # override only for a trial: the run folders'
SAVE_EVERY = int(os.environ.get("LONG_WINDOWS_SAVE_EVERY", 1000))   # names record the number of updates
HISTORY, FUTURE = 300, 180
SIZES = {30: 64, 60: 128, 90: 192}    # true states N -> state-table rows K
PROCESSES = {f"renewal-{N}": dict(process="renewal", params={"F": [1.0 / N] * N}, history_len=HISTORY,
                                  rollout_len=FUTURE, context=N - 1, n_states=K) for N, K in SIZES.items()}
PLAN = HERE / "plans" / "long_windows.json"
PARTS = {"transformer": ("encoder", "encoder_norm", "transformer_selector"), "gru": ("gru_cell", "gru_selector"),
         "shared": ("causal_head",), "heads": ("pair_head", "triple_head")}
MEMORY_GB = 2.5                       # one run's peak memory, with room (measured 1.0-2.0 GB on macOS)
ENV = dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MallocLargeExpandedCacheThreshold="1")     # the last: macOS only (main.py train-many), harmless elsewhere


# ================================================================================================ the runs
def shown(folder):
    """A run folder relative to test_models/ when it is inside it (runs may live elsewhere: TEST_MODELS_RUNS)."""
    try:
        return str(Path(folder).resolve().relative_to(HERE.resolve()))
    except ValueError:
        return str(folder)


def settings_of(model, process, seed, steps=STEPS, **changes):
    """The settings of one run of the plan."""
    return TR.settings_for(model, **dict(PROCESSES[process], seed=seed, steps=steps, **changes))


def jobs(models=None, processes=None, seeds=None):
    """Every run of the plan (or those of the given models, processes and seeds), seed 0 first (so one seed of
    everything finishes first); within a seed the longest process first."""
    return [dict(model=m, process=p, seed=s) for s in SEEDS for p in reversed(list(PROCESSES)) for m in MODELS
            if (models is None or m in models) and (processes is None or p in processes)
            and (seeds is None or s in seeds)]


def write_plan(path=PLAN):
    """plans/long_windows.json: each run as {"model", "process", "seed", "settings" (the changes to settings_for),
    "memory_gb", "folder"}."""
    out = []
    for j in jobs():
        s = settings_of(j["model"], j["process"], j["seed"])
        out.append(dict(j, settings=dict(PROCESSES[j["process"]], seed=j["seed"], steps=STEPS), memory_gb=MEMORY_GB,
                        folder=shown(TR.run_folder(s))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


# ================================================================================================ recording
def instrument(model, always=False):
    """Add `ce_t1_measured` to the loss terms of a model with a Transformer whose loss has no `ce_t1` (models 1 and B;
    every such model when always=True): the CE of the Transformer's own next-token predictions of tokens context + 1 ..
    T - 1, from the same state choices (and noise) the loss used, without gradient.  The loss is unchanged."""
    if not hasattr(model, "transformer_pass"):
        return model
    original_pass, original_loss = model.transformer_pass, model.loss

    def transformer_pass(tokens, tau):
        out = original_pass(tokens, tau)
        model.__dict__["_selection"] = out[1]
        return out

    def loss(window, tau, weight):
        total, terms, ids = original_loss(window, tau, weight)
        sel = model.__dict__.pop("_selection", None)
        if sel is not None and (always or "ce_t1" not in terms):
            C, T = model.cfg.context, window.shape[1]
            with torch.no_grad():
                terms["ce_t1_measured"] = TR.prediction_loss(
                    model.causal_head.emission(sel.representation[:, C:T - 1]), window[:, C + 1:T])
        return total, terms, ids

    model.transformer_pass, model.loss = transformer_pass, loss
    return model


def part_lengths(grads):
    """From the gradient lengths of the named parts (grad_<name>): each group's length (the root of the summed squares
    of its members') and the length over all parameters."""
    out = {}
    for part, names in PARTS.items():
        values = [grads[f"grad_{n}"] for n in names if f"grad_{n}" in grads]
        if values:
            out[f"grad_{part}_part"] = math.sqrt(sum(v * v for v in values))
    out["grad_total"] = math.sqrt(sum(v * v for v in grads.values()))
    return out


# ================================================================================================ training
def _save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(settings, folder=None, save_every=SAVE_EVERY, verbose=True):
    """training.train with every update's loss terms and gradient lengths kept, and resume points.  Returns the run
    folder (a finished run is left as it is)."""
    folder = Path(folder) if folder is not None else TR.run_folder(settings)
    if (folder / "completed.pt").exists():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    meta = dict(settings.to_dict(), code=TR.code_fingerprint(settings.model), trainer="long_windows.py")
    resume_file = folder / "resume.pt"
    if resume_file.exists():
        saved = json.loads((folder / "settings.json").read_text())
        if saved != meta:
            raise ValueError(f"{folder}: the settings or the code changed since this run started; not resuming")
    else:
        (folder / "settings.json").write_text(json.dumps(meta, indent=2))
    torch.set_num_threads(settings.threads)
    model = TR.make_model(settings)
    optimizer = TR.make_optimizer(model, settings)
    instrument(model)
    process = settings.process_object()
    schedule_steps = settings.schedule_steps or settings.steps
    train_source = process.sampler(settings.train_seed)
    validation, validation_states = TR.validation_windows(settings)
    validation_exact = TR.exact_ce_bits(process, validation)
    T, metric = settings.total_len, TR.main_metric(settings.model)
    training, hist, best, first, before = [], [], None, 0, 0.0
    if resume_file.exists():
        r = torch.load(resume_file, map_location="cpu", weights_only=False)
        model.load_state_dict(r["model_state"])
        optimizer.load_state_dict(r["optimizer_state"])
        train_source.rng.bit_generator.state = r["rng_state"]
        torch.set_rng_state(r["torch_rng_state"])
        training, hist, best, first, before = r["training"], r["hist"], tuple(r["best"]), r["update"], r["seconds"]
        if verbose:
            print(f"resumed {folder.name} at update {first:,}", flush=True)
    start = time.perf_counter() - before

    def record(update, tau, grads):                                      # training.train's record
        nonlocal best
        values = TR.evaluate(model, settings, validation, validation_states, tau, validation_exact)
        values.update(update=update, tau=tau, seconds=time.perf_counter() - start, **grads)
        hist.append(values)
        pd.DataFrame(hist).to_csv(folder / "validation_history.csv", index=False)
        (folder / "progress.json").write_text(json.dumps(values, indent=2))
        if best is None or values[metric] < best[0]:
            best = (values[metric], update)
            torch.save(dict(settings=settings.to_dict(), model_state=model.state_dict(), update=update, metric=metric,
                            value=values[metric]), folder / "best.pt")
        if verbose and (update % 5000 == 0 or update == settings.steps):
            TR._progress(settings, update, values, metric, tau)

    if first == 0:
        record(0, settings.tau_start, {})
    for step in range(first, settings.steps):
        tau = TR.temperature(step, schedule_steps, settings.tau_start, settings.tau_end)
        if settings.lr_warmup:
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * min(1.0, (step + 1) / settings.lr_warmup)
        window = train_source.draw(settings.batch_size, T)
        is_eval = (step + 1) % settings.eval_every == 0 or step + 1 == settings.steps
        out = TR.train_step(model, window, optimizer, tau, settings.weight, settings.grad_clip, record_grads=True)
        if not math.isfinite(out["loss"]):
            raise RuntimeError(f"non-finite loss at update {step + 1}")
        grads = {k: v for k, v in out.items() if k.startswith("grad_")}
        training.append(dict(update=step + 1, tau=tau, **{k: v for k, v in out.items() if not k.startswith("grad_")},
                             **grads, **part_lengths(grads)))
        if is_eval:
            record(step + 1, tau, grads)
        if save_every and (step + 1) % save_every == 0 and step + 1 < settings.steps:
            _save(dict(update=step + 1, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                       rng_state=train_source.rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
                       training=training, hist=hist, best=best, seconds=time.perf_counter() - start), resume_file)
    torch.save(dict(settings=settings.to_dict(), model_state=model.state_dict(), updates=settings.steps,
                    seconds=time.perf_counter() - start, best_update=best[1], best_value=best[0], metric=metric,
                    optimizer_state=optimizer.state_dict(), rng_state=train_source.rng.bit_generator.state,
                    torch_rng_state=torch.get_rng_state(), torch_version=str(torch.__version__)),
               folder / "completed.pt")
    pd.DataFrame(training).to_csv(folder / "training_history.csv", index=False)
    resume_file.unlink(missing_ok=True)
    return folder


# ================================================================================================ running the plan
def _folder(job):
    return TR.run_folder(settings_of(job["model"], job["process"], job["seed"]))


def _last_change(folder):
    files = [folder / n for n in ("progress.json", "resume.pt", "completed.pt")]
    return max((f.stat().st_mtime for f in files if f.exists()), default=None)


def status_table(**subset):
    """One row per run of the plan: finished, resumable (with the update of its resume point), started (no resume
    point yet) or not started; the last update recorded; minutes since its files last changed."""
    rows = []
    for j in jobs(**subset):
        folder = _folder(j)
        last = _last_change(folder)
        if (folder / "completed.pt").exists():
            state, update = "finished", STEPS
        elif (folder / "resume.pt").exists():
            state = "resumable"
            update = int(torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)["update"])
        elif (folder / "progress.json").exists():
            state, update = "started", int(json.loads((folder / "progress.json").read_text())["update"])
        else:
            state, update = "not started", 0
        rows.append(dict(j, state=state, update=update,
                         minutes_since_change=None if last is None else round((time.time() - last) / 60, 1),
                         folder=shown(folder)))
    return pd.DataFrame(rows)


def queue(workers, memory_gb, adopt_minutes=90.0, poll=30.0, **subset):
    """Run every unfinished run of the plan, each in its own process with one thread: at most `workers` at a time and
    within `memory_gb` (MEMORY_GB each).  A run whose files changed in the last `adopt_minutes` is taken to be running
    in another process and is not started again.  A failed run is reported (run queue again to resume it).  Logs:
    runs/logs/long_windows__<run folder>.log (appended to)."""
    logs = TR.RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **ENV)

    def say(text):
        print(f"{time.strftime('%Y-%m-%d %H:%M')} {text}", flush=True)

    waiting, running, failed = [], [], []
    for j in jobs(**subset):
        folder = _folder(j)
        last = _last_change(folder)
        if (folder / "completed.pt").exists():
            continue
        if last is not None and (time.time() - last) / 60 < adopt_minutes:
            running.append((j, folder, None, None))
            say(f"adopted  {folder.parent.name}/{folder.name} (changed {(time.time() - last) / 60:.0f} min ago)")
        else:
            waiting.append((j, folder))
    say(f"{len(waiting)} runs to start, {len(running)} adopted; at most {workers} at a time within {memory_gb:g} GB")
    while waiting or running:
        for j, folder in list(waiting):
            if len(running) < workers and (len(running) + 1) * MEMORY_GB <= memory_gb:
                log = open(logs / f"long_windows__{folder.parent.name}__{folder.name}.log", "a")
                cmd = [sys.executable, str(HERE / "long_windows.py"), "train", "--model", j["model"], "--process",
                       j["process"], "--seed", str(j["seed"])]
                running.append((j, folder, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE),
                                                            env=env), log))
                waiting.remove((j, folder))
                say(f"started  {folder.parent.name}/{folder.name}")
        time.sleep(poll)
        for item in list(running):
            j, folder, proc, log = item
            if proc is None:                                             # adopted: watch its files
                last = _last_change(folder)
                if (folder / "completed.pt").exists():
                    running.remove(item)
                    say(f"finished {folder.parent.name}/{folder.name}")
                elif last is None or (time.time() - last) / 60 > 2 * adopt_minutes:
                    running.remove(item)
                    waiting.append((j, folder))                          # stopped elsewhere: resume it here
                    say(f"no change for {2 * adopt_minutes:g} min, queued again: {folder.parent.name}/{folder.name}")
                continue
            if proc.poll() is not None:
                running.remove(item)
                log.close()
                if proc.returncode == 0:
                    say(f"finished {folder.parent.name}/{folder.name}")
                else:
                    failed.append(folder)
                    say(f"FAILED (exit {proc.returncode}) {folder.parent.name}/{folder.name}; see its log")
    say(f"all done; {len(failed)} failed" + "".join(f"\n  {f}" for f in failed))
    return failed


def commands(python="python", **subset):
    """One shell command per unfinished run of the plan (for a cluster's job array), run from test_models/."""
    out = []
    for j in jobs(**subset):
        if not (_folder(j) / "completed.pt").exists():
            out.append(" ".join([f"{k}={v}" for k, v in ENV.items()] + [python, "long_windows.py", "train",
                                 "--model", j["model"], "--process", j["process"], "--seed", str(j["seed"])]))
    return out


def report_one(folder, which=("latest",)):
    """analysis.report (the held-out results) at each of `which` and analysis.machine at the final parameters of one
    finished run, cached in its folder (report_<which>.json, machine_latest_<predictor>.json), as section 12 reads them."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import analysis as A
    torch.set_num_threads(1)
    for w in which:
        A.report(folder, w)
    A.machine(folder, "latest")
    return str(folder)


def make_reports(workers=4, which=("latest",), **subset):
    """report_one for every finished run of the plan (or a subset), `workers` at a time."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    folders = [f for f in (_folder(j) for j in jobs(**subset)) if (f / "completed.pt").exists()]
    print(f"{len(folders)} finished runs to report on", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(report_one, f, tuple(which)): f for f in folders}
        for done in as_completed(futures):
            print(f"{time.strftime('%Y-%m-%d %H:%M')} reported {shown(done.result())}", flush=True)


# ================================================================================================ timing
def _peak_gb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 30 if sys.platform == "darwin" else peak / 2 ** 20      # bytes on macOS, KiB on Linux


def time_one(model, process, updates=3):
    """Seconds per training update (the smallest of `updates` after one warm-up update) and per evaluation of the
    validation windows, and the peak memory of this process, for one model and process."""
    s = settings_of(model, process, 0)
    torch.set_num_threads(1)
    m = instrument(TR.make_model(s))
    opt = TR.make_optimizer(m, s)
    source = s.process_object().sampler(12345)
    times = []
    for i in range(updates + 1):
        w = source.draw(s.batch_size, s.total_len)
        t = time.perf_counter()
        TR.train_step(m, w, opt, 1.0, s.weight, s.grad_clip, record_grads=True)
        if i:
            times.append(time.perf_counter() - t)
    val, val_states = TR.validation_windows(s)
    exact = TR.exact_ce_bits(s.process_object(), val)
    t = time.perf_counter()
    TR.evaluate(m, s, val, val_states, s.tau_end, exact)
    return dict(model=model, process=process, update_seconds=min(times), evaluation_seconds=time.perf_counter() - t,
                peak_gb=_peak_gb())


def estimate_hours(update_seconds, evaluation_seconds, steps=STEPS, eval_every=500):
    return (steps * update_seconds + (steps // eval_every + 1) * evaluation_seconds) / 3600


def makespan(hours, workers):
    """Hours until the last run ends when the runs (in plan order) start as workers free up."""
    free = [0.0] * workers
    for h in hours:
        i = int(np.argmin(free))
        free[i] += h
    return max(free)


def bench(updates=3, workers=8):
    """Time every model and process here (each in a fresh process, one at a time), print each run's estimated hours,
    and the hours until the whole plan ends with `workers` runs at a time.  Runs that share the machine are slower
    than one alone; on the M3 Pro used to write this, 8 at a time made each about twice as slow."""
    rows = []
    for p in PROCESSES:
        for m in MODELS:
            out = subprocess.run([sys.executable, str(HERE / "long_windows.py"), "time-one", "--model", m, "--process",
                                  p, "--updates", str(updates)], capture_output=True, text=True, cwd=str(HERE),
                                 env=dict(os.environ, **ENV))
            if out.returncode:
                raise RuntimeError(out.stderr)
            rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
            print(f"{m:>3} {p}: {rows[-1]['update_seconds']:.2f} s per update, {rows[-1]['evaluation_seconds']:.1f} s "
                  f"per evaluation, peak {rows[-1]['peak_gb']:.2f} GB", flush=True)
    table = pd.DataFrame(rows)
    table["hours_alone"] = [estimate_hours(r.update_seconds, r.evaluation_seconds) for r in table.itertuples()]
    hours = {(r.model, r.process): r.hours_alone for r in table.itertuples()}
    plan_hours = [hours[(j["model"], j["process"])] for j in jobs()]
    print(table.round(2).to_string(index=False))
    print(f"one run alone: {min(plan_hours):.0f}-{max(plan_hours):.0f} h; all {len(plan_hours)} runs one after "
          f"another: {sum(plan_hours):.0f} h")
    for w in sorted({1, workers, 18, 27, 54}):
        print(f"  {w:>2} at a time: the plan ends after {makespan(plan_hours, w):.0f} h if runs did not slow each other "
              f"down; peak memory about {w * table.peak_gb.max():.0f} GB")
    return table


# ================================================================================================ reading and figures
def history(folder):
    """Every update's recorded values of a run (training_history.csv, with any longer training)."""
    folder = Path(folder)
    files = [folder / "training_history.csv"] + sorted(folder.glob("training_history_extended_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)


def term_table(runs, last=10_000):
    """The mean of every recorded loss term and gradient length over the last `last` updates of each run
    (runs: {name: folder})."""
    rows = []
    for name, folder in runs.items():
        h = history(folder)
        h = h[h["update"] > h["update"].max() - last]
        rows.append(dict(run=name, **h.drop(columns=["update", "tau"]).mean().to_dict()))
    return pd.DataFrame(rows).set_index("run")


def plot_parts(runs, title="", smooth=250):
    """For each model (columns): top, the GRU's CE (ce_gru) and the Transformer's own next-token CE (ce_t1, or
    ce_t1_measured where the loss does not train it) at every update; bottom, the gradient length of the GRU part and of
    the Transformer part (log scale).  Running means over `smooth` updates; seeds solid, dashed, dotted.  The grey line
    in the top row is the process's entropy rate: the CE of a model that knows the true state.  runs: {"<model>
    s<seed>": folder}, as runs_of in the walkthrough."""
    import matplotlib.pyplot as plt
    blue, orange, grey = "#2a78d6", "#eb6834", "#898781"
    by_model = {}
    for name, folder in runs.items():
        model, seed = name.rsplit(" s", 1)
        by_model.setdefault(model, []).append((int(seed), folder))
    models = [m for m in MODELS if m in by_model] + [m for m in by_model if m not in MODELS]
    fig, axes = plt.subplots(2, len(models), figsize=(3.1 * len(models), 5.2), sharex=True, squeeze=False,
                             layout="constrained")
    styles = {0: "-", 1: "--", 2: ":"}
    for c, model in enumerate(models):
        top, bottom = axes[0][c], axes[1][c]
        for seed, folder in sorted(by_model[model]):
            h = history(folder)
            u = h["update"].to_numpy()
            mean = lambda col: h[col].rolling(smooth, min_periods=1).mean().to_numpy()
            ls = styles.get(seed, "-.")
            transformer_ce = "ce_t1" if "ce_t1" in h else "ce_t1_measured" if "ce_t1_measured" in h else None
            if "ce_gru" in h:
                top.plot(u, mean("ce_gru"), ls=ls, color=blue, lw=1.1, label="GRU" if seed == 0 else None)
                bottom.plot(u, mean("grad_gru_part"), ls=ls, color=blue, lw=1.1, label="GRU" if seed == 0 else None)
            if transformer_ce:
                trained = transformer_ce == "ce_t1"
                top.plot(u, mean(transformer_ce), ls=ls, color=orange, lw=1.1,
                         label=("Transformer" if trained else "Transformer (not trained by the loss)")
                         if seed == 0 else None)
                bottom.plot(u, mean("grad_transformer_part"), ls=ls, color=orange, lw=1.1,
                            label="Transformer" if seed == 0 else None)
        s = TR.read_settings(sorted(by_model[model])[0][1])
        top.axhline(s.process_object().analytics()["entropy_rate"], color=grey, lw=0.8)
        top.set_title(f"model {model}")
        bottom.set_yscale("log")
        bottom.set_xlabel("update")
        top.legend(fontsize=7, loc="upper right")
        bottom.legend(fontsize=7, loc="upper right")
    axes[0][0].set_ylabel("CE (bits per token)")
    axes[1][0].set_ylabel("gradient length (before clipping)")
    fig.suptitle(f"{title}: the GRU's and the Transformer's loss (top) and gradient length (bottom) at every update; "
                 f"running means over {smooth} updates; seeds solid, dashed, dotted", fontsize=10)
    return fig


# ================================================================================================ command line
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    st = sub.add_parser("status")
    cm = sub.add_parser("commands")
    cm.add_argument("--python", default="python")
    q = sub.add_parser("queue")
    rp = sub.add_parser("reports")
    rp.add_argument("--workers", type=int, default=4)
    rp.add_argument("--which", default="latest", help="latest, or latest,best")
    for parser in (st, cm, q, rp):                     # a subset of the plan; default: every run
        parser.add_argument("--models", default=None, help="e.g. 1,5b")
        parser.add_argument("--processes", default=None, help="e.g. renewal-30,renewal-90")
        parser.add_argument("--seeds", default=None, help="e.g. 0")
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--memory-gb", type=float, default=12.0)
    q.add_argument("--adopt-minutes", type=float, default=90.0)
    q.add_argument("--poll", type=float, default=30.0)
    b = sub.add_parser("bench")
    b.add_argument("--updates", type=int, default=3)
    b.add_argument("--workers", type=int, default=8)
    o = sub.add_parser("time-one")
    o.add_argument("--model", required=True)
    o.add_argument("--process", required=True, choices=list(PROCESSES))
    o.add_argument("--updates", type=int, default=3)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True, choices=MODELS)
    t.add_argument("--process", required=True, choices=list(PROCESSES))
    t.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    subset = {}
    if a.command in ("status", "commands", "queue", "reports"):
        subset = dict(models=a.models.split(",") if a.models else None,
                      processes=a.processes.split(",") if a.processes else None,
                      seeds=[int(s) for s in a.seeds.split(",")] if a.seeds else None)
    if a.command == "plan":
        out = write_plan()
        print(f"wrote {PLAN.relative_to(HERE)}: {len(out)} runs")
    elif a.command == "status":
        table = status_table(**subset)
        print(table.to_string(index=False))
        print(table.groupby("state").size().to_string())
    elif a.command == "commands":
        print("\n".join(commands(a.python, **subset)))
    elif a.command == "queue":
        sys.exit(1 if queue(a.workers, a.memory_gb, a.adopt_minutes, a.poll, **subset) else 0)
    elif a.command == "reports":
        make_reports(a.workers, a.which.split(","), **subset)
    elif a.command == "bench":
        bench(a.updates, a.workers)
    elif a.command == "time-one":
        print(json.dumps(time_one(a.model, a.process, a.updates)))
    elif a.command == "train":
        folder = train(settings_of(a.model, a.process, a.seed))
        print(f"done: {folder}", flush=True)


if __name__ == "__main__":
    main()
