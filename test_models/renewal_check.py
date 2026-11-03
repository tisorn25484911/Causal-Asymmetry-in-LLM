"""
Model 1 on renewal-15, -30 and -60 with the best training settings (section 6 of model_1_study.ipynb; the user's request
of 2026-10-31: use the best training settings of the search on renewal-15, 30 and 60 with model 1, searching again if
needed, recording the gradient and the loss over training and the gradient at the starting points).

The coin search (coin_check.py) found nothing better than section 16's settings, and those learn renewal-15 on one seed
of three.  So the search is repeated on renewal-15, and the best settings found then train model 1 on renewal-15, -30
and -60.  A pilot (8 configurations, seed 0, 10,000 updates) ended at 0.042-0.051 bits above the true process on all 8
('remembers nothing': 0.055), so the search uses 50,000 updates, as section 16, with SEARCH_NAMES.

Processes       renewal-N: between two 1s there are 0 to N - 1 zeros, each equally likely (F = 1/N).  Window sizes as
                before: renewal-15 40 + 20 tokens and 32 table rows (section 16); renewal-30 60 + 30 and 64 rows
                (walkthrough section 10); renewal-60 120 + 60 and 128 rows (walkthrough section 7).  A configuration
                that changes a size (table rows K, future tokens L) changes it in proportion N / 15.
Configurations  coin_check.CONFIGS for model 1 with AdamW, and SGD without weight decay; 4 table rows (fewer than
                renewal-15's 15 states) are replaced by 16 and 64.  The three SGD configurations with weight decay are
                left out: on the coin they fail from the weight decay alone (it shrinks every weight by lr x 5e-4 per
                update, 300 times AdamW's).
Recording       every run records at every update the training loss and the gradient length (before clipping) reaching
                the whole Transformer (encoder, LayerNorm, scoring layer), the whole GRU (cell, scoring layer) and the
                state table with the output layer (steps.csv); and every 250 updates model_1_checks.probe on a separate
                stream of windows (trace.csv), with the random-number state restored, so training is unchanged
                (model_1_checks.traced_run checked this on four 50,000-update runs; test_tracing() checks it here).
After training  analysis.report and analysis.machine (test windows, the GRU's machine) and coin_check.diagnose (the
                Transformer's choice, the linear classifier on its vectors), at the final and the best parameters.
Runs            runs/renewal_check/<configuration>_renewal-<N>_n<updates>_s<seed>/: training.train's files, plus
                renewal_check.json (the configuration), steps.csv, trace.csv.

    python renewal_check.py run --names default,table_random --N 15 --steps 10000 --seeds 0 --workers 8
    python renewal_check.py run --stage search --workers 9        (SEARCH_NAMES, renewal-15, 50,000 updates, 3 seeds)
    python renewal_check.py run --stage combined --workers 11     (COMBINED_NAMES: the second stage, as the search)
    python renewal_check.py starts --names default --N 15,30,60   (the gradient at 200 starting points)
    python renewal_check.py status
Do not change training.py, process.py, causal_state_matrix.py or model_1.py while runs are going.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import training as TR                                                           # noqa: E402
import coin_check as CC                                                         # noqa: E402
import model_1_checks as C1                                                     # noqa: E402

ROOT = TR.RUNS / "renewal_check"
SEEDS = (0, 1, 2)
PILOT_STEPS = 10_000
FINAL_STEPS = 50_000
PILOT_NAMES = ["default", "table_random", "tau_5_1", "tau_const1", "lr_3e-3", "wd_5e-3", "no_noise", "K_64"]
TRACE_EVERY = 250
SIZES = {15: dict(), 30: dict(history_len=60, rollout_len=30, n_states=64, context=29),
         60: dict(history_len=120, rollout_len=60, n_states=128, context=59)}
SCALED = ("n_states", "rollout_len")                    # sizes a configuration may change, scaled by N / 15

# ------------------------------------------------------------------------------------------- configurations
LEFT_OUT = {"model_2", "feedback_GRU", "discrete_Transformer", "K_4", "sgd_m0_lr0.3", "sgd_m0.9_lr0.03",
            "sgd_m0.9_lr0.3", "even_model_1", "even_feedback_GRU"}
CONFIGS = {n: c for n, c in CC.CONFIGS.items() if n not in LEFT_OUT}
CONFIGS["K_16"] = CC.config(n_states=16, group="table rows K")
CONFIGS["K_64"] = CC.config(n_states=64, group="table rows K")
LABELS = dict(CC.LABELS, K_16="16 rows in the state table (not 32)", K_64="64 rows in the state table (not 32)",
              batch_32="batch of 32 windows (4x the updates: the same windows)",
              batch_512="batch of 512 windows (1/4 of the updates: the same windows)")
# The 50,000-update search: one change at a time from section 16's settings.  Left out, because on the coin they only
# learned more slowly (renewal-15 already needs about 30,000 updates at section 16's settings): learning rate 0.0001
# (5 times slower on the coin) and SGD without weight decay (4-9 times slower).  Also left out, to keep the search to
# about 11 hours: Adam's beta2 and beta1 0.5 (no effect on the coin) and the learning rate x beta1 grid.
# default and original_setup are section 16's runs (adopt_imported).
SEARCH_NAMES = ["table_random", "lr_3e-4", "lr_3e-3", "lr_1e-2", "beta1_0", "beta1_0.99", "wd_0", "wd_5e-3", "clip_off",
                "tau_const1", "tau_2.5_0.1", "tau_5_1", "no_noise", "no_layernorm", "batch_32", "batch_512", "K_16",
                "K_64", "L_2", "L_5"]
IMPORTED = {"default": "section16", "original_setup": "original_setup"}


def combine(*names):
    """Several configurations at once: their changed settings, optimizer and state-table start together."""
    c = CC.config(group="combinations")
    for n in names:
        c["settings"].update(CONFIGS[n]["settings"])
        c["optimizer"].update({k: v for k, v in CONFIGS[n]["optimizer"].items() if v != CC.ADAM.get(k)})
        if CONFIGS[n]["table_start"] != "equal":
            c["table_start"] = CONFIGS[n]["table_start"]
    return c


# The second stage: the changes that learned renewal-15 in the search and then lost it (learning rate 0.003, random
# table rows) together with those that kept the Transformer's choice from becoming fixed (temperature held at 1 or
# 5 to 1, no noise in the choice).
LEARNERS, KEEPERS = ["lr_3e-3", "table_random"], ["tau_const1", "tau_5_1", "no_noise"]
COMBINED_NAMES = [f"{a}+{b}" for a in LEARNERS for b in KEEPERS] + ["lr_3e-3+table_random"]
for _name in COMBINED_NAMES:
    CONFIGS[_name] = combine(*_name.split("+"))
    LABELS[_name] = " + ".join(LABELS[n] for n in _name.split("+"))


# Section 7's pairs on renewal-30, chosen the same way from the search on renewal-30: the changes that learned most of
# it at their best parameters and then lost it (learning rate 0.003, the batch of 32) with those that kept the
# Transformer's choice from becoming fixed there on some seeds (fixed on 26%, 58%, 88% of windows at the end with the
# temperature held at 1; 0%, 81%, 100% with no noise in the choice, whose GRU machines also keep 3-5 states, and whose
# training reads the states the way the tests do), and the two learners together.  The temperature from 5 to 1 left
# the choice fixed on 81-93% of windows, against 95-96% at section 16's settings.
LEARNERS_30, KEEPERS_30 = ["lr_3e-3", "batch_32"], ["tau_const1", "no_noise"]
PAIRS_30 = [f"{a}+{b}" for a in LEARNERS_30 for b in KEEPERS_30] + ["lr_3e-3+batch_32"]
for _name in PAIRS_30:
    if _name not in CONFIGS:
        CONFIGS[_name] = combine(*_name.split("+"))
        LABELS[_name] = " + ".join(LABELS[n] for n in _name.split("+"))


SHORT = {"batch_32": "batch of 32 (4x the updates)", "batch_512": "batch of 512 (1/4 of the updates)",
         "no_noise": "no noise in the choice of state"}


def label(name, N=15):
    """LABELS[name] on renewal-N: the sizes a configuration changes (SCALED) are scaled by N / 15 (6.1), and a few
    long labels are shortened (SHORT)."""
    if N == 15:
        return LABELS.get(name, name)
    words = []
    for n in name.split("+"):
        changes = CONFIGS[n]["settings"]
        if "n_states" in changes:
            words.append(f"{round(changes['n_states'] * N / 15)} rows in the state table (not {SIZES[N]['n_states']})")
        elif "rollout_len" in changes:
            words.append(f"{round(changes['rollout_len'] * N / 15)} future tokens per window "
                         f"(not {SIZES[N]['rollout_len']})")
        else:
            words.append(SHORT.get(n, LABELS.get(n, n)))
    return " + ".join(words)


def steps_for(name, steps):
    """A configuration's own number of updates: every batch size sees the same windows, `steps` x 128 (a batch of 512
    in steps / 4 updates, a batch of 32 in steps x 4; on the coin the batch of 32 kept 10,000 updates)."""
    return steps * 128 // CONFIGS[name]["settings"].get("batch_size", 128)


def settings_of(name, N, seed, steps):
    changes = dict(CONFIGS[name]["settings"])
    for k in SCALED:
        if k in changes:
            changes[k] = round(changes[k] * N / 15)
    return TR.settings_for(CONFIGS[name]["model"], process="renewal", params={"F": [1.0 / N] * N},
                           **dict(SIZES[N], **changes), seed=seed, steps=steps_for(name, steps))


def folder_of(name, N, seed, steps):
    return ROOT / f"{name}_renewal-{N}_n{steps_for(name, steps)}_s{seed}"


# ------------------------------------------------------------------------------------------- training
def traced_train(name, N, seed, steps, verbose=True):
    """Train one run with training.train (the configuration's optimizer and state-table start, coin_check.patched),
    recording steps.csv and trace.csv as described above.  A finished run is left as it is."""
    import pandas as pd
    c = CONFIGS[name]
    settings, folder = settings_of(name, N, seed, steps), folder_of(name, N, seed, steps)
    if all((folder / f).exists() for f in ("completed.pt", "steps.csv", "trace.csv")):
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "renewal_check.json").write_text(json.dumps(dict(name=name, N=N, **c), indent=2))
    sampler = settings.process_object().sampler(4242)
    rows, per_step, count = [], [], {"n": 0}
    own_step = TR.train_step

    def length(out, *names):
        return float(np.sqrt(sum(out.get(f"grad_{n}", 0.0) ** 2 for n in names)))

    def traced_step(model, window, optimizer, tau, weight, clip=0.0, record_grads=False):
        if count["n"] % TRACE_EVERY == 0:
            rng = torch.get_rng_state()
            E = model.causal_head.state_matrix.detach()
            rows.append(dict(update=count["n"], tau=tau, row_spread=float((E - E.mean(0)).norm(dim=1).max()),
                             **C1.probe(model, sampler.draw(settings.batch_size, settings.total_len), tau, weight)))
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

    with CC.patched(c["optimizer"], c["table_start"]):
        TR.train_step = traced_step
        try:
            TR.train(settings, folder=folder, force=True, verbose=verbose)
        finally:
            TR.train_step = own_step
    pd.DataFrame(rows).to_csv(folder / "trace.csv", index=False)
    pd.DataFrame(per_step).to_csv(folder / "steps.csv", index=False)
    return folder


def after_training(folder):
    """The held-out results at the final and the best parameters (cached in the run folder)."""
    import analysis as A
    torch.set_num_threads(1)
    for which in ("latest", "best"):
        A.report(folder, which)
        A.machine(folder, which)
        CC.diagnose(folder, which)


def _worker(name, N, seed, steps):
    log = ROOT / "logs" / f"{folder_of(name, N, seed, steps).name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with open(log, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        after_training(traced_train(name, N, seed, steps))
    return name, N, seed, time.perf_counter() - start


def done(name, N, seed, steps):
    return (folder_of(name, N, seed, steps) / "diagnostics_best.json").exists()


def run(jobs, workers):
    """jobs: (name, N, seed, steps); the unfinished ones run, the largest N and longest first."""
    jobs = sorted([j for j in jobs if not done(*j)], key=lambda j: (-j[1], -steps_for(j[0], j[3])))
    print(f"{len(jobs)} runs, {workers} at a time", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, *j) for j in jobs]
        for f in as_completed(futures):
            name, N, seed, seconds = f.result()
            print(f"done {name} renewal-{N} s{seed}: {seconds / 60:.1f} min", flush=True)


def test_tracing(name="table_random", N=15, steps=300):
    """Tracing changes nothing: the parameters after `steps` updates with and without it are identical."""
    import tempfile
    c = CONFIGS[name]
    settings = settings_of(name, N, 0, steps)
    with tempfile.TemporaryDirectory() as tmp, CC.patched(c["optimizer"], c["table_start"]):
        plain, _ = TR.train(settings, folder=Path(tmp) / "plain", verbose=False)
    global ROOT
    old, ROOT = ROOT, Path(tempfile.mkdtemp())
    try:
        traced, _ = TR.load_run(traced_train(name, N, 0, steps, verbose=False))[:2]
    finally:
        ROOT = old
    return all(torch.equal(v, traced.state_dict()[k]) for k, v in plain.state_dict().items())


# ------------------------------------------------------------------------------------------- reading the results
def levels(N):
    """On renewal-N's test windows (analysis.windows, the run's own window size): the excess CE over the future tokens
    of 'remembers nothing' and 'last symbol only' (analysis.reference_levels), of the best possible prediction when the
    GRU's starting state tells nothing ('handoff_useless'), and the training loss of the true probabilities
    ('best_loss') and of 'remembers nothing' ('nothing_loss'), bits per token."""
    import analysis as A
    settings = settings_of("default", N, 0, PILOT_STEPS)
    lv = A.reference_levels(settings, "test")
    x, _, exact = A.windows(settings, "test")
    H = settings.history_len
    alone = TR.exact_ce_bits(settings.process_object(), np.asarray(x)[:, H:])
    best = float(exact[:, H:].mean())
    return dict(lv, handoff_useless=float(alone.mean()) - best, best_loss=best, nothing_loss=best + lv["remembers_nothing"])


def excess_on_default_windows(folder, N, which="latest"):
    """The GRU's excess CE from the Transformer's handoff state over the future tokens of renewal-N's default test
    windows: for a configuration that changes the number of future tokens L, whose own test windows are shorter.  On
    a run with the default windows it equals report's excess_gru_from_handoff_future (checked on three runs)."""
    import analysis as A
    model, settings, _ = TR.load_run(folder, which)
    x, _, exact = A.windows(settings_of("default", N, 0, FINAL_STEPS), "test")
    H = settings.history_len
    with torch.no_grad():
        start = model.handoff_states(x[:, :H], settings.tau_end)
        _, logits = A.rollout(model, start, x[:, H:], model.step_rules()["gru_from_handoff"])
        ce = TR.ce_bits(logits, x[:, H:]).numpy()
    return float(ce.mean() - np.asarray(exact)[:, H:].mean())


def summary(jobs):
    """One row per finished run of jobs (name, N, seed, steps): test results at the final and the best parameters, the
    GRU's machine, the Transformer's choice, and the first validation update with 50% and 90% of the 'remembers
    nothing' excess closed (gap_closed = 1 - excess / remembers nothing).  A configuration that changes the number of
    future tokens L is scored on the default test windows (excess_on_default_windows; own_window_excess is its own),
    and its first_..._closed on its own validation windows against their own 'remembers nothing'."""
    import pandas as pd
    import analysis as A
    rows, lv = [], {}
    for name, N, seed, steps in jobs:
        if not done(name, N, seed, steps):
            continue
        folder = folder_of(name, N, seed, steps)
        lv.setdefault(N, levels(N))
        h = TR.history(folder)
        row = dict(config=name, group=CONFIGS[name]["group"], N=N, seed=seed, updates=int(h["update"].max()),
                   minutes=float(h["seconds"].max()) / 60)
        own_L = "rollout_len" in CONFIGS[name]["settings"]
        for which, pre in (("latest", ""), ("best", "best_")):
            m = dict(A.report(folder, which)["metrics"])
            if own_L:
                row[f"{pre}own_window_excess"] = m["excess_gru_from_handoff_future"]
                m["excess_gru_from_handoff_future"] = excess_on_default_windows(folder, N, which)
            mach = A.machine(folder, which)
            d = json.loads((folder / f"diagnostics_{which}.json").read_text())
            row.update({f"{pre}update": A.report(folder, which)["update"],
                        f"{pre}excess": m["excess_gru_from_handoff_future"],
                        f"{pre}gap_closed": 1 - m["excess_gru_from_handoff_future"] / lv[N]["remembers_nothing"],
                        f"{pre}long_excess": m["long_excess_gru_from_handoff"],
                        f"{pre}states_told_apart": m["gru_from_handoff_true_states_with_own_state"],
                        f"{pre}machine_states": mach["n_states"],
                        f"{pre}handoff_codes": m["handoff_codes"],
                        f"{pre}handoff_information": d.get("handoff_information"),
                        f"{pre}choice_locked_share": d.get("choice_locked_share"),
                        f"{pre}encoder_classifier": d.get("encoder_classifier"),
                        f"{pre}encoder_classifier_untrained": d.get("encoder_classifier_untrained")})
        metric = TR.main_metric("1")
        nothing = (A.reference_levels(settings_of(name, N, seed, steps), "validation")["remembers_nothing"] if own_L
                   else lv[N]["remembers_nothing"])
        for closed in (0.5, 0.9):
            below = h[h[metric] < (1 - closed) * nothing]
            row[f"first_{closed:.0%}_closed"] = int(below["update"].iloc[0]) if len(below) else None
        rows.append(row)
    return pd.DataFrame(rows)


def steps_and_trace(name, N, seed, steps):
    import pandas as pd
    folder = folder_of(name, N, seed, steps)
    return pd.read_csv(folder / "steps.csv"), pd.read_csv(folder / "trace.csv")


def around(name, N, seed, steps, start, stop, every=1000, window=250):
    """One run between updates start and stop, every `every` updates: the validation excess, the training loss and
    the gradient lengths (mean over the `window` updates up to it; 'largest_gradient': the longest of any part in
    them, before clipping), and from trace.csv the fraction of windows whose choice is fixed and the median top q."""
    import pandas as pd
    st, tr = steps_and_trace(name, N, seed, steps)
    st, tr = st.set_index("update"), tr.set_index("update")
    h = TR.history(folder_of(name, N, seed, steps)).set_index("update")[TR.main_metric("1")]
    mean, top = st.rolling(window).mean(), st.rolling(window).max()
    rows = []
    for u in range(start, stop + 1, every):
        rows.append(dict(update=u, validation_excess=h.get(u), training_loss=mean.loss[u],
                         grad_transformer=mean.grad_transformer[u], grad_gru=mean.grad_gru[u],
                         grad_table_output=mean.grad_table_output[u],
                         largest_gradient=top[["grad_transformer", "grad_gru", "grad_table_output"]].loc[u].max(),
                         choice_locked_share=tr.choice_locked_share.get(u), q_max_median=tr.q_max_median.get(u)))
    return pd.DataFrame(rows).set_index("update")


def losses(jobs, low=0.01, high=0.02, before=1000):
    """Every time a run loses what it learned: a validation value above `high` after one below `low` (counted again only
    after it is back below `low`).  With the longest gradient (all parameters, before clipping) in the `before`
    updates up to that validation, and the update of it."""
    import pandas as pd
    rows = []
    for job in jobs:
        folder = folder_of(*job)
        if not (folder / "steps.csv").exists():
            continue
        st = pd.read_csv(folder / "steps.csv").set_index("update")
        total = np.sqrt((st[["grad_transformer", "grad_gru", "grad_table_output"]] ** 2).sum(axis=1))
        h = TR.history(folder)
        learned = False
        for u, v in zip(h["update"], h[TR.main_metric("1")]):
            if v < low:
                learned = True
            elif learned and v > high:
                part = total.loc[u - before + 1:u]
                rows.append(dict(config=job[0], seed=job[2], update=int(u), validation_excess=v,
                                 largest_gradient=float(part.max()), at=int(part.idxmax())))
                learned = False
    return pd.DataFrame(rows)


def gradient_spikes(jobs, above=(1.0, 10.0, 1e3), spike=1e3, window=250, random_points=2000):
    """Single updates with a long gradient (all parameters, before clipping; steps.csv), per run: how many are longer
    than each of `above`, the longest finite one, and how many are too long for 32-bit numbers (recorded as infinite).
    Then what follows an update longer than `spike`: the mean change of the validation excess across the intervals
    between checks that contain one (val_change_at_spikes; val_change_95: the 95th percentile of the size of the change
    across the other intervals), and the mean training loss of the `window` updates after it minus the `window` before
    (loss_change_at_spikes), with the same difference at `random_points` random updates for comparison."""
    import pandas as pd
    rows = []
    for job in jobs:
        folder = folder_of(*job)
        if not (folder / "steps.csv").exists():
            continue
        st = pd.read_csv(folder / "steps.csv").set_index("update")
        total = np.sqrt((st[["grad_transformer", "grad_gru", "grad_table_output"]].astype("float64") ** 2).sum(axis=1))
        row = dict(config=job[0], N=job[1], seed=job[2], updates=len(st))
        row.update({f"over_{a:g}": int((total > a).sum()) for a in above})
        row.update(longest=float(total[np.isfinite(total)].max()), n_infinite=int(np.isinf(total).sum()))
        h = TR.history(folder).set_index("update")[TR.main_metric("1")]
        change = h.diff().dropna()
        at = total.index[total > spike]
        inside = sorted({int(change.index[change.index >= u][0]) for u in at if u <= change.index[-1]})
        other = change.drop(inside)

        def loss_change(u):
            return st.loss.loc[u + 1:u + window].mean() - st.loss.loc[u - window:u - 1].mean()
        last = len(st) - window
        rnd = np.random.default_rng(0).integers(window + 1, last, random_points)
        diffs = [loss_change(u) for u in at if window < u <= last]
        row.update(spikes=len(at), val_change_at_spikes=float(change.loc[inside].mean()) if inside else None,
                   val_change_95=float(other.abs().quantile(0.95)),
                   loss_change_at_spikes=float(np.mean(diffs)) if diffs else None,
                   loss_change_random=float(np.mean([loss_change(u) for u in rnd])),
                   loss_change_random_sd=float(np.std([loss_change(u) for u in rnd])))
        rows.append(row)
    return pd.DataFrame(rows)


def start_ranks(name, N, seeds=SEEDS):
    """The trained runs' own starting points among the 200 of start_gradients (with random rows, seed s's table is the
    one its training run started from): each part's gradient length and its rank, 1 = the smallest of the 200."""
    import pandas as pd
    d = start_gradients(name, N)
    r = d[d.start == "random"].set_index("seed")
    rows = []
    for s in seeds:
        row = dict(N=N, seed=s)
        for part in ("transformer", "gru", "table_output"):
            row[part] = r.loc[s, part]
            row[f"{part}_rank"] = int((r[part] < r.loc[s, part]).sum()) + 1
        rows.append(row)
    return pd.DataFrame(rows)


def search_jobs(seeds=SEEDS, N=15):
    """The search at 50,000 updates: on renewal-15 (section 6, including section 16's runs) or renewal-30 (section 7,
    every run trained here)."""
    return [(n, N, s, FINAL_STEPS) for n in list(IMPORTED) + SEARCH_NAMES for s in seeds]


def combined_jobs(seeds=SEEDS):
    """The second stage on renewal-15 at 50,000 updates: COMBINED_NAMES."""
    return [(n, 15, s, FINAL_STEPS) for n in COMBINED_NAMES for s in seeds]


def pairs_30_jobs(seeds=SEEDS):
    """Section 7's pairs on renewal-30 at 50,000 updates (a batch of 32: 200,000): PAIRS_30."""
    return [(n, 30, s, FINAL_STEPS) for n in PAIRS_30 for s in seeds]


def adopt_imported(seeds=SEEDS):
    """Section 16's runs of model 1 on renewal-15 (runs/imported/model_1/, 50,000 updates) are the search's default and
    original_setup: their settings are settings_of(name, 15, seed, 50000) exactly, and training.train reproduces their
    final parameters exactly (model_1_checks.traced_run).  Copies their files, and steps.csv / trace.csv where
    model_1_checks.traced_run recorded them (default seeds 0-2, original_setup seed 0), into the search's folders."""
    import shutil
    for name, kind in IMPORTED.items():
        for seed in seeds:
            src, folder = TR.RUNS / "imported" / "model_1" / f"renewal-15_w0_s{seed}_{kind}", folder_of(name, 15, seed,
                                                                                                        FINAL_STEPS)
            if done(name, 15, seed, FINAL_STEPS):
                continue
            folder.mkdir(parents=True, exist_ok=True)
            for f in ("settings.json", "validation_history.csv", "best.pt", "completed.pt"):
                shutil.copy2(src / f, folder / f)
            tag = f"s{seed}_{kind}"
            for f in ("steps", "trace"):
                if (TR.RUNS / "model_1_checks" / f"{f}_{tag}.csv").exists():
                    shutil.copy2(TR.RUNS / "model_1_checks" / f"{f}_{tag}.csv", folder / f"{f}.csv")
            (folder / "renewal_check.json").write_text(json.dumps(dict(name=name, N=15, copied_from=str(src),
                                                                       **CONFIGS[name]), indent=2))
            after_training(folder)


# ------------------------------------------------------------------------------------------- the starting points
def start_gradients(name, N, n=200, rows=512, force=False):
    """model_1_checks.start_gradients for configuration `name` on renewal-N: the gradient length before the first
    update at n starting points (model seeds 0 .. n-1, the starting temperature), reaching the whole Transformer, the
    whole GRU, and the state table with the output layer, with the state table as built (all rows equal) and with its
    rows drawn at random (randn / sqrt(S)).  Every starting point sees the same `rows` windows and noise."""
    import pandas as pd
    path = ROOT / f"start_gradients_{name}_renewal-{N}_n{n}.json"
    if path.exists() and not force:
        return pd.DataFrame(json.loads(path.read_text()))
    torch.set_num_threads(2)
    s = settings_of(name, N, 0, FINAL_STEPS)
    window = s.process_object().sampler(999).draw(rows, s.total_len)
    out = []
    for seed in range(n):
        for start in ("equal", "random"):
            model = TR.make_model(settings_of(name, N, seed, FINAL_STEPS))
            if start == "random":
                with torch.no_grad():
                    E = model.causal_head.state_matrix
                    E.copy_(torch.randn_like(E) / E.shape[1] ** 0.5)
            torch.manual_seed(12345)
            g = C1.probe(model, window, s.tau_start)
            out.append(dict(seed=seed, start=start, transformer=float(np.hypot(g["grad_encoder"], g["grad_selector"])),
                            gru=g["grad_gru"], table_output=float(np.hypot(g["grad_state_table"], g["grad_emission"]))))
    ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return pd.DataFrame(out)


@torch.no_grad()
def sampled_excess(name, N, seed, steps, which="latest", draws=3):
    """The GRU read the way it is trained.  In training each next state is drawn at random (Gumbel noise: state k with
    probability softmax(scores)_k; the temperature does not change the draw); in the test results it is the highest
    score.  On renewal-N's default test windows, the Transformer's handoff state the highest score both times:
    'excess_highest' and 'excess_sampled' (mean of `draws` draws, torch seeds 0, 1, ...) over the future tokens,
    'differs' (fraction of the GRU's steps whose drawn state is not the highest score), and 'p1_highest' / 'p1_sampled'
    (the mean predicted P(next token is 1) at each true state).  A configuration without noise draws nothing, so its
    two values are equal.  Cached in <folder>/sampled_<which>.json."""
    import analysis as A
    from torch.nn import functional as F
    folder = folder_of(name, N, seed, steps)
    path = folder / f"sampled_{which}.json"
    model, settings, update = TR.load_run(folder, which)
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["update"] == update and cached["draws"] == draws:
            return cached
    model.eval()
    x, st, exact = A.windows(settings_of("default", N, 0, FINAL_STEPS), "test")
    x, st = torch.as_tensor(np.asarray(x)), np.asarray(st)
    H, L, tau = settings.history_len, x.shape[1] - settings.history_len, settings.tau_end
    head, E, W = model.causal_head, model.causal_head.state_matrix, model.causal_head.emission
    start = model.handoff_states(x[:, :H], tau)
    best = float(np.asarray(exact)[:, H:].mean())
    true = st[:, H:H + L]
    out = dict(update=update, draws=draws)
    for kind, noise, seeds in (("highest", False, [0]), ("sampled", bool(model.cfg.gumbel), range(draws))):
        ce, p1, differs = [], [], []
        for d in seeds:
            torch.manual_seed(d)
            ids, logits = start, []
            for k in range(L):
                logits.append(W(E[ids]))
                scores = model.gru_selector(model.gru_cell(F.one_hot(x[:, H + k], 2).to(E.dtype), E[ids]))
                new = head.select(scores, tau, noise).ids
                differs.append(float((new != scores.argmax(-1)).float().mean()))
                ids = new
            logits = torch.stack(logits, 1)
            ce.append(float(TR.ce_bits(logits, x[:, H:]).numpy().mean()) - best)
            p1.append(logits.softmax(-1)[..., 1].numpy())
        p1 = np.mean(p1, 0)
        out[f"excess_{kind}"] = float(np.mean(ce))
        out[f"p1_{kind}"] = [float(p1[true == c].mean()) if (true == c).any() else None for c in range(N)]
        if kind == "sampled":
            out["differs"] = float(np.mean(differs))
    path.write_text(json.dumps(out))
    return out


# ------------------------------------------------------------------------------------------- figures
SEED_COLOURS = ["#eb6834", "#2a78d6", "#1baf7a"]
EXCESS = "test CE above the true process (bits/token)"


def plot_search(table, title=None, xlim=(8e-4, 1.0)):
    """table: summary() of the search.  One row per configuration, one marker per seed: the test CE above the true
    process over the 20 future tokens.  Filled: final parameters; hollow: the parameters with the best validation
    value.  Dashed: 'remembers nothing'."""
    C1.style()
    import matplotlib.pyplot as plt
    order = [n for n in CONFIGS if n in set(table.config)][::-1]
    fig, ax = plt.subplots(figsize=(11, 0.36 * len(order) + 2.0), layout="constrained")
    for i, name in enumerate(order):
        for _, r in table[table.config == name].iterrows():
            y = i + (int(r.seed) - 1) * 0.18
            ax.scatter(max(r["excess"], 1e-4), y, s=40, color=C1.ORANGE, zorder=3)
            ax.scatter(max(r["best_excess"], 1e-4), y, s=40, facecolor="none", edgecolor="#52514e", zorder=2)
    N = int(table.N.iloc[0])
    lv = levels(N)
    ax.axvline(lv["remembers_nothing"], color=C1.GREY, ls="--", lw=1.3,
               label=f"predicting each token's long-run frequency ({lv['remembers_nothing']:.3f})")
    ax.scatter([], [], s=40, color=C1.ORANGE, label="final parameters (one dot per seed)")
    ax.scatter([], [], s=40, facecolor="none", edgecolor="#52514e", label="parameters with the best validation value")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_yticks(range(len(order)), [label(n, N) for n in order])
    ax.set_xlabel(EXCESS + f", {SIZES[N].get('rollout_len', 20)} future tokens")
    ax.grid(axis="y", visible=False)
    groups = [CONFIGS[n]["group"] for n in order]
    for i in range(1, len(order)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="#e6e5e1", lw=1)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1)
    ax.set_title(title or f"Renewal-{N}, model 1: every configuration\n(values below 0.0001 drawn at 0.0001)")
    return fig


def plot_curves(panels, title=None, width=6.0, height=3.5, ylim=(1e-4, 1)):
    """panels: {title: [(seed, folder), ...]}.  The validation CE above the true process of the GRU started from the
    Transformer's state, one line per seed; dashed: 'remembers nothing'; dotted: the best possible when the GRU's
    starting state tells nothing.  Each panel `width` x `height` inches."""
    C1.style()
    fig, axes = C1._grid(len(panels), width=width, height=height)
    for ax, (name, runs) in zip(axes, panels.items()):
        lv = levels(len(TR.read_settings(runs[0][1]).params["F"]))
        for seed, f in runs:
            h = TR.history(f)
            ax.plot(h["update"] / 1000, h[TR.main_metric("1")].clip(lower=1e-4), color=SEED_COLOURS[seed],
                    label=f"seed {seed}")
        ax.axhline(lv["remembers_nothing"], color=C1.GREY, ls="--", lw=1.3,
                   label="predicting each token's long-run frequency")
        ax.axhline(lv["handoff_useless"], color=C1.GREY, ls=":", lw=1.3,
                   label="best possible when the GRU's starting state tells nothing")
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_title(name)
        ax.set_xlabel("update (thousands)")
        ax.set_ylabel("CE above true (bits/token)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3 if len(panels) > 1 else 1)
    fig.suptitle(title or "Validation CE above the true process (bits per token) over training\n"
                 "(the GRU started from the Transformer's state; values below 0.0001 drawn at 0.0001)")
    return fig


def plot_whole_training(panels, smooth=250):
    """panels: {title: (steps, N)}.  model_1_checks.plot_whole_training with the loss axis and its reference lines
    (the loss of the true probabilities, and of 'remembers nothing', on the test windows) for each panel's renewal-N."""
    C1.style()
    fig, axes = C1._grid(len(panels))
    twins = []
    for ax, (name, (st, N)) in zip(axes, panels.items()):
        lv = levels(N)
        u = st["update"] / 1000
        ax.plot(u, st["grad_transformer"].clip(lower=1e-14).rolling(smooth, min_periods=1).median(), color=C1.BLUE,
                label="gradient length: whole Transformer")
        ax.plot(u, st["grad_gru"].clip(lower=1e-14).rolling(smooth, min_periods=1).median(), color=C1.ORANGE,
                label="gradient length: whole GRU")
        ax.set_yscale("log")
        ax.set_ylim(1e-14, 1)
        ax.set_xlabel("update (thousands)")
        ax.set_ylabel("gradient length")
        ax.set_title(name)
        twin = ax.twinx()
        twins.append(twin)
        twin.plot(u, st["loss"].rolling(smooth, min_periods=1).mean(), color=C1.INK, lw=1.4,
                  label="training loss (right axis)")
        twin.axhline(lv["best_loss"], color=C1.GREY, lw=1.4, ls="--",
                     label="best possible loss (true probabilities, test windows)")
        twin.axhline(lv["nothing_loss"], color=C1.GREY, lw=1.4, ls=":",
                     label="loss when remembering nothing (test windows)")
        gap = lv["nothing_loss"] - lv["best_loss"]
        twin.set_ylim(lv["best_loss"] - 0.55 * gap, lv["nothing_loss"] + 1.7 * gap)
        twin.set_ylabel("training loss (bits per token)")
        twin.grid(False)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = twins[0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=2)
    fig.suptitle(f"Gradient length and training loss over training\n(each point: median gradient and mean loss of the "
                 f"{smooth} updates up to it; a gradient of 0 is drawn at 1e-14)")
    return fig


def plot_choice_over_training(panels):
    """panels: {title: (trace, settings)}.  model_1_checks.plot_choice_over_training with each run's own number of
    state scores and last history position."""
    C1.style()
    fig, axes = C1._grid(len(panels))
    twins = []
    for ax, (name, (t, s)) in zip(axes, panels.items()):
        u = t["update"] / 1000
        ax.plot(u, t["grad_at_handoff_vector"].clip(lower=1e-14), color=C1.INK,
                label="at the state vector the Transformer gives the GRU")
        ax.plot(u, t["grad_after_choice"].clip(lower=1e-14), color=C1.PURPLE,
                label="at the Transformer's K state scores (after the choice)")
        ax.set_yscale("log")
        ax.set_ylim(1e-14, 1)
        ax.set_xlabel("update (thousands)")
        ax.set_ylabel("gradient length")
        ax.set_title(f"{name}\n(K = {s.n_states}, last history token {s.history_len - 1})")
        twin = ax.twinx()
        twins.append(twin)
        twin.plot(u, t["choice_locked_share"], color=C1.GREY, lw=1.4, ls="--",
                  label=f"fraction of {s.batch_size} test windows where the picked state has softmax probability > 0.999")
        twin.set_ylim(-0.02, 1.02)
        twin.set_ylabel("fraction of test windows")
        twin.grid(False)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = twins[0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=1)
    fig.suptitle("Gradient length on both sides of the Transformer's choice of state at the last history token\n"
                 "(every 250 updates, on windows not used for training; 0 drawn at 1e-14)")
    return fig


def plot_spikes(panels, smooth=250, top=1e20):
    """panels: {title: steps}.  The gradient length of all parameters before clipping at every update: the median of
    the `smooth` updates up to each point (grey line) and every single update longer than the clipping length 1
    (dots); dashed: the clipping length.  A length too large for 32-bit numbers (infinite) is drawn at `top`."""
    C1.style()
    fig, axes = C1._grid(len(panels))
    for ax, (name, st) in zip(axes, panels.items()):
        total = np.sqrt((st[["grad_transformer", "grad_gru", "grad_table_output"]].astype("float64") ** 2).sum(axis=1))
        u = st["update"] / 1000
        ax.plot(u, total.rolling(smooth, min_periods=1).median(), color=C1.GREY, lw=1.2,
                label=f"median of the {smooth} updates up to each point")
        long_ = total > 1
        ax.scatter(u[long_], total[long_].clip(upper=top), s=14, color=C1.ORANGE, zorder=3,
                   label="single updates longer than 1")
        ax.axhline(1, color=C1.INK, ls="--", lw=1.2, label="clipping length 1")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, top * 10)
        ax.set_xlabel("update (thousands)")
        ax.set_ylabel("gradient length")
        ax.set_title(name)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle(f"Gradient length of all parameters, before clipping, at every update\n(a length too large for 32-bit "
                 f"numbers is drawn at {top:.0e})")
    return fig


def plot_start_gradients(data_by_N, name="default", title=None):
    """data_by_N: {N: start_gradients(name, N)}.  One row per renewal-N, one panel per state-table start: histograms
    of the gradient length before the first update."""
    C1.style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(data_by_N), 2, figsize=(12.5, 3.6 * len(data_by_N)), sharex=True, squeeze=False,
                             layout="constrained")
    bins = np.logspace(-14, 1, 61)
    for row, (N, data) in zip(axes, data_by_N.items()):
        K = settings_of(name, N, 0, FINAL_STEPS).n_states
        for ax, (start, panel) in zip(row, [("equal", f"renewal-{N}: state table as built (all {K} rows equal)"),
                                            ("random", f"renewal-{N}: state table rows drawn at random")]):
            d = data[data.start == start]
            for col, label, colour in [("transformer", "whole Transformer", C1.BLUE), ("gru", "whole GRU", C1.ORANGE),
                                       ("table_output", "state table + output layer", C1.GREY)]:
                ax.hist(d[col].clip(lower=1e-14), bins=bins, color=colour, alpha=0.75, label=label)
            ax.set_xscale("log")
            ax.set_title(panel)
            ax.set_ylabel(f"number of seeds (of {data.seed.nunique()})")
        row[1].legend(loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("gradient length at the start (0 drawn at 1e-14)")
    fig.suptitle(f"Gradient length before the first update, for {next(iter(data_by_N.values())).seed.nunique()} "
                 f"seeds (each seed = one set of starting weights; model 1, {title or LABELS.get(name, name)})")
    return fig


def _learned_state_of_true(machine_):
    """For each true state, the learned state it is most often in (-1: never seen on the alignment windows)."""
    table = np.asarray(machine_["by_true_state"])
    return np.where(table.sum(0) > 0, table.argmax(0), -1)


def draw_renewal_machine(ax, machine_, settings, colour, title=None):
    """A renewal machine on a circle, its states in the order of the true states they stand for (a0 at the top, then
    clockwise).  Each circle: the true states sent to it and the probability that the next token is a 1.  Grey arrow: the
    move after a 0; coloured arrow: the move after a 1 (only moves with probability >= 0.01).  A move back to the same
    state is written in the circle: '↺ 0' after a 0, '↺ 1' after a 1."""
    import networkx as nx
    import analysis as A
    from matplotlib.colors import to_hex, to_rgb
    M = np.asarray(machine_["M"])
    n = len(M)
    names = A._names(machine_, settings)
    table = np.asarray(machine_["by_true_state"])
    first_true = [min(c) if c else int(table[s].argmax()) + 0.5 for s, c in enumerate(machine_["sent_to"])]
    order = np.argsort(first_true)
    angle = {int(s): np.pi / 2 - 2 * np.pi * r / n for r, s in enumerate(order)}
    pos = {s: np.array([np.cos(a), np.sin(a)]) if n > 1 else np.zeros(2) for s, a in angle.items()}
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    edges, loops = {0: [], 1: []}, {s: [] for s in range(n)}
    for s in range(n):
        for t in range(n):
            for x in (0, 1):
                if M[s, t, x] < 0.01:
                    continue
                if s == t:
                    loops[s].append(str(x))
                else:
                    edges[x].append((s, t))
    size = 5200 if n <= 4 else 2900
    font = 13 if n <= 4 else 11
    tint = to_hex(0.22 * np.array(to_rgb(colour)) + 0.78 * np.array(to_rgb(A.SURFACE)))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=size, node_color=tint, edgecolors=colour, linewidths=1.6)
    def wrapped(name_):                          # a long list of true states: 3 per line
        parts = name_.split(", ")
        return ",\n".join(", ".join(parts[i:i + 3]) for i in range(0, len(parts), 3))
    labels = {s: f"{wrapped(names[s])}\n{M[s, :, 1].sum():.2f}" + ("\n↺ " + ", ".join(loops[s]) if loops[s] else "")
              for s in range(n)}
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=font, labels=labels, font_color=A.INK)
    for x, edge_colour in ((0, A.INK2), (1, colour)):
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges[x], node_size=size, arrows=True, arrowstyle="-|>",
                               arrowsize=14, width=1.2, edge_color=edge_colour, connectionstyle="arc3,rad=0.08")
    ax.set_axis_off()
    ax.set_aspect("equal")
    lim = 1.22 if n > 4 else 1.7
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if title:
        ax.set_title(title)


def plot_machines(name, Ns=(15,), seeds=SEEDS, steps=None, which="latest", ncols=4, panel=5.4):
    """For each renewal-N: the true machine (drawn for N <= 15; otherwise described), then the machine each seed's GRU
    defines (analysis.machine: its step rule tried on every table row, the part it keeps returning to, identical states
    merged), at the final parameters.  Panels in rows of `ncols`, each `panel` inches wide."""
    C1.style()
    import matplotlib.pyplot as plt
    import analysis as A
    steps = steps or FINAL_STEPS
    panels = [(N, seed) for N in Ns for seed in [None, *seeds]]
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols, (panel + 0.4) * nrows), squeeze=False,
                             layout="constrained")
    for ax, (N, seed) in zip(axes.flat, panels):
        if seed is None and N <= 15:
            settings = settings_of(name, N, 0, steps)
            draw_renewal_machine(ax, A.true_machine(settings), settings, C1.BLUE,
                                 title=f"renewal-{N}: the true machine, {N} states")
        elif seed is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"renewal-{N}: the true machine has\n{N} states, a0 to a{N - 1}, on a ring\n"
                    f"like renewal-15's: after a 0, a(c) → a(c+1);\nafter a 1, back to a0.\n"
                    f"P(1) in a(c) is 1 / ({N} − c):\n{1 / N:.3f} in a0, 1 in a{N - 1}",
                    ha="center", va="center", fontsize=13, transform=ax.transAxes)
        else:
            folder = folder_of(name, N, seed, steps)
            m = A.machine(folder, which)
            plural = "" if m["n_states"] == 1 else "s"
            draw_renewal_machine(ax, m, TR.read_settings(folder), C1.ORANGE,
                                 title=f"renewal-{N}, seed {seed}: the GRU's machine,\n{m['n_states']} state{plural} "
                                       f"(update {m['update']:,})")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("Circles: states, named by the true states sent to them (a<c>: c zeros since the last 1); the number: "
                 "the probability that the next token is a 1.\nGrey arrow: the move after a 0; coloured arrow: the "
                 "move after a 1; ↺ 0 / ↺ 1: it stays in the state after a 0 / a 1.")
    return fig


def plot_prob_of_one(name, Ns=(15, 30, 60), seeds=SEEDS, steps=None, which="latest"):
    """For each renewal-N: the probability that the next token is a 1 at each true state, in the true machine (black)
    and in the learned state that true state is most often in, for each seed's GRU machine (final parameters)."""
    C1.style()
    import matplotlib.pyplot as plt
    import analysis as A
    steps = steps or FINAL_STEPS
    colours = [C1.ORANGE, C1.BLUE, C1.PURPLE]
    fig, axes = plt.subplots(1, len(Ns), figsize=(6.2 * len(Ns), 4.8), squeeze=False, layout="constrained")
    for ax, N in zip(axes[0], Ns):
        settings = settings_of(name, N, 0, steps)
        true_p1 = settings.process_object().M[:, :, 1].sum(1)
        ax.plot(range(N), true_p1, color="black", lw=2.2, label="true machine")
        for seed, colour, marker in zip(seeds, colours, "osD"):
            folder = folder_of(name, N, seed, steps)
            m = A.machine(folder, which)
            p1 = np.asarray(m["M"])[:, :, 1].sum(1)
            state = _learned_state_of_true(m)
            seen = state >= 0
            ax.plot(np.flatnonzero(seen), p1[state[seen]], color=colour, lw=1.2, marker=marker, ms=5,
                    label=f"seed {seed}: GRU's machine, {m['n_states']} state{'' if m['n_states'] == 1 else 's'}")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 1.3)
        ax.set_xlabel("true state: zeros since the last 1")
        ax.set_ylabel("P(next token is 1)")
        ax.set_title(f"renewal-{N}")
        ax.legend(loc="upper left", fontsize=10)
    fig.suptitle("Probability that the next token is a 1, at each true state:\ntrue machine and each GRU's machine "
                 + ("(final parameters)" if which == "latest" else "(parameters with the best validation value)"))
    return fig


def plot_sampled(name, N, seeds=SEEDS, steps=None, which="latest"):
    """One panel per seed (sampled_excess): the mean predicted P(next token is 1) at each true state of the future
    tokens, with the GRU's next state the highest score (as in the test results) and drawn at random (as in training),
    and the true probability."""
    C1.style()
    import matplotlib.pyplot as plt
    steps = steps or FINAL_STEPS
    fig, axes = plt.subplots(1, len(seeds), figsize=(6.0 * len(seeds), 4.8), squeeze=False, layout="constrained")
    for ax, seed in zip(axes[0], seeds):
        d = sampled_excess(name, N, seed, steps, which)
        ax.plot(range(N), [1 / (N - c) for c in range(N)], color="black", lw=2.2, label="true")
        for kind, colour, marker, words in (("highest", C1.ORANGE, "o", "next state: highest score"),
                                            ("sampled", C1.BLUE, "s", "next state: drawn, as in training")):
            p = np.array([np.nan if v is None else v for v in d[f"p1_{kind}"]])
            ax.plot(range(N), p, color=colour, lw=1.2, marker=marker, ms=4,
                    label=f"{words} (excess {d[f'excess_{kind}']:.4f})")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 1.3)
        ax.set_xlabel("true state: zeros since the last 1")
        ax.set_ylabel("mean predicted P(next token is 1)")
        ax.set_title(f"renewal-{N}, seed {seed} (update {d['update']:,})")
        ax.legend(loc="upper left", fontsize=10)
    fig.suptitle("The GRU's predictions of the future tokens at each true state, on the test windows\n"
                 "(the Transformer's starting state is the highest score in both; excess in bits per token)")
    return fig


# ------------------------------------------------------------------------------------------- command line
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stage", choices=["search", "combined"], default=None)
    r.add_argument("--names", default=None, help="comma-separated configurations")
    r.add_argument("--N", default="15", help="e.g. 15,30,60")
    r.add_argument("--steps", type=int, default=PILOT_STEPS)
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("starts")
    s.add_argument("--names", default="default")
    s.add_argument("--N", default="15,30,60")
    sub.add_parser("status")
    sub.add_parser("test")
    a = p.parse_args()
    if a.cmd == "run":
        seeds = [int(x) for x in a.seeds.split(",")]
        if a.stage == "search":
            adopt_imported(seeds)
            jobs = search_jobs(seeds)
        elif a.stage == "combined":
            jobs = combined_jobs(seeds)
        else:
            jobs = [(n, int(N), sd, a.steps) for N in a.N.split(",") for n in a.names.split(",") for sd in seeds]
        run(jobs, a.workers)
    elif a.cmd == "starts":
        for name in a.names.split(","):
            for N in a.N.split(","):
                start_gradients(name, int(N))
                print(f"done start_gradients {name} renewal-{N}", flush=True)
    elif a.cmd == "status":
        for f in sorted(p for p in ROOT.glob("*_renewal-*") if p.is_dir()):
            state = ("finished" if (f / "diagnostics_best.json").exists() else
                     "training" if (f / "progress.json").exists() else "started")
            upd = json.loads((f / "progress.json").read_text())["update"] if (f / "progress.json").exists() else 0
            print(f"{f.name:55s} {state:9s} update {upd}")
    elif a.cmd == "test":
        print("tracing leaves training unchanged:", test_tracing())


if __name__ == "__main__":
    main()
