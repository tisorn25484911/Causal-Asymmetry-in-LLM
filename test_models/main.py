"""
Run any model of this folder from the command line.  (From test_models/; every run is saved under runs/.)

    python main.py list                                     the models (model_<name>.py) and the processes
    python main.py train  --model 2 --seed 0                train with section 16's settings (a finished run reloads)
    python main.py train  --model 5b --weight 0.1 --steps 10000
    python main.py train  --model 1 --original-setup        section 13's settings: no LayerNorm on the Transformer's
                                                            output, no weight decay, no gradient clipping
    python main.py train  --model 2 --process even --params '{"p": 0.5}' --set history_len=30 --set rollout_len=10
    python main.py train  --model B --until-settled         + 25,000 more updates while still improving, up to 100,000
    python main.py train  --model 5b --json '{"process": "flower", "params": {"n": 8, "m": 12, "dice_seed": 0}}'
    python main.py train-many --plan plans/new_processes.json --workers 9 --memory-gb 10.5
                                                            run a list of trainings in parallel (a JSON list of
                                                            {"model": ..., "settings": {...}, "memory_gb": ...}) within a
                                                            memory limit, a log for each in runs/logs/
    python main.py extend --run runs/model_2/<folder> --steps 25000
    python main.py report --run runs/model_2/<folder>       test-window results and the main figures (analysis.py)
    python main.py import-runs                              copy the trained runs of sections 13, 16 and 17 of
                                                            hybrid_model_test.ipynb into runs/imported/

A new model: copy model_2.py to model_<name>.py, change it, and run `python main.py train --model <name>`.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import training as TR                                                       # noqa: E402

OLD = HERE.parent / "model_modeularised"
IMPORTED = TR.RUNS / "imported"

# validation-history columns of sections 16-17 -> this folder's names
S16_COLUMNS = {"excess_gru_rollout": "excess_gru_from_handoff_future",
               "excess_next_state_net_rollout": "excess_next_state_from_handoff_future",
               "gru_rollout_states_used": "gru_from_handoff_states_used",
               "gru_rollout_information": "gru_from_handoff_information",
               "next_state_net_rollout_states_used": "next_state_from_handoff_states_used",
               "next_state_net_rollout_information": "next_state_from_handoff_information"}


def modular_columns(predictor):
    return {"excess_future": f"excess_{predictor}_future", "excess_all": f"excess_{predictor}_all",
            "states_used": f"{predictor}_states_used", "information": f"{predictor}_information",
            "determinism": f"{predictor}_determinism", "state_gap_median": "handoff_margin_median"}


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def cmd_list(_):
    print("models (python main.py train --model <name>):")
    for name in TR.available_models():
        m = TR.load_model_module(name)
        print(f"  {name:22s} {m.DESCRIPTION}")
    print("\nprocesses (--process <name> --params '<json>'):", ", ".join(sorted(TR.process_module.PROCESS)))


def settings_from_args(a):
    changes = {k: parse_value(v) for k, v in (pair.split("=", 1) for pair in (a.set or []))}
    if a.original_setup:
        changes.update(TR.ORIGINAL_SETUP)
    for key in ("seed", "steps", "weight"):
        if getattr(a, key) is not None:
            changes[key] = getattr(a, key)
    if a.process:
        changes["process"] = a.process
    if a.params:
        changes["params"] = json.loads(a.params)
    if a.json:
        changes.update(json.loads(a.json))
    return TR.settings_for(a.model, **changes)


def cmd_train(a):
    settings = settings_from_args(a)
    print("run folder:", TR.run_folder(settings), flush=True)
    if a.until_settled:
        TR.train_until_settled(settings)
    else:
        TR.train(settings, force=a.force)


def cmd_extend(a):
    TR.extend(Path(a.run), a.steps)


def cmd_train_many(a):
    """Run the trainings of a plan in parallel, each in its own process with one thread.  The plan is a JSON list of
    {"model": name, "settings": {setting: value}, "memory_gb": the run's peak memory (optional, default 1)}.  Trainings
    start in the order listed while fewer than --workers run and the memory_gb of the running ones stays within
    --memory-gb; when the next one does not fit, a later one that fits starts first.  Finished runs are skipped.  A run
    whose progress.json changed in the last --adopt-minutes is taken to be training in another process (e.g. started by
    an earlier train-many that was stopped): it counts as running until it finishes, and is not started again.
    Logs: runs/logs/.  Do not change training.py, process.py, causal_state_matrix.py or a model file while it runs:
    later trainings would use the changed code under a different run folder."""
    jobs = json.loads(Path(a.plan).read_text())
    logs = TR.RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               # macOS: keep freed large blocks for reuse instead of returning them to the system after every update;
               # model C on the flower (a 20^3-wide output) then trains 16% faster.  Results are bit-identical.
               MallocLargeExpandedCacheThreshold="1")

    def say(text):
        print(f"{time.strftime('%d %b %H:%M')} {text}", flush=True)

    def minutes_since_progress(folder):
        p = folder / "progress.json"
        return (time.time() - p.stat().st_mtime) / 60 if p.exists() else float("inf")

    waiting, running, failed = [], [], 0
    for job in jobs:
        folder = TR.run_folder(TR.settings_for(job["model"], **job.get("settings", {})))
        if (folder / "completed.pt").exists():
            say(f"already finished {folder.parent.name}/{folder.name}")
        elif minutes_since_progress(folder) < a.adopt_minutes:
            running.append((job, folder, None, None))                  # training in another process
            say(f"adopted  {folder.parent.name}/{folder.name} (training in another process)")
        else:
            waiting.append((job, folder))
    say(f"{len(waiting)} trainings to run, at most {a.workers} at a time within {a.memory_gb:g} GB")
    while waiting or running:
        for job, folder in list(waiting):
            used = sum(j.get("memory_gb", 1.0) for j, _, _, _ in running)
            if len(running) < a.workers and (not running or used + job.get("memory_gb", 1.0) <= a.memory_gb):
                log = open(logs / f"{folder.parent.name}__{folder.name}.log", "w")
                cmd = [sys.executable, str(HERE / "main.py"), "train", "--model", job["model"],
                       "--json", json.dumps(job.get("settings", {}))]
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE), env=env)
                running.append((job, folder, proc, log))
                waiting.remove((job, folder))
                say(f"started  {folder.parent.name}/{folder.name}")
        time.sleep(a.poll)
        for item in list(running):
            job, folder, proc, log = item
            if proc is None:                                             # adopted: watch its files
                if (folder / "completed.pt").exists():
                    running.remove(item)
                    say(f"finished {folder.parent.name}/{folder.name}")
                elif minutes_since_progress(folder) > 2 * a.adopt_minutes:
                    running.remove(item)
                    failed += 1
                    say(f"STOPPED (no progress for {2 * a.adopt_minutes:g} minutes) {folder.parent.name}/{folder.name}")
                continue
            if proc.poll() is not None:
                running.remove(item)
                log.close()
                failed += proc.returncode != 0
                status = "finished" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                say(f"{status} {folder.parent.name}/{folder.name}")
    say(f"all done; {failed} failed")


def cmd_report(a):
    import analysis as A
    folder = Path(a.run)
    rep = A.report(folder, a.which)
    print(json.dumps({k: v for k, v in rep["metrics"].items()}, indent=2))
    for path in A.save_run_figures(folder, a.which):
        print("figure:", path)


# ------------------------------------------------------------------------------------------------ importing old runs
def _adopted_state(settings, state):
    model = TR.make_model(settings)
    return TR.adopt_state_dict(model, state).state_dict()


def _renamed_history(path, columns):
    return pd.read_csv(path).rename(columns=columns)


def copy_run(src, dst, settings, columns, metric, source):
    """One run of model_modeularised -> dst in this folder's layout: settings.json, completed.pt (with the optimizer and
    random-number states when saved, so `extend` can continue it), extended_<n>.pt + extensions.json, best.pt, and the
    validation histories with this folder's column names.  The per-update training histories (large) are not copied;
    settings.json gives their location."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    keep = ("updates", "seconds", "best_update", "best_value", "optimizer_state", "rng_state", "torch_rng_state",
            "torch_version")
    saved = torch.load(src / "completed.pt", map_location="cpu", weights_only=False)
    torch.save(dict({k: saved[k] for k in keep if k in saved}, settings=settings.to_dict(), metric=metric,
                    model_state=_adopted_state(settings, saved["model_state"])), dst / "completed.pt")
    exts = json.loads((src / "extensions.json").read_text()) if (src / "extensions.json").exists() else []
    for ext in exts:
        e = torch.load(src / ext["file"], map_location="cpu", weights_only=False)
        torch.save(dict({k: e[k] for k in keep if k in e}, settings=settings.to_dict(),
                        model_state=_adopted_state(settings, e["model_state"])), dst / ext["file"])
    (dst / "extensions.json").write_text(json.dumps(exts, indent=2)) if exts else None
    if (src / "best.pt").exists():
        b = torch.load(src / "best.pt", map_location="cpu", weights_only=False)
        torch.save(dict(settings=settings.to_dict(), model_state=_adopted_state(settings, b["model_state"]),
                        update=b["update"], metric=metric, value=b["value"]), dst / "best.pt")
    _renamed_history(src / "validation_history.csv", columns).to_csv(dst / "validation_history.csv", index=False)
    if (src / "validation_history_extended.csv").exists():
        _renamed_history(src / "validation_history_extended.csv", columns).to_csv(
            dst / "validation_history_extended.csv", index=False)
    (dst / "settings.json").write_text(json.dumps(dict(
        settings.to_dict(), imported_from=str(src), source=source,
        training_histories=sorted(str(p) for p in src.glob("training_history*.csv"))), indent=2))
    return dst


def check_same_settings(settings, old, ignore=()):
    """The settings recorded with an old run must agree with the new Settings on every key they share."""
    new = settings.to_dict()
    differ = {k: (old[k], new[k]) for k in old if k in new and k not in ignore and old[k] != new[k]}
    if differ:
        raise ValueError(f"settings differ from the old run's: {differ}")


def import_runs(old=OLD, verbose=True):
    """Copy the trained runs of hybrid_model_test.ipynb's sections 16 (9 models x 3 seeds, and model 1 in the original
    setup), 17 (model 2 + L2 at 0.01 / 0.03 / 0.1, the feedback GRU and the discrete Transformer, 3 seeds each) and 13
    (model 1 + L2, H = 300, L = 40) into runs/imported/.  Returns {(model, label, seed): folder}."""
    if str(old) not in sys.path:
        sys.path.append(str(old))
    from checks import section16_experiments as S, section16_report as R, section17_experiments as S17
    by_s16 = {TR.load_model_module(k).SECTION16_NAME: k for k in TR.available_models()
              if getattr(TR.load_model_module(k), "SECTION16_NAME", None)}
    out = {}
    for (name, seed), trial in R.full_trials().items():
        original = name == "gru_loss_original"
        key = by_s16[trial["model"]]
        settings = TR.settings_for(key, seed=seed, weight=trial["weight"], **(TR.ORIGINAL_SETUP if original else {}))
        check_same_settings(settings, trial, ignore=("model",))
        label = "original_setup" if original else "section16"
        dst = IMPORTED / f"model_{key}" / f"{settings.process_object().label}_w{settings.weight:g}_s{seed}_{label}"
        out[(key, label, seed)] = copy_run(S.trial_dir(trial), dst, settings, S16_COLUMNS,
                                           TR.main_metric(key), f"section 16 ({trial['model']})")
    for name, weights in (("l2_both_losses", (0.01, 0.03, 0.1)), ("feedback_gru", (None,)),
                          ("discrete_transformer", (None,))):
        key = {"l2_both_losses": "2_L2", "feedback_gru": "feedback_GRU", "discrete_transformer": "discrete_Transformer"}[name]
        predictor = TR.load_model_module(key).MAIN_PREDICTOR
        for weight in weights:
            for seed in (0, 1, 2):
                trial = S17.trial_settings(name, seed, weight)
                settings = TR.settings_for(key, seed=seed, **({"weight": weight} if weight is not None else {}))
                check_same_settings(settings, trial, ignore=("model",))
                columns = S16_COLUMNS if name == "l2_both_losses" else modular_columns(predictor)
                dst = IMPORTED / f"model_{key}" / f"{settings.process_object().label}_w{settings.weight:g}_s{seed}_section17"
                out[(key, "section17", seed) if weight is None else (key, f"w{weight:g}", seed)] = copy_run(
                    S17.trial_dir(trial), dst, settings, columns, TR.main_metric(key), f"section 17 ({name})")
    # section 13's current run: model 1 + L2 with the original setup, 300 + 40 tokens, weight 0.03
    src = old / "notebook_results" / "renewal15_1ee548972375"
    exp = torch.load(src / "completed.pt", map_location="cpu", weights_only=False)["experiment"]
    settings = TR.settings_for("1_L2", history_len=300, rollout_len=40, steps=exp["steps"], weight=exp["l2_weight"],
                               eval_every=exp["eval_every"], threads=exp["threads"], **TR.ORIGINAL_SETUP)
    check_same_settings(settings, dict(exp["model"], **{k: exp[k] for k in ("steps", "batch_size", "lr", "tau_start",
                                                                               "tau_end", "validation_rows")}))
    dst = IMPORTED / "model_1_L2" / f"{settings.process_object().label}_H300_L40_w{settings.weight:g}_s0_section13"
    dst.mkdir(parents=True, exist_ok=True)
    saved = torch.load(src / "completed.pt", map_location="cpu", weights_only=False)
    torch.save(dict(settings=settings.to_dict(), model_state=_adopted_state(settings, saved["model_state"]),
                    updates=saved["updates"], seconds=saved["seconds"], metric=TR.main_metric("1_L2")), dst / "completed.pt")
    h = pd.read_csv(src / "validation_history.csv")
    h["excess_gru_from_handoff_future"] = h["ce_bits"] - h["exact_ce_bits"]
    h.rename(columns={"mean_l2": "l2"}).to_csv(dst / "validation_history.csv", index=False)
    (dst / "settings.json").write_text(json.dumps(dict(
        settings.to_dict(), imported_from=str(src), source="section 13 (renewal15_1ee548972375)",
        training_histories=[str(src / "training_history.csv")]), indent=2))
    out[("1_L2", "section13", 0)] = dst
    (IMPORTED / "index.json").write_text(json.dumps({"|".join(map(str, k)): str(v.relative_to(TR.RUNS))
                                                     for k, v in out.items()}, indent=2))
    if verbose:
        print(f"imported {len(out)} runs into {IMPORTED}")
    return out


def imported_runs():
    """{(model, label, seed): folder} of the imported runs (import-runs writes runs/imported/index.json)."""
    index = json.loads((IMPORTED / "index.json").read_text())
    return {(k.split("|")[0], k.split("|")[1], int(k.split("|")[2])): TR.RUNS / v for k, v in index.items()}


def cmd_import(a):
    if a.clean and IMPORTED.exists():
        shutil.rmtree(IMPORTED)
    import_runs(Path(a.old))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--seed", type=int)
    t.add_argument("--steps", type=int)
    t.add_argument("--weight", type=float)
    t.add_argument("--process")
    t.add_argument("--params", help="the process's parameters as JSON")
    t.add_argument("--set", action="append", help="any setting: key=value (see training.Settings)")
    t.add_argument("--json", help="settings as a JSON object, e.g. '{\"history_len\": 120}'")
    t.add_argument("--original-setup", action="store_true")
    t.add_argument("--until-settled", action="store_true")
    t.add_argument("--force", action="store_true")
    t.set_defaults(func=cmd_train)
    m = sub.add_parser("train-many")
    m.add_argument("--plan", required=True)
    m.add_argument("--workers", type=int, default=9)
    m.add_argument("--memory-gb", type=float, default=10.0)
    m.add_argument("--poll", type=float, default=30.0)
    m.add_argument("--adopt-minutes", type=float, default=30.0)
    m.set_defaults(func=cmd_train_many)
    e = sub.add_parser("extend")
    e.add_argument("--run", required=True)
    e.add_argument("--steps", type=int, default=TR.EXTEND_STEP)
    e.set_defaults(func=cmd_extend)
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    r.add_argument("--which", default="latest", choices=["latest", "best", "completed"])
    r.set_defaults(func=cmd_report)
    i = sub.add_parser("import-runs")
    i.add_argument("--old", default=str(OLD))
    i.add_argument("--clean", action="store_true")
    i.set_defaults(func=cmd_import)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
