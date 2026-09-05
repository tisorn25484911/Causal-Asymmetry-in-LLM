#!/usr/bin/env python
"""
Run the pipeline on one process, or on all seven, for N repeats.

    python Experimental_pipeline/run_process.py --process coin --p 0.3 --q 0.4 --repeats 30
    python Experimental_pipeline/run_process.py --process flower --n 2 --m 8 --repeats 30
    python Experimental_pipeline/run_process.py --all --repeats 30       # the seven
    python Experimental_pipeline/run_process.py --all --plots-only       # redraw only

Writes main_results/trainings/<tag>/{repeats.pkl, F1..F4}, and with --all also
main_results/arc_comparision/F4_all_processes.png.

Per repeat this trains FOUR models -- {discrete, onehot} x {forward, backward} --
on one realisation.  Measured ~130 s per repeat, so the seven at 30 repeats is
roughly 7.5 hours.  Time a short run before committing to a long one.
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import figures as FIG                                            # noqa: E402
from config import CONFIG, baseline_specs, coin_spec, flower_spec  # noqa: E402
from pipeline import run_process                                 # noqa: E402
from training import load_pkl, mkdir, save_run_config            # noqa: E402

OUT_DEFAULT = os.path.join(_ROOT, "main_results", "trainings")
ARC_DIR     = os.path.join(_ROOT, "main_results", "arc_comparision")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", choices=("coin", "flower"),
                    help="which process; omit and pass --all for the seven")
    ap.add_argument("--p", type=float); ap.add_argument("--q", type=float)
    ap.add_argument("--n", type=int);   ap.add_argument("--m", type=int)
    ap.add_argument("--all", action="store_true",
                    help="the seven baseline processes")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out-root", default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=None,
                    help="override the base seed; repeat i uses seed+i")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override max_epochs (for a quick shakedown)")
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw every figure from repeats.pkl, no training")
    return ap.parse_args(argv)


def select_specs(args, cfg) -> list:
    if args.all:
        return baseline_specs(cfg)
    if args.process == "coin":
        if args.p is None or args.q is None:
            raise SystemExit("--process coin needs --p and --q")
        return [coin_spec(cfg, args.p, args.q)]
    if args.process == "flower":
        if args.n is None or args.m is None:
            raise SystemExit("--process flower needs --n and --m")
        return [flower_spec(cfg, args.n, args.m)]
    raise SystemExit("pass --process (with its parameters) or --all")


def main(argv=None):
    args = parse_args(argv)
    cfg = dict(CONFIG)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["max_epochs"] = args.epochs

    specs = select_specs(args, cfg)
    mkdir(args.out_root)
    save_run_config(cfg, os.path.join(os.path.dirname(args.out_root),
                                      "run_config.json"))

    t0, recs = time.time(), []
    for spec in specs:
        path = os.path.join(args.out_root, spec["tag"], "repeats.pkl")
        if args.plots_only:
            if not os.path.exists(path):
                print(f"  {spec['tag']}: no repeats.pkl, skipped")
                continue
            rec = load_pkl(path)
            for p in FIG.draw_all(rec, os.path.dirname(path)):
                print(f"  redrew -> {p}")
        else:
            rec = run_process(spec, cfg, args.out_root, args.repeats)
        recs.append(rec)

    if len(recs) > 1:
        mkdir(ARC_DIR)
        print(f"\n  pooled -> {FIG.draw_pooled(recs, ARC_DIR)}")
    print(f"\n  total {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
