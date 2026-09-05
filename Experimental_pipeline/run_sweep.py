#!/usr/bin/env python
"""
Run the pipeline across a grid of processes.

    python Experimental_pipeline/run_sweep.py --sweep-coin 0.15 0.35 0.55 0.75 0.95 --repeats 5
    python Experimental_pipeline/run_sweep.py --sweep-flower 2 4 6 8 10 --repeats 5
    python Experimental_pipeline/run_sweep.py --sweep-coin 0.15 0.35 --dry-run
    python Experimental_pipeline/run_sweep.py --sweep-coin ... --plots-only

Writes main_results/sweep_params/sweep_{coin,flower}/ -- grid.pkl, the per-cell
F1-F4, and F4_complexity_grid.png.

THE GRID IS CROSSED WITH ITSELF: k values give k**2 cells, not k.  Five coin
values is 25 cells, ten is 100.  At ~130 s per repeat and 4 models per repeat
that is the difference between a long evening and most of a week, so --dry-run
prints the cost and exits.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import figures as FIG                                                   # noqa: E402
from config import CONFIG, coin_grid_specs, flower_grid_specs           # noqa: E402
from pipeline import run_sweep                                          # noqa: E402
from training import load_pkl, mkdir                                    # noqa: E402

OUT_DEFAULT = os.path.join(_ROOT, "main_results", "sweep_params")
SEC_PER_REPEAT = 130.0          # measured, 4 models


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-coin", nargs="+", type=float, metavar="P",
                    help="values crossed to give the (p,q) grid")
    ap.add_argument("--sweep-flower", nargs="+", type=int, metavar="N",
                    help="values crossed to give the (n,m) grid")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out-root", default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the cells and the estimated cost, then exit")
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw from grid.pkl, no training")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.sweep_coin and not args.sweep_flower:
        raise SystemExit("pass --sweep-coin and/or --sweep-flower")

    cfg = dict(CONFIG)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["max_epochs"] = args.epochs

    jobs = []
    if args.sweep_coin:
        jobs.append(("sweep_coin", coin_grid_specs(cfg, args.sweep_coin)))
    if args.sweep_flower:
        jobs.append(("sweep_flower", flower_grid_specs(cfg, args.sweep_flower)))

    for name, specs in jobs:
        hours = len(specs) * args.repeats * SEC_PER_REPEAT / 3600
        print(f"\n  {name}: {len(specs)} cells x {args.repeats} repeats "
              f"x 4 models = {len(specs)*args.repeats*4} trainings, ~{hours:.1f} h")
        for s in specs:
            print(f"    {s['tag']:<20} V={s['num_token']:<3} K={s['n_states']:<3} "
                  f"C+={s['C_plus']:.3f} C-={s['C_minus']:.3f} "
                  f"true k {s['true_k_fw']}/{s['true_k_bw']}")

    if args.dry_run:
        print("\n  --dry-run: nothing trained")
        return 0

    for name, specs in jobs:
        out = mkdir(os.path.join(args.out_root, name))
        if args.plots_only:
            path = os.path.join(out, "grid.pkl")
            if not os.path.exists(path):
                print(f"  {name}: no grid.pkl, skipped")
                continue
            grid = load_pkl(path)
            for rec in grid["cells"].values():
                FIG.draw_all(rec, os.path.join(out, rec["spec"]["tag"]))
            print(f"  redrew -> {FIG.draw_grid(grid, out)}")
        else:
            run_sweep(specs, cfg, out, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
