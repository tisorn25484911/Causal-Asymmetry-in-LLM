"""Compute and cache the test report and main machine of every finished new-process run, as runs finish.
Stops when all 45 are cached, or when the training runner has stopped and nothing is left to do."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TM = Path("/Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version/test_models")
sys.path.insert(0, str(TM))
import matplotlib
matplotlib.use("Agg")
import analysis as A                                                          # noqa: E402
import training as TR                                                         # noqa: E402

PLAN = sys.argv[1] if len(sys.argv) > 1 else "plans/new_processes_folders.json"   # a {seed: [run folder]} file
folders = [Path(f) for items in json.loads((TM / PLAN).read_text()).values() for f in items]
done = set()
while True:
    for f in folders:
        if f in done or not (f / "completed.pt").exists():
            continue
        start = time.time()
        rep = A.report(f)
        m = A.machine(f)
        done.add(f)
        main = TR.load_model_module(rep["model"]).MAIN_PREDICTOR
        print(f"{time.strftime('%H:%M')} {f.parent.name}/{f.name}: excess {rep['metrics'][f'excess_{main}_future']:.4f}, "
              f"machine {m['n_states']} states ({time.time() - start:.0f} s)", flush=True)
    if len(done) == len(folders):
        break
    runner = subprocess.run(["pgrep", "-f", "main.py train-many|after_new_processes"], capture_output=True).returncode == 0
    if not runner and all(f in done or not (f / "completed.pt").exists() for f in folders):
        break
    time.sleep(300)
print(f"{time.strftime('%H:%M')} cached {len(done)} of {len(folders)}", flush=True)
