"""Progress of the 45 new-process runs: update reached, seconds per update, hours left at that pace."""
import json
import os
from pathlib import Path

ROOT = Path("/Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version/test_models")
folders = json.loads((ROOT / "plans/new_processes_folders.json").read_text())
done = running = 0
for seed, items in folders.items():
    for f in map(Path, items):
        name = f"{f.parent.name[6:]:22s} {f.name.rsplit('_', 1)[0]:38s}"
        if (f / "completed.pt").exists():
            done += 1
            continue
        p = f / "progress.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        running += 1
        rate = d["seconds"] / max(d["update"], 1)
        steps = json.loads((f / "settings.json").read_text())["steps"]
        metric = next(k for k in d if k.startswith("excess_") and k.endswith("_future"))
        print(f"{name} update {d['update']:6d}  {rate:5.2f} s/update  {(steps - d['update']) * rate / 3600:5.1f} h left"
              f"  {metric} {d[metric]:+.4f}")
print(f"finished {done} of 45, running {running}")
