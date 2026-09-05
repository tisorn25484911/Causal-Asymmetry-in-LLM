# Reorganisation Fix Plan

**Status: EXECUTED 2026-08-12.** All of §5 is applied and every gate in §6 passes;
see [§10 Execution record](#10-execution-record) for the measured results, the two
deviations from the plan and what remains open.
**Written:** 2026-08-12, after the tree was split into `Transformer_model/`,
`Experimental_setup/`, `Jupyter_notebooks/`, `Run_logs/` and `All_Results/`.

**Bottom line as found: the repository was 100 % non-functional as a codebase.**
Every one of the six runners, both post-hoc scripts, the cluster-plot script and
the whole 66-test suite failed at import. No data was lost, and no fix required
retraining anything. The cause was a single structural fact — the modules are
still *flat* (`from utils import ...`) but now live in two sibling directories,
and Python only ever puts the *entry script's own* directory on `sys.path`.

Sections 1–9 are the diagnosis and the plan, written before any change and left in
the present tense as the record of what was found. §10 records the execution.

Every claim below was verified by running it, not by reading. The verification
commands and their real output are quoted inline so this document can be audited.

---

## Table of contents

1. [What the reorganisation actually did](#1-what-the-reorganisation-actually-did)
2. [Failure inventory](#2-failure-inventory-verified)
   - [Class A — import breakage (hard, loud)](#class-a--import-breakage-hard-loud)
   - [Class B — path breakage (silent, and the dangerous one)](#class-b--path-breakage-silent-and-the-dangerous-one)
   - [Class C — documentation](#class-c--documentation)
   - [Class D — git bookkeeping](#class-d--git-bookkeeping)
   - [Class E — pre-existing breakage, NOT caused by the move](#class-e--pre-existing-breakage-not-caused-by-the-move)
3. [Two things that are *not* broken (verified, and they constrain the design)](#3-two-things-that-are-not-broken)
4. [Fix design and why](#4-fix-design-and-why)
5. [File-by-file edit list](#5-file-by-file-edit-list)
6. [Verification protocol](#6-verification-protocol)
7. [Where the reorganisation does not make sense — suggestions](#7-where-the-reorganisation-does-not-make-sense--suggestions)
8. [Deliberately not changed](#8-deliberately-not-changed)
9. [Execution order](#9-execution-order)
10. [Execution record — measured results, deviations, what is still open](#10-execution-record)

---

## 1. What the reorganisation actually did

`git status` reports 51 tracked paths as deleted and five new untracked
directories. Nothing was deleted in reality — everything moved:

| New directory | Contents | Count |
|---|---|---|
| `Transformer_model/` | `Data_generation.py`, `Flower_process_generation.py`, `OneHot_model.py`, `Training_model.py`, `Model_analysis.py`, `utils.py`, `Test_data_eval.py`, `plot_state_clusters.py` | 8 `.py` |
| `Experimental_setup/` | `configs.py`, `run_experiments.py`, `run_statistical_trj.py`, `run_sweep_experiment.py`, `sanity_check.py`, `pq_experiment.py`, `LLM_asymmetry_testing.py` | 7 `.py` |
| `Jupyter_notebooks/` | 4 notebooks + `build_walkthrough.py` | 5 |
| `Run_logs/` | 9 logs, `requirements.txt`, `requirements-lock.txt`, `run_all.sh`, `run_rest.sh`, `run_all_done.marker`, a `.code-workspace` | 15 |
| `All_Results/` | `results/`, `results_smoke/`, `results_quick/`, `results_large/`, `results_trajectories/`, `results_sweep/`, `results_sweep_wd/` | 7 dirs |

Unmoved, still at the repo root: `tests/`, `implementation_logbook/`,
`lightning_logs/`, `sanity_check_flower_process/`, `README.md`, `HOW_TO_RUN.md`,
`.gitignore`.

### The intended layering, and where it leaks

The split is *almost* a clean two-layer design — model/primitives below,
experiment drivers above. Verified with an AST import graph:

```
Experimental_setup → Transformer_model     5 files   (expected direction)
Transformer_model  → Experimental_setup    2 files   (LEAK)
```

The leak: `Transformer_model/Test_data_eval.py` and
`Transformer_model/plot_state_clusters.py` both do `from configs import ...`,
and `configs.py` now lives in `Experimental_setup/`. So `Transformer_model/` is
**not** a self-contained lower layer. This does not change the fix (the
bootstrap puts both directories on the path either way) but it is a real design
smell — see [§7.1](#71-two-scripts-are-in-the-wrong-layer-recommended-fix).

---

## 2. Failure inventory (verified)

### Class A — import breakage (hard, loud)

Python inserts the **script's own directory** into `sys.path[0]`, never the cwd.
So `python Experimental_setup/run_experiments.py` can see `configs` (same
directory) but not `Data_generation` (in `Transformer_model/`).

**A1. All six runners fail immediately.** Verified from the repo root:

```
run_experiments          ModuleNotFoundError: No module named 'Data_generation'
run_statistical_trj      ModuleNotFoundError: No module named 'Data_generation'
run_sweep_experiment     ModuleNotFoundError: No module named 'Model_analysis'
sanity_check             ModuleNotFoundError: No module named 'Data_generation'
pq_experiment            ModuleNotFoundError: No module named 'Data_generation'
LLM_asymmetry_testing    ModuleNotFoundError: No module named 'Data_generation'
```

Note this is *not* fixable by `cd`-ing anywhere. `cd Experimental_setup && python
run_experiments.py` fails identically, because the script directory is still
`Experimental_setup/`.

**A2. The two `Transformer_model/` scripts fail on `configs`** (the layering
leak above):

```
Test_data_eval           ModuleNotFoundError: No module named 'configs'
plot_state_clusters      ModuleNotFoundError: No module named 'configs'
```

**A3. The entire test suite fails to collect — 0 of 66 tests run.**
`tests/test_theory.py:19` bootstraps the *repo root* onto `sys.path`:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

That was correct when the modules were at the root. They no longer are:

```
tests/test_theory.py:21: in <module>
    from Data_generation import coin_generation
E   ModuleNotFoundError: No module named 'Data_generation'
ERROR tests/test_theory.py
!!!! Interrupted: 1 error during collection !!!!
no tests collected, 1 error in 1.08s
```

**A4. All four notebooks fail at their first import cell.** They run with cwd =
`Jupyter_notebooks/` and import `Data_generation`, `Training_model`,
`Model_analysis`, `OneHot_model`, `utils`, `configs`, `pq_experiment`,
`LLM_asymmetry_testing` — none of which are reachable from there.

**A5. `build_walkthrough.py` has contradictory requirements.** Line 87 is
`sys.path.insert(0, os.getcwd())`, which only works if cwd holds the modules —
no directory does any more. Worse, its data reads (`results_quick/…`,
`sanity_check_flower_process/…`) want cwd = repo root, while its output write
`nbformat.write(nb, "walkthrough.ipynb")` (line 888) wants cwd =
`Jupyter_notebooks/`, which is where `walkthrough.ipynb` now lives. No single cwd
satisfies both; this needs explicit anchoring, not a `cd`.

### Class B — path breakage (silent, and the dangerous one)

Class A is harmless in the end — it crashes in one second and you fix it. Class B
is the part that can cost real money, because **`mkdir(..., exist_ok=True)`
turns a wrong path into a silent fresh start rather than an error.**

**B1. THE BIG ONE — a wrong `out_root` silently discards 23 h of results and
re-runs everything.** The resume mechanism is:

```python
out_root      = mkdir(args.out_root)                      # creates it if absent
combined_path = os.path.join(out_root, "all_sweep.pkl")
combined      = load_combined(combined_path)              # returns {} if absent
...
todo = [s for s in specs
        if args.redo
        or len((combined.get(s["tag"], {}) or {}).get("runs", [])) < args.repeats]
```

`load_combined` returns `{}` for a missing file *by design* (that is what makes
the first run work), so a stale relative path produces an empty `combined`, an
empty skip list, and a full re-run — with no warning at all. Verified against the
real pickle:

```
PROOF — resume filter on All_Results/results_sweep_wd/wd1.000/all_sweep.pkl
   records present : 123
   would SKIP      : 122
   would RUN       : 3    (1 partial + 2 never started)

Same filter when out_root is wrong (dir absent -> load_combined returns {}):
   would SKIP      : 0
   would RUN       : 125   <-- full 23 h re-run, silently
```

Affected defaults, all now stale:

| File | Line | Current default | Points at |
|---|---|---|---|
| `run_sweep_experiment.py` | 165 | `OUT_ROOT_DEFAULT = "results_sweep"` | nothing (data is in `All_Results/`) |
| `run_statistical_trj.py` | 145 | `OUT_ROOT_DEFAULT = "results_trajectories"` | nothing |
| `configs.py` | 112 | `out_root = "results_smoke"` | nothing |
| `configs.py` | 139 | `out_root = "results_quick"` | nothing |
| `configs.py` | 164 | `out_root = "results_large"` | nothing |

This is exactly the failure that already bit us once: the λ = 1.0 coin run
crashed with
`FileNotFoundError: 'results_sweep_wd/wd1.000/all_sweep.pkl'` when the folder
moved mid-run. That one at least crashed *loudly*, because the directory
vanished after `mkdir` had already run. A fresh invocation will not be so kind.

**B2. The baseline cross-check silently disappears.**
`run_sweep_experiment.py:987` defaults `--baseline` to
`results_trajectories/all_trajectories.pkl`. `baseline_crosscheck` opens it via
the same `load_combined`, then:

```python
base = load_combined(baseline_path)
if not base:
    return                       # <- silent
```

So the sweep's only end-to-end regression test against the 100-repeat baseline
now vanishes without printing a word. Every future sweep would look fine while
having lost its cross-check.

**B3. `test_old_checkpoints_still_load` silently degrades to a skip.**
`tests/test_theory.py:846` globs `"results_quick/models/*_fw.pt"` and calls
`pytest.skip` when it finds nothing. Verified:

```
old glob 'results_quick/models/*_fw.pt'             -> 0 files  (test SKIPS)
new glob 'All_Results/results_quick/models/*_fw.pt' -> 7 files  (test RUNS)
```

A green suite that quietly stopped checking backward compatibility is worse than
a red one.

**B4. `build_walkthrough.py` data paths** — lines 325, 546, 627, 761 read
`sanity_check_flower_process/models/coin_p05_q05_fw.pt`,
`results_quick/all_results.pkl`, `sanity_check_flower_process/{tag}/results.pkl`.
Two of those roots moved, one did not (see [§7.2](#72-sanity_check_flower_process-was-left-behind)). These crash loudly (`open()` on a
missing file), so they are annoying rather than dangerous.

**B5. `Run_logs/run_all.sh` and `Run_logs/run_rest.sh` are both dead.** They
`cd` to the repo root and then call `python sanity_check.py`,
`python run_experiments.py` — neither path exists any more. Their log redirects
(`> run_sanity.log`) would also drop logs at the repo root rather than in
`Run_logs/`.

**B6. Notebook `results_dir` literals.** `sequence_prediction.ipynb` and
`umap_analysis.ipynb` hard-code `results_dir = "results"`, which is now
`All_Results/results`.

**B7. The detached daemon scripts in `/tmp` reference the old layout.**
`/tmp/claude-501/wd_sweep_daemon.py`, `wd_coin_daemon.py`, `wd_coin_daemon2.py`
all spawn `run_sweep_experiment.py` with `--out-root results_sweep_wd/wd{λ}`.
Both the script path and the out-root are stale. They live outside the repo and
are ephemeral, but if any is relaunched as-is it hits B1 head-on. Worth
rewriting one of them into the repo as a proper launcher — see [§7.5](#75-the-launcher-daemons-live-in-tmp).

### Class C — documentation

`HOW_TO_RUN.md` is the operating manual and every command in it is now wrong.
Counted occurrences of bare `python <script>.py`:

| Doc | Stale commands | Stale result-path mentions | `requirements` refs |
|---|---|---|---|
| `HOW_TO_RUN.md` | 38 | 22 | 2 |
| `README.md` | 6 | 4 | 2 |

Plus:
- `README.md:831-855` prints a repository-structure tree that shows the **old
  flat layout** — it is now actively misleading.
- `HOW_TO_RUN.md:54` says `pip install -r requirements.txt`; the file is now
  `Run_logs/requirements.txt`.
- Both docs say the suite has **61 tests**; it has **66** (verified: `66 tests
  collected`). Stale since the weight-decay work, not the move.
- `README.md:487, 504, 1419` cite `factorial.log` and `divtest.log`, now in
  `Run_logs/`.

### Class D — git bookkeeping

The move is entirely unstaged, so git currently believes 51 tracked files were
deleted. Verified what a `git add -A` would do:

- **70 files would be added**, of which 24 are the previously tracked
  `results_trajectories/*.png` reappearing at their new path — correct and
  desirable.
- `Run_logs/run_large.log` (**101 MB**) stays ignored, matched by the existing
  `run_*.log` rule. Confirmed with `git check-ignore -v`. Good — that is the one
  file that must never enter history.
- `All_Results/results_quick/all_results.pkl` stays ignored: gitignore patterns
  without a leading slash match at **any depth**, so `results_quick/` still
  catches `All_Results/results_quick/`. The existing `.gitignore` therefore
  survives the move better than expected.
- **Newly committable and probably unwanted:** `All_Results/results_sweep/*.png`
  and `All_Results/results_sweep_wd/wd*/*.png` (12 files) plus their
  `run_config_*.json`. These were never ignored — a pre-existing gap, not
  something the move caused, but it will surface the moment the move is
  committed.

### Class E — pre-existing breakage, NOT caused by the move

Recording these so they are not mistaken for regressions, and so "works as it
used to" is honest about what "used to" means.

**E1. Three notebooks were already broken before the reorganisation.**
`metric_panel_plot.ipynb`, `sequence_prediction.ipynb` and `umap_analysis.ipynb`
all contain

```python
from Data_generation import coin_generation, flower_process_generation
```

but `flower_process_generation` was **deleted** from `Data_generation.py` during
the IMPROVEMENT_PLAN C4 de-duplication (verified: `grep -c "def
flower_process_generation" Transformer_model/Data_generation.py` → `0`; the file
carries a comment explaining the removal). These notebooks raise `ImportError`
regardless of layout. Fixing them means repointing the import at
`Flower_process_generation`, which is a *separate* repair from this plan and
should be flagged as such.

**E2. `lightning_logs/` is 34 MB of dead weight.** 32 `version_*` directories.
`Training_model.py:487-499` constructs the Trainer with `logger=False` and
`enable_checkpointing=False`, so Lightning writes nothing there any more. Purely
historical.

**E3. `run_all_done.marker` contains `done`** and is tracked. Harmless.

Not a problem, checked because it would have been: LaTeX **is** installed
(`/Library/TeX/texbin/latex`), so `LLM_asymmetry_testing.py`'s
`text.usetex = True` is fine.

---

## 3. Two things that are *not* broken

Both were verified because they would each have forced a much larger and riskier
plan. Neither does.

**3.1 No pickle references a repo class.** Loaded every result bundle through a
`pickle.Unpickler` subclass that records each `find_class` call:

```
All_Results/results_trajectories/all_trajectories.pkl
    total classes referenced: 4
    repo modules referenced : NONE
All_Results/results_sweep_wd/wd0.030/all_sweep.pkl
    total classes referenced: 4
    repo modules referenced : NONE
All_Results/results_quick/all_results.pkl
    total classes referenced: 4
    repo modules referenced : NONE
```

The four are numpy/builtins reconstructors. The bundles are plain nested dicts of
arrays and floats, so **no pickle is tied to a module path.**

**3.2 No `.pt` file references a repo class either.** `utils.save_weights` calls
`torch.save(model.state_dict(), path)` — an `OrderedDict` of tensors, not the
model object. All 34 checkpoints are portable.

**Why this matters:** it means converting the tree into real packages
(`Transformer_model/__init__.py` + `from Transformer_model.utils import …`) would
*not* invalidate any stored result. That option is genuinely open. I am still
not recommending it — see [§4](#4-fix-design-and-why) — but the reason is churn,
not data safety. Had a single pickle carried a repo class, the package rename
would have silently broken every stored result, and that would have been the
most important line in this document.

---

## 4. Fix design and why

Three candidate strategies, with the reason for the choice.

| Strategy | Cost | Verdict |
|---|---|---|
| **(a) Path bootstrap, flat modules kept** | 8 files × 6 lines, 1 new `conftest.py` | **Chosen** |
| (b) Real packages + `python -m` | rewrite ~40 import lines, all notebook cells, every documented command | Rejected — large diff, no benefit here |
| (c) `PYTHONPATH` in the shell / a `.pth` file | 0 code, but every bare `python x.py` still breaks and it is invisible to a future reader | Rejected — reintroduces the bug the moment anyone forgets |

(b) is the textbook answer and is safe from the data side (§3), but it changes
every import statement, every notebook cell and all 44 documented commands to
buy nothing this repo needs — there is one interpreter, one environment, no
distribution, no name collisions. The flat namespace is a legitimate choice for a
research tree; it just needs the two directories to be importable.

### 4.1 The import bootstrap

Six lines at the top of each **entry point**, before its first local import.
Anchored on `__file__`, so it is cwd-independent, and idempotent:

```python
# ── repo path bootstrap ───────────────────────────────────────────────────────
# The tree is split across Transformer_model/ and Experimental_setup/ but the
# modules are still flat (`from utils import ...`), so both directories must be
# importable.  Python only ever puts the *script's own* directory on sys.path,
# which is why this is needed and why cd-ing somewhere else does not help.
# Anchored on __file__, not the cwd, so the script runs from anywhere.
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

Library modules do **not** need it: once `Transformer_model/` is on the path,
`Model_analysis`'s own `from utils import …` resolves normally. Only the eight
files that are invoked directly need the block.

Verified sufficient — with both directories on the path, every module imports:

```
run_experiments          IMPORTS OK        sanity_check            IMPORTS OK
run_statistical_trj      IMPORTS OK        pq_experiment           IMPORTS OK
run_sweep_experiment     IMPORTS OK        LLM_asymmetry_testing   IMPORTS OK
Test_data_eval           IMPORTS OK        plot_state_clusters     IMPORTS OK
```

and `pytest` collects the full suite: `66 tests collected in 10.04s`.

### 4.2 The path anchor

Add to `Transformer_model/utils.py` (imported by everything that writes output):

```python
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(p: str) -> str:
    """
    Resolve a repo-relative path against the repo root rather than the cwd.

    Every output directory in this repo is written repo-relative
    ("All_Results/results_quick"), which was unambiguous while the scripts lived
    at the root and is not any more.  Absolute paths pass through untouched, so
    `--out-root /tmp/scratch` still means /tmp/scratch.
    """
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
```

Applied at exactly **one line per runner** — the audit found the out_root
resolution is already fully centralised, which is why this fix is small:

| File | Line | Becomes |
|---|---|---|
| `run_experiments.py` | 867 | `out_root = repo_path(cfg["out_root"])` |
| `run_statistical_trj.py` | 973 | `out_root = repo_path(args.out_root)` |
| `run_sweep_experiment.py` | 1015 | `out_root = mkdir(repo_path(args.out_root))` |
| `plot_state_clusters.py` | 220 | `out_root = repo_path(args.out_root or cfg["out_root"])` |
| `Test_data_eval.py` | 460 | `results_dir = repo_path(RUN["results_dir"] or cfg["out_root"])` |
| `LLM_asymmetry_testing.py` | 660 | `results_dir = repo_path(RUN["results_dir"] or cfg["out_root"])` |
| `sanity_check.py` | 768 | `mkdir(repo_path(OUT_ROOT))` and pass the resolved value onward |

**Design note — this also rewrites user-supplied relative paths.** `--out-root
foo` resolves to `<repo>/foo`, not `./foo`. That is deliberate: it makes the
documented invariant ("run from the repo root") *true by construction* instead of
a thing you must remember, and every documented command already assumes it. The
cost is one surprise for someone running from another directory with a relative
argument; absolute paths, which is what the `/tmp/traj_check` examples use, are
unaffected. This trade must be written into `HOW_TO_RUN.md` §"Where output goes".

**Keep the config strings relative.** `configs.py` gains the `All_Results/`
prefix but stays a *relative* string, because `out_root` is copied verbatim into
each run's `run_config_*.json` provenance record. Baking a machine-specific
absolute path into provenance would be a regression; resolving at the point of
use keeps the JSON portable.

---

## 5. File-by-file edit list

### 5.1 `Transformer_model/utils.py`
- Add `REPO_ROOT` and `repo_path()` as in §4.2. No other change; every existing
  helper keeps its signature.

### 5.2 New file: `conftest.py` at the repo root
```python
"""
Make the split tree importable for pytest.

The modules are flat but live in Transformer_model/ and Experimental_setup/.
pytest imports test files without running them as scripts, so the bootstrap in
tests/test_theory.py is not enough on its own for future test files -- this puts
both directories on sys.path for the whole session, once.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
```
This also makes `pytest` work from any cwd, since pytest locates `conftest.py`
by rootdir rather than by cwd.

### 5.3 Bootstrap block — 8 files
Insert the §4.1 block immediately before the first local import:

| File | Insert before line | First local import |
|---|---|---|
| `Experimental_setup/run_experiments.py` | 89 | `from configs import CONFIGS, QUICK` |
| `Experimental_setup/run_statistical_trj.py` | 116 | `from configs import CONFIGS` |
| `Experimental_setup/run_sweep_experiment.py` | 139 | `from configs import CONFIGS` |
| `Experimental_setup/sanity_check.py` | 67 | `from Data_generation import …` |
| `Experimental_setup/pq_experiment.py` | **1** | `from Data_generation import …` (file starts with it — block goes at the very top) |
| `Experimental_setup/LLM_asymmetry_testing.py` | 10 | `from configs import CONFIGS` |
| `Transformer_model/Test_data_eval.py` | 11 | `from Data_generation import …` |
| `Transformer_model/plot_state_clusters.py` | 42 | `from configs import CONFIGS, QUICK` |

Order matters: the block must sit above the local imports but may sit below the
`matplotlib.use("Agg")` calls, which are independent.

### 5.4 Default paths
- `Experimental_setup/configs.py:112` → `out_root = "All_Results/results_smoke"`
- `Experimental_setup/configs.py:139` → `out_root = "All_Results/results_quick"`
- `Experimental_setup/configs.py:164` → `out_root = "All_Results/results_large"`
- `Experimental_setup/run_statistical_trj.py:145` →
  `OUT_ROOT_DEFAULT = "All_Results/results_trajectories"`
- `Experimental_setup/run_sweep_experiment.py:165` →
  `OUT_ROOT_DEFAULT = "All_Results/results_sweep"`
- `Experimental_setup/run_sweep_experiment.py:987` → `--baseline` default
  `"All_Results/results_trajectories/all_trajectories.pkl"`, wrapped in
  `repo_path()` at the call site (line 1137).
- Apply `repo_path()` at the seven resolution sites in the §4.2 table.

### 5.5 Turn the two silent degradations into loud ones
Class B is dangerous specifically because it is quiet. Two small guards:

- **`run_sweep_experiment.py`, `baseline_crosscheck`** — replace the bare
  `if not base: return` with a warning naming the path it tried, so a missing
  baseline is visible in the log instead of absent from it.
- **`run_sweep_experiment.py`, after `load_combined(combined_path)`** — when the
  pickle is absent but the directory already existed and holds figures or a
  `run_config_*.json`, print a prominent warning that this looks like a
  populated output directory with no resumable state. This is the cheap
  insurance against a repeat of the 23 h scare. Purely additive; no behaviour
  change on a genuine first run.

### 5.6 `tests/test_theory.py`
- Line 19: replace the single root insert with the two-directory bootstrap (keep
  it even with `conftest.py` present, so `python tests/test_theory.py` and
  direct-file runs still work).
- Line 846: `glob.glob("results_quick/models/*_fw.pt")` →
  root-anchored `All_Results/results_quick/models/*_fw.pt` via `repo_path`, so
  the test actually runs (7 checkpoints available) instead of skipping.
- Grep the file for any other bare relative path before finishing — the audit
  found only this one, but it is worth a second pass at edit time.

### 5.7 `Jupyter_notebooks/build_walkthrough.py`
- Line 87: replace `sys.path.insert(0, os.getcwd())` with the §4.1 bootstrap.
- Lines 325, 546, 627, 761: wrap the four data paths in `repo_path()`.
- Line 888: `nbformat.write(nb, "walkthrough.ipynb")` →
  write next to `__file__` so the notebook lands in `Jupyter_notebooks/`
  regardless of cwd. This is the fix for the contradictory-cwd problem (A5):
  reads anchored at the repo root, the write anchored at the script.

### 5.8 The four notebooks
Add one bootstrap cell at the top of each. Notebooks have no `__file__`, so walk
up to the `.git` marker — idempotent if the cell is re-run:

```python
import os, sys
_p = os.path.abspath(os.getcwd())
while not os.path.isdir(os.path.join(_p, ".git")) and _p != os.path.dirname(_p):
    _p = os.path.dirname(_p)
ROOT = _p
os.chdir(ROOT)                        # so repo-relative data paths resolve
for _d in ("Transformer_model", "Experimental_setup"):
    if os.path.join(ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, _d))
```

`os.chdir` is right *here* and wrong in the scripts: a notebook has no CLI
arguments to surprise, and its many bare relative literals all assume the repo
root.

Then fix `results_dir = "results"` → `"All_Results/results"` in
`sequence_prediction.ipynb` and `umap_analysis.ipynb`.

**Do not** attempt E1 (the `flower_process_generation` import) in the same pass —
it is a pre-existing bug with its own correct fix (repoint at
`Flower_process_generation`), and mixing it in makes the reorganisation diff
impossible to review. Land it as a separate follow-up commit.

### 5.9 `Run_logs/run_all.sh`, `Run_logs/run_rest.sh`
Repoint the script paths and the log destinations:

```zsh
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version
python Experimental_setup/sanity_check.py                   > Run_logs/run_sanity.log 2>&1
python Experimental_setup/run_experiments.py --config QUICK > Run_logs/run_quick.log  2>&1
python Experimental_setup/run_experiments.py --config LARGE > Run_logs/run_large.log  2>&1
echo "ALL RUNS COMPLETE" > Run_logs/run_all_done.marker
```

Note the `run_*.log` gitignore rule matches at any depth, so the logs stay
ignored in their new home — verified.

### 5.10 `.gitignore`
- Add `All_Results/results_sweep/`, `All_Results/results_sweep_wd/` (or the
  looser `results_sweep*/`) if the sweep PNGs should stay out of history —
  decide deliberately, since committing the move is what first exposes them.
- Add `lightning_logs/` — already present, no change needed.
- Everything else survives the move unchanged, because the patterns are
  depth-agnostic. Verified per-path with `git check-ignore -v`.

### 5.11 Documentation
- **`README.md:831-855`** — replace the flat tree with the real one, grouped by
  the new directories, and keep the one-line description per file.
- **`README.md`** — 6 commands, 4 result paths, the `requirements.txt` reference,
  and the `factorial.log` / `divtest.log` references.
- **`HOW_TO_RUN.md`** — 38 commands, 22 result paths, `pip install -r
  Run_logs/requirements.txt`, `pytest tests/ -q`, and the `61 tests` → `66
  tests` correction in both files.
- Add a short "Repository layout" section near the top of `HOW_TO_RUN.md`
  explaining the two-directory split, that all commands are run **from the repo
  root**, and the `repo_path` semantics from §4.2.
- Update the output-tree listings at `HOW_TO_RUN.md:595, 810, 1018` to show the
  `All_Results/` prefix.

---

## 6. Verification protocol

Run in order; each step is a hard gate. Steps 1-4 cost seconds and catch
everything Class A and B.

**1. Every entry point imports.** From the repo root *and* from `/tmp`, to prove
cwd-independence:
```bash
for f in run_experiments run_statistical_trj run_sweep_experiment; do
  python Experimental_setup/$f.py --help >/dev/null || echo "FAIL $f"
done
python -c "import sys; sys.path.insert(0,'Transformer_model'); sys.path.insert(0,'Experimental_setup')" 
cd /tmp && python <repo>/Experimental_setup/run_sweep_experiment.py --help >/dev/null || echo "FAIL cwd-independence"
```
The three argparse-less scripts (`sanity_check`, `pq_experiment`,
`LLM_asymmetry_testing`) must be smoke-tested by **import, not execution** —
they have no `--help`, so an unrecognised flag is ignored and they start
training. (Learned the hard way during this audit: a `--help` probe launched a
real `sanity_check.py` run. It was killed before writing anything —
`find sanity_check_flower_process -newermt "-20 minutes"` came back empty — but
the near-miss is the reason this warning is here.)

**2. Full suite green.** `pytest tests/ -q` → expect **66 passed**, and confirm
`test_old_checkpoints_still_load` **runs rather than skips**
(`pytest tests/ -q -rs` shows skip reasons; that test must not appear).

**3. Resume is intact — the most important check.** This must be verified with
`--dry-run` before any real run:
```bash
python Experimental_setup/run_sweep_experiment.py --sweep-coin --repeats 30 \
       --weight-decay 1.0 --out-root All_Results/results_sweep_wd/wd1.000 --dry-run
```
Then confirm against the recorded truth: the log must say it is skipping
**122** processes and running **3**. If it says it will run 125, stop — B1 is not
fixed and a real run would burn 23 h.

**4. Every default out_root points at existing data.** For each of SMOKE, QUICK,
LARGE, the trajectory and the sweep default, assert the resolved directory
already exists and is non-empty. A default that resolves to a *new empty*
directory is the B1 failure mode and must fail this check.

**5. `--plots-only` reproduces existing figures.** The cheapest true end-to-end
test, and it touches no training:
```bash
python Experimental_setup/run_statistical_trj.py --plots-only
python Experimental_setup/run_sweep_experiment.py --plots-only \
       --out-root All_Results/results_sweep_wd/wd0.030
```
Both must load the existing pickle, redraw, **and** print the baseline
cross-check table (which proves §5.5 and the `--baseline` fix). Compare a
regenerated PNG against the archived one — byte-identical is not required, but
the panel count and axis ranges must match.

**6. One real short run.** `python Experimental_setup/run_experiments.py --config
SMOKE` (~2 min) exercises training, weight saving, sidecar writing, pickling and
plotting through the real paths. Confirm output lands in
`All_Results/results_smoke/` and **not** in a fresh `results_smoke/` at the root.

**7. Notebooks.** Execute the first two cells of each of the four notebooks;
expect imports to succeed for `walkthrough.ipynb` and to fail on E1 for the
other three (documenting that E1 is untouched and separate).

**8. Git.** `git add -A` then `git status` — verify the 24
`results_trajectories` PNGs reappear as renames, that `Run_logs/run_large.log`
is absent from the staged set, and that nothing over ~1 MB is staged
unintentionally. Commit the move and the fixes **separately** so the mechanical
move is reviewable on its own.

---

## 7. Where the reorganisation does not make sense — suggestions

### 7.1 Two scripts are in the wrong layer (recommended fix)
`Test_data_eval.py` and `plot_state_clusters.py` sit in `Transformer_model/` but
import `configs` from `Experimental_setup/`, making the dependency between the
two directories bidirectional. Both are *entry points* — each has
`if __name__ == "__main__"`, `plot_state_clusters.py` has its own argparse — and
neither defines anything the model layer uses. They are experiment drivers.

**Suggestion: move both into `Experimental_setup/`.** `Transformer_model/` then
becomes a true leaf layer with zero outward dependencies — six primitive modules
(`Data_generation`, `Flower_process_generation`, `OneHot_model`,
`Training_model`, `Model_analysis`, `utils`) and nothing else, which is a
property worth having and worth stating in the README. Cost: two `git mv`s, plus
their entries in the docs. The path bootstrap makes this optional rather than
required, which is exactly why it should be done deliberately now rather than
left to rot.

### 7.2 `sanity_check_flower_process/` was left behind
Every results directory moved into `All_Results/` except this one, which is still
at the repo root — even though it is produced by `sanity_check.py` and is
results in exactly the same sense as `results_quick/`. It holds the positive and
null controls, which are among the more scientifically load-bearing outputs.

**Suggestion: move it to `All_Results/sanity_check_flower_process/`.** Doing so
requires updating, in one pass: `sanity_check.py:154`,
`build_walkthrough.py:325, 627, 761`, two cells in `walkthrough.ipynb`, the
`.gitignore` entry, and the README/HOW_TO_RUN references. Not moving it costs
nothing today and is a defensible "leave it alone" — but then it is worth a
sentence in the README saying why one results tree is elsewhere, because the
next reader will assume it is an oversight.

### 7.3 `requirements.txt` does not belong in `Run_logs/`
`requirements.txt` and `requirements-lock.txt` are environment specification, not
run output, and `pip install -r requirements.txt` from the repo root is the
single most conventional command in any Python repo. Burying them under
`Run_logs/` breaks that expectation for no gain.

**Suggestion: move both back to the repo root** (or to a `Environment/`
directory if the root must stay tidy). Same argument, more weakly, for
`run_all.sh` / `run_rest.sh`: they are launchers, not logs. A `Scripts/`
directory — or leaving them at the root — reads better than filing an executable
under `Run_logs/`.

### 7.4 `Run_logs/` holds 171 MB, dominated by one file
`run_large.log` alone is 101 MB and `run_sweep_wd_coin.log` is 40 MB. Both are
correctly gitignored, so this is disk hygiene rather than a repo problem, but a
`Run_logs/archive/` subdirectory (or gzipping the finished ones — measured at
**22×** on a 20 MB sample of `run_large.log`, so 171 MB becomes under 8 MB)
would make the directory navigable. Also
worth deleting `lightning_logs/` (34 MB, 32 versions): E2 established that
Lightning no longer writes there at all, so it is pure history.

### 7.5 The launcher daemons live in `/tmp`
The three fork+`setsid` daemons that ran the λ sweep are in
`/tmp/claude-501/`, which macOS will eventually clear — and they hard-code both
the old script path and the old `--out-root`. The overnight-launch capability is
genuinely useful and is currently one directory sweep away from being lost.

**Suggestion: promote one into the repo** as `Experimental_setup/launch_sweep.py`
with the λ list and out-root as CLI arguments, and document it in
`HOW_TO_RUN.md` next to the sweep section. It is also the natural place to put
the `check=False` fix: the daemon printed `ALL COIN GRIDS COMPLETE in 23.18 h`
while its last child had died of a `FileNotFoundError`, because a non-zero exit
from `subprocess.run(..., check=False)` is silently discarded. A launcher that
cannot report a failed child is worse than no launcher.

### 7.6 Two directory names read oddly
`Experimental_setup/` contains the runners that *execute* experiments, not just
their setup — `Experiments/` or `Runners/` would describe it better.
`Transformer_model/` contains the data generators and the analysis library as
well as the model, so `Core/`, `Library/` or `src/` would be more accurate. Low
priority, cosmetic, and renaming costs another round of doc edits — but if a
rename is ever going to happen, doing it in the same commit as this fix is much
cheaper than doing it later.

---

## 8. Deliberately not changed

- **Flat module names.** Kept, per §4. §3 establishes that packaging *would* be
  safe for the stored data; it is simply not worth the churn.
- **E1, the notebooks' `flower_process_generation` import.** Pre-existing, and it
  needs a real code fix, not a path fix. Separate commit.
- **The `lightning_logs/` contents.** Deleting 34 MB of stale output is a
  judgement call for the user, not a correctness fix.
- **The λ = 1.0 coin gap.** Three cells (1 partial + 2 never started) are still
  missing. That is a compute task, not a reorganisation fix, and it is
  *blocked on this plan*: running it before §5.4 lands would trip B1 and re-run
  all 125 cells. After the fix, the resumable command is
  ```bash
  python -u Experimental_setup/run_sweep_experiment.py --sweep-coin --repeats 30 \
         --weight-decay 1.0 --out-root All_Results/results_sweep_wd/wd1.000
  ```
  which verification step 3 confirms will skip 122 and run 3 (~10 min).
- **The `run_config_QUICK_flower.json` provenance workaround.** Still needed;
  unaffected by any of this.

---

## 9. Execution order

1. `utils.py` — `REPO_ROOT` + `repo_path()` (§5.1). Nothing else works without it.
2. Root `conftest.py` (§5.2) and the 8 bootstrap blocks (§5.3). **Gate:
   verification 1.**
3. `tests/test_theory.py` (§5.6). **Gate: verification 2 — 66 passed, no skip.**
4. Default paths and `repo_path()` call sites (§5.4). **Gate: verifications 3
   and 4 — this is the gate that protects the 23 h of existing results.**
5. Loud-failure guards (§5.5). **Gate: verification 5.**
6. Shell scripts and `.gitignore` (§5.9, §5.10).
7. `build_walkthrough.py` and the notebook bootstrap cells (§5.7, §5.8).
   **Gate: verification 7.**
8. Documentation (§5.11).
9. **Gate: verification 6** (SMOKE run) and **8** (git), then commit — move and
   fixes as two separate commits.
10. Only then: the λ = 1.0 coin gap (§8).

Steps 1-4 restore the repository. Steps 5-8 are what stop it breaking the same
way next time.

---

## 10. Execution record

Applied 2026-08-12 in the order given in §9. Every gate in §6 was run; the
measured results are below.

### 10.1 Gates

| Gate | Expected | Measured |
|---|---|---|
| 1. Entry points import | 8/8 | **8/8**, and from a foreign cwd (`/private/tmp`) |
| 2. Test suite | 66 passed, checkpoint test runs | **66 passed, 0 skipped** in 20.05 s |
| 3. Resume intact | skip the complete cells, train 3 | **97 skipped, 3 to train**, partial named: `sweep_coin_p095_q075 (18/30)` |
| 3b. Negative control | a wrong out-root must shout | **3-line warning**, `nothing to resume from` |
| 4. Defaults resolve | 6/6 to existing data | **6/6** (5 dirs non-empty, baseline 2.3 MB) |
| 5. `--plots-only` | figures reproduce; cross-check prints | **23/23 identical geometry**; cross-check restored, all 4 cells agree, \|z\| ≤ 0.98 |
| 6. SMOKE run | trains, writes to `All_Results/` | **67 files**, sidecars carry `optimizer: adamw`; **no stray `results_smoke/` at the root** |
| 6b. `plot_state_clusters.py` | runs end to end | **exit 0**, 4 `*_states_js.png` written |
| 7. Notebooks | bootstrap works; 2 fail on E1 | **4/4 bootstrap OK**; 2 import OK, 2 fail on E1 as predicted |
| 8. Git | move recorded, nothing large staged | 24 deletions ↔ 24 additions; largest staged file **2.74 MB**; the 101 MB and 40 MB logs, all sweep figures and all pickles **ignored** |

The baseline cross-check output, which is the strongest single piece of evidence
that the pipeline is intact end to end — it resolves the baseline path, loads a
different results tree, matches processes by key and reproduces the 100-repeat
numbers from 30 repeats:

```
  process              sweep dCE      sem    n  baseline dCE      sem    n       z  agree?
  flower_n2_m6          +0.00103  0.00054   30      +0.00064  0.00035  100   +0.61  yes
  flower_n2_m8          +0.00233  0.00067   30      +0.00238  0.00043  100   -0.07  yes
  flower_n4_m2          -0.00057  0.00042   30      -0.00066  0.00027  100   +0.18  yes
  flower_n6_m4          -0.00142  0.00073   30      -0.00061  0.00039  100   -0.98  yes
```

### 10.2 Two deviations from the plan

**(a) `--dry-run` could not answer the question §6 gate 3 asks.** The resume
filter was computed *after* the `--dry-run` early return, so the one flag whose
job is "tell me what you are about to do" could not report whether a run would
resume or restart. The `todo`/`n_skip` computation was moved above that return and
now prints a `resume plan` line — plus the partial-cell listing and, when the
pickle is absent, the three-line warning. This was not in §5.5 as written; it is
the same intent applied to the place the gate actually exercised.

**(b) The plan asserted both harnesses resume; only one does.** Verified in the
code: `run_statistical_trj.py` has **no** repeat-level resume. It merges by
process (`combined[tag] = rec`, so other processes survive) but discards the
selected process's existing repeats and retrains all `--repeats`. `HOW_TO_RUN.md`
called this "one process, resumable", which invites exactly the wrong inference.
Both docs now carry the granularity table in §14.6, and the plan's §6 gate 3
wording ("122 skipped") was for the full 125-spec grid — the resumable command
uses `--sweep-coin`, so the correct figure is 97 of 100 coin cells.

Two smaller corrections found while executing: `build_walkthrough.py` never
imported `os` at module level (its `import os` lives inside a generated cell
*string*), so the `__file__`-anchored write raised `NameError` until `import os`
was added — caught by running the script in a scratch copy rather than in place.
And `walkthrough.ipynb` turned out to hold **16 executed cells with saved
outputs**, so it was patched in place rather than regenerated; regenerating it, as
§5.7 implied was safe, would have destroyed them.

### 10.3 Deliberately left open

- **E1** — `metric_panel_plot.ipynb` (`cannot import name 'CFG' from
  LLM_asymmetry_testing`) and `umap_analysis.ipynb` (`flower_process_generation`
  from `Data_generation`). Both were broken *before* the reorganisation and both
  are now *reachable* — the modules import, the names are stale. Separate fix,
  separate commit, per §5.8.
- **§7.1–7.6 suggestions.** None applied; all are judgement calls for the owner.
  §7.1 (moving the two entry points out of `Transformer_model/`) is now noted in
  the README structure section so it is visible rather than buried here.
- **The λ = 1.0 coin gap.** Unblocked by this work — gate 3 confirms the command
  in §8 will skip 97 and train 3.
- **`.gitignore` choice.** `results_sweep/` and `results_sweep_wd/` are now
  ignored, preserving the pre-move status quo (they were never tracked). This is
  reversible with one line if the sweep figures should enter history; it was done
  this way because ignoring is recoverable and committing 12 binaries is not.
