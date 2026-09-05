# How to run

```bash
conda activate asym
cd /Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version
```

Every command is run from the repository root.

---

## 1. Verify first

```bash
python Experimental_pipeline/checks.py           # samplers vs closed forms, ~20 s
python Experimental_pipeline/checks.py --full    # + the extraction gate, ~5 min
```

The fast checks hold the generators against the formulas in `processes.py`:
empirical transition matrices, occupancies and entropy rates from 10⁵ tokens,
all to within 1e-2, plus the identity `entropy(occupancy) == C`. The `--full`
gate trains one real discrete model and asserts its `S_emp` lands on `C+`.

**If `--full` fails, stop.** It means something in the geometry or the
bottleneck moved that the sampler checks cannot see, and no figure downstream is
trustworthy until it is found.

## 2. One process, or all seven

```bash
python Experimental_pipeline/run_process.py --process coin   --p 0.3 --q 0.4 --repeats 30
python Experimental_pipeline/run_process.py --process flower --n 2   --m 8   --repeats 30
python Experimental_pipeline/run_process.py --all --repeats 30       # the seven baselines
python Experimental_pipeline/run_process.py --all --plots-only       # redraw, no training
```

Writes `main_results/trainings/<tag>/` — `repeats.pkl` plus F1–F4. With `--all`
it also writes `main_results/arc_comparision/F4_all_processes.png`.

The seven baselines are coin (0.1,0.9), (0.3,0.4), (0.4,0.8) and flower (2,6),
(2,8), (4,2), (6,4) — the same processes the archived `00_base` used, with the
same dice seed, so the numbers are comparable to it.

## 3. A grid

```bash
python Experimental_pipeline/run_sweep.py --sweep-coin 0.15 0.35 0.55 0.75 0.95 --repeats 5
python Experimental_pipeline/run_sweep.py --sweep-flower 2 4 6 8 10 --repeats 5
python Experimental_pipeline/run_sweep.py --sweep-coin 0.15 0.55 0.95 --dry-run
```

**The grid is crossed with itself: k values give k² cells, not k.** Five coin
values is 25 cells; ten is 100. `--dry-run` prints the cells and the estimated
cost and exits — use it before every long run.

Writes `main_results/sweep_params/sweep_{coin,flower}/` — `grid.pkl`,
`F4_complexity_grid.png`, and a `<cell_tag>/` with F1–F4 per cell.

## 4. What it costs

Measured at **~130 s per repeat**, which is four models: {discrete, onehot} ×
{forward, backward}.

| Run | Cells | Repeats | Trainings | Wall |
|---|---|---|---|---|
| `--process coin --p .3 --q .4 --repeats 5` | 1 | 5 | 20 | ~11 min |
| `--all --repeats 30` | 7 | 30 | 840 | ~7.6 h |
| `--sweep-coin` 5 values, 5 repeats | 25 | 5 | 500 | ~4.5 h |
| `--sweep-coin` + `--sweep-flower`, 5 values each, 5 repeats | 50 | 5 | 1000 | ~9 h |
| `--sweep-coin` 10 values, 20 repeats | 100 | 20 | 8000 | ~4.5 days |

Time a short run before committing to a long one.

---

## 5. What you can change, and what it costs you

Everything lives in `Experimental_pipeline/config.py`. CLI flags override a few
of them per run.

### Change freely — these are the intended knobs

| Value | Default | What it does |
|---|---|---|
| `--repeats` | 30 / 5 | Statistics. Set it from the measured spread, not from precedent: `S_emp` forward has SD ≈ 0.002 (5 repeats is plenty), backward SD ≈ 0.25 (30 gives SEM ≈ 0.046). |
| `--seed` | 0 | Base seed; repeat *i* uses `seed + i`. |
| `--epochs` | 150 | Override `max_epochs` for a shakedown. |
| `--sweep-coin` / `--sweep-flower` | — | The grid values. Remember they cross. |
| `num_samples` | 500 | Sequences per repeat. The only pure data knob — cost is now proportional to it, since nothing is discarded. |
| `n_pts` | 1000 | Points clustered in F2. Cost is O(n²) in the distance matrix. |
| `trans_total_run` | 5000 | Generation steps for F3. ~5 s per arm; more only tightens the estimate. |
| `conv_tol` | 0.10 | What counts as converged. Widening it admits worse models to the figures. |
| `cluster_metric` | `euclidean` | `euclidean` / `tv` / `js`. JS has the better theory — bounded, symmetric, a true metric — but measured very slightly *worse* than plain L2 here. Thresholds are **not** transferable between metrics; their ranges differ. |

### Change with care — these are coupled

| Value | Default | The coupling |
|---|---|---|
| `seq_len` | 300 | **β tracks it.** `β = 1/(batch·seq_len)`, and β must stay under an optimisation cliff at ~6e-4 (lr=1e-3) above which the bottleneck collapses to one state. At 32×300, β = 1.04e-4, 5.8× under. Shortening `seq_len` raises β toward the cliff — at 32×49 the same rule gives 6.4e-4, already past it. |
| `batch` | 32 | Same coupling, same direction. |
| `lr` | 1e-3 | The cliff scales with **1/lr**, not with N. Raising `lr` lowers the cliff, so a β that was safe may not be. |
| `max_epochs` | 150 | A **ceiling, not a floor.** Cross-entropy on a deterministic transition has no finite optimum, so the backward arm converges and then *diverges past* H∞. Measured held-out CE above H∞ for the coin backward arm: 10 ep +0.289, 75 ep +0.022, **150 ep +0.016**, 200 ep +0.121, 300 ep +0.185. Longer is worse. |
| `weight_decay` | 0.01 | What restores a finite optimum on those deterministic transitions. At 0.0, AdamW is bit-identical to Adam and the divergence returns. Its total effect scales with `lr · weight_decay · steps`, so a value chosen for one step budget does not transfer to another. |
| `n_states_mult` | 5 (K = 5V) | A state **budget**, not an estimate. K at the exact theoretical count *fails* — flower(2,3) forward at its true K=3 recovers only 2 states. Slack is required. |
| `state_dim_mult` | 1 (S = V) | `state_matrix` then `emission` composes to a single (K,V) map, so S carries no expressive power. S=K would make `state_matrix` mostly reparameterisation freedom. |
| `accelerator` | `auto` | `auto` selects MPS, which is **not** bit-reproducible: trajectories diverge from the first backward pass. `cpu` is exactly repeatable at ~6× the wall clock. |

### The one genuinely unresolved knob

`state_tol` (default `0.10`) is the distance threshold for F2's clustering, and
**no single value is correct**. The smallest true separation between backward
states varies by an order of magnitude across these processes — 0.135 at
p=0.1,q=0.9 against 0.612 at p=q=0.5 — so a threshold that resolves one merges
another.

At this geometry 0.10 is measurably *wrong*: on coin p=0.3,q=0.4 it returns
`k_hat = 5` against a true 2 forward and 8 against a true 3 backward, biasing
`S_hat` by **+1.00** and **+1.11** bits. The pipeline therefore reports the
**plateau** — the k that survives the widest span of the tolerance grid — which
recovers 2 and 3 exactly and lands within 0.03 bits of the closed form. The
plateau is chosen without reference to the truth, so it is a mitigation, not
tuning on the answer. F4 plots the plateau; `S_hat` at `state_tol` is still in
the pickle.

State this wherever F2 or F4's clustered bar is shown. The comparison against
other extraction methods will turn on it.

---

## 6. Reading the outputs

`repeats.pkl` holds `spec`, `cfg`, `runs` (one record per repeat), `best`,
`best_fell_back`, and `transition`. Each run record carries, per architecture
and per arm: `final_ce`, `within_tol`, `diverged`, the loss curves, and the
extraction results — `S_emp` / `n_states_used` / `occupancy` / `emissions` /
`h_state_given_token` for discrete, `k_hat` / `S_hat` / `plateau` /
`S_hat_plateau` / `stability` for clustered.

Two diagnostics worth reading directly:

- **`h_state_given_token`** — H(state | current token), in bits. The true causal
  state is a deterministic function of the current token in both arms of both
  processes, so the truth scores **exactly 0**. Any positive value means the
  bottleneck is keying on context it does not need. It needs no ground truth,
  which makes it the one available unsupervised warning that states were merged.
- **`stability`** — `[(tol, k), ...]` across the tolerance grid. `k_hat = 2` is
  not defensible alone; "k_hat = 2, stable across tol ∈ [0.10, 0.60]" is.

`best_fell_back = True` means **no repeat converged** and F1–F3 were drawn from
the closest one anyway. Treat those figures as weaker evidence and say so.

## 7. If something looks wrong

| Symptom | Likely cause |
|---|---|
| `S_emp` ≈ 0 for a repeat | The bottleneck collapsed to one state. Check β against the cliff (§5) and whether that repeat diverged. |
| `S_emp` far above C | Over-splitting: the budget K is being used for distinctions the process does not have. `h_state_given_token > 0` confirms it. |
| Wide error bars on the backward bar of F4 | Expected. Backward `S_emp` has SD ≈ 0.25 across repeats on the coin — a property of the method, not a bug. It is the same in the archived results. |
| `k_hat` much larger than `plateau` | The `state_tol` problem in §5. Read the plateau. |
| Every repeat non-converged | `max_epochs` is a ceiling — check you have not raised it past 150. |
