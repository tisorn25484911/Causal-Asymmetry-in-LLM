# tau_experiment

Standalone studies of the straight-through temperature `tau` and of Adam's
momenta, for the discrete causal bottleneck.

**Nothing here modifies `Experimental_pipeline/`.** Every study imports it
read-only and injects what it varies from outside the model class:

- `tau` is a plain float attribute read fresh on every forward
  (`models.py:296`), so a Lightning callback can rewrite it per step.
- Adam's `betas` live in `optimizer.param_groups`, reached through an
  *instance-level* `configure_optimizers` override that shadows the class
  method without touching it.

The pipeline's own settings are the reference arm in every study:
`tau = const:1`, `betas = adam:0.9:0.999`.

---

## Layout

```
tau_experiment/
├── harness/                  shared, imported by every study
│   ├── schedules.py          tau and beta schedules, as spec strings
│   ├── runner.py             train one model under one arm, score it, write JSON
│   └── report.py             load JSON, aggregate, shared plot style
│
├── study0_pilot/             the original four-arm anneal test  [pre-existing]
│   ├── tau_anneal_test.py    runner            plot_tau_results.py, tau_report.py
│   └── results/ logs/ figures/
│
├── study1_tau_schedules/     how should tau vary?
│   ├── run.py  analyse.py    results/ figures/
│
├── study2_momentum/          does Adam's momentum help, tau held at the winner?
│   ├── run.py  analyse.py    results/ figures/
│
└── REPORT.md                 the written conclusion
```

Every run writes a JSON carrying its full spec, config, per-seed records and
the schedule trace, so every table and figure redraws without retraining.

---

## The arm vocabulary

An arm is fully described by the string that produced it — the JSON, the figure
legend and the command line all carry the same token.

| tau spec | meaning |
|---|---|
| `const:X` | fixed temperature |
| `geom:A:B` | geometric A → B, the exp(−rt) schedule of Jang et al. with the endpoint pinned |
| `lin:A:B`, `cos:A:B` | linear / cosine A → B |
| `hold:A:B:h` | hold at A for the first fraction `h`, then geometric to B |
| `ramp:A:B:w` | geometric A → B over the first fraction `w`, then hold B |

| optimiser spec | meaning |
|---|---|
| `adam:B1:B2` | constant betas; `adam:0.9:0.999` is the pipeline |
| `adam_b1:A:B[:B2]` | β₁ linear A → B |
| `adam_b2:A:B[:B1]` | β₂ linear A → B |

---

## What is scored, and why

The documented failure mode is a **state-count ceiling**
(`DISCRETE_STATE_EXPERIMENTS.md` 6.10): no configuration recovered more than
four distinct causal states, so cells whose true count is 5, 7 or 9 come back
merged.

The headline statistic is therefore **|S_emp − C|**, the gap between the
bottleneck's occupancy entropy and the closed-form statistical complexity, and
arms are compared by the **paired** change in it against `const:1` within each
(cell, seed) — same realisation, same initialisation, the arm the only
difference.

|S_emp − C| rather than `k_occ` because it is symmetric about the truth: states
merged (S below C) and one causal state split across several bottleneck states
(S above C, 6.9) both move it up, whereas `k_occ` scores a duplicated state as a
success. `H(state | token)` is reported beside it as the independent duplication
witness — it needs no ground truth, since the true causal state is a
deterministic function of the current token in both arms of both processes — and
is never folded into the score. CE − H∞ is the convergence gate, not a result.

---

## Reproducing

```bash
PY=/opt/anaconda3/envs/qdrug/bin/python      # the env with lightning

cd study1_tau_schedules
$PY run.py --phase 1B --list                 # 12 arms x 3 cells x 2 seeds
$PY run.py --phase 1B --cell 0               # one cell at a time; ~75 min each
$PY analyse.py                               # tables + F1-F5

cd ../study2_momentum
$PY run.py --tau geom:0.5:5 --list
$PY run.py --tau geom:0.5:5 --cell 0
$PY analyse.py
```

`accelerator=cpu` is the default and is exactly repeatable; MPS is not
(`training.py:284`). One run is ~180 s at 150 epochs, so cells are sized to be
launched in parallel — 2 threads each (`OMP_NUM_THREADS=2`) on an 11-core
machine fits about four at once.
