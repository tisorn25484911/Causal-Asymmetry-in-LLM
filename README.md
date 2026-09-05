# Transformer-based causal-state extraction

A pipeline that takes a stochastic process, trains transformers on it, and
returns **the causal structure the models extracted** — against the closed form
for the same process. One config, two commands, four figures.

This exists to be *compared against*. The next stage of the work puts other
causal-state extraction methods beside this one, and that comparison needs a
method that runs at fixed, stated settings and reports a number — not a
parameter search. The parameter search already happened; it lives in
`model tuning experiment/` and is not part of this pipeline.

---

## What it measures

For every process it is run on, in **both** the forward and backward direction:

| | Quantity | How |
|---|---|---|
| **F1** | Causal states, extracted by a **discrete bottleneck** | Prediction is forced through a K-way state; a position's state is `argmax(state_logits)`. Exact, no clustering, no free parameter. |
| **F2** | Causal states, extracted from a **standard decoder by clustering** | Agglomerative clustering of the model's *predictive distributions*. Has one genuinely free parameter, the distance threshold. |
| **F3** | The learned **state-to-state transition matrix** | Read off the model's own free-running generation. Forward arms are drawn beside the closed form. |
| **F4** | **Theoretical vs empirical statistical complexity** | C± against both estimators, over converged repeats. The headline, and the figure the method comparison extends. |

Both arms are trained because the backward direction is where the state count
differs — the coin needs 3 backward states against 2 forward, and the flower's
backward count depends on how its dice merge — so it is where an extraction
method is actually stressed.

Cross-entropy is recorded **only as a convergence check**: is this model trained
enough for its extracted states to mean anything. A repeat outside `conv_tol` is
excluded from the figures and counted, never silently averaged in.

## The two processes

**Coin** — a 2-state hidden chain (`0→1` w.p. `p`, `1→0` w.p. `q`) emitting 3
tokens. Forward 2 causal states, backward 3.

**Flower** — alternates a uniform die selection over `n` dice with a roll of
that die over `m` faces; `V = n+m`. Forward `n+1` states, backward
`1 + #distinguishable outcomes` — *not* `m+1` in general, because dice whose
posterior columns are proportional merge into one backward state.

Both samplers and both sets of closed forms live in one file, `processes.py`, on
purpose: "how to draw it" and "what its ε-machine provably is" are the same
subject, and separating them is how a generator change silently invalidates a
formula. `checks.py` holds the samplers against the formulas.

## The flow

```
run_process.py / run_sweep.py          the terminal commands
        │   pick specs from config.CONFIG
        ▼
pipeline.run_process(spec, cfg, repeats)
        │
        ├── for each repeat i, seed = cfg["seed"] + i:
        │     pipeline.one_repeat
        │       processes.generate ──► SequenceDataset ──► ONE seeded split
        │       │                        (all four models see the same data)
        │       └── for arch in (discrete, onehot):
        │             for arm in (forward, backward):
        │                 training.train_model          ← models.build_model
        │                 training.eval_ce              ← convergence check
        │                 discrete → extraction.causal_state_report    (S_emp)
        │                 onehot   → extraction.recover_causal_states  (S_hat)
        │
        ├── pipeline.best_repeat_index    closest to H∞ among converged
        ├── extraction.transition_matrix_extraction   on that repeat's models
        ├── training.save_pkl  ──►  repeats.pkl
        └── figures.draw_all   ──►  F1 F2 F3 F4
```

`run_sweep.py` is the same loop over a grid of specs, plus one aggregate figure.
Every figure is drawn **from the saved pickle**, so `--plots-only` redraws
without retraining.

## Layout

```
Experimental_pipeline/
├── run_process.py   ENTRY  one process (or all seven) x N repeats
├── run_sweep.py     ENTRY  (p,q) and (n,m) grids
├── pipeline.py      the experiment: one_repeat, run_process, run_sweep
├── config.py        the master config + per-process specs
├── processes.py     generation, Datasets, and every closed form
├── models.py        OneHotDecoder + DiscreteCausalDecoder, one shared stack
├── training.py      train loop, split, eval, seeding, divergence, io
├── extraction.py    the three extractors
├── figures.py       the four figures + the two aggregates
└── checks.py        self-checks that can fail loudly

main_results/
├── run_config.json
├── trainings/<tag>/           repeats.pkl, F1–F4
├── sweep_params/sweep_{coin,flower}/   grid.pkl, F4_complexity_grid.png, <cell>/F1–F4
└── arc_comparision/           F4_all_processes.png

model tuning experiment/       the superseded exploratory tree, frozen
```

`Experimental_pipeline/` imports nothing from `model tuning experiment/`. It
depends only on numpy, torch, lightning, matplotlib and scikit-learn.

## Quick start

```bash
conda activate asym
python Experimental_pipeline/checks.py                                  # ~20 s
python Experimental_pipeline/run_process.py --process coin --p 0.3 --q 0.4 --repeats 5
```

See `HOW_TO_RUN.md` for every command, every tunable value, and what each one
costs.
