# Sweep Experiment — Implementation Plan

Extends `run_statistical_trj.py` with two parameter sweeps and two cross-process
figures. Date: 2026-08-10.

---

## 0. The question this answers

The seven-process harness answers *"what is ΔCE for this process?"* — precision at
few points. The sweep answers a different and stronger question:

> **Across many processes spanning a wide range of C⁻ − C⁺, does the measured
> asymmetry track the theoretical one?**

A single process cannot show a trend. 125 can. This is the strongest claim the
whole design is capable of supporting.

### The two response variables

**1. Final ΔCE** — already computed (`paired["mean"]`). Endpoint asymmetry.

**2. Area under the trajectory-difference curve** — new, and the more interesting
of the two.

For repeat *r* at gradient step *s*, let `D_r(s) = CE_BW,r(s) − CE_FW,r(s)`. The
area is its mean over the settled portion of training:

```
A_r = mean_{s >= burn} D_r(s)          units: bits
```

**Why this metric earns its place.** ΔCE at the endpoint is a difference of
*residuals*: once both arms converge to H∞ the residuals vanish and ΔCE → 0
whatever C⁻ − C⁺ is. That is exactly why the current result is a null and why the
null is ambiguous. But the *path* to convergence need not be symmetric — if the
backward direction is genuinely harder, it should sit above the forward arm
throughout optimisation even when both endpoints agree. `A` measures that. It is
a strictly richer statistic than the endpoint, computed from data already stored.

---

## 1. Measured budget

| quantity | measured |
|---|---|
| `one_repeat`, coin, 2 arms × 130 steps | 7.8 s |
| coin data generation | 0.3 s |
| flower data generation (n=m=10, 500 × 2000 cycles) | 7.1 s |
| `one_repeat`, flower (estimated) | ~15 s |

| sweep | processes | R=10 | R=20 | R=100 |
|---|---|---|---|---|
| coin 10×10 | 100 | 2.2 h | 4.3 h | 21.7 h |
| flower 5×5 | 25 | 1.0 h | 2.1 h | 10.4 h |
| **total** | **125** | **3.2 h** | **6.4 h** | **32 h** |

**Recommendation: R = 20 for sweeps**, baseline stays at 100. A sem over 20
repeats is ~2.2× wider than over 100 — irrelevant when the claim is a trend over
125 points rather than a value at one.

### Optional prerequisite (Step 0)

`flower_process_generation` is a pure-Python double loop calling `np.random.randint`
and `np.random.choice` once per cycle — 1.005M iterations for one process. It is
fully vectorisable: die selections are one `rng.integers(0, n, size=(N, T))`, and
outcomes can be drawn per-die with a vectorised gather. Expect 7.1 s → ~0.05 s,
which removes ~1 hour from the flower sweep at R=20.

This is a D2-class change. **Not required.** If done, it must be pinned by a test
asserting the vectorised generator reproduces the loop's distribution (the
existing `test_flower_generator_alternates_select_and_roll` is the starting point,
and a chi-square against `dice_probs` is the real check).

---

## 2. Coverage the grids actually give

Computed from the closed forms, not assumed.

**Coin, p,q ∈ {0.05, 0.15, …, 0.95}** — 100 processes.
C⁻ − C⁺ ∈ [0.0143, 0.6826], median 0.335, near-uniformly distributed across eight
bins (10–14 processes each). **Every point is positive** — the coin is causally
asymmetric for all (p,q), so this grid alone cannot test the sign, only the
magnitude trend.

**Flower, n,m ∈ {2, 4, 6, 8, 10}** — 25 processes.
C⁻ − C⁺ ∈ [−1.1643, +1.0688]: 15 negative, 1 near zero, 10 positive.
**This is the grid that tests the sign**, because `n > m` reverses the prediction.

Combined x-axis coverage: **−1.16 to +1.07 with points near zero.** That is a
well-designed sweep — do not change the grids without re-checking this.

Caveat to record with the flower results: each (n,m) uses ONE dice realisation
drawn from `cfg["flower_dice_seed"]`, so scatter in C⁻ − C⁺ at fixed (n,m) is
dice luck, not sampling. Deliberate — it keeps the sweep comparable to
`run_experiments.py`, which trains on the same dice.

---

## 3. Design decisions, fixed in advance

| decision | choice | why |
|---|---|---|
| Sweep is a longer spec list, not a new code path | reuse `main()` | `process_specs` is the only thing deciding what runs; everything downstream is spec-driven |
| Area computed at **analysis** time, not in `one_repeat` | from stored `traj` | the burn-in becomes tunable without re-running hours of training |
| Area **normalised** (mean, not sum) | `mean(D[burn:])` | units of bits, directly comparable to ΔCE, and robust if L ever differs between processes |
| Area burn-in shared with the plot | one module constant | the number and the picture must not drift apart |
| Per-repeat area, then aggregate | `_stats([A_r])` | preserves pairing; gives sd/sem across repeats |
| Sweeps write to their own `out_root` | `results_sweep` | tags are not config-scoped; sharing a root with the baseline silently overwrites |
| Tag prefix `sweep_` not `traj_` | `coin_spec(..., prefix)` | defence in depth against the same collision |
| Per-process figures **suppressed** in sweep mode | new flag | 125 × 3 = 375 near-useless PNGs, and `plot_grid_summary` would build a 550-inch-wide figure |

### The burn-in

`_draw_paired_diff` already scales its y-axis to `int(L / 4.3)` onward, because the
first ~30 steps swing by whole bits (30–90× the settled spread, up to 8.8 bits on
flower configs) and that swing is optimisation transient, not asymmetry.

**The area must use the same cutoff.** Integrating from step 0 would make `A`
mostly a measurement of the transient. Promote `1/4.3` to a named module constant
and have both the plot and the area function read it.

Report **both** the full-trajectory and post-burn-in area, so the choice is visible
rather than buried.

---

## 4. Implementation steps

### Step 1 — CLI surface and grid parsing
Add `--sweep-coin` and `--sweep-flower` with `nargs="*"`, plus two default-grid
constants and validation. **No wiring yet** — the flags are inert.
*Done when:* `--help` reads well, bad input is rejected, and existing behaviour is
byte-identical.

### Step 2 — make `process_specs` sweep-aware
Add a `prefix` parameter to `coin_spec`/`flower_spec`; give `process_specs` the two
optional grids. When a sweep is requested, return the swept specs *instead of* the
default seven.
*Done when:* a `--dry-run`-style print shows 100 / 25 / 125 specs with the right
tags and the C⁻ − C⁺ values from §2.

### Step 3 — sweep-mode output hygiene
Default `out_root` to `results_sweep` when sweeping; suppress per-process figures
and `plot_grid_summary`; keep the incremental pickle save (a 6-hour run must not
lose everything to one crash).
*Done when:* `--sweep-flower --repeats 1` completes and writes exactly one pickle
plus the summary figures.

### Step 4 — the area metric
A module constant for the burn-in fraction, a `trajectory_area(rec, burn_frac)`
returning per-repeat areas, and a refactor of `_draw_paired_diff` to read the same
constant.
*Done when:* the area on an existing `results_trajectories` pickle is finite,
sign-consistent with that process's ΔCE, and unchanged by re-running (it is a pure
function of stored data).

### Step 5 — the scatter figure (`sweep_scatter.png`)
Panels, all sharing x = C⁻ − C⁺:
1. final ΔCE, error bars ±1 sem over repeats
2. trajectory area, error bars ±1 sem
3. *(optional, requested earlier — confirm you still want it)* `S_emp − C_theory`
   per arm, i.e. how far the empirical complexity estimator sits from the closed
   form, as a function of the asymmetry

Colour by family (coin / flower) so the two grids are separable. Draw x=0 and y=0
lines; the hypothesis predicts points confined to the **lower-left and upper-right
quadrants**. Add Spearman ρ with its p-value per panel — that rank correlation,
not the eyeball, is the statistic that answers "does it track?".

### Step 6 — the trajectory-difference overlay (`sweep_trajectories.png`)
One mean `D(s)` curve per process, all overlaid, **coloured by C⁻ − C⁺ on a
diverging colormap centred at zero** (matching `plot_diff_heatmap`'s existing
`TwoSlopeNorm` convention), with a colorbar. If the hypothesis holds the curves
stratify by colour; if not they are a grey tangle. Clip y to the settled range as
`_draw_paired_diff` does.

### Step 7 — timing probe, then the real run
`--sweep-coin --repeats 2` first; multiply out; commit to the full run only then.

---

## 5. What will bite

1. **`plot_grid_summary` at 125 columns** — `figsize=(4.4*125, 12.8)`. Must be
   guarded, not merely "not called by accident".
2. **`combined` merges across runs.** The summary reads the whole pickle, so a
   coin sweep followed by a flower sweep into the same root produces one combined
   scatter — *desirable here*, but it means `--repeats` can differ between points.
   Record `n` per point and show it.
3. **Divergence.** At 130 steps the rate is low but nonzero, and one diverged
   repeat can dominate an area computed over 20. Report the converged-only
   statistic alongside, as `paired_delta_ce` already does.
4. **`build_dataset` is defined twice in the repo** (here and in
   `plot_state_clusters.py`) with different seeding semantics behind the same name.
   Do not add a third.
5. **Memory.** 125 processes × 20 repeats × 2 arms × 130 floats of trajectory is
   only ~1.3 MB, but the pickle also holds `val_traj`. Fine — just do not add
   latents.

---

## 6. Verification checklist

- [ ] `pytest tests/ -q` still passes (61 tests) after every step
- [ ] no-flag invocation still runs the original seven processes
- [ ] swept C⁻ − C⁺ values match §2 exactly
- [ ] area is a pure function of the stored pickle (recompute = identical)
- [ ] the scatter's Spearman ρ is computed over processes, not repeats
- [ ] sweep output is in its own `out_root`, and the baseline pickle is untouched
