# Discrete Decoder — Restructured Experiment Plan (`All_Results/discrete_v2`)

Plan only. Nothing applied. Supersedes the folder layout in
`DISCRETE_RUNTHROUGH_PLAN.md`; that document's §0a (the regulariser change) and
§4 (code sanity check) still stand and are referenced rather than repeated.

Every number below was measured, not assumed. Provenance is marked:
**[cfg]** read from `configs.py`, **[calc]** computed from the closed forms,
**[meas]** measured in a training run this session.

---

## 1. Verified constants

**[cfg]** DISCRETE geometry — `train_chunk_len=256`, `coin_batch=flower_batch=32`,
`d_model=32`, `lr=1e-3`, `weight_decay=0.01`, `n_layers=2`, `tau=1.0`,
`n_folds=5`, `coin/flower_max_epochs=150`, `flower_dice_seed=42`.

**[meas]** `ChunckDataset` slices the *already-shifted* (input,target) pair, so
**T = chunk_len = 256**, not 255. Reading the live loader:

```
  batches/epoch = 16
  batch 0..14 : B=32  T=256   N = B*T = 8192   1/N = 1.2207e-4
  batch 15    : B=20  T=256   N = B*T = 5120   1/N = 1.9531e-4
```

**β = 1/N = 1.2207e-4 at this geometry — computed from `batch × chunk_len`, never
written as a literal.** Any change to batch or chunk moves it (§6.6).

**[calc]** The seven baseline processes:

```
             process   V  5V=K  S=V ||  Kfw  Kbw      C+      C-      gap   H_inf ||  K/Kfw  K/Kbw
      coin_p010_q090   3    15    3 ||    2    3  0.4690  0.8911  +0.4221  0.4690 ||    7.5    5.0
      coin_p030_q040   3    15    3 ||    2    3  0.9852  1.4888  +0.5036  0.9197 ||    7.5    5.0
      coin_p040_q080   3    15    3 ||    2    3  0.9183  1.5656  +0.6473  0.8879 ||    7.5    5.0
        flower_n2_m6   8    40    8 ||    3    7  1.5000  2.1137  +0.6137  1.5343 ||   13.3    5.7
        flower_n2_m8  10    50   10 ||    3    9  1.5000  2.4765  +0.9765  1.7800 ||   16.7    5.6
        flower_n4_m2   6    30    6 ||    5    3  2.0000  1.4952  -0.5048  1.3364 ||    6.0   10.0
        flower_n6_m4  10    50   10 ||    7    5  2.2925  1.9899  -0.3026  2.0985 ||    7.1   10.0
```

`K = 5V` gives 5.0x–16.7x slack over the true state counts. Note `n2_m8`
backward needs 9 states — the largest true count in the set, and the arm most
likely to fail (§2.3).

---

## 2. Why these base values

### 2.1 `n_states = 5V` — directly validated

**[meas]** The two best state recoveries in the entire session both landed
exactly on 5V, without that rule being in mind at the time:

| process | V | 5V | measured at K=5V, S=V | result |
|---|---|---|---|---|
| flower(2,3) | 5 | **25** | ARI **1.000**, k=3/3 | S_emp − C⁺ = **+0.0050** |
| flower(3,5) | 8 | **40** | ARI **0.982**, k=4/4 | S_emp − C⁺ = **+0.0064** |

**[meas]** K at or near the *true count* is what fails: flower(2,3) forward at
K=3 (the exact count) gives ARI 0.745 with k=2; K=4 also 0.745. Slack is
required, and 5V supplies 5.0x–16.7x of it across the baseline set (§1).

**Provenance caveat:** both validating processes — flower(2,3) and flower(3,5) —
are *pilot* processes, not members of the baseline seven, and each result is a
single seed. 5V is the best-supported starting value, not an established optimum;
`01_ksweep` is what tests it on the processes that matter.

### 2.2 `state_dim = V` — better than the alternatives tested

**[meas]** Three settings compared on all 28 flower cells:

| `state_dim` | cells with `k_found == trueK` | best (3,5) fw |
|---|---|---|
| `None` (= K) | 3 | 0.877, k=3 |
| **V** | **6** | **0.982, k=4** |
| round(1.3V) | 5 | 0.849, k=3 |

1.3V is not better than V, and it loses the K=40 cell that matters. `state_dim=K`
at K=40 is a 40×40 matrix that is almost entirely reparameterisation freedom —
`state_matrix @ Q` with `inv(Q) @ emission` is the same model for any invertible
Q, so only the K×V emission table is identified.

### 2.3 What is *not* solved by these defaults

**[meas]** No backward arm has ever recovered its full state set — at any K, β,
`state_dim`, budget, sequence length or seed tried. Best cases:

```
  flower(2,3) backward  trueK=4  best k=3, ARI 0.949, S_emp - C- = -0.1675
  flower(3,5) backward  trueK=6  best k=4, ARI 0.907 (at 2x seq len)
  coin        backward  trueK=3  best k=3, ARI 1.000 (K=16) but 0.157 by 12k steps
```

`flower_n2_m8` backward needs **9** states — more than any arm yet recovered.
Expect it to fail; the plan must record that as a result rather than retry until
it passes.

---

## 3. Folder structure

```
All_Results/discrete_v2/
  00_base/            7 processes, base config, repeats=30      <- the reference run
  01_ksweep/          K vs k_discovered, the headline figure
  02_statedim/        state_dim in {V/2, V, 2V, K}
  03_seqlen/          chunk_len in {128, 256, 512}   (T varies, B fixed)
  03b_batch/          batch in {2,4,8,16,32,64}      (B varies, T fixed)
  04_capacity/        d_model in {16, 32, 64}
  05_budget/          epochs in {50, 100, 150, 300}, checkpointed
  06_beta_verify/     beta in {0.5, 1, 2, 4} x 1/N  -- VERIFICATION, not tuning
  07_seeds/           base config, 5 seeds, the noise floor
  ── the science, not the tuning ──────────────────────────────────────
  08_pqsweep/         coin (p,q) grid   -> scatter vs C- - C+
  09_nmsweep/         flower (n,m) grid -> scatter vs C- - C+
  10_trajectories/    7 baseline processes, repeats=30, loss curves + delta_CE
```

**Folders 00–07 exist to make 08–10 interpretable.** 08–10 are the experiments
that answer the study's actual question; everything before them establishes the
settings and the error bars they need.

`All_Results/discrete/` is kept read-only as the historical record. Nothing is
re-run into it.

**Budget: the per-folder hours below sum to ~61 h and are ESTIMATES, not
measurements.** They are extrapolated from a d_model=20, single-split, T=99
scratchpad harness; the real configuration is d_model=32, 5-fold, T=256. Before
committing, time one repeat of `00_base` and rescale everything by the ratio.
That timing run is the first task, not an afterthought.

**Base config for every folder except where it is the swept axis:**

```
embed_type   discrete        state_dim    V  (per process)
lr           1e-3            n_states     5V (per process)
weight_decay 0.01            usage_beta   1/(batch * chunk_len)   COMPUTED
tau          1.0             epochs       150
d_model      32              n_folds      5
```

---

## 4. The experiments

### 00_base — the reference run (~4 h)

Seven processes, both arms, `repeats=30`. Everything else compares to this.

**Report per arm:** CE−H∞, `k_found`, `S_emp`, `S_emp − C`, `H(state|token)`,
occupancy vs the closed-form stationary distribution, emission table vs the
closed-form conditionals.

**Done when:** every arm inside `conv_tol=0.1`, and the `k_found` column is
recorded for all 14 arms whether or not it equals the true count.

### 01_ksweep — K vs k_discovered (the headline figure) (~6 h)

Per process, per arm. **The grid must be V-relative, not absolute** — an
absolute K=60 is 20V for the coin (V=3) but only 6V for `flower_n2_m8` (V=10),
so a fixed grid would compare different amounts of slack across processes:

```
  below the true count :  K in {1, 2, 3, 4, 6}          absolute, to see the y=x regime
  above it             :  K in {V, 2V, 3V, 5V, 8V, 12V}  relative, to see the plateau
```

For the coin (V=3) that is {1,2,3,4,6,9,15,24,36}; for `flower_n2_m8` (V=10),
{1,2,3,4,6,10,20,30,50,80,120}. `state_dim=V`, β=1/N, repeats=5.

**Figure F1 — K vs k_discovered.** x = K, y = `k_found`, one line per
(process, arm). Reference lines: `y = x` (the budget ceiling) and `y = trueK`
(the target). Point colour = ARI, so a saturated line at the right height reads
as success and a saturated line at the *wrong* height reads as confident error.

**[meas] What it should show, from the pilot data:** the curve rises from
`y = x` while K < trueK, then flattens. Where it flattens is the question. On
flower(3,5) forward the plateau sat at k=3 for K ∈ {4…30} and jumped to k=4 only
at K=40. If that shape is general, **the plateau height is the estimator and the
onset of the plateau is the confidence measure** — which would be a genuinely
new read-out, since it needs no ground truth.

**F1b:** K vs `S_emp`, same layout, with `y = C` as the reference. This
is the calibration view of the same sweep.

**Repeats matter here more than anywhere.** **[meas]** `k_found` is wildly
non-monotonic in K at one seed — flower(3,5) forward fails at K=8 (ARI 0.568)
while succeeding at K=6 and K=12. Five seeds minimum, and the figure must show
the spread, not the mean alone.

### 02_statedim — S in {V/2, V, 2V, K} (~3 h)

**[meas]** V beat both K and 1.3V (§2.2), but V/2 and 2V are untested and the
comparison so far is single-seed. Sweep at K=5V fixed, repeats=5.

**Done when:** we can say whether S=V is a real optimum or was a lucky basin.

### 03_seqlen — chunk_len in {128, 256, 512} (~5 h)

**[meas]** Sequence length matters, and it moves β because β = 1/(B·T).

**Scale caveat:** the pilot below ran at T ∈ {49,99,199,299} on a *scratchpad*
harness (single split, no folds, `flower(3,5)`), not at the repo's T=256. It
establishes the mechanism and the cliff location, not the values to expect at the
planned `chunk_len ∈ {128, 256, 512}` — where β is 2.4e-4 / 1.2e-4 / 6.1e-5, all
safely below the cliff. Pilot, 9 K values per arm, B=32:

```
  chunk   T    beta=1/(B*T)    collapsed cells (peak k=1)   best bw ARI
     50   49      6.38e-4              1                       0.857
    100   99      3.16e-4              0                       0.861
    200  199      1.57e-4              0                       0.907
    300  299      1.05e-4              0                       0.953
```

**The cliff prediction is confirmed across all four lengths.** Collapses occur
only at T=49, where β = 6.38e-4 is past the ~6e-4 cliff; the other three, all
below it, give zero. β and collapse move together while context moves the other
way, so the collapse is a β effect. §03b is still the decisive control because it
holds context fixed.

**[meas] The backward arm improves monotonically with context** — best ARI
0.857 → 0.861 → 0.907 → **0.953** across the four lengths. At 3×, `K=25` backward
reaches **ARI 0.953 with k=5 of 6**, the closest any backward arm has come to its
full state set. Longer context is the only lever that has consistently helped it.

Forward does *not* share that trend (best 1.000 / 0.982 / 0.956 / 0.880), and the
K value that succeeds moves with length — K=40 at 0.5× and 1×, K=20 at 2×, K=25
at 3×. That scatter is more consistent with seed noise relocating the basin than
with a length effect at fixed K, which is why `07_seeds` gates everything.

**This experiment is confounded by construction** — changing `chunk_len` changes
β. Run it *both* ways: with β tracking 1/N (the honest default) and with β pinned
to the 256-value, so the length effect and the β effect are separable. That is
one of only two places a β other than 1/N is justified, and it is a control, not
a tuning knob. The other is §03b, which is the cleaner separation.

### 03b_batch — B in {2, 4, 8, 16, 32, 64} at FIXED T = 256 (~4 h)

**The control that §03_seqlen cannot be.** Holding `chunk_len` fixed and varying
the batch size sweeps β across the *same* cliff, with the context length
identical in every cell — so any effect is attributable to β alone.

**[calc]** At the base geometry, `chunk_len = T = 256`:

```
    B    T   N=B*T    beta=1/N   vs cliff 6e-4
    2  256     512   1.953e-03   PAST CLIFF
    4  256    1024   9.766e-04   PAST CLIFF
    8  256    2048   4.883e-04   near cliff
   16  256    4096   2.441e-04         safe
   32  256    8192   1.221e-04         safe   <- base
   64  256   16384   6.104e-05         safe
```

Two cells past the cliff, one on it, three safely below — the same β span the
length sweep covers, with context held constant at 256 tokens.

**Note the grid is {2…64}, not {4…128}.** At T=256 a batch of 128 gives
β = 3.05e-5, twenty times below the cliff and uninformative, while B=2 is needed
to reach past it at all. The earlier draft's {4…128} was computed at T=99 and
does not span the cliff at the real geometry.

**The prediction this tests.** If the single short-sequence collapse was a β
effect, then B ∈ {2, 4} should collapse at T=256 *despite* having full context.
If they train fine, that collapse was about short context, and β = 1/N is safer
than the cliff estimate implies.

**Two confounds to control, both mechanical:**

* **Steps, not epochs.** At `n_seq=500`, `batches/epoch` ranges 250 (B=2) to 8
  (B=64) — a 31× difference in gradient steps for a fixed epoch count. Fix the
  *step* budget (2400, matching the base config) and let epochs vary.
* **Gradient noise.** Small B is noisier per step independently of β. So run the
  same B grid a second time with **β pinned at 1/(32·99)** — then B varies while
  β does not, and subtracting the two panels isolates the β effect from the
  batch-noise effect.

**Done when:** the two panels can be compared cell by cell. If they agree, batch
size acts only through β and the rule "β = 1/N, with N ≳ 2000" is sufficient. If
they differ, batch size has an independent effect and the base config needs a
minimum B as well as a minimum N.

**Figure F2 — β vs collapse rate.** x = β on a log axis, y = fraction of cells
with `k_found = 1`, one series for the length sweep and one for the batch sweep.
If the two series lie on top of each other, β is the whole story and the cliff
has a single location; if they separate, T and B matter independently. This is
the figure that justifies the base β for every future run.

### 04_capacity — d_model in {16, 32, 64} (~8 h)

**Completely untested for this architecture.** Every result in the session is at
d_model ∈ {20, 32}. Guard: the runners already refuse unless `out_root` names the
capacity.

**[meas]** Motivation: no arm needing ≥5 states has been recovered. Whether that
is capacity or optimisation is unknown, and this is the experiment that decides.

### 05_budget — epochs in {50, 100, 150, 300}, checkpointed (~4 h)

**[meas]** More training is *harmful* past a point, and CE gives no warning:

```
flower(2,3) forward, 20k steps
  step    100    250    500   1000   2000   4000   6000  10000  15000  20000
  ARI   0.525  0.672  0.748  1.000  1.000  0.986  0.952  0.910  0.906  0.857
  CE-H +1.089 +0.657 +0.363 +0.082 +0.006 +0.011 +0.016 -0.004 -0.015 -0.040
```

ARI peaks at 1–2k steps and decays; CE falls monotonically and ends *below* H∞.
Five of ten long runs ended sub-H∞, every one with degraded ARI.

**[meas] The stopping rule to validate here:** stop at the **first** checkpoint
with `|CE − H∞| < conv_tol`. Scored over 10 runs it lost a mean of **0.046** ARI
against the true peak, versus 0.071 (worst 0.545) for a `CE ≈ H∞+0.10` rule and
0.217 for an `H(state|token)` rule. It needs **no new hyperparameter** —
`conv_tol=0.1` already exists; it is applied as an early stop rather than a
post-hoc filter.

**150 epochs × 16 batches = 2400 steps** [calc] sits inside the measured good
window, so the current budget is close to right and this experiment is a check,
not a search.

### 06_beta_verify — β ∈ {0.5, 1, 2, 4} × 1/N (~3 h)

**Verification that 1/N is in the safe band. Not a tuning sweep.**

**[meas]** Why the band is narrow, and why it must be re-checked at any geometry
change:

```
  collapse cliff at lr=1e-3   ~6e-4          beta=1/N at chunk 256   1.22e-4  (5x below)
  beta_cliff * lr ~ 6e-7      -> the cliff scales with 1/lr, NOT with N
  beta = 0.01 (retired rule)  82x the safe value  -> collapses k to 1 in 3 of 4 pilot cells
  beta = 0.2  (retired rule)  1638x               -> collapses everywhere
```

**Done when:** 1× and 2× behave alike and 4× shows the first degradation, which
would confirm 1/N sits comfortably inside the band. If 2× already degrades, 1/N
is closer to the edge than the cliff estimate suggests and the base value should
drop to 0.5/N.

### 07_seeds — 5 seeds at base config (~4 h)

**[meas]** The noise floor is large and currently unquantified. Same seed, same
config, differing only in checkpoint schedule gave ARI 0.730 vs 0.943 in my
scratchpad harness. `k_found` flips between 2 and 4 across neighbouring K.

**Nothing in 01–06 is interpretable without this.** Run it early, not last.

### 08_pqsweep — coin (p,q) grid (~7 h)

**[cfg]** `COIN_GRID_DEFAULT = [0.05, 0.15, … 0.95]`; the existing discrete run
used a **5x5 = 25-process** subset at `repeats=5`, which is the scale to keep.

`n_states = 5V = 15` for every cell (V=3 is constant on the coin), `state_dim=V=3`,
β = 1/N, both arms.

**[calc]** Coverage: C⁻−C⁺ ∈ [0.014, 0.683] and **every point is positive** — the
coin is causally asymmetric for all (p,q), so this grid tests magnitude, not sign.
**[meas, prior]** It is also confounded: corr(gap, H∞) = +0.590, and the raw
Spearman ρ has the *wrong sign* (−0.415), going to +0.068 (p=0.50) once H∞ is
partialled out. **This grid is a confound control, not evidence** — carry that
sentence into the results.

### 09_nmsweep — flower (n,m) grid (~7 h)

**[cfg]** `FLOWER_GRID_DEFAULT = [2, 4, 6, 8, 10]` → 25 processes.
`n_states = 5V` per cell, so K ranges **20 → 100** as V = n+m ranges 4 → 20;
`state_dim = V` likewise. β = 1/N.

**[calc]** This is the grid that tests the **sign**: C⁻−C⁺ ∈ [−1.164, +1.069],
15 negative, 10 positive. **[meas, prior]** Its confound is different and worse:
corr(C⁻−C⁺, m−n) = +0.977, so a trend here is equally consistent with tracking
m−n. `run_dice_experiment` exists to break that, and the discrete model should be
run on it too once 09 is in hand.

**Warning specific to this grid** **[calc]**: backward true state counts run to
1+m = 11, and **[meas]** no arm needing ≥5 states has ever been recovered. Expect
the high-m cells to under-count. That is a result about the instrument, and the
recovery rate must be reported per cell, not averaged away.

### 10_trajectories — statistical loss curves (~6 h)

The seven baseline processes, `repeats=30`, base config. Reproduces the existing
`All_Results/discrete/trajectories` at the corrected β and the new K/state_dim.

Per process it produces the full per-step loss trajectory for both arms, from
which come the paired-difference curve D(s) = CE_bw(s) − CE_fw(s), the settled
ΔCE, the trajectory area, and d_step.

**[meas] Read ΔCE from this model with heavy caution.** Measured on the pilot,
ΔCE spanned **−0.13 to +0.26 bits purely as a function of (K, β)** against a real
effect of 0.001–0.003 — a setting-dependent bias 20x–250x the signal, with a
consistent sign. The continuous model gives ΔCE ∈ [−0.0011, +0.0024] on the same
processes. **The discrete model is not currently a ΔCE instrument**; it is run
here for the trajectory shape and for S_emp, not to beat the baseline on ΔCE.

---

## 4b. What the discrete architecture actually buys

This is the reason 08–10 are worth the compute, and it is not ΔCE.

ΔCE is a **difference of residuals**: once both arms converge to H∞ it vanishes
regardless of C⁻−C⁺, which is exactly why the existing null is ambiguous.

`S_emp` is **not** a residual. S_emp_fw estimates C⁺ and S_emp_bw estimates C⁻
directly, so

```
        Delta_S = S_emp_bw - S_emp_fw     estimates    C- - C+     DIRECTLY
```

**[meas]** When the state set is recovered, S_emp is calibrated to **±0.006 bits**
(flower(2,3) fw: 1.5050 vs C⁺ 1.5000; flower(3,5) fw: 1.7989 vs C⁺ 1.7925). That
is a direct measurement of the asymmetry at a precision the residual route cannot
reach, and it sidesteps the ambiguity in the headline null entirely.

**The catch, stated plainly** **[meas]**: S_emp is calibrated *only* when
`k_found == trueK`, which held in 6 of 28 pilot cells. When the state set merges,
S_emp − C runs −0.17 to −1.04 bits. There is no partial credit. So **Δ_S is only
usable on cells where both arms recovered their counts**, and the fraction of such
cells is itself a headline number.

---

## 5. Figures

**House style, applied to every figure below.** These are the rules the pilot
figures broke, each costing legibility rather than buying it:

| rule | why |
|---|---|
| Encode each variable **once** | the current `plot_causal_states` shows occupancy as bar height, as a printed number, *and* as a colorbar — three encodings of one number |
| No colorbar when values are printed or the scale is fixed 0–1 | panel 3's colorbar is pure decoration |
| **Annotate reference lines in place**, never in the legend | `y = trueK` labelled at the line's right edge costs one word; in a legend it costs a legend |
| Legend only at **≥5 series**; below that, label the last point of each line | 14 arms need a legend, 4 do not |
| Axis labels are **units, not sentences** | `S_emp (bits)` not `Empirical statistical complexity in bits` |
| Title is the **identity**, not a summary | `flower_n2_m6 · backward` |
| Numbers to **3 dp**, aligned, no thousands separators inside plots | `1.799` not `1,799.0` |

Colour: one hue for forward, one for backward, grey for unused/reference. Marker
shape carries process family (circle = coin, square = flower). Never both colour
and shape for the same variable.

### F0 — 00_base: calibration (one figure, the reference)

```
  x : C (bits)                      closed form
  y : S_emp (bits)                  measured
  line: y = x, grey, annotated "y = x" at its top-right end
  point: colour = arm, shape = family, no fill if k_found != trueK
  label: last point of each arm only, e.g. "n2_m8 bw"
```

Title `S_emp vs closed form · 14 arms`. **No legend** — two colours and two
shapes are self-evident from the four labelled points. Hollow markers do the
work that a fourth encoding would otherwise need.

### F1 — 01_ksweep: **K vs k_discovered** (the headline)

```
  x : K  (log2)                     state budget
  y : k_discovered                  states actually used
  lines: y = x       grey dashed, annotated "budget" at upper-left
         y = trueK   grey dotted per panel, annotated "true"
  panels: one per process, 2 columns (fw | bw), shared axes
  series: median over 5 seeds, band = min–max
```

Title per panel `flower_n2_m6 · fw`. Axis labels `K` and `k`. **No legend at
all** — the two reference lines are annotated in place and there is one series
per panel.

**What the figure is for.** The curve should track `y = x` while K < trueK, then
plateau. **[meas]** On flower(3,5) forward the plateau sat at k=3 for
K ∈ {4…30} and stepped to 4 only at K=40. If that shape is general, the
**plateau height estimates the state count and the plateau onset measures
confidence**, neither needing ground truth. The min–max band is essential, not
decorative: `k_found` flipped between 2 and 4 across neighbouring K at one seed.

### F1b — 01_ksweep: calibration view

Same layout, `y = S_emp (bits)`, reference `y = C` annotated `C`. Read together
with F1: F1 says whether the count is right, F1b whether the entropy is.

### F2 — 02_statedim

```
  x : state_dim / V   at {0.5, 1, 2, K/V}
  y : ARI
  panels: fw | bw ; series: one per process, direct-labelled at the right edge
```

Seven series is above the legend threshold, but direct labels at the right edge
still beat a legend here because the lines separate. Second row of the same
figure: `y = k_discovered − trueK`, reference `y = 0` annotated `exact`.

### F3 — 03_seqlen

```
  x : chunk_len  {50, 100, 200, 300}   log
  y : ARI  (top)  /  collapse rate     (bottom, shared x)
  twin annotation: beta printed under each x tick, e.g. "6.4e-4"
```

Printing β under the tick is what makes this figure honest — it shows that x is
confounded with β by construction. Two panels, no legend, series direct-labelled.

### F4 — 03b_batch: **β vs collapse rate** (the control)

```
  x : beta  (log)
  y : fraction of cells with k_found = 1
  series: "vary T" and "vary B"      <- exactly 2, direct-labelled, no legend
  line: measured cliff ~6e-4, vertical grey, annotated "cliff"
  point: annotated with its (B,T) e.g. "8x99"
```

**The whole plan's β justification rests on this figure.** If the two series
superimpose, β is the entire story and one cliff location covers both axes. If
they separate, T and B act independently and the base config needs a minimum B
as well as a minimum N.

### F5 — 04_capacity

```
  x : d_model  {16, 32, 64}   log2
  y : k_discovered − trueK    reference y = 0 annotated "exact"
  panels: fw | bw ; one line per process, direct-labelled
```

Deliberately **not** ARI: the question is whether capacity lifts the state count
on the arms needing ≥5 states, and the signed deficit answers it directly.

### F6 — 05_budget: the early-stopping figure

```
  x : gradient steps  (log)
  y (left)  : ARI               solid
  y (right) : CE - H_inf (bits) dashed, zero line annotated "H_inf"
  band: |CE - H_inf| < conv_tol, shaded, annotated "conv_tol"
  marker: first entry into the band -> the proposed stopping point
```

One panel per representative arm (4 panels, not 14). **The two-axis pairing is
the point** — ARI peaks and falls while CE keeps dropping through H∞, so the
figure has to show both to make the case that CE is the wrong stopping signal.
**[meas]** ARI peaked at 1–2k steps and decayed to 0.857 by 20k while CE fell
monotonically to −0.040 below H∞.

### F7 — 06_beta_verify

```
  x : beta / (1/N)   at {0.5, 1, 2, 4}   log2, tick labels "0.5x 1x 2x 4x"
  y : ARI ; second row y = k_discovered
  line: x = 1 vertical, annotated "1/N"
```

Expresses β as a **ratio**, so the figure reads the same at any geometry. Absolute
β belongs in the caption, not on the axis.

### F8 — 07_seeds: the noise floor

```
  strip plot: x = arm (14), y = ARI, one dot per seed, median bar
  annotation: the max spread, e.g. "widest 0.21", placed once
```

This figure exists to be **quoted in every other caption**. Without it none of
F1–F7 has an error bar, and **[meas]** the observed spread already exceeds most
differences reported so far.

### F9 — 08/09: scatter vs C⁻ − C⁺ (four panels, the repo's existing layout)

```
  x : C- - C+ (bits, closed form)    shared by all four
  p1: delta_CE          y-err = sem      <- the only panel the theory speaks to
  p2: trajectory area   EXPLORATORY, label it so
  p3: d_step
  p4: Delta_S = S_emp_bw - S_emp_fw      <- NEW headline for the discrete model
  lines: x=0 and y=0, grey, unlabelled (quadrants are the point)
  colour: family (coin / flower); marker hollow if k_found != trueK on EITHER arm
  per panel: Spearman rho + n, printed top-left, 3 dp
```

**Panel 4 replaces the repo's current `S_emp − closed form` panel.** **[meas]**
That panel reads `r["fw"]["S_emp"]` (`run_sweep_experiment.py:480`) — the k-means
estimate — while the discrete value is stored as `S_emp_states`. Its axis note
still says *"S_emp is k-means at an assumed k"*, false for a discrete run. Panel 4
must read `S_emp_states`, and add `y = x` annotated *exact* since Δ_S is being
compared to the x-axis itself.

Hollow-when-unrecovered is doing real work here: it shows at a glance what
fraction of the grid the S_emp panel is even entitled to speak about.

### F10 — 08/09: trajectory overlay

```
  x : gradient step        y : D(s) = CE_bw - CE_fw (bits)
  one mean curve per process, colour = C- - C+ on a diverging map centred at 0
  colorbar: yes -- it is the only encoding of C- - C+ here, so it earns its place
  y-limit: settled range only (burn-in already excluded)
```

Title `D(s) by C− − C+`. **No legend** — 25 processes; the colorbar *is* the
legend. This is the one figure in the plan where a colorbar is not redundant.

### F11 — 10_trajectories: per-process loss curves

Keep the repo's existing three, at the new settings:

```
  <tag>_trajectories.png   all runs at alpha 0.3, two mean curves bold
  summary_delta_ce.png     delta_CE with sem, per process
  summary_grid.png         all three quantities, all processes
```

Minimal-label changes only: axis labels to `step` / `CE (bits)`, titles to the
bare tag, and **H∞ drawn as an annotated horizontal line rather than a legend
entry**. Add one series the continuous version cannot have: `k_found` per arm as
a thin step-line on a right-hand axis, which shows *when* the state set was
discovered relative to the loss settling.

### F12 — 08/09: recovery rate (small, and it gates F9 panel 4)

```
  x : C- - C+        y : fraction of repeats with k_found == trueK
  two series: fw, bw, direct-labelled
  reference: y = 1 annotated "recovered"
```

Small panel, but F9's panel 4 is uninterpretable without it — it says which part
of the x-range the Δ_S measurement is entitled to.

### Figures NOT to make

* Per-arm `plot_causal_states` for all 14 arms in every folder — 8 folders × 14
  arms is 112 near-identical figures. Produce them for `00_base` only.
* ARI vs CE scatter — they are related through the state assignment, and the
  relationship is already the point of F6.
* Anything with purity on it unless ARI is beside it: **[meas]** purity 94.1%
  with ARI 0.353 on a run that had split one true state six ways.

---

## 6. Rules that fall out of the measurements

1. **β is computed, never written.** `usage_beta = None` → `1/(batch·chunk_len)`,
   resolved by the runner and written to `run_config`. A literal is a bug.
2. **Report `k_found` beside every `S_emp`.** They fail together: S_emp − C is
   +0.005 when the state set is right and −0.17 to −1.04 when it merges. There is
   no partial credit and no middle ground **[meas]**.
3. **Report `H(state|token)`.** 0.000 in every successful cell, 0.11–0.48 in every
   merged one, and computable without ground truth **[meas]**.
4. **Early-stop on first `conv_tol` entry**, not on the final value (§05).
5. **Never quote the penalised objective as a cross-entropy** — the penalty is
   `β·S_emp` and the total is not a CE.
6. **Re-check β whenever batch, chunk or lr changes.** β is pinned to 1/N but the
   *cliff* scales with 1/lr, so the safety margin is not invariant **[meas]**.
7. **No single-seed claim.** §07 exists because the noise floor exceeds most of
   the differences reported so far.

---

## 7. Order, and what blocks what

```
  07_seeds  ->  00_base  ->  01_ksweep  ->  02_statedim
                            \-> 05_budget
                            \-> 06_beta_verify -> 03b_batch  ->  04_capacity
                                              \-> 03_seqlen  ->/
```

`03b_batch` and `03_seqlen` are a pair: both sweep beta across the cliff, one by
varying T and one by varying B.  Run 03b FIRST -- it holds context constant, so
it attributes any collapse to beta unambiguously, and its answer tells you how
to read 03.

`07_seeds` first: it sets the error bar that decides whether anything else is a
finding. `06_beta_verify` before `03_seqlen`, because seqlen changes β and the
two must not be confounded. `04_capacity` last — it is the most expensive and
only matters if 01–03 leave a deficit on the arms needing ≥5 states.

Prerequisite for all of it: the dead β machinery (`usage_target`,
`usage_beta_gap/high/low`, `usage_beta_shared`, `target_occupancy`) must be
removed first — see `DISCRETE_RUNTHROUGH_PLAN.md` §0a. Until then
`run_statistical_trj` writes a β into provenance that the model never used.

---

## 8. Open questions — decide before implementing

1. **Nominal or per-batch β?** The last batch of each epoch is partial (B=20 of
   32), so a literal per-step `1/(B·T)` is 1.95e-4 on 1 step in 16 and 1.22e-4 on
   the other 15. Both below the cliff here, so probably immaterial — but at T=49
   the remainder batch was *past* it. Pilot runs used per-batch.
2. **Is `k_found` or `S_emp` the deliverable?** They answer different questions —
   the count is discrete and comparable to `causal_state_count`, `S_emp` is
   continuous and comparable to C. F1 and F1b split them; the writeup should
   commit to one as primary.
3. **A minimum N.** β = 1/N is unsafe below roughly N ≈ 2000 at lr=1e-3
   **[meas]**. Should the runner refuse, warn, or silently proceed?
