# Remaining Problems and Improvement Plan

**Written:** 2026-08-12, immediately after the reorganisation fix
(`REORGANISATION_FIX_PLAN.md`) was committed.
**Status:** audit complete, nothing here is implemented yet.

Every number in this document was computed from the stored results or read out of
the code during this audit. Where something is an inference rather than a
measurement it says so. Two of my own earlier claims were **wrong** and are
corrected here — see [S3](#s3-the-slope-estimator-is-fragile-and-the-repo-reports-the-fragile-one) and [C4](#c4-wd1000s-figures-are-stale-and-three-cells-are-missing).

The headline: the code is now sound, and the **science has one design flaw that
matters more than everything else combined** — the flower sweep cannot distinguish
"ΔCE tracks the theoretical asymmetry" from "ΔCE tracks `m − n`", because those two
correlate at **+0.977**. §S1 gives a verified, cheap experiment that settles it.

---

## Contents

- [Part I — Implementation](#part-i--implementation)
  - [C1 `save_pkl` is not atomic](#c1-save_pkl-is-not-atomic--a-kill-during-the-write-destroys-the-bundle) · [C2 no repeat-level resume](#c2-the-trajectory-harness-cannot-resume) · [C3 provenance is overwritten](#c3-provenance-is-overwritten-per-invocation) · [C4 stale figures](#c4-wd1000s-figures-are-stale-and-three-cells-are-missing) · [C5 launchers in `/tmp`](#c5-the-launchers-live-in-tmp-and-swallow-child-failures) · [C6 two notebooks](#c6-two-notebooks-import-names-that-no-longer-exist) · [C7 dead flag](#c7-reverse_pos_for_backward-is-a-dead-flag) · [C8 no test for the path fix](#c8-nothing-tests-the-path-fix-that-was-just-made) · [C9 structure](#c9-structural-tidying-carried-over) · [C10 brittle merge test](#c10-the-distinguishability-merge-uses-exact-float-equality)
- [Part II — Science](#part-ii--science)
  - [S1 the m−n confound](#s1-critical--the-design-cannot-separate-c-c-from-m--n) · [S2 the coin grid](#s2-critical--the-coin-grid-measures-h-not-asymmetry) · [S3 the slope estimator](#s3-the-slope-estimator-is-fragile-and-the-repo-reports-the-fragile-one) · [S4 the λ sweep is a null](#s4-the-weight-decay-sweep-is-a-null-and-that-is-a-result) · [S5 no capacity sweep](#s5-the-capacity-sweep-has-never-been-run-and-has-no-code-path) · [S6 effective df](#s6-the-p-values-are-anti-conservative) · [S7 x measured with error](#s7-x-is-measured-with-error-which-attenuates-the-slope) · [S8 distinguishability never fires](#s8-the-distinguishability-refinement-never-fires-on-random-dice) · [S9 divergence](#s9-the-divergence-trigger-is-inferred-never-measured) · [S10 S_emp](#s10-s_emp-remains-the-weakest-measurement-in-the-repo)
- [Part III — The plan](#part-iii--the-plan)

---

# Part I — Implementation

Ordered by what it costs you if it goes wrong, not by effort.

## C1 `save_pkl` is not atomic — a kill during the write destroys the bundle

**Verified.** `Transformer_model/utils.py`:

```python
def save_pkl(obj, path: str):
    with open(path, "wb") as f:          # truncates immediately
        pickle.dump(obj, f, protocol=4)
```

`open(path, "wb")` truncates the existing file *before* a single byte is written.
A 12 MB bundle holding 122 completed sweep cells takes a non-trivial moment to
serialise, and for that whole window the only copy on disk is truncated. Ctrl-C, a
laptop sleep, a full disk, or an `OSError` mid-`dump` leaves an unreadable pickle,
and `load_combined` then treats it as "nothing done yet" — silently discarding the
run.

This is not hypothetical. The λ = 1.0 coin run crashed **inside this function**
(`FileNotFoundError` from `open`). It survived only because the failure was on
`open` rather than during `dump`: the directory had moved, so there was nothing to
truncate. Had the directory still existed and the process been killed a second
later, 23 hours would be gone.

**Fix.** Write to a sibling temp file and `os.replace()`, which is atomic on the
same filesystem:

```python
def save_pkl(obj, path: str):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
```

Ten lines including the comment. It makes every incremental save crash-safe, and
the throttled-save design (`SAVE_EVERY_S`) that already limits data loss to one
interval becomes actually trustworthy. **Do this first.**

## C2 the trajectory harness cannot resume

**Verified** by reading the code: `run_statistical_trj.py` builds
`rec = dict(..., runs=[])` unconditionally and loops `range(args.repeats)`. It
merges by process (`combined[tag] = rec`, so *other* processes survive) but throws
away any repeats the selected process already has.

The sweep, by contrast, skips processes already at `--repeats`. The manual called
the trajectory behaviour "one process, resumable", which invites exactly the wrong
inference; the docs now carry a granularity table, but the underlying asymmetry
remains. At 100 repeats a mistaken re-run costs ~20 minutes per process, ~2.25 h
for all seven.

**Fix.** Lift the sweep's `todo` filter into the trajectory harness — the same six
lines, plus extending `rec["runs"]` from `combined` rather than starting empty. The
two harnesses then behave identically, which is worth more than the time saved.

## C3 provenance is overwritten per invocation

**Verified, and it has already destroyed information.**
`run_config_<CONFIG>.json` is rewritten on every invocation, and the sweep records
the grids of *that* invocation only. Because the pickle merges across invocations,
the two disagree:

| folder | pickle actually holds | `run_config_QUICK.json` claims |
|---|---|---|
| `All_Results/results_sweep` | **100 coin + 81 flower** | `coin_grid: null`, flower `[2..10]` |
| `All_Results/results_sweep_wd/wd0.030` | 100 coin + 25 flower | coin `[0.05..0.95]`, `flower_grid: null` |

For the `wd*` folders the daemon worked around this by copying the flower record
aside as `run_config_QUICK_flower.json`. **`results_sweep` had no such workaround,
so the provenance for its 100 coin cells is simply gone** — the parameters are
recoverable from the pickle's `spec` entries, but the file a reader would consult
says no coin data exists.

**Fix.** Merge rather than overwrite: read the existing JSON, update the keys this
invocation actually set, keep the others, and append an `invocations` list of
`{utc, grids, repeats}`. Then one file describes the pickle beside it. Retroactively,
regenerate `results_sweep`'s JSON from the pickle's specs so the record matches.

## C4 `wd1.000`'s figures are stale, and three cells are missing

**Verified, and this corrects an earlier claim of mine.** I previously said the
λ = 0.1 and 0.3 figures predated the `unstable_mask` fix. They do not. `unstable_mask`
landed in commit `1ccbaed` at 08-11 12:38, and:

| folder | figures written | verdict |
|---|---|---|
| `wd0.030` | Aug 12 03:57 | current (redrawn during the fix verification) |
| `wd0.100` | Aug 11 14:32 | current |
| `wd0.300` | Aug 11 21:41 | current |
| **`wd1.000`** | **Aug 11 06:11** | **stale — predates the fix by 6 h, and drawn from a flower-only pickle** |

`wd1.000` is the one folder affected, on both counts: its `sweep_rows.pkl` and both
PNGs are from 06:11 while its `all_sweep.pkl` is from Aug 12 03:11, and it is
missing 3 of 100 coin cells (1 partial at 18/30, 2 never started).

**Fix.** One command, ~10 minutes, which trains the 3 and redraws everything:

```bash
python -u Experimental_setup/run_sweep_experiment.py --sweep-coin --repeats 30 \
       --weight-decay 1.0 --out-root All_Results/results_sweep_wd/wd1.000
```

The dry-run already confirms it will skip 97 and train 3.

## C5 the launchers live in `/tmp` and swallow child failures

**Verified.** The three fork+`setsid` daemons that ran the λ sweep are in
`/tmp/claude-501/`, hard-code the pre-reorganisation script path and out-root, and
call `subprocess.run(..., check=False)`. That last detail is why
`wd_coin_daemon2.py` printed `ALL COIN GRIDS COMPLETE in 23.18 h` while its final
child had died of a `FileNotFoundError`. macOS will eventually clear `/tmp`.

**Fix.** Promote one into the repo as `Experimental_setup/launch_sweep.py` with the
λ list, family and out-root as CLI arguments; check each child's return code and
refuse to print a completion banner if any failed; keep the fork+`setsid`
detachment (macOS has no `setsid(1)`, and `nohup` blocks only `SIGHUP`). Document
it beside §8 of `HOW_TO_RUN.md`.

## C6 two notebooks import names that no longer exist

**Verified precisely, so the fix is now unambiguous:**

| notebook | broken import | what exists |
|---|---|---|
| `umap_analysis.ipynb`, and the same line in others | `from Data_generation import flower_process_generation` | it lives in `Flower_process_generation` (removed from `Data_generation` by IMPROVEMENT_PLAN C4) |
| `metric_panel_plot.ipynb` | `from LLM_asymmetry_testing import CFG` | that module exports `CONFIGS`, `EVAL_CFG`, `RUN` — no `CFG` |

Both predate the reorganisation. They are now *reachable* (the modules import; the
names are stale), which is why they surface as `ImportError` rather than
`ModuleNotFoundError`.

**Fix.** Repoint the generator import. For `CFG`, read what the notebook does with
it and bind the right one of `EVAL_CFG` / `RUN` / a `CONFIGS[...]` entry — do not
guess, because picking the wrong config is precisely the A4 failure the sidecars
exist to catch.

## C7 `reverse_pos_for_backward` is a dead flag

**Verified:** declared in `OneHot_model.__init__`, consumed at line 270 of
`forward`, and **never set by any caller** — the only other mention is a note in
`build_walkthrough.py` saying it is never used. It is the last remaining
architectural asymmetry between the arms: the forward model's positional encoding
increases with the direction it reads, the backward model's decreases.

**Fix.** Decide, don't leave it ambiguous. Either (a) run one paired A/B at
`reverse_pos_for_backward=True` on two or three processes and record the result —
it is a genuine confound and ~20 minutes answers it — or (b) delete the flag and
say in the README that the PE is deliberately shared. (a) is better; the flag exists
because someone already suspected this.

## C8 nothing tests the path fix that was just made

The reorganisation fix is currently protected by nothing. A future move, or a new
script added without the bootstrap block, reproduces the whole failure.

**Fix.** Three cheap tests in `tests/test_theory.py`:
1. `repo_path` leaves absolute paths alone and anchors relative ones to `REPO_ROOT`.
2. Every default out-root in `CONFIGS` plus both `OUT_ROOT_DEFAULT`s resolves to an
   existing directory.
3. Every `.py` in `Experimental_setup/` with an `if __name__ == "__main__"` block
   imports cleanly in a subprocess with a scrubbed `PYTHONPATH`. This is the one
   that would have caught the original breakage.

## C9 structural tidying carried over

From `REORGANISATION_FIX_PLAN.md` §7, none applied, all judgement calls:

- **§7.1** `Test_data_eval.py` and `plot_state_clusters.py` sit in
  `Transformer_model/` but import `configs` from `Experimental_setup/`, so the
  directory dependency is bidirectional. Both are entry points. Moving them makes
  `Transformer_model/` a true leaf layer. Two `git mv`s plus doc lines.
- **§7.2** `sanity_check_flower_process/` is the only results tree left at the root.
- **§7.3** `requirements.txt` is under `Run_logs/`, where nobody looks for it.
- **§7.4** `lightning_logs/` is 34 MB of output Lightning no longer writes
  (`logger=False`, `enable_checkpointing=False`); `Run_logs/` is 171 MB and gzips
  22×.
- **§7.6** `Experimental_setup/` holds runners, not setup; `Transformer_model/`
  holds generators and the analysis library as well as the model.

## C10 the distinguishability merge uses exact float equality

**Verified.** `flower_complexity` decides whether two outcomes share a backward
causal state by rounding the posterior to 9 decimals and comparing tuples for
**exact equality**. On 1800 random dice draws it fired **0 times** (see [S8](#s8-the-distinguishability-refinement-never-fires-on-random-dice)) — correct for
generic dice, which really are distinguishable. But it means the merge only ever
fires on exactly-proportional columns, and the `n=1, m=2` **null control depends on
it** (verified: gap = +0.0000 there, because every posterior is exactly 1.0).

The risk is a hand-crafted near-degenerate process — two dice intended to be
identical, arrived at through arithmetic that leaves a 1e-12 discrepancy — silently
scoring as fully distinguishable and inflating C⁻. The null control would then read
as a positive.

**Fix.** Cluster posteriors within a tolerance instead of matching exactly (the repo
already has `DEFAULT_STATE_TOL` and clustering machinery), and add a test that dice
perturbed by 1e-10 still merge.

---

# Part II — Science

## S1 CRITICAL — the design cannot separate C⁻−C⁺ from `m − n`

This is the most important item in this document.

**Verified** on the 81-cell flower grid:

```
corr(C--C+, m - n     ) = +0.977    spearman +0.986
corr(C--C+, vocab n+m ) = +0.017    spearman +0.008
corr(C--C+, H_inf     ) = -0.010    spearman -0.026
```

The good news is real: within the flower family the theoretical gap is beautifully
decoupled from vocabulary and from entropy rate, which is what makes it a far
better axis than the coin family. **But it is nearly identical to `m − n`.** So the
headline result — ρ = +0.875 over 76 stable cells — is equally consistent with
"ΔCE tracks causal asymmetry" and with "ΔCE tracks the excess of faces over dice",
and no amount of extra repeats on this grid can tell them apart. Any mechanism that
scales with `m − n` (token-frequency imbalance, how often each die is revisited,
the ratio of selection to outcome tokens) reproduces the result.

### The experiment that settles it — verified constructible

Hold `(n, m)` **completely fixed** and vary only the dice realisation. Then `m − n`,
vocabulary, sequence structure, the number of forward causal states and C⁺ are all
identical by construction, and C⁻ − C⁺ still moves, because

> C⁻ − C⁺ = ½·[ H(π_outcome) − log₂ n ]

and π_outcome depends on how much the n dice overlap.

Measured, with `flower_complexity` itself:

| (n,m) | Dirichlet α | gap range from dice alone | share of the whole 5×5 grid's range |
|---|---|---|---|
| (2,6) | 0.2 | **−0.034 → +0.721** | 35 % |
| (2,8) | 0.2 | **−0.051 → +0.886** | 44 % |
| (4,6) | 0.2 | **−0.331 → +0.269** | 28 % |
| (6,4) | 0.2 | −0.710 → −0.296 | 19 % |

Three of those four **cross zero** — the sign of the prediction flips at fixed
`(n, m)`. That is a within-cell sign test, which the current design cannot perform
at all.

**It also decouples H∞, which is the confound that ruined the coin family.** At
(n,m) = (2,8), α = 0.35, over 4000 draws:

```
H_inf overall range                     : 0.7555 – 1.8482
inside the H_inf band 1.4484 – 1.4662   : gap spans +0.5109 → +0.9041   (0.39 bits)
   corr(gap, H_inf) inside the band     : +0.061
   corr(gap, H_inf) over all draws      : +0.772
```

So with H∞ matched to ~1 %, the gap still spans 0.39 bits. Two concrete processes,
identical `(n,m)`, H∞ equal to 3 decimals:

```
dice seed 2509 : gap = +0.5109   H_inf = 1.4594
dice seed 1428 : gap = +0.9041   H_inf = 1.4551
```

**This is the decisive experiment, and it is cheap.** At the measured throughput
(~0.12 h per flower process at 30 repeats) a 20-point design is **≈ 2.5 hours** —
less than a third of one λ of the sweep already run. It requires no new process
family and no new theory: only a dice-seed axis in the spec builder and an H∞-matched
selection step.

If ΔCE tracks the gap under those conditions, the nuisance explanations are
exhausted. If it does not, the flower trend was `m − n` all along — and that is
worth knowing before anything is written up.

## S2 CRITICAL — the coin grid measures H∞, not asymmetry

**Verified** on the 100-cell coin grid:

```
corr(C--C+, H_inf)          = +0.590
raw     spearman(gap, dCE)  = -0.415   p = 1.8e-05
partial spearman | H_inf    = +0.041   p = 0.68
        spearman(H_inf,dCE) = -0.793   p = 8.3e-23
```

Three things follow. The raw correlation is **the wrong sign** — the coin's
C⁻ − C⁺ is positive everywhere, so the hypothesis predicts ΔCE rising with the gap,
and it falls. Controlling for H∞ the relationship vanishes entirely (+0.041,
p = 0.68). And H∞ on its own predicts ΔCE at ρ = −0.793.

So the coin grid is measuring *how hard the process is*, not how asymmetric — at a
fixed 130-step budget, higher-entropy processes sit further from convergence and
their two arms' residuals differ differently. It contributes **no evidence** about
causal asymmetry.

Worse, weight decay strengthens the artefact rather than suppressing it: the coin
ρ goes −0.415 (λ=0) → −0.399 → −0.342 → −0.465 → **−0.579** (λ=1), and the
intercept rises from +0.00043 to +0.00137.

**Fix — reporting, not compute.** The coin grid should be presented as a *positive
control for the confound*: a demonstration that this measurement responds strongly
to entropy rate, which is exactly why the flower family (corr(gap, H∞) = −0.010) is
the axis that carries the claim. Stating it that way turns a wrong-sign
embarrassment into the strongest argument for the design. Any coin-family ρ quoted
without its partial correlation is misleading and should be removed.

## S3 the slope estimator is fragile, and the repo reports the fragile one

**This corrects a claim I made an hour ago.** I said the flower slope was not
significant at any λ. That is true of *ordinary least squares on all cells*, and it
is an artefact of that estimator, not a fact about the data:

| estimator | n | slope b | t |
|---|---|---|---|
| OLS, all cells | 81 | +0.00344 ± 0.00248 | **+1.38 (null)** |
| OLS, `unstable_mask` applied | 76 | +0.00317 ± 0.00025 | +12.69 |
| **inverse-variance WLS, all cells** | **81** | **+0.00294 ± 0.00016** | **+18.87** |

The per-cell sems span **249×** (0.00038 → 0.09442) before filtering and 4× after.
OLS assumes they are equal, so five unstable cells dominate the residual variance
and flatten the t-statistic. Inverse-variance weighting gives t = +18.9 and —
importantly — returns **the same estimate with or without the filter**
(+0.00294 ± 0.00016 both times), because it down-weights those cells automatically
instead of discarding them.

**Fix.** Report the WLS slope as primary. `unstable_mask` is doing crudely, by
deletion, what weighting does properly; keep it for the figures' y-limits, where it
genuinely helps readability, but stop letting a hard threshold decide which cells
enter the inference. Add the WLS slope and its t to `print_sweep_summary`, which
currently reports Spearman ρ and an unweighted fit.

## S4 the weight-decay sweep is a null, and that is a result

The a(λ)/b(λ) decomposition that `WEIGHT_DECAY_PLAN.md` §6.2 asked for was never
built. Computed here for the first time, fitting ΔCE = a(λ) + b(λ)·(C⁻−C⁺) per λ
per family (OLS, so read the slopes with [S3](#s3-the-slope-estimator-is-fragile-and-the-repo-reports-the-fragile-one) in mind):

| λ | family | n | intercept a | slope b | ρ |
|---|---|---|---|---|---|
| 0 | flower | 81 | −0.00134 ± 0.00123 | +0.00344 ± 0.00248 | +0.827 |
| 0.03 | flower | 25 | −0.00386 ± 0.00310 | +0.00355 ± 0.00552 | +0.728 |
| 0.10 | flower | 25 | +0.01024 ± 0.00905 | +0.00659 ± 0.01612 | +0.751 |
| 0.30 | flower | 25 | −0.00955 ± 0.00865 | +0.00744 ± 0.01539 | +0.811 |
| 1.00 | flower | 25 | −0.00195 ± 0.00995 | +0.01188 ± 0.01771 | +0.625 |
| 0 | coin | 100 | +0.00043 ± 0.00022 | −0.00266 ± 0.00054 | −0.415 |
| 1.00 | coin | 98 | +0.00137 ± 0.00029 | −0.00498 ± 0.00072 | −0.579 |

**Reading.** Every flower slope overlaps every other within one standard error, and
b(λ)/se ≤ 0.67 for λ ≥ 0.1 — the λ ≥ 0.1 folders are too noisy to constrain b at
all. The intercepts swing in sign with no pattern and errors larger than the swing.
Meanwhile ρ stays +0.63…+0.83 throughout.

So: **weight decay changes the variance, not the sensitivity.** It neither reveals
nor destroys the asymmetry signal at any λ that leaves both arms trainable. That is
a clean negative result and it is worth stating — it closes off regularisation as an
explanation for the near-null ΔCE, which is what the λ sweep was for.

One caveat on the comparison: λ = 0 has 81 flower cells and the others 25, so the
λ = 0 slope is better determined for reasons unrelated to λ. Comparing like with
like would mean restricting λ = 0 to the same 25 cells.

## S5 the capacity sweep has never been run, and has no code path

**Verified:** `configs.py` offers three fixed `d_model` values (16/32/64) across
four configs, and `run_sweep_experiment.py` has no capacity axis — `d_model` appears
only in a printout and in a docstring caveat. `README.md` line 103 still says the
`d_model` sweep is "not yet run".

This is the **direct test of the repo's own central argument.** §1.2 of the README
derives that ΔCE is a difference of *residuals*, so it must shrink toward zero as
capacity grows and vanish for an unbounded predictor. Until that is measured,
ΔCE ≈ 0 is ambiguous between the two readings that matter most:

- "this process pair is symmetric" (a result), and
- "d_model = 32 was already enough for both directions" (no result).

Nothing else in the repo can break that tie, and the sanity-check docstring already
flags the same gap for the positive control ("this needs the d_model sweep to
interpret").

**Fix.** Add `--d-model` to the sweep (it is one `cfg` override), then run a small
grid — say d_model ∈ {8, 16, 32, 64} × the 4–6 flower cells with the largest
|C⁻−C⁺| × 30 repeats. ~100 processes, and larger models are slower, so budget
6–8 hours. The prediction is monotone shrinkage of |ΔCE| with d_model; a flat
profile would falsify the residual argument itself.

## S6 the p-values are anti-conservative

The 81 flower cells are a 9×9 grid over `(n,m) ∈ {2..10}²` of a **smooth
deterministic function** of (n, m) and the dice, with one dice draw per cell. So
neighbouring cells are near-duplicate processes and the effective degrees of freedom
are far below 81. The module docstring says exactly this — and then reports
p = 1.9 × 10⁻²¹, which is nominal.

**Fix.** Report a permutation test that shuffles ΔCE across cells while preserving
the (n,m) neighbourhood structure, or a block-bootstrap over `m − n` strata, and
quote that interval instead. Cheap — it is analysis-time only, no retraining. The
dice-axis design in [S1](#s1-critical--the-design-cannot-separate-c-c-from-m--n) largely dissolves the problem, because independent dice
draws at fixed (n,m) *are* independent processes.

## S7 x is measured with error, which attenuates the slope

Each (n,m) uses one dice realisation from `cfg["flower_dice_seed"]`, so C⁻ − C⁺
carries realisation noise at fixed (n,m) — the x-axis is measured with error, which
biases any fitted slope **toward zero**. This was a deliberate choice (it keeps the
processes identical to those the other runners train on, which is what makes the
baseline cross-check possible) and it is documented. Its size has never been
quantified.

It is now quantifiable from the [S1](#s1-critical--the-design-cannot-separate-c-c-from-m--n) measurements: at α = 1.0 the within-cell gap
spread is 0.13–0.45 bits depending on (n,m), against a between-cell range of 2.14.
That is a substantial errors-in-variables problem, and it means the true slope is
**larger** than the reported +0.0029. The dice-axis design fixes this at the root by
making the dice the independent variable rather than a nuisance.

## S8 the distinguishability refinement never fires on random dice

**Verified: 0 of 1800 random dice draws.** For every generic dice matrix,
C⁻ = 1 + ½·H(π_outcome) exactly, and the merge over "distinguishable outcomes" —
the correction `IMPROVEMENT_PLAN.md` A1 presents as the repo's key theoretical fix,
and which genuinely repaired the old `m+1` formula — changes nothing.

This is not a bug. Distinct continuous dice really do induce distinct posteriors, so
no merging is *correct*. But it has a consequence worth writing down: the sweep's
entire x-axis is a function of the marginal outcome entropy and n, and the
distinguishability machinery is load-bearing **only** for the degenerate cases —
above all the `n=1` null control, whose correctness depends on it (see [C10](#c10-the-distinguishability-merge-uses-exact-float-equality)).

**Fix.** Documentation, plus the C10 tolerance. State the closed form
C⁻ − C⁺ = ½[H(π_outcome) − log₂ n] in the README where the flower process is
introduced. It makes the design transparent and it is what makes [S1](#s1-critical--the-design-cannot-separate-c-c-from-m--n) obviously
constructible rather than a lucky search.

## S9 the divergence trigger is inferred, never measured

The current account — no finite minimiser, so Adam normalises a vanishing gradient
into a non-vanishing step, weights grow, the logit surface becomes exponentially
asymmetric, and a normal step falls off a cliff — is consistent with everything
measured (attn/ffn weight norms ×2.29/×1.97, LN gains *shrinking*, residual stream
×1.57, max|logit| ×2.11) but the **trigger itself has never been observed**. Two
mechanisms I proposed earlier were tested and refuted: the effective step *falls*
(9.9e-3 → 2.7e-4), and a stale `v̂` cannot produce more than `(1−β₁)/√(1−β₂)` = 3.16×
`lr` on a single excursion.

It matters because the step budget (130) sits at the low end of the observed onset
window, so the divergence rate is a tunable that nobody controls, and because
`converged_mask` and the converged-only statistics are downstream of it.

**Fix.** Log per-step `‖g‖`, `‖θ‖` and `max|logit|` for a handful of runs that are
known to diverge. That is a recorder hook and one figure — cheap, and it either
confirms the account or replaces it.

## S10 S_emp remains the weakest measurement in the repo

k-means at a hand-specified k, bounded above by log₂k, and biased upward when state
occupancy is unbalanced. It is the headline complexity number in every figure.
`--khat` already implements the better estimator (recovering k̂ from the predictive
distribution). Nothing new to build — the remaining work is to make k̂ the reported
quantity and demote S_emp to a cross-check, which is a reporting decision plus one
sweep re-run at analysis time.

---

# Part III — The plan

Sequenced so that cheap insurance lands before long runs, and so the decisive
science is not gated behind tidying.

### Phase 0 — insurance and cleanup (≈2 h, no training)

| | item | why now |
|---|---|---|
| 1 | [C1](#c1-save_pkl-is-not-atomic--a-kill-during-the-write-destroys-the-bundle) atomic `save_pkl` | every later phase writes bundles; this is 10 lines and removes the one failure that loses a whole run |
| 2 | [C8](#c8-nothing-tests-the-path-fix-that-was-just-made) tests for the path fix | stops the reorganisation breakage recurring silently |
| 3 | [C2](#c2-the-trajectory-harness-cannot-resume) repeat-level resume in the trajectory harness | same six lines as the sweep; removes a 2.25 h footgun |
| 4 | [C3](#c3-provenance-is-overwritten-per-invocation) merge provenance, regenerate `results_sweep`'s JSON | the coin provenance is currently *lost*, not just stale |
| 5 | [C4](#c4-wd1000s-figures-are-stale-and-three-cells-are-missing) finish λ=1.0 (3 cells) and redraw | ~10 min; completes the λ sweep |
| 6 | [C10](#c10-the-distinguishability-merge-uses-exact-float-equality) tolerance in the merge + a test | the null control's correctness rests on it |

### Phase 1 — analysis-only science (≈3 h, no training)

| | item | output |
|---|---|---|
| 7 | [S3](#s3-the-slope-estimator-is-fragile-and-the-repo-reports-the-fragile-one) WLS slope as the primary statistic | a slope you can defend: +0.00294 ± 0.00016 |
| 8 | [S4](#s4-the-weight-decay-sweep-is-a-null-and-that-is-a-result) a(λ)/b(λ) table into the repo and the README | closes out `WEIGHT_DECAY_PLAN` §6.2 |
| 9 | [S2](#s2-critical--the-coin-grid-measures-h-not-asymmetry) reframe the coin grid as a confound control | turns a wrong-sign result into the argument for the design |
| 10 | [S6](#s6-the-p-values-are-anti-conservative) permutation / block-bootstrap p-values | honest uncertainty on ρ |
| 11 | [S8](#s8-the-distinguishability-refinement-never-fires-on-random-dice) document the closed form for C⁻−C⁺ | makes the design legible, and Phase 2 obvious |

Phases 0 and 1 together make everything already measured reportable. Nothing in
them requires a GPU-hour.

### Phase 2 — the decisive experiment (≈3 h training)

**[S1](#s1-critical--the-design-cannot-separate-c-c-from-m--n): the fixed-(n,m), dice-varying design.** Add a dice axis to the sweep's spec
builder; select ~20 dice realisations at (n,m) = (2,8) and (4,6) spanning the widest
gap range inside an H∞ band matched to ~1 %; 30 repeats each. Include at least three
cells with a **negative** gap at the same (n,m) as positive ones, which the current
design cannot offer anywhere.

This is the highest-value item in the document. It is the only experiment that can
distinguish the project's hypothesis from `m − n`, and it simultaneously removes the
H∞ confound ([S2](#s2-critical--the-coin-grid-measures-h-not-asymmetry)), the errors-in-variables attenuation ([S7](#s7-x-is-measured-with-error-which-attenuates-the-slope)) and most of the
effective-df problem ([S6](#s6-the-p-values-are-anti-conservative)).

### Phase 3 — the capacity axis (≈8 h training)

**[S5](#s5-the-capacity-sweep-has-never-been-run-and-has-no-code-path): the d_model sweep.** `--d-model` flag, then d_model ∈ {8,16,32,64} on the
highest-|gap| flower cells. Tests the residual argument directly and disambiguates
every near-null ΔCE the repo has reported. Run it after Phase 2 because Phase 2
tells you which cells are worth the capacity axis.

### Phase 4 — mechanism and tidying (opportunistic)

[S9](#s9-the-divergence-trigger-is-inferred-never-measured) per-step norm logging · [C7](#c7-reverse_pos_for_backward-is-a-dead-flag) settle the PE flag with one A/B · [S10](#s10-s_emp-remains-the-weakest-measurement-in-the-repo) promote k̂ over
S_emp · [C5](#c5-the-launchers-live-in-tmp-and-swallow-child-failures) a real launcher in the repo · [C6](#c6-two-notebooks-import-names-that-no-longer-exist) the two notebooks · [C9](#c9-structural-tidying-carried-over) the structural
moves.

### What I would not do

- **More repeats on the existing flower grid.** ρ = +0.875 at n = 76 is not
  limited by precision; it is limited by [S1](#s1-critical--the-design-cannot-separate-c-c-from-m--n). More repeats buy a tighter estimate
  of a quantity whose interpretation is contested.
- **More λ values.** [S4](#s4-the-weight-decay-sweep-is-a-null-and-that-is-a-result) shows b(λ) is flat across two orders of magnitude and
  the λ ≥ 0.1 folders cannot constrain it. λ = 3 and above destroy both arms.
- **Packaging the repo.** Verified during the reorganisation fix that no pickle or
  checkpoint references a repo class, so it is *safe* — but it buys nothing this
  project needs.
