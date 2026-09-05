# Work Logbook

Period covered: **4 August – 14 August 2026**. Compiled from the commit history
and diffs. (The preceding commits are from March 2026 and fall outside this
window.)

Each row is one task, sized to fit a logbook table cell.

| Work and/or Tasks Performed | Date (From) | Date (To) |
|---|---|---|
| Audited the codebase against the causal-asymmetry hypothesis it was meant to test, and wrote a plan for the repairs. | 2026-08-04 | 2026-08-04 |
| Pinned a reproducible environment (requirements + lock file) after finding the base environment could not import `umap`, so nothing in the repo would run. | 2026-08-04 | 2026-08-04 |
| Removed ~1 GB of generated output (training logs, caches, result images) from version control and stopped writing dead checkpoints. | 2026-08-04 | 2026-08-04 |
| Fixed a NaN in the cross-entropy: a `0 × −inf` term appeared on saturated logits, reachable because both processes contain deterministic transitions with no finite optimum. | 2026-08-04 | 2026-08-04 |
| Paired the forward and backward arms. They had been drawing different train/test splits and context windows, so the headline difference carried split noise larger than the effect being measured. | 2026-08-04 | 2026-08-04 |
| Corrected the backward model's read position. Under a `triu` mask the maximum-context position is the first, not the last — every backward complexity number in the repo had been computed from a single token. | 2026-08-04 | 2026-08-04 |
| Moved analysis to the training chunk length. Evaluating at full sequence length asked the model to extrapolate to positional indices it had never been trained on. | 2026-08-04 | 2026-08-04 |
| Derived the flower closed forms over *distinguishable* outcomes rather than `m+1` states. This reversed the sign prediction and showed every flower configuration in the repo had been testing the negation of the hypothesis. | 2026-08-04 | 2026-08-04 |
| Added a paired ΔCE statistic that separates optimisation failures from measurements, after one diverged fold was found inflating a result by 75×. | 2026-08-04 | 2026-08-04 |
| Consolidated two near-duplicate 700-line runners into one runner plus a single config file. They had shared an output root and a tag at different parameters, silently overwriting each other's checkpoints. | 2026-08-04 | 2026-08-04 |
| Vectorised three Python double-loops (~10⁶ iterations per call), separated the validation cadence from the training-curve cadence, cached attention masks, and made perplexity accumulate incrementally instead of holding 80 MB of logits. | 2026-08-04 | 2026-08-04 |
| Wrote the first regression tests covering the closed forms and the loss path. | 2026-08-04 | 2026-08-04 |
| Replaced the assumed causal-state count with a discovered one (silhouette-selected k̂), and added total-variation and Jensen–Shannon distances between recovered and theoretical conditionals. | 2026-08-05 | 2026-08-05 |
| Rewrote the latent (UMAP) plots, which had been embedding a single process realisation — any cluster structure previously shown was an artefact. Added state-cluster plots coloured by token and marked by recovered state. | 2026-08-05 | 2026-08-05 |
| Made training divergence reported rather than left to be discovered, and recorded that the `LARGE` config's cross-validation is dominated by it and therefore unusable. Added a `QUICK_LARGE_HMM` config and the walkthrough notebook. | 2026-08-05 | 2026-08-05 |
| Wrote `HOW_TO_RUN.md` as a formal tutorial and rewrote `README.md` around the experimental design rather than the code structure. | 2026-08-06 | 2026-08-08 |
| Built the repeat-statistics harness (`run_statistical_trj.py`): *N* independent repeats per process with seed-paired arms, recording the full per-step loss trajectory. Needed because the fold-level error bar used until then was a lower bound — folds share a training set. | 2026-08-08 | 2026-08-08 |
| Produced the first trajectory results over seven processes. **The headline result is a null:** ΔCE of order 0.001–0.003 bits, with error bars of the same size. | 2026-08-08 | 2026-08-08 |
| Built the parameter-sweep harness (`run_sweep_experiment.py`), sweeping the coin over (p,q) and the flower over (n,m), to ask whether measured asymmetry *tracks* theoretical C⁻−C⁺ across many processes — the question a single process cannot answer. Planned first, with a measured compute budget and closed-form coverage per grid. | 2026-08-10 | 2026-08-10 |
| Added the area under the forward/backward trajectory-difference curve as a second response variable, on the argument that endpoint ΔCE is a difference of *residuals* and must vanish once both arms converge, whereas the path to convergence need not be symmetric. | 2026-08-10 | 2026-08-10 |
| Added weight decay (AdamW) as a training option, threaded through model, trainer, config and both runners, defaulting off so existing results stay bit-identical. Chosen over label smoothing and confidence penalties, which were computed to tax the two arms asymmetrically and manufacture a spurious trend. Planned, then run as a λ sweep. | 2026-08-11 | 2026-08-11 |
| Reorganised the tree into role-based directories and repaired the fallout, committing the move and the fix separately so the breakage is reviewable in isolation. | 2026-08-12 | 2026-08-12 |
| Fixed repo-relative path resolution. Output roots resolved against the working directory, and a missing results file reads as "nothing done yet" — so moving the tree mid-run had silently discarded a 23-hour sweep and retrained from scratch. | 2026-08-12 | 2026-08-12 |
| Audited the remaining implementation and scientific gaps (10 of each) and planned the four phases that follow. | 2026-08-12 | 2026-08-12 |
| Made the pipeline crash-safe: atomic result writes (the old path truncated before writing, so a kill mid-dump destroyed the bundle), repeat-level resume with eight tested refusal paths, and run provenance that merges instead of overwriting. | 2026-08-12 | 2026-08-12 |
| Replaced the reported slope estimator with inverse-variance weighted least squares. Per-cell standard errors span 249×, so the unweighted fit read as a null (t = +1.4) where the weighted fit gives t = +18.9 — the repo had been reporting the estimator that hides its own result. | 2026-08-12 | 2026-08-12 |
| Replaced the nominal p-value with a block bootstrap over grid rows, since neighbouring cells of a lattice are near-duplicate processes and cell-level resampling overstates significance. | 2026-08-12 | 2026-08-12 |
| Analysed the weight-decay sweep and **corrected the earlier conclusion**: the slope is flat for λ ≤ 0.3 but roughly doubles at λ = 1 in *both* families — including the coin, a known artefact — so it is a scale effect from degraded training, not regularisation revealing causal structure. | 2026-08-12 | 2026-08-12 |
| Established that the coin grid is a confound control rather than evidence: its raw trend has the wrong sign for the hypothesis and vanishes once entropy rate is partialled out. | 2026-08-12 | 2026-08-12 |
| Designed and launched the dice experiment, which can actually settle the headline question. C⁻−C⁺ and (m−n) correlate at +0.977 on the flower grid, so the observed trend is equally consistent with either explanation; varying how much the dice *overlap* moves C⁻−C⁺ at fixed (n,m), pinning vocabulary, capacity and step budget by construction. Cells chosen by surveying 3000+ draws for the widest gap range inside a ±1% entropy-rate band containing both signs. | 2026-08-12 | 2026-08-12 |
| Added a model-capacity axis and its analysis — the direct test of the repo's own residual argument, which had never been run and had no code path. Until it is, every near-null ΔCE stays ambiguous between "symmetric process" and "capacity already sufficient". | 2026-08-12 | 2026-08-12 |
| Repaired two notebooks that had been broken since before the reorganisation, importing names and referencing tags that no longer existed. | 2026-08-12 | 2026-08-12 |
| Documented the new experiments and recorded the three places where executing the plan overturned what the plan itself had asserted. | 2026-08-12 | 2026-08-12 |
| Implementing a discretised training strategy for the decoder: an argmax one-hot bottleneck over a fixed number of causal states, with a straight-through estimator carrying the gradient past the non-differentiable argmax. The aim is to make the model learn an explicit, countable state set rather than a continuous latent, so the state count and its emission table can be read off directly instead of being clustered out of the latents. **Work in progress, not committed.** | 2026-08-13 | 2026-08-14 |
| Diagnosed the first prototype of the discretised decoder. Three faults found: the bottleneck was discretised twice with the straight-through estimator wrapped around the second argmax, so the graph was cut and only 2 of 32 parameter tensors received gradient while the transformer stayed at random initialisation; the loss was taken on the one-hot itself, which caps confidence and imposes a vocabulary-dependent cross-entropy floor (0.80 bits at V=3, 2.34 at V=12) along the same axis the experiments sweep; and the state count was never a free parameter, being fixed to the vocabulary size. Restructured it to the specified pipeline — logits, straight-through argmax one-hot, a learnable vocab × K state matrix, then an emission layer decoding the state vector back to next-token logits — and added a usage-entropy penalty against collapse. Added the occupancy, state-vector and per-state emission plots, and a check against the closed form. On the coin process the prototype recovers exactly 2 causal states, assigns them with 100% purity against the true hidden states, matches the theoretical emission table to within 0.005, and sits 0.012 bits above H∞. **Work in progress, not committed.** | 2026-08-14 | 2026-08-14 |
| Dice experiment running (20 processes × 30 repeats); notebook and plotting work in progress. | 2026-08-13 | 2026-08-14 |

---

## Notes

- Regression test count over the period: **61 → 74**.
- Open items in the working tree at time of writing:
  - `Transformer_model/Training_model.py` uses a package-qualified import, which
    conflicts with the flat-module convention settled during the reorganisation
    fix; it would load the module a second time under a different name.
  - `All_Results/results_dice/dice_scatter.png` is untracked but is a real result.
