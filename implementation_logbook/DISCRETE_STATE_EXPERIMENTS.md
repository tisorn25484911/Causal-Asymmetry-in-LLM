# Discrete Causal-State Decoder — Experiments, Methodology, Figures

Everything measured on `Transformer_model/DiscreteCausal_model.py` between
2026-08-14 and 2026-08-25, the methodology those measurements settle on, and the
figures and tables worth producing from them.

Nothing in this document is a proposal without evidence: every number below was
measured, and every claim that failed is recorded as such.

---

## 0. The question this model exists to answer

`OneHotDecoder` produces a continuous latent, and the causal states have to be
*clustered out of it* — `Model_analysis.recover_causal_states`, with a
silhouette-selected `k̂` and a `state_tol`. Two hyperparameters stand between the
model and the number being reported.

`DiscreteCausalDecoder` forces the prediction through a hard one-hot over `K`
states, so the state of a position is just `argmax(state_logits)`. Then:

* the number of causal states is **counted**, not clustered;
* `S_emp` is the entropy of the occupancy distribution, comparable directly to
  C⁺ / C⁻, with no clustering hyperparameters at all;
* `output_prj.weight` is a K×V **emission table** — one predictive distribution
  per state — directly comparable to the ε-machine's.

That is a stronger instrument than ΔCE *if it works*. Sections 5–7 establish
where it works and where it does not.

---

## 1. The architecture, and the three faults already fixed

```
x            (B,T,D)   transformer stack        shared with OneHotDecoder
state_lgt    (B,T,K)   state_head    D -> K     the state code
onehot       (B,T,K)   argmax + straight-through
rep          (B,T,S)   onehot @ state_matrix
out          (B,T,V)   emission      S -> V     <- the cross-entropy is here
```

Three faults were found in the first prototype and are recorded because each
would have produced plausible-looking but meaningless numbers:

| fault | symptom | evidence |
|---|---|---|
| Two discretisations, straight-through wrapped around the **second** argmax | graph severed; **2 of 33** parameter tensors trained while the transformer stayed at initialisation — and the loss curve still fell | gradient audit: `WITH gradient: ['embedding_map.weight', 'embedding_map.bias']`, 30 tensors dead |
| Loss taken on the one-hot itself | softmax of a one-hot caps confidence, putting a **vocabulary-dependent floor** under CE: 0.796 bits at V=3, 1.305 at V=5, 2.335 at V=12 | closed form `−log₂(1/(1+(V−1)/e))` |
| `p_bar` computed from the **softmax** rather than the straight-through tensor | the anti-collapse penalty read 2.87 bits when the true occupancy entropy was 0.70 — blind through exactly the window in which collapse is decided | at init: soft 2.8715 vs hard 0.7043; after one epoch 2.6888 vs 0.8452 |

The floor issue matters specifically for this study: the flower grid varies V, so
a V-dependent CE floor would sit along the sweep's own axis.

---

## 2. Ground truth — why these processes are exactly checkable

For **both** the coin and the flower, in **both** directions, the causal state is
a deterministic function of the current token. That makes state recovery a
scoreable classification problem rather than a judgement call.

### Coin (p=0.6, q=0.4), V=3

```
forward : token 1 -> s1 ;  tokens 0,2 -> s0        K=2, occupancy [0.40,0.60], C+ = 0.9710
backward: token 0 -> s0 ;  token 1 -> s1 ; token 2 -> s2
                                                    K=3, occupancy [0.16,0.60,0.24], C- = 1.3593
H_inf = 0.9710 (time-reversal invariant, so BOTH arms share this floor)
```

Measured conditionals:

```
forward   token 0: [0.4009, 0.5991, 0.0000]   H = 0.9715
          token 1: [0.0000, 0.6001, 0.3999]   H = 0.9709
          token 2: [0.3992, 0.6008, 0.0000]   H = 0.9705   <- IDENTICAL to token 0
backward  token 0: [0.4014, 0.0000, 0.5986]   H = 0.9718
          token 1: [0.1598, 0.6002, 0.2400]   H = 1.3590
          token 2: [0.0000, 1.0000, 0.0000]   H = 0.0000   <- DETERMINISTIC
```

### Flower (n=2, m=3, Dirichlet α=0.5 seed 42), V=5

```
forward : selection token i -> state i+1 ; outcome token -> state 0
                                          K = n+1 = 3, occupancy [0.50,0.25,0.25], C+ = 1.5000
backward: selection token -> state 0 ; outcome j -> state j+1
                                          K = m+1 = 4, occupancy [0.50,0.3038,0.1574,0.0388],
                                          C- = 1.6240
H_inf = 0.9340,  gap C- - C+ = +0.1240
```

### The reference bounds this yields

The CE achievable by **every** grouping of the true states, computed in closed
form. This is the single most useful diagnostic in the whole set-up, because it
converts "the model did badly" into "the model is sitting at the best 2-state
solution" or "the model is not even at a partition optimum".

```
             3 states   best 2   1 state
forward        0.9709   0.9709    1.3591     <- forward reaches H_inf with TWO states
backward       0.9709   1.1133    1.3591     <- backward needs THREE
```

**This asymmetry is the root of everything in §6.** Forward's optimal solution
needs only 2 states because tokens 0 and 2 are forward-equivalent, and it beats
the next-best 2-state grouping by 0.226 bits — a wide, unambiguous basin.
Backward has no 2-state shortcut, and its two best 2-state groupings are
separated by only 0.034 bits.

---

## 3. Metrics — formal definitions

Let `pred[b,t] = argmax(state_logits[b,t])` and `truth[b,t] = smap(token[b,t])`.

| metric | definition | reads as |
|---|---|---|
| **CE** | mean cross-entropy in bits over all B·T positions | compare to H∞ and to the §2 bounds |
| **k_found** | number of states with non-zero occupancy | compare to the true K |
| **occupancy** | `mean_{b,t} onehot(pred)` — a normalised COUNT | compare to the theoretical stationary distribution |
| **H_occ** | entropy of the occupancy | **this is the S_emp estimate**; compare to C |
| **ARI** | adjusted Rand index of `pred` vs `truth` | permutation- and count-invariant; 1.0 = exact recovery |
| **purity** | each model state votes for its most common true state | tolerant of over-splitting, so report it *with* ARI, never instead |
| **H(state\|token)** | conditional entropy of the assignment given the token | **0 iff the state is a pure function of the token**, which the truth is. >0 means the model is keying on context it does not need |
| **emission table** | `emission(state_matrix).softmax(-1)` | K×V, compare row-wise to the true conditionals |

Two notes that cost time to learn:

* **purity alone is misleading.** Backward K=8 β=0 scores 94.1% purity with
  ARI 0.353 — it had split one true state into six.
* **H(state|token) is the diagnostic that localises the failure.** Forward scores
  0.0000 in every configuration ever run; backward's failures all have it > 0.

---

## 4. Protocol

Fixed across every run reported here, so that K and β are the only variables:

```
d_model 20, n_layers 2, tau 1.0, lr 1e-3, AdamW, weight_decay 0
300 epochs x 10 steps = 3000 gradient steps
300 sequences x 100 tokens, batch 32  ->  T = 99, N = B*T = 3168
seed 0, CPU, penalty applied as `loss + beta * H_occupancy`
```

Rules the measurements forced:

1. **Never quote the penalised objective as a cross-entropy.** Log CE separately.
2. **Report `k_found` and `H(state|token)` beside every CE.** A CE number alone
   cannot distinguish "converged" from "collapsed to the marginal".
3. **Check the states-used trace, not just the final count.** Two of twenty runs
   were still moving at epoch 300; both look like results in a summary table.
4. **Every run collapses to k=1 within 1–3 epochs, then re-expands.** Sampling the
   trace every 30 epochs hides this — it caused one wrong claim in this study
   ("backward never collapses"), corrected here.

---

## 5. Experiments run, and what each established

| # | experiment | settled |
|---|---|---|
| E1 | Gradient audit of the prototype | STE around the wrong argmax; 2/33 tensors training |
| E2 | Closed-form CE floor of a one-hot output | V-dependent floor 0.80–2.34 bits; emission must follow the bottleneck |
| E3 | Soft vs straight-through `p_bar` | 2.17-bit disagreement at init; the penalty was blind during collapse |
| E4 | β ∈ {0, 1/T, 1} | 1/T and 1 give **bit-identical** failures: `H=0` is absorbing under `+β·H` |
| E5 | β = 1/N and threshold bracket | works; threshold between 3.16e-4 and 1e-3 |
| E6 | K ∈ {2,3,4,8,16} × β ∈ {0,1/N} × both arms | the central result set, §6 |
| E7 | Partition bounds (closed form) | forward needs 2 states, backward 3 |
| E8 | Baseline `OneHotDecoder` control | ΔCE ∈ [−0.0011, +0.0024]; backward is NOT intrinsically harder |
| E9 | β sweep 0 → 1e-2 (8 values), both arms, K ∈ {2,4,8}, coin **and flower** — 80 runs | threshold is 5e-4–7e-4, non-monotonic; rare-state blindness; structural-zero rule confirmed. §6.6–6.8 |

---

## 6. Results

### 6.1 Forward is a fixed point

Ten configurations — K ∈ {2,3,4,8,16} × β ∈ {0, 1/N}:

```
k_found  = 2          in all ten
ARI      = 1.000      in all ten
H(s|tok) = 0.0000     in all ten
occupancy= [0.6034, 0.3966]   bit-identical in all ten
H_occ    = 0.9689     vs C+ = 0.9710   (the -0.0020 is finite-sample, not model error)
```

Only the CE varies, and only through convergence speed. Two runs of the ten had
not converged at 3000 steps (K=2 β=1/N at +0.0136; K=4 β=1/N at +0.1874).

### 6.2 Backward needs slack in K, and β to protect it

```
   K  beta | final CE  CE-Hinf | k_found | H_occ    C     H_occ-C |   ARI  H(s|tok)
   2     0 |  1.2275   +0.2565 |    2    | 0.7479 1.3593  -0.6114 | 0.332   0.2779
   3     0 |  1.2271   +0.2561 |    2    | 0.6982 1.3593  -0.6611 | 0.377   0.1879
   3   1/N |  1.2311   +0.2601 |    2    | 0.6618 1.3593  -0.6975 | 0.409   0.1124
   4     0 |  1.0375   +0.0665 |    4    | 1.9696 1.3593  +0.6103 | 0.523   0.8289
   4   1/N |  1.0239   +0.0529 |    3    | 1.3590 1.3593  -0.0003 | 0.980   0.0474   <- BEST
   8     0 |  1.0287   +0.0578 |    8    | 2.6159 1.3593  +1.2565 | 0.353   1.5101
   8   1/N |  1.0442   +0.0732 |    3    | 1.4621 1.3593  +0.1028 | 0.737   0.4358
  16     0 |  1.0229   +0.0519 |   10    | 2.3645 1.3593  +1.0052 | 0.487   1.1663
  16   1/N |  1.0289   +0.0580 |    3    | 1.3989 1.3593  +0.0396 | 0.910   0.1942
```

Three separate facts, often confused:

**(a) K = the true count fails.** K=3 backward — the value the class docstring
tells you to pass — never opens its third state within 3000 steps, at either β.
K=4 succeeds. Slack is what allows the 2→3 escape.

**(b) At β=0 the slack fills with duplicates.** K=8 β=0 builds five near-identical
copies of token 1's row; K=16 β=0 builds seven. Duplicating a state costs
**exactly zero** cross-entropy, so nothing at β=0 discourages it. Consequence:
`H_occ` is inflated by up to 1.26 bits and is **not** an estimate of C.

**(c) β=1/N kills the duplicates without triggering collapse.** It pins k_found at
exactly 3 for every K ≥ 4, stable from epoch 23–45 onward.

The best cell recovers the backward ε-machine:

```
backward K=4 beta=1/N:  H_occ = 1.3590  vs  C- = 1.3593        error 0.0003 bits
  state 1 (occ 0.5976): [0.1558, 0.5991, 0.2451]   true token 1: [0.1598, 0.6002, 0.2400]
  state 2 (occ 0.2477): [0.0101, 0.9757, 0.0142]   true token 2: [0.0000, 1.0000, 0.0000]
  state 0 (occ 0.1546): [0.3819, 0.0365, 0.5816]   true token 0: [0.4014, 0.0000, 0.5986]
```

### 6.3 Why the duplicates are always of token 1

At K=8 and K=16, β=0, **every** duplicate is a copy of token 1's row and there is
never a duplicate of token 0 or token 2. Token 1 is the only backward conditional
with no structural zero. Splitting a zero-free row costs nothing; copies of a row
containing a hard zero would each have to reproduce that zero, which the softmax
resists. Same reason the merge penalties are asymmetric: token 2's true entropy
is 0, so any merge takes it from 0 to 0.375 bits.

### 6.4 The two knobs do opposite jobs in the two arms

| | failure mode | what `+β·H` does |
|---|---|---|
| forward | collapse (entropy too low) | **hurts** — deepens the collapsed basin, delays escape |
| backward, K ≥ 4 | over-splitting (entropy too high) | **helps** — removes free duplicates |

This is why a single β cannot be optimal for both arms, and why `+β·H` — which is
wrong for the anti-collapse purpose the docstring still describes — is *right*
for the minimum-complexity purpose.

### 6.5 The consequence for ΔCE

```
   K |  beta=0    beta=1/N        true value = 0 (H_inf is time-reversal invariant)
   2 | +0.2511    +0.2482
   3 | +0.2558    +0.2599
   4 | +0.0632    -0.1345 (fw not converged)
   8 | +0.0600    +0.0754
  16 | +0.0541    +0.0601
```

Measured ΔCE spans **−0.13 to +0.26 bits purely as a function of (K, β)**, against
a real effect of 0.001–0.003 bits in the main study — a setting-dependent bias
20× to 250× the signal, with a consistent sign. The baseline model gives
[−0.0011, +0.0024] on the same processes.

**Conclusion: this model is a good state-recovery instrument and is currently not
a ΔCE instrument.** Those are different uses and only the first is supported.

### 6.6 The β threshold, refined — and it is not monotonic

80 runs over β ∈ {0, 1e-4, 1/N=3.16e-4, 5e-4, 7e-4, 1e-3, 2e-3, 1e-2}.

**Coin forward** collapses between **5e-4 and 7e-4** at K=2 and K=8, and between
1e-4 and 5e-4 at K=4. So the usable band is roughly `β ≤ 5e-4`, i.e. `1/N` sits
comfortably inside it at N=3168 but with less than a factor of 2 to spare.

**Coin backward is not monotonic in β.** At K=4: β=1/N gives 3 states (ARI 0.980),
β=5e-4 and 7e-4 give 1 state, and β=1e-3 gives 3 states again (ARI 0.814). This is
basin sensitivity, not a threshold — **a single-seed β sweep cannot be read as a
tuning curve.** Any β recommendation must come from repeats.

The most robust backward cell is K=8, which holds k_found=3 across the whole range
1e-4 → 1e-3 (ARI 0.54–0.74) but never matches K=4's best (ARI 0.980).

### 6.7 The flower transfers — and exposes rare-state blindness

Flower n=2, m=3, Dirichlet α=0.5 seed 42. V=5, H∞=0.9340, C⁺=1.5000, C⁻=1.6240.

**Forward, K=8, β=0 is an exact recovery:**

```
k_found = 3 = true K,  ARI = 1.000,  H_occ = 1.5050  vs  C+ = 1.5000   (+0.0050)
```

But **K=4 fails at every β** — stuck at 2 states, ARI 0.745, H_occ 0.9999, for all
eight β values identically. True K is 3, so K=4 is true_K+1 and it is *not enough*.
The "+1 slack" rule from the coin does **not** generalise; flower forward needed
K=8.

**Backward finds only 3 of its 4 states, at every K and every β.** And the miss is
exactly identifiable:

```
theory occupancy  [0.5000, 0.3038, 0.1574, 0.0388]   C- = 1.6240
merge two smallest [0.5000, 0.3038, 0.1962]          H  = 1.4832   <- prediction
measured H_occ over the 14 non-collapsed runs: mean 1.4784, range [1.4614, 1.5000]
```

The model merges the two lowest-occupancy states. The missed state carries
**3.9%** of positions; for comparison the coin backward's smallest state carries
16% and *is* found (measured occupancy 0.1546 against a theoretical 0.16).

**This is a systematic downward bias in H_occ as an estimator of C.** My first
reading of it — that the cause is *rare-state blindness*, an occupancy floor below
which states are missed — is **wrong**, and §6.10 records the experiment that
disproved it. The bias is real; the mechanism is a state-count ceiling.

### 6.8 β does not scale with N — it scales with 1/lr

`β = 1/N` was used throughout as a *fixed constant* `1/3168`, computed from the
nominal batch. Two things are wrong with tying β to N.

**(a) N is not even constant across steps.** 300 sequences at batch 32 gives nine
full batches and a remainder of 12:

```
  steps 0-8: B=32  N = B*T = 3168   1/N = 0.000316
  step  9  : B=12  N = B*T = 1188   1/N = 0.000842   <- PAST the 5e-4..7e-4 cliff
```

So a literal per-step `1/N` spends one step in ten inside the collapse regime. The
per-batch control run gave the same answers, but that is one seed and should not
be assumed to hold.

**(b) There is nothing for 1/N to normalise.** The CE is `F.cross_entropy(...,
reduction="mean")` — bits per token, intensive. `H(p_bar)` is built from a mean
over positions — bits, intensive. Neither scales with N, so dividing by N is not
a units correction; it just picks a small number. Statistically it picks one that
is *too* small: the Miller–Madow bias of `H(p_bar)` is `(K−1)/(2N ln2)` = 0.0016
bits at K=8, N=3168, while the penalty at β=1/N is 0.00032 bits — five times below
the estimation bias of the quantity being penalised.

**What the threshold actually scales with.** The energetic balance predicts a
threshold 670× too high: opening the second forward state gains
`1.3591 − 0.9709 = 0.3882` bits and costs `H = 0.9710`, so collapse should win at
β > 0.3998. Measured, it wins at 6e-4. The barrier crossing, not the energetics,
is binding — and barrier crossings depend on the optimiser:

```
      lr      beta        CE  states     ARI
   0.001   0.0e+00    0.9687       2   1.000
   0.001   1.0e-04    0.9687       2   1.000
   0.001   3.0e-04    0.9706       2   1.000
   0.001   5.0e-04       --        2   1.000
   0.001   7.0e-04       --        1   0.000     <- cliff in (5e-4, 7e-4)
   0.003   0.0e+00    0.9682       2   1.000     <- lr=3e-3 is NOT broken on its own
   0.003   1.0e-04    0.9683       2   1.000
   0.003   3.0e-04    1.3492       1   0.000     <- cliff in (1e-4, 3e-4)
```

Tripling lr moves the cliff down ~3×, and `β_cliff · lr ≈ 6e-7` at both points.
On two points and one seed that is suggestive, not established — but the direction
is unambiguous and the β=0 control rules out "lr=3e-3 is simply broken".

**Guidance:** β is a property of the *training configuration*, not of the data
shape. Do not write it as a formula in N. Calibrate it against the collapse cliff
at the lr and step budget actually used, record the absolute value in the run
config, and re-calibrate whenever lr or the step budget changes.

### 6.9 Structural zeros predict duplication exactly

The runaway duplication in §6.2–6.3 occurs in **one** of the four (process, arm)
combinations, and the structural-zero rule accounts for all four:

| process | arm | any zero-free conditional row? | k_found at K=8, β=0 | true K |
|---|---|---|---|---|
| coin | forward | no (both rows have a zero) | 2 | 2 |
| coin | backward | **yes** — token 1 is [0.160, 0.600, 0.240] | **8** | 3 |
| flower | forward | no | 3 | 3 |
| flower | backward | no | 3 | 4 |

Only the arm with a zero-free row duplicates, and it duplicates *only that row*
(five copies at K=8, seven at K=16 — never a copy of a row containing a zero).
Copies of a row with a hard zero would each have to reproduce that zero, which the
softmax resists; copies of a zero-free row cost exactly nothing.

Practical consequence: **β>0 is needed only when the process has a zero-free
conditional row.** For the flower — both arms, every β tested — β=0 already gives
the right state count, and β buys nothing.

### 6.10 The real limit is a state-count ceiling of about three

The rare-state hypothesis of §6.7 predicted that flatter dice — raising the
smallest occupancy — would recover more states. It was tested on flower (3,5),
the cell the dice experiment actually uses, at α=0.2 (smallest backward occupancy
0.0232) against α=2.0 (smallest 0.0665, 2.9× larger). K ∈ {4,6,8,12,16}, β ∈
{0, 1/N}, both arms, 28 runs.

**The flatter dice did not help.** Process B's best backward run reached ARI
0.765 against Process A's 0.841 — if anything worse.

And the decisive case is the **forward** arm at (3,5), whose occupancy is
`[0.5, 0.1667, 0.1667, 0.1667]`: its smallest state carries 16.7% of positions,
so there is **no rare state at all**, and it still never finds its fourth state at
any K or β:

```
  A forward K=4  b=1/N -> k=3  ARI 0.862   H_occ 1.4849
  A forward K=6  b=1/N -> k=3  ARI 0.876   H_occ 1.4669
  A forward K=8  b=1/N -> k=2  ARI 0.660   H_occ 0.9999
  A forward K=4  b=0   -> k=3  ARI 0.866   H_occ 1.4800
  A forward K>=6 b=0   -> k=2  ARI 0.660   H_occ 0.9999
  merging two of the three die-states predicts H = 1.4591   <- matches the k=3 runs
```

Across everything tested:

```
           process       arm  true K  best distinct found  complete?
              coin   forward       2                    2      yes
              coin  backward       3                    3      yes
        flower 2,3   forward       3                    3      yes
        flower 2,3  backward       4                    3       NO
  flower 3,5 a=0.2   forward       4                    3       NO
  flower 3,5 a=0.2  backward       6                    4       NO
  flower 3,5 a=2.0   forward       4                    3       NO
  flower 3,5 a=2.0  backward       6                    3       NO
```

**Every process with ≤3 true states is recovered exactly; every process with ≥4 is
not.** No configuration anywhere in this study has recovered more than four
distinct causal states, and only three reliably. (Coin backward K=8 β=0 "found" 8
states, but those were duplicates at ARI 0.353 — see §6.9.)

That single ceiling explains the (2,3) backward miss without any appeal to
occupancy, and it explains a case the occupancy story cannot. At d_model=20,
2 layers and 3000 steps, the instrument saturates at about three states.

**The ceiling turned out to be the STEP BUDGET** — §6.11 lifts it by training
longer, and finds that lifting it does not help.

### 6.11 Longer training raises the state count and *lowers* the recovery

Four probes trained to **30,000 steps**, 10× the standard budget, checkpointed
throughout. β=0, d_model=20, lr 1e-3, everything else as §4.

```
  k_found         trueK    300   1000   3000   6000  10000  15000  20000  25000  30000
  35 fw K=4           4      2      2      3      3      4      4      4      4      4
  35 fw K=6           4      2      2      4      4      5      5      5      6      6
  35 bw K=6           6      2      2      4      6      6      6      6      6      6
  23 bw K=6           4      2      2      2      4      5      5      5      6      6

  ARI                       300   1000   3000   6000  10000  15000  20000  25000  30000
  35 fw K=4                0.660  0.660  0.869  0.863  0.862  0.538  0.531  0.538  0.531
  35 fw K=6                0.660  0.660  0.661  0.785  0.583  0.515  0.505  0.528  0.519
  35 bw K=6                0.621  0.621  0.733  0.508  0.493  0.489  0.495  0.481  0.490
  23 bw K=6                0.730  0.730  0.730  0.840  0.712  0.732  0.733  0.672  0.652
```

**The count is not bounded at three.** With enough steps every probe reaches its
true K and then *overshoots* it — 35 fw K=6 and 23 bw K=6 both end at 6 states for
processes with 4. So §6.10's "ceiling of about three" was a budget artefact, and
the rare-state reading of §6.7 is dead for a second reason: 23 bw *does* find its
fourth state — the 3.9% one — by 6,000 steps.

**But ARI peaks early and then falls, in all four probes.** The best recovery in
the entire study is 0.869 at 3,000 steps — the original budget, reached by
accident. Every probe is worse at 30,000 steps than at its peak.

What the model does with the extra steps is split states on context rather than
isolate causal states:

```
                       at the ARI peak          at 30,000 steps
  35 fw K=4     k=3  ARI 0.869  H(s|tok) 0.06   k=4  ARI 0.531  H(s|tok) 0.63
  35 fw K=6     k=4  ARI 0.785  H(s|tok) 0.26   k=6  ARI 0.519  H(s|tok) 0.73
  35 bw K=6     k=4  ARI 0.733  H(s|tok) 0.26   k=6  ARI 0.490  H(s|tok) 0.92
  23 bw K=6     k=4  ARI 0.840  H(s|tok) 0.28   k=6  ARI 0.652  H(s|tok) 0.78
```

`H_occ` overshoots C correspondingly (1.948 vs 1.7925; 2.109 vs 1.7925; 2.352 vs
2.0387; 2.088 vs 1.6240), so **longer training turns an under-estimate of C into an
over-estimate** and passes through the right answer on the way.

**CE gives no warning.** It falls monotonically throughout and ends *below* H∞ in
every probe (−0.041, −0.060, −0.078, −0.031) — the memorisation effect of §6.5.
A CE-based stopping rule would stop at exactly the worst point for state recovery.

**Where the peak sits.** Three of four probes peak where CE − H∞ ≈ **+0.10**
(+0.103, +0.100, +0.098) and the fourth at +0.041. That is a usable early-stopping
target when H∞ is known in closed form, which it is for every process in this
repo. `H(state|token)` rising off zero is the theory-free version of the same
signal and needs no ground truth — it is 0.06–0.28 at the peaks and 0.63–0.92 at
the end.

**Consequences.**

* **State recovery requires early stopping, and the criterion is not the loss.**
  Train to CE ≈ H∞ + 0.1, or stop when `H(state|token)` starts climbing. Training
  to convergence is actively wrong for this purpose.
* The (3,5) cell is reachable after all — but only in a window, and even at its
  best it reaches ARI 0.87 (forward) and 0.73 (backward), well short of the 1.000
  the coin and (2,3) forward achieve.
* **`H_occ` is a biased C estimator in both directions** depending on where you
  stop. It is trustworthy only for processes where the peak ARI is ~1.0, i.e. ≤3
  true states at this capacity.
* Capacity (d_model, n_layers) is now the untested axis. Everything above is at
  d_model=20, 2 layers. The open question is whether more capacity widens the
  window or raises the peak ARI — §9.

---

## 7. Figures and tables worth producing

Ordered by how much each earns its place. Every one is computable from the
quantities in §3 with no extra training.

### F1 — The state-count trace (the single most diagnostic figure)

`k_found` against epoch, one line per K, panelled forward | backward, coloured by
β. Log-x helps because everything happens in the first 60 epochs.

Shows in one picture: the universal collapse to 1, the escape, and whether the
curve settles or keeps accreting. **Annotate the true K as a horizontal rule** —
the entire result is whether the curve lands on that line.

### F2 — H_occ vs C, the calibration plot

Scatter of measured `H_occ` (y) against theoretical C (x), one point per
(process, arm, K, β), with the y=x line. Marker shape = arm, colour = β, size = K.

This is the plot that says whether the instrument is *calibrated*. Forward points
pile on y=x; backward β=0 points sit far above it (duplicates); backward β=1/N
points return to it. Extending across flower (n,m) cells populates the x-axis
properly and turns this into a real calibration curve.

### F3 — CE against the partition bounds

CE trajectory per run with **horizontal rules at the 1-, 2- and 3-state bounds**
from §2. Converts every plateau into a statement about which partition the model
found. This is what revealed that backward K=2,3 sit *worse* than the best 2-state
model rather than at it.

### F4 — Emission table heatmap

K×V heatmap per run beside the true conditional rows, states sorted by occupancy.
Duplicate rows are instantly visible as repeated stripes — this is how the five
copies of token 1's row were found. Far more legible than the numeric dump.

### F5 — β threshold curve

x = β on a log axis (0 plotted as a decade below the smallest non-zero value),
y = ARI on the left axis and `k_found` on the right; one panel per arm, one line
per K. The collapse threshold appears as a cliff. **This is the figure that
justifies whatever β the study ends up using.**

### F6 — H(state|token) bar chart

One bar per configuration, with a rule at 0. A single glance separates "wrong
grouping" (bar at 0, ARI < 1) from "not a function of the token at all" (bar > 0).

### T1 — Master table

Exactly the columns in §6.2: arm, K, β | CE, CE−H∞, bound, CE−bound | k_found,
k_true | H_occ, C, H_occ−C | ARI, purity, H(state|token). Every row of every
experiment. This is the table to keep.

### T2 — ΔCE table

CE_fw, CE_bw, ΔCE at matched (K, β), with the baseline model's range printed
underneath as the reference. The point of the table is the *spread*, so sort by
|ΔCE| and show min/max explicitly.

### T3 — Occupancy vs theory

Measured occupancy vector beside the theoretical stationary distribution, sorted
descending. Reads as a direct ε-machine comparison.

### What NOT to plot

* Purity without ARI beside it (§3).
* CE alone against β — it hides collapse, because a collapsed model has a
  perfectly respectable-looking CE equal to the marginal entropy (1.359 here).
* Any run whose states-used trace is still moving at the last epoch, unless it is
  labelled as such.

---

## 8. What will bite

1. **`n_states=None` falls back to `token_size`.** For the coin backward that is
   3 = the true count, which is exactly the value that fails. The default is the
   trap.
2. **Two of twenty runs were unconverged at 3000 steps** and are indistinguishable
   from results in a summary table. Always carry the "still moving?" column.
3. **~1800–3000 gradient steps** to converge, against roughly 130 in QUICK. Any
   integration with the existing harness needs the step budget to grow by more
   than an order of magnitude, which changes the sweep compute arithmetic
   substantially.
4. **β is not transferable across batch or chunk size.** β=1/N works because
   N=3168 is large; the measured cliff is at N ≈ 1000–3000, so a smaller batch or
   chunk crosses it silently. If β is tied to N at all, record the effective value
   in the run config.
5. **`target_occupancy` is currently dead code** — the KL branch was removed but
   the buffer, its validation and the docstring remain.
6. **State labels are arbitrary.** Never compare model state k to theory state k;
   always score with ARI or a sorted comparison.

---

## 9. Open questions

* **Can the backward arm reach H∞?** Its best is +0.053 while forward reaches
  −0.002. Until that closes, ΔCE from this model is dominated by optimisation
  difficulty. Candidates: longer training, τ annealing, a two-term objective
  (diversity floor + complexity penalty), or a better initialisation of
  `state_head`.
* **What is the occupancy floor?** §6.7 shows a state at 3.9% is missed and one at
  16% is found. The threshold between them is the single most important unknown,
  because it sets the bias of `H_occ` as a C estimator and that bias moves with
  process skew. Measure it directly: flower cells with a deliberately tuned
  smallest-outcome mass, sweeping it across 1%–20%.
* **The K-slack rule does not generalise** (§6.7): coin backward needs true_K+1,
  flower forward needs more than true_K+1 and works at 2×true_K+2. There is no
  known rule yet; until there is, K must be swept per process rather than set from
  the theoretical count.
* **A single-seed β sweep is not a tuning curve** (§6.6). Repeats are required
  before any β is recommended — the non-monotonicity is basin noise.
* **Can the backward coin arm reach H∞?** Its best is +0.053 while forward reaches
  −0.002. Candidates: longer training, τ annealing, a two-term objective
  (diversity floor + complexity penalty), better `state_head` initialisation.
* **Is the structural-zero rule causal?** §6.8 accounts for all four arms tested,
  but four is a small sample. A synthetic process with a tunable number of
  zero-containing rows would settle it, and it is cheap to build.
* **Does `H_occ` stay calibrated across the flower grid**, where C⁻ varies
  continuously with the dice? That is the measurement that would make this model
  useful to the main study — a direct C⁻ − C⁺ estimator, bypassing ΔCE entirely.
  §6.7 says this cannot be attempted until the occupancy floor is characterised.
