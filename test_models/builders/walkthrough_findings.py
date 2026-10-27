"""The demonstration sections of walkthrough.ipynb.  Every number in the 'after' texts was read from the cached reports
of the imported runs (runs/imported/test_table.csv, machine_table.csv, first_below.csv, test_table_best.csv)."""

PERF = '''r = runs_of("{model}", "{label}")
display(performance(r))
display(A.first_below(r).set_index("seed")[["final_update", "best", "best_update", "below 0.04", "below 0.02", "below 0.01"]])
fig = A.plot_machines(list(r.values())); plt.show()'''

KEY_TEXT = '''
* **Performance** (first table): each predictor's test excess CE over the 20 future tokens and over 200 (`long_excess`),
  the true states that get a learned state of their own, the size of each step rule's machine, and D (hybrids with a
  GRU).
* **Convergence** (second table): the best validation value of the main predictor and when it first fell below 0.04,
  0.02 and 0.01 bits (empty: never).
* **Properties** (figure): learning curves; the machine each seed ends with; its transitions on the true states; and the
  difference from the true transitions (hatched rows: true states with no learned state of their own).
'''

FINDINGS = [
    dict(markdown='''
### 6.1 Model 1: the GRU's predictions only (`model_1.py`), and the original setup

**Loss:** the cross-entropy of the GRU's predictions of the 20 future tokens. The GRU starts from the Transformer's state
after the history. The Transformer is trained only through that starting state, so its own predictions at other
positions are not trained; they are measured anyway.
''' + KEY_TEXT, code=PERF.format(model="1", label="section16"), after='''
**What the runs show**
* **With the 16.1 changes, only seed 0's GRU learns most of the counts.** It ends with 12 long-run states covering 12
  true states, at a test excess of 0.0175 bits. At its best validation update (34,000) it was at 0.0068.
* **Seeds 1 and 2 end with a single long-run state**, at 0.0453 and 0.0529, near the two reference levels.
* **The Transformer's own predictions are not trained by this loss:** 0.0540–0.1719 bits.

The next cell shows the same model in the original setup (no LayerNorm on the Transformer's output, no weight decay, no
gradient clipping).
'''),
    dict(markdown='''
#### Model 1 in the original setup
''', code='''o = runs_of("1", "original_setup")
display(performance(o))
fig = A.plot_machines(list(o.values())); plt.show()''', after='''
**What the runs show:** all three GRUs end with one long-run state, at 0.0417–0.0549 bits. The Transformer's own
predictions are 0.0490–0.3247 bits.
'''),
    dict(markdown='''
### 6.2 Model 2: + the Transformer's own next-token predictions (`model_2.py`)

**Loss:** the GRU's future-token CE + the CE of the Transformer's own next-token predictions at every position from 19
on. The Transformer's choice of state is now trained at every position, not only through the GRU at the start.
''' + KEY_TEXT, code=PERF.format(model="2", label="section16"), after='''
**What the runs show**
* **Seed 0's GRU ends with the true machine.**
  * 15 long-run states cover all 15 true states, and its transitions differ from the true ones by 0.0008 on average.
  * Test excess 0.0013 bits (0.0005 at its best validation update, 62,000), and 0.0005 over 200 tokens.
  * It first fell below 0.02 bits at update 36,500.
* **Seeds 2 and 1 do not.** Seed 2's GRU has 11 long-run states (0.0184; 0.0043 at its best, update 54,000). Seed 1's has
  one (0.0500).
* **The Transformer's own predictions are good on every seed:** 0.0005, 0.0024 and 0.0076 bits, giving 15, 7 and 13 true
  states their own state.
'''),
    dict(markdown='''
### 6.3 Model 3: + a next-state network following the Transformer (`model_3.py`)

**Loss:** model 2's + weight × (entropy of a next-state network's output + its CE against the state the Transformer
actually picks next), with weight 0.001. The network reads a state vector and the next token, and gives probabilities for
the next state. The idea: if the Transformer's states follow a fixed rule, the network can learn it and step through the
future like the GRU.
''' + KEY_TEXT, code=PERF.format(model="3", label="section16") + '''
fig = A.plot_extra_terms(list(r.values())); plt.show()''', after='''
**What the runs show**
* **The GRU ends with 1, 11 and 13 long-run states**, at 0.0418, 0.0277 and 0.0175 bits. Seeds 1 and 2 first fell below
  0.02 bits at updates 61,000 and 43,000. No seed reached 0.01.
* **The next-state network stays at the "remembers nothing" level:** 0.0561, 0.0578 and 0.0585 bits.
* **The Transformer's own predictions are good** (0.0026–0.0045 bits), but it gives only 4–5 true states their own state:
  the early counts, whose next-token probabilities are close, share one state.
'''),
    dict(markdown='''
### 6.4 Model 4: model 3 without the GRU (`model_4.py`)

**Loss:** the Transformer's own next-token CE + weight × (the next-state terms of model 3), with weight 0.001. Can the
next-state network replace the GRU?
''' + KEY_TEXT, code=PERF.format(model="4", label="section16") + '''
fig = A.plot_extra_terms(list(r.values())); plt.show()''', after='''
**What the runs show**
* **The Transformer's predictions** are at 0.0025, 0.0019 and 0.0024 bits. They first fell below 0.01 bits at updates
  2,000, 1,000 and 11,000.
* **Its machine, read off its states, has 5, 6 and 5 states**, covering the same numbers of true states: the early
  counts share one state.
* **The next-state network does not replace the GRU:** 0.0552, 0.0591 and 0.0582 bits, the "remembers nothing" level.
'''),
    dict(markdown='''
### 6.5 Model 5a: one- and two-step predictions + the spread of the inferred next state (`model_5a.py`)

**Loss:** the GRU's future-token CE + the Transformer's one- and two-step CE + weight × the entropy of the next state
worked out from those predictions with Bayes' rule, with weight 0.01. The three steps are in `model_5a.py`. The next cell
also prints the value this penalty would have on the true machine, and its floor, the crypticity.
''' + KEY_TEXT, code=PERF.format(model="5a", label="section16") + '''
fig = A.plot_extra_terms(list(r.values())); plt.show()
display(pd.Series(A.inferred_entropy_on_true_machine(settings, 1)).round(4))''', after='''
**What the runs show**
* **The GRU ends with one long-run state on seeds 0 and 1** (0.0418, 0.0446 bits) and 3 on seed 2 (0.0254). Seed 2 was
  at 0.0042 at its best validation update, 54,000.
* **The Transformer's own predictions are good:** 0.0002, 0.0075 and 0.0004 bits, giving 11, 11 and 13 true states their
  own state.
* **The penalty favours fewer states.** On the true machine it is 3.6645 bits, while a single state scores 0. No number
  of predicted tokens can bring the true machine below the crypticity, 3.2363 bits.
'''),
    dict(markdown='''
### 6.6 Model 5b: model 5a without the GRU (`model_5b.py`)

**Loss:** the Transformer's one- and two-step CE + weight × the entropy of the inferred next state, with weight 0.1.
''' + KEY_TEXT, code=PERF.format(model="5b", label="section16") + '''
fig = A.plot_extra_terms(list(r.values())); plt.show()''', after='''
**What the runs show**
* **The Transformer's predictions** are at 0.0009, 0.0014 and 0.0014 bits. They first fell below 0.01 bits at updates
  1,000–1,500.
* **Its machine has 10, 8 and 8 states**, covering the same numbers of true states.
'''),
    dict(markdown='''
### 6.7 Model A: model 3 with the Transformer's next state as a fixed target (`model_A.py`)

**Loss:** model 3's, with weight 0.01. The state the Transformer picks next is a fixed target: no gradient of the
next-state terms reaches the Transformer through it. Holding the target fixed is the known remedy against a model
predicting its own next state collapsing to a constant state (Tang et al. 2023; Chen & He 2021).
''' + KEY_TEXT, code=PERF.format(model="A", label="section16") + '''
fig = A.plot_extra_terms(list(r.values())); plt.show()''', after='''
**What the runs show**
* **The GRU ends with 9, 1 and 13 long-run states**, at 0.0395, 0.0284 and 0.0163 bits. At their best validation updates
  they were at 0.0110, 0.0051 and 0.0036. They first fell below 0.02 bits at updates 32,000, 18,500 and 23,000.
* **On seed 1 the next-state network predicts well** (0.0067 bits), and the Transformer gives all 15 true states their
  own state, but the GRU ends with one long-run state.
'''),
    dict(markdown='''
### 6.8 Model B: the Transformer's starting state trained to copy the GRU's own state (`model_B.py`)

**Loss:** the GRU's future-token CE + weight × the CE of the Transformer's scores at the last history position against the
copy target, with weight 3.
* **The copy target** is the state the GRU itself reaches after reading the 40 history tokens from row 0 of the table.
* The copy term trains only the Transformer.

The extra cell shows, for seed 2, what the copy target looks like at the starting weights and at the end:
* how many different values it takes on the test windows;
* how many true states get their own value;
* how often the Transformer's starting state equals it.
''' + KEY_TEXT, code=PERF.format(model="B", label="section16") + '''
display(A.copy_target_stats(runs[("B", "section16", 2)]))
fig = A.plot_extra_terms(list(r.values())); plt.show()''', after='''
**What the runs show**
* **The GRU ends with 9, 13 and 14 long-run states**, covering the same numbers of true states, at 0.0355, 0.0117 and
  0.0070 bits.
* **Seed 2 reached 0.0004 bits** at its best validation update (95,000). Seeds 1 and 2 first fell below 0.02 bits at
  updates 58,000 and 45,000.
* **The Transformer's own predictions are not trained** except at the last history position: 0.0572–0.0679 bits.
'''),
    dict(markdown='''
### 6.9 Model C: model 5b with the next state worked out from two predicted tokens (`model_C.py`)

**Loss:** the Transformer's one-, two- and three-step CE + weight × the entropy of the inferred next state, with weight
0.01. Here each candidate state is identified by its predicted next two tokens. `model_C.py` lists the three ways it
differs from 5b.
''' + KEY_TEXT, code=PERF.format(model="C", label="section16") + '''
display(pd.Series(A.inferred_entropy_on_true_machine(settings, 2)).round(4))''', after='''
**What the runs show**
* **The Transformer's predictions** are at 0.0019, 0.0070 and 0.0002 bits. They first fell below 0.01 bits at updates
  1,000–2,000.
* **Its machine has 15, 13 and 14 states**, covering 13, 13 and 14 true states. Its transitions differ from the true
  ones by 0.0322, 0.0182 and 0.0132 on average.
* **On the true machine the penalty is 3.6571 bits** from two tokens, against 3.6645 from one.
'''),
    dict(markdown='''
### 6.10 Model 1 + L2: section 13's run (`model_1_L2.py`)

**Loss:** the GRU's future-token CE + weight × D, the distance between the Transformer's and the GRU's state
probabilities after each future token, with weight 0.03.

There is one run: section 13's, with 300 history and 40 future tokens, the original setup, and 50,000 updates. Section
13 saved no best-validation parameters, so the convergence table has none.
''' + KEY_TEXT, code=PERF.format(model="1_L2", label="section13") + '''
fig = A.plot_extra_terms(list(r.values())); plt.show()''', after='''
**What the run shows:** the GRU and the Transformer both end at 0.0511 bits with D = 0. Both use the same 2 states,
which follow the last token only (2 long-run states covering 2 true states).
'''),
    dict(markdown='''
### 6.11 Model 2 + L2: weights 0.01, 0.03 and 0.1 (`model_2_L2.py`)

**Loss:** model 2's + weight × D over the GRU's 19 steps (section 17). The next cells show four things:
* the test results of all nine runs;
* the learning curves and D over training;
* the machines at weight 0.03;
* for seed 0 of each weight, and of model 2 without the term, how often the GRU picks the same state as the Transformer,
  and how certain each part is (`A.state_agreement`).
''', code='''r = runs_of("2_L2")
display(performance(r).sort_values(["weight", "seed"]))
fig = A.plot_learning_curves(list(r.values())); plt.show()
fig = A.plot_extra_terms(list(r.values())); plt.show()
fig = A.plot_machines(list(runs_of("2_L2", "w0.03").values())); plt.show()
display(pd.DataFrame({f"{m} {l} s{s}": A.state_agreement(runs[(m, l, s)])
                      for m, l, s in [("2", "section16", 0), ("2_L2", "w0.01", 0), ("2_L2", "w0.03", 0),
                                      ("2_L2", "w0.1", 0)]}).T.round(3))''', after='''
**What the runs show**
* **At weight 0.01** the GRU ends with one long-run state on seeds 0 and 1 (0.0428, 0.0446 bits) and 13 on seed 2
  (0.0165).
* **At weights 0.03 and 0.1** every GRU ends with one long-run state, at 0.0431–0.0481 bits. Model 2 without the term
  ended with 15, 1 and 11.
* **The Transformer's own predictions get worse as the weight grows:** 0.0050–0.0074 bits at 0.01, 0.0089–0.0094 at 0.03
  and 0.0158–0.0172 at 0.1, against 0.0005–0.0076 without the term.
* **Agreement is not learning the count.** With the term, the GRU picks the same state as the Transformer at most future
  positions (last table; both parts are nearly certain of their choice), so D is small. But the GRU's own machine is a
  single state.
'''),
    dict(markdown='''
### 6.12 The feedback GRU alone (`model_feedback_GRU.py`)

**The model:** at every position a GRU computes a proposal from (the previous state's vector, the token). Its scoring
layer picks a state, and that state's vector is fed back. It starts blank at every window and is trained on its own
next-token predictions. There is no Transformer, so its state after the 40 history tokens comes from its own reading.
''' + KEY_TEXT, code=PERF.format(model="feedback_GRU", label="section17") + '''
fig = A.plot_states_over_training(list(r.values())); plt.show()''', after='''
**What the runs show:** it learns the true machine on every seed.
* Its machine has 15, 16 and 15 long-run states, covering all 15 true states. Its transitions differ from the true ones
  by 0.0004–0.0006 on average.
* Test excess 0.0001, 0.0001 and 0.0002 bits, and 0.0001–0.0002 over 200 tokens.
* It first fell below 0.01 bits between updates 19,500 and 26,500.
'''),
    dict(markdown='''
### 6.13 The discrete Transformer alone (`model_discrete_Transformer.py`)

**The model:** a Transformer that picks a state at every position and predicts the next token from it. It is trained on
its own next-token predictions. Its machine is read off the states it picks, so it need not be deterministic.
''' + KEY_TEXT, code=PERF.format(model="discrete_Transformer", label="section17"), after='''
**What the runs show**
* **Its predictions** are at 0.0002, 0.0064 and 0.0003 bits. They first fell below 0.01 bits at updates 2,500–3,000.
* **Its machine has 13, 13 and 11 states**, covering 13, 11 and 11 true states. Its transitions differ from the true
  ones by 0.0330, 0.0595 and 0.0468 on average.
'''),
    dict(markdown='''
### 6.14 Across the models
''', code='''everything = {k: f for k, f in runs.items()}
fig = A.plot_machine_summary(A.machine_table(everything), levels); plt.show()''', after='''
* **Only two runs end with a machine covering all 15 true states at under 0.002 bits:**
  * model 2, seed 0 (the hybrid's GRU);
  * the feedback GRU alone, on all three seeds.
* **The hybrid's GRU learns the counts on some seeds only.** Ending machines with at least 11 long-run states:
  * model 1: seed 0;
  * model 2: seeds 0 and 2;
  * model 3: seeds 1 and 2;
  * model A: seed 2;
  * model B: seeds 1 and 2;
  * model 2 + L2 at weight 0.01: seed 2.
* **The Transformer-only models predict well but merge the early counts.**
  * Test excess: 0.0002–0.0070 bits.
  * True states covered: model 4, 5–6; model 5b, 8–10; model C, 13–14; the discrete Transformer, 11–13.
* **The terms that pull the two parts together** (D in models 1 + L2 and 2 + L2) make the GRU follow the Transformer.
  They leave the GRU's own machine at a single state, except at the smallest weight on one seed.
'''),
]
