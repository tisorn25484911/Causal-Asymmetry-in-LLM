"""Section 10 of test_models/walkthrough.ipynb: the de Bruijn hybrid (model_de_bruijn.py, model_de_bruijn_bic.py) on
renewal-5, 10, 15, 30, 60 and the flower processes (3, 5) and (8, 14), each read forward and backward.  CELLS is a list
of ("md" | "code", text); FINDINGS (markdown) closes the section.  It uses the shared code of sections 7-9."""

CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
## 10. The de Bruijn hybrid (`model_de_bruijn.py`)

This section tests a new model, built from the plan `DEBRUIJN_HYBRID_IMPLEMENTATION_PLAN.md` (its sections 1–9).

**The idea.** If a process's next token depends only on its last R tokens (its Markov order is R), those R tokens, the
*node*, tell everything the past can tell about the future. The model therefore reads its state from the node instead
of learning a state from the whole history. It also does not start from random weights: before training it builds a
machine by counting, and training starts there.

**Before training, by counting** (on 40,000 fresh windows of the process, a stream of their own):
1. Count how often each node is followed by each token.
2. Merge the nodes into states, round by round. Two groups merge when (i) their next-token frequencies agree within a
   statistical tolerance (a Hoeffding test at level 0.001) and (ii) for every token both have been followed by, the
   nodes they lead to are in the same group. Only nodes counted at least 100 times are merged; a rarer node takes the
   state of the frequent node with the longest common ending (*suffix back-off*). For a process of Markov order R this
   merging stops at the causal states (the plan's Theorems B and C). The result is the *quotient*.

**The network** has three parts:
* **the reader**: a table with one row of state scores per node (no Transformer). Its scores depend on the window only
  through the node;
* **the shared state table E and output layer W** of every model in this folder (`causal_state_matrix.py`);
* **the transition head G**: a table of next-state scores for every (state, token), the model's step rule.

All three start at the quotient: each node's row scores its state 12 and every other state 0; E's rows give the counted
next-token probabilities (at least 0.001 each); G scores the counted next state of every (state, token) 12.

**The loss**, at every position that has a node (all in bits):
L = L_pred + β L_use + λ (L_G + κ L_E), where
* L_pred is the CE of the reader's prediction of the next token;
* L_use is the entropy of the reader's mean state probabilities (β = 1/9600);
* L_G is the CE of G's next state against the reader's actual next state;
* L_E pulls the reader's next-state probabilities toward G's prediction (κ = 0.25);
* λ = 1 (the run's weight).

**Predictors.**
* *node table*: the reader's state at every position.
* *next-state network from the handoff*: the reader picks the state after the H history tokens, then G steps from
  state to state on each real future token. This is the plan's evaluation and the main predictor, as for the hybrids.

**Two variants of R** (two model files that differ in one line):
* `model_de_bruijn.py`: R is the process's Markov order (renewal-N: N − 1; the flower: 1). The model is given the
  order, as the plan's Experiment C does for renewal-15.
* `model_de_bruijn_bic.py`: R is estimated from the counting windows by the plan's information criterion (Bayesian
  information criterion, Csiszár & Shields 2000). It uses no knowledge of the process, but its penalty grows like
  2^R, so it under-estimates long orders.

**An honest comparison.** The training is exactly that of every other model here: 50,000 updates of 128 fresh windows,
AdamW with learning rate 0.001 and weight decay 0.0005, gradient clipping at 1, temperature 2.5 → 0.5, random noise in
the state choice, and the same validation and test windows. On renewal-15 and renewal-60 the window sizes and table rows
are those of sections 16 and 7. The plan fine-tuned differently (Adam, learning rate 0.0003, temperature 0.5 → 0.1,
3,000 updates). The de Bruijn model also reads 40,000 counting windows before training, 0.6% of the 6.4 million
training windows.

**Two changes from the plan**, both forced by size:
* The plan keeps arrays over all V^R nodes, which is 2^59 for renewal-60. Here only the nodes seen in the counting
  windows have rows; any other node reads the row of its suffix representative.
* The merge compares a group only with the groups that have the same next groups. That gives the plan's result
  exactly: `tests/test_de_bruijn.py` checks it against the plan's own code on nine processes, together with the plan's
  contract tests (29 tests).

### 10.1 One model, step by step

Building the model does the counting, the merging and the warm start. On renewal-15 (Markov order 14):
""")

code(r"""
import model_de_bruijn as DB
DB_MODELS = ["de_bruijn", "de_bruijn_bic"]
F_uniform = lambda n: {"F": [1.0 / n] * n}
PROCESSES.update({
    "renewal-5": dict(process="renewal", params=F_uniform(5)),
    "renewal-10": dict(process="renewal", params=F_uniform(10)),
    "renewal-15": dict(process="renewal", params=F_uniform(15)),
    "renewal-30": dict(process="renewal", params=F_uniform(30), history_len=60, rollout_len=30, n_states=64,
                       context=29),
    "flower(3,5)": dict(process="flower", params={"n": 3, "m": 5, "dice_seed": 0}),
    "flower(3,5)-backward": dict(process="flower", params={"n": 3, "m": 5, "dice_seed": 0, "direction": "backward"}),
    "flower(8,14)": dict(process="flower", params={"n": 8, "m": 14, "dice_seed": 0}),
    "flower(8,14)-backward": dict(process="flower", params={"n": 8, "m": 14, "dice_seed": 0, "direction": "backward"})})
DB_PROCESSES = ["renewal-5", "renewal-10", "renewal-15", "renewal-30", "renewal-60", "flower(3,5)",
                "flower(3,5)-backward", "flower(8,14)", "flower(8,14)-backward"]

s15 = TR.settings_for("de_bruijn", **PROCESSES["renewal-15"])
model15 = TR.make_model(s15)
display(pd.Series(model15.info, name="renewal-15").to_frame().T)
x, st, exact = A.windows(s15, "test")
start = TR.evaluate(model15, s15, x, st, s15.tau_end, exact)
print("before any training, on the 4,096 test windows:",
      {k: round(v, 5) for k, v in start.items() if k.startswith(("excess_", "U_", "head_", "defect"))})
""")

md(r"""
`merge_rounds` is the number of groups after each round: 470 frequent nodes, then 379, 301, … down to the 15 causal
states. Before any training both predictors are already exact (excess about 0), and the square commutes: U = 0 and G
agrees with the reader at every step.

On renewal-5 (Markov order 4) every node can be listed. Each state of the quotient is a group of nodes; it predicts
the counted P(next token = 1) and moves to a counted next state on each token:
""")

code(r"""
s5 = TR.settings_for("de_bruijn", **PROCESSES["renewal-5"])
model5 = TR.make_model(s5)
q5 = model5.quotient
nodes = pd.DataFrame({"node": [format(int(u), "04b") for u in q5.nodes], "state": q5.node_state})
display(pd.DataFrame({"nodes (oldest token first)": nodes.groupby("state")["node"].apply(", ".join),
                      "P(next token = 1)": q5.rows[:, 1].round(4), "next state after 0": q5.successor[:, 0],
                      "next state after 1": q5.successor[:, 1], "counted positions": q5.visits.astype(int)}))
""")

md(r"""
The state of a node is the number of 0s after its last 1: every node ending in 1 is state "0 zeros", every node ending
in 10 is "1 zero", and so on. The P(next token = 1) column is the counted version of the true 1/(5 − c) (0.2, 0.25,
0.333, 0.5, 1 for c = 0 … 4); 0000 can only be followed by a 1.

### 10.2 The runs

54 runs: the two variants × nine processes (renewal-5, 10, 15, 30, 60; the flower (3, 5) and (8, 14), each read
forward and backward) × seeds 0, 1, 2, in `plans/de_bruijn.json`, launched with
`python main.py train-many --plan plans/de_bruijn.json`. Window sizes and table rows grow with the renewal process
as in sections 16 and 7: 40 + 20 tokens and 32 rows up to renewal-15, 60 + 30 and 64 rows for renewal-30, 120 + 60
and 128 rows for renewal-60; the flowers use 40 + 20 and 32 rows.
""")

code(r"""
section10_runs = [dict(model=m, settings=dict(PROCESSES[label], seed=s))
                  for label in DB_PROCESSES for m in DB_MODELS for s in (0, 1, 2)]
print(len(section10_runs), "runs; all in plans/de_bruijn.json:", check_plan(section10_runs, "plans/de_bruijn.json"))
LAUNCH = False
if LAUNCH:
    launch("plans/de_bruijn.json")
print("finished seeds (of 3):")
display(status(DB_PROCESSES, DB_MODELS).T)
""")

md(r"""
### 10.3 The starting machine of every process

For every process: its true number of states, the order of each variant, how many nodes the counting saw and how many
were frequent enough to merge, and the number of states of each variant's quotient (where training starts).
""")

code(r"""
rows = []
for label in DB_PROCESSES:
    s = TR.settings_for("de_bruijn", **PROCESSES[label])
    row = {"process": label, "true states": len(s.process_object().T)}
    for rule in ("markov", "bic"):
        R, q, info = DB.starting_point(s, rule)
        row.update({f"R ({rule})": R, f"nodes seen ({rule})": info["nodes_seen"],
                    f"frequent ({rule})": info["frequent_nodes"], f"rounds ({rule})": len(info["merge_rounds"]) - 1,
                    f"quotient states ({rule})": q.k})
    rows.append(row)
display(pd.DataFrame(rows).set_index("process"))
""")

md(r"""
### 10.4 Performance on held-out data

The test excess CE of both predictors of every run (4,096 test windows, future tokens), per process and variant
(mean, smallest and largest over the finished seeds), with the "remembers nothing" level of each process. `long` is the
excess of the main predictor over 200 future tokens.
""")

code(r"""
summary = []
for label in DB_PROCESSES:
    for m in DB_MODELS:
        found = runs_of(label, [m])
        if not found:
            continue
        t = A.test_table(found)
        main, reader = t["excess_next_state_from_handoff_future"], t["excess_node_table_future"]
        summary.append({"process": label, "variant": m, "seeds": len(t), "main: mean": main.mean(),
                        "main: min": main.min(), "main: max": main.max(),
                        "long (200 tokens): mean": t["long_excess_next_state_from_handoff"].mean(),
                        "node table: mean": reader.mean(),
                        "'remembers nothing'": A.reference_levels(settings_of(label, DB_MODELS), "test")["remembers_nothing"]})
if not summary:
    print("No run of this section has finished yet.")
else:
    display(pd.DataFrame(summary).set_index(["process", "variant"]).round(5))
    fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
    for j, label in enumerate(DB_PROCESSES):
        for m, dx, colour in (("de_bruijn", -0.12, A.GOLD), ("de_bruijn_bic", 0.12, A.NEUTRAL)):
            found = runs_of(label, [m])
            if found:
                t = A.test_table(found)
                ax.scatter([j + dx] * len(t), t["excess_next_state_from_handoff_future"].clip(lower=1e-5), s=18,
                           color=colour, label=m if j == 0 else None, zorder=3)
        ax.scatter(j, A.reference_levels(settings_of(label, DB_MODELS), "test")["remembers_nothing"], marker="_",
                   s=200, color=A.INK2, label="'remembers nothing'" if j == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(len(DB_PROCESSES)), DB_PROCESSES, rotation=20, fontsize=7.5)
    ax.set_ylabel("test excess CE (bits; below 1e-5 drawn at 1e-5)")
    ax.set_title("The de Bruijn hybrid: test excess of the main predictor (G from the handoff), every run")
    ax.legend(fontsize=7)
    plt.show()
""")

md(r"""
**Against the other models.** The same test windows on renewal-15 (the runs of sections 16–17 shown in section 6,
with model 2 + L2 at weight 0.03) and on renewal-60 (section 7): the main predictor of every model, mean over its seeds,
its machine's size, and the groups it sorts the true states into (*covered*).
""")

code(r"""
def compare(found):
    # main-predictor test excess, machine size and true states covered, per model (mean over seeds; min and max).
    mt = A.machine_table(found)
    mt["model"] = mt["group"]
    out = mt.groupby("model", sort=False).agg(seeds=("seed", "count"), excess_mean=("test_excess", "mean"),
                                              excess_min=("test_excess", "min"), excess_max=("test_excess", "max"),
                                              states_mean=("states", "mean"), covered_mean=("covered", "mean"))
    return out.sort_values("excess_mean")

renewal15_others = {f"{m} {l} s{s}": f for (m, l, s), f in runs.items() if l in ("section16", "section17", "w0.03")}
for label, others in (("renewal-15", renewal15_others), ("renewal-60", runs_of("renewal-60"))):
    found = dict(others, **runs_of(label, DB_MODELS))
    print(f"{label}: {len(found)} runs")
    display(compare(found).round(5))
""")

md(r"""
### 10.5 Convergence

The validation excess of both predictors over training (seeds: solid, dashed, dotted), one panel per process and
variant; then U over training, the commutation entropy of the reader's own states (0 when its next state is fixed by its
state and the token), and the first update at which the main predictor fell below a few levels.
""")

code(r"""
db_runs = {f"{label} {name}": f for label in DB_PROCESSES for m in DB_MODELS
           for name, f in runs_of(label, [m]).items()}
if db_runs:
    A.plot_learning_curves(list(db_runs.values()), title="The de Bruijn hybrid: validation excess CE (seeds: solid, dashed, dotted)")
    plt.show()
    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    for name, f in db_runs.items():
        h = TR.history(f)
        ax.plot(h["update"] / 1000, h["U_bits"], lw=0.8, alpha=0.7)
    ax.set_xlabel("update (thousands)")
    ax.set_ylabel("U (bits)")
    ax.set_title("Commutation entropy U of the reader's states over training, every run")
    plt.show()
    display(A.first_below(db_runs, (0.01, 0.001, 0.0002)).set_index("run").round(5))
""")

md(r"""
### 10.6 The machines learned

The machine of each run's main predictor (G from the handoff), read off exactly as for every model: its size, the true
states it covers, its difference from the true transitions, and *merge_cost*, the excess its groups of true states
cause by themselves (section 7.5). Then the machines of the Markov-order variant, one row per process (seeds 0, 1, 2):
graphs up to 24 states, matrices on the true states above that.
""")

code(r"""
if db_runs:
    db_machines = A.machine_table(db_runs)
    display(db_machines.set_index("run")[["states", "covered", "mean_error", "max_error", "merge_cost",
                                          "test_excess"]].round(5))
    A.plot_machine_summary(db_machines, {})
    plt.show()
    grid = {f"{label} s{s}": runs_of(label, ["de_bruijn"]).get(f"de_bruijn s{s}")
            for label in DB_PROCESSES for s in (0, 1, 2)}
    A.plot_machine_grid(grid).suptitle("model_de_bruijn (R = Markov order): the machine of every run "
                                       "(rows: processes; columns: seeds 0, 1, 2)", fontsize=11)
    plt.show()
""")

md(r"""
### 10.7 Findings

*Written when the runs are done.*
""")
