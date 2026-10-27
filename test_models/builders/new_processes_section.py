"""Section 7 of test_models/walkthrough.ipynb: five models on renewal-60 and on the flower process read forward and
backward.  CELLS is a list of ("md" | "code", text); build_walkthrough.py puts them before the References.  FINDINGS
(markdown, written after the runs were read) closes the section."""

CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
## 7. New processes: renewal-60, and the flower process read forward and backward

Sections 1–6 used one process, renewal-15. This section trains five of the models on three new processes: three seeds
each, 50,000 updates per run, 45 runs in all. The runs are listed in `plans/new_processes.json`;
`python main.py train-many --plan plans/new_processes.json` trains them, several at a time.

**The five models** (section 3 describes them in full; the weights of the extra terms are those of section 16):
* **5b**: a Transformer with a two-step head. Its extra term (weight 0.1) is the entropy of the next state worked out
  from the predicted next token.
* **C**: as 5b, with a three-step head; the next state is worked out from two predicted tokens (weight 0.01).
* **B**: a Transformer and a GRU. Its extra term (weight 3) trains the Transformer's state after the history to be the
  state the GRU itself reaches after reading the history.
* **feedback GRU**: a GRU alone. Its state is a table row, fed back at every step.
* **discrete Transformer**: a Transformer alone, choosing a state at every position.

**The three processes**
* **renewal-60**: renewal-15 with longer gaps. After each 1, the number of 0s before the next 1 is equally likely to be
  any of 0–59. Its 60 causal states are the counts of 0s since the last 1 (a0 … a59); from count c the next token is
  1 with probability 1/(60 − c) (Marzen & Crutchfield 2015). The models get 128 table rows and windows of 120 history +
  60 future tokens. There are never more than 59 0s in a row, so every 120-token history contains a 1 and the true
  state at the end of every history is known.
* **flower-9**, the flower process with 8 dice of 12 faces (Thompson et al. 2018, Appendix D), read forward. The tokens
  alternate between a die and a face. A die token (0–7) names one of the 8 dice, each with probability 1/8; the next
  token is a face (8–19) rolled with that die. Each die's 12 face probabilities were drawn at random once
  (`dice_seed` 0) and are fixed. Its 9 causal states: one after any face (the next token is a die, each 1/8), and one
  for each die just named (the next token is a face of that die).
* **flower-13-backward**: the same sequences read from right to left. Now each face comes just before the die that
  rolled it. After a face, the next token is that die, and how likely each die is depends on the face: one state per
  face, 12 in all. After a die, the next token is a face with its long-run frequency: one more state, 13 in all.

The two kinds of process test different things. On renewal-60 the state is a count up to 59, so a model must carry
information over long stretches of 0s. On the flower the last token alone fixes the state in both directions, so
predicting from the last token alone is already exact (its excess CE is 0 in the table below) and no long memory is
needed. The question there is whether a model finds the smallest set of states: 9 going forward, 13 going backward. The
same process needs more memory to predict backward than forward: its statistical complexity is 2.75 bits backward and
2.50 bits forward (Crutchfield, Ellison & Mahoney 2009 compare the two directions in general).

The section has the same three parts as section 6 (performance, convergence, the machines learned), then compares the
two directions of the flower and ends with the findings.

### 7.1 The runs and the processes
""")

code(r"""
import numpy as np
from itertools import combinations
plt.rcParams["figure.dpi"] = 80              # smaller figures from here on (this section has many)
NEW_ROOT, NEW_STEPS = None, 50_000           # where the runs are (None: runs/) and their length
NEW_PROCESSES = {
    "renewal-60": dict(process="renewal", params={"F": [1 / 60] * 60}, history_len=120, rollout_len=60,
                       n_states=128, context=59),
    "flower-9": dict(process="flower", params={"n": 8, "m": 12, "dice_seed": 0}),
    "flower-13-backward": dict(process="flower", params={"n": 8, "m": 12, "dice_seed": 0, "direction": "backward"})}
NEW_MODELS = ["5b", "B", "C", "feedback_GRU", "discrete_Transformer"]

new_runs = {}                                # {(process, model, seed): run folder}, finished runs only
for label, changes in NEW_PROCESSES.items():
    for model in NEW_MODELS:
        for folder, s in TR.find_runs(NEW_ROOT, model=model, steps=NEW_STEPS, **changes).items():
            new_runs[(label, model, s.seed)] = folder


def new_runs_of(label, model=None, seed=None):
    # The finished runs of one process (optionally of one model or one seed) as {"<model> s<seed>": folder}.
    keys = sorted((k for k in new_runs if k[0] == label and model in (None, k[1]) and seed in (None, k[2])),
                  key=lambda k: (NEW_MODELS.index(k[1]), k[2]))
    return {f"{m} s{s}": new_runs[(l, m, s)] for l, m, s in keys}


new_settings = {label: TR.read_settings(next(iter(new_runs_of(label).values()))) if new_runs_of(label)
                else TR.settings_for("discrete_Transformer", **changes) for label, changes in NEW_PROCESSES.items()}
new_levels = {label: A.reference_levels(s, "test") for label, s in new_settings.items()}

print("finished runs (seeds) per model and process:")
display(pd.DataFrame(list(new_runs), columns=["process", "model", "seed"])
        .pivot_table(index="model", columns="process", values="seed", aggfunc="count")
        .reindex(index=NEW_MODELS, columns=list(NEW_PROCESSES)))
rows = []
for label, s in new_settings.items():
    proc = s.process_object()
    a = proc.analytics()
    rows.append({"process": label, "tokens": proc.vocab_size, "causal states": a["n_causal"],
                 "C_mu (bits)": a["statistical_c"], "h (bits per token)": a["entropy_rate"],
                 "E (bits)": a["excess_entropy"], "chi = C_mu - E": a["crypticity"],
                 "'remembers nothing' (test)": new_levels[label]["remembers_nothing"],
                 "'last symbol only' (test)": round(new_levels[label]["last_symbol_only"], 6) + 0.0})   # (no -0.0)
display(pd.DataFrame(rows).set_index("process").round(4))
""")

md(r"""
The last two columns are the reference levels of section 5 on each process's test windows: the excess CE of always
predicting each token's long-run frequency, and of predicting from the last token alone. On renewal-60 the whole
gain from memory is small: remembering nothing costs only about 0.015 bits per token. A learned predictor's excess
has to be read against that number. On the flower, remembering nothing costs about 1.2 bits per token, while the last
token alone is exact.

Which tokens lead into each state of the flower's two machines (s0 is the state after a face going forward, and after a
die going backward):
""")

code(r"""
def token_names(tokens, n=8):
    # 'dice 0-7', 'face 9', ... for a sorted list of flower tokens (0..n-1: dice, n..: faces)
    parts, run = [], []
    for t in tokens + [None]:
        if run and (t is None or t != run[-1] + 1 or (t >= n) != (run[-1] >= n)):
            one, many = ("die", "dice") if run[0] < n else ("face", "faces")
            parts.append(f"{one} {run[0]}" if len(run) == 1 else f"{many} {run[0]}–{run[-1]}")
            run = []
        if t is not None:
            run.append(t)
    return ", ".join(parts)


display(pd.DataFrame({label: pd.Series([token_names(t) for t in A.entry_tokens(new_settings[label])],
                                       index=new_settings[label].process_object().state_names)
                      for label in ("flower-9", "flower-13-backward")})
        .reindex(new_settings["flower-13-backward"].process_object().state_names).fillna(""))
for label in ("flower-9", "flower-13-backward"):
    fig = A.plot_true_machine(new_settings[label]); plt.show()
proc = new_settings["renewal-60"].process_object()
fig, ax = plt.subplots(figsize=(7, 2.4), layout="constrained")
ax.plot(proc.emission()[:, 1], "o-", ms=2.5, color=A.NEUTRAL)
ax.set_xticks(range(0, 60, 5), [proc.state_names[c] for c in range(0, 60, 5)])
ax.set_yscale("log")
ax.set_xlabel("true state (0s since the last 1)")
ax.set_ylabel("P(next token = 1)")
ax.set_title("renewal-60: the probability of a 1 at each true state, 1/(60 − c)")
plt.show()
""")

md(r"""
The settings of the runs (`settings.json` of each run). Everything not listed is section 16's setup (section 4).
""")

code(r"""
keys = ["history_len", "rollout_len", "context", "n_states", "state_dim", "d_model", "steps", "batch_size", "lr",
        "tau_start", "tau_end", "weight_decay", "grad_clip", "encoder_norm", "eval_every"]
display(pd.DataFrame({label: {k: getattr(s, k) for k in keys} for label, s in new_settings.items()}))
display(pd.Series({m: TR.settings_for(m).weight for m in NEW_MODELS}, name="weight of the extra term").to_frame().T)
""")

md(r"""
### 7.2 Performance on held-out data

The excess CE of every predictor of every run on the 4,096 test windows (future tokens), at the final parameters. The
test windows have each run's own sizes: 120 history + 60 future tokens on renewal-60, 40 + 20 on the flower. Model B
has two predictors: its Transformer, and its GRU stepped from the Transformer's state after the history (its main
predictor). The other models have one each. The `long_excess` column is the excess over 200 future tokens of a
predictor that steps from state to state (B's GRU, the feedback GRU).

B's Transformer predictor is not a result of training. B's loss trains its Transformer only to pick the GRU's starting
state after the history; the states it picks at the other positions are never trained to predict the next token. On
the flower, where a wrong state can give a token almost no probability, that predictor's excess is several bits.
""")

code(r"""
new_tables = {label: A.test_table(new_runs_of(label)) for label in NEW_PROCESSES}
for label, table in new_tables.items():
    cols = [c for c in table if (c.startswith("excess_") and c.endswith("_future")) or c.startswith("long_excess_")]
    print(f"{label} | reference levels on the test windows:",
          {k: round(v, 4) + 0.0 for k, v in new_levels[label].items()})
    display(table.set_index(["model", "seed"])[cols].round(4))
    fig = A.plot_test_summary(table, new_levels[label], title=f"{label}: test excess CE of every predictor")
    plt.show()
""")

md(r"""
### 7.3 Convergence

The validation excess CE over training (512 validation windows every 500 updates; the grey dashed lines are the
reference levels), the states in use, and the first update at which each run's main predictor fell below a few levels.
""")

code(r"""
BELOW = {"renewal-60": (0.01, 0.005, 0.002, 0.001), "flower-9": (0.1, 0.01, 0.002), "flower-13-backward": (0.1, 0.01, 0.002)}
for label in NEW_PROCESSES:
    folders = list(new_runs_of(label).values())
    fig = A.plot_learning_curves(folders, title=f"{label}: validation excess CE (seeds: solid, dashed, dotted)")
    plt.show()
    fig = A.plot_states_over_training(folders)
    fig.suptitle(f"{label}: states in use and the score gap over training", fontsize=10.5)
    plt.show()
    display(A.first_below(new_runs_of(label), BELOW[label]).set_index("run").round(4))
""")

md(r"""
### 7.4 The machines learned

As in section 6: the machine of each run's main predictor, read off exactly from a step rule (B's GRU, the feedback
GRU) or from the sequence of states the Transformer picks (5b, C, the discrete Transformer). *covered* counts the true
states that have a learned state of their own; *mean_error* is the mean |learned − true| of the state-to-state
probabilities over the covered rows.

*merge_cost* is the excess the machine's merges cause by themselves. Take each learned state's group of true states
(the true states sent to it). At each true state s of a group, predict the long-run mixture of the group's next-token
distributions instead of s's own. The cost is the sum over states of π_s × KL(own distribution ‖ mixture), in bits
per token. When *merge_cost* is close to *test_excess*, the machine's error is its merges. On the flower this
accounting is exact, because the state is fixed by the last token.
""")

code(r"""
new_machines = {label: A.machine_table(new_runs_of(label)) for label in NEW_PROCESSES}
for label, mt in new_machines.items():
    print(label)
    display(mt.set_index(["model", "seed"])[["machine", "states", "covered", "mean_error", "max_error",
                                              "merge_cost", "test_excess"]].round(4))
    fig = A.plot_machine_summary(mt, new_levels[label])
    fig.suptitle(f"{label}: the machine of every run", fontsize=10.5)
    plt.show()
""")

md(r"""
Each run's machine (rows: models; columns: seeds 0, 1, 2). On the flower each machine is drawn as a graph, named by
the true states sent to each learned state (arrows without token labels: there are 20 tokens). On renewal-60 the
machines are too large to draw, so each panel shows the machine's state-to-state matrix laid out on the 60 true states
(as in section 6); the true one has 1 − 1/(60 − c) just right of the diagonal and 1/(60 − c) in column a0.
""")

code(r"""
for label in NEW_PROCESSES:
    fig = A.plot_machine_grid(new_runs_of(label), kind="matrix" if label == "renewal-60" else "graph")
    fig.suptitle(f"{label}: the machine of every run", fontsize=11)
    plt.show()
""")

md(r"""
Where the errors are: the excess CE of each model's main predictor at each true state, for seed 0 (the grey bars are
how often each true state occurs). Then, for renewal-60, each predictor's mean P(next token = 1) at each true state against the truth; for
the flower, which learned state each true state is sent to.
""")

code(r"""
for label in NEW_PROCESSES:
    fig = A.plot_excess_by_true_state(new_runs_of(label, seed=0), main_only=True,
                                      title=f"{label}: excess CE at each true state (seed 0, main predictor; test windows, future tokens)")
    plt.show()
fig = A.plot_token_probs_by_true_state(new_runs_of("renewal-60", seed=0))
plt.show()
for label in ("flower-9", "flower-13-backward"):
    fig = A.plot_state_tables(new_runs_of(label, seed=0))
    fig.suptitle(f"{label}: which learned state each true state is sent to (seed 0)", fontsize=10.5)
    plt.show()
""")

md(r"""
### 7.5 The flower forward against backward

The same runs side by side: for every model and seed, the size of its machine, the true states it covers (of 9
forward, of 13 backward) and its test excess CE, in the two directions.
""")

code(r"""
cols = ["states", "covered", "mean_error", "test_excess"]
both = new_machines["flower-9"].set_index(["model", "seed"])[cols].join(
    new_machines["flower-13-backward"].set_index(["model", "seed"])[cols], lsuffix=" (forward)", rsuffix=" (backward)")
display(both.reindex(NEW_MODELS, level=0).round(4))
pairs = pd.concat([new_machines["flower-9"].assign(direction=0), new_machines["flower-13-backward"].assign(direction=1)])
pairs = pairs.sort_values(["model", "direction", "seed"], key=lambda c: c.map(NEW_MODELS.index) if c.name == "model" else c)
fig = A.plot_machine_summary(pairs, new_levels["flower-9"])
fig.suptitle("The flower forward (9 true states) and backward (13 true states): the machine of every run", fontsize=10.5)
plt.show()
""")

md(r"""
How costly each possible merge is: for every pair of true states of the flower, the excess of merging just that pair
(*merge_cost* of that one group), and the pairs sorted from cheapest to costliest.
""")

code(r"""
pair_costs, rows = {}, []
for label in ("flower-9", "flower-13-backward"):
    s = new_settings[label]
    names = s.process_object().state_names
    costs = {(i, j): A.merge_cost(s, [[i, j]]) for i, j in combinations(range(len(names)), 2)}
    pair_costs[label] = sorted(costs.values())
    cheapest = min(costs, key=costs.get)
    rows.append({"process": label, "pairs of true states": len(costs),
                 "cheapest pair": f"{names[cheapest[0]]} + {names[cheapest[1]]}",
                 "its merge cost (bits per token)": costs[cheapest],
                 "pairs costing < 0.01": sum(v < 0.01 for v in costs.values()),
                 "pairs costing < 0.02": sum(v < 0.02 for v in costs.values()),
                 "median pair": float(np.median(list(costs.values())))})
display(pd.DataFrame(rows).set_index("process").round(4))
fig, ax = plt.subplots(figsize=(6.5, 2.8), layout="constrained")
for (label, values), colour in zip(pair_costs.items(), (A.BLUE, A.ORANGE)):
    ax.plot(range(1, len(values) + 1), values, "o-", ms=2.5, lw=1, color=colour, label=label)
ax.axhline(0.01, color=A.NEUTRAL, lw=0.8, ls=(0, (4, 2)))
ax.set_yscale("log")
ax.set_xlabel("pairs of true states, cheapest first")
ax.set_ylabel("merge cost (bits per token)")
ax.set_title("The cost of merging one pair of true states (dashed: 0.01 bits)")
ax.legend()
plt.show()
""")
