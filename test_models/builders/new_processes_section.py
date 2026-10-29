"""Sections 7-9 of test_models/walkthrough.ipynb: five models on renewal-60 (7), on the flower process with 8 dice of 12
faces read forward and backward (8), and on the flower with 20 dice of 30 faces (9).  CELLS is a list of
("md" | "code", text); build_walkthrough.py puts them before the References."""

CELLS = []


def md(text):
    CELLS.append(("md", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# ==================================================================================================== shared code
md(r"""
## Sections 7–9: new processes

Sections 1–6 used one process, renewal-15. Sections 7–9 train five of the models on new processes, with three seeds
and 50,000 updates per run:

* **7. renewal-60**: renewal-15 with gaps of up to 59 0s (60 causal states);
* **8. the flower process with 8 dice of 12 faces**, read forward (9 causal states) and backward (13);
* **9. the flower process with 20 dice of 30 faces**, read forward (21 causal states) and backward (31).

**The five models** (section 3 describes them in full; the weights of the extra terms are those of section 16):
* **5b**: a Transformer with a two-step head. Its extra term (weight 0.1) is the entropy of the next state worked out
  from the predicted next token.
* **C**: as 5b, with a three-step head; the next state is worked out from two predicted tokens (weight 0.01).
* **B**: a Transformer and a GRU. Its extra term (weight 3) trains the Transformer's state after the history to be the
  state the GRU itself reaches after reading the history.
* **feedback GRU**: a GRU alone. Its state is a table row, fed back at every step.
* **discrete Transformer**: a Transformer alone, choosing a state at every position.

Each section has the same parts: the runs (the code that defines and launches them, and how many have finished), the
process, the performance on held-out data, the convergence, the machines learned, and the findings. The runs train in
the background (`python main.py train-many`, several at a time), so a section shows the runs finished when the notebook
was last run; the findings give the date and the runs they are based on.

The next cell holds the code the three sections share.
""")

code(r"""
import json
import numpy as np
from itertools import combinations
plt.rcParams["figure.dpi"] = 80              # smaller figures from here on (sections 7-9 have many)

NEW_MODELS = ["5b", "B", "C", "feedback_GRU", "discrete_Transformer"]
FLOWER_8_12 = {"n": 8, "m": 12, "dice_seed": 0}
FLOWER_20_30 = {"n": 20, "m": 30, "dice_seed": 0}
PROCESSES = {   # the settings of each process's runs that differ from section 16's
    "renewal-60": dict(process="renewal", params={"F": [1 / 60] * 60}, history_len=120, rollout_len=60,
                       n_states=128, context=59),
    "flower-9": dict(process="flower", params=FLOWER_8_12),
    "flower-13-backward": dict(process="flower", params=dict(FLOWER_8_12, direction="backward")),
    "flower-21": dict(process="flower", params=FLOWER_20_30, n_states=64),
    "flower-31-backward": dict(process="flower", params=dict(FLOWER_20_30, direction="backward"), n_states=64)}


def plan(labels, models, seeds=(0, 1, 2)):
    # The trainings of a section, as `main.py train-many` reads them: one per process, model and seed.
    return [dict(model=m, settings=dict(PROCESSES[label], seed=s)) for s in seeds for label in labels for m in models]


def check_plan(runs, path):
    # Are these runs exactly the ones in the plan file that was launched?
    key = lambda j: (j["model"], json.dumps(j["settings"], sort_keys=True))
    stored = {key(j) for j in json.loads(Path(path).read_text())}
    return all(key(j) in stored for j in runs)


def launch(path, workers=8):
    # Start the trainings of a plan in the background (finished runs are skipped; a log for each in runs/logs/).
    import subprocess
    return subprocess.Popen([sys.executable, "main.py", "train-many", "--plan", path, "--workers", str(workers),
                             "--memory-gb", "13"], stdout=open("runs/logs/train_many_from_notebook.log", "a"),
                            stderr=subprocess.STDOUT)


def runs_of(label, models=NEW_MODELS, seed=None):
    # The finished 50,000-update runs of a process as {"<model> s<seed>": folder}, in model and seed order.
    found = []
    for m in models:
        for folder, s in TR.find_runs(model=m, steps=50_000, **PROCESSES[label]).items():
            if seed in (None, s.seed):
                found.append((models.index(m), s.seed, f"{m} s{s.seed}", folder))
    return {name: folder for _, _, name, folder in sorted(found)}


def settings_of(label, models=NEW_MODELS):
    # The settings of a finished run of this process (window sizes, ...), or of a planned one.
    found = runs_of(label, models)
    if found:
        return TR.read_settings(next(iter(found.values())))
    return TR.settings_for("discrete_Transformer", **PROCESSES[label])


def status(labels, models):
    # Finished seeds (of 3) of each model on each process.
    return pd.DataFrame({label: {m: len(runs_of(label, [m])) for m in models} for label in labels})


def process_table(labels):
    # The analytical results of each process (process.py) and the two reference levels on its test windows.
    rows = []
    for label in labels:
        s = settings_of(label)
        proc, levels = s.process_object(), A.reference_levels(s, "test")
        a = proc.analytics()
        rows.append({"process": label, "tokens": proc.vocab_size, "causal states": a["n_causal"],
                     "C_mu (bits)": a["statistical_c"], "h (bits per token)": a["entropy_rate"],
                     "E (bits)": a["excess_entropy"], "chi = C_mu - E": a["crypticity"],
                     "'remembers nothing' (test)": levels["remembers_nothing"],
                     "'last symbol only' (test)": round(levels["last_symbol_only"], 6) + 0.0})   # (no -0.0)
    display(pd.DataFrame(rows).set_index("process").round(4))


def show_performance(label, models=NEW_MODELS):
    # The test excess CE of every predictor of every finished run, and the figure.
    runs = runs_of(label, models)
    if not runs:
        return print(f"{label}: no finished runs yet")
    levels, table = A.reference_levels(settings_of(label, models), "test"), A.test_table(runs)
    cols = [c for c in table if (c.startswith("excess_") and c.endswith("_future")) or c.startswith("long_excess_")]
    print(f"{label} | reference levels on the test windows:", {k: round(v, 4) + 0.0 for k, v in levels.items()})
    display(table.set_index(["model", "seed"])[cols].round(4))
    A.plot_test_summary(table, levels, title=f"{label}: test excess CE of every predictor")
    plt.show()


def show_convergence(label, levels, models=NEW_MODELS):
    # Validation curves, states in use, and the first update below each level, of every finished run.
    runs = runs_of(label, models)
    if not runs:
        return print(f"{label}: no finished runs yet")
    A.plot_learning_curves(list(runs.values()), title=f"{label}: validation excess CE (seeds: solid, dashed, dotted)")
    plt.show()
    fig = A.plot_states_over_training(list(runs.values()))
    fig.suptitle(f"{label}: states in use and the score gap over training", fontsize=10.5)
    plt.show()
    display(A.first_below(runs, levels).set_index("run").round(4))


def show_machines(label, kind, models=NEW_MODELS):
    # The machine table, its summary figure, and every run's machine (rows: models; columns: seeds 0, 1, 2).
    runs = runs_of(label, models)
    if not runs:
        return print(f"{label}: no finished runs yet")
    mt = A.machine_table(runs)
    display(mt.set_index(["model", "seed"])[["machine", "states", "covered", "mean_error", "max_error", "merge_cost",
                                              "test_excess"]].round(4))
    fig = A.plot_machine_summary(mt, A.reference_levels(settings_of(label, models), "test"))
    fig.suptitle(f"{label}: the machine of every run", fontsize=10.5)
    plt.show()
    grid = {f"{m} s{s}": runs.get(f"{m} s{s}") for m in models if runs_of(label, [m]) for s in (0, 1, 2)}
    fig = A.plot_machine_grid(grid, kind=kind)
    fig.suptitle(f"{label}: the machine of every run (rows: models; columns: seeds 0, 1, 2)", fontsize=11)
    plt.show()


def show_states(label, models=NEW_MODELS):
    # Seed 0 of every model: the excess at each true state; then which learned state each true state is sent to
    # (the flower) or the predicted P(next token = 1) at each true state (renewal).
    seed0 = runs_of(label, models, seed=0)
    if not seed0:
        return print(f"{label}: no finished runs yet")
    A.plot_excess_by_true_state(seed0, main_only=True,
                                title=f"{label}: excess CE at each true state (seed 0, main predictor)")
    plt.show()
    if label.startswith("renewal"):
        A.plot_token_probs_by_true_state(seed0)
    else:
        A.plot_state_tables(seed0, main_only=True).suptitle(
            f"{label}: which learned state each true state is sent to (seed 0, main predictor)", fontsize=10.5)
    plt.show()


def show_groups(label, models=NEW_MODELS):
    # The groups of true states each run's machine puts together (true states sent to the same learned state).
    names = settings_of(label, models).process_object().state_names
    for name, folder in runs_of(label, models).items():
        groups = sorted((sorted(g) for g in A.machine(folder)["sent_to"] if g), key=lambda g: g[0])
        print(f"{name:24s} {len(groups):2d} groups: " + " | ".join(
            names[g[0]] if len(g) == 1 else (f"{names[g[0]]}–{names[g[-1]]}" if g == list(range(g[0], g[-1] + 1))
                                              else "{" + ",".join(names[c] for c in g) + "}") for g in groups))


def compare_directions(forward, backward, models=NEW_MODELS):
    # The same models and seeds on the flower read forward and backward: machine size, true states covered, merge cost
    # and test excess, side by side, and the summary figure with each forward row above its backward row.
    fw, bw = runs_of(forward, models), runs_of(backward, models)
    if not fw or not bw:
        return print("no finished runs yet in one of the directions")
    cols = ["states", "covered", "merge_cost", "test_excess"]
    tf, tb = A.machine_table(fw), A.machine_table(bw)
    both = tf.set_index(["model", "seed"])[cols].join(tb.set_index(["model", "seed"])[cols], how="outer",
                                                     lsuffix=" (forward)", rsuffix=" (backward)")
    display(both.reindex(models, level=0).round(4))
    pairs = pd.concat([tf.assign(direction=0), tb.assign(direction=1)])
    pairs = pairs.sort_values(["model", "direction", "seed"],
                              key=lambda c: c.map(models.index) if c.name == "model" else c)
    fig = A.plot_machine_summary(pairs, A.reference_levels(settings_of(forward, models), "test"))
    fig.suptitle(f"{forward} and {backward}: the machine of every run", fontsize=10.5)
    plt.show()


def merge_pairs(labels):
    # For every pair of true states: the excess of merging just that pair (A.merge_cost of that one group); a table
    # of the cheapest pair and how many pairs are cheap, and the costs sorted from cheapest to costliest.
    costs_of, rows = {}, []
    for label in labels:
        s = settings_of(label)
        names = s.process_object().state_names
        costs = {(i, j): A.merge_cost(s, [[i, j]]) for i, j in combinations(range(len(names)), 2)}
        costs_of[label] = sorted(costs.values())
        cheapest = min(costs, key=costs.get)
        rows.append({"process": label, "pairs of true states": len(costs),
                     "cheapest pair": f"{names[cheapest[0]]} + {names[cheapest[1]]}",
                     "its merge cost (bits per token)": costs[cheapest],
                     "pairs costing < 0.005": sum(v < 0.005 for v in costs.values()),
                     "pairs costing < 0.01": sum(v < 0.01 for v in costs.values()),
                     "median pair": float(np.median(list(costs.values())))})
    display(pd.DataFrame(rows).set_index("process").round(4))
    fig, ax = plt.subplots(figsize=(6.5, 2.8), layout="constrained")
    for (label, values), colour in zip(costs_of.items(), (A.BLUE, A.ORANGE)):
        ax.plot(np.arange(1, len(values) + 1) / len(values), values, "o-", ms=2, lw=1, color=colour,
                label=f"{label} ({len(values)} pairs)")
    ax.axhline(0.01, color=A.NEUTRAL, lw=0.8, ls=(0, (4, 2)))
    ax.set_yscale("log")
    ax.set_xlabel("share of the pairs of true states, cheapest first")
    ax.set_ylabel("merge cost (bits per token)")
    ax.set_title("The cost of merging one pair of true states (dashed: 0.01 bits)")
    ax.legend()
    plt.show()


def flower_states_table(labels):
    # Which tokens lead into each true state of a flower (tokens 0..n-1: the dice; n..n+m-1: the faces).
    def token_names(tokens, n):
        parts, run = [], []
        for t in tokens + [None]:
            if run and (t is None or t != run[-1] + 1 or (t >= n) != (run[-1] >= n)):
                one, many = ("die", "dice") if run[0] < n else ("face", "faces")
                parts.append(f"{one} {run[0]}" if len(run) == 1 else f"{many} {run[0]}–{run[-1]}")
                run = []
            if t is not None:
                run.append(t)
        return ", ".join(parts)
    longest = max(labels, key=lambda l: len(settings_of(l).process_object().state_names))
    display(pd.DataFrame({label: pd.Series([token_names(t, settings_of(label).params["n"])
                                            for t in A.entry_tokens(settings_of(label))],
                                           index=settings_of(label).process_object().state_names)
                          for label in labels}).reindex(settings_of(longest).process_object().state_names).fillna(""))
""")

# ==================================================================================================== section 7
md(r"""
## 7. Renewal-60

**The process.** After each 1, the number of 0s before the next 1 is equally likely to be any of 0–59 (renewal-15 had
0–14). Its 60 causal states are the counts of 0s since the last 1 (a0 … a59); from count c the next token is 1 with
probability 1/(60 − c) (Marzen & Crutchfield 2015). A model must carry a count of up to 59 through long stretches of 0s.

**The settings.** Section 16's, except for the sizes, which grow with the process: 128 table rows (K), windows of
120 history + 60 future tokens, and the Transformer's own predictions scored from position 59. There are never more
than 59 0s in a row, so every 120-token history contains a 1, and the true state at the end of every history is known.

### 7.1 The runs

15 runs: the five models × seeds 0, 1, 2. They were launched, together with section 8's 30 runs, from one plan file:
`python main.py train-many --plan plans/new_processes.json --workers 8 --memory-gb 13`. The next cell builds the list of
runs, checks that the plan file holds exactly these runs, and shows how many have finished. Set `LAUNCH = True` to start
the unfinished ones from here (several hours, in the background).
""")

code(r"""
section7_runs = plan(["renewal-60"], NEW_MODELS)
print(len(section7_runs), "runs; all in plans/new_processes.json:", check_plan(section7_runs, "plans/new_processes.json"))
LAUNCH = False
if LAUNCH:
    launch("plans/new_processes.json")
print("finished seeds (of 3):")
display(status(["renewal-60"], NEW_MODELS))
""")

md(r"""
### 7.2 The process

The two reference levels of section 5 on renewal-60's test windows, and the probability of a 1 at each count. The
whole gain from memory is small here: always predicting the long-run frequency of a 1 ("remembers nothing") is only
0.0145 bits per token above exact, so a learned predictor's excess has to be read against that number.
""")

code(r"""
process_table(["renewal-60"])
proc = settings_of("renewal-60").process_object()
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
### 7.3 Performance on held-out data

The excess CE of every predictor of every run on 4,096 test windows of 120 + 60 tokens (future tokens), at the final
parameters. Model B has two predictors: its Transformer, and its GRU stepped from the Transformer's state after the
history (its main predictor). The other models have one each. The `long_excess` columns are the excess over 200 future
tokens of the predictors that step from state to state (B's GRU, the feedback GRU).
""")

code(r"""
show_performance("renewal-60")
""")

md(r"""
### 7.4 Convergence

The validation excess CE over training (512 validation windows every 500 updates; grey dashed lines: the reference
levels), the states in use, the gap between the two highest state scores at the last history position (above about 10
the choice there no longer changes in training), and the first update at which each run's main predictor fell below
0.01, 0.005, 0.002 and 0.001 bits.
""")

code(r"""
show_convergence("renewal-60", (0.01, 0.005, 0.002, 0.001))
""")

md(r"""
### 7.5 The machines learned

As in section 6: the machine of each run's main predictor, read off exactly from a step rule (B's GRU, the feedback
GRU) or from the sequence of states the Transformer picks (5b, C, the discrete Transformer). The columns:
* *states*: the machine's states;
* *covered*: the true states that have a learned state assigned to them. Each learned state is assigned to the first
  true state sent to it, so this is the number of groups the true states are sorted into, not the number of true
  states that are alone in their group;
* *mean_error*, *max_error*: |learned − true| of the state-to-state probabilities, over the covered rows;
* *merge_cost*: the excess that the machine's groups cause by themselves. At each true state s of a group, predict
  the long-run mixture of the group's next-token distributions instead of s's own; the cost is the sum over states of
  π_s × KL(own distribution ‖ mixture), in bits per token. When *merge_cost* is close to *test_excess*, the error is
  the merges.

Each run's machine is too large to draw, so each panel shows its state-to-state matrix laid out on the 60 true states
(as in section 6; hatched rows: true states with no learned state assigned). The true matrix has 1 − 1/(60 − c) just
right of the diagonal and 1/(60 − c) in column a0. Then the groups of counts of each run, the excess at each true
state, and each predictor's mean P(next token = 1) at each true state against the truth (seed 0).
""")

code(r"""
show_machines("renewal-60", kind="matrix")
show_groups("renewal-60")
show_states("renewal-60")
""")

md(r"""
### 7.6 Findings so far

*Based on 11 of the 15 runs (seeds 0 and 1 of every model, seed 2 of the feedback GRU), finished by 28 Oct 2026, 08:00;
this part will be updated when the last four runs (seed 2 of 5b, B, C and the discrete Transformer) are done.*

1. **Two models learn renewal-60 on every seed so far; two never do; one depends on the seed.** 5b reaches 0.0011 and
   0.0012 bits per token and C 0.0006 on both seeds: they close 92–96% of the 0.0145-bit gap between "remembers
   nothing" and exact (7.3). The discrete Transformer reaches 0.0003 on seed 0 (98% of the gap) but stays at 0.0145
   on seed 1. The feedback GRU (seeds 0, 1, 2) and B's GRU (seeds 0, 1) stay at 0.0145–0.0146: they predict as if
   they remembered nothing.
2. **The runs that fail end with one state.** They use at most 2–7 states at any validation point and one at the end
   (7.4, states in use); their machines have one state (7.5). On renewal-15 the feedback GRU found all 15 true states on
   every seed (section 6.12).
3. **Good prediction without the true states.** The runs that learn sort the 60 counts into 8–9 (5b), 22–27 (C) and
   14 (the discrete Transformer, seed 0) groups of consecutive counts: wide groups at small counts, narrow ones near 59
   (7.5, groups). For 5b seed 0 these are 0–31, 32–43, 44–50, 51–53, 54–55, and 56, 57, 58, 59 each alone. Merging is
   cheap where the probability of a 1, 1/(60 − c), changes slowly (0.017 at a0, 0.033 at a30; 7.2). The cost of each
   run's groups (*merge_cost*: 0.0008–0.0009 for 5b, 0.0004–0.0005 for C, 0.0001 for the discrete Transformer) is
   most of its test excess (0.0011–0.0012, 0.0006, 0.0003). The discrete Transformer, with 14 groups, has a lower
   excess than C with 22–27.
4. **Speed.** 5b and C fall below 0.002 bits within 2,500 updates. The discrete Transformer's seed 0 uses one state
   until about update 13,000 and falls below 0.002 at update 14,500 (7.4).
""")

# ==================================================================================================== section 8
md(r"""
## 8. The flower process with 8 dice of 12 faces, read forward and backward

**The process** (Thompson et al. 2018, Appendix D). The tokens alternate between a die and a face. A die token (0–7)
names one of 8 dice, each with probability 1/8; the next token is a face (8–19) rolled with that die. Each die's 12
face probabilities were drawn at random once (`dice_seed` 0) and are fixed.
* **Read forward** (flower-9) it has 9 causal states: one after any face (the next token is a die, each 1/8), and one
  for each die just named (the next token is a face of that die).
* **Read backward** (flower-13-backward: the same sequences from right to left), each face comes just before the die
  that rolled it. After a face, the next token is that die, and how likely each die is depends on the face: one state
  per face, 12 in all. After a die, the next token is a face with its long-run frequency: one more state, 13 in all.

In both directions the last token alone fixes the state, so predicting from the last token alone is exact ("last
symbol only" is 0 below) and no long memory is needed. The question is whether a model finds the smallest set of
states: 9 forward, 13 backward. The same process needs more memory to predict backward than forward: its statistical
complexity is 2.75 bits backward and 2.50 bits forward (Crutchfield, Ellison & Mahoney 2009 compare the two directions
in general).

**The settings.** Section 16's: 32 table rows, windows of 40 history + 20 future tokens.

### 8.1 The runs

30 runs: the five models × the two directions × seeds 0, 1, 2, in the same plan file as section 7.
""")

code(r"""
section8_runs = plan(["flower-9", "flower-13-backward"], NEW_MODELS)
print(len(section8_runs), "runs; all in plans/new_processes.json:", check_plan(section8_runs, "plans/new_processes.json"))
print("finished seeds (of 3):")
display(status(["flower-9", "flower-13-backward"], NEW_MODELS))
""")

md(r"""
### 8.2 The process

The analytical results and reference levels, which tokens lead into each true state (s0 is the state after a face going
forward, and after a die going backward), and the two true machines.
""")

code(r"""
process_table(["flower-9", "flower-13-backward"])
flower_states_table(["flower-9", "flower-13-backward"])
for label in ("flower-9", "flower-13-backward"):
    A.plot_true_machine(settings_of(label))
    plt.show()
""")

md(r"""
### 8.3 Performance on held-out data

As in 7.3, on 4,096 test windows of 40 + 20 tokens. B's Transformer predictor is not a result of training: B's loss
trains its Transformer only to pick the GRU's starting state after the history, and the states it picks at the other
positions are never trained to predict the next token. On the flower, where a wrong state can give a token almost no
probability, that predictor's excess is several bits.
""")

code(r"""
for label in ("flower-9", "flower-13-backward"):
    show_performance(label)
""")

md(r"""
### 8.4 Convergence
""")

code(r"""
for label in ("flower-9", "flower-13-backward"):
    show_convergence(label, (0.1, 0.01, 0.002))
""")

md(r"""
### 8.5 The machines learned

As in 7.5. Here each machine is drawn as a graph, each learned state named by the true states sent to it (arrows
without token labels: there are 20 tokens). On the flower *merge_cost* is exact: the state is fixed by the last token,
so a machine that merges these groups and otherwise predicts exactly has exactly this excess.
""")

code(r"""
for label in ("flower-9", "flower-13-backward"):
    show_machines(label, kind="graph")
    show_groups(label)
    show_states(label)
""")

md(r"""
### 8.6 Forward against backward

The same models and seeds in the two directions; then, for every pair of true states, the cost of merging just that
pair; then each die's two most likely faces.
""")

code(r"""
compare_directions("flower-9", "flower-13-backward")
merge_pairs(["flower-9", "flower-13-backward"])
dice = P.make_dice(8, 12, 0)
display(pd.DataFrame({"die": range(8), "its state (forward)": [f"s{k + 1}" for k in range(8)],
                      "most likely face": [int(np.argmax(d)) + 8 for d in dice], "probability": dice.max(1),
                      "second face": [int(np.argsort(-d)[1]) + 8 for d in dice],
                      "its probability": np.sort(dice, 1)[:, -2]}).set_index("die").round(3))
""")

md(r"""
### 8.7 Findings

*Based on all 30 runs (28 Oct 2026).*

1. **Forward (9 true states), only B finds all 9 on every seed** (0.0006–0.0009 bits per token). The feedback GRU
   finds 9, 9 and 7 (0.0004, 0.0005, 0.0549); the discrete Transformer 9, 9 and 8 (0.0003, 0.0006, 0.0293); 5b 7, 9
   and 6 (0.0552, 0.0007, 0.0785); C 7, 9 and 7 (0.0571, 0.0006, 0.0549) (8.3, 8.5).
2. **Backward (13 true states) is harder.** The discrete Transformer finds 13, 12 and 13 (0.0004, 0.0086, 0.0004),
   5b 6, 13 and 12 (0.0674, 0.0006, 0.0079), B 12 on every seed (0.0040–0.0041), the feedback GRU 12, 11 and 11 (0.0079,
   0.0229, 0.0227), and C 2 on every seed (0.2462–0.2464): C puts every face state into one.
3. **The error is the merges.** Each run's *merge_cost* matches its test excess: for example 0.0553 against 0.0552
   (5b seed 0, forward), 0.0805 against 0.0785 (5b seed 2, forward), 0.0040 against 0.0040–0.0041 (B, backward) and
   0.2418 against 0.2462–0.2464 (C, backward) (8.5). The models predict the states they keep as well as those states
   can be predicted; what they miss is states.
4. **Backward offers more cheap merges.** 11 of the 78 pairs of backward states cost less than 0.01 bits per token to
   merge, against 1 of the 36 forward pairs (8.6). Finding fewer states backward fits this; it does not prove the
   cause.
5. **The same merges recur.** All six forward runs with fewer than 9 states merge dice 0 and 5 (states s1 and s6);
   five of them also merge die 7 with die 6 or with dice 0 and 5. The pair of dice 0 and 5 is only the 12th cheapest of
   36 (0.0302 bits), while the cheapest pair, dice 2 and 3 (0.0094), is never merged. Dice 6 and 7 have the same most
   likely face (15), and die 0's most likely face (17) is die 5's second; dice 2 and 3 have different most likely faces
   (11 and 8) (8.6, last table). Backward, B merges faces 13 and 14 (s6 and s7, the second cheapest pair) on all three
   seeds, and 5b and the feedback GRU merge faces 11 and 12 (s4 and s5) on five runs. These are observations from these
   runs, not a tested cause.
6. **B's Transformer predictor** has an excess of 5.3–6.0 bits on the flower (8.3); as said there, it is never trained
   to predict.
""")

# ==================================================================================================== section 9
md(r"""
## 9. The flower process with 20 dice of 30 faces, read forward and backward

**The process.** As in section 8, with 20 dice of 30 faces (`dice_seed` 0): 50 tokens, 21 causal states forward
(one after a face, one per die) and 31 backward (one after a die, one per face). Read backward, most pairs of states are
nearly alike: merging one pair costs less than 0.01 bits per token for 328 of the 465 backward pairs, against 19 of the
210 forward pairs (9.2).

**The settings.** As in section 8, with 64 table rows (K) instead of 32: about twice the 31 backward states, as 128
rows were for renewal-60's 60 states.

**Model C cannot run here.** Its three-step head predicts every combination of the next three tokens: 50³ = 125,000
outputs per position. Measured with one thread while eight other runs trained:

| flower (dice, faces) | tokens | C: seconds per update (batch 128) | peak memory | 50,000 updates |
|---|---|---|---|---|
| (8, 12) | 20 | 1.3 | 1.9 GB | about 18 h (section 8) |
| (10, 14) | 24 | 1.9 | 1.8 GB | about 26 h |
| (12, 18) | 30 | 6.3 | 2.3 GB | about 87 h |
| (20, 30) | 50 | — | — | at batch 8: 1.5 s per update and 370 s per evaluation; at batch 128 more than 10 GB |

So C's limit on this laptop is about 24 tokens, and this section has four models.

### 9.1 The runs

24 runs: four models × the two directions × seeds 0, 1, 2, in `plans/flower_20_30.json`. They start when sections 7 and
8's runs are done (`plans/after_new_processes.sh` waits for those, then runs
`python main.py train-many --plan plans/flower_20_30.json --workers 8 --memory-gb 13`).
""")

code(r"""
FLOWER_MODELS = ["5b", "B", "feedback_GRU", "discrete_Transformer"]          # model C cannot run on 50 tokens
section9_runs = plan(["flower-21", "flower-31-backward"], FLOWER_MODELS)
print(len(section9_runs), "runs; all in plans/flower_20_30.json:", check_plan(section9_runs, "plans/flower_20_30.json"))
LAUNCH = False
if LAUNCH:
    launch("plans/flower_20_30.json")
print("finished seeds (of 3):")
display(status(["flower-21", "flower-31-backward"], FLOWER_MODELS))
""")

md(r"""
### 9.2 The process
""")

code(r"""
process_table(["flower-21", "flower-31-backward"])
flower_states_table(["flower-21", "flower-31-backward"])
merge_pairs(["flower-21", "flower-31-backward"])
for label in ("flower-21", "flower-31-backward"):
    A.plot_true_machine(settings_of(label))
    plt.show()
""")

md(r"""
### 9.3 Results

Performance, convergence, the machines learned, and the two directions compared, as in section 8 (each part shows the
runs finished so far).
""")

code(r"""
if not (runs_of("flower-21") or runs_of("flower-31-backward")):
    print("No run of this section has finished yet.")
else:
    for label in ("flower-21", "flower-31-backward"):
        show_performance(label)
        show_convergence(label, (0.1, 0.01, 0.002))
        show_machines(label, kind="graph")
        show_states(label)
    compare_directions("flower-21", "flower-31-backward")
""")

md(r"""
### 9.4 Findings

*The runs of this section start when sections 7 and 8's runs are done (about 28 Oct 2026, 09:30) and take about 13
hours. The findings will be added then.*
""")
