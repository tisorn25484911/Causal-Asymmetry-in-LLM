"""
The training windows of a run, found again, and what the models were shown during training.

Runs do not save their training windows: runs/ keeps settings, parameters and measurements.  The windows can be found
again exactly.  training.train() and training.extend() draw the windows of update k as the k-th call of ONE window
sampler, process.sampler(settings.train_seed), and each call of WindowSampler.draw(batch_size, total_len) uses exactly
batch_size x (total_len + 1) random numbers: one for each window's starting state, then one per token.  So the
windows of update k are the sampler's draw after skipping (k - 1) x batch_size x (total_len + 1) numbers, which numpy's
PCG64 generator does in one step (its `advance`).  check_run() tests this on a saved run: the sampler state saved with
each of its checkpoints must equal the state after skipping all of its updates.

The windows depend only on the process with its parameters, train_seed, batch_size and total_len, not on the model.
Runs that share these five (a "stream") were trained on the same windows.

Finding windows
    windows_at(run, update, states=False)   the windows of one update (run: a run folder or a Settings); with
                                            states=True also the true states (process.WindowSampler explains them)
    windows(run, first=1, last=None, states=False)
                                            (update, windows[, states]) for every update from first to last, in order
    last_update(folder)                     the last update a saved run was trained to, continuations included
    check_run(folder)                       {checkpoint file: True when its saved sampler state equals the skip}
    stream_key(settings), stream_label(key), streams()
                                            which saved runs share their windows
    training_history(folder)                every update's training loss and loss terms (imported runs: from the
                                            source files their settings.json names)

What each model reads and is scored on
    term_positions(settings)                for each term of a model's loss, the positions it uses
    loss_positions(settings)                for each next-token CE term, the weight of every token position in it; the
                                            batch's exact CE of the same tokens is sum_j weight[j] x exact CE at j
    ExactCE(process)                        the exact CE of every token of a batch, and whether the true state is known
                                            from the tokens so far (fast forward algorithm for unifilar machines)

Statistics of a stream: `python training_data.py stats` computes every stream into runs/training_data/<label>.npz
(arrays) and <label>.json (numbers); load_statistics(label) reads them.  For every update of the stream (up to the
last update any of its runs reached):
    state_counts        (block, position j, state): how often the true state before token j is each state, summed over
                        blocks of BLOCK updates (position total_len: the state after the last token)
    exact_<term>        the batch's exact CE on the tokens each CE term of the stream's runs scores
    rare_handoff_H<H>   windows whose state before token H is one of the rarest states (lowest long-run probability)
    rare_scored_C<C>    (window, token) pairs with a rarest state, over tokens C + 1 .. total_len - 1
the repeats of whole windows and of their first H tokens (exact comparison, with the number of equal pairs the
process predicts), and the counts of every word of L consecutive tokens at every position for L in WORD_LENGTHS
(until more than WORD_CAP different words of a length are seen), with how many of the validation and test windows'
words were seen in training.

Reading the statistics (training_data.ipynb uses these)
    stream_table(), show_windows(run, update, rows)      the streams; the text of an update's windows and true states
    losses_at, loss_vs_content, loss_jumps               the runs' training losses beside their batches' exact CE
    state_check, rare_table                              the true states at each position against the long-run ones
    plot_positions, plot_known, plot_renewal_batches, plot_jumps, plot_coverage, plot_loss_vs_content    the figures
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import training as TR                                                      # noqa: E402

STATS = TR.RUNS / "training_data"
BLOCK = 1000                       # updates per block of the true-state counts
WORD_LENGTHS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)
WORD_CAP = 20_000_000              # stop counting words of a length once more than this many different ones were seen
DIRECT_CELLS = 2 ** 24             # lengths with V^L <= this are counted in a table; longer words by sorting
WORD_BUFFER = 8_000_000            # tokens collected before the words in them are counted
EVAL_SEEDS = dict(validation=None, test=4404)       # None: the run's own validation_seed (analysis.EVAL_SEEDS: test)
EVAL_ROWS = dict(validation=None, test=4096)        # None: the run's validation_rows (analysis.EVAL_ROWS: test)


# ================================================================================================ finding windows
def _settings(run):
    return run if isinstance(run, TR.Settings) else TR.read_settings(run)


def numbers_per_update(settings):
    """The random numbers one update's draw uses: one per window for its starting state, then one per token."""
    return settings.batch_size * (settings.total_len + 1)


def sampler_at(settings, update):
    """The training sampler of a run with `settings`, placed to draw the windows of `update` (1-based) next."""
    if update < 1:
        raise ValueError("updates are numbered from 1")
    sampler = settings.process_object().sampler(settings.train_seed)
    sampler.rng.bit_generator.advance((update - 1) * numbers_per_update(settings))
    return sampler


def last_update(folder):
    """The last update a saved run was trained to: its planned updates, or its last continuation."""
    return TR.checkpoint(folder, "latest")[1]


def _check_limit(run, update):
    if not isinstance(run, TR.Settings) and update > last_update(run):
        raise ValueError(f"{run} was trained for {last_update(run)} updates, so update {update} was never drawn")


def windows_at(run, update, states=False):
    """The training windows of update `update` (1-based, as in training_history.csv) of a run folder or Settings:
    a torch.long tensor (batch_size, total_len), and with states=True also the true states (numpy,
    (batch_size, total_len + 1): states[:, j] is the generating machine's state before token j)."""
    settings = _settings(run)
    _check_limit(run, update)
    return sampler_at(settings, update).draw(settings.batch_size, settings.total_len, states=states)


def windows(run, first=1, last=None, states=False):
    """(update, windows) or (update, windows, states) for every update from `first` to `last` (default: the run's last
    update), drawn in order from one sampler: faster than windows_at for many updates in a row."""
    settings = _settings(run)
    if last is None:
        if isinstance(run, TR.Settings):
            raise ValueError("give `last` when `run` is a Settings")
        last = last_update(run)
    _check_limit(run, last)
    sampler = sampler_at(settings, first)
    for update in range(first, last + 1):
        out = sampler.draw(settings.batch_size, settings.total_len, states=states)
        yield (update, *out) if states else (update, out)


def check_run(folder):
    """{checkpoint file: True/False}: does the training-sampler state saved with each checkpoint (completed.pt and any
    extended_<n>.pt) equal the state after skipping all its updates?  None for a checkpoint without a saved sampler
    state (section 13's run)."""
    folder = Path(folder)
    settings = TR.read_settings(folder)
    out = {}
    for file in sorted(folder.glob("*.pt")):
        if file.name == "best.pt":
            continue
        saved = torch.load(file, map_location="cpu", weights_only=False)
        if "rng_state" not in saved:
            out[file.name] = None
            continue
        out[file.name] = sampler_at(settings, saved["updates"] + 1).rng.bit_generator.state == saved["rng_state"]
    return out


def stream_key(settings):
    """The five settings that fix a run's training windows."""
    return (settings.process, json.dumps(settings.params, sort_keys=True), settings.train_seed, settings.batch_size,
            settings.total_len)


def stream_label(key):
    """A file name for a stream: '<process label>[_<parameters>]_T<total_len>_B<batch_size>_seed<train_seed>'."""
    name, params_json, train_seed, batch_size, total_len = key
    params = json.loads(params_json)
    label = TR.get_process(name, params_json).label
    if name == "flower":
        label += f"_n{params['n']}m{params['m']}" + (f"d{params['dice_seed']}" if params.get("dice_seed", 0) else "")
    elif not (name == "renewal" and len(set(params.get("F", [0, 1]))) == 1):
        label += "_" + hashlib.sha1(params_json.encode()).hexdigest()[:6]
    return f"{label}_T{total_len}_B{batch_size}_seed{train_seed}"


def run_folders(root=None, finished=True):
    """Every saved run folder: runs/model_*/<run> and runs/imported/model_*/<run>."""
    root = Path(root or TR.RUNS)
    out = sorted(p.parent for p in root.glob("model_*/*/settings.json")) + \
        sorted(p.parent for p in root.glob("imported/model_*/*/settings.json"))
    return [f for f in out if (f / "completed.pt").exists() or not finished]


def streams(root=None):
    """{stream key: [run folders]} of the finished saved runs."""
    out = {}
    for folder in run_folders(root):
        out.setdefault(stream_key(TR.read_settings(folder)), []).append(folder)
    return out


def training_history(folder):
    """Every update's training loss and loss terms of a saved run (its own training_history*.csv files; for an imported
    run, the files its settings.json names), one row per update from 1 to the last."""
    folder = Path(folder)
    meta = json.loads((folder / "settings.json").read_text())
    if "training_histories" in meta:
        files = [Path(p) for p in meta["training_histories"]]
    else:
        files = [folder / "training_history.csv"] + sorted(folder.glob("training_history_extended_*.csv"))
    h = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).sort_values("update", kind="stable")
    if not np.array_equal(h["update"].to_numpy(), np.arange(1, len(h) + 1)):
        raise ValueError(f"{folder}: the training history does not hold updates 1 .. {len(h)} once each")
    return h.reset_index(drop=True)


def run_name(folder):
    """A short name for a run: model, process label, seed and where it came from."""
    folder = Path(folder)
    s = TR.read_settings(folder)
    meta = json.loads((folder / "settings.json").read_text())
    source = meta.get("source", f"test_models {folder.name.rsplit('_', 1)[-1]}")      # its folder key: unique
    if folder.name.endswith("original_setup"):
        source += ", original setup"
    return f"{s.model} | {s.process_object().label} | w{s.weight:g} | seed {s.seed} | {source}"


# ================================================================================================ positions
GRU_HYBRIDS = {"1", "1_L2", "2", "2_L2", "3", "5a", "A", "B"}  # a GRU steps from the Transformer's state at H - 1
OWN_ONE_STEP = {"2", "2_L2", "3", "4", "5a", "5b", "A", "C"}  # + the Transformer's own next-token CE (ce_t1)
ALONE = {"feedback_GRU", "discrete_Transformer"}               # one network, its own next-token CE (ce)
DE_BRUIJN = {"de_bruijn", "de_bruijn_bic"}


def model_order(settings):
    """The order R of a de Bruijn model (its reader reads the last R tokens), computed as the model computes it."""
    module = TR.load_model_module(settings.model)
    return module.starting_point(settings, module.ORDER_RULE)[0]


def term_positions(settings, order=None):
    """{term: (kind, positions)} of a model's loss, positions 0-based in the window.  kind 'tokens': the CE of the
    tokens at these positions; 'tuples n': the CE of the n tokens x_{t+1} .. x_{t+n} predicted at each position t
    given; 'states': a term on the states chosen at these positions (after reading the token there); 'targets': the
    term's target is the state chosen at these positions.  The model files' loss() methods are the source."""
    H, C, T, m = settings.history_len, settings.context, settings.total_len, settings.model
    out = {}
    if m in GRU_HYBRIDS:
        out["ce_gru"] = ("tokens", list(range(H, T)))
    if m in OWN_ONE_STEP:
        out["ce_t1"] = ("tokens", list(range(C + 1, T)))
    if m in ALONE:
        out["ce"] = ("tokens", list(range(C + 1, T)))
    if m in ("5a", "5b", "C"):
        out["ce_t2"] = ("tuples 2", list(range(C, T - 2)))
        n = 2 if m == "C" else 1                                      # tokens that identify the next state
        out["inferred_entropy"] = ("states", list(range(C, T - 1 - n)))
    if m == "C":
        out["ce_t3"] = ("tuples 3", list(range(C, T - 3)))
    if m in ("3", "4", "A"):
        out["next_entropy"] = ("states", list(range(C, T - 1)))
        out["next_nll"] = ("targets", list(range(C + 1, T)))
    if m == "1_L2":
        out["l2"] = ("states", list(range(H, T)))                     # after every future token, the last included
    if m == "2_L2":
        out["l2"] = ("states", list(range(H, T - 1)))                 # after x_H .. x_{T-2}
    if m == "B":
        out["copy_ce"] = ("states", [H - 1])
    if m in DE_BRUIJN:
        R = model_order(settings) if order is None else order
        out["pred"] = ("tokens", list(range(R, T)))
        out["use"] = ("states", list(range(R - 1, T - 1)))
        out["fit"] = ("targets", list(range(R, T)))
        out["commit"] = ("targets", list(range(R, T)))
    if not out:
        raise KeyError(f"no positions are known for model {m!r}; add them to term_positions")
    return out


def loss_positions(settings, order=None):
    """{CE term: weight of each token position (length total_len)}: the term's exact counterpart on a batch is
    sum_j weight[j] x (the batch's mean exact CE at position j).  A tuple term ('ce_t2', 'ce_t3') is the CE of n tokens
    in a row, whose exact value is the sum of the n tokens' exact CEs."""
    T = settings.total_len
    out = {}
    for term, (kind, positions) in term_positions(settings, order).items():
        w = np.zeros(T)
        if kind == "tokens":
            w[positions] = 1.0 / len(positions)
        elif kind.startswith("tuples"):
            n = int(kind.split()[1])
            for j in range(1, n + 1):
                w[np.asarray(positions) + j] += 1.0 / len(positions)
        else:
            continue
        out[term] = w
    return out


class ExactCE:
    """
    -log2 P(x_j | x_0 .. x_{j-1}) for every token of a batch, from the long-run state distribution: the forward
    algorithm of process.filter_probs, written for a unifilar machine (each state and token lead to one next state),
    where one step costs O(states) instead of O(states^2 x tokens).  Equal to training.exact_ce_bits (tests).
    Also `known`: whether the state after x_0 .. x_t is certain from those tokens (the belief over the states is on one
    state).
    """

    def __init__(self, process):
        M = np.asarray(process.M, dtype=float)
        if ((M > 0).sum(axis=1) > 1).any():
            raise ValueError(f"{process.label}: the generating machine is not unifilar")
        self.emit = M.sum(axis=1)                                     # (S, V): P(token | state)
        self.succ = M.argmax(axis=1)                                  # (S, V): the next state
        self.pi = np.asarray(process.pi, dtype=float)

    def __call__(self, tokens, known=False):
        x = np.asarray(tokens)
        B, T = x.shape
        S = len(self.pi)
        belief = np.tile(self.pi, (B, 1))
        ce, sure = np.empty((B, T)), np.empty((B, T), dtype=bool)
        base = (np.arange(B) * S)[:, None]
        for t in range(T):
            w = belief * self.emit[:, x[:, t]].T                      # P(state, x_t | x_0 .. x_{t-1})
            p = w.sum(axis=1)
            ce[:, t] = -np.log2(p)
            belief = np.bincount((base + self.succ[:, x[:, t]].T).ravel(), weights=(w / p[:, None]).ravel(),
                                 minlength=B * S).reshape(B, S)
            if known:
                sure[:, t] = belief.max(axis=1) > 1 - 1e-9
        return (ce, sure) if known else ce


def known_share(process, total_len, rows=20_000, seed=7_000_000):
    """The share of windows whose state after reading x_0 .. x_t is certain from those tokens, for t = 0 .. total_len-1
    (fresh windows of their own random stream)."""
    x = process.sampler(seed).draw(rows, total_len).numpy()
    return ExactCE(process)(x, known=True)[1].mean(axis=0)


# ================================================================================================ window repeats
def pack_rows(tokens, vocab_size):
    """Each row of tokens as bytes (ceil(log2 V) bits per token), so that equal rows are equal bytes."""
    x = np.asarray(tokens, dtype=np.int64)
    bits = max(1, math.ceil(math.log2(vocab_size)))
    if bits > 8:
        raise ValueError("more than 256 tokens")
    if bits == 8:
        return x.astype(np.uint8)
    b = ((x[..., None] >> np.arange(bits)) & 1).astype(np.uint8).reshape(len(x), -1)
    return np.packbits(b, axis=1)


def unpack_rows(rows, vocab_size, length):
    bits = max(1, math.ceil(math.log2(vocab_size)))
    if bits == 8:
        return rows.astype(np.int64)[:, :length]
    b = np.unpackbits(rows, axis=1)[:, :length * bits].reshape(len(rows), length, bits).astype(np.int64)
    return (b << np.arange(bits)).sum(axis=-1)


def row_counts(rows):
    """(counts, first): how often each different row occurs, and the index of one of its rows, sorted by row."""
    n, nb = rows.shape
    pad = (-nb) % 8
    r = np.ascontiguousarray(np.pad(rows, ((0, 0), (0, pad))) if pad else rows)
    cols = r.view(np.uint64)
    order = np.lexsort(cols.T[::-1])
    s = cols[order]
    new = np.ones(n, dtype=bool)
    new[1:] = (s[1:] != s[:-1]).any(axis=1)
    starts = np.flatnonzero(new)
    return np.diff(np.append(starts, n)), order[starts]


def collision_probability(process, length):
    """sum_w P(w)^2 over the words w of `length` tokens from the long-run state distribution: the chance that two
    independent windows are equal.  A[s, s'] = sum_w P(w, ends in s) P(w, ends in s'), stepped token by token."""
    M = np.asarray(process.M, dtype=float)
    A = np.outer(process.pi, process.pi)
    for _ in range(length):
        A = sum(M[:, :, x].T @ A @ M[:, :, x] for x in range(M.shape[2]))
    return float(A.sum())


def repeats_summary(rows, process, length, top=10):
    """How many rows repeat an earlier one, the number of equal pairs against the process's prediction, and the most
    common rows (as tokens)."""
    counts, first = row_counts(rows)
    n = len(rows)
    pairs = float((counts * (counts - 1) / 2).sum())
    q = collision_probability(process, length)
    best = np.argsort(-counts, kind="stable")[:top]
    return dict(windows=n, different=int(len(counts)), repeats=int(n - len(counts)), equal_pairs=pairs,
                expected_equal_pairs=n * (n - 1) / 2 * q, collision_probability=q, max_count=int(counts.max()),
                top_counts=counts[best].tolist(),
                top_tokens=unpack_rows(rows[first[best]], process.vocab_size, length).tolist())


# ================================================================================================ word counts
def _unique_counts(a):
    """np.unique(a, return_counts=True) by sorting (numpy 2's hash-based unique is very slow on large word ids;
    model_de_bruijn.unique)."""
    s = np.sort(a)
    if not len(s):
        return s, np.empty(0, dtype=np.int64)
    new = np.ones(len(s), dtype=bool)
    new[1:] = s[1:] != s[:-1]
    starts = np.flatnonzero(new)
    return s[starts], np.diff(np.append(starts, len(s)))


def _merge(w1, c1, w2, c2):
    w, c = np.concatenate([w1, w2]), np.concatenate([c1, c2])
    order = np.argsort(w)
    w, c = w[order], c[order]
    new = np.ones(len(w), dtype=bool)
    new[1:] = w[1:] != w[:-1]
    starts = np.flatnonzero(new)
    return w[starts], np.add.reduceat(c, starts) if len(c) else c


def word_ids(tokens, V, lengths):
    """{L: ids (rows, T - L + 1)}: every word of L consecutive tokens as one base-V number, oldest token first."""
    x = np.asarray(tokens, dtype=np.int64)
    T = x.shape[1]
    out, ids = {}, x.copy()
    for L in range(1, max(lengths) + 1):
        if L > 1:
            ids = ids[:, :T - L + 1] * V + x[:, L - 1:]
        if L in lengths:
            out[L] = ids
    return out


class WordCounts:
    """Counts of every word of L consecutive tokens, at every position of every window, for several L.  Lengths whose
    ids do not fit in 63 bits are left out.  Lengths with V^L <= DIRECT_CELLS are counted in a table of all V^L words;
    a longer length is counted by sorting and dropped (capped) once more than `cap` different words are seen."""

    def __init__(self, V, lengths, cap=WORD_CAP):
        self.V, self.cap = V, cap
        self.lengths = sorted(L for L in set(lengths) if V ** L < 2 ** 62)
        self.table = {L: np.zeros(V ** L, dtype=np.int64) for L in self.lengths if V ** L <= DIRECT_CELLS}
        self.words = {L: (np.empty(0, np.int64), np.empty(0, np.int64)) for L in self.lengths if L not in self.table}
        self.capped, self.positions = set(), {L: 0 for L in self.lengths}

    def active(self):
        return [L for L in self.lengths if L not in self.capped]

    def add(self, tokens):
        lengths = [L for L in self.active() if L <= np.shape(tokens)[1]]
        if not lengths:
            return
        for L, ids in word_ids(tokens, self.V, lengths).items():
            self.positions[L] += ids.size
            if L in self.table:
                self.table[L] += np.bincount(ids.ravel(), minlength=len(self.table[L]))
                continue
            w, c = _merge(*self.words[L], *_unique_counts(ids.ravel()))
            if len(w) > self.cap:
                self.capped.add(L)
                self.words[L] = None
            else:
                self.words[L] = (w, c)

    def distinct(self, L):
        if L in self.capped:
            return None
        return int((self.table[L] > 0).sum()) if L in self.table else len(self.words[L][0])

    def counts_of(self, L, ids):
        """How often each word id in `ids` was seen (0: never); None for a capped length."""
        if L in self.capped:
            return None
        if L in self.table:
            return self.table[L][ids]
        w, c = self.words[L]
        ids = np.asarray(ids)
        if not len(w):
            return np.zeros(len(ids), dtype=np.int64)
        i = np.clip(np.searchsorted(w, ids), 0, len(w) - 1)
        return np.where(w[i] == ids, c[i], 0)

    def summary(self, L):
        """The number of different words, how many were seen 1, 2-9, 10-99, 100-999 or 1000+ times, and the share of
        all positions those words fill."""
        if L in self.capped:
            return dict(L=L, positions=self.positions[L], different=None)
        c = self.table[L][self.table[L] > 0] if L in self.table else self.words[L][1]
        out = dict(L=L, positions=self.positions[L], different=int(len(c)), fewest=int(c.min()) if len(c) else 0,
                   median=float(np.median(c)) if len(c) else 0.0)
        for name, a, b in (("1", 1, 2), ("2-9", 2, 10), ("10-99", 10, 100), ("100-999", 100, 1000),
                           ("1000+", 1000, np.inf)):
            sel = (c >= a) & (c < b)
            out[f"words_seen_{name}"] = int(sel.sum())
            out[f"positions_in_words_seen_{name}"] = float(c[sel].sum() / max(1, c.sum()))
        return out


# ================================================================================================ statistics
def _eval_windows(settings, kind):
    seed = settings.validation_seed if EVAL_SEEDS[kind] is None else EVAL_SEEDS[kind]
    rows = settings.validation_rows if EVAL_ROWS[kind] is None else EVAL_ROWS[kind]
    return settings.process_object().sampler(seed).draw(rows, settings.total_len).numpy()


def _relative(folder):
    """A run folder relative to runs/ when it is inside it."""
    try:
        return str(Path(folder).resolve().relative_to(TR.RUNS.resolve()))
    except ValueError:
        return str(folder)


def stream_statistics(key, folders, out=None, verbose=True, last=None):
    """Go through every update of a stream (up to the last update of its runs, or `last`) and save its statistics
    (module docstring) as out/<label>.npz and out/<label>.json.  Returns the json part."""
    out = Path(out or STATS)
    out.mkdir(parents=True, exist_ok=True)
    label = stream_label(key)
    runs = [TR.read_settings(f) for f in folders]
    s = runs[0]
    proc = s.process_object()
    B, T, V, S = s.batch_size, s.total_len, proc.vocab_size, len(proc.pi)
    last = last or max(last_update(f) for f in folders)
    exact = ExactCE(proc)
    pi = np.asarray(proc.pi)
    rarest = np.flatnonzero(pi <= pi.min() * (1 + 1e-9))
    is_rare = np.isin(np.arange(S), rarest)

    terms, orders = {}, {}                                            # every CE term's weights over positions
    for f, r in zip(folders, runs):
        order = model_order(r) if r.model in DE_BRUIJN else None
        if order is not None:
            orders[_relative(f)] = order
        for term, w in loss_positions(r, order).items():
            terms[exact_name(term, w, r.context)] = w
    names = sorted(terms)
    W = np.stack([terms[n] for n in names], axis=1)                   # (T, terms)
    hs = sorted({r.history_len for r in runs})
    cs = sorted({r.context for r in runs})

    n_blocks = (last + BLOCK - 1) // BLOCK
    state_counts = np.zeros((n_blocks, T + 1, S), dtype=np.int32)
    exact_terms = np.zeros((last, len(names)), dtype=np.float32)
    exact_mean = np.zeros(last, dtype=np.float32)
    rare_handoff = {H: np.zeros(last, dtype=np.int16) for H in hs}
    rare_scored = {C: np.zeros(last, dtype=np.int32) for C in cs}
    nbytes = pack_rows(np.zeros((1, T), dtype=np.int64), V).shape[1]
    window_rows = np.empty((last * B, nbytes), dtype=np.uint8)
    hbytes = {H: pack_rows(np.zeros((1, H), dtype=np.int64), V).shape[1] for H in hs}
    history_rows = {H: np.empty((last * B, hbytes[H]), dtype=np.uint8) for H in hs}
    words = WordCounts(V, [L for L in WORD_LENGTHS if L <= max(hs)] + [o for o in orders.values() if o <= max(hs)])
    buffer, buffered = [], 0
    position_index = (np.arange(T + 1) * S)[None, :]
    start = time.perf_counter()
    check = None
    for update, x, st in windows(s, 1, last, states=True):
        xn = x.numpy()
        ce = exact(xn)
        if check is None:                                             # the fast exact CE equals training's
            check = float(np.abs(ce - TR.exact_ce_bits(proc, xn)).max())
            if check > 1e-9:
                raise AssertionError(f"{label}: fast exact CE differs from training.exact_ce_bits by {check}")
        k = update - 1
        e = ce.mean(axis=0)
        exact_terms[k] = e @ W
        exact_mean[k] = e.mean()
        state_counts[k // BLOCK] += np.bincount((position_index + st).ravel(),
                                                minlength=(T + 1) * S).reshape(T + 1, S).astype(np.int32)
        for H in hs:
            rare_handoff[H][k] = is_rare[st[:, H]].sum()
            history_rows[H][k * B:(k + 1) * B] = pack_rows(xn[:, :H], V)
        for C in cs:
            rare_scored[C][k] = is_rare[st[:, C + 1:T]].sum()
        window_rows[k * B:(k + 1) * B] = pack_rows(xn, V)
        buffer.append(xn)
        buffered += xn.size
        if buffered >= WORD_BUFFER or update == last:
            words.add(np.concatenate(buffer))
            buffer, buffered = [], 0
        if verbose and (update % 5000 == 0 or update == last):
            print(f"{label}: update {update:,}/{last:,}, {(time.perf_counter() - start) / 60:.1f} min", flush=True)

    summary = dict(label=label, process=s.process, params=s.params, train_seed=s.train_seed, batch_size=B,
                   total_len=T, vocab_size=V, states=S, last_update=last, block=BLOCK,
                   runs=[_relative(f) for f in folders], de_bruijn_orders=orders,
                   history_lens=hs, contexts=cs, rarest_states=[proc.state_names[i] for i in rarest],
                   rarest_probability=float(pi.min()), exact_terms=names, fast_exact_check=check,
                   windows=repeats_summary(window_rows, proc, T),
                   histories={str(H): repeats_summary(history_rows[H], proc, H) for H in hs})
    del window_rows, history_rows
    coverage = []
    for kind in ("validation", "test"):
        held = _eval_windows(s, kind)
        for L in words.lengths:
            row = dict(kind=kind, L=L, words=int(held.shape[0] * (T - L + 1)))
            counts = words.counts_of(L, word_ids(held, V, [L])[L].ravel()) if L <= T else None
            if counts is not None:
                row.update(seen=float((counts > 0).mean()), seen_10=float((counts >= 10).mean()),
                           seen_100=float((counts >= 100).mean()), fewest=int(counts.min()))
            coverage.append(row)
    summary["word_counts"] = [words.summary(L) for L in words.lengths]
    summary["coverage"] = coverage
    summary["seconds"] = time.perf_counter() - start
    arrays = dict(state_counts=state_counts, exact_terms=exact_terms, exact_mean=exact_mean, pi=pi,
                  **{f"rare_handoff_H{H}": v for H, v in rare_handoff.items()},
                  **{f"rare_scored_C{C}": v for C, v in rare_scored.items()},
                  **{f"weights_{n}": terms[n] for n in names})
    np.savez_compressed(out / f"{label}.npz", **arrays)
    (out / f"{label}.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_statistics(label, root=None):
    """(json summary, npz arrays as a dict) of one stream's statistics."""
    root = Path(root or STATS)
    with np.load(root / f"{label}.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return json.loads((root / f"{label}.json").read_text()), arrays


# ================================================================================================ reading the statistics
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
PHASES = (("first 1,000 updates", 1, 1000), ("updates 1,001-10,000", 1001, 10_000), ("last 10,000 updates", None, None))


def style():
    """Matplotlib settings of the figures: light surface, neutral ink, hairline grid, the categorical colours in a
    fixed order (the dataviz skill's validated palette)."""
    import matplotlib as mpl
    mpl.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.prop_cycle": mpl.cycler(color=SERIES), "lines.linewidth": 1.6, "font.size": 9,
                         "axes.titlesize": 10, "legend.frameon": False})


def process_name(key):
    """The process part of a stream's label, e.g. 'flower-9_n8m12' or 'renewal-60'."""
    return stream_label(key).split("_T")[0]


def stream_table(root=None):
    """One row per stream: its label, process, seed (train_seed - train_seed_base), window length, last update and
    the runs trained on it."""
    rows = []
    for key, folders in streams(root).items():
        runs = [TR.read_settings(f) for f in folders]
        rows.append(dict(stream=stream_label(key), process=process_name(key), seed=key[2] - runs[0].train_seed_base,
                         window=key[4], batch=key[3], last_update=max(last_update(f) for f in folders),
                         runs=len(folders), models=", ".join(sorted({r.model for r in runs}))))
    return pd.DataFrame(rows).sort_values(["window", "process", "seed"]).reset_index(drop=True)


def show_windows(run, update, rows=4):
    """Text of the first `rows` windows of an update: each window's tokens and, under them, the true state before each
    token, one character per position (0-9, a-z, A-Z for 0 .. 61; a renewal state a<c> is c, any other state s<i> is
    i).  '|' stands before the first token the own-prediction CE scores (context + 1) and before the first future
    token (history_len)."""
    s = _settings(run)
    x, st = windows_at(run, update, states=True)
    proc = s.process_object()
    if max(proc.vocab_size, len(proc.pi)) > len(ALPHABET):
        raise ValueError("more tokens or states than one character each can show")
    marks = sorted({s.context + 1, s.history_len})

    def line(values):
        chars = [ALPHABET[v] for v in values]
        for m in reversed(marks):
            chars.insert(m, "|")
        return "".join(chars)

    out = [f"update {update:,} of {stream_label(stream_key(s))}: the first {rows} of its {s.batch_size} windows; "
           f"'|' before positions {', '.join(map(str, marks))}",
           f"position   {line([j % 10 for j in range(s.total_len)])}"]
    for i in range(rows):
        out += [f"tokens {i:<3} {line(x[i].tolist())}", f"states {i:<3} {line(st[i, :s.total_len].tolist())}"]
    return "\n".join(out)


@lru_cache(maxsize=None)
def _history(folder):
    return training_history(folder)


def exact_name(term, weights, context):
    """The name stream_statistics gives the batch's exact CE of a term's tokens."""
    if term in ("pred", "ce_gru", "ce_t1", "ce"):
        return f"{term}_from{int(np.flatnonzero(weights)[0])}"
    return f"{term}_C{context}" if term in ("ce_t2", "ce_t3") else term


def run_exact(folder, stats=None):
    """{CE term: the batch's exact CE of its tokens at every update the run trained (numpy)} for a saved run."""
    folder = Path(folder)
    s = TR.read_settings(folder)
    info, arrays = stats or load_statistics(stream_label(stream_key(s)))
    order = info["de_bruijn_orders"].get(_relative(folder)) if s.model in DE_BRUIJN else None
    last = last_update(folder)
    return {term: arrays["exact_terms"][:last, info["exact_terms"].index(exact_name(term, w, s.context))].astype(float)
            for term, w in loss_positions(s, order).items()}


def losses_at(folders, updates, stats=None):
    """Every run's training loss and loss terms at the given updates, with the batch's exact CE of each CE term's
    tokens ('exact <term>')."""
    rows = []
    for f in folders:
        h, exact = _history(str(f)), run_exact(f, stats)
        for u in updates:
            if u <= len(h):
                row = dict(run=run_name(f), update=u, **h.iloc[u - 1].drop("update").to_dict())
                row.update({f"exact {t}": float(e[u - 1]) for t, e in exact.items()})
                rows.append(row)
    return pd.DataFrame(rows)


def loss_vs_content(folder, stats=None):
    """For each next-token CE term of a run, in three phases of training (PHASES): how the term's per-update training
    value goes with the batch's exact CE of the same tokens.  correlation; explained = correlation^2, the share of the
    term's update-to-update variance the batch's own unpredictability accounts for; slope (term on exact); the mean
    excess (term - exact) and the spreads (standard deviations) of the term, of the exact CE and of the excess."""
    folder = Path(folder)
    s = TR.read_settings(folder)
    h, last = _history(str(folder)), last_update(folder)
    rows = []
    for term, e in run_exact(folder, stats).items():
        y = h[term].to_numpy(float)
        for phase, a, b in PHASES:
            a, b = (max(1, last - 9999), last) if a is None else (a, min(b, last))
            if b - a < 2:                                                # a short run has no updates in this phase
                continue
            ys, es = y[a - 1:b], e[a - 1:b]
            r = float(np.corrcoef(ys, es)[0, 1])
            rows.append(dict(run=run_name(folder), model=s.model, term=term, phase=phase, first=a, last=b,
                             correlation=r, explained=r ** 2, slope=float(np.polyfit(es, ys, 1)[0]),
                             mean_excess=float((ys - es).mean()), spread_term=float(ys.std()),
                             spread_exact=float(es.std()), spread_excess=float((ys - es).std())))
    return pd.DataFrame(rows)


def loss_jumps(folder, stats=None, top=3, window=101, apart=500, part="loss"):
    """The `top` updates where a run's training loss rose most above its running median (`window` updates, centred),
    after the first `window` updates and at least `apart` updates from each other.  part="loss": the whole training
    loss; part="ce": only its next-token CE terms added (the part the batch's exact CE is the floor of).  The same is
    measured for the batch's exact CE of the tokens those terms score: a jump that the batch explains has an exact jump
    of the same size.  'in spreads': the jump over the run's typical jump (1.4826 x the median absolute jump)."""
    folder = Path(folder)
    s = TR.read_settings(folder)
    info, arrays = stats or load_statistics(stream_label(stream_key(s)))
    h = _history(str(folder))
    exact_terms = run_exact(folder, (info, arrays))
    exact = sum(exact_terms.values())
    loss = (h["loss"] if part == "loss" else h[list(exact_terms)].sum(axis=1)).to_numpy(float)

    def above_median(v):
        return v - pd.Series(v).rolling(window, center=True, min_periods=window // 2 + 1).median().to_numpy()

    jump, exact_jump = above_median(loss), above_median(exact)
    valid = np.arange(len(loss)) >= window
    spread = 1.4826 * np.nanmedian(np.abs(jump[valid]))
    exact_spread = 1.4826 * np.nanmedian(np.abs(exact_jump[valid]))
    order = np.argsort(-np.where(valid & np.isfinite(jump), jump, -np.inf))
    chosen = []
    for i in order:
        if all(abs(i - j) >= apart for j in chosen):
            chosen.append(int(i))
        if len(chosen) == top:
            break
    hs = sorted({r for r in info["history_lens"]})
    rows = []
    for i in chosen:
        rows.append(dict(run=run_name(folder), model=s.model, part=part, update=i + 1, tau=float(h["tau"][i]), loss=loss[i],
                         jump=jump[i], jump_in_spreads=jump[i] / spread, exact=exact[i], exact_jump=exact_jump[i],
                         exact_jump_in_spreads=exact_jump[i] / exact_spread,
                         rare_at_handoff=int(arrays[f"rare_handoff_H{s.history_len}"][i])
                         if s.history_len in hs else None))
    return pd.DataFrame(rows)


def state_check(info, arrays):
    """Do the true states at every position follow the long-run distribution?  For each position, a chi-square test
    of the counts summed over all updates against windows x pi (the windows are independent, so each position's
    counts are a multinomial sample); and the largest standardised deviation of any (position, state) count."""
    from scipy import stats as sps
    counts = arrays["state_counts"].sum(axis=0).astype(float)          # (positions, states)
    pi, n = arrays["pi"], counts[0].sum()
    live = pi > 0
    expected = n * pi[live]
    chi2 = (((counts[:, live] - expected) ** 2) / expected).sum(axis=1)
    p = sps.chi2.sf(chi2, live.sum() - 1)
    z = (counts[:, live] - expected) / np.sqrt(expected * (1 - pi[live]))
    return dict(stream=info["label"], windows=int(n), positions=len(counts), smallest_p=float(p.min()),
                share_p_below_001=float((p < 0.01).mean()), largest_abs_z=float(np.abs(z).max()))


def rare_table(info, arrays):
    """How often the rarest states (lowest long-run probability) came up: windows per update whose state before the
    first future token (the handoff) is one of them, against batch_size x their probability; the share of updates
    with none; and the total over training."""
    rows = []
    pi_min = info["rarest_probability"] * len(info["rarest_states"])
    for H in info["history_lens"]:
        r = arrays[f"rare_handoff_H{H}"].astype(float)
        rows.append(dict(stream=info["label"], rarest_states=", ".join(info["rarest_states"]),
                         probability=pi_min, position=H, mean_per_update=r.mean(),
                         expected_per_update=info["batch_size"] * pi_min, share_of_updates_with_none=(r == 0).mean(),
                         expected_share_with_none=(1 - pi_min) ** info["batch_size"], total=int(r.sum())))
    return pd.DataFrame(rows)


# ================================================================================================ figures
TERM_GROUPS = (("the GRU's predictions of the future tokens", ("ce_gru",), SERIES[0]),
               ("own next-token predictions", ("ce_t1", "ce", "pred"), SERIES[1]),
               ("own predictions of 2 or 3 tokens ahead", ("ce_t2", "ce_t3"), SERIES[2]),
               ("extra terms on the chosen states", ("l2", "next_entropy", "next_nll", "inferred_entropy", "copy_ce",
                                                     "use", "fit", "commit"), SERIES[6]))


def plot_positions(settings_by_model, orders=None):
    """For each model (rows), each term of its loss (sub-rows): the window positions it uses.  CE terms show the tokens
    they score (a 2- or 3-token term: every token it scores); extra terms the positions of the states they use.  The
    dashed line is the last history position (the handoff); the grey bar is the window."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    orders = orders or {}
    layout = []
    for name, s in settings_by_model.items():
        terms = term_positions(s, orders.get(name))
        layout.append((name, s, [t for g in TERM_GROUPS for t in g[1] if t in terms], terms))
    height = sum(len(t) + 1 for _, _, t, _ in layout)
    fig, ax = plt.subplots(figsize=(9.5, 0.17 * height + 1.2))
    y, ticks = 0.0, []
    T = max(s.total_len for _, s, _, _ in layout)
    for name, s, order, terms in layout:
        top = y
        ax.add_patch(plt.Rectangle((-0.5, y - 0.4), s.total_len, 0.8, color=GRID, lw=0))
        ticks.append((y, name))
        for term in order:
            y += 1
            kind, positions = terms[term]
            colour = next(c for _, members, c in TERM_GROUPS if term in members)
            if kind.startswith("tuples"):
                n = int(kind.split()[1])
                positions = sorted({p + j for p in positions for j in range(1, n + 1)})
            ax.broken_barh([(p - 0.45, 0.9) for p in positions], (y - 0.35, 0.7), color=colour, lw=0)
            ax.text(T + 0.8, y, term, va="center", fontsize=7.5, color=INK2)
        if s.model in GRU_HYBRIDS:
            ax.plot([s.history_len - 1] * 2, [top - 0.5, y + 0.5], ls="--", lw=0.9, color=INK2)
        y += 1.6
    ax.set_yticks([t for t, _ in ticks], [n for _, n in ticks])
    ax.set_ylim(y - 0.6, -1)
    ax.set_xlim(-1, T + 8)
    ax.set_xlabel("position in the window (0-based)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=c, label=g) for g, _, c in TERM_GROUPS], loc="lower center",
              bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_known(shares, firsts):
    """The share of windows whose true state is certain from the tokens read so far, by position, one panel per group
    of processes.  shares: {group: {process: array}}; firsts: {process: first token the own-prediction CE scores}."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(shares), figsize=(4.2 * len(shares), 3.2), squeeze=False)
    for ax, (group, curves) in zip(axes[0], shares.items()):
        for k, (proc, share) in enumerate(curves.items()):
            ax.plot(np.arange(len(share)), share, color=SERIES[k], label=proc)
            f = firsts[proc]
            ax.plot([f], [share[f - 1]], "o", ms=5, color=SERIES[k], mec=SURFACE, mew=1.2)
        ax.set_title(group)
        ax.set_xlabel("tokens read (position t: after x_0 .. x_t)")
        ax.set_ylim(-0.03, 1.03)
        if curves:
            ax.legend(fontsize=7.5, loc="lower right")
    axes[0][0].set_ylabel("share of windows whose state is known")
    fig.tight_layout()
    return fig


def plot_renewal_batches(rows):
    """The true states (zeros since the last 1) of every window of chosen updates of renewal processes:
    rows = [(run or Settings, [updates])], one row of panels each with its own colour scale.  Dashed lines: the first
    token the own-prediction CE scores and the first future token."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
    cols = max(len(u) for _, u in rows)
    fig, axes = plt.subplots(len(rows), cols, figsize=(3.1 * cols + 0.8, 2.7 * len(rows)), squeeze=False,
                             constrained_layout=True)
    for r, (run, updates) in enumerate(rows):
        s = _settings(run)
        n = len(s.process_object().pi)
        for c, update in enumerate(updates):
            ax = axes[r][c]
            _, st = windows_at(run, update, states=True)
            im = ax.imshow(st[:, :s.total_len], aspect="auto", cmap=blues, vmin=0, vmax=n - 1,
                           interpolation="nearest")
            for p in sorted({s.context + 1, s.history_len}):
                ax.axvline(p - 0.5, color=SERIES[1], lw=1, ls="--")
            ax.set_title(f"{process_name(stream_key(s))}, seed {s.seed}, update {update:,}", fontsize=8.5)
            ax.grid(False)
            if r == len(rows) - 1 or c >= len(rows[r + 1][1]):
                ax.set_xlabel("position")
        for c in range(len(updates), cols):
            axes[r][c].axis("off")
        axes[r][0].set_ylabel("window")
        fig.colorbar(im, ax=axes[r, :len(updates)].tolist(), shrink=0.9, label="zeros since the last 1")
    return fig


def plot_jumps(cases, stats_by_label, half=300, part="loss"):
    """The training loss (part="loss") or its CE part (part="ce": its next-token CE terms added) around chosen jumps
    (cases = [(run folder, update)]), and the batch's exact CE of the same tokens (the floor of the CE part), one panel
    each."""
    import matplotlib.pyplot as plt
    cols = min(3, len(cases))
    rows = math.ceil(len(cases) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), squeeze=False)
    for ax, (folder, update) in zip(axes.ravel(), cases):
        s = TR.read_settings(folder)
        h = _history(str(folder))
        terms = run_exact(folder, stats_by_label[stream_label(stream_key(s))])
        exact = sum(terms.values())
        a, b = max(1, update - half), min(len(h), update + half)
        u = np.arange(a, b + 1)
        y = (h["loss"] if part == "loss" else h[list(terms)].sum(axis=1)).to_numpy()
        ax.plot(u, y[a - 1:b], color=SERIES[0], lw=0.8,
                label="training loss (all its terms)" if part == "loss" else "the loss's CE terms")
        ax.plot(u, exact[a - 1:b], color=SERIES[1], lw=0.8, label="the batch's exact CE (its CE terms)")
        ax.axvline(update, color=INK2, lw=0.8, ls="--")
        ax.set_title(f"model {s.model}, {process_name(stream_key(s))}\nseed {s.seed}, update {update:,}", fontsize=8.5)
        ax.set_xlabel("update")
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    axes[0][0].set_ylabel("bits")
    axes[0][0].legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def plot_coverage(summaries, kind="test"):
    """Per group of processes (columns): the number of different words of L tokens seen in training (top) and the share
    of the held-out windows' words of L tokens that were seen in training at least once (bottom), against L.
    summaries: {group: {process: json summary}}."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(summaries), figsize=(4.2 * len(summaries), 5.6), squeeze=False, sharex="col")
    for c, (group, procs) in enumerate(summaries.items()):
        for k, (proc, info) in enumerate(procs.items()):
            wc = [r for r in info["word_counts"] if r["different"] is not None]
            axes[0][c].plot([r["L"] for r in wc], [r["different"] for r in wc], "o-", ms=3, color=SERIES[k], label=proc)
            cov = [r for r in info["coverage"] if r["kind"] == kind and "seen" in r]
            axes[1][c].plot([r["L"] for r in cov], [r["seen"] for r in cov], "o-", ms=3, color=SERIES[k], label=proc)
        axes[0][c].set_yscale("log")
        axes[0][c].set_title(group)
        axes[1][c].set_xlabel("word length L (tokens)")
        axes[1][c].set_ylim(-0.03, 1.03)
        if procs:
            axes[0][c].legend(fontsize=7.5)
    axes[0][0].set_ylabel("different words seen in training")
    axes[1][0].set_ylabel(f"share of {kind} words seen in training")
    fig.tight_layout()
    return fig


def plot_loss_vs_content(cases, stats_by_label):
    """For chosen runs and CE terms (cases = [(run folder, term)]): the term's training value against the batch's exact
    CE of the same tokens over the last 10,000 updates, one panel each; the line is equality."""
    import matplotlib.pyplot as plt
    cols = min(3, len(cases))
    rows = math.ceil(len(cases) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), squeeze=False)
    for ax, (folder, term) in zip(axes.ravel(), cases):
        s = TR.read_settings(folder)
        h = _history(str(folder))
        e = run_exact(folder, stats_by_label[stream_label(stream_key(s))])[term]
        last = len(h)
        y, x = h[term].to_numpy(float)[last - 10_000:], e[last - 10_000:]
        ax.plot(x, y, ".", ms=1.5, alpha=0.35, color=SERIES[0], rasterized=True)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"model {s.model}, {process_name(stream_key(s))}, seed {s.seed}\n{term}: correlation {r:.3f}",
                     fontsize=8.5)
        ax.set_xlabel("exact CE of the batch (bits)")
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")
    axes[0][0].set_ylabel("training CE of the term (bits)")
    fig.tight_layout()
    return fig


def _stream_job(args):
    key, folders, out = args
    torch.set_num_threads(1)
    try:
        summary = stream_statistics(key, [Path(f) for f in folders], out)
        return stream_label(key), summary["seconds"], None
    except Exception as error:                                        # reported, the other streams go on
        return stream_label(key), None, repr(error)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="check the saved sampler state of every saved run against the skip")
    st = sub.add_parser("stats", help="compute the statistics of every stream into runs/training_data/")
    st.add_argument("--workers", type=int, default=4)
    st.add_argument("--only", default=None, help="only streams whose label contains this text")
    st.add_argument("--force", action="store_true", help="recompute streams that are already saved")
    st.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.command == "check":
        for folder in run_folders():
            print(folder.relative_to(TR.RUNS), check_run(folder), flush=True)
        return
    out = Path(a.out or STATS)
    jobs = []
    for k, v in streams().items():
        if (a.only is not None and a.only not in stream_label(k)) or \
                (not a.force and (out / f"{stream_label(k)}.json").exists()):
            continue
        checks = [r for f in v for r in check_run(f).values()]          # only streams whose windows are verified
        if False in checks or all(r is None for r in checks):
            print(f"skipped {stream_label(k)}: saved sampler states {checks}", flush=True)
            continue
        jobs.append((k, [str(f) for f in v], str(out)))
    jobs.sort(key=lambda j: -max(last_update(f) for f in j[1]) * j[0][4] * len(TR.get_process(j[0][0], j[0][1]).pi))
    print(f"{len(jobs)} streams to compute with {a.workers} workers", flush=True)
    if a.workers <= 1:
        results = map(_stream_job, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1)
        results = pool.imap_unordered(_stream_job, jobs)
    for label, seconds, error in results:
        print(f"done {label}: " + (f"{seconds / 60:.1f} min" if error is None else f"FAILED {error}"), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
