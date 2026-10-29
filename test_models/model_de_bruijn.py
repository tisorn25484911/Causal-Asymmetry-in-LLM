"""
The de Bruijn hybrid: the state is read from the last R tokens (the "de Bruijn node"), the states start at the merged
quotient of the nodes, and a transition head G, (state, token) -> next state, is trained to agree with the reader.
(DEBRUIJN_HYBRID_IMPLEMENTATION_PLAN.md, "the plan", sections 1-9, rewritten for this folder.)

The model, in four steps
    1. The order R and the node.  The node at position t is the last R tokens up to and including x_t,
       Z_t = x_{t-R+1..t}.  For a process of Markov order R it is all that is needed to predict x_{t+1} (the plan's
       Proposition 1).  ORDER_RULE = "markov": R is the process's Markov order, the smallest R after which every word of
       R tokens the process can produce leaves its generating machine in one known state (renewal-N: N - 1; the flower:
       1).  ORDER_RULE = "bic" (model_de_bruijn_bic.py): R is estimated from the counting windows by the Bayesian
       information criterion of the plan's section 3.  Both values are kept in `info`.
    2. Counting and merging, before training, on 40,000 fresh windows of the process (a stream of their own, seed
       train_seed + 2,000,000; the training windows are not touched).  Every position t = R-1 .. T-2 counts
       n(Z_t, x_{t+1}).  The states start as the nodes counted at least 100 times and are merged in rounds: two classes
       merge when their next-token rows pass a Hoeffding test (level 0.001) and, for every token both have seen, their
       next nodes are in the same class (the plan's Theorems B and C, section 4).  A node counted fewer times, or never,
       takes the state of the frequent node with the longest common ending ("suffix back-off").  The result, the
       quotient, is the starting machine.  The plan keeps dense arrays over all V^R nodes; here only the nodes seen in
       the counting windows have rows (V^R is 2^59 for renewal-60), and the merge compares a class only with the classes
       that have the same next classes.  Both give the plan's result exactly (tests/test_de_bruijn.py).
    3. The network.  The reader: one row of K state scores per node (its scores depend on the window only through the
       node; a node not seen in the counting windows reads the row of its suffix representative).  The shared state
       table E and output layer W (causal_state_matrix.py).  The transition head G: a K x V x K table of next-state
       scores (TRANSITION = "gru": the hybrids' GRUCell and a scoring layer instead).  Warm start at the quotient
       (the plan's section 7): node u's row has 12 at its state and 0 elsewhere; row k of E holds the counted
       log-probabilities of state k's next token in its first V coordinates (small random numbers elsewhere) and
       W = [I 0]; G has 12 at the successor of every (state, token) (a token never counted keeps the state).
    4. Training: the same updates as every model of this folder (batch, learning rate, AdamW, clipping, temperature
       schedule, random noise in the state choice), with this loss at every position t = R-1 .. T-2 at once:
           loss = L_pred + beta L_use + weight (L_G + kappa L_E)            (all in bits; the plan's section 6)
       L_pred   CE of the reader's prediction of x_{t+1} from its state s_t;
       L_use    entropy of the reader's mean state probabilities over the positions (beta = 1/9600);
       L_G      CE of G( . | s_t, x_{t+1}) against the reader's next state s_{t+1} (held fixed);
       L_E      CE of the reader's next-state probabilities against G's prediction (held fixed) (kappa = 0.25).
       The weight is the plan's lambda; DEFAULTS sets it to 1.

Predictors
    node_table               the reader's state at every position t >= R - 1 (it reads the last R tokens).
    next_state_from_handoff  the reader picks the state after the H history tokens; then G steps from state to state on
                             each real future token (the plan's evaluation protocol).  The main predictor.
measures(): U (bits), the commutation entropy H[s_{t+1} | s_t, x_{t+1}] of the reader's own states (0 when its next
state is fixed by its state and the token); the defect (share of steps not going to the most common next state); and
the share of steps where G's choice equals the reader's next state (the plan's section 9).
"""
import json
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from causal_state_matrix import CausalStateMatrix
from training import LN2, prediction_loss

NAME = "de_bruijn"
DESCRIPTION = "the de Bruijn hybrid: states read from the last R tokens, started at the merged quotient (R = Markov order)"
DEFAULTS = {"weight": 1.0}
MAIN_PREDICTOR = "next_state_from_handoff"
ORDER_RULE = "markov"                  # "markov": R = the process's Markov order; "bic": estimated from the data
TRANSITION = "table"                   # "table": G is a K x V x K table of scores; "gru": GRUCell + scoring layer
BETA, KAPPA = 1 / 9600, 0.25           # the weights of L_use and L_E (the plan's)
GAIN = 12.0                            # warm start: the score of a node's state and of a successor
COUNT_WINDOWS = 40_000                 # windows counted before training
COUNT_SEED_OFFSET = 2_000_000          # their seed: train_seed + this
ALPHA, MIN_VISITS = 1e-3, 100          # the merge's Hoeffding level and the visits a node needs to be merged
EMISSION_FLOOR = 1e-3                  # counted probabilities below this start at this value
BIC_CELLS = 2 ** 22                    # the information criterion counts orders r with V^(r+1) <= this
MEASURES_ONE_PASS = True               # measures() are not means over windows: analysis.py runs them on all at once


# ================================================================================================ order and node
def markov_order(proc, max_order):
    """The smallest R such that every word of R tokens the process can produce, read from its long-run state
    distribution, leaves its generating machine in one known state.  None if no R <= max_order does."""
    M = np.asarray(proc.M)
    sets = {frozenset(np.flatnonzero(np.asarray(proc.pi) > 0).tolist())}
    for r in range(1, max_order + 1):
        reached = set()
        for states in sets:
            rows = M[sorted(states)]
            for x in range(M.shape[-1]):
                nxt = frozenset(np.flatnonzero(rows[:, :, x].sum(0) > 0).tolist())
                if nxt:
                    reached.add(nxt)
        sets = reached
        if all(len(s) == 1 for s in sets):
            return r
    return None


def context_counts(tokens, V, r, start):
    """counts[u, x]: the r tokens before position j form u (oldest most significant) and token j is x, over every
    position j >= start of every row (the plan's section 3)."""
    x = np.asarray(tokens)
    n_rows, T = x.shape
    ids = np.zeros((n_rows, T - start), dtype=np.int64)
    for j in range(r):
        ids = ids * V + x[:, start - r + j:T - r + j]
    flat = (ids * V + x[:, start:]).ravel()
    return np.bincount(flat, minlength=V ** (r + 1)).reshape(V ** r, V)


def estimate_order(tokens, V, max_order):
    """The Bayesian information criterion over orders 0 .. max_order (Csiszar and Shields 2000), every order scored on
    the same positions (those with max_order tokens of context).  Returns (best order, scores)."""
    n = tokens.shape[0] * (tokens.shape[1] - max_order)
    scores = []
    for r in range(max_order + 1):
        c = context_counts(tokens, V, r, max_order).astype(float)
        rows = c.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            loglik = np.where(c > 0, c * np.log2(c / rows), 0.0).sum()
        scores.append(loglik - 0.5 * (V - 1) * V ** r * math.log2(n))
    return int(np.argmax(scores)), np.array(scores)


def node_ids(tokens, R, V):
    """The node of every position: the last R tokens, oldest first, as one base-V number; -1 before position R - 1."""
    B, T = tokens.shape
    ids = torch.full((B, T), -1, dtype=torch.long, device=tokens.device)
    if T >= R:
        acc = torch.zeros((B, T - R + 1), dtype=torch.long, device=tokens.device)
        for j in range(R):
            acc = acc * V + tokens[:, j:T - R + 1 + j]
        ids[:, R - 1:] = acc
    return ids


# ================================================================================================ counting and merging
def unique(a, return_index=False, return_counts=False):
    """np.unique by sorting.  (numpy 2's hash-based unique is extremely slow on the node ids of long orders, which are
    large numbers whose low digits are mostly 0.)"""
    a = np.asarray(a).ravel()
    order = np.argsort(a, kind="stable")
    s = a[order]
    first = np.ones(len(s), dtype=bool)
    first[1:] = s[1:] != s[:-1]
    out = [s[first]]
    if return_index:
        out.append(order[first])                       # the first occurrence of each value
    if return_counts:
        out.append(np.diff(np.append(np.flatnonzero(first), len(s))))
    return out[0] if len(out) == 1 else tuple(out)


class SuffixBackoff:
    """For any node, the frequent node with the longest common ending (the most visited one among ties; frequent nodes
    stand for themselves): the plan's suffix_representative, for node ids given on demand instead of all V^R."""

    def __init__(self, frequent, visits, R, V):
        self.frequent = np.asarray(frequent, dtype=np.int64)          # sorted
        order = self.frequent[np.argsort(-np.asarray(visits), kind="stable")]
        self.levels = []                                               # (V^keep, sorted endings, their node)
        for keep in range(R - 1, -1, -1):
            m = V ** keep
            endings, first = unique(order % m, return_index=True)     # first occurrence = most visited
            self.levels.append((m, endings, order[first]))

    def __call__(self, nodes):
        shape = np.shape(nodes)
        nodes = np.asarray(nodes, dtype=np.int64).ravel()
        rep = np.full(nodes.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.frequent, nodes).clip(max=len(self.frequent) - 1)
        own = self.frequent[pos] == nodes
        rep[own] = nodes[own]
        for m, endings, best in self.levels:
            todo = np.flatnonzero(rep < 0)
            if len(todo) == 0:
                break
            e = nodes[todo] % m
            p = np.searchsorted(endings, e).clip(max=len(endings) - 1)
            hit = endings[p] == e
            rep[todo[hit]] = best[p[hit]]
        return rep.reshape(shape)


def compatible(c1, n1, c2, n2, alpha):
    """ALERGIA-style Hoeffding test on every token: are two counted rows plausibly equal?"""
    width = math.sqrt(0.5 * math.log(2 * len(c1) / alpha)) * (n1 ** -0.5 + n2 ** -0.5)
    return bool(np.all(np.abs(c1 / n1 - c2 / n2) <= width))


@dataclass
class Quotient:
    nodes: np.ndarray        # [n] the nodes seen in the counting windows, sorted
    node_state: np.ndarray   # [n] the state of each (rare ones through back-off)
    rows: np.ndarray         # [k, V] next-token rows from the pooled counts
    successor: np.ndarray    # [k, V] next state, -1 where the token was never counted
    visits: np.ndarray       # [k] counted positions in each state
    sizes: list              # the number of classes after each round
    backoff: SuffixBackoff

    @property
    def k(self):
        return len(self.rows)


def count_nodes(windows, R, V):
    """Counts n(u, x) at positions t = R-1 .. T-2: (nodes seen at t = R-1 .. T-1, sorted; their counts [n, V])."""
    ids = node_ids(torch.as_tensor(windows), R, V).numpy()
    x = np.asarray(windows)
    T = x.shape[1]
    keys, n = unique(ids[:, R - 1:T - 1] * V + x[:, R:T], return_counts=True)
    nodes = unique(ids[:, R - 1:])
    counts = np.zeros((len(nodes), V), dtype=np.int64)
    counts[np.searchsorted(nodes, keys // V), keys % V] = n
    return nodes, counts


def merge_downwards(nodes, counts, R, V, alpha=ALPHA, min_visits=MIN_VISITS, max_rounds=None):
    """Bottom-up merging from the node partition: the plan's merge_downwards (section 4), with the same result.

    Only nodes counted at least `min_visits` times are merged; a successor that is not frequent is read through its
    suffix representative.  A round compares classes by (pooled row, successor classes from the previous round).
    Within a round each class, most visited first, joins the earliest-made class that has the same successor on every
    token both have counted and a compatible row; otherwise it starts a new class.  Rounds repeat until nothing merges.
    Every seen node then takes the state of its suffix representative."""
    nodes, counts = np.asarray(nodes, dtype=np.int64), np.asarray(counts)
    visits = counts.sum(axis=1)
    keep = visits >= min_visits
    frequent, f_counts = nodes[keep], counts[keep]
    if len(frequent) == 0:
        raise ValueError("no node reaches min_visits; count more windows")
    backoff = SuffixBackoff(frequent, visits[keep], R, V)
    shift = V ** (R - 1)
    succ_node = backoff((frequent[:, None] % shift) * V + np.arange(V)[None, :])      # delta(u, x), through back-off
    succ_index = np.searchsorted(frequent, succ_node)                                   # (a frequent node's position)
    seen = f_counts > 0
    cls = np.arange(len(frequent))
    sizes = [len(frequent)]

    def pooled_and_successors():
        k = cls.max() + 1
        pooled = np.zeros((k, V))
        np.add.at(pooled, cls, f_counts)
        succ = np.full((k, V), -1)
        for i in range(len(frequent)):                     # in node order, as the plan (the last member written wins)
            for x in np.flatnonzero(seen[i]):
                succ[cls[i], x] = cls[succ_index[i, x]]
        return pooled, succ

    for _ in range(10 * R + 10 if max_rounds is None else max_rounds):
        pooled, succ = pooled_and_successors()
        k = len(pooled)
        totals = pooled.sum(axis=1)
        reps, rep_succ, by_tuple, partial = [], [], {}, []
        new_label = np.empty(k, dtype=int)
        for c in np.argsort(-totals, kind="stable"):
            s = succ[c]
            if (s >= 0).all():                             # every token seen: same tuple, or a rep with gaps
                candidates = by_tuple.get(tuple(s), []) + [r for r in partial
                                                           if np.all(rep_succ[r][rep_succ[r] >= 0] == s[rep_succ[r] >= 0])]
                candidates.sort()
            else:                                          # gaps of its own: check every rep
                candidates = [r for r in range(len(reps))
                              if np.all(s[(s >= 0) & (rep_succ[r] >= 0)] == rep_succ[r][(s >= 0) & (rep_succ[r] >= 0)])]
            target = next((r for r in candidates
                           if compatible(pooled[c], totals[c], pooled[reps[r]], totals[reps[r]], alpha)), None)
            if target is None:
                new_label[c] = len(reps)
                (by_tuple.setdefault(tuple(s), []) if (s >= 0).all() else partial).append(len(reps))
                reps.append(c)
                rep_succ.append(s)
            else:
                new_label[c] = target
        if len(reps) == k:
            break
        cls = new_label[cls]
        sizes.append(len(reps))
    pooled, succ = pooled_and_successors()
    node_state = cls[np.searchsorted(frequent, backoff(nodes))]
    return Quotient(nodes=nodes, node_state=node_state, rows=pooled / pooled.sum(axis=1, keepdims=True),
                    successor=succ, visits=pooled.sum(axis=1), sizes=sizes, backoff=backoff)


@lru_cache(maxsize=16)
def _starting_point(key):
    """Order, counts and quotient of one process and window size (cached: a saved run is rebuilt when it is loaded)."""
    import training as TR
    d = json.loads(key)
    proc = TR.get_process(d["process"], json.dumps(d["params"], sort_keys=True))
    V, H, T = proc.vocab_size, d["history_len"], d["total_len"]
    windows = proc.sampler(d["seed"]).draw(COUNT_WINDOWS, T).numpy()
    max_bic = min(H - 1, max(r for r in range(T) if V ** (r + 1) <= BIC_CELLS))
    bic, _ = estimate_order(windows, V, max_bic)
    markov = markov_order(proc, H)
    if d["rule"] == "markov":
        if markov is None:
            raise ValueError(f"{proc.label} has no Markov order <= history_len = {H}")
        R = markov
    else:
        R = max(1, bic)
    if V ** (R + 1) >= 2 ** 62:
        raise ValueError(f"node ids of order {R} over {V} tokens do not fit in 64 bits")
    nodes, counts = count_nodes(windows, R, V)
    quotient = merge_downwards(nodes, counts, R, V)
    info = dict(order=R, markov_order=markov, bic_order=bic, bic_max_order=max_bic, counted_positions=int(counts.sum()),
                nodes_seen=len(nodes), frequent_nodes=quotient.sizes[0], merge_rounds=list(quotient.sizes),
                quotient_states=quotient.k)
    return R, quotient, info


def starting_point(cfg, rule):
    key = json.dumps(dict(process=cfg.process, params=cfg.params, history_len=cfg.history_len,
                          total_len=cfg.total_len, seed=cfg.train_seed + COUNT_SEED_OFFSET, rule=rule), sort_keys=True)
    return _starting_point(key)


# ================================================================================================ the network
class NodeTable(nn.Module):
    """The reader: one row of K state scores per node seen in the counting windows; any other node reads the row of
    its suffix representative.  Its scores depend on the window only through the node (the plan's condition C1)."""

    def __init__(self, quotient, R, V, K):
        super().__init__()
        self.R, self.V = R, V
        self.register_buffer("nodes", torch.as_tensor(quotient.nodes, dtype=torch.long))
        self.table = nn.Parameter(torch.zeros(len(quotient.nodes), K))
        self.backoff = quotient.backoff

    def rows(self, ids):
        """The table row of every node id (B, T); positions without a node (-1) read row 0 and are never scored."""
        flat = ids.clamp(min=0)
        pos = torch.searchsorted(self.nodes, flat).clamp(max=len(self.nodes) - 1)
        seen = self.nodes[pos] == flat
        if not seen.all():
            rep = torch.as_tensor(self.backoff(flat[~seen].cpu().numpy()), device=ids.device)
            pos = pos.clone()
            pos[~seen] = torch.searchsorted(self.nodes, rep)
        return pos

    def forward(self, tokens):
        return self.table[self.rows(node_ids(tokens, self.R, self.V))]                 # (B, T, K)


class Model(nn.Module):
    order_rule = ORDER_RULE

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        K, S, V = cfg.n_states, cfg.state_dim, cfg.vocab_size
        if S < V:
            raise ValueError("state_dim must be at least vocab_size: the warm start writes log-probabilities into E")
        self.order, self.quotient, self.info = starting_point(cfg, self.order_rule)
        if self.order > cfg.history_len:
            raise ValueError("the order must not exceed history_len (the plan's Proposition 1)")
        if self.quotient.k > K:
            raise ValueError(f"the quotient has {self.quotient.k} states but the table has {K} rows (n_states)")
        self.reader = NodeTable(self.quotient, self.order, V, K)
        self.causal_head = CausalStateMatrix(K, S, V)
        if TRANSITION == "table":
            self.transition_table = nn.Parameter(torch.zeros(K, V, K))
        else:
            self.gru_cell = nn.GRUCell(V, S)
            self.gru_selector = nn.Linear(S, K)
        self.warm_start()

    @torch.no_grad()
    def warm_start(self, gain=GAIN):
        """Selector, output and transition head set to the quotient (the plan's initialise_from_quotient)."""
        q, K, V, S = self.quotient, self.cfg.n_states, self.cfg.vocab_size, self.cfg.state_dim
        k = q.k
        self.reader.table.zero_()
        self.reader.table[torch.arange(len(q.nodes)), torch.as_tensor(q.node_state)] = gain
        rows = np.full((K, V), 1.0 / V)
        rows[:k] = np.clip(q.rows, EMISSION_FLOOR, None)
        rows /= rows.sum(axis=1, keepdims=True)
        E = torch.zeros(K, S)
        E[:, :V] = torch.as_tensor(np.log(rows), dtype=torch.float32)
        E[:, V:] = 0.1 * torch.randn(K, S - V)                    # distinct codes for the GRU form
        self.causal_head.state_matrix.copy_(E)
        self.causal_head.emission.weight.zero_()
        self.causal_head.emission.weight[:, :V] = torch.eye(V)
        self.causal_head.emission.bias.zero_()
        successor = q.successor.copy()
        for s in range(k):                                         # tokens never counted: stay
            successor[s, successor[s] < 0] = s
        if TRANSITION == "table":
            self.transition_table.zero_()
            for s in range(k):
                self.transition_table[s, torch.arange(V), torch.as_tensor(successor[s])] = gain
            for s in range(k, K):
                self.transition_table[s, :, s] = gain
        else:
            with torch.enable_grad():
                self.fit_gru_transition(successor)
        self.successor = successor

    def fit_gru_transition(self, successor, steps=1500, lr=1e-2):
        """TRANSITION == "gru": fit the cell and its scoring layer to the quotient's successor map, E fixed."""
        k, V = successor.shape
        states, tokens = torch.arange(k).repeat_interleave(V), torch.arange(V).repeat(k)
        target = torch.as_tensor(successor.reshape(-1))
        onehot = F.one_hot(states, self.cfg.n_states).float()
        optimizer = torch.optim.Adam(list(self.gru_cell.parameters()) + list(self.gru_selector.parameters()), lr=lr)
        for _ in range(steps):
            loss = F.cross_entropy(self.transition_logits(onehot, tokens), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            return float((self.transition_logits(onehot, tokens).argmax(-1) == target).float().mean())

    def choose(self, scores, tau):
        """The shared state choice (causal_state_matrix.py), and log softmax((scores + noise) / tau) for L_E."""
        perturbed = scores
        if self.training and self.cfg.gumbel:
            c = torch.finfo(scores.dtype).eps
            u = torch.rand_like(scores).clamp(min=c, max=1 - c)
            perturbed = scores - torch.log(-torch.log(u))
        return self.causal_head.select(perturbed, tau, noise=False), (perturbed / tau).log_softmax(dim=-1)

    def transition_logits(self, onehot, observed_token):
        """Scores of G( . | state, x) for (straight-through) state vectors (..., K) and tokens (...)."""
        if TRANSITION == "table":
            K, V = self.cfg.n_states, self.cfg.vocab_size
            lead = onehot.shape[:-1]
            z = onehot.new_zeros(*lead, V, K)
            z.scatter_(-2, observed_token[..., None, None].expand(*lead, 1, K), onehot.unsqueeze(-2))
            return z.reshape(*lead, V * K) @ self.transition_table.permute(1, 0, 2).reshape(V * K, K)
        symbol = F.one_hot(observed_token, self.cfg.vocab_size).to(onehot.dtype)
        proposal = self.gru_cell(symbol.reshape(-1, symbol.shape[-1]),
                                 (onehot @ self.causal_head.state_matrix).reshape(-1, self.cfg.state_dim))
        return self.gru_selector(proposal).reshape(*onehot.shape[:-1], self.cfg.n_states)

    # ---- training
    def loss(self, window, tau, weight):
        R, T, K = self.order, window.shape[1], self.cfg.n_states
        sel, log_q = self.choose(self.reader(window), tau)
        now, nxt = slice(R - 1, T - 1), slice(R, T)
        targets = window[:, nxt]                                                     # x_{t+1}
        g_logits = self.transition_logits(sel.onehot[:, now], targets)               # G( . | y_t, x_{t+1})
        terms = {"pred": prediction_loss(self.causal_head.emission(sel.representation[:, now]), targets)}
        p_bar = sel.probs[:, now].reshape(-1, K).mean(dim=0)
        terms["use"] = -(p_bar * torch.log2(p_bar.clamp_min(1e-12))).sum()
        terms["fit"] = F.cross_entropy(g_logits.reshape(-1, K), sel.ids[:, nxt].detach().reshape(-1)) / LN2
        terms["commit"] = -(g_logits.detach().softmax(-1) * log_q[:, nxt]).sum(-1).mean() / LN2
        loss = terms["pred"] + BETA * terms["use"] + weight * (terms["fit"] + KAPPA * terms["commit"])
        return loss, terms, sel.ids

    # ---- evaluation (no noise)
    def step(self, ids, tokens):
        """The fixed step rule of G: (state ids, tokens) -> next state ids (highest score)."""
        return self.transition_logits(F.one_hot(ids, self.cfg.n_states).float(), tokens).argmax(-1)

    def step_rules(self):
        return {"next_state_from_handoff": self.step}

    @torch.no_grad()
    def reader_states(self, tokens, tau):
        """The reader's scores (N, T, K) and states (N, T) without noise, in batches of 256."""
        scores, ids = [], []
        for batch in tokens.split(256):
            s = self.reader(batch)
            scores.append(s)
            ids.append(self.choose(s, tau)[0].ids)
        return torch.cat(scores), torch.cat(ids)

    @torch.no_grad()
    def handoff_states(self, history, tau):
        """The reader's state after the history: where G starts."""
        return self.reader_states(history, tau)[1][:, -1]

    @torch.no_grad()
    def predict(self, tokens, tau):
        R, H = self.order, self.cfg.history_len
        T = tokens.shape[1]
        scores, ids = self.reader_states(tokens, tau)
        E, W = self.causal_head.state_matrix, self.causal_head.emission
        state, states, logits = ids[:, H - 1], [], []
        for k in range(H, T):
            states.append(state)
            logits.append(W(E[state]))
            state = self.step(state, tokens[:, k])
        return dict(predictors={"node_table": dict(first=R, logits=W(E[ids[:, R - 1:T - 1]]), states=ids[:, R - 1:T]),
                                "next_state_from_handoff": dict(first=H, logits=torch.stack(logits, 1),
                                                                states=torch.stack(states, 1))},
                    handoff_scores=scores[:, H - 1])

    @torch.no_grad()
    def measures(self, tokens, tau, prediction):
        """U (bits), the defect and G's agreement with the reader, on the reader's own states (the plan's
        commutation_report)."""
        K, V, R = self.cfg.n_states, self.cfg.vocab_size, self.order
        ids = prediction["predictors"]["node_table"]["states"]
        s0, x1, s1 = ids[:, :-1], tokens[:, R:], ids[:, 1:]
        joint = torch.bincount(((s0 * V + x1) * K + s1).reshape(-1), minlength=K * V * K).reshape(K * V, K).double()
        joint /= joint.sum()
        pair = joint.sum(dim=1)
        live = pair > 0
        cond = joint[live] / pair[live, None]
        U = -(joint[live] * torch.log2(cond.clamp_min(1e-300))).sum()
        agree = torch.cat([(self.step(a, b) == c).float() for a, b, c in
                           zip(s0.reshape(-1).split(65536), x1.reshape(-1).split(65536), s1.reshape(-1).split(65536))])
        return {"U_bits": max(0.0, float(U)), "defect": max(0.0, float(1 - joint.max(dim=1).values.sum())),
                "head_agreement": float(agree.mean())}
