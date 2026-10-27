class WindowSampler:
    """
    Fresh windows of a process, each started from the long-run (stationary) distribution of the generating machine's
    states.  This is the training data of section 16 of hybrid_model_test.ipynb (checks/section16_experiments.Sampler)
    for any process: with the same machine and seed it draws exactly the same windows.

        draw(batch_size, total_len)               -> tokens, a torch.long tensor (batch_size, total_len)
        draw(batch_size, total_len, states=True)  -> tokens, states
            states[:, j] is the generating machine's state BEFORE token j, and states[:, total_len] the state after
            the last token (numpy, (batch_size, total_len + 1)).  Models never see the states; analyses use them.

    Every call moves the random-number stream on, so no window is drawn twice.
    """

    def __init__(self, M, pi, seed):
        self.M = np.asarray(M, dtype=float)
        self.pi = np.asarray(pi, dtype=float)
        self.cdf = self.M.reshape(len(self.M), -1).cumsum(axis=-1)       # over (next state, token) pairs
        self.cdf /= self.cdf[:, -1:]
        self.V = self.M.shape[-1]
        self.rng = np.random.default_rng(seed)

    def draw(self, batch_size, total_len, states=False):
        import torch
        state = self.rng.choice(len(self.M), size=batch_size, p=self.pi)
        tokens = np.empty((batch_size, total_len), dtype=np.int64)
        st = np.empty((batch_size, total_len + 1), dtype=np.int64)
        st[:, 0] = state
        for j in range(total_len):
            u = self.rng.random(batch_size)
            flat = (self.cdf[state] <= u[:, None]).sum(axis=-1)          # the (next state, token) pair drawn
            state, token = flat // self.V, flat % self.V
            tokens[:, j] = token
            st[:, j + 1] = state
        out = torch.from_numpy(tokens)
        return (out, st) if states else out


class Process:
    """
    One process of PROCESS (part 1) with its parameters, as used by training.py and analysis.py.

        T, E, M          the generating machine as part 1 writes it: T[s, s'] = P(s' | s), E[s, s', x] = P(x | s, s'),
                         and the joint map M[s, s', x] = P(token x and next state s' | state s)
        pi               the long-run (stationary) distribution of its states
        vocab_size       the number of different tokens
        state_names      a name for each state: 'a<c>' for a renewal process (c 0s since the last 1), else 's<i>'
        is_epsilon_machine
                         True when the generating machine is its own epsilon-machine: from every state each token leads
                         to one next state (unifilar), every state keeps recurring, and no two states have the same
                         future.  Then its states ARE the causal states, the window sampler's `states` are the true
                         causal states, and the analyses that compare learned states with true ones can be used.
        sampler(seed)    a WindowSampler of this process
        exact_probs(x)   P(x_j | x_0 .. x_{j-1}) for every position of every window (the true predictive
                         probabilities, by the forward algorithm on M started from pi), shape (N, T, V)
        emission()       P(next token | state), shape (states, V)
        analytics()      the analytical results of part 2's `generator`: number of causal states, statistical
                         complexity C_mu, entropy rate h, excess entropy E, crypticity chi = C_mu - E, ... (of the
                         backward view for a backward process)

    The time-reversed process: give "direction": "backward" in params.  The generating machine is then the reverse
    epsilon-machine M- (part 1's reverse_machine of the forward epsilon-machine, as part 2's generator.reverse() uses):
    its windows are windows of the time-reversed process, read left to right, and its states are the reverse causal
    states.
    """

    def __init__(self, name, params):
        if name not in PROCESS:
            raise ValueError(f"unknown process {name!r}; known: {sorted(PROCESS)}")
        spec = PROCESS[name]
        self.name, self.params = name, dict(params)
        self.direction = self.params.get("direction", "forward")
        if self.direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")
        self.T = np.asarray(spec["transition"](params), dtype=float)
        self.E = np.asarray(spec["state_map"](params), dtype=float)
        self.vocab_size = int(spec["vocab_size"](params))
        check_map(self.T, self.E, self.vocab_size)
        if self.direction == "backward":
            E_live = live_map(self.T, self.E)                       # as generator.__init__: drop dead edges and
            keep = closed_class(self.T)                             # transient states, then M+ and its reverse M-
            fwd = epsilon_machine(self.T[np.ix_(keep, keep)], E_live[np.ix_(keep, keep)])
            bwd = reverse_machine(fwd["T"], fwd["E"])
            if bwd["infinite"] or bwd["truncated"]:
                raise ValueError(f"the reverse epsilon-machine of {name} is not finite")
            self.T, self.E = np.asarray(bwd["T"], dtype=float), np.asarray(bwd["E"], dtype=float)
        self.M = joint_map(self.T, self.E)
        self.pi = stationary_dist(self.T)
        n = len(self.T)
        forward_renewal = name == "renewal" and self.direction == "forward"
        self.state_names = [f"a{s}" for s in range(n)] if forward_renewal else [f"s{s}" for s in range(n)]
        self.is_epsilon_machine = bool(is_unifilar(self.T, self.E) and len(closed_class(self.T)) == n
                                       and epsilon_machine(self.T, self.E)["n_states"] == n)

    @property
    def label(self):
        return f"{self.name}-{len(self.T)}" + ("-backward" if self.direction == "backward" else "")

    def sampler(self, seed):
        return WindowSampler(self.M, self.pi, seed)

    def exact_probs(self, tokens):
        return filter_probs(self.M, self.pi, np.asarray(tokens))

    def emission(self):
        return emission_dist(self.T, self.E)

    def analytics(self):
        params = {k: v for k, v in self.params.items() if k != "direction"}
        g = generator(self.name, n_sample=1, seq_len=2, params=params)
        if self.direction == "backward":
            g.reverse()
        return g.summary()
