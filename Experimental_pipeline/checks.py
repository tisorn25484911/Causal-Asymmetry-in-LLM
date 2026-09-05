#!/usr/bin/env python
"""
Self-checks for the pipeline.  Every one can fail loudly.

    python Experimental_pipeline/checks.py            # fast: processes only, ~20 s
    python Experimental_pipeline/checks.py --full     # + the extraction gate, ~5 min

The fast checks hold the SAMPLERS against their own closed forms: if a generator
change ever silently invalidates a formula in processes.py, these are what catch
it.  The full check trains one real model and asserts that the discrete
bottleneck's S_emp lands on C+; it is the gate that says the pipeline as a whole
still measures what it claims to.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import processes as P                                   # noqa: E402

FAIL = []


def _chk(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<50} {got:>9.5f} vs {want:>9.5f}")
    if not ok:
        FAIL.append(name)


def _chk_mat(name, got, want, tol):
    e = float(np.abs(np.asarray(got) - np.asarray(want)).max())
    ok = e <= tol
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<50} max cell err {e:.5f}")
    if not ok:
        FAIL.append(name)


def _states(tokens, kind, params):
    """Forward causal state of each position, recoverable from the token alone."""
    if kind == "coin":
        return (tokens == 1).astype(int)          # state 1 emits token 1
    n = params["n"]
    return np.where(tokens < n, tokens + 1, 0)    # 0 = "a roll just happened"


def check_processes(num_samples=200, seq_len=500):
    """
    Samplers vs closed forms.

    Transitions are counted PER SEQUENCE.  Ravelling them together joins the last
    token of one sequence to the first of the next, which for flower is a
    selection->selection step the process cannot produce (odd length, every
    sequence starts on a selection) -- that spurious mass inflates H(next|state)
    by ~0.02 bits and would read as a generator bug.
    """
    print("\n-- generators vs their own closed forms --")
    d = P.generate("coin", {"p": .3, "q": .4}, 5, 300, 250, np.random.default_rng(0))
    print(f"  {'OK  ' if d.shape == (5, 301) else 'FAIL'} shape is (num_samples, seq_len+1)"
          f"{'':<19} {d.shape}")
    if d.shape != (5, 301):
        FAIL.append("shape")
    a = P.generate("coin", {"p": .3, "q": .4}, 3, 100, 250, np.random.default_rng(7))
    b = P.generate("coin", {"p": .3, "q": .4}, 3, 100, 250, np.random.default_rng(7))
    print(f"  {'OK  ' if np.array_equal(a, b) else 'FAIL'} same seed -> byte-identical")
    if not np.array_equal(a, b):
        FAIL.append("determinism")

    cases = ([("coin", dict(p=p, q=q)) for p, q in ((.1, .9), (.3, .4), (.55, .55))]
             + [("flower", dict(n=n, m=m)) for n, m in ((2, 6), (4, 2))])
    for kind, params in cases:
        if kind == "flower":
            params["dice_probs"] = P.make_dice(params["n"], params["m"], 42)
            T_th = P.flower_transition_matrix(params["n"])
            C = P.flower_complexity(params["n"], params["m"], params["dice_probs"])[0]
            H = P.flower_entropy_rate(params["n"], params["m"], params["dice_probs"])
            occ = P.causal_state_occupancy("flower", "forward", n=params["n"])
            V = params["n"] + params["m"]
        else:
            T_th = P.coin_transition_matrix(params["p"], params["q"])
            C = P.coin_complexity(params["p"], params["q"])[0]
            H = P.entropy_rate_coin(params["p"], params["q"])
            occ = P.causal_state_occupancy("coin", "forward", p=params["p"], q=params["q"])
            V = 3

        seqs = P.generate(kind, params, num_samples, seq_len, 250,
                          np.random.default_rng(1))
        K = T_th.shape[0]
        Tc, cnt = np.zeros((K, K)), np.zeros(K)
        emis = np.zeros((K, V)); tot = 0
        for row in seqs:
            st = _states(row, kind, params)
            for i in range(len(st) - 1):
                Tc[st[i], st[i + 1]] += 1
                emis[st[i], row[i + 1]] += 1
                tot += 1
            for s in st:
                cnt[s] += 1
        tag = f"{kind} {params.get('p', params.get('n'))},{params.get('q', params.get('m'))}"
        print(f"  [{tag}]  {seqs.size:,} tokens")
        _chk_mat(f"    empirical T vs closed form", Tc / Tc.sum(1, keepdims=True), T_th, 1e-2)
        _chk_mat(f"    occupancy vs causal_state_occupancy", cnt / cnt.sum(), occ, 1e-2)
        _chk(f"    H(occupancy) vs C+", P.entropy_bits(cnt / cnt.sum()), C, 1e-2)
        Hemp = sum((emis[s].sum() / tot) * P.entropy_bits(emis[s] / emis[s].sum())
                   for s in range(K) if emis[s].sum() > 0)
        _chk(f"    H(next|state) vs entropy rate", Hemp, H, 1e-2)

    # entropy(occupancy) == C is an identity, so it holds exactly
    dp = P.make_dice(2, 6, 42)
    _chk("identity: H(bwd occupancy) == C-",
         P.entropy_bits(P.causal_state_occupancy("flower", "backward", n=2, m=6,
                                                 dice_probs=dp)),
         P.flower_complexity(2, 6, dp)[1], 1e-9)


def check_extraction(tol=0.02):
    """
    THE GATE.  Train one real discrete model and assert S_emp lands on C+.

    If this fails, something in the geometry or the bottleneck moved that the
    generator checks cannot see -- stop and find it before trusting any figure.
    """
    import torch, torch.utils.data as tud
    import training as T, extraction as E
    from config import CONFIG, coin_spec

    print("\n-- extraction gate (trains one model) --")
    cfg, spec = dict(CONFIG), coin_spec(dict(CONFIG), 0.3, 0.4)
    ds = P.SequenceDataset(P.generate(spec["kind"], spec["params"],
                                      cfg["num_samples"], cfg["seq_len"],
                                      cfg["burn_in"], np.random.default_rng(0)))
    tr, te = T.split_loader(ds, cfg["batch"], cfg["test_ratio"], seed=0)
    ana = tud.DataLoader(ds, batch_size=cfg["ana_batch"], shuffle=False)
    T.set_seed(0); torch.manual_seed(0)
    with T.quiet(True):
        rec = T.train_model(tr, "discrete", val_loader=te, num_token=spec["num_token"],
                            d_model=cfg["d_model"], max_len=ds.seq_len,
                            max_epochs=cfg["max_epochs"], lr=cfg["lr"], mode="forward",
                            n_layers=cfg["n_layers"], weight_decay=cfg["weight_decay"],
                            accelerator=cfg["accelerator"],
                            n_states=spec["n_states"], state_dim=spec["state_dim"],
                            tau=spec["tau"], usage_beta=spec["usage_beta"])
    T.to_cpu_for_analysis(rec.model)
    r = E.causal_state_report(rec.model, ana, min_pos=cfg["state_min_pos"])
    _chk(f"discrete forward S_emp vs C+ (k={r['n_states_used']}, true 2)",
         r["S_emp"], spec["C_plus"], tol)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also run the extraction gate")
    a = ap.parse_args(argv)
    check_processes()
    if a.full:
        check_extraction()
    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
