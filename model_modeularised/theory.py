"""
What a trained model is compared against: the closed forms of a process, computed
from HMM_processes.py for ANY process registered in HMM_processes.PROCESS.

    process_theory(process, params, burn_in, horizon)   both arms of one process

The arms come from process_generator.generator, which builds them with
HMM_processes (epsilon_machine for M+, reverse_machine for M- = U(T(M+)); Ellison,
Mahoney & Crutchfield 2009), and the per-position floor from
HMM_processes.conditional_entropies directly:

    generator_hmm         the generating HMM exactly as HMM_processes defines it,
                          M[s, s', x] = P(x, s' | s) with its stationary
                          distribution (may be non-unifilar, e.g. sns)
    forward / backward    extraction.theory_from_generator of each view: the
                          epsilon-machine (T, E) as a (next_state, emission_probs)
                          machine, its stationary occupancy, k and C
    C_plus, C_minus       statistical complexity of M+ and M-
    entropy_rate          h, the same in both directions
    excess_entropy        E = I[past; future]
    crypticity_*          chi+ = H[S+ | S-], chi- = H[S- | S+]
    floor                 h(t) = H[X_t | X_0 .. X_{t-1}], t = 1..horizon: the
                          expected cross-entropy of the EXACT model with t tokens
                          of context -- what a trained model's per-position CE is
                          compared with.  sum_t (h(t) - h) = E (Crutchfield &
                          Feldman 2003), reported as `floor_excess` as a check.
    warnings              what makes the learned-vs-theory comparison unreliable:
                          an infinite or truncated machine, or causal states whose
                          next-token distributions coincide (emission-row matching,
                          extraction.assign_states_by_emission, cannot tell those
                          apart, so "discovered" undercounts).
"""
import numpy as np

from extraction import theory_from_generator
from HMM_processes import conditional_entropies, is_unifilar, joint_map, stationary_dist
from process_generator import generator

# below this total-variation distance two theoretical states count as sharing a
# next-token distribution for emission-row matching (tests use 0.05 too)
SAME_ROW_TV = 0.05
# above this many states a machine is summarised, not drawn
DRAWABLE_K = 20


def process_theory(process, params, burn_in=100, horizon=300) -> dict:
    gen = generator(process, 1, horizon, params, burn_in=burn_in)
    floor_full = conditional_entropies(gen.gen_transition, gen.gen_state_map, horizon + 1)
    # the HMM exactly as HMM_processes defines it (live, recurrent part), before any
    # minimisation: it can be non-unifilar (sns) or larger than M+ (merged states)
    generator_hmm = {"M": joint_map(gen.gen_transition, gen.gen_state_map),
                     "stationary": stationary_dist(gen.gen_transition),
                     "unifilar": bool(is_unifilar(gen.gen_transition, gen.gen_state_map))}
    views = {}
    for arm in ("forward", "backward"):
        if arm == "backward":
            gen.reverse()
        th = theory_from_generator(gen)
        th.update(crypticity=float(gen.crypticity), truncated=bool(gen.truncated),
                  infinite=bool(gen.infinite), C_shift=float(gen.C_shift))
        views[arm] = th

    warnings = []
    for arm, th in views.items():
        if th["infinite"] or th["truncated"]:
            warnings.append(f"{arm} machine is {'infinite' if th['infinite'] else 'truncated'}: "
                            f"the {th['true_k']} states are a finite approximation")
        if np.isfinite(th["min_emission_tv"]) and th["min_emission_tv"] < SAME_ROW_TV:
            warnings.append(f"{arm}: two causal states share a next-token distribution "
                            f"(min TV {th['min_emission_tv']:.3f}), so 'discovered' undercounts")
    h = float(gen.entropy_rate)
    return {
        "process": process, "params": dict(params), "burn_in": burn_in,
        "vocab_size": int(gen.vocab_size), "generator_hmm": generator_hmm,
        "forward": views["forward"], "backward": views["backward"],
        "C_plus": views["forward"]["C"], "C_minus": views["backward"]["C"],
        "entropy_rate": h, "excess_entropy": float(gen.excess_entropy),
        "crypticity_plus": views["forward"]["crypticity"],
        "crypticity_minus": views["backward"]["crypticity"],
        "h_L": floor_full,                             # h(L), L = 1..horizon+1: L - 1 tokens of context
        "floor": floor_full[1:],                       # t = 1..horizon: t tokens of context
        "floor_excess": float(np.sum(floor_full - h)),  # should equal excess_entropy
        "warnings": warnings,
    }
