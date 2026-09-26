"""
The straight-through temperature schedule (ported from
Experimental_pipeline/schedules.py; the learning-rate schedules were left there).

tau NEVER changes what the model computes.  The forward value is
`hard = one_hot(argmax(z))` at every tau, because argmax is invariant to
positive scaling (causal_matrix._learn_causal_state.forward).  tau shapes only
the gradient:

    d p_i / d z_j = (1/tau) * p_i * (delta_ij - p_j)

so it decides which states still receive gradient once one has won a position.

Which direction is better depends on the run length.  At 150 epochs
tau_experiment/study1 ranked the high->low geom:5:0.5 last; at 2000 epochs
study5 found it the best fixed schedule (states are found DURING the descent,
and 5 -> 0.5 keeps the losing states' probability mass alive throughout).

Specs (f is the training fraction, 0 -> 1):

    a float, 'const:X'     constant
    'geom:A:B'             geometric A -> B over the whole run
    'geomhold:A:B[:DESCENT]'  geometric A -> B over the first DESCENT fraction of the
                           run (default 0.5), then held at B for the REST -- the 4th
                           number is how long the descent takes, not how long the
                           hold lasts: geomhold:5:0.5:0.8 over 1000 epochs descends
                           for 800 and holds for 200.  study5's `geomhold` rule
                           (tau_experiment/study5_adaptive_tau/adaptive.py).  On the
                           flower (3,8) backward arm, 5 -> 0.5 descended over 2000
                           epochs and held for 2000 more gave 9,8,8,8,7 of 9 states
                           against 7.1 on average for the plain 2000-epoch descent: the
                           extra updates help only AFTER the descent ("raises the floor").

The schedule is indexed by the training FRACTION, so changing max_epochs rescales it
rather than truncating it: geomhold:5:0.5:0.5 at 800 epochs descends over 400.
"""
import lightning as L

# tau <= 0 divides by zero in the head, and a geometric schedule cannot reach 0
# in any case.  0.05 is well below the measured <=0.2 collapse point.
TAU_FLOOR = 0.05


def parse_tau(spec):
    """
    'geom:5:0.5' -> (label, f -> tau), with f the training fraction in [0, 1].

    A float (or an int) is accepted and returns a constant schedule.
    """
    if isinstance(spec, (int, float)):
        v = float(spec)
        if v <= 0:
            raise ValueError(f"tau must be > 0, got {v}")
        return f"const:{v:g}", (lambda f, v=v: v)

    kind, *rest = str(spec).split(":")
    try:
        vals = [float(x) for x in rest]
    except ValueError:
        raise ValueError(f"unparseable tau schedule {spec!r}")

    if kind == "const" and len(vals) == 1:
        (a,) = vals
        fn = lambda f: a                                          # noqa: E731
    elif kind == "geom" and len(vals) == 2:
        a, b = vals
        if a <= 0 or b <= 0:
            raise ValueError(f"geom endpoints must be > 0, got {spec!r}")
        fn = lambda f: a * (b / a) ** f                            # noqa: E731
    elif kind == "geomhold" and len(vals) in (2, 3):
        a, b = vals[:2]
        descent = vals[2] if len(vals) == 3 else 0.5
        if a <= 0 or b <= 0:
            raise ValueError(f"geomhold endpoints must be > 0, got {spec!r}")
        if not 0.0 < descent <= 1.0:
            raise ValueError(f"geomhold's DESCENT is the fraction of the run the descent takes, in (0, 1]; "
                             f"got {spec!r}")
        fn = lambda f: a * (b / a) ** min(f / descent, 1.0)        # noqa: E731
    else:
        raise ValueError(
            f"unknown tau schedule {spec!r}; use a float, 'const:X', 'geom:A:B' or 'geomhold:A:B[:DESCENT]'")

    return str(spec), (lambda f: max(TAU_FLOOR, float(fn(min(1.0, max(0.0, f))))))


def is_scheduled(spec) -> bool:
    """True when `spec` actually varies, i.e. needs the callback attached."""
    if isinstance(spec, (int, float)):
        return False
    kind, *rest = str(spec).split(":")
    if kind == "const":
        return False
    if kind in ("geom", "geomhold") and len(rest) >= 2:
        return float(rest[0]) != float(rest[1])
    return True


class TauSchedule(L.Callback):
    """
    Rewrite `pl_module.tau` before every training batch.

    This works without touching the model class because tau is a plain float
    attribute that the model passes to its head on every forward.  Keep it that
    way: caching tau (in the head, or in a registered buffer) silently disables
    every schedule.

    The schedule is indexed by `trainer.global_step / (total_steps - 1)`, so it
    is defined over the whole run.
    """

    def __init__(self, fn, total_steps: int):
        super().__init__()
        self.fn = fn
        self.total_steps = max(2, int(total_steps))
        self.trace = []                      # (step, tau) every 100 steps

    def tau_at(self, step: int) -> float:
        return self.fn(step / (self.total_steps - 1))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        tau = self.tau_at(trainer.global_step)
        pl_module.tau = tau
        if trainer.global_step % 100 == 0:
            self.trace.append((int(trainer.global_step), round(tau, 5)))
