"""
The straight-through temperature schedule.

tau NEVER changes what the model computes.  The forward value is
`hard = one_hot(argmax(z))` at every tau, because argmax is invariant to
positive scaling (models.py:296-298).  tau shapes only the gradient:

    d p_i / d z_j = (1/tau) * p_i * (delta_ij - p_j)

Two consequences drive the whole effect.  The off-diagonal mass decays like
exp(-gap/tau), so a bottleneck state that has already lost receives
exponentially little gradient and can never be claimed -- a lockout.  And the
entire bottleneck gradient carries a 1/tau factor, INCLUDING the beta*H(p_bar)
usage penalty, which minimises occupancy entropy and so actively prunes states.
Both bite late, because both depend on a logit gap that training itself creates.
Low tau is therefore nearly free early and expensive late; high tau is
uninformative early and protective late, and the schedule that matches is
low -> high.

MEASURED (tau_experiment/, 320 trained models, 13 phase-cells, CPU):

    geom:0.5:5     -0.336 +- 0.246 bits vs const:1   (phase 1B, 67% win)
    const:5        +0.172 +- 0.120                   -- high tau ALONE is worse,
                                                        so it is the rise that
                                                        helps, not the level
    geom:5:0.5     +0.439 +- 0.110, 5% win rate      -- the TEXTBOOK direction
                                                        (Jang et al. anneal
                                                        high->low), and the only
                                                        effect in the study
                                                        clearing two SEM.  It is
                                                        a harm.

Why the textbook direction fails here: Gumbel-softmax anneals high->low to
harden a SOFT relaxation into a discrete decision.  This architecture is already
hard at every tau, so high->low imports the cost -- an uninformative early phase,
a locked-out late phase -- without the benefit.

SCOPE, and it is narrow.  Phase 1D ran the schedule on four cells fixed before
the winner was known and found NOTHING: -0.036 +- 0.049 bits.  Improvement
tracks how badly const:1 was already doing (r = -0.44 over 13 phase-cells):
-0.370 bits where |S_emp - C| > 0.30, +0.003 where it is already below.  Treat
this as a rescue for runs that are visibly merging states, not as a free win.

Only `const` and `geom` are implemented.  The pilot also tested lin, cos, hold
and ramp between the same endpoints; they were within noise of each other
(F3), so carrying four more shapes into the pipeline would add four things to
get wrong and buy nothing.
"""
import lightning as L

# tau <= 0 divides by zero at models.py:296, and a geometric schedule cannot
# reach 0 in any case.  0.05 is well below the measured <=0.2 collapse point.
TAU_FLOOR = 0.05


def parse_tau(spec):
    """
    'geom:0.5:5' -> (label, f -> tau), with f the training fraction in [0, 1].

    A float (or an int) is accepted and returns a constant schedule, so
    `tau=1.0` still describes exactly the pre-schedule pipeline and every
    existing result stays reproducible.
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
    else:
        raise ValueError(
            f"unknown tau schedule {spec!r}; use a float, 'const:X' or 'geom:A:B'")

    return str(spec), (lambda f: max(TAU_FLOOR, float(fn(min(1.0, max(0.0, f))))))


def is_scheduled(spec) -> bool:
    """True when `spec` actually varies, i.e. needs the callback attached."""
    if isinstance(spec, (int, float)):
        return False
    kind, *rest = str(spec).split(":")
    if kind == "const":
        return False
    if kind == "geom" and len(rest) == 2:
        return float(rest[0]) != float(rest[1])
    return True


class TauSchedule(L.Callback):
    """
    Rewrite `pl_module.tau` before every training batch.

    This works without touching the model class because tau is a plain float
    attribute read fresh on every forward (models.py:296).  Keep it that way:
    caching tau, or folding it into a registered buffer, silently disables every
    schedule.

    The schedule is indexed by `trainer.global_step / (total_steps - 1)`, so it
    is defined over the whole run.  Changing `max_epochs` therefore RESCALES the
    schedule rather than truncating it -- a 75-epoch run traverses the same tau
    range as a 150-epoch one, at twice the rate.
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
