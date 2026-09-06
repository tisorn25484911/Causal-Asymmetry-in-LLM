"""
Schedules for the straight-through temperature and for Adam's momenta.

Both are expressed as SPEC STRINGS so an arm is fully described by the string
that produced it -- the JSON, the figure legend and the command line all carry
the same token, and no arm can be mislabelled by a lookup table drifting out of
step with the code.

    tau     const:1 | geom:0.5:5 | lin:0.5:5 | cos:0.5:5
            hold:A:B:h   tau=A for the first fraction h, then geometric A->B
            ramp:A:B:w   geometric A->B over the first fraction w, then B

    opt     adam:B1:B2            constant betas (the pipeline is adam:0.9:0.999)
            adam_b1:A:B[:B2]      beta1 linear A->B, beta2 fixed
            adam_b2:A:B[:B1]      beta2 linear A->B, beta1 fixed

`geom` is the exp(-r t) schedule of Jang et al. with the endpoint pinned rather
than the rate guessed, and reproduces study0's TauSchedule exactly.

NOTHING HERE TOUCHES Experimental_pipeline.  tau is a plain float attribute read
fresh on every forward (models.py:296), and Adam's betas live in
optimizer.param_groups, so both can be rewritten per step from a Lightning
callback without subclassing or editing the model.
"""
import math

import lightning as L

# tau <= 0 divides by zero in models.py:296, and a geometric schedule cannot
# reach 0 anyway.  0.05 is well below the documented <=0.2 collapse threshold.
TAU_FLOOR = 0.05


def parse_tau(spec: str):
    """'geom:0.5:5' -> (spec, f -> tau), f the training fraction in [0, 1]."""
    kind, *rest = spec.split(":")
    v = [float(x) for x in rest]

    if kind == "const":
        (a,) = v
        fn = lambda f: a                                          # noqa: E731
    elif kind == "geom":
        a, b = v
        fn = lambda f: a * (b / a) ** f                            # noqa: E731
    elif kind == "lin":
        a, b = v
        fn = lambda f: a + (b - a) * f                             # noqa: E731
    elif kind == "cos":
        a, b = v
        fn = lambda f: a + (b - a) * (1 - math.cos(math.pi * f)) / 2   # noqa: E731
    elif kind == "hold":
        a, b, h = v

        def fn(f, a=a, b=b, h=h):
            if f <= h:
                return a
            return a * (b / a) ** ((f - h) / max(1e-9, 1.0 - h))
    elif kind == "ramp":
        a, b, w = v

        def fn(f, a=a, b=b, w=w):
            if f >= w:
                return b
            return a * (b / a) ** (f / max(1e-9, w))
    else:
        raise ValueError(f"unknown tau schedule {spec!r}")

    return spec, (lambda f: max(TAU_FLOOR, float(fn(min(1.0, max(0.0, f))))))


def parse_opt(spec: str):
    """'adam:0.9:0.999' -> (spec, f -> (beta1, beta2))."""
    kind, *rest = spec.split(":")
    v = [float(x) for x in rest]

    if kind == "adam":
        b1, b2 = v
        fn = lambda f: (b1, b2)                                    # noqa: E731
    elif kind == "adam_b1":
        a, b = v[0], v[1]
        b2 = v[2] if len(v) > 2 else 0.999
        fn = lambda f: (a + (b - a) * f, b2)                        # noqa: E731
    elif kind == "adam_b2":
        a, b = v[0], v[1]
        b1 = v[2] if len(v) > 2 else 0.9
        fn = lambda f: (b1, a + (b - a) * f)                        # noqa: E731
    else:
        raise ValueError(f"unknown optimiser schedule {spec!r}")

    def clamp(f):
        b1, b2 = fn(min(1.0, max(0.0, f)))
        return (min(0.9999, max(0.0, b1)), min(0.99999, max(0.0, b2)))

    return spec, clamp


class TauSchedule(L.Callback):
    """Rewrite `pl_module.tau` before every training batch."""

    def __init__(self, fn, total_steps: int):
        super().__init__()
        self.fn = fn
        self.total_steps = max(2, int(total_steps))
        self.trace = []

    def tau_at(self, step: int) -> float:
        return self.fn(step / (self.total_steps - 1))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        tau = self.tau_at(trainer.global_step)
        pl_module.tau = tau
        if trainer.global_step % 100 == 0:
            self.trace.append((int(trainer.global_step), round(tau, 5)))


class BetaSchedule(L.Callback):
    """
    Rewrite Adam's (beta1, beta2) in place before every training batch.

    Mutating param_groups is the only way to move betas mid-run: torch reads
    them out of the group on every step, but `exp_avg` / `exp_avg_sq` carry the
    history accumulated under the OLD betas, so a schedule reinterprets an
    existing buffer rather than restarting it.  That is the intended behaviour
    -- it is what makes a beta schedule continuous -- but it does mean the
    effective averaging window lags the nominal one by roughly 1/(1-beta) steps.
    """

    def __init__(self, fn, total_steps: int):
        super().__init__()
        self.fn = fn
        self.total_steps = max(2, int(total_steps))
        self.trace = []

    def betas_at(self, step: int):
        return self.fn(step / (self.total_steps - 1))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        b = self.betas_at(trainer.global_step)
        for opt in trainer.optimizers:
            for g in opt.param_groups:
                g["betas"] = b
        if trainer.global_step % 100 == 0:
            self.trace.append((int(trainer.global_step),
                               round(b[0], 5), round(b[1], 6)))


class StateTrace(L.Callback):
    """
    Distinct states the model actually assigned during each epoch.

    The mechanistic read-out: a schedule that de-starves the state head claims
    states EARLIER, not merely more of them at the end, so the trajectory
    separates "the schedule helped" from "the schedule got lucky at the end".
    """

    def __init__(self):
        super().__init__()
        self.per_epoch = []
        self._seen = set()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        st = getattr(pl_module, "last_states", None)
        if st is not None:
            self._seen.update(st.detach().reshape(-1).cpu().tolist())

    def on_train_epoch_end(self, trainer, pl_module):
        self.per_epoch.append(len(self._seen))
        self._seen = set()
