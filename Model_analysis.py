import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from sklearn.cluster import KMeans
from OneHot_model import cross_ent_onehot
from matplotlib.colors import TwoSlopeNorm
try:
    import umap as _umap_mod
    _warmup = _umap_mod.UMAP(n_components=2, n_neighbors=200).fit_transform(
        np.random.rand(20, 4)
    )
    del _warmup
    UMAP_AVAILABLE = True
    print("umap-learn JIT warm-up succeeded")
except Exception as _e:
    UMAP_AVAILABLE = False
    print(f"UMAP unavailable ({_e}) — PCA fallback active")

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})
"""
latent analysis:
    - latent_extraction: extract latents, inputs, targets from a model and dataloader
    - _project2d: UMAP/PCA 2D projection of latents
    - UMAP plots: color by token type FW/BW models
    - attention heatmaps: visualize attention patterns for FW/BW models

Post-train analysis:
    - perplexity_calculation: compute perplexity on a dataset
    - statistical_complexity_empirical: estimate complexity via clustering latents
    - statistical_complexity_compare: compare empirical complexity to theoretical HMM complexity

General purpose plotting:
    - plot_diff_heatmap
    - training_loss_plot
"""
# ── Helper functions ───────────────────────────────────────────────────────
# C4: mkdir / save_pkl / save_weights / cleanup / entropy_rate_coin now have a
# single definition in utils.py.  They are re-exported here because several
# modules and the notebooks import them from Model_analysis.
from utils import (            # noqa: F401  -- deliberate re-export
    cleanup,
    coin_tag,
    entropy_bits,
    entropy_rate_coin,
    flower_tag,
    mkdir,
    save_pkl,
    save_weights,
    to_cpu_for_analysis,
)


# ── Result-bundle slimming ─────────────────────────────────────────────────
# A raw `res` bundle pickles to ~700 MB, split almost evenly between:
#   * the five fold models and their recorders (~100 MB per direction) --
#     and most of that is not weights but stale `last_attention` /
#     `last_encodings` activation buffers still hanging off each module;
#   * the raw latents, which are stored *twice*, once as `latents`
#     (N, T, d_model) and again as the flattened view of the same array.
# Neither is worth keeping: the best-fold weights are already written to
# results/models/*.pt by save_weights, and latents are cheap to re-extract
# from those weights.  See IMPROVEMENT_PLAN.md Phase 0.3.
_DROP_KEYS      = ("latents", "flat_lat", "flat_inp", "inputs_arr")
_RECORDER_SERIES = ("step_loss", "step_ppl", "step_at",
                    "step_val_loss", "step_val_ppl", "step_val_at",
                    "epoch_loss", "val_loss")


def _recorder_curves(rec):
    """Lift the plottable series off a Record_training, leaving its model behind."""
    if rec is None:
        return None
    return {name: list(getattr(rec, name, None) or []) for name in _RECORDER_SERIES}


def slim_results(obj):
    """
    Return a copy of a result bundle with the heavy objects stripped out.

    Kept: every scalar and per-fold metric, the per-step train/val curves
    lifted off the recorders (as `best_curves` / `fold_curves`), the UMAP
    coordinates (expensive to recompute) and the complexity numbers.
    Dropped: fold models, recorder objects, and raw latent arrays.

    Recurses through dicts and lists, so it works equally on a single
    experiment's `res` and on the combined `all_results` map.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _DROP_KEYS or k == "best_model":
                continue
            if k == "all_recorders":
                out["fold_curves"] = [_recorder_curves(r) for r in (v or [])]
            elif k == "best_recorder":
                out["best_curves"] = _recorder_curves(v)
            else:
                out[k] = slim_results(v)
        return out
    if isinstance(obj, list):
        return [slim_results(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(slim_results(v) for v in obj)
    return obj


def paired_delta_ce(cv_fw, cv_bw, label="", theory=None, conv_tol=0.10):
    """
    Paired delta-CE = CE_BW - CE_FW, fold by fold, on the held-out test set.

    This is only meaningful because A2 made the two arms paired: with the same
    seed, fold k of the forward run and fold k of the backward run share a
    hold-out test set, fold membership, batch order and weight init, so their
    difference isolates the mask/convention change.  Before A2 the two arms had
    different splits, and differencing them mixed the effect with split noise
    of unknown size -- for an effect expected to be a few hundredths of a bit.

    Reports the mean paired difference with its standard error across folds and
    a paired t-statistic.  Note the caveat in IMPROVEMENT_PLAN.md 1.1/4.2: five
    folds of one seed share a training set, so this quantifies fold-to-fold
    variability, NOT sampling variability over datasets.  It is a lower bound
    on the uncertainty, and the seed-repeat harness (Phase 4.2) is what turns
    it into evidence.
    """
    fw = np.asarray(cv_fw.get("fold_test_loss", []), dtype=float)
    bw = np.asarray(cv_bw.get("fold_test_loss", []), dtype=float)
    if fw.size == 0 or fw.size != bw.size:
        return None

    if cv_fw.get("seed") != cv_bw.get("seed"):
        print("  ! paired_delta_ce: arms were run with different seeds "
              f"({cv_fw.get('seed')} vs {cv_bw.get('seed')}) — NOT paired")

    def _stats(diff):
        k = diff.size
        if k == 0:
            return dict(mean=float("nan"), sd=float("nan"),
                        sem=float("nan"), t=float("nan"), n=0)
        mu = float(diff.mean())
        sd = float(diff.std(ddof=1)) if k > 1 else 0.0
        se = sd / np.sqrt(k) if k > 1 else float("nan")
        return dict(mean=mu, sd=sd, sem=se,
                    t=(mu / se if se and np.isfinite(se) and se > 0 else float("nan")),
                    n=k)

    d   = bw - fw
    all_stats = _stats(d)

    # Convergence filter.  A fold that failed to escape a bad local optimum
    # contributes an optimisation failure, not a causal-asymmetry measurement,
    # and at 5 folds one such fold dominates the mean.  Observed in
    # sanity_check: most folds land within 0.01 bits of H_inf while roughly one
    # in six sits 0.4-0.7 bits above it.  Folds are called converged when BOTH
    # arms are within conv_tol of the process entropy rate.
    #
    # Nothing is hidden: both statistics are printed and both are returned, and
    # fold_ce_fw / fold_ce_bw keep every fold so any other filter can be
    # applied after the fact without re-running.
    converged = None
    conv_stats = None
    if theory is not None and np.isfinite(theory):
        converged = (np.abs(fw - theory) <= conv_tol) & (np.abs(bw - theory) <= conv_tol)
        conv_stats = _stats(d[converged])

    print(f"\n  Paired delta-CE (BW - FW) over {all_stats['n']} folds"
          f"{' — ' + label if label else ''}")
    for k, (a, b) in enumerate(zip(fw, bw)):
        flag = ""
        if converged is not None and not converged[k]:
            flag = "   <-- NOT CONVERGED, excluded from the filtered mean"
        print(f"    fold {k+1}: CE_FW={a:.4f}  CE_BW={b:.4f}  delta={b-a:+.4f}{flag}")
    print(f"    all folds      : mean={all_stats['mean']:+.4f}  "
          f"sd={all_stats['sd']:.4f}  sem={all_stats['sem']:.4f}  t={all_stats['t']:+.2f}")
    if conv_stats is not None:
        n_bad = int((~converged).sum())
        if conv_stats["n"] == 0:
            print(f"    converged only : NO fold reached within {conv_tol} bits of "
                  f"H_inf={theory:.4f} — reporting all folds, and the number "
                  f"below is an optimisation artefact, not a measurement")
        else:
            print(f"    converged only : mean={conv_stats['mean']:+.4f}  "
                  f"sd={conv_stats['sd']:.4f}  sem={conv_stats['sem']:.4f}  "
                  f"t={conv_stats['t']:+.2f}   ({conv_stats['n']}/{all_stats['n']} folds"
                  f"{f', {n_bad} dropped' if n_bad else ''})")

    out = dict(fold_ce_fw=fw.tolist(), fold_ce_bw=bw.tolist(),
               fold_delta=d.tolist(), theory=theory, conv_tol=conv_tol,
               converged=(converged.tolist() if converged is not None else None),
               all_folds=all_stats, converged_only=conv_stats)
    # Headline: the converged-fold statistic when we have a theory value to
    # judge convergence against, otherwise all folds.
    head = conv_stats if (conv_stats and conv_stats["n"] > 0) else all_stats
    out.update(mean=head["mean"], sd=head["sd"], sem=head["sem"],
               t=head["t"], n=head["n"])
    return out


def _sub(arr: np.ndarray, n: int = 1000) -> tuple:
    """
    Return (first n rows, their indices).

    DEPRECATED for latent plotting -- this is the C6 bug.  `latents.reshape(-1, d)`
    flattens (N, T, d) row-major, so the first n rows are every position of the
    first ceil(n/T) sequences.  At chunk=256 with n=1000 that is 4 sequences;
    at T=1999 (sanity_check) with n=500 it is positions 0-499 of sequence 0
    ALONE.  Any cluster structure seen was conditioned on one realisation.
    Kept only because it is a public name; use _sample_latents instead.
    """
    n = min(n, len(arr))
    return arr[:n], np.arange(n)


def _resolve_t(use_t, T: int) -> int:
    """Map 'last'/'first'/int to a concrete position index."""
    if use_t == "last":
        return T - 1
    if use_t == "first":
        return 0
    if isinstance(use_t, (int, np.integer)):
        return int(use_t) % T
    raise ValueError("use_t must be 'first', 'last', or an int index.")


def _sample_latents(latents, inputs_arr, mode="per_sequence", use_t="last",
                    n_pts=1000, burn_in=32, seed=0):
    """
    Choose which latent vectors to embed — IMPROVEMENT_PLAN.md C6.

    latents    : (N, T, d)      inputs_arr : (N, T)
    returns    : (points (M, d), tokens (M,), human-readable label)

    Modes:

    "per_sequence"
        The latent at ONE position of every sequence, namely the max-context
        position for the direction (use_t="last" for a tril mask, "first" for
        triu).  This is exactly the slice statistical_complexity_empirical
        clusters, so the picture and the S_emp number describe the same thing.
        Every point has identical context length, so any spread is structure
        rather than a mixture of context lengths.

    "random"
        Uniform without replacement over (sequence, position) pairs, covering
        all N sequences.  Positions with too little context are excluded, and
        which end counts as "too little" depends on direction: a forward model
        accumulates context left-to-right, so its first `burn_in` positions are
        dropped; a backward model accumulates right-to-left, so its LAST
        `burn_in` positions are dropped.  Richer than "per_sequence" but mixes
        context lengths, which is what produces elongated smears rather than
        tight clusters.

    "prefix"
        The old C6 behaviour, kept so the difference can be shown rather than
        asserted.  Do not use for results.
    """
    lat = np.asarray(latents)
    inp = np.asarray(inputs_arr)
    if lat.ndim != 3:
        raise ValueError(f"latents must be (N, T, d), got {lat.shape}")
    N, T, _ = lat.shape
    rng = np.random.default_rng(seed)
    backward = (use_t == "first")

    if mode == "per_sequence":
        t = _resolve_t(use_t, T)
        pts, toks = lat[:, t, :], inp[:, t]
        if len(pts) > n_pts:                       # still sample ACROSS sequences
            keep = rng.choice(len(pts), size=n_pts, replace=False)
            pts, toks = pts[keep], toks[keep]
        return pts, toks, f"1 pos/seq, t={t}, {len(pts)} seqs"

    if mode == "random":
        b = int(np.clip(burn_in, 0, max(T - 1, 0)))
        lo, hi = (0, T - b) if backward else (b, T)   # direction-aware burn-in
        if hi <= lo:
            lo, hi = 0, T
        n_avail = N * (hi - lo)
        take = min(n_pts, n_avail)
        flat = rng.choice(n_avail, size=take, replace=False)
        seq_i = flat // (hi - lo)
        pos_i = flat % (hi - lo) + lo
        pts, toks = lat[seq_i, pos_i, :], inp[seq_i, pos_i]
        span = f"[{lo},{hi})"
        return pts, toks, f"random over {N} seqs x pos {span}, {take} pts"

    if mode == "prefix":
        flat_l = lat.reshape(-1, lat.shape[-1])
        flat_i = inp.reshape(-1)
        take = min(n_pts, len(flat_l))
        n_seq_covered = int(np.ceil(take / T))
        return (flat_l[:take], flat_i[:take],
                f"PREFIX (C6 bug): first {take} rows = {n_seq_covered} seq(s)")

    raise ValueError(f"unknown sample mode {mode!r}")
def savefig(fig, path: str, dpi: int = 120):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved -> {path}")
#latent analysis:
def latent_extraction(model, data_loader, max_batches = None):
    model.eval()
    latents_all = []
    inputs_all = []
    target_all = []

    device = next(model.parameters()).device  # FIX 3: Fixed device access

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)  # FIX 3: Changed from model.device()
            targets = targets.to(device)  # FIX 3: Changed from model.device()

            _ = model(inputs)
            z = model.last_encodings.detach().cpu().numpy()  # FIX 4: Fixed to numpy directly

            latents_all.append(z)  # FIX 4: Append z directly
            inputs_all.append(inputs.detach().cpu().numpy())
            target_all.append(targets.detach().cpu().numpy())

    latents_all = np.concatenate(latents_all, axis = 0)
    inputs_all = np.concatenate(inputs_all, axis = 0)
    target_all = np.concatenate(target_all, axis = 0)

    return latents_all, inputs_all, target_all
def _project2d(flat: np.ndarray, n_neighbors: int = 200) -> tuple:
    if UMAP_AVAILABLE:
        try:
            c = _umap_mod.UMAP(
                n_components=2, random_state=42,
                n_neighbors=min(n_neighbors, len(flat) - 1),
                min_dist=0.1,
                metric="euclidean",
            ).fit_transform(flat)
            return c, "UMAP"
        except Exception as e:
            print(f"  UMAP failed ({e}), using PCA")
    return PCA(n_components=2).fit_transform(flat), "PCA"
def _scatter_tokens(ax, coords, toks, num_token, title, xlim=None, ylim=None):
    cmap = plt.cm.tab10
    for tok in range(num_token):
        mask = toks == tok
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(tok / max(num_token - 1, 1))],
                   label=f"Token {tok}", alpha=0.7, s=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, markerscale=3)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2)


def plot_umap(latents, inputs_arr, num_token, title="", save_path=None,
              xlim=None, ylim=None, n_pts: int = 1000, n_neighbors: int = 15,
              use_t="last", burn_in: int = 32, seed: int = 0,
              modes=("per_sequence", "random")):
    """
    2-D UMAP of the latents, coloured by input token, one panel per sampling
    mode — IMPROVEMENT_PLAN.md C6.

    What C6 was.  This used to embed `latents.reshape(-1, d)[:n_pts]`.  The
    reshape is row-major, so those first n_pts rows are every position of the
    first ceil(n_pts/T) sequences: 4 sequences at chunk=256, and at T=1999 with
    n_pts=500 a single sequence.  Every latent figure in the repo was therefore
    one trajectory, not a sample of the latent distribution, and any cluster
    structure in it was conditioned on one realisation.

    Two panels are drawn because they answer different questions and each is
    misleading alone:

      per_sequence  one position per sequence, at the max-context position for
                    the direction.  This is the exact slice
                    statistical_complexity_empirical clusters, so the figure
                    and the S_emp number cannot tell different stories.  All
                    points share a context length, so spread is structure.
      random        uniform over (sequence, position) pairs across all
                    sequences, excluding a direction-aware burn-in.  Shows far
                    more of the distribution, but mixes context lengths.

    `use_t` must match the direction ("last" for a tril mask, "first" for triu)
    — it selects the per_sequence position and which end the burn-in trims.

    n_neighbors: 15 by default.  This parameter was unreachable before B7 was
    fixed; every plot silently used 200, which on 1000 points smears exactly
    the local structure the figure exists to show.

    Returns (fig, {mode: coords}).
    """
    modes = tuple(modes)
    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 5.5),
                             squeeze=False)
    out = {}
    for ax, mode in zip(axes[0], modes):
        try:
            pts, toks, lbl = _sample_latents(
                latents, inputs_arr, mode=mode, use_t=use_t,
                n_pts=n_pts, burn_in=burn_in, seed=seed)
            coords, mlbl = _project2d(pts, n_neighbors=n_neighbors)
            out[mode] = coords
            _scatter_tokens(ax, coords, toks, num_token,
                            f"{mode} ({mlbl})\n{lbl}", xlim, ylim)
        except Exception as e:
            print(f"  UMAP panel {mode!r} failed: {e}")
            ax.set_title(f"{mode} — failed", fontsize=10)
            ax.text(0.5, 0.5, str(e)[:80], ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        savefig(fig, save_path)
    return fig, out
def plot_attention_heatmap(model, input_seq):
    fig, ax = plt.subplots(figsize=(8, 6))

    device = next(model.parameters()).device
    input_seq = input_seq.unsqueeze(0).to(device)  # (1, T)

    # D3: attention is not retained unless explicitly requested, and no_grad
    # so a plot does not build an autograd graph over a (1, T, T) map.
    with torch.no_grad(), model.capture_attention():
        _ = model(input_seq)
        attention = model.last_attention.squeeze(0).cpu().numpy()  # (T, T)
        n_layers = len(model.last_attention_layers)

    cax = ax.matshow(attention, cmap='viridis')
    fig.colorbar(cax)

    # C8: say which layer this is.  It has always been the last one.
    ax.set_title(f"Attention Heatmap (layer {n_layers} of {n_layers})")
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")

    return fig

#Post-train analysis:
def perplexity_calculation(model, data_loader, max_batches=None, pad_id=None):
    logits_all  = []
    targets_all = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # mirror the same convention used in training_step
            if getattr(model, "mode", "forward") == "backward":
                targets, inputs = batch
            else:
                inputs, targets = batch

            inputs  = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)          # (B, T, V)
            logits_all.append(logits.cpu())
            targets_all.append(targets.cpu())

    logits  = torch.cat(logits_all,  dim=0)
    targets = torch.cat(targets_all, dim=0)

    if pad_id is not None:
        mask    = targets != pad_id
        logits  = logits[mask]
        targets = targets[mask]

    loss, perplexity = cross_ent_onehot(logits, targets)
    return perplexity.item()

def perplexity_ind_CE(model, data_loader, p, q, num_token=3, max_batches=None):
    """
    Cross-entropy perplexity evaluated against the TRUE HMM conditional
    distribution (soft label) on GROUND-TRUTH sequences from a DataLoader.

    At every position t in every ground-truth sequence:
        cur_tok = input token at position t
        p_true  = true conditional P(next | cur_tok)  [analytical for coin HMM]
        p_model = model softmax distribution at position t
        CE_t    = -sum_x P_true(x | cur_tok) * log2 P_model(x | context_t)

    PPL = 2^(mean CE_t)
    """
    if p is None or q is None:
        raise ValueError("Both p and q must be provided.")

    model.eval()
    device = next(model.parameters()).device
    is_bw  = (getattr(model, "mode", "forward") == "backward")

    # True transition matrix: target_prob[cur_tok] = P(next/prev | cur_tok)
    target_prob = np.zeros((num_token, num_token), dtype=np.float64)
    if is_bw:
        # P(previous token | current token)
        target_prob[0] = [1-p,       0,    p  ]
        target_prob[1] = [q*(1-p),   1-q,  p*q]
        target_prob[2] = [0,         1,    0  ]
    else:
        # P(next token | current token)
        target_prob[0] = [1-p, p,   0]
        target_prob[1] = [0,   1-q, q]
        target_prob[2] = [1-p, p,   0]

    row_sums = target_prob.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"target_prob rows do not sum to 1: {row_sums}")

    total_ce = 0.0
    n_steps  = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Mirror the training convention — backward model has swapped batch
            if is_bw:
                targets, inputs = batch
            else:
                inputs, targets = batch

            inputs  = inputs.to(device)             # (B, T)
            logits  = model(inputs)                 # (B, T, V)
            p_model = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, T, V)
            inp_np  = inputs.cpu().numpy()          # (B, T)

            # D2: vectorised.  This was `for b: for t:` in pure Python.
            # LLM_asymmetry_testing calls it with max_batches=None, i.e. 500
            # sequences x 1999 positions = 1e6 iterations per call, four calls
            # per experiment.  Fancy-indexing target_prob by the input tokens
            # gives the (B, T, V) soft-label array in one step.
            p_true = target_prob[inp_np]                     # (B, T, V)
            ce_bt  = -(p_true * np.log2(p_model + 1e-12)).sum(-1)   # (B, T)
            total_ce += float(ce_bt.sum())
            n_steps  += ce_bt.size

    if n_steps == 0:
        raise ValueError("No steps accumulated — empty loader or max_batches=0.")

    return float(2 ** (total_ce / n_steps))

def coin_true_conditional(p, q, num_token=3):
    fw = np.zeros((num_token, num_token))
    fw[0] = [1-p, p,       0  ]
    fw[1] = [0,   1-q,     q  ]
    fw[2] = [1-p, p,       0  ]
    bw = np.zeros((num_token, num_token))
    bw[0] = [1-p,      0,   p  ]
    bw[1] = [q*(1-p),  1-q, p*q]
    bw[2] = [0,        1,   0  ]
    return fw, bw

def stepwise_kl_coin(model, loader, p, q, num_token=3, max_batches=None):
    """
    For every position in every ground-truth sequence:
        KL_t = D_KL( P_true(.|cur_tok) || P_model(.|context_t) )
    Returns mean_kl, per_tok_avg (V,), per_tok_count (V,).
    """
    model.eval()
    is_bw  = (getattr(model, "mode", "forward") == "backward")
    device = next(model.parameters()).device
    fw_cond, bw_cond = coin_true_conditional(p, q, num_token)
    true_cond = bw_cond if is_bw else fw_cond

    total_kl      = 0.0
    per_tok_kl    = np.zeros(num_token)
    per_tok_count = np.zeros(num_token)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if is_bw:
                targets, inputs = batch
            else:
                inputs, targets = batch
            inputs  = inputs.to(device)
            logits  = model(inputs)
            p_model = torch.softmax(logits, dim=-1).cpu().numpy()
            inp_np  = inputs.cpu().numpy()
            # D2: vectorised, same shape of fix as perplexity_ind_CE.
            # np.bincount does the per-token accumulation that the inner loop
            # was doing one element at a time.
            p_true = true_cond[inp_np]                          # (B, T, V)
            kl_bt  = (p_true * np.log2(p_true / (p_model + 1e-12) + 1e-12)).sum(-1)
            flat_tok = inp_np.reshape(-1)
            flat_kl  = kl_bt.reshape(-1)
            total_kl      += float(flat_kl.sum())
            per_tok_kl    += np.bincount(flat_tok, weights=flat_kl,
                                         minlength=num_token)
            per_tok_count += np.bincount(flat_tok, minlength=num_token)

    mean_kl     = total_kl / max(per_tok_count.sum(), 1)
    per_tok_avg = np.where(per_tok_count > 0,
                           per_tok_kl / per_tok_count, 0.0)

    print(f"    mean KL = {mean_kl:.6f} bits")
    for tok in range(num_token):
        print(f"      token {tok}: avg KL={per_tok_avg[tok]:.6f}"
              f"  (n={int(per_tok_count[tok])})")
    return mean_kl, per_tok_avg, per_tok_count


def self_generated_entropy_rate(model, len_seq=2000, start_token=0):
    """
    Estimate of the MODEL'S OWN entropy rate — NOT a comparison metric.

    IMPROVEMENT_PLAN.md C3.  This samples a sequence from the model and then
    scores the model's log-loss on that same sequence.  A degenerate,
    over-confident, low-entropy model *minimises* it, so it cannot be used to
    say which of two models is better: each model is evaluated on its own
    sequence, drawn from its own state distribution.

    LLM_asymmetry_testing.py already labelled it "informational only — not used
    for asymmetry comparison"; the old name `perplexity_ind_model` did not, and
    Test_data_eval.py used it as its headline FW-vs-BW chart (B14).  For an
    actual comparison use perplexity_calculation / teacher_forced_ppl, which
    scores both models on the same ground-truth sequences.
    """
    burn_in = 100
    print(f"Standard PPL over {len_seq - burn_in} tokens (burn-in={burn_in}) ...")
    model.eval()
    device = next(model.parameters()).device
    is_bw  = (getattr(model, "mode", "forward") == "backward")

    total_log2_prob = 0.0
    total_tokens    = 0

    # forward : context grows rightward  → start at left end
    # backward: context grows leftward   → start at right end
    context = [start_token]

    with torch.no_grad():
        for i in range(len_seq):
            x      = torch.tensor([context], device=device)   # (1, current_len)
            out    = model(x)                                  # (1, current_len, C)

            if is_bw:
                # position 0: upper-triangular mask → attends to ALL tokens in context
                # → predicts the token that comes BEFORE context[0]
                logits = out[:, 0, :]                          # (1, C)
            else:
                # position -1: lower-triangular mask → attends to full past
                # → predicts the token that comes AFTER context[-1]
                logits = out[:, -1, :]                         # (1, C)

            log2_probs = torch.nn.functional.log_softmax(logits, dim=-1) \
                         / torch.log(torch.tensor(2.0, device=device))
            probs      = torch.softmax(logits, dim=-1)

            # sample the next token in the respective direction
            next_token = torch.multinomial(probs[0], num_samples=1).item()
            next_token_log2_prob = log2_probs[0, next_token].item()

            if i >= burn_in:
                total_log2_prob += next_token_log2_prob
                total_tokens    += 1

            # forward: append to right  |  backward: prepend to left
            if is_bw:
                context = [next_token] + context
            else:
                context = context + [next_token]

    perplexity = 2 ** (-(total_log2_prob / total_tokens))
    return perplexity


# Backwards-compatible alias: metric_panel_plot.ipynb imports the old name.
perplexity_ind_model = self_generated_entropy_rate


def plot_perplexity(model_fw, model_bw, data_loader, max_batches=None,
                    save_path=None):
    perplexity_fw = perplexity_calculation(model_fw, data_loader, max_batches)
    perplexity_bw = perplexity_calculation(model_bw, data_loader, max_batches)

    print("="*70)
    print("PERPLEXITY COMPARISON")
    print("="*70)
    print(f"Forward Model Perplexity:  {perplexity_fw:.4f}")
    print(f"Backward Model Perplexity: {perplexity_bw:.4f}")
    print("="*70)

    # Bar Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    models = ['Forward', 'Backward']
    perplexities = [perplexity_fw, perplexity_bw]
    bars = ax.bar(models, perplexities, color=['blue', 'orange'], alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Model Perplexity Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    # B9: this used to end in plt.show(), which is a NO-OP under the Agg
    # backend that every runner selects — so the figure was built and then
    # silently discarded, and the function returned None.
    if save_path:
        savefig(fig, save_path)
    return fig
def statistical_complexity(p, q, mode): 
    # statistical complexity of HMM
    if mode == "forward":
        stead0 = q / (p + q)
        stead1 = p / (p + q)
        state_prob = np.array([stead0, stead1])
    elif mode == "backward":
        stead0 = (q - p*q)/(p+q)
        stead1 = (p)/(p+q)
        stead2 = p*q/(p+q)
        state_prob = np.array([stead0, stead1, stead2])
    else:
        raise ValueError(f"Invalid model mode: {mode}. Must be 'forward' or 'backward'")
    S = 0.0
    for prob in state_prob:
        S += - prob * np.log2(prob + 1e-12)
    return S
def flower_complexity(n: int, m: int, dice_probs) -> tuple[float, float]:
    """
    Closed-form (C+, C-) for the n-m flower process — IMPROVEMENT_PLAN.md A1.

    The process alternates: die i ~ Uniform{0..n-1} emitted as token i, then
    outcome j ~ dice_probs[i] emitted as token n+j.

    FORWARD causal states: one "a roll just happened" state (the next token is
    a uniform die selection, independent of everything before) plus one state
    per die i (the next token is dice_probs[i]).  Their probabilities are 1/2
    and 1/2 * 1/n each, so

        C+ = 1 + (1/2) * log2(n)

    BACKWARD causal states: one "the current token is a selection" state (the
    previous token is a roll drawn from the marginal outcome distribution,
    independent of which die) plus one state per *distinguishable* outcome.

    The word distinguishable is what the repo's old m+1 formula missed.  Two
    outcomes j and j' are the same backward state exactly when they induce the
    same posterior P(die = i | outcome = j) ~ dice_probs[i,j]/n over dice --
    i.e. when columns j and j' of dice_probs are proportional.  Merging those,

        C- = 1 + (1/2) * H(pi_merged),   pi_j = (1/n) * sum_i dice_probs[i,j]

    Two consequences the repo was getting wrong:

      * n=1 degenerates.  With a single die P(die|outcome)=1 for every j, so
        ALL m outcomes collapse to one state and C- = 1 = C+.  The n=1,m=2
        sanity-check process 0,X,0,X,... with X iid uniform{1,2} is exactly
        time-reversible; it is a null control, not a positive one.
      * C- <= 1 + (1/2)log2(m), so C- > C+ requires m > n.  Every flower config
        originally in the repo had n > m, i.e. tested the negation of the
        hypothesis.

    Verified against IMPROVEMENT_PLAN.md section 4: (1,2)->(1.0000, 1.0000),
    (4,2)->(2.0000, 1.4952), (6,4)->(2.2925, 1.9899).
    """
    dp = np.asarray(dice_probs, dtype=float)
    if dp.shape != (n, m):
        raise ValueError(f"dice_probs must have shape ({n}, {m}), got {dp.shape}")
    if not np.allclose(dp.sum(axis=1), 1.0):
        raise ValueError(f"dice_probs rows must sum to 1, got {dp.sum(axis=1)}")

    C_plus = 1.0 + 0.5 * np.log2(n)

    pi_outcome = dp.mean(axis=0)                       # (1/n) sum_i dp[i, j]
    col_mass   = dp.sum(axis=0)
    merged: dict[tuple, float] = {}
    for j in range(m):
        if col_mass[j] <= 0:                           # outcome never occurs
            continue
        posterior = np.round(dp[:, j] / col_mass[j], 9)   # P(die | outcome j)
        key = tuple(posterior)
        merged[key] = merged.get(key, 0.0) + pi_outcome[j]

    C_minus = 1.0 + 0.5 * entropy_bits(list(merged.values()))
    return float(C_plus), float(C_minus)


def flower_entropy_rate(n: int, m: int, dice_probs) -> float:
    """
    Entropy rate H_inf (bits/token) of the n-m flower process.

    Half the tokens are die selections, uniform over n, contributing log2(n)
    bits each; the other half are rolls of a uniformly chosen die,
    contributing the mean of H(dice_probs[i]).  So

        H_inf = (1/2) log2(n) + (1/2) * mean_i H(dice_probs[i])

    Time-reversal invariance means the same value applies to both directions,
    which is why the runners pass one `theory` to both arms.  Gives flower runs
    a real H_inf reference line instead of nan.  Check: n=1, m=2, fair die ->
    0 + 0.5*1 = 0.5, matching the value sanity_check.py had hard-coded.
    """
    dp = np.asarray(dice_probs, dtype=float)
    return float(0.5 * np.log2(n)
                 + 0.5 * np.mean([entropy_bits(dp[i]) for i in range(n)]))


def statistical_complexity_empirical(model, data_loader, max_batches=None, use_t="last", k=2):
    model.eval()
    latents_all = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)
            z = model.last_encodings.detach().cpu().numpy()  # (B, T, d_model)
            latents_all.append(z)

    latents = np.concatenate(latents_all, axis=0)  # (N, T, d_model)

    # --- choose a consistent time slice instead of mixing all t ---
    if use_t == "last":
        z_use = latents[:, -1, :]          # (N, d_model)
    elif use_t == "first":
        z_use = latents[:, 0, :]           # (N, d_model)
    elif isinstance(use_t, int):
        z_use = latents[:, use_t, :]       # (N, d_model)
    else:
        raise ValueError("use_t must be 'first', 'last', or an int index.")

    # --- cluster in full space (don’t PCA-crush by default) ---
    # D4: KMeans and PCA were re-imported inside functions that already import
    # them at module top.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(z_use)

    counts = np.bincount(labels, minlength=k)
    probs = counts / counts.sum()

    S = -np.sum(probs * np.log2(probs + 1e-12))
    return S
def statistical_complexity_compare(forward_model, backward_model, data_loader,
                                  p=0.6, q=0.4, max_batches=None, k_fw=2, k_bw=3,
                                  save_path=None):
    print("="*70)
    print("STATISTICAL COMPLEXITY COMPARISON")
    print("="*70)
    
    Ss_empirical = {}
    Ss_theory = {}
    
    # Forward: max-past context occurs at LAST token (causal tril)
    print("\n[1/2] Forward Model")
    print("-"*70)
    S_fw_emp = statistical_complexity_empirical(
        forward_model, data_loader, max_batches=max_batches, use_t="last", k=k_fw
    )
    S_fw_theory = statistical_complexity(p, q, mode='forward')
    
    print(f"  Empirical (use_t='last'):   {S_fw_emp:.4f} bits")
    print(f"  Theoretical:               {S_fw_theory:.4f} bits")
    print(f"  Error:                     {abs(S_fw_emp - S_fw_theory):.4f} bits")
    
    Ss_empirical['forward'] = S_fw_emp
    Ss_theory['forward'] = S_fw_theory
    
    # Backward: max-future context occurs at FIRST token (anti-causal triu)
    print("\n[2/2] Backward Model")
    print("-"*70)
    S_bw_emp = statistical_complexity_empirical(
        backward_model, data_loader, max_batches=max_batches, use_t="first", k=k_bw
    )
    S_bw_theory = statistical_complexity(p, q, mode='backward')
    
    print(f"  Empirical (use_t='first'):  {S_bw_emp:.4f} bits")
    print(f"  Theoretical:               {S_bw_theory:.4f} bits")
    print(f"  Error:                     {abs(S_bw_emp - S_bw_theory):.4f} bits")
    
    Ss_empirical['backward'] = S_bw_emp
    Ss_theory['backward'] = S_bw_theory
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nEmpirical Complexity:")
    print(f"  Forward (last):   {S_fw_emp:.4f} bits")
    print(f"  Backward (first): {S_bw_emp:.4f} bits")
    print(f"  Difference (FW - BW): {S_fw_emp - S_bw_emp:+.4f} bits")
    
    print(f"\nTheoretical Complexity:")
    print(f"  Forward:  {S_fw_theory:.4f} bits")
    print(f"  Backward: {S_bw_theory:.4f} bits")
    print(f"  Difference (FW - BW): {S_fw_theory - S_bw_theory:+.4f} bits")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    modes = ['Forward', 'Backward']
    emp_values = [Ss_empirical['forward'], Ss_empirical['backward']]
    theory_values = [Ss_theory['forward'], Ss_theory['backward']]

    x = np.arange(len(modes))
    width = 0.35

    bars1 = ax.bar(x - width/2, emp_values, width, label='Empirical', alpha=0.8, color='blue')
    bars2 = ax.bar(x + width/2, theory_values, width, label='Theoretical', alpha=0.8, color='red')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Statistical Complexity (bits)', fontsize=11)
    ax.set_title("Empirical vs Theoretical Statistical Complexity", fontsize=12, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    # B9: was plt.show() under Agg — a no-op, so this figure was discarded and
    # only the numbers came back.
    if save_path:
        savefig(fig, save_path)

    print("="*70)
    return Ss_empirical, Ss_theory, fig

#General purpose plotting:
def plot_diff_heatmap(Z, p_vals, q_vals, title, cbar_label, save_path=None, cmap="RdBu_r", vcenter=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    ext = [float(np.min(q_vals)), float(np.max(q_vals)),
           float(np.min(p_vals)), float(np.max(p_vals))]
    if vcenter is not None:                                            # ← added block
        vmin = float(np.nanmin(Z))
        vmax = float(np.nanmax(Z))
        vmin = min(vmin, vcenter - 1e-6)   # guard: vmin < vcenter < vmax
        vmax = max(vmax, vcenter + 1e-6)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        im = ax.imshow(Z, origin="lower", extent=ext,
                       cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(Z, origin="lower", extent=ext, cmap=cmap, aspect="auto")
    Qm, Pm = np.meshgrid(q_vals, p_vals)
    ax.contour(Qm, Pm, Z, levels=8, colors="white", alpha=0.4, linewidths=0.8)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("q", fontsize=12); ax.set_ylabel("p", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if save_path: savefig(fig, save_path)
    return fig
def training_loss_plot(recorder):
    fig, ax = plt.subplots(figsize=(40, 6))
    ax.plot(recorder.epoch_loss, label='Training Loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss over Epochs")
    ax.legend()
    return fig
    
#Standard Analysis functions:   


def FW_BW_attention_comparison(model_fw, model_bw, input_seq, title_prefix=""):
    """
    Compare forward and backward model attention patterns with truly SQUARE heatmaps.

    - Forces square *axes boxes* (not just square pixels) via set_box_aspect(1)
    - Uses ONE shared colorbar to avoid shrinking each subplot differently
    - Keeps FW/BW on same color scale (vmin/vmax shared)
    """

    # --- Move input to same device as model ---
    device = next(model_fw.parameters()).device
    input_seq = input_seq.unsqueeze(0).to(device)  # (1, T)

    # --- Forward / backward (D3: attention retained only inside the block) ---
    with torch.no_grad():
        with model_fw.capture_attention():
            _ = model_fw(input_seq)
            attention_fw = model_fw.last_attention.squeeze(0).cpu().numpy()  # (T,T)
            n_layers = len(model_fw.last_attention_layers)
        with model_bw.capture_attention():
            _ = model_bw(input_seq)
            attention_bw = model_bw.last_attention.squeeze(0).cpu().numpy()  # (T,T)

    # --- Shared color limits ---
    vmin = min(attention_fw.min(), attention_bw.min())
    vmax = max(attention_fw.max(), attention_bw.max())

    # --- Figure (square-ish overall layout) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Plot FW
    im0 = axes[0].imshow(attention_fw, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    axes[0].set_title(f"{title_prefix}Forward Model Attention (layer {n_layers})",
                      fontsize=14, fontweight="bold", pad=10)
    axes[0].set_xlabel("Key", fontsize=12)
    axes[0].set_ylabel("Query", fontsize=12)
    axes[0].set_box_aspect(1)          # <-- makes the axes box square
    axes[0].tick_params(labelsize=10)

    # Plot BW
    im1 = axes[1].imshow(attention_bw, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    axes[1].set_title(f"{title_prefix}Backward Model Attention (layer {n_layers})",
                      fontsize=14, fontweight="bold", pad=10)
    axes[1].set_xlabel("Key", fontsize=12)
    axes[1].set_ylabel("Query", fontsize=12)
    axes[1].set_box_aspect(1)          # <-- makes the axes box square
    axes[1].tick_params(labelsize=10)

    # --- One shared colorbar (prevents per-axes shrinking) ---
    cbar = fig.colorbar(im0, ax=axes, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    return fig


def compare_FW_BW_latents(model_fw, model_bw, data_loader, max_batches = None):
    # Create a non-shuffled loader to ensure both models see same data
    # Note: We'll just remove the assertion since both models are trained on same distribution
    latents_fw, inputs_fw, _ = latent_extraction(model_fw, data_loader, max_batches)
    latents_bw, inputs_bw, _ = latent_extraction(model_bw, data_loader, max_batches)

    # FIX 18: Removed assertion - models see different batches due to shuffle, but same distribution
    # assert np.array_equal(inputs_fw, inputs_bw), "Input sequences do not match between forward and backward models."

    token_colors = {0: 'red', 1: 'green', 2: 'purple'}
    latents_fw_flat = latents_fw.reshape(-1, latents_fw.shape[-1])
    latents_bw_flat = latents_bw.reshape(-1, latents_bw.shape[-1])

    print("Performing PCA on Forward and Backward Latents...")
    pca_fw = PCA(n_components = 2)
    latents_fw_2D = pca_fw.fit_transform(latents_fw_flat) #(N*T, 2)

    pca_bw = PCA(n_components = 2)
    latents_bw_2D = pca_bw.fit_transform(latents_bw_flat) #(N*T, 2)

    print("="*70)
    print("FW/BW LATENTS COMPARISON")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    
    for token in [0, 1, 2]:
        mask = inputs_fw.reshape(-1) == token
        axes[0].scatter(latents_fw_2D[mask, 0], latents_fw_2D[mask, 1], 
                        c = token_colors[token], label = f'Token {token}', 
                        alpha = 0.3, s = 5)
    axes[0].set_title("Forward Model Latents (PCA)")
    axes[0].legend()

    for token in [0, 1, 2]:
        mask = inputs_bw.reshape(-1) == token
        axes[1].scatter(latents_bw_2D[mask, 0], latents_bw_2D[mask, 1], 
                        c = token_colors[token], label = f'Token {token}', 
                        alpha = 0.3, s = 5)
    axes[1].set_title("Backward Model Latents (PCA)")
    axes[1].legend()

    return fig

def FW_BW_loss_comparison(recorder_fw, recorder_bw, save_path=None):
    """
    Forward vs backward training loss, and their difference.

    IMPROVEMENT_PLAN.md B10.  Four defects, all fixed here:
      * the function ended in a bare `return`, so the figure it had just built
        was unreachable and discarded;
      * both x-axes were labelled "Epoch" while step_loss is indexed by
        gradient step;
      * panel 2's title said "Backward Model Training Loss" while it plots the
        BW - FW difference;
      * panel 1's title said "Forward Model Training Loss" while it plots both.

    Note the runners do not call this — they use their own plot_loss_theory,
    which was already correct.  Kept and fixed rather than deleted so ad-hoc
    and notebook use gets a working version.
    """
    print("="*70)
    print("FW/BW LOSS COMPARISON")
    print("="*70)
    sl_fw = np.asarray(recorder_fw.step_loss, dtype=float)
    sl_bw = np.asarray(recorder_bw.step_loss, dtype=float)
    n = min(len(sl_fw), len(sl_bw))          # folds can differ by a step
    sl_fw, sl_bw = sl_fw[:n], sl_bw[:n]

    fig, ax = plt.subplots(1, 2, figsize=(20, 4))
    ax[0].plot(sl_fw, label='Forward Model Loss')
    ax[0].plot(sl_bw, label='Backward Model Loss')
    ax[1].plot(sl_bw - sl_fw, label='Loss Difference (BW - FW)')
    ax[1].axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)

    ax[0].set_xlabel("Gradient step")
    ax[0].set_ylabel("Loss (bits)")
    ax[0].set_title("Training Loss — Forward vs Backward")
    ax[1].set_xlabel("Gradient step")
    ax[1].set_ylabel("CE_BW − CE_FW (bits)")
    ax[1].set_title("Loss Difference (BW − FW)")
    ax[0].legend(); ax[1].legend()
    ax[0].grid(True, alpha=0.3); ax[1].grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        savefig(fig, save_path)
    return fig