"""Process-global patches to AutoGluon Mitra internals.

Applied unconditionally (d2h sync) or conditionally (regression CE, row caps,
gate caps, local from_pretrained) in the child subprocess before any fit runs.
"""
from __future__ import annotations

import contextlib
import inspect
import textwrap

import numpy as np


def install_d2h_sync_patch() -> None:
    """Make Mitra's GPU->CPU model move complete before the model is pickled.

    ``MitraModel._save_model_artifact`` moves a fitted bag child to the CPU
    and immediately ``torch.save``s it, and AutoGluon's Mitra trainer does
    that move with ``.to(device, non_blocking=True)`` without ever
    synchronizing. On torch >= 2.13 the destination of a non-blocking
    device-to-host copy is PINNED memory, so the copy genuinely is
    asynchronous and ``torch.save`` serializes host buffers the DMA is
    still filling. The child lands on disk with a correct PREFIX of tensors
    (in state_dict order) and stale pinned-cache contents after it --
    output head included, so the reloaded child emits all-equal logits:
    a constant 1/n_classes for every row.

    Out-of-fold probabilities come from the intact in-memory model, while
    test predictions come from the children the bag reloads from disk, so
    only the TEST side collapses. That is the "OOF healthy, test
    near-uniform" signature measured on 25-40% of TabArena tasks (1-6 of
    the 8 children per task, varying run to run because it is a timing
    race). Measured on this box: a 303 MB non-blocking D2H copy returns in
    0.1 ms and completes 43.5 ms later, so ``torch.save`` loses the race
    almost every time.

    A blocking copy plus an explicit synchronize costs a few ms per child
    fit and removes it. Patched on the base class BEFORE ``install_view``,
    so the ``ViewTrainer`` subclass -- which does not override
    ``set_device`` -- inherits the fix.
    """
    import torch
    import autogluon.tabular.models.mitra._internal.core.trainer_finetune \
        as trainer_module

    if getattr(trainer_module, "_mitra_finetune_d2h_sync_patched", False):
        return

    def set_device(self, device: str) -> None:
        self.device = device
        self.model = self.model.to(device=device, non_blocking=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    trainer_module.TrainerFinetune.set_device = set_device
    trainer_module._mitra_finetune_d2h_sync_patched = True


def install_reg_ce_patches(n_bins: int) -> None:
    """Turn AutoGluon >= 1.6 Mitra into the champion cross-entropy regressor.

    Stock ``MitraRegressor`` fits with an MSE scalar head (``dim_output=1``).
    The released regression checkpoint is instead a ``n_bins``-way
    cross-entropy head (regression-as-classification over value bins), decoded
    by the softmax-weighted mean of bin centers. AutoGluon already ships the
    full binning + cross-entropy training path (it is reached whenever
    ``task=REGRESSION`` and ``regression_loss=CROSS_ENTROPY``); it just is not
    selectable through a public hyperparameter. Two process-global patches
    switch it on, mirroring ``install_d2h_sync_patch``'s style:

    1. ``MitraBase._create_config`` forces ``regression_loss=CROSS_ENTROPY``
       and ``dim_output=n_bins`` for regression, so the trainer bins the
       target and sizes ``self.bins`` to the checkpoint's head.
    2. ``TrainerFinetune.evaluate``/``predict`` decode with the mean over bin
       centers instead of argmax (see ``_mean_decode_source``).

    ``predict`` is additionally wrapped so that, inside a
    ``capture_regression_distribution`` block, every trainer predict call (one
    per bag child) hands its bin distribution to the collector before the mean
    decode collapses it; outside such a block the wrapper is a pass-through.

    Applied on the base class BEFORE ``install_view``, so the ``ViewTrainer``
    subclass inherits the mean decode through its ``super()`` calls, and
    BEFORE the predict-time speed/cap wrappers, so an OOM retry by an outer
    wrapper re-enters the capture afresh (a failed attempt's chunks are
    discarded, never double counted).
    """
    import autogluon.tabular.models.mitra._internal.core.trainer_finetune \
        as trainer_module
    import autogluon.tabular.models.mitra.sklearn_interface as sklearn_interface
    from autogluon.tabular.models.mitra._internal.config.enums import (
        LossName,
        Task,
    )

    if getattr(sklearn_interface, "_mitra_finetune_reg_ce_patched", False):
        return

    original_create_config = sklearn_interface.MitraBase._create_config

    def create_config(self, task, dim_output, time_limit=None):
        cfg, model_cls = original_create_config(self, task, dim_output, time_limit)
        is_regression = cfg.task == Task.REGRESSION or str(task).lower().endswith(
            "regression"
        )
        if is_regression:
            cfg.task = Task.REGRESSION
            cfg.hyperparams["regression_loss"] = LossName.CROSS_ENTROPY
            cfg.hyperparams["dim_output"] = n_bins
        return cfg, model_cls

    sklearn_interface.MitraBase._create_config = create_config
    trainer_module.TrainerFinetune.evaluate = _mean_decode_source(
        trainer_module.TrainerFinetune.evaluate, numpy_variant=False
    )
    decoded_predict = _mean_decode_source(
        trainer_module.TrainerFinetune.predict, numpy_variant=True
    )

    def predict_with_distribution_capture(self, x_support, y_support, x_query):
        collector = _REG_DIST_CAPTURE["collector"]
        if collector is None:
            return decoded_predict(self, x_support, y_support, x_query)
        collector.begin_call(self, n_query=len(x_query))
        try:
            y_pred = decoded_predict(self, x_support, y_support, x_query)
        except BaseException:
            collector.abort_call()
            raise
        collector.end_call(self, y_pred)
        return y_pred

    # Module global read by the rewritten decode (its globals are the trainer
    # module's); None outside a capture block.
    trainer_module._mitra_finetune_distribution_sink = None
    trainer_module.TrainerFinetune.predict = predict_with_distribution_capture
    sklearn_interface._mitra_finetune_reg_ce_patched = True


def install_use_hf_patch() -> None:
    """Force USE_HF=True and patch Tab2D.from_pretrained for local dirs."""
    import autogluon.tabular.models.mitra.sklearn_interface as _mitra_sk
    _mitra_sk.USE_HF = True

    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D as _Tab2D
    if getattr(_Tab2D, "_local_from_pretrained_patched", False):
        return

    import json as _json_fp
    from safetensors.torch import load_file as _load_file
    _orig_from_pretrained = _Tab2D.from_pretrained.__func__

    @classmethod
    def _local_from_pretrained(cls, path_or_repo_id: str, device: str = "cuda"):
        from pathlib import Path as _P
        local = _P(path_or_repo_id)
        if local.is_dir() and (local / "config.json").exists():
            config = _json_fp.loads((local / "config.json").read_text())
            model = cls(
                dim=config["dim"],
                dim_output=config["dim_output"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"],
                task=config["task"],
                use_pretrained_weights=False,
                path_to_weights="",
                device=device,
            )
            state_dict = _load_file(str(local / "model.safetensors"), device=device)
            model.load_state_dict(state_dict)
            return model
        return _orig_from_pretrained(cls, path_or_repo_id, device=device)

    _Tab2D.from_pretrained = _local_from_pretrained
    _Tab2D._local_from_pretrained_patched = True


def install_gate_cap_patch(gate_cap: int) -> None:
    """Cap in-context support/query so the gate's holdout fits don't OOM."""
    import autogluon.tabular.models.mitra.sklearn_interface as _si_cap
    _cap_orig_cc = _si_cap.MitraBase._create_config

    def _cap_cc(self, *a, **k):
        cfg, mc = _cap_orig_cc(self, *a, **k)
        cfg.hyperparams["max_samples_support"] = gate_cap
        cfg.hyperparams["max_samples_query"] = gate_cap
        return cfg, mc

    _si_cap.MitraBase._create_config = _cap_cc


def install_support_cap_patch(cap: int) -> None:
    """Lift the stock 8192-row in-context support cap for fine-tune to ``cap``."""
    import autogluon.tabular.models.mitra.sklearn_interface as _si_rc
    if getattr(_si_rc.MitraBase, "_rc_cc_patched", False):
        return
    _rc_orig_cc = _si_rc.MitraBase._create_config

    def _rc_cc(self, *a, **k):
        cfg, mc = _rc_orig_cc(self, *a, **k)
        cfg.hyperparams["max_samples_support"] = int(cap)
        return cfg, mc

    _si_rc.MitraBase._create_config = _rc_cc
    _si_rc.MitraBase._rc_cc_patched = True


def install_support_select_patch(predict_mode: str, finetune_mode: str = "") -> None:
    """Class-balance the in-context support subsample (predict and/or fine-tune).

    When the support pool is capped below the full training set, stock
    ``DatasetFinetune.__getitem__`` keeps a uniformly random subsample, so a
    class-imbalanced table hands the model an equally imbalanced in-context
    support. ``balanced_bin`` instead fills a per-class quota -- smallest class
    first, releasing any unmet quota to the remaining classes, topping up any
    shortfall at random -- so a BINARY task's widened predict context is
    class-balanced. It is a no-op when the support is NOT subsampled
    (``support_size >= n_samples_support``), when the task is not binary
    (``!= 2`` distinct support labels; multiclass/regression fall back to the
    stock random draw), and when the governing mode string is empty.

    ``predict_mode`` governs the label-free predict draw; ``finetune_mode`` the
    labelled fine-tune draw (usually left empty so fine-tuning is byte-for-byte
    stock). Predict vs fine-tune is distinguished by whether ``y_query`` was
    ``None`` at construction -- captured in an ``__init__`` wrapper BEFORE stock
    ``__init__`` replaces a ``None`` ``y_query`` with a ``-1`` sentinel, so it
    cannot be recovered afterwards.

    Process-global monkeypatch on ``DatasetFinetune`` (``__init__`` +
    ``__getitem__``); the ``__getitem__`` draw is byte-identical to stock's
    ``self.rng.choice`` whenever the active mode is empty / not engaged, so
    fine-tune (mode empty) and non-binary tasks are unchanged. Install BEFORE
    ``install_fast_predict_patch`` so the fixed-support RNG-freeze wraps this
    balanced draw (keeping every predict query chunk on the same support rows).
    """
    import numpy as _np_ss
    import inspect as _inspect_ss
    import torch as _torch_ss
    from autogluon.tabular.models.mitra._internal.data.dataset_finetune import (
        DatasetFinetune as _DF_ss,
    )
    if getattr(_DF_ss, "_support_select_patched", False):
        return

    def _random_draw(self):
        return self.rng.choice(
            self.n_samples_support, size=self.support_size, replace=False
        )

    def _draw_support_indices(self):
        mode = predict_mode if getattr(self, "_predict_mode", False) else finetune_mode
        if not mode or self.support_size >= self.n_samples_support:
            return _random_draw(self)
        try:
            if mode == "balanced_bin":
                y = _np_ss.asarray(self.y_support).reshape(-1)
                classes, counts = _np_ss.unique(y, return_counts=True)
                if classes.shape[0] != 2:
                    return _random_draw(self)
                budget = int(self.support_size)
                n_classes = int(classes.shape[0])
                parts = []
                for i, ci in enumerate(_np_ss.argsort(counts)):
                    quota = budget // (n_classes - i)
                    take = int(min(int(counts[ci]), quota))
                    budget -= take
                    pool = _np_ss.where(y == classes[ci])[0]
                    parts.append(
                        pool if take >= pool.shape[0]
                        else self.rng.choice(pool, size=take, replace=False)
                    )
                out = _np_ss.concatenate(parts)
                if out.shape[0] < self.support_size:
                    rest = _np_ss.setdiff1d(
                        _np_ss.arange(self.n_samples_support), out
                    )
                    out = _np_ss.concatenate([
                        out,
                        self.rng.choice(
                            rest, size=self.support_size - out.shape[0], replace=False
                        ),
                    ])
                return out
        except Exception as _e_ss:  # never let a selection bug fail a fit
            print(
                f"[support-select] {mode} failed ({_e_ss!r}); random draw",
                flush=True,
            )
        return _random_draw(self)

    _orig_init = _DF_ss.__init__
    _init_sig = _inspect_ss.signature(_orig_init)

    def _init(self, *a, **k):
        # Capture predict-mode (label-free query) BEFORE stock __init__ turns a
        # None y_query into a -1 sentinel; bind by signature so arg position
        # does not matter.
        try:
            bound = _init_sig.bind(self, *a, **k)
            bound.apply_defaults()
            _yq = bound.arguments.get("y_query", None)
        except TypeError:
            _yq = k.get("y_query", None)
        self._predict_mode = _yq is None
        _orig_init(self, *a, **k)

    def _getitem(self, idx):
        support_indices = _draw_support_indices(self)
        x_support = self.x_support[support_indices]
        y_support = self.y_support[support_indices]
        return {
            "x_support": _torch_ss.as_tensor(x_support),
            "y_support": _torch_ss.as_tensor(y_support),
            "x_query": _torch_ss.as_tensor(self.x_queries[idx]),
            "y_query": _torch_ss.as_tensor(self.y_queries[idx]),
        }

    _DF_ss.__init__ = _init
    _DF_ss.__getitem__ = _getitem
    _DF_ss._support_select_patched = True


# Active predict-cap state: install_predict_support_cap_patch() is called per
# task but the TrainerFinetune.predict wrapper is process-global, so repeat
# calls RECONFIGURE this state instead of re-wrapping (the cap is
# task-type-conditioned: binary 16384, multiclass/regression 32768).
_PREDICT_CAP_STATE = {"cap": 32768, "floor": 8192}


def install_predict_support_cap_patch(pred_cap: int, pred_floor: int) -> None:
    """Predict-only support widening with an OOM-halving ratchet.

    Fine-tune keeps the stock 8192-row support cap (in-distribution, cheap);
    only the forward-only predict context is widened to ``pred_cap``. Wide
    tables (col-attention memory ~ rows x cols^2) fall back by halving the cap
    on CUDA OOM until they fit (floor ``pred_floor``, default = the stock cap),
    which makes the large cap a SAFE universal default.
    """
    import torch as _torch_rc
    from autogluon.tabular.models.mitra._internal.core.trainer_finetune import (
        TrainerFinetune as _TF_rc,
    )
    if _PREDICT_CAP_STATE["cap"] != pred_cap:
        print(f"[predict-rowcap] task cap -> {pred_cap}", flush=True)
    _PREDICT_CAP_STATE["cap"] = pred_cap
    _PREDICT_CAP_STATE["floor"] = pred_floor
    if getattr(_TF_rc, "_rc_pred_patched", False):
        return
    _rc_orig_pred = _TF_rc.predict

    def _rc_pred(self, x_support, y_support, x_query):
        _old = self.cfg.hyperparams["max_samples_support"]
        _cap = _PREDICT_CAP_STATE["cap"]
        try:
            while True:
                self.cfg.hyperparams["max_samples_support"] = _cap
                try:
                    return _rc_orig_pred(self, x_support, y_support, x_query)
                except RuntimeError as _e_rc:
                    if "out of memory" not in str(_e_rc).lower():
                        raise
                    _torch_rc.cuda.empty_cache()
                    if _cap <= _PREDICT_CAP_STATE["floor"]:
                        raise
                    _cap_next = _cap // 2
                    if _cap > 32768 and _cap_next < 32768:
                        # Snap to the proven-safe stock cap first so an
                        # oversized cap can never land BELOW baseline
                        # (49152//2=24576 would); below 32768 the plain
                        # halving resumes.
                        _cap_next = 32768
                    _cap = _cap_next
                    print(
                        f"[predict-rowcap] CUDA OOM -> halving predict "
                        f"support cap to {_cap}",
                        flush=True,
                    )
        finally:
            self.cfg.hyperparams["max_samples_support"] = _old

    _TF_rc.predict = _rc_pred
    _TF_rc._rc_pred_patched = True


def fast_predict_plan(
    n_support: int, support_cap: int, qchunk: int, qchunk_floor: int, stock_chunk: int
) -> tuple[bool, int, int]:
    """How one predict call runs: ``(fixed_support_draw, query_chunk, query_chunk_floor)``.

    When the support pool fits under the active cap no subsample is drawn, so
    one seeded permutation serves every query chunk (a single-chunk predict is
    bit-exact with stock) and the out-of-memory ratchet may shrink the chunk
    down to ``qchunk_floor``. When the pool exceeds the cap every chunk gets a
    fresh capped draw, as in stock, and the ratchet stops at the stock chunk:
    below it the prediction would only get slower, never different.
    """
    support_fits = not (0 < int(support_cap) < int(n_support))
    query_chunk = max(int(stock_chunk), int(qchunk))
    floor = int(qchunk_floor) if support_fits else int(stock_chunk)
    return support_fits, query_chunk, min(floor, query_chunk)


def install_fast_predict_patch(qchunk: int, qchunk_floor: int) -> None:
    """Predict-only speed lever: wide query chunks, one support draw when the
    support fits, with a query-chunk-first OOM ratchet.

    Stock ``TrainerFinetune.predict`` splits the query rows into 1,024-row
    chunks and ``DatasetFinetune.__getitem__`` draws a fresh support subsample
    for every chunk, so a test set of K chunks re-encodes the in-context
    support K times. Predicting in ``qchunk``-row chunks (default 16,384)
    removes most of that work without touching the support cap:

    1. Support fits under the active cap (the common case): no subsample is
       drawn at all, so the fast path changes nothing but speed. The
       ``rng`` state is snapshotted on the first ``__getitem__`` and restored
       before every later chunk, so the UNMODIFIED stock ``__getitem__`` draws
       the same support permutation for every chunk (a single-chunk predict is
       bit-exact with stock; multi-chunk predicts change only which rows the
       2nd+ chunks see, a within-noise difference). On CUDA OOM the query
       chunk is halved down to ``qchunk_floor``.
    2. Support exceeds the cap (large tables): every chunk still gets a fresh
       capped draw, as in stock, so every query row sees exactly one draw and
       the prediction is the same in expectation; the chunks are just 16
       times wider. Paired on the ten TabArena tables in this regime (90
       splits, RTX PRO 6000, same fleet): 50 wins, 40 losses against the stock
       chunking, geometric error ratio 1.0001, and 7 times less inference time
       (APSFailure 1,419 s to 185 s per split). On CUDA OOM the chunk is
       halved down to the STOCK chunk, never below it: the worst case is the
       stock path plus a few failed allocations, which is why widening the
       chunk cannot slow a bag fold past AutoGluon's time projection.

    The fixed support draw is active ONLY while a predict() call in regime 1
    is on the stack: fit-time ``evaluate`` and regime 2 run the stock per-chunk
    redraw. The ``__getitem__`` patch is process-global by necessity but inert
    unless the flag is set; the frozen RNG state is cached on the per-call
    ``DatasetFinetune`` instance, so it cannot leak across test sets. The
    chunk is read fresh from ``cfg.hyperparams`` inside ``predict`` and
    restored afterward, so fine-tuning is untouched.

    Install BEFORE ``install_predict_support_cap_patch`` so the support-cap
    wrapper becomes the OUTER predict wrapper: an OOM shrinks the query chunk
    here first, and only once the query chunk hits its floor does it escalate
    to (the quality-costing) support-cap halving.
    """
    import torch as _torch_fp
    from autogluon.tabular.models.mitra._internal.core.trainer_finetune import (
        TrainerFinetune as _TF_fp,
    )
    from autogluon.tabular.models.mitra._internal.data.dataset_finetune import (
        DatasetFinetune as _DF_fp,
    )
    if getattr(_TF_fp, "_fast_pred_patched", False):
        return

    # Predict-scope flag shared by the two closures below: set only while a
    # predict() call whose support fits under the cap is on the stack.
    _state = {"active": False}

    # (1) Fixed support draw (predict-scoped). Snapshot the RNG on the first
    # chunk and restore it before every later chunk so the STOCK __getitem__'s
    # own support draw yields identical indices across chunks; the batch it
    # returns is stock's verbatim, so it cannot drift if AutoGluon changes the
    # returned structure. Falls back to the stock per-chunk redraw when inactive.
    _orig_getitem = _DF_fp.__getitem__

    def _fixed_getitem(self, idx):
        if not _state["active"]:
            return _orig_getitem(self, idx)
        _st = getattr(self, "_fixed_rng_state", None)
        if _st is None:
            self._fixed_rng_state = self.rng.get_state()
        else:
            self.rng.set_state(_st)
        return _orig_getitem(self, idx)

    _DF_fp.__getitem__ = _fixed_getitem

    # (2) Wide query chunk in both regimes, with a query-chunk-first OOM ratchet.
    _fp_orig_pred = _TF_fp.predict

    def _fast_pred(self, x_support, y_support, x_query):
        _hp = self.cfg.hyperparams
        try:
            _cap_now = int(_hp["max_samples_support"])
        except (KeyError, TypeError, ValueError):
            _cap_now = 0
        _old_q = _hp["max_samples_query"]
        _fixed, _q, _floor = fast_predict_plan(
            len(x_support), _cap_now, qchunk, qchunk_floor, int(_old_q)
        )
        _was_active = _state["active"]
        _state["active"] = _fixed
        try:
            while True:
                _hp["max_samples_query"] = _q
                try:
                    return _fp_orig_pred(self, x_support, y_support, x_query)
                except RuntimeError as _e_fp:
                    if "out of memory" not in str(_e_fp).lower():
                        raise
                    _torch_fp.cuda.empty_cache()
                    if _q <= _floor:
                        raise
                    _q = max(_floor, _q // 2)
                    print(
                        f"[fast-predict] CUDA OOM -> halving query chunk to {_q}",
                        flush=True,
                    )
        finally:
            _hp["max_samples_query"] = _old_q
            _state["active"] = _was_active

    _TF_fp.predict = _fast_pred
    _TF_fp._fast_pred_patched = True


def _mean_decode_source(fn, *, numpy_variant: bool):
    """Rewrite one AutoGluon trainer method's argmax bin-decode to a mean decode.

    Mitra's cross-entropy regression head predicts a distribution over
    ``dim_output`` value bins; stock AutoGluon decodes it with
    ``argmax`` (the single most-likely bin center). The champion regression
    checkpoint was evaluated with the softmax-weighted MEAN over bin centers
    instead -- a smoother estimate that materially lowers RMSE. This is the
    only quality-critical delta from stock; everything else (binning, loss,
    the pretrained backbone) is stock AutoGluon Mitra cross-entropy
    regression, unchanged.

    Rather than hand-copy AutoGluon's long ``evaluate``/``predict`` methods
    (which would silently drift on an AutoGluon upgrade), this transforms the
    INSTALLED method's own source: it locates the two-line argmax decode and
    swaps only those lines, leaving the rest of the method byte-identical. If
    AutoGluon ever changes the decode, the two needles stop matching and this
    raises loudly instead of mis-decoding.

    The ``predict`` (numpy) variant also hands each chunk's softmax to
    ``_mitra_finetune_distribution_sink`` (a trainer-module global, None
    unless ``capture_regression_distribution`` is active) before the mean.
    """
    source = textwrap.dedent(inspect.getsource(fn))
    lines = source.split("\n")

    if numpy_variant:
        argmax_needle = "y_hat = np.argmax(y_hat, axis=-1)"
        bins_needle = "y_hat = (self.bins[y_hat] + self.bin_width / 2).cpu().numpy()"
        replacement = (
            "_logits = torch.as_tensor(y_hat).float()\n"
            "if not torch.isfinite(_logits).all():\n"
            "    _logits = torch.nan_to_num(_logits, nan=0.0, posinf=1e4, neginf=-1e4)\n"
            "_centers = (self.bins[:-1] + self.bin_width / 2).cpu()\n"
            "_probs = torch.softmax(_logits, dim=-1)\n"
            "if _mitra_finetune_distribution_sink is not None:\n"
            "    _mitra_finetune_distribution_sink(self, _probs)\n"
            "y_hat = (_probs * _centers).sum(dim=-1).numpy()"
        )
    else:
        argmax_needle = "y_hat = torch.argmax(y_hat, dim=-1)"
        bins_needle = "y_hat = self.bins[y_hat] + self.bin_width / 2"
        replacement = (
            "_yf = y_hat.float()\n"
            "if not torch.isfinite(_yf).all():\n"
            "    _yf = torch.nan_to_num(_yf, nan=0.0, posinf=1e4, neginf=-1e4)\n"
            "_centers = self.bins[:-1] + self.bin_width / 2\n"
            "y_hat = (torch.softmax(_yf, dim=-1) * _centers.to(_yf.device)).sum(dim=-1)"
        )

    index = next(
        (i for i, line in enumerate(lines) if line.strip() == argmax_needle),
        None,
    )
    if index is None or lines[index + 1].strip() != bins_needle:
        raise RuntimeError(
            "AutoGluon Mitra regression bin-decode changed; the mean-decode "
            f"patch for {fn.__qualname__} must be updated"
        )
    indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    block = [indent + line if line else line for line in replacement.split("\n")]
    lines[index : index + 2] = block

    namespace: dict = {}
    exec(compile("\n".join(lines), f"<mitra_finetune reg-decode {fn.__name__}>", "exec"),
         fn.__globals__, namespace)
    return namespace[fn.__name__]


# -- Regression predictive distribution capture --------------------------------
# At most one collector is active (set by capture_regression_distribution);
# the reg-CE predict wrapper installed by install_reg_ce_patches consults it.
_REG_DIST_CAPTURE = {"collector": None}


def _member_grid_in_target_units(trainer, probs: np.ndarray):
    """Map one trainer's bin grid and probabilities to the original target units.

    The trainer bins the NORMALIZED target: ``self.bins`` is
    ``linspace(-0.5, 1.5, n_bins + 1)`` over ``(y - y_min) / (y_max - y_min)``
    of the child's fit data, and the stock ``inverse_transform_y`` undoes an
    optional mirror (``y -> 1 - y``) and then the min-max scaling. Both steps
    are affine, so the bin edges map exactly; the mirror reverses the bin
    order, so the probabilities are flipped along with the edges to keep them
    ascending. Returns ``(edges [n_bins + 1] float64, probs [n, n_bins])``.
    """
    import torch

    edges = trainer.bins.detach().to(device="cpu", dtype=torch.float64).numpy()
    preprocessor = trainer.preprocessor
    mirrored = bool(getattr(preprocessor, "random_mirror_regression", False)) and bool(
        getattr(preprocessor, "regression_mirror", False)
    )
    if mirrored:
        edges = (1.0 - edges)[::-1]
        probs = probs[:, ::-1]
    y_min = float(preprocessor.y_min)
    y_max = float(preprocessor.y_max)
    edges = edges * (y_max - y_min) + y_min
    return np.ascontiguousarray(edges), np.ascontiguousarray(probs)


class _RegressionDistributionCollector:
    """Collects one bin distribution per trainer predict call (= per bag member).

    ``begin_call``/``end_call`` bracket one ``TrainerFinetune.predict`` call;
    ``on_chunk`` receives the softmax of every query chunk in between (the
    rewritten decode calls it with the chunk's ``[rows, n_bins]`` CPU tensor).
    ``end_call`` checks that the chunks cover exactly the call's query rows,
    maps the grid to target units, and verifies that the histogram mean
    reproduces the call's own decoded point predictions, which pins the
    affine mapping (mirror included) to the stock decode.
    """

    def __init__(self) -> None:
        self.members: list[tuple[np.ndarray, np.ndarray]] = []
        self.points: list[np.ndarray] = []
        self._call = None

    def begin_call(self, trainer, n_query: int) -> None:
        if self._call is not None:
            raise RuntimeError("nested trainer predict calls are not expected")
        self._call = {"trainer": trainer, "n_query": int(n_query), "chunks": []}

    def on_chunk(self, trainer, probs) -> None:
        if self._call is None or self._call["trainer"] is not trainer:
            raise RuntimeError(
                "regression distribution chunk received outside its predict call"
            )
        self._call["chunks"].append(
            np.array(probs.detach().cpu().numpy(), dtype=np.float32, copy=True)
        )

    def abort_call(self) -> None:
        self._call = None

    def end_call(self, trainer, y_pred) -> None:
        call, self._call = self._call, None
        if call is None or call["trainer"] is not trainer:
            raise RuntimeError("end_call without a matching begin_call")
        if not call["chunks"]:
            raise RuntimeError(
                "no bin distribution was produced by the predict call; the "
                "cross-entropy regression decode is not active"
            )
        probs = np.concatenate(call["chunks"], axis=0)
        if probs.shape[0] != call["n_query"]:
            raise RuntimeError(
                f"distribution rows {probs.shape[0]} != query rows {call['n_query']}"
            )
        edges, probs = _member_grid_in_target_units(trainer, probs)
        point_raw = np.asarray(y_pred)
        point = point_raw.astype(np.float64).reshape(-1)
        if point.shape[0] != probs.shape[0]:
            raise RuntimeError(
                f"point predictions {point.shape[0]} != distribution rows {probs.shape[0]}"
            )
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Compare in float64 on renormalized rows. The captured float32 softmax
        # rows sum to one only to about 1e-6, while the stock decode adds the
        # target offset y_min exactly once; an unnormalized mean would carry a
        # y_min * (sum(p) - 1) error that swamps the tolerance whenever
        # |y_min| >> y_max - y_min (timestamps, pressures in Pa).
        p64 = probs.astype(np.float64)
        p64 /= p64.sum(axis=1, keepdims=True)
        offset = float(edges[0])
        hist_mean = p64 @ (centers - offset) + offset
        span = float(edges[-1] - edges[0])
        scale = max(abs(float(edges[0])), abs(float(edges[-1])))
        # Both sides carry float64 rounding proportional to the target offset
        # (the stock decode forms m * range + y_min); without this floor a
        # target with |offset| / range beyond ~1e12 would fail the check after
        # the whole bagged fit.
        tolerance = 1e-4 * span + 1e-12 * scale
        if point_raw.dtype.itemsize < 8:
            # The decode stayed in float32 in target units (float32 targets, or
            # value-based casting); allow its representation error.
            tolerance += 1e-6 * scale
        gap = float(np.max(np.abs(hist_mean - point))) if point.size else 0.0
        if not np.isfinite(gap) or gap > tolerance:
            raise RuntimeError(
                "captured bin distribution does not reproduce the decoded point "
                f"predictions (max gap {gap:.3g} > {tolerance:.3g})"
            )
        self.members.append((edges, probs))
        self.points.append(point)

    def stacked(self):
        """``(bin_edges [M, n_bins + 1], probabilities [M, n, n_bins], points [M, n])``."""
        if not self.members:
            raise RuntimeError("no bag member predicted during the capture block")
        edges = np.stack([e for e, _ in self.members], axis=0)
        probs = np.stack([p for _, p in self.members], axis=0)
        points = np.stack(self.points, axis=0)
        return edges, probs, points


@contextlib.contextmanager
def capture_regression_distribution():
    """Collect every bag child's bin distribution during one predict pass.

    Yields a collector whose ``stacked()`` returns the members in call order
    after the block. Requires ``install_reg_ce_patches`` (the capture hooks
    live in its predict wrapper and decode). Not re-entrant.
    """
    import autogluon.tabular.models.mitra._internal.core.trainer_finetune \
        as trainer_module
    import autogluon.tabular.models.mitra.sklearn_interface as sklearn_interface

    if not getattr(sklearn_interface, "_mitra_finetune_reg_ce_patched", False):
        raise RuntimeError(
            "capture_regression_distribution needs install_reg_ce_patches first"
        )
    if _REG_DIST_CAPTURE["collector"] is not None:
        raise RuntimeError("capture_regression_distribution is not re-entrant")
    collector = _RegressionDistributionCollector()
    _REG_DIST_CAPTURE["collector"] = collector
    trainer_module._mitra_finetune_distribution_sink = collector.on_chunk
    try:
        yield collector
    finally:
        trainer_module._mitra_finetune_distribution_sink = None
        _REG_DIST_CAPTURE["collector"] = None


_SUPPORT_CACHE_FREEZE = {"on": False}


def install_support_cache_patch(budget_gb: float = 6.0) -> None:
    """Support-stream caching for chunked predict (``MITRA_SUPPORT_CACHE=1``).

    Tab2D re-encodes the full support set for every query chunk at predict
    time, although (verified in tab2d.py) the support stream never attends
    to the query: support states are query-independent and query rows are
    mutually independent. This patch

    1. freezes the loader's per-chunk support redraw for the duration of
       one ``TrainerFinetune.predict`` call (one draw per call; on tasks
       whose rows exceed the support cap the stock behavior draws a fresh
       subsample per chunk, so this is a small, documented semantic change
       there -- elsewhere it is a pure permutation fix), and
    2. encodes the support stream once per predict call, reusing its
       per-layer post-``layer_norm1`` states as the row-attention K/V for
       every query chunk.

    The cached math implements the standard-attention branch (the attention
    modules fall back to SDPA when called without varlen metadata), so on
    flash-attn builds predict numerics change kernel, not math. Measured on
    P5/H100 (support 8192x50, 20x1024 query chunks): bit-equal to the
    uncached standard-attention path, 4.95x faster; 4.7x faster than the
    uncached flash path. Fine-tuning is untouched.

    A cache larger than ``budget_gb`` (env ``MITRA_SUPPORT_CACHE_GB``) falls
    back to the stock forward for that call.
    """
    import einops
    import torch

    from autogluon.tabular.models.mitra._internal.data.dataset_finetune import (
        DatasetFinetune as _DS,
    )
    from autogluon.tabular.models.mitra._internal.core.trainer_finetune import (
        TrainerFinetune as _TF,
    )
    from autogluon.tabular.models.mitra._internal.models.tab2d import (
        Tab2D as _T2D,
        Task as _Task,
    )

    if getattr(_TF, "_support_cache_patched", False):
        return

    # -- 1. freeze the support draw within one predict call ----------------
    _orig_getitem = _DS.__getitem__

    def _frozen_getitem(self, idx):
        if not _SUPPORT_CACHE_FREEZE["on"]:
            return _orig_getitem(self, idx)
        if not hasattr(self, "_sc_support_idx"):
            self._sc_support_idx = self.rng.choice(
                self.n_samples_support, size=self.support_size, replace=False
            )
        idxs = self._sc_support_idx
        x_support = torch.as_tensor(self.x_support[idxs])
        y_support = torch.as_tensor(self.y_support[idxs])
        return {
            "x_support": x_support,
            "y_support": y_support,
            "x_query": torch.as_tensor(self.x_queries[idx]),
            "y_query": torch.as_tensor(self.y_queries[idx]),
        }

    _DS.__getitem__ = _frozen_getitem

    # -- 2. cached forward --------------------------------------------------
    _orig_forward = _T2D.forward

    def _query_half_layer(layer, query, s_ln_flat, bsz):
        residual = query
        q = layer.layer_norm1(query)
        q_flat = einops.rearrange(q, "b s f d -> (b f) s d")
        q_att = layer.attention1(q_flat, s_ln_flat, s_ln_flat)
        query = residual + einops.rearrange(
            q_att, "(b f) s d -> b s f d", b=bsz
        )
        residual = query
        q = layer.layer_norm2(query)
        q = layer.linear2(torch.nn.functional.gelu(layer.linear1(q)))
        query = residual + q
        residual = query
        q = layer.layer_norm3(query)
        q_feat = einops.rearrange(q, "b s f d -> (b s) f d")
        q_feat = layer.attention2(q_feat, q_feat, q_feat)
        query = residual + einops.rearrange(
            q_feat, "(b s) f d -> b s f d", b=bsz
        )
        residual = query
        q = layer.layer_norm4(query)
        q = layer.linear4(torch.nn.functional.gelu(layer.linear3(q)))
        return residual + q

    def _support_pass(model, support):
        bsz = support.shape[0]
        cache = []
        for layer in model.layers:
            residual = support
            s = layer.layer_norm1(support)
            s_flat = einops.rearrange(s, "b s f d -> (b f) s d")
            cache.append(s_flat)
            s_att = layer.attention1(s_flat, s_flat, s_flat)
            support = residual + einops.rearrange(
                s_att, "(b f) s d -> b s f d", b=bsz
            )
            residual = support
            s = layer.layer_norm2(support)
            s = layer.linear2(torch.nn.functional.gelu(layer.linear1(s)))
            support = residual + s
            residual = support
            s = layer.layer_norm3(support)
            s_feat = einops.rearrange(s, "b s f d -> (b s) f d")
            s_feat = layer.attention2(s_feat, s_feat, s_feat)
            support = residual + einops.rearrange(
                s_feat, "(b s) f d -> b s f d", b=bsz
            )
            residual = support
            s = layer.layer_norm4(support)
            s = layer.linear4(torch.nn.functional.gelu(layer.linear3(s)))
            support = residual + s
        return cache

    def _cached_forward(self, x_support, y_support, x_query,
                        padding_features, padding_obs_support,
                        padding_obs_query__):
        if not getattr(self, "_sc_enabled", False):
            return _orig_forward(self, x_support, y_support, x_query,
                                 padding_features, padding_obs_support,
                                 padding_obs_query__)
        n_obs_query = x_query.shape[1]
        key = (
            tuple(x_support.shape),
            float(x_support.sum()),
            float(y_support.sum()),
        )
        store = getattr(self, "_sc_store", None)
        if store is None or store["key"] != key:
            # estimate cache size; fall back to stock forward if over budget
            b, s, f = x_support.shape
            est = len(self.layers) * b * s * (f + 1) * self.dim * 2  # bf16
            if est > budget_gb * (1024 ** 3):
                return _orig_forward(self, x_support, y_support, x_query,
                                     padding_features, padding_obs_support,
                                     padding_obs_query__)
            store = None
        xs, xq = self.x_quantile(x_support, x_query, padding_obs_support,
                                 padding_features)
        xs_e = self.x_embedding(xs)
        xq_e = self.x_embedding(xq)
        ys_e, yq_e = self.y_embedding(y_support, padding_obs_support,
                                      n_obs_query)
        support, _ = einops.pack((ys_e, xs_e), "b s * d")
        query, pack_q = einops.pack((yq_e, xq_e), "b s * d")
        if store is None:
            store = {"key": key, "cache": _support_pass(self, support)}
            self._sc_store = store
        bsz = x_support.shape[0]
        for layer, s_ln in zip(self.layers, store["cache"]):
            query = _query_half_layer(layer, query, s_ln, bsz)
        query = self.final_layer_norm(query)
        query = self.final_layer(query)
        y_q, _ = einops.unpack(query, pack_q, "b s * c")
        if self.task == _Task.REGRESSION:
            if self.dim_output == 1:
                return y_q[:, :, 0, 0]
            return y_q[:, :, 0, :]
        return y_q[:, :, 0, :]

    _T2D.forward = _cached_forward

    # -- 3. scope both to one predict call ----------------------------------
    _orig_pred = _TF.predict

    def _sc_pred(self, x_support, y_support, x_query):
        _SUPPORT_CACHE_FREEZE["on"] = True
        self.model._sc_enabled = True
        try:
            return _orig_pred(self, x_support, y_support, x_query)
        finally:
            _SUPPORT_CACHE_FREEZE["on"] = False
            self.model._sc_enabled = False
            if hasattr(self.model, "_sc_store"):
                del self.model._sc_store

    _TF.predict = _sc_pred
    _TF._support_cache_patched = True


def install_protocol_patches() -> None:
    """Benchmark-protocol controls, read from the environment at fit time.

    Two knobs that the TabArena 1h protocol needs and that stock AutoGluon
    has no hyperparameter for. Both are inert unless the variable is set, so
    the patch is installed unconditionally (before ``install_view``, so the
    ``ViewTrainer`` subclass and every fold fitting path inherit it):

    - ``MITRA_FT_BUDGET_S=<seconds>``: wall-clock budget for the fine-tuning
      loop of EACH bag child, overriding the stock behaviour of handing the
      child the remaining task time limit as its budget (``MitraBase.
      _create_config`` puts ``time_limit`` into ``hyperparams["budget"]`` and
      ``TrainerFinetune.train`` stops when it is exceeded).
    - ``MITRA_BAG_SALVAGE=1``: when AutoGluon's fold fitting raises
      ``TimeLimitExceeded`` after at least one bag child has been fitted, keep
      the fitted children as the bag instead of failing the whole model.
      ``SequentialLocalFoldFittingStrategy.after_all_folds_scheduled`` stops
      fitting further folds and records which children were fitted;
      ``BaggedEnsembleModel.add_child`` then skips the never-fitted children
      that ``_fit_folds`` would otherwise register (registering them would
      break predict, since no artifact exists on disk).
    """
    import os

    import autogluon.tabular.models.mitra.sklearn_interface as _si_pr
    from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel as _BEM
    from autogluon.core.models.ensemble.fold_fitting_strategy import (
        SequentialLocalFoldFittingStrategy as _SLFFS,
    )
    from autogluon.core.utils.exceptions import TimeLimitExceeded as _TLE

    if getattr(_si_pr.MitraBase, "_protocol_patched", False):
        return

    _orig_cc = _si_pr.MitraBase._create_config

    def _budget_cc(self, *args, **kwargs):
        cfg, model_cls = _orig_cc(self, *args, **kwargs)
        budget = os.environ.get("MITRA_FT_BUDGET_S")
        if budget:
            cfg.hyperparams["budget"] = float(budget)
        return cfg, model_cls

    _si_pr.MitraBase._create_config = _budget_cc
    _si_pr.MitraBase._protocol_patched = True

    def _after_all_folds_scheduled(self):
        for job in self.jobs:
            try:
                self._fit_fold_model(job)
            except _TLE:
                if os.environ.get("MITRA_BAG_SALVAGE", "0") == "1" and len(self.models) >= 1:
                    fitted = {m if isinstance(m, str) else m.name for m in self.models}
                    self.bagged_ensemble_model._mitra_salvage_fitted = fitted
                    print(
                        f"[mitra-finetune] MITRA_BAG_SALVAGE: time limit reached after "
                        f"{len(self.models)} fitted folds; truncating bag",
                        flush=True,
                    )
                    break
                raise

    _SLFFS.after_all_folds_scheduled = _after_all_folds_scheduled
    _SLFFS._protocol_patched = True

    _orig_add_child = _BEM.add_child
    _orig_fit_folds = _BEM._fit_folds

    def _add_child(self, model, *args, **kwargs):
        fitted = getattr(self, "_mitra_salvage_fitted", None)
        if fitted is not None and isinstance(model, str):
            if model not in fitted and not any(name.endswith(model) for name in fitted):
                return None  # never fitted: the bag was truncated before this child
        return _orig_add_child(self, model, *args, **kwargs)

    def _fit_folds(self, *args, **kwargs):
        self._mitra_salvage_fitted = None
        try:
            return _orig_fit_folds(self, *args, **kwargs)
        finally:
            # transient marker only; never persisted with the model
            if hasattr(self, "_mitra_salvage_fitted"):
                del self._mitra_salvage_fitted

    _BEM.add_child = _add_child
    _BEM._fit_folds = _fit_folds
    _BEM._protocol_patched = True


# -- Fitted-context memo -------------------------------------------------------
# AutoGluon's _train_ensemble halves the fine-tuning context (max_samples_support,
# then max_samples_query) until a child fits the GPU. The eight children of a bag
# run one after another in one process on tables of the same shape, so the
# attempts that failed for the first child fail again for the other seven. The
# memo records the context that fit, keyed on the table shape and the requested
# caps, and later children start there; the ratchet stays in place as fallback.
_FITTED_CONTEXT_MEMO: dict[tuple, tuple[int, int]] = {}
_FITTED_CONTEXT_PENDING: dict[str, tuple | None] = {"key": None}


def fitted_context_key(task, n_rows: int, n_features: int) -> tuple:
    """Table-shape part of the memo key; rows are bucketed so the eight children of a bag share it."""
    return (str(task), int(n_features), int(round(int(n_rows) / 256)))


def install_fitted_context_memo_patch() -> None:
    """Start later bag children at the fine-tuning context that fit the first.

    Two wrappers on ``MitraBase``: ``_train_ensemble`` publishes the table
    shape for the duration of the call and records the caps its trainers ended
    up with; ``_create_config`` completes the key with the caps requested for
    the fit and, when an earlier fit of the same key settled on a smaller
    context, starts from that context instead. Install LAST, so this
    ``_create_config`` wrapper sees the caps every other patch set. Measured
    on the TabArena suite (RTX PRO 6000, 96 GB) only APSFailure and
    kddcup09_appetency (context 4,096) and hiva_agnostic (512) ratchet at all;
    the memo saves about a minute per split there (2,282 s over 816 splits).
    """
    import autogluon.tabular.models.mitra.sklearn_interface as _si_memo

    if getattr(_si_memo.MitraBase, "_fitted_context_memo_patched", False):
        return
    _orig_cc = _si_memo.MitraBase._create_config
    _orig_te = _si_memo.MitraBase._train_ensemble

    def _memo_cc(self, *args, **kwargs):
        cfg, model_cls = _orig_cc(self, *args, **kwargs)
        pending = _FITTED_CONTEXT_PENDING["key"]
        if pending is not None and len(pending) == 3:
            hp = cfg.hyperparams
            requested = (int(hp["max_samples_support"]), int(hp["max_samples_query"]))
            key = pending + requested
            _FITTED_CONTEXT_PENDING["key"] = key
            fitted = _FITTED_CONTEXT_MEMO.get(key)
            if fitted is not None and fitted < requested:
                print(
                    "[mitra-finetune] fitted-context memo: starting fine-tuning at "
                    f"max_samples_support={fitted[0]}, max_samples_query={fitted[1]}, the "
                    "context that fit this table shape earlier in this process",
                    flush=True,
                )
                hp["max_samples_support"], hp["max_samples_query"] = fitted
        return cfg, model_cls

    def _memo_te(self, X_train, y_train, X_valid, y_valid, task, dim_output, *args, **kwargs):
        shape = np.shape(X_train)
        _FITTED_CONTEXT_PENDING["key"] = fitted_context_key(
            task, shape[0], shape[1] if len(shape) > 1 else 1
        )
        try:
            result = _orig_te(self, X_train, y_train, X_valid, y_valid, task, dim_output, *args, **kwargs)
            key = _FITTED_CONTEXT_PENDING["key"]
            trainers = getattr(self, "trainers", None)
            if key is not None and len(key) == 5 and trainers:
                hp = trainers[0].cfg.hyperparams
                _FITTED_CONTEXT_MEMO[key] = (int(hp["max_samples_support"]), int(hp["max_samples_query"]))
            return result
        finally:
            _FITTED_CONTEXT_PENDING["key"] = None

    _si_memo.MitraBase._create_config = _memo_cc
    _si_memo.MitraBase._train_ensemble = _memo_te
    _si_memo.MitraBase._fitted_context_memo_patched = True
