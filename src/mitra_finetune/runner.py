"""Per-view runner: one AutoGluon 8-fold BAGGED Mitra fit per view.

The fit/predict protocol is TabArena's ``AGSingleBagWrapper.fit_custom``
-- the exact class the reference evaluation runs -- imported from the
installed ``tabarena`` package. This module adds no execution logic of
its own: it assembles the wrapper's arguments (checkpoint, Mitra
hyperparameters, time budget), runs ``fit_custom``, and extracts the
out-of-fold validation probabilities alongside the test probabilities.

Two execution shapes, one protocol (``child_main`` is the single source
of truth):

- **in-process** (``in_process=True``): the fit runs in the calling
  process -- one long-lived process per GPU, seeded once at startup,
  fitting its tasks sequentially (the reference harness shape).
- **subprocess** (default): the fit runs in a fresh child process that
  seeds at startup. Required for multi-view modes, whose hooks are
  process-global module patches and cannot coexist in one process.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .modes import ViewSpec
from .patches import (
    install_d2h_sync_patch,
    install_fast_predict_patch,
    install_fitted_context_memo_patch,
    install_gate_cap_patch,
    install_support_cache_patch,
    install_predict_support_cap_patch,
    install_protocol_patches,
    capture_regression_distribution,
    install_reg_ce_patches,
    install_support_cap_patch,
    install_support_select_patch,
    install_use_hf_patch,
)

NUM_BAG_FOLDS = 8

_CHILD = (
    "import sys\n"
    "from mitra_finetune.runner import child_main\n"
    "child_main(sys.argv[1])\n"
)


@contextmanager
def _override_bag_child_validation(mitra_model_cls, bagged_ensemble_cls):
    """Route a bag's external validation set into every Mitra child fit."""
    validation_stack = []
    stats = {"child_fits": 0}
    original_bagged_fit = bagged_ensemble_cls._fit
    original_mitra_fit = mitra_model_cls.fit

    def bagged_fit(self, *args, **kwargs):
        X_val = kwargs.get("X_val")
        y_val = kwargs.get("y_val")
        if X_val is None or y_val is None:
            raise RuntimeError(
                "Official validation was not forwarded to the bagged ensemble"
            )
        validation_stack.append((X_val, y_val))
        try:
            return original_bagged_fit(self, *args, **kwargs)
        finally:
            validation_stack.pop()

    def mitra_fit(self, *args, **kwargs):
        if validation_stack:
            X_val, y_val = validation_stack[-1]
            kwargs["X_val"] = X_val
            kwargs["y_val"] = y_val
            stats["child_fits"] += 1
        return original_mitra_fit(self, *args, **kwargs)

    bagged_ensemble_cls._fit = bagged_fit
    mitra_model_cls.fit = mitra_fit
    try:
        yield stats
    finally:
        mitra_model_cls.fit = original_mitra_fit
        bagged_ensemble_cls._fit = original_bagged_fit


def _as_values(a):
    return a.values if hasattr(a, "values") else np.asarray(a)


def _support_extension_parts(estimator, heldout_active, val_active):
    """Extra (X, y) support blocks for one bag child's final predict.

    Order is fixed: the child's own held-out fold first (HELDOUT_IN_SUPPORT,
    stashed as ``_heldout_in_support`` on the official-validation path, or
    riding the ``_val_in_support`` fit-side stash when the child validated
    on that fold), then the official validation rows (VAL_IN_SUPPORT). A
    toggle whose stash is absent contributes nothing.
    """
    parts = []
    if heldout_active:
        stash = getattr(estimator, "_heldout_in_support", None)
        if stash is not None:
            parts.append(stash)
    if val_active:
        stash = getattr(estimator, "_val_in_support", None)
        if stash is not None:
            parts.append(stash)
    return parts


@contextmanager
def _skip_fold_oof_predictions(fold_strategy_cls):
    """Keep bag children but omit unused held-out-fold inference.

    Official-validation path only. Each child fine-tuned on its bag-train
    fold and validated on the OFFICIAL validation set, so its own held-out
    fold is used by nothing (no OOF predict). It is stashed on the child
    here (``_heldout_in_support``) so HELDOUT_IN_SUPPORT can add it to the
    child's in-context support for the final test predictions, exactly as
    on the no-official-validation path where the fit-side stash already
    holds that fold. The rows go through the SAME ``MitraModel.preprocess``
    the child applied to its support (feature-generator output ->
    label-encoded categoricals) and the labels are the bag's internal
    LabelCleaner-encoded target the child fit on, so the stash is directly
    concatenable with the estimator's ``X``/``y``. Rides the child's pickle.
    """
    stats = {"skipped_folds": 0, "heldout_stashed": 0}
    sentinel = object()
    original_predict_oof = fold_strategy_cls._predict_oof
    original_update = fold_strategy_cls._update_bagged_ensemble
    predict_oof_was_local = "_predict_oof" in fold_strategy_cls.__dict__
    update_was_local = "_update_bagged_ensemble" in fold_strategy_cls.__dict__

    def skip_predict_oof(self, fold_model, fold_ctx):
        _, val_index = fold_ctx["fold"]
        stats["skipped_folds"] += 1

        estimator = getattr(fold_model, "model", None)
        if estimator is not None:
            X_heldout = fold_model.preprocess(self.X.iloc[val_index, :])
            y_heldout = self.y.iloc[val_index]
            estimator._heldout_in_support = (
                _as_values(X_heldout),
                _as_values(y_heldout),
            )
            stats["heldout_stashed"] += 1

        # AutoGluon's child bookkeeping expects prediction timing metadata
        # even when the prediction itself is intentionally omitted.
        fold_model.val_score = None
        fold_model.predict_time = 0.0
        fold_model.predict_1_time = None
        fold_model._predict_n_size = len(val_index)
        fold_model.reduce_memory_size(
            remove_fit=True,
            remove_info=False,
            requires_save=True,
        )
        if not self.bagged_ensemble_model.params.get(
            "save_bag_folds",
            True,
        ):
            fold_model.model = None
        return fold_model, sentinel

    def update_without_oof(self, fold_model, pred_proba, fold_ctx):
        if pred_proba is not sentinel:
            return original_update(self, fold_model, pred_proba, fold_ctx)

        model_to_append = fold_model
        if not self.save_folds:
            fold_model.model = None
        if self.bagged_ensemble_model.low_memory:
            self.bagged_ensemble_model.save_child(
                fold_model,
                verbose=False,
            )
            model_to_append = fold_model.name
        self.models.append(model_to_append)
        self.bagged_ensemble_model._add_child_times_to_bag(
            model=fold_model
        )
        self.bagged_ensemble_model._add_child_num_cpus(
            num_cpus=fold_model.fit_num_cpus
        )
        self.bagged_ensemble_model._add_child_num_gpus(
            num_gpus=fold_model.fit_num_gpus
        )

    fold_strategy_cls._predict_oof = skip_predict_oof
    fold_strategy_cls._update_bagged_ensemble = update_without_oof
    try:
        yield stats
    finally:
        if predict_oof_was_local:
            fold_strategy_cls._predict_oof = original_predict_oof
        else:
            delattr(fold_strategy_cls, "_predict_oof")
        if update_was_local:
            fold_strategy_cls._update_bagged_ensemble = original_update
        else:
            delattr(fold_strategy_cls, "_update_bagged_ensemble")


def _normalize_probabilities(proba_out, wrapper, sorted_labels) -> np.ndarray:
    """Return probabilities in sorted-label column order."""
    import pandas as pd

    if isinstance(proba_out, pd.Series):
        positive_class = wrapper.predictor.positive_class
        positive_proba = proba_out.to_numpy(dtype=np.float64)
        probabilities = np.stack(
            [1.0 - positive_proba, positive_proba],
            axis=1,
        )
        if positive_class == sorted_labels[0]:
            probabilities = probabilities[:, ::-1]
        return probabilities

    return proba_out.reindex(columns=sorted_labels).to_numpy(dtype=np.float64)


def _install_val_in_support_patches():
    """Enable train+val ICL support at test-predict time (VAL_IN_SUPPORT_PATCH).

    Two process-global patches on AutoGluon's Mitra sklearn interface, both
    inert until the module-level toggle ``_mitra_val_in_support_active`` is
    flipped on (``child_main`` does that only around the final test-query
    re-predict, gated on ``MITRA_VAL_IN_SUPPORT=1``):

    1. Fit-side stash: ``MitraClassifier.fit`` / ``MitraRegressor.fit`` keep
       the official validation rows they were checkpointed on
       (``self._val_in_support``). At that point X_val has already been
       column-selected by the API, transformed by AutoGluon's feature
       generator, and run through ``MitraModel.preprocess`` -- the SAME
       pipeline as the stored support ``self.X`` -- and y_val is in the same
       encoded label space as ``self.y``, so the stash is directly
       concatenable. It rides the child's pickle, so bag children reloaded
       from disk keep it. Stashing changes no computation.
    2. Predict-side toggle: while active, each bag child predicts with
       support = its own bag-train subset + the FULL official validation set
       (every child grows by the same val rows). The trainer-level row-cap
       policy is untouched: ``DatasetFinetune`` uniformly subsamples the
       CONCATENATED support down to ``max_samples_support`` (the
       ``MITRA_PREDICT_SUPPORT_CAP`` ratchet in ``child_main``), so when
       train+val exceeds the cap, val rows are dropped with the same
       probability as train rows. ``MitraClassifier.predict`` routes through
       ``predict_proba``, so patching ``predict_proba`` covers both.

    Returns the patched ``sklearn_interface`` module so the caller can flip
    the toggle. Idempotent; composes with the row-cap/reg-CE/view patches
    (disjoint targets: those patch config creation and trainer methods, this
    patches the sklearn estimator's fit/predict entry points).
    """
    import contextlib

    import autogluon.tabular.models.mitra.sklearn_interface as si

    if getattr(si, "_mitra_val_in_support_patched", False):
        return si

    _values = _as_values

    def _make_stash_fit(original_fit):
        def fit(self, X, y, X_val=None, y_val=None, time_limit=None):
            result = original_fit(
                self, X, y, X_val=X_val, y_val=y_val, time_limit=time_limit
            )
            if X_val is not None and y_val is not None:
                self._val_in_support = (_values(X_val), _values(y_val))
            return result

        return fit

    @contextlib.contextmanager
    def _extended_support(estimator):
        # Two toggles, two stashes (see _support_extension_parts). With a
        # single active block this is the original two-array concatenation.
        parts = _support_extension_parts(
            estimator,
            si._mitra_heldout_in_support_active,
            si._mitra_val_in_support_active,
        )
        if not parts:
            yield
            return
        X_orig, y_orig = estimator.X, estimator.y
        # Mutate only inside the try: if either concatenation raises (e.g. a
        # dtype/shape mismatch in the stash), the finally still restores the
        # originals instead of leaving the estimator permanently extended.
        try:
            estimator.X = np.concatenate(
                [np.asarray(X_orig)] + [p[0] for p in parts], axis=0
            )
            estimator.y = np.concatenate(
                [np.asarray(y_orig)] + [p[1] for p in parts], axis=0
            )
            yield
        finally:
            estimator.X, estimator.y = X_orig, y_orig

    def _make_extended_predict(original_predict):
        def predict_fn(self, X):
            with _extended_support(self):
                return original_predict(self, X)

        return predict_fn

    si._mitra_val_in_support_active = False
    si._mitra_heldout_in_support_active = False
    si.MitraClassifier.fit = _make_stash_fit(si.MitraClassifier.fit)
    si.MitraRegressor.fit = _make_stash_fit(si.MitraRegressor.fit)
    si.MitraClassifier.predict_proba = _make_extended_predict(
        si.MitraClassifier.predict_proba
    )
    si.MitraRegressor.predict = _make_extended_predict(si.MitraRegressor.predict)
    si._mitra_val_in_support_patched = True
    return si


def _install_predict_chunk_patch(chunk: int) -> None:
    """Chunk the support stream of the dense (non-flash) Layer.forward.

    Predict-time (grad disabled) only; fine-tune and the flash path delegate
    to the stock forward. Exact-math row partitioning of the three
    peak-memory regions -- the pointwise MLP blocks, the per-row feature
    attention, and the query side of row attention (full k/v kept) -- so
    predict support caps >32768 fit on 80GB for wide tables (at 65536 rows x
    172 cols the stock path peaks ~95 GiB in each MLP and ~85 GiB in row
    attention; chunked it stays under ~65 GiB).
    """
    import einops
    import torch
    from autogluon.tabular.models.mitra._internal.models.tab2d import Layer

    if getattr(Layer, "_pred_chunk_patched", False):
        return

    def _chunked(x, n, fn):
        # Apply fn to row chunks of x, gathering into a buffer that takes the
        # OUTPUT dtype (under autocast the stream is bf16 while layer_norm
        # emits fp32 -- inheriting the input dtype would change the residual
        # add's rounding vs stock and double the buffer).
        out = None
        for i in range(0, n, chunk):
            seg = fn(x, i, i + chunk)
            if out is None:
                out = x.new_empty(
                    x.shape[:1] + (n,) + seg.shape[2:], dtype=seg.dtype
                )
            out[:, i : i + chunk] = seg
        return out

    def _mlp(ln, lin1, lin2, x):
        # Fused ln -> lin1 -> gelu -> lin2 over row chunks; pointwise per
        # token, so identical to the full-tensor sequence.
        return _chunked(
            x,
            x.shape[1],
            lambda t, a, b: lin2(torch.nn.functional.gelu(lin1(ln(t[:, a:b])))),
        )

    orig_fwd = Layer.forward

    def _fwd(
        self,
        support,
        query__,
        padder_support,
        padder_query__,
        batch_size=None,
        padding_obs_support=None,
        padding_obs_query__=None,
        padding_features=None,
    ):
        if (
            (
                self.use_flash_attn
                and padder_support is not None
                and padder_query__ is not None
            )
            or torch.is_grad_enabled()
            or support.shape[1] <= chunk
        ):
            return orig_fwd(
                self,
                support,
                query__,
                padder_support,
                padder_query__,
                batch_size=batch_size,
                padding_obs_support=padding_obs_support,
                padding_obs_query__=padding_obs_query__,
                padding_features=padding_features,
            )
        if batch_size is None:
            batch_size = support.shape[0]
        n_s = support.shape[1]
        # Row attention: full k/v from the support rows; support-side queries
        # chunked (each query row's softmax spans all keys, so partitioning
        # queries is exact). Per-chunk k/v re-projection inside attention1 is
        # negligible next to the attention itself.
        support_residual = support
        query___residual = query__
        support_n = self.layer_norm1(support)
        query___n = self.layer_norm1(query__)
        support_flat = einops.rearrange(support_n, "b s f d -> (b f) s d")
        query___flat = einops.rearrange(query___n, "b s f d -> (b f) s d")
        del support_n, query___n
        # Pre-cast the attention inputs to the autocast dtype: layer_norm1
        # emits fp32 under autocast, but the q/k/v projections cast their
        # input to the autocast dtype anyway, so this is value-identical and
        # halves the dominant full-support tensor (fp32 -> bf16). Skipped
        # (harmlessly) if the autocast API probe fails.
        if support_flat.is_cuda:
            try:
                if torch.is_autocast_enabled("cuda"):
                    try:
                        _ac_dt = torch.get_autocast_dtype("cuda")
                    except (AttributeError, TypeError):
                        _ac_dt = torch.get_autocast_gpu_dtype()
                    support_flat = support_flat.to(_ac_dt)
                    query___flat = query___flat.to(_ac_dt)
            except TypeError:
                if torch.is_autocast_enabled():
                    support_flat = support_flat.to(torch.get_autocast_gpu_dtype())
                    query___flat = query___flat.to(torch.get_autocast_gpu_dtype())
        support_att_flat = _chunked(
            support_flat,
            n_s,
            lambda t, a, b: self.attention1(t[:, a:b], support_flat, support_flat),
        )
        query___att_flat = self.attention1(query___flat, support_flat, support_flat)
        del support_flat, query___flat
        support = support_residual + einops.rearrange(
            support_att_flat, "(b f) s d -> b s f d", b=batch_size
        )
        query__ = query___residual + einops.rearrange(
            query___att_flat, "(b f) s d -> b s f d", b=batch_size
        )
        del support_att_flat, query___att_flat, support_residual, query___residual
        # First MLP block
        support = support + _mlp(self.layer_norm2, self.linear1, self.linear2, support)
        query__ = query__ + _mlp(self.layer_norm2, self.linear1, self.linear2, query__)
        # Feature attention: rows are independent batch entries in the
        # (b s) f d layout, so chunking rows (with per-chunk layer_norm3,
        # itself per-token) is exact.
        def _feat_att(t, a, b):
            seg = einops.rearrange(
                self.layer_norm3(t[:, a:b]), "b s f d -> (b s) f d"
            )
            return einops.rearrange(
                self.attention2(seg, seg, seg), "(b s) f d -> b s f d", b=batch_size
            )

        support_att = _chunked(support, n_s, _feat_att)
        query___feat = einops.rearrange(
            self.layer_norm3(query__), "b s f d -> (b s) f d"
        )
        query___att = einops.rearrange(
            self.attention2(query___feat, query___feat, query___feat),
            "(b s) f d -> b s f d",
            b=batch_size,
        )
        del query___feat
        support = support + support_att
        query__ = query__ + query___att
        del support_att, query___att
        # Second MLP block
        support = support + _mlp(self.layer_norm4, self.linear3, self.linear4, support)
        query__ = query__ + _mlp(self.layer_norm4, self.linear3, self.linear4, query__)
        return support, query__

    Layer.forward = _fwd
    Layer._pred_chunk_patched = True


@contextmanager
def _distribution_scope(enabled: bool):
    """``capture_regression_distribution`` when enabled, else a no-op yielding None."""
    if not enabled:
        yield None
        return
    with capture_regression_distribution() as collector:
        yield collector


def _save_regression_distribution(work, wrapper, X_test, test_proba, collector):
    """Write every bag child's bin distribution for the final test predictions.

    ``collector`` wraps the pass that produced ``test_proba`` (the
    heldout-in-support re-predict, on by default). Without one the final
    predictions came straight out of ``fit_custom`` and the distribution is
    taken from one extra forward-only pass over ``X_test``, which reproduces
    the point predictions unless the training table exceeds the predict-time
    support cap (the gap is logged). Guard: the members' point predictions
    must average to the bagged predictions of the same pass, so no child is
    missing or counted twice (each member's own histogram mean is checked in
    the collector).
    """
    reference = np.asarray(test_proba, dtype=np.float64).reshape(-1)
    if collector is None:
        with capture_regression_distribution() as collector:
            reference = np.asarray(wrapper.predict(X_test), dtype=np.float64).reshape(-1)
        gap = float(np.max(np.abs(reference - np.asarray(test_proba).reshape(-1))))
        print(
            "[mitra-finetune] distribution: taken from a separate forward pass "
            f"(heldout-in-support off); max |gap| to the point predictions = {gap:.3g}",
            flush=True,
        )
    edges, probs, points = collector.stacked()
    bagged = points.mean(axis=0)
    span = float(edges.max() - edges.min())
    if not np.allclose(bagged, reference, rtol=1e-5, atol=1e-6 * span):
        raise RuntimeError(
            "bag members' point predictions do not average to the bagged "
            f"predictions (max gap {float(np.max(np.abs(bagged - reference))):.3g}); "
            f"{probs.shape[0]} members captured"
        )
    print(
        f"[mitra-finetune] distribution: {probs.shape[0]} bag members x "
        f"{probs.shape[1]} rows x {probs.shape[2]} bins captured",
        flush=True,
    )
    np.savez(work / "test_distribution.npz", bin_edges=edges, probabilities=probs)


def child_main(work_dir: str, reseed: bool = True) -> None:
    """Run one bagged view fit described by the payload in ``work_dir``.

    ``reseed=True`` (subprocess shape) seeds every global RNG at startup,
    exactly as the reference evaluation does when its process starts.
    ``reseed=False`` (in-process shape) leaves the process RNG stream
    untouched: the reference harness seeds once per process and its
    sequential tasks consume the shared stream -- it never reseeds per task.
    """
    work = Path(work_dir)
    spec = json.loads((work / "view.json").read_text())
    with open(work / "data.pkl", "rb") as f:
        payload = pickle.load(f)

    view = ViewSpec(**spec)
    problem_type = payload.get("problem_type")
    is_regression = problem_type == "regression"
    # Regression only (predict_distribution): also return every bag child's
    # bin distribution for the FINAL test predictions, captured around the
    # forward-only pass that produces them (see _save_regression_distribution).
    _want_dist = bool(is_regression and payload.get("return_distribution"))
    _dist_capture = None
    if not is_regression:
        # The eval harness passes problem_type="classification"; the
        # binary-vs-multiclass distinction that the quantile-transform gate
        # and the predict-time support cap below key on must come from the
        # labels themselves (mirrors the wrapper-assembly derivation).
        import pandas as _pd

        _n_cls = int(_pd.Series(list(payload["y_train"])).nunique())
        problem_type = "binary" if _n_cls == 2 else "multiclass"

    # The quantile transform is a small-binary-task lever: it wins broadly
    # there, but loses on the categorical-heavy multiclass tasks (anneal,
    # splice), on regression, and on large tables where the transform is
    # estimated on a support-cap subsample yet applied to a much larger
    # predict context (APSFailure, kddcup09). Enable it only when the child
    # fine-tunes on its whole train table within the 16384-row support cap.
    # Runtime gate, NOT a spec mutation: install_view() is once-per-process
    # and sequential tasks in one process must present the SAME ViewSpec.
    import mitra_finetune.hooks as _hooks_mod

    _hooks_mod.QT_TASK_GATE = not (
        problem_type != "binary" or len(payload["y_train"]) > 16384
    )

    # Small-binary fine-tuning learning rate: 3e-6 instead of the recipe's
    # 1e-5 on binary tasks whose whole train table fits the 16384-row support
    # cap. Screens on the rank-dense small binary tasks favoured the gentler
    # rate broadly (blood-transfusion, credit-g, diabetes, Fitness_Club);
    # multiclass (splice, anneal) and large tables (APSFailure,
    # Diabetes130US) lost with it, so they keep the recipe rate. Runtime
    # override, not a spec mutation, for the same once-per-process reason.
    _hooks_mod.LR_TASK_OVERRIDE = (
        3e-6 if (problem_type == "binary" and len(payload["y_train"]) <= 16384) else None
    )

    # Unconditional, and before install_view: the corrupted-child-save race
    # is upstream AutoGluon, so it hits stock views (base mode) too, which
    # deliberately run without any hooks.
    install_d2h_sync_patch()

    # Benchmark-protocol controls (MITRA_FT_BUDGET_S, MITRA_BAG_SALVAGE): inert
    # unless the variables are set, so installed unconditionally as well.
    install_protocol_patches()

    if is_regression:
        # Switch AutoGluon Mitra from its MSE scalar head to the champion
        # cross-entropy bin head (must precede install_view so ViewTrainer
        # inherits the mean decode). n_bins is the checkpoint's head width.
        config_path = Path(payload["checkpoint"]) / "config.json"
        n_bins = int(json.loads(config_path.read_text())["dim_output"])
        install_reg_ce_patches(n_bins)

    install_use_hf_patch()

    # A stock view (base mode) equals stock AutoGluon Mitra fine-tuning
    # exactly, so apart from the correctness fixes above it runs on UNPATCHED
    # AutoGluon -- the same execution as the reference pipeline, no view hooks
    # involved. Only non-stock views (e.g. the shipped lr=1e-5 recipe) need
    # the module patches below.

    _gate_cap = payload.get("gate_cap")
    if _gate_cap:
        # Cap the in-context support/query so the gate's wide-table holdout
        # fits don't OOM-ratchet. Lives only in this per-fit subprocess.
        install_gate_cap_patch(_gate_cap)

    # Fine-tune in-context support cap (default-on, task-type-conditioned to
    # the frozen final config): regression lifts the stock 8192-row cap to
    # 20480 and classification to 16384 so large-N tasks fine-tune on more
    # context (a no-op below ~14k train rows: child train = N * 2/3 * 7/8
    # <= 8192). The classification raise is safe only in combination with
    # the fast-predict subsample guard and the 16384 predict cap: without
    # them the first bag fold pays an OOM-restart cycle plus a wide-chunk
    # predict cascade, which pushes that fold over AutoGluon's per-fold time
    # projection and silently truncates the bag.
    # MITRA_SUPPORT_CAP overrides (0 disables).
    _sup_cap_default = 20480 if is_regression else 16384
    _sup_cap = int(os.environ.get("MITRA_SUPPORT_CAP", str(_sup_cap_default)))

    # Predict-time support cap (default-on, task-type-conditioned to the frozen
    # final config): regression and multiclass widen the forward-only predict
    # context to 32768; binary uses 16384 (with the OOM-halving ratchet below).
    # The lower binary cap is deliberate: it keeps the per-fold out-of-fold
    # predict fast enough that AutoGluon's time projection never truncates the
    # bag on large-N tasks, which evaluation showed outweighs the wider
    # context (bigger caps scored worse end to end). Multiclass takes the wide
    # cap: the only large-N multiclass task (SDSS17) scores better with it and
    # every other multiclass task sits far below the cap, so it is a no-op
    # there. MITRA_PREDICT_SUPPORT_CAP overrides.
    _pred_cap_default = 16384 if problem_type == "binary" else 32768
    _pred_cap = int(
        os.environ.get("MITRA_PREDICT_SUPPORT_CAP", str(_pred_cap_default))
    )
    _pred_floor = max(1, int(os.environ.get("MITRA_PREDICT_SUPPORT_FLOOR", "8192")))

    if _sup_cap > 0:
        # Row-side lever: lift the stock 8192-row in-context support cap for
        # fine-tune (large-N tasks truncate to ~20% of rows).
        install_support_cap_patch(_sup_cap)

    # Predict-time in-context support SELECTION (default-on for classification):
    # once the widened predict context is subsampled, class-balance the kept
    # rows on BINARY tasks (``balanced_bin``) instead of the stock uniform draw;
    # a no-op for multiclass/regression and when the support is not subsampled.
    # Regression keeps the stock random draw. Installed BEFORE fast-predict so
    # the fixed-support RNG-freeze wraps this balanced draw. MITRA_SUPPORT_SELECT
    # overrides the predict draw (empty = stock random); MITRA_SUPPORT_SELECT_FT
    # (default empty) the fine-tune draw.
    _sup_select = os.environ.get(
        "MITRA_SUPPORT_SELECT", "" if is_regression else "balanced_bin"
    )
    _sup_select_ft = os.environ.get("MITRA_SUPPORT_SELECT_FT", "")
    if _sup_select or _sup_select_ft:
        install_support_select_patch(_sup_select, _sup_select_ft)

    # FAST_PREDICT (default ON): predict in 16384-row query chunks whether or
    # not the support fits the cap (stock: 1024). When the support fits, the
    # subsample is drawn once per test set (a single-chunk predict is bit-exact
    # with stock); when it exceeds the cap, every chunk still gets a fresh
    # capped draw, as in stock, so the prediction is the same in expectation
    # and 5 to 8 times cheaper (see install_fast_predict_patch). Set
    # MITRA_FAST_PREDICT=0 to restore stock behavior everywhere. Installed
    # BEFORE the support-cap patch so it is the INNER predict wrapper: a
    # predict-time CUDA OOM shrinks the (quality-neutral) query chunk first,
    # down to the floor when the support fits and to the stock chunk when it
    # is capped, and only then escalates to the (quality-costing) support-cap
    # halving.
    _fast_predict = os.environ.get("MITRA_FAST_PREDICT", "1") == "1"
    _fast_qchunk = max(1, int(os.environ.get("MITRA_FAST_PREDICT_QCHUNK", "16384")))
    _fast_qchunk_floor = max(
        1, int(os.environ.get("MITRA_FAST_PREDICT_QCHUNK_FLOOR", "256"))
    )
    if _fast_predict:
        install_fast_predict_patch(_fast_qchunk, _fast_qchunk_floor)

    if _pred_cap > 0:
        # Predict-only support widening (default-on, cap 32768): fine-tune keeps
        # the stock 8192-row support cap; only the forward-only predict context
        # is widened, with an OOM-halving ratchet down to the floor so the large
        # cap is a SAFE universal default (wide tables fall back automatically).
        install_predict_support_cap_patch(_pred_cap, _pred_floor)

    # SUPPORT_CACHE_PATCH: default OFF; MITRA_SUPPORT_CACHE=1 encodes the
    # support stream once per predict call and reuses it for every query
    # chunk (~5x faster prediction; see patches.install_support_cache_patch
    # for the exactness argument and the flash-kernel / support-redraw
    # caveats). Budget via MITRA_SUPPORT_CACHE_GB (default 6).
    if os.environ.get("MITRA_SUPPORT_CACHE", "0") == "1":
        install_support_cache_patch(
            float(os.environ.get("MITRA_SUPPORT_CACHE_GB", "6"))
        )

    # PREDICT_CHUNK_PATCH: default OFF; MITRA_PREDICT_CHUNK=<rows> chunks the
    # support stream at predict time (see _install_predict_chunk_patch).
    # The chunk is used as given: when the fit-time support cap exceeds it,
    # mid-fit validation forwards (no_grad) also run chunked -- value-identical
    # math, accepted. (An earlier clamp to the fit cap doubled the chunked MLP
    # transients and caused predict-time OOM ratchets on wide tables.)
    _pred_chunk = int(os.environ.get("MITRA_PREDICT_CHUNK", "0"))
    if _pred_chunk > 0:
        _install_predict_chunk_patch(_pred_chunk)
    # VAL_IN_SUPPORT_PATCH: default OFF; MITRA_VAL_IN_SUPPORT=1 folds the
    # official validation rows into the ICL support of the FINAL test-query
    # predictions only (see the re-predict block after fit below).
    _val_in_sup = os.environ.get("MITRA_VAL_IN_SUPPORT", "0") == "1"
    # HELDOUT_IN_SUPPORT (default ON, part of the frozen configuration behind
    # the reported board; MITRA_HELDOUT_IN_SUPPORT=0 disables it): folds each
    # bag child's OWN held-out fold into that child's ICL support for the
    # FINAL test-query predictions only, so every child predicts with the
    # full outer training set as support. Fit, fine-tuning validation, OOF
    # and any selection all still run on train-fold-only support. Applies on
    # both paths: without an official validation set the held-out fold is the
    # X_val the child validated on (fit-side stash); with one, the child
    # validated on the official set and its held-out fold is stashed by
    # _skip_fold_oof_predictions. Reuses the VAL_IN_SUPPORT concat machinery
    # (same predict-side concat, same row-cap policy on the concatenation).
    _heldout_in_sup = os.environ.get("MITRA_HELDOUT_IN_SUPPORT", "1") == "1"
    if _val_in_sup or _heldout_in_sup:
        _si_vs = _install_val_in_support_patches()
    if not view.is_stock:
        from mitra_finetune.hooks import install_view
        install_view(view)

    # FITTED_CONTEXT_MEMO (default ON; MITRA_FITTED_CONTEXT_MEMO=0 disables):
    # later bag children start at the fine-tuning context that fit the first
    # child instead of repeating its out-of-memory attempts. Installed LAST so
    # its _create_config wrapper sees the caps every patch above set.
    if os.environ.get("MITRA_FITTED_CONTEXT_MEMO", "1") == "1":
        install_fitted_context_memo_patch()

    if reseed:
        # Mitra's fine-tuning consumes the GLOBAL torch RNG stream, so a
        # different stream at fit time yields a small systematic accuracy
        # offset (measured on churn: ~+0.007 error without this).
        import random as _random
        import torch as _torch
        _seed = int(payload.get("seed", 0))
        _random.seed(_seed)
        np.random.seed(_seed)
        _torch.manual_seed(_seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(_seed)

    import pandas as pd
    from autogluon.tabular.models.mitra.mitra_model import MitraModel

    # TabArena's wrapper IS the fit/predict protocol behind every reported
    # number: LabelCleaner label handling, the byte-matched predictor.fit,
    # deterministic test-row shuffle + inverse permutation, memory-tracked
    # fit/predict, per-task teardown. Imported, not reimplemented -- the
    # hand-rolled reconstruction of this flow was measurably not equivalent
    # (task-level collapse amplification on H100).
    from tabarena.benchmark.exec_models.autogluon import AGSingleBagWrapper

    X_train = payload["X_train"]
    X_test = payload["X_test"]
    X_val = payload.get("X_val")
    y_val = payload.get("y_val")
    if (X_val is None) != (y_val is None):
        raise ValueError("X_val and y_val must either both be provided or both be None")
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
        X_test.columns = X_train.columns
    if X_val is not None and not isinstance(X_val, pd.DataFrame):
        X_val = pd.DataFrame(X_val, columns=X_train.columns)

    # y must be a Series ALIGNED to X_train's index: the wrapper joins the
    # label column onto X by index, and a default-indexed Series against an
    # OpenML-indexed frame produces NaN labels ("label column cannot contain
    # non-finite values").
    y_train = payload["y_train"]
    if isinstance(y_train, pd.Series):
        y_train = pd.Series(
            y_train.to_numpy(), index=X_train.index, name=y_train.name
        )
    else:
        y_train = pd.Series(np.asarray(y_train), index=X_train.index)
    if y_val is not None:
        if isinstance(y_val, pd.Series):
            y_val = pd.Series(
                y_val.to_numpy(), index=X_val.index, name=y_val.name
            )
        else:
            y_val = pd.Series(np.asarray(y_val), index=X_val.index)

    if is_regression:
        problem_type = "regression"
    else:
        n_classes = int(y_train.nunique())
        problem_type = "binary" if n_classes == 2 else "multiclass"

    # Argument assembly replicated from AGModelBagExperiment.__init__ +
    # ExperimentConstructor._apply_debug_fold_fitting (what eval.py builds):
    # the time budget lives in ag_args_ensemble["ag.max_time_limit"] for a
    # bagged fit (NOT fit_kwargs["time_limit"]), fold fitting is pinned to
    # sequential_local, and raise_on_model_failure goes into fit_kwargs.
    from autogluon.core.metrics import get_metric

    # AG >= 1.6 loads custom weights via hf_model=<local dir in
    # Tab2D.save_pretrained format>; the state_dict_* hyperparameters were
    # removed (1.6 pops them with a deprecation warning and IGNORES them).
    # The checkpoint in the payload is already a converted directory --
    # resolve_checkpoint converts raw .pt files on the fly.
    model_hp = {
        "hf_model": payload["checkpoint"],
        "ag_args_fit": {"max_rows": 1000000, "max_features": 50000},
        "ag_args_ensemble": {
            "ag.max_time_limit": payload["time_limit"],
            "fold_fitting_strategy": "sequential_local",
            # This reaches MitraModel.seed and its private support/query RNG.
            "model_random_seed": int(payload.get("seed", 0)),
        },
    }
    if payload.get("fine_tune", True) is False:
        model_hp["fine_tune"] = False

    fit_kwargs = {
        "num_bag_folds": payload["num_bag_folds"],
        "num_bag_sets": 1,
        "num_gpus": 1,
        "raise_on_model_failure": True,
    }
    if X_val is not None:
        # AutoGluon otherwise rejects tuning_data with bagging. Its bagged
        # model normally ignores tuning_data during child fitting; the scoped
        # override below deliberately routes it into each Mitra child.
        fit_kwargs["use_bag_holdout"] = True
        official_X_val = X_val
        official_y_val = y_val

        class OfficialValidationBagWrapper(AGSingleBagWrapper):
            def fit(self, X, y, X_val=None, y_val=None):
                return super().fit(
                    X,
                    y,
                    X_val=official_X_val,
                    y_val=official_y_val,
                )

        wrapper_cls = OfficialValidationBagWrapper
    else:
        wrapper_cls = AGSingleBagWrapper

    wrapper = wrapper_cls(
        model_cls=MitraModel,
        model_hyperparameters=model_hp,
        problem_type=problem_type,
        eval_metric=get_metric(
            payload.get("eval_metric")
            or ("root_mean_squared_error" if is_regression else "log_loss"),
            problem_type=problem_type,
        ),
        fit_kwargs=fit_kwargs,
    )
    if X_val is None:
        out = wrapper.fit_custom(X_train, y_train, X_test)
    else:
        from autogluon.core.models.ensemble.bagged_ensemble_model import (
            BaggedEnsembleModel,
        )
        from autogluon.core.models.ensemble.fold_fitting_strategy import (
            SequentialLocalFoldFittingStrategy,
        )

        with (
            _override_bag_child_validation(
                MitraModel,
                BaggedEnsembleModel,
            ) as override_stats,
            _skip_fold_oof_predictions(
                SequentialLocalFoldFittingStrategy,
            ) as oof_stats,
        ):
            out = wrapper.fit_custom(X_train, y_train, X_test)
        expected_child_fits = int(payload["num_bag_folds"])
        if override_stats["child_fits"] != expected_child_fits:
            raise RuntimeError(
                "Official validation override reached "
                f"{override_stats['child_fits']} Mitra bag children; "
                f"expected {expected_child_fits}"
            )
        if oof_stats["skipped_folds"] != expected_child_fits:
            raise RuntimeError(
                "Held-out OOF override reached "
                f"{oof_stats['skipped_folds']} bag children; "
                f"expected {expected_child_fits}"
            )
        if _heldout_in_sup and oof_stats["heldout_stashed"] != expected_child_fits:
            raise RuntimeError(
                "Held-out fold stash reached "
                f"{oof_stats['heldout_stashed']} bag children; "
                f"expected {expected_child_fits}"
            )
        print(
            "[mitra-finetune] official validation: "
            f"rows={len(X_val)}, child_fits={override_stats['child_fits']}, "
            f"heldout_oof_skipped={oof_stats['skipped_folds']}, "
            f"heldout_stashed={oof_stats['heldout_stashed']}",
            flush=True,
        )

    if is_regression:
        # Regression yields point predictions (fit_custom returns them under
        # "predictions"; "probabilities" is None). No class-column ordering.
        test_proba = np.asarray(out["predictions"], dtype=np.float64)
        if X_val is not None:
            val_proba = np.asarray(
                wrapper.predict(X_val), dtype=np.float64
            )
        else:
            # Out-of-fold point predictions: row i predicted by the bag child
            # that did not train on it. Only used for multi-view selection;
            # the base single-view mode ignores it.
            sim = wrapper.get_oof()
            val_proba = np.asarray(sim["pred_proba_dict_val"], dtype=np.float64)
    else:
        # Column order: for multiclass the wrapper returns a DataFrame whose
        # columns are the ORIGINAL class labels (LabelCleaner inverse); for
        # binary it may return a Series of the positive-class probability
        # (AutoGluon's own predict_proba does this for binary problems) with no
        # column labels to sort. Normalize both to sorted-label
        # [P(low), P(high)] order so callers can map y via sorted(unique)
        # deterministically, independent of AutoGluon's internal ordering.
        sorted_labels = sorted(y_train.unique())
        test_proba = _normalize_probabilities(
            out["probabilities"],
            wrapper,
            sorted_labels,
        )

        if X_val is not None:
            val_proba = _normalize_probabilities(
                wrapper.predict_proba(X_val),
                wrapper,
                sorted_labels,
            )
        else:
            # Out-of-fold validation probabilities cover every training row:
            # row i is predicted by the bag child that did not train on it.
            sim = wrapper.get_oof()
            val_proba = np.asarray(sim["pred_proba_dict_val"], dtype=np.float64)
            if val_proba.ndim == 1:
                val_proba = np.stack([1.0 - val_proba, val_proba], axis=1)
            internal_labels = list(sim["ordered_class_labels"])
            order = [internal_labels.index(c) for c in sorted_labels]
            val_proba = val_proba[:, order]

    if X_val is not None and (_val_in_sup or _heldout_in_sup):
        # Official-validation path: ONE extra forward-only pass over the
        # test-role query rows with each bag child's in-context support
        # extended by
        #   (a) HELDOUT_IN_SUPPORT: the child's own held-out fold (stashed in
        #       _skip_fold_oof_predictions). The child fine-tuned on its
        #       bag-train fold and validated on the official set, so that
        #       fold was used by nothing; adding it makes every child predict
        #       with the full outer training set as support, exactly as on
        #       the no-official-validation path below;
        #   (b) VAL_IN_SUPPORT (MITRA_VAL_IN_SUPPORT=1): the official
        #       validation rows (fit-side stash).
        # Selection-before-inclusion ordering holds by construction:
        # everything computed above ran with both toggles OFF -- the fit, the
        # fit_custom query pass, and val_proba (the selection input for view
        # weights/temperature and any downstream val-based choice) all used
        # train-fold-only support. Fine-tuning is NOT repeated.
        #
        # ``val_query_rows`` marks how many LEADING X_test rows are
        # validation-role queries (the hierarchical path predicts val+test
        # in one combined table): those keep their train-fold predictions
        # from the pass above; only the true test tail is re-predicted with
        # the extended support.
        _val_head = int(payload.get("val_query_rows") or 0)
        _X_ext = X_test.iloc[_val_head:] if _val_head else X_test
        _si_vs._mitra_heldout_in_support_active = bool(_heldout_in_sup)
        _si_vs._mitra_val_in_support_active = bool(_val_in_sup)
        try:
            if is_regression:
                if _want_dist and _val_head:
                    raise RuntimeError(
                        "return_distribution does not support validation-role "
                        "query rows"
                    )
                with _distribution_scope(_want_dist) as _dist_capture:
                    _ext = np.asarray(wrapper.predict(_X_ext), dtype=np.float64)
            else:
                _ext = _normalize_probabilities(
                    wrapper.predict_proba(_X_ext),
                    wrapper,
                    sorted_labels,
                )
        finally:
            _si_vs._mitra_heldout_in_support_active = False
            _si_vs._mitra_val_in_support_active = False
        if _val_head:
            test_proba = np.concatenate([test_proba[:_val_head], _ext], axis=0)
        else:
            test_proba = _ext
        if _heldout_in_sup:
            print(
                "[mitra-finetune] heldout-in-support: re-predicted "
                f"{len(_X_ext)} query rows with each bag child's held-out "
                f"fold added to its support "
                f"(stashed on {oof_stats['heldout_stashed']} children)",
                flush=True,
            )
        if _val_in_sup:
            print(
                "[mitra-finetune] val-in-support: extended predict support by "
                f"{len(X_val)} validation rows for {len(_X_ext)} of "
                f"{len(X_test)} query rows",
                flush=True,
            )

    if _heldout_in_sup and X_val is None:
        # HELDOUT_IN_SUPPORT re-predict (no official validation set, e.g.
        # TabArena): one extra forward-only pass over the test rows with each
        # bag child's support = its train fold + its own held-out fold (= the
        # full outer training set). Here the held-out fold IS the X_val the
        # child validated on, so the fit-side stash already holds it and the
        # val toggle adds it. No fine-tuning is repeated and val_proba (OOF,
        # train-fold support) is untouched. The official-validation path
        # reaches the same support composition via the block above.
        _si_vs._mitra_val_in_support_active = True
        try:
            if is_regression:
                with _distribution_scope(_want_dist) as _dist_capture:
                    test_proba = np.asarray(
                        wrapper.predict(X_test), dtype=np.float64
                    )
            else:
                test_proba = _normalize_probabilities(
                    wrapper.predict_proba(X_test),
                    wrapper,
                    sorted_labels,
                )
        finally:
            _si_vs._mitra_val_in_support_active = False
        print(
            "[mitra-finetune] heldout-in-support: re-predicted "
            f"{len(X_test)} query rows with each bag child's held-out fold "
            "added to its support",
            flush=True,
        )

    if _want_dist:
        _save_regression_distribution(
            work, wrapper, X_test, test_proba, _dist_capture
        )

    np.save(work / "val_proba.npy", val_proba)
    np.save(work / "test_proba.npy", test_proba)

    # Reference per-task teardown (exec_models/autogluon.py cleanup(),
    # invoked by the experiment runner after every task).
    wrapper.cleanup()
    (work / "done").touch()



def _ft_env_view(view: ViewSpec) -> ViewSpec:
    """Env-gated fine-tune hyperparameter overrides for screening.

    All gates default OFF (env unset = the frozen recipe, bit-identical).
    MITRA_FT_STEPS ("none" = wall-clock only), MITRA_FT_LR, MITRA_FT_WD,
    MITRA_FT_WARMUP, MITRA_FT_VALINT, MITRA_FT_QT=1.
    """
    import dataclasses
    ov = {}
    v = os.environ.get("MITRA_FT_STEPS")
    if v:
        ov["steps"] = None if v.lower() == "none" else int(v)
    v = os.environ.get("MITRA_FT_LR")
    if v:
        ov["lr"] = float(v)
    v = os.environ.get("MITRA_FT_WD")
    if v:
        ov["weight_decay"] = float(v)
    v = os.environ.get("MITRA_FT_WARMUP")
    if v:
        ov["warmup_steps"] = int(v)
    v = os.environ.get("MITRA_FT_VALINT")
    if v:
        ov["validation_interval"] = int(v)
    if os.environ.get("MITRA_FT_QT") == "1":
        ov["quantile_transform"] = True
    if os.environ.get("MITRA_FT_CF") == "1":
        ov["category_frequency"] = True
    if not ov:
        return view
    view = dataclasses.replace(view, name=view.name + "+ftenv", **ov)
    print(f"[mitra-finetune] FT env overrides active: {ov}", flush=True)
    return view


def run_view(
    view: ViewSpec,
    checkpoint: str,
    X_train,
    y_train,
    X_test,
    time_limit: int,
    device: str,
    X_val=None,
    y_val=None,
    eval_metric: str | None = None,
    in_process: bool = False,
    seed: int = 0,
    problem_type: str | None = None,
    fine_tune: bool = True,
    num_bag_folds: int = NUM_BAG_FOLDS,
    gate_cap: int | None = None,
    val_query_rows: int = 0,
    return_distribution: bool = False,
) -> tuple:
    """Train one bagged view and return ``(oof_val_proba, test_proba)``.

    With ``return_distribution=True`` (regression only) a third element is
    returned: ``{"bin_edges": [M, n_bins + 1], "probabilities": [M, n_test,
    n_bins]}``, the bag children's bin distributions behind ``test_proba``
    (see ``_save_regression_distribution``).

    ``in_process=True`` runs the fit in the calling process (reference
    harness shape: one long-lived process per GPU, seeded once at startup,
    tasks sequential). Only valid for single-view modes; hooks are
    process-global patches, so a second distinct view in the same process
    raises. ``in_process=False`` runs it in a fresh subprocess.
    """
    if return_distribution and problem_type != "regression":
        raise ValueError(
            "return_distribution is regression-only; got "
            f"problem_type={problem_type!r}"
        )
    view = _ft_env_view(view)
    with tempfile.TemporaryDirectory(prefix="mitra_view_") as tmp:
        work = Path(tmp)
        (work / "view.json").write_text(json.dumps(asdict(view)))
        with open(work / "data.pkl", "wb") as f:
            pickle.dump(
                {
                    "checkpoint": checkpoint,
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "X_val": X_val,
                    "y_val": y_val,
                    "seed": int(seed),
                    "eval_metric": eval_metric,
                    "time_limit": time_limit,
                    "num_bag_folds": int(num_bag_folds),
                    "fine_tune": fine_tune,
                    "gate_cap": gate_cap,
                    # Leading X_test rows that are validation-role queries;
                    # only read when MITRA_VAL_IN_SUPPORT=1.
                    "val_query_rows": int(val_query_rows),
                    "return_distribution": bool(return_distribution),
                    "device": device,
                    "problem_type": problem_type,
                },
                f,
            )
        if in_process:
            child_main(str(work), reseed=False)
        else:
            result = subprocess.run(
                [sys.executable, "-c", _CHILD, str(work)],
                capture_output=True,
                text=True,
            )
            # Surface the child's diagnostics (view banner) in the parent log
            # even on success; we were blind to them before.
            for line in result.stdout.splitlines():
                if "[mitra-finetune]" in line:
                    print(line, flush=True)
            if not (work / "done").exists():
                raise RuntimeError(
                    f"View {view.name!r} failed:\n{result.stdout[-2000:]}"
                    f"\n{result.stderr[-2000:]}"
                )
        val_proba = np.load(work / "val_proba.npy")
        test_proba = np.load(work / "test_proba.npy")
        if not return_distribution:
            return val_proba, test_proba
        with np.load(work / "test_distribution.npz") as saved:
            distribution = {
                "bin_edges": saved["bin_edges"],
                "probabilities": saved["probabilities"],
            }
        return val_proba, test_proba, distribution
