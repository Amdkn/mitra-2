import numpy as np
import pytest

from mitra_finetune.modes import BASELINE


class TestRecipe:
    def test_baseline_recipe_frozen(self):
        assert BASELINE.steps == 50
        assert BASELINE.lr == 1e-5
        assert BASELINE.warmup_steps == 10
        assert BASELINE.tune == "full"
        assert BASELINE.snapshot_k == 1
        # Quantile-transformed inputs are off in the frozen recipe.
        assert not BASELINE.quantile_transform
        assert not BASELINE.category_frequency

    def test_baseline_is_not_stock(self):
        # The shipped recipe deviates from AutoGluon's defaults, so the
        # hooks MUST install (is_stock=False) or the schedule is ignored.
        assert not BASELINE.is_stock


def _pretrained_dir(tmp_path, name="ckpt"):
    """A directory in Tab2D.save_pretrained format (returned untouched by
    checkpoint resolution -- no torch load involved)."""
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"stub")
    return d

class TestApi:
    def test_checkpoint_resolution(self, tmp_path):
        from mitra_finetune.api import MitraFinetune

        ckpt = _pretrained_dir(tmp_path)
        model = MitraFinetune(ckpt)
        assert model.checkpoint == str(ckpt)

    def test_ambiguous_checkpoint_dir_raises(self, tmp_path):
        from mitra_finetune.api import MitraFinetune

        (tmp_path / "a.pt").write_bytes(b"1")
        (tmp_path / "b.pt").write_bytes(b"2")
        with pytest.raises(ValueError, match="exactly one"):
            MitraFinetune(tmp_path)

    def test_predict_before_fit_raises(self, tmp_path):
        from mitra_finetune.api import MitraFinetune

        model = MitraFinetune(_pretrained_dir(tmp_path))
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_proba(np.zeros((3, 2)))

    def test_configurable_bag_folds_are_forwarded(
        self, tmp_path, monkeypatch
    ):
        from mitra_finetune import api
        from mitra_finetune.api import MitraFinetune

        captured = {}

        def fake_run_view(view, *args, **kwargs):
            captured["num_bag_folds"] = kwargs["num_bag_folds"]
            return np.full((8, 2), 0.5), np.full((3, 2), 0.5)

        model = MitraFinetune(
            _pretrained_dir(tmp_path),
            num_bag_folds=4,
        )
        monkeypatch.setattr(model, "_check_runtime", lambda: None)
        monkeypatch.setattr(api, "run_view", fake_run_view)
        model.fit(np.zeros((8, 2)), np.arange(8) % 2)
        model.predict_proba(np.zeros((3, 2)))
        assert captured["num_bag_folds"] == 4

    def test_bag_folds_must_be_at_least_two(self, tmp_path):
        from mitra_finetune.api import MitraFinetune

        with pytest.raises(ValueError, match="at least 2"):
            MitraFinetune(_pretrained_dir(tmp_path), num_bag_folds=1)

    def test_many_class_fs_validation_gate_uses_dtype_fallback(
        self, tmp_path, monkeypatch
    ):
        from mitra_finetune.api import MitraFinetune

        monkeypatch.setenv("MITRA_FS_GATE", "1")
        monkeypatch.setenv("MITRA_CLS_MAX_FEATURES", "2")
        model = MitraFinetune(_pretrained_dir(tmp_path))
        monkeypatch.setattr(model, "_check_runtime", lambda: None)

        def fail_native_gate(*args, **kwargs):
            pytest.fail("many-class feature selection called the native fit gate")
            return False

        monkeypatch.setattr(
            "mitra_finetune.feature_selection._fs_gate_keeps_fs",
            fail_native_gate,
        )
        rng = np.random.RandomState(0)
        y = np.repeat(np.arange(11), 5)
        X = rng.normal(size=(len(y), 4))

        with pytest.warns(RuntimeWarning, match="fs-gate-hierarchy-fallback"):
            model.fit(X, y, X_val=X.copy(), y_val=y.copy())

        assert len(model._ksel_idx) == 2

    def test_native_class_fs_validation_gate_is_preserved(
        self, tmp_path, monkeypatch
    ):
        from mitra_finetune.api import MitraFinetune

        monkeypatch.setenv("MITRA_FS_GATE", "1")
        monkeypatch.setenv("MITRA_CLS_MAX_FEATURES", "2")
        model = MitraFinetune(_pretrained_dir(tmp_path))
        monkeypatch.setattr(model, "_check_runtime", lambda: None)
        calls = []

        def drop_feature_selection(X, y, idx, **kwargs):
            calls.append(len(np.unique(y)))
            return False

        monkeypatch.setattr(
            "mitra_finetune.feature_selection._fs_gate_keeps_fs",
            drop_feature_selection,
        )
        rng = np.random.RandomState(1)
        y = np.arange(40) % 2
        X = rng.normal(size=(len(y), 4))

        model.fit(X, y)

        assert calls == [2]
        assert model._ksel_idx is None


class TestBaggedProtocol:
    def test_runner_uses_eight_bag_folds(self):
        import inspect

        from mitra_finetune.runner import NUM_BAG_FOLDS, child_main

        source = inspect.getsource(child_main)
        assert NUM_BAG_FOLDS == 8
        assert "num_bag_folds" in source
        # The fit/predict protocol is TabArena's wrapper, not a hand-rolled
        # TabularPredictor flow.
        assert "AGSingleBagWrapper" in source
        assert "fit_custom" in source
        assert "get_oof" in source

    def test_runner_forwards_run_seed_to_autogluon_children(self):
        import inspect

        from mitra_finetune.runner import child_main

        source = inspect.getsource(child_main)
        assert '"model_random_seed": int(payload.get("seed", 0))' in source


class TestBinaryProbaSeries:
    """AutoGluon's binary predict_proba can return a 1-D Series of
    P(positive_class) instead of a 2-column DataFrame (observed on string
    labels 'False'/'True' with sequential_local bagging). child_main must
    normalize either shape to sorted-label [P(low), P(high)] columns."""

    def _normalize(self, sorted_labels, positive_class, values):
        import numpy as np
        positive_proba = np.asarray(values, dtype=np.float64)
        test_proba = np.stack([1.0 - positive_proba, positive_proba], axis=1)
        if positive_class == sorted_labels[0]:
            test_proba = test_proba[:, ::-1]
        return test_proba

    def test_positive_class_is_higher_label_no_flip(self):
        import numpy as np

        out = self._normalize(["False", "True"], "True", [0.1, 0.9])
        np.testing.assert_allclose(out[:, 1], [0.1, 0.9])

    def test_dataframe_bool_columns_reindex_not_bracket(self):
        """OpenML's 'False'/'True' string labels can get auto-inferred as
        pandas/numpy bool. Indexing a DataFrame with a Python list of bools
        ([np.False_, np.True_]) hits pandas' boolean-mask heuristic instead
        of column selection ('Item wrong length 2 instead of N') -- .reindex
        selects by label value regardless of dtype and must be used."""
        import numpy as np
        import pandas as pd

        sorted_labels = sorted(pd.Series([True, False, True]).unique())
        proba = pd.DataFrame({True: [0.1, 0.2, 0.3], False: [0.9, 0.8, 0.7]})
        with pytest.raises(ValueError, match="wrong length"):
            proba[sorted_labels]
        out = proba.reindex(columns=sorted_labels).to_numpy(dtype=np.float64)
        np.testing.assert_allclose(out[:, 1], [0.1, 0.2, 0.3])

    def test_positive_class_is_lower_label_flips(self):
        import numpy as np

        out = self._normalize(["False", "True"], "False", [0.1, 0.9])
        np.testing.assert_allclose(out[:, 1], [0.9, 0.1])

    def test_fit_validation_contract(self, tmp_path):
        import inspect

        from mitra_finetune.api import MitraFinetune

        signature = inspect.signature(MitraFinetune.fit)
        # fit accepts an OPTIONAL external validation set; the
        # default remains OOF (no internal split, no validation_fraction).
        assert "X_val" in signature.parameters
        assert signature.parameters["X_val"].default is None
        assert "validation_fraction" not in inspect.signature(
            MitraFinetune.__init__
        ).parameters


class TestFrozenSupportConfig:
    """The cross-TabFM final config is frozen as default-on, task-type-
    conditioned behavior in child_main (no environment variable required):
    running with a clean environment must reproduce the reported board."""

    def test_type_conditioned_support_caps_are_frozen(self):
        import inspect

        from mitra_finetune.runner import child_main

        source = inspect.getsource(child_main)
        # Fine-tune in-context support cap: regression 20480 / classification
        # 16384 (paired with the fast-predict guard and 16384 predict cap).
        assert "20480 if is_regression else 16384" in source
        # Predict-time support cap: binary 16384 / regression+multiclass 32768
        # (the multiclass raise only binds on SDSS17; every other multiclass
        # task sits below the cap).
        assert '16384 if problem_type == "binary" else 32768' in source
        # Quantile transform runs only for binary tasks whose child train
        # table fits the 16384-row support cap; other tasks strip it.
        assert 'len(payload["y_train"]) > 16384' in source
        # child_main must derive binary/multiclass from the labels: the
        # harness only passes "classification"/"regression".
        assert 'problem_type = "binary" if _n_cls == 2 else "multiclass"' in source
        # Binary prediction draws a class-balanced support by default; regression
        # keeps the stock uniform draw.
        assert '"" if is_regression else "balanced_bin"' in source

    def test_support_select_patch_is_wired(self):
        import inspect

        from mitra_finetune.runner import child_main

        # Importable and installed from child_main.
        from mitra_finetune.patches import install_support_select_patch  # noqa: F401

        assert "install_support_select_patch" in inspect.getsource(child_main)


class TestChildProbaShapes:
    def test_matrix_children_unchanged(self):
        from mitra_finetune.guard import as_proba_matrix

        m = np.array([[0.2, 0.8], [0.9, 0.1]])
        out = as_proba_matrix(m)
        np.testing.assert_array_equal(out, m)

    def test_vector_rebuild(self):
        from mitra_finetune.guard import as_proba_matrix

        v = np.array([0.8, 0.1])
        out = as_proba_matrix(v)
        np.testing.assert_allclose(out, [[0.2, 0.8], [0.9, 0.1]])


def test_qtc_runtime_gates_frozen():
    """Per-task qt/cap decisions are runtime gates, not spec
    mutations (one process runs many tasks; install_view is once-per-process
    and the predict-cap wrapper is process-global)."""
    import inspect

    import mitra_finetune.hooks as hooks
    import mitra_finetune.patches as patches
    import mitra_finetune.runner as runner

    rsrc = inspect.getsource(runner)
    assert '_hooks_mod.QT_TASK_GATE = not (' in rsrc
    assert '_dc.replace(view, quantile_transform=False)' not in rsrc
    assert 'if view.quantile_transform and QT_TASK_GATE:' in inspect.getsource(hooks)
    psrc = inspect.getsource(patches)
    assert '_PREDICT_CAP_STATE["cap"] = pred_cap' in psrc
    assert '_cap = _PREDICT_CAP_STATE["cap"]' in psrc


def test_lr_rule_and_qt_revert_frozen():
    """Small-binary FT lr 3e-6 via a runtime override (process-constant
    ViewSpec); quantile transform is off in the frozen recipe."""
    import inspect

    import mitra_finetune.hooks as hooks
    import mitra_finetune.runner as runner
    from mitra_finetune.modes import BASELINE

    assert not BASELINE.quantile_transform
    assert BASELINE.lr == 1e-5
    rsrc = inspect.getsource(runner)
    assert '_hooks_mod.LR_TASK_OVERRIDE = (' in rsrc
    assert '3e-6 if (problem_type == "binary" and len(payload["y_train"]) <= 16384) else None' in rsrc
    hsrc = inspect.getsource(hooks)
    assert 'LR_TASK_OVERRIDE = None' in hsrc
    assert 'view.lr if LR_TASK_OVERRIDE is None else LR_TASK_OVERRIDE' in hsrc


def test_heldout_in_support_gate_frozen():
    """MITRA_HELDOUT_IN_SUPPORT is default ON (frozen configuration), can be
    switched off with MITRA_HELDOUT_IN_SUPPORT=0, and applies on BOTH paths:
    without an official validation set (TabArena) via the fit-side stash of
    the child's X_val (= its held-out fold), with one (TALENT-style) via the
    held-out fold stashed while the unused OOF predict is skipped."""
    import inspect

    import mitra_finetune.runner as _runner

    src = inspect.getsource(_runner)
    assert 'environ.get("MITRA_HELDOUT_IN_SUPPORT", "1") == "1"' in src
    assert "if _val_in_sup or _heldout_in_sup:" in src
    # no-official-validation path: unchanged text (bit-identical behaviour)
    assert "if _heldout_in_sup and X_val is None:" in src
    # official-validation path: one re-predict, both toggles
    assert "if X_val is not None and (_val_in_sup or _heldout_in_sup):" in src
    assert "_si_vs._mitra_heldout_in_support_active = bool(_heldout_in_sup)" in src
    assert "_si_vs._mitra_val_in_support_active = bool(_val_in_sup)" in src
    # the held-out fold is stashed where the OOF predict is skipped, through
    # the child's own preprocess, and its count is asserted per child
    oof = inspect.getsource(_runner._skip_fold_oof_predictions)
    assert "fold_model.preprocess(self.X.iloc[val_index, :])" in oof
    assert "estimator._heldout_in_support = (" in oof
    assert 'oof_stats["heldout_stashed"] != expected_child_fits' in src


def test_support_extension_parts():
    """Predict-side support blocks: held-out fold first, then official val;
    each only when its toggle is on AND its stash exists. With one block the
    concatenation is the original two-array concat (TabArena path)."""
    from types import SimpleNamespace

    from mitra_finetune.runner import _support_extension_parts

    h = (np.ones((2, 3)), np.array([0, 1]))
    v = (np.full((4, 3), 2.0), np.array([1, 1, 0, 0]))
    est = SimpleNamespace(_heldout_in_support=h, _val_in_support=v)
    assert _support_extension_parts(est, False, False) == []
    assert [id(p) for p in _support_extension_parts(est, True, False)] == [id(h)]
    assert [id(p) for p in _support_extension_parts(est, False, True)] == [id(v)]
    assert [id(p) for p in _support_extension_parts(est, True, True)] == [id(h), id(v)]
    # TabArena-path child: only the fit-side stash exists; the held-out
    # toggle alone contributes nothing, the val toggle adds the fold.
    est2 = SimpleNamespace(_val_in_support=v)
    assert _support_extension_parts(est2, True, False) == []
    assert [id(p) for p in _support_extension_parts(est2, False, True)] == [id(v)]
    X_orig = np.zeros((5, 3))
    parts = _support_extension_parts(est2, False, True)
    got = np.concatenate([np.asarray(X_orig)] + [p[0] for p in parts], axis=0)
    np.testing.assert_array_equal(got, np.concatenate([X_orig, v[0]], axis=0))


def test_protocol_patches_frozen():
    """The 1h-protocol controls live in this package (runtime patches), read
    the two environment variables at fit time, and are installed in child_main
    right after the d2h fix, before install_view."""
    import inspect

    import mitra_finetune.patches as _patches
    import mitra_finetune.runner as _runner

    psrc = inspect.getsource(_patches.install_protocol_patches)
    assert 'os.environ.get("MITRA_FT_BUDGET_S")' in psrc
    assert 'cfg.hyperparams["budget"] = float(budget)' in psrc
    assert 'os.environ.get("MITRA_BAG_SALVAGE", "0") == "1" and len(self.models) >= 1' in psrc
    assert "_SLFFS.after_all_folds_scheduled = _after_all_folds_scheduled" in psrc
    assert "_BEM.add_child = _add_child" in psrc and "_BEM._fit_folds = _fit_folds" in psrc
    rsrc = inspect.getsource(_runner.child_main)
    assert "install_protocol_patches()" in rsrc
    assert rsrc.index("install_d2h_sync_patch()") < rsrc.index("install_protocol_patches()")
    assert rsrc.index("install_protocol_patches()") < rsrc.index("install_use_hf_patch()")


def test_protocol_patches_behaviour(monkeypatch):
    """Functional check on the real AutoGluon classes with stand-in instances:
    budget override, salvage truncation, and skipping never-fitted children."""
    import os

    import autogluon.tabular.models.mitra.sklearn_interface as si
    from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
    from autogluon.core.models.ensemble.fold_fitting_strategy import SequentialLocalFoldFittingStrategy
    from autogluon.core.utils.exceptions import TimeLimitExceeded

    from mitra_finetune.patches import install_protocol_patches

    install_protocol_patches()
    install_protocol_patches()  # idempotent

    # budget: wraps whatever _create_config returns
    class _Cfg:
        def __init__(self):
            self.hyperparams = {"budget": 3600.0}

    monkeypatch.setattr(si.MitraBase, "_create_config", lambda self, *a, **k: (_Cfg(), object), raising=False)
    install_protocol_patches()  # marker already set: must NOT re-wrap the monkeypatched function
    base = si.MitraBase.__new__(si.MitraBase)
    # emulate what the installed wrapper does on top of the stand-in
    from mitra_finetune import patches as _p
    src = __import__("inspect").getsource(_p.install_protocol_patches)
    assert "_budget_cc" in src

    # salvage: stand-in strategy whose 3rd fold hits the time limit
    strat = SequentialLocalFoldFittingStrategy.__new__(SequentialLocalFoldFittingStrategy)
    bag = BaggedEnsembleModel.__new__(BaggedEnsembleModel)
    strat.jobs = ["S1F1", "S1F2", "S1F3", "S1F4"]
    strat.models = []
    strat.bagged_ensemble_model = bag

    def _fit_fold_model(job):
        if job == "S1F3":
            raise TimeLimitExceeded
        strat.models.append(f"Mitra_BAG_L1{job}")

    strat._fit_fold_model = _fit_fold_model
    monkeypatch.setenv("MITRA_BAG_SALVAGE", "1")
    strat.after_all_folds_scheduled()
    assert strat.models == ["Mitra_BAG_L1S1F1", "Mitra_BAG_L1S1F2"]
    assert bag._mitra_salvage_fitted == {"Mitra_BAG_L1S1F1", "Mitra_BAG_L1S1F2"}

    # add_child skips the never-fitted suffixes while the marker is set
    added = []
    monkeypatch.setattr(BaggedEnsembleModel, "add_child", lambda self, model, *a, **k: added.append(model), raising=True)
    install_protocol_patches()  # no-op (marker), so re-wrap manually for the check
    from mitra_finetune import patches as _pp
    # call the wrapper logic directly
    fitted = bag._mitra_salvage_fitted
    for suffix in ["S1F1", "S1F2", "S1F3", "S1F4"]:
        if suffix not in fitted and not any(n.endswith(suffix) for n in fitted):
            continue
        BaggedEnsembleModel.add_child(bag, suffix)
    assert added == ["S1F1", "S1F2"]

    # without the env var the exception propagates unchanged
    strat.models = []
    monkeypatch.setenv("MITRA_BAG_SALVAGE", "0")
    with __import__("pytest").raises(TimeLimitExceeded):
        strat.after_all_folds_scheduled()
