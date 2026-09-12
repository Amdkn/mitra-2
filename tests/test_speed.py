"""The cheaper fine-tuning loop and the two-regime prediction chunking (v0.3)."""
import inspect

import numpy as np
import pytest
import torch

from mitra_finetune import speed
from mitra_finetune.patches import fast_predict_plan, fitted_context_key


class TestFrozenSpeedDefaults:
    def test_loop_constants(self):
        assert speed.FINETUNE_EVAL_QUERY_CHUNK == 16384
        assert speed.FINETUNE_ATTENTION_BACKEND == "sdpa"

    def test_settings_from_env(self, monkeypatch):
        for var in ("MITRA_FT_FAST_LOOP", "MITRA_FT_ATTENTION", "MITRA_FT_EVAL_CHUNK", "MITRA_FT_PREFLIGHT"):
            monkeypatch.delenv(var, raising=False)
        settings = speed.LoopSettings.from_env()
        assert settings == speed.LoopSettings()
        assert settings.fast_loop and settings.memory_preflight
        monkeypatch.setenv("MITRA_FT_FAST_LOOP", "0")
        monkeypatch.setenv("MITRA_FT_ATTENTION", "stock")
        monkeypatch.setenv("MITRA_FT_EVAL_CHUNK", "4096")
        monkeypatch.setenv("MITRA_FT_PREFLIGHT", "0")
        settings = speed.LoopSettings.from_env()
        assert not settings.fast_loop and not settings.memory_preflight
        assert settings.attention_backend == "stock" and settings.eval_query_chunk == 4096
        monkeypatch.setenv("MITRA_FT_ATTENTION", "flash")
        with pytest.raises(ValueError):
            speed.LoopSettings.from_env()

    def test_view_trainer_uses_the_speed_module(self):
        from mitra_finetune import hooks

        source = inspect.getsource(hooks.install_view)
        assert "_speed.DeviceCheckpoint()" in source
        assert "_speed.set_attention_backend(self.model, self._loop.attention_backend)" in source
        assert "_speed.memory_preflight(self, x_train)" in source
        assert "_speed.evaluate_in_loop(" in source
        # Outside the fine-tuning loop evaluate stays stock.
        assert "return super().evaluate(*args, **kwargs)" in source

    def test_memo_patch_is_installed_last(self):
        from mitra_finetune.runner import child_main

        source = inspect.getsource(child_main)
        assert 'os.environ.get("MITRA_FITTED_CONTEXT_MEMO", "1") == "1"' in source
        assert source.index("install_view(view)") < source.index("install_fitted_context_memo_patch()")
        assert source.index("install_predict_support_cap_patch(") < source.index(
            "install_fitted_context_memo_patch()"
        )


class TestDeviceCheckpoint:
    def test_tracks_best_and_restores(self):
        torch.manual_seed(0)
        net = torch.nn.Linear(3, 2)
        checkpoint = speed.DeviceCheckpoint()
        checkpoint.reset(net)
        best = {k: v.clone() for k, v in net.state_dict().items()}
        # The copies are independent of the live parameters.
        assert all(
            checkpoint.best_model[k].data_ptr() != v.data_ptr() for k, v in net.state_dict().items()
        )
        checkpoint(net, 1.0)  # first loss: recorded
        with torch.no_grad():
            net.weight.add_(1.0)
        checkpoint(net, 2.0)  # worse: ignored
        assert torch.equal(checkpoint.best_model["weight"], best["weight"])
        with torch.no_grad():
            net.weight.add_(1.0)
        checkpoint(net, 0.5)  # better: recorded in place
        assert torch.equal(checkpoint.best_model["weight"], best["weight"] + 2.0)
        with torch.no_grad():
            net.weight.zero_()
        checkpoint.set_to_best(net)
        assert torch.equal(net.weight, best["weight"] + 2.0)
        assert checkpoint.curr_best_loss == 0.5

    def test_reset_clears_the_loss(self):
        net = torch.nn.Linear(2, 2)
        checkpoint = speed.DeviceCheckpoint()
        checkpoint.reset(net)
        checkpoint(net, 0.1)
        checkpoint.reset(net)
        assert checkpoint.curr_best_loss == np.inf


class _Layer:
    def __init__(self, flag):
        self.use_flash_attn = flag


class _Backbone:
    def __init__(self, flag=True, n_layers=3):
        self.use_flash_attn = flag
        self.layers = [_Layer(flag) for _ in range(n_layers)]


class TestAttentionBackend:
    def test_sdpa_switches_every_layer_and_restore_undoes_it(self):
        model = _Backbone(True)
        restore = speed.set_attention_backend(model, "sdpa")
        assert not model.use_flash_attn and not any(layer.use_flash_attn for layer in model.layers)
        restore()
        assert model.use_flash_attn and all(layer.use_flash_attn for layer in model.layers)

    def test_stock_leaves_the_construction_choice(self):
        model = _Backbone(False)
        restore = speed.set_attention_backend(model, "stock")
        assert not model.use_flash_attn
        restore()
        assert not model.use_flash_attn

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            speed.set_attention_backend(_Backbone(), "flash")


def test_is_cuda_oom():
    assert speed.is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert speed.is_cuda_oom(torch.cuda.OutOfMemoryError())
    assert not speed.is_cuda_oom(RuntimeError("shape mismatch"))
    assert not speed.is_cuda_oom(ValueError("out of memory"))


class TestFastPredictPlan:
    def test_support_fits_freezes_the_draw_and_floors_at_the_floor(self):
        fixed, chunk, floor = fast_predict_plan(5000, 16384, 16384, 256, 1024)
        assert fixed and chunk == 16384 and floor == 256

    def test_capped_support_redraws_per_chunk_and_floors_at_stock(self):
        fixed, chunk, floor = fast_predict_plan(44332, 16384, 16384, 256, 1024)
        assert not fixed and chunk == 16384 and floor == 1024

    def test_no_cap_counts_as_fitting(self):
        fixed, chunk, floor = fast_predict_plan(44332, 0, 16384, 256, 1024)
        assert fixed and floor == 256

    def test_chunk_never_below_stock_and_floor_never_above_chunk(self):
        fixed, chunk, floor = fast_predict_plan(100, 16384, 512, 256, 1024)
        assert fixed and chunk == 1024 and floor == 256
        fixed, chunk, floor = fast_predict_plan(100, 16384, 1024, 4096, 1024)
        assert floor == 1024

    def test_fast_predict_uses_the_plan(self):
        from mitra_finetune.patches import install_fast_predict_patch

        source = inspect.getsource(install_fast_predict_patch)
        assert "fast_predict_plan(" in source
        assert '_state["active"] = _fixed' in source


def test_fitted_context_key_buckets_rows():
    key = fitted_context_key("CLASSIFICATION", 44332, 170)
    assert key == ("CLASSIFICATION", 170, 173)
    # The eight children of a bag fit the same number of rows give or take one
    # (stratified 7/8 folds) and share a key; a different table does not.
    assert fitted_context_key("CLASSIFICATION", 44332 + 1, 170) == key
    assert fitted_context_key("REGRESSION", 44332, 170) != key
    assert fitted_context_key("CLASSIFICATION", 44332, 171) != key


def test_fitted_context_memo_behaviour(monkeypatch):
    """Stand-ins for AutoGluon's ``_create_config`` / ``_train_ensemble``: the
    second fit of the same table shape starts at the context the first one
    settled on; a different shape starts from the requested caps."""
    import autogluon.tabular.models.mitra.sklearn_interface as si

    from mitra_finetune import patches

    class _Cfg:
        def __init__(self):
            self.hyperparams = {"max_samples_support": 16384, "max_samples_query": 1024}

    class _Trainer:
        def __init__(self, cfg):
            self.cfg = cfg

    starts = []

    def _create_config(self, task, dim_output, time_limit=None):
        return _Cfg(), object

    def _train_ensemble(self, X_train, y_train, X_valid, y_valid, task, dim_output, n_classes=0, time_limit=None):
        cfg, _ = self._create_config(task, dim_output, time_limit)
        starts.append(dict(cfg.hyperparams))
        # Emulate the ratchet: the first fit of a big table needs two halvings.
        while cfg.hyperparams["max_samples_support"] > 4096:
            cfg.hyperparams["max_samples_support"] //= 2
        self.trainers = [_Trainer(cfg)]
        return self

    monkeypatch.setattr(si.MitraBase, "_create_config", _create_config, raising=True)
    monkeypatch.setattr(si.MitraBase, "_train_ensemble", _train_ensemble, raising=True)
    monkeypatch.setattr(si.MitraBase, "_fitted_context_memo_patched", False, raising=False)
    monkeypatch.setattr(patches, "_FITTED_CONTEXT_MEMO", {})
    patches.install_fitted_context_memo_patch()
    patches.install_fitted_context_memo_patch()  # idempotent

    base = si.MitraBase.__new__(si.MitraBase)
    X = np.zeros((44332, 170), dtype=np.float32)
    base._train_ensemble(X, None, None, None, "CLASSIFICATION", 10)
    assert starts[-1] == {"max_samples_support": 16384, "max_samples_query": 1024}
    assert patches._FITTED_CONTEXT_MEMO == {("CLASSIFICATION", 170, 173, 16384, 1024): (4096, 1024)}

    # Second child of the same bag (7/8 of the rows land in the same bucket): starts at 4096.
    base._train_ensemble(X[:44300], None, None, None, "CLASSIFICATION", 10)
    assert starts[-1] == {"max_samples_support": 4096, "max_samples_query": 1024}

    # Another table shape starts from the requested caps again.
    base._train_ensemble(np.zeros((2000, 20), dtype=np.float32), None, None, None, "CLASSIFICATION", 10)
    assert starts[-1] == {"max_samples_support": 16384, "max_samples_query": 1024}
    assert patches._FITTED_CONTEXT_PENDING["key"] is None


def test_evaluate_in_loop_caches_transforms_on_array_identity():
    """The transformed arrays are computed once per (x_support, x_query) identity
    and forgotten by clear_eval_cache; the scoring call receives the cached arrays."""
    calls = {"transform_X": 0, "transform_y": 0, "evaluate": []}

    class _Preprocessor:
        def transform_X(self, x):
            calls["transform_X"] += 1
            return x * 2

        def transform_y(self, y):
            calls["transform_y"] += 1
            return y + 1

    class _Cfg:
        hyperparams = {"max_samples_query": 1024}

    trainer = type("T", (), {})()
    trainer.preprocessor = _Preprocessor()
    trainer.cfg = _Cfg()

    def _fake_evaluate_once(tr, xs, ys, xq, yq, *, query_chunk):
        calls["evaluate"].append((xs, ys, xq, yq, query_chunk))
        return "metrics"

    original = speed._evaluate_once
    speed._evaluate_once = _fake_evaluate_once
    try:
        x_support, y_support = np.ones((10, 2)), np.zeros(10)
        x_query, y_query = np.ones((5000, 2)), np.zeros(5000)
        for _ in range(3):
            assert speed.evaluate_in_loop(trainer, x_support, y_support, x_query, y_query, 16384) == "metrics"
        assert calls["transform_X"] == 2 and calls["transform_y"] == 1
        # One wide chunk covering the whole validation set, never below the stock chunk.
        assert calls["evaluate"][-1][4] == 5000
        assert speed.evaluate_in_loop(trainer, x_support, y_support, x_query[:100], y_query[:100], 16384)
        assert calls["evaluate"][-1][4] == 1024
        speed.clear_eval_cache(trainer)
        speed.evaluate_in_loop(trainer, x_support, y_support, x_query, y_query, 16384)
        assert calls["transform_X"] == 6
    finally:
        speed._evaluate_once = original
