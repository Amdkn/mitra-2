"""Process-local AutoGluon Mitra hooks for the fine-tuning recipe.

AutoGluon's public Mitra API exposes only ``fine_tune``/``fine_tune_steps``.
The shipped recipe needs more control (its own lr/warmup/weight-decay
schedule), so this module patches the internal Mitra modules in-process
before a fit. The baseline recipe (lr=1e-5, warmup=10) deviates from
AutoGluon's stock defaults (lr=1e-4, warmup=1000), so these hooks are
REQUIRED for the shipped recipe to take effect -- see ``ViewSpec.is_stock``.

Because the patches are process-global, EVERY FIT RUNS IN ITS OWN
SUBPROCESS (see runner.py). Within one process, install_view() may be
called at most once.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from . import speed as _speed
from .modes import ViewSpec

WALL_CLOCK_MAX_EPOCHS = 2_147_483_647

# Per-task runtime gate for the quantile transform. install_view() runs once
# per process and sequential tasks in one process must present the SAME
# ViewSpec, so the small-binary-only qt decision cannot live in the spec:
# child_main flips this gate before each fit instead of mutating the view.
QT_TASK_GATE = True

# Per-task runtime override for the fine-tuning learning rate (same reason:
# the ViewSpec is process-constant, so a task-conditioned rate cannot live in
# the spec). None = use view.lr. child_main sets it before each fit.
LR_TASK_OVERRIDE = None

def configure_trainable(model: nn.Module, view: ViewSpec) -> int:
    if view.tune != "full":
        raise ValueError(f"Unknown tune mode: {view.tune}")
    model.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def install_view(view: ViewSpec) -> None:
    """Install the hooks for one ViewSpec. Once per process."""

    import autogluon.tabular.models.mitra._internal.core.trainer_finetune as trainer_module
    import autogluon.tabular.models.mitra.sklearn_interface as sklearn_interface

    installed = getattr(sklearn_interface, "_mitra_finetune_view_installed", None)
    if installed is not None:
        # One long-lived process fits its tasks sequentially with the SAME
        # view (reference harness shape): reinstalling an identical view is
        # a no-op. Two DIFFERENT views cannot share a process -- the patches
        # are module-global.
        if installed == view:
            return
        raise RuntimeError(
            f"View {installed.name!r} is already installed in this process; "
            f"cannot install {view.name!r}. Distinct views need separate "
            "processes."
        )

    original_create_config = sklearn_interface.MitraBase._create_config
    original_trainer = sklearn_interface.TrainerFinetune

    def create_config(self, *args, **kwargs):
        cfg, model_cls = original_create_config(self, *args, **kwargs)
        cfg.hyperparams["lr"] = view.lr if LR_TASK_OVERRIDE is None else LR_TASK_OVERRIDE
        cfg.hyperparams["warmup_steps"] = view.warmup_steps
        cfg.hyperparams["weight_decay"] = view.weight_decay
        if view.wall_clock_only:
            cfg.hyperparams["max_epochs"] = WALL_CLOCK_MAX_EPOCHS
            cfg.hyperparams["early_stopping_patience"] = WALL_CLOCK_MAX_EPOCHS
        else:
            cfg.hyperparams["max_epochs"] = view.steps
        if view.quantile_transform and QT_TASK_GATE:
            cfg.hyperparams["use_quantile_transformer"] = True
        return cfg, model_cls

    def get_optimizer(hyperparams, model):
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("View selected no trainable parameters")
        return torch.optim.AdamW(
            parameters,
            lr=hyperparams["lr"],
            betas=(0.9, 0.999),
            weight_decay=hyperparams["weight_decay"],
        )

    class ViewTrainer(original_trainer):
        def __init__(self, cfg, model, *args, **kwargs):
            trainable = configure_trainable(model, view)
            super().__init__(cfg, model, *args, **kwargs)
            total = sum(p.numel() for p in model.parameters())
            print(
                f"[mitra-finetune] view={view.name} trainable={trainable:,}"
                f"/{total:,}",
                flush=True,
            )
            self._view_inside_train = False
            self._view_epoch = 0
            self._view_last_metrics = None
            # Cheaper loop (speed.py): GPU-resident best-weights checkpoint,
            # cached validation transforms in one wide chunk, memory
            # preflight, SDPA attention. Same computation per step.
            self._loop = _speed.LoopSettings.from_env()
            if self._loop.fast_loop:
                self.checkpoint = _speed.DeviceCheckpoint()
                self.checkpoint.reset(self.model)

        def train(self, x_train, y_train, x_val, y_val):
            self._view_inside_train = True
            started = time.monotonic()
            restore_attention = None
            if self._loop.fast_loop:
                restore_attention = _speed.set_attention_backend(self.model, self._loop.attention_backend)
                _speed.clear_eval_cache(self)
            try:
                if (
                    self._loop.fast_loop
                    and self._loop.memory_preflight
                    and str(self.device).startswith("cuda")
                ):
                    _speed.memory_preflight(self, x_train)
                return super().train(x_train, y_train, x_val, y_val)
            finally:
                if restore_attention is not None:
                    restore_attention()
                    _speed.clear_eval_cache(self)
                self._view_inside_train = False
                print(
                    f"[mitra-finetune] view={view.name} epochs={self._view_epoch} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        def evaluate(self, *args, **kwargs):
            if not self._view_inside_train:
                return super().evaluate(*args, **kwargs)
            if self._view_last_metrics is None:
                metrics = self._evaluate_in_loop(*args, **kwargs)
                self._view_last_metrics = metrics
                return metrics
            self._view_epoch += 1
            if self._view_epoch % view.validation_interval != 0:
                return self._view_last_metrics
            metrics = self._evaluate_in_loop(*args, **kwargs)
            self._view_last_metrics = metrics
            return metrics

        def _evaluate_in_loop(self, x_support, y_support, x_query, y_query):
            # The validation pass of one fine-tuning step: stock evaluate, or
            # the cached-transform / wide-chunk version of speed.py.
            if not self._loop.fast_loop:
                return super().evaluate(x_support, y_support, x_query, y_query)
            return _speed.evaluate_in_loop(
                self, x_support, y_support, x_query, y_query, self._loop.eval_query_chunk
            )

    # AutoGluon pickles the fitted trainer when saving bagged child models,
    # so the class must be resolvable as mitra_finetune.hooks.ViewTrainer:
    # fix __qualname__ (still "install_view.<locals>.ViewTrainer") and bind
    # it at module level. install_view runs once per process, so the global
    # is unambiguous.
    ViewTrainer.__name__ = "ViewTrainer"
    ViewTrainer.__qualname__ = "ViewTrainer"
    ViewTrainer.__module__ = __name__
    globals()["ViewTrainer"] = ViewTrainer
    sklearn_interface.TrainerFinetune = ViewTrainer
    trainer_module.TrainerFinetune = ViewTrainer
    trainer_module.get_optimizer = get_optimizer
    sklearn_interface.MitraBase._create_config = create_config
    sklearn_interface._mitra_finetune_view_installed = view
