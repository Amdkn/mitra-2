"""The frozen fine-tuning recipe.

The recipe was selected on an earlier Mitra checkpoint and prospectively
confirmed on a later checkpoint of the same pretraining run (and on a
held-out TabArena fold). Hyperparameters
are constants, not tunables: changing them invalidates the transfer
evidence.

One ``MitraFinetune`` fit is one Mitra fine-tuning run -- a 50-step full
fine-tune of every parameter (the standard recipe) -- capped at
``time_limit`` seconds (default 3,600).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewSpec:
    """The hyperparameters of one fine-tuning run."""

    name: str
    # Fine-tuning schedule
    steps: int | None = 50          # None = wall-clock only (no step cap)
    lr: float = 1e-5
    warmup_steps: int = 10
    weight_decay: float = 0.3
    validation_interval: int = 1
    # Parameter selection
    tune: str = "full"
    # Input view
    quantile_transform: bool = False
    category_frequency: bool = False
    # Prediction-time
    snapshot_k: int = 1             # >1: average the k best-validation
    #                                 checkpoints' probabilities

    @property
    def wall_clock_only(self) -> bool:
        return self.steps is None

    @property
    def is_stock(self) -> bool:
        """True when this view equals stock AutoGluon Mitra fine-tuning.

        Stock is defined by AUTOGLUON's own defaults (sklearn_interface:
        fine_tune_steps=50, LR=1e-4, WARMUP_STEPS=1000, weight_decay=0.1,
        AdamW(0.9, 0.999), full fine-tune, no input transforms, single
        checkpoint) -- NOT this class's defaults. A stock view runs on
        unpatched AutoGluon with no hooks, so AutoGluon's values apply;
        any view that deviates from AutoGluon's defaults (e.g. the current
        baseline recipe, lr=1e-5/warmup=10) MUST be non-stock or its
        schedule would be silently ignored.
        """
        return (
            self.steps == 50
            and self.lr == 1e-4
            and self.warmup_steps == 1000
            and self.weight_decay == 0.1
            and self.tune == "full"
            and not self.quantile_transform
            and not self.category_frequency
            and self.snapshot_k == 1
        )


BASELINE = ViewSpec(name="baseline")
