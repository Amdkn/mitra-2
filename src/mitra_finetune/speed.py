"""Cheaper fine-tuning loop: the same steps, computed for less.

Everything here is consumed by ``hooks.ViewTrainer`` and changes how much a
fine-tuning step costs, never what it computes. On the largest TabArena
tables a step costs about half of what it did, so about twice as many of the
50 steps fit a 250 s budget (RTX PRO 6000: Diabetes130US 11 to 26 steps,
APSFailure 15 to 30, kddcup09_appetency 15 to 26); tables that already
reached 50 steps simply finish sooner. Four changes:

1. The validation pass after every step predicts the validation set in one
   wide query chunk (``FINETUNE_EVAL_QUERY_CHUNK`` rows) instead of stock's
   1,024-row chunks with a fresh support draw each, which made that pass cost
   as much as several steps on large tables. Under CUDA out-of-memory the
   chunk is halved back to the stock size. The transformed arrays it scores
   are computed once per fit instead of once per step (the preprocessor's
   transforms are fixed when it is fit).
2. Before the first validation pass, one throw-away forward and backward
   pass at the fine-tuning context size (:func:`memory_preflight`) makes a
   context that does not fit the GPU fail in seconds instead of after a full
   validation pass at that size; AutoGluon's out-of-memory ratchet then
   halves the context as before.
3. The loop's attention runs on ``torch.nn.functional.
   scaled_dot_product_attention`` (:func:`set_attention_backend`), which
   matches flash-attn 2 to bf16 rounding, is as fast per step on an H100 and
   1.3 to 1.7 times faster on an RTX PRO 6000 Blackwell. Prediction keeps
   the construction-time kernel: at prediction shapes the flash path is
   faster and far lighter on memory.
4. The best-weights checkpoint stays on the GPU (:class:`DeviceCheckpoint`)
   instead of being copied to the host at every improving step, and the
   validation forward runs under ``inference_mode``.

All of it is on by default and read from the environment at fit time:
``MITRA_FT_FAST_LOOP=0`` restores the stock loop wholesale,
``MITRA_FT_ATTENTION=stock`` keeps the construction-time attention kernel in
the loop, ``MITRA_FT_EVAL_CHUNK=<rows>`` sets the validation chunk, and
``MITRA_FT_PREFLIGHT=0`` skips the memory preflight.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np

FINETUNE_EVAL_QUERY_CHUNK = 16384
FINETUNE_ATTENTION_BACKEND = "sdpa"


@dataclass(frozen=True)
class LoopSettings:
    """The fine-tuning loop's speed settings, resolved from the environment at fit time."""

    fast_loop: bool = True
    eval_query_chunk: int = FINETUNE_EVAL_QUERY_CHUNK
    attention_backend: str = FINETUNE_ATTENTION_BACKEND
    memory_preflight: bool = True

    @classmethod
    def from_env(cls) -> "LoopSettings":
        fast_loop = os.environ.get("MITRA_FT_FAST_LOOP", "1") == "1"
        backend = os.environ.get("MITRA_FT_ATTENTION", FINETUNE_ATTENTION_BACKEND).strip().lower()
        if backend not in ("sdpa", "stock"):
            raise ValueError(f"MITRA_FT_ATTENTION must be 'sdpa' or 'stock', got {backend!r}")
        return cls(
            fast_loop=fast_loop,
            eval_query_chunk=max(1, int(os.environ.get("MITRA_FT_EVAL_CHUNK", str(FINETUNE_EVAL_QUERY_CHUNK)))),
            attention_backend=backend,
            memory_preflight=os.environ.get("MITRA_FT_PREFLIGHT", "1") == "1",
        )


def is_cuda_oom(exc: BaseException) -> bool:
    """Whether an exception is a CUDA out-of-memory error, as the torch class or in message form."""
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def set_attention_backend(model, backend: str):
    """Select the attention kernel of a loaded Tab2D backbone; returns a callable that restores the previous one.

    Tab2D picks ``flash_attn_varlen_func`` at construction whenever ``flash-attn`` imports and the
    device is CUDA, and otherwise runs its layers on ``torch.nn.functional.scaled_dot_product_attention``.
    The choice is an instance attribute of the model and of each layer. ``"sdpa"`` forces the latter,
    ``"stock"`` leaves the construction-time choice.
    """
    if backend not in ("sdpa", "stock"):
        raise ValueError(f"attention backend must be 'sdpa' or 'stock', got {backend!r}")
    modules = [m for m in [model, *getattr(model, "layers", [])] if hasattr(m, "use_flash_attn")]
    previous = [(m, m.use_flash_attn) for m in modules]
    if backend == "sdpa":
        for m in modules:
            m.use_flash_attn = False

    def restore() -> None:
        for m, flag in previous:
            m.use_flash_attn = flag

    return restore


class DeviceCheckpoint:
    """Best-weights checkpoint kept on the model's device.

    Drop-in for AutoGluon's ``Checkpoint`` (``reset``, ``__call__``, ``set_to_best``). The stock one
    copies every tensor of the state dict to the CPU on each step that improves the validation loss,
    a synchronous host copy of about 300 MB for Mitra's 77M parameters. This one clones the state
    once on the device and copies into it in place. The same weights are restored at the end; the
    copies just never leave the GPU. ``TrainerFinetune.post_fit_optimize`` drops the checkpoint
    before a child is saved, as it does the stock one.
    """

    def __init__(self) -> None:
        self.curr_best_loss = np.inf
        self.best_model: dict = {}

    def reset(self, net) -> None:
        self.curr_best_loss = np.inf
        self.best_model = {key: value.detach().clone() for key, value in net.state_dict().items()}

    def __call__(self, net, loss: float) -> None:
        import torch

        if loss < self.curr_best_loss:
            self.curr_best_loss = loss
            with torch.no_grad():
                for key, value in net.state_dict().items():
                    self.best_model[key].copy_(value)

    def set_to_best(self, net) -> None:
        net.load_state_dict(self.best_model)


def _regression_over_bins(trainer) -> bool:
    from autogluon.tabular.models.mitra._internal.config.enums import LossName, Task

    return trainer.cfg.task == Task.REGRESSION and trainer.cfg.hyperparams["regression_loss"] == LossName.CROSS_ENTROPY


def _forward(trainer, batch: dict):
    """One forward pass over a collated batch under autocast; returns raw model outputs."""
    import torch
    from autogluon.tabular.models.mitra._internal.config.enums import ModelName

    hp = trainer.cfg.hyperparams
    with torch.autocast(device_type=trainer.device, dtype=getattr(torch, hp["precision"])):
        x_s = batch["x_support"].to(trainer.device, non_blocking=True)
        y_s = batch["y_support"].to(trainer.device, non_blocking=True)
        x_q = batch["x_query"].to(trainer.device, non_blocking=True)
        padding_features = batch["padding_features"].to(trainer.device, non_blocking=True)
        padding_obs_support = batch["padding_obs_support"].to(trainer.device, non_blocking=True)
        padding_obs_query = batch["padding_obs_query"].to(trainer.device, non_blocking=True)
        if _regression_over_bins(trainer):
            y_s = torch.bucketize(y_s, trainer.bins) - 1
            y_s = torch.clamp(y_s, 0, hp["dim_output"] - 1).to(torch.int64)
        if trainer.cfg.model_name == ModelName.TABPFN:
            return trainer.model(x_s, y_s, x_q, task=trainer.cfg.task).squeeze(-1)
        return trainer.model(x_s, y_s, x_q, padding_features, padding_obs_support, padding_obs_query)


def _mean_decode(trainer, logits):
    """Softmax-weighted mean over bin centers: the same decode ``patches._mean_decode_source`` installs."""
    import torch

    logits = logits.float()
    if not torch.isfinite(logits).all():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    centers = (trainer.bins[:-1] + trainer.bin_width / 2).to(logits.device)
    return (torch.softmax(logits, dim=-1) * centers).sum(dim=-1)


def memory_preflight(trainer, x_train) -> None:
    """One throw-away forward and backward pass at the fine-tuning context size.

    Stock fine-tuning validates the pretrained model on the whole validation set before its first
    step, so when the context does not fit the GPU the out-of-memory error arrives only after that
    full pass, about a minute per attempt on large tables, and AutoGluon's ratchet pays it at every
    context size it tries. This pass reproduces the shapes of a training step on synthetic data: the
    loop's 80/20 support and query split, capped as the loop caps them, over the training table's
    non-constant columns (the preprocessor drops constant ones). A context that cannot fit fails here
    in seconds. The model is left as it was: no optimizer step, gradients cleared, and neither the
    trainer's RNG nor the global RNGs are drawn from, so a fit that goes on is the same fit as without
    the pass.
    """
    import torch
    from autogluon.tabular.models.mitra._internal.config.enums import Task
    from autogluon.tabular.models.mitra._internal.data.dataset_finetune import DatasetFinetune

    hp = trainer.cfg.hyperparams
    x = np.asarray(x_train, dtype=np.float32)
    n_rows = x.shape[0]
    if n_rows < 2 or x.shape[1] == 0:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        n_features = int((np.nanmax(x, axis=0) != np.nanmin(x, axis=0)).sum())
    if n_features == 0:
        return
    n_support_pool = int(np.ceil(0.8 * n_rows))
    n_support = min(int(hp["max_samples_support"]), n_support_pool)
    n_query = max(1, min(int(hp["max_samples_query"]), n_rows - n_support_pool))
    rng = np.random.RandomState(0)
    x_support = rng.standard_normal((n_support, n_features)).astype(np.float32)
    x_query = rng.standard_normal((n_query, n_features)).astype(np.float32)
    if trainer.cfg.task == Task.CLASSIFICATION:
        y_support = np.zeros(n_support, dtype=np.int64)
    else:
        y_support = np.zeros(n_support, dtype=np.float32)
    dataset = DatasetFinetune(
        trainer.cfg,
        x_support=x_support,
        y_support=y_support,
        x_query=x_query,
        y_query=None,
        max_samples_support=n_support,
        max_samples_query=n_query,
        rng=rng,
    )
    batch = next(iter(trainer.make_loader(dataset, training=False)))
    was_training = trainer.model.training
    trainer.model.train()
    outputs = None
    try:
        with torch.enable_grad():
            outputs = _forward(trainer, batch)
            outputs.float().mean().backward()
    finally:
        trainer.model.train(was_training)
        outputs = None
        batch = None
        trainer.optimizer.zero_grad(set_to_none=True)
        for parameter in trainer.model.parameters():
            parameter.grad = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def clear_eval_cache(trainer) -> None:
    """Forget the transformed validation arrays cached by :func:`evaluate_in_loop`."""
    trainer._mf_eval_arrays = None


def evaluate_in_loop(trainer, x_support, y_support, x_query, y_query, query_chunk: int):
    """The validation pass of one fine-tuning step: cached transforms, one wide query chunk.

    Stock ``TrainerFinetune.train`` calls ``evaluate`` with the same training and validation arrays
    before the first step and after every step, and stock ``evaluate`` preprocesses them every time.
    The preprocessor's transforms are fixed when it is fit, so the arrays are transformed once per
    fit (cached on array identity; :func:`clear_eval_cache` drops them when ``train`` ends) and
    scored in one query chunk of up to ``query_chunk`` rows, halved back to the stock chunk under
    CUDA out-of-memory.
    """
    import torch

    cached = getattr(trainer, "_mf_eval_arrays", None)
    if cached is None or cached[0] is not x_support or cached[1] is not x_query:
        preprocessor = trainer.preprocessor
        arrays = (
            preprocessor.transform_X(x_support),
            preprocessor.transform_y(y_support),
            preprocessor.transform_X(x_query),
            np.asarray(y_query),
        )
        cached = trainer._mf_eval_arrays = (x_support, x_query, arrays)
    stock_chunk = int(trainer.cfg.hyperparams["max_samples_query"])
    chunk = max(stock_chunk, min(int(query_chunk), len(x_query)))
    while True:
        try:
            return _evaluate_once(trainer, *cached[2], query_chunk=chunk)
        except RuntimeError as exc:
            if not is_cuda_oom(exc) or chunk <= stock_chunk:
                raise
            torch.cuda.empty_cache()
            chunk = max(stock_chunk, chunk // 2)
            print(
                f"[mitra-finetune] fine-tuning validation: CUDA out of memory, halving the query chunk to {chunk}",
                flush=True,
            )


def _evaluate_once(trainer, x_support, y_support, x_query, y_query, *, query_chunk: int):
    import torch
    from autogluon.tabular.models.mitra._internal.core.prediction_metrics import PredictionMetricsTracker
    from autogluon.tabular.models.mitra._internal.data.dataset_finetune import DatasetFinetune

    trainer.model.eval()
    dataset = DatasetFinetune(
        trainer.cfg,
        x_support=x_support,
        y_support=y_support,
        x_query=x_query,
        y_query=y_query,
        max_samples_support=trainer.cfg.hyperparams["max_samples_support"],
        max_samples_query=query_chunk,
        rng=trainer.rng,
    )
    loader = trainer.make_loader(dataset, training=False)
    tracker = PredictionMetricsTracker(task=trainer.cfg.task, preprocessor=trainer.preprocessor)
    decode_bins = _regression_over_bins(trainer)
    with torch.inference_mode():
        for batch in loader:
            y_hat = _forward(trainer, batch)
            y_q = batch["y_query"].to(trainer.device, non_blocking=True)
            if decode_bins:
                y_hat = _mean_decode(trainer, y_hat)
            tracker.update(y_hat.float(), y_q, train=False)
    return tracker.get_metrics()
