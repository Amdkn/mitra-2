"""Checkpoint resolution and conversion for AG >= 1.6."""
from __future__ import annotations

from pathlib import Path


def resolve_checkpoint(checkpoint_dir: str | Path, task: str = "CLASSIFICATION") -> str:
    """Resolve to an AG>=1.6 ``hf_model`` directory, converting if needed.

    Accepts (a) a directory already in ``Tab2D.save_pretrained`` format
    (``config.json`` + ``model.safetensors``) -- returned as-is; (b) a raw
    ``.pt`` state dict, or a directory containing exactly one -- converted
    once to ``<checkpoint>.ag16/`` (classification) or
    ``<checkpoint>.ag16.reg/`` (regression) next to the file and cached
    there. ``task`` sets the saved config so AutoGluon loads the head on
    the matching classification/regression code path.
    """
    path = Path(checkpoint_dir)
    if path.is_dir() and (path / "config.json").exists() and (
        path / "model.safetensors"
    ).exists():
        return str(path)
    if path.is_dir():
        candidates = sorted(path.glob("*.pt"))
        if len(candidates) != 1:
            raise ValueError(
                f"{path} must be a Tab2D.save_pretrained directory or "
                f"contain exactly one .pt checkpoint; found "
                f"{len(candidates)} .pt files"
            )
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}")
    return _convert_checkpoint(path, task=task)


def _convert_checkpoint(pt_path: Path, task: str = "CLASSIFICATION") -> str:
    """Convert a raw ``.pt`` state dict to a cached save_pretrained dir.

    For regression the checkpoint is a value-bin cross-entropy head
    (``dim_output`` = number of bins); ``task="REGRESSION"`` makes
    ``Tab2D`` build and later run on the regression forward path.
    """
    suffix = ".ag16.reg" if task == "REGRESSION" else ".ag16"
    converted = pt_path.with_suffix(pt_path.suffix + suffix)
    if (converted / "config.json").exists() and (
        converted / "model.safetensors"
    ).exists():
        return str(converted)

    import re

    import torch

    from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

    state_dict = torch.load(
        pt_path, map_location="cpu", weights_only=True
    )
    state_dict = {
        k.replace("._orig_mod.", "."): v for k, v in state_dict.items()
    }
    n_layers = max(
        int(m.group(1))
        for k in state_dict
        if (m := re.match(r"layers\.(\d+)\.", k))
    ) + 1
    model = Tab2D(
        dim=state_dict["final_layer_norm.weight"].shape[0],
        dim_output=state_dict["final_layer.weight"].shape[0],
        n_layers=n_layers,
        n_heads=4,
        task=task,
        use_pretrained_weights=False,
        path_to_weights="",
        device="cpu",
    )
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(str(converted))
    return str(converted)
