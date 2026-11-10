"""A'Space adapter for the native mitra-2 pipeline (D-N053, pas A).

Seul code A'Space du repo. Le pipeline amont/natif (``src/mitra_finetune``)
n'est jamais modifie : l'adapter est une fine enveloppe qui epingle le gel,
impose le checkpoint local (aucun telechargement implicite depuis le Hub) et
expose la surface minimale utilisee par le Runtime A'Space.

Regles :
- Aucun import tiers au niveau module (stdlib seule). ``torch``,
  ``autogluon`` et ``mitra_finetune`` ne sont importes qu'a l'interieur des
  methodes qui en ont besoin, jamais a l'import de l'adapter.
- Aucun appel reseau. Le checkpoint doit etre un chemin local existant
  (repertoire ``Tab2D.save_pretrained`` ou ``.pt`` unique) ; un identifiant
  de repo Hub est refuse avant tout import lourd.
- ``device`` par defaut ``"cpu"`` (divergence documentee : l'amont
  ``MitraFinetune`` defaut est ``"cuda"``).
"""

from __future__ import annotations

import os
from pathlib import Path

PINNED_MITRA2_COMMIT = "3f277e5aa1b0c6b36ae44dfe4ad7c3fe1cb999c2"
PINNED_MITRA2_MAIN = "b4701e8148dc33b00ed15d7086ff59816957cde4"
PINNED_MITRA2_BIND = PINNED_MITRA2_COMMIT
FIBER_ID = "fiber:a0:mitra-2"
LICENSE = "Apache-2.0"
ADAPTER_VERSION = "0.1.0"


class Mitra2AspaceAdapter:
    """Surface A'Space minimale au-dessus de ``mitra_finetune.MitraFinetune``."""

    def describe(self) -> dict:
        """Decrit l'adapter epingle (aucun import lourd, aucune E/S)."""
        return {
            "fiber_id": FIBER_ID,
            "adapter": "aspace-mitra-2-adapter",
            "adapter_version": ADAPTER_VERSION,
            "pinned_commit": PINNED_MITRA2_COMMIT,
            "pinned_main": PINNED_MITRA2_MAIN,
            "license": LICENSE,
            "upstream_class": "mitra_finetune.MitraFinetune",
            "checkpoint": "local-only (aucun telechargement Hub implicite)",
            "default_device": "cpu",
            "telemetry": "zero par contrat (voir mitra-2-TELEMETRY-AUDIT.md)",
        }

    def create(self, checkpoint_dir, **kwargs):
        """Construit ``MitraFinetune`` sur un checkpoint LOCAL existant.

        Refuse tout chemin inexistant (dont les identifiants de repo Hub)
        avant d'importer quoi que ce soit de lourd. ``device`` vaut ``"cpu"``
        par defaut ; passer ``device="cuda"`` pour retrouver le defaut amont.
        """
        path = Path(checkpoint_dir)
        if not path.exists():
            raise FileNotFoundError(
                "mitra-2 (A'Space) : checkpoint local introuvable : "
                f"{checkpoint_dir!r} — aucun telechargement depuis le Hub "
                "n'est declenche par l'adapter."
            )
        kwargs.setdefault("device", "cpu")
        from mitra_finetune.api import MitraFinetune

        return MitraFinetune(checkpoint_dir, **kwargs)

    def recipe(self) -> dict:
        """Resume la recette gelee ``BASELINE`` (import paresseux)."""
        from mitra_finetune.modes import BASELINE

        return {
            "name": BASELINE.name,
            "steps": BASELINE.steps,
            "lr": BASELINE.lr,
            "warmup_steps": BASELINE.warmup_steps,
            "tune": BASELINE.tune,
            "snapshot_k": BASELINE.snapshot_k,
            "quantile_transform": BASELINE.quantile_transform,
            "category_frequency": BASELINE.category_frequency,
            "is_stock": BASELINE.is_stock,
        }

    def telemetry_guard(self) -> dict:
        """Garde-fous reseau de l'adapter (constat, pas une promesse)."""
        return {
            "network_calls_by_adapter": 0,
            "hub_download_by_adapter": False,
            "telemetry_sdk": None,
            "env_flags_read": ["MITRA_VAL_IN_SUPPORT"],
        }


__all__ = [
    "ADAPTER_VERSION",
    "FIBER_ID",
    "LICENSE",
    "PINNED_MITRA2_BIND",
    "PINNED_MITRA2_COMMIT",
    "PINNED_MITRA2_MAIN",
    "Mitra2AspaceAdapter",
]
