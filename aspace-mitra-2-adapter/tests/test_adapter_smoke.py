"""Smoke tests de l'adapter A'Space mitra-2 (aucun import lourd)."""

import sys

import pytest

from mitra2_aspace_adapter import (
    FIBER_ID,
    PINNED_MITRA2_COMMIT,
    Mitra2AspaceAdapter,
)


def test_import_does_not_pull_heavy_deps():
    assert "torch" not in sys.modules
    assert "mitra_finetune" not in sys.modules
    assert "autogluon" not in sys.modules


def test_describe_pins_gel():
    info = Mitra2AspaceAdapter().describe()
    assert info["fiber_id"] == FIBER_ID == "fiber:a0:mitra-2"
    assert info["pinned_commit"] == PINNED_MITRA2_COMMIT
    assert info["pinned_commit"] == "3f277e5aa1b0c6b36ae44dfe4ad7c3fe1cb999c2"
    assert info["license"] == "Apache-2.0"
    assert info["checkpoint"].startswith("local-only")


def test_create_refuses_missing_path_before_heavy_import():
    adapter = Mitra2AspaceAdapter()
    with pytest.raises(FileNotFoundError, match="checkpoint local introuvable"):
        adapter.create("/nonexistent/mitra-checkpoint")
    with pytest.raises(FileNotFoundError):
        adapter.create("Amdkn/mitra-classifier-2")
    assert "torch" not in sys.modules
    assert "mitra_finetune" not in sys.modules


def test_telemetry_guard_is_zero():
    guard = Mitra2AspaceAdapter().telemetry_guard()
    assert guard["network_calls_by_adapter"] == 0
    assert guard["hub_download_by_adapter"] is False
    assert guard["telemetry_sdk"] is None
