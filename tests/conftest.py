"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_run_index(tmp_path_factory, monkeypatch):
    """Point the cross-process run index at a throwaway file per test, so detached
    runs never read or write the developer's real ~/.agentloop/index.jsonl."""
    index = tmp_path_factory.mktemp("agentloop-index") / "index.jsonl"
    monkeypatch.setenv("AGENTLOOP_INDEX", str(index))
