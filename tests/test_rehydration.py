"""Run-registry rehydration: after the server process restarts (its in-memory
registry is gone), poll tools still resolve a run_id from the on-disk index +
run dir, instead of reporting `unknown run_id` / an empty list."""

from __future__ import annotations

import os

import agentloop.runs as runs
from agentloop.mcp_server import (
    orchestrate_list_impl,
    orchestrate_result_impl,
    orchestrate_start_impl,
    orchestrate_status_impl,
    orchestrate_tail_impl,
)


def _finished_run(tmp_path):
    started = orchestrate_start_impl("goal", "criteria", backend="mock",
                                     base_dir=str(tmp_path), max_iterations=2)
    orchestrate_result_impl(started["run_id"], wait=True, timeout=15)
    return started


def _simulate_restart():
    """Drop the in-memory registry, leaving only the on-disk record + index."""
    with runs.MANAGER._lock:
        runs.MANAGER._runs.clear()


def test_status_rehydrates_after_registry_loss(tmp_path):
    started = _finished_run(tmp_path)
    _simulate_restart()

    status = orchestrate_status_impl(started["run_id"])
    assert "error" not in status
    assert status["run_id"] == started["run_id"]
    assert status["running"] is False
    assert status["completed"] is True
    assert status["rehydrated"] is True


def test_tail_rehydrates_after_registry_loss(tmp_path):
    started = _finished_run(tmp_path)
    _simulate_restart()

    tail = orchestrate_tail_impl(started["run_id"], cursor=0)
    assert "error" not in tail
    kinds = [e["kind"] for e in tail["events"]]
    assert "run_started" in kinds and "run_finished" in kinds
    assert tail["running"] is False
    # Cursor paging still works off disk.
    page = orchestrate_tail_impl(started["run_id"], cursor=0, limit=2)
    assert len(page["events"]) == 2 and page["more"] is True
    assert all(e["seq"] > 2 for e in orchestrate_tail_impl(
        started["run_id"], cursor=page["cursor"])["events"])


def test_result_rehydrates_completed_run(tmp_path):
    started = _finished_run(tmp_path)
    _simulate_restart()

    result = orchestrate_result_impl(started["run_id"])
    assert result["completed"] is True
    assert result["stop_reason"] == "goal_complete"


def test_list_includes_rehydrated_runs(tmp_path):
    started = _finished_run(tmp_path)
    _simulate_restart()

    listing = orchestrate_list_impl()
    assert any(r["run_id"] == started["run_id"] for r in listing["runs"])


def test_interrupted_run_reported_as_terminal(tmp_path):
    # A run that was mid-flight when the process died: status.json says running,
    # no result.json, and an index entry points at it. Its loop thread is gone,
    # so rehydration must report it as terminal (interrupted), not live.
    run_id = "20260101-000000-abc123"
    run_dir = os.path.join(str(tmp_path), run_id)
    os.makedirs(run_dir)
    runs._write_json(os.path.join(run_dir, "status.json"), {
        "run_id": run_id, "run_dir": run_dir, "running": True,
        "phase": "decompose", "iteration": 1, "events": 4,
    })
    runs._index_record(run_id, run_dir)

    status = orchestrate_status_impl(run_id)
    assert status["running"] is False
    assert status["stop_reason"] == "interrupted"
    assert status["rehydrated"] is True

    result = orchestrate_result_impl(run_id)
    assert result["running"] is False
    assert result["stop_reason"] == "interrupted"


def test_unknown_run_id_still_unknown_when_not_indexed(tmp_path):
    _simulate_restart()
    assert "error" in orchestrate_status_impl("never-started")
    assert "error" in orchestrate_tail_impl("never-started")
    assert "error" in orchestrate_result_impl("never-started")
