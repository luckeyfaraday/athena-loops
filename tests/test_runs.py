"""Detached runs: a run can be watched live (status/tail) and its result fetched
once done, with a durable events.jsonl + per-worker output files on disk."""

from __future__ import annotations

import json
import os

from agentloop.mcp_server import (
    orchestrate_list_impl,
    orchestrate_resume_impl,
    orchestrate_result_impl,
    orchestrate_start_impl,
    orchestrate_status_impl,
    orchestrate_tail_impl,
)


def _start(tmp_path, **kw):
    return orchestrate_start_impl("goal", "criteria", backend="mock",
                                  base_dir=str(tmp_path), **kw)


def test_detached_run_streams_events_and_returns_result(tmp_path):
    started = _start(tmp_path, max_iterations=3)
    assert started["status"] == "running"
    run_id = started["run_id"]

    result = orchestrate_result_impl(run_id, wait=True, timeout=15)
    assert result["completed"] is True
    assert result["stop_reason"] == "goal_complete"

    # The full event stream is available from the start (cursor 0).
    tail = orchestrate_tail_impl(run_id, cursor=0)
    kinds = [e["kind"] for e in tail["events"]]
    for expected in ("run_started", "decomposed", "subagent_started",
                     "subagent_finished", "review", "iteration_finished",
                     "run_finished"):
        assert expected in kinds


def test_events_are_persisted_to_disk_for_tailing(tmp_path):
    started = _start(tmp_path, max_iterations=2)
    orchestrate_result_impl(started["run_id"], wait=True, timeout=15)

    events_path = started["events_path"]
    assert os.path.exists(events_path)
    with open(events_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert any(e["kind"] == "run_started" for e in lines)
    assert any(e["kind"] == "run_finished" for e in lines)
    # result.json is written alongside the event log.
    assert os.path.exists(os.path.join(started["run_dir"], "result.json"))


def test_tail_cursor_is_incremental(tmp_path):
    started = _start(tmp_path, max_iterations=2)
    run_id = started["run_id"]
    orchestrate_result_impl(run_id, wait=True, timeout=15)

    first = orchestrate_tail_impl(run_id, cursor=0, limit=3)
    assert len(first["events"]) == 3
    assert first["cursor"] == 3
    assert first["more"] is True

    second = orchestrate_tail_impl(run_id, cursor=first["cursor"])
    assert all(e["seq"] > 3 for e in second["events"])  # no overlap with first page


def test_status_reflects_completion(tmp_path):
    started = _start(tmp_path, max_iterations=2)
    run_id = started["run_id"]
    orchestrate_result_impl(run_id, wait=True, timeout=15)

    status = orchestrate_status_impl(run_id)
    assert status["running"] is False
    assert status["phase"] == "run_finished"
    assert status["completed"] is True


def test_worker_output_written_to_file_and_previewed(tmp_path):
    started = _start(tmp_path, max_iterations=1)
    run_id = started["run_id"]
    orchestrate_result_impl(run_id, wait=True, timeout=15)

    tail = orchestrate_tail_impl(run_id, cursor=0)
    finished = [e for e in tail["events"] if e["kind"] == "subagent_finished"]
    assert finished
    data = finished[0]["data"]
    assert "preview" in data and "output" not in data  # bulky output moved to a file
    output_path = data["output_path"]
    assert os.path.exists(output_path)
    with open(output_path, encoding="utf-8") as f:
        assert "deliverable produced" in f.read()


def test_list_includes_started_runs(tmp_path):
    started = _start(tmp_path, max_iterations=2)
    orchestrate_result_impl(started["run_id"], wait=True, timeout=15)
    listing = orchestrate_list_impl()
    assert any(r["run_id"] == started["run_id"] for r in listing["runs"])


def test_unknown_run_id_is_reported():
    assert "error" in orchestrate_status_impl("does-not-exist")
    assert "error" in orchestrate_tail_impl("does-not-exist")
    assert "error" in orchestrate_result_impl("does-not-exist")


def test_detached_start_returns_immediately_and_runs_intake_in_background(tmp_path):
    """The start call returns a run_id at once; intake then runs on the run's own
    thread (so a blocking/slow intake can never time out or freeze the caller)."""
    started = _start(tmp_path, max_iterations=2)
    assert started["status"] == "running"

    orchestrate_result_impl(started["run_id"], wait=True, timeout=15)
    kinds = [e["kind"] for e in orchestrate_tail_impl(started["run_id"], cursor=0)["events"]]
    # run_started is emitted before intake even begins, so status is observable
    # immediately; intake bookends come from the background thread.
    assert "intake_started" in kinds and "intake_finished" in kinds
    assert kinds.index("run_started") < kinds.index("intake_started")
    assert kinds.index("intake_finished") < kinds.index("decomposed")


def test_detached_resume_runs_in_background(tmp_path, monkeypatch):
    from agentloop.adapters import MockAgent

    # An agent that asks a clarifying question so intake suspends, then completes.
    monkeypatch.setattr(
        "agentloop.mcp_server._build_agent",
        lambda *a, **k: MockAgent(questions=["Which scope?"], accept_on_iteration=1),
    )

    # Detached start returns immediately; the clarification pause surfaces through
    # the run's result, not by blocking the start call.
    started = orchestrate_start_impl("goal", "", backend="mock",
                                     max_iterations=2, base_dir=str(tmp_path))
    assert started["status"] == "running"

    paused = orchestrate_result_impl(started["run_id"], wait=True, timeout=15)
    assert paused["status"] == "needs_input"
    assert paused["stop_reason"] == "needs_input"
    assert paused["questions"] == ["Which scope?"]
    assert paused["completed"] is False

    # A human tailing the run sees the pause as an event too.
    kinds = [e["kind"] for e in orchestrate_tail_impl(started["run_id"], cursor=0)["events"]]
    assert "needs_input" in kinds

    resumed = orchestrate_resume_impl(
        paused["token"], ["all of it"], detach=True, base_dir=str(tmp_path))
    assert resumed["status"] == "running"

    result = orchestrate_result_impl(resumed["run_id"], wait=True, timeout=15)
    assert result["completed"] is True
