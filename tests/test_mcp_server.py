"""MCP server: the orchestrate tool's core logic + server construction.

`orchestrate_impl` is tested with the mock backend (no SDK, no real agent);
`build_server` is exercised only if the mcp SDK is importable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

from agentloop.mcp_server import (
    BACKENDS,
    _run_with_progress,
    doctor_impl,
    orchestrate_impl,
    orchestrate_suspendable,
)


def test_orchestrate_impl_mock_completes():
    out = orchestrate_impl("goal", "criteria", backend="mock", max_iterations=3)
    assert out["completed"] is True
    assert out["stop_reason"] == "goal_complete"
    assert out["final_output"]
    assert isinstance(out["history"], list) and out["history"]
    # The result must be JSON-serializable (it goes over the wire).
    json.dumps(out)


def test_result_carries_readable_summary():
    out = orchestrate_impl("g", "c", backend="mock", max_iterations=3)
    assert "completed in" in out["summary"] and "goal_complete" in out["summary"]


def test_run_with_progress_emits_per_iteration():
    """The MCP runner relays one progress notification per loop iteration."""
    import anyio

    events: list[tuple] = []

    class FakeCtx:
        async def report_progress(self, progress, total, message):
            events.append((progress, total, message))

    async def go():
        return await _run_with_progress(
            FakeCtx(),
            lambda observer: orchestrate_suspendable(
                "g", "c", backend="mock", max_iterations=3, observer=observer
            ),
        )

    out = anyio.run(go)
    assert len(events) == out["iterations"] + 1
    # The first event is immediate, before the first blocking worker iteration.
    p0, total0, msg0 = events[0]
    assert p0 == 0 and total0 == 1 and "starting" in msg0
    # Iteration progress remains 1-based iteration / total cap.
    p1, total1, msg1 = events[1]
    assert p1 == 1 and total1 == 3 and "iteration 1/3" in msg1


def test_run_with_progress_tolerates_no_context():
    """ctx=None (no live request) just runs the loop, no progress emitted."""
    import anyio

    out = anyio.run(
        lambda: _run_with_progress(
            None,
            lambda observer: orchestrate_suspendable(
                "g", "c", backend="mock", max_iterations=2, observer=observer
            ),
        )
    )
    assert out["completed"] is True


def test_orchestrate_impl_reports_history_shape():
    out = orchestrate_impl("g", "c", backend="mock", max_iterations=2)
    h0 = out["history"][0]
    assert set(h0) == {"iteration", "subgoals", "results", "review", "verification"}
    assert set(h0["review"]) == {"gates_passed", "goal_complete", "issues"}


def test_orchestrate_impl_runs_verification_commands():
    out = orchestrate_impl(
        "g", "c", backend="mock", max_iterations=2,
        verify_commands=[f"{sys.executable} -c 'print(123)'"],
    )
    verification = out["history"][0]["verification"]
    assert verification[0]["ok"] is True
    assert verification[0]["stdout"].strip() == "123"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        orchestrate_impl("g", "c", backend="nope")


def test_orchestrate_impl_reports_cli_loop_failure_after_intake_fallback(monkeypatch):
    from agentloop.adapters import CliAgent

    bad_agent = CliAgent([sys.executable, "-c", "import sys; sys.exit(7)"])
    monkeypatch.setattr(
        "agentloop.mcp_server._build_agent",
        lambda *args, **kwargs: bad_agent,
    )

    out = orchestrate_impl("g", "c", backend="claude_code")

    assert out["completed"] is False
    assert out["iterations"] == 0
    assert out["stop_reason"] == "loop_agent_error"
    assert "CLI agent exited 7" in out["error"]
    assert "stopped in 0 iteration(s)" in out["summary"]


def test_orchestrate_impl_intake_failure_falls_back(monkeypatch):
    from agentloop.adapters import MockAgent

    class IntakeFailingAgent:
        def __init__(self):
            self.inner = MockAgent(accept_on_iteration=1)

        def run(self, req):
            if req.role in ("criteria", "clarifier"):
                raise RuntimeError("quota exhausted")
            return self.inner.run(req)

    monkeypatch.setattr(
        "agentloop.mcp_server._build_agent",
        lambda *args, **kwargs: IntakeFailingAgent(),
    )

    out = orchestrate_impl("g", "", backend="claude_code")

    assert out["completed"] is True
    assert out["stop_reason"] == "goal_complete"


def test_isolate_runs_in_worktree_and_reports_it():
    repo = tempfile.mkdtemp(prefix="agentloop-repo-")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", repo, "init", "-q"], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-qm", "init", "--allow-empty"],
                   check=True, env=env)
    out = orchestrate_impl("g", "c", backend="mock", cwd=repo, isolate=True)
    assert "worktree" in out
    assert out["worktree"]["branch"].startswith("agentloop/")
    # mock backend edits nothing -> pristine worktree, no changed files.
    assert out["worktree"]["changed_files"] == []


def test_backends_listed():
    assert "mock" in BACKENDS and "claude_code" in BACKENDS


def test_doctor_reports_backends_and_timeout_guidance(tmp_path):
    out = doctor_impl(str(tmp_path))
    assert out["ok"] is True
    assert out["cwd"]["exists"] is True
    assert "mock" in out["backends"] and out["backends"]["mock"]["available"] is True
    assert "claude_code" in out["backends"]
    assert out["timeouts"]["recommended_mcp_request_timeout_ms"] >= 600000
    assert any("-32001" in rec for rec in out["recommendations"])


def test_build_server_constructs():
    pytest.importorskip("mcp")
    from agentloop.mcp_server import build_server
    server = build_server()
    assert server.name == "agentloop"
