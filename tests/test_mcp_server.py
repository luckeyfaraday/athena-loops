"""MCP server: the orchestrate tool's core logic + server construction.

`orchestrate_impl` is tested with the mock backend (no SDK, no real agent);
`build_server` is exercised only if the mcp SDK is importable.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

from agentloop.mcp_server import (
    BACKENDS,
    _run_with_progress,
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
    import json
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
    assert len(events) == out["iterations"] >= 1
    # progress is 1-based iteration / total cap, with a human-readable message.
    p0, total0, msg0 = events[0]
    assert p0 == 1 and total0 == 3 and "iteration 1/3" in msg0


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
    assert set(h0) == {"iteration", "subgoals", "results", "review"}
    assert set(h0["review"]) == {"gates_passed", "goal_complete", "issues"}


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        orchestrate_impl("g", "c", backend="nope")


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


def test_build_server_constructs():
    pytest.importorskip("mcp")
    from agentloop.mcp_server import build_server
    server = build_server()
    assert server.name == "agentloop"
