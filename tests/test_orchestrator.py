"""Tests for the harness guarantees: the loop terminates, feedback fires,
failures are captured, and budget guards stop runaway loops."""

from __future__ import annotations

import sys

from agentloop import Budget, CommandVerifier, Orchestrator, VerifyCommand, extract_json
from agentloop.adapters import MockAgent
from agentloop.agent import AgentRequest


def test_happy_path_completes():
    orch = Orchestrator(MockAgent(accept_on_iteration=1), budget=Budget(max_iterations=3))
    result = orch.run("goal", "criteria")
    assert result.completed
    assert result.iterations == 1
    assert result.stop_reason == "goal_complete"


def test_feedback_loop_runs_until_accept():
    # MockAgent rejects pass 1, accepts pass 2 -> NO branch must fire once.
    orch = Orchestrator(MockAgent(accept_on_iteration=2), budget=Budget(max_iterations=5))
    result = orch.run("goal", "criteria")
    assert result.completed
    assert result.iterations == 2
    # The first iteration must have produced a non-passing review (the feedback signal).
    assert not result.history[0].review.goal_complete
    assert result.history[1].review.goal_complete


def test_budget_guard_stops_runaway_loop():
    # Never accepts -> must stop on the guard, not spin forever.
    orch = Orchestrator(MockAgent(accept_on_iteration=999), budget=Budget(max_iterations=3))
    result = orch.run("goal", "criteria")
    assert not result.completed
    assert result.iterations == 3
    assert "max_iterations" in result.stop_reason


def test_emit_streams_fine_grained_events():
    events: list[tuple] = []
    orch = Orchestrator(
        MockAgent(accept_on_iteration=1), budget=Budget(max_iterations=2),
        emit=lambda kind, iteration, data: events.append((kind, iteration, data)),
    )
    result = orch.run("goal", "criteria")
    assert result.completed
    kinds = [k for k, _i, _d in events]
    # Every phase of the iteration is observable, not just the final result.
    for expected in ("iteration_started", "decomposed", "subagent_started",
                     "subagent_finished", "review", "iteration_finished"):
        assert expected in kinds
    # The worker event carries what the subagent actually produced.
    finished = next(d for k, _i, d in events if k == "subagent_finished")
    assert "deliverable produced" in finished["output"]


def test_emit_errors_never_break_the_run():
    def boom(*_args):
        raise RuntimeError("event sink is down")

    orch = Orchestrator(MockAgent(accept_on_iteration=1),
                        budget=Budget(max_iterations=2), emit=boom)
    assert orch.run("goal", "criteria").completed  # survives a broken sink


def test_intake_agent_failure_falls_back_and_continues():
    class IntakeFailingAgent:
        def __init__(self):
            self.inner = MockAgent(accept_on_iteration=1)

        def run(self, req: AgentRequest):
            if req.role in ("criteria", "clarifier"):
                raise RuntimeError("quota exhausted")
            return self.inner.run(req)

    orch = Orchestrator(IntakeFailingAgent(), budget=Budget(max_iterations=2))

    result = orch.run("goal")

    assert result.completed
    assert result.stop_reason == "goal_complete"


def test_subagent_failure_is_captured_not_raised():
    def boom(_req: AgentRequest) -> str:
        raise RuntimeError("tool exploded")

    agent = MockAgent(subgoals=["only one"], subagent_fn=boom, accept_on_iteration=1)
    orch = Orchestrator(agent, budget=Budget(max_iterations=2, max_task_retries=1))
    result = orch.run("goal", "criteria")
    # The run survives; the failure is recorded and blocks completion.
    assert not result.completed
    failed = result.history[0].results[0]
    assert not failed.ok
    assert "tool exploded" in failed.error
    assert failed.attempts == 2  # 1 try + 1 retry


def test_agent_call_budget_guard():
    orch = Orchestrator(MockAgent(accept_on_iteration=999),
                        budget=Budget(max_iterations=99, max_agent_calls=3))
    result = orch.run("goal", "criteria")
    assert not result.completed
    assert "max_agent_calls" in result.stop_reason


def test_checkpoint_called_once_per_iteration():
    seen = []
    orch = Orchestrator(MockAgent(accept_on_iteration=2),
                        checkpoint=lambda st: seen.append(st.iteration),
                        budget=Budget(max_iterations=5))
    result = orch.run("goal", "criteria")
    assert result.completed
    assert seen == [1, 2]  # one checkpoint per completed iteration


def test_checkpoint_error_does_not_break_the_run():
    def boom(_state):
        raise RuntimeError("commit failed")
    orch = Orchestrator(MockAgent(accept_on_iteration=1), checkpoint=boom,
                        budget=Budget(max_iterations=2))
    assert orch.run("goal", "criteria").completed  # survives a checkpoint failure


def test_verification_failure_blocks_completion_and_feeds_back():
    verifier = CommandVerifier([
        VerifyCommand(
            "failing check",
            [sys.executable, "-c", "import sys; print('broken'); sys.exit(7)"],
        )
    ])
    orch = Orchestrator(
        MockAgent(accept_on_iteration=1),
        budget=Budget(max_iterations=2),
        verifier=verifier,
    )
    result = orch.run("goal", "criteria")
    assert not result.completed
    assert result.history[0].verification[0].exit_code == 7
    assert result.history[0].verification[0].stdout.strip() == "broken"
    assert result.iterations == 2


def test_passing_verification_allows_completion():
    verifier = CommandVerifier([
        VerifyCommand("passing check", [sys.executable, "-c", "print('ok')"])
    ])
    orch = Orchestrator(
        MockAgent(accept_on_iteration=1),
        budget=Budget(max_iterations=2),
        verifier=verifier,
    )
    result = orch.run("goal", "criteria")
    assert result.completed
    assert result.history[0].verification[0].ok


def test_extract_json_recovers_from_prose_and_fences():
    assert extract_json('here you go: {"a": 1} thanks') == {"a": 1}
    assert extract_json('```json\n[1, 2, 3]\n```') == [1, 2, 3]
    assert extract_json('[{"id": "s1", "description": "x"}]')[0]["id"] == "s1"
