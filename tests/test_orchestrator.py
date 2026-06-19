"""Tests for the harness guarantees: the loop terminates, feedback fires,
failures are captured, and budget guards stop runaway loops."""

from __future__ import annotations

import sys

from agentloop import (
    Budget,
    CommandVerifier,
    Orchestrator,
    Phase,
    VerifyCommand,
    extract_json,
)
from agentloop.adapters import MockAgent
from agentloop.agent import AgentRequest


def _phase_path(**orch_kwargs):
    """Run a loop and return the ordered list of phases it moved through."""
    moves: list[str] = []
    orch = Orchestrator(
        emit=lambda kind, _i, data: (
            moves.append(data["to"]) if kind == "phase_changed" else None
        ),
        **orch_kwargs,
    )
    result = orch.run("goal", "criteria")
    return result, moves


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


class _RecordingAgent(MockAgent):
    """A MockAgent that remembers the system prompt seen for each role."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.systems: dict[str, str] = {}

    def run(self, request: AgentRequest) -> "object":
        self.systems[request.role] = request.system
        return super().run(request)


def test_playwright_off_keeps_prompts_clean():
    agent = _RecordingAgent(accept_on_iteration=1)
    Orchestrator(agent, budget=Budget(max_iterations=1)).run("goal", "criteria")
    assert "Playwright" not in agent.systems["subagent"]
    assert "Playwright" not in agent.systems["reviewer"]


def test_playwright_on_injects_prompt_guidance():
    agent = _RecordingAgent(accept_on_iteration=1)
    Orchestrator(
        agent, budget=Budget(max_iterations=1), playwright=True
    ).run("goal", "criteria")
    # Both the worker and the reviewer get the Playwright guidance.
    assert "Playwright" in agent.systems["subagent"]
    assert "Playwright" in agent.systems["reviewer"]


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


def test_phases_move_through_the_loop_in_order_to_done():
    result, moves = _phase_path(
        agent=MockAgent(accept_on_iteration=1), budget=Budget(max_iterations=2),
    )
    assert result.completed
    # One clean pass: plan -> execute -> review -> done. No verifier, so no VERIFY.
    assert moves == [Phase.DECOMPOSE, Phase.EXECUTE, Phase.REVIEW, Phase.DONE]


def test_verify_phase_appears_only_with_a_verifier():
    verifier = CommandVerifier([
        VerifyCommand("check", [sys.executable, "-c", "print('ok')"])
    ])
    _result, moves = _phase_path(
        agent=MockAgent(accept_on_iteration=1),
        budget=Budget(max_iterations=2), verifier=verifier,
    )
    assert moves == [
        Phase.DECOMPOSE, Phase.EXECUTE, Phase.VERIFY, Phase.REVIEW, Phase.DONE,
    ]


def test_looping_back_shows_a_review_to_decompose_transition():
    # Rejected on pass 1, accepted on pass 2: the loop re-enters DECOMPOSE, which
    # is the structural signal that the feedback loop fired.
    _result, moves = _phase_path(
        agent=MockAgent(accept_on_iteration=2), budget=Budget(max_iterations=5),
    )
    assert moves == [
        Phase.DECOMPOSE, Phase.EXECUTE, Phase.REVIEW,   # pass 1: not accepted
        Phase.DECOMPOSE, Phase.EXECUTE, Phase.REVIEW,   # pass 2 (re-entry)
        Phase.DONE,
    ]


def test_exhausted_budget_ends_in_failed_phase():
    result, moves = _phase_path(
        agent=MockAgent(accept_on_iteration=999), budget=Budget(max_iterations=2),
    )
    assert not result.completed
    assert moves[-1] == Phase.FAILED
    assert Phase.DONE not in moves


def test_extract_json_recovers_from_prose_and_fences():
    assert extract_json('here you go: {"a": 1} thanks') == {"a": 1}
    assert extract_json('```json\n[1, 2, 3]\n```') == [1, 2, 3]
    assert extract_json('[{"id": "s1", "description": "x"}]')[0]["id"] == "s1"
