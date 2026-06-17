"""Human-input seam: intake, clarification, criteria proposal, suspend/resume,
and the decomposer robustness fix."""

from __future__ import annotations

import json

import pytest

from agentloop import (
    AutoInteraction,
    Budget,
    ConsoleInteraction,
    NeedInput,
    Orchestrator,
    SuspendInteraction,
)
from agentloop.adapters import MockAgent
from agentloop.mcp_server import (
    orchestrate_impl,
    orchestrate_resume_impl,
    orchestrate_suspendable,
)


# --- intake / clarification --------------------------------------------------

def test_no_questions_runs_straight_through():
    orch = Orchestrator(MockAgent(questions=[]), budget=Budget(max_iterations=3))
    assert orch.run("goal", "criteria").completed


def test_clarifying_answers_feed_into_decomposition():
    captured = {}

    def spy_subagent(req):
        return "done"

    agent = MockAgent(questions=["Per-IP or per-key?"], accept_on_iteration=1,
                      subagent_fn=spy_subagent)

    class Fixed(AutoInteraction):
        def ask(self, questions):
            captured["asked"] = list(questions)
            return ["per-key"]

    orch = Orchestrator(agent, interaction=Fixed(), budget=Budget(max_iterations=2))
    g, c, clar = orch.intake("rate limit the API", "criteria")
    assert captured["asked"] == ["Per-IP or per-key?"]
    assert "per-key" in clar and "Per-IP" in clar


def test_criteria_proposed_when_missing():
    agent = MockAgent(questions=[], proposed_criteria="429s over 100 req/min, with tests")
    orch = Orchestrator(agent, interaction=AutoInteraction(), budget=Budget(max_iterations=2))
    _, criteria, _ = orch.intake("add rate limiting")  # no criteria passed
    assert criteria == "429s over 100 req/min, with tests"


def test_console_interaction_prompts(monkeypatch):
    # criteria is provided, so the only prompt is the clarifying question.
    monkeypatch.setattr("builtins.input", lambda *a: "my-key")
    agent = MockAgent(questions=["Per-IP or per-key?"])
    orch = Orchestrator(agent, interaction=ConsoleInteraction(),
                        budget=Budget(max_iterations=2))
    _, _, clar = orch.intake("goal", "criteria")
    assert "my-key" in clar


# --- suspend / resume --------------------------------------------------------

def test_suspend_raises_needinput_when_no_answers():
    orch = Orchestrator(MockAgent(questions=["Which framework?"]),
                        interaction=SuspendInteraction())
    with pytest.raises(NeedInput) as ei:
        orch.intake("build an app", "criteria")
    assert ei.value.questions == ["Which framework?"]
    assert ei.value.goal == "build an app"


def test_suspend_consumes_preloaded_answers():
    orch = Orchestrator(MockAgent(questions=["Which framework?"]),
                        interaction=SuspendInteraction(["FastAPI"]),
                        budget=Budget(max_iterations=2))
    _, _, clar = orch.intake("build an app", "criteria")
    assert "FastAPI" in clar


def test_orchestrate_suspendable_no_questions_completes():
    # The built-in mock backend asks nothing, so suspendable runs straight through.
    out = orchestrate_suspendable("g", "c", backend="mock", max_iterations=2)
    assert out["completed"] is True
    assert "status" not in out  # i.e. not a needs_input envelope


def test_resume_round_trip_via_token():
    # Hand-build a token like the one a suspended run would emit, then resume.
    from agentloop.mcp_server import _encode_token
    token = _encode_token({
        "goal": "g", "criteria": "c", "questions": ["Which DB?"],
        "backend": "mock", "max_iterations": 2,
    })
    out = orchestrate_resume_impl(token, ["Postgres"])
    assert out["completed"] is True


# --- decomposer robustness (regression for the AttributeError bug) -----------

def test_decompose_accepts_string_items():
    # Backend returns a JSON array of bare strings, not {id,description} objects.
    agent = MockAgent(subgoals=["a", "b"], accept_on_iteration=1)
    # MockAgent already emits objects; emulate the string-list bug via a wrapper.
    class StringDecomposer(MockAgent):
        def run(self, request):
            if request.role == "decomposer":
                from agentloop.agent import AgentResponse
                return AgentResponse(json.dumps(["do A", "do B"]))
            return super().run(request)

    orch = Orchestrator(StringDecomposer(accept_on_iteration=1),
                        budget=Budget(max_iterations=2))
    result = orch.run("goal", "criteria")
    assert result.completed
    descs = [sg.description for sg in result.history[0].subgoals]
    assert descs == ["do A", "do B"]
