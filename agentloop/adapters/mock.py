"""A deterministic, dependency-free backend for tests and demos.

It implements the same `Agent` interface as the real backends. By default it
fails the review on the first pass and accepts on the second, so running the
demo actually exercises the feedback loop (NO -> refine -> YES) end to end.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from ..agent import Agent, AgentRequest, AgentResponse


class MockAgent(Agent):
    def __init__(
        self,
        *,
        subgoals: Optional[list[str]] = None,
        accept_on_iteration: int = 2,
        subagent_fn: Optional[Callable[[AgentRequest], str]] = None,
        questions: Optional[list[str]] = None,
        proposed_criteria: str = "The stated goal is fully satisfied.",
    ):
        self.subgoals = subgoals or [
            "Research the topic and gather sources",
            "Draft the deliverable",
            "Proofread and format",
        ]
        self.accept_on_iteration = accept_on_iteration
        self.subagent_fn = subagent_fn
        self.questions = questions or []   # default: no clarifying questions
        self.proposed_criteria = proposed_criteria
        self._review_calls = 0

    def run(self, request: AgentRequest) -> AgentResponse:
        if request.role == "criteria":
            return AgentResponse(self.proposed_criteria)

        if request.role == "clarifier":
            return AgentResponse(json.dumps(self.questions))

        if request.role == "decomposer":
            return AgentResponse(json.dumps(
                [{"id": f"s{i+1}", "description": d} for i, d in enumerate(self.subgoals)]
            ))

        if request.role == "subagent":
            if self.subagent_fn:
                return AgentResponse(self.subagent_fn(request))
            return AgentResponse(f"Completed: {request.context.get('subgoal_id', '?')} — "
                                 f"deliverable produced.")

        if request.role == "reviewer":
            self._review_calls += 1
            accept = self._review_calls >= self.accept_on_iteration
            return AgentResponse(json.dumps({
                "quality_ok": accept,
                "consistency_ok": accept,
                "goal_aligned": accept,
                "goal_complete": accept,
                "issues": [] if accept else ["First pass too shallow; add specifics."],
                "follow_up_questions": [],
            }))

        return AgentResponse("")
