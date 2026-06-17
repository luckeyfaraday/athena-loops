"""The orchestration loop itself — the wiring of the diagram.

This is the deterministic harness. It guarantees the things a prompt cannot:
decomposition happens, subagents are fanned out and aggregated, the review gate
is enforced, and the loop terminates. Judgement lives in `roles.py`; control
lives here.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from .agent import Agent, AgentRequest, extract_json
from .roles import (
    DECOMPOSER_SYSTEM,
    REVIEWER_SYSTEM,
    aggregate,
    build_feedback,
    decompose_prompt,
    review_prompt,
)
from .scheduler import execute
from .types import (
    Budget,
    IterationTrace,
    LoopResult,
    LoopState,
    ReviewResult,
    Subgoal,
)

# A hook called once per cycle with the live state — for logging / progress UIs.
Observer = Callable[[LoopState], None]


class Orchestrator:
    def __init__(
        self,
        agent: Agent,
        *,
        budget: Optional[Budget] = None,
        parallel: bool = True,
        observer: Optional[Observer] = None,
    ):
        self.agent = agent
        self.budget = budget or Budget()
        self.parallel = parallel
        self.observer = observer
        self._lock = threading.Lock()

    # --- the loop -----------------------------------------------------------

    def run(self, goal: str, success_criteria: str) -> LoopResult:
        state = LoopState(goal=goal, success_criteria=success_criteria, budget=self.budget)

        # Thread-safe cost counter shared by every Agent call this run makes.
        def count_call() -> None:
            with self._lock:
                state.agent_calls += 1

        while True:
            stop = state.budget_exhausted()
            if stop:
                return self._finish(state, completed=False, stop_reason=stop)

            state.iteration += 1

            # 1. Task decomposition (feedback refines it on later passes).
            subgoals = self._decompose(state, count_call)

            # 2 & 3. Fan out to subagents, execute, capture failures.
            results = execute(
                self.agent,
                subgoals,
                state.goal,
                max_retries=self.budget.max_task_retries,
                on_call=count_call,
                parallel=self.parallel,
            )

            # 4. Aggregate finished task outputs.
            aggregated = aggregate(results)

            # 5. Reviewer agent — quality / consistency / goal-alignment gates.
            review = self._review(state, aggregated, count_call)

            state.history.append(
                IterationTrace(state.iteration, subgoals, results, aggregated, review)
            )
            if self.observer:
                self.observer(state)

            # 6 & 7. Goal completed? YES -> deliver. NO -> refine and loop.
            all_ok = all(r.ok for r in results)
            if review.gates_passed and review.goal_complete and all_ok:
                return self._finish(state, completed=True, stop_reason="goal_complete",
                                    final=aggregated)

            # Feedback loop: update context, refine plan, adjust subgoals/tasks.
            state.feedback = build_feedback(results, review)

    # --- stages (each is one Agent call against the shared interface) --------

    def _decompose(self, state: LoopState, count_call: Callable[[], None]) -> list[Subgoal]:
        count_call()
        resp = self.agent.run(
            AgentRequest(
                role="decomposer",
                system=DECOMPOSER_SYSTEM,
                prompt=decompose_prompt(state.goal, state.success_criteria, state.feedback),
                expects_json=True,
            )
        )
        try:
            data = extract_json(resp.text)
            subgoals = [
                Subgoal(id=str(item.get("id", f"s{i+1}")), description=item["description"])
                for i, item in enumerate(data)
            ]
            if not subgoals:
                raise ValueError("decomposer returned no subgoals")
            return subgoals
        except (ValueError, KeyError, TypeError):
            # Degrade gracefully: treat the whole goal as a single subgoal rather
            # than crash the run on a malformed decomposition.
            return [Subgoal(id="s1", description=state.goal)]

    def _review(
        self, state: LoopState, aggregated: str, count_call: Callable[[], None]
    ) -> ReviewResult:
        count_call()
        resp = self.agent.run(
            AgentRequest(
                role="reviewer",
                system=REVIEWER_SYSTEM,
                prompt=review_prompt(state.goal, state.success_criteria, aggregated),
                expects_json=True,
            )
        )
        try:
            d = extract_json(resp.text)
            return ReviewResult(
                quality_ok=bool(d["quality_ok"]),
                consistency_ok=bool(d["consistency_ok"]),
                goal_aligned=bool(d["goal_aligned"]),
                goal_complete=bool(d["goal_complete"]),
                issues=list(d.get("issues", [])),
                follow_up_questions=list(d.get("follow_up_questions", [])),
            )
        except (ValueError, KeyError, TypeError) as exc:
            # An unparseable review is a non-pass, not a silent accept.
            return ReviewResult(
                False, False, False, False,
                issues=[f"reviewer response could not be parsed: {exc}"],
            )

    # --- helpers ------------------------------------------------------------

    def _finish(
        self, state: LoopState, *, completed: bool, stop_reason: str, final: str = ""
    ) -> LoopResult:
        if not final and state.history:
            final = state.history[-1].aggregated
        return LoopResult(
            completed=completed,
            final_output=final,
            iterations=state.iteration,
            stop_reason=stop_reason,
            history=state.history,
        )
