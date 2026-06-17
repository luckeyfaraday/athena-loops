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
from .interaction import AutoInteraction, Interaction, NeedInput
from .roles import (
    CLARIFIER_SYSTEM,
    CRITERIA_SYSTEM,
    DECOMPOSER_SYSTEM,
    REVIEWER_SYSTEM,
    aggregate,
    build_feedback,
    clarify_prompt,
    criteria_prompt,
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
from .verifier import CommandVerifier, summarize_verification

DEFAULT_SUCCESS_CRITERIA = "Complete the goal as stated."

# A hook called once per cycle with the live state — for logging / progress UIs.
Observer = Callable[[LoopState], None]
# A hook called after each iteration to persist progress (e.g. commit a worktree),
# so partial work is never lost if the run later fails or is stopped.
Checkpoint = Callable[[LoopState], None]


class Orchestrator:
    def __init__(
        self,
        agent: Agent,
        *,
        budget: Optional[Budget] = None,
        parallel: bool = True,
        observer: Optional[Observer] = None,
        checkpoint: Optional[Checkpoint] = None,
        interaction: Optional[Interaction] = None,
        max_clarifying_questions: int = 4,
        verifier: Optional[CommandVerifier] = None,
    ):
        self.agent = agent
        self.budget = budget or Budget()
        self.parallel = parallel
        self.observer = observer
        # Called after each iteration to persist partial work (worktree commit).
        self.checkpoint = checkpoint
        # How the loop reaches the human. Default headless so existing callers
        # never block; pass ConsoleInteraction/SuspendInteraction for real UX.
        self.interaction = interaction or AutoInteraction()
        self.max_clarifying_questions = max_clarifying_questions
        self.verifier = verifier
        self._lock = threading.Lock()

    # --- intake -------------------------------------------------------------

    def run(self, goal: str, success_criteria: str = "") -> LoopResult:
        """Intake (clarify) then run the loop. Convenience for Python/console use."""
        goal, success_criteria, clarifications = self.intake(goal, success_criteria)
        return self.run_loop(goal, success_criteria, clarifications)

    def intake(self, goal: str, success_criteria: str = "") -> tuple[str, str, str]:
        """Resolve criteria and gather clarifying answers before any planning.

        Realizes the diagram's 'App Follow-up Questions': the orchestrator may
        propose success criteria and ask the user questions through the
        Interaction seam. Returns (goal, criteria, clarifications). May raise
        NeedInput (via SuspendInteraction) to hand control back to the caller.
        """
        criteria = (success_criteria or "").strip()
        if not criteria:
            try:
                proposed = self.agent.run(AgentRequest(
                    role="criteria", system=CRITERIA_SYSTEM, prompt=criteria_prompt(goal),
                )).text.strip()
            except Exception:  # noqa: BLE001 - criteria drafting is best-effort intake
                proposed = DEFAULT_SUCCESS_CRITERIA
            if self.interaction.confirm(f"Proposed success criteria:\n  {proposed}\nUse these?"):
                criteria = proposed
            else:
                criteria = (self.interaction.ask(["Enter your success criteria:"])[0]
                            or proposed)

        questions = self._clarify(goal, criteria)
        clarifications = ""
        if questions:
            try:
                answers = self.interaction.ask(questions)
            except NeedInput as ni:
                ni.goal, ni.criteria = goal, criteria  # let the caller build a resume token
                raise
            clarifications = "\n".join(
                f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)
            )
        return goal, criteria, clarifications

    def _clarify(self, goal: str, criteria: str) -> list[str]:
        try:
            resp = self.agent.run(AgentRequest(
                role="clarifier", system=CLARIFIER_SYSTEM,
                prompt=clarify_prompt(goal, criteria, self.max_clarifying_questions),
                expects_json=True,
            ))
            data = extract_json(resp.text)
            return [str(q) for q in data][: self.max_clarifying_questions]
        except Exception:  # noqa: BLE001 - optional intake should not block work
            return []  # no usable questions -> nothing to ask, just proceed

    # --- the loop -----------------------------------------------------------

    def run_loop(
        self, goal: str, success_criteria: str, clarifications: str = ""
    ) -> LoopResult:
        state = LoopState(goal=goal, success_criteria=success_criteria, budget=self.budget)
        state.clarifications = clarifications

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

            # 5. Run deterministic verification commands, if configured.
            verification = self.verifier.run() if self.verifier else []
            verification_text = summarize_verification(verification)

            # 6. Reviewer agent — quality / consistency / goal-alignment gates.
            review = self._review(state, aggregated, verification_text, count_call)

            state.history.append(
                IterationTrace(
                    state.iteration, subgoals, results, aggregated, review, verification
                )
            )
            if self.observer:
                self.observer(state)

            # Persist this iteration's work before deciding/looping, so partial
            # progress survives a later failure or a budget stop. Never let a
            # checkpoint error break the run.
            if self.checkpoint:
                try:
                    self.checkpoint(state)
                except Exception:  # noqa: BLE001 — checkpointing is best-effort
                    pass

            # 7 & 8. Goal completed? YES -> deliver. NO -> refine and loop.
            all_ok = all(r.ok for r in results)
            verification_ok = all(r.ok for r in verification)
            if review.gates_passed and review.goal_complete and all_ok and verification_ok:
                return self._finish(state, completed=True, stop_reason="goal_complete",
                                    final=aggregated)

            # Feedback loop: update context, refine plan, adjust subgoals/tasks.
            state.feedback = build_feedback(results, review, verification)

    # --- stages (each is one Agent call against the shared interface) --------

    def _decompose(self, state: LoopState, count_call: Callable[[], None]) -> list[Subgoal]:
        count_call()
        resp = self.agent.run(
            AgentRequest(
                role="decomposer",
                system=DECOMPOSER_SYSTEM,
                prompt=decompose_prompt(
                    state.goal, state.success_criteria, state.feedback, state.clarifications
                ),
                expects_json=True,
            )
        )
        try:
            data = extract_json(resp.text)
            if not isinstance(data, list) or not data:
                raise ValueError("decomposer did not return a non-empty JSON array")
            subgoals = [self._to_subgoal(item, i) for i, item in enumerate(data)]
            return subgoals
        except (ValueError, KeyError, TypeError, AttributeError):
            # Degrade gracefully: treat the whole goal as a single subgoal rather
            # than crash the run on a malformed decomposition.
            return [Subgoal(id="s1", description=state.goal)]

    @staticmethod
    def _to_subgoal(item: object, i: int) -> Subgoal:
        """Accept either {"id","description"} objects or bare strings."""
        if isinstance(item, str):
            return Subgoal(id=f"s{i+1}", description=item)
        if isinstance(item, dict):
            desc = item.get("description") or item.get("task") or item.get("goal")
            if not desc:
                raise ValueError(f"subgoal item missing a description: {item!r}")
            return Subgoal(id=str(item.get("id", f"s{i+1}")), description=str(desc))
        raise TypeError(f"unsupported subgoal item type: {type(item).__name__}")

    def _review(
        self, state: LoopState, aggregated: str, verification: str,
        count_call: Callable[[], None]
    ) -> ReviewResult:
        count_call()
        resp = self.agent.run(
            AgentRequest(
                role="reviewer",
                system=REVIEWER_SYSTEM,
                prompt=review_prompt(
                    state.goal, state.success_criteria, aggregated, verification
                ),
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
