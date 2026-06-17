"""Subagent execution — the 'Task Execution (parallel or sequential)' box.

Owns fan-out, retries, and failure capture. A subagent that raises does NOT
crash the loop; it becomes a FAILED TaskResult that the reviewer and the
feedback step can see and react to (a gap the original diagram leaves open).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from .agent import Agent, AgentRequest
from .roles import subagent_prompt, subagent_system
from .types import (
    EVENT_SUBAGENT_FAILED,
    EVENT_SUBAGENT_FINISHED,
    EVENT_SUBAGENT_STARTED,
    Subgoal,
    TaskResult,
    TaskStatus,
)

# An event sink, already bound to the current iteration: (kind, data) -> None.
# May be called concurrently from worker threads, so the sink must be thread-safe.
OnEvent = Callable[[str, dict[str, Any]], None]


def _run_one(
    agent: Agent,
    subgoal: Subgoal,
    goal: str,
    max_retries: int,
    on_call: Callable[[], None],
    on_event: Optional[OnEvent] = None,
    playwright: bool = False,
) -> TaskResult:
    emit = on_event or (lambda _kind, _data: None)
    last_error = ""
    for attempt in range(1, max_retries + 2):  # 1 try + N retries
        emit(EVENT_SUBAGENT_STARTED, {
            "id": subgoal.id, "description": subgoal.description, "attempt": attempt,
        })
        try:
            on_call()
            resp = agent.run(
                AgentRequest(
                    role="subagent",
                    system=subagent_system(playwright),
                    prompt=subagent_prompt(subgoal, goal),
                    context={"subgoal_id": subgoal.id},
                )
            )
            text = resp.text.strip()
            if not text:
                raise ValueError("empty output")
            emit(EVENT_SUBAGENT_FINISHED, {
                "id": subgoal.id, "attempt": attempt, "output": text,
            })
            return TaskResult(subgoal, TaskStatus.OK, output=text, attempts=attempt)
        except Exception as exc:  # noqa: BLE001 — deliberately broad; recorded, not swallowed
            last_error = f"{type(exc).__name__}: {exc}"
            emit(EVENT_SUBAGENT_FAILED, {
                "id": subgoal.id, "attempt": attempt, "error": last_error,
            })
            # The failed attempt may have left partial work in the working dir;
            # tell the retry to inspect it and continue rather than start over.
            subgoal.notes = (
                f"A previous attempt failed partway: {last_error}. The working "
                "directory may already contain partial changes from it — inspect "
                "the current state and CONTINUE; do not restart from scratch."
            )
    return TaskResult(subgoal, TaskStatus.FAILED, error=last_error, attempts=max_retries + 1)


def execute(
    agent: Agent,
    subgoals: list[Subgoal],
    goal: str,
    max_retries: int,
    on_call: Callable[[], None],
    parallel: bool = True,
    on_event: Optional[OnEvent] = None,
    playwright: bool = False,
) -> list[TaskResult]:
    """Run all subgoals, preserving input order in the returned results."""
    if not parallel or len(subgoals) <= 1:
        return [
            _run_one(agent, sg, goal, max_retries, on_call, on_event, playwright)
            for sg in subgoals
        ]

    with ThreadPoolExecutor(max_workers=min(len(subgoals), 8)) as pool:
        futures = [
            pool.submit(_run_one, agent, sg, goal, max_retries, on_call, on_event, playwright)
            for sg in subgoals
        ]
        return [f.result() for f in futures]
