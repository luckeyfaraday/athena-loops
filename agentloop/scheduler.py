"""Subagent execution — the 'Task Execution (parallel or sequential)' box.

Owns fan-out, retries, and failure capture. A subagent that raises does NOT
crash the loop; it becomes a FAILED TaskResult that the reviewer and the
feedback step can see and react to (a gap the original diagram leaves open).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .agent import Agent, AgentRequest
from .roles import SUBAGENT_SYSTEM, subagent_prompt
from .types import Subgoal, TaskResult, TaskStatus


def _run_one(
    agent: Agent,
    subgoal: Subgoal,
    goal: str,
    max_retries: int,
    on_call: Callable[[], None],
) -> TaskResult:
    last_error = ""
    for attempt in range(1, max_retries + 2):  # 1 try + N retries
        try:
            on_call()
            resp = agent.run(
                AgentRequest(
                    role="subagent",
                    system=SUBAGENT_SYSTEM,
                    prompt=subagent_prompt(subgoal, goal),
                    context={"subgoal_id": subgoal.id},
                )
            )
            text = resp.text.strip()
            if not text:
                raise ValueError("empty output")
            return TaskResult(subgoal, TaskStatus.OK, output=text, attempts=attempt)
        except Exception as exc:  # noqa: BLE001 — deliberately broad; recorded, not swallowed
            last_error = f"{type(exc).__name__}: {exc}"
            subgoal.notes = f"Previous attempt failed: {last_error}"
    return TaskResult(subgoal, TaskStatus.FAILED, error=last_error, attempts=max_retries + 1)


def execute(
    agent: Agent,
    subgoals: list[Subgoal],
    goal: str,
    max_retries: int,
    on_call: Callable[[], None],
    parallel: bool = True,
) -> list[TaskResult]:
    """Run all subgoals, preserving input order in the returned results."""
    if not parallel or len(subgoals) <= 1:
        return [_run_one(agent, sg, goal, max_retries, on_call) for sg in subgoals]

    with ThreadPoolExecutor(max_workers=min(len(subgoals), 8)) as pool:
        futures = [
            pool.submit(_run_one, agent, sg, goal, max_retries, on_call) for sg in subgoals
        ]
        return [f.result() for f in futures]
