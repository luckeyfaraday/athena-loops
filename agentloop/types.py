"""Plain data types for the orchestration loop.

These describe *state* and *results* — the boxes and arrows of the diagram —
and deliberately contain no model-specific or backend-specific logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"


@dataclass
class Budget:
    """Termination guards. The diagram's NO-branch can loop forever; these stop it."""

    max_iterations: int = 5          # whole decompose->review cycles
    max_task_retries: int = 1        # per-subgoal retries on failure
    max_seconds: Optional[float] = None  # wall-clock ceiling for the whole run
    max_agent_calls: Optional[int] = None  # hard cap on backend calls (cost guard)


@dataclass
class Subgoal:
    """One unit of decomposed work, owned by one subagent."""

    id: str
    description: str
    # Carried across iterations so a retry/refine knows what went wrong last time.
    notes: str = ""


@dataclass
class TaskResult:
    subgoal: Subgoal
    status: TaskStatus
    output: str = ""
    error: str = ""
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.status is TaskStatus.OK


@dataclass
class ReviewResult:
    """Output of the Reviewer Agent's three gates + the completion verdict."""

    quality_ok: bool
    consistency_ok: bool
    goal_aligned: bool
    goal_complete: bool
    issues: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)

    @property
    def gates_passed(self) -> bool:
        return self.quality_ok and self.consistency_ok and self.goal_aligned


@dataclass
class IterationTrace:
    """A record of one cycle — useful for debugging and for the feedback prompt."""

    iteration: int
    subgoals: list[Subgoal]
    results: list[TaskResult]
    aggregated: str
    review: ReviewResult


@dataclass
class LoopResult:
    completed: bool
    final_output: str
    iterations: int
    stop_reason: str
    history: list[IterationTrace] = field(default_factory=list)


@dataclass
class LoopState:
    """Mutable state threaded through the loop."""

    goal: str
    success_criteria: str
    budget: Budget
    iteration: int = 0
    agent_calls: int = 0
    feedback: str = ""           # the "Update context, refine plan" signal
    clarifications: str = ""     # Q/A gathered during intake, fed into decomposition
    started_at: float = field(default_factory=time.monotonic)
    history: list[IterationTrace] = field(default_factory=list)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def budget_exhausted(self) -> Optional[str]:
        """Return a stop_reason string if any guard tripped, else None."""
        b = self.budget
        if self.iteration >= b.max_iterations:
            return f"max_iterations ({b.max_iterations}) reached"
        if b.max_seconds is not None and self.elapsed() >= b.max_seconds:
            return f"max_seconds ({b.max_seconds}s) reached"
        if b.max_agent_calls is not None and self.agent_calls >= b.max_agent_calls:
            return f"max_agent_calls ({b.max_agent_calls}) reached"
        return None
