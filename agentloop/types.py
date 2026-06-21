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


class Phase(str, Enum):
    """The loop's stages, as first-class state rather than implicit call order.

    The orchestrator already moves through these stages in a fixed order — they
    were just anonymous (the numbered comments in `run_loop`). Naming them lets a
    run answer "where am I right now?" and gives later per-phase policy (scoping
    which tools a stage may use, routing a stage to a cheaper/stronger model)
    something concrete to switch on. This enum is that shared vocabulary; it does
    not by itself change what the loop does.

    INTAKE predates the LoopState today (it runs before the loop owns any state),
    so it is part of the vocabulary but not yet stamped onto a run — wiring it is
    a follow-up that gives intake its own state object.
    """

    IDLE = "idle"            # constructed, not yet running
    INTAKE = "intake"        # clarify criteria / gather answers (pre-loop)
    DECOMPOSE = "decompose"  # plan: goal -> subgoals
    EXECUTE = "execute"      # fan out to subagents, aggregate
    VERIFY = "verify"        # run deterministic verification commands
    REVIEW = "review"        # quality / consistency / goal-alignment gates
    DONE = "done"            # terminal: goal met
    FAILED = "failed"        # terminal: budget exhausted before the goal was met


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
class VerifyCommand:
    """One deterministic command the harness runs to verify an iteration."""

    name: str
    command: list[str]
    timeout: Optional[float] = None


@dataclass
class VerifyResult:
    """Captured result of a verification command."""

    name: str
    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    duration: float = 0.0
    error: str = ""


@dataclass
class IterationTrace:
    """A record of one cycle — useful for debugging and for the feedback prompt."""

    iteration: int
    subgoals: list[Subgoal]
    results: list[TaskResult]
    aggregated: str
    review: ReviewResult
    verification: list[VerifyResult] = field(default_factory=list)


@dataclass
class LoopResult:
    completed: bool
    final_output: str
    iterations: int
    stop_reason: str
    history: list[IterationTrace] = field(default_factory=list)


# --- live events (the observability stream) ---------------------------------
#
# Without these a run is a black box: the caller sees nothing until the whole
# loop returns. Each meaningful step emits a LoopEvent so the human (via a
# tail-able events.jsonl) and the calling agent (via the orchestrate_tail tool)
# can watch the loop work in real time. The orchestrator emits the iteration
# events; the scheduler emits the subagent ones; the detached launcher emits the
# run-level bookends and the pre-loop intake events.
EVENT_RUN_STARTED = "run_started"
EVENT_INTAKE_STARTED = "intake_started"
EVENT_INTAKE_FINISHED = "intake_finished"
# Between intake and the first iteration the launcher may build an isolated git
# worktree (a checkout that can be slow or, if a filter prompts, blocking). This
# event makes that otherwise-invisible gap observable so a stall there is
# attributable rather than looking like a frozen intake.
EVENT_WORKTREE_READY = "worktree_ready"
EVENT_ITERATION_STARTED = "iteration_started"
EVENT_DECOMPOSED = "decomposed"
EVENT_SUBAGENT_STARTED = "subagent_started"
EVENT_SUBAGENT_FINISHED = "subagent_finished"
EVENT_SUBAGENT_FAILED = "subagent_failed"
EVENT_AGGREGATED = "aggregated"
EVENT_VERIFICATION = "verification"
EVENT_REVIEW = "review"
EVENT_ITERATION_FINISHED = "iteration_finished"
EVENT_RUN_FINISHED = "run_finished"
EVENT_RUN_ERROR = "run_error"
# Intake paused for clarification; the run stops with questions + a resume token
# in its result instead of blocking the start call to ask.
EVENT_NEEDS_INPUT = "needs_input"
# The loop moved from one Phase to another. Carries {"from": <phase>, "to":
# <phase>}; the run-level status uses "to" as the run's current phase, so the
# lens reports which stage the run is actually in rather than the last event.
EVENT_PHASE_CHANGED = "phase_changed"


@dataclass
class LoopEvent:
    """One observable moment in a run — the unit of live visibility.

    `seq` is a per-run monotonic counter that doubles as the tail cursor: ask for
    everything with seq greater than the last you have seen. `iteration` is 0 for
    pre-loop (intake/run) events.
    """

    seq: int
    ts: float
    kind: str
    iteration: int
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "kind": self.kind,
            "iteration": self.iteration,
            "data": self.data,
        }


@dataclass
class LoopState:
    """Mutable state threaded through the loop."""

    goal: str
    success_criteria: str
    budget: Budget
    phase: Phase = Phase.IDLE   # which stage the loop is in right now
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
