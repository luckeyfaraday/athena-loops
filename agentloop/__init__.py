"""agentloop — a backend-agnostic implementation of the AI Agent Orchestration Loop.

    from agentloop import Orchestrator, Budget
    from agentloop.adapters import MockAgent

    orch = Orchestrator(MockAgent(), budget=Budget(max_iterations=3))
    result = orch.run(goal="...", success_criteria="...")
"""

from .agent import Agent, AgentRequest, AgentResponse, extract_json
from .interaction import (
    AutoInteraction,
    ConsoleInteraction,
    Interaction,
    NeedInput,
    SuspendInteraction,
)
from .orchestrator import Orchestrator
from .worktree import Worktree, worktree
from .types import (
    Budget,
    IterationTrace,
    LoopResult,
    LoopState,
    ReviewResult,
    Subgoal,
    TaskResult,
    TaskStatus,
)

__all__ = [
    "Agent",
    "AgentRequest",
    "AgentResponse",
    "extract_json",
    "Interaction",
    "AutoInteraction",
    "ConsoleInteraction",
    "SuspendInteraction",
    "NeedInput",
    "Orchestrator",
    "Worktree",
    "worktree",
    "Budget",
    "Subgoal",
    "TaskResult",
    "TaskStatus",
    "ReviewResult",
    "IterationTrace",
    "LoopResult",
    "LoopState",
]
