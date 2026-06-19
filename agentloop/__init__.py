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
    LoopEvent,
    LoopResult,
    LoopState,
    Phase,
    ReviewResult,
    Subgoal,
    TaskResult,
    TaskStatus,
    VerifyCommand,
    VerifyResult,
)
from .verifier import CommandVerifier, parse_verify_command, summarize_verification

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
    "VerifyCommand",
    "VerifyResult",
    "CommandVerifier",
    "parse_verify_command",
    "summarize_verification",
    "ReviewResult",
    "IterationTrace",
    "LoopEvent",
    "LoopResult",
    "LoopState",
    "Phase",
]
