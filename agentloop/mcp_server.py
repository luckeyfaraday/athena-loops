"""Outward plug: expose the orchestration loop as an MCP tool.

Any MCP-aware coding agent (Claude Code, Cursor, Codex, opencode, Cline,
Windsurf) can call `orchestrate(goal, success_criteria, ...)` and get the loop's
result back. The agent that calls it need not be the agent that does the work —
pick the worker `backend` independently.

Run it:  python3 -m agentloop.mcp_server   (stdio transport)

The core (`orchestrate_impl`) is plain Python with no MCP dependency, so it is
unit-testable without the SDK; `build_server()` is the thin FastMCP wrapper.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable, Optional

from .adapters import CliAgent, MockAgent
from .interaction import AutoInteraction, Interaction, NeedInput, SuspendInteraction
from .orchestrator import Orchestrator
from .types import Budget, LoopResult, LoopState
from .worktree import Worktree, worktree

# Worker backends the server can drive. The caller picks one per request.
_CLI_PRESETS = {
    "claude_code": CliAgent.claude_code,
    "codex": CliAgent.codex,
    "opencode": CliAgent.opencode,
    "aider": CliAgent.aider,
}
BACKENDS = ["mock", "claude_api", *_CLI_PRESETS]


def _build_agent(backend: str, cwd: Optional[str], skip_permissions: bool,
                 model: Optional[str], timeout: Optional[float] = None):
    if backend == "mock":
        return MockAgent()
    if backend == "claude_api":
        from .adapters import ClaudeAgent
        return ClaudeAgent(model=model) if model else ClaudeAgent()
    if backend in _CLI_PRESETS:
        kw: dict[str, Any] = {"timeout": timeout}
        if cwd:
            kw["cwd"] = cwd
        if backend != "opencode":  # opencode's run has no skip-permissions flag
            kw["skip_permissions"] = skip_permissions
        if backend == "claude_code" and model:
            kw["model"] = model
        return _CLI_PRESETS[backend](**kw)
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


def _result_dict(result: LoopResult, wt: Optional[Worktree]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "completed": result.completed,
        "iterations": result.iterations,
        "stop_reason": result.stop_reason,
        "final_output": result.final_output,
        "history": [
            {
                "iteration": t.iteration,
                "subgoals": [sg.description for sg in t.subgoals],
                "results": [
                    {"id": r.subgoal.id, "ok": r.ok, "error": r.error} for r in t.results
                ],
                "review": {
                    "gates_passed": t.review.gates_passed,
                    "goal_complete": t.review.goal_complete,
                    "issues": t.review.issues,
                },
            }
            for t in result.history
        ],
    }
    if wt is not None:
        out["worktree"] = {
            "path": wt.path,
            "branch": wt.branch,
            "changed_files": wt.changed_files(),
        }
    return out


def _run_loop_impl(
    goal: str, criteria: str, clarifications: str, *,
    backend: str, cwd: Optional[str], max_iterations: int, max_task_retries: int,
    skip_permissions: bool, isolate: bool, model: Optional[str],
    timeout: Optional[float], max_seconds: Optional[float],
    observer: Optional[Callable[[LoopState], None]],
) -> dict[str, Any]:
    """Run the loop (intake already done) and return a JSON-serializable result."""
    budget = Budget(max_iterations=max_iterations, max_task_retries=max_task_retries,
                    max_seconds=max_seconds)

    def run_in(workdir: Optional[str], wt: Optional[Worktree]) -> dict[str, Any]:
        agent = _build_agent(backend, workdir, skip_permissions, model, timeout)
        orch = Orchestrator(agent, budget=budget, observer=observer)
        return _result_dict(orch.run_loop(goal, criteria, clarifications), wt)

    # A repo + isolation -> run inside a throwaway worktree so workers never touch
    # the caller's checkout. The worktree (and its branch) is kept iff it changed.
    if cwd and isolate:
        with worktree(cwd, cleanup="auto") as wt:
            return run_in(wt.path, wt)
    return run_in(cwd, None)


def orchestrate_impl(
    goal: str,
    success_criteria: str = "",
    *,
    backend: str = "claude_code",
    cwd: Optional[str] = None,
    max_iterations: int = 4,
    max_task_retries: int = 1,
    skip_permissions: bool = False,
    isolate: bool = True,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    max_seconds: Optional[float] = None,
    interaction: Optional[Interaction] = None,
    observer: Optional[Callable[[LoopState], None]] = None,
) -> dict[str, Any]:
    """Intake (clarify) then run the loop. No MCP dependency.

    timeout caps each worker CLI call (None = no per-call cap); max_seconds caps
    the whole run between iterations. Coding workers are slow and unpredictable,
    so prefer bounding the run with max_seconds over a short per-call timeout.

    With the default AutoInteraction this never blocks. Pass a ConsoleInteraction
    for terminal prompts, or a SuspendInteraction to make intake raise NeedInput
    when it needs answers (see `orchestrate_suspendable`).
    """
    interaction = interaction or AutoInteraction()
    # Intake needs an agent but not a worktree (it only asks/plans, never edits).
    intake_orch = Orchestrator(
        _build_agent(backend, cwd, skip_permissions, model, timeout),
        budget=Budget(max_iterations=max_iterations),
        interaction=interaction,
    )
    goal, criteria, clarifications = intake_orch.intake(goal, success_criteria)
    return _run_loop_impl(
        goal, criteria, clarifications, backend=backend, cwd=cwd,
        max_iterations=max_iterations, max_task_retries=max_task_retries,
        skip_permissions=skip_permissions, isolate=isolate, model=model,
        timeout=timeout, max_seconds=max_seconds, observer=observer,
    )


# --- suspend / resume (for stateless surfaces: MCP, scripted CLI) ------------

def _encode_token(data: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def _decode_token(token: str) -> dict[str, Any]:
    return json.loads(base64.urlsafe_b64decode(token.encode()).decode())


def orchestrate_suspendable(
    goal: str, success_criteria: str = "", *, answers: Optional[list[str]] = None, **kw
) -> dict[str, Any]:
    """Like orchestrate_impl, but returns a `needs_input` envelope (with a resume
    token) instead of blocking when the orchestrator needs to ask the user."""
    try:
        return orchestrate_impl(
            goal, success_criteria, interaction=SuspendInteraction(answers), **kw
        )
    except NeedInput as ni:
        return {
            "status": "needs_input",
            "questions": ni.questions,
            "token": _encode_token({
                "goal": ni.goal or goal, "criteria": ni.criteria or success_criteria,
                "questions": ni.questions, **kw,
            }),
        }


def orchestrate_resume_impl(
    token: str, answers: list[str], *,
    observer: Optional[Callable[[LoopState], None]] = None,
) -> dict[str, Any]:
    """Resume a suspended run: fold the answers into context and run the loop.

    Uses the questions cached in the token (no re-clarify, no double agent call).
    """
    data = _decode_token(token)
    questions = data.pop("questions", [])
    goal = data.pop("goal")
    criteria = data.pop("criteria")
    clarifications = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)
    )
    return _run_loop_impl(
        goal, criteria, clarifications, observer=observer,
        backend=data.get("backend", "claude_code"), cwd=data.get("cwd"),
        max_iterations=data.get("max_iterations", 4),
        max_task_retries=data.get("max_task_retries", 1),
        skip_permissions=data.get("skip_permissions", False),
        isolate=data.get("isolate", True), model=data.get("model"),
        timeout=data.get("timeout"), max_seconds=data.get("max_seconds"),
    )


def build_server():
    """Construct the FastMCP server. Imports the MCP SDK lazily."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("agentloop")

    @mcp.tool()
    def orchestrate(
        goal: str,
        success_criteria: str = "",
        backend: str = "claude_code",
        cwd: Optional[str] = None,
        max_iterations: int = 4,
        skip_permissions: bool = False,
        isolate: bool = True,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        """Run an orchestrator -> worker -> reviewer loop until the success
        criteria are met (or a budget guard trips), and return the result.

        The loop first does INTAKE: if it needs to clarify the task, it returns
        { status: "needs_input", questions: [...], token } WITHOUT running — call
        `orchestrate_resume(token, answers)` with the user's answers to continue.
        If no clarification is needed it runs straight through. It then decomposes
        the goal into subgoals, fans them out to `backend` workers, aggregates,
        reviews (quality/consistency/goal-alignment), and loops until complete.

        Args:
            goal: What to achieve.
            success_criteria: How completion is judged. Optional — if omitted the
                orchestrator proposes criteria itself.
            backend: Worker engine — "claude_code" | "codex" | "opencode" |
                "aider" (a coding-agent CLI), "claude_api" (Anthropic SDK), or
                "mock" (deterministic, for testing).
            cwd: Repo to work in. Required for coding tasks that edit files.
            max_iterations: Cap on decompose->review cycles (termination guard).
            skip_permissions: Let CLI workers use tools without prompting. Only
                meaningful with `cwd`; the run is isolated in a worktree.
            isolate: When `cwd` is set, run in a throwaway git worktree/branch so
                the caller's checkout is untouched (recommended).
            model: Optional model override for claude_code / claude_api.
            timeout: Seconds to cap EACH worker CLI call. None (default) = no
                per-call cap. Real coding workers are slow; don't set this low.
            max_seconds: Wall-clock cap on the WHOLE run, checked between
                iterations. Prefer this over `timeout` to bound a long build.

        Returns:
            Either { status: "needs_input", questions[], token } or
            { completed, iterations, stop_reason, final_output, history[],
              worktree?{ path, branch, changed_files } }
        """
        return orchestrate_suspendable(
            goal, success_criteria, backend=backend, cwd=cwd,
            max_iterations=max_iterations, skip_permissions=skip_permissions,
            isolate=isolate, model=model, timeout=timeout, max_seconds=max_seconds,
        )

    @mcp.tool()
    def orchestrate_resume(token: str, answers: list[str]) -> dict[str, Any]:
        """Resume a run that returned `needs_input`.

        Args:
            token: The opaque token from the `needs_input` response.
            answers: One answer per question, in the order they were asked.

        Returns: the same result shape as a completed `orchestrate` call.
        """
        return orchestrate_resume_impl(token, answers)

    @mcp.tool()
    def list_backends() -> dict[str, Any]:
        """List worker backends this server can drive."""
        return {
            "backends": BACKENDS,
            "default": "claude_code",
            "notes": "CLI backends reuse that tool's own login (incl. subscription "
                     "OAuth); claude_api needs ANTHROPIC_API_KEY; mock is for tests.",
        }

    return mcp


def main() -> None:
    build_server().run()  # stdio transport


if __name__ == "__main__":
    main()
