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

from typing import Any, Callable, Optional

from .adapters import CliAgent, MockAgent
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
                 model: Optional[str]):
    if backend == "mock":
        return MockAgent()
    if backend == "claude_api":
        from .adapters import ClaudeAgent
        return ClaudeAgent(model=model) if model else ClaudeAgent()
    if backend in _CLI_PRESETS:
        kw: dict[str, Any] = {}
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


def orchestrate_impl(
    goal: str,
    success_criteria: str,
    *,
    backend: str = "claude_code",
    cwd: Optional[str] = None,
    max_iterations: int = 4,
    max_task_retries: int = 1,
    skip_permissions: bool = False,
    isolate: bool = True,
    model: Optional[str] = None,
    observer: Optional[Callable[[LoopState], None]] = None,
) -> dict[str, Any]:
    """Run the loop and return a JSON-serializable result. No MCP dependency."""
    budget = Budget(max_iterations=max_iterations, max_task_retries=max_task_retries)

    def run_in(workdir: Optional[str], wt: Optional[Worktree]) -> dict[str, Any]:
        agent = _build_agent(backend, workdir, skip_permissions, model)
        orch = Orchestrator(agent, budget=budget, observer=observer)
        return _result_dict(orch.run(goal, success_criteria), wt)

    # A repo + isolation -> run inside a throwaway worktree so workers never touch
    # the caller's checkout. The worktree (and its branch) is kept iff it changed.
    if cwd and isolate:
        with worktree(cwd, cleanup="auto") as wt:
            return run_in(wt.path, wt)
    return run_in(cwd, None)


def build_server():
    """Construct the FastMCP server. Imports the MCP SDK lazily."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("agentloop")

    @mcp.tool()
    def orchestrate(
        goal: str,
        success_criteria: str,
        backend: str = "claude_code",
        cwd: Optional[str] = None,
        max_iterations: int = 4,
        skip_permissions: bool = False,
        isolate: bool = True,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run an orchestrator -> worker -> reviewer loop until the success
        criteria are met (or a budget guard trips), and return the result.

        The loop decomposes the goal into subgoals, fans them out to `backend`
        workers, aggregates, reviews (quality/consistency/goal-alignment), and
        loops with feedback until complete.

        Args:
            goal: What to achieve.
            success_criteria: How completion is judged (be concrete/checkable).
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

        Returns:
            { completed, iterations, stop_reason, final_output, history[],
              worktree?{ path, branch, changed_files } }
        """
        return orchestrate_impl(
            goal, success_criteria, backend=backend, cwd=cwd,
            max_iterations=max_iterations, skip_permissions=skip_permissions,
            isolate=isolate, model=model,
        )

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
