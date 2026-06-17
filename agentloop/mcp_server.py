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
import os
import shutil
import subprocess
import sys
from typing import Any, Callable, Optional

from .adapters import CliAgent, MockAgent
from .interaction import AutoInteraction, Interaction, NeedInput, SuspendInteraction
from .orchestrator import Orchestrator
from .types import Budget, LoopResult, LoopState
from .verifier import CommandVerifier, parse_verify_command
from .worktree import Worktree, worktree

# Worker backends the server can drive. The caller picks one per request.
_CLI_PRESETS = {
    "claude_code": CliAgent.claude_code,
    "codex": CliAgent.codex,
    "opencode": CliAgent.opencode,
    "aider": CliAgent.aider,
}
BACKENDS = ["mock", "claude_api", *_CLI_PRESETS]
RECOMMENDED_MCP_TIMEOUT_MS = 600_000


def _run_version(command: list[str], *, timeout: float = 5) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s"}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}

    output = (proc.stdout or proc.stderr or "").strip()
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "output": output[:500],
    }


def _backend_status(name: str) -> dict[str, Any]:
    if name == "mock":
        return {"available": True, "kind": "in_process", "notes": "deterministic test backend"}
    if name == "claude_api":
        return {
            "available": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "kind": "api",
            "env": "ANTHROPIC_API_KEY",
            "notes": "requires ANTHROPIC_API_KEY in the MCP server environment",
        }

    executable = {
        "claude_code": "claude",
        "codex": "codex",
        "opencode": "opencode",
        "aider": "aider",
    }[name]
    path = shutil.which(executable)
    status: dict[str, Any] = {
        "available": path is not None,
        "kind": "cli",
        "executable": executable,
        "path": path,
    }
    if path:
        status["version"] = _run_version([executable, "--version"])
    else:
        status["error"] = f"{executable!r} not found on PATH"
    return status


def doctor_impl(cwd: Optional[str] = None) -> dict[str, Any]:
    """Return non-invasive diagnostics for MCP hosts and worker backends."""
    cwd_status: dict[str, Any] = {"path": cwd, "provided": cwd is not None}
    if cwd:
        cwd_status.update({
            "exists": os.path.isdir(cwd),
            "readable": os.access(cwd, os.R_OK),
            "writable": os.access(cwd, os.W_OK),
        })

    backends = {name: _backend_status(name) for name in BACKENDS}
    recommendations = [
        "If an MCP client reports -32001 Request timed out, first check the host/client MCP request timeout; it does not prove the worker backend failed to spawn.",
        f"For long orchestrate runs, configure the host MCP request timeout to at least {RECOMMENDED_MCP_TIMEOUT_MS} ms.",
        "Set orchestrate(timeout=...) for a hard cap on each CLI worker call; timeout=None means no per-call cap.",
        "max_seconds is a loop budget checked between phases/iterations; it is not a hard kill for one stuck CLI subprocess.",
        "Use skip_permissions=true for trusted, isolated headless coding runs that need tool access.",
    ]
    if cwd and not cwd_status["exists"]:
        recommendations.insert(0, "The supplied cwd does not exist; worker CLIs cannot run there.")

    return {
        "ok": True,
        "server": {
            "python": sys.executable,
            "package_dir": os.path.dirname(__file__),
            "path": os.environ.get("PATH", ""),
        },
        "cwd": cwd_status,
        "backends": backends,
        "timeouts": {
            "recommended_mcp_request_timeout_ms": RECOMMENDED_MCP_TIMEOUT_MS,
            "mcp_request_timeout": "Configured in the host agent (Claude/OpenCode/Codex/etc.); this server cannot see or override it.",
            "worker_timeout": "orchestrate(timeout=seconds) caps each individual CLI worker subprocess; None disables this cap.",
            "max_seconds": "orchestrate(max_seconds=seconds) is a cooperative loop budget checked between phases/iterations, not during one blocking subprocess.run call.",
        },
        "recommendations": recommendations,
    }


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
        kw["skip_permissions"] = skip_permissions
        if backend == "claude_code" and model:
            kw["model"] = model
        return _CLI_PRESETS[backend](**kw)
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


def _summary(out: dict[str, Any]) -> str:
    """One human-readable line summarizing a finished run (for the tool result)."""
    verb = "completed" if out["completed"] else "stopped"
    parts = [f"{verb} in {out['iterations']} iteration(s)", f"reason: {out['stop_reason']}"]
    wt = out.get("worktree")
    if wt:
        n = len(wt["changed_files"])
        parts.append(f"worktree {wt['branch']} ({n} file{'' if n == 1 else 's'} changed)")
    return " · ".join(parts)


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
                "verification": [
                    {
                        "name": v.name,
                        "ok": v.ok,
                        "exit_code": v.exit_code,
                        "stdout": v.stdout,
                        "stderr": v.stderr,
                        "error": v.error,
                        "duration": v.duration,
                    }
                    for v in t.verification
                ],
            }
            for t in result.history
        ],
    }
    if wt is not None:
        out["worktree"] = {
            "path": wt.path,
            "branch": wt.branch,
            "changed_files": wt.changed_files(),
            "checkpoints": wt.commits(),  # per-iteration commits; work survives a failure
        }
    out["summary"] = _summary(out)
    return out


def _error_result(exc: Exception, *, stage: str) -> dict[str, Any]:
    """Return a normal tool result for backend/runtime failures.

    MCP callers need actionable JSON back instead of a torn-down request when a
    worker CLI is unavailable, rate-limited, misconfigured, or times out.
    """
    detail = f"{type(exc).__name__}: {exc}"
    out: dict[str, Any] = {
        "completed": False,
        "iterations": 0,
        "stop_reason": f"{stage}_agent_error",
        "final_output": detail,
        "history": [],
        "error": detail,
    }
    out["summary"] = _summary(out)
    return out


def _run_loop_impl(
    goal: str, criteria: str, clarifications: str, *,
    backend: str, cwd: Optional[str], max_iterations: int, max_task_retries: int,
    skip_permissions: bool, isolate: bool, model: Optional[str],
    timeout: Optional[float], max_seconds: Optional[float],
    verify_commands: Optional[list[str]], verify_timeout: Optional[float],
    observer: Optional[Callable[[LoopState], None]],
) -> dict[str, Any]:
    """Run the loop (intake already done) and return a JSON-serializable result."""
    budget = Budget(max_iterations=max_iterations, max_task_retries=max_task_retries,
                    max_seconds=max_seconds)

    def run_in(workdir: Optional[str], wt: Optional[Worktree]) -> dict[str, Any]:
        agent = _build_agent(backend, workdir, skip_permissions, model, timeout)
        verifier = None
        if verify_commands:
            verifier = CommandVerifier(
                [parse_verify_command(cmd, timeout=verify_timeout) for cmd in verify_commands],
                cwd=workdir,
            )
        # In a worktree, commit after each iteration so partial work is durable
        # even if a later iteration fails or a budget guard stops the run.
        checkpoint = None
        if wt is not None:
            checkpoint = lambda st: wt.commit(  # noqa: E731
                f"agentloop: iteration {st.iteration}")
        orch = Orchestrator(
            agent, budget=budget, observer=observer, checkpoint=checkpoint,
            verifier=verifier,
        )
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
    verify_commands: Optional[list[str]] = None,
    verify_timeout: Optional[float] = None,
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
    # Build once up front so programmer/input errors like an unknown backend still
    # fail loudly, while runtime CLI/API failures below are returned as JSON.
    intake_agent = _build_agent(backend, cwd, skip_permissions, model, timeout)
    try:
        # Intake needs an agent but not a worktree (it only asks/plans, never edits).
        intake_orch = Orchestrator(
            intake_agent,
            budget=Budget(max_iterations=max_iterations),
            interaction=interaction,
        )
        goal, criteria, clarifications = intake_orch.intake(goal, success_criteria)
    except NeedInput:
        raise
    except Exception as exc:  # noqa: BLE001 - preserve MCP response shape on backend failure
        return _error_result(exc, stage="intake")

    try:
        return _run_loop_impl(
            goal, criteria, clarifications, backend=backend, cwd=cwd,
            max_iterations=max_iterations, max_task_retries=max_task_retries,
            skip_permissions=skip_permissions, isolate=isolate, model=model,
            timeout=timeout, max_seconds=max_seconds, verify_commands=verify_commands,
            verify_timeout=verify_timeout, observer=observer,
        )
    except Exception as exc:  # noqa: BLE001 - preserve MCP response shape on backend failure
        return _error_result(exc, stage="loop")


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
        verify_commands=data.get("verify_commands"),
        verify_timeout=data.get("verify_timeout"),
    )


# --- live progress (so the MCP client shows more than a spinner) -------------

def _progress_observer(report: Callable[[int, int, str], None]) -> Callable[[LoopState], None]:
    """Adapt a `report(progress, total, message)` callback into a LoopState observer.

    Fires once per completed iteration with a human-readable status line — what
    the calling agent surfaces under its tool-call spinner.
    """
    def observer(state: LoopState) -> None:
        t = state.history[-1]
        ok = sum(r.ok for r in t.results)
        total = state.budget.max_iterations
        report(
            t.iteration, total,
            f"iteration {t.iteration}/{total}: {ok}/{len(t.results)} subgoals ok, "
            f"verification {'pass' if all(v.ok for v in t.verification) else 'fail'}, "
            f"gates {'pass' if t.review.gates_passed else 'fail'}, "
            f"goal {'complete' if t.review.goal_complete else 'incomplete'}",
        )
    return observer


async def _run_with_progress(
    ctx,
    thunk: Callable[[Optional[Callable]], dict[str, Any]],
    *,
    start_message: str = "starting long-running orchestration",
):
    """Run a blocking `thunk(observer)` off the event loop, relaying each
    iteration to the MCP client via `ctx.report_progress`.

    The loop is synchronous and slow, so it runs in a worker thread; the observer
    bridges back to the event loop to emit `notifications/progress`. `ctx` may be
    None (e.g. a client that injects no context) — then it just runs, no progress.
    """
    import anyio

    if ctx is None:
        return await anyio.to_thread.run_sync(lambda: thunk(None))

    async def _emit(progress: int, total: int, message: str) -> None:
        try:
            await ctx.report_progress(progress=progress, total=total, message=message)
        except Exception:
            pass  # progress is best-effort; never fail a run over a notification

    await _emit(0, 1, start_message)

    def report(progress: int, total: int, message: str) -> None:
        anyio.from_thread.run(_emit, progress, total, message)

    return await anyio.to_thread.run_sync(lambda: thunk(_progress_observer(report)))


def build_server():
    """Construct the FastMCP server. Imports the MCP SDK lazily."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("agentloop")

    def _ctx():
        # The live request's Context (for progress notifications), or None if we
        # somehow run outside a request. Kept out of the tool signature so it
        # never shows up in the tool's input schema.
        try:
            return mcp.get_context()
        except Exception:
            return None

    @mcp.tool()
    async def orchestrate(
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
        verify_commands: Optional[list[str]] = None,
        verify_timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Run an orchestrator -> worker -> reviewer loop until the success
        criteria are met (or a budget guard trips), and return the result.

        Important timeout contract for MCP callers: this is a long-running,
        synchronous tool. If your host reports MCP error -32001 "Request timed
        out", first check the host/client MCP request timeout. That error often
        means the host stopped waiting before this tool returned; it does not by
        itself prove the worker backend failed to spawn. Run doctor() to check
        CLI/backend availability before making that diagnosis.

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
            timeout: Seconds to cap EACH worker CLI subprocess call. None
                (default) = no hard per-call cap. Set this when you need stuck
                worker calls to fail cleanly instead of relying on MCP request
                timeouts.
            max_seconds: Wall-clock cap on the WHOLE run, checked between
                phases/iterations. This is cooperative and does not interrupt a
                single blocking worker subprocess call.
            verify_commands: Optional real commands to run after each worker
                iteration, e.g. ["python3 -m pytest", "npx playwright test"].
            verify_timeout: Optional seconds cap for each verification command.

        Streams a `notifications/progress` update when starting and per iteration
        as it runs, so the calling agent can show live status instead of a bare
        spinner. Some hosts still enforce a separate MCP request timeout even
        when progress notifications are emitted; configure that host timeout to
        a large value (recommended: 600000 ms or more) for real coding runs.

        Returns:
            Either { status: "needs_input", questions[], token } or
            { completed, iterations, stop_reason, final_output, summary,
              history[], worktree?{ path, branch, changed_files } }
        """
        return await _run_with_progress(_ctx(), lambda observer: orchestrate_suspendable(
            goal, success_criteria, backend=backend, cwd=cwd,
            max_iterations=max_iterations, skip_permissions=skip_permissions,
            isolate=isolate, model=model, timeout=timeout, max_seconds=max_seconds,
            verify_commands=verify_commands, verify_timeout=verify_timeout,
            observer=observer,
        ), start_message=(
            "starting orchestration: if this later fails with MCP -32001, check the "
            "host MCP request timeout before blaming worker backend spawning"
        ))

    @mcp.tool()
    async def orchestrate_resume(token: str, answers: list[str]) -> dict[str, Any]:
        """Resume a run that returned `needs_input`.

        Args:
            token: The opaque token from the `needs_input` response.
            answers: One answer per question, in the order they were asked.

        Returns: the same result shape as a completed `orchestrate` call (and
        streams per-iteration progress the same way).
        """
        return await _run_with_progress(
            _ctx(), lambda observer: orchestrate_resume_impl(token, answers, observer=observer),
            start_message="resuming orchestration from answers",
        )

    @mcp.tool()
    def list_backends() -> dict[str, Any]:
        """List worker backends this server can drive."""
        return {
            "backends": BACKENDS,
            "default": "claude_code",
            "notes": "CLI backends reuse that tool's own login (incl. subscription "
                     "OAuth); claude_api needs ANTHROPIC_API_KEY; mock is for tests. "
                     "Run doctor() before guessing about backend availability.",
            "timeouts": {
                "recommended_mcp_request_timeout_ms": RECOMMENDED_MCP_TIMEOUT_MS,
                "worker_timeout": "orchestrate(timeout=seconds) caps each CLI worker call",
                "max_seconds": "cooperative loop budget, not a hard subprocess kill",
            },
        }

    @mcp.tool()
    def doctor(cwd: Optional[str] = None) -> dict[str, Any]:
        """Diagnose this MCP server and worker backend availability without running agents.

        Use this before concluding that `orchestrate` cannot spawn backends. In
        particular, MCP error -32001 usually means the host agent's MCP request
        timeout expired while this long-running tool was still working; it does
        not by itself prove Claude Code, Codex, opencode, or aider failed.

        Args:
            cwd: Optional target workspace to check for existence/read/write access.

        Returns CLI presence/version checks, cwd access, and explicit timeout
        guidance distinguishing host MCP request timeout, worker `timeout`, and
        cooperative `max_seconds`.
        """
        return doctor_impl(cwd)

    return mcp


def main() -> None:
    build_server().run()  # stdio transport


if __name__ == "__main__":
    main()
