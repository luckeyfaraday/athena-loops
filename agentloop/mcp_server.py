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
from .runs import MANAGER, RunHandle, run_base
from .types import (
    EVENT_INTAKE_FINISHED,
    EVENT_INTAKE_STARTED,
    EVENT_NEEDS_INPUT,
    Budget,
    LoopResult,
    LoopState,
)
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

# The verify command auto-added when a run sets playwright=true. Kept timeout-free
# by default (browser launch is slow); pass verify_timeout only if you must.
PLAYWRIGHT_VERIFY_COMMAND = "npx playwright test"


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
        "orchestrate is detached by default: it returns a run_id immediately and "
        "runs until it finishes or errors. Monitor with orchestrate_status / "
        "orchestrate_tail / orchestrate_result — do not impose a run timeout.",
        "There is no wall-clock cap on a run. Prefer monitoring to completion over "
        "capping; timeout/max_seconds are optional hard kills (default off), not "
        "part of the normal flow — leave them unset for real coding runs.",
        "MCP error -32001 Request timed out only affects the opt-in blocking mode "
        "(detach=false), where one request spans the whole run. The default "
        "detached flow returns at once and cannot hit it; switch to it.",
        f"If you deliberately use detach=false, raise the host MCP request timeout "
        f"to at least {RECOMMENDED_MCP_TIMEOUT_MS} ms; detached poll calls return "
        "fast and need no such bump.",
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
            "model": "orchestrate is detached by default and returns immediately; "
                     "monitor the run_id to completion. There is no run timeout.",
            "blocking_mode": "Only orchestrate(detach=false) holds one MCP request "
                             "open for the whole run; that mode alone is bound by "
                             "the host MCP request timeout (which this server cannot "
                             "see or override).",
            "worker_timeout": "orchestrate(timeout=seconds) optionally caps each CLI "
                              "worker subprocess; None (default) = no cap.",
            "max_seconds": "orchestrate(max_seconds=seconds) is an optional cooperative "
                           "loop budget checked between phases; None (default) = no cap.",
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
    emit: Optional[Callable[[str, int, dict], None]] = None,
    playwright: bool = False,
) -> dict[str, Any]:
    """Run the loop (intake already done) and return a JSON-serializable result."""
    budget = Budget(max_iterations=max_iterations, max_task_retries=max_task_retries,
                    max_seconds=max_seconds)
    # `playwright` is one switch for both halves of browser-level testing: it adds
    # the Playwright suite as a verify gate AND turns on the matching prompt
    # guidance in the orchestrator (subagent writes tests, reviewer requires them).
    commands = list(verify_commands or [])
    if playwright and PLAYWRIGHT_VERIFY_COMMAND not in commands:
        commands.append(PLAYWRIGHT_VERIFY_COMMAND)

    def run_in(workdir: Optional[str], wt: Optional[Worktree]) -> dict[str, Any]:
        agent = _build_agent(backend, workdir, skip_permissions, model, timeout)
        verifier = None
        if commands:
            verifier = CommandVerifier(
                [parse_verify_command(cmd, timeout=verify_timeout) for cmd in commands],
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
            verifier=verifier, emit=emit, playwright=playwright,
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
    playwright: bool = False,
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
            verify_timeout=verify_timeout, playwright=playwright, observer=observer,
        )
    except Exception as exc:  # noqa: BLE001 - preserve MCP response shape on backend failure
        return _error_result(exc, stage="loop")


# --- detached runs (the default: start instantly, then monitor to completion) -
#
# A detached run owns its own thread for EVERYTHING it does — intake included.
# Nothing here may run a worker on the caller's thread: the MCP `orchestrate`
# tool is async, so a blocking subprocess on that thread would freeze the whole
# server (missed pings -> the host drops the connection, not just one request).
# So `orchestrate_start_impl` returns a run_id in milliseconds and the loop,
# including any slow or clarifying intake, happens in the background where it is
# observed through status/tail/result — never by holding the start call open.

def _short(text: str, limit: int = 200) -> str:
    """A compact, event-friendly excerpt of goal/criteria text."""
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _loop_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
    """Extract the run_loop knobs from a raw orchestrate kwarg bag (with defaults)."""
    return dict(
        backend=kw.get("backend", "claude_code"), cwd=kw.get("cwd"),
        max_iterations=kw.get("max_iterations", 4),
        max_task_retries=kw.get("max_task_retries", 1),
        skip_permissions=kw.get("skip_permissions", False),
        isolate=kw.get("isolate", True), model=kw.get("model"),
        timeout=kw.get("timeout"), max_seconds=kw.get("max_seconds"),
        verify_commands=kw.get("verify_commands"),
        verify_timeout=kw.get("verify_timeout"),
        playwright=kw.get("playwright", False),
    )


def _meta(goal: str, criteria: str, kw: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal": goal, "success_criteria": criteria,
        "backend": kw.get("backend", "claude_code"), "cwd": kw.get("cwd"),
        "max_iterations": kw.get("max_iterations", 4),
        "isolate": kw.get("isolate", True),
    }


def _intake(goal: str, success_criteria: str, kw: dict[str, Any],
            interaction: Interaction) -> tuple[str, str, str]:
    """Run intake against a freshly built agent. May raise NeedInput."""
    agent = _build_agent(
        kw.get("backend", "claude_code"), kw.get("cwd"),
        kw.get("skip_permissions", False), kw.get("model"), kw.get("timeout"),
    )
    orch = Orchestrator(
        agent, budget=Budget(max_iterations=kw.get("max_iterations", 4)),
        interaction=interaction,
    )
    return orch.intake(goal, success_criteria)


def _needs_input_result(questions: list[str], token: str) -> dict[str, Any]:
    """A terminal run result meaning intake paused for clarification.

    Detached runs never block the start call to ask; the pause surfaces here, via
    orchestrate_status / orchestrate_result, and is resumed with
    orchestrate_resume(token, answers, detach=true).
    """
    out: dict[str, Any] = {
        "completed": False, "iterations": 0, "stop_reason": "needs_input",
        "status": "needs_input", "questions": list(questions), "token": token,
        "final_output": "", "history": [],
    }
    out["summary"] = "needs input: " + " | ".join(questions)
    return out


def _running_envelope(handle: RunHandle) -> dict[str, Any]:
    """What a detached start returns: the run_id plus how to monitor it."""
    events_path = os.path.join(handle.run_dir, "events.jsonl")
    return {
        "status": "running",
        "run_id": handle.run_id,
        "run_dir": handle.run_dir,
        "events_path": events_path,
        "tail_command": f"tail -f {events_path}",
        "message": (
            "Orchestration started in the background and runs until it finishes or "
            "errors — there is no run timeout. Monitor it: poll "
            "orchestrate_status(run_id) until running is false, then "
            "orchestrate_result(run_id) for the outcome (or orchestrate_tail("
            "run_id, cursor) to watch each step live). A human can `tail -f` the "
            "events file. If the result's stop_reason is 'needs_input', read its "
            "questions + token and call orchestrate_resume(token, answers)."
        ),
    }


def _launch(goal: str, criteria: str, clarifications: str, *,
            kw: dict[str, Any], base_dir: Optional[str]) -> dict[str, Any]:
    """Start an already-intaken loop on a background thread; return its run_id."""
    def thunk(emit: Callable[[str, int, dict], None]) -> dict[str, Any]:
        try:
            return _run_loop_impl(goal, criteria, clarifications, emit=emit,
                                  observer=None, **_loop_kwargs(kw))
        except Exception as exc:  # noqa: BLE001 — a failed result, not a torn thread
            return _error_result(exc, stage="loop")

    handle = MANAGER.start(thunk=thunk, meta=_meta(goal, criteria, kw),
                           base=base_dir or run_base(kw.get("cwd")))
    return _running_envelope(handle)


def orchestrate_start_impl(
    goal: str, success_criteria: str = "", *,
    base_dir: Optional[str] = None, answers: Optional[list[str]] = None, **kw: Any,
) -> dict[str, Any]:
    """Detached `orchestrate`: return a run_id immediately; intake + loop run in
    the background. Never blocks the caller to clarify, and never runs a worker on
    the calling thread (which, for the async MCP tool, would freeze the server).

    Returns { status: "running", run_id, run_dir, events_path } right away. Watch
    it with orchestrate_status / orchestrate_tail; fetch the outcome with
    orchestrate_result(run_id). If intake needs answers, the run finishes with
    stop_reason "needs_input" (questions + a resume token in its result) — resume
    with orchestrate_resume(token, answers, detach=true). Unknown backends raise
    loudly here, before anything starts.
    """
    backend = kw.get("backend", "claude_code")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")

    def thunk(emit: Callable[[str, int, dict], None]) -> dict[str, Any]:
        # Intake runs HERE, on the run's own thread, so the start call returns
        # instantly and a slow or clarifying intake can never time out the caller.
        emit(EVENT_INTAKE_STARTED, 0, {"goal": _short(goal)})
        try:
            g, criteria, clarifications = _intake(
                goal, success_criteria, kw, SuspendInteraction(answers))
        except NeedInput as ni:
            token = _encode_token({
                "goal": ni.goal or goal, "criteria": ni.criteria or success_criteria,
                "questions": ni.questions, **kw,
            })
            emit(EVENT_NEEDS_INPUT, 0, {"questions": ni.questions})
            return _needs_input_result(ni.questions, token)
        except Exception as exc:  # noqa: BLE001 — a failed result, not a torn thread
            return _error_result(exc, stage="intake")
        emit(EVENT_INTAKE_FINISHED, 0, {
            "success_criteria": _short(criteria),
            "has_clarifications": bool(clarifications),
        })
        try:
            return _run_loop_impl(g, criteria, clarifications, emit=emit,
                                  observer=None, **_loop_kwargs(kw))
        except Exception as exc:  # noqa: BLE001
            return _error_result(exc, stage="loop")

    handle = MANAGER.start(thunk=thunk, meta=_meta(goal, success_criteria, kw),
                           base=base_dir or run_base(kw.get("cwd")))
    return _running_envelope(handle)


def orchestrate_status_impl(run_id: str) -> dict[str, Any]:
    """Light status for a detached run: phase, iteration, running, event count."""
    return MANAGER.status(run_id)


def orchestrate_tail_impl(run_id: str, cursor: int = 0, limit: int = 200) -> dict[str, Any]:
    """Events for a detached run with seq > cursor; pass back `cursor` to continue."""
    return MANAGER.tail(run_id, cursor, limit)


def orchestrate_result_impl(
    run_id: str, wait: bool = False, timeout: Optional[float] = None
) -> dict[str, Any]:
    """Final result of a detached run, or a running-status dict if not done yet."""
    return MANAGER.result(run_id, wait, timeout)


def orchestrate_list_impl() -> dict[str, Any]:
    """Status of every detached run this server has started."""
    return MANAGER.list_runs()


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
    detach: bool = False, base_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Resume a suspended run: fold the answers into context and run the loop.

    Uses the questions cached in the token (no re-clarify, no double agent call).
    With detach=true the resumed loop runs in the background and returns a run_id.
    """
    data = _decode_token(token)
    questions = data.pop("questions", [])
    goal = data.pop("goal")
    criteria = data.pop("criteria")
    clarifications = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)
    )
    if detach:
        return _launch(goal, criteria, clarifications, kw=data, base_dir=base_dir)
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
        playwright=data.get("playwright", False),
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
        playwright: bool = False,
        detach: bool = True,
    ) -> dict[str, Any]:
        """Run an orchestrator -> worker -> reviewer loop until the success
        criteria are met (or a budget guard trips).

        How to drive this (the default, detach=true): this call returns
        IMMEDIATELY with { status: "running", run_id }. The loop — intake, then
        decompose -> fan out to `backend` workers -> aggregate -> review -> repeat
        — keeps running in the background until it finishes or errors. There is NO
        run timeout. You watch it to completion:

            1. poll orchestrate_status(run_id) until "running" is false
               (or stream steps live with orchestrate_tail(run_id, cursor));
            2. then call orchestrate_result(run_id) for the outcome.

        Do not set a wall-clock cap to "be safe" — coding runs are legitimately
        long and unpredictable; monitor them, don't guillotine them. `timeout` and
        `max_seconds` exist only as optional hard kills and default to off.

        Because the start call returns at once, MCP error -32001 "Request timed
        out" does not happen on the default path. It can only happen if you opt
        into the blocking mode (detach=false), where one MCP request is held open
        for the entire run; that mode alone is bound by the host MCP request
        timeout. Prefer the default and monitor.

        Intake may pause to clarify the task. When it does, the run finishes with
        stop_reason "needs_input" and its result carries { questions[], token };
        gather answers and call orchestrate_resume(token, answers) to continue.

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
            timeout: OPTIONAL seconds to cap EACH worker CLI subprocess call. None
                (default) = no per-call cap. Leave unset for normal runs; set it
                only to force a genuinely stuck worker to fail instead of hanging.
            max_seconds: OPTIONAL wall-clock cap on the whole run, checked between
                phases. None (default) = no cap. Leave unset; the run is monitored
                to completion, not raced against a clock.
            verify_commands: Optional real commands to run after each worker
                iteration, e.g. ["python3 -m pytest", "npx playwright test"].
            verify_timeout: Optional seconds cap for each verification command.
            playwright: When true, encourage browser-level testing for web/UI
                work in one switch: adds "npx playwright test" as a verify gate
                AND tells subagents to write/extend Playwright tests and the
                reviewer to require passing ones before completing. Leave
                verify_timeout unset — browser launch is slow.
            detach: Default true — start in the background and return a run_id to
                monitor (the recommended flow above). Set false only to hold one
                MCP request open and get the final result dict directly; that mode
                streams per-iteration progress notifications but is subject to the
                host MCP request timeout, so it suits short/mock runs.

        Returns:
            detach=true (default): { status: "running", run_id, run_dir,
              events_path, message } — monitor with status/tail/result.
            detach=false: { status: "needs_input", questions[], token } or
              { completed, iterations, stop_reason, final_output, summary,
                history[], worktree?{ path, branch, changed_files } }.
        """
        if detach:
            return orchestrate_start_impl(
                goal, success_criteria, backend=backend, cwd=cwd,
                max_iterations=max_iterations, skip_permissions=skip_permissions,
                isolate=isolate, model=model, timeout=timeout, max_seconds=max_seconds,
                verify_commands=verify_commands, verify_timeout=verify_timeout,
                playwright=playwright,
            )
        return await _run_with_progress(_ctx(), lambda observer: orchestrate_suspendable(
            goal, success_criteria, backend=backend, cwd=cwd,
            max_iterations=max_iterations, skip_permissions=skip_permissions,
            isolate=isolate, model=model, timeout=timeout, max_seconds=max_seconds,
            verify_commands=verify_commands, verify_timeout=verify_timeout,
            playwright=playwright, observer=observer,
        ), start_message="starting blocking orchestration (detach=false); monitor "
           "via detach=true next time if the host request times out")

    @mcp.tool()
    async def orchestrate_resume(
        token: str, answers: list[str], detach: bool = True
    ) -> dict[str, Any]:
        """Resume a run whose intake paused with stop_reason "needs_input".

        Args:
            token: The opaque token from the needs_input result.
            answers: One answer per question, in the order they were asked.
            detach: Default true — run the resumed loop in the background and
                return a run_id to monitor (poll orchestrate_status until done,
                then orchestrate_result), with no run timeout. Set false to hold
                the MCP request open for the whole run and get the result dict
                directly (subject to the host request timeout).

        Returns: detach=true -> { status: "running", run_id, ... } to monitor;
        detach=false -> the same result shape as a completed `orchestrate` call.
        """
        if detach:
            return orchestrate_resume_impl(token, answers, detach=True)
        return await _run_with_progress(
            _ctx(), lambda observer: orchestrate_resume_impl(token, answers, observer=observer),
            start_message="resuming blocking orchestration from answers",
        )

    @mcp.tool()
    def orchestrate_status(run_id: str) -> dict[str, Any]:
        """Light status of a detached run: phase, current iteration, whether it is
        still running, and how many events it has produced. Cheap to poll often.

        Args:
            run_id: The id returned by orchestrate(detach=true).
        """
        return orchestrate_status_impl(run_id)

    @mcp.tool()
    def orchestrate_tail(run_id: str, cursor: int = 0, limit: int = 200) -> dict[str, Any]:
        """Read what the loop has done since `cursor` — the live window into a
        detached run: decomposed subgoals, each worker's start/finish and output
        preview, verification, and the reviewer's verdict.

        Args:
            run_id: The id returned by orchestrate(detach=true).
            cursor: Return only events with seq greater than this (0 = from start).
            limit: Max events to return; pass the returned `cursor` back to page on.

        Returns { events[], cursor, running, more }. Full worker output for each
        finished subagent is at data.output_path (also under <run_dir>/workers/).
        """
        return orchestrate_tail_impl(run_id, cursor, limit)

    @mcp.tool()
    def orchestrate_result(
        run_id: str, wait: bool = False, timeout: Optional[float] = None
    ) -> dict[str, Any]:
        """Fetch the final result of a detached run.

        Preferred monitoring: poll orchestrate_status(run_id) (returns instantly)
        until running is false, THEN call this with wait=false. Using wait=true
        with timeout=None (or a large timeout) holds this MCP request open for the
        rest of the run — which is exactly the blocking behavior detached mode
        avoids, and can hit the host request timeout. If you do wait, use a short
        timeout and call again; a still-running run returns { status: "running" }.

        Args:
            run_id: The id returned by orchestrate(detach=true).
            wait: If true, block up to `timeout` seconds for the run to finish.
            timeout: Seconds to wait when wait=true (None = until it finishes —
                avoid; this re-blocks the request for the whole run).

        Returns the same result shape as a blocking `orchestrate` call once done
        (including stop_reason "needs_input" with questions + token if intake
        paused), or a { status: "running", ... } dict if it is still going.
        """
        return orchestrate_result_impl(run_id, wait, timeout)

    @mcp.tool()
    def orchestrate_list() -> dict[str, Any]:
        """List every detached run this server has started, with its status."""
        return orchestrate_list_impl()

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
                "model": "orchestrate is detached by default; it returns a run_id "
                         "at once and is monitored to completion with no run timeout.",
                "worker_timeout": "orchestrate(timeout=seconds) optionally caps each "
                                  "CLI worker call (default off)",
                "max_seconds": "optional cooperative loop budget (default off), not a "
                               "hard subprocess kill",
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
