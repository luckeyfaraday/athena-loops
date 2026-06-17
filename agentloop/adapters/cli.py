"""Use any headless coding-agent CLI as an agentloop backend.

This is the "inward" plug: instead of calling an LLM API directly, each role
(decomposer / subagent / reviewer) is executed by whatever coding agent you have
installed — Claude Code, Codex, opencode, Aider — so the workers get that agent's
real tools, file access, and repo context. Nothing in the loop changes; this just
implements the same `Agent` seam by shelling out.

A non-zero exit or timeout raises, which the scheduler turns into a FAILED
TaskResult (with retries) rather than a silent wrong answer.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Callable, Optional, Sequence

from ..agent import Agent, AgentRequest, AgentResponse

# Placeholders substituted into the command template (per-arg, plain string replace
# so JSON braces in prompts are never touched):
#   {prompt}    -> the task/question
#   {system}    -> the role system prompt
#   {combined}  -> system + "\n\n" + prompt  (for CLIs with no system-prompt flag)
_PROMPT_KEYS = ("{prompt}", "{combined}")


def _parse_claude_json(out: str) -> str:
    data = json.loads(out)
    if data.get("is_error"):
        status = data.get("api_error_status")
        result = str(data.get("result") or "").strip()
        detail = f"status {status}: {result}" if status else result
        raise RuntimeError(f"Claude Code reported an error: {detail}")
    return data["result"]


class CliAgent(Agent):
    def __init__(
        self,
        command: Sequence[str],
        *,
        parse_output: Optional[Callable[[str], str]] = None,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        extra_env: Optional[dict[str, str]] = None,
    ):
        # timeout=None means no per-worker cap: a coding agent's runtime is
        # unpredictable, so the right global guard is Budget.max_seconds on the
        # whole run, not an arbitrary per-call limit. Set a number to cap a call.
        self.command = list(command)
        self.parse_output = parse_output
        self.timeout = timeout
        self.cwd = cwd
        self.extra_env = extra_env
        # If no prompt placeholder appears in the template, feed the text on stdin.
        self._use_stdin = not any(
            key in arg for arg in self.command for key in _PROMPT_KEYS
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        combined = (
            f"{request.system}\n\n{request.prompt}" if request.system else request.prompt
        )
        subs = {"{prompt}": request.prompt, "{system}": request.system, "{combined}": combined}
        argv = [self._sub(arg, subs) for arg in self.command]

        env = {**os.environ, **self.extra_env} if self.extra_env else None
        run_kw: dict[str, object] = {
            "capture_output": True,
            "text": True,
            "timeout": self.timeout,
            "cwd": self.cwd,
            "env": env,
        }
        if self._use_stdin:
            run_kw["input"] = combined
        else:
            # Never let nested CLIs inherit the MCP server's JSON-RPC stdin.
            run_kw["stdin"] = subprocess.DEVNULL
        try:
            proc = subprocess.run(argv, **run_kw)
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "")
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            hint = f" (partial output: {partial.strip()[:200]!r})" if partial.strip() else ""
            raise RuntimeError(
                f"CLI agent timed out after {self.timeout}s: {argv[0]}{hint}"
            ) from exc

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()[:500]
            raise RuntimeError(f"CLI agent exited {proc.returncode}: {err}")

        raw = proc.stdout.strip()
        text = self.parse_output(raw) if self.parse_output else raw
        return AgentResponse(text=text, raw=proc)

    @staticmethod
    def _sub(arg: str, subs: dict[str, str]) -> str:
        for key, val in subs.items():
            arg = arg.replace(key, val)
        return arg

    # --- presets -----------------------------------------------------------
    # Starting templates for common coding agents. CLI flags change between
    # versions — confirm against your installed agent and adjust as needed.
    #
    # Two knobs matter for autonomous coding workers:
    #   cwd=...            -> run the worker in a specific repo (forwarded to __init__;
    #                         subprocess cwd is the working dir for every preset).
    #   skip_permissions=True -> let the worker use tools without prompting. Headless
    #                         coding runs need this, but it bypasses ALL safety
    #                         prompts/sandboxing — only use against a repo you intend
    #                         the agent to edit (ideally a worktree/branch).

    @classmethod
    def claude_code(
        cls, *, model: Optional[str] = None, skip_permissions: bool = False, **kw
    ) -> "CliAgent":
        cmd = ["claude", "-p", "{prompt}", "--append-system-prompt", "{system}",
               "--output-format", "json"]
        if model:
            cmd += ["--model", model]
        if skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        # `--output-format json` wraps the reply in an envelope; pull out `.result`.
        return cls(cmd, parse_output=_parse_claude_json, **kw)

    @classmethod
    def codex(cls, *, skip_permissions: bool = False, **kw) -> "CliAgent":
        cmd = ["codex", "exec"]
        if skip_permissions:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        cmd.append("{combined}")
        return cls(cmd, **kw)

    @classmethod
    def opencode(cls, *, skip_permissions: bool = False, **kw) -> "CliAgent":
        cmd = ["opencode", "run"]
        if skip_permissions:
            cmd.append("--dangerously-skip-permissions")
        cmd.append("{combined}")
        return cls(cmd, **kw)

    @classmethod
    def aider(cls, *, skip_permissions: bool = False, **kw) -> "CliAgent":
        cmd = ["aider", "--message", "{combined}", "--no-auto-commits"]
        cmd.append("--yes-always" if skip_permissions else "--yes")
        return cls(cmd, **kw)
