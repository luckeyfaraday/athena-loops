"""Deterministic verification commands for the orchestration loop."""

from __future__ import annotations

import shlex
import subprocess
import time
from typing import Iterable, Optional

from .types import VerifyCommand, VerifyResult


class CommandVerifier:
    """Run configured commands and capture their output without raising."""

    def __init__(self, commands: Iterable[VerifyCommand], *, cwd: Optional[str] = None):
        self.commands = list(commands)
        self.cwd = cwd

    def run(self) -> list[VerifyResult]:
        return [self._run_one(cmd) for cmd in self.commands]

    def _run_one(self, cmd: VerifyCommand) -> VerifyResult:
        started = time.monotonic()
        try:
            proc = subprocess.run(
                cmd.command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=cmd.timeout,
                check=False,
            )
            return VerifyResult(
                name=cmd.name,
                ok=proc.returncode == 0,
                stdout=_trim(proc.stdout),
                stderr=_trim(proc.stderr),
                exit_code=proc.returncode,
                duration=time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as exc:
            return VerifyResult(
                name=cmd.name,
                ok=False,
                stdout=_trim(_decode(exc.stdout)),
                stderr=_trim(_decode(exc.stderr)),
                duration=time.monotonic() - started,
                error=f"timed out after {cmd.timeout}s",
            )
        except Exception as exc:  # noqa: BLE001 - verifier failures are loop feedback
            return VerifyResult(
                name=cmd.name,
                ok=False,
                duration=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )


def parse_verify_command(text: str, *, timeout: Optional[float] = None) -> VerifyCommand:
    """Parse a CLI/MCP verification string into argv form."""
    parts = shlex.split(text)
    if not parts:
        raise ValueError("verification command cannot be empty")
    return VerifyCommand(name=text, command=parts, timeout=timeout)


def summarize_verification(results: list[VerifyResult]) -> str:
    """Compact text suitable for reviewer context and next-iteration feedback."""
    if not results:
        return ""
    chunks: list[str] = []
    for r in results:
        status = "passed" if r.ok else "failed"
        line = f"- {r.name}: {status}"
        if r.exit_code is not None:
            line += f" (exit {r.exit_code})"
        if r.error:
            line += f"; {r.error}"
        details = []
        if r.stdout:
            details.append("stdout:\n" + r.stdout)
        if r.stderr:
            details.append("stderr:\n" + r.stderr)
        chunks.append(line + ("\n" + "\n".join(details) if details else ""))
    return "\n\n".join(chunks)


def _trim(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _decode(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)
