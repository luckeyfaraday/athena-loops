"""Isolated git worktrees for autonomous coding runs.

Pairing `skip_permissions=True` with a throwaway worktree is the safe default:
the loop's workers edit a separate checkout on their own branch, never your main
working tree. Cleanup mirrors the harness's own worktrees — kept if the run
produced changes (so you can inspect/merge), auto-removed if it left nothing.

    from agentloop.worktree import worktree
    from agentloop.adapters import CliAgent
    from agentloop import Orchestrator

    with worktree("/path/to/repo") as wt:
        agent = CliAgent.claude_code(cwd=wt.path, skip_permissions=True)
        Orchestrator(agent).run(goal="...", success_criteria="...")
        print(wt.changed_files())   # what the run touched
        wt.commit("agentloop run")  # optional: persist on the branch
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


# A stuck git call must surface as an error, never an indefinite hang. The cap is
# generous because `worktree add` checks out a full tree (large repos, slow disks,
# AV scanning), but finite so a prompting/wedged git can't freeze the run forever.
_GIT_TIMEOUT = 300.0


def _git(repo: str, *args: str, timeout: float = _GIT_TIMEOUT) -> str:
    # stdin=DEVNULL: when the MCP server runs git, the child would otherwise inherit
    # the server's JSON-RPC stdin pipe. A checkout filter that prompts (Git LFS
    # smudge, a credential helper) then blocks reading a pipe that never answers and
    # hangs the whole run silently. DEVNULL + GIT_TERMINAL_PROMPT=0 make any such
    # prompt fail fast with a clear error instead.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        proc = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
            env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(args)} timed out after {timeout}s "
            f"(possibly blocked on a credential/LFS prompt)"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


@dataclass
class Worktree:
    path: str          # the isolated checkout (use this as the worker's cwd)
    branch: str        # the branch created for this run
    repo: str          # the source repo's top-level
    base_sha: str      # commit the worktree branched from
    _holder: str       # temp dir that owns `path`, removed on cleanup

    # --- inspection ---------------------------------------------------------

    def changed_files(self) -> list[str]:
        """Paths touched in the worktree (modified, added, or untracked)."""
        out = _git(self.path, "status", "--porcelain")
        return [line[3:] for line in out.splitlines()] if out else []

    def is_dirty(self) -> bool:
        return bool(_git(self.path, "status", "--porcelain"))

    def has_new_commits(self) -> bool:
        count = _git(self.path, "rev-list", "--count", f"{self.base_sha}..HEAD")
        return count not in ("", "0")

    def changed(self) -> bool:
        """True if the run produced anything worth keeping."""
        return self.is_dirty() or self.has_new_commits()

    def diff(self) -> str:
        return _git(self.path, "diff", self.base_sha)

    def commits(self) -> list[str]:
        """Checkpoint commits made on this branch since it was created (newest first)."""
        out = _git(self.path, "log", "--oneline", f"{self.base_sha}..HEAD")
        return out.splitlines() if out else []

    # --- actions ------------------------------------------------------------

    def commit(self, message: str) -> Optional[str]:
        """Stage everything and commit on the run's branch. Returns the new SHA."""
        _git(self.path, "add", "-A")
        if not _git(self.path, "status", "--porcelain"):
            return None  # nothing to commit
        _git(
            self.path,
            "-c", "user.name=agentloop",
            "-c", "user.email=agentloop@local",
            "commit", "-m", message,
        )
        return _git(self.path, "rev-parse", "HEAD")

    def remove(self) -> None:
        # --force because the worktree is throwaway and may have uncommitted edits.
        try:
            _git(self.repo, "worktree", "remove", "--force", self.path)
        finally:
            shutil.rmtree(self._holder, ignore_errors=True)


@contextmanager
def worktree(
    repo: str,
    *,
    branch: Optional[str] = None,
    base: str = "HEAD",
    cleanup: str = "auto",
) -> Iterator[Worktree]:
    """Create an isolated git worktree for the duration of the block.

    cleanup:
      "auto"   (default) keep the worktree iff the run changed something, else remove
      "always" always remove on exit
      "never"  always keep on exit (you remove it via wt.remove())
    """
    if cleanup not in ("auto", "always", "never"):
        raise ValueError(f"cleanup must be auto|always|never, got {cleanup!r}")

    try:
        top = _git(repo, "rev-parse", "--show-toplevel")
    except RuntimeError as exc:
        raise RuntimeError(f"{repo!r} is not a git repository: {exc}") from exc

    branch = branch or f"agentloop/{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    base_sha = _git(top, "rev-parse", base)

    holder = tempfile.mkdtemp(prefix="agentloop-wt-")
    path = os.path.join(holder, branch.replace("/", "-"))
    _git(top, "worktree", "add", "-b", branch, path, base_sha)

    wt = Worktree(path=path, branch=branch, repo=top, base_sha=base_sha, _holder=holder)
    try:
        yield wt
    finally:
        keep = cleanup == "never" or (cleanup == "auto" and wt.changed())
        if keep:
            # Leave it on disk and registered; tell the caller where it is.
            print(f"[agentloop] worktree kept at {wt.path} (branch {wt.branch})")
        else:
            wt.remove()
