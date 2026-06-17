"""Drive the orchestration loop using a real coding agent as the worker backend.

The loop's decomposer/subagents/reviewer are executed by whatever coding-agent
CLI you point it at — so the workers get that agent's tools and repo access.

    python3 -m examples.run_with_cli_agent claude  [REPO_DIR]
    python3 -m examples.run_with_cli_agent codex   [REPO_DIR]
    python3 -m examples.run_with_cli_agent opencode

Pass REPO_DIR to run the worker against a real repo. The run happens in an
isolated git worktree on its own branch (permissions skipped), so your main
checkout is never touched; the worktree is kept iff the run changed something.
With no REPO_DIR it runs in a throwaway dir with prompts left on.

Note: presets are starting templates; confirm your CLI's flags and adjust
agentloop/adapters/cli.py if a call errors. Requires the chosen agent installed.
"""

from __future__ import annotations

import sys

from agentloop import Budget, Orchestrator, worktree
from agentloop.adapters import CliAgent

BUILDERS = {
    "claude": CliAgent.claude_code,
    "codex": CliAgent.codex,
    "opencode": CliAgent.opencode,
    "aider": CliAgent.aider,
}

GOAL = "Add a /health endpoint that returns {status: ok} and a test for it."
CRITERIA = "Endpoint exists, returns 200 with the JSON, and its test passes."


def run_against(agent_name: str, cwd: str | None) -> None:
    build = BUILDERS[agent_name]
    kw = {}
    if cwd:
        kw["cwd"] = cwd
        if agent_name != "opencode":  # opencode's run has no skip-permissions flag
            kw["skip_permissions"] = True
    orch = Orchestrator(build(**kw), budget=Budget(max_iterations=3))
    result = orch.run(goal=GOAL, success_criteria=CRITERIA)
    print(f"\ncompleted={result.completed} iterations={result.iterations} "
          f"stop_reason={result.stop_reason}\n")
    print(result.final_output)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "claude"
    repo = sys.argv[2] if len(sys.argv) > 2 else None
    if which not in BUILDERS:
        sys.exit(f"unknown agent {which!r}; choose from {list(BUILDERS)}")

    if repo:
        # Isolate the whole run in a throwaway worktree/branch off the repo.
        with worktree(repo) as wt:
            print(f"[isolated worktree] {wt.path} (branch {wt.branch})")
            run_against(which, wt.path)
            if wt.changed():
                print(f"\nchanged files: {wt.changed_files()}")
    else:
        run_against(which, None)


if __name__ == "__main__":
    main()
