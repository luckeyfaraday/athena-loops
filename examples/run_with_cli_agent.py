"""Drive the orchestration loop using a real coding agent as the worker backend.

The loop's decomposer/subagents/reviewer are executed by whatever coding-agent
CLI you point it at — so the workers get that agent's tools and repo access.

    python3 -m examples.run_with_cli_agent claude  [REPO_DIR]
    python3 -m examples.run_with_cli_agent codex   [REPO_DIR]
    python3 -m examples.run_with_cli_agent opencode

Pass REPO_DIR to run the worker against a real repo with permissions skipped
(it WILL edit files there — use a worktree/throwaway branch). With no REPO_DIR it
runs read-only-ish with prompts left on.

Note: presets are starting templates; confirm your CLI's flags and adjust
agentloop/adapters/cli.py if a call errors. Requires the chosen agent installed.
"""

from __future__ import annotations

import sys

from agentloop import Budget, Orchestrator
from agentloop.adapters import CliAgent

BUILDERS = {
    "claude": CliAgent.claude_code,
    "codex": CliAgent.codex,
    "opencode": CliAgent.opencode,
    "aider": CliAgent.aider,
}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "claude"
    repo = sys.argv[2] if len(sys.argv) > 2 else None
    if which not in BUILDERS:
        sys.exit(f"unknown agent {which!r}; choose from {list(BUILDERS)}")

    # Skip permissions only when pointed at an explicit repo to edit.
    kw = {}
    if repo:
        kw["cwd"] = repo
        if which != "opencode":  # opencode's run has no skip-permissions flag
            kw["skip_permissions"] = True
    agent = BUILDERS[which](**kw)  # e.g. CliAgent.claude_code(cwd=..., skip_permissions=True)
    orch = Orchestrator(agent, budget=Budget(max_iterations=3))
    result = orch.run(
        goal="Add a /health endpoint that returns {status: ok} and a test for it.",
        success_criteria="Endpoint exists, returns 200 with the JSON, and its test passes.",
    )
    print(f"\ncompleted={result.completed} iterations={result.iterations} "
          f"stop_reason={result.stop_reason}\n")
    print(result.final_output)


if __name__ == "__main__":
    main()
