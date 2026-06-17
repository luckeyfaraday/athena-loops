"""Stable command-line surface — the universal outward plug.

Any coding agent that can run a shell command can drive the loop, even with no
MCP support:

    agentloop run --goal "..." --criteria "..." --backend claude_code --cwd . --json
    agentloop backends

`--json` prints the full result dict to stdout (machine-readable); `--progress`
streams one NDJSON line per iteration to stderr. Exit code is 0 if the goal
completed, 1 if it stopped on a budget guard, 2 on error — so scripts can branch.
Wraps the same `orchestrate_impl` the MCP server uses; the contract is identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .mcp_server import BACKENDS, orchestrate_impl
from .types import LoopState


def _read(value: Optional[str], path: Optional[str], what: str) -> str:
    if path:
        with open(path) as f:
            return f.read().strip()
    if value == "-":
        return sys.stdin.read().strip()
    if value:
        return value
    raise SystemExit(f"error: provide --{what} or --{what}-file")


def _progress(state: LoopState) -> None:
    t = state.history[-1]
    line = {
        "iteration": t.iteration,
        "subgoals_ok": sum(r.ok for r in t.results),
        "subgoals_total": len(t.results),
        "goal_complete": t.review.goal_complete,
    }
    print(json.dumps(line), file=sys.stderr, flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    goal = _read(args.goal, args.goal_file, "goal")
    criteria = _read(args.criteria, args.criteria_file, "criteria")

    result = orchestrate_impl(
        goal, criteria,
        backend=args.backend,
        cwd=args.cwd,
        max_iterations=args.max_iterations,
        skip_permissions=args.skip_permissions,
        isolate=not args.no_isolate,
        model=args.model,
        observer=_progress if args.progress else None,
    )

    if args.json:
        json.dump(result, sys.stdout)
        print()
    else:
        print(f"completed={result['completed']} iterations={result['iterations']} "
              f"stop_reason={result['stop_reason']}")
        wt = result.get("worktree")
        if wt:
            print(f"worktree: {wt['path']} (branch {wt['branch']}) "
                  f"changed={wt['changed_files']}")
        print("---")
        print(result["final_output"])

    return 0 if result["completed"] else 1


def cmd_backends(args: argparse.Namespace) -> int:
    if args.json:
        json.dump({"backends": BACKENDS, "default": "claude_code"}, sys.stdout)
        print()
    else:
        print("\n".join(BACKENDS))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentloop",
                                description="Run the AI agent orchestration loop.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the orchestration loop")
    r.add_argument("--goal", help="goal text (use '-' to read stdin)")
    r.add_argument("--goal-file", help="read goal from a file")
    r.add_argument("--criteria", help="success criteria (use '-' to read stdin)")
    r.add_argument("--criteria-file", help="read success criteria from a file")
    r.add_argument("--backend", default="claude_code",
                   help=f"worker backend ({' | '.join(BACKENDS)})")
    r.add_argument("--cwd", help="repo to work in (coding tasks that edit files)")
    r.add_argument("--max-iterations", type=int, default=4,
                   help="cap on decompose->review cycles (termination guard)")
    r.add_argument("--skip-permissions", action="store_true",
                   help="let CLI workers use tools without prompting (needs --cwd)")
    r.add_argument("--no-isolate", action="store_true",
                   help="do NOT run in a throwaway worktree when --cwd is set")
    r.add_argument("--model", help="model override for claude_code / claude_api")
    r.add_argument("--json", action="store_true", help="emit full result as JSON")
    r.add_argument("--progress", action="store_true",
                   help="stream per-iteration NDJSON to stderr")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("backends", help="list worker backends")
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_backends)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # surface failures with exit code 2
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
