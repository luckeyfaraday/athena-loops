"""Run the orchestration loop end to end and print a trace.

    python -m examples.run_demo            # uses the zero-dependency MockAgent
    python -m examples.run_demo --claude   # uses Claude (needs anthropic + API key)
"""

from __future__ import annotations

import sys

from agentloop import Budget, Orchestrator
from agentloop.adapters import MockAgent
from agentloop.types import LoopState


def trace(state: LoopState) -> None:
    t = state.history[-1]
    print(f"\n── iteration {t.iteration} "
          f"({len([r for r in t.results if r.ok])}/{len(t.results)} subgoals ok) ──")
    for r in t.results:
        mark = "✓" if r.ok else "✗"
        print(f"  {mark} {r.subgoal.id}: {r.subgoal.description}")
    g = t.review
    print(f"  review: quality={g.quality_ok} consistency={g.consistency_ok} "
          f"aligned={g.goal_aligned} complete={g.goal_complete}")
    if g.issues:
        print(f"  issues: {g.issues}")


def main() -> None:
    if "--claude" in sys.argv:
        from agentloop.adapters import ClaudeAgent
        agent = ClaudeAgent()
    else:
        agent = MockAgent()

    orch = Orchestrator(agent, budget=Budget(max_iterations=4), observer=trace)
    result = orch.run(
        goal="Write a short briefing on the orchestrator-worker agent pattern.",
        success_criteria="A clear briefing covering decomposition, execution, review, "
                         "and the feedback loop, with concrete specifics.",
    )

    print("\n" + "=" * 60)
    print(f"completed   : {result.completed}")
    print(f"iterations  : {result.iterations}")
    print(f"stop_reason : {result.stop_reason}")
    print("=" * 60)
    print(result.final_output)


if __name__ == "__main__":
    main()
