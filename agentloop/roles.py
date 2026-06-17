"""Role prompts — the model-facing layer (the "skills").

These are the *contents* of the boxes in the diagram: how to decompose, how a
subagent should work, and the reviewer's rubric. They are intentionally plain
strings so you can tune them without touching the harness, or load them from
files/config per deployment.
"""

from __future__ import annotations

from .types import Subgoal, TaskResult

DECOMPOSER_SYSTEM = """\
You are the Orchestrator. Break the user's goal into the SMALLEST set of \
independent subgoals that, if each is completed, fully satisfy the success \
criteria. Prefer fewer subgoals. Each must be self-contained (a subagent will \
work on it with no shared memory).

Respond with ONLY a JSON array, each item: {"id": "s1", "description": "..."}."""

SUBAGENT_SYSTEM = """\
You are a Subagent. Complete the ONE subgoal you are given using your tools, \
knowledge, and reasoning. Be concrete and produce the actual deliverable, not a \
plan to produce it. If you cannot complete it, say exactly why."""

REVIEWER_SYSTEM = """\
You are the Reviewer. Judge the aggregated outputs against the goal and success \
criteria across three gates — quality, consistency, and goal alignment — and \
decide whether the goal is COMPLETE.

Respond with ONLY a JSON object:
{
  "quality_ok": bool,
  "consistency_ok": bool,
  "goal_aligned": bool,
  "goal_complete": bool,
  "issues": ["..."],            // concrete, actionable; empty if none
  "follow_up_questions": ["..."] // info needed from the user; empty if none
}
Be strict: goal_complete is true only if every success criterion is met."""


def decompose_prompt(goal: str, success_criteria: str, feedback: str) -> str:
    parts = [f"GOAL:\n{goal}", f"\nSUCCESS CRITERIA:\n{success_criteria}"]
    if feedback:
        parts.append(
            "\nThis is a REFINEMENT pass. The previous attempt was not accepted. "
            "Adjust the subgoals to address this feedback:\n" + feedback
        )
    return "\n".join(parts)


def subagent_prompt(subgoal: Subgoal, goal: str) -> str:
    prompt = (
        f"OVERALL GOAL (for context only):\n{goal}\n\n"
        f"YOUR SUBGOAL:\n{subgoal.description}"
    )
    if subgoal.notes:
        prompt += f"\n\nNotes from a previous attempt:\n{subgoal.notes}"
    return prompt


def review_prompt(goal: str, success_criteria: str, aggregated: str) -> str:
    return (
        f"GOAL:\n{goal}\n\n"
        f"SUCCESS CRITERIA:\n{success_criteria}\n\n"
        f"AGGREGATED OUTPUTS FROM SUBAGENTS:\n{aggregated}"
    )


def aggregate(results: list[TaskResult]) -> str:
    """Deterministic aggregation of subagent outputs (the 'Finished Task Outputs' box)."""
    lines: list[str] = []
    for r in results:
        header = f"### {r.subgoal.id}: {r.subgoal.description}"
        if r.ok:
            lines.append(f"{header}\n{r.output}")
        else:
            lines.append(f"{header}\n[FAILED after {r.attempts} attempt(s)] {r.error}")
    return "\n\n".join(lines)


def build_feedback(results: list[TaskResult], review) -> str:
    """Turn this iteration's problems into the 'refine plan' signal for the next."""
    parts: list[str] = []
    failed = [r for r in results if not r.ok]
    if failed:
        parts.append(
            "Failed subgoals:\n"
            + "\n".join(f"- {r.subgoal.description}: {r.error}" for r in failed)
        )
    if review.issues:
        parts.append("Reviewer issues:\n" + "\n".join(f"- {i}" for i in review.issues))
    if not review.gates_passed:
        gates = []
        if not review.quality_ok:
            gates.append("quality")
        if not review.consistency_ok:
            gates.append("consistency")
        if not review.goal_aligned:
            gates.append("goal alignment")
        parts.append("Gates not passed: " + ", ".join(gates))
    return "\n\n".join(parts) if parts else "Goal not yet complete; refine the outputs."
