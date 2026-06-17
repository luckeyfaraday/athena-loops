"""Role prompts — the model-facing layer (the "skills").

These are the *contents* of the boxes in the diagram: how to decompose, how a
subagent should work, and the reviewer's rubric. They are intentionally plain
strings so you can tune them without touching the harness, or load them from
files/config per deployment.
"""

from __future__ import annotations

from .types import Subgoal, TaskResult, VerifyResult

CRITERIA_SYSTEM = """\
You turn a user's goal into clear, checkable success criteria — the concrete \
conditions that mean the goal is done. Keep it short. Respond with ONLY the \
criteria as a brief paragraph or bullet list, no preamble."""

CLARIFIER_SYSTEM = """\
You are doing intake before any work begins. Identify only the questions whose \
answers you genuinely need from the user to proceed well — real ambiguities or \
missing decisions that would change the plan, not nice-to-haves. If the task is \
already clear enough to start, return an empty list.

Respond with ONLY a JSON array of question strings."""

DECOMPOSER_SYSTEM = """\
You are the Orchestrator. Break the user's goal into the SMALLEST set of \
independent subgoals that, if each is completed, fully satisfy the success \
criteria. Prefer fewer subgoals. Each must be self-contained (a subagent will \
work on it with no shared memory).

Respond with ONLY a JSON array, each item: {"id": "s1", "description": "..."}."""

SUBAGENT_SYSTEM = """\
You are a Subagent. Complete the ONE subgoal you are given using your tools, \
knowledge, and reasoning. Be concrete and produce the actual deliverable, not a \
plan to produce it.

IMPORTANT: your working directory may already contain partial work from an \
earlier attempt or a previous iteration. ALWAYS inspect the current state first \
and CONTINUE from it — extend and fix what is there; never restart from scratch, \
re-create files that already exist, or undo prior progress. If you cannot \
complete the subgoal, do as much as you safely can, then say exactly what \
remains and why."""

# Opt-in Playwright guidance, appended to the subagent/reviewer system prompts
# only when a run enables it (CLI --playwright / orchestrate playwright=true).
# Kept out of the base prompts so non-web runs stay free of irrelevant testing
# instructions; see subagent_system / reviewer_system below.
PLAYWRIGHT_SUBAGENT_NOTE = """

TESTING: if your subgoal touches a web UI or browser-observable behavior, write \
or extend Playwright tests that exercise the change end-to-end, and run them \
(`npx playwright test`) before reporting done. Treat a passing Playwright suite \
as part of the deliverable, not an afterthought; report the command you ran and \
its result. Prefer extending the existing spec files over creating parallel \
ones."""

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

PLAYWRIGHT_REVIEWER_NOTE = """ For work that touches a web UI or \
browser-observable behavior, do NOT mark goal_complete unless there is evidence \
of passing Playwright tests that exercise the change; if such tests are missing \
or failing, set goal_complete to false and add a concrete issue asking for them."""


def subagent_system(playwright: bool = False) -> str:
    """The subagent system prompt, with optional Playwright testing guidance."""
    return SUBAGENT_SYSTEM + (PLAYWRIGHT_SUBAGENT_NOTE if playwright else "")


def reviewer_system(playwright: bool = False) -> str:
    """The reviewer system prompt, with optional Playwright evidence gate."""
    return REVIEWER_SYSTEM + (PLAYWRIGHT_REVIEWER_NOTE if playwright else "")


def criteria_prompt(goal: str) -> str:
    return f"GOAL:\n{goal}\n\nWrite the success criteria."


def clarify_prompt(goal: str, success_criteria: str, max_questions: int) -> str:
    return (
        f"GOAL:\n{goal}\n\nSUCCESS CRITERIA:\n{success_criteria}\n\n"
        f"List at most {max_questions} questions you need answered before planning "
        f"(JSON array of strings; return [] if the task is clear enough to start)."
    )


def decompose_prompt(
    goal: str, success_criteria: str, feedback: str, clarifications: str = ""
) -> str:
    parts = [f"GOAL:\n{goal}", f"\nSUCCESS CRITERIA:\n{success_criteria}"]
    if clarifications:
        parts.append("\nCLARIFICATIONS FROM THE USER:\n" + clarifications)
    if feedback:
        parts.append(
            "\nThis is a REFINEMENT pass. Earlier iterations already ran and their "
            "work is preserved in the working directory. Plan subgoals ONLY for "
            "what still REMAINS — fixes and unfinished pieces — and do not re-plan "
            "work that is already done. Address this feedback:\n" + feedback
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


def review_prompt(
    goal: str, success_criteria: str, aggregated: str, verification: str = ""
) -> str:
    prompt = (
        f"GOAL:\n{goal}\n\n"
        f"SUCCESS CRITERIA:\n{success_criteria}\n\n"
        f"AGGREGATED OUTPUTS FROM SUBAGENTS:\n{aggregated}"
    )
    if verification:
        prompt += (
            "\n\nVERIFICATION RESULTS FROM REAL COMMANDS:\n"
            f"{verification}\n\n"
            "If any required verification failed, goal_complete must be false."
        )
    return prompt


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


def build_feedback(
    results: list[TaskResult], review, verification: list[VerifyResult] | None = None
) -> str:
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
    failed_verification = [r for r in (verification or []) if not r.ok]
    if failed_verification:
        lines = []
        for r in failed_verification:
            msg = f"- {r.name}"
            if r.exit_code is not None:
                msg += f" exited {r.exit_code}"
            if r.error:
                msg += f": {r.error}"
            elif r.stderr:
                msg += f": {r.stderr.strip()}"
            elif r.stdout:
                msg += f": {r.stdout.strip()}"
            lines.append(msg)
        parts.append("Verification failures:\n" + "\n".join(lines))
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
