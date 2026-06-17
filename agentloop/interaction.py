"""The human-input seam — the mirror of the `Agent` seam.

The loop talks to models through `Agent`; it talks to the *user* through
`Interaction`. The control flow stays deterministic, while HOW a human answers
becomes a swappable implementation per surface:

  AutoInteraction     headless — never blocks; proceeds with best judgment
  ConsoleInteraction  interactive terminal — prompts via input()
  SuspendInteraction  stateless surfaces (MCP / scripted CLI) — replays
                      pre-supplied answers, else raises NeedInput to hand
                      control back so the caller can gather answers and resume
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable


class NeedInput(Exception):
    """Suspend signal: human input is required but no answers are queued.

    Carries the pending questions plus the goal/criteria resolved so far, so the
    caller can present the questions and resume the run with answers.
    """

    def __init__(self, questions: Sequence[str], *, goal: str = "", criteria: str = ""):
        super().__init__(f"needs input: {len(questions)} question(s)")
        self.questions = list(questions)
        self.goal = goal
        self.criteria = criteria


@runtime_checkable
class Interaction(Protocol):
    def ask(self, questions: Sequence[str]) -> list[str]:
        """Return one answer per question (same order)."""
        ...

    def confirm(self, prompt: str) -> bool:
        """Yes/no sign-off (e.g. to accept proposed success criteria)."""
        ...


class AutoInteraction:
    """Headless default: never blocks, accepts proposals, defers judgement."""

    def __init__(self, default_answer: str = "(no preference — use your best judgment)"):
        self.default_answer = default_answer

    def ask(self, questions: Sequence[str]) -> list[str]:
        return [self.default_answer for _ in questions]

    def confirm(self, prompt: str) -> bool:
        return True


class ConsoleInteraction:
    """Interactive terminal: ask the human directly."""

    def ask(self, questions: Sequence[str]) -> list[str]:
        answers: list[str] = []
        print("\nA few questions before I start:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
            answers.append(input("     > ").strip() or "(no answer)")
        return answers

    def confirm(self, prompt: str) -> bool:
        return input(f"{prompt} [Y/n] ").strip().lower() in ("", "y", "yes")


class SuspendInteraction:
    """Stateless surfaces: replay queued answers, else suspend via NeedInput.

    Confirmations auto-proceed (a proposed-criteria prompt shouldn't block a
    headless caller); only open questions cause a suspend.
    """

    def __init__(self, answers: Optional[Sequence[str]] = None):
        self._answers = list(answers or [])

    def ask(self, questions: Sequence[str]) -> list[str]:
        if len(self._answers) >= len(questions):
            taken = self._answers[: len(questions)]
            self._answers = self._answers[len(questions):]
            return taken
        raise NeedInput(questions)

    def confirm(self, prompt: str) -> bool:
        return True
