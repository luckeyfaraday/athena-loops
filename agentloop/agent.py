"""The backend-agnostic agent seam.

Everything in the orchestrator talks to this interface and *only* this
interface. Swap the implementation (Claude, OpenAI, a local model, a mock)
without touching the loop. Roles in the diagram — Orchestrator, Subagent,
Reviewer — are not separate classes; they are the same `Agent` invoked with a
different system prompt, which is what makes one harness drive "any agent".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class AgentRequest:
    role: str                 # "decomposer" | "subagent" | "reviewer" — metadata + mock dispatch
    system: str               # role system prompt (the model-facing "skill")
    prompt: str               # the actual task / question
    context: dict[str, Any] = field(default_factory=dict)
    expects_json: bool = False  # hint: response should be a single JSON value


@dataclass
class AgentResponse:
    text: str
    raw: Any = None           # backend-native response object, for logging/debugging


class Agent(Protocol):
    """Implement this once per backend."""

    def run(self, request: AgentRequest) -> AgentResponse: ...


def extract_json(text: str) -> Any:
    """Best-effort parse of a JSON value from a model response.

    Models wrap JSON in prose or ```json fences; we recover the first complete
    object/array. Raises ValueError if nothing parseable is found so callers can
    treat it as a (handleable) task failure rather than a silent wrong answer.
    """
    text = text.strip()
    # Fast path: the whole thing is JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip a ```json ... ``` fence if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Scan for the first balanced {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)

    raise ValueError(f"no parseable JSON found in response: {text[:200]!r}")
