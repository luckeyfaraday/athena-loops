"""Claude backend via the Anthropic Python SDK.

This is the only file that knows about Anthropic. The orchestrator never imports
it. Requires `pip install anthropic` and ANTHROPIC_API_KEY in the environment.

Model IDs (most capable first): claude-opus-4-8, claude-sonnet-4-6,
claude-haiku-4-5-20251001. Workers default to Sonnet for cost; bump the
`model` arg to Opus for the orchestrator/reviewer if you want stronger judgement.
"""

from __future__ import annotations

import os
from typing import Optional

from ..agent import Agent, AgentRequest, AgentResponse


class ClaudeAgent(Agent):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        *,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        client=None,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ClaudeAgent requires the anthropic SDK: pip install anthropic"
            ) from exc
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def run(self, request: AgentRequest) -> AgentResponse:
        system = request.system
        if request.expects_json:
            # Reinforce the format contract; extract_json still tolerates stray prose.
            system += "\n\nReturn ONLY the JSON value, no prose, no code fences."
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": request.prompt}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        return AgentResponse(text=text, raw=msg)
