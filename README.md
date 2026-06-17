# agentloop

A backend-agnostic implementation of the **AI Agent Orchestration Loop** —
the orchestrator → worker → reviewer pattern with a closed feedback loop.

The design principle: **the loop is a harness (deterministic code), not a skill.**
A prompt can *describe* "decompose, review, loop until done" but can't *guarantee*
it. So the control flow lives in code, and the model-facing judgement (how to
decompose, the review rubric) lives in swappable prompts. One harness drives any
backend through a single `Agent` interface.

```
            ┌──────────── harness (this package) ────────────┐
goal ─▶ decompose ─▶ fan-out to subagents ─▶ aggregate ─▶ review gate ─▶ done?
            ▲                                                          │ no
            └──────────────── feedback: refine plan ◀──────────────────┘
```

## Quick start

```bash
python3 -m examples.run_demo        # zero-dependency MockAgent
python3 -m pytest                   # 6 tests, no deps
```

```python
from agentloop import Orchestrator, Budget
from agentloop.adapters import MockAgent

orch = Orchestrator(MockAgent(), budget=Budget(max_iterations=4))
result = orch.run(
    goal="Write a briefing on the orchestrator-worker pattern.",
    success_criteria="Covers decomposition, execution, review, and the feedback loop.",
)
print(result.completed, result.iterations, result.stop_reason)
print(result.final_output)
```

## Use a real model

```bash
pip install -e ".[claude]"
export ANTHROPIC_API_KEY=sk-...
python3 -m examples.run_demo --claude
```

## Plug into any coding agent

The loop is pluggable in two directions, both thin wrappers over the `Agent` seam:

**Inward — coding agents *are* the workers.** `CliAgent` runs each role
(decomposer / subagent / reviewer) through a headless coding-agent CLI, so the
workers get that agent's tools, file access, and repo context:

```python
from agentloop import Orchestrator
from agentloop.adapters import CliAgent

orch = Orchestrator(CliAgent.claude_code())   # or .codex() / .opencode() / .aider()
result = orch.run(goal="Add a /health endpoint + test", success_criteria="test passes")
```

```bash
python3 -m examples.run_with_cli_agent claude   # codex | opencode | aider
```

Custom CLI? It's just a command template (`{prompt}`, `{system}`, `{combined}`;
no prompt placeholder ⇒ text is piped on stdin):

```python
CliAgent(["my-agent", "--system", "{system}", "--ask", "{prompt}"])
```

Presets are starting points — CLI flags vary by version; confirm yours and tweak
`agentloop/adapters/cli.py`. A non-zero exit or timeout becomes a FAILED task
(with retries), not a silent wrong answer.

**Outward — a coding agent *calls* the loop.** Wrap `Orchestrator.run()` behind a
stable CLI (`agentloop run --goal … --json`) for universal shell access, or an MCP
server exposing `orchestrate(goal, criteria, budget)` for native tool integration.
*(planned — say the word and I'll add them.)*

## The seam (where to put what)

| Layer | Lives in | What it owns |
|-------|----------|--------------|
| **Harness** | `orchestrator.py`, `scheduler.py`, `types.py` | the loop, fan-out, aggregation, review gate, termination guards, failure capture |
| **Agent seam** | `agent.py` + `adapters/` | one `Agent.run(request) -> response` per backend (Mock, Claude, …) |
| **Skills** | `roles.py` | the prompts *inside* each box: decomposer, subagent, reviewer rubric |

To support a new backend, implement one method:

```python
from agentloop.agent import Agent, AgentRequest, AgentResponse

class MyAgent(Agent):
    def run(self, request: AgentRequest) -> AgentResponse:
        text = call_your_model(system=request.system, prompt=request.prompt)
        return AgentResponse(text=text)
```

The three roles (Orchestrator, Subagent, Reviewer) are the *same* `Agent`
invoked with different system prompts — not separate classes.

## Two gaps in the original diagram, handled here

- **Termination guards** — `Budget` caps iterations, wall-clock time, and total
  agent calls so the NO-branch can't spin forever.
- **Subagent failure handling** — a subagent that raises becomes a `FAILED`
  `TaskResult` (with retries), visible to the reviewer and the feedback step,
  instead of crashing the run or being silently dropped.

## Layout

```
agentloop/
  orchestrator.py   # the loop (deterministic harness)
  scheduler.py      # parallel/sequential subagent execution + retries
  roles.py          # role prompts — the tunable "skills"
  agent.py          # Agent interface + robust JSON extraction
  types.py          # Budget, Subgoal, TaskResult, ReviewResult, LoopState, ...
  adapters/
    mock.py         # deterministic, dependency-free (demo + tests)
    claude.py       # Anthropic SDK backend
examples/run_demo.py
tests/test_orchestrator.py
```
