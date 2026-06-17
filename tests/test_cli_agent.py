"""CliAgent: the inward plug — drive the loop with a coding-agent CLI.

These use a tiny Python stub as the "CLI" so no real agent needs to be installed.
"""

from __future__ import annotations

import os
import sys

import pytest

from agentloop import Budget, Orchestrator
from agentloop.adapters import CliAgent
from agentloop.agent import AgentRequest

# A stub "coding agent": branches on the role's system prompt and emits the
# shape each role expects. Reads the combined system+prompt from stdin.
ROLE_STUB = r"""
import sys, json
text = sys.stdin.read()
if "JSON array" in text:                       # decomposer system prompt
    print(json.dumps([{"id": "s1", "description": "implement the thing"}]))
elif "JSON object" in text:                    # reviewer system prompt
    print(json.dumps({"quality_ok": True, "consistency_ok": True,
                      "goal_aligned": True, "goal_complete": True,
                      "issues": [], "follow_up_questions": []}))
else:                                           # subagent
    print("edited files and ran the build: OK")
"""


def _req(role="subagent", system="", prompt="hello"):
    return AgentRequest(role=role, system=system, prompt=prompt)


def test_arg_substitution_passes_prompt_as_argument():
    agent = CliAgent([sys.executable, "-c", "import sys; print(sys.argv[1])", "{prompt}"])
    assert agent.run(_req(prompt="ship it")).text == "ship it"


def test_stdin_mode_when_no_placeholder_sends_combined():
    # No {prompt}/{combined} in the template -> text goes to stdin.
    agent = CliAgent([sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"])
    out = agent.run(_req(system="SYS", prompt="BODY")).text
    assert "SYS" in out and "BODY" in out


def test_parse_output_extracts_from_json_envelope():
    stub = "import json; print(json.dumps({'result': 'inner text'}))"
    agent = CliAgent([sys.executable, "-c", stub],
                     parse_output=lambda out: __import__("json").loads(out)["result"])
    assert agent.run(_req()).text == "inner text"


def test_nonzero_exit_raises():
    agent = CliAgent([sys.executable, "-c", "import sys; sys.exit(3)"])
    with pytest.raises(RuntimeError, match="exited 3"):
        agent.run(_req())


def test_timeout_raises():
    agent = CliAgent([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.3)
    with pytest.raises(RuntimeError, match="timed out"):
        agent.run(_req())


def test_default_has_no_per_call_timeout():
    # Coding workers are slow; the default must not cap them at some short value.
    assert CliAgent([sys.executable, "-c", "pass"]).timeout is None
    assert CliAgent.claude_code().timeout is None


def test_timeout_flows_through_build_agent():
    from agentloop.mcp_server import _build_agent
    agent = _build_agent("claude_code", cwd=None, skip_permissions=False,
                         model=None, timeout=600.0)
    assert agent.timeout == 600.0


def test_cwd_is_honored():
    import tempfile
    d = tempfile.mkdtemp()
    agent = CliAgent([sys.executable, "-c", "import os; print(os.getcwd())"], cwd=d)
    assert os.path.realpath(agent.run(_req()).text) == os.path.realpath(d)


def test_skip_permissions_adds_bypass_flag():
    assert "--dangerously-skip-permissions" in CliAgent.claude_code(skip_permissions=True).command
    assert "--dangerously-skip-permissions" not in CliAgent.claude_code().command
    assert "--dangerously-bypass-approvals-and-sandbox" in CliAgent.codex(skip_permissions=True).command
    assert "--yes-always" in CliAgent.aider(skip_permissions=True).command


def test_cwd_forwards_through_presets():
    agent = CliAgent.claude_code(cwd="/tmp/repo", skip_permissions=True)
    assert agent.cwd == "/tmp/repo"


def test_cli_agent_drives_the_full_loop():
    # The whole orchestration loop, executed entirely through a CLI backend.
    agent = CliAgent([sys.executable, "-c", ROLE_STUB])
    orch = Orchestrator(agent, budget=Budget(max_iterations=3))
    result = orch.run("build a feature", "feature works and build passes")
    assert result.completed
    assert result.iterations == 1
    assert "edited files" in result.final_output
