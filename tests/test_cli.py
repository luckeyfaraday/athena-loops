"""The plain CLI surface: JSON output, exit codes, stdin, and backends."""

from __future__ import annotations

import json
import sys

from agentloop.cli import main


def test_run_json_completes(capsys):
    code = main(["run", "--goal", "g", "--criteria", "c",
                 "--backend", "mock", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["completed"] is True
    assert out["stop_reason"] == "goal_complete"


def test_exit_code_1_when_budget_stops_it(capsys):
    # mock accepts on iteration 2; cap at 1 -> not completed -> exit 1.
    code = main(["run", "--goal", "g", "--criteria", "c",
                 "--backend", "mock", "--max-iterations", "1", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["completed"] is False
    assert "max_iterations" in out["stop_reason"]


def test_human_output(capsys):
    code = main(["run", "--goal", "g", "--criteria", "c", "--backend", "mock"])
    text = capsys.readouterr().out
    assert code == 0
    assert "completed=True" in text


def test_goal_from_stdin(capsys, monkeypatch):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("build the thing"))
    code = main(["run", "--goal", "-", "--criteria", "c",
                 "--backend", "mock", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["completed"] is True


def test_goal_from_file(tmp_path, capsys):
    f = tmp_path / "goal.txt"
    f.write_text("ship it")
    code = main(["run", "--goal-file", str(f), "--criteria", "c",
                 "--backend", "mock", "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["completed"] is True


def test_missing_goal_errors():
    import pytest
    with pytest.raises(SystemExit):
        main(["run", "--criteria", "c", "--backend", "mock"])


def test_progress_streams_ndjson_to_stderr(capsys):
    main(["run", "--goal", "g", "--criteria", "c", "--backend", "mock",
          "--max-iterations", "2", "--progress", "--json"])
    err = capsys.readouterr().err.strip().splitlines()
    lines = [json.loads(l) for l in err if l.startswith("{")]
    assert [l["iteration"] for l in lines] == [1, 2]


def test_verify_option_runs_command(capsys):
    code = main([
        "run", "--goal", "g", "--criteria", "c", "--backend", "mock", "--json",
        "--verify", f"{sys.executable} -c 'print(456)'",
    ])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["history"][0]["verification"][0]["stdout"].strip() == "456"


def test_backends_subcommand(capsys):
    assert main(["backends"]) == 0
    assert "mock" in capsys.readouterr().out
