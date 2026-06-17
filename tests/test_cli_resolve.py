"""Windows CLI-shim resolution in adapters.cli (`_deshim_windows` / `_resolve_program`).

`_deshim_windows` is pure file parsing, so it is exercised here on any OS by
writing sample npm-style `.cmd` shims into a tmp dir next to real target files.
Two faithfulness notes baked into the fixtures:

* Real npm shims reference the node binary through a `%_prog%` variable (or a
  bare, unquoted `node`), never a literally-quoted `"node.exe"` — that is exactly
  why the `.exe`-preference branch skips it and returns `[node, script]`. The
  node-style fixtures use `%_prog%` so they match production shims.
* Fixtures spell paths as `%~dp0/<file>` (forward slash) so the resolved tokens
  land on real files on POSIX CI. On Windows the `\\` form parses identically;
  the separator handling there is `os.path.normpath`'s job, not ours.

`_resolve_program`'s Windows-only branch is covered by monkeypatching `os.name`
and `shutil.which`.
"""

from __future__ import annotations

import pytest

from agentloop.adapters import cli as cli_mod
from agentloop.adapters.cli import _deshim_windows, _resolve_program


def _shim(tmp_path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


def _touch(tmp_path, name: str):
    (tmp_path / name).write_text("", encoding="utf-8")
    return str(tmp_path / name)


# --- _deshim_windows ---------------------------------------------------------

def test_deshim_node_style_returns_node_and_script(tmp_path):
    # codex-style: bundled node.exe execs the wrapped cli .js. node is behind
    # `%_prog%` (as in real shims), so the .js is what we key off.
    node = _touch(tmp_path, "node.exe")
    script = _touch(tmp_path, "codex.js")
    shim = _shim(tmp_path, "codex.cmd",
                 '@ECHO off\r\n"%_prog%"  "%~dp0/codex.js" %*\r\n')
    assert _deshim_windows(shim) == [node, script]


def test_deshim_node_falls_back_to_path_node(tmp_path, monkeypatch):
    # No bundled node.exe next to the shim -> resolve `node` from PATH instead.
    script = _touch(tmp_path, "codex.js")
    path_node = _touch(tmp_path, "elsewhere-node")
    monkeypatch.setattr(cli_mod.shutil, "which",
                        lambda name: path_node if name == "node" else None)
    shim = _shim(tmp_path, "codex.cmd", 'node "%~dp0/codex.js" %*\r\n')
    assert _deshim_windows(shim) == [path_node, script]


def test_deshim_exe_style_returns_exe(tmp_path):
    # claude/opencode-style: the shim execs a real .exe directly.
    exe = _touch(tmp_path, "claude.exe")
    shim = _shim(tmp_path, "claude.cmd", '@ECHO off\r\n"%~dp0/claude.exe" %*\r\n')
    assert _deshim_windows(shim) == [exe]


def test_deshim_picks_last_exec_line(tmp_path):
    # When several lines carry `%*`, the reversed scan must take the last one.
    _touch(tmp_path, "old.exe")
    claude = _touch(tmp_path, "claude.exe")
    shim = _shim(tmp_path, "claude.cmd",
                 '"%~dp0/old.exe" %*\r\n"%~dp0/claude.exe" %*\r\n')
    assert _deshim_windows(shim) == [claude]


def test_deshim_unrecognized_target_returns_none(tmp_path):
    # The `%*` line references no existing .exe/.js -> None (caller then runs the
    # .cmd as-is, which is correct for single-line args).
    shim = _shim(tmp_path, "weird.cmd", '"%~dp0/nope.exe" %*\r\n')
    assert _deshim_windows(shim) is None


def test_deshim_no_exec_line_returns_none(tmp_path):
    shim = _shim(tmp_path, "weird.cmd", '@ECHO off\r\nREM nothing to exec here\r\n')
    assert _deshim_windows(shim) is None


def test_deshim_missing_file_returns_none(tmp_path):
    assert _deshim_windows(str(tmp_path / "does-not-exist.cmd")) is None


# --- _resolve_program --------------------------------------------------------

def test_resolve_program_empty_argv():
    assert _resolve_program([]) == []


def test_resolve_program_unresolved_left_unchanged(monkeypatch):
    # which() finds nothing -> argv is returned as-is so the original
    # FileNotFoundError still surfaces with the bare name.
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _name: None)
    assert _resolve_program(["claude", "--print"]) == ["claude", "--print"]


def test_resolve_program_posix_returns_resolved_path(monkeypatch):
    monkeypatch.setattr(cli_mod.os, "name", "posix")
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _name: "/usr/bin/claude")
    assert _resolve_program(["claude", "-p"]) == ["/usr/bin/claude", "-p"]


def test_resolve_program_windows_deshims_cmd_and_keeps_args(tmp_path, monkeypatch):
    exe = _touch(tmp_path, "claude.exe")
    shim = _shim(tmp_path, "claude.cmd", '"%~dp0/claude.exe" %*\r\n')
    monkeypatch.setattr(cli_mod.os, "name", "nt")
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _name: shim)
    # The multi-line prompt arg must pass through intact after the resolved exe
    # (the whole point: cmd.exe never sees it to truncate at the newline).
    out = _resolve_program(["claude", "-p", "line1\nline2"])
    assert out == [exe, "-p", "line1\nline2"]


def test_resolve_program_windows_plain_exe_not_deshimmed(tmp_path, monkeypatch):
    # which() resolved straight to an .exe (not a .cmd) -> use it, no de-shim.
    exe = _touch(tmp_path, "claude.exe")
    monkeypatch.setattr(cli_mod.os, "name", "nt")
    monkeypatch.setattr(cli_mod.shutil, "which", lambda _name: exe)
    assert _resolve_program(["claude", "-p"]) == [exe, "-p"]
