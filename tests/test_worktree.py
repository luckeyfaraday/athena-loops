"""Worktree helper: isolation, change detection, commit, and cleanup policies."""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

from agentloop.worktree import worktree


def _make_repo() -> str:
    repo = tempfile.mkdtemp(prefix="agentloop-repo-")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", repo, "init", "-q"], check=True)
    with open(os.path.join(repo, "README.md"), "w") as f:
        f.write("base\n")
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-qm", "init"], check=True, env=env)
    return repo


def test_worktree_is_isolated_from_main_checkout():
    repo = _make_repo()
    with worktree(repo) as wt:
        assert os.path.isdir(wt.path)
        assert wt.path != repo
        with open(os.path.join(wt.path, "new.txt"), "w") as f:
            f.write("hi\n")
        # The change exists in the worktree but NOT in the source checkout.
        assert "new.txt" in wt.changed_files()
        assert not os.path.exists(os.path.join(repo, "new.txt"))


def test_changed_and_commit_and_diff():
    repo = _make_repo()
    with worktree(repo, cleanup="never") as wt:
        assert not wt.changed()
        with open(os.path.join(wt.path, "f.txt"), "w") as f:
            f.write("content\n")
        assert wt.changed()
        sha = wt.commit("add f")
        assert sha and wt.has_new_commits()
        assert "f.txt" in wt.diff()
        wt.remove()
    assert not os.path.exists(wt.path)


def test_cleanup_auto_removes_when_pristine():
    repo = _make_repo()
    with worktree(repo, cleanup="auto") as wt:
        path = wt.path  # touch nothing
    assert not os.path.exists(path)


def test_cleanup_auto_keeps_when_changed(capsys):
    repo = _make_repo()
    with worktree(repo, cleanup="auto") as wt:
        path = wt.path
        with open(os.path.join(wt.path, "x.txt"), "w") as f:
            f.write("x\n")
    assert os.path.exists(path)  # kept because the run changed something
    assert "worktree kept" in capsys.readouterr().out


def test_cleanup_always_removes_even_when_changed():
    repo = _make_repo()
    with worktree(repo, cleanup="always") as wt:
        path = wt.path
        with open(os.path.join(wt.path, "x.txt"), "w") as f:
            f.write("x\n")
    assert not os.path.exists(path)


def test_non_git_dir_raises():
    d = tempfile.mkdtemp()
    with pytest.raises(RuntimeError, match="not a git repository"):
        with worktree(d):
            pass
