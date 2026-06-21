"""Detached runs + the live event stream — so callers aren't blind until done.

A normal `orchestrate` call blocks until the whole loop finishes, hiding
everything that happens inside: which subgoals were planned, what each worker
produced, what the reviewer decided. This module runs the loop on a background
thread and records every step as a `LoopEvent`, written to an append-only
`events.jsonl` a human can `tail -f` and exposed to the calling agent through the
`orchestrate_status` / `orchestrate_tail` / `orchestrate_result` tools.

Layout per run (under ``<base>/<run_id>/``, where base defaults to
``<cwd>/.agentloop/runs``):

    meta.json       goal, criteria, backend, cwd, started_at
    events.jsonl    one LoopEvent per line, append-only (the live stream)
    status.json     latest phase / iteration / running flag (atomic rewrite)
    result.json     the final result dict (written once, at the end)
    workers/        full output of each worker call: iter<N>_<subgoal>.out

The manager keeps an in-process registry so the poll tools can resolve a
``run_id`` to its writer; the files are the durable, cross-process record a human
can watch from any terminal. When the registry misses (a fresh process after the
MCP host reconnects), the poll tools fall back to a user-global run index that
maps ``run_id`` -> run dir and rehydrate status/tail/result straight off disk.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .types import (
    EVENT_PHASE_CHANGED,
    EVENT_RUN_ERROR,
    EVENT_RUN_FINISHED,
    EVENT_RUN_STARTED,
    EVENT_SUBAGENT_FINISHED,
    LoopEvent,
)

# A thunk runs the actual loop. It receives an emit callable to thread through
# the orchestrator and returns the final, JSON-serializable result dict.
Emit = Callable[[str, int, dict], None]
Thunk = Callable[[Emit], dict[str, Any]]

RUNS_DIRNAME = ".agentloop"
_PREVIEW_LIMIT = 600
_TAIL_LIMIT = 200


def run_base(cwd: Optional[str]) -> str:
    """Default location for run directories: alongside the worked-in repo."""
    root = os.path.abspath(cwd) if cwd else os.path.expanduser("~")
    return os.path.join(root, RUNS_DIRNAME, "runs")


def new_run_id() -> str:
    """A sortable, unique run id: timestamp + short random suffix."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _write_json(path: str, obj: Any) -> None:
    """Atomically (write-temp-then-rename) write a JSON file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _preview(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"


# --- cross-process run index + on-disk rehydration ---------------------------
# The in-memory registry vanishes when the server process restarts (e.g. the MCP
# host reconnects mid-run), but the run dirs persist. To resolve a run_id back to
# its dir without the caller passing a cwd, every started run appends one line to
# a stable, user-global index; read views fall back to reading the run dir off
# disk when the registry misses. See `RunManager` for how the fallback is wired.

_index_lock = threading.Lock()


def _index_path() -> str:
    # AGENTLOOP_INDEX overrides the location (tests, or pinning it to a shared
    # spot when runs span several working dirs); defaults under the user home so
    # one index spans every cwd a run was started from.
    override = os.environ.get("AGENTLOOP_INDEX")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), RUNS_DIRNAME, "index.jsonl")


def _index_record(run_id: str, run_dir: str) -> None:
    path = _index_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps({"run_id": run_id, "run_dir": run_dir}, ensure_ascii=False) + "\n"
    with _index_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _index_entries() -> list[tuple[str, str]]:
    """All indexed (run_id, run_dir) pairs, newest write wins on duplicates."""
    path = _index_path()
    if not os.path.exists(path):
        return []
    latest: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # tolerate a torn final line from a concurrent append
            rid, rdir = rec.get("run_id"), rec.get("run_dir")
            if rid and rdir:
                latest[rid] = rdir
    return list(latest.items())


def _index_lookup(run_id: str) -> Optional[str]:
    for rid, rdir in _index_entries():
        if rid == run_id:
            return rdir
    return None


def _load_json(path: str) -> Optional[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _disk_status(run_dir: str, run_id: str) -> Optional[dict[str, Any]]:
    """Reconstruct a status dict from a run dir, for a run no longer in memory.

    A rehydrated run has no live thread in this process (handles are never
    evicted, so a registry miss means a different/previous process owned it).
    Thus a recorded `running: true` with no result.json was cut off by the
    restart and can never resume — report it as a terminal `interrupted` rather
    than a live run, so callers stop polling and `result(wait=...)` won't try to
    join a thread that no longer exists.
    """
    status = _load_json(os.path.join(run_dir, "status.json"))
    if status is None:
        return None
    status["rehydrated"] = True
    result = _load_json(os.path.join(run_dir, "result.json"))
    if result is not None:
        status["running"] = False
        status["completed"] = result.get("completed")
        status["stop_reason"] = result.get("stop_reason")
    elif status.get("running"):
        status["running"] = False
        status["stop_reason"] = "interrupted"
    return status


def _disk_tail(run_dir: str, run_id: str, cursor: int, limit: int) -> Optional[dict[str, Any]]:
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(path):
        return None
    newer: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("seq", 0) > cursor:
                newer.append(ev)
    page = newer[:limit]
    status = _disk_status(run_dir, run_id) or {}
    return {
        "run_id": run_id,
        "events": page,
        "cursor": page[-1]["seq"] if page else cursor,
        "running": bool(status.get("running")),
        "more": len(newer) > len(page),
        "rehydrated": True,
    }


def _disk_result(run_dir: str, run_id: str) -> Optional[dict[str, Any]]:
    result = _load_json(os.path.join(run_dir, "result.json"))
    if result is not None:
        return result
    # No result.json: the run never finished. _disk_status turns an orphaned
    # `running` into a terminal `interrupted`; surface that (we can't fabricate
    # the full history/final_output the run would have produced).
    return _disk_status(run_dir, run_id)


class RunWriter:
    """Owns one run's on-disk record and in-memory event buffer.

    Thread-safe: workers fan out in parallel, so `emit` may be called
    concurrently. A reentrant lock guards the seq counter, the buffer, the
    append to events.jsonl, and the status rewrite.
    """

    def __init__(self, run_id: str, run_dir: str):
        self.run_id = run_id
        self.run_dir = run_dir
        self.events_path = os.path.join(run_dir, "events.jsonl")
        self._workers_dir = os.path.join(run_dir, "workers")
        self._lock = threading.RLock()
        self._seq = 0
        self.events: list[LoopEvent] = []
        self.running = True
        self.phase = "starting"
        self.iteration = 0
        self.result: Optional[dict[str, Any]] = None
        self.started_at = time.time()
        os.makedirs(self._workers_dir, exist_ok=True)

    # --- lifecycle ---------------------------------------------------------

    def begin(self, meta: dict[str, Any]) -> None:
        _write_json(os.path.join(self.run_dir, "meta.json"), meta)
        self.emit(EVENT_RUN_STARTED, 0, {
            k: meta.get(k)
            for k in ("goal", "success_criteria", "backend", "cwd", "max_iterations")
        })

    def finish(self, result: dict[str, Any]) -> None:
        with self._lock:
            self.result = result
            self.running = False
            _write_json(os.path.join(self.run_dir, "result.json"), result)
            self.emit(EVENT_RUN_FINISHED, self.iteration, {
                "completed": result.get("completed"),
                "stop_reason": result.get("stop_reason"),
                "iterations": result.get("iterations"),
                "summary": result.get("summary"),
            })

    def fail(self, exc: BaseException) -> None:
        """Record an unexpected crash so the run never just vanishes."""
        detail = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self.result = {
                "completed": False,
                "iterations": self.iteration,
                "stop_reason": "run_error",
                "final_output": detail,
                "history": [],
                "error": detail,
                "summary": f"stopped: {detail}",
            }
            self.running = False
            _write_json(os.path.join(self.run_dir, "result.json"), self.result)
            self.emit(EVENT_RUN_ERROR, self.iteration, {"error": detail})

    # --- the event sink ----------------------------------------------------

    def emit(self, kind: str, iteration: int, data: dict[str, Any]) -> LoopEvent:
        with self._lock:
            self._seq += 1
            event = LoopEvent(
                seq=self._seq, ts=time.time(), kind=kind,
                iteration=iteration, data=self._persist(kind, iteration, data),
            )
            self.events.append(event)
            if iteration:
                self.iteration = iteration
            # A phase_changed event names the loop's actual stage in data["to"];
            # use that so status reports the real phase. Every other event just
            # marks the run by its most recent moment (the event kind).
            self.phase = data.get("to", kind) if kind == EVENT_PHASE_CHANGED else kind
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_json(), ensure_ascii=False) + "\n")
            # Reentrant lock: status() re-acquires it safely on this same thread.
            _write_json(os.path.join(self.run_dir, "status.json"), self.status())
            return event

    def _persist(self, kind: str, iteration: int, data: dict[str, Any]) -> dict[str, Any]:
        """Move bulky worker output to its own file; keep the event stream light."""
        if kind == EVENT_SUBAGENT_FINISHED and "output" in data:
            output = data["output"]
            path = os.path.join(self._workers_dir, f"iter{iteration}_{data.get('id', 'task')}.out")
            with open(path, "w", encoding="utf-8") as f:
                f.write(output)
            slim = {k: v for k, v in data.items() if k != "output"}
            slim["preview"] = _preview(output)
            slim["chars"] = len(output)
            slim["output_path"] = path
            return slim
        return data

    # --- read views --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {
                "run_id": self.run_id,
                "run_dir": self.run_dir,
                "running": self.running,
                "phase": self.phase,
                "iteration": self.iteration,
                "events": self._seq,
                "started_at": self.started_at,
            }
            if self.result is not None:
                out["completed"] = self.result.get("completed")
                out["stop_reason"] = self.result.get("stop_reason")
            return out

    def tail(self, cursor: int = 0, limit: int = _TAIL_LIMIT) -> dict[str, Any]:
        with self._lock:
            newer = [e for e in self.events if e.seq > cursor]
            page = newer[:limit]
            next_cursor = page[-1].seq if page else cursor
            return {
                "run_id": self.run_id,
                "events": [e.to_json() for e in page],
                "cursor": next_cursor,
                "running": self.running,
                "more": len(newer) > len(page),
            }


@dataclass
class RunHandle:
    run_id: str
    run_dir: str
    writer: RunWriter
    thread: threading.Thread


class RunManager:
    """In-process registry of detached runs, keyed by run_id."""

    def __init__(self) -> None:
        self._runs: dict[str, RunHandle] = {}
        self._lock = threading.Lock()

    def start(self, *, thunk: Thunk, meta: dict[str, Any], base: str) -> RunHandle:
        run_id = new_run_id()
        run_dir = os.path.join(base, run_id)
        os.makedirs(run_dir, exist_ok=True)
        writer = RunWriter(run_id, run_dir)
        # Record the dir before any work so a poll after an immediate reconnect
        # can still resolve this run_id off disk.
        _index_record(run_id, run_dir)
        writer.begin(meta)

        def target() -> None:
            try:
                writer.finish(thunk(writer.emit))
            except BaseException as exc:  # noqa: BLE001 — never lose a crashed run
                writer.fail(exc)

        thread = threading.Thread(target=target, name=f"agentloop-run-{run_id}", daemon=True)
        handle = RunHandle(run_id, run_dir, writer, thread)
        with self._lock:
            self._runs[run_id] = handle
        thread.start()
        return handle

    def _get(self, run_id: str) -> Optional[RunHandle]:
        with self._lock:
            return self._runs.get(run_id)

    def _rehydrate_dir(self, run_id: str) -> Optional[str]:
        """Locate a run dir for a run_id this process never started, off the index."""
        run_dir = _index_lookup(run_id)
        return run_dir if run_dir and os.path.isdir(run_dir) else None

    def status(self, run_id: str) -> dict[str, Any]:
        handle = self._get(run_id)
        if handle is not None:
            return handle.writer.status()
        run_dir = self._rehydrate_dir(run_id)
        disk = _disk_status(run_dir, run_id) if run_dir else None
        return disk if disk is not None else _unknown(run_id)

    def tail(self, run_id: str, cursor: int = 0, limit: int = _TAIL_LIMIT) -> dict[str, Any]:
        handle = self._get(run_id)
        if handle is not None:
            return handle.writer.tail(cursor, limit)
        run_dir = self._rehydrate_dir(run_id)
        disk = _disk_tail(run_dir, run_id, cursor, limit) if run_dir else None
        return disk if disk is not None else _unknown(run_id)

    def result(
        self, run_id: str, wait: bool = False, timeout: Optional[float] = None
    ) -> dict[str, Any]:
        handle = self._get(run_id)
        if handle is None:
            # No live thread to join; serve the last persisted outcome (or the
            # interrupted state) straight from disk.
            run_dir = self._rehydrate_dir(run_id)
            disk = _disk_result(run_dir, run_id) if run_dir else None
            return disk if disk is not None else _unknown(run_id)
        if wait and handle.thread.is_alive():
            handle.thread.join(timeout)
        if handle.writer.result is not None:
            return handle.writer.result
        status = handle.writer.status()
        status["status"] = "running"
        return status

    def list_runs(self) -> dict[str, Any]:
        with self._lock:
            handles = list(self._runs.values())
        runs = [h.writer.status() for h in handles]
        seen = {h.run_id for h in handles}
        # Fold in runs from previous processes that the registry no longer holds.
        for run_id, run_dir in _index_entries():
            if run_id in seen or not os.path.isdir(run_dir):
                continue
            disk = _disk_status(run_dir, run_id)
            if disk is not None:
                runs.append(disk)
                seen.add(run_id)
        return {"runs": runs}


def _unknown(run_id: str) -> dict[str, Any]:
    return {"error": f"unknown run_id {run_id!r}", "run_id": run_id}


# One registry per server process. Detached runs live here while the process runs;
# the run index + each run's events.jsonl / result.json are the durable record that
# outlives it, so the poll tools rehydrate a run_id from disk after a restart.
MANAGER = RunManager()
