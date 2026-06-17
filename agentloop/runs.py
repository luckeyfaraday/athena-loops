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
can watch from any terminal.
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
            self.phase = kind
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

    def status(self, run_id: str) -> dict[str, Any]:
        handle = self._get(run_id)
        if handle is None:
            return _unknown(run_id)
        return handle.writer.status()

    def tail(self, run_id: str, cursor: int = 0, limit: int = _TAIL_LIMIT) -> dict[str, Any]:
        handle = self._get(run_id)
        if handle is None:
            return _unknown(run_id)
        return handle.writer.tail(cursor, limit)

    def result(
        self, run_id: str, wait: bool = False, timeout: Optional[float] = None
    ) -> dict[str, Any]:
        handle = self._get(run_id)
        if handle is None:
            return _unknown(run_id)
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
        return {"runs": [h.writer.status() for h in handles]}


def _unknown(run_id: str) -> dict[str, Any]:
    return {"error": f"unknown run_id {run_id!r}", "run_id": run_id}


# One registry per server process. Detached runs live here until the process exits;
# the events.jsonl / result.json files are the durable record that outlives it.
MANAGER = RunManager()
