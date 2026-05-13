# SPDX-License-Identifier: Apache-2.0
"""Append-only JSONL events for engine vs worker timeline (async overlap).

When ``SENDNN_INFERENCE_OVERLAP_TRACE_PATH`` is set to a file path, participating
call sites emit one JSON object per line with ``time.time_ns()`` timestamps so
downstream tools can check whether the Spyre scheduler ran while a worker
forward was still in flight (multiproc + batch queue).

This complements vLLM's torch Chrome traces, which are process-local.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from typing import Any

_path_checked: str | None = None
_path_active: str | None = None


def _trace_path() -> str | None:
    global _path_checked, _path_active
    p = os.environ.get("SENDNN_INFERENCE_OVERLAP_TRACE_PATH", "").strip()
    if p != _path_checked:
        _path_checked = p
        _path_active = p if p else None
    return _path_active


def _append_line(path: str, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, separators=(",", ":"), sort_keys=False) + "\n"
    try:
        import fcntl  # type: ignore[import-not-found]

        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode())
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    except (ImportError, OSError):
        with open(path, "a", encoding="utf-8", buffering=1) as f:
            f.write(line)


def emit(event: str, **fields: Any) -> None:
    path = _trace_path()
    if not path:
        return
    row: dict[str, Any] = {
        "ts_ns": time.time_ns(),
        "event": event,
        "pid": os.getpid(),
        "proc": multiprocessing.current_process().name,
    }
    row.update(fields)
    try:
        _append_line(path, row)
    except OSError:
        # Tracing must never break inference.
        return


def analyze_overlap_jsonl(
    path: os.PathLike[str] | str,
) -> tuple[int, int, int, int]:
    """Return counts for overlap interpretation.

    Returns:
        (schedule_begins_during_execute, total_schedule_begins,
         execute_begins, execute_ends)
    """
    p = os.fspath(path)
    if not os.path.isfile(p):
        return 0, 0, 0, 0
    rows: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: int(r.get("ts_ns", 0)))

    execute_depth = 0
    overlap_schedule_begins = 0
    total_schedule_begins = 0
    execute_begins = 0
    execute_ends = 0

    for r in rows:
        ev = r.get("event", "")
        if ev == "engine.spy_schedule_begin":
            total_schedule_begins += 1
            if execute_depth > 0:
                overlap_schedule_begins += 1
        elif ev == "worker.execute_model_begin":
            execute_begins += 1
            execute_depth += 1
        elif ev == "worker.execute_model_end":
            execute_ends += 1
            execute_depth = max(0, execute_depth - 1)

    return overlap_schedule_begins, total_schedule_begins, execute_begins, execute_ends
