# SPDX-License-Identifier: Apache-2.0
"""Append-only JSONL snapshots of vLLM scheduler queue depths (engine process).

When ``SENDNN_INFERENCE_SCHEDULER_QUEUE_TRACE_PATH`` is set, the Spyre chunked-prefill
scheduler emits queue depths on each ``schedule()`` call so you can debug async vs sync.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

_path_checked: str | None = None
_path_active: str | None = None


def _trace_path() -> str | None:
    global _path_checked, _path_active
    p = os.environ.get("SENDNN_INFERENCE_SCHEDULER_QUEUE_TRACE_PATH", "").strip()
    if p != _path_checked:
        _path_checked = p
        _path_active = p if p else None
    return _path_active


def _include_request_ids() -> bool:
    return os.environ.get("SENDNN_INFERENCE_SCHEDULER_QUEUE_TRACE_INCLUDE_IDS", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


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
    }
    row.update(fields)
    try:
        _append_line(path, row)
    except OSError:
        return


def _skipped_waiting_len(scheduler: Any) -> int:
    skipped = getattr(scheduler, "skipped_waiting", None)
    if skipped is None:
        return 0
    try:
        return len(skipped)
    except TypeError:
        return 0


def _req_ids(requests: Any) -> list[str]:
    out: list[str] = []
    for r in requests:
        rid = getattr(r, "request_id", None)
        if rid is not None:
            out.append(str(rid))
    return out


def emit_chunked_prefill_snapshot(
    phase: str,
    scheduler: Any,
    *,
    step: int,
    holdback_len: int = 0,
    running_holdback_len: int = 0,
    outputs: Any | None = None,
) -> None:
    """Record queue depths for ``ChunkedPrefillSpyreScheduler`` scheduling."""
    if not _trace_path():
        return

    inflight = getattr(scheduler, "_inflight_prefill_tokens", None) or {}
    row: dict[str, Any] = {
        "phase": phase,
        "step": step,
        "running": len(scheduler.running),
        "waiting": len(scheduler.waiting),
        "holdback": holdback_len,
        "running_holdback": running_holdback_len,
        "ongoing_prefills": len(getattr(scheduler, "ongoing_prefills", []) or []),
        "skipped_waiting": _skipped_waiting_len(scheduler),
        "inflight_prefill_reqs": len(inflight),
    }

    if outputs is not None:
        row["total_scheduled_tokens"] = int(getattr(outputs, "total_num_scheduled_tokens", 0) or 0)
        num_sched = getattr(outputs, "num_scheduled_tokens", None) or {}
        row["num_scheduled_reqs"] = len(num_sched)
        if _include_request_ids():
            row["scheduled_req_ids"] = list(num_sched.keys())

    if _include_request_ids():
        row["running_req_ids"] = _req_ids(scheduler.running)
        row["waiting_req_ids"] = _req_ids(scheduler.waiting)

    emit("engine.scheduler_queues", **row)


def load_snapshots(path: os.PathLike[str] | str) -> list[dict[str, Any]]:
    p = os.fspath(path)
    if not os.path.isfile(p):
        return []
    rows: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "engine.scheduler_queues":
                rows.append(row)
    rows.sort(key=lambda r: (int(r.get("step", 0)), str(r.get("phase", "")), int(r.get("ts_ns", 0))))
    return rows


def _rel_ms(ts_ns: int, t0_ns: int) -> float:
    return (ts_ns - t0_ns) / 1_000_000.0


def format_timeline_table(rows: list[dict[str, Any]], max_rows: int | None = None) -> str:
    if not rows:
        return "(no scheduler queue snapshots — is SENDNN_INFERENCE_SCHEDULER_QUEUE_TRACE_PATH set?)"

    t0 = int(rows[0].get("ts_ns", 0))
    lines = [
        "step | t_ms   | phase          | run | wait | hold | rbk | pref | skip | sched_tok | #sched",
        "-----+--------+---------------+-----+------+------+-----+------+------+-----------+-------",
    ]
    shown = rows if max_rows is None else rows[:max_rows]
    for r in shown:
        lines.append(
            f"{int(r.get('step', 0)):4d} | "
            f"{_rel_ms(int(r.get('ts_ns', 0)), t0):6.1f} | "
            f"{str(r.get('phase', '')):13s} | "
            f"{int(r.get('running', 0)):3d} | "
            f"{int(r.get('waiting', 0)):4d} | "
            f"{int(r.get('holdback', 0)):4d} | "
            f"{int(r.get('running_holdback', 0)):3d} | "
            f"{int(r.get('ongoing_prefills', 0)):4d} | "
            f"{int(r.get('skipped_waiting', 0)):4d} | "
            f"{int(r.get('total_scheduled_tokens', 0)):9d} | "
            f"{int(r.get('num_scheduled_reqs', 0)):5d}"
        )
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows in JSONL)")
    return "\n".join(lines)


def summarize_queue_trace(path: os.PathLike[str] | str) -> str:
    rows = load_snapshots(path)
    if not rows:
        return "No queue snapshots found."

    end_rows = [r for r in rows if r.get("phase") == "schedule_end"]
    pre_rows = [r for r in rows if r.get("phase") == "pre_delegate"]
    max_running = max(int(r.get("running", 0)) for r in rows)
    max_waiting = max(int(r.get("waiting", 0)) for r in rows)
    max_holdback = max(int(r.get("holdback", 0)) for r in rows)
    steps = len({int(r.get("step", -1)) for r in rows if "step" in r})

    lines = [
        f"schedule() steps with snapshots: {steps}",
        f"peak running (visible to base scheduler): {max_running}",
        f"peak waiting: {max_waiting}",
        f"peak holdback (not in waiting yet): {max_holdback}",
        f"snapshots per step: typically {len(rows) // max(steps, 1)} rows",
    ]
    if end_rows:
        nonempty = sum(1 for r in end_rows if int(r.get("total_scheduled_tokens", 0)) > 0)
        lines.append(f"schedule_end with tokens scheduled: {nonempty} / {len(end_rows)}")
    if pre_rows:
        both = sum(
            1
            for r in pre_rows
            if int(r.get("waiting", 0)) > 0 and int(r.get("running_holdback", 0)) > 0
        )
        lines.append(
            f"pre_delegate: waiting>0 while decodes held back (running_holdback>0): "
            f"{both} / {len(pre_rows)}"
        )
    return "\n".join(lines)
