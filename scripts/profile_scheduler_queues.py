#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Spyre + vLLM and record scheduler queue depths over time.

Each ``ChunkedPrefillSpyreScheduler.schedule()`` appends JSONL rows with
``running``, ``waiting``, ``holdback``, ``running_holdback``, and
``ongoing_prefills`` counts. Use this to compare async vs sync scheduling.

Example (async, TP=4)::

    python scripts/profile_scheduler_queues.py \\
        -tp 4 \\
        --model ibm-granite/granite-3.3-8b-instruct \\
        --num-prompts 4 \\
        --max-tokens 16 \\
        --output-dir /tmp/async_queues

    python scripts/profile_scheduler_queues.py \\
        --sync -tp 4 \\
        --model ibm-granite/granite-3.3-8b-instruct \\
        --num-prompts 4 \\
        --output-dir /tmp/sync_queues

Outputs: ``scheduler_queue_trace.jsonl``, ``scheduler_queue_timeline.tsv``
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
import socket
import sys
import time
from pathlib import Path


def _engine_args_param_names() -> set[str]:
    try:
        from vllm.engine.arg_utils import EngineArgs

        return set(inspect.signature(EngineArgs.__init__).parameters)
    except Exception:
        return set()


def _apply_engine_scheduler_kwargs(kwargs: dict, args: argparse.Namespace) -> None:
    names = _engine_args_param_names()
    if "async_scheduling" in names:
        kwargs["async_scheduling"] = not args.sync
    elif not args.sync:
        print(
            "Warning: EngineArgs has no 'async_scheduling' — using default scheduling.",
            file=sys.stderr,
        )
    if args.sync:
        return
    if "batch_queue_size" in names:
        kwargs["batch_queue_size"] = args.batch_queue_size
    else:
        print(
            f"Warning: EngineArgs has no 'batch_queue_size' (requested {args.batch_queue_size}).",
            file=sys.stderr,
        )


def _pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _configure_dist_env(master_port: int | None) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)
        return
    prev = os.environ.get("MASTER_PORT", "")
    chosen = _pick_free_tcp_port()
    os.environ["MASTER_PORT"] = str(chosen)
    if prev and prev != str(chosen):
        print(f"Note: MASTER_PORT was {prev!r}; using {chosen}.", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=128)
    p.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, metavar="N")
    p.add_argument("--no-multiproc", action="store_true")
    p.add_argument(
        "--dynamo-backend",
        type=str,
        default=os.environ.get("SENDNN_INFERENCE_DYNAMO_BACKEND", "eager"),
    )
    p.add_argument("--sync", action="store_true")
    p.add_argument("--batch-queue-size", type=int, default=2)
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--sequential", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("scheduler_queue_profile"))
    p.add_argument("--master-port", type=int, default=None)
    p.add_argument("--include-req-ids", action="store_true")
    p.add_argument("--max-table-rows", type=int, default=120)
    p.add_argument("--no-request-metrics", action="store_true")
    return p.parse_args()


def _build_prompts(args: argparse.Namespace) -> list[str]:
    word = " benchmark"
    text = word * max(1, args.prompt_tokens // len(word))
    return [f"{args.model} idx={i} {text}" for i in range(args.num_prompts)]


def _llm_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {
        "model": args.model,
        "tokenizer": args.model,
        "revision": args.revision,
        "tokenizer_revision": args.revision,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
    }
    _apply_engine_scheduler_kwargs(kwargs, args)
    return kwargs


def _write_timeline_tsv(rows: list[dict], path: Path) -> None:
    fields = [
        "step",
        "t_ms",
        "phase",
        "running",
        "waiting",
        "holdback",
        "running_holdback",
        "ongoing_prefills",
        "skipped_waiting",
        "inflight_prefill_reqs",
        "total_scheduled_tokens",
        "num_scheduled_reqs",
    ]
    if not rows:
        path.write_text("")
        return
    t0 = int(rows[0].get("ts_ns", 0))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            ts = int(r.get("ts_ns", 0))
            w.writerow(
                {
                    "step": r.get("step", ""),
                    "t_ms": f"{(ts - t0) / 1_000_000.0:.3f}",
                    "phase": r.get("phase", ""),
                    "running": r.get("running", ""),
                    "waiting": r.get("waiting", ""),
                    "holdback": r.get("holdback", ""),
                    "running_holdback": r.get("running_holdback", ""),
                    "ongoing_prefills": r.get("ongoing_prefills", ""),
                    "skipped_waiting": r.get("skipped_waiting", ""),
                    "inflight_prefill_reqs": r.get("inflight_prefill_reqs", ""),
                    "total_scheduled_tokens": r.get("total_scheduled_tokens", ""),
                    "num_scheduled_reqs": r.get("num_scheduled_reqs", ""),
                }
            )


def main() -> int:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    args = _parse_args()
    if args.tensor_parallel_size > 1:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" if args.no_multiproc else "1"

    _configure_dist_env(args.master_port)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["SENDNN_INFERENCE_DYNAMO_BACKEND"] = args.dynamo_backend

    queue_path = output_dir / "scheduler_queue_trace.jsonl"
    try:
        if queue_path.exists():
            queue_path.unlink()
    except OSError:
        pass
    os.environ["SENDNN_INFERENCE_SCHEDULER_QUEUE_TRACE_PATH"] = str(queue_path.resolve())
    if args.include_req_ids:
        os.environ["SENDNN_INFERENCE_SCHEDULER_QUEUE_TRACE_INCLUDE_IDS"] = "1"
    else:
        os.environ.pop("SENDNN_INFERENCE_SCHEDULER_QUEUE_TRACE_INCLUDE_IDS", None)

    if not args.no_request_metrics:
        os.environ["SENDNN_INFERENCE_PERF_METRIC_LOGGING_ENABLED"] = "1"
        os.environ["SENDNN_INFERENCE_PERF_METRIC_LOGGING_DIR"] = str(output_dir)

    from vllm import LLM, SamplingParams

    mode = "sync (--sync)" if args.sync else "async (default)"
    print(
        f"profile_scheduler_queues: {mode} tp={args.tensor_parallel_size} -> {queue_path}",
        file=sys.stderr,
    )

    llm = LLM(**_llm_kwargs(args))
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, ignore_eos=True)
    prompts = _build_prompts(args)

    t0 = time.perf_counter()
    if args.sequential:
        for p in prompts:
            llm.generate([p], sampling_params=sampling)
    else:
        llm.generate(prompts, sampling_params=sampling)
    wall = time.perf_counter() - t0
    print(f"generate() wall time: {wall:.3f}s\n")

    from sendnn_inference.perf.queue_trace import (
        format_timeline_table,
        load_snapshots,
        summarize_queue_trace,
    )

    rows = load_snapshots(queue_path)
    tsv_path = output_dir / "scheduler_queue_timeline.tsv"
    _write_timeline_tsv(rows, tsv_path)

    print("=" * 72)
    print("Scheduler queue trace summary")
    print("=" * 72)
    print(summarize_queue_trace(queue_path))
    print()
    print(f"JSONL: {queue_path.resolve()}")
    print(f"TSV:   {tsv_path.resolve()}")
    print("=" * 72)
    print()
    max_rows = None if args.max_table_rows <= 0 else args.max_table_rows
    print(format_timeline_table(rows, max_rows=max_rows))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
