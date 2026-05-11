#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile vLLM + Spyre inference with PyTorch profiler (container-friendly).

Two modes:

1) **Default** — ``torch.profiler`` around ``LLM.generate()`` on this process.
   Good for seeing CPU stacks in the driver / in-process engine path (TP=1).
   For multi-worker setups, most kernel time may live in child processes; use
   mode (2) or ``py-spy`` on worker PIDs in that case.

2) **``--vllm-torch-profiler``** — vLLM's built-in Torch profiler (same idea as
   ``tests/e2e/test_profiler.py``), which can capture worker-side activity
   depending on vLLM version and executor layout.

**Scheduler visibility (async vs sync):** the default profiler wraps
``LLM.generate()``, so you see whatever runs in that process. To label Spyre
chunked-prefill scheduler time in the Chrome trace (including how much is
``super().schedule()`` vs ``get_grammar_bitmask``), set::

    export SPYRE_PROFILE_SCHEDULER=1

This enables ``torch.profiler.record_function`` spans in
``sendnn_inference/v1/core/scheduler.py``. For multi-process engines, the
scheduler may run on a different process than the one ``torch.profiler`` wraps;
use ``py-spy`` on the engine PID in that case.

Examples (inside the repo / container with vLLM + sendnn_inference on PYTHONPATH)::

    export SENDNN_INFERENCE_DYNAMO_BACKEND=eager
    python scripts/profile_llm_torch.py \\
        --model ibm-granite/granite-3.3-8b-instruct \\
        --max-model-len 512 --max-num-seqs 4 --max-num-batched-tokens 128 \\
        --output-dir /tmp/torch_prof

    python scripts/profile_llm_torch.py \\
        --model ibm-granite/granite-3.3-8b-instruct \\
        --async-scheduling --batch-queue-size 2 \\
        --output-dir /tmp/torch_prof_async

Chrome trace (default mode): ``chrome_trace.json`` under ``--output-dir``.
Open in ``chrome://tracing`` or Edge's ``edge://tracing``.

Builtin mode writes ``*.pt.trace.json`` under ``--output-dir``; open in
https://ui.perfetto.dev/ or PyTorch TensorBoard plugin.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Safer defaults when not running under pytest (matches tests/conftest.py intent).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, required=True, help="HF model id or local path")
    p.add_argument("--revision", type=str, default=None, help="Optional model revision")
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=128)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--dynamo-backend",
        type=str,
        default=os.environ.get("SENDNN_INFERENCE_DYNAMO_BACKEND", "eager"),
        help="Sets SENDNN_INFERENCE_DYNAMO_BACKEND for this process (default: env or eager)",
    )
    p.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, passed through to LLM(...) when supported (omit to use vLLM defaults)",
    )
    p.add_argument(
        "--batch-queue-size",
        type=int,
        default=None,
        help="Optional batch_queue_size for async scheduling experiments",
    )
    p.add_argument("--warmup-generations", type=int, default=1, help="Unprofiled generate calls before profiling")
    p.add_argument("--profile-generations", type=int, default=1, help="Profiled generate calls")
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=128,
        help="Synthetic prompt length (repeated token id 42)",
    )
    p.add_argument("--max-tokens", type=int, default=32, help="Sampling max_tokens per request")
    p.add_argument("--num-prompts", type=int, default=2, help="Batch size for generate()")
    p.add_argument("--output-dir", type=Path, default=Path("torch_profile_out"))
    p.add_argument(
        "--with-cuda",
        action="store_true",
        help="Include ProfilerActivity.CUDA (no-op if no CUDA in container)",
    )
    p.add_argument("--record-shapes", action="store_true", help="Torch profiler record_shapes (more overhead)")
    p.add_argument("--profile-memory", action="store_true", help="Torch profiler profile_memory")
    p.add_argument(
        "--vllm-torch-profiler",
        action="store_true",
        help="Use vLLM ProfilerConfig + start_profile/stop_profile instead of wrapping generate()",
    )
    p.add_argument(
        "--table-rows",
        type=int,
        default=40,
        help="Rows in printed torch.profiler summary table",
    )
    return p.parse_args()


def _build_prompts(args: argparse.Namespace) -> list[str]:
    # Cheap synthetic text; length controlled roughly by repeating a word.
    word = " benchmark"
    text = word * max(1, args.prompt_tokens // len(word))
    return [f"{args.model} {i} {text}" for i in range(args.num_prompts)]


def _make_llm(args: argparse.Namespace):
    from vllm import LLM
    from vllm import SamplingParams

    os.environ["SENDNN_INFERENCE_DYNAMO_BACKEND"] = args.dynamo_backend

    llm_kwargs: dict = {
        "model": args.model,
        "tokenizer": args.model,
        "revision": args.revision,
        "tokenizer_revision": args.revision,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_prefix_caching": False,
    }
    if args.async_scheduling is not None:
        llm_kwargs["async_scheduling"] = args.async_scheduling
    if args.batch_queue_size is not None:
        llm_kwargs["batch_queue_size"] = args.batch_queue_size

    if args.vllm_torch_profiler:
        from vllm.config import ProfilerConfig

        llm_kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(args.output_dir.resolve()),
        )

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, ignore_eos=True)
    return llm, sampling


def _run_vllm_builtin_profiler(llm, sampling, prompts: list[str]) -> None:
    llm.start_profile()
    try:
        llm.generate(prompts, sampling_params=sampling)
    finally:
        llm.stop_profile()


def _run_torch_profiler_wrap(llm, sampling, prompts: list[str], args: argparse.Namespace) -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    args.output_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if args.with_cuda and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    elif args.with_cuda:
        print("Warning: --with-cuda set but torch.cuda.is_available() is False; CPU only.", file=sys.stderr)

    trace_path = args.output_dir / "chrome_trace.json"

    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=True,
    ) as prof:
        llm.generate(prompts, sampling_params=sampling)

    prof.export_chrome_trace(str(trace_path))
    print(f"Wrote Chrome trace: {trace_path.resolve()}")
    print()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.table_rows))


def main() -> int:
    import torch

    args = _parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = _build_prompts(args)
    llm, sampling = _make_llm(args)

    for _ in range(max(0, args.warmup_generations)):
        llm.generate(prompts, sampling_params=sampling)

    if args.vllm_torch_profiler:
        for i in range(max(1, args.profile_generations)):
            _run_vllm_builtin_profiler(llm, sampling, prompts)
            print(
                f"vLLM torch profiler run {i + 1}/{args.profile_generations} complete; "
                f"traces under {args.output_dir}"
            )
        traces = list(args.output_dir.glob("*.pt.trace.json*"))
        if not traces:
            print(
                "No *.pt.trace.json* found yet (vLLM may write async). "
                "Check directory after process exits.",
                file=sys.stderr,
            )
        else:
            for t in traces[:20]:
                print(f"  trace: {t}")
        return 0

    for i in range(max(1, args.profile_generations)):
        t0 = time.perf_counter()
        _run_torch_profiler_wrap(llm, sampling, prompts, args)
        dt = time.perf_counter() - t0
        print(f"Profiled generate {i + 1}/{args.profile_generations} in {dt:.2f}s")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
