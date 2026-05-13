#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-request timing under **async scheduling** + optional Torch traces.

Why this exists
---------------
PyTorch / vLLM traces (Perfetto) show **threads and stacks**; they do not print a
clean **per-request** table. This script runs Spyre + vLLM with async scheduling
enabled and writes **one JSON line per finished request** using the existing
``FileStatLogger`` (``SENDNN_INFERENCE_PERF_METRIC_LOGGING_*``), then prints a
summary table.

vLLM profiling references
-------------------------
* Legacy env (v0.8.x style): set ``VLLM_TORCH_PROFILER_DIR`` and open traces in
  `Perfetto <https://ui.perfetto.dev/>`__ — see
  https://docs.vllm.ai/en/v0.8.5/contributing/profiling/profiling_index.html
* Newer vLLM: ``ProfilerConfig(profiler="torch", torch_profiler_dir=...)`` plus
  ``start_profile`` / ``stop_profile`` — see stable contributing/profiling.

This script supports **both**: use ``--legacy-torch-profiler-env`` for the env
var, or ``--vllm-torch-profiler`` for ``ProfilerConfig`` (when your vLLM build
has it). You can use neither and only collect per-request JSONL.

Requirements
--------------
* ``sendnn_inference`` on ``PYTHONPATH`` and Spyre platform registered (normal
  Spyre offline run).
* For JSONL: ``SENDNN_INFERENCE_PERF_METRIC_LOGGING_ENABLED=1`` is set by this
  script into ``--output-dir``; the platform patch must run (first ``LLM`` init).

**torch.distributed / EngineCore:** if you see ``EADDRINUSE`` on ``MASTER_PORT``,
this script picks a **free local port** each run (unless you pass ``--master-port``).
Unset a stale ``MASTER_PORT`` in your shell or reuse the script default.

Example::

    export SENDNN_INFERENCE_DYNAMO_BACKEND=eager
    python scripts/profile_async_per_request.py \\
        --model ibm-granite/granite-3.3-8b-instruct \\
        --batch-queue-size 2 \\
        --num-prompts 4 \\
        --output-dir /tmp/async_req_prof \\
        --vllm-torch-profiler \\
        --profile-prefix myrun

Outputs under ``--output-dir``::

    request_metrics.jsonl   # one JSON object per finished request (not Perfetto)
    per_request_summary.tsv # flat columns (not Perfetto)
    torch_traces/*.pt.trace.json*   # Chrome trace → open at https://ui.perfetto.dev/
    torch_traces_legacy_env/...     # same, if ``--legacy-torch-profiler-env``
    engine_worker_overlap.jsonl     # engine vs worker wall-clock events (unless --no-overlap-trace)

**Perfetto:** only the ``*.pt.trace.json*`` files are Chrome traces. In the browser
use **Open trace file** (or drag-and-drop). ``request_metrics.jsonl`` is separate
per-request stats from ``FileStatLogger`` — use ``jq``/a spreadsheet, not Perfetto.

Engine vs worker overlap (multiproc / async batch queue)
--------------------------------------------------------
vLLM's torch Chrome traces are **per process**; they do not by themselves prove that
the **EngineCore scheduler** ran while a **worker** ``execute_model`` was still in
flight. When overlap capture is enabled (default), this script sets
``SENDNN_INFERENCE_OVERLAP_TRACE_PATH`` so the Spyre scheduler and worker emit
**wall-clock JSONL** events into ``engine_worker_overlap.jsonl``. After the run it
prints how often ``engine.spy_schedule_begin`` occurred while
``worker.execute_model_*`` indicated an in-flight forward — the quantity Joe asked
to verify with profiling instead of guessing from benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import socket
import sys
import time
from pathlib import Path


def _engine_args_param_names() -> set[str]:
    """Parameter names accepted by ``EngineArgs.__init__`` (vLLM version-dependent)."""
    try:
        from vllm.engine.arg_utils import EngineArgs

        return set(inspect.signature(EngineArgs.__init__).parameters)
    except Exception:
        return set()


def _apply_async_engine_kwargs(kwargs: dict, args: argparse.Namespace) -> None:
    """Only pass ``async_scheduling`` / ``batch_queue_size`` if this vLLM supports them."""
    names = _engine_args_param_names()
    if "async_scheduling" in names:
        kwargs["async_scheduling"] = True
    else:
        print(
            "Warning: EngineArgs has no 'async_scheduling' — running with default scheduling "
            "(upgrade vLLM for async_scheduling + batch_queue_size).",
            file=sys.stderr,
        )
    if "batch_queue_size" in names:
        kwargs["batch_queue_size"] = args.batch_queue_size
    else:
        print(
            "Warning: EngineArgs has no 'batch_queue_size' — argument ignored "
            f"(requested {args.batch_queue_size}).",
            file=sys.stderr,
        )


def _pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _configure_dist_env(master_port: int | None) -> None:
    """EngineCore uses env://; a busy inherited MASTER_PORT causes EADDRINUSE."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)
        return
    prev = os.environ.get("MASTER_PORT", "")
    chosen = _pick_free_tcp_port()
    os.environ["MASTER_PORT"] = str(chosen)
    if prev and prev != str(chosen):
        print(
            f"Note: MASTER_PORT was {prev!r} (often busy); using {chosen} for torch.distributed.",
            file=sys.stderr,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=128)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dynamo-backend", type=str, default=os.environ.get("SENDNN_INFERENCE_DYNAMO_BACKEND", "eager"))
    p.add_argument(
        "--batch-queue-size",
        type=int,
        default=2,
        help="Async batch queue depth (only if EngineArgs supports batch_queue_size)",
    )
    p.add_argument(
        "--warmup-generations",
        type=int,
        default=0,
        help="Extra generate() rounds before the measured run (writes extra JSONL lines).",
    )
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument(
        "--sequential",
        action="store_true",
        help="Call generate() once per prompt (clearer wall clock per client request; less batching stress).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("async_per_request_profile"))
    p.add_argument(
        "--vllm-torch-profiler",
        action="store_true",
        help="Use vLLM ProfilerConfig + start_profile/stop_profile (newer vLLM).",
    )
    p.add_argument(
        "--legacy-torch-profiler-env",
        action="store_true",
        help="Set VLLM_TORCH_PROFILER_DIR under output-dir (v0.8.x style; see vLLM profiling docs).",
    )
    p.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="MASTER_PORT for torch.distributed (default: pick a free port; avoids EADDRINUSE).",
    )
    p.add_argument(
        "--profile-prefix",
        type=str,
        default=None,
        help="Optional prefix passed to start_profile(profile_prefix=...) so trace "
        "filenames are easy to spot (vLLM builds that support it).",
    )
    p.add_argument(
        "--no-overlap-trace",
        action="store_true",
        help="Disable engine/worker overlap JSONL (see SENDNN_INFERENCE_OVERLAP_TRACE_PATH).",
    )
    p.add_argument(
        "--table-width",
        type=int,
        default=120,
        help="Truncate wide rows when printing to the terminal.",
    )
    return p.parse_args()


def _build_prompts(args: argparse.Namespace) -> list[str]:
    word = " benchmark"
    text = word * max(1, args.prompt_tokens // len(word))
    return [f"{args.model} idx={i} {text}" for i in range(args.num_prompts)]


def _llm_kwargs(args: argparse.Namespace, output_dir: Path) -> dict:
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
    _apply_async_engine_kwargs(kwargs, args)

    if args.vllm_torch_profiler:
        try:
            from vllm.config import ProfilerConfig

            torch_dir = output_dir / "torch_traces"
            torch_dir.mkdir(parents=True, exist_ok=True)
            kwargs["profiler_config"] = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=str(torch_dir.resolve()),
            )
        except Exception as e:  # pragma: no cover - version skew
            print(f"Warning: ProfilerConfig unavailable ({e}); skip --vllm-torch-profiler.", file=sys.stderr)

    return kwargs


def _start_profile(llm, profile_prefix: str | None) -> None:
    if profile_prefix is None:
        llm.start_profile()
        return
    try:
        sig = inspect.signature(llm.start_profile)
    except (TypeError, ValueError):
        llm.start_profile()
        return
    if "profile_prefix" in sig.parameters:
        llm.start_profile(profile_prefix=profile_prefix)
    else:
        print(
            "Warning: LLM.start_profile has no profile_prefix=; starting profile without prefix.",
            file=sys.stderr,
        )
        llm.start_profile()


def _run_generate(
    llm,
    sampling,
    prompts: list[str],
    sequential: bool,
    use_vllm_prof: bool,
    profile_prefix: str | None,
) -> None:
    if use_vllm_prof:
        _start_profile(llm, profile_prefix)
    try:
        if sequential:
            for p in prompts:
                llm.generate([p], sampling_params=sampling)
        else:
            llm.generate(prompts, sampling_params=sampling)
    finally:
        if use_vllm_prof:
            llm.stop_profile()


def _collect_chrome_traces(*roots: Path) -> list[Path]:
    """Chrome trace blobs vLLM/torch may write (Perfetto-compatible)."""
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.pt.trace.json", "*.pt.trace.json.gz"):
            out.extend(sorted(root.rglob(pattern)))
    # de-dupe, stable order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _print_engine_worker_overlap_report(overlap_path: Path | None) -> None:
    """Summarize JSONL from ``sendnn_inference.perf.overlap_trace``."""
    if overlap_path is None:
        return
    from sendnn_inference.perf.overlap_trace import analyze_overlap_jsonl

    print("\n" + "=" * 72)
    print("Engine vs worker overlap (multiproc / async queue evidence)")
    print("=" * 72)
    print(f"  Raw events: {overlap_path.resolve()}")
    print(
        "  Interpretation: `schedule_begin while execute_model in flight` counts how "
        "often the Spyre scheduler entered `schedule()` while at least one worker "
        "`execute_model` had started and not yet finished (wall-clock `time.time_ns`)."
    )
    ovl, total_s, ex_b, ex_e = analyze_overlap_jsonl(overlap_path)
    if total_s == 0 and ex_b == 0:
        print("  (no overlap events captured — engine may not use Spyre scheduler, or trace disabled.)")
    else:
        frac = ovl / total_s if total_s else 0.0
        print(f"  schedule_begin while worker execute_model in flight: {ovl} / {total_s}  ({frac:.1%})")
        print(f"  worker execute_model_begin / end: {ex_b} / {ex_e}")
        if ex_b != ex_e:
            print(
                "  Warning: begin/end mismatch (truncated run, crash, or missing events).",
                file=sys.stderr,
            )
    print("=" * 72 + "\n")


def _print_perfetto_instructions(traces: list[Path]) -> None:
    url = "https://ui.perfetto.dev/"
    print("\n" + "=" * 72)
    print("Perfetto (Chrome trace)")
    print("=" * 72)
    print(f"  1. Open {url}")
    print('  2. Use "Open trace file" or drag the trace onto the page.')
    print("  3. Pick a file ending in .pt.trace.json or .pt.trace.json.gz below.\n")
    if traces:
        print("Trace files:")
        for t in traces:
            print(f"  {t}")
    else:
        print(
            "  (none found yet — ensure you passed --vllm-torch-profiler or "
            "--legacy-torch-profiler-env; workers may flush traces shortly after exit.)"
        )
    print("=" * 72 + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _priority_columns() -> list[str]:
    """Stable-ish column order for TSV / print (all keys still preserved in JSONL)."""
    return [
        "request_id",
        "e2e_latency_seconds",
        "queued_time_seconds",
        "prefill_time_seconds",
        "inference_time_seconds",
        "decode_time_seconds",
        "mean_time_per_output_token_seconds",
        "num_prompt_tokens",
        "num_generation_tokens",
        "prefill_interrupt_seconds",
        "decode_only_itl_seconds",
        "timestamp",
    ]


def _write_tsv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    for c in reversed(_priority_columns()):
        if c in keys:
            keys.remove(c)
            keys.insert(0, c)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _print_table(rows: list[dict], max_width: int) -> None:
    if not rows:
        print("No rows in request_metrics.jsonl (stats logger may be off or no requests finished).")
        return
    cols = [c for c in _priority_columns() if any(c in r for r in rows)]
    if not cols:
        cols = sorted({k for r in rows for k in r})
    header = " | ".join(cols)
    print(header)
    print("-" * min(max_width, len(header)))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            s = str(v)
            if len(s) > 24:
                s = s[:21] + "..."
            cells.append(s)
        line = " | ".join(cells)
        if len(line) > max_width:
            line = line[: max_width - 3] + "..."
        print(line)


def main() -> int:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    args = _parse_args()
    _configure_dist_env(args.master_port)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["SENDNN_INFERENCE_DYNAMO_BACKEND"] = args.dynamo_backend
    os.environ["SENDNN_INFERENCE_PERF_METRIC_LOGGING_ENABLED"] = "1"
    os.environ["SENDNN_INFERENCE_PERF_METRIC_LOGGING_DIR"] = str(output_dir)

    overlap_path: Path | None = None
    if not args.no_overlap_trace:
        overlap_path = output_dir / "engine_worker_overlap.jsonl"
        try:
            if overlap_path.exists():
                overlap_path.unlink()
        except OSError:
            pass
        os.environ["SENDNN_INFERENCE_OVERLAP_TRACE_PATH"] = str(overlap_path.resolve())
    else:
        os.environ.pop("SENDNN_INFERENCE_OVERLAP_TRACE_PATH", None)

    legacy_dir = output_dir / "torch_traces_legacy_env"
    if args.legacy_torch_profiler_env:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_TORCH_PROFILER_DIR"] = str(legacy_dir.resolve())
        print(f"Set VLLM_TORCH_PROFILER_DIR={os.environ['VLLM_TORCH_PROFILER_DIR']}")

    if args.vllm_torch_profiler and args.legacy_torch_profiler_env:
        print("Note: both --vllm-torch-profiler and --legacy-torch-profiler-env set; traces may go to two dirs.", file=sys.stderr)

    if args.profile_prefix and not (args.vllm_torch_profiler or args.legacy_torch_profiler_env):
        print(
            "Warning: --profile-prefix only applies when profiling is enabled "
            "(--vllm-torch-profiler or --legacy-torch-profiler-env); ignoring prefix.",
            file=sys.stderr,
        )

    from vllm import LLM, SamplingParams

    prompts = _build_prompts(args)
    llm = LLM(**_llm_kwargs(args, output_dir))
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, ignore_eos=True)

    for _ in range(max(0, args.warmup_generations)):
        if args.sequential:
            for p in prompts:
                llm.generate([p], sampling_params=sampling)
        else:
            llm.generate(prompts, sampling_params=sampling)

    metrics_path = output_dir / "request_metrics.jsonl"

    t0 = time.perf_counter()
    _run_generate(
        llm,
        sampling,
        prompts,
        args.sequential,
        args.vllm_torch_profiler,
        args.profile_prefix,
    )
    wall = time.perf_counter() - t0
    print(f"generate() wall time: {wall:.3f}s ({'sequential' if args.sequential else 'batched'})\n")

    rows = _load_jsonl(metrics_path)
    tsv_path = output_dir / "per_request_summary.tsv"
    _write_tsv(rows, tsv_path)
    print(f"Wrote {len(rows)} finished-request rows -> {metrics_path}")
    print(f"Flat summary -> {tsv_path}\n")
    _print_table(rows, args.table_width)

    _print_engine_worker_overlap_report(overlap_path)

    torch_dir = output_dir / "torch_traces"
    traces = _collect_chrome_traces(torch_dir, legacy_dir)

    want_perfetto = args.vllm_torch_profiler or args.legacy_torch_profiler_env
    if want_perfetto:
        if not traces:
            # Workers sometimes finish writing after stop_profile returns.
            time.sleep(0.5)
            traces = _collect_chrome_traces(torch_dir, legacy_dir)
        _print_perfetto_instructions(traces)
    else:
        print(
            "\nTip: for a Chrome trace to load in Perfetto, re-run with "
            "--vllm-torch-profiler (or --legacy-torch-profiler-env).\n",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
