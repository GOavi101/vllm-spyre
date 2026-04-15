# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class AsyncSpyreScheduler(AsyncScheduler):
    """Base class inheriting from the V1 async scheduler to support static
    and continuous batching respecting AIU Spyre constraints."""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM async scheduler
        super().__init__(*args, **kwargs)
        self.model_config = self.vllm_config.model_config


# Import implementation classes from scheduler_impl and re-export with async names
# These are the same classes as in scheduler.py, but they will behave as async
# schedulers when VLLM_SPYRE_ENABLE_ASYNC_SCHEDULING=1
from vllm_spyre.v1.core.scheduler_impl import (
    ChunkedPrefillSpyreScheduler as AsyncChunkedPrefillSpyreScheduler,
    PoolingSpyreScheduler as AsyncPoolingSpyreScheduler,
)

__all__ = [
    "AsyncPoolingSpyreScheduler",
    "AsyncChunkedPrefillSpyreScheduler",
]
