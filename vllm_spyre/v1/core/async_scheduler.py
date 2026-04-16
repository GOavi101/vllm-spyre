# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_spyre.v1.core.scheduler_impl import (
    _create_chunked_prefill_scheduler,
    _create_pooling_scheduler,
)


class AsyncSpyreScheduler(AsyncScheduler):
    """Base class inheriting from the V1 async scheduler to support static
    and continuous batching respecting AIU Spyre constraints."""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM async scheduler
        super().__init__(*args, **kwargs)
        self.model_config = self.vllm_config.model_config


# Default async schedulers (backward compatibility)
# Platform.py will use these when scheduler_config.async_scheduling is True
AsyncPoolingSpyreScheduler = _create_pooling_scheduler(AsyncScheduler)
AsyncChunkedPrefillSpyreScheduler = _create_chunked_prefill_scheduler(AsyncScheduler)

# Fix __module__, __name__, and __qualname__ so classes are importable from this module
# and resolve correctly when converted to string paths
AsyncPoolingSpyreScheduler.__module__ = __name__
AsyncPoolingSpyreScheduler.__name__ = "AsyncPoolingSpyreScheduler"
AsyncPoolingSpyreScheduler.__qualname__ = "AsyncPoolingSpyreScheduler"

AsyncChunkedPrefillSpyreScheduler.__module__ = __name__
AsyncChunkedPrefillSpyreScheduler.__name__ = "AsyncChunkedPrefillSpyreScheduler"
AsyncChunkedPrefillSpyreScheduler.__qualname__ = "AsyncChunkedPrefillSpyreScheduler"

__all__ = [
    "AsyncPoolingSpyreScheduler",
    "AsyncChunkedPrefillSpyreScheduler",
]
