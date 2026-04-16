# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_spyre.v1.core.scheduler_impl import (
    _create_chunked_prefill_scheduler,
    _create_pooling_scheduler,
)


class SpyreScheduler(Scheduler):
    """Base class inheriting from the V1 scheduler to support static
    and continuous batching respecting AIU Spyre constraints."""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM scheduler
        super().__init__(*args, **kwargs)
        self.model_config = self.vllm_config.model_config


# Default sync schedulers (backward compatibility)
# Platform.py will override these based on scheduler_config.async_scheduling
PoolingSpyreScheduler = _create_pooling_scheduler(Scheduler)
ChunkedPrefillSpyreScheduler = _create_chunked_prefill_scheduler(Scheduler)

# Fix __module__, __name__, and __qualname__ so classes are importable from this module
# and resolve correctly when converted to string paths
PoolingSpyreScheduler.__module__ = __name__
PoolingSpyreScheduler.__name__ = "PoolingSpyreScheduler"
PoolingSpyreScheduler.__qualname__ = "PoolingSpyreScheduler"

ChunkedPrefillSpyreScheduler.__module__ = __name__
ChunkedPrefillSpyreScheduler.__name__ = "ChunkedPrefillSpyreScheduler"
ChunkedPrefillSpyreScheduler.__qualname__ = "ChunkedPrefillSpyreScheduler"

__all__ = [
    "PoolingSpyreScheduler",
    "ChunkedPrefillSpyreScheduler",
]
