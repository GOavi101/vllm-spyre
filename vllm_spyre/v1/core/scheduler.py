# SPDX-License-Identifier: Apache-2.0

from vllm.v1.core.sched.scheduler import Scheduler


class SpyreScheduler(Scheduler):
    """Base class inheriting from the V1 scheduler to support static
    and continuous batching respecting AIU Spyre constraints."""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM scheduler
        super().__init__(*args, **kwargs)
        self.model_config = self.vllm_config.model_config


# Import implementation classes from scheduler_impl
# These classes can inherit from either sync or async base schedulers
# based on the VLLM_SPYRE_ENABLE_ASYNC_SCHEDULING environment variable
from vllm_spyre.v1.core.scheduler_impl import (
    ChunkedPrefillSpyreScheduler,
    PoolingSpyreScheduler,
)

__all__ = [
    "PoolingSpyreScheduler",
    "ChunkedPrefillSpyreScheduler",
]
