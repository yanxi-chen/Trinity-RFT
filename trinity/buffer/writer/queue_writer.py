"""Writer of the Queue buffer."""
from typing import List

import ray

from trinity.buffer.buffer_writer import BufferWriter
from trinity.buffer.storage.queue import QueueStorage
from trinity.common.config import StorageConfig
from trinity.common.constants import StorageType


class QueueWriter(BufferWriter):
    """Writer of the Queue buffer."""

    def __init__(self, config: StorageConfig):
        assert config.storage_type == StorageType.QUEUE
        self.queue = QueueStorage.get_wrapper(config)

    def write(self, data: List) -> None:
        ray.get(self.queue.put_batch.remote(data))

    async def write_async(self, data):
        return await self.queue.put_batch.remote(data)

    async def acquire(self) -> int:
        return await self.queue.acquire.remote()

    async def release(self) -> int:
        return await self.queue.release.remote()
