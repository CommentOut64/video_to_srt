"""
Bridge 句子队列封装。
V3.2.0+dev.20260201.04
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from app.services.punctuation.semantic_buffer import SemanticChunk


class SentenceQueue:
    """句子队列封装（门面模式）：统一处理容量控制与背压信号。"""

    def __init__(
        self,
        maxsize: int = 50,
        backpressure_timeout: float = 5.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._queue = asyncio.Queue(maxsize=max(1, int(maxsize)))
        self._items: List[SemanticChunk] = []
        self._backpressure_event = asyncio.Event()
        self._capacity_event = asyncio.Event()
        self._capacity_event.set()
        self._backpressure_timeout = max(0.1, float(backpressure_timeout))
        self._logger = logger or logging.getLogger(__name__)

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[SemanticChunk]:
        return list(self._items)

    def is_full(self) -> bool:
        return self._queue.full()

    def is_backpressure_active(self) -> bool:
        return self._backpressure_event.is_set()

    async def push(self, chunk: SemanticChunk) -> bool:
        """尝试推入队列，超时返回 False 表示触发背压。"""
        if self._queue.full():
            self._capacity_event.clear()
            try:
                await asyncio.wait_for(
                    self._capacity_event.wait(),
                    timeout=self._backpressure_timeout,
                )
            except asyncio.TimeoutError:
                self._backpressure_event.set()
                self._logger.warning("Bridge 句子队列触发背压：等待容量超时")
                return False

        await self._queue.put(1)
        self._items.append(chunk)
        if not self._queue.full():
            self._capacity_event.set()
        return True

    async def force_push(self, chunk: SemanticChunk) -> None:
        """强制推入队列（阻塞等待容量）。"""
        await self._queue.put(1)
        self._items.append(chunk)
        if not self._queue.full():
            self._capacity_event.set()
        if self._backpressure_event.is_set():
            self._backpressure_event.clear()

    async def wait_for_capacity(self) -> None:
        """等待队列有容量。"""
        if not self._queue.full():
            self._capacity_event.set()
            if self._backpressure_event.is_set():
                self._backpressure_event.clear()
            return
        await self._capacity_event.wait()
        if self._backpressure_event.is_set():
            self._backpressure_event.clear()

    def drop_oldest(self) -> Optional[SemanticChunk]:
        """丢弃最旧元素（仅用于紧急丢帧）。"""
        if not self._items:
            return None
        dropped = self._items.pop(0)
        try:
            if not self._queue.empty():
                self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        self._capacity_event.set()
        if self._backpressure_event.is_set():
            self._backpressure_event.clear()
        return dropped

    def pop_all(self) -> List[SemanticChunk]:
        """取出并清空所有元素。"""
        items = list(self._items)
        self._items.clear()
        try:
            while not self._queue.empty():
                self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        self._capacity_event.set()
        if self._backpressure_event.is_set():
            self._backpressure_event.clear()
        return items
