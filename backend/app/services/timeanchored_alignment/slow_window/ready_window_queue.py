"""ReadySlowWindow 队列视图。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadyWindowQueueSnapshot:
    depth: int


class ReadyWindowQueue:
    """轻量记录 ready queue 水位。"""

    def __init__(self) -> None:
        self._depth = 0

    def observe(self, depth: int | None = None) -> ReadyWindowQueueSnapshot:
        if depth is not None:
            self._depth = max(0, int(depth))
        return ReadyWindowQueueSnapshot(depth=self._depth)

    def push(self) -> ReadyWindowQueueSnapshot:
        self._depth += 1
        return ReadyWindowQueueSnapshot(depth=self._depth)

    def pop(self, count: int = 1) -> ReadyWindowQueueSnapshot:
        self._depth = max(0, self._depth - max(0, int(count)))
        return ReadyWindowQueueSnapshot(depth=self._depth)

    @property
    def depth(self) -> int:
        return self._depth
