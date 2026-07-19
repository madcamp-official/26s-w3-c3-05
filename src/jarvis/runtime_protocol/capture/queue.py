"""Bounded, latest-frame queue for a single consumer.

Real-time gaze/gesture consumers must never fall behind on stale frames
(development-principles 5.2). When a consumer cannot keep up, the queue keeps
the newest frames and drops the oldest, counting every drop for telemetry.
"""

from __future__ import annotations

from collections import deque
from threading import Condition
from typing import Generic, TypeVar

T = TypeVar("T")


class BoundedLatestQueue(Generic[T]):
    """Thread-safe bounded queue with an explicit drop-oldest policy.

    One producer (the capture pipeline) calls :meth:`put`; one consumer calls
    :meth:`get`. When full, :meth:`put` discards the oldest item to make room
    for the newest so the consumer always advances toward the latest frame
    rather than replaying a backlog.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._items: deque[T] = deque()
        self._not_empty = Condition()
        self._dropped = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dropped(self) -> int:
        """Total items discarded due to backpressure since creation."""
        with self._not_empty:
            return self._dropped

    def __len__(self) -> int:
        with self._not_empty:
            return len(self._items)

    def put(self, item: T) -> bool:
        """Enqueue ``item``, dropping the oldest if full.

        Returns ``True`` if the queue was already full and an old item was
        dropped to admit this one, ``False`` otherwise.
        """
        with self._not_empty:
            dropped = False
            if len(self._items) >= self._capacity:
                self._items.popleft()
                self._dropped += 1
                dropped = True
            self._items.append(item)
            self._not_empty.notify()
            return dropped

    def get(self, timeout: float | None = None) -> T | None:
        """Pop the oldest retained item, waiting up to ``timeout`` seconds.

        Returns ``None`` if no item becomes available within ``timeout``.
        A ``timeout`` of ``None`` waits indefinitely.
        """
        with self._not_empty:
            if not self._items:
                if not self._not_empty.wait_for(lambda: bool(self._items), timeout):
                    return None
            return self._items.popleft()

    def get_nowait(self) -> T | None:
        """Pop the oldest retained item, or ``None`` if empty."""
        with self._not_empty:
            if not self._items:
                return None
            return self._items.popleft()
