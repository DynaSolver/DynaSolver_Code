"""Byte-budget LRU cache for per-frame CFD arrays (dataloader workers).

Scales to large point counts by caching subsetted frames only, not whole
trajectories. Each DataLoader worker should own its own instance.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Hashable


def _nbytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (tuple, list)):
        return sum(_nbytes(v) for v in value)
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return 0


class FrameLRUCache:
    """Simple LRU keyed by hashable keys with a soft byte budget."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError(f"max_bytes must be >= 0, got {max_bytes}")
        self.max_bytes = int(max_bytes)
        self._store: OrderedDict[Hashable, tuple[int, Any]] = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
        self.current_bytes = 0

    def get(self, key: Hashable) -> Any | None:
        item = self._store.get(key)
        if item is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return item[1]

    def put(self, key: Hashable, value: Any) -> None:
        if self.max_bytes <= 0:
            return
        size = _nbytes(value)
        if size > self.max_bytes:
            # Single entry larger than budget: do not store.
            return
        old = self._store.pop(key, None)
        if old is not None:
            self.current_bytes -= old[0]
        while self._store and self.current_bytes + size > self.max_bytes:
            _, (evicted_size, _) = self._store.popitem(last=False)
            self.current_bytes -= evicted_size
        self._store[key] = (size, value)
        self.current_bytes += size

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits) / float(total) if total else 0.0
