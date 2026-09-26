"""Unit tests for byte-budget frame LRU used by TemporalCFDWindowDataset."""
from __future__ import annotations

import numpy as np

from data_provider.frame_cache import FrameLRUCache


def test_frame_lru_hit_and_evict():
    cache = FrameLRUCache(max_bytes=100)
    a = np.zeros(10, dtype=np.uint8)  # 10 bytes
    b = np.zeros(20, dtype=np.uint8)  # 20 bytes
    c = np.zeros(80, dtype=np.uint8)  # 80 bytes

    assert cache.get("a") is None
    cache.put("a", a)
    assert cache.get("a") is a
    assert cache.hits == 1
    assert cache.misses == 1

    cache.put("b", b)
    cache.put("c", c)  # 10+20+80=110 > 100 → evict oldest until fits
    # After putting c, a should be gone (10+20+80), maybe b+c=100
    assert cache.get("a") is None
    assert cache.get("c") is c
    assert cache.current_bytes <= 100


def test_frame_lru_rejects_oversized_entry():
    cache = FrameLRUCache(max_bytes=50)
    big = np.zeros(100, dtype=np.uint8)
    cache.put("big", big)
    assert len(cache) == 0
    assert cache.get("big") is None
