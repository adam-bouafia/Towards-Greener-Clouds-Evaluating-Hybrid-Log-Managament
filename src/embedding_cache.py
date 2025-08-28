# src/embedding_cache.py
from __future__ import annotations
import os, hashlib, numpy as np
from collections import OrderedDict
from pathlib import Path

class EmbeddingCache:
    def __init__(self, base_dir: str = ".cache/embeddings", mem_limit: int = 2048):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.mem_limit = int(mem_limit)
        self._lru: OrderedDict[str, np.ndarray] = OrderedDict()

    def _key(self, text: str) -> str:
        return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

    def _disk_path(self, key: str) -> Path:
        return self.base / f"{key}.npy"

    def get(self, text: str, compute_fn) -> np.ndarray:
        k = self._key(text)
        # 1) memory
        if k in self._lru:
            v = self._lru.pop(k)
            self._lru[k] = v
            return v
        # 2) disk
        p = self._disk_path(k)
        if p.exists():
            v = np.load(p)
            self._insert_mem(k, v)
            return v
        # 3) compute
        v = compute_fn()
        v = np.asarray(v, dtype=np.float32)
        np.save(p, v)
        self._insert_mem(k, v)
        return v

    def _insert_mem(self, k: str, v: np.ndarray):
        self._lru[k] = v
        if len(self._lru) > self.mem_limit:
            self._lru.popitem(last=False)
