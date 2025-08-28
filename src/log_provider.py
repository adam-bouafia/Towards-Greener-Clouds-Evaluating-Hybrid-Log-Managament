# src/log_provider.py
import csv
import itertools
import os
import random
from typing import Dict, Iterable, Iterator, List, Optional


class LogProvider:
    """
    - synthetic: yields generated logs (unchanged behavior)
    - real_world: reads CSV and yields rows as dicts.
      sample_mode:
        * 'head'     -> original order
        * 'random'   -> random order
        * 'balanced' -> round-robin across LogSource groups (good for small caps)
    """
    def __init__(
        self,
        log_source_type: str = "synthetic",
        filepath: Optional[str] = None,
        sample_mode: str = "head",
    ):
        self.log_source_type = log_source_type
        self.filepath = filepath
        self.sample_mode = sample_mode

    # ---- public API ----
    def get_log_stream(self, num_logs: int = 100) -> Iterator[Dict]:
        if self.log_source_type == "synthetic":
            yield from self._synthetic_stream(num_logs=num_logs)
            return

        if not self.filepath or not os.path.isfile(self.filepath):
            raise FileNotFoundError(f"Log file not found: {self.filepath}")

        rows = self._read_csv(self.filepath)

        if self.sample_mode == "random":
            random.shuffle(rows)
            for row in rows:
                yield row
        elif self.sample_mode == "balanced":
            yield from self._round_robin_by(rows, key="LogSource")
        else:  # 'head'
            for row in rows:
                yield row

    # ---- helpers ----

    def _read_csv(self, path: str) -> List[Dict]:
        out: List[Dict] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(row)
        return out

    def _round_robin_by(self, rows: List[Dict], key: str) -> Iterator[Dict]:
        buckets: Dict[str, List[Dict]] = {}
        for r in rows:
            buckets.setdefault(r.get(key, "unknown"), []).append(r)
        iters = [iter(b) for b in buckets.values()]
        while iters:
            nxt = []
            for it in iters:
                try:
                    yield next(it)
                    nxt.append(it)
                except StopIteration:
                    pass
            iters = nxt

    def _synthetic_stream(self, num_logs: int) -> Iterator[Dict]:
        # Your existing synthetic generator (placeholder minimal version):
        for i in range(num_logs):
            yield {
                "LineId": str(i),
                "Time": "0",
                "EventId": "0",
                "Level": "INFO",
                "EventTemplate": "Synthetic",
                "Content": "Synthetic log line",
                "Component": "app",
                "Date": "2025-01-01",
                "Node": "localhost",
                "LogSource": "synthetic",
            }
