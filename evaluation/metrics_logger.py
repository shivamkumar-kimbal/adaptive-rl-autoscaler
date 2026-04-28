"""
Structured metrics logger — writes per-step data to CSV and lifecycle events to JSON.

Files written:
  LOG_DIR/metrics.csv   — per-step observations, actions, rewards
  LOG_DIR/events.jsonl  — lifecycle transitions and notable events
"""
import csv
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from autoscaler.config import settings

logger = logging.getLogger(__name__)

_STEP_FIELDS = [
    "timestamp", "state", "controller", "action",
    "cpu_pct", "mem_pct", "req_rate", "p95_latency_ms", "replicas",
    "reward", "pattern",
]


class MetricsLogger:
    def __init__(self, csv_path: str = settings.METRICS_CSV_PATH):
        self.csv_path = csv_path
        self.events_path = os.path.join(
            os.path.dirname(csv_path), "events.jsonl"
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._csv_file = None
        self._csv_writer = None
        self._init_csv()

    def _init_csv(self):
        file_exists = os.path.isfile(self.csv_path)
        self._csv_file = open(self.csv_path, "a", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=_STEP_FIELDS
        )
        if not file_exists:
            self._csv_writer.writeheader()

    def log_step(
        self,
        state: str,
        controller: str,
        action: int,
        reward: float,
        pattern: str,
        cpu_pct: float = 0.0,
        mem_pct: float = 0.0,
        req_rate: float = 0.0,
        p95_latency_ms: float = 0.0,
        replicas: float = 1.0,
        **_kwargs,
    ):
        row = {
            "timestamp": time.time(),
            "state": state,
            "controller": controller,
            "action": action,
            "cpu_pct": round(cpu_pct, 4),
            "mem_pct": round(mem_pct, 4),
            "req_rate": round(req_rate, 2),
            "p95_latency_ms": round(p95_latency_ms, 1),
            "replicas": int(replicas),
            "reward": round(reward, 4),
            "pattern": pattern,
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def log_event(self, event_type: str, data: Dict[str, Any]):
        record = {"timestamp": time.time(), "event": event_type, **data}
        with open(self.events_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def close(self):
        if self._csv_file:
            self._csv_file.close()
