"""
FastAPI workload application — the target service being autoscaled.

Endpoints:
  GET  /           — welcome
  GET  /health     — liveness probe (Kubernetes)
  GET  /ready      — readiness probe
  POST /load       — generate synthetic CPU/memory load
  GET  /metrics    — Prometheus metrics (via instrumentator)

Metrics exposed (auto-instrumented):
  http_requests_total
  http_request_duration_seconds (histogram with buckets for p95)
"""
import hashlib
import os
import time
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="RL Autoscaler Workload", version="1.0.0")

# Auto-instrument all routes → exposes /metrics in Prometheus format
Instrumentator().instrument(app).expose(app)

_START_TIME = time.time()


@app.get("/")
def root():
    return {"service": "workload-app", "uptime_s": round(time.time() - _START_TIME, 1)}


@app.get("/health")
def health():
    """Kubernetes liveness probe — always 200 while process is alive."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Kubernetes readiness probe."""
    return {"status": "ready"}


@app.post("/load")
def generate_load(
    duration: float = Query(default=0.5, ge=0.0, le=10.0, description="CPU burn seconds"),
    cpu_burn: float = Query(default=0.5, ge=0.0, le=1.0, description="CPU intensity fraction"),
    memory_mb: int = Query(default=0, ge=0, le=512, description="Allocate N MB of memory"),
):
    """
    Generate configurable synthetic CPU and memory load.
    Used by the load generator to create realistic resource pressure.
    """
    start = time.time()

    # Memory allocation (held for duration)
    blob = None
    if memory_mb > 0:
        blob = bytearray(memory_mb * 1024 * 1024)

    # CPU burn: hash iterations scaled by cpu_burn intensity
    iterations = int(cpu_burn * 50_000)
    data = os.urandom(64)
    burned = 0
    deadline = start + duration
    while time.time() < deadline:
        for _ in range(min(iterations, 1000)):
            data = hashlib.sha256(data).digest()
        burned += 1

    elapsed = time.time() - start
    del blob  # free memory allocation
    return {
        "elapsed_s": round(elapsed, 3),
        "cpu_burn": cpu_burn,
        "memory_mb": memory_mb,
        "hash_rounds": burned,
    }


@app.get("/simulate")
def simulate_request(
    latency_ms: float = Query(default=10.0, ge=0.0, le=5000.0),
):
    """
    Simulate an application request with a given processing latency.
    Used by the load generator to produce realistic latency distributions.
    """
    time.sleep(latency_ms / 1000.0)
    return {"latency_ms": latency_ms, "status": "processed"}
