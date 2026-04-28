"""
Generate synthetic demo metrics CSV simulating a full experiment run.

Phase 1 (HPA, 300 steps): higher latency, more replicas, more SLO violations
Phase 2 (RL ONLINE_TRAINING, 150 steps): transitioning, improving
Phase 3 (RL SERVING, 200 steps): lower latency, fewer replicas, fewer violations

Output: /tmp/rl_autoscaler/logs/metrics.csv
"""
import csv
import math
import os
import random
import time

import numpy as np

OUTPUT_CSV = "/tmp/rl_autoscaler/logs/metrics.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

FIELDS = [
    "timestamp", "state", "controller", "action",
    "cpu_pct", "mem_pct", "req_rate", "p95_latency_ms", "replicas",
    "reward", "pattern",
]

SLO_MS = 500.0
ALPHA = 0.7
MAX_REPLICAS = 10
MIN_REPLICAS = 1

random.seed(42)
np.random.seed(42)


def compute_reward(p95, replicas, action, prev_action):
    slo_score = 1.0 if p95 <= SLO_MS else max(0.0, 1.0 - (p95 - SLO_MS) / SLO_MS)
    util_score = 1.0 - (replicas - MIN_REPLICAS) / (MAX_REPLICAS - MIN_REPLICAS)
    osc = 0.1 if (prev_action is not None and action != prev_action and action != 1) else 0.0
    return ALPHA * slo_score + (1 - ALPHA) * util_score - osc


rows = []
t = time.time() - 3600 * 3  # start 3 hours ago
prev_action = None

# ── Phase 1: HPA bootstrap (300 steps) ────────────────────────────────────────
replicas = 4
for i in range(300):
    # Simulate traffic pattern (ramp then periodic)
    phase_frac = i / 300
    req_rate = 50 + 80 * math.sin(2 * math.pi * phase_frac * 3) + np.random.normal(0, 5)
    req_rate = max(5.0, req_rate)

    # HPA reacts to CPU with hysteresis
    cpu = 0.45 + 0.4 * math.sin(2 * math.pi * phase_frac * 3 + 0.5) + np.random.normal(0, 0.05)
    cpu = float(np.clip(cpu, 0.05, 0.95))

    if cpu > 0.70 and replicas < MAX_REPLICAS:
        action = 2; replicas = min(MAX_REPLICAS, replicas + 1)
    elif cpu < 0.45 and replicas > MIN_REPLICAS:
        action = 0; replicas = max(MIN_REPLICAS, replicas - 1)
    else:
        action = 1

    mem = float(np.clip(0.35 + 0.1 * cpu + np.random.normal(0, 0.03), 0.1, 0.9))
    # HPA is reactive → latency often spikes before scaling
    latency = max(10.0, 120 + 350 * max(0, cpu - 0.5) + req_rate * 1.5 + np.random.normal(0, 40))
    # HPA over-provisions: replicas tend high
    reward = compute_reward(latency, replicas, action, prev_action)

    pattern = "periodic" if i % 60 < 30 else "steady"
    rows.append({
        "timestamp": t, "state": "INITIALIZED" if i < 50 else "OFFLINE_TRAINING",
        "controller": "HPA", "action": action,
        "cpu_pct": round(cpu, 4), "mem_pct": round(mem, 4),
        "req_rate": round(req_rate, 2), "p95_latency_ms": round(latency, 1),
        "replicas": replicas, "reward": round(reward, 4), "pattern": pattern,
    })
    prev_action = action
    t += 30

# ── Phase 2: RL ONLINE_TRAINING (150 steps, improving) ────────────────────────
replicas = 3
for i in range(150):
    phase_frac = i / 150
    req_rate = 50 + 70 * math.sin(2 * math.pi * phase_frac * 2) + np.random.normal(0, 5)
    req_rate = max(5.0, req_rate)
    cpu = 0.40 + 0.35 * math.sin(2 * math.pi * phase_frac * 2 + 0.3) + np.random.normal(0, 0.04)
    cpu = float(np.clip(cpu, 0.05, 0.90))

    # RL learns to pre-emptively scale — better than reactive HPA
    learning_factor = phase_frac  # improves over training
    if cpu > 0.65 - 0.1 * learning_factor and replicas < MAX_REPLICAS:
        action = 2; replicas = min(MAX_REPLICAS, replicas + 1)
    elif cpu < 0.40 + 0.05 * learning_factor and replicas > MIN_REPLICAS:
        action = 0; replicas = max(MIN_REPLICAS, replicas - 1)
    else:
        action = 1

    mem = float(np.clip(0.30 + 0.1 * cpu + np.random.normal(0, 0.03), 0.1, 0.85))
    # RL converges → latency improving
    latency = max(10.0, 100 + 280 * max(0, cpu - 0.5) + req_rate * 1.2
                  + np.random.normal(0, 35) - 20 * learning_factor)
    reward = compute_reward(latency, replicas, action, prev_action)

    pattern = "ramp" if i < 60 else "periodic"
    rows.append({
        "timestamp": t, "state": "ONLINE_TRAINING",
        "controller": "RL", "action": action,
        "cpu_pct": round(cpu, 4), "mem_pct": round(mem, 4),
        "req_rate": round(req_rate, 2), "p95_latency_ms": round(latency, 1),
        "replicas": replicas, "reward": round(reward, 4), "pattern": pattern,
    })
    prev_action = action
    t += 30

# ── Phase 3: RL SERVING (200 steps, optimised) ────────────────────────────────
replicas = 2
for i in range(200):
    phase_frac = i / 200
    # Spike workload — challenging for HPA but RL handles it
    if 80 < i < 110:
        req_rate = 180 + np.random.normal(0, 10)  # spike
        pattern = "spike"
    elif 140 < i < 180:
        req_rate = 30 + 100 * phase_frac + np.random.normal(0, 8)  # ramp
        pattern = "ramp"
    else:
        req_rate = 55 + 40 * math.sin(2 * math.pi * phase_frac * 4) + np.random.normal(0, 5)
        pattern = "periodic"

    req_rate = max(5.0, req_rate)
    cpu = float(np.clip(0.25 + req_rate / 400 + np.random.normal(0, 0.03), 0.05, 0.85))

    # RL proactively scales: predicts demand from req_rate signal
    if req_rate > 120 or cpu > 0.62:
        action = 2; replicas = min(MAX_REPLICAS, replicas + 1)
    elif req_rate < 40 and cpu < 0.30:
        action = 0; replicas = max(MIN_REPLICAS, replicas - 1)
    else:
        action = 1

    mem = float(np.clip(0.25 + 0.08 * cpu + np.random.normal(0, 0.02), 0.1, 0.80))
    # RL achieves much lower latency at fewer replicas
    latency = max(10.0, 80 + 220 * max(0, cpu - 0.4) + req_rate * 0.9 + np.random.normal(0, 25))
    reward = compute_reward(latency, replicas, action, prev_action)

    rows.append({
        "timestamp": t, "state": "SERVING",
        "controller": "RL", "action": action,
        "cpu_pct": round(cpu, 4), "mem_pct": round(mem, 4),
        "req_rate": round(req_rate, 2), "p95_latency_ms": round(latency, 1),
        "replicas": replicas, "reward": round(reward, 4), "pattern": pattern,
    })
    prev_action = action
    t += 30

# ── Write CSV ──────────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} rows → {OUTPUT_CSV}")
print(f"  HPA steps:  {sum(1 for r in rows if r['controller'] == 'HPA')}")
print(f"  RL steps:   {sum(1 for r in rows if r['controller'] == 'RL')}")
