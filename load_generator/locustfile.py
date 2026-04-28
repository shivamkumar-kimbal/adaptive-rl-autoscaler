"""
Locust load generator for the AWARE-inspired RL autoscaler demo.

Run:
  locust -f locustfile.py --host=http://localhost:8000 --headless \
         -u 50 -r 5 --run-time 30m

The PatternScheduler cycles through steady / ramp / periodic / spike
workload shapes automatically, driving realistic scaling events.
"""
import os
import random
import time

from locust import HttpUser, between, events, task
from locust.env import Environment

from workload_patterns import PatternConfig, PatternScheduler

_scheduler = PatternScheduler(
    PatternConfig(
        name="demo",
        base_users=int(os.getenv("BASE_USERS", "5")),
        peak_users=int(os.getenv("PEAK_USERS", "50")),
        period_seconds=float(os.getenv("PERIOD_SECONDS", "180")),
        spike_duration_seconds=float(os.getenv("SPIKE_DURATION_SECONDS", "60")),
        ramp_duration_seconds=float(os.getenv("RAMP_DURATION_SECONDS", "240")),
    )
)


class WorkloadUser(HttpUser):
    """
    Simulates a realistic mix of:
      - lightweight health checks
      - moderate API calls (simulate endpoint)
      - heavy CPU-burning requests (load endpoint)
    """

    wait_time = between(0.5, 2.0)

    @task(5)
    def simulate_request(self):
        """Standard request with variable latency — main traffic."""
        latency = random.gauss(mu=50, sigma=20)
        latency = max(5.0, min(latency, 500.0))
        self.client.get(f"/simulate?latency_ms={latency:.1f}", name="/simulate")

    @task(2)
    def heavy_request(self):
        """CPU-intensive request — drives CPU-based scaling."""
        cpu_intensity = random.uniform(0.3, 0.9)
        duration = random.uniform(0.2, 1.0)
        self.client.post(
            f"/load?duration={duration:.2f}&cpu_burn={cpu_intensity:.2f}",
            name="/load",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")


# ── Dynamic spawn-rate adjustment ─────────────────────────────────────────────

@events.init.add_listener
def on_locust_init(environment: Environment, **kwargs):
    """Kick off a background thread that adjusts user count per pattern."""
    import threading

    def _adjust_loop():
        while True:
            target = _scheduler.current_user_count()
            pattern = _scheduler.current_pattern()
            try:
                if hasattr(environment, "runner") and environment.runner:
                    if abs(environment.runner.user_count - target) > 2:
                        environment.runner.target_user_count = target
            except Exception:
                pass
            time.sleep(10)

    t = threading.Thread(target=_adjust_loop, daemon=True)
    t.start()
