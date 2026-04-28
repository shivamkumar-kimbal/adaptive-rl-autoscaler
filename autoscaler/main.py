"""
Entrypoint for the AWARE-inspired RL Kubernetes autoscaler.

Usage:
  python -m autoscaler.main
  # or via Docker: CMD ["python", "-m", "autoscaler.main"]
"""
import logging
import os
import signal
import sys

from autoscaler.config import settings
from autoscaler.controller.orchestrator import Orchestrator

# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs(settings.LOG_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(settings.LOG_DIR, "autoscaler.log")),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting AWARE-inspired RL Autoscaler")
    logger.info("Config: SLO=%.0fms  α=%.2f  min_replicas=%d  max_replicas=%d",
                settings.SLO_LATENCY_MS, settings.ALPHA,
                settings.MIN_REPLICAS, settings.MAX_REPLICAS)

    orchestrator = Orchestrator()

    # Graceful shutdown on SIGTERM (Kubernetes sends SIGTERM on pod termination)
    def _shutdown(signum, frame):
        logger.info("Received signal %d — shutting down.", signum)
        orchestrator.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    orchestrator.run()


if __name__ == "__main__":
    main()
