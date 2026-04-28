"""
Prometheus metrics client for the autoscaler.
Queries CPU, memory, request rate, p95 latency, and replica count.
"""
import logging
from typing import Optional

import requests

from autoscaler.config import settings

logger = logging.getLogger(__name__)


class PrometheusClient:
    """Thin wrapper around the Prometheus HTTP API."""

    def __init__(self, url: str = settings.PROMETHEUS_URL):
        self.url = url.rstrip("/")
        self._session = requests.Session()

    def _query(self, promql: str) -> Optional[float]:
        """Execute an instant query; return the first scalar value or None."""
        try:
            resp = self._session.get(
                f"{self.url}/api/v1/query",
                params={"query": promql},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
        except Exception as exc:
            logger.warning("Prometheus query failed [%s]: %s", promql, exc)
        return None

    def get_cpu_pct(
        self,
        namespace: str = settings.K8S_NAMESPACE,
        container: str = settings.PROMETHEUS_CONTAINER_NAME,
    ) -> float:
        """Return CPU utilisation fraction (0-1) for the target container."""
        promql = (
            f'sum(rate(container_cpu_usage_seconds_total{{'
            f'namespace="{namespace}",container="{container}"}}[1m])) '
            f'/ sum(kube_pod_container_resource_limits{{'
            f'namespace="{namespace}",container="{container}",resource="cpu"}})'
        )
        val = self._query(promql)
        if val is None:
            # Fallback: raw CPU rate without normalisation (still useful signal)
            promql_raw = (
                f'sum(rate(container_cpu_usage_seconds_total{{'
                f'namespace="{namespace}",container="{container}"}}[1m]))'
            )
            val = self._query(promql_raw) or 0.0
        return min(max(float(val), 0.0), 1.0)

    def get_memory_pct(
        self,
        namespace: str = settings.K8S_NAMESPACE,
        container: str = settings.PROMETHEUS_CONTAINER_NAME,
    ) -> float:
        """Return memory utilisation fraction (0-1)."""
        promql = (
            f'sum(container_memory_working_set_bytes{{'
            f'namespace="{namespace}",container="{container}"}}) '
            f'/ sum(kube_pod_container_resource_limits{{'
            f'namespace="{namespace}",container="{container}",resource="memory"}})'
        )
        val = self._query(promql) or 0.0
        return min(max(float(val), 0.0), 1.0)

    def get_request_rate(
        self, job: str = settings.PROMETHEUS_JOB_WORKLOAD
    ) -> float:
        """Return HTTP requests per second over the last minute."""
        promql = (
            f'sum(rate(http_requests_total{{job="{job}"}}[1m]))'
        )
        val = self._query(promql) or 0.0
        return max(float(val), 0.0)

    def get_p95_latency_ms(
        self, job: str = settings.PROMETHEUS_JOB_WORKLOAD
    ) -> float:
        """Return p95 HTTP request latency in milliseconds."""
        promql = (
            f'histogram_quantile(0.95, sum(rate('
            f'http_request_duration_seconds_bucket{{job="{job}"}}[1m])) by (le)) * 1000'
        )
        val = self._query(promql) or 0.0
        return max(float(val), 0.0)

    def get_current_replicas(
        self,
        namespace: str = settings.K8S_NAMESPACE,
        deployment: str = settings.K8S_DEPLOYMENT_NAME,
    ) -> int:
        """Return ready replica count from kube-state-metrics."""
        promql = (
            f'kube_deployment_status_replicas_ready{{'
            f'namespace="{namespace}",deployment="{deployment}"}}'
        )
        val = self._query(promql)
        if val is None:
            return 1
        return max(int(val), 1)
