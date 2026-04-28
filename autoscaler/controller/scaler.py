"""
Kubernetes deployment scaler using the official Python client.
Automatically selects in-cluster config (when running as a Pod),
local kubeconfig (development), or a MockScaler (docker-compose mode).
"""
import logging
import time
from typing import Optional

from autoscaler.config import settings

logger = logging.getLogger(__name__)


class _MockAppsV1Api:
    """In-memory fake of the Kubernetes AppsV1Api for docker-compose mode."""

    def __init__(self, initial_replicas: int = 2):
        self._replicas = initial_replicas

    def read_namespaced_deployment(self, name, namespace):
        class _Spec:
            replicas = self._replicas  # noqa: E741

        class _Dep:
            spec = _Spec()

        return _Dep()

    def patch_namespaced_deployment_scale(self, name, namespace, body):
        self._replicas = body["spec"]["replicas"]
        logger.info("[MockScaler] %s/%s → %d replicas", namespace, name, self._replicas)


def _load_k8s_client():
    """Lazy-import kubernetes and configure the client."""
    if settings.K8S_CONFIG_MODE == "mock":
        logger.info("Using MockScaler (docker-compose mode)")
        return _MockAppsV1Api()

    try:
        from kubernetes import client, config as k8s_config

        if settings.K8S_CONFIG_MODE == "incluster":
            k8s_config.load_incluster_config()
        else:
            k8s_config.load_kube_config()
        return client.AppsV1Api()
    except Exception as exc:
        logger.error("Failed to load Kubernetes config: %s", exc)
        raise


class KubernetesScaler:
    """Thin wrapper around the Kubernetes Apps API for scaling deployments."""

    def __init__(
        self,
        namespace: str = settings.K8S_NAMESPACE,
        deployment_name: str = settings.K8S_DEPLOYMENT_NAME,
        min_replicas: int = settings.MIN_REPLICAS,
        max_replicas: int = settings.MAX_REPLICAS,
        cooldown_seconds: int = settings.SCALE_COOLDOWN_SECONDS,
    ):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.cooldown_seconds = cooldown_seconds
        self._last_scale_time: float = 0.0
        self._api: Optional[object] = None

    @property
    def api(self):
        if self._api is None:
            self._api = _load_k8s_client()
        return self._api

    def get_replicas(self) -> int:
        """Return the current desired replica count."""
        try:
            deployment = self.api.read_namespaced_deployment(
                name=self.deployment_name, namespace=self.namespace
            )
            return deployment.spec.replicas or 1
        except Exception as exc:
            logger.warning("Could not read replicas: %s", exc)
            return 1

    def scale(self, target_replicas: int) -> bool:
        """
        Scale to target_replicas, respecting bounds and cooldown.
        Returns True if scale was applied, False if skipped.
        """
        target_replicas = max(self.min_replicas, min(self.max_replicas, target_replicas))

        now = time.monotonic()
        if now - self._last_scale_time < self.cooldown_seconds:
            logger.debug(
                "Scale suppressed (cooldown): %ds remaining",
                int(self.cooldown_seconds - (now - self._last_scale_time)),
            )
            return False

        current = self.get_replicas()
        if current == target_replicas:
            return False

        try:
            body = {"spec": {"replicas": target_replicas}}
            self.api.patch_namespaced_deployment_scale(
                name=self.deployment_name,
                namespace=self.namespace,
                body=body,
            )
            self._last_scale_time = now
            logger.info(
                "Scaled %s/%s: %d → %d",
                self.namespace,
                self.deployment_name,
                current,
                target_replicas,
            )
            return True
        except Exception as exc:
            logger.error("Scale failed: %s", exc)
            return False

    def apply_delta(self, delta: int) -> int:
        """Apply a +1/0/-1 replica delta. Returns the new replica count."""
        current = self.get_replicas()
        target = current + delta
        self.scale(target)
        return max(self.min_replicas, min(self.max_replicas, target))
