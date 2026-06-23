from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from kubernetes import client, config
from kubernetes.client import ApiException

from models.remediation_plan import RemediationAction, RemediationExecutionResult


LOGGER = logging.getLogger(__name__)


class RemediationExecutionError(RuntimeError):
    """Raised when a remediation action cannot be executed safely."""


@dataclass(slots=True)
class RemediationExecutor:
    """Executes allowlisted Kubernetes remediation actions after user approval."""

    core_v1_api: client.CoreV1Api | None = None
    apps_v1_api: client.AppsV1Api | None = None

    def __post_init__(self) -> None:
        """Load Kubernetes config and create API clients when not injected."""
        if self.core_v1_api is None or self.apps_v1_api is None:
            try:
                self._load_kubernetes_config()
            except Exception as exc:
                LOGGER.exception("Failed to load Kubernetes configuration")
                raise RemediationExecutionError(
                    "Unable to load Kubernetes config from in-cluster config or kubeconfig."
                ) from exc

        if self.core_v1_api is None:
            self.core_v1_api = client.CoreV1Api()
        if self.apps_v1_api is None:
            self.apps_v1_api = client.AppsV1Api()

    def execute(self, action: RemediationAction) -> RemediationExecutionResult:
        """Execute one previously validated remediation action."""
        if not action.executable:
            return RemediationExecutionResult(
                success=False,
                message=f"Action is not executable: {action.policy_reason}",
            )

        if action.action_type == "DELETE_POD":
            return self._delete_pod(action)
        if action.action_type == "RESTART_DEPLOYMENT":
            return self._restart_deployment(action)

        return RemediationExecutionResult(
            success=False,
            message=f"Unsupported action type: {action.action_type}",
        )

    def _delete_pod(self, action: RemediationAction) -> RemediationExecutionResult:
        """Delete a pod only when it is owned by a controller."""
        try:
            pod = self._core_v1_api.read_namespaced_pod(
                name=action.target_name,
                namespace=action.namespace,
            )
            if not pod.metadata.owner_references:
                return RemediationExecutionResult(
                    success=False,
                    message=(
                        "Refusing to delete pod because it has no owner references. "
                        "Standalone pod deletion may be destructive."
                    ),
                )

            self._core_v1_api.delete_namespaced_pod(
                name=action.target_name,
                namespace=action.namespace,
            )
        except ApiException as exc:
            LOGGER.warning("Failed to delete pod: %s", exc)
            return RemediationExecutionResult(
                success=False,
                message=f"Failed to delete pod: {self._format_api_exception(exc)}",
            )

        return RemediationExecutionResult(
            success=True,
            message=f"Deleted pod {action.namespace}/{action.target_name}.",
        )

    def _restart_deployment(self, action: RemediationAction) -> RemediationExecutionResult:
        """Trigger a Deployment rollout restart via pod-template annotation."""
        restarted_at = datetime.now(timezone.utc).isoformat()
        patch_body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": restarted_at,
                        }
                    }
                }
            }
        }

        try:
            self._apps_v1_api.patch_namespaced_deployment(
                name=action.target_name,
                namespace=action.namespace,
                body=patch_body,
            )
        except ApiException as exc:
            LOGGER.warning("Failed to restart deployment: %s", exc)
            return RemediationExecutionResult(
                success=False,
                message=f"Failed to restart deployment: {self._format_api_exception(exc)}",
            )

        return RemediationExecutionResult(
            success=True,
            message=(
                f"Triggered rollout restart for deployment "
                f"{action.namespace}/{action.target_name}."
            ),
        )

    @property
    def _core_v1_api(self) -> client.CoreV1Api:
        if self.core_v1_api is None:
            raise RemediationExecutionError("CoreV1 API client is not initialized.")
        return self.core_v1_api

    def _load_kubernetes_config(self) -> None:
        """Load in-cluster config first, then local kubeconfig."""
        try:
            config.load_incluster_config()
            LOGGER.info("Loaded Kubernetes in-cluster configuration")
        except config.ConfigException:
            config.load_kube_config()
            LOGGER.info("Loaded Kubernetes configuration from local kubeconfig")

    @property
    def _apps_v1_api(self) -> client.AppsV1Api:
        if self.apps_v1_api is None:
            raise RemediationExecutionError("AppsV1 API client is not initialized.")
        return self.apps_v1_api

    def _format_api_exception(self, exc: ApiException) -> str:
        """Format an ApiException without leaking excessive response payloads."""
        reason = exc.reason or "Unknown Kubernetes API error"
        return f"status={exc.status}, reason={reason}"
