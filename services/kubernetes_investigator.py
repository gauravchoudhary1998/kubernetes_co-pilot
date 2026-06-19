from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kubernetes import client, config
from kubernetes.client import ApiException

from models.investigation_context import InvestigationContext


LOGGER = logging.getLogger(__name__)


class KubernetesInvestigationError(RuntimeError):
    """Raised when Kubernetes evidence cannot be collected safely."""


@dataclass(slots=True)
class KubernetesInvestigator:
    """Collects deterministic troubleshooting evidence from a Kubernetes cluster."""

    core_v1_api: client.CoreV1Api | None = None

    def __post_init__(self) -> None:
        """Load kubeconfig and create the CoreV1 API client when not injected."""
        if self.core_v1_api is None:
            try:
                config.load_kube_config()
            except Exception as exc:
                LOGGER.exception("Failed to load local Kubernetes configuration")
                raise KubernetesInvestigationError(
                    "Unable to load Kubernetes config from local kubeconfig."
                ) from exc

            self.core_v1_api = client.CoreV1Api()

    def get_pod(self, namespace: str, pod_name: str) -> client.V1Pod:
        """Return the requested pod or raise a user-safe investigation error."""
        self._ensure_namespace_exists(namespace)
        try:
            return self._api.read_namespaced_pod(name=pod_name, namespace=namespace)
        except ApiException as exc:
            if exc.status == 404:
                raise KubernetesInvestigationError(
                    f"Pod '{pod_name}' was not found in namespace '{namespace}'."
                ) from exc
            raise self._api_error("retrieve pod", exc) from exc

    def get_pod_events(self, namespace: str, pod_name: str) -> str:
        """Return Kubernetes events related to the pod."""
        field_selector = (
            f"involvedObject.name={pod_name},"
            f"involvedObject.namespace={namespace},"
            "involvedObject.kind=Pod"
        )

        try:
            events = self._api.list_namespaced_event(
                namespace=namespace,
                field_selector=field_selector,
            )
        except ApiException as exc:
            LOGGER.warning("Failed to retrieve pod events: %s", exc)
            return f"Unable to retrieve pod events: {self._format_api_exception(exc)}"

        sorted_events = sorted(
            events.items,
            key=lambda event: self._event_timestamp(event) or datetime.min,
        )
        if not sorted_events:
            return "No pod events found."

        return "\n".join(self._format_event(event) for event in sorted_events)

    def get_pod_logs(self, namespace: str, pod_name: str) -> str:
        """Return current pod logs, handling log retrieval errors gracefully."""
        pod = self.get_pod(namespace=namespace, pod_name=pod_name)
        return self._read_all_container_logs(pod=pod, previous=False)

    def get_previous_logs(self, namespace: str, pod_name: str) -> str:
        """Return logs from the previous container instance when available."""
        pod = self.get_pod(namespace=namespace, pod_name=pod_name)
        return self._read_all_container_logs(pod=pod, previous=True)

    def investigate_pod(self, namespace: str, pod_name: str) -> InvestigationContext:
        """Collect evidence for a pod based on the pod's current failure signal."""
        pod = self.get_pod(namespace=namespace, pod_name=pod_name)
        pod_phase = pod.status.phase or "Unknown"
        pod_status_reason = self._pod_status_reason(pod)
        container_status_reasons = self._container_status_reasons(pod)
        describe_output = self._describe_pod(pod)

        logs = ""
        previous_logs = ""
        events = ""

        if pod_status_reason in {"CrashLoopBackOff", "OOMKilled"}:
            logs = self._read_all_container_logs(pod=pod, previous=False)
            previous_logs = self._read_all_container_logs(pod=pod, previous=True)
            events = self.get_pod_events(namespace=namespace, pod_name=pod_name)
        elif pod_status_reason in {"Pending", "ImagePullBackOff", "ErrImagePull"}:
            events = self.get_pod_events(namespace=namespace, pod_name=pod_name)
        elif pod_phase == "Pending":
            events = self.get_pod_events(namespace=namespace, pod_name=pod_name)
        else:
            logs = self._read_all_container_logs(pod=pod, previous=False)
            events = self.get_pod_events(namespace=namespace, pod_name=pod_name)

        LOGGER.info(
            "Collected investigation evidence for pod %s/%s with phase=%s reason=%s",
            namespace,
            pod_name,
            pod_phase,
            pod_status_reason,
        )

        return InvestigationContext(
            namespace=namespace,
            pod_name=pod_name,
            pod_phase=pod_phase,
            pod_status_reason=pod_status_reason,
            container_status_reasons=container_status_reasons,
            describe_output=describe_output,
            logs=logs,
            previous_logs=previous_logs,
            events=events,
        )

    @property
    def _api(self) -> client.CoreV1Api:
        if self.core_v1_api is None:
            raise KubernetesInvestigationError("Kubernetes API client is not initialized.")
        return self.core_v1_api

    def _ensure_namespace_exists(self, namespace: str) -> None:
        """Raise a clear error if the namespace does not exist."""
        try:
            self._api.read_namespace(name=namespace)
        except ApiException as exc:
            if exc.status == 404:
                raise KubernetesInvestigationError(
                    f"Namespace '{namespace}' was not found."
                ) from exc
            raise self._api_error("retrieve namespace", exc) from exc

    def _read_all_container_logs(self, pod: client.V1Pod, previous: bool) -> str:
        """Read logs for each container in a pod."""
        namespace = pod.metadata.namespace
        pod_name = pod.metadata.name
        containers = pod.spec.containers or []
        if not containers:
            return "No containers found on pod."

        log_sections = []
        for container in containers:
            logs = self._read_container_logs(
                namespace=namespace,
                pod_name=pod_name,
                container_name=container.name,
                previous=previous,
            )
            log_sections.append(f"===== container: {container.name} =====\n{logs}")

        return "\n\n".join(log_sections)

    def _read_container_logs(
        self,
        namespace: str,
        pod_name: str,
        container_name: str,
        previous: bool,
    ) -> str:
        """Read logs from one container's current or previous instance."""
        try:
            return self._api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                container=container_name,
                previous=previous,
                timestamps=True,
                tail_lines=200,
            )
        except ApiException as exc:
            log_type = "previous logs" if previous else "logs"
            LOGGER.warning("Failed to retrieve pod %s: %s", log_type, exc)
            return f"Unable to retrieve pod {log_type}: {self._format_api_exception(exc)}"

    def _pod_status_reason(self, pod: client.V1Pod) -> str:
        """Derive the most useful status reason from pod and container state."""
        for status in pod.status.container_statuses or []:
            current_reason = self._nested_attr(status.state, "terminated", "reason")
            previous_reason = self._nested_attr(
                status.last_state,
                "terminated",
                "reason",
            )
            if "OOMKilled" in {current_reason, previous_reason}:
                return "OOMKilled"

        for status in pod.status.container_statuses or []:
            state = status.state
            last_state = status.last_state

            waiting_reason = self._nested_attr(state, "waiting", "reason")
            if waiting_reason:
                return waiting_reason

            terminated_reason = self._nested_attr(state, "terminated", "reason")
            if terminated_reason:
                return terminated_reason

            last_terminated_reason = self._nested_attr(last_state, "terminated", "reason")
            if last_terminated_reason:
                return last_terminated_reason

        return pod.status.reason or pod.status.phase or "Unknown"

    def _container_status_reasons(self, pod: client.V1Pod) -> str:
        """Return per-container status reasons for LLM evidence."""
        sections: list[str] = []
        sections.extend(
            self._format_container_status_reasons(
                title="Init container status reasons",
                statuses=pod.status.init_container_statuses or [],
            )
        )
        sections.extend(
            self._format_container_status_reasons(
                title="Container status reasons",
                statuses=pod.status.container_statuses or [],
            )
        )
        sections.extend(
            self._format_container_status_reasons(
                title="Ephemeral container status reasons",
                statuses=pod.status.ephemeral_container_statuses or [],
            )
        )
        return "\n".join(sections).strip() or "No container statuses reported."

    def _format_container_status_reasons(
        self,
        title: str,
        statuses: list[client.V1ContainerStatus],
    ) -> list[str]:
        """Format a group of container status reasons."""
        if not statuses:
            return [f"{title}: <none>"]

        lines = [f"{title}:"]
        for status in statuses:
            lines.append(
                "  - "
                f"Name: {status.name}, "
                f"Ready: {status.ready}, "
                f"Restart Count: {status.restart_count}, "
                f"Current Reason: {self._container_state_reason(status.state)}, "
                f"Last Reason: {self._container_state_reason(status.last_state)}"
            )
        return lines

    def _container_state_reason(self, state: Any) -> str:
        """Extract the reason from a container state object."""
        if state is None:
            return "<none>"
        waiting_reason = self._nested_attr(state, "waiting", "reason")
        if waiting_reason:
            return waiting_reason
        terminated_reason = self._nested_attr(state, "terminated", "reason")
        if terminated_reason:
            return terminated_reason
        if getattr(state, "running", None) is not None:
            return "Running"
        return "<none>"

    def _describe_pod(self, pod: client.V1Pod) -> str:
        """Create a compact, deterministic pod description for LLM analysis."""
        metadata = pod.metadata
        spec = pod.spec
        status = pod.status
        lines = [
            f"Name: {metadata.name}",
            f"Namespace: {metadata.namespace}",
            f"Node: {spec.node_name or '<none>'}",
            f"Service Account: {spec.service_account_name or '<none>'}",
            f"Phase: {status.phase or '<unknown>'}",
            f"Reason: {status.reason or '<none>'}",
            f"Message: {status.message or '<none>'}",
            f"QoS Class: {status.qos_class or '<unknown>'}",
            f"Pod IP: {status.pod_ip or '<none>'}",
            f"Host IP: {status.host_ip or '<none>'}",
            f"Node Selector: {spec.node_selector or '<none>'}",
            f"Restart Policy: {spec.restart_policy or '<unknown>'}",
            "Owner References:",
        ]
        lines.extend(self._describe_owner_references(metadata.owner_references or []))
        lines.append("Conditions:")
        lines.extend(self._describe_pod_conditions(status.conditions or []))
        lines.append("Tolerations:")
        lines.extend(self._describe_tolerations(spec.tolerations or []))
        lines.append("Volumes:")
        lines.extend(self._describe_volumes(spec.volumes or []))
        lines.append(
            "Containers:",
        )

        container_statuses = {
            status.name: status for status in status.container_statuses or []
        }
        for container in spec.containers or []:
            container_status = container_statuses.get(container.name)
            lines.extend(self._describe_container(container, container_status))

        return "\n".join(lines)

    def _describe_container(
        self,
        container: client.V1Container,
        status: client.V1ContainerStatus | None,
    ) -> list[str]:
        """Format container spec and status fields relevant to troubleshooting."""
        lines = [
            f"  - Name: {container.name}",
            f"    Image: {container.image}",
            f"    Resources: {self._format_resources(container.resources)}",
            f"    Liveness Probe: {self._format_probe(container.liveness_probe)}",
            f"    Readiness Probe: {self._format_probe(container.readiness_probe)}",
            f"    Startup Probe: {self._format_probe(container.startup_probe)}",
        ]

        if status is None:
            lines.append("    Status: <missing>")
            return lines

        lines.extend(
            [
                f"    Ready: {status.ready}",
                f"    Restart Count: {status.restart_count}",
                f"    State: {self._format_container_state(status.state)}",
                f"    Last State: {self._format_container_state(status.last_state)}",
            ]
        )
        return lines

    def _describe_owner_references(
        self,
        owner_references: list[client.V1OwnerReference],
    ) -> list[str]:
        """Format pod owner references for remediation planning."""
        if not owner_references:
            return ["  <none>"]

        return [
            "  - "
            f"Kind: {owner.kind}, "
            f"Name: {owner.name}, "
            f"Controller: {owner.controller}"
            for owner in owner_references
        ]

    def _describe_pod_conditions(
        self,
        conditions: list[client.V1PodCondition],
    ) -> list[str]:
        """Format pod conditions relevant to scheduling and readiness."""
        if not conditions:
            return ["  <none>"]

        return [
            "  - "
            f"Type: {condition.type}, "
            f"Status: {condition.status}, "
            f"Reason: {condition.reason or '<none>'}, "
            f"Message: {condition.message or '<none>'}"
            for condition in conditions
        ]

    def _describe_tolerations(
        self,
        tolerations: list[client.V1Toleration],
    ) -> list[str]:
        """Format pod tolerations in a compact way."""
        if not tolerations:
            return ["  <none>"]

        return [
            "  - "
            f"Key: {toleration.key or '<none>'}, "
            f"Operator: {toleration.operator or '<none>'}, "
            f"Value: {toleration.value or '<none>'}, "
            f"Effect: {toleration.effect or '<none>'}"
            for toleration in tolerations
        ]

    def _describe_volumes(self, volumes: list[client.V1Volume]) -> list[str]:
        """Format pod volume names and source types."""
        if not volumes:
            return ["  <none>"]

        return [
            f"  - Name: {volume.name}, Type: {self._volume_type(volume)}"
            for volume in volumes
        ]

    def _volume_type(self, volume: client.V1Volume) -> str:
        """Return the configured source type for a volume."""
        volume_sources = (
            "config_map",
            "secret",
            "persistent_volume_claim",
            "empty_dir",
            "host_path",
            "projected",
            "downward_api",
            "csi",
        )
        for source in volume_sources:
            if getattr(volume, source, None) is not None:
                return source
        return "<unknown>"

    def _format_resources(self, resources: client.V1ResourceRequirements | None) -> str:
        """Format container resource requests and limits."""
        if resources is None:
            return "requests=<none>, limits=<none>"

        requests = resources.requests or {}
        limits = resources.limits or {}
        return f"requests={requests or '<none>'}, limits={limits or '<none>'}"

    def _format_probe(self, probe: client.V1Probe | None) -> str:
        """Format a container probe without dumping the full object."""
        if probe is None:
            return "<none>"

        return (
            f"{self._probe_handler(probe)}, "
            f"initial_delay={probe.initial_delay_seconds}, "
            f"period={probe.period_seconds}, "
            f"timeout={probe.timeout_seconds}, "
            f"failure_threshold={probe.failure_threshold}"
        )

    def _probe_handler(self, probe: client.V1Probe) -> str:
        """Return the configured probe handler type."""
        if probe.http_get is not None:
            return f"http_get path={probe.http_get.path}, port={probe.http_get.port}"
        if probe.tcp_socket is not None:
            return f"tcp_socket port={probe.tcp_socket.port}"
        if probe.exec is not None:
            return f"exec command={probe.exec.command}"
        if probe.grpc is not None:
            return f"grpc port={probe.grpc.port}, service={probe.grpc.service}"
        return "handler=<unknown>"

    def _format_container_state(self, state: Any) -> str:
        """Format a Kubernetes container state object."""
        if state is None:
            return "<none>"
        if getattr(state, "waiting", None) is not None:
            waiting = state.waiting
            return f"Waiting(reason={waiting.reason}, message={waiting.message})"
        if getattr(state, "running", None) is not None:
            running = state.running
            return f"Running(started_at={running.started_at})"
        if getattr(state, "terminated", None) is not None:
            terminated = state.terminated
            return (
                "Terminated("
                f"reason={terminated.reason}, "
                f"exit_code={terminated.exit_code}, "
                f"message={terminated.message}"
                ")"
            )
        return "<unknown>"

    def _format_event(self, event: client.CoreV1Event) -> str:
        """Format a Kubernetes event in a stable single-line representation."""
        timestamp = self._event_timestamp(event) or "<timestamp-unknown>"
        count = event.count if event.count is not None else 1
        return (
            f"{timestamp} {event.type or '<type-unknown>'} "
            f"{event.reason or '<reason-unknown>'} count={count}: "
            f"{event.message or '<no message>'}"
        )

    def _event_timestamp(self, event: client.CoreV1Event) -> Any:
        """Return the best available timestamp from a Kubernetes event."""
        return (
            getattr(event, "last_timestamp", None)
            or getattr(event, "event_time", None)
            or getattr(event, "first_timestamp", None)
        )

    def _nested_attr(self, obj: Any, *attrs: str) -> str:
        """Return a nested string attribute or an empty string."""
        current = obj
        for attr in attrs:
            current = getattr(current, attr, None)
            if current is None:
                return ""
        return current if isinstance(current, str) else ""

    def _api_error(self, action: str, exc: ApiException) -> KubernetesInvestigationError:
        """Create a normalized Kubernetes API error."""
        LOGGER.exception("Failed to %s from Kubernetes", action)
        return KubernetesInvestigationError(
            f"Unable to {action}: {self._format_api_exception(exc)}"
        )

    def _format_api_exception(self, exc: ApiException) -> str:
        """Format an ApiException without leaking excessive response payloads."""
        reason = exc.reason or "Unknown Kubernetes API error"
        return f"status={exc.status}, reason={reason}"
