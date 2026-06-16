from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InvestigationContext:
    """Evidence collected for a Kubernetes pod investigation."""

    namespace: str
    pod_name: str
    pod_phase: str
    pod_status_reason: str
    container_status_reasons: str
    describe_output: str
    logs: str
    previous_logs: str
    events: str
