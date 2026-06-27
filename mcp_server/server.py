from mcp.server.fastmcp import FastMCP

from services.investigation_service import InvestigationService
from services.kubernetes_investigator import KubernetesInvestigationError


mcp = FastMCP("Kubernetes Troubleshooting Copilot")
_service = InvestigationService()


@mcp.tool()
def investigate_pod(namespace: str, pod_name: str, cluster_name: str | None = None) -> str:
    """
    Investigate a failing Kubernetes pod.

    Collects cluster evidence (pod status, logs, events), analyzes it with an LLM,
    and automatically executes any low-risk remediation actions. Returns the full
    analysis and the status of each recommended action.

    Args:
        namespace: The Kubernetes namespace the pod lives in.
        pod_name: The name of the failing pod.
        cluster_name: Kubeconfig context name for a remote cluster. Omit to
                      investigate the cluster this server is running in.
    """
    try:
        investigation_id, record = _service.investigate(
            namespace=namespace,
            pod_name=pod_name,
            cluster_name=cluster_name,
        )
    except KubernetesInvestigationError as exc:
        return f"Investigation failed: {exc}"

    cluster_label = cluster_name or "local cluster"
    lines = [
        f"Investigation ID: {investigation_id}",
        f"Cluster: {cluster_label}",
        f"Pod: {namespace}/{pod_name}",
        f"Failure Class: {record.failure_class}",
        "",
        "ANALYSIS:",
        record.analysis,
        "",
        "REMEDIATION ACTIONS:",
    ]

    if not record.remediation_plan.actions:
        lines.append("  No actions proposed.")
    else:
        for action in record.remediation_plan.actions:
            auto_executed = action.executable and action.risk_level != "HIGH"
            status = (
                "Auto-executed by system"
                if auto_executed
                else f"Recommendation only — {action.policy_reason}"
            )
            lines += [
                f"  • [{action.risk_level}] {action.candidate_id}",
                f"    Description: {action.description}",
                f"    Rationale: {action.rationale}",
                f"    Status: {status}",
                "",
            ]

    if record.remediation_plan.parse_error:
        lines += [f"Note: {record.remediation_plan.parse_error}"]

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
