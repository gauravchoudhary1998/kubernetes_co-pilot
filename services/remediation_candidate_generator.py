from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from models.investigation_context import InvestigationContext
from models.remediation_plan import RemediationCandidate


class FailureClassification(StrEnum):
    """Known Kubernetes failure classes used for deterministic remediation policy."""

    MISSING_CONFIG = "MISSING_CONFIG"
    IMAGE_PULL_FAILURE = "IMAGE_PULL_FAILURE"
    OOM_KILLED = "OOM_KILLED"
    PROBE_FAILURE = "PROBE_FAILURE"
    SCHEDULING_FAILURE = "SCHEDULING_FAILURE"
    VOLUME_MOUNT_FAILURE = "VOLUME_MOUNT_FAILURE"
    CRASHING_APPLICATION = "CRASHING_APPLICATION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class RemediationCandidateSet:
    """Failure classification plus local remediation candidates."""

    classification: FailureClassification
    candidates: list[RemediationCandidate]


class RemediationCandidateGenerator:
    """Generates bounded remediation candidates from collected evidence."""

    def generate(self, context: InvestigationContext) -> RemediationCandidateSet:
        """Classify the failure and return locally approved remediation candidates."""
        classification = self._classify(context)
        return RemediationCandidateSet(
            classification=classification,
            candidates=self._candidates_for_classification(classification, context),
        )

    def _classify(self, context: InvestigationContext) -> FailureClassification:
        """Classify common pod failures using deterministic evidence checks."""
        evidence = "\n".join(
            [
                context.pod_phase,
                context.pod_status_reason,
                context.container_status_reasons,
                context.describe_output,
                context.logs,
                context.previous_logs,
                context.events,
            ]
        ).lower()

        if any(term in evidence for term in ("imagepullbackoff", "errimagepull")):
            return FailureClassification.IMAGE_PULL_FAILURE
        if any(term in evidence for term in ("pull access denied", "manifest unknown")):
            return FailureClassification.IMAGE_PULL_FAILURE
        if "oomkilled" in evidence:
            return FailureClassification.OOM_KILLED
        if any(term in evidence for term in ("failedscheduling", "nodes are available")):
            return FailureClassification.SCHEDULING_FAILURE
        if context.pod_phase.lower() == "pending":
            return FailureClassification.SCHEDULING_FAILURE
        if any(term in evidence for term in ("failedmount", "mountvolume", "unmountvolume")):
            return FailureClassification.VOLUME_MOUNT_FAILURE
        if "volume" in evidence and "not found" in evidence:
            return FailureClassification.VOLUME_MOUNT_FAILURE
        if any(term in evidence for term in ("liveness probe failed", "readiness probe failed")):
            return FailureClassification.PROBE_FAILURE
        if any(
            term in evidence
            for term in (
                "createcontainerconfigerror",
                "environment variable is not set",
                "required environment variable",
            )
        ):
            return FailureClassification.MISSING_CONFIG
        if any(term in evidence for term in ("configmap", "secret")) and "not found" in evidence:
            return FailureClassification.MISSING_CONFIG
        if "crashloopbackoff" in evidence:
            return FailureClassification.CRASHING_APPLICATION

        return FailureClassification.UNKNOWN

    def _candidates_for_classification(
        self,
        classification: FailureClassification,
        context: InvestigationContext,
    ) -> list[RemediationCandidate]:
        """Return remediation candidates for one failure class."""
        candidate = {
            FailureClassification.MISSING_CONFIG: self._manual_candidate(
                context=context,
                candidate_id="fix_missing_config",
                description=(
                    "Fix the missing or invalid Secret, ConfigMap, or environment "
                    "reference in the owning workload manifest."
                ),
                rationale=(
                    "The evidence points to configuration required by the container. "
                    "Restarting or deleting the pod will recreate the same failure."
                ),
            ),
            FailureClassification.IMAGE_PULL_FAILURE: self._manual_candidate(
                context=context,
                candidate_id="fix_image_pull",
                description=(
                    "Fix the image reference, registry credentials, image tag, or "
                    "imagePullSecret used by the owning workload."
                ),
                rationale=(
                    "Image pull failures are resolved by correcting image or registry "
                    "configuration, not by recycling the pod."
                ),
            ),
            FailureClassification.OOM_KILLED: self._manual_candidate(
                context=context,
                candidate_id="fix_memory_pressure",
                description=(
                    "Review memory usage and adjust application behavior or memory "
                    "requests and limits in the owning workload."
                ),
                rationale=(
                    "OOMKilled means the container exceeded its memory limit or node "
                    "memory pressure killed it. Pod restart alone does not address the cause."
                ),
            ),
            FailureClassification.PROBE_FAILURE: self._manual_candidate(
                context=context,
                candidate_id="fix_probe_or_app_health",
                description=(
                    "Fix the failing health endpoint, probe path, probe timing, or "
                    "application startup behavior in the workload spec."
                ),
                rationale=(
                    "Probe failures require correcting health behavior or probe config. "
                    "A rollout restart is only useful after that fix is made."
                ),
            ),
            FailureClassification.SCHEDULING_FAILURE: self._manual_candidate(
                context=context,
                candidate_id="fix_scheduling_constraints",
                description=(
                    "Resolve scheduling constraints such as insufficient resources, "
                    "taints, node selectors, affinity, or PVC binding issues."
                ),
                rationale=(
                    "Pending pods usually need scheduling constraints or cluster "
                    "capacity fixed before Kubernetes can place them."
                ),
            ),
            FailureClassification.VOLUME_MOUNT_FAILURE: self._manual_candidate(
                context=context,
                candidate_id="fix_volume_mount",
                description=(
                    "Fix the referenced volume, PVC, Secret, ConfigMap, CSI driver, "
                    "or mount configuration in the workload."
                ),
                rationale=(
                    "Volume mount failures are caused by storage or reference issues. "
                    "Recreating the pod usually repeats the same mount failure."
                ),
            ),
            FailureClassification.CRASHING_APPLICATION: self._manual_candidate(
                context=context,
                candidate_id="fix_application_crash",
                description=(
                    "Fix the application error shown in current or previous logs, "
                    "then roll out the corrected workload."
                ),
                rationale=(
                    "The pod is crashing after the container starts. Recycling it is "
                    "not a root-cause fix unless evidence shows a transient node issue."
                ),
            ),
            FailureClassification.UNKNOWN: self._manual_candidate(
                context=context,
                candidate_id="collect_more_evidence",
                description=(
                    "Collect more targeted evidence before executing remediation."
                ),
                rationale=(
                    "The current evidence does not support a safe automated fix."
                ),
            ),
        }[classification]

        return [candidate]

    def _manual_candidate(
        self,
        context: InvestigationContext,
        candidate_id: str,
        description: str,
        rationale: str,
    ) -> RemediationCandidate:
        """Build a recommendation-only candidate."""
        return RemediationCandidate(
            candidate_id=candidate_id,
            action_type="NO_ACTION",
            risk_level="LOW",
            action_category="FIX",
            description=description,
            rationale=rationale,
            target_kind="Unknown",
            target_name="",
            namespace=context.namespace,
            solves_root_cause=True,
            executable=False,
        )
