from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

from kubernetes import client as k8s_client

from api.schemas import ActionResponse, InvestigationResponse
from clients.ollama_client import OllamaClient
from models.investigation_context import InvestigationContext
from models.remediation_plan import RemediationAction, RemediationPlan
from services.cluster_registry import get_api_client
from services.kubernetes_investigator import KubernetesInvestigationError, KubernetesInvestigator
from services.remediation_candidate_generator import RemediationCandidateGenerator
from services.remediation_executor import RemediationExecutionError, RemediationExecutor
from services.remediation_planner import RemediationPlanner
from services.troubleshooting_copilot import TroubleshootingCopilot


LOGGER = logging.getLogger(__name__)


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TOKEN = ""


@dataclass(frozen=True, slots=True)
class InvestigationRecord:
    """Stored investigation state used for approved remediation execution."""

    context: InvestigationContext
    failure_class: str
    analysis: str
    remediation_plan: RemediationPlan
    cluster_name: str | None = None


class InvestigationService:
    """Coordinates evidence collection, LLM analysis, and remediation planning."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        token: str | None = None,
    ) -> None:
        self._base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
        self._model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self._token = token if token is not None else os.getenv("OLLAMA_TOKEN", DEFAULT_TOKEN)
        self._candidate_generator = RemediationCandidateGenerator()
        self._remediation_planner = RemediationPlanner()
        self._investigations: dict[str, InvestigationRecord] = {}

    def get_investigation(self, investigation_id: str) -> InvestigationRecord | None:
        """Return a stored investigation record, or None if not found."""
        return self._investigations.get(investigation_id)

    def investigate(
        self,
        namespace: str,
        pod_name: str,
        cluster_name: str | None = None,
    ) -> tuple[str, InvestigationRecord]:
        """Run a full investigation against the given cluster and return the record."""
        api_client = get_api_client(cluster_name)
        ollama_client = OllamaClient(
            base_url=self._base_url,
            model=self._model,
            token=self._token,
        )
        copilot = TroubleshootingCopilot(ollama_client=ollama_client)
        investigator = KubernetesInvestigator(
            core_v1_api=k8s_client.CoreV1Api(api_client),
        )

        context = investigator.investigate_pod(namespace=namespace, pod_name=pod_name)
        candidate_set = self._candidate_generator.generate(context)
        analysis = copilot.analyze(
            context=context,
            remediation_candidates=candidate_set.candidates,
        )
        remediation_plan = self._remediation_planner.parse(
            llm_response=analysis,
            context=context,
            candidates=candidate_set.candidates,
        )
        investigation_id = str(uuid.uuid4())
        record = InvestigationRecord(
            context=context,
            failure_class=str(candidate_set.classification),
            analysis=analysis,
            remediation_plan=remediation_plan,
            cluster_name=cluster_name,
        )
        self._investigations.clear()
        self._investigations[investigation_id] = record
        self._auto_execute(remediation_plan, api_client)
        return investigation_id, record

    def _auto_execute(
        self,
        remediation_plan: RemediationPlan,
        api_client: k8s_client.ApiClient,
    ) -> None:
        """Execute low-risk eligible actions automatically on the target cluster."""
        for action in remediation_plan.actions:
            if action.risk_level == "HIGH" or not action.executable:
                LOGGER.info(
                    "Skipping auto-execution of action %s (risk=%s, executable=%s)",
                    action.candidate_id,
                    action.risk_level,
                    action.executable,
                )
                continue
            try:
                executor = RemediationExecutor(
                    core_v1_api=k8s_client.CoreV1Api(api_client),
                    apps_v1_api=k8s_client.AppsV1Api(api_client),
                )
                result = executor.execute(action)
                LOGGER.info("Auto-executed %s: %s", action.candidate_id, result.message)
            except RemediationExecutionError as exc:
                LOGGER.warning("Auto-execution failed for %s: %s", action.candidate_id, exc)

    def investigate_stream(
        self,
        namespace: str,
        pod_name: str,
        cluster_name: str | None = None,
    ) -> Iterator[str]:
        """Stream LLM tokens then yield the final investigation result as NDJSON."""
        api_client = get_api_client(cluster_name)
        ollama_client = OllamaClient(
            base_url=self._base_url,
            model=self._model,
            token=self._token,
        )
        copilot = TroubleshootingCopilot(ollama_client=ollama_client)
        investigator = KubernetesInvestigator(
            core_v1_api=k8s_client.CoreV1Api(api_client),
        )

        try:
            context = investigator.investigate_pod(namespace=namespace, pod_name=pod_name)
        except KubernetesInvestigationError as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
            return

        candidate_set = self._candidate_generator.generate(context)

        tokens: list[str] = []
        for token in copilot.analyze_stream(
            context=context,
            remediation_candidates=candidate_set.candidates,
        ):
            tokens.append(token)
            yield json.dumps({"type": "token", "content": token}) + "\n"

        analysis = "".join(tokens)
        remediation_plan = self._remediation_planner.parse(
            llm_response=analysis,
            context=context,
            candidates=candidate_set.candidates,
        )
        investigation_id = str(uuid.uuid4())
        record = InvestigationRecord(
            context=context,
            failure_class=str(candidate_set.classification),
            analysis=analysis,
            remediation_plan=remediation_plan,
            cluster_name=cluster_name,
        )
        self._investigations.clear()
        self._investigations[investigation_id] = record

        for action in remediation_plan.actions:
            if action.risk_level == "HIGH" or not action.executable:
                continue
            try:
                executor = RemediationExecutor(
                    core_v1_api=k8s_client.CoreV1Api(api_client),
                    apps_v1_api=k8s_client.AppsV1Api(api_client),
                )
                result = executor.execute(action)
                yield json.dumps({
                    "type": "action_executed",
                    "candidate_id": action.candidate_id,
                    "success": result.success,
                    "message": result.message,
                }) + "\n"
            except RemediationExecutionError as exc:
                yield json.dumps({
                    "type": "action_error",
                    "candidate_id": action.candidate_id,
                    "error": str(exc),
                }) + "\n"

        response = self.to_response(investigation_id, record)
        yield json.dumps({"type": "result", **response.model_dump()}) + "\n"

    def to_response(
        self,
        investigation_id: str,
        record: InvestigationRecord,
    ) -> InvestigationResponse:
        """Convert a stored investigation record into an API response."""
        return InvestigationResponse(
            investigation_id=investigation_id,
            cluster_name=record.cluster_name,
            namespace=record.context.namespace,
            pod_name=record.context.pod_name,
            failure_class=record.failure_class,
            analysis=record.analysis,
            actions=[
                self._action_to_response(action)
                for action in record.remediation_plan.actions
            ],
            remediation_parse_error=record.remediation_plan.parse_error,
        )

    def _action_to_response(self, action: RemediationAction) -> ActionResponse:
        """Convert an internal remediation action into an API schema."""
        return ActionResponse(
            candidate_id=action.candidate_id,
            action_type=action.action_type,
            action_category=action.action_category,
            target_kind=action.target_kind,
            target_name=action.target_name,
            namespace=action.namespace,
            risk_level=action.risk_level,
            solves_root_cause=action.solves_root_cause,
            executable=action.executable,
            policy_reason=action.policy_reason,
            description=action.description,
            rationale=action.rationale,
        )
