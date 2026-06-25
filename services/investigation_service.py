from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from api.schemas import ActionResponse, InvestigationResponse
from clients.litellm_client import LiteLLMClient
from clients.ollama_client import OllamaClient
from models.investigation_context import InvestigationContext
from models.remediation_plan import RemediationAction, RemediationPlan
from services.kubernetes_investigator import KubernetesInvestigationError, KubernetesInvestigator
from services.remediation_candidate_generator import RemediationCandidateGenerator
from services.remediation_planner import RemediationPlanner
from services.troubleshooting_copilot import LLMClient, TroubleshootingCopilot


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LITELLM_BASE_URL = "http://localhost:4000"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TOKEN = ""


@dataclass(frozen=True, slots=True)
class InvestigationRecord:
    """Stored investigation state used for approved remediation execution."""

    context: InvestigationContext
    failure_class: str
    analysis: str
    remediation_plan: RemediationPlan


class InvestigationService:
    """Coordinates evidence collection, LLM analysis, and remediation planning."""

    def __init__(self) -> None:
        self._provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self._ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self._ollama_token = os.getenv("OLLAMA_TOKEN", DEFAULT_TOKEN)
        self._litellm_base_url = os.getenv("LITELLM_BASE_URL", DEFAULT_LITELLM_BASE_URL)
        self._litellm_model = os.getenv("LITELLM_MODEL", DEFAULT_MODEL)
        self._litellm_token = os.getenv("LITELLM_TOKEN", DEFAULT_TOKEN)
        self._candidate_generator = RemediationCandidateGenerator()
        self._remediation_planner = RemediationPlanner()
        self._investigations: dict[str, InvestigationRecord] = {}

    def _create_client(self) -> LLMClient:
        """Return the configured LLM client based on LLM_PROVIDER."""
        if self._provider == "litellm":
            return LiteLLMClient(
                base_url=self._litellm_base_url,
                model=self._litellm_model,
                token=self._litellm_token,
            )
        return OllamaClient(
            base_url=self._ollama_base_url,
            model=self._ollama_model,
            token=self._ollama_token,
        )

    def get_investigation(self, investigation_id: str) -> InvestigationRecord | None:
        """Return a stored investigation record, or None if not found."""
        return self._investigations.get(investigation_id)

    def investigate(self, namespace: str, pod_name: str) -> tuple[str, InvestigationRecord]:
        """Run a full investigation and return the stored record."""
        copilot = TroubleshootingCopilot(llm_client=self._create_client())
        investigator = KubernetesInvestigator()

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
        )
        return investigation_id, record

    def investigate_stream(self, namespace: str, pod_name: str) -> Iterator[str]:
        """Stream LLM tokens then yield the final investigation result as NDJSON."""
        copilot = TroubleshootingCopilot(llm_client=self._create_client())
        investigator = KubernetesInvestigator()

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
        )
        self._investigations.clear()
        self._investigations[investigation_id] = record

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
