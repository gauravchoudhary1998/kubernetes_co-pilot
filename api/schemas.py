from __future__ import annotations

from pydantic import BaseModel, Field


class InvestigationRequest(BaseModel):
    """Request to investigate one Kubernetes pod."""

    namespace: str = Field(min_length=1)
    pod_name: str = Field(min_length=1)


class ActionResponse(BaseModel):
    """Remediation action returned by the API."""

    candidate_id: str
    action_type: str
    action_category: str
    target_kind: str
    target_name: str
    namespace: str
    risk_level: str
    solves_root_cause: bool
    executable: bool
    policy_reason: str
    description: str
    rationale: str


class InvestigationResponse(BaseModel):
    """Response for a pod investigation."""

    investigation_id: str
    namespace: str
    pod_name: str
    failure_class: str
    analysis: str
    actions: list[ActionResponse]
    remediation_parse_error: str = ""


class ExecuteActionRequest(BaseModel):
    """Request to execute a stored remediation action."""

    approved: bool = False


class ExecuteActionResponse(BaseModel):
    """Response after attempting action execution."""

    success: bool
    message: str
