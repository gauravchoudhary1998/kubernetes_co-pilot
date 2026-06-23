from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

from api.schemas import (
    ExecuteActionRequest,
    ExecuteActionResponse,
    InvestigationRequest,
)
from services.investigation_service import InvestigationService
from services.remediation_executor import RemediationExecutionError, RemediationExecutor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = FastAPI(title="Kubernetes Troubleshooting Copilot", version="0.1.0")
investigation_service = InvestigationService()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Return basic service health."""
    return {"status": "ok"}


@app.post("/investigations", status_code=status.HTTP_201_CREATED)
def create_investigation(request: InvestigationRequest) -> StreamingResponse:
    """Investigate a failing pod, streaming LLM tokens then the final result as NDJSON."""
    return StreamingResponse(
        investigation_service.investigate_stream(
            namespace=request.namespace,
            pod_name=request.pod_name,
        ),
        media_type="application/x-ndjson",
        status_code=status.HTTP_201_CREATED,
    )


@app.post(
    "/investigations/{investigation_id}/actions/{candidate_id}/execute",
    response_model=ExecuteActionResponse,
)
def execute_action(
    investigation_id: str,
    candidate_id: str,
    request: ExecuteActionRequest,
) -> ExecuteActionResponse:
    """Execute a stored remediation action after explicit API approval."""
    if not request.approved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Action execution requires approved=true.",
        )

    record = investigation_service.get_investigation(investigation_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Investigation was not found.",
        )

    action = next(
        (
            action
            for action in record.remediation_plan.actions
            if action.candidate_id == candidate_id
        ),
        None,
    )
    if action is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Action candidate was not found for this investigation.",
        )

    if action.risk_level == "HIGH":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="High-risk actions are recommendation-only.",
        )
    if not action.executable:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Action is not executable: {action.policy_reason}",
        )

    try:
        result = RemediationExecutor().execute(action)
    except RemediationExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return ExecuteActionResponse(success=result.success, message=result.message)
