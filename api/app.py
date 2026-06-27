from __future__ import annotations

import logging

from fastapi import FastAPI, status
from fastapi.responses import StreamingResponse

from api.schemas import InvestigationRequest
from mcp_server.server import mcp
from services.investigation_service import InvestigationService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = FastAPI(title="Kubernetes Troubleshooting Copilot", version="0.1.0")
app.mount("/mcp", mcp.streamable_http_app())
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
            cluster_name=request.cluster_name,
        ),
        media_type="application/x-ndjson",
        status_code=status.HTTP_201_CREATED,
    )


