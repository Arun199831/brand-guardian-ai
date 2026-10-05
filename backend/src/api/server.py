import logging
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.src.graph.workflow import app as audit_graph

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
for noisy in ("azure", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)  # hide HTTP chatter
logger = logging.getLogger("brand-guardian-api")

app = FastAPI(
    title="Brand Guardian AI",
    description="Audits YouTube videos against brand and advertising compliance rules.",
    version="1.0.0",
)


# ---------- Request / response models ----------
class AuditRequest(BaseModel):
    video_url: str = Field(..., min_length=1, description="YouTube video URL to audit")


class ComplianceIssueOut(BaseModel):
    category: str
    description: str
    severity: str
    timestamp: Optional[str] = None


class AuditResponse(BaseModel):
    video_id: str
    status: str
    final_report: str
    compliance_results: List[ComplianceIssueOut]
    errors: List[str]


# ---------- Endpoints ----------
@app.post("/audit", response_model=AuditResponse)
def audit_video(request: AuditRequest) -> AuditResponse:
    """Run the full audit. Plain `def` on purpose: invoke() blocks for minutes."""
    video_id = f"vid_{uuid.uuid4().hex[:8]}"
    logger.info(f"Audit request received: video_id={video_id}")

    initial_state = {
        "video_url": request.video_url,
        "video_id": video_id,
        "compliance_results": [],
        "errors": [],
    }

    try:
        final_state = audit_graph.invoke(initial_state)
    except Exception:
        logger.exception(f"Unexpected failure for video_id={video_id}")
        raise HTTPException(
            status_code=500, detail="Audit failed unexpectedly. Check server logs."
        )

    return _build_response(video_id, final_state)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


# ---------- Helpers ----------
def _build_response(video_id: str, final_state: Dict[str, Any]) -> AuditResponse:
    issues = [
        _to_issue_out(issue) for issue in final_state.get("compliance_results") or []
    ]
    return AuditResponse(
        video_id=video_id,
        status=final_state.get("final_status") or "FAIL",
        final_report=final_state.get("final_report") or "No report generated.",
        compliance_results=issues,
        errors=final_state.get("errors") or [],
    )


def _to_issue_out(issue: Dict[str, Any]) -> ComplianceIssueOut:
    timestamp = issue.get("timestamp")
    if timestamp in (None, "", "null", "None"):
        timestamp = None
    return ComplianceIssueOut(
        category=str(issue.get("category", "Unknown")),
        description=str(issue.get("description", "")),
        severity=str(issue.get("severity", "medium")),
        timestamp=str(timestamp) if timestamp is not None else None,
    )
