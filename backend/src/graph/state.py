import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class ComplianceIssue(TypedDict):
    """A single violation detected by the Auditor node."""

    category: str
    description: str
    severity: str
    timestamp: Optional[str]


class VideoAuditState(TypedDict):
    """Execution context for the video compliance audit graph."""

    # --- Input (set by main.py / API before the graph starts) ---
    video_url: str
    video_id: str

    # --- Ingestion & extraction (written by the Indexer node) ---
    local_file_path: Optional[str]
    video_metadata: Optional[Dict[str, Any]]
    transcript: Optional[str]
    ocr_text: List[str]

    # --- Analysis (written by the Auditor node) ---
    compliance_results: Annotated[List[ComplianceIssue], operator.add]

    # final_status
    final_status: str
    final_report: str

    errors: Annotated[List[str], operator.add]
