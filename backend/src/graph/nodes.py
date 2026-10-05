"""LangGraph nodes for the Brand Guardian compliance workflow.

Two "contractors":
  1. index_video_node   - YouTube URL -> transcript + OCR (Azure Video Indexer)
  2. audit_content_node - transcript + OCR + compliance rules (RAG) -> violations
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import AzureSearch
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from backend.src.graph.state import ComplianceIssue, VideoAuditState
from backend.src.services.video_indexer import VideoIndexerService

logger = logging.getLogger("brand-guardian")

MAX_QUERY_CHARS = 2000  # keep the RAG query well under embedding token limits
RETRIEVED_RULES_K = 3  # how many rule chunks to pull from Azure AI Search
VALID_SEVERITIES = {"low", "medium", "high"}


# --------------------------------------------------------------------------
# NODE 1: Indexer
# --------------------------------------------------------------------------
def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    """Download the YouTube video, send it to Video Indexer, return transcript + OCR."""
    video_url = state.get("video_url", "")
    video_id = state.get("video_id", "vid_unknown")
    logger.info(f"[Indexer] Starting for video_id={video_id}")

    if not _is_youtube_url(video_url):
        return _indexer_failure(video_id, video_url, "Invalid or missing YouTube URL.")

    local_path = f"temp_{video_id}.mp4"

    try:
        service = VideoIndexerService()
        service.download_youtube_video(video_url, local_path)
        vi_video_id = service.upload_video(local_path, video_name=video_id)
        vi_json = service.wait_for_processing(vi_video_id)
        extracted = service.extract_data(vi_json)
        logger.info(f"[Indexer] Done for video_id={video_id}")
        return extracted  # {transcript, ocr_text, video_metadata}
    except Exception as e:
        logger.error(f"[Indexer] Failed: {e}")
        return _indexer_failure(video_id, video_url, str(e))
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


# --------------------------------------------------------------------------
# NODE 2: Auditor
# --------------------------------------------------------------------------
def audit_content_node(state: VideoAuditState) -> Dict[str, Any]:
    """Check the video content against retrieved compliance rules using GPT-4o."""
    transcript = state.get("transcript") or ""
    ocr_text = state.get("ocr_text") or ""
    video_metadata = state.get("video_metadata", {})
    logger.info("[Auditor] Starting compliance audit")

    if not transcript:
        return {
            "final_status": "FAIL",
            "final_report": "Audit skipped: no transcript was extracted from the video.",
        }

    # 1) Connect to the models and the knowledge base
    try:
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.0,
        )
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        vector_store = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            embedding_function=embeddings.embed_query,
        )
    except Exception as e:
        return _audit_failure(f"Could not initialise Azure clients: {e}")

    # 2) RAG: find the rules most relevant to this video
    try:
        query = _build_search_query(transcript, ocr_text)
        docs = vector_store.similarity_search(query, k=RETRIEVED_RULES_K)
    except Exception as e:
        return _audit_failure(f"Rule retrieval failed: {e}")

    if not docs:
        return _audit_failure(
            "No compliance rules found in the knowledge base. "
            "Did you run backend/scripts/index_documents.py?"
        )
    retrieved_rules = "\n\n".join(doc.page_content for doc in docs)

    # 3) Ask GPT-4o to find violations
    system_prompt = f"""You are a Brand Compliance Auditor.

OFFICIAL REGULATORY RULES:
{retrieved_rules}

INSTRUCTIONS:
1. Analyze the transcript and on-screen text below.
2. Identify ANY violations of the rules above.
3. Only report violations you can CONFIRM from the transcript, on-screen text or metadata provided.
   Do NOT report "potential" violations, and do NOT flag rules you cannot verify from the data.
4. Return ONLY a JSON array. No prose, no markdown.
   Each item must look like:
   {{"category": "...", "description": "...", "severity": "low|medium|high", "timestamp": "mm:ss or null"}}
5. If there are no violations, return [].
"""
    user_message = f"""VIDEO METADATA: {video_metadata}
TRANSCRIPT: {transcript}
ON-SCREEN TEXT (OCR): {_as_text(ocr_text)}
"""

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        )
    except Exception as e:
        return _audit_failure(f"LLM call failed: {e}")

    # 4) Parse the answer, then let CODE (not the LLM) decide PASS / FAIL
    try:
        issues = _parse_compliance_json(response.content)
    except Exception as e:
        return _audit_failure(f"Could not parse LLM response as JSON: {e}")

    status, report = _determine_status_and_report(issues)
    logger.info(f"[Auditor] Finished: {status} ({len(issues)} issue(s))")
    return {
        "compliance_results": issues,
        "final_status": status,
        "final_report": report,
    }


# --------------------------------------------------------------------------
# Helpers (stepdown rule: used above, defined below)
# --------------------------------------------------------------------------
def _is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def _indexer_failure(
    video_id: str, video_url: str, error_message: str
) -> Dict[str, Any]:
    """Standard return shape when the indexer cannot produce data."""
    return {
        "errors": [error_message],
        "final_status": "FAIL",
        "final_report": f"Indexing failed: {error_message}",
        "transcript": "",
        "ocr_text": [],
        "video_metadata": {"video_id": video_id, "video_url": video_url},
    }


def _audit_failure(error_message: str) -> Dict[str, Any]:
    """Standard return shape when the auditor cannot finish."""
    logger.error(f"[Auditor] {error_message}")
    return {
        "errors": [error_message],
        "final_status": "FAIL",
        "final_report": f"Audit failed: {error_message}",
    }


def _as_text(value: Any) -> str:
    """OCR may be a list of strings or one string - handle both."""
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    return str(value)


def _build_search_query(transcript: str, ocr_text: Any) -> str:
    """Combine transcript + OCR into one query, truncated to stay under token limits."""
    combined = f"{transcript} {_as_text(ocr_text)}".strip()
    return combined[:MAX_QUERY_CHARS]


def _parse_compliance_json(raw: str) -> List[ComplianceIssue]:
    """Turn the LLM's text into a clean list of ComplianceIssue dicts."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of violations.")

    issues: List[ComplianceIssue] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "medium")).lower()
        if severity not in VALID_SEVERITIES:
            severity = "medium"
        issues.append(
            {
                "category": str(item.get("category", "Unknown")),
                "description": str(item.get("description", "")),
                "severity": severity,
                "timestamp": item.get("timestamp"),
            }
        )
    return issues


def _determine_status_and_report(issues: List[ComplianceIssue]) -> Tuple[str, str]:
    """Code decides PASS/FAIL - never trust the LLM with the verdict."""
    if not issues:
        return "PASS", "No compliance violations detected."

    counts = {level: 0 for level in VALID_SEVERITIES}
    for issue in issues:
        counts[issue["severity"]] += 1

    lines = [
        f"Found {len(issues)} violation(s): "
        f"{counts['high']} high, {counts['medium']} medium, {counts['low']} low."
    ]
    for issue in issues:
        when = f" (at {issue['timestamp']})" if issue.get("timestamp") else ""
        lines.append(
            f"- [{issue['severity'].upper()}] {issue['category']}: {issue['description']}{when}"
        )
    return "FAIL", "\n".join(lines)
