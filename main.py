"""CLI runner: audit one YouTube video end to end.

Usage:
    uv run python main.py "https://www.youtube.com/shorts/ZZgIOqZXdSA"
"""

import argparse
import logging
import uuid
from typing import Any, Dict

from dotenv import load_dotenv

from backend.src.graph.workflow import app


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = _parse_args()
    final_state = run_audit(args.url)
    _print_report(final_state)


def run_audit(video_url: str) -> Dict[str, Any]:
    """Run the full graph for one video and return the final state."""
    initial_state = {
        "video_url": video_url,
        "video_id": f"vid_{uuid.uuid4().hex[:8]}",  # unique name per run
        "compliance_results": [],
        "errors": [],
    }
    return app.invoke(initial_state)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brand Guardian AI - video compliance audit"
    )
    parser.add_argument("url", help="YouTube video URL to audit")
    return parser.parse_args()


def _print_report(final_state: Dict[str, Any]) -> None:
    print("\n=== COMPLIANCE AUDIT REPORT ===")
    print(f"Video ID : {final_state.get('video_id')}")
    print(f"Status   : {final_state.get('final_status')}")
    print(f"Metadata : {final_state.get('video_metadata')}")
    print("\nReport:")
    print(final_state.get("final_report"))

    errors = final_state.get("errors") or []
    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
