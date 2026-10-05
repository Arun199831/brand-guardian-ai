from typing import Literal

from langgraph.graph import END, StateGraph

from backend.src.graph.nodes import audit_content_node, index_video_node
from backend.src.graph.state import VideoAuditState


def create_graph():
    """Build and compile the compliance-audit graph."""
    workflow = StateGraph(VideoAuditState)

    # Contractors (nodes)
    workflow.add_node("indexer", index_video_node)
    workflow.add_node("auditor", audit_content_node)

    # Wiring (edges)
    workflow.set_entry_point("indexer")
    workflow.add_conditional_edges(
        "indexer",
        _route_after_indexer,
        {"auditor": "auditor", "end": END},
    )
    workflow.add_edge("auditor", END)

    return workflow.compile()


def _route_after_indexer(state: VideoAuditState) -> Literal["auditor", "end"]:
    """Skip the (paid) GPT-4o audit if indexing failed."""
    if state.get("errors"):
        return "end"
    return "auditor"


# Compiled graph, imported by main.py and the API server
app = create_graph()
