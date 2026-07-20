# world_model/temporal_versioning.py
# ============================================================================
# Bi-temporal edge validity and supersession logic.
# Spec Reference: Implementation Plan v2.0, Section 5.3 (Edge Schema)
#                 Graphiti-style non-destructive versioning.
# ============================================================================

from typing import Optional

from shared.models import Edge
from shared.enums import EdgeStatus


def close_validity(edge: Edge, turn_id: int, superseded_by_id: str, reason: str) -> Edge:
    """
    Mark an existing edge as superseded — close its validity window.
    
    This is the ONLY way an edge's status transitions from ACTIVE to SUPERSEDED.
    The edge is never deleted — only its temporal fields are updated.

    Args:
        edge: The edge to supersede (will be mutated)
        turn_id: The turn at which supersession occurs
        superseded_by_id: ID of the new edge replacing this one
        reason: Human-readable revision reason

    Returns:
        The mutated edge (same object, for chaining)
    """
    edge.status = EdgeStatus.SUPERSEDED
    edge.t_valid_until = turn_id
    edge.superseded_by = superseded_by_id
    edge.revision_reason = reason
    return edge


def extend_validity(edge: Edge, turn_id: int) -> Edge:
    """
    Extend an edge's validity window (corroboration — same fact seen again).

    Args:
        edge: The edge to reinforce (will be mutated)
        turn_id: The turn confirming this edge

    Returns:
        The mutated edge
    """
    edge.corroboration_count += 1
    edge.t_valid_until = None  # Still active — no end
    # Confidence boost: diminishing returns on corroboration
    boost = 0.05 / edge.corroboration_count  # 0.05, 0.025, 0.0167, ...
    edge.confidence = min(0.99, edge.confidence + boost)
    return edge


def is_edge_active_at_turn(edge: Edge, turn_id: int) -> bool:
    """
    Check if an edge was active at a specific turn (time-travel query).

    An edge is active at turn T if:
      - t_valid_from <= T
      - AND (t_valid_until is None OR T < t_valid_until)
    """
    if edge.t_valid_from > turn_id:
        return False
    if edge.t_valid_until is not None and turn_id >= edge.t_valid_until:
        return False
    return True


def create_replacement_edge(
    old_edge: Edge,
    new_object: str,
    new_confidence: float,
    turn_id: int,
    extraction_method,
    direction: Optional[str] = None,
) -> Edge:
    """
    Create a new edge that replaces (supersedes) an old one.
    
    Inherits subject and relation from the old edge,
    but gets a new object value and fresh temporal fields.

    Args:
        old_edge: The edge being superseded
        new_object: The new value for the object field
        new_confidence: Confidence of the new belief
        turn_id: Current turn number
        extraction_method: How this new fact was extracted
        direction: Optional direction (for connects_to edges)

    Returns:
        A new Edge instance with fresh ID and temporal fields
    """
    return Edge(
        subject=old_edge.subject,
        relation=old_edge.relation,
        object=new_object,
        confidence=new_confidence,
        source_turn_id=turn_id,
        extraction_method=extraction_method,
        t_observed=turn_id,
        t_valid_from=turn_id,
        t_valid_until=None,  # Active until superseded
        status=EdgeStatus.ACTIVE,
        corroboration_count=1,
        direction=direction or old_edge.direction,
    )
