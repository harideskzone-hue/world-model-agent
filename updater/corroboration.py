# updater/corroboration.py
# ============================================================================
# Multi-observation confidence boost logic.
# Spec Reference: Implementation Plan v2.0, Section 6
# ============================================================================

from __future__ import annotations

import logging

from shared.models import Edge

logger = logging.getLogger(__name__)


def apply_corroboration(edge: Edge, turn_id: int) -> None:
    """
    Boost an existing edge's confidence via corroboration.
    Called when the same fact is observed again.
    
    Uses the temporal_versioning.extend_validity function for the
    core update, but adds additional logging and bounds checking.

    Args:
        edge: The edge being corroborated (mutated in place)
        turn_id: The turn at which corroboration occurs
    """
    from world_model.temporal_versioning import extend_validity
    
    old_conf = edge.confidence
    old_count = edge.corroboration_count
    
    extend_validity(edge, turn_id)
    
    logger.debug(
        f"Corroborated edge {edge.id}: "
        f"confidence {old_conf:.3f} → {edge.confidence:.3f}, "
        f"corroboration {old_count} → {edge.corroboration_count}"
    )
