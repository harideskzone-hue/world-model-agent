# updater/contradiction_detector.py
# ============================================================================
# Slot-based contradiction detection.
# Spec Reference: Implementation Plan v2.0, Section 6.1
# ============================================================================

from __future__ import annotations

import logging
from typing import Optional

from shared.models import CandidateFact, Edge, ContradictionResult
from shared.enums import (
    RelationType, ContradictionType, ConflictCategory, EdgeStatus,
)
from world_model.graph_store import GraphStoreBase
from world_model.schema import are_states_conflicting, is_state_relation, is_single_occupancy_relation, normalize_entity_name

logger = logging.getLogger(__name__)


class ContradictionDetector:
    """
    Detects contradictions between a candidate fact and existing beliefs.

    Algorithm:
      1. Look up existing active edges in the same (subject, relation) slot
      2. If slot is empty → EXPAND (new fact, no conflict)
      3. If exact match → CORROBORATE (reinforce existing belief)
      4. If value mismatch → check conflict type and return REVISE

    Conflict types:
      - STATE_MUTUAL_EXCLUSION: e.g., locked vs unlocked
      - LOCATION_CHANGE: object moved rooms
      - VALUE_MISMATCH: general different value in same slot
    """

    def __init__(self, graph: GraphStoreBase):
        self._graph = graph

    def detect(self, candidate: CandidateFact) -> ContradictionResult:
        """
        Check if a candidate fact contradicts any existing active belief.

        Args:
            candidate: A validated CandidateFact from the Extractor

        Returns:
            ContradictionResult indicating EXPAND, CORROBORATE, or REVISE
        """
        candidate.subject = normalize_entity_name(candidate.subject)
        candidate.object = normalize_entity_name(candidate.object)
        # 1. Slot lookup: find existing edges with same (subject, relation)
        existing_edges = self._graph.get_active_edges_by_slot(
            candidate.subject, candidate.relation
        )

        # 2. Empty slot → EXPAND
        if not existing_edges:
            return ContradictionResult(
                contradiction_type=ContradictionType.EXPAND,
            )

        # 3. Check each existing edge for match or conflict
        for existing in existing_edges:
            # 3a. Exact match → CORROBORATE
            if existing.object == candidate.object:
                return ContradictionResult(
                    contradiction_type=ContradictionType.CORROBORATE,
                    existing_edge=existing,
                )

            # 3b. State conflict check (for has_state relations)
            if is_state_relation(candidate.relation):
                if are_states_conflicting(existing.object, candidate.object):
                    return ContradictionResult(
                        contradiction_type=ContradictionType.REVISE,
                        existing_edge=existing,
                        conflict_category=ConflictCategory.STATE_MUTUAL_EXCLUSION,
                    )

            # 3c. Location conflict (for located_in — single occupancy)
            if is_single_occupancy_relation(candidate.relation):
                if existing.object != candidate.object:
                    return ContradictionResult(
                        contradiction_type=ContradictionType.REVISE,
                        existing_edge=existing,
                        conflict_category=ConflictCategory.LOCATION_CHANGE,
                    )

            # 3d. General value mismatch for single-valued slots
            # has_state with different non-conflicting values is still a mismatch
            if is_state_relation(candidate.relation):
                return ContradictionResult(
                    contradiction_type=ContradictionType.REVISE,
                    existing_edge=existing,
                    conflict_category=ConflictCategory.VALUE_MISMATCH,
                )

        # 4. For multi-valued slots (e.g., contains), different values are not conflicts
        #    A room can contain multiple objects. This is just EXPAND.
        return ContradictionResult(
            contradiction_type=ContradictionType.EXPAND,
        )
