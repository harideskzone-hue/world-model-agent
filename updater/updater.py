# updater/updater.py
# ============================================================================
# Main Updater class — processes candidate facts and updates the world model.
# Spec Reference: Implementation Plan v2.0, Section 6.3
# ============================================================================

from __future__ import annotations

import logging
from typing import List

from shared.models import (
    CandidateFact, Node, Edge, UpdateReport, RevisionAction,
)
from shared.enums import (
    NodeType, NodeStatus, ContradictionType, RevisionActionType,
)
from world_model.graph_store import GraphStoreBase
from world_model.schema import normalize_entity_name
from updater.contradiction_detector import ContradictionDetector
from updater.revision_policy import RevisionPolicy
from updater.corroboration import apply_corroboration

logger = logging.getLogger(__name__)


class Updater:
    """
    Processes candidate facts from the Extractor and updates the World Model.

    For each candidate fact:
      1. Ensure subject and object nodes exist in the graph
      2. Run ContradictionDetector against current graph
      3. Apply RevisionPolicy to determine action (EXPAND/CORROBORATE/SUPERSEDE)
      4. Execute the action on the GraphStore

    Produces an UpdateReport for the Orchestrator's logging and budget checks.
    """

    def __init__(self, graph: GraphStoreBase):
        self._graph = graph
        self._detector = ContradictionDetector(graph)
        self._policy = RevisionPolicy()

    def update(
        self, candidates: List[CandidateFact], turn_id: int
    ) -> UpdateReport:
        """
        Process all candidate facts from this turn's extraction.

        Args:
            candidates: Output of Extractor for this turn
            turn_id: Current turn number

        Returns:
            UpdateReport with counts and detailed revision log
        """
        report = UpdateReport(turn_id=turn_id)

        for candidate in candidates:
            try:
                action = self._process_candidate(candidate, turn_id)
                self._execute_action(action, turn_id)
                self._update_report(report, action)
            except Exception as e:
                logger.error(f"Failed to process candidate {candidate}: {e}")
                report.rejected += 1

        self._graph.set_current_turn(turn_id)

        logger.info(
            f"Turn {turn_id}: expanded={report.expanded}, "
            f"corroborated={report.corroborated}, revised={report.revised}, "
            f"rejected={report.rejected}"
        )
        return report

    def _process_candidate(
        self, candidate: CandidateFact, turn_id: int
    ) -> RevisionAction:
        """
        Process a single candidate fact through detection and policy.
        """
        candidate.subject = normalize_entity_name(candidate.subject)
        candidate.object = normalize_entity_name(candidate.object)
        # Ensure nodes exist, using explicit types provided by the Extractor
        self._ensure_node(
            candidate.subject, turn_id, node_type=candidate.subject_type
        )
        self._ensure_node(
            candidate.object, turn_id, is_state_value=(
                candidate.relation.value == "has_state"
            ), node_type=candidate.object_type
        )

        # Detect contradiction
        result = self._detector.detect(candidate)

        # Apply revision policy
        action = self._policy.resolve(result, candidate, turn_id)

        return action

    def _execute_action(self, action: RevisionAction, turn_id: int) -> None:
        """Execute a revision action on the graph store."""

        if action.action_type == RevisionActionType.EXPAND:
            if action.new_edge:
                self._graph.add_edge(action.new_edge)

        elif action.action_type == RevisionActionType.CORROBORATE:
            if action.existing_edge_id:
                self._graph.corroborate_edge(action.existing_edge_id, turn_id)

        elif action.action_type in (
            RevisionActionType.SUPERSEDE,
            RevisionActionType.PROVISIONAL_SUPERSEDE,
        ):
            if action.existing_edge_id and action.new_edge:
                self._graph.supersede_edge(
                    action.existing_edge_id,
                    action.new_edge,
                    action.revision_reason,
                )

    def _update_report(self, report: UpdateReport, action: RevisionAction) -> None:
        """Update the report counters based on the action type."""
        if action.action_type == RevisionActionType.EXPAND:
            report.expanded += 1
        elif action.action_type == RevisionActionType.CORROBORATE:
            report.corroborated += 1
        elif action.action_type in (
            RevisionActionType.SUPERSEDE,
            RevisionActionType.PROVISIONAL_SUPERSEDE,
        ):
            report.revised += 1
            report.revisions.append(action)

    def _ensure_node(
        self, name: str, turn_id: int, is_state_value: bool = False,
        node_type: NodeType | None = None
    ) -> None:
        """
        Ensure a node exists in the graph for this entity.
        State values (e.g., "locked") are not nodes — skip them.
        """
        if is_state_value:
            return  # State values aren't separate nodes

        normalized = normalize_entity_name(name)
        existing = self._graph.get_node(normalized)
        if existing is None:
            # Use explicit node_type from extraction, default to OBJECT if unknown
            final_type = node_type if node_type else NodeType.OBJECT
            
            self._graph.add_node(Node(
                id=normalized,
                name=name,
                node_type=final_type,
                confidence=0.9,  # Initial confidence
                first_observed_turn=turn_id,
                last_observed_turn=turn_id,
                corroboration_count=1,
            ))
