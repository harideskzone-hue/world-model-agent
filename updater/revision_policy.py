# updater/revision_policy.py
# ============================================================================
# Three named revision rules (R1, R2, R3).
# Spec Reference: Implementation Plan v2.0, Section 6.2
# ============================================================================

from __future__ import annotations

import logging

from shared.models import CandidateFact, Edge, ContradictionResult, RevisionAction
from shared.enums import (
    ContradictionType, RevisionActionType, EdgeStatus, ExtractionMethod,
)
from world_model.temporal_versioning import create_replacement_edge

logger = logging.getLogger(__name__)

# ── Corroboration threshold for R2 ──────────────────────────────────────────
# If the old edge has been independently confirmed this many times or more,
# R2 fires instead of R1 (provisional supersede with confidence penalty).
CORROBORATION_THRESHOLD = 3
PROVISIONAL_CONFIDENCE_PENALTY = 0.20


class RevisionPolicy:
    """
    Applies the locked 3-rule revision policy to resolve contradictions.

    R1: Recency-Wins
        New observation contradicts older belief with low corroboration.
        → Hard supersede: old → superseded, new → active.

    R2: Corroboration-Override
        New observation contradicts older belief with HIGH corroboration (≥3).
        → Provisional accept: new → active but with confidence reduced by 0.20.
          Old → superseded with a note that a strong prior was overridden.

    R3: Unknown-Preservation
        No information about a slot. Never fill absence as false.
        → No action. Slot remains absent. Query returns UNKNOWN.
    """

    def resolve(
        self,
        result: ContradictionResult,
        candidate: CandidateFact,
        turn_id: int,
    ) -> RevisionAction:
        """
        Determine the appropriate revision action for a candidate fact.

        Args:
            result: ContradictionResult from ContradictionDetector
            candidate: The new candidate fact
            turn_id: Current turn number

        Returns:
            RevisionAction specifying what to do
        """
        # ── EXPAND: no conflict ──
        if result.contradiction_type == ContradictionType.EXPAND:
            new_edge = self._candidate_to_edge(candidate, turn_id)
            return RevisionAction(
                action_type=RevisionActionType.EXPAND,
                new_edge=new_edge,
                revision_reason=f"new fact at turn {turn_id}",
            )

        # ── CORROBORATE: same fact confirmed again ──
        if result.contradiction_type == ContradictionType.CORROBORATE:
            return RevisionAction(
                action_type=RevisionActionType.CORROBORATE,
                existing_edge_id=result.existing_edge.id if result.existing_edge else None,
                revision_reason=f"corroborated at turn {turn_id}",
            )

        # ── REVISE: contradiction detected ──
        existing = result.existing_edge
        if existing is None:
            # Shouldn't happen, but handle gracefully
            new_edge = self._candidate_to_edge(candidate, turn_id)
            return RevisionAction(
                action_type=RevisionActionType.EXPAND,
                new_edge=new_edge,
                revision_reason=f"revision fallback (no existing edge) at turn {turn_id}",
            )

        # Decide between R1 and R2 based on corroboration count
        if existing.corroboration_count >= CORROBORATION_THRESHOLD:
            # ── R2: Corroboration-Override ──
            # The old belief had strong evidence. Accept new but with penalty.
            penalized_confidence = max(
                0.10,
                candidate.confidence - PROVISIONAL_CONFIDENCE_PENALTY,
            )
            new_edge = self._candidate_to_edge(
                candidate, turn_id, override_confidence=penalized_confidence,
            )
            reason = (
                f"R2: provisional supersede at turn {turn_id} — "
                f"old had {existing.corroboration_count} corroborations, "
                f"confidence penalized by {PROVISIONAL_CONFIDENCE_PENALTY}"
            )
            logger.info(reason)
            return RevisionAction(
                action_type=RevisionActionType.PROVISIONAL_SUPERSEDE,
                existing_edge_id=existing.id,
                new_edge=new_edge,
                revision_reason=reason,
            )
        else:
            # ── R1: Recency-Wins ──
            # The old belief had weak evidence. Hard supersede.
            new_edge = self._candidate_to_edge(candidate, turn_id)
            reason = (
                f"R1: recency-wins supersede at turn {turn_id} — "
                f"old corroboration={existing.corroboration_count} < threshold={CORROBORATION_THRESHOLD}"
            )
            logger.info(reason)
            return RevisionAction(
                action_type=RevisionActionType.SUPERSEDE,
                existing_edge_id=existing.id,
                new_edge=new_edge,
                revision_reason=reason,
            )

    def _candidate_to_edge(
        self,
        candidate: CandidateFact,
        turn_id: int,
        override_confidence: float | None = None,
    ) -> Edge:
        """Convert a CandidateFact into a new Edge."""
        return Edge(
            subject=candidate.subject,
            relation=candidate.relation,
            object=candidate.object,
            confidence=override_confidence or candidate.confidence,
            source_turn_id=turn_id,
            extraction_method=candidate.extraction_method,
            t_observed=turn_id,
            t_valid_from=turn_id,
            t_valid_until=None,
        )
