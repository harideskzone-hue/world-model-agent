# extractor/confidence_scorer.py
# ============================================================================
# Stage 4: Calibrated confidence assignment.
# Spec Reference: Implementation Plan v2.0, Section 4, Stage 4
# ============================================================================

from __future__ import annotations

import logging
from typing import List, Set

from shared.models import RawExtraction, CandidateFact
from shared.enums import RelationType, ExtractionType, ExtractionMethod, SegmentType
from shared.config import ExtractorConfig

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Stage 4 of the Extractor pipeline.

    Assigns calibrated confidence scores to validated extractions based on:
      - Extraction type (direct/implied/negation)
      - Extraction method (SLM vs rule_fallback)
      - Entity novelty (previously unseen entities get a penalty)
      - Relation-specific boosts (navigation → high certainty)

    Scoring formula:
      confidence = clamp(base + adjustments, 0.10, 0.99)
    """

    def __init__(self, config: ExtractorConfig | None = None):
        self._config = config or ExtractorConfig()
        self._known_entities: Set[str] = set()

    def score(
        self,
        extractions: List[RawExtraction],
        known_entities: Set[str] | None = None,
    ) -> List[CandidateFact]:
        """
        Assign confidence scores and convert to CandidateFact objects.

        Args:
            extractions: Validated extractions from Stage 3
            known_entities: Set of entity names already in the world model.
                           Used for novelty penalty calculation.

        Returns:
            List of CandidateFact with calibrated confidence values
        """
        if known_entities is not None:
            self._known_entities = known_entities

        results: List[CandidateFact] = []

        for ext in extractions:
            confidence = self._calculate_confidence(ext)
            extraction_type = self._parse_extraction_type(ext.extraction_type)
            extraction_method = self._parse_extraction_method(ext.extraction_method)

            # Parse relation string to RelationType enum
            try:
                relation = RelationType(ext.relation)
            except ValueError:
                logger.warning(f"Invalid relation '{ext.relation}' slipped past validator")
                continue

            fact = CandidateFact(
                subject=ext.subject,
                relation=relation,
                object=ext.object,
                confidence=confidence,
                source_turn_id=(
                    ext.source_segment.source_turn_id if ext.source_segment else 0
                ),
                extraction_type=extraction_type,
                extraction_method=extraction_method,
            )
            results.append(fact)

            # Track entities for future novelty checks
            self._known_entities.add(ext.subject)
            self._known_entities.add(ext.object)

        logger.debug(f"ConfidenceScorer: scored {len(results)} facts")
        return results

    def _calculate_confidence(self, ext: RawExtraction) -> float:
        """
        Calculate calibrated confidence score.

        Base confidence by extraction_type:
          direct   → 0.90
          implied  → 0.65
          negation → 0.80

        Adjustments:
          +0.05 if extraction_method == "slm"
          -0.10 if extraction_method == "rule_fallback"
          -0.15 if entity is previously unseen (novel entity)
          +0.05 if relation == "located_in" and segment is NAVIGATION
        """
        cfg = self._config

        # Base confidence
        ext_type = ext.extraction_type.lower()
        if ext_type == "direct":
            base = cfg.confidence_direct
        elif ext_type == "implied":
            base = cfg.confidence_implied
        elif ext_type == "negation":
            base = cfg.confidence_negation
        else:
            base = cfg.confidence_implied  # Unknown → conservative

        # Method adjustment
        if ext.extraction_method == "slm":
            base += cfg.confidence_boost_slm
        elif ext.extraction_method == "rule_fallback":
            base += cfg.confidence_penalty_rule

        # Novelty penalty
        if (ext.subject not in self._known_entities or
                ext.object not in self._known_entities):
            base += cfg.confidence_penalty_novel_entity

        # Navigation boost
        if (ext.relation == "located_in" and
                ext.source_segment and
                ext.source_segment.segment_type == SegmentType.NAVIGATION):
            base += cfg.confidence_boost_navigation

        # Clamp
        return max(cfg.confidence_min, min(cfg.confidence_max, round(base, 4)))

    def _parse_extraction_type(self, raw: str) -> ExtractionType:
        """Parse extraction type string to enum."""
        mapping = {
            "direct": ExtractionType.DIRECT,
            "implied": ExtractionType.IMPLIED,
            "negation": ExtractionType.NEGATION,
        }
        return mapping.get(raw.lower(), ExtractionType.IMPLIED)

    def _parse_extraction_method(self, raw: str) -> ExtractionMethod:
        """Parse extraction method string to enum."""
        if raw.lower() == "slm":
            return ExtractionMethod.SLM
        return ExtractionMethod.RULE_FALLBACK
