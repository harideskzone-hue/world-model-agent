# extractor/text_extractor.py
# ============================================================================
# TextExtractor: Orchestrates the full 4-stage extraction pipeline.
# Spec Reference: Implementation Plan v2.0, Section 4 (Complete API)
# ============================================================================

from __future__ import annotations

import logging
from typing import List, Set, Optional

from shared.models import Observation, CandidateFact, WorkingMemory
from shared.enums import SegmentType
from shared.config import ExtractorConfig
from extractor.base import ExtractorBase
from extractor.preprocessor import Preprocessor
from extractor.slm_extractor import SLMExtractor
from extractor.rule_fallback import RuleFallback
from extractor.schema_validator import SchemaValidator
from extractor.confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)


class TextExtractor(ExtractorBase):
    """
    Track 1 implementation: text observation → candidate facts.

    Orchestrates the 4-stage pipeline:
      Stage 1: Preprocessor   — clean, segment, classify, coreference
      Stage 2: SLMExtractor   — SLM-guided JSON extraction
      Stage 2b: RuleFallback  — regex fallback if SLM fails
      Stage 3: SchemaValidator — ontology conformance validation
      Stage 4: ConfidenceScorer — calibrated confidence assignment

    The SLM is optional — the pipeline gracefully degrades to rule-based
    extraction when unavailable, ensuring the agent always works.
    """

    def __init__(
        self,
        slm_runner=None,
        config: Optional[ExtractorConfig] = None,
        known_entities: Optional[Set[str]] = None,
    ):
        """
        Args:
            slm_runner: An SLMRunner instance for Stage 2. If None, only
                        rule-based fallback is used.
            config: Extraction configuration
            known_entities: Pre-existing entities in the world model
        """
        self._config = config or ExtractorConfig()
        self._preprocessor = Preprocessor()
        self._slm_extractor = SLMExtractor(slm_runner, self._config)
        self._rule_fallback = RuleFallback()
        self._schema_validator = SchemaValidator()
        self._confidence_scorer = ConfidenceScorer(self._config)
        self._known_entities = known_entities or set()

    def extract(
        self, observation: Observation, working_memory: WorkingMemory
    ) -> List[CandidateFact]:
        """
        Full 4-stage extraction pipeline.

        Args:
            observation: Current turn's Observation from Environment Wrapper
            working_memory: Current room facts + recent context for coreference

        Returns:
            List of CandidateFact, each with confidence and provenance metadata

        Guarantees:
            - All returned facts conform to OntologySchema
            - No fact has confidence outside [0.10, 0.99]
            - Entity names are normalized (lowercase, no articles)
            - At least attempts extraction; empty list for trivial observations
        """
        # ── Stage 1: Preprocess ──
        segments = self._preprocessor.process(observation, working_memory)

        if not segments:
            logger.debug(f"Turn {observation.turn_id}: no extractable segments")
            return []

        # ── Stage 2: SLM Extraction (with fallback) ──
        raw_extractions = self._slm_extractor.extract(segments, working_memory)

        # If SLM produced nothing or fewer results than segments, supplement with rule fallback
        extractable_count = sum(
            1 for s in segments if s.segment_type != SegmentType.FEEDBACK
        )
        if len(raw_extractions) < max(1, extractable_count):
            logger.debug("Supplementing with rule-based fallback extraction")
            rule_extractions = self._rule_fallback.extract(
                segments,
                current_room=working_memory.current_room or observation.location,
            )
            raw_extractions.extend(rule_extractions)

        if not raw_extractions:
            logger.debug(f"Turn {observation.turn_id}: no facts extracted")
            return []

        # ── Stage 3: Schema Validation ──
        validated = self._schema_validator.validate(raw_extractions)

        if not validated:
            logger.debug(f"Turn {observation.turn_id}: all extractions failed validation")
            return []

        # ── Stage 4: Confidence Scoring ──
        candidates = self._confidence_scorer.score(
            validated,
            known_entities=self._known_entities,
        )

        # Update known entities for future calls
        for fact in candidates:
            self._known_entities.add(fact.subject)
            self._known_entities.add(fact.object)

        logger.info(
            f"Turn {observation.turn_id}: extracted {len(candidates)} candidate facts"
        )
        return candidates

    def update_known_entities(self, entities: Set[str]) -> None:
        """Update the set of known entities (called when world model changes)."""
        self._known_entities.update(entities)
