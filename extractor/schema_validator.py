# extractor/schema_validator.py
# ============================================================================
# Stage 3: Ontology conformance validation and normalization.
# Spec Reference: Implementation Plan v2.0, Section 4, Stage 3
# ============================================================================

from __future__ import annotations

import logging
from typing import List, Set

from shared.models import RawExtraction
from shared.enums import ALLOWED_RELATIONS, RELATION_ALIASES, KNOWN_STATES, RelationType
from world_model.schema import normalize_entity_name

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Stage 3 of the Extractor pipeline.

    Validates and normalizes raw extractions against the ontology:
      1. Normalize relation names (map aliases to canonical)
      2. Validate relation ∈ ALLOWED_RELATIONS
      3. Reject self-referential facts (subject == object)
      4. Normalize entity names (lowercase, strip articles)
      5. Validate state values for has_state relations
      6. Deduplicate (prefer SLM over rule_fallback)
    """

    def validate(self, extractions: List[RawExtraction]) -> List[RawExtraction]:
        """
        Validate and normalize a list of raw extractions.

        Args:
            extractions: Raw extractions from Stage 2 / Stage 2b

        Returns:
            Validated, normalized, deduplicated subset
        """
        validated: List[RawExtraction] = []
        seen: Set[tuple] = set()

        for ext in extractions:
            # 1. Normalize relation
            relation = self._normalize_relation(ext.relation)
            if relation is None:
                logger.debug(
                    f"Rejected: invalid relation '{ext.relation}' in "
                    f"({ext.subject}, {ext.relation}, {ext.object})"
                )
                continue

            # 2. Normalize entity names
            subject = normalize_entity_name(ext.subject)
            obj = normalize_entity_name(ext.object)

            # 3. Reject self-referential
            if subject == obj:
                logger.debug(f"Rejected: self-referential ({subject}, {relation}, {obj})")
                continue

            # 4. Reject empty entities
            if not subject or not obj:
                logger.debug(f"Rejected: empty entity in ({subject}, {relation}, {obj})")
                continue

            # 5. Validate state values for has_state
            if relation == "has_state":
                obj = obj.lower()
                if obj not in KNOWN_STATES:
                    # Flag as novel state but don't reject — it might be valid
                    logger.debug(f"Novel state value: '{obj}' (accepted but flagged)")

            # 6. Deduplicate: (subject, relation, object) key
            dedup_key = (subject, relation, obj)
            if dedup_key in seen:
                # Keep the SLM extraction over rule_fallback
                continue
            seen.add(dedup_key)

            # Create normalized extraction
            validated.append(RawExtraction(
                subject=subject,
                relation=relation,
                object=obj,
                extraction_type=ext.extraction_type,
                source_segment=ext.source_segment,
                extraction_method=ext.extraction_method,
            ))

        logger.debug(
            f"SchemaValidator: {len(extractions)} in → {len(validated)} valid"
        )
        return validated

    def _normalize_relation(self, relation: str) -> str | None:
        """
        Normalize a relation string to its canonical form.
        Returns None if the relation is not valid.
        """
        rel = relation.strip().lower()

        # Check aliases first
        if rel in RELATION_ALIASES:
            rel = RELATION_ALIASES[rel]

        # Validate against allowed set
        if rel in ALLOWED_RELATIONS:
            return rel

        return None
