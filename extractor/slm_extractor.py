# extractor/slm_extractor.py
# ============================================================================
# Stage 2: SLM-guided structured extraction.
# Spec Reference: Implementation Plan v2.0, Section 4, Stage 2
#
# Uses the SLM to convert preprocessed text segments into structured
# candidate facts via schema-constrained JSON generation.
# ============================================================================

from __future__ import annotations

import json
import logging
from typing import List, Optional

from shared.models import TextSegment, RawExtraction, WorkingMemory
from shared.enums import SegmentType
from shared.config import ExtractorConfig

logger = logging.getLogger(__name__)

# ── Extraction prompt template ──────────────────────────────────────────────

EXTRACTION_PROMPT = """Extract facts from this text adventure observation as JSON.

SCHEMA:
- subject: entity name (room, object, or character name)
- relation: one of [contains, connects_to, located_in, holds, has_state, is_type]
- object: target entity name or state value
- extraction_type: one of [direct, implied, negation]

TEXT: "{text}"

CONTEXT:
- Current room: {current_room}
- Inventory: {inventory}

Output ONLY a JSON array of facts. No explanation.
Example: [{{"subject": "kitchen", "relation": "contains", "object": "brass key", "extraction_type": "direct"}}]

FACTS:"""

SIMPLIFIED_PROMPT = """Extract entities and relationships from this text as JSON array.
Text: "{text}"
Room: {current_room}

Format: [{{"subject": "name", "relation": "contains|connects_to|located_in|holds|has_state", "object": "name", "extraction_type": "direct"}}]

FACTS:"""


class SLMExtractor:
    """
    Stage 2 of the Extractor pipeline.

    Uses the SLM to parse natural language observations into structured
    (subject, relation, object) triples with extraction type metadata.

    Falls back to simplified prompt on first failure, then to rule-based
    extraction on second failure.
    """

    def __init__(self, slm_runner=None, config: Optional[ExtractorConfig] = None):
        """
        Args:
            slm_runner: An SLMRunner instance (injected). If None, only
                        rule-based fallback is available.
            config: Extraction configuration
        """
        self._slm = slm_runner
        self._config = config or ExtractorConfig()

    def extract(
        self,
        segments: List[TextSegment],
        working_memory: WorkingMemory,
    ) -> List[RawExtraction]:
        """
        Extract facts from preprocessed segments using the SLM.

        Algorithm:
          1. Filter out FEEDBACK segments (no facts)
          2. Batch segments (max 3 per SLM call)
          3. For each batch: construct prompt, call SLM, parse JSON
          4. On JSON parse failure: retry with simplified prompt
          5. On second failure: return empty (caller should use rule fallback)

        Args:
            segments: Preprocessed TextSegments from Stage 1
            working_memory: Current agent context

        Returns:
            List of RawExtraction with extraction_method="slm"
        """
        if self._slm is None:
            logger.warning("SLM not available — returning empty extractions")
            return []

        # Filter out feedback segments
        extractable = [
            s for s in segments if s.segment_type != SegmentType.FEEDBACK
        ]
        if not extractable:
            return []

        results: List[RawExtraction] = []

        # Batch segments
        batch_size = self._config.max_segments_per_batch
        for i in range(0, len(extractable), batch_size):
            batch = extractable[i:i + batch_size]
            batch_text = " ".join(s.resolved_text for s in batch)

            # Build context
            current_room = working_memory.current_room or "unknown"
            inventory_items = [e.object for e in working_memory.inventory_facts]
            inventory_str = ", ".join(inventory_items) if inventory_items else "empty"

            # Attempt extraction
            extractions = self._extract_with_retry(
                text=batch_text,
                current_room=current_room,
                inventory=inventory_str,
                source_segments=batch,
            )
            results.extend(extractions)

        logger.debug(f"SLMExtractor: extracted {len(results)} facts")
        return results

    def _extract_with_retry(
        self,
        text: str,
        current_room: str,
        inventory: str,
        source_segments: List[TextSegment],
    ) -> List[RawExtraction]:
        """
        Try full prompt, then simplified prompt, then give up.
        """
        # Attempt 1: Full prompt
        prompt = EXTRACTION_PROMPT.format(
            text=text, current_room=current_room, inventory=inventory,
        )
        result = self._call_and_parse(prompt, source_segments)
        if result is not None:
            return result

        # Attempt 2: Simplified prompt
        logger.debug("Full prompt failed, trying simplified prompt")
        prompt = SIMPLIFIED_PROMPT.format(
            text=text, current_room=current_room,
        )
        result = self._call_and_parse(prompt, source_segments)
        if result is not None:
            return result

        # Attempt 3: Give up — caller should use rule fallback
        logger.warning("SLM extraction failed after retries, returning empty")
        return []

    def _call_and_parse(
        self,
        prompt: str,
        source_segments: List[TextSegment],
    ) -> Optional[List[RawExtraction]]:
        """
        Call the SLM and parse JSON output.
        Returns None if parsing fails.
        """
        try:
            raw_output = self._slm.generate(
                prompt,
                temperature=self._config.confidence_direct,  # Low temp for extraction
            )
        except Exception as e:
            logger.error(f"SLM call failed: {e}")
            return None

        return self._parse_json_output(raw_output, source_segments)

    def _parse_json_output(
        self,
        raw_output: str,
        source_segments: List[TextSegment],
    ) -> Optional[List[RawExtraction]]:
        """
        Parse SLM output as JSON array of fact objects.
        Returns None if parsing fails.
        """
        # Clean up output — find JSON array
        text = raw_output.strip()

        # Try to find JSON array in the output
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.debug(f"No JSON array found in SLM output: {text[:100]}...")
            return None

        json_text = text[start:end + 1]

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse failed: {e}")
            return None

        if not isinstance(parsed, list):
            logger.debug("SLM output is not a JSON array")
            return None

        results = []
        first_segment = source_segments[0] if source_segments else None

        for item in parsed:
            if not isinstance(item, dict):
                continue

            subject = item.get("subject", "").strip()
            relation = item.get("relation", "").strip()
            obj = item.get("object", "").strip()
            ext_type = item.get("extraction_type", "direct").strip()

            if not subject or not relation or not obj:
                continue

            results.append(RawExtraction(
                subject=subject,
                relation=relation,
                object=obj,
                extraction_type=ext_type,
                source_segment=first_segment,
                extraction_method="slm",
            ))

        return results if results else None
