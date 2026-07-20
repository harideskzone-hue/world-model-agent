# extractor/preprocessor.py
# ============================================================================
# Stage 1: Text preprocessing, segmentation, and coreference resolution.
# Spec Reference: Implementation Plan v2.0, Section 4, Stage 1
# ============================================================================

from __future__ import annotations

import re
import logging
from typing import List, Optional

from shared.models import Observation, TextSegment, WorkingMemory
from shared.enums import SegmentType

logger = logging.getLogger(__name__)

# ── Boilerplate patterns to strip from TextWorld output ─────────────────────

# Pattern to EXTRACT room name from banner before stripping
_BANNER_PATTERN = re.compile(r"-=\s*(.+?)\s*=-")

_BOILERPLATE_PATTERNS = [
    re.compile(r"^-=.*=-.*$", re.MULTILINE),      # TextWorld banners: -= Kitchen =-
    re.compile(r"^\s*$", re.MULTILINE),             # Empty lines
    re.compile(r"^\u003e.*$", re.MULTILINE),              # Prompt lines
    re.compile(r"^[|\\/$\s_]+$", re.MULTILINE),   # ASCII art lines
    re.compile(r"^-=.*=-\d+/\d+$", re.MULTILINE),  # Score lines: -= Kitchen =-0/1
    re.compile(r"^\s*\d+/\d+\s*$", re.MULTILINE),  # Bare score lines: 0/1
]

# ── Segment classification patterns ─────────────────────────────────────────

_ROOM_PATTERNS = [
    re.compile(r"^you (?:are|'re) (?:in|standing in|at)\b", re.IGNORECASE),
    re.compile(r"^you(?:'ve)? (?:entered|arrived)", re.IGNORECASE),
    re.compile(r"^(?:well,?\s+)?here we are\b", re.IGNORECASE),  # TextWorld casual
    re.compile(r"^this (?:is|looks like)\b", re.IGNORECASE),
    re.compile(r"^you see\b", re.IGNORECASE),
    re.compile(r"^you (?:can )?make out\b", re.IGNORECASE),  # TextWorld "You make out a safe"
    re.compile(r"^there (?:is|are)\b", re.IGNORECASE),
    re.compile(r"^(?:a|an|the) .+ (?:is|are) here\b", re.IGNORECASE),
    re.compile(r"^you can (?:see|make out)\b", re.IGNORECASE),
    re.compile(r"^(?:you don't like|why not try)\b", re.IGNORECASE),  # TextWorld exit hints
    re.compile(r"^what a\b", re.IGNORECASE),  # "What a letdown!" (container empty)
]

_NAVIGATION_PATTERNS = [
    re.compile(r"^you go\b", re.IGNORECASE),
    re.compile(r"^you (?:head|walk|move|travel)\b", re.IGNORECASE),
    re.compile(r"^you enter\b", re.IGNORECASE),
    re.compile(r"^you leave\b", re.IGNORECASE),
]

_INVENTORY_PATTERNS = [
    re.compile(r"^you (?:pick up|take|grab)\b", re.IGNORECASE),
    re.compile(r"^you (?:drop|put down)\b", re.IGNORECASE),
    re.compile(r"^you(?:'re| are) carrying\b", re.IGNORECASE),
    re.compile(r"^you have\b", re.IGNORECASE),
]

_STATE_CHANGE_PATTERNS = [
    re.compile(r"^you (?:open|close|lock|unlock|turn on|turn off)\b", re.IGNORECASE),
    re.compile(r"^the .+ is now\b", re.IGNORECASE),
    re.compile(r"^you (?:cook|slice|dice|chop|eat)\b", re.IGNORECASE),
    re.compile(r"^(?:with a .+,? )?you (?:unlock|open)\b", re.IGNORECASE),
]

_FEEDBACK_PATTERNS = [
    re.compile(r"^(?:that|it)(?:'s| is)? not\b", re.IGNORECASE),
    re.compile(r"^you (?:can't|cannot)\b", re.IGNORECASE),
    re.compile(r"^(?:i )?don't (?:understand|know)\b", re.IGNORECASE),
    re.compile(r"^nothing (?:happens|useful)\b", re.IGNORECASE),
    re.compile(r"^you don't see\b", re.IGNORECASE),
]

# ── Coreference patterns ────────────────────────────────────────────────────

_PRONOUN_PATTERNS = {
    "it": re.compile(r"\bit\b", re.IGNORECASE),
    "them": re.compile(r"\bthem\b", re.IGNORECASE),
    "here": re.compile(r"\bhere\b", re.IGNORECASE),
}


class Preprocessor:
    """
    Stage 1 of the Extractor pipeline.

    Responsibilities:
      1. Clean/strip TextWorld boilerplate
      2. Segment text into sentences
      3. Classify each sentence by information type
      4. Resolve short-range coreferences using Working Memory
    """

    def process(
        self, observation: Observation, working_memory: WorkingMemory
    ) -> List[TextSegment]:
        """
        Preprocess an observation into classified, coreference-resolved segments.

        Args:
            observation: Raw observation from the environment
            working_memory: Current agent context for coreference resolution

        Returns:
            List of TextSegment ready for Stage 2 extraction
        """
        # 0. Extract room name from banner BEFORE stripping
        raw_for_banner = observation.description or observation.feedback or ""
        banner_room = self._extract_room_from_banner(raw_for_banner)
        if banner_room and not working_memory.current_room:
            working_memory.current_room = banner_room.lower()

        # 1. Combine observation text fields
        raw_text = self._combine_observation_text(observation)

        # 2. Strip boilerplate
        cleaned = self._strip_boilerplate(raw_text)

        # 3. Segment into sentences
        sentences = self._segment_sentences(cleaned)

        # 4. Classify and resolve coreferences
        segments = []
        for sentence in sentences:
            if not sentence.strip():
                continue

            segment_type = self._classify_segment(sentence)
            resolved = self._resolve_coreference(sentence, working_memory)
            entities = self._extract_entity_mentions(resolved, working_memory)

            segments.append(TextSegment(
                raw_text=sentence.strip(),
                resolved_text=resolved.strip(),
                segment_type=segment_type,
                referenced_entities=entities,
                source_turn_id=observation.turn_id,
            ))

        logger.debug(
            f"Preprocessor: {len(segments)} segments from turn {observation.turn_id}"
        )
        return segments

    def _combine_observation_text(self, observation: Observation) -> str:
        """
        Combine observation fields, prioritizing description over feedback
        to avoid ASCII art pollution from the TextWorld welcome banner.
        """
        parts = []
        # Use description first (cleaner — has room banner + text)
        if observation.description:
            parts.append(observation.description.strip())
        # Only add feedback if it differs from description (action responses)
        if observation.feedback:
            fb = observation.feedback.strip()
            # Skip if feedback is just a repeat of description or contains ASCII art
            if observation.description and observation.description.strip() in fb:
                # Feedback contains description — extract only the unique prefix
                unique_part = fb.replace(observation.description.strip(), "").strip()
                # Filter out ASCII art lines from the unique part
                clean_lines = []
                for line in unique_part.split("\n"):
                    line_s = line.strip()
                    if not line_s:
                        continue
                    # Skip lines that look like ASCII art
                    if re.match(r'^[|\\/$\s_{}()\[\]]+$', line_s):
                        continue
                    # Skip very long lines with mostly special chars
                    if len(line_s) > 20 and sum(c in '|\\/$_{}[]()' for c in line_s) > len(line_s) * 0.3:
                        continue
                    clean_lines.append(line_s)
                if clean_lines:
                    parts.append("\n".join(clean_lines))
            else:
                parts.append(fb)
        if observation.inventory:
            parts.append(observation.inventory.strip())
        return "\n".join(parts)

    def _extract_room_from_banner(self, text: str) -> Optional[str]:
        """
        Extract room name from TextWorld banner format: -= Kitchen =-
        Called BEFORE stripping boilerplate.
        """
        match = _BANNER_PATTERN.search(text)
        if match:
            room_name = match.group(1).strip()
            logger.debug(f"Extracted room from banner: '{room_name}'")
            return room_name
        return None

    def _strip_boilerplate(self, text: str) -> str:
        """Remove TextWorld banners, ASCII art, and formatting artifacts."""
        for pattern in _BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)
        # Also strip any remaining lines that are purely non-alphanumeric
        lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and re.search(r'[a-zA-Z]', stripped):
                lines.append(stripped)
        return "\n".join(lines)

    def _segment_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using rule-based approach.
        Handles newlines and period-capital boundaries.
        """
        # First split on newlines
        lines = text.split("\n")
        sentences = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Split on sentence boundaries (period/!/? followed by space + capital)
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', line)
            sentences.extend(parts)
        return sentences

    def _classify_segment(self, sentence: str) -> SegmentType:
        """Classify a sentence by information type using pattern matching."""
        s = sentence.strip()

        for pattern in _FEEDBACK_PATTERNS:
            if pattern.search(s):
                return SegmentType.FEEDBACK

        for pattern in _NAVIGATION_PATTERNS:
            if pattern.search(s):
                return SegmentType.NAVIGATION

        for pattern in _INVENTORY_PATTERNS:
            if pattern.search(s):
                return SegmentType.INVENTORY_UPDATE

        for pattern in _STATE_CHANGE_PATTERNS:
            if pattern.search(s):
                return SegmentType.STATE_CHANGE

        for pattern in _ROOM_PATTERNS:
            if pattern.search(s):
                return SegmentType.ROOM_DESCRIPTION

        # Default: treat as room description (most common in TextWorld)
        return SegmentType.ROOM_DESCRIPTION

    def _resolve_coreference(
        self, sentence: str, working_memory: WorkingMemory
    ) -> str:
        """
        Resolve short-range coreferences using Working Memory context.
        - "it" → last mentioned object
        - "here" → current room name
        """
        resolved = sentence

        # Resolve "here" → current room
        if working_memory.current_room:
            resolved = _PRONOUN_PATTERNS["here"].sub(
                working_memory.current_room, resolved
            )

        # Resolve "it" / "them" → most recently mentioned entity
        # Use the last entity from recent observations or current room facts
        last_entity = self._get_last_mentioned_entity(working_memory)
        if last_entity:
            resolved = _PRONOUN_PATTERNS["it"].sub(last_entity, resolved)
            resolved = _PRONOUN_PATTERNS["them"].sub(last_entity, resolved)

        return resolved

    def _get_last_mentioned_entity(self, working_memory: WorkingMemory) -> Optional[str]:
        """Get the most recently mentioned object entity from working memory."""
        # Look at inventory facts first (most recently manipulated)
        if working_memory.inventory_facts:
            return working_memory.inventory_facts[-1].object

        # Then current room contents
        from shared.enums import RelationType
        contains_facts = [
            e for e in working_memory.current_room_facts
            if e.relation == RelationType.CONTAINS
        ]
        if contains_facts:
            return contains_facts[-1].object

        return None

    def _extract_entity_mentions(
        self, text: str, working_memory: WorkingMemory
    ) -> List[str]:
        """
        Find entity names mentioned in text using Working Memory as a dictionary.
        Simple substring matching against known entities.
        """
        text_lower = text.lower()
        entities = []

        # Check current room
        if working_memory.current_room and working_memory.current_room.lower() in text_lower:
            entities.append(working_memory.current_room)

        # Check all entities from room facts and inventory
        known_entities = set()
        for edge in working_memory.current_room_facts:
            known_entities.add(edge.subject)
            known_entities.add(edge.object)
        for edge in working_memory.inventory_facts:
            known_entities.add(edge.object)

        for entity in known_entities:
            if entity.lower() in text_lower:
                entities.append(entity)

        return list(set(entities))
