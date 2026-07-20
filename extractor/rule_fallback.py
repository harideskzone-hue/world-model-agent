# extractor/rule_fallback.py
# ============================================================================
# Stage 2b: Rule-based fallback extraction for when SLM fails or is unavailable.
# Spec Reference: Implementation Plan v2.0, Section 4, Stage 2b
#
# These regex rules cover the most common TextWorld observation patterns.
# They serve as a reliable baseline and fallback when SLM output is malformed.
# ============================================================================

from __future__ import annotations

import re
import logging
from typing import List, Optional

from shared.models import TextSegment, RawExtraction
from shared.enums import SegmentType

logger = logging.getLogger(__name__)


# ── Navigation Rules ────────────────────────────────────────────────────────

_NAV_PATTERNS = [
    # "You go north." → player moved
    (re.compile(r"you (?:go|head|walk|move|travel)\s+(north|south|east|west|up|down)",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject="player", relation="located_in",
             object=ctx.get("new_room", "unknown room"),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "You enter the garden." → player in garden
    (re.compile(r"you enter (?:the |a |an )?(.+?)\.?$", re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject="player", relation="located_in",
             object=m.group(1).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
]

# ── Room Description Rules ──────────────────────────────────────────────────

_ROOM_PATTERNS = [
    # "You are in the kitchen." → player located_in kitchen
    (re.compile(r"you (?:are|'re) (?:in|standing in|at) (?:the |a |an )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject="player", relation="located_in",
             object=m.group(1).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "You've entered a kitchen." / "You have entered the kitchen."
    # TextWorld's primary room entry phrasing
    (re.compile(r"you(?:'ve| have)? entered (?:the |a |an )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject="player", relation="located_in",
             object=m.group(1).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "Well, here we are in the attic." / "here we are in the kitchen."
    # TextWorld's casual room description style
    (re.compile(r"(?:well,?\s+)?here we are in (?:the |a |an )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject="player", relation="located_in",
             object=m.group(1).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "You see a brass key." → current_room contains brass key
    (re.compile(r"you (?:see|can see|can make out|notice) (?:a |an |the )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: _parse_object_list(m.group(1), ctx.get("current_room", "unknown"))),
    # "You make out a closed safe in the corner." (TextWorld object + state)
    (re.compile(r"you (?:make out|see) (?:a |an |the )?(?:(opened?|closed|locked) )?(.+?)(?:\s+in the (?:corner|room|area))?\.?$",
                re.IGNORECASE),
     lambda m, ctx: _parse_object_with_state(
         m.group(2).strip().lower(), m.group(1), ctx)),
    # "On the table you see a key." (TextWorld container listing)
    (re.compile(r"on (?:the |a |an )?(.+?) you (?:see|can see|make out) (.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: _parse_container_list(
         m.group(1).strip().lower(), m.group(2), ctx)),
    # "There is a key on the table." → table contains key
    (re.compile(r"there (?:is|are) (?:a |an |the )?(.+?)\s+(?:on|in|inside) (?:the |a |an )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=m.group(2).strip().lower(), relation="contains",
             object=m.group(1).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "There is a key here." → current_room contains key
    (re.compile(r"there (?:is|are) (?:a |an |the )?(.+?) here\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=ctx.get("current_room", "unknown"), relation="contains",
             object=m.group(1).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "The safe contains a shadfly." → safe contains shadfly
    (re.compile(r"(?:the |a |an )?(.+?) contains? (?:a |an |the )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=m.group(1).strip().lower(), relation="contains",
             object=m.group(2).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "You open the safe, revealing a shadfly." → safe contains shadfly + safe has_state open
    (re.compile(r"you open (?:the |a |an )?(.+?),?\s*revealing (?:a |an |the )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=m.group(1).strip().lower(), relation="contains",
             object=m.group(2).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
         RawExtraction(
             subject=m.group(1).strip().lower(), relation="has_state",
             object="open",
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
]

# ── Inventory Rules ─────────────────────────────────────────────────────────

_INVENTORY_PATTERNS = [
    # "You pick up the brass key." → player holds brass key
    (re.compile(r"you (?:pick up|take|grab) (?:the |a |an )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject="player", relation="holds",
             object=m.group(1).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "You drop the brass key." → current_room contains brass key, player no longer holds
    (re.compile(r"you (?:drop|put down) (?:the |a |an )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=ctx.get("current_room", "unknown"), relation="contains",
             object=m.group(1).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
]

# ── State Change Rules ──────────────────────────────────────────────────────

_STATE_PATTERNS = [
    # "You unlock the door." → door has_state unlocked
    (re.compile(r"you (open|close|lock|unlock) (?:the |a |an )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=m.group(2).strip().lower(), relation="has_state",
             object=_action_to_state(m.group(1).strip().lower()),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "The door is now open." → door has_state open
    (re.compile(r"(?:the |a |an )?(.+?) is (?:now )?(\w+)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=m.group(1).strip().lower(), relation="has_state",
             object=m.group(2).strip().lower(),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "You cook the potato." → potato has_state cooked
    (re.compile(r"you (cook|slice|dice|chop|fry|roast|grill) (?:the |a |an )?(.+?)\.?$",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=m.group(2).strip().lower(), relation="has_state",
             object=_action_to_state(m.group(1).strip().lower()),
             extraction_type="direct", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
]

# ── Exit/Connection Rules ───────────────────────────────────────────────────

_EXIT_PATTERNS = [
    # "There is an exit to the north." / "You can go north."
    # "There is an exit to the north." / "unblocked exit to the east"
    (re.compile(r"(?:exit|door|passage|opening|way|entranceway)\s+(?:to the\s+)?(north|south|east|west|up|down)",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=ctx.get("current_room", "unknown"), relation="connects_to",
             object=f"room_{m.group(1).strip().lower()}",
             extraction_type="implied", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "try going south" / "try going east" (TextWorld exit hints)
    (re.compile(r"(?:try )?going\s+(north|south|east|west|up|down)",
                re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=ctx.get("current_room", "unknown"), relation="connects_to",
             object=f"room_{m.group(1).strip().lower()}",
             extraction_type="implied", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
]


# ── Helper Functions ────────────────────────────────────────────────────────

def _action_to_state(action: str) -> str:
    """Map an action verb to the resulting state."""
    mapping = {
        "open": "open",
        "close": "closed",
        "lock": "locked",
        "unlock": "unlocked",
        "cook": "cooked",
        "slice": "sliced",
        "dice": "diced",
        "chop": "chopped",
        "fry": "fried",
        "roast": "roasted",
        "grill": "grilled",
    }
    return mapping.get(action, action)


def _parse_object_with_state(
    obj_name: str, state: str | None, ctx: dict
) -> List[RawExtraction]:
    """
    Parse a TextWorld object with optional state.
    E.g., 'closed safe' → room contains safe, safe has_state closed.
    """
    results = []
    current_room = ctx.get("current_room", "unknown")

    # Object in room
    results.append(RawExtraction(
        subject=current_room, relation="contains",
        object=obj_name,
        extraction_type="direct", source_segment=None,
        extraction_method="rule_fallback",
    ))

    # Object state if present
    if state:
        # Normalize "opened" → "open"
        normalized_state = "open" if state.lower().startswith("open") else state.lower()
        results.append(RawExtraction(
            subject=obj_name, relation="has_state",
            object=normalized_state,
            extraction_type="direct", source_segment=None,
            extraction_method="rule_fallback",
        ))

    return results


def _parse_object_list(text: str, current_room: str) -> List[RawExtraction]:
    """Parse a comma/and-separated list of objects."""
    # Split on ", " and " and "
    parts = re.split(r",\s*|\s+and\s+", text)
    results = []
    for part in parts:
        obj = re.sub(r"^(?:a|an|the)\s+", "", part.strip(), flags=re.IGNORECASE)
        obj = obj.strip().rstrip(".")
        if obj:
            results.append(RawExtraction(
                subject=current_room, relation="contains",
                object=obj.lower(),
                extraction_type="direct", source_segment=None,
                extraction_method="rule_fallback",
            ))
    return results


def _parse_container_list(
    container: str, objects_text: str, ctx: dict
) -> List[RawExtraction]:
    """
    Parse TextWorld's container listing format.
    E.g., 'On the table you see a key, a lamp and a book.'
    Produces: table contains key, table contains lamp, table contains book
    Also registers container in current room.
    """
    results = []
    current_room = ctx.get("current_room", "unknown")

    # Container is in the room
    results.append(RawExtraction(
        subject=current_room, relation="contains",
        object=container,
        extraction_type="implied", source_segment=None,
        extraction_method="rule_fallback",
    ))

    # Parse the object list
    parts = re.split(r",\s*|\s+and\s+", objects_text)
    for part in parts:
        obj = re.sub(r"^(?:a|an|the)\s+", "", part.strip(), flags=re.IGNORECASE)
        obj = obj.strip().rstrip(".")
        if obj:
            results.append(RawExtraction(
                subject=container, relation="contains",
                object=obj.lower(),
                extraction_type="direct", source_segment=None,
                extraction_method="rule_fallback",
            ))
    return results


# ── Negation Rules ──────────────────────────────────────────────────────────
# These patterns fire when a sentence explicitly states ABSENCE or BLOCKING.
# They produce RawExtractions with extraction_type="negation".

_NEGATION_PATTERNS = [
    # "What a letdown! The dresser is empty!" → (dresser, contains, nothing)
    # Must be FIRST to prevent the generic 'is empty' pattern from capturing too much
    (re.compile(r"what a letdown.*?(?:the |a |an )?([\w\s]+?) is empty", re.IGNORECASE | re.DOTALL),
     lambda m, ctx: [
         RawExtraction(
             subject=m.group(1).strip().lower(), relation="contains",
             object="nothing",
             extraction_type="negation", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "The safe is empty" / "The dresser is empty!" → (safe, contains, nothing)
    # [\w\s]+ limits match to word chars/spaces only — prevents whole-sentence capture
    (re.compile(r"(?:^|(?<=\.\s)|(?<=!\s))(?:the |a |an )?([\w\s]+?) is empty[.!]?$", re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=m.group(1).strip().lower(), relation="contains",
             object="nothing",
             extraction_type="negation", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "There is nothing here" / "There's nothing here" → (room, contains, nothing)
    # Must be checked BEFORE the generic "there is X here" room pattern
    (re.compile(r"there(?:'s| is) nothing (?:here|in this room)[.!]?$", re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=ctx.get("current_room", "unknown room"), relation="contains",
             object="nothing",
             extraction_type="negation", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "You can't see any brass key" / "You cannot see a key here" → (key, has_state, absent)
    (re.compile(r"you can(?:not|'t) see (?:any |a |an )?(.+?)(?:\s+here)?[.!]?$", re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=m.group(1).strip().lower(), relation="has_state",
             object="absent",
             extraction_type="negation", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "You can't go north" / "You cannot go east" → (room, connects_to, north) blocked
    (re.compile(r"you can(?:not|'t) go (north|south|east|west|up|down)[.!]?$", re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject=ctx.get("current_room", "unknown room"), relation="connects_to",
             object=f"room_{m.group(1).strip().lower()}_blocked",
             extraction_type="negation", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
    # "That's fixed in place" / "That is fixed in place" → target immovable
    (re.compile(r"that(?:'s| is) fixed in place[.!]?$", re.IGNORECASE),
     lambda m, ctx: [
         RawExtraction(
             subject="unknown", relation="has_state",
             object="fixed",
             extraction_type="negation", source_segment=None,
             extraction_method="rule_fallback",
         ),
     ]),
]


# ── Main Fallback Extractor ─────────────────────────────────────────────────

class RuleFallback:
    """
    Stage 2b: Rule-based extraction fallback.

    Used when:
      - SLM is unavailable
      - SLM output is malformed (JSON parse failure after retries)
      - As a baseline for common TextWorld patterns

    The rules are ordered by specificity. First match wins per sentence.
    """

    def extract(
        self,
        segments: List[TextSegment],
        current_room: str = "unknown",
    ) -> List[RawExtraction]:
        """
        Apply rule-based extraction to preprocessed segments.

        Args:
            segments: Preprocessed TextSegments from Stage 1
            current_room: Current room name for context

        Returns:
            List of RawExtraction with extraction_method="rule_fallback"
        """
        context = {"current_room": current_room}
        results: List[RawExtraction] = []

        for segment in segments:
            # Skip feedback segments — no facts to extract
            if segment.segment_type == SegmentType.FEEDBACK:
                continue

            text = segment.resolved_text
            extracted = self._match_rules(text, context, segment)
            results.extend(extracted)

        logger.debug(f"RuleFallback: extracted {len(results)} facts")
        return results

    def _match_rules(
        self, text: str, context: dict, segment: TextSegment
    ) -> List[RawExtraction]:
        """Apply rules based on segment type, first match wins.
        Negation patterns are checked as a second-pass fallback on all segments.
        """
        rule_sets = {
            SegmentType.NAVIGATION: _NAV_PATTERNS,
            SegmentType.ROOM_DESCRIPTION: _ROOM_PATTERNS + _EXIT_PATTERNS,
            SegmentType.INVENTORY_UPDATE: _INVENTORY_PATTERNS,
            SegmentType.STATE_CHANGE: _STATE_PATTERNS,
            SegmentType.OBJECTIVE_INFO: [],
        }

        # Primary rules: type-specific
        patterns = rule_sets.get(segment.segment_type, _ROOM_PATTERNS)

        # Check negation FIRST for sentences that start with known negation signals.
        # This prevents "There is nothing here" being caught by the "there is X here" room pattern.
        negation_triggers = re.compile(
            r"^(?:there(?:'s| is) nothing|you can(?:not|'t)|what a letdown|.+? is empty)",
            re.IGNORECASE
        )
        if negation_triggers.match(text):
            for pattern, extractor in _NEGATION_PATTERNS:
                match = pattern.search(text)
                if match:
                    extractions = extractor(match, context)
                    for ext in extractions:
                        ext.source_segment = segment
                    return extractions

        for pattern, extractor in patterns:
            match = pattern.search(text)
            if match:
                extractions = extractor(match, context)
                for ext in extractions:
                    ext.source_segment = segment
                return extractions

        # Secondary pass: negation patterns apply to ALL segment types
        for pattern, extractor in _NEGATION_PATTERNS:
            match = pattern.search(text)
            if match:
                extractions = extractor(match, context)
                for ext in extractions:
                    ext.source_segment = segment
                return extractions

        return []
