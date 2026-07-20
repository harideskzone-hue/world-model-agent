# query_layer/context_formatter.py
# ============================================================================
# Step 5: Convert selected edges into structured text for the SLM prompt.
# Spec Reference: Implementation Plan v2.0, Section 7.1, Step 5
# ============================================================================

from __future__ import annotations

import logging
from typing import List, Dict
from collections import defaultdict

from shared.models import Edge
from shared.enums import RelationType

logger = logging.getLogger(__name__)


class ContextFormatter:
    """
    Step 5 of the Query Layer pipeline.
    Converts selected edges into a structured, human-readable text block
    suitable for insertion into the SLM prompt.

    Output format:
      LOCATION: Kitchen
      ROOM CONTENTS: brass key (locked), apple
      CONNECTED ROOMS: north → Garden, south → Hallway
      INVENTORY: silver coin, old map
      RECENT CHANGES: [Turn 5] door: locked → unlocked
      OBJECTIVE: Find and take the brass key
    """

    def format(
        self,
        edges: List[Edge],
        current_room: str,
        objective: str,
        recent_changes: List[str] | None = None,
    ) -> str:
        """
        Format edges into structured text.

        Args:
            edges: Selected edges from budget truncation
            current_room: Current room name
            objective: Quest objective text
            recent_changes: Optional list of recent change descriptions

        Returns:
            Formatted text string ready for SLM prompt
        """
        lines: List[str] = []

        # Group edges by type
        contains: Dict[str, List[Edge]] = defaultdict(list)
        connects: List[Edge] = []
        inventory: List[Edge] = []
        states: Dict[str, str] = {}  # entity → state
        location_edges: List[Edge] = []
        other: List[Edge] = []

        for edge in edges:
            if edge.relation == RelationType.CONTAINS:
                contains[edge.subject].append(edge)
            elif edge.relation == RelationType.CONNECTS_TO:
                connects.append(edge)
            elif edge.relation == RelationType.HOLDS:
                inventory.append(edge)
            elif edge.relation == RelationType.HAS_STATE:
                states[edge.subject] = edge.object
            elif edge.relation == RelationType.LOCATED_IN:
                location_edges.append(edge)
            else:
                other.append(edge)

        # LOCATION
        lines.append(f"LOCATION: {current_room}")

        # ROOM CONTENTS
        room_items = contains.get(current_room, [])
        if room_items:
            items = []
            for e in room_items:
                state = states.get(e.object)
                if state:
                    items.append(f"{e.object} ({state})")
                else:
                    items.append(e.object)
            lines.append(f"ROOM CONTENTS: {', '.join(items)}")

        # CONNECTED ROOMS
        room_connections = [e for e in connects if e.subject == current_room]
        if room_connections:
            dirs = []
            for e in room_connections:
                if e.direction:
                    dirs.append(f"{e.direction} → {e.object}")
                else:
                    dirs.append(e.object)
            lines.append(f"CONNECTED ROOMS: {', '.join(dirs)}")

        # INVENTORY
        if inventory:
            inv_items = [e.object for e in inventory]
            lines.append(f"INVENTORY: {', '.join(inv_items)}")

        # OTHER ROOM CONTENTS (not current room)
        for room, items in contains.items():
            if room != current_room:
                item_names = [e.object for e in items]
                lines.append(f"NEARBY ({room}): {', '.join(item_names)}")

        # RECENT CHANGES
        if recent_changes:
            lines.append(f"RECENT CHANGES: {'; '.join(recent_changes[-3:])}")

        # OBJECTIVE
        if objective:
            lines.append(f"OBJECTIVE: {objective}")

        formatted = "\n".join(lines)
        logger.debug(f"ContextFormatter: {len(lines)} lines, {len(formatted)} chars")
        return formatted
