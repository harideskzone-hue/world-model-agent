# query_layer/anchor_resolver.py
# ============================================================================
# Step 1: Determine traversal starting points for context retrieval.
# Spec Reference: Implementation Plan v2.0, Section 7.1, Step 1
# ============================================================================

from __future__ import annotations

import logging
from typing import List

from shared.models import AnchorNode, WorkingMemory, Edge
from shared.enums import RelationType
from world_model.graph_store import GraphStoreBase

logger = logging.getLogger(__name__)


class AnchorResolver:
    """
    Step 1 of the Query Layer pipeline.
    Determines starting points for graph traversal.
    """

    def __init__(self, graph: GraphStoreBase):
        self._graph = graph

    def resolve(
        self, working_memory: WorkingMemory, sub_goal: str
    ) -> List[AnchorNode]:
        """
        Determine traversal anchor nodes.

        Priority order:
          1. PRIMARY: Agent's current room (highest priority)
          2. GOAL: Entities mentioned in the current sub-goal
          3. INVENTORY: Objects the agent currently holds
        """
        anchors: List[AnchorNode] = []
        seen: set = set()

        # 1. Primary anchor: current room
        if working_memory.current_room:
            room_id = working_memory.current_room
            if self._graph.get_node(room_id) is not None:
                anchors.append(AnchorNode(
                    id=room_id, anchor_type="location", priority=1.0,
                ))
                seen.add(room_id)

        # 2. Goal anchors: parse sub-goal for entity mentions
        goal_entities = self._extract_goal_entities(sub_goal)
        for entity_id in goal_entities:
            if entity_id not in seen and self._graph.get_node(entity_id) is not None:
                anchors.append(AnchorNode(
                    id=entity_id, anchor_type="goal", priority=0.8,
                ))
                seen.add(entity_id)

        # 3. Inventory anchors
        for edge in working_memory.inventory_facts:
            obj_id = edge.object
            if obj_id not in seen:
                anchors.append(AnchorNode(
                    id=obj_id, anchor_type="inventory", priority=0.5,
                ))
                seen.add(obj_id)

        logger.debug(f"AnchorResolver: {len(anchors)} anchors resolved")
        return anchors

    def _extract_goal_entities(self, sub_goal: str) -> List[str]:
        """
        Extract entity names from a sub-goal string using simple word matching
        against the world model's known nodes.
        """
        if not sub_goal:
            return []

        entities = []
        goal_lower = sub_goal.lower()

        for node in self._graph.get_all_nodes():
            if node.id.lower() in goal_lower:
                entities.append(node.id)

        return entities
