# orchestrator/working_memory.py
# ============================================================================
# WorkingMemory builder — constructs per-turn agent context.
# Spec Reference: Implementation Plan v2.0, Section 8.3
# ============================================================================

from __future__ import annotations

import logging
from typing import List, Optional
from collections import deque

from shared.models import WorkingMemory, Edge
from shared.enums import RelationType
from world_model.graph_store import GraphStoreBase
from shared.config import OrchestratorConfig

logger = logging.getLogger(__name__)


class WorkingMemoryBuilder:
    """
    Builds the per-turn WorkingMemory from the current graph state.
    
    WorkingMemory contains:
      - Current room name
      - Active facts for the current room
      - Inventory facts (what the player holds)
      - Last N raw observations (for coreference resolution)
      - Current sub-goal and objective
    """

    def __init__(
        self,
        graph: GraphStoreBase,
        config: Optional[OrchestratorConfig] = None,
    ):
        self._graph = graph
        self._config = config or OrchestratorConfig()
        self._recent_observations: deque = deque(
            maxlen=self._config.working_memory_observation_window
        )
        self._failed_actions: deque = deque(maxlen=10)
        self._current_sub_goal: str = ""
        self._objective: str = ""

    def build(self) -> WorkingMemory:
        """Build WorkingMemory from current graph state."""
        current_room = self._get_current_room()
        room_facts = self._get_room_facts(current_room) if current_room else []
        inventory = self._get_inventory_facts()

        return WorkingMemory(
            current_room=current_room or "",
            current_room_facts=room_facts,
            inventory_facts=inventory,
            recent_observations=list(self._recent_observations),
            failed_actions=list(self._failed_actions),
            current_sub_goal=self._current_sub_goal,
            objective=self._objective,
        )

    def add_failed_action(self, action: str) -> None:
        """Record an action that failed so the agent avoids repeating it."""
        if action:
            self._failed_actions.append(action)

    def add_observation(self, text: str) -> None:
        """Add a raw observation to the recent history."""
        if text.strip():
            self._recent_observations.append(text.strip())

    def set_objective(self, objective: str) -> None:
        """Set the quest objective."""
        self._objective = objective

    def set_sub_goal(self, sub_goal: str) -> None:
        """Set the current sub-goal."""
        self._current_sub_goal = sub_goal

    def _get_current_room(self) -> Optional[str]:
        """Find the player's current room from the graph."""
        player_location = self._graph.get_active_edges_by_slot(
            "player", RelationType.LOCATED_IN
        )
        if player_location:
            return player_location[0].object
        return None

    def _get_room_facts(self, room_id: str) -> List[Edge]:
        """Get all active facts about the current room."""
        return self._graph.get_active_edges_for_entity(room_id)

    def _get_inventory_facts(self) -> List[Edge]:
        """Get all objects the player currently holds."""
        return self._graph.get_active_edges_by_slot(
            "player", RelationType.HOLDS
        )
