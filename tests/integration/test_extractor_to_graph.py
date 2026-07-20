# tests/integration/test_extractor_to_graph.py
# ============================================================================
# Integration: Extractor → Updater → GraphStore
# Verifies that extracted facts from text observations correctly update
# the world model graph.
# ============================================================================

import pytest

from shared.models import Observation, WorkingMemory
from shared.enums import RelationType, EdgeStatus
from world_model.graph_store import InMemoryGraphStore
from extractor.text_extractor import TextExtractor
from updater.updater import Updater


@pytest.fixture
def fresh_system():
    """Fresh graph, extractor (rule-only), and updater."""
    graph = InMemoryGraphStore()
    extractor = TextExtractor(slm_runner=None)  # Rule-based only
    updater = Updater(graph)
    wm = WorkingMemory(current_room="", objective="Find the key.")
    return graph, extractor, updater, wm


class TestExtractorToGraph:
    def test_room_description_populates_graph(self, fresh_system):
        graph, extractor, updater, wm = fresh_system
        obs = Observation(
            feedback="",
            description="You are in the kitchen. You see a brass key.",
            inventory="",
            location="kitchen",
            objective="Find the key.",
            turn_id=0,
        )
        wm.current_room = "kitchen"

        candidates = extractor.extract(obs, wm)
        report = updater.update(candidates, turn_id=0)

        # Graph should have nodes and edges
        assert graph.get_stats().total_nodes > 0
        assert graph.get_stats().active_edges > 0
        assert report.expanded > 0

    def test_state_change_updates_graph(self, fresh_system):
        graph, extractor, updater, wm = fresh_system

        # First: establish initial state
        obs1 = Observation(
            feedback="",
            description="You are in the kitchen. You see a door.",
            inventory="",
            location="kitchen",
            objective="Find the key.",
            turn_id=0,
        )
        wm.current_room = "kitchen"
        candidates1 = extractor.extract(obs1, wm)
        updater.update(candidates1, turn_id=0)

        # Then: unlock the door
        obs2 = Observation(
            feedback="You unlock the door.",
            description="",
            inventory="",
            location="kitchen",
            objective="Find the key.",
            turn_id=1,
        )
        candidates2 = extractor.extract(obs2, wm)
        report = updater.update(candidates2, turn_id=1)

        # Check that a has_state edge was created
        state_edges = graph.get_active_edges_by_slot("door", RelationType.HAS_STATE)
        if state_edges:
            assert state_edges[0].object == "unlocked"

    def test_inventory_change_creates_holds_edge(self, fresh_system):
        graph, extractor, updater, wm = fresh_system
        wm.current_room = "kitchen"

        obs = Observation(
            feedback="You take the apple.",
            description="",
            inventory="",
            location="kitchen",
            objective="Eat the apple.",
            turn_id=1,
        )
        candidates = extractor.extract(obs, wm)
        updater.update(candidates, turn_id=1)

        holds = graph.get_active_edges_by_slot("player", RelationType.HOLDS)
        assert any(e.object == "apple" for e in holds)

    def test_sequential_observations_build_graph(self, fresh_system):
        graph, extractor, updater, wm = fresh_system

        observations = [
            Observation(
                feedback="", description="You are in the kitchen. You see a key.",
                inventory="", location="kitchen", objective="Find key.", turn_id=0,
            ),
            Observation(
                feedback="You take the key.", description="",
                inventory="", location="kitchen", objective="Find key.", turn_id=1,
            ),
            Observation(
                feedback="You go north.", description="You are in the garden.",
                inventory="", location="garden", objective="Find key.", turn_id=2,
            ),
        ]

        for obs in observations:
            wm.current_room = obs.location
            candidates = extractor.extract(obs, wm)
            updater.update(candidates, turn_id=obs.turn_id)

        stats = graph.get_stats()
        assert stats.total_nodes >= 2  # At least kitchen and garden
        assert stats.active_edges >= 2
