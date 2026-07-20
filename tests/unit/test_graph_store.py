# tests/unit/test_graph_store.py
# ============================================================================
# Unit tests for the World Model graph store.
# Tests: CRUD, bi-temporal versioning, supersession, time-travel, BFS,
#        serialization round-trip, and performance constraints.
# ============================================================================

import json
import pytest
from world_model.graph_store import InMemoryGraphStore
from world_model.schema import normalize_entity_name, are_states_conflicting
from shared.models import Node, Edge
from shared.enums import (
    NodeType, NodeStatus, EdgeStatus, RelationType, ExtractionMethod,
)


@pytest.fixture
def graph():
    """Fresh graph store for each test."""
    return InMemoryGraphStore()


@pytest.fixture
def populated_graph(graph):
    """
    Graph with a small world: 2 rooms, 2 objects, 1 character.
    Kitchen → brass key, apple. Kitchen connects to Garden (north).
    """
    # Nodes
    graph.add_node(Node(
        id="kitchen", name="Kitchen", node_type=NodeType.ROOM,
        confidence=0.95, first_observed_turn=0, last_observed_turn=0,
    ))
    graph.add_node(Node(
        id="garden", name="Garden", node_type=NodeType.ROOM,
        confidence=0.90, first_observed_turn=1, last_observed_turn=1,
    ))
    graph.add_node(Node(
        id="brass key", name="brass key", node_type=NodeType.OBJECT,
        confidence=0.90, first_observed_turn=0, last_observed_turn=0,
    ))
    graph.add_node(Node(
        id="apple", name="apple", node_type=NodeType.OBJECT,
        confidence=0.85, first_observed_turn=0, last_observed_turn=0,
    ))
    graph.add_node(Node(
        id="player", name="player", node_type=NodeType.CHARACTER,
        confidence=1.0, first_observed_turn=0, last_observed_turn=0,
    ))

    # Edges
    graph.add_edge(Edge(
        id="e1", subject="kitchen", relation=RelationType.CONTAINS,
        object="brass key", confidence=0.90, source_turn_id=0,
        extraction_method=ExtractionMethod.SLM,
        t_observed=0, t_valid_from=0,
    ))
    graph.add_edge(Edge(
        id="e2", subject="kitchen", relation=RelationType.CONTAINS,
        object="apple", confidence=0.85, source_turn_id=0,
        extraction_method=ExtractionMethod.SLM,
        t_observed=0, t_valid_from=0,
    ))
    graph.add_edge(Edge(
        id="e3", subject="kitchen", relation=RelationType.CONNECTS_TO,
        object="garden", confidence=0.90, source_turn_id=1,
        extraction_method=ExtractionMethod.SLM,
        t_observed=1, t_valid_from=1, direction="north",
    ))
    graph.add_edge(Edge(
        id="e4", subject="brass key", relation=RelationType.HAS_STATE,
        object="locked", confidence=0.90, source_turn_id=0,
        extraction_method=ExtractionMethod.SLM,
        t_observed=0, t_valid_from=0,
    ))
    graph.add_edge(Edge(
        id="e5", subject="player", relation=RelationType.LOCATED_IN,
        object="kitchen", confidence=0.95, source_turn_id=0,
        extraction_method=ExtractionMethod.SLM,
        t_observed=0, t_valid_from=0,
    ))

    return graph


# ═══════════════════════════════════════════════════════════════════════════
# NODE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeOperations:
    def test_add_node(self, graph):
        node_id = graph.add_node(Node(
            id="kitchen", name="Kitchen", node_type=NodeType.ROOM,
        ))
        assert node_id == "kitchen"
        assert graph.get_node("kitchen") is not None
        assert graph.get_node("kitchen").name == "Kitchen"

    def test_add_node_normalizes_id(self, graph):
        node_id = graph.add_node(Node(
            id="The Kitchen", name="The Kitchen", node_type=NodeType.ROOM,
        ))
        assert node_id == "kitchen"  # Normalized: lowercase, no article

    def test_add_node_idempotent(self, graph):
        graph.add_node(Node(
            id="kitchen", name="Kitchen", node_type=NodeType.ROOM,
            confidence=0.5, first_observed_turn=0, last_observed_turn=0,
        ))
        graph.add_node(Node(
            id="kitchen", name="Kitchen", node_type=NodeType.ROOM,
            confidence=0.5, first_observed_turn=5, last_observed_turn=5,
        ))
        # Should update, not duplicate
        assert len(graph.get_all_nodes()) == 1
        node = graph.get_node("kitchen")
        assert node.last_observed_turn == 5
        assert node.corroboration_count == 1  # Incremented from 0

    def test_get_nonexistent_node(self, graph):
        assert graph.get_node("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════════
# EDGE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeOperations:
    def test_add_edge(self, graph):
        edge = Edge(
            id="e1", subject="kitchen", relation=RelationType.CONTAINS,
            object="key", confidence=0.9, source_turn_id=0,
            extraction_method=ExtractionMethod.SLM,
            t_observed=0, t_valid_from=0,
        )
        edge_id = graph.add_edge(edge)
        assert edge_id == "e1"

    def test_get_active_edges_for_entity(self, populated_graph):
        edges = populated_graph.get_active_edges_for_entity("kitchen")
        # Kitchen is subject for e1, e2, e3; and object for e5
        assert len(edges) == 4

    def test_get_active_edges_by_slot(self, populated_graph):
        edges = populated_graph.get_active_edges_by_slot("kitchen", RelationType.CONTAINS)
        assert len(edges) == 2
        objects = {e.object for e in edges}
        assert objects == {"brass key", "apple"}

    def test_empty_slot_returns_empty(self, populated_graph):
        edges = populated_graph.get_active_edges_by_slot("garden", RelationType.CONTAINS)
        assert len(edges) == 0


# ═══════════════════════════════════════════════════════════════════════════
# SUPERSESSION (BI-TEMPORAL VERSIONING)
# ═══════════════════════════════════════════════════════════════════════════

class TestSupersession:
    def test_supersede_edge(self, populated_graph):
        """Superseding 'brass key has_state locked' with 'unlocked'."""
        new_edge = Edge(
            id="e4_v2", subject="brass key", relation=RelationType.HAS_STATE,
            object="unlocked", confidence=0.90, source_turn_id=5,
            extraction_method=ExtractionMethod.SLM,
            t_observed=5, t_valid_from=5,
        )
        populated_graph.supersede_edge("e4", new_edge, "newer direct observation")

        # Old edge should be superseded
        old_edge = populated_graph._edges["e4"]
        assert old_edge.status == EdgeStatus.SUPERSEDED
        assert old_edge.t_valid_until == 5
        assert old_edge.superseded_by == "e4_v2"

        # New edge should be active
        new = populated_graph._edges["e4_v2"]
        assert new.status == EdgeStatus.ACTIVE
        assert new.object == "unlocked"

    def test_superseded_edge_excluded_from_active_query(self, populated_graph):
        new_edge = Edge(
            id="e4_v2", subject="brass key", relation=RelationType.HAS_STATE,
            object="unlocked", confidence=0.90, source_turn_id=5,
            extraction_method=ExtractionMethod.SLM,
            t_observed=5, t_valid_from=5,
        )
        populated_graph.supersede_edge("e4", new_edge, "newer observation")

        active = populated_graph.get_active_edges_by_slot(
            "brass key", RelationType.HAS_STATE
        )
        assert len(active) == 1
        assert active[0].object == "unlocked"

    def test_supersede_nonexistent_raises(self, populated_graph):
        new_edge = Edge(id="x", subject="y", relation=RelationType.CONTAINS, object="z")
        with pytest.raises(KeyError):
            populated_graph.supersede_edge("nonexistent", new_edge, "test")


# ═══════════════════════════════════════════════════════════════════════════
# CORROBORATION
# ═══════════════════════════════════════════════════════════════════════════

class TestCorroboration:
    def test_corroborate_increases_count(self, populated_graph):
        original_count = populated_graph._edges["e1"].corroboration_count
        populated_graph.corroborate_edge("e1", turn_id=5)
        assert populated_graph._edges["e1"].corroboration_count == original_count + 1

    def test_corroborate_boosts_confidence(self, populated_graph):
        original_conf = populated_graph._edges["e1"].confidence
        populated_graph.corroborate_edge("e1", turn_id=5)
        assert populated_graph._edges["e1"].confidence > original_conf

    def test_confidence_capped_at_099(self, populated_graph):
        for i in range(100):
            populated_graph.corroborate_edge("e1", turn_id=i)
        assert populated_graph._edges["e1"].confidence <= 0.99


# ═══════════════════════════════════════════════════════════════════════════
# TIME-TRAVEL QUERIES
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeTravel:
    def test_edges_at_turn_initial(self, populated_graph):
        """At turn 0, brass key should be locked."""
        edges = populated_graph.get_edges_at_turn("brass key", turn_id=0)
        state_edges = [e for e in edges if e.relation == RelationType.HAS_STATE]
        assert len(state_edges) == 1
        assert state_edges[0].object == "locked"

    def test_edges_at_turn_after_supersession(self, populated_graph):
        """After superseding locked→unlocked at turn 5, time-travel should still see locked at turn 3."""
        new_edge = Edge(
            id="e4_v2", subject="brass key", relation=RelationType.HAS_STATE,
            object="unlocked", confidence=0.90, source_turn_id=5,
            extraction_method=ExtractionMethod.SLM,
            t_observed=5, t_valid_from=5,
        )
        populated_graph.supersede_edge("e4", new_edge, "test")

        # At turn 3, should see "locked"
        edges_t3 = populated_graph.get_edges_at_turn("brass key", turn_id=3)
        state_t3 = [e for e in edges_t3 if e.relation == RelationType.HAS_STATE]
        assert len(state_t3) == 1
        assert state_t3[0].object == "locked"

        # At turn 5+, should see "unlocked"
        edges_t5 = populated_graph.get_edges_at_turn("brass key", turn_id=5)
        state_t5 = [e for e in edges_t5 if e.relation == RelationType.HAS_STATE]
        assert len(state_t5) == 1
        assert state_t5[0].object == "unlocked"


# ═══════════════════════════════════════════════════════════════════════════
# BFS SUBGRAPH
# ═══════════════════════════════════════════════════════════════════════════

class TestSubgraph:
    def test_room_subgraph_depth_0(self, populated_graph):
        nodes, edges = populated_graph.get_room_subgraph("kitchen", depth=0)
        node_ids = {n.id for n in nodes}
        assert "kitchen" in node_ids

    def test_room_subgraph_depth_1(self, populated_graph):
        nodes, edges = populated_graph.get_room_subgraph("kitchen", depth=1)
        node_ids = {n.id for n in nodes}
        # Should include kitchen + directly connected entities
        assert "kitchen" in node_ids
        assert "brass key" in node_ids or "apple" in node_ids

    def test_edges_are_deduplicated(self, populated_graph):
        _, edges = populated_graph.get_room_subgraph("kitchen", depth=2)
        edge_ids = [e.id for e in edges]
        assert len(edge_ids) == len(set(edge_ids))


# ═══════════════════════════════════════════════════════════════════════════
# SERIALIZATION ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_serialize_produces_valid_json(self, populated_graph):
        data = populated_graph.serialize()
        parsed = json.loads(data)
        assert "nodes" in parsed
        assert "edges" in parsed
        assert len(parsed["nodes"]) == 5
        assert len(parsed["edges"]) == 5

    def test_round_trip(self, populated_graph):
        """Serialize → new store → deserialize → same state."""
        data = populated_graph.serialize()
        new_graph = InMemoryGraphStore()
        new_graph.deserialize(data)

        # Verify nodes
        for nid in ["kitchen", "garden", "brass key", "apple", "player"]:
            original = populated_graph.get_node(nid)
            restored = new_graph.get_node(nid)
            assert restored is not None, f"Node {nid} missing after round-trip"
            assert original.name == restored.name
            assert original.node_type == restored.node_type
            assert original.confidence == restored.confidence

        # Verify edges
        orig_active = populated_graph.get_all_active_edges()
        rest_active = new_graph.get_all_active_edges()
        assert len(orig_active) == len(rest_active)

    def test_round_trip_preserves_supersession(self, populated_graph):
        """Supersede an edge, then round-trip. Both old and new should be preserved."""
        new_edge = Edge(
            id="e4_v2", subject="brass key", relation=RelationType.HAS_STATE,
            object="unlocked", confidence=0.90, source_turn_id=5,
            extraction_method=ExtractionMethod.SLM,
            t_observed=5, t_valid_from=5,
        )
        populated_graph.supersede_edge("e4", new_edge, "test reason")

        data = populated_graph.serialize()
        new_graph = InMemoryGraphStore()
        new_graph.deserialize(data)

        # Old edge should be superseded
        old = new_graph._edges["e4"]
        assert old.status == EdgeStatus.SUPERSEDED
        assert old.revision_reason == "test reason"

        # New edge should be active
        new = new_graph._edges["e4_v2"]
        assert new.status == EdgeStatus.ACTIVE
        assert new.object == "unlocked"


# ═══════════════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════════════

class TestStats:
    def test_stats_counts(self, populated_graph):
        stats = populated_graph.get_stats()
        assert stats.total_nodes == 5
        assert stats.total_edges == 5
        assert stats.active_edges == 5
        assert stats.superseded_edges == 0
        assert stats.avg_confidence > 0

    def test_stats_after_supersession(self, populated_graph):
        new_edge = Edge(
            id="e4_v2", subject="brass key", relation=RelationType.HAS_STATE,
            object="unlocked", confidence=0.90, source_turn_id=5,
            extraction_method=ExtractionMethod.SLM,
            t_observed=5, t_valid_from=5,
        )
        populated_graph.supersede_edge("e4", new_edge, "test")
        stats = populated_graph.get_stats()
        assert stats.total_edges == 6  # 5 original + 1 new
        assert stats.active_edges == 5  # e4 superseded, e4_v2 active
        assert stats.superseded_edges == 1


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaUtils:
    def test_normalize_entity_name(self):
        assert normalize_entity_name("The Brass Key") == "brass key"
        assert normalize_entity_name("A golden apple") == "golden apple"
        assert normalize_entity_name("kitchen") == "kitchen"
        assert normalize_entity_name("  An Old Map  ") == "old map"

    def test_conflicting_states(self):
        assert are_states_conflicting("open", "closed") is True
        assert are_states_conflicting("locked", "unlocked") is True
        assert are_states_conflicting("open", "locked") is False
        assert are_states_conflicting("raw", "cooked") is True
        assert are_states_conflicting("open", "open") is False

    def test_non_conflicting_states(self):
        assert are_states_conflicting("hot", "cold") is False
        assert are_states_conflicting("open", "hot") is False
