# tests/integration/test_graph_to_query.py
# ============================================================================
# Integration: GraphStore → QueryLayer → ContextSlice
# Verifies that the query layer correctly retrieves from a populated graph.
# ============================================================================

import pytest

from shared.models import Node, Edge, WorkingMemory
from shared.enums import NodeType, RelationType, ExtractionMethod
from shared.config import QueryConfig
from world_model.graph_store import InMemoryGraphStore
from query_layer.query_layer import QueryLayer


@pytest.fixture
def rich_graph():
    """Graph with multiple rooms, objects, and connections."""
    g = InMemoryGraphStore()
    rooms = ["kitchen", "garden", "cellar", "hallway"]
    for r in rooms:
        g.add_node(Node(id=r, name=r.title(), node_type=NodeType.ROOM))

    g.add_node(Node(id="player", name="player", node_type=NodeType.CHARACTER))
    g.add_node(Node(id="key", name="key", node_type=NodeType.OBJECT))
    g.add_node(Node(id="apple", name="apple", node_type=NodeType.OBJECT))
    g.add_node(Node(id="sword", name="sword", node_type=NodeType.OBJECT))

    g.add_edge(Edge(
        id="e1", subject="player", relation=RelationType.LOCATED_IN,
        object="kitchen", confidence=0.95,
        extraction_method=ExtractionMethod.SLM, t_observed=5, t_valid_from=5,
    ))
    g.add_edge(Edge(
        id="e2", subject="kitchen", relation=RelationType.CONTAINS,
        object="key", confidence=0.9,
        extraction_method=ExtractionMethod.SLM, t_observed=5, t_valid_from=5,
    ))
    g.add_edge(Edge(
        id="e3", subject="kitchen", relation=RelationType.CONTAINS,
        object="apple", confidence=0.85,
        extraction_method=ExtractionMethod.SLM, t_observed=3, t_valid_from=3,
    ))
    g.add_edge(Edge(
        id="e4", subject="kitchen", relation=RelationType.CONNECTS_TO,
        object="garden", confidence=0.9, direction="north",
        extraction_method=ExtractionMethod.SLM, t_observed=1, t_valid_from=1,
    ))
    g.add_edge(Edge(
        id="e5", subject="kitchen", relation=RelationType.CONNECTS_TO,
        object="hallway", confidence=0.9, direction="south",
        extraction_method=ExtractionMethod.SLM, t_observed=1, t_valid_from=1,
    ))
    g.add_edge(Edge(
        id="e6", subject="player", relation=RelationType.HOLDS,
        object="sword", confidence=0.9,
        extraction_method=ExtractionMethod.SLM, t_observed=4, t_valid_from=4,
    ))
    g.add_edge(Edge(
        id="e7", subject="key", relation=RelationType.HAS_STATE,
        object="locked", confidence=0.9,
        extraction_method=ExtractionMethod.SLM, t_observed=5, t_valid_from=5,
    ))
    return g


class TestGraphToQuery:
    def test_retrieval_includes_current_room(self, rich_graph):
        wm = WorkingMemory(
            current_room="kitchen",
            current_room_facts=rich_graph.get_active_edges_for_entity("kitchen"),
            inventory_facts=rich_graph.get_active_edges_by_slot("player", RelationType.HOLDS),
            objective="Find the key.",
        )
        ql = QueryLayer(rich_graph)
        ctx = ql.retrieve(wm, current_turn=5)
        assert "LOCATION: kitchen" in ctx.formatted_text

    def test_retrieval_includes_room_contents(self, rich_graph):
        wm = WorkingMemory(
            current_room="kitchen",
            current_room_facts=rich_graph.get_active_edges_for_entity("kitchen"),
            inventory_facts=[],
            objective="Find the key.",
        )
        ql = QueryLayer(rich_graph)
        ctx = ql.retrieve(wm, current_turn=5)
        assert "key" in ctx.formatted_text.lower()

    def test_retrieval_includes_connections(self, rich_graph):
        wm = WorkingMemory(
            current_room="kitchen",
            current_room_facts=rich_graph.get_active_edges_for_entity("kitchen"),
            inventory_facts=[],
            objective="Explore.",
        )
        ql = QueryLayer(rich_graph)
        ctx = ql.retrieve(wm, current_turn=5)
        assert "garden" in ctx.formatted_text.lower() or "CONNECTED" in ctx.formatted_text

    def test_budget_respected(self, rich_graph):
        config = QueryConfig(token_budget=30)  # Very tight budget
        wm = WorkingMemory(
            current_room="kitchen",
            current_room_facts=rich_graph.get_active_edges_for_entity("kitchen"),
            inventory_facts=[],
            objective="Find key.",
        )
        ql = QueryLayer(rich_graph, config=config)
        ctx = ql.retrieve(wm, current_turn=5)
        assert ctx.total_tokens_estimate <= 30

    def test_metadata_has_counts(self, rich_graph):
        wm = WorkingMemory(
            current_room="kitchen",
            current_room_facts=rich_graph.get_active_edges_for_entity("kitchen"),
            inventory_facts=[],
            objective="Find key.",
        )
        ql = QueryLayer(rich_graph)
        ctx = ql.retrieve(wm, current_turn=5)
        assert ctx.retrieval_metadata["anchor_count"] >= 1
        assert ctx.retrieval_metadata["selected_count"] >= 1
