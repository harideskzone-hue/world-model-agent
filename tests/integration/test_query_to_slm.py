# tests/integration/test_query_to_slm.py
# ============================================================================
# Integration: QueryLayer → ActionSelector → SLMDecision
# Tests without a live SLM (heuristic fallback mode).
# ============================================================================

import pytest

from shared.models import Observation, WorkingMemory, Node, Edge, ContextSlice
from shared.enums import NodeType, RelationType, ExtractionMethod
from world_model.graph_store import InMemoryGraphStore
from query_layer.query_layer import QueryLayer
from slm.action_selector import ActionSelector


@pytest.fixture
def system():
    """Complete system minus the SLM (heuristic mode)."""
    g = InMemoryGraphStore()
    g.add_node(Node(id="kitchen", name="Kitchen", node_type=NodeType.ROOM))
    g.add_node(Node(id="player", name="player", node_type=NodeType.CHARACTER))
    g.add_node(Node(id="key", name="key", node_type=NodeType.OBJECT))
    g.add_edge(Edge(
        id="e1", subject="player", relation=RelationType.LOCATED_IN,
        object="kitchen", confidence=0.95,
        extraction_method=ExtractionMethod.SLM, t_observed=0, t_valid_from=0,
    ))
    g.add_edge(Edge(
        id="e2", subject="kitchen", relation=RelationType.CONTAINS,
        object="key", confidence=0.9,
        extraction_method=ExtractionMethod.SLM, t_observed=0, t_valid_from=0,
    ))
    ql = QueryLayer(g)
    selector = ActionSelector(slm=None)
    wm = WorkingMemory(
        current_room="kitchen",
        current_room_facts=g.get_active_edges_for_entity("kitchen"),
        inventory_facts=[],
        objective="Find the key.",
    )
    obs = Observation(
        feedback="You are in the kitchen.",
        description="You see a key on the table.",
        inventory="",
        location="kitchen",
        objective="Find the key.",
        turn_id=0,
    )
    return g, ql, selector, wm, obs


class TestQueryToSLM:
    def test_full_chain_produces_action(self, system):
        g, ql, selector, wm, obs = system
        ctx = ql.retrieve(wm, current_turn=0)
        decision = selector.select_action(ctx, wm, obs)
        assert decision.action_text == "INVALID_ACTION"
        assert isinstance(decision.action_text, str)

    def test_decision_has_latency(self, system):
        g, ql, selector, wm, obs = system
        ctx = ql.retrieve(wm, current_turn=0)
        decision = selector.select_action(ctx, wm, obs)
        assert decision.latency_ms >= 0

    def test_context_slice_not_empty(self, system):
        g, ql, selector, wm, obs = system
        ctx = ql.retrieve(wm, current_turn=0)
        assert len(ctx.formatted_text) > 0
        assert "LOCATION:" in ctx.formatted_text
