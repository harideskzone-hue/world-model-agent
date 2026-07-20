# tests/unit/test_query_layer.py
# ============================================================================
# Unit tests for the Query Layer (5-step retrieval pipeline).
# ============================================================================

import pytest

from world_model.graph_store import InMemoryGraphStore
from shared.models import Node, Edge, WorkingMemory, AnchorNode
from shared.enums import (
    NodeType, RelationType, ExtractionMethod, EdgeStatus,
)
from shared.config import QueryConfig
from query_layer.anchor_resolver import AnchorResolver
from query_layer.graph_traverser import GraphTraverser
from query_layer.relevance_scorer import RelevanceScorer
from query_layer.budget_manager import BudgetManager
from query_layer.context_formatter import ContextFormatter
from query_layer.query_layer import QueryLayer


@pytest.fixture
def populated_graph():
    """Graph with kitchen, garden, 3 objects, player."""
    g = InMemoryGraphStore()
    g.add_node(Node(id="kitchen", name="Kitchen", node_type=NodeType.ROOM))
    g.add_node(Node(id="garden", name="Garden", node_type=NodeType.ROOM))
    g.add_node(Node(id="brass key", name="brass key", node_type=NodeType.OBJECT))
    g.add_node(Node(id="apple", name="apple", node_type=NodeType.OBJECT))
    g.add_node(Node(id="sword", name="sword", node_type=NodeType.OBJECT))
    g.add_node(Node(id="player", name="player", node_type=NodeType.CHARACTER))

    g.add_edge(Edge(
        id="e1", subject="kitchen", relation=RelationType.CONTAINS,
        object="brass key", confidence=0.9,
        extraction_method=ExtractionMethod.SLM, t_observed=0, t_valid_from=0,
    ))
    g.add_edge(Edge(
        id="e2", subject="kitchen", relation=RelationType.CONTAINS,
        object="apple", confidence=0.85,
        extraction_method=ExtractionMethod.SLM, t_observed=0, t_valid_from=0,
    ))
    g.add_edge(Edge(
        id="e3", subject="kitchen", relation=RelationType.CONNECTS_TO,
        object="garden", confidence=0.9, direction="north",
        extraction_method=ExtractionMethod.SLM, t_observed=1, t_valid_from=1,
    ))
    g.add_edge(Edge(
        id="e4", subject="brass key", relation=RelationType.HAS_STATE,
        object="locked", confidence=0.9,
        extraction_method=ExtractionMethod.SLM, t_observed=0, t_valid_from=0,
    ))
    g.add_edge(Edge(
        id="e5", subject="player", relation=RelationType.LOCATED_IN,
        object="kitchen", confidence=0.95,
        extraction_method=ExtractionMethod.SLM, t_observed=0, t_valid_from=0,
    ))
    g.add_edge(Edge(
        id="e6", subject="player", relation=RelationType.HOLDS,
        object="sword", confidence=0.9,
        extraction_method=ExtractionMethod.SLM, t_observed=2, t_valid_from=2,
    ))
    g.add_edge(Edge(
        id="e7", subject="garden", relation=RelationType.CONTAINS,
        object="rose", confidence=0.7,
        extraction_method=ExtractionMethod.RULE_FALLBACK, t_observed=1, t_valid_from=1,
    ))
    g.add_node(Node(id="rose", name="rose", node_type=NodeType.OBJECT))
    return g


@pytest.fixture
def working_memory(populated_graph):
    return WorkingMemory(
        current_room="kitchen",
        current_room_facts=populated_graph.get_active_edges_for_entity("kitchen"),
        inventory_facts=populated_graph.get_active_edges_by_slot("player", RelationType.HOLDS),
        recent_observations=["You are in the kitchen.", "You see a brass key."],
        current_sub_goal="Find the brass key",
        objective="Find the key and escape.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: ANCHOR RESOLVER
# ═══════════════════════════════════════════════════════════════════════════

class TestAnchorResolver:
    def test_resolves_current_room(self, populated_graph, working_memory):
        resolver = AnchorResolver(populated_graph)
        anchors = resolver.resolve(working_memory, "Find the brass key")
        location_anchors = [a for a in anchors if a.anchor_type == "location"]
        assert len(location_anchors) == 1
        assert location_anchors[0].id == "kitchen"

    def test_resolves_goal_entities(self, populated_graph, working_memory):
        resolver = AnchorResolver(populated_graph)
        anchors = resolver.resolve(working_memory, "Find the brass key")
        goal_anchors = [a for a in anchors if a.anchor_type == "goal"]
        assert any(a.id == "brass key" for a in goal_anchors)

    def test_resolves_inventory(self, populated_graph, working_memory):
        resolver = AnchorResolver(populated_graph)
        anchors = resolver.resolve(working_memory, "")
        inv_anchors = [a for a in anchors if a.anchor_type == "inventory"]
        assert any(a.id == "sword" for a in inv_anchors)

    def test_priority_ordering(self, populated_graph, working_memory):
        resolver = AnchorResolver(populated_graph)
        anchors = resolver.resolve(working_memory, "Find the brass key")
        # Location should have highest priority
        if len(anchors) >= 2:
            location = [a for a in anchors if a.anchor_type == "location"]
            others = [a for a in anchors if a.anchor_type != "location"]
            if location and others:
                assert location[0].priority >= max(o.priority for o in others)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: GRAPH TRAVERSER
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphTraverser:
    def test_traverses_from_anchors(self, populated_graph):
        traverser = GraphTraverser(populated_graph, max_depth=1)
        anchors = [AnchorNode(id="kitchen", anchor_type="location", priority=1.0)]
        edges = traverser.traverse(anchors)
        assert len(edges) > 0

    def test_edges_are_deduplicated(self, populated_graph):
        traverser = GraphTraverser(populated_graph, max_depth=2)
        anchors = [
            AnchorNode(id="kitchen", anchor_type="location", priority=1.0),
            AnchorNode(id="garden", anchor_type="goal", priority=0.8),
        ]
        edges = traverser.traverse(anchors)
        ids = [e.id for e in edges]
        assert len(ids) == len(set(ids))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: RELEVANCE SCORER
# ═══════════════════════════════════════════════════════════════════════════

class TestRelevanceScorer:
    def test_scores_are_between_0_and_1(self, populated_graph):
        config = QueryConfig()
        scorer = RelevanceScorer(populated_graph, config)
        edges = populated_graph.get_all_active_edges()
        anchors = [AnchorNode(id="kitchen", anchor_type="location", priority=1.0)]
        scored = scorer.score(edges, anchors, current_turn=5, sub_goal="Find key")
        for edge, score in scored:
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_anchor_adjacent_edges_score_higher(self, populated_graph):
        config = QueryConfig()
        scorer = RelevanceScorer(populated_graph, config)
        edges = populated_graph.get_all_active_edges()
        anchors = [AnchorNode(id="kitchen", anchor_type="location", priority=1.0)]
        scored = scorer.score(edges, anchors, current_turn=5, sub_goal="")
        # Edges involving "kitchen" should score higher
        kitchen_scores = [s for e, s in scored if e.subject == "kitchen" or e.object == "kitchen"]
        other_scores = [s for e, s in scored if e.subject != "kitchen" and e.object != "kitchen"]
        if kitchen_scores and other_scores:
            assert max(kitchen_scores) >= max(other_scores)

    def test_sorted_descending(self, populated_graph):
        config = QueryConfig()
        scorer = RelevanceScorer(populated_graph, config)
        edges = populated_graph.get_all_active_edges()
        anchors = [AnchorNode(id="kitchen", anchor_type="location", priority=1.0)]
        scored = scorer.score(edges, anchors, current_turn=5, sub_goal="Find key")
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: BUDGET MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class TestBudgetManager:
    def test_respects_token_budget(self, populated_graph):
        config = QueryConfig(token_budget=50)  # Very tight budget
        bm = BudgetManager(config)
        edges = populated_graph.get_all_active_edges()
        scored = [(e, 1.0 - i * 0.1) for i, e in enumerate(edges)]
        selected, tokens, excluded = bm.truncate(scored)
        assert tokens <= 50
        assert excluded >= 0

    def test_selects_highest_scored_first(self, populated_graph):
        config = QueryConfig(token_budget=30)  # Very tight
        bm = BudgetManager(config)
        edges = populated_graph.get_all_active_edges()
        scored = [(e, 1.0 - i * 0.1) for i, e in enumerate(edges)]
        selected, _, _ = bm.truncate(scored)
        if len(selected) >= 2:
            # First selected should be highest scored
            assert selected[0] == scored[0][0]


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: CONTEXT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════

class TestContextFormatter:
    def test_includes_location(self, populated_graph):
        formatter = ContextFormatter()
        edges = populated_graph.get_all_active_edges()
        text = formatter.format(edges, "kitchen", "Find the key.")
        assert "LOCATION: kitchen" in text

    def test_includes_objective(self, populated_graph):
        formatter = ContextFormatter()
        edges = populated_graph.get_all_active_edges()
        text = formatter.format(edges, "kitchen", "Find the key.")
        assert "OBJECTIVE: Find the key." in text

    def test_includes_room_contents(self, populated_graph):
        formatter = ContextFormatter()
        edges = populated_graph.get_all_active_edges()
        text = formatter.format(edges, "kitchen", "Find the key.")
        assert "ROOM CONTENTS:" in text

    def test_includes_inventory(self, populated_graph):
        formatter = ContextFormatter()
        edges = populated_graph.get_all_active_edges()
        text = formatter.format(edges, "kitchen", "Find the key.")
        assert "INVENTORY:" in text


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class TestQueryLayerFull:
    def test_end_to_end_retrieval(self, populated_graph, working_memory):
        ql = QueryLayer(populated_graph)
        context = ql.retrieve(working_memory, current_turn=5)
        assert context.formatted_text != ""
        assert context.total_tokens_estimate <= 512
        assert "LOCATION:" in context.formatted_text
        assert "OBJECTIVE:" in context.formatted_text

    def test_token_budget_enforced(self, populated_graph, working_memory):
        config = QueryConfig(token_budget=50)
        ql = QueryLayer(populated_graph, config=config)
        context = ql.retrieve(working_memory, current_turn=5)
        assert context.total_tokens_estimate <= 50

    def test_metadata_populated(self, populated_graph, working_memory):
        ql = QueryLayer(populated_graph)
        context = ql.retrieve(working_memory, current_turn=5)
        assert "anchor_count" in context.retrieval_metadata
        assert "selected_count" in context.retrieval_metadata
        assert context.retrieval_metadata["anchor_count"] > 0
