# tests/unit/test_contradiction_detector.py
# ============================================================================
# Unit tests for the ContradictionDetector and RevisionPolicy.
# ============================================================================

import pytest

from world_model.graph_store import InMemoryGraphStore
from shared.models import Node, Edge, CandidateFact
from shared.enums import (
    NodeType, RelationType, ExtractionType, ExtractionMethod,
    ContradictionType, ConflictCategory, RevisionActionType, EdgeStatus,
)
from updater.contradiction_detector import ContradictionDetector
from updater.revision_policy import RevisionPolicy
from updater.updater import Updater


@pytest.fixture
def graph():
    g = InMemoryGraphStore()
    g.add_node(Node(id="kitchen", name="Kitchen", node_type=NodeType.ROOM))
    g.add_node(Node(id="brass key", name="brass key", node_type=NodeType.OBJECT))
    g.add_node(Node(id="player", name="player", node_type=NodeType.CHARACTER))
    g.add_node(Node(id="garden", name="Garden", node_type=NodeType.ROOM))
    # kitchen contains brass key
    g.add_edge(Edge(
        id="e1", subject="kitchen", relation=RelationType.CONTAINS,
        object="brass key", confidence=0.9,
        extraction_method=ExtractionMethod.SLM, t_observed=0, t_valid_from=0,
    ))
    # brass key is locked
    g.add_edge(Edge(
        id="e2", subject="brass key", relation=RelationType.HAS_STATE,
        object="locked", confidence=0.9,
        extraction_method=ExtractionMethod.SLM, t_observed=0, t_valid_from=0,
    ))
    # player located_in kitchen
    g.add_edge(Edge(
        id="e3", subject="player", relation=RelationType.LOCATED_IN,
        object="kitchen", confidence=0.95,
        extraction_method=ExtractionMethod.SLM, t_observed=0, t_valid_from=0,
    ))
    return g


def make_candidate(subject, relation, obj, confidence=0.9):
    return CandidateFact(
        subject=subject,
        relation=RelationType(relation) if isinstance(relation, str) else relation,
        object=obj,
        confidence=confidence,
        source_turn_id=5,
        extraction_type=ExtractionType.DIRECT,
        extraction_method=ExtractionMethod.SLM,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CONTRADICTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class TestContradictionDetector:
    def test_new_fact_is_expand(self, graph):
        detector = ContradictionDetector(graph)
        candidate = make_candidate("garden", RelationType.CONTAINS, "rose")
        result = detector.detect(candidate)
        assert result.contradiction_type == ContradictionType.EXPAND

    def test_same_fact_is_corroborate(self, graph):
        detector = ContradictionDetector(graph)
        candidate = make_candidate("kitchen", RelationType.CONTAINS, "brass key")
        result = detector.detect(candidate)
        assert result.contradiction_type == ContradictionType.CORROBORATE
        assert result.existing_edge is not None

    def test_state_conflict_is_revise(self, graph):
        detector = ContradictionDetector(graph)
        candidate = make_candidate("brass key", RelationType.HAS_STATE, "unlocked")
        result = detector.detect(candidate)
        assert result.contradiction_type == ContradictionType.REVISE
        assert result.conflict_category == ConflictCategory.STATE_MUTUAL_EXCLUSION

    def test_location_change_is_revise(self, graph):
        detector = ContradictionDetector(graph)
        candidate = make_candidate("player", RelationType.LOCATED_IN, "garden")
        result = detector.detect(candidate)
        assert result.contradiction_type == ContradictionType.REVISE
        assert result.conflict_category == ConflictCategory.LOCATION_CHANGE

    def test_multi_valued_slot_is_expand(self, graph):
        """Room can contain multiple objects — new object is EXPAND, not REVISE."""
        detector = ContradictionDetector(graph)
        candidate = make_candidate("kitchen", RelationType.CONTAINS, "apple")
        result = detector.detect(candidate)
        assert result.contradiction_type == ContradictionType.EXPAND


# ═══════════════════════════════════════════════════════════════════════════
# REVISION POLICY
# ═══════════════════════════════════════════════════════════════════════════

class TestRevisionPolicy:
    def test_expand_creates_new_edge(self, graph):
        detector = ContradictionDetector(graph)
        policy = RevisionPolicy()
        candidate = make_candidate("garden", RelationType.CONTAINS, "rose")
        result = detector.detect(candidate)
        action = policy.resolve(result, candidate, turn_id=5)
        assert action.action_type == RevisionActionType.EXPAND
        assert action.new_edge is not None
        assert action.new_edge.object == "rose"

    def test_corroborate_references_existing(self, graph):
        detector = ContradictionDetector(graph)
        policy = RevisionPolicy()
        candidate = make_candidate("kitchen", RelationType.CONTAINS, "brass key")
        result = detector.detect(candidate)
        action = policy.resolve(result, candidate, turn_id=5)
        assert action.action_type == RevisionActionType.CORROBORATE
        assert action.existing_edge_id == "e1"

    def test_r1_recency_wins_low_corroboration(self, graph):
        """R1: When old edge has low corroboration, hard supersede."""
        detector = ContradictionDetector(graph)
        policy = RevisionPolicy()
        candidate = make_candidate("brass key", RelationType.HAS_STATE, "unlocked")
        result = detector.detect(candidate)
        action = policy.resolve(result, candidate, turn_id=5)
        assert action.action_type == RevisionActionType.SUPERSEDE
        assert action.existing_edge_id == "e2"
        assert action.new_edge.object == "unlocked"

    def test_r2_corroboration_override_high_corroboration(self, graph):
        """R2: When old edge has high corroboration, provisional supersede with penalty."""
        # Boost the old edge's corroboration count
        for i in range(5):
            graph.corroborate_edge("e2", turn_id=i)
        assert graph._edges["e2"].corroboration_count >= 3

        detector = ContradictionDetector(graph)
        policy = RevisionPolicy()
        candidate = make_candidate("brass key", RelationType.HAS_STATE, "unlocked")
        result = detector.detect(candidate)
        action = policy.resolve(result, candidate, turn_id=10)
        assert action.action_type == RevisionActionType.PROVISIONAL_SUPERSEDE
        # New edge should have reduced confidence
        assert action.new_edge.confidence < candidate.confidence


# ═══════════════════════════════════════════════════════════════════════════
# FULL UPDATER
# ═══════════════════════════════════════════════════════════════════════════

class TestUpdater:
    def test_expand_adds_to_graph(self, graph):
        updater = Updater(graph)
        candidates = [make_candidate("garden", RelationType.CONTAINS, "rose")]
        report = updater.update(candidates, turn_id=5)
        assert report.expanded == 1
        # Verify the edge is in the graph
        edges = graph.get_active_edges_by_slot("garden", RelationType.CONTAINS)
        assert any(e.object == "rose" for e in edges)

    def test_corroborate_preserves_confidence(self, graph):
        updater = Updater(graph)
        original_conf = graph._edges["e1"].confidence
        candidates = [make_candidate("kitchen", RelationType.CONTAINS, "brass key")]
        report = updater.update(candidates, turn_id=5)
        assert report.corroborated == 1
        assert graph._edges["e1"].confidence == original_conf

    def test_supersede_changes_state(self, graph):
        updater = Updater(graph)
        candidates = [make_candidate("brass key", RelationType.HAS_STATE, "unlocked")]
        report = updater.update(candidates, turn_id=5)
        assert report.revised == 1
        # Old edge superseded
        assert graph._edges["e2"].status == EdgeStatus.SUPERSEDED
        # New edge active with "unlocked"
        active_states = graph.get_active_edges_by_slot("brass key", RelationType.HAS_STATE)
        assert len(active_states) == 1
        assert active_states[0].object == "unlocked"

    def test_location_update(self, graph):
        updater = Updater(graph)
        candidates = [make_candidate("player", RelationType.LOCATED_IN, "garden")]
        report = updater.update(candidates, turn_id=5)
        assert report.revised == 1
        # Player now in garden
        active = graph.get_active_edges_by_slot("player", RelationType.LOCATED_IN)
        assert len(active) == 1
        assert active[0].object == "garden"

    def test_multiple_candidates_in_one_turn(self, graph):
        updater = Updater(graph)
        candidates = [
            make_candidate("garden", RelationType.CONTAINS, "rose"),
            make_candidate("garden", RelationType.CONTAINS, "fountain"),
            make_candidate("kitchen", RelationType.CONTAINS, "brass key"),  # corroborate
        ]
        report = updater.update(candidates, turn_id=5)
        assert report.expanded == 2
        assert report.corroborated == 1

    def test_update_report_has_revision_details(self, graph):
        updater = Updater(graph)
        candidates = [make_candidate("brass key", RelationType.HAS_STATE, "unlocked")]
        report = updater.update(candidates, turn_id=5)
        assert len(report.revisions) == 1
        assert "R1" in report.revisions[0].revision_reason

    def test_nodes_auto_created(self, graph):
        updater = Updater(graph)
        candidates = [make_candidate("new room", RelationType.CONTAINS, "magic sword")]
        report = updater.update(candidates, turn_id=5)
        assert graph.get_node("new room") is not None
        assert graph.get_node("magic sword") is not None
