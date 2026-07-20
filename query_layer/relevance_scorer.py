# query_layer/relevance_scorer.py
# ============================================================================
# Step 3: Multi-factor relevance scoring for graph edges.
# Spec Reference: Implementation Plan v2.0, Section 7.1, Step 3
# ============================================================================

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Set

from shared.models import Edge, AnchorNode
from shared.enums import RelationType
from shared.config import QueryConfig
from world_model.graph_store import GraphStoreBase

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """
    Step 3 of the Query Layer pipeline.
    Scores edges by multi-factor relevance formula:

      score = α × spatial + β × recency + γ × corroboration + δ × goal_relevance
    """

    def __init__(self, graph: GraphStoreBase, config: QueryConfig):
        self._graph = graph
        self._config = config

    def score(
        self,
        edges: List[Edge],
        anchors: List[AnchorNode],
        current_turn: int,
        sub_goal: str,
    ) -> List[Tuple[Edge, float]]:
        """
        Score all candidate edges by relevance.

        Args:
            edges: Candidate edges from graph traversal
            anchors: Anchor nodes (for spatial distance calculation)
            current_turn: Current turn number (for recency)
            sub_goal: Current sub-goal (for goal relevance)

        Returns:
            List of (Edge, score) tuples, sorted descending by score
        """
        w = self._config.scoring_weights
        alpha = w.get("spatial", 0.35)
        beta = w.get("recency", 0.25)
        gamma = w.get("corroboration", 0.15)
        delta = w.get("goal", 0.25)

        anchor_ids = {a.id for a in anchors}
        goal_terms = set(sub_goal.lower().split()) if sub_goal else set()

        scored: List[Tuple[Edge, float]] = []
        for edge in edges:
            spatial = self._spatial_relevance(edge, anchor_ids)
            recency = self._recency_score(edge, current_turn)
            corroboration = self._corroboration_score(edge)
            goal = self._goal_relevance(edge, goal_terms, sub_goal)

            total = alpha * spatial + beta * recency + gamma * corroboration + delta * goal
            scored.append((edge, round(total, 4)))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _spatial_relevance(self, edge: Edge, anchor_ids: Set[str]) -> float:
        """
        Spatial relevance: higher if edge involves an anchor node.
        Full graph BFS distance is expensive — use a heuristic:
          - 0-hop (edge involves anchor directly): 1.0
          - Otherwise: 0.4 (we assume all traversed edges are within 2 hops)
        """
        if edge.subject in anchor_ids or edge.object in anchor_ids:
            return 1.0
        return 0.4

    def _recency_score(self, edge: Edge, current_turn: int) -> float:
        """
        Recency: more recently observed facts score higher.
        recency = max(0.1, 1.0 - turns_ago × decay)
        """
        turns_ago = max(0, current_turn - edge.t_observed)
        score = max(0.1, 1.0 - turns_ago * self._config.recency_decay)
        return score

    def _corroboration_score(self, edge: Edge) -> float:
        """
        Corroboration: multiply-confirmed facts score higher.
        corroboration = min(1.0, count × scale)
        """
        return min(1.0, edge.corroboration_count * self._config.corroboration_scale)

    def _goal_relevance(self, edge: Edge, goal_terms: Set[str], sub_goal: str) -> float:
        """
        Goal relevance: edges mentioning goal entities score higher.
        """
        if not goal_terms:
            return 0.1

        # Check if edge entities match goal terms
        edge_terms = set(edge.subject.lower().split()) | set(edge.object.lower().split())
        overlap = edge_terms & goal_terms
        if overlap:
            return 1.0

        # Navigation edges get a boost when goal involves movement
        nav_terms = {"go", "find", "get", "reach", "navigate"}
        if edge.relation == RelationType.CONNECTS_TO and goal_terms & nav_terms:
            return 0.8

        return 0.1
