# query_layer/query_layer.py
# ============================================================================
# Main QueryLayer class — orchestrates the 5-step retrieval pipeline.
# Spec Reference: Implementation Plan v2.0, Section 7.2
# ============================================================================

from __future__ import annotations

import logging
from typing import Optional

from shared.models import WorkingMemory, ContextSlice
from shared.config import QueryConfig
from world_model.graph_store import GraphStoreBase
from query_layer.anchor_resolver import AnchorResolver
from query_layer.graph_traverser import GraphTraverser
from query_layer.relevance_scorer import RelevanceScorer
from query_layer.budget_manager import BudgetManager
from query_layer.context_formatter import ContextFormatter

logger = logging.getLogger(__name__)


class QueryLayer:
    """
    Retrieves a minimal, relevant context slice from the World Model
    within the token budget.

    5-step pipeline:
      1. AnchorResolver    — determine traversal starting points
      2. GraphTraverser     — BFS fact collection from anchors
      3. RelevanceScorer    — multi-factor scoring
      4. BudgetManager      — token-budget truncation
      5. ContextFormatter   — edge → human-readable text

    Guarantees:
      - Output never exceeds config.token_budget tokens
      - Agent's current location is always included
      - Active inventory is always included
      - Objective is always included
    """

    def __init__(
        self,
        graph: GraphStoreBase,
        config: Optional[QueryConfig] = None,
    ):
        self._graph = graph
        self._config = config or QueryConfig()
        self._anchor_resolver = AnchorResolver(graph)
        self._traverser = GraphTraverser(graph, self._config.max_traversal_depth)
        self._scorer = RelevanceScorer(graph, self._config)
        self._budget_manager = BudgetManager(self._config)
        self._formatter = ContextFormatter()

    def retrieve(
        self,
        working_memory: WorkingMemory,
        current_turn: int,
    ) -> ContextSlice:
        """
        Execute the 5-step retrieval pipeline.

        Args:
            working_memory: Current agent context
            current_turn: Current turn number

        Returns:
            ContextSlice with formatted text and metadata
        """
        sub_goal = working_memory.current_sub_goal or working_memory.objective

        # Step 1: Anchor Resolution
        anchors = self._anchor_resolver.resolve(working_memory, sub_goal)

        # Step 2: Graph Traversal
        candidate_edges = self._traverser.traverse(anchors)

        # Step 3: Relevance Scoring
        scored_edges = self._scorer.score(
            candidate_edges, anchors, current_turn, sub_goal,
        )

        # Step 4: Budget Truncation
        selected, total_tokens, excluded = self._budget_manager.truncate(scored_edges)

        # Step 5: Context Formatting
        formatted = self._formatter.format(
            edges=selected,
            current_room=working_memory.current_room,
            objective=working_memory.objective,
        )

        context_slice = ContextSlice(
            formatted_text=formatted,
            included_facts=selected,
            excluded_count=excluded,
            total_tokens_estimate=total_tokens,
            retrieval_metadata={
                "anchor_count": len(anchors),
                "candidate_count": len(candidate_edges),
                "selected_count": len(selected),
                "scoring_weights": self._config.scoring_weights,
            },
        )

        logger.info(
            f"QueryLayer: {len(selected)} facts, ~{total_tokens} tokens, "
            f"{excluded} excluded"
        )
        return context_slice
