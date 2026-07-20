# query_layer/budget_manager.py
# ============================================================================
# Step 4: Token-budget truncation.
# Spec Reference: Implementation Plan v2.0, Section 7.1, Step 4
# ============================================================================

from __future__ import annotations

import logging
from typing import List, Tuple

from shared.models import Edge
from shared.config import QueryConfig

logger = logging.getLogger(__name__)


class BudgetManager:
    """
    Step 4 of the Query Layer pipeline.
    Enforces the hard token budget ceiling by truncating lower-scored edges.
    """

    def __init__(self, config: QueryConfig):
        self._config = config

    def truncate(
        self, scored_edges: List[Tuple[Edge, float]]
    ) -> Tuple[List[Edge], int, int]:
        """
        Select edges that fit within the token budget.

        Args:
            scored_edges: List of (Edge, score) sorted descending by score

        Returns:
            Tuple of (selected_edges, total_tokens, excluded_count)
        """
        budget = self._config.token_budget
        token_count = 0
        selected: List[Edge] = []
        excluded = 0

        for edge, score in scored_edges:
            est_tokens = self._estimate_tokens(edge)
            if token_count + est_tokens <= budget:
                selected.append(edge)
                token_count += est_tokens
            else:
                excluded += 1

        logger.debug(
            f"BudgetManager: {len(selected)} edges selected, "
            f"{excluded} excluded, ~{token_count} tokens of {budget} budget"
        )
        return selected, token_count, excluded

    def _estimate_tokens(self, edge: Edge) -> int:
        """
        Rough token estimate for an edge's text representation.
        Approximation: words × 1.3 (sub-word tokenization factor).
        """
        text = f"{edge.subject} {edge.relation.value} {edge.object}"
        if edge.direction:
            text += f" ({edge.direction})"
        word_count = len(text.split())
        return int(word_count * 1.3) + 2  # +2 for formatting overhead
