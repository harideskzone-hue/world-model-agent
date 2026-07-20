# query_layer/graph_traverser.py
# ============================================================================
# Step 2: BFS graph traversal to collect candidate facts.
# Spec Reference: Implementation Plan v2.0, Section 7.1, Step 2
# ============================================================================

from __future__ import annotations

import logging
from typing import List, Set

from shared.models import Edge, AnchorNode
from world_model.graph_store import GraphStoreBase

logger = logging.getLogger(__name__)


class GraphTraverser:
    """
    Step 2 of the Query Layer pipeline.
    Bounded BFS from anchor nodes to collect candidate edges.
    """

    def __init__(self, graph: GraphStoreBase, max_depth: int = 2):
        self._graph = graph
        self._max_depth = max_depth

    def traverse(self, anchors: List[AnchorNode]) -> List[Edge]:
        """
        BFS from anchor nodes, collecting active edges within max_depth hops.

        Args:
            anchors: Starting points (sorted by priority)

        Returns:
            Deduplicated list of active edges
        """
        all_edges: List[Edge] = []
        seen_edge_ids: Set[str] = set()

        for anchor in sorted(anchors, key=lambda a: -a.priority):
            _, edges = self._graph.get_room_subgraph(
                anchor.id, depth=self._max_depth
            )
            for edge in edges:
                if edge.id not in seen_edge_ids:
                    seen_edge_ids.add(edge.id)
                    all_edges.append(edge)

        logger.debug(
            f"GraphTraverser: {len(all_edges)} edges from {len(anchors)} anchors"
        )
        return all_edges
