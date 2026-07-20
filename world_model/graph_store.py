# world_model/graph_store.py
# ============================================================================
# In-memory typed graph store with bi-temporal versioned edges.
# Spec Reference: Implementation Plan v2.0, Section 5.6 (GraphStoreBase API)
#
# This is the physical substrate of the agent's beliefs.
# ============================================================================

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Set

from shared.models import Node, Edge, GraphStats
from shared.enums import (
    NodeType, NodeStatus, EdgeStatus, RelationType, ExtractionMethod,
)
from world_model.temporal_versioning import (
    close_validity, extend_validity, is_edge_active_at_turn,
)
from world_model.schema import normalize_entity_name


class GraphStoreBase(ABC):
    """
    Abstract interface for the world model graph.
    Decoupled behind this interface so the in-memory implementation can be
    swapped for a heavier backend without touching Extractor/Updater/Query code.
    """

    # ── Write Operations ──

    @abstractmethod
    def add_node(self, node: Node) -> str:
        """Add or update a node. Returns node ID. Idempotent on ID."""

    @abstractmethod
    def add_edge(self, edge: Edge) -> str:
        """Add a new edge (never overwrites — new version created). Returns edge ID."""

    @abstractmethod
    def supersede_edge(self, edge_id: str, new_edge: Edge, reason: str) -> str:
        """Mark edge as superseded, create replacement. Returns new edge ID."""

    @abstractmethod
    def corroborate_edge(self, edge_id: str, turn_id: int) -> None:
        """Increase confidence and corroboration count of an existing edge."""

    # ── Read Operations ──

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Node]:
        """Return node by ID, or None."""

    @abstractmethod
    def get_all_nodes(self) -> List[Node]:
        """Return all nodes."""

    @abstractmethod
    def get_active_edges_for_entity(self, entity_id: str) -> List[Edge]:
        """All edges where entity is subject or object AND status=ACTIVE."""

    @abstractmethod
    def get_active_edges_by_slot(
        self, subject: str, relation: RelationType
    ) -> List[Edge]:
        """Active edges matching a subject-relation slot. Used by contradiction detector."""

    @abstractmethod
    def get_edges_at_turn(self, entity_id: str, turn_id: int) -> List[Edge]:
        """Time-travel query: edges valid at a specific turn."""

    @abstractmethod
    def get_room_subgraph(
        self, room_id: str, depth: int = 1
    ) -> Tuple[List[Node], List[Edge]]:
        """Return all nodes and edges within `depth` hops of a room."""

    @abstractmethod
    def get_all_active_edges(self) -> List[Edge]:
        """Return all edges with status=ACTIVE. Used for evaluation."""

    # ── Persistence ──

    @abstractmethod
    def serialize(self) -> str:
        """Serialize to JSON string."""

    @abstractmethod
    def deserialize(self, data: str) -> None:
        """Load from JSON string."""

    # ── Metrics ──

    @abstractmethod
    def get_stats(self) -> GraphStats:
        """Return storage size, node count, edge count, active/superseded ratio."""


class InMemoryGraphStore(GraphStoreBase):
    """
    In-memory implementation of the world model graph.

    Storage structure:
      - _nodes: Dict[node_id → Node]
      - _edges: Dict[edge_id → Edge]
      - _subject_index: Dict[subject_id → Set[edge_id]]
      - _object_index: Dict[object_id → Set[edge_id]]
      - _slot_index: Dict[(subject, relation) → Set[edge_id]]
    """

    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}

        # ── Indexes for fast lookup ──
        self._subject_index: Dict[str, Set[str]] = defaultdict(set)
        self._object_index: Dict[str, Set[str]] = defaultdict(set)
        self._slot_index: Dict[Tuple[str, RelationType], Set[str]] = defaultdict(set)

    # ═══════════════════════════════════════════════════════════════════════
    # WRITE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════

    def add_node(self, node: Node) -> str:
        """
        Add or update a node. Idempotent on ID.
        If the node already exists, update its last_observed_turn and corroboration.
        """
        node.id = normalize_entity_name(node.id)
        if node.id in self._nodes:
            existing = self._nodes[node.id]
            existing.last_observed_turn = max(
                existing.last_observed_turn, node.last_observed_turn
            )
            existing.corroboration_count += 1
            existing.confidence = min(0.99, existing.confidence + 0.02)
            if node.status == NodeStatus.ACTIVE:
                existing.status = NodeStatus.ACTIVE
            # Merge attributes (new values take precedence)
            existing.attributes.update(node.attributes)
            return existing.id
        else:
            self._nodes[node.id] = node
            return node.id

    def add_edge(self, edge: Edge) -> str:
        """
        Add a new edge to the graph. Never overwrites existing edges.
        Updates all indexes.
        """
        self._edges[edge.id] = edge
        self._subject_index[edge.subject].add(edge.id)
        self._object_index[edge.object].add(edge.id)
        self._slot_index[(edge.subject, edge.relation)].add(edge.id)
        return edge.id

    def supersede_edge(self, edge_id: str, new_edge: Edge, reason: str) -> str:
        """
        Mark an existing edge as superseded and add its replacement.

        The old edge is NEVER deleted — its status becomes SUPERSEDED,
        its validity window is closed, and a pointer to the new edge is stored.
        """
        if edge_id not in self._edges:
            raise KeyError(f"Edge {edge_id} not found in graph store")

        old_edge = self._edges[edge_id]
        # Close the old edge's validity
        close_validity(
            old_edge,
            turn_id=new_edge.t_observed,
            superseded_by_id=new_edge.id,
            reason=reason,
        )

        # Add the new replacement edge
        self.add_edge(new_edge)
        return new_edge.id

    def corroborate_edge(self, edge_id: str, turn_id: int) -> None:
        """Increase confidence and corroboration count of an existing edge."""
        if edge_id not in self._edges:
            raise KeyError(f"Edge {edge_id} not found in graph store")
        extend_validity(self._edges[edge_id], turn_id)

    # ═══════════════════════════════════════════════════════════════════════
    # READ OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════

    def get_node(self, node_id: str) -> Optional[Node]:
        """Return node by normalized ID, or None."""
        return self._nodes.get(normalize_entity_name(node_id))

    def get_all_nodes(self) -> List[Node]:
        """Return all nodes."""
        return list(self._nodes.values())

    def get_active_edges_for_entity(self, entity_id: str) -> List[Edge]:
        """All ACTIVE edges where entity is subject or object."""
        entity_id = normalize_entity_name(entity_id)
        edge_ids = self._subject_index.get(entity_id, set()) | \
                   self._object_index.get(entity_id, set())
        return [
            self._edges[eid]
            for eid in edge_ids
            if eid in self._edges and self._edges[eid].status == EdgeStatus.ACTIVE
        ]

    def get_active_edges_by_slot(
        self, subject: str, relation: RelationType
    ) -> List[Edge]:
        """Active edges matching a subject-relation slot."""
        subject = normalize_entity_name(subject)
        edge_ids = self._slot_index.get((subject, relation), set())
        return [
            self._edges[eid]
            for eid in edge_ids
            if eid in self._edges and self._edges[eid].status == EdgeStatus.ACTIVE
        ]

    def get_edges_at_turn(self, entity_id: str, turn_id: int) -> List[Edge]:
        """Time-travel query: edges valid at a specific turn."""
        entity_id = normalize_entity_name(entity_id)
        edge_ids = self._subject_index.get(entity_id, set()) | \
                   self._object_index.get(entity_id, set())
        return [
            self._edges[eid]
            for eid in edge_ids
            if eid in self._edges and is_edge_active_at_turn(self._edges[eid], turn_id)
        ]

    def get_room_subgraph(
        self, room_id: str, depth: int = 1
    ) -> Tuple[List[Node], List[Edge]]:
        """
        BFS traversal from a room node, collecting all nodes and active edges
        within `depth` hops.
        """
        room_id = normalize_entity_name(room_id)
        visited_nodes: Set[str] = set()
        collected_edges: List[Edge] = []
        frontier: Set[str] = {room_id}

        for _ in range(depth + 1):
            next_frontier: Set[str] = set()
            for node_id in frontier:
                if node_id in visited_nodes:
                    continue
                visited_nodes.add(node_id)

                # Collect active edges for this node
                active_edges = self.get_active_edges_for_entity(node_id)
                for edge in active_edges:
                    collected_edges.append(edge)
                    # Add connected nodes to next frontier
                    next_frontier.add(edge.subject)
                    next_frontier.add(edge.object)

            frontier = next_frontier - visited_nodes

        # Collect the actual Node objects
        nodes = [
            self._nodes[nid]
            for nid in visited_nodes
            if nid in self._nodes
        ]

        # Deduplicate edges by ID
        seen_edge_ids: Set[str] = set()
        unique_edges = []
        for e in collected_edges:
            if e.id not in seen_edge_ids:
                seen_edge_ids.add(e.id)
                unique_edges.append(e)

        return nodes, unique_edges

    def get_all_active_edges(self) -> List[Edge]:
        """Return all edges with status=ACTIVE."""
        return [e for e in self._edges.values() if e.status == EdgeStatus.ACTIVE]

    # ═══════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════

    def serialize(self) -> str:
        """Serialize the entire graph to a JSON string."""

        def node_to_dict(n: Node) -> dict:
            return {
                "id": n.id,
                "name": n.name,
                "node_type": n.node_type.value,
                "status": n.status.value,
                "confidence": n.confidence,
                "first_observed_turn": n.first_observed_turn,
                "last_observed_turn": n.last_observed_turn,
                "corroboration_count": n.corroboration_count,
                "attributes": n.attributes,
            }

        def edge_to_dict(e: Edge) -> dict:
            return {
                "id": e.id,
                "subject": e.subject,
                "relation": e.relation.value,
                "object": e.object,
                "confidence": round(e.confidence, 4),
                "source_turn_id": e.source_turn_id,
                "extraction_method": e.extraction_method.value,
                "t_observed": e.t_observed,
                "t_valid_from": e.t_valid_from,
                "t_valid_until": e.t_valid_until,
                "status": e.status.value,
                "corroboration_count": e.corroboration_count,
                "superseded_by": e.superseded_by,
                "revision_reason": e.revision_reason,
                "direction": e.direction,
            }

        data = {
            "nodes": [node_to_dict(n) for n in self._nodes.values()],
            "edges": [edge_to_dict(e) for e in self._edges.values()],
        }
        return json.dumps(data, indent=2)

    def deserialize(self, data: str) -> None:
        """Load graph from JSON string. Replaces current state."""
        parsed = json.loads(data)

        self._nodes.clear()
        self._edges.clear()
        self._subject_index.clear()
        self._object_index.clear()
        self._slot_index.clear()

        for nd in parsed.get("nodes", []):
            node = Node(
                id=nd["id"],
                name=nd["name"],
                node_type=NodeType(nd["node_type"]),
                status=NodeStatus(nd["status"]),
                confidence=nd["confidence"],
                first_observed_turn=nd["first_observed_turn"],
                last_observed_turn=nd["last_observed_turn"],
                corroboration_count=nd["corroboration_count"],
                attributes=nd.get("attributes", {}),
            )
            self._nodes[node.id] = node

        for ed in parsed.get("edges", []):
            edge = Edge(
                id=ed["id"],
                subject=ed["subject"],
                relation=RelationType(ed["relation"]),
                object=ed["object"],
                confidence=ed["confidence"],
                source_turn_id=ed["source_turn_id"],
                extraction_method=ExtractionMethod(ed["extraction_method"]),
                t_observed=ed["t_observed"],
                t_valid_from=ed["t_valid_from"],
                t_valid_until=ed.get("t_valid_until"),
                status=EdgeStatus(ed["status"]),
                corroboration_count=ed["corroboration_count"],
                superseded_by=ed.get("superseded_by"),
                revision_reason=ed.get("revision_reason"),
                direction=ed.get("direction"),
            )
            self._edges[edge.id] = edge
            self._subject_index[edge.subject].add(edge.id)
            self._object_index[edge.object].add(edge.id)
            self._slot_index[(edge.subject, edge.relation)].add(edge.id)

    # ═══════════════════════════════════════════════════════════════════════
    # METRICS
    # ═══════════════════════════════════════════════════════════════════════

    def get_stats(self) -> GraphStats:
        """Return diagnostic statistics."""
        active = sum(1 for e in self._edges.values() if e.status == EdgeStatus.ACTIVE)
        superseded = sum(
            1 for e in self._edges.values() if e.status == EdgeStatus.SUPERSEDED
        )
        confidences = [e.confidence for e in self._edges.values()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        storage = sys.getsizeof(self.serialize())

        return GraphStats(
            total_nodes=len(self._nodes),
            total_edges=len(self._edges),
            active_edges=active,
            superseded_edges=superseded,
            storage_bytes=storage,
            avg_confidence=round(avg_conf, 4),
        )
