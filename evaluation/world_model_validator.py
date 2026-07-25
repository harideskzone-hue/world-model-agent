from dataclasses import dataclass
from typing import List
from world_model.graph_store import InMemoryGraphStore
from shared.enums import EdgeStatus, NodeType

@dataclass
class ConsistencyViolation:
    violation_type: str  # "duplicate_active_edge", "temporal_invalid", etc.
    description: str
    affected_nodes: List[str]

@dataclass
class ConsistencyReport:
    is_valid: bool
    violations: List[ConsistencyViolation]
    total_nodes: int
    total_edges: int
    active_edges: int
    superseded_edges: int

class WorldModelValidator:
    """
    Validates world model consistency per CONSISTENCY_RULES.md & EVALUATION_PLAN.md.
    
    Checks:
    1. No two ACTIVE edges with identical (subject, relation, target)
    2. Temporal validity: t_valid_from <= t_observed for every edge
    3. Temporal validity: if t_valid_until set, t_valid_from <= t_valid_until <= current_turn
    4. Exactly one PLAYER node exists
    5. Superseded edges have valid superseded_by pointers
    6. No orphaned superseded edges (pointing to non-existent edges)
    """
    
    def validate(self, graph_store: InMemoryGraphStore, current_turn: int) -> ConsistencyReport:
        """Validate entire graph store."""
        violations = []
        
        # Check 1: No duplicate active edges
        active_edges = graph_store.get_all_active_edges()
        seen = {}
        for edge in active_edges:
            key = (edge.subject, edge.relation, edge.object)
            if key in seen:
                violations.append(ConsistencyViolation(
                    violation_type="duplicate_active_edge",
                    description=f"Duplicate active edge: {key}",
                    affected_nodes=[edge.subject, edge.object]
                ))
            seen[key] = edge.id
        
        # Check 2 & 3: Temporal validity
        all_edges = graph_store._edges.values()
        for edge in all_edges:
            if edge.t_valid_from > edge.t_observed:
                violations.append(ConsistencyViolation(
                    violation_type="temporal_invalid_from",
                    description=f"t_valid_from ({edge.t_valid_from}) > t_observed ({edge.t_observed})",
                    affected_nodes=[edge.subject, edge.object]
                ))
            
            if edge.t_valid_until is not None:
                if edge.t_valid_until < edge.t_valid_from:
                    violations.append(ConsistencyViolation(
                        violation_type="temporal_invalid_until",
                        description=f"t_valid_until ({edge.t_valid_until}) < t_valid_from ({edge.t_valid_from})",
                        affected_nodes=[edge.subject, edge.object]
                    ))
                if edge.t_valid_until > current_turn:
                    violations.append(ConsistencyViolation(
                        violation_type="temporal_future_until",
                        description=f"t_valid_until ({edge.t_valid_until}) > current_turn ({current_turn})",
                        affected_nodes=[edge.subject, edge.object]
                    ))
        
        # Check 4: Exactly one PLAYER node (id="player", type=CHARACTER)
        player_nodes = [n for n in graph_store.get_all_nodes() if n.id == "player" and n.node_type == NodeType.CHARACTER]
        if len(player_nodes) != 1:
            violations.append(ConsistencyViolation(
                violation_type="player_node_count",
                description=f"Expected 1 PLAYER node, found {len(player_nodes)}",
                affected_nodes=[]
            ))
        
        # Check 5 & 6: Supersession validity
        for edge in all_edges:
            if edge.status == EdgeStatus.SUPERSEDED:
                if not edge.superseded_by:
                    violations.append(ConsistencyViolation(
                        violation_type="superseded_no_pointer",
                        description=f"SUPERSEDED edge {edge.id} has no superseded_by pointer",
                        affected_nodes=[edge.subject, edge.object]
                    ))
                elif edge.superseded_by not in graph_store._edges:
                    violations.append(ConsistencyViolation(
                        violation_type="superseded_orphaned",
                        description=f"SUPERSEDED edge {edge.id} points to non-existent {edge.superseded_by}",
                        affected_nodes=[edge.subject, edge.object]
                    ))
        
        # Compile report
        stats = graph_store.get_stats()
        return ConsistencyReport(
            is_valid=len(violations) == 0,
            violations=violations,
            total_nodes=stats.total_nodes,
            total_edges=stats.total_edges,
            active_edges=stats.active_edges,
            superseded_edges=stats.superseded_edges,
        )
