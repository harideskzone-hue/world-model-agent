# world_model/persistence/serializer.py
# ============================================================================
# JSON serialization/deserialization for world model snapshots.
# Spec Reference: Implementation Plan v2.0, Section 5.6
# ============================================================================

import json
import os
from pathlib import Path
from typing import Optional

from world_model.graph_store import GraphStoreBase


def save_snapshot(graph: GraphStoreBase, filepath: str) -> None:
    """
    Save a world model snapshot to a JSON file.
    Creates parent directories if needed.
    
    Args:
        graph: The graph store to serialize
        filepath: Path to the output JSON file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = graph.serialize()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(data)


def load_snapshot(graph: GraphStoreBase, filepath: str) -> None:
    """
    Load a world model snapshot from a JSON file.
    Replaces the current graph state.
    
    Args:
        graph: The graph store to load into
        filepath: Path to the JSON file
    
    Raises:
        FileNotFoundError: If the snapshot file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Snapshot file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()
    graph.deserialize(data)


def save_human_readable(graph: GraphStoreBase, filepath: str) -> None:
    """
    Save a human-readable summary of the world model.
    This is what judges inspect — not the full JSON.
    
    Output format:
      ROOMS:
        kitchen (confidence: 0.95, observed: turns 0-15)
          contains: brass key (locked), apple
          connects_to: garden (north), hallway (south)
      
      INVENTORY:
        silver coin, old map
      
      REVISION HISTORY:
        [Turn 12] door: locked → unlocked (reason: newer direct observation)
    """
    from shared.enums import EdgeStatus, RelationType, NodeType
    
    lines = ["=" * 60, "WORLD MODEL — Human-Readable Summary", "=" * 60, ""]
    
    all_edges = graph.get_all_active_edges()
    all_nodes = graph.get_all_nodes()
    
    # Group by rooms
    rooms = [n for n in all_nodes if n.node_type == NodeType.ROOM]
    for room in sorted(rooms, key=lambda r: r.name):
        lines.append(f"📍 {room.name.upper()} (confidence: {room.confidence:.2f})")
        room_edges = graph.get_active_edges_for_entity(room.id)
        
        contains = [e for e in room_edges if e.relation == RelationType.CONTAINS and e.subject == room.id]
        connects = [e for e in room_edges if e.relation == RelationType.CONNECTS_TO and e.subject == room.id]
        
        if contains:
            items = []
            for e in contains:
                # Find state for this object
                state_edges = graph.get_active_edges_by_slot(e.object, RelationType.HAS_STATE)
                state_str = f" ({state_edges[0].object})" if state_edges else ""
                items.append(f"{e.object}{state_str}")
            lines.append(f"  Contains: {', '.join(items)}")
        
        if connects:
            dirs = [f"{e.object} ({e.direction or '?'})" for e in connects]
            lines.append(f"  Exits: {', '.join(dirs)}")
        
        lines.append("")
    
    # Inventory
    inventory_edges = [e for e in all_edges if e.relation == RelationType.HOLDS]
    if inventory_edges:
        items = [e.object for e in inventory_edges]
        lines.append(f"🎒 INVENTORY: {', '.join(items)}")
        lines.append("")
    
    # Revision history (superseded edges)
    all_edges_full = list(graph._edges.values()) if hasattr(graph, '_edges') else []
    revisions = [e for e in all_edges_full if e.status == EdgeStatus.SUPERSEDED and e.revision_reason]
    if revisions:
        lines.append("📝 REVISION HISTORY:")
        for rev in sorted(revisions, key=lambda e: e.t_valid_until or 0):
            lines.append(f"  [Turn {rev.t_valid_until}] {rev.subject}.{rev.relation.value}: "
                        f"{rev.object} → superseded ({rev.revision_reason})")
        lines.append("")
    
    # Stats
    stats = graph.get_stats()
    lines.append(f"📊 Stats: {stats.total_nodes} nodes, {stats.active_edges} active / "
                f"{stats.superseded_edges} superseded edges, "
                f"avg confidence: {stats.avg_confidence:.2f}")
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
