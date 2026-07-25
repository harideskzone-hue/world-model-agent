from typing import List, Set
from shared.models import WorkingMemory
from shared.enums import RelationType, NodeType
from world_model.graph_store import GraphStoreBase
from query_engine.context_slice import ContextSlice

class QueryLayer:
    """
    Extracts relevant world state for the SLM without heuristics.
    """
    def retrieve(self, world_model: GraphStoreBase, working_memory: WorkingMemory) -> ContextSlice:
        # 1. Find player node location
        current_room = working_memory.current_room
        player_location_edges = world_model.get_active_edges_by_slot("player", RelationType.LOCATED_IN)
        if player_location_edges:
            current_room = player_location_edges[0].object
            
        # Deduplicate sets for lists
        reachable_rooms_set: Set[str] = set()
        inventory_set: Set[str] = set()
        objects_in_room_list = []
        locked_doors_set: Set[str] = set()
        
        # 2. Find reachable rooms
        if current_room:
            connects_edges = world_model.get_active_edges_by_slot(current_room, RelationType.CONNECTS_TO)
            for edge in connects_edges:
                reachable_rooms_set.add(edge.object)
                
        # 3. Extract inventory
        inventory_edges = world_model.get_active_edges_by_slot("player", RelationType.HOLDS)
        for edge in inventory_edges:
            inventory_set.add(edge.object)
            
        # 4. Extract objects in room
        if current_room:
            contains_edges = world_model.get_active_edges_by_slot(current_room, RelationType.CONTAINS)
            
            # Sort edges by source_turn_id (as proxy for t_observed) descending
            sorted_contains = sorted(contains_edges, key=lambda e: getattr(e, 'source_turn_id', 0), reverse=True)
            
            seen_objs = set()
            for edge in sorted_contains:
                if edge.object not in seen_objs:
                    seen_objs.add(edge.object)
                    objects_in_room_list.append(edge.object)
                
        # 5. Extract locked/closed doors
        if current_room:
            doors_in_room = self._get_doors_in_room(world_model, current_room)
            for door_node in doors_in_room:
                door_name = door_node.id
                states = world_model.get_active_edges_by_slot(door_name, RelationType.HAS_STATE)
                for state_edge in states:
                    if state_edge.object in ("locked", "closed"):
                        locked_doors_set.add(door_name)
                        break
                        
        # 6. Recent changes
        # Gather recent action feedback and edges from the graph without rules or heuristics.
        recent_changes: List[str] = list(working_memory.recent_observations)
        all_edges = world_model.get_all_active_edges()
        if all_edges:
            sorted_edges = sorted(all_edges, key=lambda e: getattr(e, 'source_turn_id', 0), reverse=True)
            # Take up to 5 most recent
            for e in sorted_edges[:5]:
                item = f"{e.subject} {e.relation.value} {e.object}"
                if item not in recent_changes:
                    recent_changes.append(item)
        
        return ContextSlice(
            objective=working_memory.objective,
            current_room=current_room,
            reachable_rooms=sorted(list(reachable_rooms_set)),
            inventory=sorted(list(inventory_set)),
            objects_in_current_room=objects_in_room_list,
            locked_doors=sorted(list(locked_doors_set)),
            recent_changes=recent_changes
        )

    def _get_doors_in_room(self, world_model: GraphStoreBase, room_id: str) -> List:
        """
        Find all DOOR nodes (or conceptually doors) that are:
        1. In the room via CONTAINS edge, OR
        2. Connected to the room via CONNECTS_TO edge
        """
        doors = []
        
        # We will collect node IDs of things contained or connected
        candidate_ids = set()
        
        contains_edges = world_model.get_active_edges_by_slot(room_id, RelationType.CONTAINS)
        for edge in contains_edges:
            candidate_ids.add(edge.object)
            
        connects_edges = world_model.get_active_edges_by_slot(room_id, RelationType.CONNECTS_TO)
        for edge in connects_edges:
            candidate_ids.add(edge.object)
            
        for node_id in candidate_ids:
            node = world_model.get_node(node_id)
            if node:
                # If there's a strict NodeType.DOOR, check it. But standard enum is ROOM/OBJECT/CHARACTER.
                # So we consider any OBJECT that connects rooms or has open/closed/locked states as a potential door.
                # Actually, the user snippet says `node.node_type == NodeType.DOOR`, we'll try to just return all 
                # candidates, and let the outer loop filter by HAS_STATE -> locked/closed.
                doors.append(node)
                
        return doors
