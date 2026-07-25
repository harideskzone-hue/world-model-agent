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
        else:
            # Check inverted semantic extractions (e.g. room contains player or room located_in player)
            for e in world_model.get_all_active_edges():
                if e.relation == RelationType.CONTAINS and e.object in ("player", "you", "agent", "self"):
                    current_room = e.subject
                    break
                elif e.relation == RelationType.LOCATED_IN and e.object in ("player", "you", "agent", "self"):
                    current_room = e.subject
                    break
            
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
            located_edges = [e for e in world_model.get_all_active_edges() if e.relation == RelationType.LOCATED_IN and e.object == current_room and e.subject not in ("player", "you", "agent", "self")]
            
            # Sort edges by source_turn_id (as proxy for t_observed) descending
            all_room_edges = sorted(contains_edges + located_edges, key=lambda e: getattr(e, 'source_turn_id', 0), reverse=True)
            
            seen_objs = set()
            for edge in all_room_edges:
                obj_name = edge.object if edge.relation == RelationType.CONTAINS else edge.subject
                if obj_name not in seen_objs and obj_name not in ("player", "you", "agent", "self", current_room, "nothing", "empty"):
                    seen_objs.add(obj_name)
                    objects_in_room_list.append(obj_name)
                    # Check for 2-hop containment (objects on supporters or in open containers)
                    sub_contains = world_model.get_active_edges_by_slot(obj_name, RelationType.CONTAINS)
                    sub_located = [e for e in world_model.get_all_active_edges() if e.relation == RelationType.LOCATED_IN and e.object == obj_name]
                    for sub_e in sub_contains + sub_located:
                        sub_obj = sub_e.object if sub_e.relation == RelationType.CONTAINS else sub_e.subject
                        if sub_obj not in seen_objs and sub_obj not in ("player", "you", "agent", "self", current_room, "nothing", "empty"):
                            seen_objs.add(sub_obj)
                            objects_in_room_list.append(f"{sub_obj} (in/on {obj_name})")
                
        # 5. Extract locked/closed doors and gateways
        if current_room:
            doors_in_room = self._get_doors_in_room(world_model, current_room)
            for door_node in doors_in_room:
                door_name = door_node.id
                states = world_model.get_active_edges_by_slot(door_name, RelationType.HAS_STATE)
                is_locked_or_closed = False
                for state_edge in states:
                    if state_edge.object in ("locked", "closed"):
                        is_locked_or_closed = True
                        break
                if is_locked_or_closed:
                    locked_doors_set.add(door_name)
                # Make sure doors and portals are visible in the room objects list
                if door_name not in seen_objs and door_name != current_room:
                    seen_objs.add(door_name)
                    objects_in_room_list.append(door_name)
                        
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
            
        located_edges = [e for e in world_model.get_all_active_edges() if e.relation == RelationType.LOCATED_IN and e.object == room_id]
        for edge in located_edges:
            candidate_ids.add(edge.subject)
            
        for edge in world_model.get_all_active_edges():
            if edge.relation == RelationType.HAS_STATE and edge.object in ("locked", "closed"):
                other_room_edges = [e2 for e2 in world_model.get_active_edges_by_slot(edge.subject, RelationType.LOCATED_IN) if e2.object != room_id and e2.object not in ("player", "player inventory", "you", "me")]
                if not other_room_edges:
                    candidate_ids.add(edge.subject)
            elif edge.relation == RelationType.CONNECTS_TO and edge.subject not in (room_id, "player", "you", "me"):
                other_room_edges = [e2 for e2 in world_model.get_active_edges_by_slot(edge.subject, RelationType.LOCATED_IN) if e2.object != room_id and e2.object not in ("player", "player inventory", "you", "me")]
                if not other_room_edges:
                    candidate_ids.add(edge.subject)
            
        for node_id in candidate_ids:
            node = world_model.get_node(node_id)
            if node:
                doors.append(node)
                
        return doors
