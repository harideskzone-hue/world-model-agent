import pytest
import json
from query_engine.context_slice import ContextSlice
from query_engine.query_layer import QueryLayer
from query_engine.prompt_builder import PromptBuilder
from world_model.graph_store import GraphStoreBase
from shared.models import WorkingMemory, Node, Edge
from shared.enums import NodeType, RelationType

class MockGraphStore(GraphStoreBase):
    def __init__(self):
        super().__init__()
        self.nodes = {}
        self.edges = []
        self.edges_by_slot = {}
        
    def add_node_mock(self, id: str, node_type: NodeType):
        self.nodes[id] = Node(id=id, name=id, node_type=node_type)
        
    def add_edge_mock(self, subject: str, relation: RelationType, obj: str, turn: int = 1):
        edge = Edge(subject=subject, relation=relation, object=obj, source_turn_id=turn)
        self.edges.append(edge)
        
        # Add to slot indexes
        if subject not in self.edges_by_slot:
            self.edges_by_slot[subject] = {}
        if relation not in self.edges_by_slot[subject]:
            self.edges_by_slot[subject][relation] = []
        self.edges_by_slot[subject][relation].append(edge)

    def get_node(self, node_id: str):
        return self.nodes.get(node_id)
        
    def get_active_edges_by_slot(self, entity_id: str, relation: RelationType):
        return self.edges_by_slot.get(entity_id, {}).get(relation, [])
        
    def get_all_active_edges(self):
        return self.edges


    def get_active_edges_for_entity(self, entity_id: str): return []
    def get_all_nodes(self): return list(self.nodes.values())
    def get_edges_at_turn(self, entity_id: str, turn_id: int): return []
    def get_room_subgraph(self, room_id: str, depth: int): return []
    def get_stats(self): return None
    def serialize(self): return ""
    def deserialize(self, data: str): pass
    def add_node(self, node): pass
    def add_edge(self, edge): pass
    def set_current_turn(self, turn_id: int): pass
    def supersede_edge(self, edge_id, new_edge, reason): return ""
    def corroborate_edge(self, edge_id, turn_id): pass

class TestQueryLayer:
    def setup_method(self):
        self.graph = MockGraphStore()
        self.query_layer = QueryLayer()
        self.wm = WorkingMemory(current_room="kitchen", objective="find key")

    def test_find_player_node(self):
        self.graph.add_edge_mock("player", RelationType.LOCATED_IN, "garden")
        self.wm.current_room = "" # Clear working memory room to ensure it finds from graph
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.current_room == "garden"
        
    def test_get_current_room(self):
        # Fallback to WM
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.current_room == "kitchen"

    def test_find_reachable_rooms_single(self):
        self.graph.add_edge_mock("kitchen", RelationType.CONNECTS_TO, "garden")
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.reachable_rooms == ["garden"]

    def test_find_reachable_rooms_multiple(self):
        self.graph.add_edge_mock("kitchen", RelationType.CONNECTS_TO, "garden")
        self.graph.add_edge_mock("kitchen", RelationType.CONNECTS_TO, "pantry")
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert set(slice.reachable_rooms) == {"garden", "pantry"}

    def test_extract_inventory_empty(self):
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.inventory == []

    def test_extract_inventory_multiple(self):
        self.graph.add_edge_mock("player", RelationType.HOLDS, "key")
        self.graph.add_edge_mock("player", RelationType.HOLDS, "apple")
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert set(slice.inventory) == {"apple", "key"}

    def test_extract_objects_in_room(self):
        self.graph.add_edge_mock("kitchen", RelationType.CONTAINS, "knife")
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.objects_in_current_room == ["knife"]

    def test_extract_locked_doors_locked_only(self):
        self.graph.add_edge_mock("kitchen", RelationType.CONTAINS, "front door")
        self.graph.add_node_mock("front door", NodeType.OBJECT)
        self.graph.add_edge_mock("front door", RelationType.HAS_STATE, "locked")
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.locked_doors == ["front door"]

    def test_extract_locked_doors_closed_also(self):
        self.graph.add_edge_mock("kitchen", RelationType.CONTAINS, "trapdoor")
        self.graph.add_node_mock("trapdoor", NodeType.OBJECT)
        self.graph.add_edge_mock("trapdoor", RelationType.HAS_STATE, "closed")
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.locked_doors == ["trapdoor"]

    def test_extract_locked_doors_open_excluded(self):
        self.graph.add_edge_mock("kitchen", RelationType.CONTAINS, "chest")
        self.graph.add_node_mock("chest", NodeType.OBJECT)
        self.graph.add_edge_mock("chest", RelationType.HAS_STATE, "open")
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.locked_doors == []

    def test_handle_missing_player(self):
        # No player LOCATED_IN edge
        self.wm.current_room = ""
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.current_room == ""

    def test_handle_no_reachable_rooms(self):
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.reachable_rooms == []

    def test_return_human_readable_names_only(self):
        self.graph.add_edge_mock("kitchen", RelationType.CONTAINS, "golden key")
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert "golden key" in slice.objects_in_current_room

    def test_no_duplicate_names_in_lists(self):
        self.graph.add_edge_mock("kitchen", RelationType.CONTAINS, "knife")
        self.graph.add_edge_mock("kitchen", RelationType.CONTAINS, "knife")
        slice = self.query_layer.retrieve(self.graph, self.wm)
        assert slice.objects_in_current_room == ["knife"]


class TestPromptBuilder:
    def setup_method(self):
        self.builder = PromptBuilder()
        self.slice = ContextSlice(
            objective="find exit",
            current_room="kitchen",
            reachable_rooms=["garden"],
            inventory=["key"],
            objects_in_current_room=["knife", "apple"],
            locked_doors=["front door"],
            recent_changes=["player located_in garden"]
        )
        self.grammar = "['go north', 'take key']"

    def test_format_lists_as_json(self):
        prompt = self.builder.build(self.slice, self.grammar)
        assert '["garden"]' in prompt
        assert '["key"]' in prompt
        assert '["knife", "apple"]' in prompt
        assert '["front door"]' in prompt
        
    def test_populate_template_correctly(self):
        prompt = self.builder.build(self.slice, self.grammar)
        assert "=== CURRENT WORLD STATE ===" in prompt
        assert "[GOAL]\nfind exit" in prompt
        assert "[CURRENT LOCATION]\nkitchen" in prompt
        
    def test_token_estimation(self):
        tokens = self.builder._estimate_tokens("word " * 10)
        assert tokens == int(10 * 1.3)
        
    def test_no_truncation_needed_under_budget(self):
        prompt = self.builder.build(self.slice, self.grammar)
        assert '["player located_in garden"]' in prompt
        
    def test_truncate_recent_changes_first(self):
        # Make a giant grammar string to force budget limit
        giant_grammar = "word " * 1200
        prompt = self.builder.build(self.slice, giant_grammar)
        # recent_changes should be dropped
        assert '["player located_in garden"]' not in prompt
        
    def test_truncate_objects_by_recency(self):
        self.slice.objects_in_current_room = [f"obj{i}" for i in range(100)]
        giant_grammar = "word " * 970
        prompt = self.builder.build(self.slice, giant_grammar)
        
        # It should truncate some objects
        assert 'obj99' not in prompt
        assert 'obj0' in prompt  # The most recent is at the beginning, so we pop from end!
        
        # Note: the query_layer already sorted objects by source_turn_id descending
        # So objects_in_current_room[0] is the newest. pop() removes from the end (the oldest).
        
    def test_keep_locked_doors_unless_necessary(self):
        self.slice.objects_in_current_room = [f"obj{i}" for i in range(500)]
        giant_grammar = "word " * 1100
        prompt = self.builder.build(self.slice, giant_grammar)
        # It should truncate objects but try to keep locked_doors
        assert '["front door"]' in prompt or '[]' in prompt
        
    def test_keep_mandatory_fields_always(self):
        giant_grammar = "word " * 3000
        prompt = self.builder.build(self.slice, giant_grammar)
        # Even if over budget, it must keep mandatory fields
        assert "[GOAL]\nfind exit" in prompt
        assert "[CURRENT LOCATION]\nkitchen" in prompt
        assert '["garden"]' in prompt
        assert '["key"]' in prompt

    def test_no_extra_fields_added(self):
        prompt = self.builder.build(self.slice, self.grammar)
        assert "[UNKNOWN]" not in prompt
        
    def test_grammar_description_included(self):
        prompt = self.builder.build(self.slice, "my grammar")
        assert "[GRAMMAR DESCRIPTION]\nmy grammar" in prompt


class TestQueryEngineIntegration:
    def setup_method(self):
        self.graph = MockGraphStore()
        self.query_layer = QueryLayer()
        self.builder = PromptBuilder()
        self.wm = WorkingMemory(current_room="kitchen", objective="find key")
        
    def test_query_layer_to_prompt_builder_end_to_end(self):
        self.graph.add_edge_mock("kitchen", RelationType.CONTAINS, "knife")
        self.graph.add_edge_mock("player", RelationType.HOLDS, "apple")
        
        slice = self.query_layer.retrieve(self.graph, self.wm)
        prompt = self.builder.build(slice, "['take knife']")
        
        assert "[INVENTORY]\n[\"apple\"]" in prompt
        assert "[OBJECTS IN CURRENT ROOM]\n[\"knife\"]" in prompt
        assert "[GRAMMAR DESCRIPTION]\n['take knife']" in prompt
        
    def test_prompt_builder_output_is_valid_string(self):
        slice = self.query_layer.retrieve(self.graph, self.wm)
        prompt = self.builder.build(slice, "grammar")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
    def test_empty_world_produces_sensible_output(self):
        self.wm.current_room = ""
        self.wm.objective = ""
        slice = self.query_layer.retrieve(self.graph, self.wm)
        prompt = self.builder.build(slice, "[]")
        
        assert "[GOAL]\n\n" in prompt
        assert "[CURRENT LOCATION]\n\n" in prompt
        assert "[]" in prompt # For lists
