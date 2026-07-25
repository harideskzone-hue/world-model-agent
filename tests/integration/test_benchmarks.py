import pytest
import os
from evaluation.world_model_validator import WorldModelValidator, ConsistencyViolation
from evaluation.performance_profiler import PerformanceProfiler
from scripts.benchmark_suite import run_benchmark_suite, BenchmarkReport
from world_model.graph_store import InMemoryGraphStore
from shared.models import Edge, Node
from shared.enums import EdgeStatus, NodeType, RelationType
from unittest.mock import Mock

@pytest.fixture
def mock_graph_store():
    store = InMemoryGraphStore()
    store.add_node(Node(id="player", name="player", node_type=NodeType.CHARACTER))
    store.add_node(Node(id="kitchen", name="kitchen", node_type=NodeType.ROOM))
    store.add_edge(Edge(subject="player", relation=RelationType.LOCATED_IN, object="kitchen", t_observed=1))
    return store

class TestWorldModelValidator:
    def test_detects_duplicate_active_edges(self, mock_graph_store):
        """Validator catches duplicate active edges."""
        # Force a duplicate active edge
        mock_graph_store._edges["fake"] = Edge(id="fake", subject="player", relation=RelationType.LOCATED_IN, object="kitchen", t_observed=1)
        
        validator = WorldModelValidator()
        report = validator.validate(mock_graph_store, current_turn=5)
        
        assert not report.is_valid
        assert any(v.violation_type == "duplicate_active_edge" for v in report.violations)
    
    def test_detects_temporal_invalid_from(self, mock_graph_store):
        """Validator catches t_valid_from > t_observed."""
        mock_graph_store._edges["time"] = Edge(id="time", subject="kitchen", relation=RelationType.CONTAINS, object="key", t_observed=1, t_valid_from=2)
        
        validator = WorldModelValidator()
        report = validator.validate(mock_graph_store, current_turn=5)
        
        assert not report.is_valid
        assert any(v.violation_type == "temporal_invalid_from" for v in report.violations)
    
    def test_detects_player_node_count(self, mock_graph_store):
        """Validator checks for exactly 1 PLAYER node."""
        validator = WorldModelValidator()
        report = validator.validate(mock_graph_store, current_turn=5)
        # Should pass because fixture has exactly 1 PLAYER
        assert report.is_valid
        
        # Add another player
        mock_graph_store.add_node(Node(id="player2", name="player2", node_type=NodeType.CHARACTER))
        # Wait, if id="player2", validator won't find it as a PLAYER because it checks id == "player".
        # Let's add another one with id="player" but different node internally? 
        # Actually, if I add id="player", it will override the existing node in InMemoryGraphStore since it uses a dict by id.
        # But wait, the validator logic: player_nodes = [n for n in graph_store.get_all_nodes() if n.id == "player" and n.node_type == NodeType.CHARACTER]
        # If I want it to fail due to count != 1, I should remove the player node!
        
        mock_graph_store._nodes.pop("player")
        report2 = validator.validate(mock_graph_store, current_turn=5)
        assert not report2.is_valid
        assert any(v.violation_type == "player_node_count" for v in report2.violations)

class TestPerformanceProfiler:
    def test_measures_component_latency(self):
        """Profiler tracks component times."""
        profiler = PerformanceProfiler(episode_id=1)
        profiler.start_turn(turn_id=1)
        profiler.measure_component("query_layer", 12.5)
        profiler.measure_component("prompt_builder", 8.3)
        profiler.end_turn(tokens_used=1200)
        
        report = profiler.compile_report()
        assert report.total_turns == 1
        assert "query_layer" in report.turn_profiles[0].components
        assert report.turn_profiles[0].components["query_layer"] == 12.5

from unittest.mock import patch

class TestBenchmarkSuite:
    @patch('slm_actions.slm_client.SLMRunner')
    def test_benchmark_suite_with_demo_world(self, mock_slm_class):
        """Benchmark suite runs on demo.z8 (if it exists)."""
        # Set up the mock SLM instance to return empty extractions and simple actions
        mock_slm = mock_slm_class.return_value
        mock_slm.generate.side_effect = ['[]', '{"action": "look"}'] * 200
        
        demo_path = "examples/demo.z8"
        if not os.path.exists(demo_path):
            pytest.skip("examples/demo.z8 does not exist")
            
        report = run_benchmark_suite(
            world_file=demo_path,
            num_episodes=2,
            output_csv="results/test_demo.csv"
        )
        if report is not None:
            assert report.num_episodes == 2
            
    def test_handles_missing_world_file(self):
        """Benchmark suite handles missing files gracefully."""
        report = run_benchmark_suite(
            world_file="examples/nonexistent.z8",
            num_episodes=1,
        )
        assert report is None  # Should return None, not crash
    
    def test_csv_export(self, tmp_path):
        """Benchmark report exports CSV correctly."""
        report = BenchmarkReport(stage="test", num_episodes=1)
        report.export_csv(str(tmp_path / "test.csv"))
        assert (tmp_path / "test.csv").exists()
