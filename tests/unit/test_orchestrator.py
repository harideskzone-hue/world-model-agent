import pytest
from unittest.mock import Mock, MagicMock
from orchestrator.config import OrchestratorConfig
from orchestrator.orchestrator import Orchestrator, BudgetExceeded
from shared.models import Observation, WorkingMemory
from slm_actions.environment_validator import ActionResult

@pytest.fixture
def mock_env():
    env = Mock()
    obs = Observation(
        feedback="Welcome", description="A room", inventory="nothing",
        location="room", objective="win the game", won=False, lost=False
    )
    env.reset.return_value = obs
    env.step.return_value = obs
    env.get_available_commands.return_value = ["go north", "take key"]
    return env

@pytest.fixture
def mock_slm():
    slm = Mock()
    slm.generate.return_value = '{"action": "go north"}'
    return slm

class TestOrchestratorSetup:
    def test_initializes_all_modules(self, mock_env, mock_slm):
        config = OrchestratorConfig()
        orch = Orchestrator(mock_env, mock_slm, config)
        
        assert orch.env == mock_env
        assert orch.slm == mock_slm
        assert orch.graph_store is not None
        assert orch.updater is not None
        assert orch.preprocessor is not None
        assert orch.json_parser is not None
        assert orch.schema_validator is not None
        assert orch.confidence_assigner is not None
        assert orch.query_layer is not None
        assert orch.prompt_builder is not None
        assert orch.slm_extractor is not None
        assert orch.slm_agent is not None
        assert orch.action_parser is not None
        assert orch.env_validator is not None

    def test_loads_config_correctly(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=50, max_tokens_per_episode=1000)
        orch = Orchestrator(mock_env, mock_slm, config)
        assert orch.config.max_turns == 50
        assert orch.config.max_tokens_per_episode == 1000


class TestOrchestratorTurn:
    def test_single_turn_flow(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=1)
        orch = Orchestrator(mock_env, mock_slm, config)
        
        # We need mock slm to return empty facts then valid action
        mock_slm.generate.side_effect = ['[]', '{"action": "go north"}']
        
        result = orch.run_episode()
        
        assert result.steps_taken == 1
        assert mock_env.step.call_count == 1
        assert orch.metrics.total_valid_actions == 1
        assert orch.metrics.total_invalid_actions == 0

    def test_invalid_action_recorded(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=1)
        orch = Orchestrator(mock_env, mock_slm, config)
        
        mock_slm.generate.side_effect = ['[]', '{"action": "go west"}']
        
        result = orch.run_episode()
        
        assert result.steps_taken == 1
        assert mock_env.step.call_count == 0  # Should skip step if invalid
        assert orch.metrics.total_valid_actions == 0
        assert orch.metrics.total_invalid_actions == 1

    def test_tokens_tracked_correctly(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=1)
        orch = Orchestrator(mock_env, mock_slm, config)
        
        mock_slm.generate.side_effect = ['[]', '{"action": "go north"}']
        
        result = orch.run_episode()
        
        assert result.total_tokens_used > 0
        assert orch.metrics.total_tokens_used > 0


class TestOrchestratorTerminal:
    def test_terminates_on_win(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=10)
        orch = Orchestrator(mock_env, mock_slm, config)
        
        mock_slm.generate.return_value = '{"action": "go north"}'
        
        obs_win = Observation(
            feedback="", description="", inventory="", location="", objective="", won=True, lost=False
        )
        mock_env.step.return_value = obs_win
        
        result = orch.run_episode()
        
        assert result.win is True
        assert result.steps_taken == 1

    def test_terminates_on_loss(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=10)
        orch = Orchestrator(mock_env, mock_slm, config)
        
        mock_slm.generate.return_value = '{"action": "go north"}'
        
        obs_loss = Observation(
            feedback="", description="", inventory="", location="", objective="", won=False, lost=True
        )
        mock_env.step.return_value = obs_loss
        
        result = orch.run_episode()
        
        assert result.lost is True
        assert result.steps_taken == 1

    def test_terminates_on_max_turns(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=5)
        orch = Orchestrator(mock_env, mock_slm, config)
        
        mock_slm.generate.return_value = '{"action": "go north"}'
        
        result = orch.run_episode()
        
        assert result.steps_taken == 5
        assert result.win is False
        assert result.lost is False

    def test_terminates_on_budget_exceeded(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_tokens_per_episode=1) # Too small
        orch = Orchestrator(mock_env, mock_slm, config)
        
        mock_slm.generate.return_value = '{"action": "go north"}'
        
        result = orch.run_episode()
        
        assert result.steps_taken == 0 # fails before completing turn 1
        assert orch.metrics.is_over_budget(config.max_tokens_per_episode) is True

    def test_terminates_on_stuck_invalid(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=20, max_invalid_actions_in_row=3)
        orch = Orchestrator(mock_env, mock_slm, config)
        
        # Always output invalid action
        mock_slm.generate.return_value = '{"action": "invalid"}'
        
        result = orch.run_episode()
        
        assert result.steps_taken == 3
        assert result.invalid_action_count == 3
        assert orch.metrics.is_stuck(config.max_invalid_actions_in_row) is True


class TestMetrics:
    def test_metrics_calculated_correctly(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=2)
        orch = Orchestrator(mock_env, mock_slm, config)
        
        mock_slm.generate.side_effect = [
            '[]', '{"action": "invalid"}',
            '[]', '{"action": "go north"}'
        ]
        
        result = orch.run_episode()
        
        assert result.invalid_action_count == 1
        assert result.invalid_action_rate == 0.5  # 1 invalid / 2 turns
        assert result.avg_tokens_per_turn == result.total_tokens_used / 2


class TestSerialization:
    def test_world_model_exported_as_json(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=1, export_world_model=True)
        orch = Orchestrator(mock_env, mock_slm, config)
        
        mock_slm.generate.return_value = '{"action": "go north"}'
        result = orch.run_episode()
        
        assert isinstance(result.world_model_json, str)
        assert len(result.world_model_json) > 0

    def test_export_disabled(self, mock_env, mock_slm):
        config = OrchestratorConfig(max_turns=1, export_world_model=False)
        orch = Orchestrator(mock_env, mock_slm, config)
        
        mock_slm.generate.return_value = '{"action": "go north"}'
        result = orch.run_episode()
        
        assert result.world_model_json == ""
