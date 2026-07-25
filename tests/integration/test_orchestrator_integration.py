import pytest
from unittest.mock import Mock
from orchestrator.config import OrchestratorConfig
from orchestrator.orchestrator import Orchestrator
from shared.models import Observation

class MockEnvironment:
    def __init__(self):
        self.turns = 0
        self.won = False
        self.lost = False
        
    def reset(self):
        self.turns = 0
        return Observation(
            feedback="Welcome to integration world.",
            description="You are in a square room.",
            inventory="You are empty-handed.",
            location="square room",
            objective="find the key",
            won=False,
            lost=False
        )
        
    def step(self, action):
        self.turns += 1
        if action == "take key":
            self.won = True
        return Observation(
            feedback=f"You executed: {action}",
            description="You are in a square room.",
            inventory="You hold a key." if self.won else "You are empty-handed.",
            location="square room",
            objective="find the key",
            won=self.won,
            lost=self.lost
        )
        
    def get_available_commands(self):
        return ["look", "inventory", "take key", "go north"]


@pytest.fixture
def integration_env():
    return MockEnvironment()

@pytest.fixture
def mock_slm_integration():
    slm = Mock()
    # It takes 1 extractor call and 1 agent call per turn
    # Turn 1: extract [], decide "look"
    # Turn 2: extract [], decide "take key"
    slm.generate.side_effect = [
        '[]', '{"action": "look"}',
        '[]', '{"action": "take key"}'
    ]
    return slm


class TestOrchestratorIntegration:
    def test_end_to_end_simple_world(self, integration_env, mock_slm_integration):
        config = OrchestratorConfig(max_turns=10)
        orch = Orchestrator(integration_env, mock_slm_integration, config)
        
        result = orch.run_episode()
        
        assert result.win is True
        assert result.steps_taken == 2
        assert result.invalid_action_count == 0
        assert result.total_tokens_used > 0
        
    def test_handles_empty_observations(self, integration_env, mock_slm_integration):
        config = OrchestratorConfig(max_turns=10)
        orch = Orchestrator(integration_env, mock_slm_integration, config)
        
        # Override step to return empty strings
        integration_env.step = Mock(return_value=Observation(
            feedback="", description="", inventory="", location="", objective="", won=False, lost=True
        ))
        
        result = orch.run_episode()
        
        assert result.lost is True
        assert result.steps_taken == 1
