import pytest
from unittest.mock import Mock
from slm_actions.slm_extractor import SLMExtractor
from slm_actions.slm_agent import SLMAgent
from slm_actions.action_parser import ActionParser
from slm_actions.environment_validator import EnvironmentValidator, ActionResult

@pytest.fixture
def mock_slm():
    """Mock SLMRunner."""
    slm = Mock()
    slm.generate = Mock(return_value='')
    return slm

# ═══════════════════════════════════════════════════════════════════════════
# SLM EXTRACTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSLMExtractor:
    def test_extract_calls_generate_with_temp_zero(self, mock_slm):
        extractor = SLMExtractor(mock_slm)
        mock_slm.generate.return_value = '[]'
        
        extractor.extract("You are in a kitchen.")
        
        mock_slm.generate.assert_called_once()
        assert mock_slm.generate.call_args[1]['temperature'] == 0.0

    def test_extract_includes_observation_in_prompt(self, mock_slm):
        extractor = SLMExtractor(mock_slm)
        observation = "You are in a kitchen."
        mock_slm.generate.return_value = '[]'
        
        extractor.extract(observation)
        
        prompt = mock_slm.generate.call_args[1]['prompt']
        assert observation in prompt
        
    def test_extract_includes_schema_in_prompt(self, mock_slm):
        extractor = SLMExtractor(mock_slm)
        extractor.extract("You are in a kitchen.")
        prompt = mock_slm.generate.call_args[1]['prompt']
        assert "SCHEMA:" in prompt
        assert "subject_type: strictly one of [ROOM, OBJECT, CHARACTER]" in prompt
        
    def test_extract_includes_rules_in_prompt(self, mock_slm):
        extractor = SLMExtractor(mock_slm)
        extractor.extract("You are in a kitchen.")
        prompt = mock_slm.generate.call_args[1]['prompt']
        assert "RULES:" in prompt
        assert "Output ONLY a valid JSON array" in prompt

    def test_extract_returns_raw_response(self, mock_slm):
        extractor = SLMExtractor(mock_slm)
        expected_response = '{"some": "json"}'
        mock_slm.generate.return_value = expected_response
        result = extractor.extract("Test")
        assert result == expected_response
        
    def test_extract_handles_slm_error(self, mock_slm):
        extractor = SLMExtractor(mock_slm)
        mock_slm.generate.side_effect = RuntimeError("SLM down")
        with pytest.raises(RuntimeError):
            extractor.extract("Test")
            
    def test_extract_prompt_ends_with_facts(self, mock_slm):
        extractor = SLMExtractor(mock_slm)
        extractor.extract("Test")
        prompt = mock_slm.generate.call_args[1]['prompt']
        assert prompt.endswith("FACTS:")
        
    def test_extract_passes_observation_exactly(self, mock_slm):
        extractor = SLMExtractor(mock_slm)
        obs = 'Test "quotes" and {braces}'
        extractor.extract(obs)
        prompt = mock_slm.generate.call_args[1]['prompt']
        assert f'TEXT: "{obs}"' in prompt

# ═══════════════════════════════════════════════════════════════════════════
# SLM AGENT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSLMAgent:
    def test_decide_calls_generate_with_temp_zero(self, mock_slm):
        agent = SLMAgent(mock_slm)
        mock_slm.generate.return_value = '{"action": "go north"}'
        agent.decide("formatted prompt")
        mock_slm.generate.assert_called_once()
        assert mock_slm.generate.call_args[1]['temperature'] == 0.0

    def test_decide_returns_raw_response(self, mock_slm):
        agent = SLMAgent(mock_slm)
        expected_response = '{"action": "go north"}'
        mock_slm.generate.return_value = expected_response
        result = agent.decide("formatted prompt")
        assert result == expected_response
        
    def test_decide_preserves_prompt_exactly(self, mock_slm):
        agent = SLMAgent(mock_slm)
        prompt = "test prompt 123"
        agent.decide(prompt)
        assert mock_slm.generate.call_args[1]['prompt'] == prompt
        
    def test_decide_handles_slm_error(self, mock_slm):
        agent = SLMAgent(mock_slm)
        mock_slm.generate.side_effect = RuntimeError("timeout")
        with pytest.raises(RuntimeError):
            agent.decide("prompt")
            
    def test_decide_handles_empty_prompt(self, mock_slm):
        agent = SLMAgent(mock_slm)
        agent.decide("")
        assert mock_slm.generate.call_args[1]['prompt'] == ""
        
    def test_decide_handles_large_prompt(self, mock_slm):
        agent = SLMAgent(mock_slm)
        prompt = "A" * 10000
        agent.decide(prompt)
        assert mock_slm.generate.call_args[1]['prompt'] == prompt
        
    def test_decide_handles_malformed_response(self, mock_slm):
        agent = SLMAgent(mock_slm)
        mock_slm.generate.return_value = "not json"
        result = agent.decide("prompt")
        assert result == "not json"

    def test_decide_multiple_calls(self, mock_slm):
        agent = SLMAgent(mock_slm)
        agent.decide("p1")
        agent.decide("p2")
        assert mock_slm.generate.call_count == 2

# ═══════════════════════════════════════════════════════════════════════════
# ACTION PARSER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestActionParser:
    def test_parse_valid_json_returns_action(self):
        parser = ActionParser()
        result = parser.parse('{"action": "go north"}')
        assert result == "go north"

    def test_parse_missing_action_returns_empty(self):
        parser = ActionParser()
        result = parser.parse('{"foo": "bar"}')
        assert result == ""

    def test_parse_malformed_json_returns_empty(self):
        parser = ActionParser()
        result = parser.parse('not json')
        assert result == ""
        
    def test_parse_action_not_string(self):
        parser = ActionParser()
        result = parser.parse('{"action": 123}')
        assert result == ""
        
    def test_parse_whitespace_trimmed(self):
        parser = ActionParser()
        result = parser.parse('{"action": "  go north  "}')
        assert result == "go north"
        
    def test_parse_empty_string(self):
        parser = ActionParser()
        result = parser.parse('')
        assert result == ""
        
    def test_parse_empty_object(self):
        parser = ActionParser()
        result = parser.parse('{}')
        assert result == ""
        
    def test_parse_extra_fields_ignored(self):
        parser = ActionParser()
        result = parser.parse('{"action": "take key", "reasoning": "I need it"}')
        assert result == "take key"
        
    def test_parse_list_returns_empty(self):
        parser = ActionParser()
        result = parser.parse('["take key"]')
        assert result == ""
        
    def test_parse_null_returns_empty(self):
        parser = ActionParser()
        result = parser.parse('null')
        assert result == ""
        
    def test_parse_boolean_returns_empty(self):
        parser = ActionParser()
        result = parser.parse('true')
        assert result == ""
        
    def test_parse_json_with_newlines(self):
        parser = ActionParser()
        result = parser.parse('{\n"action": \n"take key"\n}')
        assert result == "take key"

# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VALIDATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvironmentValidator:
    def test_validate_empty_action_invalid(self):
        validator = EnvironmentValidator()
        result = validator.validate("", ["go north"])
        assert result.valid == False
        assert result.action == ""
        assert "empty" in result.error.lower()

    def test_validate_action_in_list_is_valid(self):
        validator = EnvironmentValidator()
        result = validator.validate("go north", ["go north", "take key"])
        assert result.valid == True
        assert result.error is None
        assert result.action == "go north"

    def test_validate_action_not_in_list_invalid(self):
        validator = EnvironmentValidator()
        result = validator.validate("go west", ["go north", "take key"])
        assert result.valid == False
        assert "not in admissible" in result.error.lower()
        assert result.action == "go west"
        
    def test_validate_case_sensitive_matching(self):
        validator = EnvironmentValidator()
        result = validator.validate("Go North", ["go north"])
        assert result.valid == False
        
    def test_validate_no_list_provided_accepts_nonempty(self):
        validator = EnvironmentValidator()
        result = validator.validate("go north")
        assert result.valid == True
        assert result.error is None
        
    def test_validate_no_list_provided_rejects_empty(self):
        validator = EnvironmentValidator()
        result = validator.validate("   ")
        assert result.valid == False
        
    def test_validate_whitespace_stripped(self):
        validator = EnvironmentValidator()
        result = validator.validate("  go north  ", ["go north"])
        assert result.valid == True
        assert result.action == "go north"
        
    def test_validate_preserves_original_action(self):
        validator = EnvironmentValidator()
        result = validator.validate("invalid", ["go north"])
        assert result.valid == False
        assert result.action == "invalid"
        
    def test_validate_exact_match_required(self):
        validator = EnvironmentValidator()
        result = validator.validate("go", ["go north"])
        assert result.valid == False
        
    def test_validate_substring_not_matched(self):
        validator = EnvironmentValidator()
        result = validator.validate("go north east", ["go north"])
        assert result.valid == False
        
    def test_validate_list_with_empty_string(self):
        validator = EnvironmentValidator()
        result = validator.validate("", [""])
        assert result.valid == False # Fails the empty check first
        
    def test_validate_returns_correct_result_type(self):
        validator = EnvironmentValidator()
        result = validator.validate("test")
        assert isinstance(result, ActionResult)
