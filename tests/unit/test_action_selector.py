# tests/unit/test_action_selector.py
# ============================================================================
# Unit tests for the SLM ActionSelector.
# Tests without actual SLM (heuristic mode) and prompt parsing.
# ============================================================================

import pytest

from shared.models import Observation, ContextSlice, Edge
from shared.enums import RelationType, ExtractionMethod
from slm.action_selector import ActionSelector
from slm.action_grammar import ActionGrammar


@pytest.fixture
def context_slice():
    return ContextSlice(
        formatted_text=(
            "LOCATION: kitchen\n"
            "ROOM CONTENTS: brass key (locked), apple\n"
            "CONNECTED ROOMS: north → garden\n"
            "INVENTORY: sword\n"
            "OBJECTIVE: Find and take the brass key"
        ),
        included_facts=[],
        excluded_count=0,
        total_tokens_estimate=30,
    )


@pytest.fixture
def observation():
    return Observation(
        feedback="You are in the kitchen. You see a brass key on the table.",
        description="",
        inventory="You are carrying: a sword",
        location="kitchen",
        objective="Find and take the brass key.",
        turn_id=5,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ACTION GRAMMAR
# ═══════════════════════════════════════════════════════════════════════════

class TestActionGrammar:
    def test_grammar_description_not_empty(self):
        grammar = ActionGrammar()
        desc = grammar.get_grammar_description()
        assert len(desc) > 50
        assert "take" in desc
        assert "go" in desc

    def test_simple_action_list(self):
        grammar = ActionGrammar()
        simple = grammar.get_simple_action_list()
        assert "look" in simple
        assert "take" in simple


# ═══════════════════════════════════════════════════════════════════════════
# ACTION SELECTOR (NO SLM — HEURISTIC MODE)
# ═══════════════════════════════════════════════════════════════════════════

class TestActionSelectorNoSLM:
    def test_returns_valid_action(self, context_slice, observation):
        selector = ActionSelector(slm=None)
        decision = selector.select_action(context_slice, observation)
        assert decision.action_text != ""
        assert isinstance(decision.action_text, str)

    def test_heuristic_fallback_is_look(self, context_slice, observation):
        selector = ActionSelector(slm=None)
        decision = selector.select_action(context_slice, observation)
        assert decision.action_text == "look"

    def test_latency_recorded(self, context_slice, observation):
        selector = ActionSelector(slm=None)
        decision = selector.select_action(context_slice, observation)
        assert decision.latency_ms >= 0


# ═══════════════════════════════════════════════════════════════════════════
# ACTION PARSING
# ═══════════════════════════════════════════════════════════════════════════

class TestActionParsing:
    def test_parse_bare_action(self):
        selector = ActionSelector(slm=None)
        result = selector._parse_action("take the brass key")
        assert result == "take the brass key"

    def test_parse_with_action_prefix(self):
        selector = ActionSelector(slm=None)
        result = selector._parse_action("ACTION: go north")
        assert result == "go north"

    def test_parse_with_quotes(self):
        selector = ActionSelector(slm=None)
        result = selector._parse_action('"take apple"')
        assert result == "take apple"

    def test_parse_multiline_takes_first(self):
        selector = ActionSelector(slm=None)
        result = selector._parse_action("go north\nThis is because...")
        assert result == "go north"

    def test_parse_direction_normalized(self):
        selector = ActionSelector(slm=None)
        result = selector._parse_action("north")
        assert result == "go north"

    def test_parse_invalid_returns_none(self):
        selector = ActionSelector(slm=None)
        result = selector._parse_action("I think we should probably")
        assert result is None

    def test_parse_empty_returns_none(self):
        selector = ActionSelector(slm=None)
        result = selector._parse_action("")
        assert result is None
