import pytest
from shared.models import ContextSlice, WorkingMemory, Observation, Edge
from shared.enums import RelationType
from slm.prompt_builder import PromptBuilder

def test_prompt_builder_basic():
    # Setup ContextSlice with edges
    edges = [
        Edge(subject="apple", relation=RelationType.HOLDS, object="apple"),
        Edge(subject="kitchen", relation=RelationType.CONNECTS_TO, object="pantry", direction="north"),
        Edge(subject="chest", relation=RelationType.HAS_STATE, object="open"),
        Edge(subject="chest", relation=RelationType.CONTAINS, object="coin")
    ]
    context_slice = ContextSlice(formatted_text="", included_facts=edges)
    
    # Setup WorkingMemory
    wm = WorkingMemory(
        current_room="kitchen",
        current_sub_goal="take apple",
        failed_actions=["go west", "open door"]
    )
    
    # Setup Observation
    obs = Observation(
        turn_id=1,
        description="You are in a kitchen.",
        feedback="You took the apple.",
        inventory="an apple",
        location="kitchen",
        objective="Find the apple.",
        score=0, max_score=10, won=False, lost=False
    )
    
    grammar = "Valid commands: go north, take <item>"
    
    prompt = PromptBuilder.build_prompt(context_slice, wm, obs, grammar)
    
    # Assertions
    assert "CURRENT WORLD STATE" in prompt
    assert "[GOAL]\ntake apple" in prompt
    assert "[CURRENT LOCATION]\nroom_1" in prompt
    assert "room_1" in prompt  # abstract location
    assert "object_1" in prompt  # apple
    assert "room_2" in prompt  # pantry
    assert "container_1" in prompt  # chest
    assert "object_2" in prompt  # coin
    assert '"go west"' in prompt
    assert '"open door"' in prompt
    assert grammar in prompt

def test_build_retry_prompt():
    grammar = "Valid commands: go north, take <item>"
    invalid_action = "I think I should go north"
    prompt = PromptBuilder.build_retry_prompt(invalid_action, grammar)
    
    assert invalid_action in prompt
    assert grammar in prompt
    assert "exactly one valid TextWorld command" in prompt
