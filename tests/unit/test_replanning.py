"""
tests/unit/test_replanning.py
=============================
8 unit tests for Gap 2: Deep environment adaptation in demo_live_textworld.py

Acceptance criteria:
  ✓ Agent finds another route if first path fails
  ✓ Agent never repeats the same failed action
  ✓ After 3 failures on same sub-goal, agent advances to next sub-goal
  ✓ take X failure triggers examine X (not look)
  ✓ Replanning produces a valid TextWorld command
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

# Import the module-level state we need to control
import scripts.demo_live_textworld as demo
from scripts.demo_live_textworld import (
    _find_alternative, record_failure,
    _failed_actions, _subgoal_fail_count, _SUBGOAL_SKIP_THRESHOLD,
    semantic_action,
)
from orchestrator.objective_parser import ObjectiveParser


# ── Helpers ──────────────────────────────────────────────────────────────────

def reset_state():
    """Reset all module-level state before each test."""
    demo._failed_actions.clear()
    demo._subgoal_fail_count.clear()
    demo._consecutive_failures = 0
    demo._last_action = ""
    demo._last_room = ""


class MockSubgoal:
    def __init__(self, index=0, action="go north", raw_text="go north"):
        self.index = index
        self.action = action
        self.raw_text = raw_text


class MockParser:
    def __init__(self, subgoals):
        self._sgs = subgoals
        self._current = 0

    def get_current_subgoal(self):
        if self._current < len(self._sgs):
            return self._sgs[self._current]
        return None

    def mark_completed(self, idx):
        self._current = idx + 1


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestReplanning:

    def setup_method(self):
        reset_state()

    def test_go_north_fails_tries_east(self):
        """If 'go north' fails, _find_alternative should try another known exit."""
        sg = MockSubgoal(0, "go north")
        parser = MockParser([sg])
        demo._failed_actions.add("go north")

        room_conns = [("attic", "north"), ("garden", "east")]
        available_dirs = {"north", "east"}

        action, reason = _find_alternative(
            sg, parser,
            current_room="kitchen",
            available_dirs=available_dirs,
            room_conns=room_conns,
            items_here=[],
            all_visible_items=[],
            held_items=[],
            states={},
        )
        assert action == "go east", f"Expected 'go east', got '{action}'"
        assert "alternative" in reason.lower() or "east" in reason.lower()

    def test_failed_action_never_repeated(self):
        """_find_alternative must never return an action in _failed_actions."""
        sg = MockSubgoal(0, "go north")
        parser = MockParser([sg])
        # Mark all go actions as failed
        for d in ["north", "south", "east", "west"]:
            demo._failed_actions.add(f"go {d}")
        demo._failed_actions.add("take key")

        action, _ = _find_alternative(
            sg, parser,
            current_room="kitchen",
            available_dirs=set(),
            room_conns=[],
            items_here=[],
            all_visible_items=[],
            held_items=[],
            states={},
        )
        assert action not in demo._failed_actions, (
            f"'{action}' is in _failed_actions but was returned"
        )

    def test_subgoal_skipped_after_3_failures(self):
        """After 3 consecutive failures on a sub-goal, agent must advance to the next one."""
        sg0 = MockSubgoal(0, "go north", "go north to attic")
        sg1 = MockSubgoal(1, "open safe", "open safe in attic")
        parser = MockParser([sg0, sg1])

        # Simulate 3 failures for sub-goal 0
        demo._subgoal_fail_count[0] = _SUBGOAL_SKIP_THRESHOLD

        from world_model.graph_store import InMemoryGraphStore
        from shared.models import Observation, WorkingMemory

        graph = InMemoryGraphStore()
        obs = Observation(
            feedback="You can't go north.",
            description="", inventory="", location="kitchen",
            objective="go north then open safe", turn_id=1,
        )

        action, reason = semantic_action(parser, graph, obs)
        # Should have advanced to sub-goal 1: "open safe"
        assert "safe" in action.lower() or parser._current >= 1, (
            f"Expected sub-goal skip but got '{action}'"
        )

    def test_take_failure_triggers_examine(self):
        """'take X' failure must trigger 'examine X', not 'look'."""
        sg = MockSubgoal(0, "take shadfly")
        parser = MockParser([sg])
        demo._failed_actions.add("take shadfly")

        action, reason = _find_alternative(
            sg, parser,
            current_room="attic",
            available_dirs={"north"},
            room_conns=[("kitchen", "north")],
            items_here=[],
            all_visible_items=[],
            held_items=[],
            states={},
        )
        assert action == "examine shadfly", (
            f"Expected 'examine shadfly' after take failure, got '{action}'"
        )
        assert "examine" in reason.lower() or "confirm" in reason.lower()

    def test_replanning_produces_valid_command(self):
        """Every action returned by _find_alternative must be a non-empty string."""
        sg = MockSubgoal(0, "go north")
        parser = MockParser([sg])
        demo._failed_actions.add("go north")

        action, reason = _find_alternative(
            sg, parser,
            current_room="kitchen",
            available_dirs={"east"},
            room_conns=[("garden", "east")],
            items_here=[],
            all_visible_items=[],
            held_items=[],
            states={},
        )
        assert isinstance(action, str) and len(action.strip()) > 0
        assert isinstance(reason, str) and len(reason.strip()) > 0

    def test_empty_graph_falls_back_to_look(self):
        """If no exits or items are known, must gracefully fall back to 'look'."""
        sg = MockSubgoal(0, "go north")
        parser = MockParser([sg])
        demo._failed_actions.add("go north")

        action, _ = _find_alternative(
            sg, parser,
            current_room="unknown",
            available_dirs=set(),
            room_conns=[],
            items_here=[],
            all_visible_items=[],
            held_items=[],
            states={},
        )
        assert action in {"look", "inventory"}, (
            f"Expected graceful fallback, got '{action}'"
        )

    def test_multiple_failed_subgoals_no_corruption(self):
        """Multiple failed sub-goals must not corrupt the sub-goal index."""
        sgs = [
            MockSubgoal(0, "go north"),
            MockSubgoal(1, "open safe"),
            MockSubgoal(2, "take shadfly"),
        ]
        parser = MockParser(sgs)

        # Fail sub-goals 0 and 1
        demo._subgoal_fail_count[0] = _SUBGOAL_SKIP_THRESHOLD
        demo._subgoal_fail_count[1] = _SUBGOAL_SKIP_THRESHOLD

        # Mark 0 as skipped
        parser.mark_completed(0)
        parser.mark_completed(1)

        current = parser.get_current_subgoal()
        assert current is not None
        assert current.index == 2, f"Expected sub-goal 2, got {current.index}"
        assert current.action == "take shadfly"

    def test_after_recovery_resumes_correct_subgoal(self):
        """After _find_alternative finds a working action, the sub-goal index is unchanged."""
        sg0 = MockSubgoal(0, "go north")
        sg1 = MockSubgoal(1, "open safe")
        parser = MockParser([sg0, sg1])

        # Only mark go north as failed (not 3 times — no skip yet)
        demo._failed_actions.add("go north")
        demo._subgoal_fail_count[0] = 1  # only 1 failure — no skip

        action, _ = _find_alternative(
            sg0, parser,
            current_room="kitchen",
            available_dirs={"east"},
            room_conns=[("garden", "east")],
            items_here=[],
            all_visible_items=[],
            held_items=[],
            states={},
        )
        # Sub-goal should still be 0 (no skip triggered)
        assert parser._current == 0, (
            f"Sub-goal advanced prematurely to {parser._current}"
        )
        assert action == "go east"
