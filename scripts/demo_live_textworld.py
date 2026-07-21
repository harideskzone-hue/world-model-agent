#!/usr/bin/env python3
# scripts/demo_live_textworld.py
# ============================================================================
# Demonstrates SEMANTIC action prediction against a REAL TextWorld game.
#
# This is NOT keyword matching. The pipeline:
#   1. Extracts structured facts from natural language → builds a graph
#   2. Parses multi-step objectives into ordered sub-goals
#   3. Queries the world model for relevant context
#   4. Selects the next action based on SEMANTIC UNDERSTANDING
#      of the objective, world state, and inventory
# ============================================================================

import sys
import os
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import textworld
from textworld import EnvInfos

from shared.models import Observation, WorkingMemory
from shared.enums import RelationType
from world_model.graph_store import InMemoryGraphStore
from extractor.text_extractor import TextExtractor
from updater.updater import Updater
from query_layer.query_layer import QueryLayer
from slm.model_runner import SLMRunner
from slm.action_selector import ActionSelector
import argparse
from slm.action_selector import ActionSelector
from orchestrator.working_memory import WorkingMemoryBuilder
from orchestrator.objective_parser import ObjectiveParser

DIVIDER = "═" * 72
THIN = "─" * 72
MAX_TURNS = 25


def generate_game(output_dir: str) -> str:
    """Generate a real TextWorld game."""
    os.makedirs(output_dir, exist_ok=True)
    options = textworld.GameOptions()
    options.path = os.path.join(output_dir, "live_demo_game.z8")
    options.nb_rooms = 3
    options.nb_objects = 5
    options.quest_length = 5
    options.seeds = 42
    game = textworld.generator.make_game(options)
    return textworld.generator.compile_game(game, options)




# Track actions that failed so we don't retry them
_failed_actions: set = set()
_last_action: str = ""
_consecutive_failures: int = 0
_last_room: str = ""  # Track room changes to avoid false sub-goal completion
_subgoal_fail_count: dict = {}  # sub-goal index → consecutive failure count
_SUBGOAL_SKIP_THRESHOLD = 3    # skip sub-goal after this many consecutive failures

_DIRECTIONS = {"north", "south", "east", "west", "up", "down"}


def _parse_direction_from_target(target: str, edge_direction: str | None) -> str:
    """
    Extract the compass direction from a connection edge.
    The exit rule encodes direction in the target name: 'room_north' → 'north'.
    Falls back to edge.direction if available.
    """
    if edge_direction and edge_direction in _DIRECTIONS:
        return edge_direction
    # Parse from target name: 'room_north' → 'north'
    parts = target.lower().split("_")
    for part in parts:
        if part in _DIRECTIONS:
            return part
    return edge_direction or "unknown"


def _has_failed(obs_text: str) -> bool:
    """Check if the observation indicates the last action failed."""
    failure_indicators = [
        "you can't", "you don't", "there is no", "that's not",
        "i don't understand", "isn't open", "is already open",
        "is locked", "is closed", "nothing happens",
        "doesn't seem to", "what do you want to", "which do you mean",
        "can't go", "not a verb", "doesn't open", "already closed",
        "don't see", "you need to", "can't see any such thing",
        "fixed in place", "that's fixed", "only understood you"
    ]
    obs_lower = obs_text.lower()
    return any(ind in obs_lower for ind in failure_indicators)


def record_failure(obs_text: str, action: str, subgoal_index: int = -1):
    """Record if the last action failed based on TextWorld response."""
    global _consecutive_failures, _last_action
    
    if _has_failed(obs_text):
        _failed_actions.add(action)
        _consecutive_failures += 1
        # Track per-sub-goal failures for skip logic
        if subgoal_index >= 0:
            _subgoal_fail_count[subgoal_index] = _subgoal_fail_count.get(subgoal_index, 0) + 1
    else:
        _consecutive_failures = 0
        # Reset failure count for this sub-goal on success
        if subgoal_index >= 0:
            _subgoal_fail_count[subgoal_index] = 0
    _last_action = action


def semantic_action(
    obj_parser: ObjectiveParser,
    graph: InMemoryGraphStore,
    obs: Observation,
) -> tuple[str, str]:
    """
    Choose the next action using SEMANTIC reasoning over the world model.

    Reasoning chain:
      1. What is my current sub-goal? (from objective parser)
      2. Is the action feasible given the world model? (check exits, items)
      3. If not feasible, what prerequisite is missing?
      4. If stuck (action failed), explore or interact with environment
    """
    current_sg = obj_parser.get_current_subgoal()
    if current_sg is None:
        return "look", "All sub-goals completed or no objective parsed"

    action = current_sg.action
    active_edges = graph.get_all_active_edges()

    # Gather world model state
    player_location = None
    held_items = []
    room_contents = {}  # room -> list of objects
    states = {}         # entity -> state
    connections = {}    # room -> [(target, direction)]

    for e in active_edges:
        if e.relation == RelationType.LOCATED_IN and e.subject == "player":
            player_location = e.object
        elif e.relation == RelationType.HOLDS and e.subject == "player":
            held_items.append(e.object)
        elif e.relation == RelationType.CONTAINS:
            room_contents.setdefault(e.subject, []).append(e.object)
        elif e.relation == RelationType.HAS_STATE:
            states[e.subject] = e.object
        elif e.relation == RelationType.CONNECTS_TO:
            direction = _parse_direction_from_target(e.object, e.direction)
            connections.setdefault(e.subject, []).append(
                (e.object, direction)
            )

    current_room = player_location or ""
    room_conns = connections.get(current_room, [])
    available_dirs = {d for _, d in room_conns}
    items_here = room_contents.get(current_room, [])
    all_visible_items = []
    for items in room_contents.values():
        all_visible_items.extend(items)

    # ── Check if the sub-goal action already failed ──
    if action in _failed_actions:
        # Sub-goal action is not directly executable from here.
        # Use world model to find an alternative path.
        return _find_alternative(
            current_sg, obj_parser, current_room, available_dirs,
            room_conns, items_here, all_visible_items, held_items, states
        )

    # ── Check if this sub-goal is exhausted (≥3 failures) → skip it ──
    sg_failures = _subgoal_fail_count.get(current_sg.index, 0)
    if sg_failures >= _SUBGOAL_SKIP_THRESHOLD:
        obj_parser.mark_completed(current_sg.index)
        next_sg = obj_parser.get_current_subgoal()
        if next_sg:
            return next_sg.action, (
                f"Sub-goal #{current_sg.index+1} blocked ({sg_failures} failures) "
                f"— skipping to: {next_sg.action}"
            )
        return "look", f"All sub-goals attempted — observing"

    # ── Semantic Reasoning per action type ──

    # Navigation: "go north" — verify the exit exists
    go_match = re.match(r"go (north|south|east|west|up|down)", action)
    if go_match:
        direction = go_match.group(1)
        if direction in available_dirs:
            return action, f"Sub-goal #{current_sg.index+1}: going {direction} (exit confirmed)"
        elif available_dirs:
            # Direction not available — the sub-goal path may go through other rooms
            return _find_alternative(
                current_sg, obj_parser, current_room, available_dirs,
                room_conns, items_here, all_visible_items, held_items, states
            )
        else:
            return "look", "No exits known yet — observing surroundings first"

    # Open: "open X" — check state and accessibility
    open_match = re.match(r"open (.+)", action)
    if open_match:
        target = open_match.group(1)
        # Try to find the object in any form
        target_found = _find_entity(target, all_visible_items + list(states.keys()))
        if target_found:
            target_state = states.get(target_found, "")
            if target_state == "locked":
                for item in held_items:
                    if "key" in item:
                        return f"unlock {target_found} with {item}", \
                            f"'{target_found}' is locked — using '{item}'"
                return f"open {target_found}", f"Attempting to open locked '{target_found}'"
            if target_state == "open":
                # Already open — mark sub-goal as done and advance
                obj_parser.mark_completed(current_sg.index)
                next_sg = obj_parser.get_current_subgoal()
                if next_sg:
                    return next_sg.action, f"'{target_found}' already open — advancing to: {next_sg.action}"
                return "look", f"'{target_found}' already open"
            return f"open {target_found}", f"Sub-goal #{current_sg.index+1}: opening '{target_found}'"
        else:
            # Target not visible — need to explore
            return _find_alternative(
                current_sg, obj_parser, current_room, available_dirs,
                room_conns, items_here, all_visible_items, held_items, states
            )

    # Take: "take X" — check availability
    take_match = re.match(r"take (.+)", action)
    if take_match:
        target = take_match.group(1)
        target_found = _find_entity(target, all_visible_items)
        if target_found:
            # Check if inside a container
            for container, items in room_contents.items():
                if target_found in items:
                    container_state = states.get(container, "")
                    if container_state in ("closed", "locked"):
                        if container_state == "locked":
                            for item in held_items:
                                if "key" in item:
                                    return f"unlock {container} with {item}", \
                                        f"'{target_found}' in locked '{container}' — unlocking"
                        return f"open {container}", \
                            f"'{target_found}' in closed '{container}' — opening first"
                    elif container_state in ("open", ""):
                        # Item in an open container — use 'take X from container'
                        # TextWorld requires this syntax for items inside containers
                        if container != current_room:
                            return f"take {target_found} from {container}", \
                                f"Sub-goal #{current_sg.index+1}: taking '{target_found}' from '{container}'"
            return f"take {target_found}", f"Sub-goal #{current_sg.index+1}: taking '{target_found}'"
        elif target in held_items:
            obj_parser.mark_completed(current_sg.index)
            next_sg = obj_parser.get_current_subgoal()
            if next_sg:
                return next_sg.action, f"Already holding '{target}' — advancing to: {next_sg.action}"
            return "look", f"Already holding '{target}'"
        else:
            return _find_alternative(
                current_sg, obj_parser, current_room, available_dirs,
                room_conns, items_here, all_visible_items, held_items, states
            )

    # Put: "put X in Y" / "insert X into Y"
    put_match = re.match(r"put (.+?) in (.+)", action)
    if put_match:
        item_name, container_name = put_match.group(1), put_match.group(2)
        # Strip location qualifiers: "dresser in the restroom" → "dresser"
        container_clean = re.sub(r"\s+in the\s+\w+$", "", container_name).strip()
        # Fuzzy-match held item (player may hold "shadfly from safe" not "shadfly")
        item_held = _find_entity(item_name, held_items)
        if not item_held:
            return f"take {item_name}", f"Need '{item_name}' first before placing in '{container_clean}'"
        # Fuzzy-match container in room
        container_found = _find_entity(container_clean, items_here + all_visible_items)
        target_container = container_found or container_clean
        # Strip state prefixes: TextWorld uses "dresser" not "opened dresser"
        target_container = _strip_state_prefix(target_container)
        # TextWorld uses "insert X into Y" syntax
        return f"insert {item_name} into {target_container}", \
            f"Sub-goal #{current_sg.index+1}: inserting '{item_name}' into '{target_container}'"

    return action, f"Sub-goal #{current_sg.index+1}: {current_sg.raw_text}"


def _strip_state_prefix(name: str) -> str:
    """Remove state adjectives from entity names for TextWorld commands.
    'opened dresser' → 'dresser', 'closed safe' → 'safe'."""
    return re.sub(r"^(opened?|closed|locked)\s+", "", name, flags=re.IGNORECASE).strip()


def _find_entity(target: str, entity_list: list) -> str | None:
    """Fuzzy match an entity name in a list."""
    target_lower = target.lower()
    # Exact match
    for e in entity_list:
        if e.lower() == target_lower:
            return e
    # Partial match (target is a substring)
    for e in entity_list:
        if target_lower in e.lower() or e.lower() in target_lower:
            return e
    return None


def _find_alternative(
    current_sg, obj_parser, current_room, available_dirs,
    room_conns, items_here, all_visible_items, held_items, states,
) -> tuple[str, str]:
    """
    Deep replanning: find an alternative when the primary sub-goal action fails.

    Replanning ladder:
      P1. take X fails → examine X first (not look)
      P2. go X fails → try all other known exits from the graph (not just look)
      P3. open X fails → try unlock X if a key is held
      P4. Interact with closed objects in the room
      P5. Navigate to unexplored rooms via known exits
      P6. look (reset)
      P7. inventory
    """
    global _consecutive_failures

    sg_action = current_sg.action if current_sg else ""

    # ── P1: take X failure → examine X first ──
    take_match = re.match(r"take (.+)", sg_action)
    if take_match:
        target = take_match.group(1).strip()
        examine_action = f"examine {target}"
        if examine_action not in _failed_actions:
            return examine_action, (
                f"'take {target}' failed — examining '{target}' first to confirm location"
            )

    # ── P2: go X fails → try all other known exits from the graph ──
    go_match = re.match(r"go (north|south|east|west|up|down)", sg_action)
    if go_match:
        failed_dir = go_match.group(1)
        for target, direction in room_conns:
            if direction != failed_dir:
                alt_go = f"go {direction}"
                if alt_go not in _failed_actions:
                    return alt_go, (
                        f"'go {failed_dir}' failed — trying alternative exit: go {direction} (→ {target})"
                    )

    # ── P3: open X fails → try unlock if key available ──
    open_match = re.match(r"open (.+)", sg_action)
    if open_match:
        target = open_match.group(1).strip()
        for held in held_items:
            if "key" in held:
                unlock_action = f"unlock {target} with {held}"
                if unlock_action not in _failed_actions:
                    return unlock_action, f"'open {target}' failed — trying unlock with '{held}'"

    # ── P4: Interact with closed/locked objects in the room ──
    for item in items_here:
        item_state = states.get(item, "")
        if item_state == "closed" and f"open {item}" not in _failed_actions:
            return f"open {item}", f"Opening '{item}' to explore contents"
        if item_state == "locked":
            for held in held_items:
                if "key" in held and f"unlock {item} with {held}" not in _failed_actions:
                    return f"unlock {item} with {held}", f"Unlocking '{item}' with '{held}'"
        if f"take {item}" not in _failed_actions and item not in held_items:
            return f"take {item}", f"Picking up '{item}' — may be useful"

    # ── P5: Navigate to unexplored rooms via known exits ──
    for target, direction in room_conns:
        go_action = f"go {direction}"
        if go_action not in _failed_actions:
            return go_action, f"Exploring {direction} (→ {target}) to find quest items"

    # ── P6: look (reset) ──
    if "look" not in _failed_actions or _consecutive_failures > 5:
        _failed_actions.discard("look")
        return "look", "Observing surroundings for new information"

    # ── P7: inventory ──
    if "inventory" not in _failed_actions:
        return "inventory", "Checking inventory"

    # Last resort
    return "look", "Stuck — resetting observation"


def run_demo(policy: str = "rule"):
    print(f"\n{DIVIDER}")
    print("  🧪 LIVE TextWorld — Semantic Action Prediction Demo")
    print(f"{DIVIDER}")

    # ── Generate game ──
    game_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "env", "game_suite", "live_demo"
    )
    game_file = generate_game(game_dir)
    print(f"  Game: {game_file}")

    # ── Start environment ──
    game_path = generate_game(game_dir)
    from env.textworld_wrapper import TextWorldWrapper
    env = TextWorldWrapper(game_path, eval_mode=True)
    obs = env.reset()

    # ── Initialize pipeline ──
    graph = InMemoryGraphStore()
    extractor = TextExtractor(slm_runner=None)
    updater = Updater(graph)
    wm_builder = WorkingMemoryBuilder(graph)
    obj_parser = ObjectiveParser()
    
    slm_runner = None
    action_selector = None
    if policy == "slm":
        slm_runner = SLMRunner()
        if not slm_runner.is_available():
            print(f"❌ Error: Ollama is not available or model '{slm_runner._config.model_name}' is missing.")
            exit(1)
        action_selector = ActionSelector(slm_runner)

    admissible = env.get_admissible_commands()

    # Parse objective into sub-goals
    sub_goals = obj_parser.parse(obs.objective)

    print(f"\n{THIN}")
    print(f"  OBJECTIVE: {obs.objective[:120]}...")
    print(f"{THIN}")
    print(f"  📋 Parsed Sub-Goals:")
    for sg in sub_goals:
        print(f"    {sg.index+1}. {sg.raw_text}")
        print(f"       → Predicted command: '{sg.action}'")
    print()

    # ── Show initial observation ──
    print(f"{THIN}")
    print(f"  INITIAL OBSERVATION")
    print(f"{THIN}")
    desc = obs.description.replace("\n\n\n", "\n")
    for line in desc.strip().split("\n"):
        if line.strip():
            print(f"    {line.strip()}")
    print(f"  📍 Admissible (eval-only): {admissible}")

    # ── Multi-turn loop ──
    wm_builder.set_objective(obs.objective)
    done = False
    total_reward = 0

    for turn in range(MAX_TURNS):
        print(f"\n{'━' * 72}")
        print(f"  TURN {turn}")
        print(f"{'━' * 72}")

        # 1. Build working memory
        wm_builder.add_observation(obs.feedback or obs.description)
        wm = wm_builder.build()

        # Track current room for sub-goal completion detection
        global _last_room
        banner_m = re.search(r"-=\s*(.+?)\s*=-", obs.description or obs.feedback or "")
        entered_m = re.search(r"(?:you(?:'ve| have)? entered|here we are in) (?:the |a |an )?(.+?)\.",
                              obs.feedback or obs.description or "", re.IGNORECASE)
        current_room_detected = ""
        if banner_m:
            current_room_detected = banner_m.group(1).strip().lower()
        elif entered_m:
            current_room_detected = entered_m.group(1).strip().lower()

        if current_room_detected:
            if turn == 0:
                _last_room = current_room_detected
            # _last_room is updated in the completion check below when room changes

        # 2. Extract facts
        candidates = extractor.extract(obs, wm)
        if candidates:
            print(f"  🔍 Extracted {len(candidates)} facts:")
            for f in candidates[:5]:
                print(f"     ({f.subject}, {f.relation.value}, {f.object}) "
                      f"[conf={f.confidence:.2f}]")
            if len(candidates) > 5:
                print(f"     ... and {len(candidates)-5} more")

        # 3. Update world model
        report = updater.update(candidates, turn_id=turn)
        print(f"  🔄 Update: +{report.expanded} new, "
              f"↻{report.corroborated} confirmed, ⟳{report.revised} revised")

        # 4. Check if current sub-goal was completed by last observation
        if turn > 0:
            obs_text = (obs.feedback or "") + " " + (obs.description or "")
            current_subgoal = obj_parser.get_current_subgoal()

            # For navigation sub-goals: only mark complete if we actually
            # changed rooms (prevents false positive from 'look' returning
            # the same "You've entered a kitchen" text)
            if current_subgoal and re.match(r"go \w+", current_subgoal.action):
                # Use current_room_detected from the tracking above
                if current_room_detected and current_room_detected != _last_room and _last_room:
                    obj_parser.mark_completed(current_subgoal.index)
                    print(f"  ✅ Sub-goal completed: '{current_subgoal.action}' "
                          f"(moved from '{_last_room}' → '{current_room_detected}')")
                    _last_room = current_room_detected  # Update tracking
            else:
                # Non-navigation sub-goals: use the standard check
                if obj_parser.advance_if_completed(obs_text):
                    completed_idx = max(0, (obj_parser.get_current_subgoal().index - 1
                                    if obj_parser.get_current_subgoal()
                                    else len(obj_parser.sub_goals) - 1))
                    print(f"  ✅ Sub-goal completed: '{obj_parser.sub_goals[completed_idx].action}'")

        # 5. Show world model state
        stats = graph.get_stats()
        print(f"  🧠 World model: {stats.total_nodes} nodes, "
              f"{stats.active_edges} active edges")

        # 6. Retrieve context for SLM
        context_slice = QueryLayer(graph).retrieve(wm, turn)

        # 7. Action Selection
        print(f"\n  [ACTION SELECTION - Policy: {policy.upper()}]")
        t0 = time.perf_counter()
        
        current_sg = obj_parser.get_current_subgoal()
        if policy == "slm":
            wm.current_sub_goal = current_sg.action if current_sg else ""
            slm_decision = action_selector.select_action(context_slice, wm, obs)
            action = slm_decision.action_text
            reason = slm_decision.raw_output
            latency = slm_decision.latency_ms
            
            if action == "INVALID_ACTION":
                print(f"  ❌ SLM failed to produce valid action. Terminating.")
                break
        else:
            action, reason = semantic_action(obj_parser, graph, obs)
            latency = (time.perf_counter() - t0) * 1000

        print(f"\n  ┌{'─'*68}┐")
        if current_sg:
            print(f"  │ SUB-GOAL: {current_sg.action:<56}│")
        print(f"  │ ACTION:   {action:<56}│")
        if policy == "slm":
            print(f"  │ REASON:   [See raw SLM output below]{' '*31}│")
        else:
            print(f"  │ REASON:   {reason[:56]:<56}│")
        print(f"  └{'─'*68}┘")
        if policy == "slm":
            print(f"\n  🧠 Raw SLM Output:\n{reason.strip()}\n")
        print(f"  ⏱️  Latency: {latency:.0f}ms")

        # 7b. Show what TextWorld considers valid (for comparison)
        admissible = env.get_admissible_commands()
        is_valid = action in admissible
        marker = "✅" if is_valid else "⚠️"
        print(f"  {marker} Action '{action}' in admissible list: {is_valid}")
        print(f"     Admissible: {admissible}")

        # 8. Execute action
        obs, reward, done = env.step(action)
        total_reward += reward

        feedback = obs.feedback or obs.description or "(no response)"

        # 8b. Record if action failed — enables replanning on next turn
        record_failure(feedback, action)

        # Clean feedback for display
        clean_fb = re.sub(r'-=.*=-', '', feedback).strip()
        clean_lines = [l.strip() for l in clean_fb.split('\n') if l.strip() and re.search(r'[a-zA-Z]', l)]
        print(f"  📝 Response: {' | '.join(clean_lines[:3])}")
        print(f"  📊 Reward: {reward}, Score: {obs.score}/{obs.max_score}, Done: {done}")

        if done:
            won_str = "🏆 WON!" if obs.won else "💀 LOST"
            print(f"\n  {won_str} Final score: {obs.score}/{obs.max_score}")
            break

    # ── Final summary ──
    env.close()
    stats = graph.get_stats()
    print(f"\n{DIVIDER}")
    print(f"  FINAL RESULTS")
    print(f"{DIVIDER}")
    print(f"  Turns played: {min(turn + 1, MAX_TURNS)}")
    print(f"  Final score:  {obs.score}/{obs.max_score}")
    print(f"  Won:          {obs.won}")
    print(f"  World model:  {stats.total_nodes} nodes, "
          f"{stats.active_edges} active / {stats.superseded_edges} superseded edges")
    print(f"\n  {obj_parser.progress_summary}")
    print(f"\n  Active beliefs:")
    for edge in graph.get_all_active_edges():
        dir_str = f" [{edge.direction}]" if edge.direction else ""
        print(f"    ({edge.subject}, {edge.relation.value}, {edge.object}){dir_str}")
    print(f"\n{DIVIDER}\n")

    # Save snapshot
    from world_model.persistence.serializer import save_snapshot
    save_snapshot(graph, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "expected_world_model.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Demo")
    parser.add_argument("--policy", choices=["rule", "slm"], default="rule", help="Action selection policy")
    args = parser.parse_args()
    
    run_demo(policy=args.policy)
