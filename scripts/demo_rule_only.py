#!/usr/bin/env python3
# scripts/demo_rule_only.py
# ============================================================================
# Demonstrates the full agent pipeline in rule-only mode (no SLM, no TextWorld).
# Uses synthetic observations to show the data flow end-to-end.
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from shared.models import Observation, WorkingMemory
from shared.enums import RelationType
from world_model.graph_store import InMemoryGraphStore
from extractor.text_extractor import TextExtractor
from updater.updater import Updater
from query_layer.query_layer import QueryLayer
from slm.action_selector import ActionSelector
from orchestrator.working_memory import WorkingMemoryBuilder
from world_model.persistence.serializer import save_human_readable


def main():
    print("=" * 70)
    print("  TextWorld AI Agent — Rule-Only Pipeline Demo")
    print("  (No SLM, No TextWorld — Synthetic Observations)")
    print("=" * 70)
    print()

    # Initialize system
    graph = InMemoryGraphStore()
    extractor = TextExtractor(slm_runner=None)
    updater = Updater(graph)
    query_layer = QueryLayer(graph)
    action_selector = ActionSelector(slm=None)
    wm_builder = WorkingMemoryBuilder(graph)

    # Synthetic game observations simulating a TextWorld episode
    observations = [
        Observation(
            feedback="",
            description="You are in the kitchen. You see a brass key on the counter. "
                        "There is an apple here. You can see a door to the north.",
            inventory="You are carrying nothing.",
            location="kitchen",
            objective="Find the brass key and unlock the garden shed.",
            turn_id=0,
        ),
        Observation(
            feedback="You take the brass key.",
            description="",
            inventory="You are carrying: a brass key",
            location="kitchen",
            objective="Find the brass key and unlock the garden shed.",
            turn_id=1,
        ),
        Observation(
            feedback="You go north.",
            description="You are in the garden. There is a shed here.",
            inventory="You are carrying: a brass key",
            location="garden",
            objective="Find the brass key and unlock the garden shed.",
            turn_id=2,
        ),
        Observation(
            feedback="You unlock the shed.",
            description="The shed is now unlocked.",
            inventory="You are carrying: a brass key",
            location="garden",
            objective="Find the brass key and unlock the garden shed.",
            turn_id=3,
        ),
    ]

    for obs in observations:
        print(f"{'─' * 70}")
        print(f"  TURN {obs.turn_id}")
        print(f"{'─' * 70}")

        if obs.feedback:
            print(f"  Feedback: {obs.feedback}")
        if obs.description:
            print(f"  Description: {obs.description}")

        # 1. Build working memory
        wm_builder.set_objective(obs.objective)
        wm_builder.add_observation(obs.feedback or obs.description)
        wm = wm_builder.build()
        wm.current_room = obs.location  # Ensure we know where we are

        # 2. Extract facts
        candidates = extractor.extract(obs, wm)
        print(f"\n  📋 Extracted {len(candidates)} candidate facts:")
        for fact in candidates:
            print(f"     ({fact.subject}, {fact.relation.value}, {fact.object}) "
                  f"[conf={fact.confidence:.2f}, method={fact.extraction_method.value}]")

        # 3. Update world model
        report = updater.update(candidates, turn_id=obs.turn_id)
        print(f"\n  📊 Update: +{report.expanded} expanded, "
              f"↻{report.corroborated} corroborated, "
              f"⟳{report.revised} revised")

        # 4. Query for context
        wm = wm_builder.build()  # Refresh after update
        context = query_layer.retrieve(wm, current_turn=obs.turn_id)
        print(f"\n  🔍 Context (~{context.total_tokens_estimate} tokens, "
              f"{len(context.included_facts)} facts):")
        for line in context.formatted_text.split("\n"):
            print(f"     {line}")

        # 5. Select action (heuristic)
        decision = action_selector.select_action(context, obs)
        print(f"\n  🎮 Action: '{decision.action_text}' "
              f"(latency={decision.latency_ms:.1f}ms)")

        print()

    # Final summary
    print(f"{'═' * 70}")
    print(f"  FINAL WORLD MODEL")
    print(f"{'═' * 70}")
    stats = graph.get_stats()
    print(f"  Nodes: {stats.total_nodes}")
    print(f"  Active edges: {stats.active_edges}")
    print(f"  Superseded edges: {stats.superseded_edges}")
    print(f"  Storage: {stats.storage_bytes} bytes")
    print(f"  Avg confidence: {stats.avg_confidence:.3f}")

    # Print all active edges
    print(f"\n  Active beliefs:")
    for edge in graph.get_all_active_edges():
        state = f" [{edge.direction}]" if edge.direction else ""
        print(f"    ({edge.subject}, {edge.relation.value}, {edge.object}){state} "
              f"[conf={edge.confidence:.2f}, turn={edge.source_turn_id}]")

    # Save human-readable snapshot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(output_dir, exist_ok=True)
    save_human_readable(graph, os.path.join(output_dir, "demo_world_model.txt"))
    print(f"\n  💾 Human-readable snapshot saved to logs/demo_world_model.txt")
    print()


if __name__ == "__main__":
    main()
