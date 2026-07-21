#!/usr/bin/env python3
import os
import argparse
import random
from pathlib import Path
import textworld

# Define difficulty tiers for training data
TIER_CONFIGS = {
    "easy":   {"nb_rooms": 2, "nb_objects": 2, "quest_length": 2},
    "medium": {"nb_rooms": 5, "nb_objects": 5, "quest_length": 4},
    "hard":   {"nb_rooms": 10, "nb_objects": 10, "quest_length": 6},
    "expert": {"nb_rooms": 15, "nb_objects": 15, "quest_length": 8},
}

def generate_corpus(num_games: int, output_dir: str):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    difficulties = list(TIER_CONFIGS.keys())
    
    stats = {
        "total_games": 0,
        "tier_counts": {t: 0 for t in difficulties},
        "total_rooms": 0,
        "total_objects": 0,
        "total_quest_length": 0
    }
    
    print(f"Generating {num_games} TextWorld games into {output_dir}...")
    for i in range(num_games):
        seed = random.randint(1000, 999999)
        tier = random.choice(difficulties)
        cfg = TIER_CONFIGS[tier]
        
        options = textworld.GameOptions()
        options.path = str(out_path / f"game_{tier}_s{seed}.z8")
        options.seeds = seed
        options.nb_rooms = cfg["nb_rooms"]
        options.nb_objects = cfg["nb_objects"]
        options.quest_length = cfg["quest_length"]
        
        # Add diverse quest templates and random topographies implicitly handled by TextWorld's procedural generator when not fixing the grammar.
        
        try:
            game = textworld.generator.make_game(options)
            compiled_path = textworld.generator.compile_game(game, options)
            
            walkthrough = []
            if hasattr(game, 'quests') and game.quests:
                walkthrough = list(game.quests[0].commands)
                
            # Save metadata separately from TextWorld's internal JSON
            metadata = {
                "game_id": f"game_{tier}_s{seed}",
                "difficulty": tier,
                "rooms": cfg["nb_rooms"],
                "objects": cfg["nb_objects"],
                "quest_length": cfg["quest_length"],
                "seed": seed,
                "walkthrough": walkthrough
            }
            import json
            with open(out_path / f"metadata_game_{tier}_s{seed}.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            # Aggregate stats
            stats["total_games"] += 1
            stats["tier_counts"][tier] += 1
            stats["total_rooms"] += cfg["nb_rooms"]
            stats["total_objects"] += cfg["nb_objects"]
            stats["total_quest_length"] += cfg["quest_length"]
                
            if i % 10 == 0:
                print(f"[{i}/{num_games}] Generated {tier} game (seed={seed}) -> {compiled_path}")
        except Exception as e:
            print(f"Failed to generate game {i} (seed={seed}): {e}")

    # Compute averages and save distribution summary
    if stats["total_games"] > 0:
        summary = {
            "total_games": stats["total_games"],
            "difficulty_distribution": {t: f"{(c / stats['total_games']) * 100:.1f}%" for t, c in stats["tier_counts"].items()},
            "avg_rooms": round(stats["total_rooms"] / stats["total_games"], 2),
            "avg_objects": round(stats["total_objects"] / stats["total_games"], 2),
            "avg_quest_length": round(stats["total_quest_length"] / stats["total_games"], 2)
        }
        with open(out_path / "distribution_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\nDataset Generation Complete.")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diverse TextWorld corpus for Phase 2 Semantic Policy Learning.")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to generate.")
    parser.add_argument("--output-dir", type=str, default="data/training_corpus", help="Output directory for generated games.")
    args = parser.parse_args()
    
    generate_corpus(args.num_games, args.output_dir)
