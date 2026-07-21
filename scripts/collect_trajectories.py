#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
from pathlib import Path

from env.textworld_wrapper import TextWorldWrapper
from extractor.text_extractor import TextExtractor
from updater.updater import Updater
from world_model.graph_store import InMemoryGraphStore
from orchestrator.working_memory import WorkingMemoryBuilder
from query_layer.query_layer import QueryLayer
from slm.prompt_builder import PromptBuilder

def extract_state(obs, graph, extractor, updater, wm_builder, query_layer):
    """Helper to extract semantic state without advancing the main graph state if needed."""
    wm_builder.set_objective(obs.objective)
    wm_builder.add_observation(obs.feedback or obs.description)
    wm = wm_builder.build()
    
    candidates = extractor.extract(obs, wm)
    report = updater.update(candidates, turn_id=obs.turn_id)
    
    # Rebuild wm after graph update to get the latest facts
    wm = wm_builder.build()
    context_slice = query_layer.retrieve(wm, obs.turn_id)
    return PromptBuilder.build_prompt(context_slice, wm, obs, "")

def collect_from_game(game_path, metadata, output_file):
    env = TextWorldWrapper(game_path, eval_mode=True)
    obs = env.reset()
    
    # Initialize World Model components
    graph = InMemoryGraphStore()
    extractor = TextExtractor(slm_runner=None)
    updater = Updater(graph)
    wm_builder = WorkingMemoryBuilder(graph)
    query_layer = QueryLayer(graph)
    
    walkthrough = metadata.get("walkthrough", [])
    if not walkthrough:
        print(f"Skipping {game_path} — no walkthrough found.")
        env.close()
        return
        
    trajectories = []
    
    # Bootstrap turn 0
    semantic_state = extract_state(obs, graph, extractor, updater, wm_builder, query_layer)
    
    for step_num, optimal_action in enumerate(walkthrough):
        # We step the environment
        next_obs, reward, done = env.step(optimal_action)
        
        # Extract next state
        next_semantic_state = extract_state(next_obs, graph, extractor, updater, wm_builder, query_layer)
        
        trajectory_step = {
            "episode_id": metadata.get("game_id"),
            "step_number": step_num,
            "observation": obs.description,
            "semantic_world_state": semantic_state,
            "sub_goal": optimal_action, # As an expert policy, the optimal action resolves the immediate sub-goal
            "optimal_action": optimal_action,
            "reward": reward,
            "next_semantic_state": next_semantic_state,
            "done": done
        }
        trajectories.append(trajectory_step)
        
        # Advance state
        obs = next_obs
        semantic_state = next_semantic_state

    env.close()
    
    # Append trajectories to output file
    with open(output_file, "a") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")

def collect_trajectories(corpus_dir: str, output_file: str):
    print(f"Collecting trajectories from {corpus_dir} into {output_file}...")
    
    json_files = glob.glob(os.path.join(corpus_dir, "metadata_*.json"))
    json_files = [j for j in json_files if "distribution_summary" not in j]
    
    # Ensure output file is empty initially
    if os.path.exists(output_file):
        os.remove(output_file)
        
    for j_idx, json_file in enumerate(json_files):
        with open(json_file, "r") as f:
            metadata = json.load(f)
            
        game_path = json_file.replace("metadata_", "").replace(".json", ".z8")
        if not os.path.exists(game_path):
            continue
            
        collect_from_game(game_path, metadata, output_file)
        
        if (j_idx + 1) % 10 == 0:
            print(f"[{j_idx+1}/{len(json_files)}] Collected trajectories.")
            
    print(f"Done. Saved to {output_file}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect semantic state trajectories using expert policy.")
    parser.add_argument("--corpus-dir", type=str, default="data/training_corpus", help="Directory containing generated games.")
    parser.add_argument("--output-file", type=str, default="data/trajectories.jsonl", help="Output JSONL file.")
    args = parser.parse_args()
    
    # Create data dir if not exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    collect_trajectories(args.corpus_dir, args.output_file)
