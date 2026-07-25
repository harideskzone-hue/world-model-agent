import sys
import os
# Ensure the root directory is in sys.path when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from orchestrator.orchestrator import Orchestrator
from orchestrator.config import OrchestratorConfig
from orchestrator.metrics import EpisodeResult
from evaluation.world_model_validator import WorldModelValidator
from evaluation.performance_profiler import PerformanceProfiler

class BenchmarkReport:
    """Aggregated results from N episodes."""
    def __init__(self, stage: str, num_episodes: int):
        self.stage = stage
        self.num_episodes = num_episodes
        self.results: List[EpisodeResult] = []
        self.consistency_reports = []
        self.performance_profiles = []
        self.timestamp = datetime.now().isoformat()
    
    def add_episode(self, result: EpisodeResult, consistency, profile):
        self.results.append(result)
        self.consistency_reports.append(consistency)
        self.performance_profiles.append(profile)
    
    def calculate_metrics(self) -> dict:
        """Calculate aggregate metrics per EVALUATION_PLAN.md."""
        if not self.results:
            return {}
        
        wins = sum(1 for r in self.results if r.win)
        losses = sum(1 for r in self.results if r.lost)
        valid_steps = [r.steps_taken for r in self.results if r.win]
        
        return {
            "win_rate": (wins / self.num_episodes) * 100,
            "loss_rate": (losses / self.num_episodes) * 100,
            "avg_steps_on_win": sum(valid_steps) / len(valid_steps) if valid_steps else 0,
            "avg_tokens_per_turn": sum(r.avg_tokens_per_turn for r in self.results) / self.num_episodes if self.num_episodes > 0 else 0,
            "invalid_action_rate": sum(r.invalid_action_rate for r in self.results) / self.num_episodes if self.num_episodes > 0 else 0,
            "world_model_consistency": sum(1 for r in self.consistency_reports if r.is_valid) / self.num_episodes * 100 if self.num_episodes > 0 else 0,
        }
    
    def export_csv(self, output_path: str):
        """Export episode-level results to CSV."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode_id', 'win', 'lost', 'steps', 'total_tokens',
                'avg_tokens_per_turn', 'invalid_actions', 'invalid_action_rate',
                'consistency_valid'
            ])
            
            for i, result in enumerate(self.results):
                consistency_valid = self.consistency_reports[i].is_valid if i < len(self.consistency_reports) else False
                writer.writerow([
                    i + 1,
                    result.win,
                    result.lost,
                    result.steps_taken,
                    result.total_tokens_used,
                    round(result.avg_tokens_per_turn, 2),
                    result.invalid_action_count,
                    round(result.invalid_action_rate, 4),
                    consistency_valid,
                ])
    
    def print_summary(self):
        """Print human-readable summary table."""
        metrics = self.calculate_metrics()
        if not metrics:
            print("No metrics available (0 episodes completed).")
            return
            
        print(f"\n{'='*70}")
        print(f"{self.stage} - {self.num_episodes} episodes")
        print(f"{'='*70}")
        print(f"Metric                          Result      Status")
        print(f"{'-'*70}")
        print(f"Win Rate                        {metrics.get('win_rate', 0):.1f}%")
        print(f"Loss Rate                       {metrics.get('loss_rate', 0):.1f}%")
        print(f"Avg Steps (wins)                {metrics.get('avg_steps_on_win', 0):.1f}")
        print(f"Avg Tokens/Turn                 {metrics.get('avg_tokens_per_turn', 0):.0f}")
        print(f"Invalid Action Rate             {metrics.get('invalid_action_rate', 0):.2f}%")
        print(f"World Model Consistency         {metrics.get('world_model_consistency', 0):.1f}%")
        print(f"{'='*70}\n")

class EnvWrapper:
    """Simple wrapper to match our Orchestrator interface."""
    def __init__(self, raw_env):
        self.raw_env = raw_env
        self.last_obs = None
        self.last_info = None
        self.turns = 0
        
    def reset(self):
        obs, info = self.raw_env.reset()
        self.last_obs = obs
        self.last_info = info
        self.turns = 0
        return self._build_observation()
        
    def step(self, action):
        obs, score, done, info = self.raw_env.step(action)
        self.last_obs = obs
        self.last_info = info
        self.turns += 1
        return self._build_observation()
        
    def get_available_commands(self):
        if not self.last_info or 'admissible_commands' not in self.last_info:
            return []
        return self.last_info['admissible_commands']
        
    def _build_observation(self):
        from shared.models import Observation
        info = self.last_info or {}
        desc = info.get('description') or ''
        if not desc and isinstance(self.last_obs, str):
            desc = self.last_obs
        return Observation(
            feedback=info.get('feedback') or (self.last_obs if isinstance(self.last_obs, str) else ''),
            description=desc,
            inventory=info.get('inventory') or '',
            location=info.get('location') or '',
            objective=info.get('objective') or '',
            won=info.get('won', False),
            lost=info.get('lost', False),
            turn_id=self.turns
        )

def run_benchmark_suite(
    world_file: str,
    num_episodes: int = 10,
    output_csv: str = None,
) -> BenchmarkReport:
    """
    Core benchmark engine.
    
    Args:
        world_file: Path to .z8 TextWorld file
        num_episodes: Number of episodes to run
        output_csv: Optional path to save CSV results
    
    Returns:
        BenchmarkReport with aggregated metrics
    """
    
    # Validate world file exists
    if not Path(world_file).exists():
        print(f"❌ ERROR: World file not found: {world_file}")
        print(f"ℹ️  Skipping benchmark. Stage files will be provided at evaluation time.")
        return None
    
    # Initialize environment
    try:
        from textworld import EnvInfos
        import textworld.gym
        # Register and make env
        request_infos = EnvInfos(
            feedback=True, description=True, inventory=True, 
            location=True, objective=True, won=True, lost=True, 
            admissible_commands=True,
        )
        env_id = textworld.gym.register_game(world_file, request_infos)
        raw_env = textworld.gym.make(env_id)
    except ImportError:
        print(f"❌ ERROR: textworld not installed. Cannot load {world_file}")
        return None
    except Exception as e:
        print(f"❌ ERROR: Failed to load world {world_file}: {e}")
        return None
        
    env = EnvWrapper(raw_env)
    
    # Initialize SLM client
    try:
        from slm_actions.slm_client import SLMRunner
        slm = SLMRunner()
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize SLM: {e}")
        return None
    
    # Run episodes
    report = BenchmarkReport(world_file, num_episodes)
    
    for episode_id in range(num_episodes):
        print(f"Running episode {episode_id + 1}/{num_episodes}...")
        
        try:
            # Configure orchestrator
            config = OrchestratorConfig(
                max_turns=100,
                max_tokens_per_episode=5000,
                max_invalid_actions_in_row=5,
                export_world_model=True,
            )
            
            # Run orchestrator
            orchestrator = Orchestrator(env, slm, config)
            orchestrator.episode_id = episode_id + 1
            result = orchestrator.run_episode()
            
            # Validate world model
            validator = WorldModelValidator()
            consistency = validator.validate(orchestrator.graph_store, result.steps_taken)
            
            # Profile performance
            profile = PerformanceProfiler(episode_id + 1)
            # We don't hook it properly in this script, but we instantiate it.
            
            # Add to report
            report.add_episode(result, consistency, profile)
            
        except Exception as e:
            print(f"  ⚠️  Episode {episode_id + 1} failed: {e}")
            continue
    
    # Export results
    if output_csv and report.results:
        report.export_csv(output_csv)
        print(f"✅ CSV exported to {output_csv}")
    
    # Print summary
    report.print_summary()
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument("world_file", help="Path to .z8 world file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--output", type=str, help="Output CSV path")
    
    args = parser.parse_args()
    run_benchmark_suite(args.world_file, args.episodes, args.output)
