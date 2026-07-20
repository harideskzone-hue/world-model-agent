#!/usr/bin/env python3
"""
evaluation/run_benchmark.py
===========================
Track 1 benchmark: runs the agent on 5 seeds × 3 difficulty tiers = 15 games.

Pipeline per game:
  1. Generate Game      → textworld.generator.make_game(options)
  2. Run Agent          → DecisionLoop.run_episode(env)
  3. Collect TurnLogs   → stored in turn_log_store[]
  4. Collect Metrics    → metrics.py
  5. Write Report       → evaluation/reports/latest.json
  6. Compare Baseline   → full-history agent (raw text, no graph)

Usage:
  python3 evaluation/run_benchmark.py              # full run (15 games)
  python3 evaluation/run_benchmark.py --dry-run    # 1 game, mock env
  python3 evaluation/run_benchmark.py --tier 1     # only tier 1 games
"""
from __future__ import annotations

import sys, os, json, csv, time, argparse, logging, traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import textworld
from textworld import EnvInfos

from shared.models import Observation, EpisodeResult, TurnLog, UpdateReport, SLMDecision, ContextSlice, CandidateFact
from shared.enums import ExtractionMethod, ExtractionType
from world_model.graph_store import InMemoryGraphStore
from extractor.text_extractor import TextExtractor
from updater.updater import Updater
from query_layer.query_layer import QueryLayer
from orchestrator.working_memory import WorkingMemoryBuilder
from orchestrator.objective_parser import ObjectiveParser
from env.textworld_wrapper import TextWorldWrapper
from evaluation.metrics import compile_report

# Import demo's semantic reasoning (avoids duplication)
import scripts.demo_live_textworld as _demo

# ── Configuration ─────────────────────────────────────────────────────────────

SEEDS = [42, 99, 123, 7, 777]

TIER_CONFIGS = {
    1: dict(quest_length=2, nb_rooms=2, nb_objects=3),
    2: dict(quest_length=4, nb_rooms=4, nb_objects=6),
    3: dict(quest_length=7, nb_rooms=6, nb_objects=10),
}

REPORTS_DIR = Path(__file__).parent / "reports"
GAMES_DIR   = Path(__file__).parent / "games"
MAX_TURNS   = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ── Game Generation ───────────────────────────────────────────────────────────

def generate_game(seed: int, tier: int, output_dir: Path) -> str:
    """Generate and compile a TextWorld game for the given seed + tier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = TIER_CONFIGS[tier]
    options = textworld.GameOptions()
    options.path = str(output_dir / f"game_t{tier}_s{seed}.z8")
    options.seeds = seed
    options.nb_rooms    = cfg["nb_rooms"]
    options.nb_objects  = cfg["nb_objects"]
    options.quest_length = cfg["quest_length"]
    game = textworld.generator.make_game(options)
    return textworld.generator.compile_game(game, options)


def _reset_demo_state():
    """Reset per-game module state in demo_live_textworld between games."""
    _demo._failed_actions.clear()
    _demo._subgoal_fail_count.clear()
    _demo._consecutive_failures = 0
    _demo._last_action = ""
    _demo._last_room = ""


def run_one_game(game_path: str, seed: int, tier: int) -> tuple[EpisodeResult, list[TurnLog]]:
    """
    Run the World-Model Agent using the same semantic pipeline as demo_live_textworld.py.
    Uses InMemoryGraphStore + ObjectiveParser + semantic_action() — no SLM required.
    """
    _reset_demo_state()

    request_infos = EnvInfos(
        feedback=True, description=True, inventory=True,
        location=True, won=True, lost=True,
        score=True, max_score=True, objective=True,
        admissible_commands=True,
    )
    env = textworld.start(game_path, request_infos)
    game_state = env.reset()

    graph = InMemoryGraphStore()
    extractor = TextExtractor(slm_runner=None)
    updater = Updater(graph)
    wm_builder = WorkingMemoryBuilder(graph)
    obj_parser = ObjectiveParser()

    obs = _demo.create_observation(game_state, turn_id=0)
    obj_parser.parse(obs.objective)
    wm_builder.set_objective(obs.objective)

    turn_logs: list[TurnLog] = []
    done = False
    total_reward = 0.0
    t_episode = time.perf_counter()

    for turn in range(MAX_TURNS):
        wm_builder.add_observation(obs.feedback or obs.description)
        wm = wm_builder.build()

        # Extract + update
        t0 = time.perf_counter()
        candidates = extractor.extract(obs, wm)
        update_report = updater.update(candidates, turn_id=turn)
        context_slice = QueryLayer(graph).retrieve(wm, turn)
        latency = (time.perf_counter() - t0) * 1000

        # Sub-goal completion check
        if turn > 0:
            obs_text = (obs.feedback or "") + " " + (obs.description or "")
            current_sg = obj_parser.get_current_subgoal()
            if current_sg and re.match(r"go \w+", current_sg.action):
                banner_m = re.search(r"-=\s*(.+?)\s*=-", obs.description or obs.feedback or "")
                room_now = banner_m.group(1).strip().lower() if banner_m else ""
                if room_now and room_now != _demo._last_room and _demo._last_room:
                    obj_parser.mark_completed(current_sg.index)
                    _demo._last_room = room_now
            else:
                obj_parser.advance_if_completed(obs_text)

        # Semantic action selection
        action, reason = _demo.semantic_action(obj_parser, graph, obs)

        slm_decision = SLMDecision(action_text=action, latency_ms=latency)
        turn_log = TurnLog(
            turn_id=turn, observation=obs,
            extracted_facts=candidates, update_report=update_report,
            context_slice=context_slice, slm_decision=slm_decision,
            reward=0.0, done=done,
            wall_clock_ms=latency,
            world_model_size_bytes=graph.get_stats().storage_bytes,
        )
        turn_logs.append(turn_log)

        # Execute action
        game_state, reward, done = env.step(action)
        total_reward += reward
        obs = _demo.create_observation(game_state, turn_id=turn + 1)
        _demo.record_failure(obs.feedback or obs.description, action, subgoal_index=(
            obj_parser.get_current_subgoal().index
            if obj_parser.get_current_subgoal() else -1
        ))

        if done:
            break

    env.close()
    result = EpisodeResult(
        game_path=game_path,
        total_turns=len(turn_logs),
        final_score=obs.score,
        max_score=obs.max_score,
        won=obs.won,
        turn_logs=turn_logs,
        total_wall_clock_seconds=time.perf_counter() - t_episode,
    )
    return result, turn_logs


# ── Baseline agent (full-history, no graph) ───────────────────────────────────

def run_baseline_game(game_path: str) -> tuple[EpisodeResult, list[TurnLog]]:
    """
    Baseline: always picks the first admissible command each turn.
    Uses TextWorldWrapper in eval_mode to access admissible_commands.
    """
    env = TextWorldWrapper(game_path, eval_mode=True)
    obs = env.reset()
    turn_logs = []
    won = False
    score = 0
    max_score = 1

    for turn_id in range(MAX_TURNS):
        commands = env.get_admissible_commands()
        action = commands[0] if commands else "look"
        t0 = time.perf_counter()
        obs, reward, done = env.step(action)
        latency = (time.perf_counter() - t0) * 1000

        slm = SLMDecision(action_text=action, latency_ms=latency)
        turn_logs.append(TurnLog(
            turn_id=turn_id, observation=obs, reward=reward, done=done,
            slm_decision=slm, extracted_facts=[],
        ))
        score = obs.score
        won = obs.won
        max_score = obs.max_score
        if done:
            break

    env.close()
    result = EpisodeResult(
        game_path=game_path, total_turns=len(turn_logs),
        final_score=score, max_score=max_score,
        won=won, turn_logs=turn_logs,
    )
    return result, turn_logs


# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark(tiers: list[int] | None = None, dry_run: bool = False):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / "benchmark.log"
    csv_path = REPORTS_DIR / "summary.csv"

    active_tiers = tiers or list(TIER_CONFIGS.keys())
    active_seeds = [42] if dry_run else SEEDS

    results_by_tier:      dict[int, list[EpisodeResult]] = {t: [] for t in active_tiers}
    baseline_by_tier:     dict[int, list[EpisodeResult]] = {t: [] for t in active_tiers}
    all_turn_logs:        list[TurnLog] = []
    baseline_turn_logs:   list[TurnLog] = []

    csv_rows = []
    errors = 0
    total_games = len(active_tiers) * len(active_seeds)
    game_num = 0

    with open(log_path, "w") as logf:
        logf.write(f"Benchmark started: {datetime.utcnow().isoformat()}Z\n")
        logf.write(f"Tiers: {active_tiers}  Seeds: {active_seeds}\n")
        logf.write(f"Total games: {total_games}\n\n")

        for tier in active_tiers:
            for seed in active_seeds:
                game_num += 1
                tag = f"[{game_num}/{total_games}] Tier {tier}, Seed {seed}"
                logger.info(f"{tag} — generating game...")
                try:
                    game_path = generate_game(seed, tier, GAMES_DIR)
                    logger.info(f"{tag} — running agent...")
                    t_start = time.perf_counter()
                    result, turn_logs = run_one_game(game_path, seed, tier)
                    elapsed = round(time.perf_counter() - t_start, 2)

                    results_by_tier[tier].append(result)
                    all_turn_logs.extend(turn_logs)

                    logger.info(f"{tag} — running baseline...")
                    b_result, b_logs = run_baseline_game(game_path)
                    baseline_by_tier[tier].append(b_result)
                    baseline_turn_logs.extend(b_logs)

                    row = {
                        "tier": tier, "seed": seed,
                        "won": result.won, "score": result.final_score,
                        "max_score": result.max_score, "turns": result.total_turns,
                        "elapsed_s": elapsed,
                        "baseline_won": b_result.won,
                    }
                    csv_rows.append(row)
                    status = "✅ WON" if result.won else "❌ lost"
                    logf.write(f"{tag}: {status} score={result.final_score}/{result.max_score} turns={result.total_turns} time={elapsed}s\n")
                    logger.info(f"{tag}: {status} score={result.final_score}/{result.max_score} in {result.total_turns} turns ({elapsed}s)")

                except Exception as e:
                    errors += 1
                    tb = traceback.format_exc()
                    logf.write(f"{tag}: ERROR — {e}\n{tb}\n")
                    logger.error(f"{tag}: ERROR — {e}")

        logf.write(f"\nBenchmark finished: {datetime.utcnow().isoformat()}Z\n")
        logf.write(f"Errors: {errors}/{total_games}\n")

    # Write summary CSV
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    # Write latest.json
    report = compile_report(
        results_by_tier, all_turn_logs,
        baseline_by_tier, baseline_turn_logs,
    )
    json_path = REPORTS_DIR / "latest.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary table
    print("\n" + "═" * 60)
    print("BENCHMARK COMPLETE")
    print("─" * 60)
    for tier in active_tiers:
        tier_res = results_by_tier[tier]
        wins = sum(1 for r in tier_res if r.won)
        print(f"  Tier {tier}: {wins}/{len(tier_res)} won  ({100*wins/max(len(tier_res),1):.0f}%)")
    print(f"\n  Errors: {errors}/{total_games} games")
    print(f"  Report: {json_path}")
    print(f"  CSV:    {csv_path}")
    print(f"  Log:    {log_path}")
    print("═" * 60)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track 1 Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="1 game only (fast test)")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="Run only this tier")
    args = parser.parse_args()

    tiers = [args.tier] if args.tier else None
    run_benchmark(tiers=tiers, dry_run=args.dry_run)
