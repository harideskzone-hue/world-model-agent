# orchestrator/decision_loop.py
# ============================================================================
# Main per-turn decision loop — the heart of the agent.
# Spec Reference: Implementation Plan v2.0, Section 8.3 (Decision Loop)
# ============================================================================

from __future__ import annotations

import logging
import time
from typing import Optional, List

from shared.models import (
    Observation, TurnLog, EpisodeResult,
    CandidateFact, UpdateReport, ContextSlice, SLMDecision,
)
from shared.config import GlobalConfig, DEFAULT_CONFIG
from world_model.graph_store import InMemoryGraphStore
from extractor.text_extractor import TextExtractor
from updater.updater import Updater
from query_layer.query_layer import QueryLayer
from slm.model_runner import SLMRunner
from slm.action_selector import ActionSelector
from orchestrator.working_memory import WorkingMemoryBuilder
from orchestrator.budget_enforcer import BudgetEnforcer

logger = logging.getLogger(__name__)


class DecisionLoop:
    """
    Main per-turn decision loop. Orchestrates the full pipeline:

      Observation → Extractor → Updater → QueryLayer → ActionSelector → Action

    Each module is decoupled and communicates only through shared dataclasses.
    """

    def __init__(
        self,
        config: Optional[GlobalConfig] = None,
        slm_runner: Optional[SLMRunner] = None,
    ):
        self._config = config or DEFAULT_CONFIG

        # Initialize the graph store (fresh per episode)
        self._graph = InMemoryGraphStore()

        # Initialize modules
        self._extractor = TextExtractor(
            slm_runner=slm_runner,
            config=self._config.extractor,
        )
        self._updater = Updater(self._graph)
        self._query_layer = QueryLayer(
            self._graph, config=self._config.query,
        )
        self._action_selector = ActionSelector(
            slm=slm_runner, config=self._config.slm,
        )
        self._wm_builder = WorkingMemoryBuilder(
            self._graph, config=self._config.orchestrator,
        )
        self._budget_enforcer = BudgetEnforcer(
            self._graph, config=self._config.orchestrator,
        )

        # SLM runner for model switching
        self._slm_runner = slm_runner

    def run_episode(
        self, env, game_path: str = "",
    ) -> EpisodeResult:
        """
        Run a complete episode (game) from start to finish.

        Args:
            env: A TextWorldWrapper instance (already constructed)
            game_path: Path to the game file (for logging)

        Returns:
            EpisodeResult with full turn logs
        """
        episode_start = time.time()
        turn_logs: List[TurnLog] = []

        # Reset environment
        obs = env.reset()
        self._wm_builder.set_objective(obs.objective)
        self._wm_builder.add_observation(
            obs.feedback or obs.description
        )

        # Initial extraction from the reset observation
        self._process_initial_observation(obs)

        done = False
        max_turns = self._config.orchestrator.max_turns

        while not done and obs.turn_id < max_turns:
            turn_start = time.time()

            # ── 1. Build Working Memory ──
            working_memory = self._wm_builder.build()

            # ── 2. Extract facts from observation ──
            candidates = self._extractor.extract(obs, working_memory)

            # ── 3. Update world model ──
            update_report = self._updater.update(candidates, obs.turn_id)

            # ── 4. Query for context ──
            context_slice = self._query_layer.retrieve(
                working_memory, obs.turn_id,
            )

            # ── 5. Select action ──
            slm_decision = self._action_selector.select_action(
                context_slice, obs,
            )

            # ── 6. Execute action ──
            obs, reward, done = env.step(slm_decision.action_text)
            self._wm_builder.add_observation(
                obs.feedback or obs.description
            )

            # ── 7. Budget enforcement ──
            turn_elapsed = time.time() - turn_start
            self._budget_enforcer.check_latency(turn_elapsed)
            self._budget_enforcer.check_memory_growth()

            # Switch model if needed
            if (self._budget_enforcer.should_switch_to_fallback() and
                    self._slm_runner is not None):
                self._slm_runner.switch_model(self._config.slm.fallback_model)
                logger.warning("Switched to fallback SLM model")

            # ── 8. Log turn ──
            turn_log = TurnLog(
                turn_id=obs.turn_id,
                observation=obs,
                extracted_facts=candidates,
                update_report=update_report,
                context_slice=context_slice,
                slm_decision=slm_decision,
                reward=reward,
                done=done,
                wall_clock_ms=turn_elapsed * 1000,
                world_model_size_bytes=self._graph.get_stats().storage_bytes,
            )
            turn_logs.append(turn_log)

            logger.info(
                f"Turn {obs.turn_id}: action='{slm_decision.action_text}' "
                f"reward={reward} done={done} "
                f"facts={len(candidates)} "
                f"latency={turn_elapsed*1000:.0f}ms"
            )

        # Episode complete
        total_time = time.time() - episode_start

        result = EpisodeResult(
            game_path=game_path,
            total_turns=obs.turn_id,
            final_score=obs.score,
            max_score=obs.max_score,
            won=obs.won,
            turn_logs=turn_logs,
            final_world_model_json=self._graph.serialize(),
            total_wall_clock_seconds=total_time,
        )

        logger.info(
            f"Episode complete: {result.total_turns} turns, "
            f"score={result.final_score}/{result.max_score}, "
            f"won={result.won}, time={total_time:.1f}s"
        )
        return result

    def _process_initial_observation(self, obs: Observation) -> None:
        """Process the initial reset observation (turn 0)."""
        working_memory = self._wm_builder.build()
        candidates = self._extractor.extract(obs, working_memory)
        self._updater.update(candidates, turn_id=0)

    @property
    def graph(self) -> InMemoryGraphStore:
        """Expose graph for testing/evaluation."""
        return self._graph
