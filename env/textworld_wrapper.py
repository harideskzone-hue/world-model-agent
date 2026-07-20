# env/textworld_wrapper.py
# ============================================================================
# Thin adapter between TextWorld's Gym-style API and our pipeline's I/O contract.
# Spec Reference: Implementation Plan v2.0, Section 8.1 (Environment Wrapper API)
#
# CRITICAL: ground-truth queries are isolated behind get_ground_truth_state()
#           and must NEVER be called from live agent code.
# ============================================================================

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

from shared.models import Observation, GroundTruthFact

logger = logging.getLogger(__name__)

# TextWorld may not be installed during early development/testing.
# Guard import so unit tests for other modules don't fail.
try:
    import textworld
    from textworld import EnvInfos
    TEXTWORLD_AVAILABLE = True
except ImportError:
    TEXTWORLD_AVAILABLE = False
    logger.warning("textworld not installed — TextWorldWrapper will be non-functional")


class TextWorldWrapper:
    """
    Adapter between TextWorld's Gym-style Environment and our pipeline.

    Produces: Observation dataclass
    Consumed by: Extractor (for fact extraction), Orchestrator (for turn sequencing)

    Usage:
        wrapper = TextWorldWrapper("path/to/game.z8")
        obs = wrapper.reset()
        obs, reward, done = wrapper.step("take apple")
    """

    def __init__(self, game_path: str, eval_mode: bool = False):
        """
        Initialize the wrapper.

        Args:
            game_path: Path to a TextWorld game file (.z8 or .json)
            eval_mode: If True, request ground-truth facts and admissible
                       commands for evaluation. NEVER True in live agent.
        """
        if not TEXTWORLD_AVAILABLE:
            raise RuntimeError(
                "textworld is not installed. Run: pip install textworld"
            )

        self._game_path = game_path
        self._eval_mode = eval_mode
        self._turn_id = 0
        self._env = None
        self._last_game_state = None

        # Configure what info to request from TextWorld
        self._request_infos = EnvInfos(
            feedback=True,
            description=True,
            inventory=True,
            location=True,
            won=True,
            lost=True,
            score=True,
            max_score=True,
            objective=True,
        )

        # Evaluation-only info — never used in the live agent path
        if eval_mode:
            self._request_infos.facts = True
            self._request_infos.admissible_commands = True
            self._request_infos.policy_commands = True

    def reset(self) -> Observation:
        """
        Reset the environment and return the initial observation.
        Sets turn_id to 0.
        """
        if self._env is not None:
            self._env.close()

        self._env = textworld.start(self._game_path, self._request_infos)
        self._last_game_state = self._env.reset()
        self._turn_id = 0

        return self._build_observation(self._last_game_state)

    def step(self, action_text: str) -> Tuple[Observation, float, bool]:
        """
        Execute an action and return (observation, reward, done).
        Increments turn_id.

        Args:
            action_text: The text command to send to the environment

        Returns:
            Tuple of (Observation, reward float, done bool)
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._last_game_state, reward, done = self._env.step(action_text)
        self._turn_id += 1

        obs = self._build_observation(self._last_game_state)
        logger.debug(
            f"Turn {self._turn_id}: action='{action_text}' reward={reward} done={done}"
        )

        return obs, reward, done

    def get_ground_truth_state(self) -> List[GroundTruthFact]:
        """
        EVALUATION ONLY — Return TextWorld's internal ground-truth facts.

        These map to TextWorld's Proposition objects and represent the
        simulator's actual world state, NOT the agent's beliefs.

        Raises:
            RuntimeError: If eval_mode is False
        """
        if not self._eval_mode:
            raise RuntimeError(
                "Ground truth access requires eval_mode=True. "
                "This method must NEVER be called from live agent code."
            )
        if self._last_game_state is None:
            return []

        facts = []
        for prop in self._last_game_state.get("facts", []):
            facts.append(GroundTruthFact(
                name=prop.name if hasattr(prop, 'name') else str(prop),
                arguments=[
                    v.name if hasattr(v, 'name') else str(v)
                    for v in (prop.arguments if hasattr(prop, 'arguments') else [])
                ],
            ))
        return facts

    def get_admissible_commands(self) -> List[str]:
        """
        EVALUATION ONLY — Return TextWorld's admissible commands.
        The live agent must NEVER use this.
        """
        if not self._eval_mode:
            raise RuntimeError(
                "Admissible commands access requires eval_mode=True."
            )
        if self._last_game_state is None:
            return []
        return list(self._last_game_state.get("admissible_commands", []))

    def close(self) -> None:
        """Release environment resources."""
        if self._env is not None:
            self._env.close()
            self._env = None

    @property
    def turn_id(self) -> int:
        return self._turn_id

    def _build_observation(self, game_state) -> Observation:
        """Convert TextWorld GameState into our Observation dataclass."""
        return Observation(
            feedback=game_state.get("feedback", ""),
            description=game_state.get("description", ""),
            inventory=game_state.get("inventory", ""),
            location=game_state.get("location", ""),
            objective=game_state.get("objective", ""),
            score=game_state.get("score", 0),
            max_score=game_state.get("max_score", 0),
            won=game_state.get("won", False),
            lost=game_state.get("lost", False),
            turn_id=self._turn_id,
        )
