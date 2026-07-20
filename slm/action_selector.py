# slm/action_selector.py
# ============================================================================
# Action selection via SLM with structured retry logic.
# Spec Reference: Implementation Plan v2.0, Section 8.2
# ============================================================================

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from shared.models import Observation, ContextSlice, SLMDecision
from shared.config import SLMConfig
from slm.model_runner import SLMRunner
from slm.action_grammar import ActionGrammar

logger = logging.getLogger(__name__)

# ── Prompt templates ────────────────────────────────────────────────────────

DECISION_PROMPT = """You are an agent in a text adventure game.

OBJECTIVE: {objective}

CURRENT WORLD STATE:
{context}

CURRENT OBSERVATION:
{observation}

{grammar}

Choose the single best action to make progress toward your objective.
Respond with ONLY the action text on one line.
ACTION:"""

SIMPLIFIED_PROMPT = """You are playing a text adventure. Pick ONE action.

Current situation: {observation}
Goal: {objective}

Actions you can take: {simple_actions}

Reply with ONLY the action. Example: "go north" or "take key"
ACTION:"""

LAST_RESORT_PROMPT = """Pick one word: look, inventory, north, south, east, west
WORD:"""


class ActionSelector:
    """
    Constructs prompts from context + observation, calls SLM, parses output.

    Retry policy:
      Attempt 1: Full prompt with grammar and world state
      Attempt 2: Simplified prompt with compact action list
      Attempt 3: Last-resort single-word prompt
      Fallback:  "look" (safe no-op that produces new observation)
    """

    def __init__(
        self,
        slm: Optional[SLMRunner] = None,
        config: Optional[SLMConfig] = None,
    ):
        self._slm = slm
        self._config = config or SLMConfig()
        self._grammar = ActionGrammar()

    def select_action(
        self,
        context_slice: ContextSlice,
        observation: Observation,
    ) -> SLMDecision:
        """
        Select the next action using the SLM.

        Args:
            context_slice: Retrieved world model context
            observation: Current turn's raw observation

        Returns:
            SLMDecision with the chosen action text
        """
        start_time = time.time()

        if self._slm is None:
            # No SLM available — use a simple heuristic
            return self._heuristic_action(observation, start_time)

        # ── Attempt 1: Full prompt ──
        action = self._try_full_prompt(context_slice, observation)
        if action:
            elapsed = (time.time() - start_time) * 1000
            return SLMDecision(
                action_text=action,
                raw_output=action,
                latency_ms=elapsed,
                retry_count=0,
            )

        # ── Attempt 2: Simplified prompt ──
        action = self._try_simplified_prompt(observation)
        if action:
            elapsed = (time.time() - start_time) * 1000
            return SLMDecision(
                action_text=action,
                raw_output=action,
                latency_ms=elapsed,
                retry_count=1,
            )

        # ── Attempt 3: Last resort ──
        action = self._try_last_resort()
        if action:
            elapsed = (time.time() - start_time) * 1000
            return SLMDecision(
                action_text=action,
                raw_output=action,
                latency_ms=elapsed,
                retry_count=2,
            )

        # ── Fallback: look ──
        elapsed = (time.time() - start_time) * 1000
        logger.warning("All SLM attempts failed — falling back to 'look'")
        return SLMDecision(
            action_text="look",
            raw_output="FALLBACK",
            latency_ms=elapsed,
            retry_count=3,
        )

    def _try_full_prompt(
        self, context_slice: ContextSlice, observation: Observation,
    ) -> Optional[str]:
        """Attempt 1: Full prompt with world state and grammar."""
        try:
            prompt = DECISION_PROMPT.format(
                objective=observation.objective,
                context=context_slice.formatted_text,
                observation=observation.feedback or observation.description,
                grammar=self._grammar.get_grammar_description(),
            )
            raw = self._slm.generate(
                prompt, temperature=self._config.temperature_decision,
            )
            return self._parse_action(raw)
        except Exception as e:
            logger.warning(f"Full prompt failed: {e}")
            return None

    def _try_simplified_prompt(self, observation: Observation) -> Optional[str]:
        """Attempt 2: Simplified prompt."""
        try:
            prompt = SIMPLIFIED_PROMPT.format(
                observation=observation.feedback or observation.description,
                objective=observation.objective,
                simple_actions=self._grammar.get_simple_action_list(),
            )
            raw = self._slm.generate(
                prompt, temperature=self._config.temperature_decision,
            )
            return self._parse_action(raw)
        except Exception as e:
            logger.warning(f"Simplified prompt failed: {e}")
            return None

    def _try_last_resort(self) -> Optional[str]:
        """Attempt 3: Extremely simple prompt."""
        try:
            raw = self._slm.generate(
                LAST_RESORT_PROMPT, temperature=0.1,
            )
            word = raw.strip().split()[0].lower() if raw.strip() else None
            if word in {"look", "inventory", "north", "south", "east", "west"}:
                if word in {"north", "south", "east", "west"}:
                    return f"go {word}"
                return word
            return None
        except Exception as e:
            logger.warning(f"Last resort prompt failed: {e}")
            return None

    def _parse_action(self, raw_output: str) -> Optional[str]:
        """
        Parse SLM output to extract the action text.
        Handles common output formats: bare text, "ACTION: ...", etc.
        """
        if not raw_output:
            return None

        text = raw_output.strip()

        # Remove "ACTION:" prefix if present
        text = re.sub(r"^(?:ACTION|action|Action)\s*:\s*", "", text)

        # Take only the first line
        text = text.split("\n")[0].strip()

        # Remove quotes
        text = text.strip("\"'`")

        # Basic validation — must look like an action
        if not text or len(text) > 100:
            return None

        # Must start with a known verb or direction
        first_word = text.split()[0].lower() if text else ""
        known_starts = {
            "look", "inventory", "go", "take", "drop", "open", "close",
            "unlock", "examine", "put", "eat", "cook", "slice", "dice",
            "chop", "north", "south", "east", "west", "up", "down",
        }
        if first_word not in known_starts:
            logger.debug(f"Parsed action '{text}' doesn't start with known verb")
            return None

        # Normalize bare directions to "go {direction}"
        if first_word in {"north", "south", "east", "west", "up", "down"}:
            text = f"go {first_word}"

        return text.lower()

    def _heuristic_action(
        self, observation: Observation, start_time: float,
    ) -> SLMDecision:
        """Simple heuristic when no SLM is available (for testing)."""
        elapsed = (time.time() - start_time) * 1000
        return SLMDecision(
            action_text="look",
            raw_output="HEURISTIC_FALLBACK",
            latency_ms=elapsed,
            retry_count=0,
        )
