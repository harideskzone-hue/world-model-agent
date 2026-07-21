# slm/action_selector.py
# ============================================================================
# Action selection via SLM with structured retry logic.
# Spec Reference: Implementation Plan (SLM-Only Semantic Agent)
# ============================================================================

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from shared.models import Observation, ContextSlice, SLMDecision, WorkingMemory
from shared.config import SLMConfig
from slm.model_runner import SLMRunner
from slm.action_grammar import ActionGrammar
from slm.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ActionSelector:
    """
    Constructs prompts from semantic ContextSlice + WorkingMemory, calls SLM, parses output.
    Contains retry logic for invalid outputs. There is no rule-based fallback.
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
        working_memory: WorkingMemory,
        observation: Observation,
    ) -> SLMDecision:
        """
        Select the next action using the SLM based purely on semantic state.

        Args:
            context_slice: Retrieved world model context
            working_memory: The current memory and goals
            observation: The raw observation (used for admissible commands if needed)

        Returns:
            SLMDecision with the chosen action text
        """
        start_time = time.time()

        if self._slm is None or not self._slm.is_available():
            logger.error("SLM is unavailable. SLM-only policy cannot proceed.")
            elapsed = (time.time() - start_time) * 1000
            return SLMDecision(
                action_text="INVALID_ACTION",
                raw_output="Ollama Unavailable",
                latency_ms=elapsed,
                retry_count=0,
            )

        grammar_desc = self._grammar.get_grammar_description()
        
        # Build the initial semantic prompt
        prompt = PromptBuilder.build_prompt(
            context_slice, working_memory, observation, grammar_desc
        )

        max_retries = self._config.max_retries
        raw_output = ""
        action_text = ""
        
        for attempt in range(max_retries):
            try:
                raw_output = self._slm.generate(
                    prompt, temperature=self._config.temperature_decision
                )
                
                parsed_action = self._parse_action(raw_output)
                
                if parsed_action:
                    # Valid action parsed!
                    elapsed = (time.time() - start_time) * 1000
                    return SLMDecision(
                        action_text=parsed_action,
                        raw_output=raw_output,
                        latency_ms=elapsed,
                        retry_count=attempt,
                    )
                
                logger.warning(f"Invalid SLM output on attempt {attempt+1}: {raw_output}")
                
                # Setup retry prompt
                prompt = PromptBuilder.build_retry_prompt(raw_output, grammar_desc)
                
            except Exception as e:
                logger.error(f"SLM Generation failed: {e}")
                break
                
        # All retries exhausted or catastrophic failure
        elapsed = (time.time() - start_time) * 1000
        logger.error("All SLM attempts failed to produce a valid action.")
        return SLMDecision(
            action_text="INVALID_ACTION",
            raw_output=raw_output or "FAILURE",
            latency_ms=elapsed,
            retry_count=max_retries,
        )

    def _parse_action(self, raw_output: str) -> Optional[str]:
        """
        Parse SLM output to extract the action text.
        Aggressively strips markdown, conversational filler, reasoning, etc.
        """
        if not raw_output:
            return None

        text = raw_output.strip()

        # 1. Strip markdown blocks if they wrapped the action
        text = re.sub(r'```[^\n]*\n', '', text)
        text = text.replace('```', '')

        # 2. Extract first logical line that might be an action (ignore reasoning)
        # Sometimes models say "Based on the state, I will: \n go north"
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        candidate = None
        for line in reversed(lines): # often the final line is the action
            line_clean = re.sub(r"^(?:ACTION|action|Action|Command|command)\s*[:\-]?\s*", "", line)
            line_clean = line_clean.strip("\"'`* ")
            if len(line_clean) > 0 and len(line_clean) < 100:
                candidate = line_clean
                break
        
        if not candidate:
            candidate = lines[0] if lines else text

        candidate = re.sub(r"^(?:ACTION|action|Action|Command|command)\s*[:\-]?\s*", "", candidate)
        candidate = candidate.strip("\"'`*. \t")

        # Must start with a known verb or direction
        first_word = candidate.split()[0].lower() if candidate else ""
        known_starts = {
            "look", "inventory", "go", "take", "drop", "open", "close",
            "unlock", "examine", "put", "eat", "cook", "slice", "dice",
            "chop", "north", "south", "east", "west", "up", "down", "insert"
        }
        
        if first_word not in known_starts:
            logger.debug(f"Parsed action '{candidate}' doesn't start with known verb")
            return None

        # Normalize bare directions to "go {direction}"
        if first_word in {"north", "south", "east", "west", "up", "down"}:
            candidate = f"go {first_word}"

        return candidate.lower()
