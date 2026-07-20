# slm/model_runner.py
# ============================================================================
# Ollama/LLM backend wrapper.
# Spec Reference: Implementation Plan v2.0, Section 8.2
# ============================================================================

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from shared.config import SLMConfig

logger = logging.getLogger(__name__)

# Guard import — Ollama may not be installed during testing
try:
    import urllib.request
    import urllib.error
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


class SLMRunner:
    """
    Wrapper around a local SLM backend (Ollama by default).

    Provides a simple generate() method used by both the Extractor
    and the ActionSelector.

    Communication is via HTTP to the Ollama REST API — no Python SDK needed.
    """

    def __init__(self, config: Optional[SLMConfig] = None):
        self._config = config or SLMConfig()
        self._model = self._config.model_name
        self._base_url = self._config.base_url

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from the SLM.

        Args:
            prompt: Input prompt
            temperature: Override temperature (uses config default if None)
            max_tokens: Override max tokens

        Returns:
            Generated text string

        Raises:
            RuntimeError: If the model server is unavailable or returns an error
        """
        temp = temperature if temperature is not None else self._config.temperature_decision
        max_tok = max_tokens or self._config.max_tokens

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
            },
        }

        url = f"{self._base_url}/api/generate"
        start_time = time.time()

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(
                req, timeout=self._config.timeout_seconds
            ) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self._base_url}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"SLM generation failed: {e}") from e

        elapsed_ms = (time.time() - start_time) * 1000
        result = response_data.get("response", "")

        logger.debug(
            f"SLM generate: {len(prompt)} chars prompt → {len(result)} chars response "
            f"in {elapsed_ms:.0f}ms"
        )
        return result

    def is_available(self) -> bool:
        """Health check — is the model server running?"""
        try:
            url = f"{self._base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                available = any(self._model in m for m in models)
                if not available:
                    logger.warning(
                        f"Model '{self._model}' not found in Ollama. "
                        f"Available: {models}"
                    )
                return available
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def switch_model(self, model_name: str) -> None:
        """Switch to a different model (e.g., fallback from 4b to 2b)."""
        logger.info(f"Switching SLM model: {self._model} → {model_name}")
        self._model = model_name
