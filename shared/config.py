# shared/config.py
# ============================================================================
# Global configuration with locked non-functional budgets.
# Spec Reference: Implementation Plan v2.0, Section 3 (Resolved Decisions)
#                 + Section 10 (Non-Functional Budgets)
# ============================================================================

from dataclasses import dataclass, field
from typing import Dict


import os

@dataclass
class SLMConfig:
    """Configuration for the small language model."""
    model_name: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    fallback_model: str = field(default_factory=lambda: os.getenv("OLLAMA_FALLBACK", "qwen2.5:3b"))
    backend: str = "ollama"                         # "ollama" | "llamacpp"
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"))
    temperature_decision: float = 0.3               # For action selection
    temperature_extraction: float = 0.1             # For fact extraction (near-deterministic)
    max_tokens: int = 512                           # Increased from 100 to prevent JSON truncation
    timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT", "15.0"))
    max_retries: int = 3                            # Retries on malformed output


@dataclass
class QueryConfig:
    """Configuration for the Query Layer retrieval."""
    token_budget: int = 512                         # Hard ceiling on context tokens
    max_traversal_depth: int = 2                    # BFS depth from anchor nodes
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "spatial": 0.35,
        "recency": 0.25,
        "corroboration": 0.15,
        "goal": 0.25,
    })
    # Spatial decay: relevance = max(0, 1.0 - hop_distance * spatial_decay)
    spatial_decay: float = 0.3
    # Recency decay: relevance = max(0.1, 1.0 - turns_ago * recency_decay)
    recency_decay: float = 0.05
    # Corroboration scaling: relevance = min(1.0, count * corroboration_scale)
    corroboration_scale: float = 0.25


@dataclass
class OrchestratorConfig:
    """Configuration for the Orchestrator / Decision Loop."""
    max_turns: int = 100                            # Safety limit per episode
    latency_budget_seconds: float = 10.0            # Max wall-clock per turn
    memory_growth_ceiling_kb: float = 5.0           # Max KB per turn average growth
    enable_logging: bool = True                     # Full per-turn logging
    log_dir: str = "logs/"
    # After N consecutive latency violations, switch to fallback model
    latency_violation_threshold: int = 3
    # Number of recent observations kept in WorkingMemory for coreference
    working_memory_observation_window: int = 3


@dataclass
class ExtractorConfig:
    """Configuration for the Extractor pipeline."""
    # Confidence base scores by extraction type
    confidence_direct: float = 0.90
    confidence_implied: float = 0.65
    confidence_negation: float = 0.80
    # Adjustments
    confidence_boost_slm: float = 0.05
    confidence_penalty_rule: float = -0.10
    confidence_penalty_novel_entity: float = -0.15
    confidence_boost_navigation: float = 0.05
    # Clamp range
    confidence_min: float = 0.10
    confidence_max: float = 0.99
    # Max segments per SLM batch call
    max_segments_per_batch: int = 3


@dataclass
class GlobalConfig:
    """Master configuration — single source of truth for all budgets and parameters."""
    slm: SLMConfig = field(default_factory=SLMConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)

    # Project-wide paths
    project_root: str = "."
    game_suite_dir: str = "env/game_suite"
    test_fixtures_dir: str = "tests/fixtures"


# ── Singleton-style default config ──────────────────────────────────────────

DEFAULT_CONFIG = GlobalConfig()
