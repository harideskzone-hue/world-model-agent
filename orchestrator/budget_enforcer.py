# orchestrator/budget_enforcer.py
# ============================================================================
# Non-functional budget enforcement (latency, memory growth, tokens).
# Spec Reference: Implementation Plan v2.0, Section 10
# ============================================================================

from __future__ import annotations

import logging
import sys
from typing import Optional

from shared.config import OrchestratorConfig
from world_model.graph_store import GraphStoreBase

logger = logging.getLogger(__name__)


class BudgetEnforcer:
    """
    Enforces non-functional budgets per turn:
      - Latency: ≤ 10 seconds per turn
      - Memory growth: ≤ 5 KB per turn average
      - Consecutive latency violations → switch to fallback model
    """

    def __init__(
        self,
        graph: GraphStoreBase,
        config: Optional[OrchestratorConfig] = None,
    ):
        self._graph = graph
        self._config = config or OrchestratorConfig()
        self._consecutive_latency_violations = 0
        self._initial_size: Optional[int] = None
        self._turn_count = 0

    def check_latency(self, wall_clock_seconds: float) -> bool:
        """
        Check if a turn violated the latency budget.

        Returns True if within budget, False if violated.
        """
        budget = self._config.latency_budget_seconds
        if wall_clock_seconds > budget:
            self._consecutive_latency_violations += 1
            logger.warning(
                f"Latency violation: {wall_clock_seconds:.2f}s > {budget}s "
                f"({self._consecutive_latency_violations} consecutive)"
            )
            return False
        else:
            self._consecutive_latency_violations = 0
            return True

    def should_switch_to_fallback(self) -> bool:
        """Check if we should switch to the smaller fallback model."""
        return (
            self._consecutive_latency_violations
            >= self._config.latency_violation_threshold
        )

    def check_memory_growth(self) -> bool:
        """
        Check if average memory growth per turn is within budget.

        Returns True if within budget.
        """
        stats = self._graph.get_stats()
        current_size = stats.storage_bytes

        if self._initial_size is None:
            self._initial_size = current_size
            return True

        self._turn_count += 1
        if self._turn_count == 0:
            return True

        growth = current_size - self._initial_size
        avg_growth_per_turn = growth / self._turn_count
        budget_bytes = self._config.memory_growth_ceiling_kb * 1024

        if avg_growth_per_turn > budget_bytes:
            logger.warning(
                f"Memory growth: {avg_growth_per_turn:.0f} B/turn > "
                f"{budget_bytes:.0f} B/turn budget"
            )
            return False
        return True

    def get_status(self) -> dict:
        """Return current budget status."""
        stats = self._graph.get_stats()
        return {
            "consecutive_latency_violations": self._consecutive_latency_violations,
            "should_switch_model": self.should_switch_to_fallback(),
            "graph_storage_bytes": stats.storage_bytes,
            "graph_active_edges": stats.active_edges,
            "graph_total_edges": stats.total_edges,
        }
