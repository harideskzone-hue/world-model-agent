"""
tests/unit/test_benchmark.py
============================
6 unit tests for Gap 3+4: Benchmark pipeline and evaluation metrics.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from shared.models import (
    EpisodeResult, TurnLog, Observation, UpdateReport,
    SLMDecision, ContextSlice, CandidateFact, Edge,
)
from shared.enums import ExtractionMethod, ExtractionType, RelationType, EdgeStatus
from evaluation.metrics import (
    task_success_rate, state_tracking_precision, state_tracking_recall,
    contradiction_handling_pass_rate, memory_growth_kb_per_turn,
    context_efficiency, avg_latency_seconds, compile_report,
)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'evaluation', 'reports')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs():
    return Observation(feedback="ok", description="", inventory="", location="kitchen", objective="test")


def _episode(won: bool, score: int = 1, max_score: int = 1) -> EpisodeResult:
    return EpisodeResult(
        game_path="test.z8", total_turns=5,
        final_score=score, max_score=max_score, won=won, turn_logs=[]
    )


def _turn(facts=0, revised=0, expanded=1, tokens=256,
          latency_ms=50.0, corroborated=0, rejected=0) -> TurnLog:
    # CandidateFact requires: subject, relation, object, confidence, source_turn_id, extraction_type, extraction_method
    extracted = [
        CandidateFact(
            subject=f"obj{i}", relation=RelationType.CONTAINS, object="room",
            confidence=0.8, source_turn_id=0,
            extraction_type=ExtractionType.DIRECT,
            extraction_method=ExtractionMethod.RULE_FALLBACK,
        )
        for i in range(facts)
    ]
    rpt = UpdateReport(
        expanded=expanded, corroborated=corroborated,
        revised=revised, rejected=rejected, revisions=[]
    )
    # ContextSlice uses: formatted_text, included_facts, excluded_count, total_tokens_estimate
    ctx = ContextSlice(
        formatted_text="...",
        included_facts=[],
        excluded_count=0,
        total_tokens_estimate=tokens,
    )
    slm = SLMDecision(action_text="go north", latency_ms=latency_ms)
    return TurnLog(
        turn_id=0, observation=_obs(), reward=0.0, done=False,
        extracted_facts=extracted, update_report=rpt,
        context_slice=ctx, slm_decision=slm,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_task_success_rate(self):
        """Correct success rate from EpisodeResult list."""
        results = [_episode(True), _episode(False), _episode(True), _episode(True)]
        assert task_success_rate(results) == pytest.approx(0.75)
        assert task_success_rate([]) == 0.0
        assert task_success_rate([_episode(True)]) == 1.0

    def test_state_tracking_precision_recall(self):
        """Precision and recall produce values in [0, 1] for valid turn logs."""
        turns = [
            _turn(facts=3, expanded=3, revised=0),
            _turn(facts=2, expanded=2, revised=1),
        ]
        p = state_tracking_precision(turns)
        r = state_tracking_recall(turns)
        assert 0.0 <= p <= 1.0, f"Precision out of range: {p}"
        assert 0.0 <= r <= 1.0, f"Recall out of range: {r}"
        # Both turns have facts → recall should be 1.0
        assert r == 1.0

    def test_contradiction_handling_pass_rate(self):
        """Pass rate = 1.0 when contradictions are resolved."""
        turns_no_conflict = [_turn(revised=0), _turn(revised=0)]
        assert contradiction_handling_pass_rate(turns_no_conflict) == 1.0
        turns_with_conflict = [_turn(revised=1), _turn(revised=1)]
        assert contradiction_handling_pass_rate(turns_with_conflict) == 1.0

    def test_memory_growth_kb_per_turn(self):
        """Memory growth is non-negative and grows with more facts."""
        turns_few  = [_turn(facts=1), _turn(facts=1)]
        turns_many = [_turn(facts=10), _turn(facts=10)]
        assert memory_growth_kb_per_turn(turns_few)  >= 0
        assert memory_growth_kb_per_turn(turns_many) > memory_growth_kb_per_turn(turns_few)

    def test_latest_json_schema_matches_viewer_contract(self):
        """compile_report() output must contain all keys consumed by /api/evaluation."""
        results_by_tier = {
            1: [_episode(True), _episode(True), _episode(False)],
            2: [_episode(True), _episode(False)],
            3: [_episode(False), _episode(False)],
        }
        turns = [_turn(facts=3, revised=1), _turn(facts=2, revised=0)]
        report = compile_report(results_by_tier, turns)

        required_keys = [
            "task_success_rate", "state_tracking_precision", "state_tracking_recall",
            "contradiction_handling_pass_rate", "memory_growth_kb_per_turn",
            "context_efficiency_pct_of_budget", "avg_latency_seconds",
            "model_size_compliant", "total_games", "total_wins", "overall_success_rate",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: '{key}'"

        tsr = report["task_success_rate"]
        for tier in [1, 2, 3]:
            assert f"tier{tier}" in tsr

        # JSON-serialisable
        assert len(json.dumps(report)) > 0

    def test_benchmark_dry_run(self):
        """run_benchmark() dry_run=True must produce latest.json with valid content."""
        from evaluation.run_benchmark import run_benchmark

        report = run_benchmark(tiers=[1], dry_run=True)

        assert isinstance(report, dict), "Benchmark must return a dict"

        json_path = os.path.join(REPORTS_DIR, 'latest.json')
        assert os.path.exists(json_path), f"latest.json not written at {json_path}"

        with open(json_path) as f:
            data = json.load(f)
        assert "task_success_rate" in data
        assert "overall_success_rate" in data
        assert 0.0 <= data["overall_success_rate"] <= 1.0
