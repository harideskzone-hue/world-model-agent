# tests/unit/test_extractor.py
# ============================================================================
# Unit tests for the Extractor pipeline (Stages 1-4).
# Tests preprocessor, rule fallback, schema validation, confidence scoring,
# and end-to-end pipeline without SLM (rule-only mode).
# ============================================================================

import pytest

from shared.models import Observation, WorkingMemory, TextSegment, RawExtraction, Edge
from shared.enums import (
    SegmentType, RelationType, ExtractionType, ExtractionMethod, EdgeStatus,
)
from shared.config import ExtractorConfig
from extractor.preprocessor import Preprocessor
from extractor.rule_fallback import RuleFallback
from extractor.schema_validator import SchemaValidator
from extractor.confidence_scorer import ConfidenceScorer
from extractor.text_extractor import TextExtractor


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def empty_wm():
    """Empty working memory."""
    return WorkingMemory(
        current_room="kitchen",
        objective="Find the key.",
    )


@pytest.fixture
def rich_wm():
    """Working memory with some facts."""
    return WorkingMemory(
        current_room="kitchen",
        current_room_facts=[
            Edge(
                id="e1", subject="kitchen", relation=RelationType.CONTAINS,
                object="brass key", confidence=0.9,
                extraction_method=ExtractionMethod.SLM,
                t_observed=0, t_valid_from=0,
            ),
        ],
        inventory_facts=[
            Edge(
                id="e2", subject="player", relation=RelationType.HOLDS,
                object="silver coin", confidence=0.9,
                extraction_method=ExtractionMethod.SLM,
                t_observed=0, t_valid_from=0,
            ),
        ],
        recent_observations=["You are in the kitchen."],
        current_sub_goal="Find the brass key",
        objective="Find the key and escape.",
    )


def make_observation(
    feedback="", description="", inventory="",
    location="kitchen", turn_id=0,
) -> Observation:
    """Helper to create test observations."""
    return Observation(
        feedback=feedback,
        description=description,
        inventory=inventory,
        location=location,
        objective="Find the key.",
        turn_id=turn_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

class TestPreprocessor:
    def test_basic_segmentation(self, empty_wm):
        pp = Preprocessor()
        obs = make_observation(
            description="You are in the kitchen. You see a brass key on the table."
        )
        segments = pp.process(obs, empty_wm)
        assert len(segments) >= 1
        assert all(isinstance(s, TextSegment) for s in segments)

    def test_room_description_classified(self, empty_wm):
        pp = Preprocessor()
        obs = make_observation(description="You are in the kitchen.")
        segments = pp.process(obs, empty_wm)
        room_segs = [s for s in segments if s.segment_type == SegmentType.ROOM_DESCRIPTION]
        assert len(room_segs) >= 1

    def test_navigation_classified(self, empty_wm):
        pp = Preprocessor()
        obs = make_observation(feedback="You go north.")
        segments = pp.process(obs, empty_wm)
        nav_segs = [s for s in segments if s.segment_type == SegmentType.NAVIGATION]
        assert len(nav_segs) >= 1

    def test_inventory_classified(self, empty_wm):
        pp = Preprocessor()
        obs = make_observation(feedback="You pick up the brass key.")
        segments = pp.process(obs, empty_wm)
        inv_segs = [s for s in segments if s.segment_type == SegmentType.INVENTORY_UPDATE]
        assert len(inv_segs) >= 1

    def test_state_change_classified(self, empty_wm):
        pp = Preprocessor()
        obs = make_observation(feedback="You unlock the door.")
        segments = pp.process(obs, empty_wm)
        state_segs = [s for s in segments if s.segment_type == SegmentType.STATE_CHANGE]
        assert len(state_segs) >= 1

    def test_feedback_classified(self, empty_wm):
        pp = Preprocessor()
        obs = make_observation(feedback="You can't do that.")
        segments = pp.process(obs, empty_wm)
        fb_segs = [s for s in segments if s.segment_type == SegmentType.FEEDBACK]
        assert len(fb_segs) >= 1

    def test_boilerplate_stripped(self, empty_wm):
        pp = Preprocessor()
        obs = make_observation(description="-= Kitchen =-\nYou are in the kitchen.")
        segments = pp.process(obs, empty_wm)
        # Banner should be stripped; "You are in the kitchen." should remain
        assert any("kitchen" in s.resolved_text.lower() for s in segments)
        assert not any("-=" in s.resolved_text for s in segments)

    def test_coreference_resolution(self, rich_wm):
        pp = Preprocessor()
        obs = make_observation(feedback="You see it here.")
        segments = pp.process(obs, rich_wm)
        # "here" should be resolved to "kitchen", "it" to an entity
        assert any("kitchen" in s.resolved_text.lower() for s in segments)

    def test_empty_observation(self, empty_wm):
        pp = Preprocessor()
        obs = make_observation()  # All fields empty
        segments = pp.process(obs, empty_wm)
        assert len(segments) == 0


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2b: RULE FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

class TestRuleFallback:
    def test_navigation_extraction(self):
        rf = RuleFallback()
        segments = [TextSegment(
            raw_text="You go north.",
            resolved_text="You go north.",
            segment_type=SegmentType.NAVIGATION,
            source_turn_id=1,
        )]
        results = rf.extract(segments, current_room="kitchen")
        assert len(results) >= 1
        assert any(r.relation == "located_in" for r in results)

    def test_room_contains_extraction(self):
        rf = RuleFallback()
        segments = [TextSegment(
            raw_text="You see a brass key.",
            resolved_text="You see a brass key.",
            segment_type=SegmentType.ROOM_DESCRIPTION,
            source_turn_id=0,
        )]
        results = rf.extract(segments, current_room="kitchen")
        assert len(results) >= 1
        assert any(r.subject == "kitchen" and r.relation == "contains" for r in results)

    def test_inventory_pickup_extraction(self):
        rf = RuleFallback()
        segments = [TextSegment(
            raw_text="You take the apple.",
            resolved_text="You take the apple.",
            segment_type=SegmentType.INVENTORY_UPDATE,
            source_turn_id=2,
        )]
        results = rf.extract(segments, current_room="kitchen")
        assert len(results) >= 1
        assert any(r.relation == "holds" and r.object == "apple" for r in results)

    def test_state_change_extraction(self):
        rf = RuleFallback()
        segments = [TextSegment(
            raw_text="You unlock the door.",
            resolved_text="You unlock the door.",
            segment_type=SegmentType.STATE_CHANGE,
            source_turn_id=3,
        )]
        results = rf.extract(segments)
        assert len(results) >= 1
        assert any(
            r.relation == "has_state" and r.object == "unlocked"
            for r in results
        )

    def test_feedback_skipped(self):
        rf = RuleFallback()
        segments = [TextSegment(
            raw_text="You can't do that.",
            resolved_text="You can't do that.",
            segment_type=SegmentType.FEEDBACK,
            source_turn_id=0,
        )]
        results = rf.extract(segments)
        assert len(results) == 0

    def test_multiple_objects(self):
        rf = RuleFallback()
        segments = [TextSegment(
            raw_text="You see a key, an apple, and a sword.",
            resolved_text="You see a key, an apple, and a sword.",
            segment_type=SegmentType.ROOM_DESCRIPTION,
            source_turn_id=0,
        )]
        results = rf.extract(segments, current_room="kitchen")
        # Should extract at least 3 contains relationships
        contains = [r for r in results if r.relation == "contains"]
        assert len(contains) >= 3


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: SCHEMA VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaValidator:
    def test_valid_extraction_passes(self):
        sv = SchemaValidator()
        extractions = [RawExtraction(
            subject="kitchen", relation="contains", object="key",
            extraction_type="direct", extraction_method="slm",
        )]
        result = sv.validate(extractions)
        assert len(result) == 1
        assert result[0].subject == "kitchen"

    def test_invalid_relation_rejected(self):
        sv = SchemaValidator()
        extractions = [RawExtraction(
            subject="kitchen", relation="smells_like", object="roses",
            extraction_type="direct", extraction_method="slm",
        )]
        result = sv.validate(extractions)
        assert len(result) == 0

    def test_alias_normalized(self):
        sv = SchemaValidator()
        extractions = [RawExtraction(
            subject="kitchen", relation="in", object="house",
            extraction_type="direct", extraction_method="slm",
        )]
        result = sv.validate(extractions)
        assert len(result) == 1
        assert result[0].relation == "located_in"

    def test_self_referential_rejected(self):
        sv = SchemaValidator()
        extractions = [RawExtraction(
            subject="kitchen", relation="contains", object="kitchen",
            extraction_type="direct", extraction_method="slm",
        )]
        result = sv.validate(extractions)
        assert len(result) == 0

    def test_entity_names_normalized(self):
        sv = SchemaValidator()
        extractions = [RawExtraction(
            subject="The Kitchen", relation="contains", object="A Brass Key",
            extraction_type="direct", extraction_method="slm",
        )]
        result = sv.validate(extractions)
        assert len(result) == 1
        assert result[0].subject == "kitchen"
        assert result[0].object == "brass key"

    def test_deduplication(self):
        sv = SchemaValidator()
        extractions = [
            RawExtraction(
                subject="kitchen", relation="contains", object="key",
                extraction_type="direct", extraction_method="slm",
            ),
            RawExtraction(
                subject="kitchen", relation="contains", object="key",
                extraction_type="direct", extraction_method="rule_fallback",
            ),
        ]
        result = sv.validate(extractions)
        assert len(result) == 1  # Deduplicated


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: CONFIDENCE SCORER
# ═══════════════════════════════════════════════════════════════════════════

class TestConfidenceScorer:
    def test_direct_gets_high_confidence(self):
        cs = ConfidenceScorer()
        extractions = [RawExtraction(
            subject="kitchen", relation="contains", object="key",
            extraction_type="direct", extraction_method="slm",
        )]
        results = cs.score(extractions, known_entities={"kitchen", "key"})
        assert len(results) == 1
        assert results[0].confidence >= 0.85

    def test_implied_gets_lower_confidence(self):
        cs = ConfidenceScorer()
        ext_direct = [RawExtraction(
            subject="kitchen", relation="contains", object="key",
            extraction_type="direct", extraction_method="slm",
        )]
        ext_implied = [RawExtraction(
            subject="kitchen", relation="connects_to", object="garden",
            extraction_type="implied", extraction_method="slm",
        )]
        known = {"kitchen", "key", "garden"}
        direct = cs.score(ext_direct, known_entities=known)
        implied = cs.score(ext_implied, known_entities=known)
        assert direct[0].confidence > implied[0].confidence

    def test_rule_fallback_penalized(self):
        cs = ConfidenceScorer()
        ext_slm = [RawExtraction(
            subject="kitchen", relation="contains", object="key",
            extraction_type="direct", extraction_method="slm",
        )]
        ext_rule = [RawExtraction(
            subject="kitchen", relation="contains", object="apple",
            extraction_type="direct", extraction_method="rule_fallback",
        )]
        known = {"kitchen", "key", "apple"}
        slm_result = cs.score(ext_slm, known_entities=known)
        rule_result = cs.score(ext_rule, known_entities=known)
        assert slm_result[0].confidence > rule_result[0].confidence

    def test_confidence_clamped(self):
        cs = ConfidenceScorer()
        extractions = [RawExtraction(
            subject="kitchen", relation="contains", object="key",
            extraction_type="direct", extraction_method="slm",
        )]
        results = cs.score(extractions, known_entities={"kitchen", "key"})
        assert 0.10 <= results[0].confidence <= 0.99

    def test_novel_entity_penalized(self):
        cs = ConfidenceScorer()
        # "new_entity" not in known_entities → penalty
        extractions = [RawExtraction(
            subject="kitchen", relation="contains", object="new_entity",
            extraction_type="direct", extraction_method="slm",
        )]
        results = cs.score(extractions, known_entities={"kitchen"})
        assert results[0].confidence < 0.95  # Should be penalized


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE (RULE-ONLY MODE — NO SLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestTextExtractorRuleOnly:
    """Test the full pipeline in rule-only mode (no SLM)."""

    def test_end_to_end_room_description(self, empty_wm):
        extractor = TextExtractor(slm_runner=None)
        obs = make_observation(
            description="You are in the kitchen. You see a brass key.",
            location="kitchen",
        )
        results = extractor.extract(obs, empty_wm)
        assert len(results) >= 1
        # Should find at least "kitchen contains brass key"
        contains = [f for f in results if f.relation == RelationType.CONTAINS]
        assert len(contains) >= 1

    def test_end_to_end_state_change(self, empty_wm):
        extractor = TextExtractor(slm_runner=None)
        obs = make_observation(feedback="You unlock the door.")
        results = extractor.extract(obs, empty_wm)
        state = [f for f in results if f.relation == RelationType.HAS_STATE]
        assert len(state) >= 1
        assert state[0].object == "unlocked"

    def test_end_to_end_inventory(self, empty_wm):
        extractor = TextExtractor(slm_runner=None)
        obs = make_observation(feedback="You take the apple.")
        results = extractor.extract(obs, empty_wm)
        holds = [f for f in results if f.relation == RelationType.HOLDS]
        assert len(holds) >= 1

    def test_feedback_only_returns_empty(self, empty_wm):
        extractor = TextExtractor(slm_runner=None)
        obs = make_observation(feedback="You can't do that.")
        results = extractor.extract(obs, empty_wm)
        assert len(results) == 0

    def test_all_confidences_in_range(self, empty_wm):
        extractor = TextExtractor(slm_runner=None)
        obs = make_observation(
            description="You are in the garden. There is a sword here.",
            feedback="You go north.",
            location="garden",
        )
        results = extractor.extract(obs, empty_wm)
        for fact in results:
            assert 0.10 <= fact.confidence <= 0.99, \
                f"Confidence {fact.confidence} out of range for {fact}"
