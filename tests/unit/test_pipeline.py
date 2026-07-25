import pytest
import json
from pipeline.preprocessor import MinimalPreprocessor
from pipeline.json_parser import JSONParser
from pipeline.schema_validator import SchemaValidator
from pipeline.confidence_assigner import ConfidenceAssigner
from shared.models import CandidateFact
from shared.enums import RelationType, NodeType, ExtractionType, ExtractionMethod

class TestMinimalPreprocessor:
    def setup_method(self):
        self.preprocessor = MinimalPreprocessor()

    def test_empty_string(self):
        assert self.preprocessor.preprocess("") == ""

    def test_whitespace_only(self):
        assert self.preprocessor.preprocess("   \n\t  ") == ""

    def test_collapse_spaces(self):
        assert self.preprocessor.preprocess("This   is  a   test.") == "This is a test."

    def test_collapse_newlines(self):
        assert self.preprocessor.preprocess("This\n\nis\n\n\na test.") == "This is a test."

    def test_utf8_preserved(self):
        # Even with normal utf-8 strings, they shouldn't be mangled
        text = "Café \u2014 test"
        assert self.preprocessor.preprocess(text) == "Café \u2014 test"
        
    def test_mixed_newlines_spaces(self):
        assert self.preprocessor.preprocess("   Hello \n \n  World   ") == "Hello World"

class TestJSONParser:
    def setup_method(self):
        self.parser = JSONParser()

    def test_valid_array(self):
        result = self.parser.parse('[{"subject": "test"}]')
        assert result == [{"subject": "test"}]

    def test_valid_object_not_array(self):
        result = self.parser.parse('{"subject": "test"}')
        assert result == []

    def test_malformed_json(self):
        result = self.parser.parse('[{"subject": "test"') # missing bracket
        assert result == []

    def test_empty_string(self):
        result = self.parser.parse('')
        assert result == []

    def test_non_string_input(self):
        result = self.parser.parse(None)
        assert result == []
        
    def test_trailing_comma(self):
        # standard json.loads fails on trailing comma
        result = self.parser.parse('[{"subject": "test"},]')
        assert result == []

class TestSchemaValidator:
    def setup_method(self):
        self.validator = SchemaValidator()
        self.turn_id = 5

    def test_missing_subject(self):
        raw = [{"relation": "contains", "object": "key"}]
        assert self.validator.validate(raw, self.turn_id) == []

    def test_missing_relation(self):
        raw = [{"subject": "chest", "object": "key"}]
        assert self.validator.validate(raw, self.turn_id) == []

    def test_missing_object(self):
        raw = [{"subject": "chest", "relation": "contains"}]
        assert self.validator.validate(raw, self.turn_id) == []

    def test_invalid_relation_string(self):
        raw = [{"subject": "chest", "relation": "not_a_relation", "object": "key"}]
        assert self.validator.validate(raw, self.turn_id) == []

    def test_extra_fields_rejected(self):
        raw = [{"subject": "chest", "relation": "contains", "object": "key", "extra": "field"}]
        assert self.validator.validate(raw, self.turn_id) == []

    def test_valid_with_optional_fields(self):
        raw = [{
            "subject": "kitchen", 
            "relation": "contains", 
            "object": "key",
            "subject_type": "ROOM",
            "object_type": "OBJECT",
            "extraction_type": "DIRECT"
        }]
        facts = self.validator.validate(raw, self.turn_id)
        assert len(facts) == 1
        fact = facts[0]
        assert fact.subject == "kitchen"
        assert fact.relation == RelationType.CONTAINS
        assert fact.object == "key"
        assert fact.subject_type == NodeType.ROOM
        assert fact.object_type == NodeType.OBJECT
        assert fact.extraction_type == ExtractionType.DIRECT

    def test_invalid_optional_field_rejected(self):
        raw = [{
            "subject": "kitchen", 
            "relation": "contains", 
            "object": "key",
            "subject_type": "NOT_A_TYPE"
        }]
        assert self.validator.validate(raw, self.turn_id) == []

    def test_order_preserved(self):
        raw = [
            {"subject": "a", "relation": "contains", "object": "b"},
            {"subject": "c", "relation": "is_in", "object": "d"}
        ]
        facts = self.validator.validate(raw, self.turn_id)
        assert len(facts) == 2
        assert facts[0].subject == "a"
        assert facts[1].subject == "c"

    def test_field_mapping_object_to_target(self):
        raw = [{"subject": "a", "relation": "contains", "object": "b"}]
        facts = self.validator.validate(raw, self.turn_id)
        assert len(facts) == 1
        assert facts[0].object == "b"

    def test_enum_conversion(self):
        raw = [{"subject": "a", "relation": "connects_to", "object": "b"}]
        facts = self.validator.validate(raw, self.turn_id)
        assert facts[0].relation == RelationType.CONNECTS_TO
        
        # Test located_in alias mapping
        raw2 = [{"subject": "a", "relation": "located_in", "object": "b"}]
        facts2 = self.validator.validate(raw2, self.turn_id)
        assert facts2[0].relation == RelationType.LOCATED_IN

class TestConfidenceAssigner:
    def setup_method(self):
        self.assigner = ConfidenceAssigner()

    def _make_fact(self):
        return CandidateFact(
            subject="a",
            relation=RelationType.CONTAINS,
            object="b",
            confidence=0.0,
            source_turn_id=1,
            extraction_type=ExtractionType.DIRECT,
            extraction_method=ExtractionMethod.SLM
        )

    def test_single_fact_gets_09(self):
        fact = self._make_fact()
        result = self.assigner.assign([fact])
        assert result[0].confidence == 0.9

    def test_multiple_facts_all_get_09(self):
        facts = [self._make_fact(), self._make_fact()]
        result = self.assigner.assign(facts)
        assert result[0].confidence == 0.9
        assert result[1].confidence == 0.9

    def test_overwrites_existing_confidence(self):
        fact = self._make_fact()
        fact.confidence = 0.5
        result = self.assigner.assign([fact])
        assert result[0].confidence == 0.9

    def test_no_heuristics(self):
        fact = self._make_fact()
        # Even if it's the same fact processed multiple times, it shouldn't change
        self.assigner.assign([fact])
        self.assigner.assign([fact])
        assert fact.confidence == 0.9
