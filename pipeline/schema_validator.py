from typing import List
from shared.models import CandidateFact
from shared.enums import RelationType, NodeType, ExtractionType, ExtractionMethod

class SchemaValidator:
    """
    Stage 3: Strict ontology conformance.
    """
    
    RELATION_MAPPING = {
        "contains": RelationType.CONTAINS,
        "is_in": RelationType.LOCATED_IN,
        "located_in": RelationType.LOCATED_IN,  # Fallback for old output
        "located_at": getattr(RelationType, "LOCATED_AT", RelationType.LOCATED_IN),
        "connects_to": RelationType.CONNECTS_TO,
        "carries": getattr(RelationType, "CARRIES", RelationType.HOLDS),
        "holds": RelationType.HOLDS,
        "has_state": RelationType.HAS_STATE,
    }
    
    ALLOWED_FIELDS = {
        "subject", "relation", "object", 
        "subject_type", "object_type", "extraction_type"
    }

    def validate(self, raw_objects: List[dict], turn_id: int) -> List[CandidateFact]:
        valid_facts = []
        for obj in raw_objects:
            # 1. Reject any object with extra/unrecognized fields
            if not isinstance(obj, dict):
                continue
                
            obj_keys = set(obj.keys())
            if not obj_keys.issubset(self.ALLOWED_FIELDS):
                continue
                
            # 2. Check required fields
            if "subject" not in obj or "relation" not in obj or "object" not in obj:
                continue
                
            # 3. Convert relation string to enum
            relation_str = obj.get("relation")
            if not isinstance(relation_str, str):
                continue
                
            relation_str = relation_str.lower()
            if relation_str not in self.RELATION_MAPPING:
                continue
                
            relation_enum = self.RELATION_MAPPING[relation_str]
            
            # 4. Validate optional fields
            subject_type_str = obj.get("subject_type")
            target_type_str = obj.get("object_type")
            extraction_type_str = obj.get("extraction_type")
            
            subject_type = None
            if subject_type_str is not None:
                if not isinstance(subject_type_str, str):
                    continue
                try:
                    subject_type = NodeType[subject_type_str.upper()]
                except KeyError:
                    continue
                    
            target_type = None
            if target_type_str is not None:
                if not isinstance(target_type_str, str):
                    continue
                try:
                    target_type = NodeType[target_type_str.upper()]
                except KeyError:
                    continue
                    
            extraction_type = None
            if extraction_type_str is not None:
                if not isinstance(extraction_type_str, str):
                    continue
                try:
                    extraction_type = ExtractionType[extraction_type_str.upper()]
                except KeyError:
                    continue

            # 5. Create CandidateFact
            fact = CandidateFact(
                subject=str(obj["subject"]),
                subject_type=subject_type,
                relation=relation_enum,
                object=str(obj["object"]),
                object_type=target_type,
                extraction_type=extraction_type,
                extraction_method=ExtractionMethod.SLM,
                source_turn_id=turn_id,
                confidence=0.0  # Will be assigned in the next stage
            )
            valid_facts.append(fact)
            
        return valid_facts
