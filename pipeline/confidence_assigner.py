from typing import List
from shared.models import CandidateFact

class ConfidenceAssigner:
    """
    Stage 4: Fixed confidence application.
    """
    def assign(self, facts: List[CandidateFact]) -> List[CandidateFact]:
        """
        Assign a fixed confidence of 0.9 to each fact.
        """
        for fact in facts:
            fact.confidence = 0.9
        return facts
