class SLMExtractor:
    """
    Calls the local SLM to extract facts from a raw observation.
    Spec: SLM_INTERFACES.md Section 1
    """
    
    def __init__(self, slm_client):
        """
        Args:
            slm_client: Local SLM client (e.g., SLMRunner)
            Must support: client.generate(prompt=..., temperature=...) -> str
        """
        self.slm = slm_client
    
    def extract(self, observation: str) -> str:
        """
        Build the extractor prompt and call the SLM.
        
        INPUT: Raw observation string
        OUTPUT: Raw SLM text
        """
        prompt = self._build_extractor_prompt(observation)
        raw_response = self.slm.generate(prompt=prompt, temperature=0.0)
        return raw_response
    
    def _build_extractor_prompt(self, observation: str) -> str:
        """Zero-shot extraction prompt without static examples or keywords."""
        return f"""You are a strict data extractor for a TextWorld environment.
Your ONLY task is to extract facts from the observation text and return them as a valid JSON array using zero-shot semantic comprehension.

SCHEMA:
- subject: entity name (lowercase)
- subject_type: strictly one of [ROOM, OBJECT, CHARACTER]
- relation: strictly one of [contains, connects_to, located_in, carries, has_state]
- object: target entity name or state value (lowercase)
- object_type: strictly one of [ROOM, OBJECT, CHARACTER, STATE]
- extraction_type: strictly one of [direct, implied, negation]

RULES:
1. Output ONLY a valid JSON array. No explanations, no markdown blocks, no conversational text.
2. If no facts are present, output an empty array: []
3. Perform direct semantic inference without relying on hardcoded examples or keywords.

TEXT: "{observation}"
FACTS:"""
