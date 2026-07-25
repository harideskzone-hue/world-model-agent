class SLMAgent:
    """
    Calls the local SLM to select the next action.
    Spec: SLM_INTERFACES.md Section 2, MODULE_RESPONSIBILITIES.md
    """
    
    def __init__(self, slm_client):
        self.slm = slm_client
    
    def decide(self, prompt: str) -> str:
        """
        Call the SLM with a formatted prompt and return the raw response.
        
        INPUT: Formatted prompt string (from Prompt Builder)
        OUTPUT: Raw SLM text (should be JSON with "action" field, but may be malformed)
        """
        return self.slm.generate(prompt=prompt, temperature=0.0)
