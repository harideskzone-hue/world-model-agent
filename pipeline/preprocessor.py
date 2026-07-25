import re

class MinimalPreprocessor:
    """
    Stage 1: Cleans raw observation strings deterministically.
    """
    def preprocess(self, observation: str) -> str:
        """
        1. Decode as UTF-8 (or assert already UTF-8)
        2. Strip leading/trailing whitespace
        3. Replace multiple spaces/newlines with single space
        4. Return empty string if result is empty/whitespace-only
        5. Return cleaned string
        """
        if not observation:
            return ""

        # Ensure valid UTF-8 by encoding and decoding
        try:
            if isinstance(observation, bytes):
                observation = observation.decode('utf-8', 'ignore')
            else:
                observation = observation.encode('utf-8', 'ignore').decode('utf-8')
        except Exception:
            pass
            
        # Replace multiple spaces/newlines with a single space
        cleaned = re.sub(r'\s+', ' ', observation)
        
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
