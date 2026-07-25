import json

class ActionParser:
    """
    Extracts the "action" field from SLM's JSON response.
    Spec: MODULE_RESPONSIBILITIES.md
    """
    
    def parse(self, raw_response: str) -> str:
        """
        Parse SLM's JSON and extract the "action" field.
        
        INPUT: Raw SLM text
        OUTPUT: Action string (trimmed), or empty string if missing
        """
        try:
            data = json.loads(raw_response)
            if not isinstance(data, dict):
                return ""
            action = data.get("action", "")
            if not isinstance(action, str):
                return ""
            return action.strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            return ""
