import json
from typing import List

class JSONParser:
    """
    Stage 2: Fail-safe JSON extraction.
    """
    def parse(self, raw_text: str) -> List[dict]:
        """
        1. Attempt `json.loads(raw_text)`.
        2. If parse fails (JSONDecodeError or TypeError), return `[]`.
        3. If parsed object is not a list, return `[]`.
        4. Return the parsed list of dictionaries.
        """
        try:
            data = json.loads(raw_text)
            if not isinstance(data, list):
                return []
            return data
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
