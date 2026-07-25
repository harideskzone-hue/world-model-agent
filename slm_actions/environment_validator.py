from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ActionResult:
    action: str
    valid: bool
    error: Optional[str] = None

class EnvironmentValidator:
    """
    Validates that an action is syntactically valid and admissible.
    Spec: MODULE_RESPONSIBILITIES.md, DATA_CONTRACTS.md
    """
    
    def validate(self, action: str, admissible_commands: Optional[List[str]] = None) -> ActionResult:
        """
        Validate an action string against the environment's admissible commands.
        """
        action = action.strip()
        if not action:
            return ActionResult(action=action, valid=False, error="Action is empty")
            
        if admissible_commands is not None:
            if action not in admissible_commands:
                return ActionResult(action=action, valid=False, error="Action not in admissible list")
                
        return ActionResult(action=action, valid=True, error=None)
