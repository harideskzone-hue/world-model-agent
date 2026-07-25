from dataclasses import dataclass, field
from typing import List

@dataclass
class ContextSlice:
    """
    Data contract for the state extracted by the Query Layer.
    Only contains human-readable names, no internal IDs.
    """
    objective: str
    current_room: str
    reachable_rooms: List[str] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    objects_in_current_room: List[str] = field(default_factory=list)
    locked_doors: List[str] = field(default_factory=list)
    recent_changes: List[str] = field(default_factory=list)
