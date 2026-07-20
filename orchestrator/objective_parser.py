# orchestrator/objective_parser.py
# ============================================================================
# Parses TextWorld multi-step objectives into ordered sub-goals.
# Spec Reference: Implementation Plan v2.0, Section 8.3
# ============================================================================

from __future__ import annotations

import re
import logging
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Sequence markers TextWorld uses in objectives
_SEQUENCE_MARKERS = [
    r"first(?:\s+off)?",
    r"with that (?:accomplished|done|over(?: with)?)",
    r"(?:and )?then",
    r"after that",
    r"following that",
    r"after which",
    r"once (?:that's|that is) (?:all )?(?:handled|done|complete)",
    r"finally",
    r"next",
    r"(?:and )?(?:lastly|last)",
]

# Compile the split pattern
_SPLIT_PATTERN = re.compile(
    r"(?:^|[.!]\s+)(?:" + "|".join(_SEQUENCE_MARKERS) + r")[\s,]*",
    re.IGNORECASE,
)

# Action patterns to extract from sub-goal text
_SUBGOAL_ACTION_PATTERNS = [
    # "take a trip north" → go north
    (re.compile(r"take a trip\s+(north|south|east|west|up|down)", re.IGNORECASE),
     lambda m: f"go {m.group(1).lower()}"),
    # "go to the north" / "go to the south" → go south
    (re.compile(r"go\s+to\s+the\s+(north|south|east|west|up|down)", re.IGNORECASE),
     lambda m: f"go {m.group(1).lower()}"),
    # "head to the north" / "head north"
    (re.compile(r"head\s+(?:to\s+the\s+)?(north|south|east|west|up|down)", re.IGNORECASE),
     lambda m: f"go {m.group(1).lower()}"),
    # "go north/south/east/west"
    (re.compile(r"go\s+(north|south|east|west|up|down)", re.IGNORECASE),
     lambda m: f"go {m.group(1).lower()}"),
    # "ensure the safe is open" → open safe
    (re.compile(r"ensure (?:that )?(?:the |a |an )?(.+?) (?:is|are) (open|closed|locked|unlocked)",
                re.IGNORECASE),
     lambda m: f"{'open' if m.group(2).lower() == 'open' else 'close'} {m.group(1).strip().lower()}"),
    # "open/close the X"
    (re.compile(r"(open|close|unlock|lock) (?:the |a |an )?(.+?)(?:\.|$)", re.IGNORECASE),
     lambda m: f"{m.group(1).lower()} {m.group(2).strip().lower()}"),
    # "retrieve/take/get the X from the Y"
    (re.compile(r"(?:retrieve|take|get|grab) (?:the |a |an )?(.+?) from (?:the |a |an )?(.+?)(?:\.|$)",
                re.IGNORECASE),
     lambda m: f"take {m.group(1).strip().lower()}"),
    # "take/get the X"
    (re.compile(r"(?:retrieve|take|get|grab) (?:the |a |an )?(.+?)(?:\.|$)", re.IGNORECASE),
     lambda m: f"take {m.group(1).strip().lower()}"),
    # "insert/put the X in/inside/on the Y"
    (re.compile(r"(?:insert|put|place) (?:the |a |an )?(.+?) (?:in(?:side)?|on|into) (?:the |a |an )?(.+?)(?:\.|$)",
                re.IGNORECASE),
     lambda m: f"put {m.group(1).strip().lower()} in {m.group(2).strip().lower()}"),
    # "examine/look at the X"
    (re.compile(r"(?:examine|look at|inspect) (?:the |a |an )?(.+?)(?:\.|$)", re.IGNORECASE),
     lambda m: f"examine {m.group(1).strip().lower()}"),
    # "eat/cook the X"
    (re.compile(r"(eat|cook|slice|dice|chop) (?:the |a |an )?(.+?)(?:\.|$)", re.IGNORECASE),
     lambda m: f"{m.group(1).lower()} {m.group(2).strip().lower()}"),
]


@dataclass
class SubGoal:
    """A single parsed sub-goal from the objective."""
    index: int
    raw_text: str
    action: str         # Predicted TextWorld command
    completed: bool = False


class ObjectiveParser:
    """
    Parses TextWorld multi-step objectives into ordered sub-goals.

    Example objective:
      "First off, attempt to take a trip north. With that accomplished,
       ensure that the safe within the attic is open. And then, retrieve
       the shadfly from the safe inside the attic."

    Parsed into:
      [SubGoal(0, "take a trip north", "go north"),
       SubGoal(1, "ensure the safe is open", "open safe"),
       SubGoal(2, "retrieve the shadfly from the safe", "take shadfly")]
    """

    def __init__(self):
        self._sub_goals: List[SubGoal] = []
        self._current_index: int = 0

    def parse(self, objective: str) -> List[SubGoal]:
        """Parse the full objective string into ordered sub-goals."""
        if not objective:
            return []

        # Split on sequence markers
        parts = _SPLIT_PATTERN.split(objective)
        parts = [p.strip().rstrip(".!") for p in parts if p.strip()]

        # Filter out TextWorld boilerplate intro/outro
        # Strip trailing filler phrases before evaluating
        _TRAILING_FILLER = re.compile(
            r"\s*[.!]\s*(?:Got that\??|Good[.!]?|Have fun[.!]?|Good luck[.!]?)\s*$",
            re.IGNORECASE,
        )
        _PREAMBLE_ONLY = [
            "you are now playing",
            "profound game of textworld",
            "you've just entered textworld",
            "here is your task",
            "you can stop",
            "i hope you're ready",
            "good luck",
            "have fun",
        ]
        filtered = []
        for p in parts:
            # Strip trailing filler
            p = _TRAILING_FILLER.sub("", p).strip().rstrip(".!")
            p_lower = p.lower()
            # Skip if the whole segment is just preamble/outro
            if any(skip in p_lower for skip in _PREAMBLE_ONLY):
                continue
            # Skip fragments with no recognisable action verb
            has_action = any(kw in p_lower for kw in [
                "go ", "take ", "get ", "put ", "open ", "close ", "unlock ",
                "retrieve ", "insert ", "place ", "examine ", "eat ", "cook ",
                "head ", "north", "south", "east", "west",
            ])
            if not has_action:
                continue
            if len(p) > 5:  # Skip very short fragments
                filtered.append(p)

        # Extract action from each sub-goal
        self._sub_goals = []
        for i, text in enumerate(filtered):
            action = self._extract_action(text)
            self._sub_goals.append(SubGoal(
                index=i,
                raw_text=text,
                action=action,
            ))

        self._current_index = 0
        logger.info(
            f"ObjectiveParser: {len(self._sub_goals)} sub-goals parsed: "
            + ", ".join(sg.action for sg in self._sub_goals)
        )
        return self._sub_goals

    def get_current_subgoal(self) -> SubGoal | None:
        """Get the current (first incomplete) sub-goal."""
        for sg in self._sub_goals:
            if not sg.completed:
                return sg
        return None

    def mark_completed(self, index: int) -> None:
        """Mark a sub-goal as completed."""
        if 0 <= index < len(self._sub_goals):
            self._sub_goals[index].completed = True
            logger.info(f"Sub-goal {index} completed: {self._sub_goals[index].action}")

    def advance_if_completed(self, observation_text: str) -> bool:
        """
        Check if the current sub-goal was accomplished based on
        the observation, and advance if so.
        """
        current = self.get_current_subgoal()
        if current is None:
            return False

        obs_lower = observation_text.lower()
        action_lower = current.action.lower()

        # Check for success indicators
        success_patterns = [
            # Navigation success: we entered a new room
            (r"go (north|south|east|west|up|down)",
             [r"you've entered", r"you are in", r"you enter"]),
            # Take success
            (r"take (.+)", [r"you take", r"you pick up"]),
            # Open success
            (r"open (.+)", [r"you open", r"is now open", r"is already open"]),
            # Unlock success
            (r"unlock (.+)", [r"you unlock", r"is now unlocked"]),
            # Put success
            (r"put (.+) in (.+)", [r"you put"]),
        ]

        for action_pat, obs_pats in success_patterns:
            if re.match(action_pat, action_lower):
                for obs_pat in obs_pats:
                    if re.search(obs_pat, obs_lower):
                        self.mark_completed(current.index)
                        return True
        return False

    def _extract_action(self, text: str) -> str:
        """Extract a TextWorld command from sub-goal text."""
        for pattern, formatter in _SUBGOAL_ACTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return formatter(match)

        # Fallback: return raw text truncated
        return text[:50].lower().strip()

    @property
    def sub_goals(self) -> List[SubGoal]:
        return self._sub_goals.copy()

    @property
    def progress_summary(self) -> str:
        """Human-readable progress summary."""
        if not self._sub_goals:
            return "No objective parsed."
        completed = sum(1 for sg in self._sub_goals if sg.completed)
        total = len(self._sub_goals)
        current = self.get_current_subgoal()
        lines = [f"Progress: {completed}/{total} sub-goals completed"]
        if current:
            lines.append(f"Current: {current.action} (step {current.index + 1})")
        return "\n".join(lines)
