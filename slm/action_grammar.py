# slm/action_grammar.py
# ============================================================================
# Fixed action grammar — verb templates the agent can produce.
# Spec Reference: Implementation Plan v2.0, Section 8.2 (ActionGrammar)
#
# This is NOT TextWorld's admissible commands list.
# The agent generates actions from this fixed grammar + world model entities.
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ActionGrammar:
    """
    Fixed set of verb templates the agent can produce.
    NOT TextWorld's admissible list — that's forbidden in the live agent.
    """
    verbs: List[str] = field(default_factory=lambda: [
        "look",
        "inventory",
        "go {direction}",
        "take {object}",
        "drop {object}",
        "open {object}",
        "close {object}",
        "unlock {object} with {object}",
        "examine {object}",
        "put {object} on {object}",
        "put {object} in {object}",
        "eat {object}",
        "cook {object}",
        "slice {object}",
        "dice {object}",
        "chop {object}",
    ])

    directions: List[str] = field(default_factory=lambda: [
        "north", "south", "east", "west", "up", "down",
    ])

    def get_grammar_description(self) -> str:
        """Return a compact text description of available actions."""
        lines = ["Available action templates:"]
        for verb in self.verbs:
            lines.append(f"  - {verb}")
        lines.append(f"Directions: {', '.join(self.directions)}")
        return "\n".join(lines)

    def get_simple_action_list(self) -> str:
        """Return a very compact list for fallback prompts."""
        simple = [
            "look", "inventory",
            "go north/south/east/west",
            "take [item]", "drop [item]",
            "open [item]", "close [item]",
            "examine [item]", "eat [item]",
        ]
        return ", ".join(simple)
