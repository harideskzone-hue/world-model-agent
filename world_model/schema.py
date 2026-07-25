# world_model/schema.py
# ============================================================================
# World Model ontology constants and schema utilities.
# Spec Reference: Implementation Plan v2.0, Section 5.4 (State Values)
# ============================================================================

from shared.enums import STATE_CONFLICTS, KNOWN_STATES, RelationType


# ── Schema Utilities ────────────────────────────────────────────────────────

def are_states_conflicting(state_a: str, state_b: str) -> bool:
    """
    Check if two state values are mutually exclusive.

    Returns True if both values belong to the same conflict group.
    E.g., "open" and "closed" → True; "open" and "hot" → False.
    """
    if state_a == state_b:
        return False
    for group in STATE_CONFLICTS.values():
        if state_a in group and state_b in group:
            return True
    return False


def is_known_state(state: str) -> bool:
    """Check if a state value is in the known vocabulary."""
    return state.lower() in KNOWN_STATES


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity names: lowercase, strip articles, collapse whitespace.
    "The Brass Key" → "brass key"
    """
    articles = {"the", "a", "an"}
    words = name.strip().lower().split()
    words = [w for w in words if w not in articles]
    res = " ".join(words) if words else name.strip().lower()
    if res in ("you", "i", "me", "self", "agent", "player", "the player", "player's inventory"):
        return "player"
    return res


def is_single_occupancy_relation(relation: RelationType) -> bool:
    """
    Check if a relation type implies single occupancy for the subject.
    E.g., an object can only be located_in ONE room at a time.
    A character can only be located_in ONE room at a time.
    """
    return relation in {RelationType.LOCATED_IN}


def is_state_relation(relation: RelationType) -> bool:
    """Check if this relation tracks object state (subject to conflict groups)."""
    return relation == RelationType.HAS_STATE
