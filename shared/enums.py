# shared/enums.py
# ============================================================================
# All enumerations for the TextWorld AI Agent system.
# Spec Reference: Implementation Plan v2.0, Section 5 (Ontology) + Section 4 (Extractor)
# ============================================================================

from enum import Enum
from typing import Dict, FrozenSet


# ── Node Types ──────────────────────────────────────────────────────────────

class NodeType(Enum):
    """Types of entities in the world model graph."""
    ROOM = "room"
    OBJECT = "object"
    CHARACTER = "character"
    STATE = "state"


# ── Node / Edge Status ──────────────────────────────────────────────────────

class NodeStatus(Enum):
    """Belief status for a node in the world model."""
    ACTIVE = "active"           # Currently believed to exist
    SUPERSEDED = "superseded"   # Replaced by newer belief
    UNKNOWN = "unknown"         # Existence uncertain


class EdgeStatus(Enum):
    """Belief status for an edge (fact) in the world model."""
    ACTIVE = "active"           # Currently believed true
    SUPERSEDED = "superseded"   # Replaced by newer observation
    UNKNOWN = "unknown"         # Truth value uncertain


# ── Relation Types ──────────────────────────────────────────────────────────

class RelationType(Enum):
    """Types of relationships (edges) in the world model graph."""
    CONTAINS = "contains"           # Room/Container → Object/Character
    CONNECTS_TO = "connects_to"     # Room → Room (with optional direction)
    LOCATED_IN = "located_in"       # Object/Character → Room
    HOLDS = "holds"                 # Character → Object (inventory)
    HAS_STATE = "has_state"         # Object → State value
    IS_TYPE = "is_type"             # Object → Type category


# Set of valid relation strings for fast validation
ALLOWED_RELATIONS: frozenset = frozenset(r.value for r in RelationType)



# ── Text Segment Types (Extractor Stage 1) ─────────────────────────────────

class SegmentType(Enum):
    """Classification of observation text segments."""
    ROOM_DESCRIPTION = "room_description"   # "You are in the...", "You see..."
    INVENTORY_UPDATE = "inventory_update"   # "You pick up...", "You drop..."
    NAVIGATION = "navigation"               # "You go north...", "You enter..."
    STATE_CHANGE = "state_change"           # "The door is now...", "You unlock..."
    FEEDBACK = "feedback"                   # "That doesn't work", "You can't..."
    OBJECTIVE_INFO = "objective_info"        # Quest objective text


# ── Extraction Types ────────────────────────────────────────────────────────

class ExtractionType(Enum):
    """How a fact was extracted from text."""
    DIRECT = "direct"       # Explicitly stated: "There is a key on the table"
    IMPLIED = "implied"     # Inferred: going north implies room connectivity
    NEGATION = "negation"   # Stated absence: "There is nothing here"


class ExtractionMethod(Enum):
    """Which extraction mechanism produced a fact."""
    SLM = "slm"                     # SLM-guided structured extraction


# ── State Conflicts (Mutual Exclusion Groups) ──────────────────────────────

# Values within each group are mutually exclusive — observing one supersedes another.
STATE_CONFLICTS: Dict[str, FrozenSet[str]] = {
    "open_closed":  frozenset({"open", "closed"}),
    "lock_state":   frozenset({"locked", "unlocked"}),
    "switch":       frozenset({"on", "off"}),
    "cook_state":   frozenset({"raw", "cooked", "burned", "fried", "roasted", "grilled"}),
    "cut_state":    frozenset({"uncut", "chopped", "sliced", "diced"}),
}

# All known state values (union of all conflict groups + misc)
KNOWN_STATES: frozenset = frozenset(
    state
    for group in STATE_CONFLICTS.values()
    for state in group
) | frozenset({"edible", "inedible", "portable", "fixed", "hot", "cold"})


# ── Contradiction Result Types ──────────────────────────────────────────────

class ContradictionType(Enum):
    """Result of checking a candidate fact against existing beliefs."""
    EXPAND = "expand"               # No existing edge — add new fact
    CORROBORATE = "corroborate"     # Same fact already exists — reinforce
    REVISE = "revise"               # Contradicting fact — apply revision policy


class ConflictCategory(Enum):
    """Category of detected conflict."""
    STATE_MUTUAL_EXCLUSION = "state_mutual_exclusion"   # e.g., locked vs unlocked
    LOCATION_CHANGE = "location_change"                 # Object moved rooms
    VALUE_MISMATCH = "value_mismatch"                   # General value difference


# ── Revision Action Types ───────────────────────────────────────────────────

class RevisionActionType(Enum):
    """Action taken by the revision policy."""
    EXPAND = "expand"                           # Add new fact
    CORROBORATE = "corroborate"                 # Reinforce existing fact
    SUPERSEDE = "supersede"                     # Hard supersede (R1: recency-wins)
    PROVISIONAL_SUPERSEDE = "provisional_supersede"  # Cautious supersede (R2: corroboration-override)
