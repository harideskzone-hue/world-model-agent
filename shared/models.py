# shared/models.py
# ============================================================================
# All shared dataclasses for inter-module communication.
# Spec Reference: Implementation Plan v2.0, Section 9 (Shared Data Models)
#
# Data flow:
#   Observation → CandidateFact → UpdateReport → Edge → ContextSlice → SLMDecision → TurnLog
# ============================================================================

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from shared.enums import (
    NodeType, NodeStatus, EdgeStatus, RelationType,
    SegmentType, ExtractionType, ExtractionMethod,
    ContradictionType, ConflictCategory, RevisionActionType,
)


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT LAYER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Observation:
    """
    Structured observation from the TextWorld environment wrapper.
    Produced by: TextWorldWrapper.step() / .reset()
    Consumed by: Extractor, Orchestrator
    """
    feedback: str               # Text response to last action
    description: str            # Current room description (output of 'look')
    inventory: str              # Player's inventory text
    location: str               # Current room name (string)
    objective: str              # Quest objective text
    score: int = 0              # Current game score
    max_score: int = 0          # Maximum possible score
    won: bool = False           # Win flag
    lost: bool = False          # Lose flag
    turn_id: int = 0            # Sequential turn counter (0-indexed)


@dataclass
class GroundTruthFact:
    """
    Mirrors TextWorld's internal Proposition format.
    EVALUATION ONLY — never consumed by live agent modules.
    """
    name: str                   # Predicate name: "at", "in", "on", "open", "locked"
    arguments: List[str] = field(default_factory=list)  # Entity names


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTOR LAYER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TextSegment:
    """
    A preprocessed text segment with resolved coreferences.
    Produced by: Preprocessor (Stage 1)
    Consumed by: SLMExtractor (Stage 2)
    """
    raw_text: str                           # Original sentence
    resolved_text: str                      # After coreference resolution
    segment_type: SegmentType               # ROOM_DESCRIPTION, NAVIGATION, etc.
    referenced_entities: List[str] = field(default_factory=list)
    source_turn_id: int = 0


@dataclass
class RawExtraction:
    """
    Unvalidated extraction output from SLM or rule-based fallback.
    Produced by: SLMExtractor (Stage 2) or RuleFallback (Stage 2b)
    Consumed by: SchemaValidator (Stage 3)
    """
    subject: str
    relation: str
    object: str
    subject_type: Optional[str] = None
    object_type: Optional[str] = None
    extraction_type: str = "direct"         # "direct" | "implied" | "negation"
    source_segment: Optional[TextSegment] = None
    extraction_method: str = "slm"          # "slm"


@dataclass
class CandidateFact:
    """
    A fully validated, confidence-scored candidate fact ready for the Updater.
    Produced by: TextExtractor (full 4-stage pipeline)
    Consumed by: Updater
    """
    subject: str                            # Normalized entity name (lowercase, no articles)
    relation: RelationType                  # Validated relation type
    object: str                             # Target entity or state value
    confidence: float                       # [0.10, 0.99]
    source_turn_id: int                     # Turn that produced this
    extraction_type: ExtractionType         # DIRECT | IMPLIED | NEGATION
    extraction_method: ExtractionMethod     # SLM
    subject_type: Optional[NodeType] = None # Explicit node type from extractor
    object_type: Optional[NodeType] = None  # Explicit node type from extractor


# ═══════════════════════════════════════════════════════════════════════════
# WORLD MODEL LAYER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    """
    An entity in the world model graph (room, object, or character).
    """
    id: str                                     # Unique identifier (normalized name)
    name: str                                   # Display name
    node_type: NodeType                         # ROOM | OBJECT | CHARACTER
    status: NodeStatus = NodeStatus.ACTIVE
    confidence: float = 0.5                     # Belief confidence [0.0–1.0]
    first_observed_turn: int = 0                # When first seen
    last_observed_turn: int = 0                 # Most recent observation
    corroboration_count: int = 0                # Independent observation count
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Edge:
    """
    A fact (relationship) in the world model graph.
    Bi-temporal, versioned, non-destructive — following the Graphiti pattern.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subject: str = ""                           # Source node ID
    relation: RelationType = RelationType.CONTAINS
    object: str = ""                            # Target node ID or state value

    # ── Confidence & Provenance ──
    confidence: float = 0.5                     # [0.0–1.0]
    source_turn_id: int = 0                     # Turn that produced this edge
    extraction_method: ExtractionMethod = ExtractionMethod.SLM

    # ── Bi-Temporal Fields ──
    t_observed: int = 0                         # Episodic time: when extracted
    t_valid_from: int = 0                       # Semantic time: when belief became active
    t_valid_until: Optional[int] = None         # None = still active; int = superseded at turn N

    # ── Status & Revision ──
    status: EdgeStatus = EdgeStatus.ACTIVE
    corroboration_count: int = 1                # How many times independently confirmed
    superseded_by: Optional[str] = None         # ID of the edge that replaced this one
    revision_reason: Optional[str] = None       # "superseded: newer observation at turn 42"

    # ── Direction (for connects_to edges) ──
    direction: Optional[str] = None             # "north", "south", etc.


@dataclass
class GraphStats:
    """
    Diagnostic statistics for the world model graph.
    """
    total_nodes: int = 0
    total_edges: int = 0
    active_edges: int = 0
    superseded_edges: int = 0
    storage_bytes: int = 0
    avg_confidence: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# UPDATER LAYER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ContradictionResult:
    """
    Result of checking a candidate fact against existing beliefs.
    Produced by: ContradictionDetector
    Consumed by: RevisionPolicy
    """
    contradiction_type: ContradictionType       # EXPAND | CORROBORATE | REVISE
    existing_edge: Optional[Edge] = None        # The conflicting edge (if any)
    conflict_category: Optional[ConflictCategory] = None


@dataclass
class RevisionAction:
    """
    Decision made by the revision policy about how to handle a candidate fact.
    Produced by: RevisionPolicy
    Consumed by: Updater (to execute the action on the GraphStore)
    """
    action_type: RevisionActionType             # EXPAND | CORROBORATE | SUPERSEDE | PROVISIONAL
    existing_edge_id: Optional[str] = None      # Edge being superseded (if any)
    new_edge: Optional[Edge] = None             # Edge being created
    revision_reason: str = ""                   # Human-readable log string


@dataclass
class UpdateReport:
    """
    Summary of all updates applied in a single turn.
    Produced by: Updater
    Consumed by: Orchestrator (for logging and budget enforcement)
    """
    turn_id: int = 0
    expanded: int = 0                           # New facts added
    corroborated: int = 0                       # Existing facts reinforced
    revised: int = 0                            # Contradictions resolved
    rejected: int = 0                           # Invalid candidates dropped
    revisions: List[RevisionAction] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# QUERY LAYER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnchorNode:
    """
    A starting point for graph traversal in the Query Layer.
    """
    id: str                                     # Node ID
    anchor_type: str                            # "location" | "goal" | "inventory"
    priority: float = 1.0                       # Higher = traversed first


@dataclass
class ContextSlice:
    """
    The bounded, formatted context slice delivered to the SLM.
    Produced by: QueryLayer
    Consumed by: ActionSelector
    """
    formatted_text: str                         # Ready to insert into SLM prompt
    included_facts: List[Edge] = field(default_factory=list)
    excluded_count: int = 0                     # How many edges were truncated
    total_tokens_estimate: int = 0              # Estimated token count
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# SLM LAYER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SLMDecision:
    """
    The SLM's output: a chosen action and optional sub-goal restatement.
    Produced by: ActionSelector
    Consumed by: Orchestrator
    """
    action_text: str                            # "take brass key", "go north"
    restated_sub_goal: Optional[str] = None     # "Now find the exit"
    raw_output: str = ""                        # Full SLM response (debugging)
    latency_ms: float = 0.0                     # Inference time
    retry_count: int = 0                        # How many retries were needed


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR LAYER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WorkingMemory:
    """
    The agent's per-turn working context, built from the current graph state.
    Produced by: Orchestrator.get_working_memory()
    Consumed by: Extractor (for coreference), QueryLayer (for anchoring)
    """
    current_room: str = ""
    current_room_facts: List[Edge] = field(default_factory=list)
    inventory_facts: List[Edge] = field(default_factory=list)
    recent_observations: List[str] = field(default_factory=list)  # Last 3 turns
    failed_actions: List[str] = field(default_factory=list)       # History of failed actions to avoid repeating
    current_sub_goal: str = ""
    objective: str = ""


@dataclass
class TurnLog:
    """
    Complete record of a single turn for evaluation and visualization.
    """
    turn_id: int = 0
    observation: Optional[Observation] = None
    extracted_facts: List[CandidateFact] = field(default_factory=list)
    update_report: Optional[UpdateReport] = None
    context_slice: Optional[ContextSlice] = None
    slm_decision: Optional[SLMDecision] = None
    reward: float = 0.0
    done: bool = False
    wall_clock_ms: float = 0.0
    world_model_size_bytes: int = 0


@dataclass
class EpisodeResult:
    """
    Complete result of running one episode (game).
    """
    game_path: str = ""
    total_turns: int = 0
    final_score: int = 0
    max_score: int = 0
    won: bool = False
    turn_logs: List[TurnLog] = field(default_factory=list)
    final_world_model_json: str = ""            # Serialized graph for inspection
    total_wall_clock_seconds: float = 0.0
