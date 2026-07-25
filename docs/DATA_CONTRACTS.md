# Data Contracts (Language‑agnostic)

All contracts are expressed as plain data structures (think TypeScript interfaces or UML structs).  
s: name, type, required (✓) or optional (prod
Optional key invariants.

---

Observation
 – Description, Producer, Consumer, Key Invariant.

## Observation
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| feedback | string | ✓ | Text response to the last action |
| description | string | ✓ | Output of `look` – current room description |
| inventory | string | ✓ | Player’s inventory (free‑form) |
| objective | string | ✓ | Quest objective (set at reset) |
| score | int | ○ | Current game score |
| max_score | int | ○ | Maximum possible score |
| won | bool | ✓ | Win flag |
| lost | bool | ✓ | Lose flag |
| turn_id | int | ✓ | 0‑indexed turn counter |

**Producer**: `TextWorldWrapper.step()` / `reset()`  
**Consumer**: Extractor, Orchestrator (working memory)  
**Invariants**: `won` ⊕ `lost` never both true; `turn_id` increments by exactly 1 each turn.

---

## CandidateFact
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| subject | string | ✓ | Normalized entity name (lower‑case, no articles) |
| relation | RelationType | ✓ | Ontology relation (e.g., CONTAINS, IS_IN, HAS_STATE) |
| target | string | ✓ | Target entity name or literal state value |
| confidence | float ∈ [0,1] | ✓ | Belief strength (fixed 0.9 for SLM facts after rebuild) |
| source_turn_id | int | ✓ | Turn when fact was extracted |
| extraction_method | ExtractionMethod (only `SLM`) | ✓ | Production method |
| subject_type | NodeType? | ○ | Explicit node type for subject (if known) |
| target_type | NodeType? | ○ | Explicit node type for target (if known) |
| extraction_type | ExtractionType? | ○ | DIRECT, IMPLIED, NEGATION (optional) |

**Producer**: SLMExtractor → JSON Parser → Schema Validator → Confidence Assigner  
**Consumer**: Updater  
**Invariants**: confidence ∈ [0,1]; subject/target normalized; if *_type present must be a valid NodeType.

---

## Node
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | string | ✓ | Normalized identifier (same rules as `subject`) |
| name | string | ✓ | Human‑readable display name |
| node_type | NodeType (`ROOM|OBJECT|DOOR|PLAYER|NPC|ITEM|STATE`) | ✓ | Ontology class |
| status | NodeStatus (`ACTIVE`) | ✓ | Belief status |
| confidence | float ∈ [0,1] | ✓ | Belief strength |
| first_observed_turn | int | ✓ | Turn first seen |
| last_observed_turn | int | ✓ | Most recent turn observed |
| corroboration_count | int ≥0 | ✓ | Independent observation count |
| attributes | Map<string, any> | ○ | Arbitrary key‑value properties (bool, int, float, string) |

**Producer**: Updater (`add_node`)  
**Consumer**: Query Layer, GraphStore (read‑only), any reader  
**Invariants**: id = normalized(name); last_observed_turn ≥ first_observed_turn; exactly one node with node_type = PLAYER exists at any time; confidence ∈ [0,1].

---

## Edge
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | string | ✓ | Unique edge identifier (UUID‑8) |
| subject | string | ✓ | source node `id` |
| relation | RelationType | ✓ | Ontology relation |
| target | string | ✓ | target node `id` **or** literal state value |
| confidence | float ∈ [0,1] | ✓ | Belief strength |
| source_turn_id | int | ✓ | Turn when fact observed |
| extraction_method | ExtractionMethod (`SLM`) | ✓ | Production method |
| t_observed | int | ✓ | Episodic time (same as source_turn_id) |
| t_valid_from | int | ✓ | Semantic start time |
| t_valid_until | int? | ○ | Semantic end time (`null` = still active) |
| status | EdgeStatus (`ACTIVE` | `SUPERSEDED`) | ✓ | Current belief status |
| corroboration_count | int ≥0 | ✓ | Independent confirmations |
| superseded_by | string? | ○ | ID of edge that supersedes this one (if SUPERSEDED) |
| revision_reason | string? | ○ | Human‑readable reason for supersession |
| direction | string? | ○ | For `connects_to`: compass direction (`north`, `south`, …) |

**Producer**: Updater (`add_edge` / `supersede_edge`)  
**Consumer**: Query Layer, GraphStore, Updater (contradiction check)  
**Invariants**:  
- If status = ACTIVE → superseded_by = null and t_valid_until = null.  
- If status = SUPERSEDED → superseded_by points to an existing edge and t_valid_until ≤ current turn.  
- t_valid_from ≤ t_observed.  
- confidence ∈ [0,1].  
- No two **active** edges share identical (subject, relation, target).

---

## WorldModel (belief‑revision + ontology)
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| (no storage) | – | – | Provides methods: apply_fact(CandidateFact), get_active_edges(...), get_nodes_by_type(...), temporal_valid(edge, turn), etc. |
| current_turn | int | ✓ | Turn number of the most recent processed observation (mirrors GraphStore.current_turn) |

**Producer**: Updater (after delegating to GraphStore)  
**Consumer**: Query Layer (needs to read beliefs), any module requiring logical queries  
**Invariants**: Delegates all persistence to GraphStore; maintains no extra cached state.

---

## GraphStore (storage + indexes)
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| nodes | Map<string, Node> | ✓ | All nodes keyed by `id` |
| edges | Map<string, Edge> | ✓ | All edges keyed by `id` |
| indexes | `{ subjectIdx: Map<string, Set<string>>, objectIdx: Map<string, Set<string>>, slotIdx: Map<(string,RelationType), Set<string>> }` | ✓ | Internal lookup structures (kept in sync) |
| current_turn | int | ✓ | Turn number of the most recent processed observation |

**Producer**: Updater (via add_node/add_edge/supersede_edge)  
**Consumer**: WorldModel (for reads), Query Layer (needs to iterate over relevant subgraphs)  
**Invariants**:  
- Every Node.id appears as a key in `nodes`; every Edge.id appears as a key in `edges`.  
- Indexes are always consistent with the underlying maps (internal invariant).  
- current_turn equals the turn of the last applied fact.

---

## Working Memory (transient execution state)
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| previous_action | string? | ○ | The action string executed in the previous turn |
| retry_count | int | ✓ | Number of retries needed to obtain a valid SLM JSON in the current turn |
| current_sub_goal | string? | ○ | Short description of the sub‑goal being pursued (derived from objective & progress) |
| last_observation | string? | ○ | Raw observation text from the previous turn (for debugging) |
| objective | string | ✓ | Copied from Observation.objective (constant during episode) |

**Producer**: Orchestrator (after each turn, before calling Prompt Builder)  
**Consumer**: Prompt Builder (may include in prompt for context, but must not treat as persistent memory)  
**Invariants**: All fields are cleared/reset at the start of a new episode; they never survive beyond the current turn.

---

## ContextSlice (output of Query Layer, input to Prompt Builder)
All values are **human‑readable strings** (node names or literal state values).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| objective | string | ✓ | Copied from Observation.objective |
| current_room | string | ✓ | Name of the room where the player resides |
| reachable_rooms | string[] | ✓ | Names of rooms reachable via an active `connects_to` edge (depth = 1) |
| inventory | string[] | ✓ | Names of objects the player holds |
| objects_in_current_room | string[] | ✓ | Names of objects present in the current room.
| locked_doors | string[] | ✓ | Names of doors in the current room whose state is `Locked` (or `Closed` per ontology) |
| recent_changes | string[]? | ○ | IDs or short descriptions of edges added/updated in the last *N* turns (debug only) |

**Producer**: QueryLayer.retrieve(workingMemory, turnId)  
**Consumer**: Prompt Builder  
**Invariants**: Every string must correspond to an existing Node.name (or a literal state value) in the current GraphStore; no duplicates inside a list; current_room must be a node with node_type = ROOM.

---

## SLMDecision
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| action_text | string | ✓ | Verb phrase chosen as the next action |
| restated_sub_goal | string? | ○ | Optional one‑sentence clarification (debug) |
| reasoning_summary | string? | ○ | One‑sentence explanation of why the action was chosen (debug/telemetry) |
| raw_output | string? | ○ | Full SLM response before parsing |
| latency_ms | float? | ○ | Inference time in ms |
| retry_count | int? | ○ | Number of retries needed to obtain valid JSON |

**Producer**: SLM Agent (`ActionSelector`)  
**Consumer**: Action Parser  
**Invariants**: action_text non‑empty after trim (validated by Action Parser).

---

## ActionResult (output of Environment Validator)
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| action | string | ✓ | String to send to `env.step()` |
| valid | bool | ✓ | `true` if the action is syntactically valid and non‑empty |
| error | string? | ○ | Explanation when `valid = false` |

**Producer**: Environment Validator (after parsing SLM JSON)  
**Consumer**: `TextWorldWrapper.step()`  
**Invariants**: If valid = true → error must be null/empty and action non‑empty; if valid = false → action may be empty and error non‑empty.

---

## Enums
- **NodeType**: ROOM, OBJECT, DOOR, PLAYER, NPC, ITEM, STATE  
- **NodeStatus**: ACTIVE (future‑proof)  
- **RelationType**: e.g., CONTAINS, IS_IN, HAS_STATE, CONNECTS_TO (direction stored separately) – *Note: LOCKED/OPEN/CLOSED are modeled via HAS_STATE + STATE node*  
- **ExtractionMethod**: only `SLM` (enum kept for extensibility)  
- **ExtractionType**: DIRECT, IMPLIED, NEGATION (optional)  
- **EdgeStatus**: ACTIVE, SUPERSEDED  

---
