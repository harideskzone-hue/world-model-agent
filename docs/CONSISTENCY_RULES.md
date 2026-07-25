# Consistency Rules (Ontology Constraints)

These rules are **static constraints** on the graph structure. They are checked by the Updater before applying a new fact; violations are treated as contradictions that trigger the appropriate update rule (usually supersede).

## Mutually Exclusive States
A single entity cannot simultaneously hold two mutually exclusive state values.

- For any entity `E` (node of type OBJECT, DOOR, etc.) and state property `P` (represented by a STATE node), if there exists an active edge  
  `E --HAS_STATE--> S1` and another active edge  
  `E --HAS_STATE--> S2` where `S1.name ≠ S2.name` and the pair `(S1.name, S2.name)` is listed as mutually exclusive, then the newer edge **supersedes** the older one.

**Pre‑defined mutually exclusive pairs** (can be extended per game):
- (`"locked"`, `"unlocked"`)
- (`"open"`, `"closed"`)
- (`"on"`, `"off"`)
- (`"intact"`, `"broken"`)

## Impossible Relations
Certain relation–target combinations are nonsensical and must be rejected.

| Subject Type | Relation | Invalid Target Types |
|--------------|----------|----------------------|
| ROOM | HAS_STATE | Invalid (rooms do not have states in TextWorld). |
| OBJECT | CONTAINS | OBJECT, DOOR, PLAYER, NPC (only ITEM/STATE or nothing meaningful) – a plain object cannot “contain” another object in TextWorld semantics. |
| DOOR | CONTAINS | Only ROOM (a door does not contain items; items are in rooms). |
| PLAYER | HAS_STATE | Only STATES that apply to the player (e.g., `"alive"`/`"dead"`). Other states like `"locked"` are invalid. |
| STATE | Any relation except HAS_* | STATE nodes cannot have outgoing HAS_STATE edges (they are leaf values). |

## Valid Entity‑Relation Combos (Whitelist)
The following subject‑relation‑target_type combinations are considered **valid** (anything else is treated as a contradiction and will cause the new fact to supersede any existing conflicting fact, or be rejected if no prior fact exists).

| Subject Type | Relation | Allowed Target Types |
|--------------|----------|----------------------|
| ROOM | CONTAINS | OBJECT, ITEM, DOOR (if modeling a door as an object inside a room) |
| ROOM | CONNECTS_TO | ROOM |
| OBJECT | IS_IN | ROOM |
| OBJECT | HAS_STATE | STATE |
| DOOR | HAS_STATE | STATE |
| DOOR | CONNECTS_TO | ROOM |
| PLAYER | IS_IN | ROOM |
| PLAYER | HAS_STATE | STATE (e.g., "alive") |
| ITEM | IS_IN | ROOM |
| ITEM | HAS_STATE | STATE |
| STATE | (none) | – |

When a candidate fact matches a valid combo but conflicts with an existing active fact (same subject & relation, different target), the newer fact **supersedes** the older per the contradiction rules above.

## Temporal Consistency
- For any edge, `t_valid_from` must be ≤ `t_observed`.
- If `t_valid_until` is set, it must be ≥ `t_valid_from` and ≤ current turn.
- An edge marked `SUPERSEDED` must have a non‑null `superseded_by` pointing to a later edge with the same subject and relation.

These constraints are enforced automatically by the GraphStore when edges are added or superseded.

---
