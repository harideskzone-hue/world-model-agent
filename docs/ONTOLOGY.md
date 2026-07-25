# Ontology

The world model uses a **typed directed graph**. Nodes represent entities; edges represent facts (relationships) between nodes or between a node and a literal state.

## Node Types
| Type | Meaning | Example |
|------|---------|---------|
| ROOM | A location in the game (can contain objects, doors, NPCs). | `"Kitchen"` |
| OBJECT | Portable item that can be taken/dropped. | `"rusty key"` |
| DOOR | Traversable barrier between rooms (may have state). | `"North Door"` |
| PLAYER | The agent’s avatar – exactly one instance. | `"player"` |
| NPC | Non‑player character (if present). | `"guard"` |
| ITEM | Generic object subclass – kept separate from OBJECT if the game distinguishes “item” vs “object”. | `"torch"` |
| STATE | A property that can be true/false or have a value (e.g., open, closed, locked, on, off). Represented as a node whose `name` is the state value. | `"open"` |

## Relation Types
Relations are directed from **subject** to **target**. The target may be another node or a literal state value (when the target is a STATE node, its `name` holds the value).

| Relation | Typical Use (subject → target) | Notes |
|----------|-------------------------------|-------|
| CONTAINS | Room → Object/Owner (room contains an object) | Inverse of IS_IN |
| IS_IN | Object → Room (object is inside a room) | Inverse of CONTAINS |
| HAS_STATE | Object/Door → STATE (e.g., door HAS_STATE "locked") | Generic state relation; the STATE node’s name is the value (`"locked"` etc.) |
| CONNECTS_TO | Room → Room (bidirectional via two edges) | Direction stored in the `direction` field of each edge (`"north"`, `"south"`). |
| CARRIES | Player → Object (player holds object) | Can also be modeled via IS_IN with player as container. |
| LOCATED_AT | NPC/Object → Room (alternative to IS_IN for non‑portable entities) |  |
| ON / OFF | Object → STATE "on" / "off" (for switches, lamps) | Modeled as HAS_STATE + STATE node. |

### State Modeling
States are **first‑class nodes** of type STATE.  
Example: a door that is locked is represented by:

- Node: `{id: "north_door", name: "North Door", node_type: DOOR}`
- Node: `{id: "locked_state", name: "locked", node_type: STATE}`
- Edge: `{subject: "north_door", relation: HAS_STATE, target: "locked_state", ...}`

To query “which doors are locked?” the system looks for edges with relation `HAS_STATE` whose target node’s `name` equals `"locked"`.

### Why No Separate LOCKED/OPEN/CLOSED Relations
Using a single generic `HAS_STATE` relation eliminates redundancy and makes the ontology extensible (new states like `"charged"`, `"broken"` require no new relation types).

---
