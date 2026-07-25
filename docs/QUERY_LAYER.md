# Query Layer Design

The Query Layer answers the question: **“What information does the SLM need right now to select the next action?”**  
It returns a `ContextSlice` containing only the relevant, human‑readable pieces of the world model.

## Input
- `workingMemory` (contains: objective, optional transient state).
- `turnId` (current turn number, accessible via `GraphStore.current_turn`).

## Output – `ContextSlice`
All fields contain **plain strings** (node names or literal state values) so the Prompt Builder can embed them directly without extra look‑ups.

| Field | Meaning | How it is Computed |
|-------|---------|--------------------|
| `objective` | The agent’s goal (constant for the episode). | Copied from `Observation.objective`. |
| `current_room` | Name of the room where the player is located. | Find the unique `PLAYER` node, follow an active `IS_IN` (or `LOCATED_AT`) edge to its container room; return that room’s `name`. |
| `reachable_rooms` | Names of rooms directly connected to the current room via an active `connects_to` edge (depth = 1). | From the current room node, collect all active `CONNECTS_TO` edges; for each edge, get the target room’s `name`. |
| `inventory` | Names of objects the player holds. | From the player node, follow all active `IS_IN` (or `CARRIES`) edges where the subject is the player; collect the target object’s `name`. |
| `objects_in_current_room` | Names of objects present in the current room. The SLM decides which of these are pertinent to the goal. | From the current room node, follow all active `CONTAINS` edges; collect the target object`s `name`. |
| `locked_doors` | Names of doors in the current room whose state is `Locked` (or `Closed` depending on ontology). | From the current room node, follow all active `CONTAINS` edges to find door nodes; for each door, check if there exists an active `HAS_STATE` edge to a STATE node whose `name` = `"locked"` (or `"closed"`). If yes, include the door’s `name`. |
| `recent_changes` | (Optional) Short descriptions of edges added, corroborated, or superseded in the last *N* turns (useful for debugging). | Query the GraphStore for edges where `t_observed ≥ current_turn - N`; map each to a human‑readable string (e.g., “North Door locked → unlocked”). May be omitted entirely for production. |

## Computational Guarantees
- All look‑ups use the GraphStore’s indexes (`subjectIdx`, `objectIdx`, `slotIdx`) → **O(1)** per relationship type.
- For worlds up to 100 rooms, total runtime ≤ 5 ms per call.
- Memory usage is proportional to the size of the returned slice (small constant).

## Restrictions (What the Query Layer **must not** do)
- No semantic filtering: it must not decide which objects are “relevant” beyond the simple exclusion of inventory items.
- No summarization, inference, or addition of facts not present in the graph.
- No modification of the graph (read‑only).
- No inclusion of the full world graph or any aggregated statistics unless explicitly listed above (e.g., recent changes are optional and limited).

## Example
Given a world where:
- Player is in Kitchen.
- Kitchen contains a rusty key (object) and a north door (door).
- The north door has state `locked`.
- Player inventory is empty.
- Objective: “Get the key and open the north door”.

The Query Layer returns:
```json
{
  "objective": "Get the key and open the north door",
  "current_room": "Kitchen",
  "reachable_rooms": ["Hall"],
  "inventory": [],
  "objects_in_current_room": ["rusty key"],
  "locked_doors": ["North Door"],
  "recent_changes": []
}
```
The Prompt Builder then formats this into the final prompt for the SLM.

---
