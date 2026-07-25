# Update Rules (Belief Revision)

The Updater applies each `CandidateFact` using deterministic rules that maintain a consistent belief state. No planning or inference is performed—only maintenance of the graph.

## Overview
For each incoming fact the Updater:
1. Ensures the subject and target nodes exist (creates them with appropriate `node_type` if missing).
2. Constructs a provisional edge from the fact.
3. Checks for conflicts with existing **active** edges sharing the same `(subject, relation)` slot.
4. Applies one of four actions: **Add**, **Corroborate**, **Supersede**, or **Retract** (retract is a special case of supersede when the new fact negates an existing belief).
5. Updates timestamps and stores the edge in the GraphStore.
6. Increments `WorldModel.current_turn` after processing all facts for the turn.

## Detailed Rules

### 1. Node Creation / Update
- If the fact’s `subject` (or `target` when it denotes a node) does not exist, create a node:
  - `id` = normalized name.
  - `name` = original name (preserving case).
  - `node_type` taken from `subject_type`/`target_type` if provided; otherwise default to `OBJECT`.
  - Set `confidence = 0.9`, `first_observed_turn = source_turn_id`, `last_observed_turn = source_turn_id`, `corroboration_count = 0`.
- If the node already exists, update:
  - `last_observed_turn = max(last_observed_turn, source_turn_id)`.
  - Increase `corroboration_count` by 1.
  - Confidence remains unchanged (remains at the value set on creation, typically 0.9 for SLM‑origin nodes).
  - Merge `attributes` (new values overwrite old).

### 2. Edge Construction
Create a provisional edge with:
- `subject` = normalized subject name.
- `relation` = relation from fact.
- `target` = normalized target name (or literal state value).
- `confidence` = 0.9 (fixed for SLM facts).
- `source_turn_id` = fact.source_turn_id.
- `extraction_method` = SLM.
- `t_observed` = source_turn_id.
- `t_valid_from` = source_turn_id.
- `t_valid_until` = null.
- `status` = ACTIVE.
- `corroboration_count` = 1.
- `superseded_by` = null.
- `revision_reason` = "".
- `direction` = extracted from fact if relation is CONNECTS_TO (compass direction); otherwise null.

### 3. Conflict Detection
Query the GraphStore for all **active** edges where `edge.subject == provisional.subject` and `edge.relation == provisional.relation`.

- **No active edges** → **Add**: simply insert the provisional edge.
- **One or more active edges** → compare each existing edge’s `target` (or state value) with the provisional edge’s `target`.

  - **Exact match** (target strings equal, ignoring case) → **Corroborate**:
    * Increment existing edge’s `corroboration_count`.
    * Confidence remains at 0.9 (or its current value); no adjustment is made.
    * Set `t_valid_from = min(old.t_valid_from, provisional.t_valid_from)`.
    * Ensure `t_valid_until = null` and `status = ACTIVE`.
    * Discard the provisional edge (no new edge added).

  - **Different target** → **Supersede**:
    * For each conflicting active edge:
      - Set `status = SUPERSEDED`.
      - Set `t_valid_until = provisional.t_observed` (the turn the new fact was observed).
      - Set `superseded_by = provisional.edge.id`.
      - Set `revision_reason = "superseded: newer observation at turn X"` where X = provisional.t_observed.
    * Insert the provisional edge as a new active edge.
    * Note: If multiple conflicting edges exist (should not happen under normal consistency rules), all are superseded.

### 4. Retraction (Negation)
If the incoming fact expresses a negation (e.g., `extraction_type = NEGATION`) and its target negates an existing active edge (same subject & relation, opposite state), treat it as a supersede where the new fact’s target is the opposite STATE node (e.g., `"open"` vs `"closed"`). The same supersede logic applies.

### 5. Timestamp Update
After all facts of a turn have been processed:
- Set `WorldModel.current_turn = observation.turn_id`.
- Propagate this value to `GraphStore.current_turn`.

## Guarantees
- No two **active** edges share identical `(subject, relation, target)`.
- Temporal validity reflects the most recent observation for each fact.
- The graph remains a single source of truth for the agent’s beliefs.
- All updates are deterministic; given the same sequence of facts, the final graph is identical.

