# Module Responsibilities

Each module has a strictly defined role. The list below describes **what the module must do** (responsibilities) and **what it must never do** (forbidden logic).  
All modules are deterministic except where explicitly noted (SLM Extractor and SLM Agent).

---

## Minimal Preprocessor
**Responsibilities**
- Accept an `Observation` (raw strings).
- Apply UTF‑8 normalization, trim whitespace, collapse repeated spaces/newlines.
- Detect an empty observation and return an empty string or a single `TextSegment` with empty text.
- Output exactly one `TextSegment` (raw_text == normalized observation, resolved_text identical).

**Forbidden**
- Any segmentation, tokenisation, coreference resolution, classification, or semantic interpretation.
- Adding, removing, or altering content beyond whitespace/encoding normalization.

---

## SLM Extractor
**Responsibilities**
- Build the extractor prompt (see `SLM_INTERFACES.md`).
- Call the local SLM with that prompt and capture the raw text output.
- Pass the raw output to the JSON Parser (no modification).

**Forbidden**
- Performing keyword matching, regex extraction, rule‑based fallback, or any heuristic that attempts to parse the observation directly.
- Modifying the prompt based on domain heuristics (e.g., adding “look for keys” prefixes).
- Attempting to repair or complete malformed JSON.

---

## JSON Parser
**Responsibilities**
- Try to parse the SLM’s raw output as JSON.
- On success, return the parsed value.
- On any parse error (malformed JSON, not an array, etc.), return an empty array `[]`.
- Never throw; always return a value.

**Forbidden**
- Trying to fix or supplement missing fields.
- Inferring the schema from partial data.
- Returning anything other than the parsed array or an empty array.

---

## Schema Validator
**Responsibilities**
- Receive an array of raw objects from the JSON Parser.
- For each object, validate:
  - Presence of required fields (`subject`, `relation`, `target`).
  - `relation` belongs to the allowed `RelationType` enum.
  - Optional `subject_type`/`target_type` (if present) belong to `NodeType`.
  - Optional `extraction_type` (if present) belongs to `ExtractionType`.
  - No extra fields are permitted (`additionalProperties: false`).
- Return a new array containing only the objects that pass validation.
- Preserve the original order.

**Forbidden**
- Adding default values for missing fields.
- Converting or coercing types (e.g., turning a number into a string).
- Adding new relations or node types not declared in the ontology.
- Performing any semantic validation (e.g., checking whether a “door” can actually “contain” an object) – that is the job of the Consistency Rules (see `CONSISTENCY_RULES.md`).

---

## Confidence Assigner
**Responsibilities**
- For each validated `CandidateFact`, set `confidence = 0.9` (fixed constant for all SLM‑origin facts).
            - **Design note**: Confidence is intended as audit metadata; the system relies on `corroboration_count` (incremented by the Updater) to reflect repeated observations. The fixed confidence simplifies belief revision while still allowing the Updater to weigh evidence via corroboration.
- Leave `confidence` unchanged if the fact came from a non‑SLM source (not used in the current design but kept for extensibility).

**Forbidden**
- Using heuristics such as entity familiarity, word overlap, or contextual cues to adjust confidence.
- Applying different values based on relation type, source, or any other property.

---

## Updater
**Responsibilities**
- Receive an array of `CandidateFact`s.
- For each fact, execute the node/edge creation/update logic described in `UPDATE_RULES.md`.
- Detect contradictions with existing active edges and apply Add/Corroborate/Supersede per the rules.
- After processing all facts, increment `WorldModel.current_turn` (and propagate to `GraphStore.current_turn`).
- Return an `UpdateReport` (counts of added, corroborated, superseded, rejected) for logging/metrics.

**Forbidden**
- Performing any planning, goal reasoning, or action selection.
- Modifying the graph based on anything other than the incoming facts (e.g., adding inferred links).
- Applying temporary or heuristic weights that persist beyond the current update.
- Removing edges except via the formal Supersede mechanism.

---

## WorldModel
**Responsibilities**
- Provide read‑only query methods:
  - `get_active_edges(subject?, relation?)`
  - `get_nodes_by_type(type)`
  - `get_node(id)`
  - `temporal_valid(edge, turn)` – check if an edge is active at a given turn.
  - Expose `current_turn`.
- Contain the ontology (enums) and delegate persistence to `GraphStore`.
- Offer transaction‑like semantics if needed (but current design uses a simple apply‑then‑commit flow).

**Forbidden**
- Storing any derived or cached information that is not a direct lookup in the underlying graph (e.g., pre‑computed reachability matrices, summarized descriptions).
- Performing belief revision itself; that logic resides in the `Updater`.
- Exposing internal mutators that bypass the `Updater`.

---

## GraphStore
**Responsibilities**
- Store `nodes: Map<string, Node>` and `edges: Map<string, Edge>`.
- Maintain three indexes:
  - `subjectIdx`: subject_id → set of edge_ids.
  - `objectIdx`: object_id → set of edge_ids.
  - `slotIdx`: (subject_id, relation) → set of edge_ids.
- Provide atomic add/node/update operations used by the `Updater`.
- Ensure indexes stay in sync after every mutation.
- Track `current_ton` (mirrored from `WorldModel`).

**Forbidden**
- Computing or storing any semantic summaries (e.g., “all doors are locked”).
- Exposing raw internal maps without encapsulation that could allow external mutation.
- Implementing belief‑revision rules; those belong to the `Updater`.

---

## Query Layer
**Responsibilities**
- Receive the current `WorldModel` (read‑only) and `workingMemory`.
- Construct a `ContextSlice` as defined in `QUERY_LAYER.md`:
  - `objective` from `observation.objective` (passed via working memory).
  - `current_room` by locating the player node and following an active `IS_IN`/`LOCATED_AT` edge.
  - `reachable_rooms` via outgoing active `CONNECTS_TO` edges from the current room.
  - `inventory` via active `IS_IN`/`CARRIES` edges from the player.
  - \`objects_in_current_room\` via active \`CONTAINS\` edges from the current room.
  - `locked_doors` by checking doors in the current room for an active `HAS_STATE` edge to a STATE node named `"locked"` (or `"closed"` per ontology).
  - `recent_anges` (optional) – last *N* changed edges.
- Return the `Slice`; **do not** alter any model state.

**Forbidden**
- Performing any inference, summarization, or addition of facts not present in the model.
- Filtering based on guessed relevance to the objective (that is the SLM’s job).
- Modifying the graph or any persistent state.
- Returning internal IDs instead of human‑readable names (the slice must contain names/strings).

---

## Working Memory
**Responsibilities**
- Hold transient execution state for the current turn:
  - `previous_action` (string or null).
  - `retry_count` (integer, number of retries needed to obtain a valid SLM JSON this turn).
  - `current_sub_goal` (string or null, if the agent wishes to track sub‑goals).
  - `last_observation` (the most recent `Observation`).
  - `objective` (copied from the observation for easy access).
- Be cleared or re‑initialized at the start of each episode.
- Provide read‑only access to the `Prompt Builder` and any other module that needs transient info.
- Lifecycle: created at episode start, updated each turn, destroyed at episode end.

**Forbidden**
- Persisting any of this information beyond the current turn (i.e., writing it to the `WorldModel` or `GraphStore`).
- Using it to influence belief revision (the `WorldModel` must remain the sole source of persistent truth).
- Performing any inference or reasoning.

---

## Prompt Builder
**Responsibilities**
- Receive a `ContextSlice` (from the `Query Layer`).
- Render it into a single prompt string using the template defined in `SLM_INTERFACES.md` Appendix A.
- Ensure the final prompt length does not exceed the token budget (e.g., 1500 tokens). If overflow occurs, remove lower‑priority fields in this order: `recent_changes`, then perhaps truncate lists (preserving order) until fit.
- Never add information that is not present in the `Slice`.
- Never re‑phrase, summarise, or infer beyond the given data.

**Forbidden**
- Adding examples, hints, or few‑shot demonstrations that were not part of the original template.
- Performing any semantic summarisation (e.g., “you see a key and a door”).
- Altering the meaning of any field (e.g., changing “locked” to “closed” unless the ontology defines them as equivalent).
- Injecting randomness or creativity.

---

## SLM Agent (the local LLM)
**Responsibilities**
- Receive the prompt string from the `Prompt Builder`.
- Generate a response (typically a short JSON object).
- Return the raw text to the `Action Parser`.
- The model is free to use its internal knowledge and reasoning; this is the sole locus of semantic reasoning.

**Forbidden**
- No constraints from the system side; the model may output anything. The system will handle invalid outputs via parsing/validation.
- (Note: The model should be encouraged to produce valid JSON, but enforcement is post‑hoc.)

---

## Action Parser
**Responsibilities**
- Parse the SLM’s raw output as JSON.
- Extract the `"action"` field (default to empty string if missing or not a string).
- Trim whitespace.
- Pass the raw string (trimmed) to the `Environment Validator`.

**Forbidden**
- Attempting to correct or complete a malformed action string.
- Inferring the intended action from context.
- Returning anything other than the extracted string (even if empty).

---

## Environment Validator
**Responsibilities**
- Receive the action string from the `Action Parser`.
- Verify:
  1. The string is non‑empty after trimming.
  2. If the environment exposes an `admissible_commands` list (via `env.get_available_commands()`), the string must exactly match one of those entries (case‑sensitive).
  3. If no such list is available, any non‑empty string is considered syntactically valid (the environment will reject it and return appropriate feedback).
- Return an `ActionResult`:
  - `{ action: <trimmed string>, valid: true, error: null }` if checks pass.
  - `{ action: "", valid: false, error: <reason> }` otherwise (e.g., empty string, not in admissible list).

**Forbidden**
- Guessing whether the action is “wise” or likely to succeed based on world model.
- Modifying the action string (e.g., adding prefixes, correcting typos).
- Never rewrite or alter the action string.
- Consulting the world model to decide validity beyond the admissible list check.
- Providing any strategic advice; the validator is purely a syntactic/semantic gate.

---


## Orchestrator (Decision Loop) – implicit in `decision_loop.py`
**Responsibilities**
- Initialise modules, reset environment, set up `workingMemory`.
- Loop per turn:
  1. Get `Observation` from env.
  2. Run `Preprocessor`.
  3. Run `SLM Extractor`.
  4. Run `JSON Parser`.
  5. Run `Schema Validator`.
  6. Run `Confidence Assigner` → `CandidateFact[]`.
  7. Run `Updater` (which updates `WorldModel` via `GraphStore`).
  8. Run `Query Layer` → `ContextSlice`.
  9. (Update `workingMemory` with latest observation, increment `retry_count` as needed, store `previous_action` after validation.)
  10. Run `Prompt Builder` → prompt.
  11. Run `SLM Agent`.
  12. Run `Action Parser`.
  13. Run `Environment Validator` → `ActionResult`.
  14. **If `valid == true`**, execute `action` via `env.step()`; else, treat as a no‑op and let the environment produce feedback (no `env.step` call).
  15. Record metrics, continue until terminal.

**On Invalid Action**:
1. Environment Validator returns valid=false
2. Orchestrator does NOT execute env.step()
3. No new Observation is generated
4. Next turn begins with same Observation as input
5. SLM can select a different action (retry implicit, no explicit counter)
6. Log invalid-action count for evaluation metrics

**Guarantee**: Every action sent to env.step() is non-empty and has passed validation.

**Forbidden**
- Embedding any belief‑revision or planning logic outside of the designated modules.
- Modifying the data flow order without updating the interface documents.
- Bypassing validation steps to “help” the agent.
