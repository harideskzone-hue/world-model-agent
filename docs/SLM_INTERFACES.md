# SLM Interfaces

## Section 1: Extractor Prompt Template

The SLM Extractor uses the following prompt template (see `extractor/slm_extractor.py`) to extract facts from raw observation text.

```
You are a strict data extractor for a TextWorld environment.
Your ONLY task is to extract facts from the observation text and return them as a valid JSON array.

SCHEMA:
- subject: entity name (lowercase)
- subject_type: strictly one of [ROOM, OBJECT, CHARACTER]
- relation: strictly one of [contains, connects_to, located_in, holds, has_state, is_type]
- object: target entity name or state value (lowercase)
- object_type: strictly one of [ROOM, OBJECT, CHARACTER, STATE]
- extraction_type: strictly one of [direct, implied, negation]

RULES:
1. Output ONLY a valid JSON array. No explanations, no markdown blocks, no conversational text.
2. If no facts are present, output an empty array: []

EXAMPLES:
Text: "You've entered a kitchen. There is an exit to the north. Don't worry, it is unblocked."
[
  {"subject": "kitchen", "subject_type": "ROOM", "relation": "located_in", "object": "you", "object_type": "CHARACTER", "extraction_type": "implied"},
  {"subject": "kitchen", "subject_type": "ROOM", "relation": "has_state", "object": "exit to north is unblocked", "object_type": "STATE", "extraction_type": "direct"}
]

Text: "You open the safe. Inside, you see a brass key."
[
  {"subject": "safe", "subject_type": "OBJECT", "relation": "has_state", "object": "open", "object_type": "STATE", "extraction_type": "direct"},
  {"subject": "safe", "subject_type": "OBJECT", "relation": "contains", "object": "brass key", "object_type": "OBJECT", "extraction_type": "direct"}
]

TEXT: "{text}"
CONTEXT:
- Current room: {current_room}
- Inventory: {inventory}

FACTS:
```

## Section 2: Planner Prompt Template (Action Selection)

The Prompt Builder constructs the action selection prompt using the template defined in Appendix A. 
The SLM Agent (Planner) receives this prompt and must output a JSON object with an `"action"` field containing the selected TextWorld command.

The prompt contains the current world state in a structured format that includes:
- The agent's goal/objective
- Current location
- Reachable rooms
- Inventory
- Objects in the current room
- Locked doors
- Recent changes (debug information)
- Grammar description of valid actions

The SLM must analyze this state and select exactly one valid action to progress toward the goal.

## Appendix A: Prompt Builder Template

The Prompt Builder renders a `ContextSlice` (from the Query Layer) into a single prompt string using the following section‑ordered template. Placeholders are replaced with the corresponding field values from the `ContextSlice` and `WorkingMemory`.

```
You are an AI agent reasoning over a semantic world model to play a text adventure game.
Analyze the structured Current World State below and select exactly one valid action.
Do NOT output conversational filler, markdown formatting, or explanations.

=== CURRENT WORLD STATE ===

[GOAL]
{objective}

[CURRENT LOCATION]
{current_room}

[REACHABLE ROOMS]
{reachable_rooms}

[INVENTORY]
{inventory}

[OBJECTS IN CURRENT ROOM]
{objects_in_current_room}

[LOCKED DOORS]
{locked_doors}

[RECENT CHANGES]
{recent_changes}

[GRAMMAR DESCRIPTION]
{grammar_description}

Analyze the state and select the single best action to progress toward the [GOAL].
Respond with ONLY the exact text of the action you want to take.
ACTION:
```

Placeholders are replaced as follows:
- `{objective}`: from `ContextSlice.objective` (or `WorkingMemory.objective`)
- `{current_room}`: from `ContextSlice.current_room`
- `{reachable_rooms}`: JSON array of strings from `ContextSlice.reachable_rooms`
- `{inventory}`: JSON array of strings from `ContextSlice.inventory`
- `{objects_in_current_room}`: JSON array of strings from `ContextSlice.objects_in_current_room`
- `{locked_doors}`: JSON array of strings from `ContextSlice.locked_doors`
- `{recent_changes}`: JSON array of strings from `ContextSlice.recent_changes` (may be empty)
- `{grammar_description}`: a string describing the valid action commands (provided by the environment)

If a field is empty, it is rendered as an empty array `[]` or empty string as appropriate.

## Appendix B – Prompt Builder Truncation Policy

If the final prompt exceeds the token budget (1500 tokens for Stage 3), truncate in this order:

1. **Keep (mandatory)**:
   - Objective
   - Current room name
   - Reachable rooms list
   - Inventory list

2. **Then truncate (in order)**:
   - `objects_in_current_room`: keep first N items (sorted by recency of first observation)
   - `locked_doors`: keep all (structural safety critical)
   - `recent_changes`: omit entirely (debug only)

**Token estimation**: `len(prompt.split()) × 1.3` (conservative estimate for SLM tokenizer).
If estimate > 1500, apply truncation before inference.

---