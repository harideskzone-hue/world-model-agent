# Testing Strategy

## Overview
This document outlines the testing strategy for the World-Model Agent system. The approach emphasizes unit testing of deterministic components and integration testing aligned with the evaluation plan in `EVALUATION_PLAN.md`.

## Unit Testing Strategy

Each deterministic module should be unit tested in isolation. The SLM Extractor and SLM Agent are not unit-tested for semantic correctness (as they rely on the SLM), but their input/output handling and fallback logic should be tested.

### 1. Minimal Preprocessor
- **Test cases**: 
  - Normal text with various whitespace combinations
  - Empty strings
  - UTF-8 encoded characters
  - Text with special characters
- **Assertions**: Output should be normalized (trimmed, collapsed whitespace) but semantically identical
- **Forbidden logic verification**: Ensure no semantic processing occurs

### 2. JSON Parser
- **Test cases**:
  - Valid JSON arrays → return parsed array
  - Malformed JSON (missing brackets, invalid syntax) → return empty array
  - Non-array JSON (objects, strings, numbers) → return empty array
  - Empty input → return empty array
  - JSON with trailing commas → return empty array (strict parsing)
- **Assertions**: Always returns an array; never throws exceptions

### 3. Schema Validator
- **Test cases**:
  - Valid CandidateFact objects → pass through
  - Missing required fields (subject, relation, target) → filtered out
  - Invalid RelationType values → filtered out
  - Invalid NodeType values in optional fields → filtered out
  - Extra fields present → filtered out (additionalProperties: false)
  - Preserve input order in output
- **Assertions**: Only validates structure; does not perform semantic validation

### 4. Confidence Assigner
- **Test cases**:
  - Valid CandidateFact → confidence set to 0.9
  - Non-SLM source facts (if implemented) → confidence unchanged
  - Edge cases: confidence already set to 0.9, 0.0, 1.0
- **Assertions**: Confidence is always 0.9 for SLM facts; never uses heuristics

### 5. Updater
- **Test cases**:
  - Adding new facts → creates nodes/edges with corroboration_count=1
  - Corroborating existing facts → increments corroboration_count
  - Contradictory facts (same subject/relation, different target) → supersedes older edge
  - Mutually exclusive states (per CONSISTENCY_RULES.md) → supersedes older edge
  - Invalid relations (per CONSISTENCY_RULES.md) → treated as contradictions
  - Empty input → returns empty UpdateReport
- **Assertions**: 
  - Never removes edges except via supersede
  - Never adds inferred links
  - current_turn incremented after processing
  - Follows ADD/Corroborate/SUPERSEDE rules precisely

### 6. Query Layer
- **Test cases** (requires mock GraphStore/WorldModel):
  - current_room: correctly identifies player location via IS_IN edge
  - reachable_rooms: returns rooms via active CONNECTS_TO edges
  - inventory: returns objects via IS_IN/CARRIES edges from player
  - objects_in_current_room: returns objects via CONTAINS edges from current room
  - locked_doors: returns doors with HAS_STATE to "locked" or "closed" STATE nodes
  - recent_changes: returns last N edge IDs (when implemented)
 
- **Assertions**:
  - Returns only human-readable names (no internal IDs)
  - No duplicates in lists
  - Does not modify model state
  - No filtering based on goal relevance

### 7. Prompt Builder
- **Test cases**:
  - Normal case: renders all sections correctly
  - Empty fields: renders as empty arrays/strings
  - Token budget exceeded: truncates in correct order (recent_changes first, then objects_in_current_room)
  - Token estimation: matches len(prompt.split()) × 1.3
  - No extra information added beyond ContextSlice
  - No rephrasing or summarization of content
- **Assertions**:
  - Output matches template exactly with filled placeholders
  - Truncation follows specified priority order
  - Never adds examples, hints, or semantic interpretation

### 8. Environment Validator
- **Test cases**:
  - Non-empty string in admissible_commands → valid=true
  - Empty string → valid=false, error message
  - String not in admissible_commands → valid=false, error message
  - When no admissible_commands provided: any non-empty string → valid=true
  - Whitespace-only strings → valid=false after trimming
- **Assertions**:
  - Never modifies action string
  - Never consults world model for validity
  - Only checks syntactic/admissible validity

## Integration Testing Strategy

Integration tests should verify the pipeline works correctly end-to-end using the evaluation worlds from `EVALUATION_PLAN.md`.

### Test Worlds
Use the three staged worlds defined in `EVALUATION_PLAN.md`:
1. **2-room world**: Simple navigation and object manipulation
2. **6-room world**: Moderate complexity with multiple objects and doors
3. **10-room world**: Complex scenario requiring planning and memory

### Metrics to Validate
For each world, measure:
1. **Success rate**: Percentage of episodes where the agent achieves the objective
2. **Steps to solution**: Efficiency of the agent's path
3. **Invalid action rate**: Frequency of syntactically invalid actions
4. **World model accuracy**: 
   - Precision/recall of extracted facts vs. ground truth
   - Correctness of spatial relationships (room connections)
   - Correctness of object states (open/closed, on/off, etc.)
5. **Belief revision correctness**:
   - Proper handling of contradictory observations
   - Correct temporal reasoning (what was true when)
   - Proper superseding of outdated facts

### Test Procedure
1. **Deterministic seed testing**: 
   - Fix SLM temperature to 0.0 for reproducible outputs
   - Run each world 10-20 times with different seeds
   - Measure success rate and variance

2. **Ablation studies** (where applicable):
   - Test with SLM Extractor disabled (fallback to rule-based only)
   - Test with confidence assignment varied
   - Test with different truncation thresholds

3. **Edge case testing**:
   - Empty observations
   - Contradictory observations in same turn
   - Very long observations requiring prompt truncation
   - Ambiguous language requiring coreference resolution (should fail gracefully)

### Regression Testing
- Create test suites that run on every commit
- Fail builds if:
  - Any deterministic module fails unit tests
  - Success rate drops below baseline in evaluation worlds
  - Invalid action rate increases beyond threshold
  - World model accuracy metrics degrade

### Test Environment Requirements
- Fixed SLM version (for reproducibility)
- Deterministic random seeds
- Isolated test database/state for each test
- Mock SLM available for unit testing (returns predefined responses)
- Benchmark scripts to rerun evaluation scenarios

### Continuous Integration
- Unit tests run on every push
- Integration tests run nightly (due to SLM inference time)
- Performance benchmarks tracked over time
- Alert on significant regression in success rate (>10% drop)

## Acceptance Criteria for Architecture Freeze
1. All unit tests pass for deterministic modules
2. Integration tests show:
   - >80% success rate on 2-room world
   - >60% success rate on 6-room world  
   - >40% success rate on 10-room world
   - <5% invalid action rate across all worlds
3. No forbidden logic detected in any module (via code review/test inspection)
4. Deterministic behavior confirmed with fixed seeds

This testing strategy ensures the architecture is both correctly implemented and empirically validated before moving to integration with the full SLM system.