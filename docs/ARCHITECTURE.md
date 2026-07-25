# Architecture Overview

The World‑Model Agent follows a strict pipeline where **only the SLM performs semantic reasoning**. All other blocks are deterministic infrastructure (validation, updates, queries, prompting).

```
Observation
        │
        ▼
Minimal Preprocessor          ← UTF‑8/whitespace trim, empty‑check
        │
        ▼
SLM Extractor                 ← Prompt → JSON array of facts
        │
        ▼
JSON Parser                   ← try/catch → [] on failure
        │
        ▼
Schema Validator              ← against NodeType/RelationType
        │
        ▼
Confidence Assigner           ← fixed 0.9 for SLM‑produced facts
        │
        ▼
CandidateFact[]
        │
        ▼
Updater                       ← add / corroborate / supersede
        │
        ▼
WorldModel                    ← belief revision + ontology (no storage)
        │
        ▼
GraphStore                    ← node/edge storage + indexes
        │
        ▼
Query Layer                   ← returns ContextSlice (human‑readable names)
        │
        ▼
Working Memory                ← transient execution state (prev action, retry, sub‑goal, …)
        │
        ▼
Prompt Builder                ← compact prompt text for the SLM
        │
        ▼
SLM                           ← receives: Objective + Prompt
        │
        ▼
SLMDecision                   ← JSON: {action, reasoning_summary?, …}
        │
        ▼
Action Parser                 ← extract "action" field
        │
        ▼
Environment Validator         ← → ActionResult {action, valid, error}
        │
        ▼
TextWorld Environment         ← executes action → new Observation
        │
        ▼
(loop back to Observation)
```

## Key Separations
- **WorldModel**: contains the ontology, belief‑revision logic, temporal reasoning, and graph queries (no raw storage).  
- **GraphStore**: pure storage layer – nodes, edges, indexes.  
- **Working Memory**: short‑lived execution state (does **not** persist beyond the current turn).  
- **Environment Validator**: checks that the parsed action is syntactically valid and belongs to the set of admissible TextWorld commands (does not evaluate strategic merit).

## Deterministic Infrastructure (allowed logic)
- UTF‑8/whitespace normalization  
- JSON parsing (fallback to empty array on failure)  
- Schema validation against the fixed ontology  
- Fixed confidence assignment (0.9 for SLM facts)  
- Graph storage & indexing  
- Belief‑revision rules (add, corroborate, supersede)  
- Context slice extraction (current room, adjacent rooms, inventory, relevant objects, locked doors)  
- Prompt building (template filling, token‑budget truncation)  
- Action parsing (`"action"` field extraction)  
- Action validation (non‑empty string, admissible command list)

## Semantic Reasoning (SLM‑only)
1. **Extractor** – turns raw observation into a JSON array of facts (entities, relations, states).  
2. **Planner (SLM Agent)** – given the objective and the current world slice, selects exactly one action.

## Non‑Goals (what the system will never do)
- No keyword‑matching, regex‑based, or rule‑based semantic extraction.  
- No rule‑based planning or action selection (e.g., “if key then open door”).  
- No game‑specific heuristics or hard‑coded navigation logic.  
- No semantic filtering inside the Query Layer or Prompt Builder.  
- No duplicate persistent memory outside the WorldModel/GraphStore.  
- No cloud APIs.  
- No manually curated walkthroughs or hardcoded solutions for benchmark worlds.  
- No storage of derived or cached information (e.g., pre‑computed room descriptions) in the WorldModel; only indexes are kept.

## Modules (see MODULE_RESPONSIBILITIES.md for details)
- Minimal Preprocessor  
- SLM Extractor  
- JSON Parser  
- Schema Validator  
- Confidence Assigner  
- Updater  
- WorldModel  
- GraphStore  
- Query Layer  
- Working Memory  
- Prompt Builder  
- SLM Agent  
- Action Parser  
- Environment Validator  

## Evaluation
Three staged worlds (2‑room, 6‑room, 10‑room) with explicit success metrics are defined in EVALUATION_PLAN.md.

---
*This document captures the frozen architecture as of version 2.0. See CHANGELOG_ARCHITECTURE.md for a history of changes.*

## Dependency Diagram

The following diagram shows which modules depend on others (arrows point from consumer to provider).

```
Orchestrator
│
├──► Minimal Preprocessor
├──► SLM Extractor
├──► JSON Parser
├──► Schema Validator
├──► Confidence Assigner
├──► Updater
│   │
│   ├──► WorldModel
│   │       ▲
│   │       │
│   │       ▼
│   └──► GraphStore
│
├──► Query Layer
│       ▲
│       │
│       ▼
│   WorldModel
│
├──► Working Memory
│
├──► Prompt Builder
│       ▲
│       │
│       ▼
│   ContextSlice (from Query Layer)
│
├──► SLM Agent
│       ▲
│       │
│       ▼
│   Prompt (from Prompt Builder)
│
├──► Action Parser
├──► Environment Validator
│       ▲
│       │
│       ▼
│   Action (to Environment)
│
Environment
│
▼
Observation (back to Orchestrator)
```
```

