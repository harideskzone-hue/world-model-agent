# Design Principles

## ✅ Allowed (what the system **may** do)

- **Local SLM reasoning only** – All semantic understanding (entity/relation extraction, action selection) is performed by the locally‑run language model; no external APIs or calls to cloud services.
- **Deterministic validation, parsing, updating, querying** – Steps such as JSON parsing, schema validation, confidence assignment, graph storage/indexing, belief‑revision rules, context‑slice extraction, prompt building, and action validation must be deterministic (same input → same output).
- **Structured, typed world model** – The persisted knowledge is a directed graph whose nodes and edges belong to a fixed ontology (see `ONTOLOGY.md`).
- **Belief revision with temporal versioning** – Contradictory facts create new edge versions; old edges are marked `SUPERSEDED` and retained for audit/time‑travel queries.
- **Compact context for the SLM** – The query layer returns only the information the model needs right now (objective, current room, reachable rooms, inventory, relevant objects, locked doors). Nothing more.
- **Strict separation of concerns** – Semantic reasoning lives exclusively in the SLM; everything else is infrastructure.
- **Clear interfaces** – Every module communicates via explicitly defined data contracts (see `DATA_CONTRACTS.md`). Changes to those contracts require a version bump and explicit agreement.
- **Fail‑safe defaults** – When the SLM emits malformed JSON, the system treats it as an empty fact list or empty action rather than crashing.
- **Observable metrics** – Timing, token counts, validation outcomes, and world‑model consistency are logged for debugging and performance tuning.

## ❌ Forbidden (what the system **must never** do)

- **Keyword‑matching, regex‑based, or rule‑based semantic extraction** – No handcrafted patterns that map substrings to facts.
- **Rule‑based planning or action selection** – No logic of the form “if I have a key and see a door then output ‘unlock door’”.
- **Game‑specific heuristics** – No code that assumes particular object names, room layouts, or puzzle solutions for the benchmark worlds.
- **Hard‑coded navigation logic** – No predetermined movement strategies (e.g., always explore north first).
- **Semantic filtering inside the Query Layer** – The query layer must not decide which objects are “relevant” beyond the simple rule of excluding inventory items.
- **Semantic reasoning inside the Prompt Builder** – The builder may only format existing data; it cannot infer, summarize, or inject new facts.
- **Duplicate persistent memory outside the World Model/GraphStore** – No caching of derived facts, summaries, or histories that could serve as an alternative memory source.
- **Cloud APIs** – No calls to external LLM providers, translation services, knowledge bases, etc., during inference.
- **Manually curated walkthroughs or hardcoded solutions** – The agent must discover solutions through its perception‑action loop; no embedded scripts for the benchmark worlds.
- **Storing denormalised or computed data in the core model** – Attributes like “room description” or “number of objects in a room” must not be stored as node/edge properties unless they are true primitives of the domain; such values must be computed on‑the‑fly when needed.
- **Using confidence as a heuristic for belief weighting** – Confidence is a fixed constant (0.9) for SLM‑derived facts; no learned or context‑dependent weighting schemes are permitted in the core logic. (Future extensions may introduce a learned confidence model, but it would live outside the deterministic core and be clearly marked as experimental.)

These principles constitute the architectural contract. Any proposed change must be evaluated against them; violations require explicit justification and consensus before adoption.

---
