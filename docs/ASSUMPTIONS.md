# Assumptions

The following assumptions underlie the architecture and are held to be true for the target environment (TextWorld games used in HackTronix Track 1). If any assumption is violated, the system may exhibit degraded performance or require adaptation.

1. **Truthful observations** – The TextWorld environment returns accurate descriptions and feedback; it does not deliberately deceive the agent.
2. **Deterministic dynamics** – Given the same state and action, the environment produces the same next state and observation (no stochasticity unless explicitly modelled as part of the state).
3. **Structured‑JSON capable SLM** – The local SLM (e.g., via Ollama) can be prompted to return syntactically valid JSON arrays/objects as specified in `SLM_INTERFACES.md`. Malformed outputs are rejected by the JSON Parser, and the system proceeds without updating the World Model for that observation.
4. **Admissible command exposure** – The environment provides a method (`get_available_commands()` or equivalent) to retrieve the set of syntactically valid verbs/phrases for the current state, enabling the environment validator to prune impossible actions.
5. **Singleton player** – There is exactly one player entity in the world (node with `node_type = PLAYER`). All inventory and location relationships are anchored to this node.
6. **Sufficient context window** – The SLM’s context length (e.g., 2048 tokens for typical LLMs) is large enough to contain the objective plus the maximal expected `ContextSlice` after truncation; the stream‑wise token budget is enforced by the prompt builder.
7. **No simultaneous multi‑agent interactions** – The agent is the sole decision‑making entity; other NPCs, if present, are treated as ordinary objects or state‑bearing entities.
8. **Static ontology for a given game** – The set of node types, relation types, and allowed state values does not change during an episode; new types are only introduced via a versioned update of the ontology files.
9. ** Axiomatic temporal model** – Time advances monotonically with each turn (`t` increments by 1); there is no time‑skipping or simultaneous sub‑turn actions.
10. ** Observation language comprehensible to SLM** – The natural language emitted by the environment is within the linguistic competence of the chosen SLM (i.e., the model has seen similar text during pretraining or fine‑tuning).

If any of these assumptions are discovered to be false for a specific game or deployment, the relevant design document (e.g., `ASSUMPTIONS.md`, `DATA_CONTRACTS.md`, or `SLM_INTERFACES.md`) should be updated and the impact assessed before proceeding.

---

11. **SLM determinism** – The local SLM is run with temperature = 0 (or the minimum setting that produces stable, deterministic JSON output). Given identical prompts, the model produces identical responses across runs.
12. **Language determinism** – The implementation uses Python 3.7+ (dict/set insertion-order preservation guaranteed by language spec). GraphStore index iteration is deterministic; relying on this for reproducible world-model output.
