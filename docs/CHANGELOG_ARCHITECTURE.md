# Changelog – Architecture

All notable changes to the frozen architecture are documented here.
The format follows the pattern: **Version – Date – Summary**.

## v2.0 – 2026-07-24
- **Split WorldModel** into `WorldModel` (belief revision + ontology) and `GraphStore` (storage + indexes).
- **Added Working Memory** layer between Query Layer and Prompt Builder to hold transient execution state.
- **Simplified ontology**: removed `LOCKED`, `OPEN`, `CLOSED` relation types; all states modeled via a single `HAS_STATE` relation to `STATE` nodes.
- **Renamed Action Validator → Environment Validator** and enabled validation against the environment’s `admissible_commands` when available.
- **Renamed PROMPTS.md → SLM_INTERFACES.md** to emphasize that the interface (JSON schemas, retry policy) is stable while the prompt text is an appendix.
- **Separated Ontology from Consistency Rules**: created `ONTOLOGY.md` and `CONSISTENCY_RULES.md`.
- **Separated Update Rules**: created `UPDATE_RULES.md` (pure belief‑revision mechanics).
- **Added ASSUMPTIONS.md** to capture external expectations about the environment and the SLM.
- **Expanded Non‑Goals** section in ARCHITECTURE.md with explicit prohibited practices (keyword matching, rule‑based planning, etc.).
- Updated all interface documents (`DATA_CONTRACTS.md`, `SLM_INTERFACES.md`, `MODULE_RESPONSIBILITIES.md`, `DESIGN_PRINCIPLES.md`) to reflect the above.
- **Versioned** the architecture as v2.0 to mark the post‑redesign, competition‑ready baseline.

## v1.0 – 2026-07-01 (baseline)
- Initial architecture draft (pre‑redesign) that combined storage and reasoning, used keyword/rule‑based fallback, and had a monolithic validation step.
- Provided for reference only; superseded by v2.0.

