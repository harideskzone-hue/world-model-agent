# Repository Structure

This document provides a map of the repository, highlighting how each module corresponds to the Track 1 architectural requirements.

```text
world-model-agent/
│
├── extractor/
│   # Responsible for semantic extraction. Converts unstructured text 
│   # observations into semantic graph triples (Subject, Relation, Object).
│
├── world_model/
│   # Persistent graph store. Maintains the global knowledge graph and 
│   # temporal versions of the agent's beliefs.
│
├── updater/
│   # Belief revision logic. Resolves contradictions and corroborates 
│   # new extractions before applying them to the World Model.
│
├── query_layer/
│   # Produces the current world slice. Filters the massive world graph 
│   # down to only the locally relevant context for the SLM.
│
├── orchestrator/
│   # Agent loop and memory. Ties the pipeline together and manages 
│   # the step-by-step execution cycle.
│
├── slm/
│   # Local SLM interface. Handles prompt building, formatting, and 
│   # action selection using local models (e.g., via Ollama).
│
├── env/
│   # TextWorld environment wrapper. Connects the agent to the game.
│
├── shared/
│   # Core configuration, enums, and data models used across layers.
│
├── examples/
│   # Demo game files, reference outputs, and expected world models.
│
├── scripts/
│   # Entrypoints for execution:
│   #  - run_demo.sh (1-command launcher)
│   #  - validate_submission.py (sanity checker)
│   #  - demo_live_textworld.py (core demo loop)
│
├── tests/
│   # Smoke, integration, and unit tests ensuring pipeline integrity.
│
├── docs/
│   # Reference documentation including the original problem statement.
│
├── README.md
├── SUBMISSION.md
├── PROJECT_STRUCTURE.md
├── DECISIONS.md
├── LICENSE
├── requirements.txt
├── requirements-lock.txt
└── pyproject.toml
```
