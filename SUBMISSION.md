# Hackathon Submission Guide

Welcome! This guide is designed to help you quickly set up and evaluate the **World Model Agent** submission for Track 1. 

## 1. Install Dependencies
Clone the repository and install the minimal required packages:
```bash
pip install -r requirements.txt
```

## 2. Run the Validation Script
We have provided an automated script to verify that your environment (Ollama, TextWorld) is correctly configured and to execute a lightweight smoke test of the agent loop.
```bash
python scripts/validate_submission.py
```

## 3. Run the Live Demo
Execute the full agent pipeline on a TextWorld environment. The agent will extract observations, build a graph-based world model, formulate queries, and determine actions using a local SLM.
```bash
./scripts/run_demo.sh
```

## 4. Observe the World Model
As the agent runs, pay attention to the logs. You will see:
- The **Extractor** converting raw text into semantic triples.
- The **Updater** adding/removing edges in the persistent world graph.
- The **Query Layer** filtering the graph into a minimal contextual slice.

## 5. Expected Output
Check the `examples/` directory for a reference of what a successful execution looks like:
- `examples/demo_output.txt`: A trace of a successful run.
- `examples/expected_world_model.json`: The expected graph state.

## 6. Repository Layout
Please see `PROJECT_STRUCTURE.md` for a complete breakdown of how the architecture maps to the repository folders.

## 7. Known Limitations
- The Extractor relies on a fallback heuristic rule system if the SLM fails to generate valid JSON or encounters a context boundary.
- Visualizations are limited to terminal output during standard runs; web UI is optional.
- Apple Silicon (M1) may have slow inference times for larger context models.
