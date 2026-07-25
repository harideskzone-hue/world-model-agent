# Architectural Decisions

This document briefly outlines the core design decisions made for the World Model Agent, specifically answering *why* the architecture was built this way for the Track 1 submission.

## Why a structured world model?
Traditional LLM agents rely on a rolling context window of raw text observations. This leads to unbounded context growth, hallucination over long horizons, and exponential inference costs. A structured world model decouples "what the agent knows" from "what the agent is currently prompted with." By maintaining a separate state representation, the agent's context window size is strictly bounded by its immediate surroundings, allowing it to run indefinitely without performance degradation.

## Why not conversation history?
Conversation history is a poor representation of a physical environment. If the agent moves back and forth between two rooms, conversation history duplicates the descriptions, confusing the model. A graph-based world model correctly merges these observations, maintaining a single source of truth.

## Why semantic extraction?
Raw text from a TextWorld game contains flavor text and irrelevant details. The Extractor distills this into a rigid ontology (nodes, edges, properties) using `Subject-Relation-Object` triples. This structural enforcement ensures that the Updater can reliably diff new observations against the existing graph.

## Why belief revision?
Environments change. A door that was previously closed might now be open. The Updater module explicitly checks for logical contradictions between new extractions and the existing world graph. Instead of just appending new facts, it implements a belief revision policy that actively overwrites stale edges, preventing the world model from becoming a garbage dump of historical facts.

## Why graph storage?
TextWorld environments are inherently spatial and relational (Rooms contain Containers; Containers contain Items; Rooms connect to Rooms). A graph database (even implemented simply in-memory) perfectly mirrors this topology. It allows for efficient sub-graph extraction via BFS/DFS traversal when the agent needs context.

## Why a query layer?
The entire world model might encompass hundreds of rooms and thousands of items—far too large for a local SLM's context window. The Query Layer acts as a filter, injecting only the *locally relevant* subset of the graph (typically 1-2 hops from the agent's current location) into the prompt.

## Why a local SLM?
Using a local SLM (e.g., `qwen2.5:3b` via Ollama) proves that this architecture allows highly capable autonomy without massive 100B+ parameter models. Because the context is meticulously curated by the Query Layer, the SLM only has to perform the simplest task: Next Action Prediction based on a pristine, isolated state slice.

## Known limitations
1. **Extraction Bottlenecks**: The system is highly dependent on the initial extraction step. If the SLM fails to extract a key object, the Updater cannot add it to the graph, and the agent becomes permanently blind to it.
2. **Zero-Shot Only**: The system uses purely zero-shot SLM prompts with no few-shot examples or keyword heuristics, which maximizes generalization but may reduce recall on edge-case observations.
3. **Graph Serialization**: Currently, the graph is serialized into a textual list of triples for the SLM prompt. While effective, a native graph-neural approach might yield better spatial reasoning in future iterations.
