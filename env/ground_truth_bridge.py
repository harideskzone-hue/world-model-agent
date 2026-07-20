# env/ground_truth_bridge.py
# ============================================================================
# EVALUATION-ONLY bridge to TextWorld ground truth.
# Spec Reference: Implementation Plan v2.0, Section 2 (Risks & Mitigations)
#
# ██████████████████████████████████████████████████████████████████████████
# ██  WARNING: This module exists ONLY for evaluation/scoring.            ██
# ██  It must NEVER be imported by any module in the live agent path:     ██
# ██    extractor/, world_model/, updater/, query_layer/, slm/,           ██
# ██    orchestrator/                                                     ██
# ██████████████████████████████████████████████████████████████████████████
# ============================================================================

from __future__ import annotations

from typing import List, Set, Tuple

from shared.models import GroundTruthFact, Edge
from shared.enums import RelationType, EdgeStatus


def convert_tw_facts_to_triples(
    facts: List[GroundTruthFact],
) -> Set[Tuple[str, str, str]]:
    """
    Convert TextWorld GroundTruthFacts into normalized (subject, relation, object) triples
    for comparison against the agent's beliefs.

    Args:
        facts: TextWorld's internal propositions

    Returns:
        Set of (subject, relation, object) tuples, normalized to lowercase
    """
    triples = set()
    for fact in facts:
        name = fact.name.lower()
        args = [a.lower() for a in fact.arguments]
        if len(args) >= 2:
            triples.add((args[0], name, args[1]))
        elif len(args) == 1:
            triples.add((args[0], name, "true"))
    return triples


def convert_beliefs_to_triples(
    edges: List[Edge],
) -> Set[Tuple[str, str, str]]:
    """
    Convert the agent's active belief edges into normalized triples
    for comparison against ground truth.
    """
    triples = set()
    for edge in edges:
        if edge.status == EdgeStatus.ACTIVE:
            triples.add((
                edge.subject.lower(),
                edge.relation.value.lower(),
                edge.object.lower(),
            ))
    return triples


def compute_belief_accuracy(
    agent_beliefs: List[Edge],
    ground_truth: List[GroundTruthFact],
) -> dict:
    """
    Compute precision, recall, and F1 of agent beliefs vs ground truth.

    Returns:
        Dict with keys: precision, recall, f1, true_positives, false_positives, false_negatives
    """
    belief_triples = convert_beliefs_to_triples(agent_beliefs)
    truth_triples = convert_tw_facts_to_triples(ground_truth)

    true_positives = belief_triples & truth_triples
    false_positives = belief_triples - truth_triples
    false_negatives = truth_triples - belief_triples

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }
