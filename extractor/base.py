# extractor/base.py
# ============================================================================
# Abstract base class for modality-agnostic extraction.
# Spec Reference: Implementation Plan v2.0, Section 4 (ExtractorBase)
#
# Track 1 (text) and future Track 2 (vision) both implement this interface.
# ============================================================================

from abc import ABC, abstractmethod
from typing import List

from shared.models import Observation, CandidateFact, WorkingMemory


class ExtractorBase(ABC):
    """
    Abstract base for modality-agnostic fact extraction.

    Any extractor (text, vision, or hybrid) must implement this interface.
    The downstream modules (Updater, Query Layer) are agnostic to the
    extraction modality — they only consume List[CandidateFact].
    """

    @abstractmethod
    def extract(
        self, observation: Observation, working_memory: WorkingMemory
    ) -> List[CandidateFact]:
        """
        Extract candidate facts from an observation.

        Args:
            observation: Current turn's Observation from the Environment Wrapper
            working_memory: Current room facts + recent context for coreference

        Returns:
            List of CandidateFact, each with confidence and provenance metadata.

        Guarantees (enforced by implementations):
            - All returned facts conform to the ontology schema
            - No fact has confidence outside [0.10, 0.99]
            - Entity names are normalized (lowercase, no articles)
            - Empty list returned for trivial/unparseable observations
        """
        ...
