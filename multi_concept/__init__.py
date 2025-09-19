"""
Multi-Concept Zero-Shot YOLO Package

This package implements concept-specific zero-shot detection where each concept vector
is used to detect its corresponding object class.

Key Innovation: Concept-Class Mapping for True Zero-Shot Detection
"""

from .detect import MultiConceptGuidedDetect
from .mapper import ConceptSpecificMapper
from .model import MultiConceptZeroShotYOLO
from .results import MultiConceptSemanticResults

__all__ = [
    "MultiConceptSemanticResults",
    "ConceptSpecificMapper",
    "MultiConceptGuidedDetect",
    "MultiConceptZeroShotYOLO",
]
