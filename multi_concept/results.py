"""
Multi-Concept Semantic Results

Extended Results class with multi-concept semantic detection support.
"""

from ultralytics.engine.results import Results


class MultiConceptSemanticResults(Results):
    """Extended Results class with multi-concept semantic detection support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concept_detections = {}  # concept_name -> detections
