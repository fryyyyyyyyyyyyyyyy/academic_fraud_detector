"""
Custom tools for the Academic Fraud Detection Crew.

Imports are lazy/guarded to allow testing individual tool modules
without requiring the full CrewAI dependency chain.
"""

import logging

logger = logging.getLogger(__name__)

__all__ = [
    # Paper fetching
    "ArxivSearchTool",
    "CrossrefSearchTool",
    "SemanticScholarSearchTool",
    "PaperLookupTool",
    "LocalPaperLoaderTool",
    # Text similarity
    "SemanticSimilarityTool",
    "lexical_plagiarism_check",
    # Image forensics
    "ELATool",
    "CloneDetectionTool",
    "AIImageDetectionTool",
    "CrossImageDuplicateTool",
    "BackgroundConsistencyTool",
    # Statistical
    "BenfordLawTool",
    "PValueDistributionTool",
    "GRIMTestTool",
    "AnomalousPrecisionTool",
    "StatisticalConsistencyTool",
    # Citation
    "CitationGraphTool",
    "SelfCitationTool",
    # Peer review
    "ReviewTextAnalyzerTool",
    "ReviewerCredentialCheckerTool",
    "ReviewTemplateDetectorTool",
]

try:
    from .paper_fetching import (
        ArxivSearchTool,
        CrossrefSearchTool,
        SemanticScholarSearchTool,
        PaperLookupTool,
        LocalPaperLoaderTool,
    )
except ImportError as e:
    logger.debug(f"Paper fetching tools not available: {e}")

try:
    from .text_similarity import SemanticSimilarityTool, lexical_plagiarism_check
except ImportError as e:
    logger.debug(f"Text similarity tools not available: {e}")

try:
    from .image_forensics import (
        ELATool, CloneDetectionTool, AIImageDetectionTool,
        CrossImageDuplicateTool, BackgroundConsistencyTool,
    )
except ImportError as e:
    logger.debug(f"Image forensics tools not available: {e}")

try:
    from .statistical_analysis import (
        BenfordLawTool, PValueDistributionTool, GRIMTestTool,
        AnomalousPrecisionTool, StatisticalConsistencyTool,
    )
except ImportError as e:
    logger.debug(f"Statistical analysis tools not available: {e}")

try:
    from .citation_analysis import CitationGraphTool, SelfCitationTool
except ImportError as e:
    logger.debug(f"Citation analysis tools not available: {e}")

try:
    from .peer_review_analysis import (
        ReviewTextAnalyzerTool,
        ReviewerCredentialCheckerTool,
        ReviewTemplateDetectorTool,
    )
except ImportError as e:
    logger.debug(f"Peer review analysis tools not available: {e}")
