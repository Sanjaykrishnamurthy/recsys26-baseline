"""mcrs.evaluation — Evaluation metrics for Music CRS.

Integrated from the official music-crs-evaluator:
https://github.com/nlp4musa/music-crs-evaluator

Provides retrieval quality (nDCG) and diversity metrics used by the
RecSys Challenge 2026 Codabench leaderboard.
"""

from .metrics_recsys import compute_recsys_metrics
from .metrics_diversity import compute_catalog_diversity, compute_lexical_diversity

__all__ = [
    "compute_recsys_metrics",
    "compute_catalog_diversity",
    "compute_lexical_diversity",
]
