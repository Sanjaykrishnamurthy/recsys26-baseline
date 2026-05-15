"""mcrs — Music Conversational Recommender System Baselines.

Quick start::

    from mcrs import load_crs_baseline

    crs = load_crs_baseline(
        lm_type="none",
        retrieval_type="bm25",
        device="mps",
    )
    result = crs.chat("Play me something upbeat")
"""

__version__ = "0.2.0"

import torch
from .crs_baseline import CRS_BASELINE
from .utils import load_local_or_hf_dataset

__all__ = [
    "CRS_BASELINE",
    "load_crs_baseline",
    "load_local_or_hf_dataset",
]


def load_crs_baseline(
    lm_type: str = "meta-llama/Llama-3.2-1B-Instruct",
    retrieval_type: str = "bm25",
    item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    track_split_types: list[str] = ["all_tracks"],
    user_split_types: list[str] = ["all_users"],
    corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
    cache_dir: str = "./cache",
    device: str = "cuda",
    attn_implementation: str = "eager",
    dtype: torch.dtype = torch.bfloat16,
    use_user_reranker: bool = False,
) -> CRS_BASELINE:
    """Create and return a fully initialized CRS baseline instance.

    Args:
        lm_type: LLM identifier — HF model ID, ``"ollama:<model>"``, or
            ``"none"`` for retrieval-only mode.
        retrieval_type: ``"bm25"`` (sparse) or ``"bert"`` (dense).
        item_db_name: HF dataset ID or local path for track metadata.
        user_db_name: HF dataset ID or local path for user profiles.
        track_split_types: Must be ``["all_tracks"]`` (competition rule).
        user_split_types: User dataset splits to load.
        corpus_types: Track metadata fields used for retrieval indexing.
        cache_dir: Directory for caching BM25/BERT indexes.
        device: ``"cuda"``, ``"mps"``, or ``"cpu"``.
        attn_implementation: ``"eager"`` or ``"flash_attention_2"``.
        dtype: Torch dtype for the LLM weights.
        use_user_reranker: Rerank candidates by user-profile similarity
            (requires ``retrieval_type="bert"``).

    Returns:
        A ready-to-use :class:`CRS_BASELINE` instance.
    """
    return CRS_BASELINE(
        lm_type, retrieval_type, item_db_name, user_db_name,
        track_split_types, user_split_types, corpus_types,
        cache_dir, device, attn_implementation, dtype,
        use_user_reranker=use_user_reranker,
    )
