"""Hybrid BM25 + SBERT retrieval with Reciprocal Rank Fusion (RRF).

Combines sparse (BM25) and dense (SBERT) retrieval to capture both exact
keyword matches and semantic similarity.  Results are merged using RRF,
a simple, parameter-free fusion method that is robust across domains.

RRF formula:
    score(doc) = Σ  1 / (k + rank_i(doc))
where k is a smoothing constant (default 60, per the original paper) and
the sum runs over each retrieval system i that returned the document.

Reference:
    Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms
    Condorcet and Individual Rank Learning Methods", SIGIR 2009.
"""

from __future__ import annotations

from typing import Dict, List

from .bm25 import BM25_MODEL
from .sbert import SBERT_MODEL


class HYBRID_MODEL:
    """Hybrid BM25 + SBERT retriever with Reciprocal Rank Fusion.

    Internally instantiates a BM25_MODEL and an SBERT_MODEL over the same
    corpus, retrieves a wide candidate pool from each, then fuses the two
    ranked lists via RRF to produce a single combined ranking.
    """

    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
        retrieval_depth: int = 100,
    ) -> None:
        """Initialize both BM25 and SBERT retrievers.

        Args:
            dataset_name: Hugging Face dataset name or local path for track metadata.
            split_types: Dataset splits to load and concatenate.
            corpus_types: Metadata fields to include in the text corpus.
            cache_dir: Directory to cache indexes and embeddings.
            sbert_model_name: Hugging Face model ID for the sentence-transformer.
            rrf_k: RRF smoothing constant (default 60, per the original paper).
            retrieval_depth: Number of candidates to fetch from each retriever
                before fusion. Should be >= the final topk you'll request.
        """
        self.rrf_k = rrf_k
        self.retrieval_depth = retrieval_depth

        print("[HYBRID] Initializing BM25 retriever ...")
        self.bm25 = BM25_MODEL(dataset_name, split_types, corpus_types, cache_dir)

        print("[HYBRID] Initializing SBERT retriever ...")
        self.sbert = SBERT_MODEL(
            dataset_name, split_types, corpus_types, cache_dir,
            model_name=sbert_model_name,
        )
        print(f"[HYBRID] Ready — RRF(k={rrf_k}), depth={retrieval_depth}")

    @staticmethod
    def _rrf_fuse(
        ranked_lists: list[list[str]],
        k: int = 60,
        topk: int = 20,
    ) -> list[str]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            ranked_lists: List of ranked ID lists (one per retriever).
            k: RRF smoothing constant.
            topk: Number of items to return after fusion.

        Returns:
            A single ranked list of IDs ordered by decreasing RRF score.
        """
        scores: Dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

        # Sort by score descending, break ties by earliest appearance
        sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        return sorted_docs[:topk]

    def text_to_item_retrieval(self, query: str, topk: int = 20) -> list[str]:
        """Retrieve top-k track IDs using hybrid BM25+SBERT with RRF fusion.

        Args:
            query: The user's natural language query.
            topk: Number of items to return.

        Returns:
            A list of track IDs ordered by decreasing RRF score.
        """
        depth = max(self.retrieval_depth, topk)
        bm25_results = self.bm25.text_to_item_retrieval(query, topk=depth)
        sbert_results = self.sbert.text_to_item_retrieval(query, topk=depth)
        return self._rrf_fuse([bm25_results, sbert_results], k=self.rrf_k, topk=topk)

    def batch_text_to_item_retrieval(
        self, queries: list[str], topk: int = 20
    ) -> list[list[str]]:
        """Retrieve top-k track IDs for multiple queries using hybrid retrieval.

        Args:
            queries: List of user text queries.
            topk: Number of items to retrieve per query.

        Returns:
            A list of lists of track IDs, each ordered by decreasing RRF score.
        """
        depth = max(self.retrieval_depth, topk)
        bm25_batch = self.bm25.batch_text_to_item_retrieval(queries, topk=depth)
        sbert_batch = self.sbert.batch_text_to_item_retrieval(queries, topk=depth)

        results = []
        for bm25_results, sbert_results in zip(bm25_batch, sbert_batch):
            fused = self._rrf_fuse(
                [bm25_results, sbert_results], k=self.rrf_k, topk=topk
            )
            results.append(fused)
        return results
