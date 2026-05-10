"""User-profile BERT reranker for personalized track reranking.

After Stage 1 retrieval returns a candidate pool (e.g. top-100), this module
reranks the candidates by cosine similarity between a BERT embedding of the
user's profile and each candidate track's pre-computed embedding.

It deliberately shares the BERT encoder and track-embedding matrix already
built by BERT_MODEL, so no extra model loading or disk access is required.

Typical usage inside CRS_BASELINE:
    retrieval_items = retrieval.text_to_item_retrieval(query, topk=100)
    if reranker and user_id:
        user_profile = user_db.id_to_profile_str(user_id)
        retrieval_items = reranker.rerank(user_profile, retrieval_items, topk=20)
    else:
        retrieval_items = retrieval_items[:20]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List

# Avoid a circular import: import the class for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bert import BERT_MODEL


class UserBERTReranker:
    """Reranks retrieved track candidates using user-profile BERT embeddings.

    Borrows the encoder model and pre-computed track embedding matrix from an
    already-initialised BERT_MODEL instance — no extra GPU memory is used for
    the track side.

    Attributes:
        model: BERT encoder model (shared reference from BERT_MODEL).
        tokenizer: BERT tokenizer (shared reference from BERT_MODEL).
        embeddings: Pre-computed L2-normalised track embeddings [N, hidden].
        track_ids: List of track IDs aligned with `embeddings` rows.
        track_id_to_idx: Reverse lookup from track_id → row index in embeddings.
        device: Torch device string.
        max_length: Max tokenisation length (shared from BERT_MODEL).
    """

    def __init__(self, bert_retriever: "BERT_MODEL") -> None:
        """Initialise the reranker from an existing BERT_MODEL instance.

        Args:
            bert_retriever: A fully initialised BERT_MODEL whose encoder and
                            embedding matrix are reused directly.
        """
        self.model = bert_retriever.model
        self.tokenizer = bert_retriever.tokenizer
        self.embeddings = bert_retriever.embeddings          # [N, h], on CPU
        self.track_ids = bert_retriever.track_ids
        self.track_id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}
        self.device = bert_retriever.device
        self.max_length = bert_retriever.max_length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode a text string to an L2-normalised embedding vector.

        Args:
            text: Any plain-text string (user profile, query, etc.).

        Returns:
            A 1-D CPU tensor of shape [hidden_dim], L2-normalised.
        """
        self.model.eval()
        with torch.no_grad():
            batch = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            # Mean pool over non-padding tokens
            mask = batch["attention_mask"].unsqueeze(-1).expand(
                outputs.last_hidden_state.size()
            ).float()
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts                          # [1, h]
            emb = F.normalize(pooled, p=2, dim=1).cpu().squeeze(0)  # [h]
        return emb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        user_profile: str,
        candidate_ids: List[str],
        topk: int = 20,
    ) -> List[str]:
        """Rerank candidate track IDs by similarity to the user profile.

        Args:
            user_profile: Plain-text user profile string, e.g. the output of
                          ``UserProfileDB.id_to_profile_str(user_id)``.
            candidate_ids: Ordered list of track IDs from Stage 1 retrieval.
                           Typically 50–100 candidates.
            topk: Number of top tracks to return after reranking.

        Returns:
            A list of up to ``topk`` track IDs, ordered by descending
            cosine similarity to the user profile embedding.
        """
        # Encode user profile
        user_emb = self._encode_text(user_profile)           # [h]

        # Build index lookup for candidates; skip any IDs not in the matrix
        valid_candidates = [
            (tid, self.track_id_to_idx[tid])
            for tid in candidate_ids
            if tid in self.track_id_to_idx
        ]

        if not valid_candidates:
            # Fallback: return original order truncated to topk
            return candidate_ids[:topk]

        candidate_track_ids, candidate_indices = zip(*valid_candidates)
        candidate_embs = self.embeddings[list(candidate_indices)]  # [k, h]

        # Cosine similarity (embeddings are already L2-normalised)
        scores = torch.matmul(candidate_embs, user_emb)      # [k]

        k = min(topk, scores.shape[0])
        top_local_indices = torch.topk(scores, k=k).indices.tolist()

        return [candidate_track_ids[i] for i in top_local_indices]

    def batch_rerank(
        self,
        user_profiles: List[str],
        batch_candidate_ids: List[List[str]],
        topk: int = 20,
    ) -> List[List[str]]:
        """Rerank candidates for a batch of users.

        Args:
            user_profiles: List of user profile strings, one per item.
            batch_candidate_ids: List of candidate ID lists, one per item.
            topk: Number of tracks to return per item.

        Returns:
            List of reranked track ID lists, one per item.
        """
        return [
            self.rerank(profile, candidates, topk)
            for profile, candidates in zip(user_profiles, batch_candidate_ids)
        ]
