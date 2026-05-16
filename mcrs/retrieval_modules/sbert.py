"""Sentence-Transformer embedding-based retrieval for music track metadata.

This module replaces the vanilla BERT retriever with a proper retrieval encoder
(`sentence-transformers/all-MiniLM-L6-v2` by default) that was trained with a
contrastive objective for cosine-similarity search.  Token embeddings are pooled
by the model's built-in pooling layer (typically mean pooling), and the resulting
384-dim vectors are L2-normalised so that cosine similarity reduces to a dot product.
"""
import os
import json
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from datasets import concatenate_datasets
from mcrs.utils import load_local_or_hf_dataset
from sentence_transformers import SentenceTransformer


class SBERT_MODEL:
    """Sentence-Transformer retriever over track metadata.

    Builds an embedding index from specified corpus fields (e.g., `track_name`,
    `artist_name`, `album_name`, `tag_list`) and provides text-to-item retrieval
    via cosine similarity over sentence embeddings.
    """
    def __init__(self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        """Initialize the Sentence-Transformer retriever.

        Args:
            dataset_name: Hugging Face dataset name containing track metadata.
            split_types: Dataset splits to load and concatenate.
            corpus_types: Metadata fields to include in the text corpus.
            cache_dir: Directory to cache the embedding index and artifacts.
            model_name: Hugging Face model id for the sentence-transformer encoder.
            device: Torch device string. If None, chooses CUDA if available else CPU.
            batch_size: Batch size for embedding computation when building index.
        """
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.corpus_name = "_".join(corpus_types)
        self.cache_dir = cache_dir
        self.index_dir = os.path.join(self.cache_dir, "sbert", self.corpus_name)
        self.model_name = model_name
        self.batch_size = batch_size
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.metadata_dict = self._load_corpus()
        self.model = SentenceTransformer(self.model_name, device=self.device)

        if os.path.exists(os.path.join(self.index_dir, "embeddings.pt")) and \
           os.path.exists(os.path.join(self.index_dir, "track_ids.json")):
            self.embeddings, self.track_ids = self._load_index()
        else:
            self.build_index()
            self.embeddings, self.track_ids = self._load_index()

        # Move model to CPU after indexing — query encoding is fast on CPU
        # and this frees GPU VRAM for the LLM.
        if self.device != "cpu":
            print("[INFO] Moving SBERT retriever to CPU to free GPU memory for LLM.")
            self.model = SentenceTransformer(self.model_name, device="cpu")
            self.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_index(self) -> Tuple[torch.Tensor, List[str]]:
        """Load cached embedding matrix and track id list.

        Returns:
            A tuple of (embeddings [num_items, dim], track_ids).
        """
        embeddings = torch.load(os.path.join(self.index_dir, "embeddings.pt"), map_location="cpu")
        track_ids = json.load(open(os.path.join(self.index_dir, "track_ids.json"), "r"))
        return embeddings, track_ids

    def _load_corpus(self) -> Dict[str, Dict]:
        """Load and combine metadata splits from the configured dataset.

        Returns:
            A mapping from `track_id` to its metadata dictionary.
        """
        metadata_dataset = load_local_or_hf_dataset(self.dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in self.split_types])
        metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}
        return metadata_dict

    def _stringify_metadata(self, metadata: Dict[str, object]) -> str:
        """Convert a metadata dict into a multi-line string for indexing.

        Args:
            metadata: Track metadata with fields listed in `self.corpus_types`.

        Returns:
            A newline-separated string with `field: value` per selected field.
        """
        metadata_str = ""
        for corpus_type in self.corpus_types:
            entity = metadata[corpus_type]
            if isinstance(entity, list):
                entity = ", ".join(entity)
            metadata_str += f"{corpus_type}: {entity}\n"
        return metadata_str

    def build_index(self) -> None:
        """Build and persist an embedding index over the loaded corpus."""
        track_ids = list(self.metadata_dict.keys())
        corpus_texts = []
        for track_id in track_ids:
            metadata = self.metadata_dict[track_id]
            corpus_texts.append(self._stringify_metadata(metadata))

        os.makedirs(self.index_dir, exist_ok=True)
        print(f"[INFO] Building SBERT index for {len(corpus_texts)} tracks...")

        # SentenceTransformer.encode handles batching internally
        embedding_mat = self.model.encode(
            corpus_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,  # L2-normalise so dot product = cosine sim
        ).cpu()

        torch.save(embedding_mat, os.path.join(self.index_dir, "embeddings.pt"))
        with open(os.path.join(self.index_dir, "track_ids.json"), "w") as f:
            json.dump(track_ids, f, indent=2)
        print(f"[INFO] SBERT index built and saved to {self.index_dir}")

    def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
        """Retrieve top-k track IDs for a natural language query via cosine similarity.

        Args:
            query: The user text query to embed and compare against the corpus.
            topk: Number of items to retrieve.

        Returns:
            A list of track IDs ordered by decreasing cosine similarity.
        """
        query_emb = self.model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).cpu().squeeze(0)  # [dim]

        # dot product = cosine similarity since embeddings are L2-normalised
        scores = torch.matmul(self.embeddings, query_emb)  # [N]
        topk = min(topk, scores.shape[0])
        top_indices = torch.topk(scores, k=topk).indices.tolist()
        return [self.track_ids[i] for i in top_indices]

    def batch_text_to_item_retrieval(self, queries: List[str], topk: int) -> List[List[str]]:
        """Retrieve top-k track IDs for multiple queries in batch via cosine similarity.

        Args:
            queries: List of user text queries to embed and compare against the corpus.
            topk: Number of items to retrieve per query.

        Returns:
            A list of lists, where each inner list contains track IDs ordered by
            decreasing cosine similarity.
        """
        query_embs = self.model.encode(
            queries,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).cpu()  # [batch, dim]

        # [N, dim] @ [dim, batch] -> [N, batch]
        scores = torch.matmul(self.embeddings, query_embs.T)
        results = []
        topk = min(topk, scores.shape[0])
        for i in range(len(queries)):
            top_indices = torch.topk(scores[:, i], k=topk).indices.tolist()
            results.append([self.track_ids[idx] for idx in top_indices])
        return results
