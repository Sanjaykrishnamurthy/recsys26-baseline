from .bm25 import BM25_MODEL
from .bert import BERT_MODEL
from .sbert import SBERT_MODEL
from .hybrid import HYBRID_MODEL
from .user_reranker import UserBERTReranker

def load_retrieval_module(
        retrieval_type: str,
        dataset_name: str,
        track_split_types: list[str],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir: str = "./cache",
        retrieval_model_name: str | None = None,
    ):
    if retrieval_type == "bm25":
        return BM25_MODEL(dataset_name, track_split_types, corpus_types, cache_dir)
    elif retrieval_type == "bert":
        kwargs = {}
        if retrieval_model_name:
            kwargs["model_name"] = retrieval_model_name
        return BERT_MODEL(dataset_name, track_split_types, corpus_types, cache_dir, **kwargs)
    elif retrieval_type == "sbert":
        kwargs = {}
        if retrieval_model_name:
            kwargs["model_name"] = retrieval_model_name
        return SBERT_MODEL(dataset_name, track_split_types, corpus_types, cache_dir, **kwargs)
    elif retrieval_type == "hybrid":
        kwargs = {}
        if retrieval_model_name:
            kwargs["sbert_model_name"] = retrieval_model_name
        return HYBRID_MODEL(dataset_name, track_split_types, corpus_types, cache_dir, **kwargs)
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type}")
