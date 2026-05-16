# Hybrid BM25+SBERT Retrieval with Reciprocal Rank Fusion

**Date**: 2026-05-16

## Problem

BM25 (sparse, keyword-matching) and SBERT (dense, semantic similarity) each have complementary strengths:

| Retriever | Good at | Bad at |
|-----------|---------|--------|
| **BM25** | Exact entity matches ("play Queen", "Bohemian Rhapsody") | Abstract semantic queries ("something chill for a rainy day") |
| **SBERT** | Semantic similarity, mood/vibe queries | Exact entity names get diluted across 384-dim embedding space |

On the Codabench leaderboard, BM25 outperforms SBERT for this task because music queries are heavily entity-based. But neither alone is optimal.

## Solution: Reciprocal Rank Fusion (RRF)

We combine both retrievers using RRF, a simple and robust rank fusion method:

```
score(doc) = Σ  1 / (k + rank_i(doc))
```

where `k = 60` (smoothing constant from the original SIGIR 2009 paper) and the sum runs over each retrieval system that returned the document.

### Why RRF over score-level fusion?

- **No normalization needed** — BM25 and cosine similarity scores are on different scales; RRF only uses ranks
- **No hyperparameter tuning** — `k=60` is the standard default
- **Robust across domains** — proven in web search, academic IR, and RAG systems

### Pipeline

```
User Query
    │
    ├──→ BM25        → top-100 ranked tracks
    │
    ├──→ SBERT       → top-100 ranked tracks
    │
    └──→ RRF Fusion  → merged top-20 tracks
```

## Implementation

### New file: `mcrs/retrieval_modules/hybrid.py`

- `HYBRID_MODEL` class wraps both `BM25_MODEL` and `SBERT_MODEL`
- Constructor initializes both retrievers over the same corpus
- `text_to_item_retrieval(query, topk)` — fetches 100 candidates from each, fuses with RRF
- `batch_text_to_item_retrieval(queries, topk)` — batched version
- `_rrf_fuse(ranked_lists, k=60, topk=20)` — static method implementing RRF

### Factory registration: `mcrs/retrieval_modules/__init__.py`

```python
elif retrieval_type == "hybrid":
    return HYBRID_MODEL(dataset_name, track_split_types, corpus_types, cache_dir, ...)
```

### Configs created

| Config | Purpose |
|--------|---------|
| `config/local_hybrid_devset.yaml` | Local Mac (no GPU, no LLM) — retrieval-only eval |
| `config/local_sbert_devset.yaml` | Local SBERT-only baseline for comparison |
| `config/kaggle_llama1b_hybrid_devset.yaml` | Kaggle GPU — full pipeline with Llama |
| `config/kaggle_llama1b_hybrid_blindset_A.yaml` | Kaggle GPU — blind set A submission |

## Usage

### Local (requires one-time SBERT model download off corporate network)

```bash
# ONE-TIME: Download SBERT model at home (off Walmart network)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('./models/all-MiniLM-L6-v2')"

# Then run locally (works offline forever)
python run_inference_devset.py --tid local_hybrid_devset --batch_size 16
python run_evaluate_devset.py --tid local_hybrid_devset --config local_hybrid_devset
```

### Kaggle (no setup needed — SBERT auto-downloads)

```python
# In Kaggle notebook:
!python run_inference_devset.py --tid kaggle_llama1b_hybrid_devset --batch_size 2
```

## Expected Results

The hybrid approach should improve nDCG@20 over both individual retrievers by:
- Retaining BM25's exact match strength for entity queries
- Adding SBERT's semantic understanding for vibe/mood queries
- The RRF fusion naturally up-ranks documents that both systems agree on (consensus signal)

## Reference

Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods", SIGIR 2009.
