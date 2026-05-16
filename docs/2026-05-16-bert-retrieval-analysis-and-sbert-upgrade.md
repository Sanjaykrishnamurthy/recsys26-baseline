# BERT Retrieval Failure Analysis & Sentence-Transformer Upgrade

**Date**: 2026-05-16
**Status**: Implementing

---

## Problem

The `Bert_Llama` baseline scored **nDCG@20 = 0** on Codabench (rank 63, composite 0.06), while `bm25_llama` scored **nDCG@20 = 0.15** (rank 49, composite 0.19). A zero nDCG means the BERT retriever never places the ground-truth track in the top-20 results — effectively random retrieval.

## Root Cause Analysis

### 1. `bert-base-uncased` is not a retrieval model (Primary cause)

`bert-base-uncased` was pre-trained with Masked Language Modeling (MLM). Its raw mean-pooled embeddings are designed for language *understanding* tasks (NER, classification), **not** for semantic similarity search. Cosine similarity between these embeddings is essentially meaningless for retrieval — a well-documented issue that motivated the Sentence-BERT paper (Reimers & Gurevych, 2019).

BM25 doesn't have this problem since it uses direct lexical matching (TF-IDF term overlap).

### 2. 128-token truncation loses later turns

`bert.py` uses `max_length=128`. The retrieval query is the **full conversation history** concatenated as `"role: content\n..."`. Each turn is ~20–50 tokens, so by turn 3–4 the conversation exceeds 128 tokens. The tokenizer truncates from the **end**, meaning BERT loses the latest (most relevant) user query for turns 4–8. BM25 has no length limit.

### 3. Query–document length asymmetry

Mean-pooling over a long multi-turn conversation dilutes the embedding into a vague average of everything discussed, rather than capturing the user's current intent. The document embeddings are short metadata strings (~20 tokens). This asymmetry degrades cosine similarity quality.

### 4. BM25 explicit lowercasing

Minor: BM25 calls `query.lower()` explicitly. BERT's uncased tokenizer should handle this, but it's one less edge case for BM25.

## Solution: Three-pronged fix

### A. Swap to `sentence-transformers/all-MiniLM-L6-v2`

A proper retrieval encoder trained with contrastive learning for semantic similarity. Key properties:
- **384-dim embeddings** (vs. 768 for BERT) — faster cosine similarity, less memory
- **Trained on 1B+ sentence pairs** with contrastive objective
- **Designed for cosine similarity** — embeddings are meaningful in similarity space
- Small model (~80MB) — minimal GPU/memory impact

### B. Use only the latest user query for retrieval

Instead of concatenating the full conversation history, pass only `user_query` to the retriever. This:
- Eliminates truncation issues entirely
- Removes noise from assistant responses and earlier off-topic turns
- Focuses the embedding on the user's current intent
- Applied at the `crs_baseline.py` orchestration level so it benefits ALL retrieval modules

### C. Add `tag_list` to corpus_types

`tag_list` contains genre/mood/style tags (e.g., `["jazz", "smooth", "instrumental"]`). Adding it to the indexed metadata gives the retriever richer signal to match against user queries like "I want something jazzy and chill."

## Files Changed

| File | Change |
|------|--------|
| `mcrs/retrieval_modules/sbert.py` | New retrieval module using `sentence-transformers` |
| `mcrs/retrieval_modules/__init__.py` | Register `sbert` in the factory |
| `mcrs/crs_baseline.py` | Use `user_query` instead of full history for retrieval |
| `pyproject.toml` | Add `sentence-transformers` dependency |
| `config/llama1b_sbert_devset.yaml` | New config for dev set inference |
| `config/llama1b_sbert_blindset_A.yaml` | New config for blind set submission |

## Expected Impact

| Metric | Before (BERT) | Expected | Why |
|--------|---------------|----------|-----|
| nDCG@20 | 0 | > 0.15 | Proper retrieval embeddings + focused query |
| Catalog Diversity | 0.01 | ~0.03+ | Semantic retrieval covers more catalog |
| Lexical Diversity | 0.61 | ~0.61 | Unchanged (LLM-side) |
| LLM Score | 1 | ~1.5+ | Better top-1 recommendation improves response |

## Validation

Run a small batch to verify before full inference:
```bash
python run_inference_devset.py --tid llama1b_sbert_devset --batch_size 2
```
Inspect `exp/inference/devset/llama1b_sbert_devset.json` — confirm `predicted_track_ids` are non-trivial and vary across sessions.
