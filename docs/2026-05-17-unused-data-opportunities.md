# Unused Data & Low-Hanging Fruit Opportunities

**Date**: 2026-05-17

## Current Data Usage

Our pipeline currently uses only a fraction of the available datasets:

| Dataset | Used? | What we use |
|---------|:-----:|-------------|
| Track-Metadata | ✅ | `track_name`, `artist_name`, `album_name`, `release_date`, `tag_list` → BM25/SBERT retrieval corpus |
| Challenge-Dataset (test) | ✅ | `conversations` → chat history, user queries, ground truth |
| User-Metadata | ✅ | `age_group`, `gender`, `country_name` → system prompt personalization |
| Blind-A | ✅ | Blind test conversations → Codabench submission |
| **Challenge-Dataset (train)** | ❌ | **15,199 sessions completely ignored** |
| **Track-Embeddings** | ❌ | **6 pre-computed embedding types for all 47,071 tracks** |
| **User-Embeddings** | ❌ | **Collaborative filtering vectors for users** |

Even within the datasets we do use, several rich fields are ignored.

---

## Low-Hanging Fruit (ordered by effort vs impact)

### 1. Inject `conversation_goal` into system prompt

**Effort**: Low (prompt template change)
**Impact**: High (directly improves LLM-Judge score — 30% of composite)
**Available in test**: ✅ Yes

Every session (including test) has a `conversation_goal`:
```json
{
  "category": "H",
  "listener_goal": "The listener wants to explore different artists and discover new songs from a broad collection, focusing on general moods and styles, without specific genre constraints.",
  "specificity": "LL"
}
```

The LLM-Judge evaluates **personalization** and **explanation quality**. If the LLM knows the user's goal, it can tailor responses ("Since you're looking to explore new artists..." vs generic recommendations).

**Implementation**: Add `conversation_goal` to `_get_system_prompt()` and pass it through `batch_chat()`. Create a new prompt template `system_prompts/conversation_goal.txt`.

---

### 2. Use `preferred_musical_culture` from `user_profile`

**Effort**: Low (already in the dataset, just not extracted)
**Impact**: Medium (improves both retrieval relevance and LLM personalization)
**Available in test**: ✅ Yes

The `user_profile` embedded in each session contains:
```json
{
  "age": 20,
  "age_group": "20s",
  "country_name": "Brazil",
  "gender": "female",
  "preferred_language": "English",
  "preferred_musical_culture": "Western Alternative Rock"
}
```

We currently get `age_group`, `gender`, `country_name` from User-Metadata. But `preferred_musical_culture` is only in the Challenge-Dataset's `user_profile` field and we never read it.

**Implementation**: Extract `preferred_musical_culture` from the session data and inject it into the system prompt alongside the existing user profile.

---

### 3. Use `thought` field for few-shot prompting

**Effort**: Medium (prompt engineering)
**Impact**: Medium (improves LLM response quality)
**Available in test**: ❌ Only in train (but can be used as few-shot examples)

The `music` role in training conversations has a `thought` field explaining **why** a track was recommended:
```
"You enjoyed the last Alesana track, so I'm sticking with your
 favorite artist and genre. 'The Fiend' carries that same intense
 post-hardcore energy and screamo vocals that define Alesana's sound."
```

This is ground-truth reasoning that could be used as few-shot examples in the system prompt to teach the LLM how to explain recommendations.

**Implementation**: Select 2-3 high-quality train examples, format them as few-shot demonstrations in `system_prompts/response_generation.txt`.

---

### 4. Use pre-computed Track-Embeddings for retrieval

**Effort**: Medium (new retrieval module)
**Impact**: High (richer signal than text-only retrieval)
**Available in test**: ✅ Yes (embeddings cover all 47,071 tracks)

Six pre-computed embedding types per track:

| Embedding | Dim | Signal |
|-----------|-----|--------|
| `audio-laion_clap` | 512 | Actual audio content (sound similarity) |
| `image-siglip2` | 768 | Album art visual style |
| `cf-bpr` | 128 | Collaborative filtering ("users who liked X also liked Y") |
| `attributes-qwen3_embedding_0.6b` | 1024 | Track attributes via Qwen3 |
| `lyrics-qwen3_embedding_0.6b` | 1024 | Lyrics content via Qwen3 |
| `metadata-qwen3_embedding_0.6b` | 1024 | Metadata via Qwen3 |

The `metadata-qwen3_embedding_0.6b` embeddings are especially interesting — they're Qwen3-generated embeddings of track metadata, likely much stronger than our all-MiniLM-L6-v2 SBERT embeddings (1024-dim vs 384-dim, trained on a more powerful base model).

**Implementation**: Create a new retrieval module that loads pre-computed embeddings and does cosine similarity search. Can be fused with BM25 via RRF (same as our current hybrid approach).

---

### 5. Use `cf-bpr` User-Embeddings for personalized reranking

**Effort**: Medium (new reranker module)
**Impact**: High (personalized recommendations — directly improves nDCG)
**Available in test**: ⚠️ Partial (371 "warm" users have embeddings, 129 "cold" users don't)

Both tracks and users have `cf-bpr` (128-dim) collaborative filtering embeddings. For warm users, you can compute `user_cf · track_cf` cosine similarity to score how likely a user is to enjoy a track based on collaborative signals.

**Implementation**: After Stage 1 retrieval (BM25/SBERT/hybrid), rerank candidates by `cf-bpr` similarity for warm users. Fall back to original ranking for cold users.

---

### 6. Mine training data for popularity priors

**Effort**: Low-Medium
**Impact**: Medium (improves catalog diversity and nDCG for common tracks)
**Available in test**: N/A (derived from train)

The 15,199 training sessions contain ground-truth track recommendations. Mining which tracks are recommended most frequently gives a popularity prior that can be used as a tiebreaker in retrieval or as a fallback when retrieval confidence is low.

**Implementation**: Count track frequencies in train conversations. Add a small popularity bonus to retrieval scores.

---

## Priority Matrix

| Opportunity | Effort | Impact on nDCG (50%) | Impact on LLM-Judge (30%) | Impact on Diversity (20%) |
|-------------|--------|:--------------------:|:-------------------------:|:-------------------------:|
| 1. conversation_goal in prompt | Low | — | ⬆️ High | — |
| 2. preferred_musical_culture | Low | ⬆️ Low | ⬆️ Medium | — |
| 3. Few-shot from thought field | Medium | — | ⬆️ Medium | ⬆️ Low |
| 4. Pre-computed track embeddings | Medium | ⬆️ High | — | ⬆️ Medium |
| 5. CF-BPR personalized reranking | Medium | ⬆️ High | — | ⬆️ Medium |
| 6. Popularity priors from train | Low | ⬆️ Low | — | ⬇️ Negative |

**Recommended order**: 1 → 2 → 4 → 5 → 3 → 6
