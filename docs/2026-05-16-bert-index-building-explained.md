# BERT Index Building — How It Works and Why GPU→CPU Matters

> **Last updated:** 2026-05-16  
> A deep-dive into how the BERT retrieval module builds its embedding index, why GPU is needed for indexing but not for inference, and how we move the model to CPU to free VRAM for Llama.

---

## Table of Contents

- [The Big Idea](#the-big-idea)
- [Phase 1: Index Building (one-time, GPU)](#phase-1-index-building-one-time-gpu)
  - [Step-by-Step Walkthrough](#step-by-step-walkthrough)
  - [Why GPU is Needed](#why-gpu-is-needed)
  - [Code Walkthrough: `build_index()`](#code-walkthrough-build_index)
- [Phase 2: Query Retrieval (every call, CPU)](#phase-2-query-retrieval-every-call-cpu)
  - [Why CPU is Fine Here](#why-cpu-is-fine-here)
  - [Code Walkthrough: `text_to_item_retrieval()`](#code-walkthrough-text_to_item_retrieval)
- [The GPU→CPU Move](#the-gpucpu-move)
  - [Memory Timeline](#memory-timeline)
  - [Code: How the Move Happens](#code-how-the-move-happens)
- [Key Takeaways](#key-takeaways)

---

## The Big Idea

Think of index building like creating a library's card catalog:

1. **Build the catalog once** — read every book, write a summary card for each (slow, heavy work)
2. **Look things up forever** — flip through the cards to find what you need (fast, lightweight)

In our system:
- **"Books"** = 47,071 tracks in the music catalog
- **"Summary cards"** = 768-dimensional embedding vectors (numerical fingerprints)
- **"Looking things up"** = computing cosine similarity between a user query and all track vectors

---

## Phase 1: Index Building (one-time, GPU)

The goal is to convert every track into a **numerical fingerprint** (embedding vector) that captures its semantic meaning.

### Step-by-Step Walkthrough

```
47,071 tracks in the catalog
        │
        ▼
Each track's metadata gets stringified:
    "track_name: Bohemian Rhapsody
     artist_name: Queen
     album_name: A Night at the Opera
     tag_list: rock, classic, epic"
        │
        ▼
BERT reads each string and produces a 768-dimensional vector:
    [0.023, -0.156, 0.891, ..., 0.044]
    ↑ this IS the track's meaning represented as numbers
        │
        ▼
All 47,071 vectors stacked into a matrix: [47071 × 768]
        │
        ▼
Saved to disk: cache/bert/embeddings.pt (~138 MB file)
```

### Why GPU is Needed

BERT must process **47,071 texts** through a neural network. Each text passes through 12 transformer layers, each performing millions of matrix multiplications.

| | GPU (T4) | CPU |
|---|---|---|
| Process 47K tracks | ~2–3 minutes | ~30–60 minutes |
| Parallelism | Thousands of CUDA cores | 4–16 CPU cores |
| Matrix multiply speed | Optimized tensor cores | General-purpose ALUs |

The GPU's massive parallelism makes index building **10–20× faster**. For a one-time job over 47K items, that speedup matters.

### Code Walkthrough: `build_index()`

From `mcrs/retrieval_modules/bert.py`:

```python
def build_index(self) -> None:
    track_ids = list(self.metadata_dict.keys())
    corpus_texts = []
    for track_id in track_ids:
        metadata = self.metadata_dict[track_id]
        corpus_texts.append(self._stringify_metadata(metadata))

    # Process in batches of 32 to fit in GPU memory
    embeddings: List[torch.Tensor] = []
    self.model.eval()
    with torch.no_grad():
        for start in range(0, len(corpus_texts), self.batch_size):
            batch_texts = corpus_texts[start:start + self.batch_size]

            # Convert text → token IDs (numbers BERT understands)
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Send tokens to GPU — this is where GPU speed matters
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Run through all 12 BERT layers — HEAVY computation
            # For 32 texts × ~128 tokens × 12 layers × 768 dimensions
            # = millions of matrix multiplications. GPU does these in parallel.
            outputs = self.model(**batch)

            # Average all token vectors into ONE vector per track
            # "Bohemian Rhapsody by Queen" → [0.023, -0.156, ..., 0.044]
            pooled = self._mean_pool(
                outputs.last_hidden_state, batch["attention_mask"]
            )

            # Normalize to unit length (for cosine similarity later)
            pooled = F.normalize(pooled, p=2, dim=1)

            # Move result to CPU (save GPU memory for next batch)
            embeddings.append(pooled.detach().cpu())

    # Stack all vectors and save to disk
    embedding_mat = torch.cat(embeddings, dim=0)       # [47071, 768]
    torch.save(embedding_mat, "cache/bert/embeddings.pt")
```

**Key details:**

- `self.model.eval()` — disables dropout (we're not training, just encoding)
- `torch.no_grad()` — skips gradient computation (saves ~50% memory)
- `pooled.detach().cpu()` — moves each batch's result to CPU immediately, so GPU only holds one batch at a time
- The final `embedding_mat` is a `[47071 × 768]` matrix (~138 MB) saved to disk

---

## Phase 2: Query Retrieval (every call, CPU)

When a user asks *"something chill and jazzy"*, we:

1. **Encode** that one short text through BERT → 768-dim vector
2. **Compare** it against all 47,071 track vectors via cosine similarity
3. **Return** the top-20 highest-scoring track IDs

```python
# Simplified view of text_to_item_retrieval()
query_emb = BERT("something chill and jazzy")  # → [768] vector, ~5ms on CPU
scores = embeddings @ query_emb                  # → [47071] similarity scores
top_20 = scores.topk(20)                         # → 20 best track IDs
```

### Why CPU is Fine Here

| Metric | Index Building | Query Encoding |
|--------|---------------|----------------|
| Inputs | 47,071 texts | 1 text |
| BERT forward passes | ~1,471 batches | 1 pass |
| Time on GPU | ~2–3 min | ~1 ms |
| Time on CPU | ~30–60 min | **~5 ms** |
| GPU worth it? | ✅ Absolutely | ❌ 4ms savings, not worth 440MB VRAM |

Encoding 1 text through BERT takes ~5ms on CPU vs ~1ms on GPU. That 4ms difference is negligible — but keeping BERT on GPU permanently costs **440MB of VRAM** that Llama desperately needs for generation.

### Code Walkthrough: `text_to_item_retrieval()`

```python
def text_to_item_retrieval(self, query: str, topk: int) -> List[str]:
    self.model.eval()
    with torch.no_grad():
        # Tokenize the single query
        batch = self.tokenizer(
            [query], padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        # Send to whatever device BERT is on (CPU after our fix)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # One forward pass through BERT — trivial on CPU for 1 text
        outputs = self.model(**batch)

        # Mean-pool → single 768-dim vector
        query_emb = self._mean_pool(
            outputs.last_hidden_state, batch["attention_mask"]
        )
        query_emb = F.normalize(query_emb, p=2, dim=1).cpu().squeeze(0)

    # Cosine similarity: [47071, 768] @ [768] → [47071] scores
    # Both embeddings and query_emb are on CPU — fast dot product
    scores = torch.matmul(self.embeddings, query_emb)

    # Return top-20 track IDs
    top_indices = torch.topk(scores, k=topk).indices.tolist()
    return [self.track_ids[i] for i in top_indices]
```

**Why cosine similarity works here:**
- Both the track embeddings and query embedding are L2-normalized (unit length)
- For unit vectors: `cosine_similarity(a, b) = a · b` (just a dot product)
- So `embeddings @ query_emb` gives us cosine similarity scores directly
- Tracks whose embeddings point in a similar direction to the query get high scores

---

## The GPU→CPU Move

### Memory Timeline

```
TIME →
                    Index Building              Inference (×80 sessions)
                    ──────────────              ────────────────────────
GPU 0 VRAM:
  Llama (7 GB)      ████████████████████████████████████████████████████
  BERT  (440 MB)    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                                    ↑
                            model.to("cpu") — BERT leaves GPU
                            empty_cache()   — VRAM reclaimed

CPU RAM:
  BERT  (440 MB)    ░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  Embeddings (138MB)████████████████████████████████████████████████████
```

### Code: How the Move Happens

From `BERT_MODEL.__init__()`:

```python
# Step 1: BERT loads on GPU for fast index building
self.model.to("cuda").eval()

# Step 2: Build or load the index (47K embeddings)
if cached:
    self.embeddings, self.track_ids = self._load_index()   # From disk → CPU
else:
    self.build_index()       # Uses GPU — 2-3 min instead of 30-60 min
    self.embeddings, ... = self._load_index()

# Step 3: Index is done. BERT's heavy job is finished.
# Move it to CPU — we only need it for tiny query encoding from now on.
if self.device != "cpu":
    print("[INFO] Moving BERT retriever to CPU to free GPU memory for LLM.")
    self.model.to("cpu")        # Moves all 440MB of weights to CPU RAM
    self.device = "cpu"         # Future .to(self.device) calls go to CPU
    torch.cuda.empty_cache()    # Tell PyTorch to release the freed VRAM
```

**What each line does:**

| Line | Effect |
|------|--------|
| `self.model.to("cpu")` | Copies all BERT weight tensors from GPU VRAM to CPU RAM, then frees the GPU copies |
| `self.device = "cpu"` | Ensures all future tokenized inputs are sent to CPU (not GPU) via `batch.to(self.device)` |
| `torch.cuda.empty_cache()` | PyTorch's GPU allocator holds freed memory in a pool for reuse. This call releases it back to CUDA so Llama can use it |

---

## Key Takeaways

1. **Index building = heavy batch job** — 47K items through a neural network. GPU makes it 10–20× faster. Worth the VRAM cost temporarily.

2. **Query encoding = tiny per-request job** — 1 short text through the same network. ~5ms on CPU. Not worth 440MB of permanent GPU residency.

3. **`model.to("cpu")` is not destructive** — the model works identically on CPU, just slightly slower per forward pass. For 1-text encoding, the difference is imperceptible.

4. **`torch.cuda.empty_cache()` is the cleanup step** — without it, PyTorch holds the freed VRAM in its allocator pool and Llama can't use it.

5. **Embeddings are always on CPU** — the `[47071 × 768]` matrix lives on CPU from the start (loaded with `map_location="cpu"`). The cosine similarity dot product runs on CPU regardless. Only the BERT model itself was unnecessarily on GPU.

6. **This pattern generalizes** — whenever you have a "build once, query many" pipeline on a memory-constrained GPU, use GPU for the build phase, then move the model to CPU for serving. Reserve GPU for the model that actually needs it at inference time (in our case, Llama).
