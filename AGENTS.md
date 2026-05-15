# AGENTS.md — Music CRS Baselines

> Context guide for AI assistants (Wibey, Claude Code, Copilot, etc.) working in this repository.

---

## Project Overview

**music-crs-baselines** is the official baseline and evaluation framework for the **RecSys Challenge 2026: Conversational Music Recommendation System (Music-CRS)**.

The challenge tackles an evolving paradigm in music discovery: replacing static recommendation lists with dynamic, multi-turn conversational interactions. Systems must integrate Natural Language Understanding (NLU) with high-precision Recommender Systems (RecSys) to understand nuanced user preferences and provide contextually relevant track recommendations through dialogue.

- **Challenge Website**: https://nlp4musa.github.io/music-crs-challenge/
- **Dataset Hub**: https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge
- **Evaluator Repo**: https://github.com/nlp4musa/music-crs-evaluator

### Challenge Timeline (2026)
| Date | Milestone |
|------|-----------|
| 10 Apr | Dataset release (Train, Dev, Blind A) |
| 15 Apr | Submission system open / Leaderboard live |
| 15 Jun | Blind B released |
| 30 Jun | End of challenge |
| 9 Jul | Upload final prediction code |

---

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | ≥ 3.8 (`.venv` runs 3.12) |
| Package manager | `uv` | latest |
| Deep learning | PyTorch | ≥ 2.5.1 |
| LLM inference | Hugging Face Transformers | ≥ 4.46.3 |
| Sparse retrieval | bm25s | ≥ 0.2.13 |
| Dataset access | Hugging Face Datasets | ≥ 3.1.0 |
| Config management | OmegaConf | ≥ 2.3.0 |
| System monitoring | psutil | ≥ 7.2.2 |
| Fast attention | flash-attn | optional, recommended for GPU |

### Default Models
- **LLM**: `meta-llama/Llama-3.2-1B-Instruct` (swappable to any HF causal LM, e.g. `Qwen/Qwen3-4B`)
- **Dense retriever**: `bert-base-uncased` (BERT module)
- **Sparse retriever**: BM25 via `bm25s`

---

## Repository Structure

```
music-crs-baselines-main/
│
├── mcrs/                          # Core installable Python package
│   ├── __init__.py                # Exports load_crs_baseline()
│   ├── crs_baseline.py            # CRS_BASELINE: main orchestration class
│   │
│   ├── retrieval_modules/         # Stage 1 — track retrieval
│   │   ├── __init__.py            # load_retrieval_module() factory
│   │   ├── bm25.py                # BM25_MODEL — sparse TF-IDF retrieval
│   │   └── bert.py                # BERT_MODEL — dense cosine-sim retrieval
│   │
│   ├── lm_modules/                # Stage 2 — response generation
│   │   ├── __init__.py            # load_lm_module() factory
│   │   └── llama.py               # LLAMA_MODEL — causal LM wrapper
│   │
│   ├── db_item/                   # Track/item database
│   │   ├── __init__.py
│   │   └── music_catalog.py       # MusicCatalogDB: track_id → metadata
│   │
│   ├── db_user/                   # User profile database
│   │   ├── __init__.py
│   │   └── user_profile.py        # UserProfileDB: user_id → profile string
│   │
│   └── system_prompts/            # Plain-text LLM prompt templates
│       ├── roleplay.txt           # Persona / base role prompt
│       ├── personalization.txt    # User-profile injection segment
│       └── response_generation.txt# Response formatting instructions
│
├── config/                        # OmegaConf YAML experiment configs
│   ├── llama1b_bm25_devset.yaml
│   ├── llama1b_bert_devset.yaml
│   ├── llama1b_bm25_blindset_A.yaml
│   └── llama1b_bert_blindset_A.yaml
│
├── lowerbound/                    # Non-neural baseline scripts
│   ├── random_sample.py           # Random track recommendations
│   └── popularity.py              # Most-popular tracks from training set
│
├── tips/                          # Improvement directions (read-only guides)
│   ├── improve_item_representation.md
│   ├── add_reranker.md
│   └── use_genrec_semantic_ids.md
│
├── run_inference_devset.py        # Batch inference on the dev/test split
├── run_inference_blindset.py      # Batch inference for blind submission sets
├── pyproject.toml                 # Package metadata + dependencies
└── .venv/                         # Local virtual environment (Python 3.12)
```

---

## Architecture: Two-Stage Pipeline

```
User Query + Chat History
         │
         ▼
┌─────────────────────┐
│   Stage 1: RecSys   │  text_to_item_retrieval(query, topk=20)
│   (BM25 or BERT)    │──────────────────────────────────────────► Top-20 track IDs
└─────────────────────┘
         │  retrieval_items[0]  →  item metadata string
         ▼
┌─────────────────────┐
│   Stage 2: LLM      │  response_generation(system_prompt, history, recommend_item)
│   (Llama / Qwen)    │──────────────────────────────────────────► Natural language response
└─────────────────────┘
         │
         ▼
Output: { user_id, user_query, retrieval_items, recommend_item, response }
```

### Key Data Flow Details
- **System prompt** = `roleplay.txt` + `response_generation.txt` + (optionally) `personalization.txt` + user profile string
- **Retrieval input** = full conversation history joined as `"role: content\n..."` strings
- **LLM input** = system prompt + chat history + top-1 recommended item metadata (formatted as `track_id: ..., track_name: ..., artist_name: ...`)
- **Output JSON** (submitted for evaluation):
  ```json
  {
    "session_id": "...",
    "user_id": "...",
    "turn_number": 3,
    "predicted_track_ids": ["id1", "id2", ...],   // top-20 track IDs
    "predicted_response": "Here is a great song ..."
  }
  ```

---

## Core Classes & APIs

### `CRS_BASELINE` (`mcrs/crs_baseline.py`)
The main orchestration class. Instantiated via `load_crs_baseline()`.

```python
from mcrs import load_crs_baseline

crs = load_crs_baseline(
    lm_type="meta-llama/Llama-3.2-1B-Instruct",
    retrieval_type="bm25",            # "bm25" | "bert"
    item_db_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    user_db_name="talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    track_split_types=["all_tracks"], # MUST always be ["all_tracks"]
    user_split_types=["all_users"],
    corpus_types=["track_name", "artist_name", "album_name"],
    cache_dir="./cache",
    device="cuda",
    attn_implementation="flash_attention_2",
)
```

Key methods:
- `crs.chat(user_query, user_id=None)` — single-turn inference
- `crs.batch_chat(batch_data)` — batch inference (preferred for experiments)
- `crs._get_system_prompt(user_id)` — builds the full system prompt
- `crs._reset_session_memory()` — clears in-memory session history

### `BM25_MODEL` (`mcrs/retrieval_modules/bm25.py`)
- Builds a BM25 index over concatenated metadata fields on first run, caches to `./cache/bm25/<corpus_name>/`
- `text_to_item_retrieval(query, topk)` and `batch_text_to_item_retrieval(queries, topk)`

### `BERT_MODEL` (`mcrs/retrieval_modules/bert.py`)
- Encodes all tracks with mean-pooled BERT embeddings; uses cosine similarity for retrieval
- Caches embeddings to `./cache/bert/<corpus_name>/embeddings.pt`
- `text_to_item_retrieval(query, topk)` and `batch_text_to_item_retrieval(queries, topk)`

### `LLAMA_MODEL` (`mcrs/lm_modules/llama.py`)
- Wraps any HF causal LM (Llama, Qwen, etc.) using `AutoModelForCausalLM`
- Formats messages using the tokenizer's built-in chat template
- `response_generation(sys_prompt, chat_history, recommend_item)` — single sample
- `batch_response_generation(sys_prompts, chat_histories, recommend_items)` — batched

### `MusicCatalogDB` (`mcrs/db_item/music_catalog.py`)
- `id_to_metadata(track_id)` — returns a formatted string like `track_id: X, track_name: ..., artist_name: ...`

### `UserProfileDB` (`mcrs/db_user/user_profile.py`)
- `id_to_profile_str(user_id)` — returns `"user_id: ...\nage_group: ...\ngender: ...\ncountry_name: ..."`

---

## Configuration System

All experiments are driven by YAML files in `config/`. Config is loaded with OmegaConf:

```yaml
lm_type: "meta-llama/Llama-3.2-1B-Instruct"
retrieval_type: "bm25"                   # "bm25" or "bert"
test_dataset_name: "talkpl-ai/TalkPlayData-Challenge-Dataset"
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
track_split_types:
  - "all_tracks"                         # CRITICAL — never change this
user_split_types:
  - "all_users"
corpus_types:
  - "track_name"
  - "artist_name"
  - "album_name"
  - "release_date"
cache_dir: "./cache"
device: "cuda"
attn_implementation: "flash_attention_2" # "eager" if flash-attn not installed
```

The `--tid` argument matches a filename in `config/` (e.g., `--tid llama1b_bm25_devset` loads `config/llama1b_bm25_devset.yaml`).

---

## Development Commands

### Installation
```bash
# Create and activate virtual environment (Python 3.10 recommended)
uv venv .venv --python=3.10
source .venv/bin/activate

# Install the mcrs package in editable mode
uv pip install -e .

# Optional: fast attention (requires CUDA + compatible GPU)
uv pip install flash-attn --no-build-isolation
```

### Run Inference — Development Set
```bash
# BM25 retrieval baseline
python run_inference_devset.py --tid llama1b_bm25_devset --batch_size 16

# BERT retrieval baseline
python run_inference_devset.py --tid llama1b_bert_devset --batch_size 16
```
Output saved to: `exp/inference/devset/<tid>.json`

### Run Inference — Blind Sets (for submission)
```bash
# Blind Set A — BM25
python run_inference_blindset.py --tid llama1b_bm25_blindset_A --batch_size 16

# Blind Set A — BERT
python run_inference_blindset.py --tid llama1b_bert_blindset_A --batch_size 16
```
Output saved to: `exp/inference/<eval_dataset>/<tid>.json`

### Run Lower-Bound Baselines
```bash
# Random recommendation (lower bound floor)
python lowerbound/random_sample.py

# Popularity-based recommendation
python lowerbound/popularity.py
```
Output saved to: `exp/inference/random.json` and `exp/inference/popularity.json`

### Custom Experiment
```bash
# 1. Create config/my_experiment.yaml with desired settings
# 2. Run inference
python run_inference_devset.py --tid my_experiment --batch_size 8
```

### GPU Memory Tips
- Reduce `--batch_size` (try 4 or 8) if you hit CUDA out-of-memory errors
- Switch `attn_implementation: "eager"` if `flash_attention_2` is unavailable
- The scripts automatically `rm -rf cache` on startup to prevent stale memory issues

---

## Critical Rules & Constraints

> **Breaking these rules will invalidate your submission.**

1. **Always use `all_tracks`**: `track_split_types` MUST be `["all_tracks"]`. Never filter, subset, or restrict the track catalog during inference. This is enforced by the evaluation server.

2. **Output format is strict**: The JSON output must contain exactly these keys per record:
   - `session_id` (string)
   - `user_id` (string)
   - `turn_number` (int, 1–8)
   - `predicted_track_ids` (list of strings, typically 20 track IDs)
   - `predicted_response` (string, can be empty for retrieval-only submissions)

3. **8 turns per session**: Every session in the dataset has exactly 8 conversation turns. Inference scripts iterate `range(1, 9)` over each session.

4. **Cache invalidation**: The `run_inference_*.py` scripts delete `./cache` at startup. If you modify the retrieval index logic, manually delete `./cache` to force a rebuild.

5. **No secrets in code**: Never hardcode HuggingFace tokens, API keys, or credentials. Use environment variables (`HF_TOKEN`, etc.) or `huggingface-cli login`.

---

## Extension Points (Tips)

The `tips/` directory documents three main improvement directions:

### 1. Improve Item Representation (`tips/improve_item_representation.md`)
- Add fields like `tag_list`, mood labels, or popularity scores to `corpus_types`
- Integrate audio features using CLAP (text+audio understanding)
- Swap `bert-base-uncased` for stronger encoders: `Qwen2.5-Embedding`, `E5`, `BGE`, `ColBERT`

### 2. Add a Reranker (`tips/add_reranker.md`)
- **Embedding-based**: Compute user embeddings from listening history, score candidates by user-item similarity
- **LLM-based**: Prompt an LLM to rank top-k candidates by relevance to the user query
- Hook point: after `retrieval.text_to_item_retrieval(query, topk=100)`, before selecting top-20

### 3. Generative Retrieval (`tips/use_genrec_semantic_ids.md`)
- Assign hierarchical semantic IDs (e.g., `jazz/smooth/piano/0042`) to tracks
- Fine-tune the LLM to directly generate track IDs from user queries
- Replaces the two-stage retrieve-then-generate architecture with a single unified model

---

## Adding a New Retrieval Module

1. Create `mcrs/retrieval_modules/my_retriever.py` implementing:
   - `text_to_item_retrieval(query: str, topk: int) -> list[str]`
   - `batch_text_to_item_retrieval(queries: list[str], topk: int) -> list[list[str]]` (optional but recommended)
2. Register it in `mcrs/retrieval_modules/__init__.py` `load_retrieval_module()` factory
3. Set `retrieval_type: "my_retriever"` in your config YAML
4. Run inference as normal — the factory handles the rest

## Adding a New LM Module

1. Create `mcrs/lm_modules/my_lm.py` with `response_generation()` and optionally `batch_response_generation()`
2. Register in `mcrs/lm_modules/__init__.py` `load_lm_module()` factory
3. Set `lm_type: "my_lm"` in your config YAML (or map to it in the factory)

---

## Datasets (Hugging Face)

All datasets are loaded at runtime via the `datasets` library. Ensure `HF_TOKEN` is set if datasets require authentication.

| Dataset | HF ID | Purpose |
|---------|-------|---------|
| Conversations | `talkpl-ai/TalkPlayData-Challenge-Dataset` | Train / test dialogues |
| Track Metadata | `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` | Item DB (split: `all_tracks`) |
| User Profiles | `talkpl-ai/TalkPlayData-Challenge-User-Metadata` | User DB (split: `all_users`) |
| Blind A | `talkpl-ai/TalkPlayData-Challenge-Blind-A` | Submission set A |

Track metadata fields available for `corpus_types`:
- `track_id`, `track_name`, `artist_name`, `album_name`, `release_date`, `tag_list`

User profile fields: `user_id`, `age_group`, `gender`, `country_name`

---

## Code Style & Conventions

- **Formatting**: Follow existing code style — no linter config is present; match surrounding indentation and conventions
- **Type hints**: Use them consistently; existing code uses Python 3.10+ union syntax (`str | None`) in some places but also `Optional` — be consistent with the file you're editing
- **Docstrings**: Google-style docstrings are used throughout (Args/Returns sections)
- **No tests**: There is no test suite. Validate changes by running a small batch inference (`--batch_size 2`) and inspecting the output JSON
- **No CI/CD**: This is a research/challenge repository — no pipelines to worry about
- **Dependencies**: Add new dependencies to `pyproject.toml` under `[project] dependencies`; install with `uv pip install -e .`

---

## Documentation Rules

- **"Document" always means**: create a `.md` file inside the `docs/` folder
- **Filename convention**: prepend today's date → `docs/YYYY-MM-DD-<descriptive-slug>.md`
- **Auto-document on major changes**: whenever a significant feature, script, or architecture change is made, always create a corresponding doc in `docs/` with the date prefix. Do not wait to be asked.
- **Existing docs**: update the relevant existing doc if the change modifies something already documented; create a new doc if it's a new topic
- **Examples**:
  - `docs/2026-05-09-how-the-baseline-works.html` — architecture overview
  - `docs/2026-05-14-inference-runbook.md` — running inference locally
  - `docs/2026-05-14-kaggle-gpu-setup.md` — Kaggle GPU setup
  - `docs/2026-05-15-execution-guide.md` — local + remote execution
  - `docs/2026-05-15-prepare-submission.md` — submission packaging

---

## Common Pitfalls

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `KeyError: 'all_tracks'` | Wrong split name in config | Use exactly `"all_tracks"` |
| CUDA OOM during inference | Batch size too large | Reduce `--batch_size` to 4 or 8 |
| Stale BM25/BERT index | Changed `corpus_types` but cache exists | `rm -rf cache` |
| `flash_attention_2` import error | flash-attn not installed | Set `attn_implementation: "eager"` in config |
| Missing track in `id_to_metadata` | Track not in `all_tracks` split | Ensure `track_split_types: ["all_tracks"]` |
| Wrong output path | `eval_dataset` arg not matching directory | Check `--eval_dataset` on blindset script |
