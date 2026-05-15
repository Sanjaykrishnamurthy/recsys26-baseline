# Inference Runbook — Music CRS Baselines

> **Last updated:** 2026-05-14  
> Step-by-step guide for running all supported inference configurations and packaging a competition submission.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Setup](#2-data-setup)
3. [Configuration Reference](#3-configuration-reference)
4. [Running Inference — Dev Set](#4-running-inference--dev-set)
5. [Running Inference — Blind Set A](#5-running-inference--blind-set-a)
6. [Running with Ollama (Local LLM)](#6-running-with-ollama-local-llm)
7. [Evaluating Dev Set Output](#7-evaluating-dev-set-output)
8. [Packaging the Submission](#8-packaging-the-submission)
9. [Uploading the Submission](#9-uploading-the-submission)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Environment Setup

```bash
# Create virtual environment (Python 3.10+)
uv venv .venv --python=3.10
source .venv/bin/activate

# Install the mcrs package in editable mode
uv pip install -e .
```

> **Apple Silicon (M-series Mac):** the default `device: "mps"` in all `local_*` configs uses the GPU cores built into the chip. No CUDA needed.  
> **NVIDIA GPU:** change `device: "cuda"` in the config and optionally install `flash-attn` for faster inference.

---

## 2. Data Setup

Download the four HuggingFace datasets and place them under `data/raw/`:

```
data/
└── raw/
    ├── TalkPlayData-Challenge-Dataset/          ← train/dev conversations
    │   └── data/
    │       ├── test-00000-of-00001.parquet
    │       └── train-00000-of-00001.parquet
    ├── TalkPlayData-Challenge-Track-Metadata/   ← item database
    │   └── data/
    │       ├── all_tracks-00000-of-00001.parquet
    │       └── test_tracks-00000-of-00001.parquet
    ├── TalkPlayData-Challenge-User-Metadata/    ← user profiles
    │   └── data/
    │       └── all_users-00000-of-00001.parquet
    └── TalkPlayData-Challenge-Blind-A/          ← blind submission set A
        └── data/
            └── test-00000-of-00001.parquet
```

All `local_*` configs point to `./data/raw/...` so inference runs fully offline — no network needed after this step.

> **Note:** `data/` is git-ignored. The parquet files stay local only.

---

## 3. Configuration Reference

All configs live in `config/`. Pick the one that matches your setup:

| Config file | Retriever | LLM | Dataset | Notes |
|---|---|---|---|---|
| `local_bm25_devset.yaml` | BM25 | none (retrieval-only) | Dev set | **Recommended first run** |
| `local_bert_devset.yaml` | BERT | none | Dev set | Slower build, better recall |
| `local_bm25_blindset_A.yaml` | BM25 | none | Blind A | For submission |
| `local_bert_blindset_A.yaml` | BERT | none | Blind A | For submission |
| `ollama_bm25_devset.yaml` | BM25 | Ollama `llama3.2:1b` | Dev set | Needs Ollama running locally |
| `qwen4b_bert_devset.yaml` | BERT | Qwen3-4B (HF) | Dev set | Needs HuggingFace access |
| `bm25_retrieval_only_devset.yaml` | BM25 | none | Dev set (HF) | Requires network |
| `bert_retrieval_only_devset.yaml` | BERT | none | Dev set (HF) | Requires network |

**Key config fields:**

```yaml
lm_type: "none"          # "none" = retrieval-only | "ollama:<model>" | HF model ID
retrieval_type: "bm25"   # "bm25" or "bert"
device: "mps"            # "mps" (Apple Silicon) | "cuda" (NVIDIA) | "cpu"
use_user_reranker: false # true = BERT reranks top-100 using user profile embedding
                         # only effective when retrieval_type is "bert"
```

> **Critical:** Never change `track_split_types` from `["all_tracks"]` — the evaluator enforces this.

---

## 4. Running Inference — Dev Set

### 4a. BM25 Retrieval-Only (fastest, recommended start)

```bash
python run_inference_devset.py \
  --tid local_bm25_devset \
  --batch_size 16
```

Output → `exp/inference/devset/local_bm25_devset.json`

**What happens:**
1. Deletes `cache/` to avoid stale indexes
2. Builds BM25 index over `track_name + artist_name + album_name + release_date + tag_list`
3. For every session × 8 turns, retrieves top-20 tracks using the conversation history as query
4. Saves one JSON record per (session, turn)

BM25 index is rebuilt on first run (~30 sec) and cached at `cache/bm25/`.

---

### 4b. BERT Dense Retrieval

```bash
python run_inference_devset.py \
  --tid local_bert_devset \
  --batch_size 16
```

Output → `exp/inference/devset/local_bert_devset.json`

**What happens:**
1. Encodes all tracks with mean-pooled `bert-base-uncased` embeddings on first run
2. Embeddings cached at `cache/bert/`; subsequent runs skip encoding
3. Retrieves top-20 tracks via cosine similarity

First run is slower (~2–5 min on MPS to encode the full track catalog). Subsequent runs use the cached `.pt` file.

---

### 4c. BERT + User Reranker

Edit `config/local_bert_devset.yaml` and set `use_user_reranker: true`, then run the same command. The reranker:
- Retrieves top-100 candidates via BERT
- Encodes the user profile string with the same BERT encoder
- Reranks the 100 candidates by user-profile similarity
- Returns the top-20 reranked list

---

### 4d. With Ollama LLM

First, make sure Ollama is running:

```bash
# Install Ollama from https://ollama.com (run once)
ollama pull llama3.2:1b    # ~1.3 GB download

# Start the server (leave running in a separate terminal)
ollama serve
```

Then run inference:

```bash
python run_inference_devset.py \
  --tid ollama_bm25_devset \
  --batch_size 4
```

> Use `--batch_size 4` for Ollama — it processes requests sequentially so larger batches don't help and can cause timeouts.

Each record now has a non-empty `predicted_response` generated by the LLM.

---

## 5. Running Inference — Blind Set A

### 5a. BM25 (retrieval-only)

```bash
python run_inference_blindset.py \
  --tid local_bm25_blindset_A \
  --eval_dataset blindset_A \
  --batch_size 16
```

Output → `exp/inference/blindset_A/local_bm25_blindset_A.json`

### 5b. BERT (retrieval-only)

```bash
python run_inference_blindset.py \
  --tid local_bert_blindset_A \
  --eval_dataset blindset_A \
  --batch_size 16
```

Output → `exp/inference/blindset_A/local_bert_blindset_A.json`

**Blind set structure:** each session appears once at the specific conversation turn it was captured. The dataset has 80 sessions across turns 1–8 — this is expected and correct. You will see a one-prediction-per-session summary.

---

## 6. Running with Ollama (Local LLM)

Ollama lets you run LLMs locally without needing HuggingFace access — useful on corporate networks where `huggingface.co` is blocked.

### Setup

```bash
# macOS — download from https://ollama.com/download
# Or via Homebrew:
brew install ollama

# Pull a model (run once; downloads to ~/.ollama/models/)
ollama pull llama3.2:1b     # ~1.3 GB — fast, lightweight
ollama pull llama3.2:3b     # ~2.0 GB — better quality

# Start the server
ollama serve
# Server runs at http://localhost:11434
```

### Config

The `ollama_bm25_devset.yaml` uses `lm_type: "ollama:llama3.2:1b"`. To use a different model, create a new config and change that field:

```yaml
lm_type: "ollama:llama3.2:3b"    # or any model pulled via `ollama pull`
```

### Run

```bash
python run_inference_devset.py \
  --tid ollama_bm25_devset \
  --batch_size 4
```

---

## 7. Evaluating Dev Set Output

Use the official evaluator repo to score dev set predictions:

```bash
# Clone the evaluator (run once, outside this repo)
git clone https://github.com/nlp4musa/music-crs-evaluator

# Follow its README to score your output JSON
# Point it at: exp/inference/devset/<tid>.json
```

Key metrics: **Recall@5**, **Recall@10**, **Recall@20** for retrieval; **BLEU / ROUGE** for response quality.

---

## 8. Packaging the Submission

After running blind set inference, package the output:

```bash
# Dry-run first — validate without writing files
python prepare_submission.py \
  --input exp/inference/blindset_A/local_bm25_blindset_A.json \
  --dry-run

# If validation passes, create the zip
python prepare_submission.py \
  --input exp/inference/blindset_A/local_bm25_blindset_A.json \
  --output submissions/submission.zip
```

The packager enforces all competition rules automatically:
- Deduplicates `predicted_track_ids` (preserving ranking order)
- Truncates to 20 track IDs
- Validates required fields (`session_id`, `user_id`, `turn_number`, `predicted_track_ids`, `predicted_response`)
- Writes `predictions.json` inside `submission.zip`

**Expected output (BM25 blind set A, retrieval-only):**
```
✅  All records passed validation — no issues found.
ℹ️  Blind-set mode: each session has one prediction (single-turn evaluation).

Summary:
   Total records  : 80
   Unique sessions: 80
   Turns covered  : [1, 2, 3, 4, 5, 6, 7, 8]
```

---

## 9. Uploading the Submission

> ⚠️ **Corporate network note:** Walmart's SSL inspection proxy blocks uploads to external competition servers. You will get `"Server returned: 0 error"` on the Walmart Wi-Fi / VPN.

**Fix:** Disconnect from Walmart VPN and use a personal network (phone hotspot or home Wi-Fi) to upload.

**Steps:**
1. Go to the competition leaderboard: https://nlp4musa.github.io/music-crs-challenge/
2. Log in to your registered account
3. Upload `submissions/submission.zip`
4. Wait for the leaderboard to update (typically a few minutes)

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'all_tracks'` | Wrong split name | Ensure `track_split_types: ["all_tracks"]` in config |
| CUDA / MPS out of memory | Batch too large | Reduce `--batch_size` to `4` or `8` |
| Stale BM25 / BERT index | Changed `corpus_types` | `rm -rf cache/` |
| `flash_attention_2` error | Not installed | Set `attn_implementation: "eager"` |
| `[Errno 61] Connection refused` | Ollama not running | Run `ollama serve` in a separate terminal |
| `Server returned: 0 error` on upload | Corporate network blocking | Disconnect from Walmart VPN, use personal hotspot |
| HuggingFace SSL error | Corporate proxy intercepts HTTPS | Use local data configs (`local_*`) or Ollama |
| `SSH: Operation timed out` on git push | Port 22 blocked | Use HTTPS remote: `git remote set-url origin https://github.com/...` |
| Missing track in `id_to_metadata` | Track not in `all_tracks` split | Check `track_split_types: ["all_tracks"]` |

---

## Quick Reference — Commands

```bash
# ── Dev set ───────────────────────────────────────────────────────
# BM25 retrieval-only (local data, fastest)
python run_inference_devset.py --tid local_bm25_devset --batch_size 16

# BERT retrieval-only (local data)
python run_inference_devset.py --tid local_bert_devset --batch_size 16

# BM25 + Ollama LLM (requires: ollama serve)
python run_inference_devset.py --tid ollama_bm25_devset --batch_size 4

# ── Blind Set A ───────────────────────────────────────────────────
# BM25 retrieval-only
python run_inference_blindset.py --tid local_bm25_blindset_A --eval_dataset blindset_A --batch_size 16

# BERT retrieval-only
python run_inference_blindset.py --tid local_bert_blindset_A --eval_dataset blindset_A --batch_size 16

# ── Submission ────────────────────────────────────────────────────
# Validate
python prepare_submission.py --input exp/inference/blindset_A/local_bm25_blindset_A.json --dry-run

# Package
python prepare_submission.py --input exp/inference/blindset_A/local_bm25_blindset_A.json --output submissions/submission.zip
```
