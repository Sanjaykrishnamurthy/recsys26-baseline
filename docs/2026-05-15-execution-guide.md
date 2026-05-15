# Execution Guide — Local & Remote (Kaggle)

> **Last updated:** 2026-05-15  
> Complete procedures for running Music CRS baselines on your local machine (Mac / Linux) and on Kaggle's free GPU.

---

## Table of Contents

- [Local Execution](#local-execution)
  - [Prerequisites](#prerequisites)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Setup](#2-data-setup)
  - [3. Configure Environment Variables](#3-configure-environment-variables)
  - [4. Run Lower-Bound Baselines](#4-run-lower-bound-baselines)
  - [5. Run BM25 Retrieval-Only](#5-run-bm25-retrieval-only)
  - [6. Run BERT Retrieval-Only](#6-run-bert-retrieval-only)
  - [7. Run with Ollama LLM](#7-run-with-ollama-llm)
  - [8. Package Submission](#8-package-submission)
- [Remote Execution (Kaggle)](#remote-execution-kaggle)
  - [Why Kaggle?](#why-kaggle)
  - [1. Create a Kaggle Notebook](#1-create-a-kaggle-notebook)
  - [2. Attach the Llama Model](#2-attach-the-llama-model)
  - [3. Clone and Install](#3-clone-and-install)
  - [4. Run BM25 + Llama Inference](#4-run-bm25--llama-inference)
  - [5. Run BERT + Llama Inference](#5-run-bert--llama-inference)
  - [6. Package and Download](#6-package-and-download)
- [Upload Submission](#upload-submission)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

# Local Execution

Best for: retrieval-only experiments (BM25, BERT) and Ollama-based LLM runs.  
Works fully offline once data is downloaded.

## Prerequisites

| Requirement | Version | Check |
|---|---|---|
| Python | ≥ 3.10 | `python --version` |
| uv (package manager) | latest | `uv --version` |
| Git | any | `git --version` |
| Disk space | ~2 GB | For data + cache |

## 1. Environment Setup

```bash
# Clone the repo
git clone https://github.com/Sanjaykrishnamurthy/recsys26-baseline.git
cd recsys26-baseline

# Create virtual environment
uv venv .venv --python=3.10
source .venv/bin/activate

# Install the mcrs package (editable mode)
uv pip install -e .
```

Verify the install:

```bash
python -c "import mcrs; print('mcrs', mcrs.__version__)"
# Expected: mcrs 0.2.0
```

## 2. Data Setup

Download the four HuggingFace datasets and place them under `data/raw/`:

```
data/
└── raw/
    ├── TalkPlayData-Challenge-Dataset/
    │   └── data/
    │       ├── test-00000-of-00001.parquet
    │       └── train-00000-of-00001.parquet
    ├── TalkPlayData-Challenge-Track-Metadata/
    │   └── data/
    │       ├── all_tracks-00000-of-00001.parquet
    │       └── test_tracks-00000-of-00001.parquet
    ├── TalkPlayData-Challenge-User-Metadata/
    │   └── data/
    │       └── all_users-00000-of-00001.parquet
    └── TalkPlayData-Challenge-Blind-A/
        └── data/
            └── test-00000-of-00001.parquet
```

> `data/` is git-ignored — it stays local only.

## 3. Configure Environment Variables

The `.env` file sets local data paths for the lowerbound scripts:

```bash
# .env (already created at project root, git-ignored)
MCRS_TEST_DB=./data/raw/TalkPlayData-Challenge-Dataset
MCRS_TRACK_DB=./data/raw/TalkPlayData-Challenge-Track-Metadata
MCRS_USER_DB=./data/raw/TalkPlayData-Challenge-User-Metadata
MCRS_BLIND_A_DB=./data/raw/TalkPlayData-Challenge-Blind-A
```

Load it before running:

```bash
source .env
```

> The `local_*` YAML configs already point to `./data/raw/...` so the main inference scripts don't need env vars.

## 4. Run Lower-Bound Baselines

These generate non-neural baseline predictions for comparison scoring.

### Random Baseline

Picks 20 random tracks per turn. This is the absolute floor.

```bash
source .env
python lowerbound/random_sample.py
```

Output → `exp/inference/random.json`

### Popularity Baseline

Returns the 20 most frequently recommended tracks from the training set.

```bash
source .env
python lowerbound/popularity.py
```

Output → `exp/inference/popularity.json`

## 5. Run BM25 Retrieval-Only

BM25 is a sparse keyword-matching retriever. Fast to build (~30 sec), no GPU needed.

### Dev Set (for local scoring)

```bash
python run_inference_devset.py \
  --tid local_bm25_devset \
  --batch_size 16
```

Output → `exp/inference/devset/local_bm25_devset.json`

**What happens:**
1. Deletes `cache/` to prevent stale indexes
2. Builds a BM25 index over `track_name + artist_name + album_name + release_date + tag_list`
3. For each session × 8 turns, retrieves top-20 tracks
4. `predicted_response` is `""` (retrieval-only mode)

### Blind Set A (for submission)

```bash
python run_inference_blindset.py \
  --tid local_bm25_blindset_A \
  --eval_dataset blindset_A \
  --batch_size 16
```

Output → `exp/inference/blindset_A/local_bm25_blindset_A.json`

> The blind set has 80 sessions, each at a different conversation stage. One prediction per session — this is expected.

## 6. Run BERT Retrieval-Only

BERT is a dense cosine-similarity retriever. Better recall than BM25, but slower first run.

### Dev Set

```bash
python run_inference_devset.py \
  --tid local_bert_devset \
  --batch_size 16
```

Output → `exp/inference/devset/local_bert_devset.json`

**First run:** encodes the full track catalog with `bert-base-uncased` (~2–5 min on Apple MPS). Embeddings are cached at `cache/bert/` — subsequent runs skip encoding.

### Blind Set A

```bash
python run_inference_blindset.py \
  --tid local_bert_blindset_A \
  --eval_dataset blindset_A \
  --batch_size 16
```

Output → `exp/inference/blindset_A/local_bert_blindset_A.json`

### BERT + User Reranker

To rerank BM25/BERT candidates by user-profile similarity, edit the config and set `use_user_reranker: true`. This retrieves top-100 candidates, then reranks to top-20 using the user's profile embedding.

> Only works with `retrieval_type: "bert"`.

## 7. Run with Ollama LLM

Ollama runs LLMs locally. Useful when HuggingFace is blocked (corporate network).

### Setup Ollama

```bash
# Install (macOS)
brew install ollama

# Pull a model (one-time download)
ollama pull llama3.2:1b     # ~1.3 GB
ollama pull llama3.2:3b     # ~2.0 GB (better quality)

# Start the server (keep running in a separate terminal)
ollama serve
```

### Run Inference

```bash
python run_inference_devset.py \
  --tid ollama_bm25_devset \
  --batch_size 4
```

> Use `--batch_size 4` — Ollama processes sequentially, larger batches just queue.

Each record now has a real `predicted_response` generated by the LLM instead of an empty string.

## 8. Package Submission

```bash
# Validate first (no files written)
python prepare_submission.py \
  --input exp/inference/blindset_A/local_bm25_blindset_A.json \
  --dry-run

# Create the zip
python prepare_submission.py \
  --input exp/inference/blindset_A/local_bm25_blindset_A.json \
  --output submissions/submission.zip
```

Expected output:

```
✅  All records passed validation — no issues found.
ℹ️  Blind-set mode: each session has one prediction (single-turn evaluation).

Summary:
   Total records  : 80
   Unique sessions: 80
   Turns covered  : [1, 2, 3, 4, 5, 6, 7, 8]

🎉  submission.zip is ready to upload!
```

---

# Remote Execution (Kaggle)

Best for: full pipeline with Llama 3.2 1B LLM on a free GPU.  
Generates real conversational responses (not empty strings).

## Why Kaggle?

| | Local (Mac) | Kaggle (free GPU) |
|---|---|---|
| GPU | Apple MPS (no CUDA) | NVIDIA T4 / P100 |
| Llama 3.2 1B | Blocked by Walmart VPN | Pre-loaded via Kaggle Models |
| HuggingFace datasets | Blocked by Walmart VPN | Direct access |
| Cost | Free | Free (30 hrs/week GPU quota) |

## 1. Create a Kaggle Notebook

1. Go to [kaggle.com](https://kaggle.com) → **Create** → **New Notebook**
2. Settings (right panel):
   - **Accelerator:** GPU T4 x2
   - **Internet:** On

## 2. Attach the Llama Model

In the notebook sidebar → **+ Add Input** → **New Model**:

1. Search: `meta-llama/llama-3.2`
2. Select: **Llama 3.2** → **Transformers** → **1B Instruct** → **Version 1**
3. Click **Add**

The model mounts at:
```
/kaggle/input/llama-3.2/transformers/1b-instruct/1/
```

No download wait, no HuggingFace token, no SSL issues.

## 3. Clone and Install

```python
# Cell 1 — Setup
!git clone https://github.com/Sanjaykrishnamurthy/recsys26-baseline.git
%cd recsys26-baseline
!pip install bm25s omegaconf psutil -e . -q
```

**If datasets require HuggingFace authentication** (add this before inference):

```python
# Cell 2 — HF auth (only if datasets are private)
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

> Set up the `HF_TOKEN` secret at: Kaggle → Account → Settings → Secrets

## 4. Run BM25 + Llama Inference

### Dev Set (check quality first)

```python
# Cell 3 — Dev set inference
!python run_inference_devset.py \
    --tid kaggle_llama1b_bm25_devset \
    --batch_size 16
```

Output → `exp/inference/devset/kaggle_llama1b_bm25_devset.json`

### Blind Set A (for submission)

```python
# Cell 4 — Blind set inference
!python run_inference_blindset.py \
    --tid kaggle_llama1b_bm25_blindset_A \
    --eval_dataset blindset_A \
    --batch_size 16
```

Output → `exp/inference/blindset_A/kaggle_llama1b_bm25_blindset_A.json`

## 5. Run BERT + Llama Inference

```python
# Cell 5 — BERT retrieval + Llama response
!python run_inference_blindset.py \
    --tid kaggle_llama1b_bert_blindset_A \
    --eval_dataset blindset_A \
    --batch_size 16
```

Output → `exp/inference/blindset_A/kaggle_llama1b_bert_blindset_A.json`

> First run encodes the full track catalog (~2–3 min on T4). Cached after that.

## 6. Package and Download

```python
# Cell 6 — Package
!python prepare_submission.py \
    --input exp/inference/blindset_A/kaggle_llama1b_bm25_blindset_A.json \
    --output submissions/submission.zip
```

Download `submissions/submission.zip` from the Kaggle file browser (left panel → right-click → Download).

---

# Upload Submission

> **Corporate network blocks the upload endpoint.** Disconnect from Walmart VPN and use a personal network (phone hotspot or home Wi-Fi).

1. Go to the competition leaderboard: https://nlp4musa.github.io/music-crs-challenge/
2. Log in to your registered account
3. Upload `submission.zip`
4. Wait for the leaderboard to update (typically a few minutes)

---

# Configuration Reference

## Local Configs (offline, no network)

| Config | Retriever | LLM | Dataset | Device |
|---|---|---|---|---|
| `local_bm25_devset` | BM25 | none | Dev set | mps |
| `local_bert_devset` | BERT | none | Dev set | mps |
| `local_bm25_blindset_A` | BM25 | none | Blind A | mps |
| `local_bert_blindset_A` | BERT | none | Blind A | mps |
| `ollama_bm25_devset` | BM25 | Ollama llama3.2:1b | Dev set | mps |

## Kaggle Configs (free GPU + LLM)

| Config | Retriever | LLM | Dataset | Device |
|---|---|---|---|---|
| `kaggle_llama1b_bm25_devset` | BM25 | Llama 3.2 1B | Dev set | cuda |
| `kaggle_llama1b_bm25_blindset_A` | BM25 | Llama 3.2 1B | Blind A | cuda |
| `kaggle_llama1b_bert_blindset_A` | BERT | Llama 3.2 1B | Blind A | cuda |

## HuggingFace Hub Configs (requires network + HF access)

| Config | Retriever | LLM | Dataset | Device |
|---|---|---|---|---|
| `llama1b_bm25_devset` | BM25 | Llama 3.2 1B (HF) | Dev set | cuda |
| `llama1b_bert_devset` | BERT | Llama 3.2 1B (HF) | Dev set | cuda |
| `llama1b_bm25_blindset_A` | BM25 | Llama 3.2 1B (HF) | Blind A | cuda |
| `llama1b_bert_blindset_A` | BERT | Llama 3.2 1B (HF) | Blind A | cuda |
| `qwen4b_bert_devset` | BERT | Qwen3-4B (HF) | Dev set | cuda |
| `bm25_retrieval_only_devset` | BM25 | none | Dev set (HF) | mps |
| `bert_retrieval_only_devset` | BERT | none | Dev set (HF) | mps |

## Environment Variables

| Variable | Used by | Default |
|---|---|---|
| `MCRS_TEST_DB` | `lowerbound/*.py` | `talkpl-ai/TalkPlayData-Challenge-Dataset` |
| `MCRS_TRACK_DB` | `lowerbound/random_sample.py` | `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` |
| `MCRS_USER_DB` | reserved for future use | `talkpl-ai/TalkPlayData-Challenge-User-Metadata` |
| `MCRS_BLIND_A_DB` | reserved for future use | `talkpl-ai/TalkPlayData-Challenge-Blind-A` |
| `HF_TOKEN` | HuggingFace auth (private datasets) | unset |

Set these in `.env` and run `source .env` before executing scripts.

---

# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: mcrs` | Package not installed | `pip install -e .` from repo root |
| `KeyError: 'all_tracks'` | Wrong split name | Ensure `track_split_types: ["all_tracks"]` in config |
| CUDA / MPS out of memory | Batch too large | Reduce `--batch_size` to `4` or `8` |
| Stale BM25 / BERT index | Changed `corpus_types` but cache exists | `rm -rf cache/` |
| `flash_attention_2` error | Not installed | Set `attn_implementation: "eager"` in config |
| `[Errno 61] Connection refused` | Ollama not running | Run `ollama serve` in a separate terminal |
| `Server returned: 0 error` on upload | Corporate network blocking | Disconnect from Walmart VPN, use personal hotspot |
| HuggingFace SSL error | Corporate proxy intercepts HTTPS | Use `local_*` configs or Ollama |
| `SSH: Operation timed out` on git push | Port 22 blocked by corporate firewall | Use HTTPS: `git remote set-url origin https://github.com/...` |
| `FileNotFoundError: /kaggle/input/llama-3.2/...` | Model not attached | Add Llama model via Kaggle sidebar → + Add Input → New Model |
| `AuthenticationError` on HuggingFace | Private dataset, no token | Set `HF_TOKEN` in Kaggle secrets or `.env` |
| GPU quota exhausted on Kaggle | 30 hr/week limit | Resets Sunday; use CPU or wait |
| `predicted_response` is empty | Using `lm_type: "none"` | Expected for retrieval-only; use Ollama or Kaggle LLM configs for responses |
