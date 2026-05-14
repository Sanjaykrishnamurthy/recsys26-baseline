# Kaggle GPU Setup — Llama 3.2 1B Inference

> Run the full BM25 + Llama 3.2 1B pipeline on a free Kaggle T4 GPU.  
> This gives you real LLM-generated responses, not empty strings.

---

## Why Kaggle?

| | Mac (local) | Kaggle (free GPU) |
|---|---|---|
| GPU | Apple MPS (no CUDA) | NVIDIA T4 / P100 |
| Llama 3.2 1B | ❌ Blocked by Walmart VPN | ✅ Pre-loaded via Kaggle Models |
| HuggingFace datasets | ❌ Blocked by Walmart VPN | ✅ Direct access |
| Cost | Free | Free (30 hrs/week GPU quota) |

---

## Step 1 — Create a Kaggle Notebook

1. Go to [kaggle.com](https://kaggle.com) → **Create** → **New Notebook**
2. In **Settings** (right panel):
   - **Accelerator**: T4 GPU x2  *(or P100 if T4 not available)*
   - **Internet**: On (required to load HuggingFace datasets)

---

## Step 2 — Add the Llama 3.2 1B Model

In the notebook sidebar → **+ Add Data** → **Models** tab:

1. Search: `meta-llama/llama-3.2`
2. Select: **Llama 3.2** → **Transformers** → **1B Instruct** → **Version 1**
3. Click **Add**

The model is now mounted at:
```
/kaggle/input/llama-3.2/transformers/1b-instruct/1/
```
No download, no HuggingFace token, no SSL issues.

---

## Step 3 — Add the TalkPlay Datasets

You have two options:

### Option A — HuggingFace Hub (easier, requires HF token)

If the `talkpl-ai/` datasets are public or you have access:

1. Go to your Kaggle **Account** → **Settings** → **Secrets**
2. Add a secret: `HF_TOKEN` = your HuggingFace read token
3. In the notebook, expose it before running inference:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

The configs `kaggle_llama1b_bm25_blindset_A.yaml` already use HF Hub dataset IDs — nothing else to change.

### Option B — Upload Your Local Parquet Files (reliable, no token)

1. Go to [kaggle.com/datasets](https://kaggle.com/datasets) → **New Dataset**
2. Upload the four folders from your local `data/raw/`:
   - `TalkPlayData-Challenge-Dataset/`
   - `TalkPlayData-Challenge-Track-Metadata/`
   - `TalkPlayData-Challenge-User-Metadata/`
   - `TalkPlayData-Challenge-Blind-A/`
3. Name the dataset (e.g., `talkplay-challenge-data`) and make it **Private**
4. In your notebook → **+ Add Data** → **Datasets** → find your dataset

Then create custom configs with the Kaggle mount paths:
```yaml
test_dataset_name: "/kaggle/input/talkplay-challenge-data/TalkPlayData-Challenge-Blind-A"
item_db_name:      "/kaggle/input/talkplay-challenge-data/TalkPlayData-Challenge-Track-Metadata"
user_db_name:      "/kaggle/input/talkplay-challenge-data/TalkPlayData-Challenge-User-Metadata"
```

---

## Step 4 — Clone the Repo & Install

In your Kaggle notebook code cell:

```python
# Clone the repo
!git clone https://github.com/Sanjaykrishnamurthy/recsys26-baseline.git
%cd recsys26-baseline

# Install dependencies (Kaggle already has torch, transformers, datasets)
!pip install bm25s omegaconf psutil -q
!pip install -e . -q
```

---

## Step 5 — Run Inference

### BM25 + Llama 3.2 1B — Blind Set A

```python
!python run_inference_blindset.py \
    --tid kaggle_llama1b_bm25_blindset_A \
    --eval_dataset blindset_A \
    --batch_size 16
```

Output → `exp/inference/blindset_A/kaggle_llama1b_bm25_blindset_A.json`

### BERT + Llama 3.2 1B — Blind Set A

```python
!python run_inference_blindset.py \
    --tid kaggle_llama1b_bert_blindset_A \
    --eval_dataset blindset_A \
    --batch_size 16
```

> **Tip:** Start with BM25 — it builds its index in ~30 sec. BERT encodes all tracks on first run (~2–3 min on T4) and caches them.

### Dev Set (to check quality before blind submission)

```python
!python run_inference_devset.py \
    --tid kaggle_llama1b_bm25_devset \
    --batch_size 16
```

---

## Step 6 — Package and Download

```python
!python prepare_submission.py \
    --input exp/inference/blindset_A/kaggle_llama1b_bm25_blindset_A.json \
    --output submissions/submission.zip
```

Then download `submissions/submission.zip` from the Kaggle file browser (right-click → Download).

---

## Step 7 — Upload Submission

Upload `submission.zip` from your personal network (not Walmart VPN — it blocks the upload endpoint).

---

## Config Reference

| Config | Retriever | Dataset source |
|---|---|---|
| `kaggle_llama1b_bm25_blindset_A.yaml` | BM25 | HuggingFace Hub |
| `kaggle_llama1b_bert_blindset_A.yaml` | BERT | HuggingFace Hub |
| `kaggle_llama1b_bm25_devset.yaml` | BM25 | HuggingFace Hub |

All three use:
- `lm_type: "/kaggle/input/llama-3.2/transformers/1b-instruct/1"` — local Kaggle model path
- `device: "cuda"`
- `attn_implementation: "eager"` — compatible with all Kaggle GPU types

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'mcrs'` | Run `pip install -e .` from the repo root |
| `CUDA out of memory` | Reduce `--batch_size` to `8` or `4` |
| `FileNotFoundError: /kaggle/input/llama-3.2/...` | Model not added to notebook — go to **+ Add Data → Models** |
| `AuthenticationError` on HuggingFace dataset | Set `HF_TOKEN` secret (see Step 3 Option A) |
| GPU quota exhausted | Kaggle resets 30 hr/week on Sunday; use CPU in the meantime |
| Slow BERT embedding build | Normal — ~2–3 min on T4 for full track catalog; subsequent runs use cache |

---

## Expected Output (with LLM)

Each record now has a real response:

```json
{
  "session_id": "9c37dcd7-...",
  "user_id": "u_1234",
  "turn_number": 1,
  "predicted_track_ids": ["t_001", "t_002", ...],
  "predicted_response": "Here's a track that should get you moving! 'Uptown Funk' by Bruno Mars..."
}
```

Compare this to the retrieval-only output where `predicted_response` is always `""`. The LLM adds conversational quality that improves the BLEU/ROUGE response metrics.
