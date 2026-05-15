# Prepare Submission — `prepare_submission.py`

> **Last updated:** 2026-05-15  
> Validates inference output and packages it into a competition-ready `submission.zip`.

---

## Overview

`prepare_submission.py` takes the raw JSON produced by `run_inference_blindset.py`, validates it against the official competition rules, and packages it as `submission.zip` containing a single file named `predictions.json`.

---

## Usage

### Validate first (recommended)

```bash
python prepare_submission.py \
  --input exp/inference/blindset_A/local_bm25_blindset_A.json \
  --dry-run
```

This runs all validation checks and prints a summary without writing any files.

### Create the submission zip

```bash
python prepare_submission.py \
  --input exp/inference/blindset_A/local_bm25_blindset_A.json
```

Output → `submissions/submission.zip` (default)

### Custom output path

```bash
python prepare_submission.py \
  --input exp/inference/blindset_A/local_bm25_blindset_A.json \
  --output my_submission.zip
```

---

## Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--input` | Yes | — | Path to the inference output JSON |
| `--output` | No | `submissions/submission.zip` | Path for the output zip file |
| `--dry-run` | No | off | Validate only — do not write any files |

---

## What It Does

The script applies the following steps in order:

### 1. Load

Reads the input JSON file (a list of prediction dicts).

### 2. Validate & Clean

For each record:

| Check | Action |
|---|---|
| Missing required fields | Record is **skipped** with a warning |
| Duplicate `(session_id, turn_number)` pair | Record is **skipped** with a warning |
| `turn_number` not in 1–8 | Warning logged |
| Duplicate track IDs in `predicted_track_ids` | Deduplicated (order preserved) |
| More than 20 track IDs | Truncated to first 20 |
| Fewer than 20 track IDs | Warning logged (not an error) |
| `predicted_response` is not a string | Coerced to string |

### 3. Coverage Check

- **Blind-set mode** (1 prediction per session): no warnings — this is expected
- **Dev-set mode** (multiple predictions per session): warns if any turn within the covered range is missing

### 4. Write

- Creates `predictions.json` in the output directory
- Zips it as `submission.zip` with the archive name `predictions.json`

---

## Required Fields

Every record in the input JSON must contain:

```json
{
  "session_id": "string",
  "user_id": "string",
  "turn_number": 1,
  "predicted_track_ids": ["track_id_1", "track_id_2", "..."],
  "predicted_response": "string or empty string"
}
```

---

## Examples

### BM25 blind set (retrieval-only, local data)

```bash
python run_inference_blindset.py --tid local_bm25_blindset_A --eval_dataset blindset_A --batch_size 16
python prepare_submission.py --input exp/inference/blindset_A/local_bm25_blindset_A.json
```

### BERT blind set (retrieval-only, local data)

```bash
python run_inference_blindset.py --tid local_bert_blindset_A --eval_dataset blindset_A --batch_size 16
python prepare_submission.py --input exp/inference/blindset_A/local_bert_blindset_A.json
```

### Kaggle Llama + BM25 blind set

```bash
# On Kaggle notebook:
!python run_inference_blindset.py --tid kaggle_llama1b_bm25_blindset_A --eval_dataset blindset_A --batch_size 16
!python prepare_submission.py --input exp/inference/blindset_A/kaggle_llama1b_bm25_blindset_A.json
```

---

## Expected Output

### Blind set (80 sessions, 1 turn each)

```
Loaded 80 records from 'exp/inference/blindset_A/local_bm25_blindset_A.json'
✅  All records passed validation — no issues found.
ℹ️  Blind-set mode: each session has one prediction (single-turn evaluation).

Summary:
   Total records  : 80
   Unique sessions: 80
   Turns covered  : [1, 2, 3, 4, 5, 6, 7, 8]

Wrote predictions.json → '/path/to/submissions/predictions.json'
Created 'submissions/submission.zip' (0.0 MB)

🎉  submission.zip is ready to upload!
```

### Dev set (1000 sessions × 8 turns)

```
Loaded 8000 records from 'exp/inference/devset/local_bm25_devset.json'
✅  All records passed validation — no issues found.

Summary:
   Total records  : 8000
   Unique sessions: 1000
   Turns covered  : [1, 2, 3, 4, 5, 6, 7, 8]
```

---

## Upload

> **Walmart corporate network blocks the upload endpoint.** Disconnect from VPN and use a personal network (phone hotspot or home Wi-Fi).

1. Go to the competition leaderboard
2. Upload `submissions/submission.zip`
3. Wait for the leaderboard to refresh

---

## File Structure

```
submissions/
├── predictions.json     ← cleaned JSON (human-readable, indent=2)
└── submission.zip      ← competition upload (contains predictions.json)
```

Both files are git-ignored via `.gitignore`.
