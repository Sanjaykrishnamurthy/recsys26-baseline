# Local Evaluation Integration

**Date**: 2026-05-16

## Summary

Integrated the official [music-crs-evaluator](https://github.com/nlp4musa/music-crs-evaluator) into the baselines project as a first-class subpackage (`mcrs.evaluation`). This eliminates the need to maintain a separate evaluator directory, copy prediction files between projects, or rely on HuggingFace Hub connectivity for evaluation.

## What Changed

| Change | Description |
|--------|-------------|
| **`mcrs/evaluation/`** | New subpackage containing `metrics_recsys.py` and `metrics_diversity.py`, ported verbatim from the official evaluator |
| **`run_evaluate_devset.py`** | New root-level script — reads from the shared `exp/` directory, auto-generates ground truth if missing, and prints a summary with partial composite score |
| **`exp/ground_truth/devset.json`** | Pre-built ground truth file (8,000 records: 1,000 sessions × 8 turns) — copied from the evaluator so the first run works offline |
| **`pyproject.toml`** | Added `numpy` and `scipy` to dependencies (required by metrics) |

## Architecture

```
run_inference_devset.py   →  exp/inference/devset/<tid>.json
                                       │
run_evaluate_devset.py    ←────────────┘
        │
        ├── reads exp/ground_truth/devset.json  (auto-generates if missing)
        ├── computes nDCG@{1,10,20}, catalog diversity, lexical diversity
        └── writes exp/scores/devset/<tid>.json
```

Both inference and evaluation now share a single `exp/` directory — no file copying required.

## Usage

### Basic (uses HF Hub for track metadata catalog size)

```bash
python run_evaluate_devset.py --tid local_bm25_devset
```

### With config (resolves local dataset paths — works offline)

```bash
python run_evaluate_devset.py --tid local_bm25_devset --config llama1b_bm25_devset
```

The `--config` flag loads `config/<name>.yaml` and uses `item_db_name` and `test_dataset_name` from it, so local directory paths are resolved automatically via `load_local_or_hf_dataset()`.

### Force-regenerate ground truth

```bash
python run_evaluate_devset.py --tid local_bm25_devset --regenerate-gt
```

## Output

Results are saved to `exp/scores/devset/<tid>.json` and printed to stdout:

```
============================================================
  Evaluation Results: local_bm25_devset
============================================================
  nDCG@1:              0.009750
  nDCG@10:             0.062721
  nDCG@20:             0.081495
  Catalog Diversity:   0.379491
  Lexical Diversity:   0.254897
  Catalog Size:        47071
------------------------------------------------------------
  Partial Composite:   0.104187  (70% weight, excl. LLM-Judge)
============================================================
```

### Partial Composite Score

The script computes a **partial composite** covering 70% of the Codabench weight:

```
partial_composite = 0.50 × nDCG@20 + 0.10 × catalog_diversity + 0.10 × lexical_diversity
```

The remaining 30% (LLM-as-a-Judge) is only available via Codabench submission.

## Metrics Reference

| Metric | Weight | Description |
|--------|--------|-------------|
| nDCG@20 | 0.50 | Retrieval ranking quality |
| Catalog Diversity | 0.10 | Unique tracks / catalog size |
| Lexical Diversity | 0.10 | Distinct-2 bigram ratio |
| LLM-Judge | 0.30 | *Codabench only* — Gemini scores personalization + explanation |

## Programmatic Access

The evaluation function can also be called directly from Python:

```python
from mcrs.evaluation import compute_recsys_metrics, compute_catalog_diversity, compute_lexical_diversity

# Or use the full evaluation pipeline:
from run_evaluate_devset import evaluate

results = evaluate(tid="local_bm25_devset", item_db_name="./data/raw/TrackMetadata")
```
