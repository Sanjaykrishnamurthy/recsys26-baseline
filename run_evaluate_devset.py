"""
Local evaluation script for Music CRS predictions.

Computes the same metrics used by the Codabench leaderboard (except LLM-Judge):
  - nDCG@1, nDCG@10, nDCG@20  (retrieval quality, 50% of composite score)
  - Catalog Diversity           (recommendation breadth, 10%)
  - Lexical Diversity           (response variety, 10%)

Usage:
    # Evaluate a specific prediction file (reads exp/inference/devset/<tid>.json):
    python run_evaluate_devset.py --tid local_bm25_devset

    # With a config file (resolves dataset paths from config/<tid>.yaml):
    python run_evaluate_devset.py --tid llama1b_sbert_devset --config llama1b_sbert_devset

    # Regenerate ground truth before evaluating:
    python run_evaluate_devset.py --tid local_bm25_devset --regenerate-gt

Prediction file format (JSON array):
    [
        {
            "session_id": "...",
            "user_id": "...",
            "turn_number": 1,
            "predicted_track_ids": ["id1", "id2", ...],
            "predicted_response": "Here is a song ..."
        },
        ...
    ]

Output:
    exp/scores/devset/<tid>.json          — per-run JSON scores
    exp/scores/leaderboard.csv            — cumulative comparison log
"""

import os
import csv
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm

from mcrs.evaluation import (
    compute_recsys_metrics,
    compute_catalog_diversity,
    compute_lexical_diversity,
)
from mcrs.utils import load_local_or_hf_dataset


# ---------------------------------------------------------------------------
# Leaderboard CSV logger
# ---------------------------------------------------------------------------

LEADERBOARD_PATH = "exp/scores/leaderboard.csv"
LEADERBOARD_COLUMNS = [
    "timestamp",
    "tid",
    "lm_type",
    "retrieval_type",
    "eval_dataset",
    "ndcg@1",
    "ndcg@10",
    "ndcg@20",
    "catalog_diversity",
    "lexical_diversity",
    "partial_composite",
    "notes",
]


def _append_to_leaderboard(
    tid: str,
    eval_dataset: str,
    metrics: dict,
    lm_type: str = "",
    retrieval_type: str = "",
    notes: str = "",
) -> str:
    """Append a row to the cumulative leaderboard CSV.

    Creates the file with headers if it doesn't exist yet.  Duplicate
    (tid, eval_dataset) entries are kept — each run is a separate row
    stamped with the current time so you can track improvements over time.

    Args:
        tid: Experiment identifier.
        eval_dataset: Dataset partition (e.g. ``"devset"``).
        metrics: Dictionary returned by ``evaluate()``.
        lm_type: LLM model name (from config, or empty).
        retrieval_type: Retrieval method name (from config, or empty).
        notes: Optional free-text notes for this run.

    Returns:
        Path to the leaderboard CSV file.
    """
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)
    file_exists = os.path.isfile(LEADERBOARD_PATH)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tid": tid,
        "lm_type": lm_type,
        "retrieval_type": retrieval_type,
        "eval_dataset": eval_dataset,
        "ndcg@1": f"{metrics.get('ndcg@1', 0):.6f}",
        "ndcg@10": f"{metrics.get('ndcg@10', 0):.6f}",
        "ndcg@20": f"{metrics.get('ndcg@20', 0):.6f}",
        "catalog_diversity": f"{metrics.get('catalog_diversity', 0):.6f}",
        "lexical_diversity": f"{metrics.get('lexical_diversity', 0):.6f}",
        "partial_composite": f"{metrics.get('partial_composite_score', 0):.6f}",
        "notes": notes,
    }

    with open(LEADERBOARD_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return LEADERBOARD_PATH


def print_leaderboard() -> None:
    """Print the leaderboard CSV as a formatted table to stdout."""
    if not os.path.isfile(LEADERBOARD_PATH):
        print("[leaderboard] No runs logged yet.")
        return

    df = pd.read_csv(LEADERBOARD_PATH)
    # Sort by partial composite descending for easy comparison
    df = df.sort_values("partial_composite", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based rank
    df.index.name = "rank"

    print("\n" + "=" * 120)
    print("  LOCAL LEADERBOARD  (sorted by partial composite score)")
    print("=" * 120)
    print(df.to_string())
    print("=" * 120 + "\n")


# ---------------------------------------------------------------------------
# Ground-truth generation (ported from music-crs-evaluator/make_ground_truth.py)
# ---------------------------------------------------------------------------

def _parse_ground_truth_turn(
    conversations: List[Dict[str, Any]], target_turn_number: int
) -> Tuple[str, str]:
    """Extract the ground-truth track ID and response for a single turn.

    Args:
        conversations: List of conversation turn dictionaries.
        target_turn_number: Turn to extract (1-8).

    Returns:
        (ground_truth_track_id, ground_truth_response)
    """
    df = pd.DataFrame(conversations)
    df_turn = df[df["turn_number"] == target_turn_number]
    recommend_music = df_turn.iloc[1]["content"]   # 'music' role row
    response = df_turn.iloc[2]["content"]           # 'assistant' role row
    return recommend_music, response


def generate_ground_truth(
    dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Dataset",
    split: str = "test",
    output_path: str = "exp/ground_truth/devset.json",
) -> str:
    """Build the ground-truth file from the TalkPlayData conversation dataset.

    Uses ``load_local_or_hf_dataset`` so both HF Hub IDs and local directory
    paths work transparently (no network required when using local data).

    Args:
        dataset_name: HF dataset ID or local path.
        split: Dataset split to use (``"test"`` for devset evaluation).
        output_path: Where to write the JSON ground-truth file.

    Returns:
        The resolved ``output_path``.
    """
    print(f"[evaluate] Generating ground truth from '{dataset_name}' (split={split}) ...")
    db = load_local_or_hf_dataset(dataset_name)[split]

    ground_truth_tracks = []
    for item in tqdm(db, desc="Building ground truth"):
        for turn in range(1, 9):
            gt_track_id, _ = _parse_ground_truth_turn(item["conversations"], turn)
            ground_truth_tracks.append({
                "session_id": item["session_id"],
                "user_id": item["user_id"],
                "turn_number": turn,
                "ground_truth_track_id": gt_track_id,
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_tracks, f, indent=4)

    print(f"[evaluate] Ground truth saved → {output_path}  ({len(ground_truth_tracks)} records)")
    return output_path


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    tid: str,
    eval_dataset: str = "devset",
    item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    test_dataset_name: str = "talkpl-ai/TalkPlayData-Challenge-Dataset",
    regenerate_gt: bool = False,
    lm_type: str = "",
    retrieval_type: str = "",
    notes: str = "",
) -> dict:
    """Score a prediction file against ground truth.

    Args:
        tid: Experiment identifier — matches ``exp/inference/<eval_dataset>/<tid>.json``.
        eval_dataset: Dataset partition name (``"devset"``).
        item_db_name: HF ID or local path for track metadata (used for catalog size).
        test_dataset_name: HF ID or local path for conversation dataset (used for
            ground truth generation if the file doesn't exist yet).
        regenerate_gt: Force-regenerate ground truth even if it already exists.
        lm_type: LLM model name (logged to leaderboard CSV).
        retrieval_type: Retrieval method name (logged to leaderboard CSV).
        notes: Optional free-text notes for this run.

    Returns:
        Dictionary of macro-averaged metric scores.
    """
    # --- Paths ---
    gt_path = f"exp/ground_truth/{eval_dataset}.json"
    pred_path = f"exp/inference/{eval_dataset}/{tid}.json"
    score_path = f"exp/scores/{eval_dataset}/{tid}.json"

    # --- Validate prediction file ---
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}\n"
            f"Run inference first, e.g.:\n"
            f"  python run_inference_devset.py --tid {tid}"
        )

    # --- Ground truth (generate if missing) ---
    if regenerate_gt or not os.path.isfile(gt_path):
        generate_ground_truth(
            dataset_name=test_dataset_name,
            split="test",
            output_path=gt_path,
        )
    else:
        print(f"[evaluate] Using existing ground truth: {gt_path}")

    # --- Load data ---
    print(f"[evaluate] Loading predictions: {pred_path}")
    ground_truth = json.load(open(gt_path, "r"))
    predictions = json.load(open(pred_path, "r"))

    df_predictions = pd.DataFrame(predictions)
    df_ground_truth = pd.DataFrame(ground_truth)

    print(f"[evaluate] Ground truth records: {len(df_ground_truth)}")
    print(f"[evaluate] Prediction records:   {len(df_predictions)}")

    # --- Per-turn scoring ---
    results = []
    list_of_recommended_track_ids = []
    list_of_responses = []

    for _, row in tqdm(df_ground_truth.iterrows(), total=len(df_ground_truth), desc="Scoring"):
        session_id = row["session_id"]
        turn_number = row["turn_number"]

        # Look up prediction for this session + turn
        mask = (df_predictions["session_id"] == session_id) & (
            df_predictions["turn_number"] == turn_number
        )
        matched = df_predictions[mask]
        if matched.empty:
            raise ValueError(
                f"Missing prediction for session_id={session_id}, turn_number={turn_number}"
            )

        pred_row = matched.iloc[0]
        prediction_track_ids = pred_row["predicted_track_ids"]
        ground_truth_track_id = row["ground_truth_track_id"]

        recsys_metrics = compute_recsys_metrics(
            prediction_track_ids, [ground_truth_track_id], [1, 10, 20]
        )
        list_of_recommended_track_ids.extend(prediction_track_ids)
        list_of_responses.append(pred_row["predicted_response"])

        results.append({
            "session_id": session_id,
            "turn_number": turn_number,
            **recsys_metrics,
        })

    # --- Aggregate: macro-average (turn-wise, then session-wise) ---
    df_results = pd.DataFrame(results)
    df_turn_wise = df_results.drop(columns=["session_id"]).groupby("turn_number").agg("mean")
    macro_results = df_turn_wise.mean(axis=0).to_dict()

    # --- Diversity metrics ---
    print("[evaluate] Computing diversity metrics ...")
    catalog = load_local_or_hf_dataset(item_db_name)["all_tracks"]
    total_catalog_size = len(catalog)

    macro_results["catalog_diversity"] = compute_catalog_diversity(
        list_of_recommended_track_ids, total_catalog_size
    )
    macro_results["lexical_diversity"] = compute_lexical_diversity(list_of_responses)
    macro_results["total_catalog_size"] = total_catalog_size

    # --- Partial composite score (70% of total — excludes LLM-Judge) ---
    partial_composite = (
        0.50 * macro_results["ndcg@20"]
        + 0.10 * macro_results["catalog_diversity"]
        + 0.10 * macro_results["lexical_diversity"]
    )
    macro_results["partial_composite_score"] = partial_composite

    # --- Save ---
    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(macro_results, f, indent=2)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print(f"  Evaluation Results: {tid}")
    print("=" * 60)
    print(f"  nDCG@1:              {macro_results['ndcg@1']:.6f}")
    print(f"  nDCG@10:             {macro_results['ndcg@10']:.6f}")
    print(f"  nDCG@20:             {macro_results['ndcg@20']:.6f}")
    print(f"  Catalog Diversity:   {macro_results['catalog_diversity']:.6f}")
    print(f"  Lexical Diversity:   {macro_results['lexical_diversity']:.6f}")
    print(f"  Catalog Size:        {macro_results['total_catalog_size']}")
    print("-" * 60)
    print(f"  Partial Composite:   {partial_composite:.6f}  (70% weight, excl. LLM-Judge)")
    print("=" * 60)
    print(f"  Saved → {score_path}")

    # --- Append to leaderboard CSV ---
    lb_path = _append_to_leaderboard(
        tid=tid,
        eval_dataset=eval_dataset,
        metrics=macro_results,
        lm_type=lm_type,
        retrieval_type=retrieval_type,
        notes=notes,
    )
    print(f"  Logged → {lb_path}")

    # --- Show full leaderboard ---
    print_leaderboard()

    return macro_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Music CRS predictions locally (same metrics as Codabench, minus LLM-Judge)."
    )
    parser.add_argument(
        "--tid",
        type=str,
        default=None,
        help="Experiment identifier — reads exp/inference/devset/<tid>.json",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="devset",
        help="Dataset partition name (default: devset)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML name in config/ (e.g. 'llama1b_bm25_devset'). "
             "If provided, resolves item_db_name and test_dataset_name from it "
             "so local paths work automatically.",
    )
    parser.add_argument(
        "--regenerate-gt",
        action="store_true",
        help="Force-regenerate ground truth even if it already exists.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes for this run (logged to leaderboard CSV).",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Just print the leaderboard and exit (no evaluation).",
    )
    args = parser.parse_args()

    # Show leaderboard only
    if args.leaderboard:
        print_leaderboard()
        return

    # --tid is required for evaluation (but not for --leaderboard)
    if not args.tid:
        parser.error("--tid is required (unless using --leaderboard)")

    # Resolve dataset paths and model info from config if provided
    item_db_name = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
    test_dataset_name = "talkpl-ai/TalkPlayData-Challenge-Dataset"
    lm_type = ""
    retrieval_type = ""

    if args.config:
        from omegaconf import OmegaConf
        config_path = f"config/{args.config}.yaml"
        if os.path.isfile(config_path):
            cfg = OmegaConf.load(config_path)
            item_db_name = getattr(cfg, "item_db_name", item_db_name)
            test_dataset_name = getattr(cfg, "test_dataset_name", test_dataset_name)
            lm_type = getattr(cfg, "lm_type", "")
            retrieval_type = getattr(cfg, "retrieval_type", "")
            print(f"[evaluate] Config loaded: {config_path}")
            print(f"[evaluate]   lm_type:            {lm_type}")
            print(f"[evaluate]   retrieval_type:     {retrieval_type}")
            print(f"[evaluate]   item_db_name:       {item_db_name}")
            print(f"[evaluate]   test_dataset_name:  {test_dataset_name}")
        else:
            print(f"[evaluate] Warning: config file not found: {config_path}, using HF defaults.")

    evaluate(
        tid=args.tid,
        eval_dataset=args.eval_dataset,
        item_db_name=item_db_name,
        test_dataset_name=test_dataset_name,
        regenerate_gt=args.regenerate_gt,
        lm_type=lm_type,
        retrieval_type=retrieval_type,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
