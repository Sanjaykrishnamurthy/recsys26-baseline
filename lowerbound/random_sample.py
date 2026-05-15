import os
import json
import random

# Import mcrs.utils FIRST — it auto-loads .env before we read env vars
from mcrs.utils import load_local_or_hf_dataset
from tqdm import tqdm

# Default dataset IDs — override via .env or explicit env vars
TRACK_DB = os.environ.get("MCRS_TRACK_DB", "talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
TEST_DB = os.environ.get("MCRS_TEST_DB", "talkpl-ai/TalkPlayData-Challenge-Dataset")


def load_track_pools(dataset_name: str = TRACK_DB) -> list[str]:
    """Load all available track IDs from the track metadata dataset.

    Args:
        dataset_name: HF dataset ID or local path.

    Returns:
        A list of all track IDs available in the dataset.
    """
    db = load_local_or_hf_dataset(dataset_name)["all_tracks"]
    track_ids = list(db['track_id'])
    return track_ids


def main():
    """Generate random baseline predictions for music recommendation evaluation.

    For each test session × 8 turns, randomly samples 20 tracks from the full
    track pool.  Results are saved to ``exp/inference/random.json``.

    Supports local data via environment variables::

        MCRS_TRACK_DB=./data/raw/TalkPlayData-Challenge-Track-Metadata \\
        MCRS_TEST_DB=./data/raw/TalkPlayData-Challenge-Dataset \\
        python lowerbound/random_sample.py
    """
    track_pools = load_track_pools()
    db = load_local_or_hf_dataset(TEST_DB)["test"]
    inference_results = []
    for item in tqdm(db):
        user_id = item['user_id']
        session_id = item['session_id']
        for target_turn_number in range(1, 9):
            retrieval_track_ids = random.sample(track_pools, 20)
            inference_results.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": target_turn_number,
                "predicted_track_ids": retrieval_track_ids,
                "predicted_response": ""
            })
    os.makedirs("exp/inference", exist_ok=True)
    with open("exp/inference/random.json", "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
