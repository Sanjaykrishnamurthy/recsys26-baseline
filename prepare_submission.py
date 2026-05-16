"""Prepare a competition-ready submission zip from inference output.

Takes the raw JSON produced by run_inference_blindset.py, validates it against
the official submission rules, and packages it as submission.zip containing
a single file named prediction.json.

Submission rules enforced:
  - Every (session_id, turn_number) pair must have a prediction
  - predicted_track_ids must be unique within each prediction
  - predicted_track_ids are capped at 20 (most relevant first)
  - predicted_response must be a string (empty string is allowed)
  - Output file must be named prediction.json inside submission.zip

Usage:
    python prepare_submission.py \\
        --input exp/inference/blindset_A/local_bm25_blindset_A.json \\
        --output submission.zip

    # Preview only (no file written)
    python prepare_submission.py \\
        --input exp/inference/blindset_A/local_bm25_blindset_A.json \\
        --dry-run
"""

import argparse
import json
import os
import zipfile
from collections import defaultdict


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_TRACK_IDS = 20
REQUIRED_FIELDS = {"session_id", "user_id", "turn_number", "predicted_track_ids", "predicted_response"}
VALID_TURNS = set(range(1, 9))  # turns 1–8


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_and_clean(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Validate submission records and apply required transformations.

    Transformations applied:
      - Deduplicate predicted_track_ids (preserve order)
      - Truncate predicted_track_ids to MAX_TRACK_IDS
      - Coerce predicted_response to str if needed

    Args:
        records: Raw list of prediction dicts from inference output.

    Returns:
        Tuple of (cleaned_records, warnings).
    """
    warnings = []
    cleaned = []
    seen_pairs: set[tuple] = set()

    for i, rec in enumerate(records):
        prefix = f"Record {i} (session={rec.get('session_id', '?')}, turn={rec.get('turn_number', '?')})"

        # 1. Required fields
        missing = REQUIRED_FIELDS - rec.keys()
        if missing:
            warnings.append(f"{prefix}: Missing fields {missing} — SKIPPED")
            continue

        # 2. Duplicate (session_id, turn_number) pairs
        pair = (rec["session_id"], rec["turn_number"])
        if pair in seen_pairs:
            warnings.append(f"{prefix}: Duplicate pair — SKIPPED")
            continue
        seen_pairs.add(pair)

        # 3. Turn number range
        if rec["turn_number"] not in VALID_TURNS:
            warnings.append(f"{prefix}: turn_number {rec['turn_number']} not in 1–8")

        # 4. Deduplicate track IDs (preserve order, log if changed)
        track_ids = rec["predicted_track_ids"]
        deduped = list(dict.fromkeys(track_ids))  # ordered dedup
        if len(deduped) < len(track_ids):
            warnings.append(f"{prefix}: Removed {len(track_ids) - len(deduped)} duplicate track ID(s)")

        # 5. Truncate to 20
        if len(deduped) > MAX_TRACK_IDS:
            warnings.append(f"{prefix}: Truncated track IDs from {len(deduped)} → {MAX_TRACK_IDS}")
            deduped = deduped[:MAX_TRACK_IDS]

        # 6. Fewer than 20 track IDs (warn only)
        if len(deduped) < MAX_TRACK_IDS:
            warnings.append(f"{prefix}: Only {len(deduped)} track IDs (expected {MAX_TRACK_IDS})")

        # 7. Ensure predicted_response is a string
        response = rec["predicted_response"]
        if not isinstance(response, str):
            warnings.append(f"{prefix}: predicted_response coerced to string")
            response = str(response)

        cleaned.append({
            "session_id":         rec["session_id"],
            "user_id":            rec["user_id"],
            "turn_number":        int(rec["turn_number"]),
            "predicted_track_ids": deduped,
            "predicted_response": response,
        })

    return cleaned, warnings


def check_coverage(records: list[dict]) -> list[str]:
    """Warn about coverage gaps for sessions that have multiple predictions.

    For blind-set submissions each session typically appears once at a specific
    turn (the conversation stage exposed to participants).  Single-turn sessions
    are NOT flagged — only sessions that have more than one prediction but are
    still missing some turns in the covered range get a warning.

    Args:
        records: Cleaned prediction records.

    Returns:
        List of warning strings (empty if no gaps detected).
    """
    session_turns: dict[str, set] = defaultdict(set)
    for rec in records:
        session_turns[rec["session_id"]].add(rec["turn_number"])

    warnings = []
    for session_id, turns in session_turns.items():
        if len(turns) <= 1:
            # Single prediction per session — blind-set style, no gap to check.
            continue
        # Multi-turn session: warn if any turn between min and max is absent.
        span = set(range(min(turns), max(turns) + 1))
        missing = span - turns
        if missing:
            warnings.append(
                f"Session '{session_id}': has turns {sorted(turns)} "
                f"but missing {sorted(missing)} within that range"
            )
    return warnings


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    print(f"Loaded {len(raw_records)} records from '{args.input}'")

    # ── Validate & clean ──────────────────────────────────────────────────────
    cleaned, warnings = validate_and_clean(raw_records)
    coverage_warnings = check_coverage(cleaned)
    all_warnings = warnings + coverage_warnings

    if all_warnings:
        print(f"\n⚠️  {len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"   • {w}")
    else:
        print("✅  All records passed validation — no issues found.")

    # Coverage note: blind-set submissions have one prediction per session at
    # the exposed turn — that is expected and not a gap.
    n_single = sum(1 for r in cleaned if True)  # total records
    multi_session_count = sum(
        1 for sid in {r["session_id"] for r in cleaned}
        if sum(1 for r in cleaned if r["session_id"] == sid) > 1
    )
    if multi_session_count == 0:
        print("ℹ️  Blind-set mode: each session has one prediction (single-turn evaluation).")

    print(f"\nSummary:")
    print(f"   Total records  : {len(cleaned)}")
    print(f"   Unique sessions: {len({r['session_id'] for r in cleaned})}")
    print(f"   Turns covered  : {sorted({r['turn_number'] for r in cleaned})}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ── Write prediction.json ──────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"\nWrote prediction.json → '{pred_path}'")

    # ── Zip as submission.zip ─────────────────────────────────────────────────
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")

    zip_size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Created '{args.output}' ({zip_size_mb:.1f} MB)")
    print("\n🎉  submission.zip is ready to upload!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package inference output into a competition-ready submission.zip"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the inference output JSON (e.g. exp/inference/blindset_A/local_bm25_blindset_A.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submissions/submission.zip",
        help="Path for the output zip file (default: submissions/submission.zip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only — do not write any files",
    )
    main(parser.parse_args())
