"""Dataset loading utilities supporting both local parquet files and HuggingFace Hub.

When running on a network where HuggingFace is unreachable (e.g. corporate
proxies), pass a local directory path in the config instead of a HF dataset ID.
The loader auto-detects whether the value is a local path or a HF identifier
and handles each case correctly.

Local directory structure expected (standard HF clone layout):
    data/raw/TalkPlayData-Challenge-Track-Metadata/
        data/
            all_tracks-00000-of-00001.parquet
            test_tracks-00000-of-00001.parquet
        README.md

Parquet filenames must follow the HuggingFace shard convention:
    {split_name}-{shard:05d}-of-{total:05d}.parquet
"""

from __future__ import annotations

import glob
import os
from datasets import load_dataset, DatasetDict


def load_local_or_hf_dataset(name_or_path: str) -> DatasetDict:
    """Load a dataset from a local directory or from the HuggingFace Hub.

    If ``name_or_path`` resolves to an existing directory on disk, parquet
    files inside a ``data/`` subdirectory are discovered and grouped into a
    ``DatasetDict`` automatically.  Otherwise the value is forwarded to
    ``datasets.load_dataset()`` for Hub access.

    Args:
        name_or_path: HuggingFace dataset ID (e.g. ``"talkpl-ai/..."``), or a
            local directory path (absolute or relative, e.g.
            ``"./data/raw/TalkPlayData-Challenge-Track-Metadata"``).

    Returns:
        A ``DatasetDict`` whose keys are split names (e.g. ``"all_tracks"``,
        ``"train"``, ``"test"``).

    Raises:
        FileNotFoundError: If the local directory contains no parquet files.
    """
    resolved = os.path.expanduser(name_or_path)

    if os.path.isdir(resolved):
        return _load_from_local_dir(resolved)

    # Fall through to HuggingFace Hub
    return load_dataset(name_or_path)


def _load_from_local_dir(dataset_dir: str) -> DatasetDict:
    """Build a DatasetDict by scanning parquet files in a local directory.

    Args:
        dataset_dir: Path to the root of the cloned HF dataset repository.

    Returns:
        DatasetDict with one entry per discovered split.

    Raises:
        FileNotFoundError: If no parquet files are found.
    """
    # HF repos keep data inside a `data/` sub-folder; fall back to root
    data_dir = os.path.join(dataset_dir, "data")
    if not os.path.isdir(data_dir):
        data_dir = dataset_dir

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(
            f"No .parquet files found in '{data_dir}'. "
            "Check that the dataset folder is placed correctly under data/raw/."
        )

    # Group shards by split name.
    # Filename pattern: {split_name}-{shard:05d}-of-{total:05d}.parquet
    # Example: all_tracks-00000-of-00004.parquet  →  split "all_tracks"
    split_to_files: dict[str, list[str]] = {}
    for filepath in parquet_files:
        stem = os.path.basename(filepath).replace(".parquet", "")
        parts = stem.split("-")
        # Last 3 parts are always: <shard_num>, "of", <total_shards>
        split_name = "-".join(parts[:-3])
        split_to_files.setdefault(split_name, []).append(filepath)

    # Load each split into a Dataset
    dataset_dict: dict = {}
    for split_name, files in split_to_files.items():
        dataset_dict[split_name] = load_dataset(
            "parquet",
            data_files={split_name: sorted(files)},
            split=split_name,
        )

    print(f"[INFO] Loaded local dataset from '{dataset_dir}' — splits: {list(dataset_dict.keys())}")
    return DatasetDict(dataset_dict)
