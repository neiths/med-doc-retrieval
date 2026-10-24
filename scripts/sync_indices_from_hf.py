"""Sync new pre-built multi-lingual indices (2.3M vectors) from HF Bucket nieths/ViBioMIR.

Usage:
    # Sync lightweight SQ8 index + slim SQLite (~7.4 GB, recommended for inference):
    python scripts/sync_indices_from_hf.py --mode sq8 --clean-old

    # Sync full unquantized FP32 index (~19.3 GB):
    python scripts/sync_indices_from_hf.py --mode full --clean-old

    # Sync all files including BM25s:
    python scripts/sync_indices_from_hf.py --mode all --clean-old
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi
from loguru import logger


SQ8_FILES = [
    ("indices/dense_index_sq8.faiss", "data/indices/dense_index_sq8.faiss"),
    ("indices/chunks_meta_slim.sqlite", "data/indices/chunks_meta_slim.sqlite"),
    ("indices/dense_metadata_sq8.json", "data/indices/dense_metadata_sq8.json"),
]

FULL_FILES = [
    ("indices/dense_index.faiss", "data/indices/dense_index.faiss"),
    ("indices/chunks_meta.sqlite", "data/indices/chunks_meta.sqlite"),
    ("indices/dense_metadata.json", "data/indices/dense_metadata.json"),
]

BM25S_FILES = [
    ("indices/bm25s_index/data.csc.index.npy", "data/indices/bm25s_index/data.csc.index.npy"),
    ("indices/bm25s_index/indices.csc.index.npy", "data/indices/bm25s_index/indices.csc.index.npy"),
    ("indices/bm25s_index/indptr.csc.index.npy", "data/indices/bm25s_index/indptr.csc.index.npy"),
    ("indices/bm25s_index/params.index.json", "data/indices/bm25s_index/params.index.json"),
    ("indices/bm25s_index/vocab.index.json", "data/indices/bm25s_index/vocab.index.json"),
    ("indices/bm25s_metadata.json", "data/indices/bm25s_metadata.json"),
]

ENRICHED_FILES = [
    ("processed/queries_enriched.jsonl", "data/processed/queries_enriched.jsonl"),
]


def clean_old_indices(indices_dir: Path):
    logger.info(f"Cleaning obsolete/incompatible index files in {indices_dir}...")
    patterns = [
        "bm25_index.pkl",
    ]
    for pat in patterns:
        target = indices_dir / pat
        if target.exists():
            size_mb = target.stat().st_size / (1024 * 1024)
            logger.info(f"Removing old file: {target.name} ({size_mb:.2f} MB)")
            target.unlink()
    logger.info("Obsolete index files cleaned.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=["sq8", "hybrid", "bm25s", "full", "all"],
        default="hybrid",
        help="Which index package to download (sq8, hybrid, bm25s, full, all). Default: hybrid.",
    )
    parser.add_argument(
        "--clean-old",
        action="store_true",
        default=False,
        help="Remove outdated legacy pickle indices before downloading.",
    )
    parser.add_argument(
        "--bucket-id",
        type=str,
        default="nieths/ViBioMIR",
        help="Hugging Face Bucket ID.",
    )
    parser.add_argument(
        "--indices-dir",
        type=Path,
        default=Path("data/indices"),
        help="Local directory to store indices.",
    )

    args = parser.parse_args()
    indices_dir = args.indices_dir
    indices_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_old:
        clean_old_indices(indices_dir)

    # Determine files to download
    files_to_sync = []
    if args.mode == "sq8":
        files_to_sync = SQ8_FILES
    elif args.mode == "bm25s":
        files_to_sync = BM25S_FILES + ENRICHED_FILES
    elif args.mode == "hybrid":
        files_to_sync = SQ8_FILES + BM25S_FILES + ENRICHED_FILES
    elif args.mode == "full":
        files_to_sync = FULL_FILES
    elif args.mode == "all":
        files_to_sync = SQ8_FILES + FULL_FILES + BM25S_FILES + ENRICHED_FILES

    # Filter out files that already exist and are non-empty
    needed_files = []
    for remote, local in files_to_sync:
        p = Path(local)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or p.stat().st_size == 0:
            needed_files.append((remote, str(p)))
        else:
            logger.info(f"Already downloaded: {local} ({p.stat().st_size / (1024*1024):.2f} MB)")

    if not needed_files:
        logger.info("All requested index files are already present and non-empty!")
        return

    logger.info(f"Connecting to Hugging Face Bucket '{args.bucket_id}' to sync {len(needed_files)} files...")
    api = HfApi()

    t0 = time.time()
    try:
        api.download_bucket_files(
            bucket_id=args.bucket_id,
            files=needed_files,
        )
        elapsed = time.time() - t0
        logger.info(f"Sync completed successfully in {elapsed / 60:.2f} minutes!")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 INDEX SYNC COMPLETE!")
    for _, local in files_to_sync:
        p = Path(local)
        if p.exists():
            print(f"  ✓ {p.name:<30} ({p.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
