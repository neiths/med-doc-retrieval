"""Utility script to validate, clean, deduplicate, and convert JSONL crawl data into optimized Parquet shards.

Supports automatic synchronization with Hugging Face Storage Buckets:
  hf sync ./data/processed/parquet_corpus hf://buckets/nieths/ViBioMIR/corpus
"""

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jsonl_to_parquet")

# Schema specification for ViBioMIR articles
PARQUET_SCHEMA = pa.schema([
    pa.field("doc_id", pa.int64()),
    pa.field("url", pa.string()),
    pa.field("title", pa.string()),
    pa.field("text", pa.string()),
    pa.field("lang", pa.string()),
    pa.field("domain", pa.string()),
    pa.field("char_count", pa.int32()),
    pa.field("status", pa.string()),
])


def normalize_doc_id(doc_id_val: Any) -> int | None:
    """Ensures doc_id is converted to an integer for BTC corpus."""
    try:
        return int(doc_id_val)
    except (ValueError, TypeError):
        return None


def convert_and_validate(
    input_paths: list[Path],
    output_dir: Path,
    chunk_size: int = 100_000,
    min_chars: int = 50,
) -> list[Path]:
    """Reads JSONL files, performs data cleaning, deduplicates by doc_id, and writes Parquet shards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[int] = set()
    current_batch: list[dict[str, Any]] = []
    shard_index = 0
    created_files: list[Path] = []

    stats = {
        "total_read": 0,
        "valid_kept": 0,
        "duplicate_skipped": 0,
        "invalid_id_skipped": 0,
        "too_short_skipped": 0,
        "domain_counts": Counter(),
        "lang_counts": Counter(),
    }

    def flush_batch(batch_records: list[dict[str, Any]], shard_num: int) -> Path:
        out_file = output_dir / f"corpus_part_{shard_num:03d}.parquet"
        logger.info(f"Writing {len(batch_records):,} records to {out_file} (ZSTD compression)...")

        # Build PyArrow Table directly for high-performance zero-copy write
        arrays = [
            pa.array([r["doc_id"] for r in batch_records], type=pa.int64()),
            pa.array([r["url"] for r in batch_records], type=pa.string()),
            pa.array([r["title"] for r in batch_records], type=pa.string()),
            pa.array([r["text"] for r in batch_records], type=pa.string()),
            pa.array([r["lang"] for r in batch_records], type=pa.string()),
            pa.array([r["domain"] for r in batch_records], type=pa.string()),
            pa.array([r["char_count"] for r in batch_records], type=pa.int32()),
            pa.array([r["status"] for r in batch_records], type=pa.string()),
        ]
        table = pa.Table.from_arrays(arrays, schema=PARQUET_SCHEMA)

        pq.write_table(
            table,
            out_file,
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
        )

        size_mb = out_file.stat().st_size / (1024 * 1024)
        logger.info(f"-> Created {out_file.name}: {size_mb:.2f} MB")
        return out_file

    for input_file in input_paths:
        logger.info(f"Processing input JSONL: {input_file} ...")
        with open(input_file, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {input_file.name}", unit="lines"):
                line = line.strip()
                if not line:
                    continue

                stats["total_read"] += 1
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                raw_id = record.get("doc_id") or record.get("id")
                clean_id = normalize_doc_id(raw_id)
                if clean_id is None:
                    stats["invalid_id_skipped"] += 1
                    continue

                if clean_id in seen_ids:
                    stats["duplicate_skipped"] += 1
                    continue

                text = record.get("text", "") or ""
                text_len = len(text.strip())
                if text_len < min_chars:
                    stats["too_short_skipped"] += 1
                    continue

                seen_ids.add(clean_id)

                url = str(record.get("url") or "")
                domain = "unknown"
                if url:
                    try:
                        domain = urlparse(url).netloc.lower() or "unknown"
                    except Exception:
                        pass

                lang = str(record.get("lang") or "vi")
                stats["domain_counts"][domain] += 1
                stats["lang_counts"][lang] += 1
                stats["valid_kept"] += 1

                current_batch.append({
                    "doc_id": clean_id,
                    "url": url,
                    "title": str(record.get("title") or ""),
                    "text": text,
                    "lang": lang,
                    "domain": domain,
                    "char_count": text_len,
                    "status": str(record.get("status") or "success"),
                })

                if len(current_batch) >= chunk_size:
                    created = flush_batch(current_batch, shard_index)
                    created_files.append(created)
                    shard_index += 1
                    current_batch = []

    if current_batch:
        created = flush_batch(current_batch, shard_index)
        created_files.append(created)

    logger.info("=" * 60)
    logger.info("DATA VALIDATION & CONVERSION SUMMARY:")
    logger.info(f"  Total lines read:           {stats['total_read']:,}")
    logger.info(f"  Valid articles kept:        {stats['valid_kept']:,}")
    logger.info(f"  Duplicates removed:         {stats['duplicate_skipped']:,}")
    logger.info(f"  Articles < {min_chars} chars:       {stats['too_short_skipped']:,}")
    logger.info(f"  Total Parquet shards:       {len(created_files)}")
    logger.info("Top domains:")
    for dom, cnt in stats["domain_counts"].most_common(10):
        logger.info(f"    - {dom}: {cnt:,} articles")
    logger.info(f"Languages: {dict(stats['lang_counts'])}")
    logger.info("=" * 60)

    return created_files


def sync_to_hf_bucket(local_dir: Path, bucket_uri: str):
    """Syncs local folder to Hugging Face Storage Bucket using hf CLI."""
    logger.info(f"Syncing {local_dir} -> {bucket_uri} ...")

    hf_bin = shutil.which("hf")
    if not hf_bin:
        logger.warning(
            "'hf' CLI tool is not found. To install: run 'uv tool install hf' or 'pip install huggingface_hub[cli]'"
        )
        return

    cmd = [hf_bin, "sync", str(local_dir), bucket_uri]
    logger.info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        logger.info(f"Successfully synced data to {bucket_uri}!")
    else:
        logger.error(f"Sync failed with exit code: {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Clean, validate, and convert ViBioMIR JSONL to Parquet.")
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="One or more JSONL files or directories containing JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/processed/parquet_corpus"),
        help="Output directory for Parquet shards (default: data/processed/parquet_corpus).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Maximum rows per Parquet file shard (default: 100,000).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Minimum article character length to keep (default: 50).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="hf://buckets/nieths/ViBioMIR/corpus",
        help="Target Hugging Face Bucket URI (default: hf://buckets/nieths/ViBioMIR/corpus).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Automatically trigger 'hf sync' to the Hugging Face Bucket after conversion.",
    )

    args = parser.parse_args()

    # Expand any glob patterns or directories in inputs
    input_files: list[Path] = []
    for item in args.input:
        p = Path(item)
        if p.is_dir():
            input_files.extend(sorted(p.glob("*.jsonl")))
        elif "*" in str(item):
            for match in glob.glob(str(item)):
                input_files.append(Path(match))
        elif p.exists():
            input_files.append(p)
        else:
            logger.warning(f"Input path not found: {item}")

    if not input_files:
        logger.error("No valid JSONL files found. Exiting.")
        return

    # Convert to Parquet
    created = convert_and_validate(
        input_paths=input_files,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        min_chars=args.min_chars,
    )

    # Sync to HF Bucket if requested
    if args.sync and created:
        sync_to_hf_bucket(args.output_dir, args.bucket)


if __name__ == "__main__":
    main()
