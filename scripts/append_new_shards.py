"""Script to append new crawled JSONL shards to the Parquet corpus as new parts
without overwriting or modifying existing corpus shards.
"""

import argparse
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

from src.ingestion.cleaner import normalize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("append_new_shards")

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
    try:
        return int(doc_id_val)
    except (ValueError, TypeError):
        return None


def extract_domain(url: str) -> str:
    if not url:
        return "unknown"
    try:
        return urlparse(url).netloc.lower() or "unknown"
    except Exception:
        return "unknown"


def load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                val = line.split("=", 1)[1].strip()
                if val:
                    return val
    return None


def main():
    parser = argparse.ArgumentParser(description="Append new crawled shards to Parquet corpus.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/processed/parquet_corpus"),
        help="Directory with existing parquet shards.",
    )
    parser.add_argument(
        "--jsonl-files",
        type=Path,
        nargs="+",
        required=True,
        help="New JSONL crawl shards to process.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Max rows per new Parquet shard.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Minimum character threshold.",
    )
    parser.add_argument(
        "--sync-to-bucket",
        action="store_true",
        help="Sync updated parquet corpus to HF Bucket.",
    )
    parser.add_argument(
        "--bucket-uri",
        type=str,
        default="hf://buckets/nieths/ViBioMIR/corpus",
        help="HF Bucket URI.",
    )

    args = parser.parse_args()

    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    existing_files = sorted(args.corpus_dir.glob("corpus_part_*.parquet"))
    logger.info(f"Found {len(existing_files)} existing Parquet shards in {args.corpus_dir}")

    seen_ids: set[int] = set()
    for ef in existing_files:
        t = pq.read_table(ef, columns=["doc_id"])
        ids = t["doc_id"].to_pylist()
        seen_ids.update(ids)
        logger.info(f"  - {ef.name}: {len(ids):,} rows")

    logger.info(f"Total existing unique doc_ids loaded: {len(seen_ids):,}")

    # Determine starting index for new parts
    next_shard_idx = 0
    if existing_files:
        indices = []
        for ef in existing_files:
            try:
                num = int(ef.stem.split("_")[-1])
                indices.append(num)
            except ValueError:
                pass
        if indices:
            next_shard_idx = max(indices) + 1

    logger.info(f"New shards will start naming at corpus_part_{next_shard_idx:03d}.parquet")

    stats = {
        "jsonl_total_read": 0,
        "duplicate_in_existing": 0,
        "duplicate_in_new": 0,
        "too_short_skipped": 0,
        "invalid_id_skipped": 0,
        "new_valid_kept": 0,
        "domain_counts": Counter(),
        "lang_counts": Counter(),
    }

    current_batch: list[dict[str, Any]] = []
    created_files: list[Path] = []

    def flush_batch(batch: list[dict[str, Any]], shard_num: int) -> Path:
        out_file = args.corpus_dir / f"corpus_part_{shard_num:03d}.parquet"
        logger.info(f"Writing {len(batch):,} new records to {out_file.name} (ZSTD compression)...")
        arrays = [
            pa.array([r["doc_id"] for r in batch], type=pa.int64()),
            pa.array([r["url"] for r in batch], type=pa.string()),
            pa.array([r["title"] for r in batch], type=pa.string()),
            pa.array([r["text"] for r in batch], type=pa.string()),
            pa.array([r["lang"] for r in batch], type=pa.string()),
            pa.array([r["domain"] for r in batch], type=pa.string()),
            pa.array([r["char_count"] for r in batch], type=pa.int32()),
            pa.array([r["status"] for r in batch], type=pa.string()),
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
        logger.info(f"-> Created {out_file.name}: {size_mb:.2f} MB ({len(batch):,} rows)")
        return out_file

    for jsonl_file in args.jsonl_files:
        if not jsonl_file.exists():
            logger.warning(f"File not found: {jsonl_file}")
            continue

        logger.info(f"Processing JSONL shard: {jsonl_file.name} ...")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {jsonl_file.name}", unit="lines"):
                line = line.strip()
                if not line:
                    continue
                stats["jsonl_total_read"] += 1

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
                    stats["duplicate_in_existing"] += 1
                    continue

                raw_text = record.get("text", "") or ""
                text = normalize_text(raw_text, strip_boilerplate=True)
                text_len = len(text.strip())
                if text_len < args.min_chars:
                    stats["too_short_skipped"] += 1
                    continue

                seen_ids.add(clean_id)
                url = str(record.get("url") or "")
                domain = extract_domain(url)
                lang = str(record.get("lang") or "zh")

                stats["new_valid_kept"] += 1
                stats["domain_counts"][domain] += 1
                stats["lang_counts"][lang] += 1

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

                if len(current_batch) >= args.chunk_size:
                    cf = flush_batch(current_batch, next_shard_idx)
                    created_files.append(cf)
                    next_shard_idx += 1
                    current_batch = []

    if current_batch:
        cf = flush_batch(current_batch, next_shard_idx)
        created_files.append(cf)

    logger.info("=" * 60)
    logger.info("SUMMARY:")
    logger.info(f"  Existing records in corpus:   {len(seen_ids) - stats['new_valid_kept']:,}")
    logger.info(f"  Lines read from new shards:   {stats['jsonl_total_read']:,}")
    logger.info(f"  Duplicates skipped:           {stats['duplicate_in_existing']:,}")
    logger.info(f"  Too short (<{args.min_chars}):              {stats['too_short_skipped']:,}")
    logger.info(f"  NEW UNIQUE ARTICLES ADDED:    {stats['new_valid_kept']:,}")
    logger.info(f"  New Shards Created:           {len(created_files)}")
    for cf in created_files:
        logger.info(f"    - {cf.name} ({cf.stat().st_size / (1024*1024):.2f} MB)")
    logger.info(f"  TOTAL CORPUS NOW:             {len(seen_ids):,} unique articles")
    logger.info("=" * 60)

    if args.sync_to_bucket and created_files:
        hf_bin = shutil.which("hf") or "hf"
        token = load_hf_token()
        env = os.environ.copy()
        if token:
            env["HF_TOKEN"] = token
        logger.info(f"Syncing updated corpus to {args.bucket_uri} ...")
        res = subprocess.run([hf_bin, "sync", str(args.corpus_dir), args.bucket_uri], env=env)
        if res.returncode == 0:
            logger.info("Hugging Face Bucket sync completed successfully!")
        else:
            logger.error("Hugging Face Bucket sync failed. Please check credentials.")


if __name__ == "__main__":
    main()
