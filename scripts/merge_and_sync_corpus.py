"""Script to merge existing corpus parquet with new JSONL shards (shards 4 & 5),
deduplicate by doc_id, generate uniform 100k-row Parquet shards, and sync to Hugging Face Bucket.
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
logger = logging.getLogger("merge_and_sync")

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
    """Reads HF_TOKEN from environment or .env file."""
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


def merge_and_create_parquet(
    existing_parquet: Path | None,
    jsonl_files: list[Path],
    output_dir: Path,
    chunk_size: int = 100_000,
    min_chars: int = 50,
) -> list[Path]:
    """Merges existing parquet data and new JSONL shards, deduplicates, and writes clean Parquet shards."""
    staging_dir = output_dir.parent / f"{output_dir.name}_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[int] = set()
    current_batch: list[dict[str, Any]] = []
    shard_index = 0
    created_files: list[Path] = []

    stats = {
        "existing_parquet_records": 0,
        "jsonl_total_read": 0,
        "duplicate_skipped": 0,
        "too_short_skipped": 0,
        "invalid_id_skipped": 0,
        "total_valid_kept": 0,
        "domain_counts": Counter(),
        "lang_counts": Counter(),
    }

    def flush_batch(batch_records: list[dict[str, Any]], shard_num: int) -> Path:
        out_file = staging_dir / f"corpus_part_{shard_num:03d}.parquet"
        logger.info(f"Writing {len(batch_records):,} records to {out_file.name} (ZSTD compression)...")

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
        logger.info(f"-> Created {out_file.name}: {size_mb:.2f} MB ({len(batch_records):,} rows)")
        return out_file

    # Step 1: Load existing records from existing parquet file if present
    if existing_parquet and existing_parquet.exists():
        logger.info(f"Loading existing records from {existing_parquet} ...")
        table = pq.read_table(existing_parquet)
        num_existing = len(table)
        logger.info(f"Found {num_existing:,} existing records.")

        doc_ids = table["doc_id"].to_pylist()
        urls = table["url"].to_pylist()
        titles = table["title"].to_pylist()
        texts = table["text"].to_pylist()
        langs = table["lang"].to_pylist() if "lang" in table.column_names else ["vi"] * num_existing
        domains = table["domain"].to_pylist() if "domain" in table.column_names else ["unknown"] * num_existing
        char_counts = table["char_count"].to_pylist() if "char_count" in table.column_names else [len(t) for t in texts]
        statuses = table["status"].to_pylist() if "status" in table.column_names else ["success"] * num_existing

        for i in range(num_existing):
            cid = doc_ids[i]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            stats["existing_parquet_records"] += 1
            stats["total_valid_kept"] += 1
            stats["lang_counts"][langs[i] or "vi"] += 1
            stats["domain_counts"][domains[i] or "unknown"] += 1

            current_batch.append({
                "doc_id": cid,
                "url": urls[i] or "",
                "title": titles[i] or "",
                "text": texts[i] or "",
                "lang": langs[i] or "vi",
                "domain": domains[i] or "unknown",
                "char_count": char_counts[i] or len(texts[i] or ""),
                "status": statuses[i] or "success",
            })

            if len(current_batch) >= chunk_size:
                created = flush_batch(current_batch, shard_index)
                created_files.append(created)
                shard_index += 1
                current_batch = []

    # Step 2: Read JSONL files and deduplicate against seen_ids
    for jsonl_file in jsonl_files:
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
                    stats["duplicate_skipped"] += 1
                    continue

                raw_text = record.get("text", "") or ""
                text = normalize_text(raw_text, strip_boilerplate=True)
                text_len = len(text.strip())
                if text_len < min_chars:
                    stats["too_short_skipped"] += 1
                    continue

                seen_ids.add(clean_id)
                url = str(record.get("url") or "")
                domain = extract_domain(url)
                lang = str(record.get("lang") or "zh")

                stats["total_valid_kept"] += 1
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

                if len(current_batch) >= chunk_size:
                    created = flush_batch(current_batch, shard_index)
                    created_files.append(created)
                    shard_index += 1
                    current_batch = []

    # Flush remaining records
    if current_batch:
        created = flush_batch(current_batch, shard_index)
        created_files.append(created)

    logger.info("=" * 60)
    logger.info("CORPUS MERGE & DEDUPLICATION SUMMARY:")
    logger.info(f"  Existing records loaded:    {stats['existing_parquet_records']:,}")
    logger.info(f"  JSONL lines read:           {stats['jsonl_total_read']:,}")
    logger.info(f"  Duplicates skipped:         {stats['duplicate_skipped']:,}")
    logger.info(f"  Too short skipped (<{min_chars}):   {stats['too_short_skipped']:,}")
    logger.info(f"  Invalid ID skipped:         {stats['invalid_id_skipped']:,}")
    logger.info(f"  TOTAL UNIQUE ARTICLES KEPT: {stats['total_valid_kept']:,}")
    logger.info(f"  Total Parquet Shards:       {len(created_files)}")
    logger.info("Top domains:")
    for dom, cnt in stats["domain_counts"].most_common(10):
        logger.info(f"    - {dom}: {cnt:,} articles")
    logger.info(f"Languages: {dict(stats['lang_counts'])}")
    logger.info("=" * 60)

    # Verification step
    logger.info("Verifying created Parquet shards...")
    total_verified_rows = 0
    all_verified_ids: set[int] = set()
    final_files: list[Path] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    # Move from staging to output_dir
    for staged_file in staging_dir.glob("*.parquet"):
        dest_file = output_dir / staged_file.name
        t = pq.read_table(staged_file)
        rows = len(t)
        total_verified_rows += rows
        ids = t["doc_id"].to_pylist()
        all_verified_ids.update(ids)
        shutil.move(str(staged_file), str(dest_file))
        final_files.append(dest_file)
        logger.info(f"  ✓ {dest_file.name}: {rows:,} rows, size: {dest_file.stat().st_size / (1024*1024):.2f} MB")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    assert total_verified_rows == stats["total_valid_kept"], (
        f"Row count mismatch! {total_verified_rows} vs {stats['total_valid_kept']}"
    )
    assert len(all_verified_ids) == total_verified_rows, (
        f"Duplicate IDs detected in output shards! {len(all_verified_ids)} vs {total_verified_rows}"
    )
    logger.info(f"Verification PASSED! Exactly {total_verified_rows:,} unique articles across {len(final_files)} shards.")

    return sorted(final_files)


def sync_to_hf_bucket(local_dir: Path, bucket_uri: str, token: str | None = None) -> bool:
    """Syncs local parquet corpus directory to Hugging Face Storage Bucket using 'hf sync'."""
    logger.info(f"Syncing {local_dir} -> {bucket_uri} ...")

    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/.local/bin:{env.get('PATH', '')}"
    if token:
        env["HF_TOKEN"] = token

    hf_bin = shutil.which("hf", path=env["PATH"])
    if not hf_bin:
        logger.error("'hf' CLI executable not found in PATH.")
        return False

    cmd = [hf_bin, "sync", str(local_dir), bucket_uri, "--no-truncate"]
    logger.info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode == 0:
        logger.info(f"Successfully synced all shards to {bucket_uri}!")
        return True
    else:
        logger.error(f"Sync failed with return code {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Merge existing corpus and shard 4, 5, deduplicate, and sync.")
    parser.add_argument(
        "--existing-parquet",
        type=Path,
        default=Path("data/processed/parquet_corpus/corpus_part_000.parquet"),
        help="Path to existing Parquet corpus file (default: data/processed/parquet_corpus/corpus_part_000.parquet).",
    )
    parser.add_argument(
        "--jsonl-files",
        type=Path,
        nargs="+",
        default=[
            Path("data/raw/crawled_shard_4.jsonl"),
            Path("data/raw/crawled_shard_5.jsonl"),
        ],
        help="List of new raw JSONL shard files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/parquet_corpus"),
        help="Destination directory for output Parquet shards.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Max number of records per Parquet shard (default: 100,000).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Minimum character count to keep an article (default: 50).",
    )
    parser.add_argument(
        "--bucket-uri",
        type=str,
        default="hf://buckets/nieths/ViBioMIR/corpus",
        help="Target Hugging Face Bucket URI (default: hf://buckets/nieths/ViBioMIR/corpus).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Trigger hf sync after parquet generation.",
    )

    args = parser.parse_args()

    files = merge_and_create_parquet(
        existing_parquet=args.existing_parquet if args.existing_parquet.exists() else None,
        jsonl_files=args.jsonl_files,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        min_chars=args.min_chars,
    )

    if args.sync and files:
        token = load_hf_token()
        sync_to_hf_bucket(args.output_dir, args.bucket_uri, token=token)


if __name__ == "__main__":
    main()
