"""Script to integrate clean, valid records from DataByHien into the Parquet corpus
chunked into 100,000-row shards (corpus_part_008.parquet onwards), update scraped_ids artifacts,
and sync to Hugging Face Storage Bucket.
"""

import argparse
import glob
import logging
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.ingestion.cleaner import detect_language, normalize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("integrate_databyhien")

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


def flush_batch(batch: list[dict[str, Any]], shard_num: int, corpus_dir: Path) -> Path:
    out_file = corpus_dir / f"corpus_part_{shard_num:03d}.parquet"
    logger.info(f"Writing {len(batch):,} records to {out_file.name} (ZSTD level 9)...")
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


def main():
    parser = argparse.ArgumentParser(description="Integrate DataByHien into Parquet corpus.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw/DataByHien"),
        help="Directory with DataByHien parquet shards.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/processed/parquet_corpus"),
        help="Directory with existing parquet shards.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Maximum rows per new Parquet shard.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Minimum character threshold for article text.",
    )
    parser.add_argument(
        "--sync-to-bucket",
        action="store_true",
        help="Sync updated parquet corpus and filters to HF Bucket.",
    )
    args = parser.parse_args()

    # 1. Load existing scraped IDs
    scraped_pq = Path("data/processed/scraped_ids.parquet")
    existing_ids = set()
    if scraped_pq.exists():
        t = pq.read_table(scraped_pq)
        existing_ids.update(t["doc_id"].to_pylist())
        logger.info(f"Loaded {len(existing_ids):,} existing IDs from {scraped_pq}")
    else:
        for f in sorted(args.corpus_dir.glob("corpus_part_*.parquet")):
            t = pq.read_table(f, columns=["doc_id"])
            existing_ids.update(t["doc_id"].to_pylist())
        logger.info(f"Loaded {len(existing_ids):,} existing IDs from corpus shards")

    # 2. Determine starting index for new parts
    existing_files = sorted(args.corpus_dir.glob("corpus_part_*.parquet"))
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

    # 3. Find all DataByHien shards
    shards = sorted(glob.glob(str(args.source_dir / "train_shard_*.parquet")))
    logger.info(f"Found {len(shards)} shards in {args.source_dir}")
    if not shards:
        logger.error("No shards found to process!")
        return

    new_seen_ids = set()
    current_batch: list[dict[str, Any]] = []
    created_files: list[Path] = []

    stats = {
        "total_read": 0,
        "failed_or_error_skipped": 0,
        "already_in_existing_skipped": 0,
        "duplicate_in_batch_skipped": 0,
        "too_short_skipped": 0,
        "valid_kept": 0,
        "domain_counts": Counter(),
        "lang_counts": Counter(),
    }

    logger.info("Processing DataByHien shards...")
    for sf in tqdm(shards, desc="Scanning shards"):
        tbl = pq.read_table(sf)
        stats["total_read"] += len(tbl)

        ids = tbl["id"].to_pylist()
        urls = tbl["url"].to_pylist()
        domains = tbl["domain"].to_pylist()
        successes = tbl["success"].to_pylist()
        statuses = tbl["status_code"].to_pylist()
        titles = tbl["title"].to_pylist()
        contents = tbl["content"].to_pylist()

        for did, url, dom, succ, stat, tit, cont in zip(
            ids, urls, domains, successes, statuses, titles, contents
        ):
            if not (succ is True and stat == 200 and cont):
                stats["failed_or_error_skipped"] += 1
                continue

            did = int(did)
            if did in existing_ids:
                stats["already_in_existing_skipped"] += 1
                continue

            if did in new_seen_ids:
                stats["duplicate_in_batch_skipped"] += 1
                continue

            norm_text = normalize_text(str(cont), strip_boilerplate=True)
            text_len = len(norm_text.strip())
            if text_len < args.min_chars:
                stats["too_short_skipped"] += 1
                continue

            new_seen_ids.add(did)
            lang = "en" if "pubmed" in dom else detect_language(norm_text)
            clean_title = str(tit or "").strip()
            clean_url = str(url or "").strip()
            clean_dom = str(dom or "").strip().lower()

            stats["valid_kept"] += 1
            stats["domain_counts"][clean_dom] += 1
            stats["lang_counts"][lang] += 1

            current_batch.append({
                "doc_id": did,
                "url": clean_url,
                "title": clean_title,
                "text": norm_text,
                "lang": lang,
                "domain": clean_dom,
                "char_count": text_len,
                "status": "success",
            })

            if len(current_batch) >= args.chunk_size:
                cf = flush_batch(current_batch, next_shard_idx, args.corpus_dir)
                created_files.append(cf)
                next_shard_idx += 1
                current_batch = []

    if current_batch:
        cf = flush_batch(current_batch, next_shard_idx, args.corpus_dir)
        created_files.append(cf)
        next_shard_idx += 1
        current_batch = []

    logger.info("=" * 65)
    logger.info("PROCESSING SUMMARY:")
    logger.info(f"  Total records read:                 {stats['total_read']:,}")
    logger.info(f"  Failed/Error status skipped:        {stats['failed_or_error_skipped']:,}")
    logger.info(f"  Already in corpus skipped:          {stats['already_in_existing_skipped']:,}")
    logger.info(f"  Duplicate in new batch skipped:     {stats['duplicate_in_batch_skipped']:,}")
    logger.info(f"  Too short (<{args.min_chars} chars) skipped:       {stats['too_short_skipped']:,}")
    logger.info(f"  VALID NEW ARTICLES KEPT:            {stats['valid_kept']:,}")
    logger.info(f"  New Shards Created:                 {len(created_files)}")
    for cf in created_files:
        logger.info(f"    - {cf.name} ({cf.stat().st_size / (1024*1024):.2f} MB)")
    logger.info(f"  Languages: {dict(stats['lang_counts'].most_common())}")
    logger.info("=" * 65)

    if not created_files:
        logger.warning("No new valid shards created.")
        return

    # 4. Update scraped_ids artifacts
    all_seen = existing_ids | new_seen_ids
    sorted_all = sorted(all_seen)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    scraped_ids_pq = processed_dir / "scraped_ids.parquet"
    id_table = pa.Table.from_arrays([pa.array(sorted_all, type=pa.int64())], names=["doc_id"])
    pq.write_table(id_table, scraped_ids_pq, compression="zstd")
    logger.info(f"Updated {scraped_ids_pq} with {len(sorted_all):,} IDs ({scraped_ids_pq.stat().st_size / (1024*1024):.2f} MB)")

    scraped_ids_txt = processed_dir / "scraped_ids.txt"
    with open(scraped_ids_txt, "w", encoding="utf-8") as f:
        for did in sorted_all:
            f.write(f"{did}\n")
    logger.info(f"Updated {scraped_ids_txt} with {len(sorted_all):,} IDs ({scraped_ids_txt.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Sync to Hugging Face Storage Bucket
    if args.sync_to_bucket and created_files:
        hf_bin = shutil.which("hf") or "hf"
        token = load_hf_token()
        env = os.environ.copy()
        if token:
            env["HF_TOKEN"] = token

        logger.info("Syncing updated corpus to hf://buckets/nieths/ViBioMIR/corpus ...")
        res_corpus = subprocess.run(
            [hf_bin, "sync", str(args.corpus_dir), "hf://buckets/nieths/ViBioMIR/corpus"],
            env=env,
        )
        if res_corpus.returncode == 0:
            logger.info("Corpus synced to Hugging Face Bucket successfully!")
        else:
            logger.error("Failed to sync corpus to Hugging Face Bucket.")

        logger.info("Uploading updated scraped_ids filter to Hugging Face Bucket...")
        subprocess.run(
            [hf_bin, "buckets", "cp", str(scraped_ids_pq), "hf://buckets/nieths/ViBioMIR/scraped_ids.parquet"],
            env=env,
        )
        subprocess.run(
            [hf_bin, "buckets", "cp", str(scraped_ids_txt), "hf://buckets/nieths/ViBioMIR/scraped_ids.txt"],
            env=env,
        )
        logger.info("Hugging Face Bucket filters sync completed successfully!")

    logger.info(f"TOTAL CORPUS UNIQUE ARTICLES: {len(sorted_all):,}")


if __name__ == "__main__":
    main()
