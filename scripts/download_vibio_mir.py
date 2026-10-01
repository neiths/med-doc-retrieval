"""Script to download, inspect, and extract the ViBioMIR dataset from Hugging Face.

Dataset: AIGuruTinix/ViBioMIR
(Vietnamese-Centric Multilingual Biomedical Information Retrieval Dataset)

Files:
- query.parquet (1,200 Vietnamese medical queries)
- links_corpus.parquet (~4.4 million URLs across Vietnamese and Chinese medical sources)
- README.md & figures
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vibio_mir_downloader")


def download_dataset(repo_id: str = "AIGuruTinix/ViBioMIR", output_dir: str | Path = "data/raw/vibio_mir") -> Path:
    """Downloads the ViBioMIR dataset from Hugging Face."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading dataset '{repo_id}' to '{out_path}'...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out_path),
    )
    logger.info(f"Dataset successfully downloaded to: {local_dir}")
    return Path(local_dir)


def inspect_and_export_queries(
    parquet_path: Path,
    export_jsonl_path: Path | None = None,
) -> int:
    """Reads queries parquet, displays stats, and optionally exports to JSONL."""
    if not parquet_path.exists():
        logger.error(f"Queries file not found at {parquet_path}")
        return 0

    table = pq.read_table(parquet_path)
    records = table.to_pylist()
    num_queries = len(records)
    logger.info(f"Loaded {num_queries} queries from {parquet_path}")

    lengths = [len(r["query"].split()) for r in records if "query" in r]
    if lengths:
        lengths_sorted = sorted(lengths)
        median_len = lengths_sorted[len(lengths_sorted) // 2]
        p95_len = lengths_sorted[int(len(lengths_sorted) * 0.95)]
        logger.info(
            f"Query word length stats: Min={min(lengths)}, Max={max(lengths)}, "
            f"Median={median_len}, P95={p95_len}"
        )

    if export_jsonl_path:
        export_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Exported {num_queries} queries to {export_jsonl_path}")

    return num_queries


def inspect_and_export_corpus(
    parquet_path: Path,
    export_sample_path: Path | None = None,
    sample_size: int = 10000,
) -> int:
    """Reads corpus parquet, calculates domain statistics, and optionally exports a sample JSONL."""
    if not parquet_path.exists():
        logger.error(f"Corpus file not found at {parquet_path}")
        return 0

    table = pq.read_table(parquet_path, columns=["id", "url"])
    total_urls = len(table)
    logger.info(f"Corpus contains {total_urls:,} total URLs")

    # Domain sampling analysis (sample every 20th URL)
    url_col = table["url"]
    step = 20
    domains = Counter()
    sampled_count = 0
    for i in range(0, total_urls, step):
        u = url_col[i].as_py()
        if u:
            sampled_count += 1
            try:
                domain = urlparse(u).netloc.lower()
                domains[domain] += 1
            except Exception:
                pass

    logger.info(f"Domain distribution (sampled {sampled_count:,} URLs across corpus):")
    for d, count in domains.most_common(15):
        pct = (count / sampled_count) * 100
        logger.info(f"  {d:<30} {count:>6} ({pct:>5.2f}%)")

    if export_sample_path and sample_size > 0:
        export_sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_records = table.slice(0, sample_size).to_pylist()
        with open(export_sample_path, "w", encoding="utf-8") as f:
            for r in sample_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Exported sample of {len(sample_records)} URLs to {export_sample_path}")

    return total_urls


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ViBioMIR dataset from Hugging Face.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="AIGuruTinix/ViBioMIR",
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/vibio_mir",
        help="Directory to save the downloaded dataset.",
    )
    parser.add_argument(
        "--queries-jsonl",
        type=str,
        default="data/raw/queries.jsonl",
        help="Path to export queries as JSONL for the pipeline.",
    )
    parser.add_argument(
        "--sample-urls-jsonl",
        type=str,
        default="data/raw/sample_urls.jsonl",
        help="Path to export sample URLs JSONL.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of URLs to export in sample JSONL.",
    )

    args = parser.parse_args()

    # 1. Download dataset
    dataset_dir = download_dataset(repo_id=args.repo_id, output_dir=args.output_dir)

    # 2. Inspect & export queries
    query_file = dataset_dir / "query.parquet"
    queries_jsonl = Path(args.queries_jsonl) if args.queries_jsonl else None
    inspect_and_export_queries(query_file, export_jsonl_path=queries_jsonl)

    # 3. Inspect & export sample corpus
    corpus_file = dataset_dir / "links_corpus.parquet"
    sample_urls_jsonl = Path(args.sample_urls_jsonl) if args.sample_urls_jsonl else None
    inspect_and_export_corpus(corpus_file, export_sample_path=sample_urls_jsonl, sample_size=args.sample_size)

    logger.info("ViBioMIR preparation completed successfully!")


if __name__ == "__main__":
    main()
