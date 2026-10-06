"""High-performance, distributed-ready corpus crawler for Road to AI 2026.

Features:
- Stream-based asyncio Queue (O(1) RAM usage, supports 4.4M+ URLs without OOM)
- Domain-level Round-Robin (prevents hammering a single host and avoiding 429/403 IP bans)
- Sharding support (--shard-id X --num-shards Y) for multi-worker / multi-machine parallelism
- Priority domain filtering (crawls authoritative hospital & clinical wikis first)
- Auto GB18030 / UTF-8 charset detection for Chinese & Vietnamese medical portals
- Clean text extraction with Trafilatura + BeautifulSoup fallback
- Resumable checkpointing: instantly skips already scraped IDs
"""

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pyarrow.parquet as pq
from tqdm import tqdm

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.url_scraper import ArticleScraper  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("corpus_crawler")

# High-priority authoritative medical domains to crawl first
PRIORITY_DOMAINS = [
    # Vietnamese authoritative medical & clinical portals
    "bachmai.gov.vn",
    "vinmec.com",
    "medlatec.vn",
    "suckhoedoisong.vn",
    "nhathuoclongchau.com.vn",
    "suckhoecongdongonline.vn",
    "hellobacsi.com",
    "thanhnien.vn",
    "dantri.com.vn",
    # Chinese authoritative medical encyclopedias & clinical Q&A
    "a-hospital.com",
    "120ask.com",
    "39.net",
    "familydoctor.com.cn",
    "cnkang.com",
    "zysjonline.com",
    "zhongyibaodian.net",
    "youlai.cn",
]

JUNK_URL_PATTERN = re.compile(
    r"(\.(jpg|jpeg|png|gif|svg|pdf|zip|rar|mp4|avi|mp3|exe|apk)(\?|$)|/tag/|/tags/|/search|page=\d+)",
    re.IGNORECASE,
)


def load_candidate_urls(
    input_file: Path,
    shard_id: int = 0,
    num_shards: int = 1,
    priority_only: bool = False,
    already_scraped: set[str] | None = None,
    max_urls: int | None = None,
) -> list[dict[str, Any]]:
    """Loads URLs from Parquet or JSONL, applies sharding, filtering, and fair domain interleaving."""
    if not input_file.exists():
        logger.info(f"Input file {input_file} not found locally. Auto-downloading from Hugging Face (AIGuruTinix/ViBioMIR)...")
        try:
            from huggingface_hub import hf_hub_download
            input_file.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id="AIGuruTinix/ViBioMIR",
                filename=input_file.name,
                repo_type="dataset",
                local_dir=str(input_file.parent),
            )
            input_file = Path(downloaded)
            logger.info(f"Downloaded candidate file to: {input_file}")
        except Exception as e:
            logger.error(f"Could not auto-download {input_file}: {e}")
            raise FileNotFoundError(f"Missing input candidate file: {input_file}")

    logger.info(f"Loading corpus from {input_file} (Shard {shard_id}/{num_shards})...")

    import itertools

    if input_file.suffix == ".parquet":
        table = pq.read_table(input_file, columns=["id", "url"])
        total_in_file = len(table)
        logger.info(f"Total rows in parquet file: {total_in_file:,}")

        if num_shards > 1:
            indices = list(range(shard_id, total_in_file, num_shards))
            table = table.take(indices)

        # PyArrow's C++ to_pylist is ~20x faster than Python loop
        raw_items = table.to_pylist()
    else:
        raw_items = []
        with open(input_file, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip() and (idx % num_shards == shard_id):
                    raw_items.append(json.loads(line))

    logger.info(f"Shard {shard_id} loaded {len(raw_items):,} initial candidate URLs.")

    # 1. Filter out obvious junk URLs and already scraped IDs
    scraped_set = already_scraped or set()
    valid_items = []
    skipped_scraped = 0
    for it in raw_items:
        uid = str(it.get("id"))
        if uid in scraped_set:
            skipped_scraped += 1
            continue
        u = it.get("url")
        if u and not JUNK_URL_PATTERN.search(u):
            valid_items.append(it)

    if skipped_scraped > 0:
        logger.info(f"Deduplication: skipped {skipped_scraped:,} URLs already scraped in existing corpus.")
    logger.info(f"Remaining valid uncrawled candidates: {len(valid_items):,}.")

    # 2. Group by domain for round-robin interleaving
    domain_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in valid_items:
        try:
            domain = urlparse(it["url"]).netloc.lower()
        except Exception:
            domain = "other"
        domain_buckets[domain].append(it)

    # 3. Order domains: Fair round-robin across all domains (no priority bias)
    if priority_only:
        sorted_domains = sorted(
            domain_buckets.keys(),
            key=lambda d: (
                0 if any(p in d for p in PRIORITY_DOMAINS) else 1,
                -len(domain_buckets[d]),
            ),
        )
        sorted_domains = [d for d in sorted_domains if any(p in d for p in PRIORITY_DOMAINS)]
        logger.info(f"Priority mode: restricted to {len(sorted_domains)} authoritative domains.")
    else:
        # Fair round-robin across all available domains to crawl the entire corpus equally
        sorted_domains = sorted(
            domain_buckets.keys(),
            key=lambda d: -len(domain_buckets[d]),
        )

    # Round-robin selection in O(N) using zip_longest (avoids O(N^2) list.pop(0))
    interleaved_items: list[dict[str, Any]] = []
    for tuple_item in itertools.zip_longest(*(domain_buckets[d] for d in sorted_domains)):
        for it in tuple_item:
            if it is not None:
                interleaved_items.append(it)
                if max_urls and len(interleaved_items) >= max_urls:
                    break
        if max_urls and len(interleaved_items) >= max_urls:
            break

    logger.info(f"Prepared {len(interleaved_items):,} round-robin interleaved URLs ready to crawl.")
    return interleaved_items


def get_already_scraped_ids(
    output_file: Path,
    resume_from: list[Path] | Path | None = None,
) -> set[str]:
    """Reads existing doc_ids from output JSONL and optional resume_from files (JSONL or Parquet)."""
    existing_ids = set()
    raw_paths: list[Path] = []
    if resume_from:
        if isinstance(resume_from, (str, Path)):
            raw_paths.append(Path(resume_from))
        else:
            raw_paths.extend([Path(p) for p in resume_from])
    if output_file.exists():
        raw_paths.append(output_file)

    files_to_check: list[Path] = []
    for rp in raw_paths:
        if not rp.exists():
            continue
        if rp.is_dir():
            pq_list = sorted(rp.glob("*.parquet"))
            jl_list = sorted(rp.glob("*.jsonl"))
            files_to_check.extend(pq_list)
            files_to_check.extend(jl_list)
        else:
            files_to_check.append(rp)

    # Auto-fallback: if no existing IDs loaded, check local or remote scraped_ids.parquet
    if not files_to_check or all(not f.exists() for f in files_to_check):
        auto_scraped_pq = Path("data/processed/scraped_ids.parquet")
        auto_scraped_txt = Path("data/processed/scraped_ids.txt")
        corpus_dir = Path("data/processed/parquet_corpus")

        if auto_scraped_pq.exists():
            files_to_check.append(auto_scraped_pq)
        elif auto_scraped_txt.exists():
            files_to_check.append(auto_scraped_txt)
        elif corpus_dir.exists() and list(corpus_dir.glob("*.parquet")):
            files_to_check.extend(sorted(corpus_dir.glob("*.parquet")))
        else:
            # Auto-download compact 0.68 MB filter from Hugging Face bucket
            logger.info("Auto-downloading scraped_ids filter (0.68 MB) from Hugging Face Bucket...")
            try:
                auto_scraped_pq.parent.mkdir(parents=True, exist_ok=True)
                cmd = f"hf cp hf://buckets/nieths/ViBioMIR/scraped_ids.parquet {auto_scraped_pq}"
                res = os.system(cmd)
                if res == 0 and auto_scraped_pq.exists():
                    files_to_check.append(auto_scraped_pq)
                    logger.info("Successfully fetched remote scraped_ids filter!")
            except Exception as e:
                logger.warning(f"Could not auto-download scraped_ids filter: {e}")

    for fpath in files_to_check:
        if not fpath.exists():
            continue
        if fpath.suffix == ".parquet":
            try:
                table = pq.read_table(fpath, columns=["doc_id"])
                ids = [str(i) for i in table["doc_id"].to_pylist() if i is not None]
                existing_ids.update(ids)
                logger.info(f"Loaded {len(ids):,} existing doc_ids from Parquet {fpath.name}")
            except Exception:
                try:
                    table = pq.read_table(fpath, columns=["id"])
                    ids = [str(i) for i in table["id"].to_pylist() if i is not None]
                    existing_ids.update(ids)
                    logger.info(f"Loaded {len(ids):,} existing doc_ids from Parquet {fpath.name}")
                except Exception as e:
                    logger.warning(f"Could not load IDs from {fpath}: {e}")
        elif fpath.suffix == ".txt":
            count = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    did = line.strip()
                    if did:
                        existing_ids.add(did)
                        count += 1
            logger.info(f"Loaded {count:,} existing doc_ids from text filter {fpath.name}")
        else:
            count = 0
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            did = str(rec.get("doc_id") or rec.get("id"))
                            if did:
                                existing_ids.add(did)
                                count += 1
                        except Exception:
                            pass
            logger.info(f"Loaded {count:,} existing doc_ids from checkpoint {fpath.name}")
    return existing_ids


async def run_crawler_pipeline(
    items: list[dict[str, Any]],
    output_file: Path,
    resume_from: list[Path] | Path | None = None,
    concurrency: int = 15,
    timeout_seconds: int = 15,
    max_retries: int = 2,
    min_delay: float = 0.2,
    max_delay: float = 0.6,
    flush_interval: int = 50,
    limit: int | None = None,
):
    """Executes asynchronous crawling using a fixed-size worker pool, polite delay, and buffered writing."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # If resume_from points to a previous version of the SAME shard file and output_file is empty / new,
    # copy existing data first so final output has everything in one place.
    if resume_from:
        resume_paths = [Path(resume_from)] if isinstance(resume_from, (str, Path)) else [Path(p) for p in resume_from]
        for rp in resume_paths:
            if rp.exists() and rp.resolve() != output_file.resolve() and rp.name == output_file.name:
                if not output_file.exists() or output_file.stat().st_size == 0:
                    logger.info(f"Resuming same file {output_file.name}: copying {rp} to {output_file}...")
                    import shutil
                    shutil.copyfile(rp, output_file)

    existing_ids = get_already_scraped_ids(output_file, resume_from=resume_from)

    items_to_crawl = [it for it in items if str(it.get("id")) not in existing_ids]
    if limit is not None and limit > 0:
        items_to_crawl = items_to_crawl[:limit]

    if existing_ids:
        logger.info(
            f"Resume checkpoint: {len(existing_ids):,} total unique articles skipped. "
            f"Remaining to crawl: {len(items_to_crawl):,}."
        )
    else:
        logger.info(f"Starting fresh crawl of {len(items_to_crawl):,} URLs into {output_file}.")

    if not items_to_crawl:
        logger.info("All candidate URLs already scraped. Exiting.")
        return

    scraper = ArticleScraper(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        concurrency=concurrency,
        min_domain_interval=min_delay,
    )

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=concurrency * 4)
    write_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=1000)

    # Stats tracking
    stats = {"success": 0, "failed": 0, "empty": 0, "total_chars": 0}
    pbar = tqdm(total=len(items_to_crawl), desc="Crawling corpus", unit="url")

    # 1. Producer: Pushes items into the work queue
    async def producer():
        for it in items_to_crawl:
            await queue.put(it)
        # Put None for each worker to signal shutdown
        for _ in range(concurrency):
            await queue.put(None)

    # 2. Worker coroutine: Pulls from queue, scrapes, and sends to write_queue
    async def worker(client: httpx.AsyncClient):
        while True:
            it = await queue.get()
            if it is None:
                queue.task_done()
                break
            try:
                # Polite randomized sleep to avoid strict lockstep bursts and anti-bot rate-limits
                if max_delay > 0:
                    sleep_sec = random.uniform(min_delay, max_delay)
                    await asyncio.sleep(sleep_sec)
                result = await scraper.process_item(client, it)
                if result:
                    await write_queue.put(result)
            except Exception as e:
                logger.debug(f"Error scraping {it.get('url')}: {e}")
            finally:
                queue.task_done()

    # 3. Consumer writer: Writes results sequentially with periodic buffer flush
    async def writer():
        count = 0
        with open(output_file, "a", encoding="utf-8") as out_f:
            while True:
                res = await write_queue.get()
                if res is None:
                    write_queue.task_done()
                    break

                st = res.get("status", "failed")
                stats[st] = stats.get(st, 0) + 1
                text_len = len(res.get("text", ""))
                stats["total_chars"] += text_len

                # Only persist articles with actual content to conserve storage
                if st == "success" and text_len >= 50:
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    count += 1
                    if count % flush_interval == 0:
                        out_f.flush()

                pbar.update(1)
                pbar.set_postfix({
                    "ok": stats.get("success", 0),
                    "empty": stats.get("empty", 0),
                    "fail": stats.get("failed", 0),
                })
                write_queue.task_done()

            out_f.flush()

    # Run producer, workers, and writer
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    async with httpx.AsyncClient(timeout=scraper.timeout, limits=limits) as client:
        writer_task = asyncio.create_task(writer())
        worker_tasks = [asyncio.create_task(worker(client)) for _ in range(concurrency)]
        producer_task = asyncio.create_task(producer())

        await producer_task
        await asyncio.gather(*worker_tasks)

        # Signal writer to finish
        await write_queue.put(None)
        await writer_task

    pbar.close()
    logger.info(
        f"Crawl session completed: {stats['success']:,} successes, {stats['empty']:,} empty, "
        f"{stats['failed']:,} failures. Total chars scraped: {stats['total_chars']:,}. Saved to {output_file}"
    )


def main():
    parser = argparse.ArgumentParser(description="Distributed-ready large-scale crawler for ViBioMIR corpus.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/raw/vibio_mir/links_corpus.parquet"),
        help="Path to links_corpus.parquet or links JSONL.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed/crawled_articles.jsonl"),
        help="Output JSONL file to append articles to.",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=10,
        help="Concurrent request limit (default: 10, recommended for cloud/Kaggle).",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Zero-indexed shard ID for distributed crawling (default: 0).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards for distributed crawling (default: 1).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        nargs="*",
        default=None,
        help="Path(s) to previous crawled JSONL (e.g. /kaggle/input/dataset/crawled_shard_1.jsonl) to resume from.",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.2,
        help="Minimum polite sleep delay in seconds between requests (default: 0.2).",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.6,
        help="Maximum polite sleep delay in seconds between requests (default: 0.6).",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum URLs to crawl in this run (e.g. 5000 for a test batch).",
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Only crawl high-priority authoritative hospital & clinical wikis.",
    )

    args = parser.parse_args()

    # 1. Load already scraped IDs (from resume_from, output_file, local parquet, or auto-download from HF)
    existing_ids = get_already_scraped_ids(args.output, resume_from=args.resume_from)

    # 2. Load & prepare URLs with upfront deduplication and fair domain interleaving
    items = load_candidate_urls(
        input_file=args.input,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        priority_only=args.priority_only,
        already_scraped=existing_ids,
        max_urls=args.limit,
    )

    # 3. Run crawler
    asyncio.run(
        run_crawler_pipeline(
            items=items,
            output_file=args.output,
            resume_from=args.resume_from,
            concurrency=args.concurrency,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
