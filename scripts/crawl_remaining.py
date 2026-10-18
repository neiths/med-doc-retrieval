"""Dedicated crawler for all remaining missing URLs across Vietnamese and Chinese sources.

Features:
1. Dynamic Remaining URL Detection:
   - Scans data/processed/v2/crawled/*.jsonl to skip all already-crawled doc_ids.
   - Automatically skips PubMed abstracts (already 100% completed).
2. Scope Selection:
   - --scope vi  : Crawls all remaining Vietnamese medical sites (~381k URLs) [Default].
   - --scope zh  : Crawls all remaining Chinese medical sites (~222k URLs).
   - --scope all : Crawls all remaining URLs (~603k URLs).
3. Domain Round-Robin Interleaving:
   - Rotates requests across domains (e.g. Domain A -> Domain B -> Domain C...)
   - Eliminates single-domain semaphore contention and maximizes throughput (15-30 req/s).
4. Adaptive Circuit-Breaker:
   - If any domain returns 429/403, pauses ONLY that specific domain for 45s.
   - Other domains continue at full speed.
5. Sharded Output:
   - Rotates JSONL shard files every 5,000 articles into data/processed/v2/crawled/.

Usage:
    # 1. Check remaining URLs breakdown:
    python scripts/crawl_remaining.py --stats-only

    # 2. Crawl all remaining Vietnamese URLs (30 workers, fast throughput):
    python scripts/crawl_remaining.py --scope vi --concurrency 30

    # 3. Crawl remaining Vietnamese URLs except Long Chau (if running Long Chau separately):
    python scripts/crawl_remaining.py --scope vi --exclude-hosts nhathuoclongchau.com.vn --concurrency 30

    # 4. Crawl remaining Chinese Q&A and disease encyclopedias:
    python scripts/crawl_remaining.py --scope zh --concurrency 25

    # 5. Crawl ALL remaining URLs:
    python scripts/crawl_remaining.py --scope all --concurrency 35
"""

import argparse
import asyncio
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from scripts.crawl_missing_urls import (
    AdaptiveCrawlerManager,
    QueueItem,
    interleave_by_host,
)

ZH_DOMAINS = {
    "120ask.com",
    "cnkang.com",
    "39.net",
    "familydoctor.com.cn",
    "zysjonline.com",
    "zhongyibaodian.net",
    "a-hospital.com",
    "youlai.cn",
    "chunyuyisheng.com",
    "haodf.com",
}


def load_crawled_doc_ids(out_dir: Path) -> set[int]:
    """Scans all shard files and returns all successfully crawled doc_ids."""
    crawled_ids = set()
    for f in out_dir.glob("*.jsonl"):
        if "failed_urls" in f.name:
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        crawled_ids.add(json.loads(line)["doc_id"])
        except Exception:
            pass
    return crawled_ids


def get_remaining_data(missing_file: Path, out_dir: Path):
    """Loads missing parquet, filters already crawled, and partitions by language."""
    logger.info(f"Loading URL registry from {missing_file}...")
    tbl = pq.read_table(missing_file)
    rows = tbl.to_pylist()

    # Exclude PubMed (already 100% completed)
    rows = [r for r in rows if "pubmed" not in r["host"]]

    crawled_ids = load_crawled_doc_ids(out_dir)
    remaining = [r for r in rows if r["id"] not in crawled_ids]

    vi_rows = [r for r in remaining if not any(d in r["host"] for d in ZH_DOMAINS)]
    zh_rows = [r for r in remaining if any(d in r["host"] for d in ZH_DOMAINS)]

    return rows, crawled_ids, vi_rows, zh_rows


def print_stats(total_rows, crawled_ids, vi_rows, zh_rows):
    print("=" * 70)
    print("           REMAINING MEDICAL CORPUS URL BREAKDOWN")
    print("=" * 70)
    print(f"Total Registry Non-PubMed URLs: {len(total_rows):,}")
    print(f"Already Crawled in Local Shards: {len(crawled_ids):,}")
    print(f"Total Remaining URLs:            {len(vi_rows) + len(zh_rows):,}")
    print(f" - Remaining Vietnamese URLs:    {len(vi_rows):,}")
    print(f" - Remaining Chinese URLs:       {len(zh_rows):,}")
    print("-" * 70)

    print("\n[Top Remaining Vietnamese Domains]:")
    for h, c in Counter(r["host"] for r in vi_rows).most_common(15):
        print(f"  {h:38s}: {c:,} URLs")

    print("\n[Top Remaining Chinese Domains]:")
    for h, c in Counter(r["host"] for r in zh_rows).most_common(10):
        print(f"  {h:38s}: {c:,} URLs")
    print("=" * 70)


async def main_async(args):
    total_rows, crawled_ids, vi_rows, zh_rows = get_remaining_data(args.missing_file, args.out_dir)

    if args.stats_only:
        print_stats(total_rows, crawled_ids, vi_rows, zh_rows)
        return

    # Filter target pool based on scope
    if args.scope == "vi":
        target_rows = vi_rows
        scope_desc = "Vietnamese Medical Websites"
    elif args.scope == "zh":
        target_rows = zh_rows
        scope_desc = "Chinese Medical Q&A & Encyclopedias"
    else:
        target_rows = vi_rows + zh_rows
        scope_desc = "All Remaining Sources (VI + ZH)"

    # Filter by hosts if provided
    if args.hosts:
        include_set = set(args.hosts)
        target_rows = [r for r in target_rows if r["host"] in include_set]
        logger.info(f"Applied --hosts filter: {len(target_rows):,} URLs matching {include_set}")

    # Exclude hosts if provided
    if args.exclude_hosts:
        ex_set = set(args.exclude_hosts)
        target_rows = [r for r in target_rows if r["host"] not in ex_set]
        logger.info(f"Applied --exclude-hosts filter: {len(target_rows):,} URLs remaining after excluding {ex_set}")

    if not target_rows:
        logger.success(f"No URLs remaining to crawl for scope '{args.scope}'! All requested URLs are already collected.")
        return

    # Create queue items and round-robin interleave across domains
    queue_items = [QueueItem(doc_id=r["id"], url=r["url"], host=r["host"]) for r in target_rows]
    if args.interleave:
        queue_items = interleave_by_host(queue_items)

    if args.limit and args.limit > 0:
        queue_items = queue_items[: args.limit]
        logger.info(f"Applied --limit {args.limit}: fetching {len(queue_items):,} URLs.")

    unique_domains = len(set(it.host for it in queue_items))

    print("=" * 70)
    print(f"Starting Crawl for: {scope_desc}")
    print(f"Target URLs:        {len(queue_items):,}")
    print(f"Unique Domains:     {unique_domains}")
    print(f"Concurrency:        {args.concurrency} workers (Max {args.domain_concurrency}/domain)")
    print(f"Interleave Mode:    Round-Robin across domains enabled")
    print("=" * 70)

    # Initialize manager
    manager = AdaptiveCrawlerManager(
        concurrency=args.concurrency,
        max_domain_concurrency=args.domain_concurrency,
        domain_interval=args.domain_interval,
        cooldown_seconds=args.cooldown_seconds,
        max_retries=args.max_retries,
        timeout=args.timeout,
        shard_size=args.shard_size,
        out_dir=args.out_dir,
    )
    await manager.init_files()

    for item in queue_items:
        await manager.work_queue.put(item)

    pbar = tqdm(total=len(queue_items), desc=f"Crawling ({args.scope})", unit="url", dynamic_ncols=True)

    limits = httpx.Limits(
        max_keepalive_connections=args.concurrency * 2,
        max_connections=args.concurrency * 3,
    )
    timeout = httpx.Timeout(args.timeout)

    stop_retry_event = asyncio.Event()
    feeder_task = asyncio.create_task(manager.retry_feeder(stop_retry_event))

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        workers = [
            asyncio.create_task(manager.fetch_worker(i, client, pbar))
            for i in range(args.concurrency)
        ]

        await manager.work_queue.join()

        while True:
            async with manager.retry_lock:
                active_retries = len(manager.retry_list)
            if active_retries == 0 and manager.work_queue.empty():
                break
            await asyncio.sleep(1.0)
            await manager.work_queue.join()

        for _ in range(args.concurrency):
            await manager.work_queue.put(None)
        await asyncio.gather(*workers)

    stop_retry_event.set()
    await feeder_task
    await manager.close()
    pbar.close()

    logger.info("=" * 70)
    logger.info(f"Crawl Completed for scope '{args.scope}'!")
    logger.info(f" - Successfully Saved:   {manager.stats['success']:,} articles")
    logger.info(f" - Permanent Failures:    {manager.stats['failed_perm']:,} (logged in failed_urls.jsonl)")
    logger.info(f" - Total Retries Queued:  {manager.stats['retries_queued']:,}")
    logger.info(f" - Circuit Breaker Trips: {manager.stats['rate_limits_hit']:,}")
    logger.info(f" - Output Directory:      {args.out_dir}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scope",
        type=str,
        default="vi",
        choices=["vi", "zh", "all"],
        help="Target scope to crawl: 'vi' (Vietnamese - 381k), 'zh' (Chinese - 222k), 'all' (all 603k).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print remaining URLs statistics and exit.",
    )
    parser.add_argument(
        "--missing-file",
        type=Path,
        default=Path("data/processed/v2/missing_urls.parquet"),
        help="Path to missing_urls.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/v2/crawled"),
        help="Directory to save crawled JSONL shards.",
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=None,
        help="Explicit list of hosts to include.",
    )
    parser.add_argument(
        "--exclude-hosts",
        nargs="+",
        default=None,
        help="List of hosts to exclude (e.g. nhathuoclongchau.com.vn).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max URLs to fetch (0 for all remaining URLs).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=30,
        help="Total concurrent HTTP workers across all domains (default: 30).",
    )
    parser.add_argument(
        "--domain-concurrency",
        type=int,
        default=3,
        help="Max concurrent connections per individual domain (default: 3).",
    )
    parser.add_argument(
        "--domain-interval",
        type=float,
        default=0.25,
        help="Base pacing interval between requests to the same domain (default: 0.25s).",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=45.0,
        help="Cooldown duration when a domain returns 429/403 (default: 45s).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for failed/banned URLs before marking permanent failure (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="HTTP request timeout in seconds (default: 12s).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Number of articles per JSONL shard file before rotating (default: 5000).",
    )
    parser.add_argument(
        "--interleave",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interleave URLs across domains in round-robin fashion (default: True).",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
