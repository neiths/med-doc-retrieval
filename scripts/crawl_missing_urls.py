"""B2 - High-performance resumable crawler for missing URLs with structured extractors (Mentor Day 1).

1. Reads missing URLs from data/processed/v2/missing_urls.parquet (or custom file).
2. Supports filtering by host (--hosts nhathuoclongchau.com.vn suckhoedoisong.vn ...).
3. Applies domain-level rate limiting and cooldown backoff.
4. Uses structured extractors (preserving H2/H3 headings and Q&A pairings).
5. Appends output to streaming JSONL shards with auto-resume.

Usage:
    # Quick test on local (30 articles across top hosts):
    python scripts/crawl_missing_urls.py --limit 30

    # Crawl specific host locally:
    python scripts/crawl_missing_urls.py --hosts nhathuoclongchau.com.vn --limit 100

    # Run full crawl:
    python scripts/crawl_missing_urls.py --concurrency 20
"""

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pyarrow.parquet as pq
from loguru import logger
from tqdm.asyncio import tqdm

from src.crawler.extractors import extract_article_structured
from src.ingestion.cleaner import detect_language

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


class AsyncArticleCrawler:
    """Async crawler with per-domain rate limiting and structured extraction."""

    def __init__(
        self,
        concurrency: int = 15,
        max_domain_concurrency: int = 3,
        timeout: float = 12.0,
        out_dir: Path = Path("data/processed/v2/crawled"),
    ):
        self.concurrency = concurrency
        self.max_domain_concurrency = max_domain_concurrency
        self.timeout = timeout
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.global_sem = asyncio.Semaphore(concurrency)
        self.domain_sems: dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(max_domain_concurrency)
        )
        self.domain_last_time: dict[str, float] = defaultdict(float)
        self.domain_interval = 0.35  # seconds minimum between requests to same domain

    async def fetch_one(
        self,
        client: httpx.AsyncClient,
        item: dict,
    ) -> dict | None:
        doc_id = item["id"]
        url = item["url"]
        host = item["host"]

        async with self.global_sem:
            async with self.domain_sems[host]:
                # Polite pacing per domain
                now = time.monotonic()
                elapsed = now - self.domain_last_time[host]
                if elapsed < self.domain_interval:
                    await asyncio.sleep(self.domain_interval - elapsed)
                self.domain_last_time[host] = time.monotonic()

                headers = {
                    "User-Agent": random.choice(USER_AGENTS),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "vi,zh-CN,zh;q=0.9,en;q=0.8",
                }

                for attempt in range(2):
                    try:
                        r = await client.get(url, headers=headers, follow_redirects=True)
                        if r.status_code == 200 and len(r.content) > 300:
                            # Extract structured text
                            extracted = extract_article_structured(r.text, url)
                            text = extracted.get("text", "")
                            title = extracted.get("title", "")
                            if len(text.strip()) >= 50:
                                lang = detect_language(text)
                                return {
                                    "doc_id": doc_id,
                                    "url": url,
                                    "title": title,
                                    "text": text,
                                    "lang": lang,
                                    "domain": host,
                                    "char_count": len(text),
                                    "status": "success",
                                }
                        elif r.status_code == 429:
                            await asyncio.sleep(2.0 * (attempt + 1))
                    except Exception:
                        await asyncio.sleep(0.5)

        return None


async def run_crawler(args):
    # 1. Load missing URLs
    logger.info(f"Loading missing URLs from {args.missing_file}...")
    tbl = pq.read_table(args.missing_file)
    rows = tbl.to_pylist()

    # Exclude PubMed (already handled via B1 E-utilities)
    rows = [r for r in rows if "pubmed" not in r["host"]]

    # Filter by hosts if specified
    if args.hosts:
        target_hosts = set(args.hosts)
        rows = [r for r in rows if r["host"] in target_hosts]
        logger.info(f"Filtered for hosts {target_hosts}: {len(rows):,} URLs remaining.")

    # 2. Check already crawled IDs for resuming
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crawled_ids = set()
    for f in args.out_dir.glob("*.jsonl"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        crawled_ids.add(json.loads(line)["doc_id"])
        except Exception:
            pass

    todo = [r for r in rows if r["id"] not in crawled_ids]
    logger.info(f"Already crawled: {len(crawled_ids):,} | Remaining to crawl: {len(todo):,}")

    if args.limit and args.limit > 0:
        todo = todo[: args.limit]
        logger.info(f"Applied --limit {args.limit}: fetching {len(todo)} URLs.")

    if not todo:
        logger.info("No URLs to crawl!")
        return

    # 3. Setup client and run async queue
    crawler = AsyncArticleCrawler(
        concurrency=args.concurrency,
        max_domain_concurrency=args.domain_concurrency,
        timeout=args.timeout,
        out_dir=args.out_dir,
    )

    out_file = args.out_dir / f"crawled_shard_{int(time.time())}.jsonl"
    logger.info(f"Writing outputs to {out_file}")

    limits = httpx.Limits(max_keepalive_connections=args.concurrency * 2, max_connections=args.concurrency * 3)
    timeout = httpx.Timeout(args.timeout)

    success_count = 0
    fail_count = 0

    with open(out_file, "a", encoding="utf-8") as fp:
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            tasks = [crawler.fetch_one(client, item) for item in todo]
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Crawling URLs"):
                res = await coro
                if res:
                    fp.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fp.flush()
                    success_count += 1
                else:
                    fail_count += 1

    logger.info(
        f"Completed crawl run! Success: {success_count:,} | Failed: {fail_count:,} | Output: {out_file}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--missing-file",
        type=Path,
        default=Path("data/processed/v2/missing_urls.parquet"),
        help="Path to missing_urls.parquet generated by audit_coverage.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/v2/crawled"),
        help="Directory to save crawled shards.",
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=None,
        help="List of specific hosts to crawl (e.g. nhathuoclongchau.com.vn suckhoedoisong.vn).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max URLs to fetch (0 for all).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=15,
        help="Global HTTP request concurrency.",
    )
    parser.add_argument(
        "--domain-concurrency",
        type=int,
        default=2,
        help="Max concurrency per domain.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="HTTP request timeout in seconds.",
    )

    args = parser.parse_args()
    asyncio.run(run_crawler(args))


if __name__ == "__main__":
    main()
