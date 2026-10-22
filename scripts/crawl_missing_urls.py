"""B2 - High-performance resumable crawler with adaptive rate limiting and retry queue.

Features:
1. Continuous high-volume async crawling with configurable concurrency.
2. Smart Ban & Failure Detection:
   - Catches HTTP 429 (Rate Limit), 403 (Forbidden/Cloudflare), 502/503/504, and Timeouts.
   - Puts failed/rate-limited URLs into an async Retry Queue with exponential backoff.
   - Triggers per-domain cooldown (e.g. 45s pause ONLY for that domain) without stopping other domains!
3. Automatic Re-try Feeder:
   - Periodically re-injects delayed retry URLs back into active workers.
4. Permanent Failure Logging:
   - Real 404s or URLs exceeding max retries are cleanly logged to failed_urls.jsonl.
5. Sharded Storage & Auto-Resume:
   - Automatically writes to rotated JSONL shards (e.g. 5,000 articles per shard).
   - Skips already crawled doc_ids on startup.

Usage:
    # Continuous high-volume crawl for Long Chau with adaptive retry queue:
    python scripts/crawl_missing_urls.py --hosts nhathuoclongchau.com.vn --concurrency 15

    # Multi-host crawl with adaptive backoff:
    python scripts/crawl_missing_urls.py --hosts nhathuoclongchau.com.vn suckhoedoisong.vn --concurrency 25
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from src.crawler.extractors import extract_article_structured
from src.ingestion.cleaner import detect_language


def interleave_by_host(items: list) -> list:
    """Interleaves queue items across domains in round-robin fashion.
    
    Prevents single-domain semaphore contention and distributes network load
    evenly across all target servers.
    """
    domain_buckets = defaultdict(deque)
    for it in items:
        domain_buckets[it.host].append(it)

    interleaved = []
    active_keys = list(domain_buckets.keys())
    while active_keys:
        next_active = []
        for d in active_keys:
            q = domain_buckets[d]
            interleaved.append(q.popleft())
            if q:
                next_active.append(d)
        active_keys = next_active
    return interleaved

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]


def normalize_target_url(url: str) -> str:
    """Applies official organizer URL migrations.
    
    BTC Announcement:
    https://pmc-ecm-healthblog.beta.pharmacity.io/<slug>/ -> https://www.pharmacity.vn/<slug>.htm
    """
    if "pmc-ecm-healthblog.beta.pharmacity.io" in url:
        m = re.match(r"https?://pmc-ecm-healthblog\.beta\.pharmacity\.io/([^/?#]+)/?", url)
        if m:
            slug = m.group(1).rstrip("/")
            return f"https://www.pharmacity.vn/{slug}.htm"
    return url


@dataclass
class QueueItem:
    doc_id: int
    url: str
    host: str
    fetch_url: str = ""
    retry_count: int = 0
    retry_after: float = 0.0


class AdaptiveCrawlerManager:
    """Manages continuous crawling, per-domain cooldowns, and the retry queue."""

    def __init__(
        self,
        concurrency: int = 15,
        max_domain_concurrency: int = 3,
        domain_interval: float = 0.25,
        cooldown_seconds: float = 45.0,
        max_retries: int = 3,
        timeout: float = 12.0,
        shard_size: int = 5000,
        out_dir: Path = Path("data/processed/v2/crawled"),
    ):
        self.concurrency = concurrency
        self.max_domain_concurrency = max_domain_concurrency
        self.base_domain_interval = domain_interval
        self.cooldown_seconds = cooldown_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self.shard_size = shard_size
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.work_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        self.retry_list: list[QueueItem] = []
        self.retry_lock = asyncio.Lock()

        self.domain_sems: dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(max_domain_concurrency)
        )
        self.domain_last_time: dict[str, float] = defaultdict(float)
        self.domain_cooldown_until: dict[str, float] = defaultdict(float)
        self.domain_interval: dict[str, float] = defaultdict(lambda: domain_interval)

        # File handles & stats
        self.current_shard_file: Path | None = None
        self.current_shard_fp = None
        self.shard_count = 0
        self.file_lock = asyncio.Lock()

        self.failed_file = self.out_dir / "failed_urls.jsonl"
        self.failed_fp = None

        self.stats = {
            "success": 0,
            "failed_perm": 0,
            "retries_queued": 0,
            "rate_limits_hit": 0,
        }

    async def init_files(self):
        self._rotate_shard()
        self.failed_fp = open(self.failed_file, "a", encoding="utf-8")

    def _rotate_shard(self):
        if self.current_shard_fp:
            try:
                self.current_shard_fp.close()
            except Exception:
                pass
        self.current_shard_file = self.out_dir / f"crawled_shard_{int(time.time())}.jsonl"
        self.current_shard_fp = open(self.current_shard_file, "a", encoding="utf-8")
        self.shard_count = 0
        logger.info(f"Opened new output shard: {self.current_shard_file.name}")

    async def write_success(self, record: dict):
        async with self.file_lock:
            self.current_shard_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.current_shard_fp.flush()
            self.shard_count += 1
            self.stats["success"] += 1
            if self.shard_count >= self.shard_size:
                self._rotate_shard()

    async def write_failure(self, item: QueueItem, reason: str, status_code: int = 0):
        async with self.file_lock:
            fail_record = {
                "doc_id": item.doc_id,
                "url": item.url,
                "host": item.host,
                "status_code": status_code,
                "reason": reason,
                "retries": item.retry_count,
                "timestamp": int(time.time()),
            }
            self.failed_fp.write(json.dumps(fail_record, ensure_ascii=False) + "\n")
            self.failed_fp.flush()
            self.stats["failed_perm"] += 1

    async def trigger_domain_cooldown(self, host: str, reason: str):
        now = time.monotonic()
        cur_until = self.domain_cooldown_until[host]
        # Debounce: if domain is already in active cooldown, do not stack additively
        if cur_until > now:
            return
        cooldown = self.cooldown_seconds
        self.domain_cooldown_until[host] = now + cooldown
        # Temporarily increase request interval for this domain
        self.domain_interval[host] = min(self.domain_interval[host] * 1.5, 3.0)
        self.stats["rate_limits_hit"] += 1
        logger.warning(
            f"[CircuitBreaker] {host} triggered {reason}! Pausing {host} for {cooldown:.1f}s. "
            f"(Interval adjusted to {self.domain_interval[host]:.2f}s)"
        )

    async def schedule_retry(self, item: QueueItem, delay: float, reason: str, count_as_attempt: bool = True) -> bool:
        """Schedule an item for retry. Returns True if queued, False if permanently failed."""
        if count_as_attempt:
            item.retry_count += 1
            if item.retry_count > self.max_retries:
                logger.error(f"[MaxRetriesExceeded] ID {item.doc_id} ({item.url}) permanently failed: {reason}")
                await self.write_failure(item, reason=f"Max retries ({self.max_retries}) exceeded: {reason}")
                return False

        item.retry_after = time.monotonic() + delay
        async with self.retry_lock:
            self.retry_list.append(item)
            self.stats["retries_queued"] += 1
        return True

    async def fetch_worker(self, worker_id: int, client: httpx.AsyncClient, pbar: tqdm):
        while True:
            item = await self.work_queue.get()
            if item is None:
                # Sentinel to terminate worker
                self.work_queue.task_done()
                break

            host = item.host
            url = item.url
            doc_id = item.doc_id

            # 1. Check if domain is under cooldown
            now = time.monotonic()
            if now < self.domain_cooldown_until[host]:
                cooldown_remaining = self.domain_cooldown_until[host] - now
                # Re-queue with a delay without exhausting URL retry count
                await self.schedule_retry(item, delay=cooldown_remaining + 0.5, reason="Domain in cooldown", count_as_attempt=False)
                self.work_queue.task_done()
                pbar.set_postfix(
                    ok=self.stats["success"],
                    retry=len(self.retry_list),
                    fail=self.stats["failed_perm"],
                    bans=self.stats["rate_limits_hit"],
                )
                await asyncio.sleep(0.05)
                continue

            # 2. Acquire domain semaphore and respect pacing
            async with self.domain_sems[host]:
                now = time.monotonic()
                elapsed = now - self.domain_last_time[host]
                interval = self.domain_interval[host]
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
                self.domain_last_time[host] = time.monotonic()

                headers = {
                    "User-Agent": random.choice(USER_AGENTS),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "vi,zh-CN,zh;q=0.9,en;q=0.8",
                }

                try:
                    target_url = item.fetch_url or normalize_target_url(url)
                    r = await client.get(target_url, headers=headers, follow_redirects=True)

                    # Check for Lao Dong JS cookie challenge
                    if "document.cookie=" in r.text and len(r.text) < 500:
                        m = re.search(r'document\.cookie="([^"]+)"', r.text)
                        if m:
                            cookie_str = m.group(1)
                            k, v = cookie_str.split(";")[0].split("=", 1)
                            client.cookies.set(k.strip(), v.strip(), domain=host)
                            r = await client.get(url, headers=headers, follow_redirects=True)

                    if r.status_code == 200 and len(r.content) > 200:
                        raw_html = r.text
                        ct = r.headers.get("content-type", "").lower()
                        if "charset=gb" in ct or "gbk" in raw_html[:1000].lower() or "gb2312" in raw_html[:1000].lower():
                            try:
                                raw_html = r.content.decode("gb18030", errors="ignore")
                            except Exception:
                                pass

                        extracted = extract_article_structured(raw_html, url) or {}
                        text = extracted.get("text", "")
                        title = extracted.get("title", "")
                        if len(text.strip()) >= 50:
                            lang = detect_language(text)
                            record = {
                                "doc_id": doc_id,
                                "url": url,
                                "title": title,
                                "text": text,
                                "lang": lang,
                                "domain": host,
                                "char_count": len(text),
                                "status": "success",
                            }
                            await self.write_success(record)
                            pbar.update(1)
                            # Slightly recover domain interval on repeated success
                            if self.domain_interval[host] > self.base_domain_interval:
                                self.domain_interval[host] = max(
                                    self.base_domain_interval, self.domain_interval[host] * 0.95
                                )
                        else:
                            await self.write_failure(item, reason="Empty text extracted (<50 chars)", status_code=200)
                            pbar.update(1)

                    elif r.status_code in (429, 403):
                        # Domain rate limited or Cloudflare blocked!
                        await self.trigger_domain_cooldown(host, f"HTTP {r.status_code}")
                        if not await self.schedule_retry(item, delay=self.cooldown_seconds, reason=f"HTTP {r.status_code}"):
                            pbar.update(1)

                    elif r.status_code in (500, 502, 503, 504):
                        # Temporary server error: retry in 15 seconds
                        if not await self.schedule_retry(item, delay=15.0, reason=f"HTTP {r.status_code}"):
                            pbar.update(1)

                    elif r.status_code in (404, 410):
                        # Permanent missing page: do not retry
                        await self.write_failure(item, reason=f"HTTP {r.status_code}", status_code=r.status_code)
                        pbar.update(1)

                    else:
                        if not await self.schedule_retry(item, delay=10.0, reason=f"HTTP {r.status_code}"):
                            pbar.update(1)

                except (httpx.TimeoutException, httpx.ConnectTimeout):
                    if not await self.schedule_retry(item, delay=10.0, reason="Timeout"):
                        pbar.update(1)
                except (httpx.ConnectError, httpx.ReadError) as e:
                    if not await self.schedule_retry(item, delay=15.0, reason=f"Network error: {type(e).__name__}"):
                        pbar.update(1)
                except Exception as e:
                    if not await self.schedule_retry(item, delay=10.0, reason=f"Exception: {e}"):
                        pbar.update(1)

            self.work_queue.task_done()
            pbar.set_postfix(
                ok=self.stats["success"],
                retry=len(self.retry_list),
                fail=self.stats["failed_perm"],
                bans=self.stats["rate_limits_hit"],
            )

    async def retry_feeder(self, stop_event: asyncio.Event):
        """Background task: periodically checks retry_list and re-injects ready items."""
        while not stop_event.is_set():
            await asyncio.sleep(2.0)
            now = time.monotonic()
            ready_items = []
            async with self.retry_lock:
                remaining = []
                for item in self.retry_list:
                    if item.retry_after <= now:
                        ready_items.append(item)
                    else:
                        remaining.append(item)
                self.retry_list = remaining

            for item in ready_items:
                await self.work_queue.put(item)

    async def close(self):
        async with self.file_lock:
            if self.current_shard_fp:
                try:
                    self.current_shard_fp.close()
                except Exception:
                    pass
            if self.failed_fp:
                try:
                    self.failed_fp.close()
                except Exception:
                    pass


async def run_adaptive_crawler(args):
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
        logger.info(f"Filtered for hosts {target_hosts}: {len(rows):,} URLs matching target filter.")

    if getattr(args, "exclude_hosts", None):
        ex_hosts = set(args.exclude_hosts)
        rows = [r for r in rows if r["host"] not in ex_hosts]
        logger.info(f"Excluded hosts {ex_hosts}: {len(rows):,} URLs remaining.")

    # 2. Check already crawled IDs and permanently failed IDs
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
        logger.info("No URLs to crawl! All requested URLs are already crawled.")
        return

    # 3. Setup Manager and Worker Pool
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

    # Feed initial URLs to queue (interleaved across domains for maximum concurrency and politeness)
    queue_items = []
    for r in todo:
        fetch_u = normalize_target_url(r["url"])
        fetch_h = urlparse(fetch_u).netloc or r["host"]
        queue_items.append(QueueItem(doc_id=r["id"], url=r["url"], host=fetch_h, fetch_url=fetch_u))
    if getattr(args, "interleave", True):
        queue_items = interleave_by_host(queue_items)

    for it in queue_items:
        await manager.work_queue.put(it)

    total_tasks = len(todo)
    pbar = tqdm(total=total_tasks, desc="Crawling", unit="url", dynamic_ncols=True)

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

        # Wait until initial work queue is empty
        await manager.work_queue.join()

        # Wait for any in-flight retries to be resolved
        while True:
            async with manager.retry_lock:
                active_retries = len(manager.retry_list)
            if active_retries == 0 and manager.work_queue.empty():
                break
            await asyncio.sleep(1.0)
            await manager.work_queue.join()

        # Stop workers
        for _ in range(args.concurrency):
            await manager.work_queue.put(None)
        await asyncio.gather(*workers)

    stop_retry_event.set()
    await feeder_task
    await manager.close()
    pbar.close()

    logger.info("=" * 60)
    logger.info(f"Crawl Completed Successfully!")
    logger.info(f" - Saved Articles:       {manager.stats['success']:,}")
    logger.info(f" - Permanent Failures:   {manager.stats['failed_perm']:,} (logged in failed_urls.jsonl)")
    logger.info(f" - Total Retries Queued: {manager.stats['retries_queued']:,}")
    logger.info(f" - Rate Limit Pauses:    {manager.stats['rate_limits_hit']:,}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        help="Directory to save crawled shards.",
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=None,
        help="List of specific hosts to crawl (e.g. nhathuoclongchau.com.vn suckhoedoisong.vn).",
    )
    parser.add_argument(
        "--exclude-hosts",
        nargs="+",
        default=None,
        help="List of hosts to exclude from crawl (e.g. nhathuoclongchau.com.vn).",
    )
    parser.add_argument(
        "--interleave",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interleave URLs across domains in round-robin fashion for maximum throughput (default: True).",
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
        default=20,
        help="Total concurrent HTTP requests across all domains (default: 20).",
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
        help="Base interval between requests to the same domain in seconds (default: 0.25s).",
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

    args = parser.parse_args()
    asyncio.run(run_adaptive_crawler(args))


if __name__ == "__main__":
    main()
