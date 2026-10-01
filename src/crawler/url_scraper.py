"""Asynchronous web scraper for Vietnamese and Chinese medical articles provided by BTC."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import trafilatura
from bs4 import BeautifulSoup
from loguru import logger
from tqdm.asyncio import tqdm


class ArticleScraper:
    """Fetches and parses medical web articles given lists of URLs."""

    def __init__(
        self,
        user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        timeout_seconds: int = 15,
        max_retries: int = 3,
        concurrency: int = 10,
    ):
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,zh-CN;q=0.8,zh;q=0.7,en-US;q=0.6,en;q=0.5",
            "Upgrade-Insecure-Requests": "1",
        }
        self.timeout = httpx.Timeout(timeout_seconds)
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(concurrency)

    async def fetch_url(self, client: httpx.AsyncClient, url: str) -> str | None:
        """Fetch raw HTML with retry logic and rate limit backoff."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.get(url, headers=self.headers, follow_redirects=True)
                if response.status_code == 200:
                    content_bytes = response.content
                    lower_head = content_bytes[:2048].lower()
                    if b"charset=gb" in lower_head or b"charset=\"gb" in lower_head or b"charset='gb" in lower_head:
                        try:
                            return content_bytes.decode("gb18030", errors="replace")
                        except Exception:
                            pass
                    try:
                        return content_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            return content_bytes.decode("gb18030", errors="replace")
                        except Exception:
                            return response.text
                elif response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "3")
                    wait_time = float(retry_after) if retry_after.isdigit() else 3.0
                    logger.warning(f"Rate limited (429) on {url}. Backing off for {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                elif response.status_code in [404, 410]:
                    logger.warning(f"URL returned {response.status_code}, skipping: {url}")
                    return None
                elif response.status_code in [500, 502, 503, 504]:
                    await asyncio.sleep(1.0 * attempt)
                    continue
            except Exception as e:
                if attempt == self.max_retries:
                    logger.warning(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(1.0 * attempt)
        return None

    def extract_text(self, html: str, fallback_url: str = "") -> dict[str, str]:
        """Extract clean text and title using trafilatura with BeautifulSoup fallback."""
        if not html:
            return {"title": "", "text": ""}

        # 1. Primary: Trafilatura (state-of-the-art for boilerplate & ads removal)
        try:
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                output_format="txt",
            )
            title = trafilatura.extract_metadata(html).title if html else ""
            if extracted and len(extracted.strip()) > 30:
                return {"title": title or "", "text": extracted.strip()}
        except Exception:
            pass

        # 2. Fallback: BeautifulSoup with aggressive boilerplate decomposition
        try:
            import re
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup([
                "script", "style", "nav", "footer", "header", "noscript",
                "aside", "form", "iframe", "svg"
            ]):
                tag.decompose()

            # Remove noise classes common in VN/ZH news portals
            for noisy in soup.find_all(class_=re.compile(r"(advertisement|banner|sidebar|social-share|comment|newsletter|footer-link)", re.I)):
                noisy.decompose()

            # Extract title
            title = ""
            h1 = soup.find("h1")
            if h1 and h1.get_text(strip=True):
                title = h1.get_text(strip=True)
            elif soup.title and soup.title.string:
                title = soup.title.string.strip()

            body_text = soup.get_text(separator="\n", strip=True)
            return {"title": title, "text": body_text}
        except Exception as e:
            logger.debug(f"Parsing failed for {fallback_url}: {e}")
            return {"title": "", "text": ""}

    async def process_item(
        self, client: httpx.AsyncClient, item: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Fetch and extract a single URL item."""
        doc_id = str(item.get("id"))
        url = item.get("url")
        if not url:
            return None

        async with self.semaphore:
            html = await self.fetch_url(client, url)
            if not html:
                return {"doc_id": doc_id, "url": url, "title": "", "text": "", "status": "failed"}

            content = self.extract_text(html, fallback_url=url)
            raw_text = content.get("text", "")

            # Normalize text and detect language
            from src.ingestion.cleaner import detect_language, normalize_text

            cleaned_text = normalize_text(raw_text) if raw_text else ""
            lang = detect_language(cleaned_text) if cleaned_text else "unknown"

            return {
                "doc_id": doc_id,
                "url": url,
                "title": content.get("title", ""),
                "text": cleaned_text,
                "lang": lang,
                "status": "success" if cleaned_text else "empty",
            }

    async def scrape_urls_jsonl(
        self,
        input_file: Path | str,
        output_file: Path | str,
        resume: bool = True,
    ) -> list[dict[str, Any]]:
        """Scrapes all URLs listed in input JSONL with resumable checkpointing and incremental flushing."""
        in_path = Path(input_file)
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        items = []
        if in_path.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(in_path, columns=["id", "url"])
            items = table.to_pylist()
        else:
            with open(in_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))

        # Check existing progress if resume is enabled
        existing_results: list[dict[str, Any]] = []
        existing_ids: set[str] = set()
        if resume and out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            existing_results.append(rec)
                            if "doc_id" in rec:
                                existing_ids.add(str(rec["doc_id"]))
                        except Exception:
                            pass

        items_to_scrape = [it for it in items if str(it.get("id")) not in existing_ids]

        if existing_ids:
            logger.info(
                f"Resuming crawler: {len(existing_ids)}/{len(items)} items already scraped in {out_path}. "
                f"Remaining to scrape: {len(items_to_scrape)}."
            )
        else:
            logger.info(f"Loaded {len(items)} items from {in_path} to scrape.")

        if not items_to_scrape:
            logger.info(f"All {len(items)} articles already present in {out_path}. Skipping crawl.")
            return existing_results

        new_results = []
        # Append mode ensures existing data is preserved, flush ensures no data loss on interrupt
        mode = "a" if (resume and out_path.exists()) else "w"
        with open(out_path, mode, encoding="utf-8") as out_f:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                tasks = [self.process_item(client, item) for item in items_to_scrape]
                for future in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc="Scraping URLs (resumable)",
                ):
                    res = await future
                    if res:
                        new_results.append(res)
                        out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        out_f.flush()

        all_results = existing_results + new_results
        success_count = sum(1 for r in all_results if r.get("status") == "success")
        logger.info(
            f"Scraping completed: {success_count}/{len(items)} articles successfully available -> {out_path}"
        )
        return all_results

