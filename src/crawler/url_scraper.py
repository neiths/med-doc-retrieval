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
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        timeout_seconds: int = 15,
        max_retries: int = 3,
        concurrency: int = 10,
    ):
        self.headers = {"User-Agent": user_agent}
        self.timeout = httpx.Timeout(timeout_seconds)
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(concurrency)

    async def fetch_url(self, client: httpx.AsyncClient, url: str) -> str | None:
        """Fetch raw HTML with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.get(url, headers=self.headers, follow_redirects=True)
                if response.status_code == 200:
                    return response.text
                elif response.status_code in [404, 410]:
                    logger.warning(f"URL returned {response.status_code}, skipping: {url}")
                    return None
            except Exception as e:
                if attempt == self.max_retries:
                    logger.warning(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(1.0 * attempt)
        return None

    def extract_text(self, html: str, fallback_url: str = "") -> dict[str, str]:
        """Extract clean text and title using trafilatura with BeautifulSoup fallback."""
        try:
            # Trafilatura is state-of-the-art for boilerplate removal and main text extraction
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

        # Fallback to BeautifulSoup
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
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
            return {
                "doc_id": doc_id,
                "url": url,
                "title": content.get("title", ""),
                "text": content.get("text", ""),
                "status": "success" if content.get("text") else "empty",
            }

    async def scrape_urls_jsonl(
        self,
        input_file: Path | str,
        output_file: Path | str,
    ) -> list[dict[str, Any]]:
        """Scrapes all URLs listed in input JSONL and saves results to output JSONL."""
        in_path = Path(input_file)
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        items = []
        with open(in_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))

        logger.info(f"Loaded {len(items)} items from {in_path} to scrape.")

        results = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self.process_item(client, item) for item in items]
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Scraping URLs"):
                res = await future
                if res:
                    results.append(res)

        # Write results
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)

        success_count = sum(1 for r in results if r.get("status") == "success")
        logger.info(
            f"Scraping completed: {success_count}/{len(items)} articles extracted successfully -> {out_path}"
        )
        return results
