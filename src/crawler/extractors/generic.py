"""Generic fallback extractor using Trafilatura in Markdown mode."""

from typing import Any
from bs4 import BeautifulSoup
import trafilatura

from src.crawler.extractors.base import BaseExtractor
from src.ingestion.cleaner import clean_boilerplate, fix_mojibake


class GenericExtractor(BaseExtractor):
    """Fallback extractor using Trafilatura with Markdown heading preservation."""

    doc_type: str = "generic"

    def extract(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        raw_html = str(soup)

        try:
            extracted = trafilatura.extract(
                raw_html,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                no_fallback=False,
                output_format="txt",
            )
            title = ""
            meta = trafilatura.extract_metadata(raw_html)
            if meta and meta.title:
                title = meta.title.strip()
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)
            elif soup.title and soup.title.string:
                title = soup.title.string.strip()

            if extracted and len(extracted.strip()) > 50:
                text = fix_mojibake(clean_boilerplate(extracted.strip()))
                return {
                    "title": title,
                    "text": text,
                    "doc_type": self.doc_type,
                    "headings": [],
                    "metadata": {"source": "trafilatura"},
                }
        except Exception:
            pass

        # Final soup fallback
        self.clean_common_noise(soup)
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else (soup.title.string.strip() if soup.title and soup.title.string else "")
        body_text = soup.get_text(separator="\n", strip=True)
        if len(body_text) < 50:
            return None

        return {
            "title": title,
            "text": fix_mojibake(clean_boilerplate(body_text)),
            "doc_type": self.doc_type,
            "headings": [],
            "metadata": {"source": "soup_fallback"},
        }
