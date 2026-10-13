"""Specialized extractor for Medical News portals (Sức Khỏe Cộng Đồng, Thanh Niên, Lao Động).

Focuses on:
- Title (H1)
- Sapo / Lead paragraph (crucial concise summary of the health article)
- Body paragraphs
- Decomposes editor notes, news carousels, related articles, and comment boxes.
"""

import re
from typing import Any
from bs4 import BeautifulSoup

from src.crawler.extractors.base import BaseExtractor
from src.ingestion.cleaner import clean_boilerplate


class NewsArticleExtractor(BaseExtractor):
    """Extracts news articles preserving lead/sapo and main report body."""

    doc_type: str = "news"

    def extract(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        self.clean_common_noise(soup)

        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""

        # Locate lead summary / sapo
        sapo_box = soup.find(class_=re.compile(r"sapo|summary|lead|description|short-intro"))
        sapo = sapo_box.get_text(strip=True) if sapo_box else ""

        # Locate body container
        container = (
            soup.find(class_=re.compile(r"detail-content|content-detail|article-body|fck_detail|main-content"))
            or soup.find("article")
        )
        if not container:
            return None

        # Clean noise widgets common to news portals
        for noisy in container.find_all(class_=re.compile(r"(relate|box-share|adv|banner|author|source|tags|comment|social|sticky)", re.I)):
            noisy.decompose()

        parts = []
        if title:
            parts.append(f"# {title}")
        if sapo:
            parts.append(f"**Tóm tắt**: {sapo}")

        paragraphs = []
        for p in container.find_all(["p", "h2", "h3"]):
            t = p.get_text(strip=True)
            if not t or len(t) < 15:
                continue
            if p.name in ["h2", "h3"]:
                paragraphs.append(f"## {t}")
            else:
                paragraphs.append(t)

        if not paragraphs:
            body = container.get_text(separator="\n", strip=True)
            if len(body) > 50:
                paragraphs.append(body)

        parts.append("\n\n".join(paragraphs))
        full_text = "\n\n".join(parts)

        if len(full_text) < 80:
            return None

        return {
            "title": title,
            "text": clean_boilerplate(full_text),
            "doc_type": self.doc_type,
            "headings": [h.get_text(strip=True) for h in container.find_all(["h2", "h3"]) if h.get_text(strip=True)],
            "metadata": {"sapo": sapo},
        }
