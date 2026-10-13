"""Base class and common types for medical document extractors."""

from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from src.ingestion.cleaner import clean_boilerplate, fix_mojibake, normalize_text


class BaseExtractor(ABC):
    """Abstract base extractor interface for all document archetypes."""

    doc_type: str = "generic"

    @abstractmethod
    def extract(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        """Extract structured text and metadata from BeautifulSoup DOM.

        Returns:
            dict containing:
                - title: str
                - text: str (Markdown formatted with headings)
                - doc_type: str ('qa', 'disease_encyclopedia', 'news', 'generic')
                - headings: list[str] (list of H2/H3 section titles)
                - qa_pairs: list[dict[str, str]] (optional for Q&A sites)
                - metadata: dict[str, Any] (domain, author, etc.)
            or None if extraction failed / page is invalid.
        """
        pass

    def clean_common_noise(self, soup: BeautifulSoup):
        """Removes script, style, ads, iframe, and svg elements from DOM."""
        for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "button"]):
            tag.decompose()

    def get_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower() or "unknown"
        except Exception:
            return "unknown"
