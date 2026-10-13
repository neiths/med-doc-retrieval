"""Unified Extractor registry and dispatcher for medical documents."""

from typing import Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from src.crawler.extractors.base import BaseExtractor
from src.crawler.extractors.qa import QAExtractor
from src.crawler.extractors.disease import DiseaseArticleExtractor
from src.crawler.extractors.news import NewsArticleExtractor
from src.crawler.extractors.generic import GenericExtractor

_QA_EXTRACTOR = QAExtractor()
_DISEASE_EXTRACTOR = DiseaseArticleExtractor()
_NEWS_EXTRACTOR = NewsArticleExtractor()
_GENERIC_EXTRACTOR = GenericExtractor()

# Domain routing mapping
_QA_DOMAINS = {
    "120ask.com",
    "39.net",
    "familydoctor.com.cn",
}

_DISEASE_DOMAINS = {
    "nhathuoclongchau.com.vn",
    "suckhoedoisong.vn",
    "vinmec.com",
    "medlatec.vn",
    "cnkang.com",
    "hellobacsi.com",
    "youmed.vn",
    "tamanhhospital.vn",
    "zhongyibaodian.net",
    "zysjonline.com",
    "a-hospital.com",
}

_NEWS_DOMAINS = {
    "suckhoecongdongonline.vn",
    "thanhnien.vn",
    "laodong.vn",
    "vietnamnet.vn",
    "baonghean.vn",
    "baohaiphong.vn",
    "baodanang.vn",
    "baocantho.com.vn",
    "baoangiang.com.vn",
    "tuoitre.vn",
    "vov.vn",
    "nhandan.vn",
}


def get_extractor(url: str) -> BaseExtractor:
    """Selects the optimal extractor based on domain and URL structure."""
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        domain = ""

    if any(d in domain for d in _QA_DOMAINS):
        return _QA_EXTRACTOR

    if any(d in domain for d in _DISEASE_DOMAINS):
        return _DISEASE_EXTRACTOR

    if any(d in domain for d in _NEWS_DOMAINS):
        return _NEWS_EXTRACTOR

    return _GENERIC_EXTRACTOR


def extract_article(html: str, url: str) -> dict[str, Any] | None:
    """Convenience function: parses HTML with the appropriate extractor."""
    if not html:
        return None

    extractor = get_extractor(url)
    soup = BeautifulSoup(html, "html.parser")
    result = extractor.extract(soup, url)

    if not result:
        # Fallback to generic if specialized extractor produced nothing
        result = _GENERIC_EXTRACTOR.extract(soup, url)

    return result


extract_article_structured = extract_article


__all__ = [
    "BaseExtractor",
    "QAExtractor",
    "DiseaseArticleExtractor",
    "NewsArticleExtractor",
    "GenericExtractor",
    "get_extractor",
    "extract_article",
    "extract_article_structured",
]
