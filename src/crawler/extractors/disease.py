"""Specialized extractor for Disease Encyclopedias and Pharmacy Portals.

Handles:
- nhathuoclongchau.com.vn
- suckhoedoisong.vn
- vinmec.com
- medlatec.vn
- cnkang.com

In accordance with Mentor Day 1 guidance:
Preserves section headers (H2/H3) alongside subsequent paragraphs so chunking
can segment along semantic disease aspects (Symptoms, Causes, Diagnosis, Treatment).
"""

import re
from typing import Any
from bs4 import BeautifulSoup

from src.crawler.extractors.base import BaseExtractor
from src.ingestion.cleaner import clean_boilerplate, fix_mojibake


class DiseaseArticleExtractor(BaseExtractor):
    """Extracts structured medical encyclopedia and clinical disease guides."""

    doc_type: str = "disease_encyclopedia"

    def extract(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        self.clean_common_noise(soup)
        domain = self.get_domain(url)

        if "longchau" in domain:
            return self._extract_longchau(soup, url)
        elif "suckhoedoisong" in domain:
            return self._extract_suckhoedoisong(soup, url)
        elif "vinmec" in domain:
            return self._extract_vinmec(soup, url)
        elif "medlatec" in domain:
            return self._extract_medlatec(soup, url)
        elif "cnkang" in domain:
            return self._extract_cnkang(soup, url)
        else:
            return self._extract_generic_disease(soup, url)

    def _extract_longchau(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""

        container = soup.find("article") or soup.find(class_=re.compile(r"content-post|article-body|detail-content"))
        if not container:
            return None

        # Decompose noisy pharmacist cards, rating stars, and product banners
        for noisy in container.find_all(class_=re.compile(r"(author|pharmacist|rating|related|product-card|banner|breadcrumb|tag-list)", re.I)):
            noisy.decompose()

        parts = [f"# {title}"] if title else []
        headings = []
        current_section = []

        for elem in container.children:
            if not getattr(elem, "name", None):
                continue
            tag_name = elem.name.lower()
            if tag_name in ["h2", "h3", "h4"]:
                if current_section:
                    parts.append("\n".join(current_section))
                    current_section = []
                h_text = elem.get_text(strip=True)
                if h_text:
                    prefix = "##" if tag_name == "h2" else "###"
                    parts.append(f"{prefix} {h_text}")
                    headings.append(h_text)
            elif tag_name in ["p", "ul", "ol", "div"]:
                text_val = elem.get_text(separator="\n", strip=True)
                if text_val:
                    current_section.append(text_val)

        if current_section:
            parts.append("\n".join(current_section))

        full_text = "\n\n".join(parts)
        if len(full_text) < 80:
            return None

        return {
            "title": title,
            "text": clean_boilerplate(full_text),
            "doc_type": self.doc_type,
            "headings": headings,
            "metadata": {"source": "nhathuoclongchau"},
        }

    def _extract_suckhoedoisong(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""

        container = soup.find(class_=re.compile(r"detail-content|fck|article-content")) or soup.find("article")
        if not container:
            return None

        for noisy in container.find_all(class_=re.compile(r"(relate|hotline|box-share|adv|banner|author|tag)", re.I)):
            noisy.decompose()

        parts = [f"# {title}"] if title else []
        headings = []

        for elem in container.find_all(["h2", "h3", "p"]):
            tag = elem.name.lower()
            t = elem.get_text(strip=True)
            if not t:
                continue
            if tag == "h2":
                parts.append(f"## {t}")
                headings.append(t)
            elif tag == "h3":
                parts.append(f"### {t}")
                headings.append(t)
            else:
                parts.append(t)

        full_text = "\n\n".join(parts)
        if len(full_text) < 80:
            return None

        return {
            "title": title,
            "text": clean_boilerplate(full_text),
            "doc_type": self.doc_type,
            "headings": headings,
            "metadata": {"source": "suckhoedoisong"},
        }

    def _extract_vinmec(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        # Check if page is language redirect notification (Mentor Day 1 example)
        page_text = soup.get_text()
        if "This content is not available in English" in page_text or "Yêu cầu tiếp tục để đọc tiếng Việt" in page_text:
            return None

        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""

        container = soup.find(class_=re.compile(r"article-body|content-detail|detail-content")) or soup.find("article")
        if not container:
            return None

        for noisy in container.find_all(class_=re.compile(r"(booking|doctor|contact|banner|social|relate)", re.I)):
            noisy.decompose()

        parts = [f"# {title}"] if title else []
        headings = []
        for elem in container.find_all(["h2", "h3", "p"]):
            t = elem.get_text(strip=True)
            if not t:
                continue
            tag = elem.name.lower()
            if tag in ["h2", "h3"]:
                parts.append(f"## {t}")
                headings.append(t)
            else:
                parts.append(t)

        full_text = "\n\n".join(parts)
        if len(full_text) < 80:
            return None

        return {
            "title": title,
            "text": clean_boilerplate(full_text),
            "doc_type": self.doc_type,
            "headings": headings,
            "metadata": {"source": "vinmec"},
        }

    def _extract_medlatec(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""

        container = soup.find(class_=re.compile(r"content-detail|article-body|detail-news")) or soup.find("article")
        if not container:
            return None

        for noisy in container.find_all(class_=re.compile(r"(booking|doctor|banner|social|relate|hotline)", re.I)):
            noisy.decompose()

        parts = [f"# {title}"] if title else []
        headings = []
        for elem in container.find_all(["h2", "h3", "p"]):
            t = elem.get_text(strip=True)
            if not t:
                continue
            tag = elem.name.lower()
            if tag in ["h2", "h3"]:
                parts.append(f"## {t}")
                headings.append(t)
            else:
                parts.append(t)

        full_text = "\n\n".join(parts)
        if len(full_text) < 80:
            return None

        return {
            "title": title,
            "text": clean_boilerplate(full_text),
            "doc_type": self.doc_type,
            "headings": headings,
            "metadata": {"source": "medlatec"},
        }

    def _extract_cnkang(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""

        container = soup.find(class_=re.compile(r"disease_content|art_content|content")) or soup.find("article")
        if not container:
            return None

        body = container.get_text(separator="\n", strip=True)
        if len(body) < 50:
            return None

        full_text = f"# {title}\n\n{body}" if title else body
        return {
            "title": title,
            "text": fix_mojibake(clean_boilerplate(full_text)),
            "doc_type": self.doc_type,
            "headings": [],
            "metadata": {"source": "cnkang"},
        }

    def _extract_generic_disease(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""
        body = soup.get_text(separator="\n", strip=True)
        if len(body) < 80:
            return None
        return {
            "title": title,
            "text": clean_boilerplate(body),
            "doc_type": self.doc_type,
            "headings": [],
            "metadata": {"source": "generic_disease"},
        }
