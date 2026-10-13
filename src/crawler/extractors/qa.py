"""Specialized extractor for Medical Q&A portals (120ask, 39.net, FamilyDoctor).

In accordance with Mentor Day 1 guidance:
Preserves Question and Doctor Answer pairs together so retrieval context is not lost.
"""

import re
from typing import Any
from bs4 import BeautifulSoup

from src.crawler.extractors.base import BaseExtractor
from src.ingestion.cleaner import clean_boilerplate, fix_mojibake


class QAExtractor(BaseExtractor):
    """Extracts medical question-and-answer pairs with doctor replies."""

    doc_type: str = "qa"

    def extract(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        self.clean_common_noise(soup)
        domain = self.get_domain(url)

        if "120ask.com" in domain:
            return self._extract_120ask(soup, url)
        elif "39.net" in domain:
            return self._extract_39net(soup, url)
        elif "familydoctor" in domain:
            return self._extract_familydoctor(soup, url)
        else:
            return self._extract_generic_qa(soup, url)

    def _extract_120ask(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""

        # Question description
        q_box = (
            soup.find(class_="b_askbox")
            or soup.find(class_="crazy_new")
            or soup.find(class_="ms_content")
        )
        q_text = ""
        if q_box:
            for noisy in q_box.find_all(class_=re.compile(r"(b_askbtn|b_askti|complaint|b_user)", re.I)):
                noisy.decompose()
            q_text = q_box.get_text(separator="\n", strip=True)

        # Doctor answers
        answers = []
        ans_boxes = soup.find_all(class_=re.compile(r"b_answerli|b_anscont|b_ansbox"))
        for ab in ans_boxes:
            txt_box = ab.find(class_=re.compile(r"b_anscont|b_cont")) or ab
            ans_soup = BeautifulSoup(str(txt_box), "html.parser")
            for tag in ans_soup(["script", "style", "a", "form", "iframe", "button"]):
                tag.decompose()
            for noisy in ans_soup.find_all(class_=re.compile(r"(complaint|b_answerti|doctor_detail|good_list)", re.I)):
                noisy.decompose()
            ans_str = ans_soup.get_text(separator="\n", strip=True)
            if len(ans_str) > 15:
                answers.append(ans_str)

        if not q_text and not answers and not title:
            return None

        # Build clean markdown
        parts = []
        headings = []
        if title:
            parts.append(f"# {title}")
        if q_text:
            headings.append("患者提问")
            parts.append(f"## 患者提问\n{q_text}")
        for idx, ans in enumerate(answers, 1):
            h = f"医生回答 {idx}"
            headings.append(h)
            parts.append(f"## {h}\n{ans}")

        text = fix_mojibake("\n\n".join(parts))
        qa_pairs = [{"question": q_text, "answer": ans} for ans in answers]

        return {
            "title": title,
            "text": clean_boilerplate(text),
            "doc_type": self.doc_type,
            "headings": headings,
            "qa_pairs": qa_pairs,
            "metadata": {
                "source": "120ask",
                "answer_count": len(answers),
            },
        }

    def _extract_39net(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        h1 = soup.find("h1") or soup.find(class_=re.compile(r"ask_tit|txt_title"))
        title = h1.get_text(strip=True) if h1 else ""

        q_box = soup.find(class_=re.compile(r"txt_ms|txt_cure|ask_content|sub_cont"))
        q_text = q_box.get_text(separator="\n", strip=True) if q_box else ""

        answers = []
        ans_boxes = soup.find_all(class_=re.compile(r"show_all_ans|doc_ans|ans_content|b_anscont"))
        for ab in ans_boxes:
            ans_soup = BeautifulSoup(str(ab), "html.parser")
            for tag in ans_soup(["script", "style", "a", "form"]):
                tag.decompose()
            ans_str = ans_soup.get_text(separator="\n", strip=True)
            if len(ans_str) > 15:
                answers.append(ans_str)

        if not q_text and not answers:
            return None

        parts = []
        headings = []
        if title:
            parts.append(f"# {title}")
        if q_text:
            headings.append("患者提问")
            parts.append(f"## 患者提问\n{q_text}")
        for idx, ans in enumerate(answers, 1):
            h = f"医生回答 {idx}"
            headings.append(h)
            parts.append(f"## {h}\n{ans}")

        text = fix_mojibake("\n\n".join(parts))
        qa_pairs = [{"question": q_text, "answer": ans} for ans in answers]

        return {
            "title": title,
            "text": clean_boilerplate(text),
            "doc_type": self.doc_type,
            "headings": headings,
            "qa_pairs": qa_pairs,
            "metadata": {
                "source": "ask.39.net",
                "answer_count": len(answers),
            },
        }

    def _extract_familydoctor(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""

        q_box = soup.find(class_=re.compile(r"question-detail|ask-content|q-desc"))
        q_text = q_box.get_text(separator="\n", strip=True) if q_box else ""

        answers = []
        for ab in soup.find_all(class_=re.compile(r"answer-content|reply-content|doc-reply")):
            ans_str = ab.get_text(separator="\n", strip=True)
            if len(ans_str) > 15:
                answers.append(ans_str)

        if not q_text and not answers and not title:
            return None

        parts = []
        headings = []
        if title:
            parts.append(f"# {title}")
        if q_text:
            headings.append("患者提问")
            parts.append(f"## 患者提问\n{q_text}")
        for idx, ans in enumerate(answers, 1):
            h = f"医生回答 {idx}"
            headings.append(h)
            parts.append(f"## {h}\n{ans}")

        text = fix_mojibake("\n\n".join(parts))
        return {
            "title": title,
            "text": clean_boilerplate(text),
            "doc_type": self.doc_type,
            "headings": headings,
            "qa_pairs": [{"question": q_text, "answer": a} for a in answers],
            "metadata": {"source": "familydoctor", "answer_count": len(answers)},
        }

    def _extract_generic_qa(self, soup: BeautifulSoup, url: str) -> dict[str, Any] | None:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""
        body = soup.get_text(separator="\n", strip=True)
        if len(body) < 50:
            return None
        return {
            "title": title,
            "text": clean_boilerplate(body),
            "doc_type": self.doc_type,
            "headings": [],
            "qa_pairs": [],
            "metadata": {"source": "generic_qa"},
        }
