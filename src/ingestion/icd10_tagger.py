"""Multilingual ICD-10 Ontology Tagger for Medical Documents and Queries.

Based on Mentor Day 1 guidance:
Maps documents across Vietnamese, Chinese, and English into the 22 WHO ICD-10 Chapters
(A00-B99, C00-D48, ..., N00-N99, etc.) to enable topic routing, domain clustering,
and cross-lingual semantic alignment.
"""

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ICD10TagResult:
    primary_chapter: str | None = None
    primary_chapter_code: str | None = None  # e.g. "XIV" or "N00-N99"
    matched_chapters: list[tuple[str, int]] = field(default_factory=list)
    matched_codes: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_chapter": self.primary_chapter,
            "primary_chapter_code": self.primary_chapter_code,
            "matched_chapters": self.matched_chapters,
            "matched_codes": self.matched_codes,
            "matched_terms": self.matched_terms,
        }


STOPWORDS_MEDICAL = {
    "clinical", "diagnosis", "symptoms", "management", "severe", "acute", "chronic",
    "patient", "patients", "other", "specified", "without", "with", "complications",
    "unspecified", "syndrome", "disorder", "disease", "treatment", "therapy",
    "adult", "adults", "child", "children", "pediatric", "bệnh", "hội chứng", "triệu chứng",
    "điều trị", "chẩn đoán", "nguyên nhân", "cách", "dấu hiệu", "không", "các", "những",
    "stones", "calculi", "failure",
}


CLINICAL_CHAPTER_KEYWORDS = {
    "(XI) K00-K93: Bệnh hệ tiêu hóa": [
        "dạ dày", "thực quản", "tá tràng", "đại tràng", "tiêu hóa", "răng", "nướu", "nha chu",
        "gan", "men gan", "mật", "túi mật", "ruột", "trĩ", "táo bón", "viêm ruột", "nôn", "buồn nôn",
        "ợ chua", "trào ngược", "xung huyết", "loét", "hành tá tràng", "chảy máu chân răng",
    ],
    "(X) J00-J99: Bệnh hệ hô hấp": [
        "phổi", "hô hấp", "phế quản", "viêm phổi", "viêm phế quản", "hen suyễn", "bị ho", "ho khan", "ho có đờm", "họng",
        "viêm họng", "amidan", "xoang", "viêm xoang", "sổ mũi", "ngạt mũi", "khó thở", "đờm",
    ],
    "(IX) I00-I99: Bệnh hệ tuần hoàn": [
        "tim", "huyết áp", "tăng huyết áp", "đột quỵ", "tai biến", "mạch vành", "thiếu máu cơ tim",
        "nhồi máu", "suy tim", "xơ vữa", "rối loạn nhịp tim", "mạch máu",
    ],
    "(XIV) N00-N99: Bệnh hệ tiết niệu, sinh dục": [
        "thận", "suy thận", "sỏi thận", "tiết niệu", "bàng quang", "tiểu buốt", "tiểu rắt",
        "phụ khoa", "tử cung", "buồng trứng", "kinh nguyệt", "tinh hoàn", "tiền liệt tuyến", "bao quy đầu",
    ],
    "(XIII) M00-M99: Bệnh hệ cơ, xương, khớp": [
        "xương", "khớp", "cột sống", "thoái hóa", "thoát vị", "đĩa đệm", "đau lưng", "gãy xương",
        "gout", "viêm khớp", "loãng xương", "đau vai gáy", "đau cơ", "cơ bắp", "gân",
    ],
    "(XII) L00-L99: Bệnh của da và mô dưới da": [
        "ngứa", "mề đay", "mụn", "vảy nến", "chàm", "dị ứng da", "viêm da", "zona",
        "ghẻ", "nấm da", "sẹo lồi", "nổi mẩn",
    ],
    "(IV) E00-E90: Bệnh nội tiết, dinh dưỡng và chuyển hóa": [
        "tiểu đường", "đái tháo đường", "tuyến giáp", "bướu cổ", "mỡ máu", "cholesterol",
        "axit uric", "béo phì", "suy dinh dưỡng", "hạ đường huyết", "rối loạn chuyển hóa",
    ],
    "(II) C00-D48: Khối u / Ung thư": [
        "ung thư", "u ác tính", "u lành", "khối u", "carcinoma", "sarcoma", "polyp", "hạch", "di căn",
    ],
    "(VI) G00-G99: Bệnh hệ thần kinh": [
        "não", "thần kinh", "đau đầu", "chóng mặt", "mất ngủ", "động kinh", "liệt", "parkinson",
        "viêm màng não", "dây thần kinh",
    ],
    "(VII) H00-H59: Bệnh của mắt và phần phụ": [
        "mắt", "thị lực", "cận thị", "viêm kết mạc", "đau mắt", "giác mạc", "đục thủy tinh thể",
    ],
    "(VIII) H60-H95: Bệnh của tai và xương chũm": [
        "tai", "thính lực", "ù tai", "viêm tai", "chảy mủ tai", "nghe kém",
    ],
    "(V) F00-F99: Rối loạn tâm thần và hành vi": [
        "trầm cảm", "lo âu", "stress", "tâm thần", "hoang tưởng", "rối loạn cảm xúc", "ảo giác",
    ],
    "(XV) O00-O99: Thai nghén, sinh đẻ và hậu sản": [
        "mang thai", "có bầu", "sinh con", "thai nhi", "sau sinh", "sảy thai", "tiền sản giật",
    ],
    "(XVIII) R00-R99: Triệu chứng, dấu hiệu và kết quả lâm sàng": [
        "kết quả", "xét nghiệm", "chỉ số", "beta-hcg", "scc", "siêu âm", "sinh thiết", "chụp x-quang", "chụp ct",
    ],
    "(I) A00-B99: Bệnh nhiễm trùng và ký sinh trùng": [
        "bệnh lao", "kháng thuốc", "hiv", "viêm gan b", "viêm gan c", "sốt xuất huyết", "covid", "cúm", "giun", "sán", "nhiễm trùng",
    ],
}


class ICD10Tagger:
    """Fast multilingual entity matcher and chapter classifier using WHO ICD-10."""

    DEFAULT_ONTOLOGY_PATH = Path("data/lexicon/icd10_ontology.json")

    def __init__(self, ontology_path: Path | str | None = None):
        self.ontology_path = Path(ontology_path) if ontology_path else self.DEFAULT_ONTOLOGY_PATH
        self.entries: list[dict[str, Any]] = []
        self.term_to_entry: dict[str, list[dict[str, Any]]] = {}
        self.regex_vi: re.Pattern | None = None
        self.regex_cn: re.Pattern | None = None
        self.regex_en: re.Pattern | None = None
        self._load_ontology()

    def _load_ontology(self):
        if not self.ontology_path.exists():
            logger.warning(f"ICD-10 ontology file not found at {self.ontology_path}")
            return

        with open(self.ontology_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            self.entries = data.get("entries", [])

        terms_vi_list = []
        terms_cn_list = []
        terms_en_list = []

        for e in self.entries:
            # Vietnamese term
            vn = (e.get("name_vn") or "").strip().lower()
            if vn and len(vn) >= 3 and not vn.isdigit() and vn not in STOPWORDS_MEDICAL:
                vn_norm = unicodedata.normalize("NFC", vn)
                self.term_to_entry.setdefault(vn_norm, []).append(e)
                terms_vi_list.append(vn_norm)

                # Strip 'bệnh ' prefix so colloquial mentions ('đái tháo đường', 'suy thận') match
                if vn_norm.startswith("bệnh "):
                    sub = vn_norm[5:].strip()
                    if len(sub) >= 3 and sub not in STOPWORDS_MEDICAL:
                        self.term_to_entry.setdefault(sub, []).append(e)
                        terms_vi_list.append(sub)

            # Chinese term
            cn = (e.get("name_cn") or "").strip()
            if cn and len(cn) >= 2 and cn not in STOPWORDS_MEDICAL:
                self.term_to_entry.setdefault(cn, []).append(e)
                terms_cn_list.append(cn)

            # English term
            en = (e.get("name_en") or "").strip().lower()
            if en and len(en) >= 3 and not en.isdigit() and en not in STOPWORDS_MEDICAL:
                en_norm = unicodedata.normalize("NFC", en)
                self.term_to_entry.setdefault(en_norm, []).append(e)
                terms_en_list.append(en_norm)

        # Also load bilingual lexicons to link Chinese and English colloquial / clinic terms
        lexicon_dir = self.ontology_path.parent
        vi_zh_path = lexicon_dir / "icd10_vi_zh.json"
        if vi_zh_path.exists():
            try:
                with open(vi_zh_path, "r", encoding="utf-8") as f:
                    vi_zh = json.load(f)
                for vn_key, zh_val in vi_zh.items():
                    vn_norm = unicodedata.normalize("NFC", vn_key.strip().lower())
                    zh_val = zh_val.strip()
                    if vn_norm in self.term_to_entry and zh_val and len(zh_val) >= 2 and zh_val not in STOPWORDS_MEDICAL:
                        matched_entries = self.term_to_entry[vn_norm]
                        self.term_to_entry.setdefault(zh_val, []).extend(matched_entries)
                        terms_cn_list.append(zh_val)
            except Exception as e:
                logger.debug(f"Could not load icd10_vi_zh.json: {e}")

        vi_en_path = lexicon_dir / "icd10_vi_en.json"
        if vi_en_path.exists():
            try:
                with open(vi_en_path, "r", encoding="utf-8") as f:
                    vi_en = json.load(f)
                for vn_key, en_val in vi_en.items():
                    vn_norm = unicodedata.normalize("NFC", vn_key.strip().lower())
                    en_val = en_val.strip().lower()
                    if vn_norm in self.term_to_entry and en_val and len(en_val) >= 3 and en_val not in STOPWORDS_MEDICAL:
                        matched_entries = self.term_to_entry[vn_norm]
                        self.term_to_entry.setdefault(en_val, []).extend(matched_entries)
                        terms_en_list.append(en_val)
            except Exception as e:
                logger.debug(f"Could not load icd10_vi_en.json: {e}")

        med_terms_path = lexicon_dir / "medical_terms.json"
        if med_terms_path.exists():
            try:
                with open(med_terms_path, "r", encoding="utf-8") as f:
                    med_terms = json.load(f)
                for vn_key, syns in med_terms.items():
                    vn_norm = unicodedata.normalize("NFC", vn_key.strip().lower())
                    matched_entries = self.term_to_entry.get(vn_norm) or self.term_to_entry.get(f"bệnh {vn_norm}")
                    if matched_entries:
                        # Extract multi-word phrases and key medical words
                        candidates = [s.strip().lower() for s in re.split(r",|;|\bor\b", syns) if s.strip()]
                        if not candidates:
                            candidates = [syns.strip().lower()]
                        # Also split into 2-word ngrams if candidate is long
                        for c in candidates:
                            words = c.split()
                            if len(words) >= 2:
                                for k in range(len(words) - 1):
                                    bigram = f"{words[k]} {words[k+1]}"
                                    if len(bigram) >= 6 and bigram not in STOPWORDS_MEDICAL:
                                        self.term_to_entry.setdefault(bigram, []).extend(matched_entries)
                                        terms_en_list.append(bigram)
                            elif len(c) >= 5 and c not in STOPWORDS_MEDICAL:
                                self.term_to_entry.setdefault(c, []).extend(matched_entries)
                                terms_en_list.append(c)
            except Exception as e:
                logger.debug(f"Could not load medical_terms.json: {e}")

        # Sort longest terms first to give priority to compound medical concepts
        terms_vi_list = sorted(set(terms_vi_list), key=lambda x: -len(x))
        terms_cn_list = sorted(set(terms_cn_list), key=lambda x: -len(x))
        terms_en_list = sorted(set(terms_en_list), key=lambda x: -len(x))

        logger.info(
            f"Loaded {len(self.entries):,} ICD-10 entries: "
            f"VI={len(terms_vi_list):,}, ZH={len(terms_cn_list):,}, EN={len(terms_en_list):,} unique terms."
        )

        # Compile regexes with word boundaries for latin languages (avoid substring false positives)
        if terms_vi_list:
            escaped_vi = [re.escape(t) for t in terms_vi_list if len(t) >= 3]
            self.regex_vi = re.compile(r"(?:\b|_)(?:" + "|".join(escaped_vi) + r")(?:\b|_)", re.IGNORECASE)

        if terms_cn_list:
            escaped_cn = [re.escape(t) for t in terms_cn_list if len(t) >= 2]
            self.regex_cn = re.compile("|".join(escaped_cn))

        if terms_en_list:
            escaped_en = [re.escape(t) for t in terms_en_list if len(t) >= 3]
            self.regex_en = re.compile(r"\b(?:" + "|".join(escaped_en) + r")\b", re.IGNORECASE)

    def extract_chapter_code(self, chapter_str: str) -> str:
        """Extracts code range like (N00-N99) or (E00-E90) from full chapter text."""
        if not chapter_str:
            return ""
        m = re.search(r"\(([A-Z]\d{2}-[A-Z]\d{2})\)", chapter_str)
        if m:
            return m.group(1)
        return chapter_str[:10]

    def tag_document(
        self,
        text: str,
        title: str = "",
        headings: list[str] | None = None,
        lang: str | None = None,
    ) -> ICD10TagResult:
        """Tags a document with its primary ICD-10 Chapter, codes, and matching entities."""
        if not text and not title:
            return ICD10TagResult()

        headings = headings or []
        combined_title = f"{title} {' '.join(headings)}".strip()

        title_norm = unicodedata.normalize("NFC", combined_title.lower())
        body_sample = unicodedata.normalize("NFC", text[:3000].lower())

        chapter_scores: Counter[str] = Counter()
        matched_codes: set[str] = set()
        matched_terms: set[str] = set()

        def scan_text(txt: str, weight: int):
            if not txt:
                return

            matches = []
            if self.regex_vi:
                for m in self.regex_vi.finditer(txt):
                    matches.append(m.group(0).strip(" _").lower())

            if self.regex_cn:
                for m in self.regex_cn.finditer(txt):
                    matches.append(m.group(0))

            if self.regex_en:
                for m in self.regex_en.finditer(txt):
                    matches.append(m.group(0).lower())

            for term in matches:
                matched_entries = self.term_to_entry.get(term, [])
                for e in matched_entries:
                    chap = e.get("chapter")
                    code = e.get("icd10_code")
                    if chap:
                        chapter_scores[chap] += weight
                    if code:
                        matched_codes.add(code)
                    matched_terms.add(term)

        # Title and section headings get 3x weight
        scan_text(title_norm, weight=3)
        # Main text get 1x weight
        scan_text(body_sample, weight=1)

        # If exact ontology did not match, fallback to clinical keywords
        if not chapter_scores:
            combined_txt = f"{title_norm} {body_sample}"
            for chap, kws in CLINICAL_CHAPTER_KEYWORDS.items():
                for kw in kws:
                    if kw in combined_txt:
                        chapter_scores[chap] += 2
                        matched_terms.add(kw)

        if not chapter_scores:
            return ICD10TagResult()

        sorted_chapters = chapter_scores.most_common()
        top_chapter, _ = sorted_chapters[0]
        top_code = self.extract_chapter_code(top_chapter)

        return ICD10TagResult(
            primary_chapter=top_chapter,
            primary_chapter_code=top_code,
            matched_chapters=sorted_chapters,
            matched_codes=sorted(list(matched_codes))[:10],
            matched_terms=sorted(list(matched_terms))[:15],
        )

    # Alias for convenience
    tag = tag_document
