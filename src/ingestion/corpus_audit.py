"""Per-document quality signals for the crawled medical corpus.

The audit answers one question per document: is this worth chunking and indexing?
Signals are deliberately coarse and cheap — this is a triage table, not a classifier.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"\w+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SPACE_RE = re.compile(r"\s+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VI_DIACRITIC_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)
BASE64_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=]{80,}")
BINARY_MAGIC = ("PK\x03\x04", "%PDF", "\x89PNG", "GIF8", "\xff\xd8\xff")

# Chinese roots are matched as substrings because Chinese has no word delimiters here.
ZH_MEDICAL_ROOTS = (
    "治疗 诊断 患者 病人 疾病 症状 感染 手术 肿瘤 癌症 医院 医生 药物 临床 病例 病情 病因 发病 "
    "并发症 综合症 综合征 血压 心脏 肾脏 肝脏 肺炎 结石 炎症 化疗 放疗 儿科 妇科 内科 外科 "
    "血液 糖尿病 免疫 病理 急诊 康复"
).split()

# Classical medicine writes 前胡散方, 去芦头, 炙微赤 instead of 治疗/诊断/患者, so the modern
# root list scores genuine 古籍 at zero. These roots cover that register.
ZH_TCM_ROOTS = (
    "中医 中药 方剂 方药 药材 本草 汤剂 丸剂 散剂 膏方 医案 医论 辨证 症候 病机 经络 针灸 艾灸 "
    "穴位 气血 脏腑 脾胃 肝肾 心脾 肺气 湿热 寒湿 风热 风寒 阴虚 阳虚 气虚 血虚 痰湿 淤血 瘀血 "
    "处方 剂量 服法 主治 功效 归经 禁忌 配伍 脉象 舌苔 治愈 病症 方书 内经 伤寒"
).split()

# Materia medica entries list ingredients and nothing else; without these a 方剂 page scores 0.
# Food and lifestyle pages that mention 人参 or 山药 also score here, which is acceptable: the
# signal only routes documents to review.
ZH_HERB_ROOTS = (
    "甘草 当归 黄芪 人参 党参 白术 茯苓 陈皮 半夏 生姜 大枣 桂枝 麻黄 柴胡 白芍 川芎 地黄 "
    "黄连 黄芩 栀子 附子 细辛 石膏 知母 牛膝 红花 桃仁 丹参 五味子 枸杞子 山药 杜仲 续断 "
    "防风 羌活 独活 前胡 桔梗 杏仁 薄荷 菊花 金银花 连翘 泽泻 车前子 山茱萸 龙骨 牡蛎"
).split()

# Titles that repeat across a site's navigation/template pages rather than describing content.
TEMPLATE_TITLES = {
    "trang chủ",
    "home",
    "首页",
    "药方",
    "danh sách",
    "mục lục",
    "chuyên mục",
    "tìm kiếm",
    "đăng nhập",
    "liên hệ",
    "giới thiệu",
}
TEMPLATE_TITLE_RE = re.compile(
    r"^(trang chủ|home|首页|danh sách|mục lục|chuyên mục|tìm kiếm|đăng nhập|liên hệ)\b"
    r"|(详情|列表|目录|分类)\s*-\s*\S+\s*-",
    re.IGNORECASE,
)

MIN_BINARY_RATIO = 0.01
MIN_BINARY_CTRL_CHARS = 5
LINE_LIMIT = 40
PAYLOAD_MIN_CHARS = 300
PAYLOAD_MIN_SHARE = 0.4


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text or "").strip()


def is_binary(text: str) -> bool:
    """Detects crawler accidents: decoded archives, PDFs and blobs saved as 'text'."""
    sample = (text or "")[:4000]
    if not sample:
        return False
    if sample.startswith(BINARY_MAGIC):
        return True
    ctrl = len(CTRL_RE.findall(sample))
    return ctrl > MIN_BINARY_CTRL_CHARS or ctrl / len(sample) > MIN_BINARY_RATIO


def short_line_char_ratio(text: str) -> float:
    """Share of characters sitting on lines shorter than LINE_LIMIT.

    Navigation, index and listing pages are dominated by short link labels, while an
    article's characters are concentrated in long paragraph lines.
    """
    lines = [line.strip() for line in (text or "").split("\n") if line.strip()]
    if not lines:
        return 1.0
    total = sum(len(line) for line in lines)
    short = sum(len(line) for line in lines if len(line) < LINE_LIMIT)
    return short / total


def looks_like_payload(text: str) -> bool:
    """Detects pages whose extracted text is a long base64/hex-ish blob instead of prose."""
    text = text or ""
    tokens = BASE64_TOKEN_RE.findall(text)
    if not tokens:
        return False
    encoded = sum(len(token) for token in tokens)
    return encoded >= PAYLOAD_MIN_CHARS and encoded / max(len(text.strip()), 1) >= PAYLOAD_MIN_SHARE


def detect_language(text: str, sample_size: int = 2000) -> str:
    """Heuristic language detection matching the ingestion cleaner, on a prefix for speed."""
    prefix = (text or "")[:sample_size]
    if len(CJK_RE.findall(prefix)) > 10:
        return "zh"
    if len(VI_DIACRITIC_RE.findall(prefix)) > 5:
        return "vi"
    return "en"


def language_mismatch(text: str, declared: str | None) -> bool:
    return bool(declared) and detect_language(text) != declared


def load_medical_vocabularies(lexicon_dir: str | Path) -> dict[str, set[str]]:
    """Builds Vietnamese and English matched-word sets from the ICD/medical lexicons."""
    lexicon_dir = Path(lexicon_dir)
    vi_terms: set[str] = set()
    en_words: set[str] = set()

    terms_file = lexicon_dir / "medical_terms.json"
    if terms_file.exists():
        with open(terms_file, encoding="utf-8") as f:
            for vi, en in json.load(f).items():
                vi_terms.add(vi)
                en_words.update(WORD_RE.findall(str(en).lower()))

    vi_zh_file = lexicon_dir / "icd10_vi_zh.json"
    if vi_zh_file.exists():
        with open(vi_zh_file, encoding="utf-8") as f:
            vi_terms.update(json.load(f).keys())

    def words(terms: set[str], min_len: int) -> set[str]:
        counts: Counter[str] = Counter()
        for term in terms:
            for word in WORD_RE.findall(term.lower()):
                if len(word) >= min_len:
                    counts[word] += 1
        return set(counts)

    return {"vi": words(vi_terms, 3), "en": {w for w in en_words if len(w) >= 4}}


def medical_hits(text: str, lang: str, vocab: dict[str, set[str]]) -> int:
    """Counts distinct medical terms found in the text (coarse topic signal)."""
    text = text or ""
    if lang == "zh":
        modern = sum(text.count(root) for root in ZH_MEDICAL_ROOTS)
        classical = sum(text.count(root) for root in ZH_TCM_ROOTS)
        classical += sum(text.count(root) for root in ZH_HERB_ROOTS)
        return max(modern, classical)
    matched = vocab.get(lang) or vocab.get("vi") or set()
    return len(matched.intersection(WORD_RE.findall(text.lower())))


def template_title_kind(title: str) -> str:
    """Returns 'exact' for a bare navigation title, 'pattern' for a label-like one, else ''.

    'exact' titles ("trang chủ", "首页") are never article headings, so they can be dropped
    outright. 'pattern' titles are often a constant site-wide prefix ("疾病详情-疾病-…") that
    still heads real content, so they only count against a document when nothing else marks
    it as an article.
    """
    normalized = (title or "").strip().lower()
    if not normalized:
        return ""
    if normalized in TEMPLATE_TITLES:
        return "exact"
    return "pattern" if TEMPLATE_TITLE_RE.search(normalized) else ""


def is_template_title(title: str) -> bool:
    return bool(template_title_kind(title))


def head_fingerprint(text: str) -> str:
    """Fingerprint of the opening 500 normalized chars — catches syndicated copies."""
    head = normalize_text(text)[:500].lower()
    return hashlib.blake2b(head.encode("utf-8"), digest_size=8).hexdigest()


def resolve_duplicates(
    doc_ids: list[str],
    sizes: list[int],
    fingerprints: list[str],
) -> dict[str, str]:
    """Maps every duplicated doc_id to the canonical one (longest text, then smallest id)."""
    groups: dict[str, list[int]] = {}
    for position, fingerprint in enumerate(fingerprints):
        groups.setdefault(fingerprint, []).append(position)

    canonical: dict[str, str] = {}
    for positions in groups.values():
        if len(positions) < 2:
            continue
        best = min(positions, key=lambda i: (-sizes[i], doc_ids[i]))
        for i in positions:
            canonical[doc_ids[i]] = doc_ids[best]
    return canonical


def decide_verdict(
    signals: dict[str, Any],
    canonical_doc_id: str | None,
    min_chars: int,
    min_chars_hard: int,
    min_med_density: float,
    max_short_line_ratio: float,
) -> tuple[str, list[str]]:
    """Returns (keep|review|drop, reasons). Drops first, then reviews, so reasons stay ordered.

    Length is graded rather than thresholded: below ``min_chars_hard`` a document cannot hold
    a usable answer, between the two thresholds it is short but still readable, so it is
    flagged for a human instead of discarded.
    """
    doc_id = signals["doc_id"]
    template_kind = signals.get("template_title") or ""
    drop_reasons: list[str] = []
    if signals.get("status") not in (None, "success"):
        drop_reasons.append(f"crawl_status:{signals['status']}")
    if signals["is_binary"]:
        drop_reasons.append("binary_blob")
    if signals["payload_like"]:
        drop_reasons.append("encoded_payload")
    if signals["char_count"] < min_chars_hard:
        drop_reasons.append("fragment")
    if template_kind == "exact":
        drop_reasons.append("template_title")
    elif template_kind == "pattern" and (
        signals.get("med_hits", 0) <= 0 or signals["short_line_ratio"] >= max_short_line_ratio
    ):
        # A label-like title only condemns a page that carries no medical content or is all nav.
        drop_reasons.append("template_title")
    if canonical_doc_id and canonical_doc_id != doc_id:
        drop_reasons.append(f"duplicate_of:{canonical_doc_id}")
    if drop_reasons:
        return "drop", drop_reasons

    review_reasons: list[str] = []
    if signals["char_count"] < min_chars:
        review_reasons.append("too_short")
    if template_kind == "pattern":
        review_reasons.append("template_title")
    if signals["med_density"] <= min_med_density:
        review_reasons.append("no_medical_terms")
    if signals["short_line_ratio"] >= max_short_line_ratio:
        review_reasons.append("listing_page")
    if signals["lang_mismatch"]:
        review_reasons.append("lang_mismatch")
    if review_reasons:
        return "review", review_reasons
    return "keep", []
