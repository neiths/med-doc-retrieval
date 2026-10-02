"""Multilingual text normalization and cleaning utilities for biomedical documents."""

import re
import unicodedata

# Common web boilerplate patterns across Vietnamese and Chinese medical sites
BOILERPLATE_PATTERNS = [
    # Vietnamese hospital & clinic navigation / ads
    r"(?i)^.*(để đặt lịch khám|bấm số hotline|tải ứng dụng .* để đặt lịch|theo dõi chúng tôi|chia sẻ bài viết|xem thêm:|bài viết liên quan|bản quyền thuộc về|chỉ mang tính chất tham khảo|hotline tư vấn|đăng ký nhận bản tin).*$",
    r"(?i)^.*(fanpage:|hotline:|zalo:|email liên hệ:|tổng đài tư vấn:).*$",
    r"(?i)^.*(bác sĩ tư vấn 24/7|đặt mua gói dịch vụ|khám bệnh trực tuyến).*$",
    # Chinese medical encyclopedia & Q&A navigation / ads
    r"^.*(扫描二维码|手机浏览|用手机扫描|关注微信公众号|相关推荐|没有更多了|免责声明|以上内容仅供参考|目前暂无留言|添加留言|订阅讨论RSS|关于“.*”的留言).*$",
    r"^.*(A\+医学百科 >>|查看全文|目录|手机版|返回顶部).*$",
    r"^.*(血液内科医生推荐|内科医生推荐|儿科医生推荐|妇科医生推荐|医生建议仅供参考).*$",
    # Markdown table borders and separator artifacts
    r"^\s*\|.*\|\s*$",
    r"^\s*[-=_]{3,}\s*$",
]

COMPILED_BOILERPLATE = [re.compile(p) for p in BOILERPLATE_PATTERNS]


def fix_mojibake(text: str) -> str:
    """Detects and repairs UTF-8 bytes mistakenly decoded as GB18030 in Chinese articles."""
    if not text:
        return ""
    # Check for signature mojibake prefixes like 鐧界櫆 or high density of rare GBK characters
    if "鐧" in text or "鑳" in text or "鐤" in text or "鎴" in text:
        try:
            recovered = text.encode("gb18030", errors="ignore").decode("utf-8", errors="ignore")
            # If recovered text contains common Chinese punctuation/characters, accept it
            if len(re.findall(r"[\u4e00-\u9fff]", recovered)) > 10:
                return recovered
        except Exception:
            pass
    return text


def clean_boilerplate(text: str) -> str:
    """Filters out advertising headers, footers, appointment phone numbers, and navigation text."""
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Drop lines that match boilerplate patterns
        if any(pat.match(stripped) for pat in COMPILED_BOILERPLATE):
            continue
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def normalize_text(text: str, strip_boilerplate: bool = True) -> str:
    """Normalizes multilingual text (Unicode NFC, whitespace normalization, boilerplate cleanup)."""
    if not text:
        return ""

    # Attempt mojibake repair for Chinese text
    text = fix_mojibake(text)

    # Normalize unicode to NFC
    text = unicodedata.normalize("NFC", text)

    # Strip web boilerplate if requested
    if strip_boilerplate:
        text = clean_boilerplate(text)

    # Replace zero-width spaces, non-breaking spaces, and tabs
    text = text.replace("\u200b", "").replace("\xa0", " ").replace("\t", " ")

    # Collapse excessive consecutive spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Collapse excessive newlines (more than 2 newlines into 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_language(text: str) -> str:
    """Heuristic language detection between Chinese (zh), Vietnamese (vi), and English (en)."""
    if not text:
        return "en"

    # Chinese character check (CJK Unified Ideographs range)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    if cjk_count > 10:
        return "zh"

    # Vietnamese specific diacritic letters
    vi_diacritics = re.findall(
        r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
        text,
        re.IGNORECASE,
    )
    if len(vi_diacritics) > 5:
        return "vi"

    return "en"
