"""Multilingual text normalization and cleaning utilities."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalizes multilingual text (Unicode NFC, whitespace normalization)."""
    if not text:
        return ""

    # Normalize unicode to NFC
    text = unicodedata.normalize("NFC", text)

    # Replace zero-width spaces, non-breaking spaces, and tabs
    text = text.replace("\u200b", "").replace("\xa0", " ").replace("\t", " ")

    # Collapse excessive consecutive spaces (preserving newlines)
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
