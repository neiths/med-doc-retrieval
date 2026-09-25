"""Unit tests for query translation and medical keyword extraction."""

from src.crawler.query_translator import VI_EN_MEDICAL_LEXICON, QueryTranslator


def test_lexicon_keywords():
    translator = QueryTranslator()
    vi_query = "Bệnh nhân bị tắc nghẽn đường tiết niệu do sỏi thận kèm đau quặn thận"

    keywords = translator.extract_pubmed_keywords(vi_query)
    assert len(keywords) > 0
    # Verify that domain keywords were extracted
    assert any(term in keywords.lower() for term in ["kidney", "calculi", "urinary", "obstruction", "colic"])


def test_lexicon_coverage():
    assert "sỏi thận" in VI_EN_MEDICAL_LEXICON
    assert "tắc nghẽn đường tiết niệu" in VI_EN_MEDICAL_LEXICON
    assert "đau quặn thận" in VI_EN_MEDICAL_LEXICON
