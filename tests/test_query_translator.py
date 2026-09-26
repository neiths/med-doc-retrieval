"""Unit tests for query translation and medical keyword extraction."""

from src.crawler.query_translator import VI_EN_MEDICAL_LEXICON, QueryTranslator


def test_lexicon_keywords():
    translator = QueryTranslator()
    vi_query = "Bệnh nhân bị tắc nghẽn đường tiết niệu do sỏi thận kèm đau quặn thận"

    keywords = translator.extract_pubmed_keywords(vi_query)
    assert len(keywords) > 0
    # Verify that domain keywords were extracted
    assert any(
        term in keywords.lower()
        for term in ["kidney", "calculi", "urinary", "obstruction", "colic"]
    )


def test_lexicon_coverage():
    assert "sỏi thận" in VI_EN_MEDICAL_LEXICON
    assert "tắc nghẽn đường tiết niệu" in VI_EN_MEDICAL_LEXICON
    assert "đau quặn thận" in VI_EN_MEDICAL_LEXICON


def test_external_lexicon_and_stopwords():
    translator = QueryTranslator(
        lexicon_path="data/lexicon/medical_terms.json",
        stopwords_path="configs/pubmed_stopwords.txt",
    )
    # Check that expanded terms are present
    assert "sonde jj" in translator.lexicon
    assert "nhồi máu cơ tim" in translator.lexicon
    assert "tán sỏi qua da" in translator.lexicon

    # Check stopwords
    assert "protocol" in translator.stopwords
    assert "guidelines" in translator.stopwords

    # Check keyword extraction with expanded terms
    query = "Điều trị sonde jj và tán sỏi qua da protocol"
    keywords = translator.extract_pubmed_keywords(query).lower()
    assert (
        "stent" in keywords
        or "ureteral" in keywords
        or "lithotomy" in keywords
        or "pcnl" in keywords
    )
    # Stopword should be filtered out
    assert "protocol" not in keywords.split()


def test_custom_lexicon_loading(tmp_path):
    custom_lex = tmp_path / "custom_lex.json"
    custom_lex.write_text('{"hội chứng thận hư": "nephrotic syndrome"}', encoding="utf-8")

    custom_stop = tmp_path / "custom_stop.txt"
    custom_stop.write_text("# Comment\nxyzstopword\n", encoding="utf-8")

    translator = QueryTranslator(lexicon_path=custom_lex, stopwords_path=custom_stop)
    assert "hội chứng thận hư" in translator.lexicon
    assert "xyzstopword" in translator.stopwords
