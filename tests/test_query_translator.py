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


def test_prompt_prefix_auto_detection():
    # MarianMT default: empty prefix
    tr_marian = QueryTranslator(model_name="Helsinki-NLP/opus-mt-vi-en")
    assert tr_marian.prompt_prefix == ""

    # T5 / ndhieu models: auto-detect "vi: "
    tr_t5 = QueryTranslator(
        model_name="ndhieu1101/medical-bidirectional-machine-translation-checkpoints-511042"
    )
    assert tr_t5.prompt_prefix == "vi: "

    # Explicit override
    tr_custom = QueryTranslator(model_name="google/mt5-base", prompt_prefix="translate: ")
    assert tr_custom.prompt_prefix == "translate: "


def test_extract_chinese_keywords():
    translator = QueryTranslator(zh_lexicon_path="data/lexicon/icd10_vi_zh.json")
    query = "Điều trị bệnh nhân bị tắc nghẽn đường tiết niệu do sỏi thận"
    zh_kw = translator.extract_chinese_keywords(query)

    assert len(zh_kw) > 0
    # Must contain Chinese keywords for kidney stone and urinary obstruction
    assert "肾结石" in zh_kw or "结石" in zh_kw
    assert "尿路梗阻" in zh_kw or "梗阻" in zh_kw


def test_expand_acronyms():
    translator = QueryTranslator(acronyms_path="data/lexicon/medical_acronyms.json")
    query = "Điều trị bệnh nhân suy tim HFrEF giai đoạn cuối kèm STEMI"
    exp = translator.expand_acronyms(query)

    assert "suy tim phân suất tống máu giảm" in exp["vi"]
    assert "heart failure with reduced ejection fraction" in exp["en"]
    assert "射血分数降低的心力衰竭" in exp["zh"]
    assert "nhồi máu cơ tim có st chênh lên" in exp["vi"]

    # Verify PubMed keyword integration
    pubmed_kw = translator.extract_pubmed_keywords("Bệnh nhân HFrEF cấp cứu").lower()
    assert "heart" in pubmed_kw or "failure" in pubmed_kw or "hfref" in pubmed_kw

    # Verify Chinese keyword integration
    zh_kw = translator.extract_chinese_keywords("Bệnh nhân HFrEF")
    assert "射血分数降低的心力衰竭" in zh_kw

    # Verify short / ambiguous acronym case-sensitivity safeguard:
    # Uppercase CAP should match, lowercase 'cap' (as in 'cap cuu') must NOT match
    exp_cap_upper = translator.expand_acronyms("Bệnh nhân mắc CAP")
    assert "viêm phổi mắc phải tại cộng đồng" in exp_cap_upper["vi"]

    exp_cap_lower = translator.expand_acronyms("bệnh nhân cap cứu khẩn cấp")
    assert "viêm phổi" not in exp_cap_lower["vi"]

    # Short acronym PE: uppercase matches, lowercase does not
    exp_pe_upper = translator.expand_acronyms("Chẩn đoán PE cấp")
    assert "thuyên tắc phổi" in exp_pe_upper["vi"]

    exp_pe_lower = translator.expand_acronyms("phương pháp pe thắt lưng")
    assert "thuyên tắc phổi" not in exp_pe_lower["vi"]

