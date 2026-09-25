"""Unit tests for multilingual tokenization."""

from src.retrieval.sparse_index import tokenize_multilingual


def test_tokenize_vietnamese():
    text = "Bệnh viện Bạch Mai điều trị sỏi thận"
    tokens = tokenize_multilingual(text, lang="vi")
    assert len(tokens) > 0
    # Checks that tokens are extracted
    assert any("thận" in t for t in tokens)


def test_tokenize_chinese():
    text = "肾结石的治疗方法有哪些"
    tokens = tokenize_multilingual(text, lang="zh")
    assert len(tokens) > 0
    # Jieba splits 肾结石 into 肾结石 or 肾 / 结石
    assert "肾结石" in tokens or "结石" in tokens


def test_tokenize_english():
    text = "Kidney stones obstruction treatment in hospital"
    tokens = tokenize_multilingual(text, lang="en")
    assert "kidney" in tokens
    assert "stones" in tokens
    assert "obstruction" in tokens
