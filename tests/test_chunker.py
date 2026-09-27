"""Unit tests for document chunking."""

from src.ingestion.chunker import DocumentChunker


def test_chunker_basic():
    chunker = DocumentChunker(max_chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    text = (
        "Bệnh sỏi thận là tình trạng lắng đọng chất khoáng trong thận. "
        "Triệu chứng phổ biến bao gồm đau quặn thận, tiểu buốt và sốt. "
        "Bệnh nhân cần uống nhiều nước và đi khám bác sĩ chuyên khoa tiết niệu kịp thời."
    )

    chunks = chunker.chunk_document(doc_id="doc_test_1", text=text, lang="vi")
    assert len(chunks) > 0

    for c in chunks:
        assert c.doc_id == "doc_test_1"
        assert len(c.chunk_text) >= 10
        # Check that chunk_text is strictly an extracted substring from original text
        assert c.chunk_text in text


def test_chunker_short_text():
    chunker = DocumentChunker(max_chunk_size=200, min_chunk_size=10)
    text = "Cần uống đủ 2 lít nước mỗi ngày để phòng sỏi thận."

    chunks = chunker.chunk_document(doc_id="short_doc", text=text)
    assert len(chunks) == 1
    assert chunks[0].chunk_text == text
    assert chunks[0].doc_id == "short_doc"


def test_contextual_chunking_vi():
    chunker = DocumentChunker(
        max_chunk_size=100, chunk_overlap=20, min_chunk_size=10, enable_contextual=True
    )
    text = (
        "Bệnh sỏi thận là tình trạng lắng đọng chất khoáng trong thận. "
        "Triệu chứng phổ biến bao gồm đau quặn thận, tiểu buốt và sốt."
    )
    title = "Tổng quan về bệnh sỏi thận"

    chunks = chunker.chunk_document(
        doc_id="doc_vi_ctx",
        text=text,
        metadata={"title": title},
        lang="vi",
    )
    assert len(chunks) > 0
    for c in chunks:
        # Crucial submission requirement: chunk_text must remain 100% exact raw substring
        assert c.chunk_text in text
        assert "Tiêu đề:" not in c.chunk_text

        # contextual_text must have the prepended metadata
        assert c.contextual_text is not None
        assert f"Tiêu đề: {title}\nNội dung: {c.chunk_text}" == c.contextual_text


def test_contextual_chunking_multilingual():
    # Chinese
    chunker_zh = DocumentChunker(enable_contextual=True)
    text_zh = "糖尿病是一种以高血糖为特征的代谢性疾病。"
    title_zh = "糖尿病指南"
    chunks_zh = chunker_zh.chunk_document(
        doc_id="doc_zh",
        text=text_zh,
        metadata={"title": title_zh},
        lang="zh",
    )
    assert len(chunks_zh) == 1
    assert chunks_zh[0].chunk_text == text_zh
    assert chunks_zh[0].contextual_text == f"标题: {title_zh}\n内容: {text_zh}"

    # English
    chunker_en = DocumentChunker(enable_contextual=True)
    text_en = "Hypertension is a long-term medical condition in which blood pressure is persistently elevated."
    title_en = "Hypertension Clinical Guidelines"
    chunks_en = chunker_en.chunk_document(
        doc_id="doc_en",
        text=text_en,
        metadata={"title": title_en},
        lang="en",
    )
    assert len(chunks_en) == 1
    assert chunks_en[0].chunk_text == text_en
    assert chunks_en[0].contextual_text == f"Title: {title_en}\nContent: {text_en}"


def test_contextual_chunking_disabled():
    chunker = DocumentChunker(enable_contextual=False)
    text = "Uống nhiều nước có lợi cho sức khỏe."
    chunks = chunker.chunk_document(
        doc_id="doc_off",
        text=text,
        metadata={"title": "Lời khuyên sức khỏe"},
        lang="vi",
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_text == text
    assert chunks[0].contextual_text is None
