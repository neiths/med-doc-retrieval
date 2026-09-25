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
