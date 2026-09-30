"""Unit tests for BGEReranker with balanced chunk selection and recall safeguard."""

from src.reranker.bge_reranker import BGEReranker


def test_two_pass_chunk_selection_multi_doc():
    """When multiple docs exist, Pass 1 prioritizes diversity up to max_chunks_per_doc."""
    reranker = BGEReranker(device="cpu")
    # Mock compute_scores to return descending scores based on index
    reranker.compute_scores = lambda pairs: [10.0 - i * 0.5 for i in range(len(pairs))]

    candidates = [
        {"doc_id": "doc1", "chunk_text": "doc1 chunk 1"},
        {"doc_id": "doc1", "chunk_text": "doc1 chunk 2"},
        {"doc_id": "doc1", "chunk_text": "doc1 chunk 3"},
        {"doc_id": "doc2", "chunk_text": "doc2 chunk 1"},
        {"doc_id": "doc2", "chunk_text": "doc2 chunk 2"},
        {"doc_id": "doc3", "chunk_text": "doc3 chunk 1"},
    ]

    doc_ids, chunks = reranker.rerank(
        query="test query",
        candidates=candidates,
        top_k_chunks=4,
        top_k_docs=3,
        score_threshold=0.0,
        max_chunks_per_doc=2,
    )

    assert doc_ids == ["doc1", "doc2", "doc3"]
    assert len(chunks) == 4
    # All chunks must belong to doc_ids
    assert all(c["doc_id"] in doc_ids for c in chunks)
    # Doc1 gets at most 2 in initial pass
    doc1_chunks = [c for c in chunks if c["doc_id"] == "doc1"]
    assert len(doc1_chunks) == 2


def test_two_pass_recall_safeguard_single_doc():
    """When only 1 document is relevant, Pass 2 fills remaining quota to safeguard Recall for F2."""
    reranker = BGEReranker(device="cpu")
    # Only doc1 passes the threshold
    candidates = [
        {"doc_id": "doc1", "chunk_text": "doc1 section A"},
        {"doc_id": "doc1", "chunk_text": "doc1 section B"},
        {"doc_id": "doc1", "chunk_text": "doc1 section C"},
        {"doc_id": "doc1", "chunk_text": "doc1 section D"},
        {"doc_id": "doc2", "chunk_text": "doc2 irrevelant"},
    ]

    reranker.compute_scores = lambda pairs: [8.0, 7.0, 6.0, 5.0, -10.0]

    doc_ids, chunks = reranker.rerank(
        query="test query",
        candidates=candidates,
        top_k_chunks=4,
        top_k_docs=2,
        score_threshold=-5.0,
        max_chunks_per_doc=2,
    )

    assert doc_ids == ["doc1"]
    # Thanks to Pass 2 (recall safeguard), doc1 fills all 4 available slots rather than being restricted to 2!
    assert len(chunks) == 4
    assert [c["chunk_text"] for c in chunks] == [
        "doc1 section A",
        "doc1 section B",
        "doc1 section C",
        "doc1 section D",
    ]
