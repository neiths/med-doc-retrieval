"""Unit tests for Qdrant Local Index with Native Hybrid Search and RRF."""

from pathlib import Path

import numpy as np

from src.retrieval.qdrant_index import (
    QdrantLocalIndex,
    lexical_weights_to_sparse_vector,
    text_to_sparse_vector,
)


def test_text_to_sparse_vector():
    vec = text_to_sparse_vector("sỏi thận và đường tiết niệu", lang="vi")
    assert len(vec.indices) > 0
    assert len(vec.values) == len(vec.indices)
    assert all(v > 0 for v in vec.values)


def test_lexical_weights_to_sparse_vector():
    # Test with integer token IDs (from BGE-M3 head)
    weights = {"104": 2.5, "2055": 1.1, "9999": 0.4}
    vec = lexical_weights_to_sparse_vector(weights)
    assert len(vec.indices) == 3
    assert set(vec.indices) == {104, 2055, 9999}
    assert vec.values == [2.5, 1.1, 0.4]

    # Test with empty
    empty_vec = lexical_weights_to_sparse_vector({})
    assert len(empty_vec.indices) == 0
    assert len(empty_vec.values) == 0

    # Test with string token fallback
    str_weights = {"sỏi": 1.8, "thận": 2.0}
    str_vec = lexical_weights_to_sparse_vector(str_weights)
    assert len(str_vec.indices) == 2
    assert all(isinstance(idx, int) for idx in str_vec.indices)


def test_qdrant_local_index(tmp_path: Path):
    storage_dir = tmp_path / "qdrant_test"
    index = QdrantLocalIndex(
        storage_path=storage_dir,
        collection_name="test_med_chunks",
        dimension=4,
    )

    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "chunk_text": "Điều trị sỏi thận bằng tán sỏi ngoài cơ thể",
            "lang": "vi",
            "metadata": {"title": "Sỏi thận", "source": "test"},
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "chunk_text": "Management of acute ureteral colic",
            "lang": "en",
            "metadata": {"title": "Colic", "source": "pubmed"},
        },
    ]

    embeddings = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )

    # Supply native sparse weights directly
    sparse_weights = [
        {"101": 2.0, "102": 1.5},
        {"201": 3.0, "202": 1.0},
    ]

    index.add_chunks(chunks, embeddings, sparse_weights=sparse_weights)
    assert index.count() == 2

    # Test Hybrid Search with native query sparse vector
    q_emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    q_sparse = {"101": 1.8}
    results = index.search(
        query_text="sỏi thận",
        query_embedding=q_emb,
        query_sparse=q_sparse,
        top_k=2,
    )

    assert len(results) > 0
    assert "doc_id" in results[0]
    assert "chunk_text" in results[0]
    assert results[0]["doc_id"] == "d1"
