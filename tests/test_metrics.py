"""Unit tests for competition evaluation metrics."""

import pytest

from src.evaluation.metrics import compute_prf, evaluate_predictions


def test_compute_prf_basic():
    retrieved = {"doc1", "doc2", "doc3"}
    relevant = {"doc1", "doc2", "doc4", "doc5"}

    # TP = 2, retrieved = 3 -> P = 2/3
    # TP = 2, relevant = 4 -> R = 2/4 = 0.5
    # F2 = (5 * (2/3) * 0.5) / (4 * (2/3) + 0.5) = 1.6666 / (2.6666 + 0.5) = 1.6666 / 3.1666 ≈ 0.5263
    p, r, f2 = compute_prf(retrieved, relevant, beta=2.0)
    assert pytest.approx(p, 0.001) == 2 / 3
    assert pytest.approx(r, 0.001) == 0.5
    assert pytest.approx(f2, 0.001) == (5 * (2/3) * 0.5) / (4 * (2/3) + 0.5)


def test_compute_prf_edge_cases():
    # Both empty
    p, r, f2 = compute_prf(set(), set())
    assert (p, r, f2) == (1.0, 1.0, 1.0)

    # Empty retrieved
    p, r, f2 = compute_prf(set(), {"doc1"})
    assert (p, r, f2) == (0.0, 0.0, 0.0)

    # Empty relevant
    p, r, f2 = compute_prf({"doc1"}, set())
    assert (p, r, f2) == (0.0, 0.0, 0.0)


def test_evaluate_predictions_macro():
    predictions = [
        {
            "id": 1,
            "relevant_docs": ["d1", "d2"],
            "relevant_chunks": [{"doc_id": "d1", "chunk_text": "c1 text"}],
        },
        {
            "id": 2,
            "relevant_docs": ["d3"],
            "relevant_chunks": [{"doc_id": "d3", "chunk_text": "c3 text"}],
        },
    ]

    ground_truth = [
        {
            "id": 1,
            "relevant_docs": ["d1"],
            "relevant_chunks": [{"doc_id": "d1", "chunk_text": "c1 text"}],
        },
        {
            "id": 2,
            "relevant_docs": ["d3"],
            "relevant_chunks": [{"doc_id": "d3", "chunk_text": "c3 text"}],
        },
    ]

    results = evaluate_predictions(predictions, ground_truth)
    assert results["num_queries_evaluated"] == 2
    assert results["macro_chunk_f2"] == 1.0
    assert 0.0 < results["macro_doc_f2"] <= 1.0
    assert "combined_f2" in results
