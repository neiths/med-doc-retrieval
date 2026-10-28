"""Unit tests for competition evaluation metrics."""

import pytest

from src.evaluation.metrics import compute_chunk_prf, compute_prf, evaluate_predictions, lcs_length


def test_compute_prf_basic():
    retrieved = {"doc1", "doc2", "doc3"}
    relevant = {"doc1", "doc2", "doc4", "doc5"}

    # TP = 2, retrieved = 3 -> P = 2/3
    # TP = 2, relevant = 4 -> R = 2/4 = 0.5
    # F2 = (5 * (2/3) * 0.5) / (4 * (2/3) + 0.5) = 1.6666 / (2.6666 + 0.5) = 1.6666 / 3.1666 ≈ 0.5263
    p, r, f2 = compute_prf(retrieved, relevant, beta=2.0)
    assert pytest.approx(p, 0.001) == 2 / 3
    assert pytest.approx(r, 0.001) == 0.5
    assert pytest.approx(f2, 0.001) == (5 * (2 / 3) * 0.5) / (4 * (2 / 3) + 0.5)


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


def test_chunk_containment_and_overlap():
    # Predicted chunk contains the gold sentence
    pred_chunks = [("doc1", "Sentence A. Gold sentence here. Sentence B.")]
    gt_chunks = [("doc1", "Gold sentence here.")]

    # Overlap / containment match
    res_overlap = evaluate_predictions(
        [
            {
                "id": 1,
                "relevant_docs": ["doc1"],
                "relevant_chunks": [{"doc_id": "doc1", "chunk_text": pred_chunks[0][1]}],
            }
        ],
        [
            {
                "id": 1,
                "relevant_docs": ["doc1"],
                "relevant_chunks": [{"doc_id": "doc1", "chunk_text": gt_chunks[0][1]}],
            }
        ],
        chunk_match_mode="overlap",
    )
    assert res_overlap["macro_chunk_f2"] == 1.0

    # Exact match mode requires 100% string equality
    exact_res = evaluate_predictions(
        [
            {
                "id": 1,
                "relevant_docs": ["doc1"],
                "relevant_chunks": [{"doc_id": "doc1", "chunk_text": pred_chunks[0][1]}],
            }
        ],
        [
            {
                "id": 1,
                "relevant_docs": ["doc1"],
                "relevant_chunks": [{"doc_id": "doc1", "chunk_text": gt_chunks[0][1]}],
            }
        ],
        chunk_match_mode="exact",
    )
    assert exact_res["macro_chunk_f2"] == 0.0


def test_lcs_length_matches_bruteforce():
    def brute(a, b):
        m = len(b)
        prev = [0] * (m + 1)
        for x in a:
            cur = [0] * (m + 1)
            for j in range(1, m + 1):
                cur[j] = prev[j - 1] + 1 if x == b[j - 1] else max(prev[j], cur[j - 1])
            prev = cur
        return prev[m]

    import random

    for _ in range(200):
        a = [random.choice("abcde") for _ in range(random.randint(0, 12))]
        b = [random.choice("abcde") for _ in range(random.randint(0, 12))]
        assert lcs_length(a, b) == brute(a, b)


def test_overlap_accepts_prediction_inside_gold():
    gold = "one two three four five six seven eight nine ten"
    pred = "three four five six seven"
    p, r, _ = compute_chunk_prf([("d", pred)], [("d", gold)])
    assert (p, r) == (1.0, 1.0)


def test_overlap_rejects_prediction_longer_than_2_5x_gold():
    gold = "one two three four five"
    pred = gold + " " + " ".join(f"noise{i}" for i in range(8))
    p, r, _ = compute_chunk_prf([("d", pred)], [("d", gold)])
    assert (p, r) == (0.0, 0.0)


def test_overlap_rejects_prediction_covering_too_little_gold():
    gold = "one two three four five six seven eight nine ten"
    pred = "one two"
    p, r, _ = compute_chunk_prf([("d", pred)], [("d", gold)])
    assert (p, r) == (0.0, 0.0)


def test_overlap_ignores_chunks_from_other_documents():
    p, r, _ = compute_chunk_prf([("docA", "one two three")], [("docB", "one two three")])
    assert (p, r) == (0.0, 0.0)
