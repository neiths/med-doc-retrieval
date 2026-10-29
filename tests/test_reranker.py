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


def test_expansion_scores_unsampled_chunks_of_top_docs():
    """A document's unretrieved passage can win once the whole document is rescored."""
    reranker = BGEReranker(device="cpu")
    candidates = [
        {"doc_id": "doc1", "chunk_text": "retrieved but weak passage"},
        {"doc_id": "doc2", "chunk_text": "other doc passage"},
    ]
    requested_docs = []

    def provider(doc_ids, max_per_doc):
        requested_docs.append((doc_ids, max_per_doc))
        return [
            {"doc_id": "doc1", "chunk_text": "the actual answer passage"},
            {"doc_id": "doc1", "chunk_text": "retrieved but weak passage"},
        ]

    def fake_scores(pairs):
        return [9.0 if "answer" in text else 0.5 for _, text in pairs]

    reranker.compute_scores = fake_scores

    doc_ids, chunks = reranker.rerank(
        query="q",
        candidates=candidates,
        top_k_chunks=2,
        top_k_docs=1,
        score_threshold=0.0,
        max_chunks_per_doc=2,
        expand_provider=provider,
        expand_top_docs=1,
        expand_max_chunks_per_doc=8,
    )

    assert requested_docs == [(["doc1"], 8)]
    assert doc_ids == ["doc1"]
    assert chunks[0]["chunk_text"] == "the actual answer passage"
    # The duplicate candidate must not be scored or emitted twice.
    assert [c["chunk_text"] for c in chunks].count("retrieved but weak passage") == 1


def test_expansion_disabled_by_default():
    reranker = BGEReranker(device="cpu")
    reranker.compute_scores = lambda pairs: [1.0 for _ in pairs]
    called = []
    doc_ids, chunks = reranker.rerank(
        query="q",
        candidates=[{"doc_id": "doc1", "chunk_text": "only passage"}],
        top_k_chunks=1,
        top_k_docs=1,
        score_threshold=0.0,
        expand_provider=lambda doc_ids, n: called.append(doc_ids) or [],
    )
    assert called == []
    assert doc_ids == ["doc1"]


def test_length_sorted_batches_cover_all_pairs_once():
    reranker = BGEReranker(device="cpu", batch_size=3)
    pairs = [
        ("q", "x" * 100),
        ("q", "x" * 10),
        ("q", "x" * 50),
        ("q", "x" * 1000),
        ("q", "x" * 20),
        ("q", "x" * 200),
        ("q", "x" * 5),
    ]

    batches = reranker._batches_by_length(pairs)

    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == list(range(len(pairs)))
    assert all(len(batch) <= 3 for batch in batches)
    lengths = [[len(pairs[i][1]) for i in batch] for batch in batches]
    assert lengths == [[5, 10, 20], [50, 100, 200], [1000]]


def test_round_robin_deals_batches_to_every_device():
    batches = [[0], [1], [2], [3], [4]]
    assert BGEReranker._round_robin(batches, 2) == [[[0], [2], [4]], [[1], [3]]]
    assert BGEReranker._round_robin(batches, 1) == [batches]


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return _FakeInputs(texts=list(texts))


class _FakeModel:
    def __call__(self, texts):
        import torch

        return type("Out", (), {"logits": torch.tensor([[float(len(t[1]))] for t in texts])})()


def test_scores_reassemble_in_original_order_across_devices():
    """Parallel workers write into their own index set; the merged list must match `pairs` order."""
    reranker = BGEReranker(device="cpu", batch_size=3)
    reranker._tokenizer = _FakeTokenizer()
    reranker._models = [_FakeModel(), _FakeModel()]
    reranker._devices = ["cpu", "cpu"]

    pairs = [("q", "x" * n) for n in (7, 1, 40, 3, 25, 11, 90, 2)]
    assert reranker.compute_scores(pairs) == [7.0, 1.0, 40.0, 3.0, 25.0, 11.0, 90.0, 2.0]
