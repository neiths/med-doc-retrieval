import json

from src.pipeline import MedicalRetrievalPipeline


def _fake_pipeline():
    pipeline = object.__new__(MedicalRetrievalPipeline)
    pipeline.search_query = lambda q: {
        "relevant_docs": [q],
        "relevant_chunks": [{"doc_id": q, "chunk_text": q}],
    }
    return pipeline


def _write_queries(path, count=10):
    path.write_text(
        "\n".join(json.dumps({"id": i, "query": f"q{i}"}) for i in range(1, count + 1)),
        encoding="utf-8",
    )


def test_offset_and_limit_select_one_shard(tmp_path):
    queries = tmp_path / "queries.jsonl"
    _write_queries(queries)

    predictions = _fake_pipeline().predict_queries_jsonl(queries, offset=3, limit=4)

    assert [p["id"] for p in predictions] == [4, 5, 6, 7]


def test_checkpoint_resumes_without_rescoring(tmp_path):
    queries = tmp_path / "queries.jsonl"
    checkpoint = tmp_path / "ckpt.jsonl"
    _write_queries(queries)

    pipeline = _fake_pipeline()
    first = pipeline.predict_queries_jsonl(queries, offset=0, limit=4, checkpoint_path=checkpoint)
    assert [p["id"] for p in first] == [1, 2, 3, 4]
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 4

    rescored = []
    pipeline.search_query = lambda q: (
        rescored.append(q)
        or {
            "relevant_docs": [q],
            "relevant_chunks": [],
        }
    )
    resumed = pipeline.predict_queries_jsonl(queries, offset=0, limit=6, checkpoint_path=checkpoint)

    assert [p["id"] for p in resumed] == [1, 2, 3, 4, 5, 6]
    assert rescored == ["q5", "q6"]
