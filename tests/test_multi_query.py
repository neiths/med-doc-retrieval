import pytest

from src.retrieval.multi_query import build_subqueries, fuse_candidates


def test_build_subqueries_splits_on_sentence_boundaries():
    query = (
        "Em bị viêm xung huyết thực quản mạn tính. "
        "Cho em hỏi bệnh này có nặng không. "
        "Em hay bị nghẹn ở cổ và buồn nôn."
    )
    assert build_subqueries(query) == [
        "Em bị viêm xung huyết thực quản mạn tính",
        "Cho em hỏi bệnh này có nặng không",
        "Em hay bị nghẹn ở cổ và buồn nôn",
    ]


def test_build_subqueries_falls_back_to_commas_for_single_sentence():
    query = "Răng có lỗ chấm đen, chảy máu khi thức dậy là sao?"
    assert build_subqueries(query) == [
        "Răng có lỗ chấm đen",
        "chảy máu khi thức dậy là sao",
    ]


def test_build_subqueries_dedupes_and_caps():
    query = (
        "Viêm gan B mạn tính. Viêm gan B mạn tính. "
        "Xơ gan do rượu. Suy gan cấp. Ung thư gan giai đoạn cuối."
    )
    assert build_subqueries(query, max_subqueries=2) == [
        "Viêm gan B mạn tính",
        "Xơ gan do rượu",
    ]


def test_build_subqueries_short_query_returns_empty():
    assert build_subqueries("Đau đầu?") == []


def test_fuse_candidates_prefers_chunks_found_by_multiple_queries():
    primary = [{"chunk_id": 1}, {"chunk_id": 2}]
    sub = [{"chunk_id": 2}, {"chunk_id": 3}]

    fused = fuse_candidates([(1.0, primary), (0.6, sub)], top_k=10, rrf_k=60)

    assert [c["chunk_id"] for c in fused] == [2, 1, 3]
    assert fused[0]["score"] == pytest.approx(1.0 / 62 + 0.6 / 61)
    assert "score" not in primary[0]


def test_fuse_candidates_truncates_to_top_k():
    primary = [{"chunk_id": cid} for cid in range(5)]
    fused = fuse_candidates([(1.0, primary)], top_k=2, rrf_k=60)
    assert [c["chunk_id"] for c in fused] == [0, 1]
