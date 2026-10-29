import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "merge_submission_shards",
    Path(__file__).resolve().parent.parent / "scripts" / "merge_submission_shards.py",
)
merge_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_mod)


def test_merge_predictions_sorts_and_keeps_last(tmp_path):
    first = tmp_path / "shard0.json"
    second = tmp_path / "shard1.json"
    first.write_text(
        json.dumps([{"id": 3, "relevant_docs": [1]}, {"id": 1, "relevant_docs": [1]}]),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps([{"id": 2, "relevant_docs": [2]}, {"id": 3, "relevant_docs": [9]}]),
        encoding="utf-8",
    )

    merged = merge_mod.merge_predictions([first, second])

    assert [m["id"] for m in merged] == [1, 2, 3]
    assert merged[2]["relevant_docs"] == [9]
