"""Unit tests for submission validation and packaging."""

import json
import zipfile
from pathlib import Path

import pytest

from src.submission.formatter import SubmissionPackage


def test_submission_package(tmp_path: Path):
    sample_preds = [
        {
            "id": 1,
            "relevant_docs": ["doc_123", "PMC456"],
            "relevant_chunks": [
                {"doc_id": "doc_123", "chunk_text": "Sample chunk 1"},
                {"doc_id": "PMC456", "chunk_text": "Sample chunk 2"},
            ],
        },
        {
            "id": 2,
            "relevant_docs": [],
            "relevant_chunks": [],
        },
    ]

    json_file, zip_file = SubmissionPackage.save_and_package(
        predictions=sample_preds,
        output_dir=tmp_path,
        submission_filename="test_sub.json",
    )

    assert json_file.exists()
    assert zip_file.exists()

    # Verify JSON content
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["id"] == 1
    assert data[1]["relevant_docs"] == []

    # Verify ZIP structure (must contain strictly 1 file at root, no folder)
    with zipfile.ZipFile(zip_file, "r") as zf:
        namelist = zf.namelist()
        assert len(namelist) == 1
        assert namelist[0] == "test_sub.json"


def test_submission_validation_error():
    invalid_preds = [
        {
            "id": "not-an-int",  # invalid id
            "relevant_docs": ["doc_1"],
            "relevant_chunks": [],
        }
    ]
    with pytest.raises(ValueError):
        SubmissionPackage.validate_submission_data(invalid_preds)
