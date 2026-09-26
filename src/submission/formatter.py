"""Submission generator, validator, and packager for R2AI competition."""

import json
import zipfile
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class ChunkSubmission(BaseModel):
    doc_id: str = Field(
        ..., description="Original document ID (Vietnamese/Chinese BTC ID or English PMID)"
    )
    chunk_text: str = Field(..., description="Exact extracted chunk text from the source document")


class QuerySubmission(BaseModel):
    id: int = Field(..., description="Query integer ID")
    relevant_docs: list[str] = Field(
        default_factory=list, description="List of predicted document IDs"
    )
    relevant_chunks: list[ChunkSubmission] = Field(
        default_factory=list, description="List of predicted chunk objects"
    )


class SubmissionPackage:
    """Manages validation and ZIP packaging of official competition submissions."""

    @staticmethod
    def validate_submission_data(data: list[dict[str, Any]]) -> list[QuerySubmission]:
        """Validates that predictions conform to the official competition schema."""
        validated = []
        for idx, item in enumerate(data):
            try:
                sub = QuerySubmission(**item)
                validated.append(sub)
            except Exception as e:
                raise ValueError(f"Validation failed for record at index {idx}: {item}. Error: {e}")
        return validated

    @classmethod
    def save_and_package(
        cls,
        predictions: list[dict[str, Any]],
        output_dir: str | Path = "outputs/submissions",
        submission_filename: str = "submission.json",
        zip_filename: str | None = None,
    ) -> tuple[Path, Path]:
        """Saves submission JSON and creates a flat ZIP archive without subdirectories.

        Args:
            predictions: List of query prediction dicts.
            output_dir: Directory where output files will be written.
            submission_filename: Name of the json file (default: submission.json).
            zip_filename: Name of the zip file (default: matches json stem + .zip).

        Returns:
            Tuple of (Path to json file, Path to zip file).
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Validate
        validated = cls.validate_submission_data(predictions)
        json_file = out_path / submission_filename

        serialized_data = [item.model_dump() for item in validated]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serialized_data, f, ensure_ascii=False, indent=2)

        if zip_filename is None:
            zip_filename = f"{Path(submission_filename).stem}.zip"
        zip_file = out_path / zip_filename

        # Zip containing strictly one file without nested directories
        with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_file, arcname=submission_filename)

        logger.info(
            f"Successfully packaged {len(validated)} query predictions -> JSON: {json_file} | ZIP: {zip_file}"
        )
        return json_file, zip_file
