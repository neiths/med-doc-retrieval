"""Submission generator, validator, and packager for R2AI competition."""

import json
import zipfile
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


def normalize_doc_id(doc_id: Any) -> int | str:
    """Ensures Vietnamese/Chinese corpus IDs (<= 4,420,561) are integers, and PubMed remains str/int."""
    try:
        val = int(doc_id)
        if val <= 4420561:
            return val
        return str(doc_id)
    except (ValueError, TypeError):
        return str(doc_id)


def clean_chunk_text(text: str) -> str:
    """Ensures chunk_text contains purely the verbatim passage from document without any contextual headers."""
    if not text:
        return ""
    if text.startswith("Tiêu đề:") and "\nNội dung: " in text:
        text = text.split("\nNội dung: ", 1)[-1]
    return text.strip()


def stitch_two_texts(
    t1: str,
    t2: str,
    min_overlap: int = 25,
    max_overlap: int = 350,
    max_merged_len: int = 3500,
) -> str | None:
    """Merges two text chunks if they overlap or one contains the other."""
    if not t1 or not t2:
        return None
    if t2 in t1:
        return t1
    if t1 in t2:
        return t2

    limit = min(len(t1), len(t2), max_overlap)
    for ov in range(limit, min_overlap - 1, -1):
        if t1[-ov:] == t2[:ov]:
            merged = t1 + t2[ov:]
            if len(merged) <= max_merged_len:
                return merged

    for ov in range(limit, min_overlap - 1, -1):
        if t2[-ov:] == t1[:ov]:
            merged = t2 + t1[ov:]
            if len(merged) <= max_merged_len:
                return merged

    return None


def stitch_chunks_for_query(
    chunks: list[dict[str, Any]],
    min_overlap: int = 25,
    max_overlap: int = 350,
    max_merged_len: int = 3500,
) -> list[dict[str, Any]]:
    """Groups chunks by doc_id and stitches contiguous/overlapping passages."""
    if not chunks:
        return []

    by_doc: dict[Any, list[str]] = {}
    doc_order: list[Any] = []

    for c in chunks:
        did = normalize_doc_id(c.get("doc_id"))
        text = clean_chunk_text(c.get("chunk_text", ""))
        if not text:
            continue
        if did not in by_doc:
            by_doc[did] = []
            doc_order.append(did)
        by_doc[did].append(text)

    stitched_chunks: list[dict[str, Any]] = []

    for did in doc_order:
        texts = by_doc[did]
        if len(texts) <= 1:
            for t in texts:
                stitched_chunks.append({"doc_id": did, "chunk_text": t})
            continue

        merged = list(texts)
        changed = True
        while changed:
            changed = False
            new_merged = []
            skip = set()
            for i in range(len(merged)):
                if i in skip:
                    continue
                cur = merged[i]
                for j in range(i + 1, len(merged)):
                    if j in skip:
                        continue
                    nxt = merged[j]
                    res = stitch_two_texts(
                        cur,
                        nxt,
                        min_overlap=min_overlap,
                        max_overlap=max_overlap,
                        max_merged_len=max_merged_len,
                    )
                    if res is not None:
                        cur = res
                        skip.add(j)
                        changed = True
                new_merged.append(cur)
            merged = new_merged

        for t in merged:
            clipped_t = t[:max_merged_len].strip() if len(t) > max_merged_len else t
            if clipped_t:
                stitched_chunks.append({"doc_id": did, "chunk_text": clipped_t})

    return stitched_chunks


class ChunkSubmission(BaseModel):
    doc_id: int | str = Field(
        ..., description="Original document ID (Vietnamese/Chinese BTC ID as int or English PMID)"
    )
    chunk_text: str = Field(..., description="Exact extracted chunk text from the source document")


class QuerySubmission(BaseModel):
    id: int = Field(..., description="Query integer ID")
    relevant_docs: list[int | str] = Field(
        default_factory=list, description="List of predicted document IDs"
    )
    relevant_chunks: list[ChunkSubmission] = Field(
        default_factory=list, description="List of predicted chunk objects"
    )


class SubmissionPackage:
    """Manages validation and ZIP packaging of official competition submissions."""

    @staticmethod
    def validate_submission_data(
        data: list[dict[str, Any]],
        stitch_adjacent: bool = True,
    ) -> list[QuerySubmission]:
        """Validates that predictions conform to the official competition schema."""
        validated = []
        for idx, item in enumerate(data):
            try:
                norm_item = dict(item)
                if "relevant_docs" in norm_item:
                    norm_item["relevant_docs"] = [normalize_doc_id(d) for d in norm_item["relevant_docs"]]
                if "relevant_chunks" in norm_item:
                    raw_chunks = norm_item["relevant_chunks"]
                    if stitch_adjacent:
                        raw_chunks = stitch_chunks_for_query(raw_chunks)
                    norm_item["relevant_chunks"] = [
                        {
                            **c,
                            "doc_id": normalize_doc_id(c["doc_id"]),
                            "chunk_text": clean_chunk_text(c.get("chunk_text", "")),
                        }
                        for c in raw_chunks
                    ]
                sub = QuerySubmission(**norm_item)
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
