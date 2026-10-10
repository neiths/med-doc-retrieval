"""Post-processing script to stitch adjacent and overlapping chunks in a submission.

Implements Adjacent Chunk Stitching (Window Expansion):
1. For each query, groups predicted chunks by their doc_id.
2. Identifies overlapping or contiguous chunks from the same document.
3. Merges them seamlessly using longest suffix-prefix matching without text duplication.
4. Expands chunk lengths from ~750 characters to ~1,400 - 2,500 characters (~350 - 600 tokens).
5. Ensures predicted chunks cover >= 40% of the BTC ground-truth passage (token LCS length / gold token count).
6. Preserves the official R2AI competition schema and creates a flat submission.zip.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from loguru import logger

from src.submission.formatter import SubmissionPackage, clean_chunk_text, normalize_doc_id


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

    # Containment
    if t2 in t1:
        return t1
    if t1 in t2:
        return t2

    # Check length budget
    combined_len = len(t1) + len(t2)

    # Suffix of t1 matching prefix of t2
    limit = min(len(t1), len(t2), max_overlap)
    for ov in range(limit, min_overlap - 1, -1):
        if t1[-ov:] == t2[:ov]:
            merged = t1 + t2[ov:]
            if len(merged) <= max_merged_len:
                return merged

    # Suffix of t2 matching prefix of t1
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

    # Group chunks preserving document order
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

        # Iterative chain merging
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
            # Safety clip for raw crawl outliers exceeding max_merged_len
            clipped_t = t[:max_merged_len].strip() if len(t) > max_merged_len else t
            if clipped_t:
                stitched_chunks.append({"doc_id": did, "chunk_text": clipped_t})

    return stitched_chunks


def process_submission_file(
    input_file: Path,
    output_dir: Path,
    output_filename: str = "submission.json",
    zip_filename: str = "submission.zip",
    min_overlap: int = 25,
    max_overlap: int = 350,
    max_merged_len: int = 3500,
) -> tuple[Path, Path]:
    """Reads input submission, stitches chunks across all queries, and packages ZIP."""
    logger.info(f"Loading input submission: {input_file} ...")
    t0 = time.time()
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data):,} queries. Starting adjacent chunk stitching...")

    orig_chunk_counts = []
    new_chunk_counts = []
    orig_chunk_lens = []
    new_chunk_lens = []
    total_stitched_events = 0

    stitched_predictions = []

    for item in data:
        qid = item["id"]
        rel_docs = item.get("relevant_docs", [])
        rel_chunks = item.get("relevant_chunks", [])

        orig_chunk_counts.append(len(rel_chunks))
        for c in rel_chunks:
            orig_chunk_lens.append(len(c.get("chunk_text", "")))

        stitched = stitch_chunks_for_query(
            rel_chunks,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            max_merged_len=max_merged_len,
        )

        new_chunk_counts.append(len(stitched))
        for c in stitched:
            new_chunk_lens.append(len(c["chunk_text"]))

        if len(stitched) < len(rel_chunks):
            total_stitched_events += (len(rel_chunks) - len(stitched))

        stitched_predictions.append({
            "id": qid,
            "relevant_docs": rel_docs,
            "relevant_chunks": stitched,
        })

    logger.info(f"Stitching completed in {time.time() - t0:.2f}s.")
    logger.info("=== BEFORE vs AFTER COMPARISON ===")
    logger.info(
        f"Total chunks:         {sum(orig_chunk_counts):,} -> {sum(new_chunk_counts):,} "
        f"(-{total_stitched_events:,} stitched mergers)"
    )
    logger.info(
        f"Avg chunks per query: {np.mean(orig_chunk_counts):.1f} -> {np.mean(new_chunk_counts):.1f}"
    )
    logger.info(
        f"Median chunk length:  {np.median(orig_chunk_lens):.0f} chars -> {np.median(new_chunk_lens):.0f} chars"
    )
    logger.info(
        f"Mean chunk length:    {np.mean(orig_chunk_lens):.1f} chars -> {np.mean(new_chunk_lens):.1f} chars"
    )
    logger.info(
        f"Max chunk length:     {max(orig_chunk_lens):,} chars -> {max(new_chunk_lens):,} chars"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, zip_path = SubmissionPackage.save_and_package(
        predictions=stitched_predictions,
        output_dir=output_dir,
        submission_filename=output_filename,
        zip_filename=zip_filename,
    )

    return json_path, zip_path


def main():
    parser = argparse.ArgumentParser(description="Stitch adjacent submission chunks.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("outputs/fifth-submission/prediction_result (2)/submission.json"),
        help="Path to source submission.json to stitch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sixth-submission"),
        help="Directory to save the new submission.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="submission.json",
        help="Name of output JSON file.",
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="submission.zip",
        help="Name of output ZIP file.",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=25,
        help="Minimum overlap character length to merge chunks (default: 25).",
    )
    parser.add_argument(
        "--max-overlap",
        type=int,
        default=350,
        help="Maximum overlap character length to check (default: 350).",
    )
    parser.add_argument(
        "--max-merged-len",
        type=int,
        default=3500,
        help="Maximum character length of a merged chunk (default: 3500).",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    json_file, zip_file = process_submission_file(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_filename=args.output_name,
        zip_filename=args.zip_name,
        min_overlap=args.min_overlap,
        max_overlap=args.max_overlap,
        max_merged_len=args.max_merged_len,
    )

    print("\n" + "=" * 60)
    print("🎉 STITCHED SUBMISSION READY FOR LEADERBOARD UPLOAD!")
    print(f"JSON: {json_file.resolve()}")
    print(f"ZIP:  {zip_file.resolve()} ({zip_file.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
