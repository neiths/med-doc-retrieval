"""Merges sharded inference outputs into one validated submission.zip.

Usage:
    python scripts/merge_submission_shards.py \
        --inputs outputs/submissions/shard0.json outputs/submissions/shard1.json \
        --output-name submission.json --expect 1200
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

from src.submission.formatter import SubmissionPackage


def merge_predictions(paths: list[Path]) -> list[dict[str, Any]]:
    """Concatenates shard files, keeping one prediction per query id, sorted by id."""
    merged: dict[int, dict[str, Any]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for item in json.load(f):
                merged[int(item["id"])] = item
    return [merged[qid] for qid in sorted(merged)]


def main():
    parser = argparse.ArgumentParser(description="Merge sharded submission JSON files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-name", type=str, default="submission.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/submissions"))
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="Fail if the merged submission does not contain exactly this many queries.",
    )
    args = parser.parse_args()

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        raise SystemExit(f"Missing shard files: {missing}")

    predictions = merge_predictions(args.inputs)
    if args.expect is not None and len(predictions) != args.expect:
        raise SystemExit(
            f"Expected {args.expect} queries but merged {len(predictions)} "
            f"(ids {predictions[0]['id']}..{predictions[-1]['id']} if non-empty)."
        )

    json_file, zip_file = SubmissionPackage.save_and_package(
        predictions=predictions,
        output_dir=args.output_dir,
        submission_filename=args.output_name,
    )

    abs_zip = Path(zip_file).resolve()
    for dest_dir in [Path("/kaggle/working"), Path("/content")]:
        if dest_dir.exists():
            try:
                shutil.copy(abs_zip, dest_dir / abs_zip.name)
                logger.info(f"Copied {abs_zip.name} to {dest_dir}")
            except Exception:
                pass

    print(f"Merged {len(predictions)} predictions from {len(args.inputs)} shards -> {json_file}")
    print(f"Submission ZIP: {abs_zip}")


if __name__ == "__main__":
    main()
