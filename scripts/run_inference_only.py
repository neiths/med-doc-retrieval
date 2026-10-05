"""Dedicated High-Speed Inference Runner for Medical Document Retrieval (R2AI 2026).

Runs pure inference on GPU without any index rebuilding or corpus parsing:
1. Loads pre-built FAISS dense index + BM25s sparse index via zero-RAM mmap.
2. Uses precomputed enriched queries & PubMed cache for 100% fast offline retrieval (~0.2s/query).
3. Hybrid Retrieval (RRF) + BGE-Reranker-Large cross-encoder on GPU.
4. Produces validated submission.json and packages submission.zip ready to submit.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
from loguru import logger
from tqdm import tqdm

from src.config import load_config
from src.pipeline import MedicalRetrievalPipeline
from src.submission.formatter import SubmissionPackage


def setup_inference_env():
    """Configures GPU and PyTorch flags for maximum inference throughput."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        logger.info(
            f"Using GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) | Allocated: {allocated_gb:.2f} GB | Reserved: {reserved_gb:.2f} GB"
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)
    else:
        logger.warning("No CUDA GPU detected! Running inference on CPU (slower).")


def main():
    parser = argparse.ArgumentParser(description="Dedicated Inference-Only Runner.")
    parser.add_argument(
        "--indices-dir",
        type=Path,
        default=Path("data/indices"),
        help="Directory containing dense_index.faiss and bm25s_index.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Path to queries file (queries_enriched.jsonl or query.parquet). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="submission.json",
        help="Output submission filename.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for cross-encoder reranker on GPU (default 16 to avoid OOM).",
    )
    parser.add_argument(
        "--disable-pubmed-network",
        action="store_true",
        help="Skip slow live PubMed HTTP API calls, rely purely on local cache and corpus.",
    )

    args = parser.parse_args()
    setup_inference_env()

    config = load_config("configs/config.yaml")
    config.embedding.batch_size = args.batch_size
    config.reranker.batch_size = args.batch_size
    config.embedding.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.reranker.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.disable_pubmed_network:
        logger.info("Live PubMed network requests disabled for maximum speed. Using offline cache only.")
        config.pubmed.source_api = "none"

    logger.info("=== Loading Pipeline and Indices ===")
    pipeline = MedicalRetrievalPipeline(config=config)

    indices_dir = args.indices_dir
    dense_file = indices_dir / "dense_index.faiss"
    bm25s_dir = indices_dir / "bm25s_index"

    if not dense_file.exists():
        raise FileNotFoundError(f"FAISS index {dense_file} not found in {indices_dir}!")

    pipeline.load_indices(indices_dir)

    # Resolve queries path
    queries_path = args.queries
    if queries_path is None or not queries_path.exists():
        enriched_file = Path("data/processed/queries_enriched.jsonl")
        query_parquet = Path("data/raw/vibio_mir/query.parquet")
        raw_queries = Path("data/raw/queries.jsonl")

        if enriched_file.exists():
            queries_path = enriched_file
            logger.info(f"Using precomputed enriched queries: {queries_path}")
        elif query_parquet.exists():
            queries_path = query_parquet
            logger.info(f"Using query parquet: {queries_path}")
        elif raw_queries.exists():
            queries_path = raw_queries
            logger.info(f"Using raw queries: {queries_path}")
        else:
            logger.info("Queries file not found locally. Auto-downloading query.parquet from Hugging Face...")
            try:
                from huggingface_hub import hf_hub_download
                query_parquet.parent.mkdir(parents=True, exist_ok=True)
                downloaded = hf_hub_download(
                    repo_id="AIGuruTinix/ViBioMIR",
                    filename="query.parquet",
                    repo_type="dataset",
                    local_dir=str(query_parquet.parent),
                )
                queries_path = Path(downloaded)
                logger.info(f"Downloaded queries to: {queries_path}")
            except Exception as e:
                raise FileNotFoundError(f"Could not find or download queries: {e}")

    logger.info(f"Running batch prediction on {queries_path}...")
    predictions = pipeline.predict_queries_jsonl(queries_path)

    logger.info("Packaging official submission...")
    json_file, zip_file = SubmissionPackage.save_and_package(
        predictions=predictions,
        output_dir=config.paths.submissions_dir,
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

    print("\n" + "=" * 60)
    print(f"🎉 INFERENCE COMPLETE!")
    print(f"Submission ZIP: {abs_zip} ({abs_zip.stat().st_size / (1024*1024):.2f} MB)")
    if Path("/kaggle/working").exists():
        print(f"Kaggle Download: /kaggle/working/{abs_zip.name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
