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
    parser.add_argument(
        "--hybrid-top-k",
        type=int,
        default=None,
        help="Override hybrid_top_k candidates retrieved before reranking (e.g. 350).",
    )
    parser.add_argument(
        "--top-k-chunks",
        type=int,
        default=None,
        help="Override top_k_chunks to output in submission (e.g. 150).",
    )
    parser.add_argument(
        "--top-k-docs",
        type=int,
        default=None,
        help="Override top_k_docs to output in submission (e.g. 70).",
    )
    parser.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=None,
        help="Override max_chunks_per_doc (e.g. 3).",
    )

    args = parser.parse_args()
    setup_inference_env()

    config = load_config("configs/config.yaml")
    config.embedding.batch_size = args.batch_size
    config.reranker.batch_size = args.batch_size
    config.embedding.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.reranker.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.hybrid_top_k is not None:
        config.retrieval.hybrid_top_k = args.hybrid_top_k
        config.retrieval.dense_top_k = args.hybrid_top_k
        config.retrieval.sparse_top_k = args.hybrid_top_k
    if args.top_k_chunks is not None:
        config.reranker.top_k_chunks = args.top_k_chunks
    if args.top_k_docs is not None:
        config.reranker.top_k_docs = args.top_k_docs
    if args.max_chunks_per_doc is not None:
        config.reranker.max_chunks_per_doc = args.max_chunks_per_doc

    logger.info(
        f"Retrieval Settings: hybrid_top_k={config.retrieval.hybrid_top_k} | "
        f"top_k_chunks={config.reranker.top_k_chunks} | "
        f"top_k_docs={config.reranker.top_k_docs} | "
        f"max_chunks_per_doc={getattr(config.reranker, 'max_chunks_per_doc', 2)}"
    )

    if args.disable_pubmed_network:
        logger.info("Live PubMed network requests disabled for maximum speed. Using offline cache only.")
        config.pubmed.source_api = "none"

    logger.info("=== Loading Pipeline and Indices ===")
    pipeline = MedicalRetrievalPipeline(config=config)

    indices_dir = args.indices_dir
    dense_file = None

    # Check candidates in indices_dir
    for candidate in [
        indices_dir / "dense_index_sq8.faiss",
        indices_dir / "dense_index.faiss",
    ]:
        if candidate.exists():
            dense_file = candidate
            break

    if dense_file is None:
        faiss_matches = list(indices_dir.glob("*.faiss")) if indices_dir.exists() else []
        if faiss_matches:
            dense_file = faiss_matches[0]
        elif Path("/kaggle/input").exists():
            k_faiss = list(Path("/kaggle/input").rglob("*.faiss"))
            if k_faiss:
                dense_file = k_faiss[0]
                indices_dir = dense_file.parent
                logger.info(f"Auto-detected FAISS index in Kaggle input: {dense_file}")

    # If still not found, try auto-downloading lightweight SQ8 index from HF Bucket
    if dense_file is None:
        logger.info(f"FAISS index not found in {indices_dir}. Checking for HF_TOKEN to auto-download from HF Bucket...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            try:
                from kaggle_secrets import UserSecretsClient
                hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                pass

        if hf_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                indices_dir.mkdir(parents=True, exist_ok=True)
                download_files = [
                    ("indices/dense_index_sq8.faiss", str(indices_dir / "dense_index_sq8.faiss")),
                    ("indices/chunks_meta_slim.sqlite", str(indices_dir / "chunks_meta_slim.sqlite")),
                    ("indices/dense_metadata_sq8.json", str(indices_dir / "dense_metadata_sq8.json")),
                ]
                logger.info("📥 Auto-downloading lightweight SQ8 index + slim SQLite (~4.9 GB) from HF Bucket 'nieths/ViBioMIR'...")
                api.download_bucket_files(
                    bucket_id="nieths/ViBioMIR",
                    files=download_files,
                    token=hf_token,
                )
                if (indices_dir / "dense_index_sq8.faiss").exists():
                    dense_file = indices_dir / "dense_index_sq8.faiss"
                    logger.info("✅ Successfully downloaded SQ8 index from Hugging Face Bucket!")
            except Exception as e:
                logger.warning(f"Could not auto-download from HF Bucket: {e}")

    if dense_file is None or not dense_file.exists():
        raise FileNotFoundError(
            f"❌ Không tìm thấy file FAISS index (*.faiss) trong '{indices_dir}'!\n"
            f"👉 Cách khắc phục:\n"
            f"1. Hãy đảm bảo bạn đã chạy cell tải file từ HF Bucket trong notebook 'inference-queries.ipynb' trước.\n"
            f"2. Hoặc kiểm tra xem đã tích chọn secret HF_TOKEN trong menu Add-ons -> Secrets trên Kaggle chưa."
        )

    logger.info(f"Using FAISS index: {dense_file} ({dense_file.stat().st_size / (1024*1024):.2f} MB)")
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
        elif Path("/kaggle/input").exists():
            k_query = list(Path("/kaggle/input").rglob("query.parquet"))
            if k_query:
                queries_path = k_query[0]
                logger.info(f"Found query parquet in Kaggle input: {queries_path}")
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
