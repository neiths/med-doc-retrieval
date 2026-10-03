"""Google Colab End-to-End Runner for Medical Document Retrieval (R2AI 2026).

Designed to run on Google Colab (T4 / L4 / A100 GPU) for fast indexing and inference:
1. Syncs corpus Parquet shards from Hugging Face Bucket: hf://buckets/nieths/ViBioMIR/corpus
2. Builds GPU-accelerated FAISS dense index + BM25 sparse index
3. Performs hybrid retrieval + cross-encoder reranking with F2-optimized top-k parameters
4. Validates and generates submission.zip ready for the leaderboard.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from loguru import logger



def setup_colab_environment():
    """Verifies GPU acceleration and sets up optimal runtime flags."""
    logger.info("=== Checking Colab Hardware & Runtime ===")
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU detected! Please change runtime type to GPU (T4 / L4 / A100).")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Detected GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Ensure sparse BM25 tokenization dependencies are installed
    missing_deps = []
    for pkg in ["rank_bm25", "pyvi", "jieba"]:
        try:
            __import__(pkg)
        except ImportError:
            missing_deps.append(pkg.replace("_", "-"))

    if missing_deps:
        logger.info(f"Installing missing sparse retrieval dependencies: {missing_deps} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing_deps], check=True)



def sync_corpus_from_hf_bucket(
    bucket_uri: str = "hf://buckets/nieths/ViBioMIR/corpus",
    local_dir: Path = Path("data/processed/parquet_corpus"),
    hf_token: str | None = None,
):
    """Downloads all corpus Parquet shards from Hugging Face Storage Bucket using hf CLI."""
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Syncing corpus from {bucket_uri} -> {local_dir} ...")

    hf_bin = shutil.which("hf") or str(Path(sys.executable).parent / "hf")
    if not shutil.which("hf") and not Path(hf_bin).exists():
        logger.info("Installing huggingface_hub[cli]...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[cli]"], check=True)
        hf_bin = shutil.which("hf") or str(Path(sys.executable).parent / "hf")

    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token

    cmd = [hf_bin, "sync", bucket_uri, str(local_dir)]
    logger.info(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        logger.warning("HF sync returned non-zero code. Trying S3 fallback or local check...")

    parquet_files = list(local_dir.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} Parquet shards in {local_dir}")
    for pf in parquet_files:
        logger.info(f"  - {pf.name} ({pf.stat().st_size / (1024 * 1024):.2f} MB)")

    # Also download precomputed pubmed_cache.jsonl if not present locally
    pubmed_cache = Path("data/processed/pubmed_cache.jsonl")
    if not pubmed_cache.exists():
        logger.info("Downloading precomputed pubmed_cache.jsonl from HF bucket for 100% offline retrieval...")
        pubmed_remote = "hf://buckets/nieths/ViBioMIR/processed/pubmed_cache.jsonl"
        subprocess.run([hf_bin, "cp", pubmed_remote, str(pubmed_cache)], env=env)
        if pubmed_cache.exists():
            logger.info(f"Loaded pubmed_cache.jsonl: {pubmed_cache.stat().st_size / (1024*1024):.2f} MB")

    return parquet_files


def run_pipeline(
    queries_file: Path,
    corpus_dir: Path,
    submission_name: str = "submission.json",
    batch_size: int = 64,
    rebuild_indices: bool = True,
):
    """Executes chunking, GPU indexing, hybrid search, reranking, and packages submission."""
    from src.config import load_config
    from src.pipeline import MedicalRetrievalPipeline

    config = load_config("configs/config.yaml")

    # Adapt batch sizes for Colab GPU
    config.embedding.batch_size = batch_size
    config.reranker.batch_size = batch_size
    config.embedding.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.reranker.device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Configuration loaded with F2-tuned parameters:")
    logger.info(f"  top_k_docs:         {config.reranker.top_k_docs}")
    logger.info(f"  top_k_chunks:       {config.reranker.top_k_chunks}")
    logger.info(f"  hybrid_top_k:       {config.retrieval.hybrid_top_k}")
    logger.info(f"  max_chunk_size:     {config.chunking.max_chunk_size} chars")
    logger.info(f"  batch_size:         {batch_size}")

    pipeline = MedicalRetrievalPipeline(config=config)

    dense_exists = (config.paths.indices_dir / "dense_index.faiss").exists()
    sparse_exists = (config.paths.indices_dir / "bm25_index.pkl").exists()

    if rebuild_indices or not dense_exists:
        logger.info(f"Building FAISS & BM25 indices from {corpus_dir}...")
        pipeline.build_indices(articles_file=corpus_dir, output_indices_dir=config.paths.indices_dir)
    else:
        logger.info(f"Using existing indices in {config.paths.indices_dir}...")
        pipeline.load_indices(config.paths.indices_dir)
        # If dense exists but BM25 is missing, quickly construct BM25 from chunks.jsonl
        if pipeline.sparse_index is None:
            chunks_file = Path(config.paths.processed_data_dir) / "chunks.jsonl"
            if chunks_file.exists():
                logger.info(f"Dense index loaded but SparseIndex missing. Building BM25 index from {chunks_file}...")
                import json
                from src.retrieval.sparse_index import SparseIndex
                from src.retrieval.hybrid import HybridRetriever
                chunks = []
                with open(chunks_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            chunks.append(json.loads(line))
                sparse_idx = SparseIndex()
                sparse_idx.build(chunks)
                sparse_idx.save(config.paths.indices_dir)
                pipeline.sparse_index = sparse_idx
                if pipeline.dense_index is not None:
                    pipeline.hybrid_retriever = HybridRetriever(
                        dense_index=pipeline.dense_index,
                        sparse_index=pipeline.sparse_index,
                        fusion_method=config.retrieval.fusion_method,
                        rrf_k=config.retrieval.rrf_k,
                        dense_weight=config.retrieval.dense_weight,
                        sparse_weight=config.retrieval.sparse_weight,
                    )
                logger.info("Successfully built and initialized Sparse BM25 index!")
            else:
                logger.warning("Could not find chunks.jsonl to rebuild SparseIndex. Operating in dense-only mode.")


    logger.info(f"Running inference on queries: {queries_file} ...")
    zip_path = pipeline.generate_submission(
        queries_file=queries_file,
        submission_filename=submission_name,
    )
    logger.info(f"SUCCESS! Submission package created: {zip_path}")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Colab Runner for Medical Document Retrieval.")
    parser.add_argument(
        "--bucket",
        type=str,
        default="hf://buckets/nieths/ViBioMIR/corpus",
        help="HF Storage Bucket URI containing corpus Parquet files.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/processed/parquet_corpus"),
        help="Local directory for Parquet corpus.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/raw/queries.jsonl"),
        help="Path to queries JSONL or Parquet file.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="submission_colab.json",
        help="Submission JSON file name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding and reranking on Colab GPU.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip downloading from HF bucket if data already exists locally.",
    )
    parser.add_argument(
        "--skip-build-index",
        action="store_true",
        help="Skip index building if already built.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Optional Hugging Face access token.",
    )
    parser.add_argument(
        "--sync-index-to-bucket",
        action="store_true",
        help="Upload built indices from data/indices to HF bucket after completion.",
    )
    parser.add_argument(
        "--sync-index-from-bucket",
        action="store_true",
        help="Download pre-built indices from HF bucket to skip embedding if available.",
    )
    parser.add_argument(
        "--index-bucket-uri",
        type=str,
        default="hf://buckets/nieths/ViBioMIR/indices",
        help="HF Storage Bucket URI for persisting pre-built indices.",
    )


    args = parser.parse_args()

    setup_colab_environment()

    if not args.skip_sync:
        sync_corpus_from_hf_bucket(
            bucket_uri=args.bucket,
            local_dir=args.corpus_dir,
            hf_token=args.hf_token,
        )

    # Fallback to local crawled articles if bucket is empty or sync skipped
    corpus_target = args.corpus_dir
    if not list(args.corpus_dir.glob("*.parquet")):
        local_crawled = Path("data/processed/crawled_articles.jsonl")
        if local_crawled.exists():
            logger.info(f"Using local crawled articles fallback: {local_crawled}")
            corpus_target = local_crawled
        else:
            raise FileNotFoundError(f"No corpus found in {args.corpus_dir} or {local_crawled}")

    queries_path = args.queries
    if not queries_path.exists():
        query_parquet = Path("data/raw/vibio_mir/query.parquet")
        if query_parquet.exists():
            queries_path = query_parquet
        else:
            logger.info("Queries file not found locally. Auto-downloading query.parquet from Hugging Face (AIGuruTinix/ViBioMIR)...")
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
                logger.info(f"Successfully downloaded queries to: {queries_path}")
            except Exception as e:
                logger.error(f"Could not auto-download queries: {e}")
                raise FileNotFoundError(f"Queries file not found at {args.queries} and auto-download failed.")


    indices_dir = Path("data/indices")
    rebuild = not args.skip_build_index

    if args.sync_index_from_bucket:
        logger.info(f"Attempting to download pre-built index from {args.index_bucket_uri} -> {indices_dir} ...")
        indices_dir.mkdir(parents=True, exist_ok=True)
        hf_bin = shutil.which("hf") or str(Path(sys.executable).parent / "hf")
        env = os.environ.copy()
        if args.hf_token:
            env["HF_TOKEN"] = args.hf_token
        res = subprocess.run([hf_bin, "sync", args.index_bucket_uri, str(indices_dir)], env=env)
        dense_found = (indices_dir / "dense_index.faiss").exists()
        sparse_found = (indices_dir / "bm25_index.pkl").exists()
        if dense_found and sparse_found:
            logger.info("Found pre-built FAISS + BM25 index from HF bucket! Skipping rebuild.")
            rebuild = False
        elif dense_found:
            logger.info("Found pre-built FAISS dense index from HF bucket. Will reuse dense and verify BM25.")
            rebuild = False

    zip_file = run_pipeline(
        queries_file=queries_path,
        corpus_dir=corpus_target,
        submission_name=args.output_name,
        batch_size=args.batch_size,
        rebuild_indices=rebuild,
    )

    if args.sync_index_to_bucket and indices_dir.exists():
        logger.info(f"Uploading built index from {indices_dir} -> {args.index_bucket_uri} ...")
        hf_bin = shutil.which("hf") or str(Path(sys.executable).parent / "hf")
        env = os.environ.copy()
        if args.hf_token:
            env["HF_TOKEN"] = args.hf_token
        subprocess.run([hf_bin, "sync", str(indices_dir), args.index_bucket_uri], env=env)
        logger.info("Index sync to HF bucket completed successfully!")


    abs_zip = Path(zip_file).resolve()
    # If running on Colab or Kaggle, copy directly to working root for easy 1-click download
    for root_dir in [Path("/content"), Path("/kaggle/working")]:
        if root_dir.exists():
            try:
                dest = root_dir / abs_zip.name
                shutil.copy(abs_zip, dest)
                logger.info(f"Copied submission zip to {root_dir}: {dest}")
            except Exception as e:
                logger.debug(f"Could not copy to {root_dir}: {e}")

    print("\n" + "=" * 60)
    print(f"🎉 COMPLETED! Ready to submit:")
    print(f"ZIP File: {abs_zip}")
    if Path("/content").exists():
        print(f"Colab Shortcut: /content/{abs_zip.name}")
    if Path("/kaggle/working").exists():
        print(f"Kaggle Shortcut: /kaggle/working/{abs_zip.name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
