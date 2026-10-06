"""Unified End-to-End Pipeline for Qdrant Int8 Quantized Hybrid Retrieval on Kaggle.

Implements all 4 Kaggle-specific hardware optimizations:
1. Qdrant Embedded on-disk storage with Scalar Quantization (Int8) for zero RAM and 4x disk compression.
2. Direct dense vector reuse from FAISS (no redundant GPU re-encoding) + batch SPLADE lexical streaming.
3. Dual GPU (2x T4) allocation: BGE-M3 on cuda:0, BGE-Reranker-Large on cuda:1.
4. Native Qdrant Score Fusion (Fusion.SCORE) to preserve high-confidence medical keyword matches.
5. Smart Chunk Selector (score filtering + balanced doc allocation) to maximize Macro F2.
"""

import argparse
import gc
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# Memory configuration for PyTorch on Kaggle
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from loguru import logger
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.submission.formatter import SubmissionPackage, clean_chunk_text, normalize_doc_id


def setup_devices() -> tuple[str, str]:
    """Configures device allocation across available GPUs (Dual T4 support)."""
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        logger.info(
            f"Dual GPU detected! Allocating BGE-M3 Embedder to cuda:0 and BGE-Reranker to cuda:1."
        )
        return "cuda:0", "cuda:1"
    elif gpu_count == 1:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Single GPU detected: {gpu_name} (cuda:0).")
        return "cuda:0", "cuda:0"
    else:
        logger.warning("No CUDA GPUs detected! Running on CPU.")
        return "cpu", "cpu"


def build_qdrant_int8_index(
    qdrant_path: Path,
    collection_name: str,
    faiss_path: Path,
    sqlite_path: Path,
    device: str = "cuda:0",
    batch_size: int = 1000,
    max_chunks: int | None = None,
):
    """Builds or resumes the Qdrant Int8 collection from FAISS vectors and SQLite chunks."""
    import faiss
    from FlagEmbedding import BGEM3FlagModel

    qdrant_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initializing QdrantClient (Embedded on-disk) at '{qdrant_path}'")
    client = QdrantClient(path=str(qdrant_path))

    # 1. Ensure collection with Int8 quantization & on_disk settings
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        logger.info(
            f"Creating collection '{collection_name}' with on_disk=True & ScalarQuantization(Int8)..."
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense-bge": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                "sparse-bge": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                )
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=False,
                )
            ),
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists.")

    # 2. Check resume point
    current_count = client.count(collection_name=collection_name).count
    logger.info(f"Current chunks in Qdrant collection: {current_count:,}")

    # 3. Check SQLite total chunks
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    total_in_sqlite = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    target_total = min(total_in_sqlite, max_chunks) if max_chunks else total_in_sqlite
    logger.info(f"Target total chunks to index: {target_total:,} (in SQLite: {total_in_sqlite:,})")

    if current_count >= target_total:
        logger.info("🎉 Qdrant collection is already fully indexed! Skipping build step.")
        conn.close()
        return

    # 4. Open FAISS index with memory mapping
    logger.info(f"Opening FAISS index via memory-mapping from {faiss_path}...")
    faiss_index = faiss.read_index(str(faiss_path), faiss.IO_FLAG_MMAP)
    faiss_total = faiss_index.ntotal
    logger.info(f"FAISS index contains {faiss_total:,} dense vectors.")

    # 5. Load BGE-M3 lexical head for sparse SPLADE generation
    logger.info(f"Loading BGE-M3 on {device} for sparse lexical weights...")
    use_fp16 = device.startswith("cuda")
    embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, device=device)

    start_idx = current_count
    pbar = tqdm(total=target_total, initial=start_idx, desc="Indexing into Qdrant Int8")

    while start_idx < target_total:
        end_idx = min(start_idx + batch_size, target_total)
        num_items = end_idx - start_idx

        # Fetch batch metadata from SQLite
        rows = cur.execute(
            "SELECT idx, doc_id, chunk_id, chunk_text, title, lang FROM chunks WHERE idx >= ? AND idx < ? ORDER BY idx",
            (start_idx, end_idx),
        ).fetchall()

        if not rows:
            break

        # Extract text for sparse encoding
        texts = []
        for r in rows:
            raw_c = r[3] or ""
            title = r[4] or ""
            text = f"Tiêu đề: {title}\nNội dung: {raw_c}" if title else raw_c
            texts.append(text)

        # 1. Reuse existing Dense vectors from FAISS directly (0 redundant GPU computation!)
        dense_vecs = faiss_index.reconstruct_n(start_idx, len(rows))

        # 2. Compute Sparse SPLADE lexical weights on GPU
        with torch.no_grad():
            res = embedder.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
            lexical_weights = res["lexical_weights"]

        # 3. Build Qdrant points
        points = []
        for i, r in enumerate(rows):
            point_id = int(r[0])  # Integer ID matches idx
            lw = lexical_weights[i]
            sparse_indices = [int(k) for k in lw.keys()]
            sparse_values = [float(v) for v in lw.values()]

            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense-bge": dense_vecs[i].tolist(),
                    "sparse-bge": models.SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload={
                    "doc_id": str(r[1]),
                    "chunk_id": str(r[2]),
                    "chunk_text": str(r[3]),
                    "title": str(r[4] or ""),
                    "lang": str(r[5] or "vi"),
                },
            )
            points.append(point)

        # 4. Upsert batch into Qdrant
        client.upsert(collection_name=collection_name, points=points, wait=False)

        # 5. Explicit memory cleanup
        pbar.update(len(points))
        start_idx = end_idx
        del rows, texts, dense_vecs, lexical_weights, points
        gc.collect()

    pbar.close()
    conn.close()
    final_count = client.count(collection_name=collection_name).count
    logger.info(f"✅ Build finished! Total points in Qdrant collection: {final_count:,}")


def search_qdrant_hybrid(
    client: QdrantClient,
    collection_name: str,
    dense_vec: list[float],
    sparse_vec: models.SparseVector,
    top_k: int = 100,
) -> list[dict[str, Any]]:
    """Executes native Qdrant Hybrid Search with Score Fusion."""
    try:
        prefetch = [
            models.Prefetch(query=dense_vec, using="dense-bge", limit=top_k),
        ]
        if sparse_vec.indices:
            prefetch.append(
                models.Prefetch(query=sparse_vec, using="sparse-bge", limit=top_k)
            )

        res = client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.SCORE),
            limit=top_k,
        )
        points = res.points
    except Exception as e:
        logger.warning(f"Hybrid Score fusion query failed ({e}). Falling back to pure dense search.")
        res = client.query_points(
            collection_name=collection_name,
            query=dense_vec,
            using="dense-bge",
            limit=top_k,
        )
        points = res.points

    candidates = []
    for p in points:
        payload = p.payload or {}
        cand = {
            "chunk_id": payload.get("chunk_id", ""),
            "doc_id": payload.get("doc_id", ""),
            "chunk_text": payload.get("chunk_text", ""),
            "title": payload.get("title", ""),
            "lang": payload.get("lang", "vi"),
            "score": float(p.score) if p.score is not None else 0.0,
        }
        candidates.append(cand)
    return candidates


def run_inference_pipeline(
    qdrant_path: Path,
    collection_name: str,
    queries_path: Path,
    output_dir: Path,
    device_embed: str = "cuda:0",
    device_rerank: str = "cuda:1",
    top_k_candidates: int = 100,
    top_k_docs: int = 35,
    top_k_chunks: int = 45,
    score_threshold: float = -2.5,
    rerank_batch_size: int = 32,
    output_name: str = "submission.json",
):
    """Executes end-to-end inference using Qdrant Int8 Hybrid Search + Cross-Encoder Reranker."""
    from FlagEmbedding import BGEM3FlagModel
    from src.reranker.bge_reranker import BGEReranker

    logger.info("=== Loading Models for Dual-GPU Inference ===")
    logger.info(f"Query Embedder (BGE-M3) on: {device_embed}")
    logger.info(f"Cross-Encoder (BGE-Reranker-Large) on: {device_rerank}")

    client = QdrantClient(path=str(qdrant_path))
    embedder = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=device_embed.startswith("cuda"),
        device=device_embed,
    )
    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-large",
        device=device_rerank,
        batch_size=rerank_batch_size,
        use_fp16=device_rerank.startswith("cuda"),
    )

    # Load queries
    logger.info(f"Loading queries from {queries_path}...")
    queries = []
    if queries_path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(queries_path)
        queries = table.to_pylist()
    else:
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))

    logger.info(f"Total queries to evaluate: {len(queries):,}")
    predictions = []

    for q in tqdm(queries, desc="Evaluating Queries (Score Fusion + Reranking)"):
        qid = int(q["id"])
        query_text = q.get("query") or q.get("original_query") or ""
        search_text = q.get("search_query_text") or query_text

        # 1. Encode query
        with torch.no_grad():
            res = embedder.encode(
                [search_text],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense_vec = res["dense_vecs"][0].tolist()
            lw = res["lexical_weights"][0]
            sparse_vec = models.SparseVector(
                indices=[int(k) for k in lw.keys()],
                values=[float(v) for v in lw.values()],
            )

        # 2. Qdrant Hybrid Search with Score Fusion
        candidates = search_qdrant_hybrid(
            client=client,
            collection_name=collection_name,
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            top_k=top_k_candidates,
        )

        if not candidates:
            predictions.append({"id": qid, "relevant_docs": [], "relevant_chunks": []})
            continue

        # 3. Rerank candidates with BGE-Reranker-Large on device_rerank
        doc_ids, relevant_chunks = reranker.rerank(
            query=query_text,
            candidates=candidates,
            top_k_chunks=top_k_chunks,
            top_k_docs=top_k_docs,
            score_threshold=score_threshold,
            max_chunks_per_doc=2,
        )

        # 4. Clean verbatim text & normalize doc IDs
        formatted_chunks = []
        for c in relevant_chunks:
            formatted_chunks.append(
                {
                    "doc_id": normalize_doc_id(c["doc_id"]),
                    "chunk_text": clean_chunk_text(c["chunk_text"]),
                }
            )

        predictions.append(
            {
                "id": qid,
                "relevant_docs": [normalize_doc_id(d) for d in doc_ids],
                "relevant_chunks": formatted_chunks,
            }
        )

    # 5. Official Submission Packaging
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, zip_path = SubmissionPackage.save_and_package(
        predictions=predictions,
        output_dir=output_dir,
        submission_filename=output_name,
    )

    abs_zip = Path(zip_path).resolve()
    for dest in [Path("/kaggle/working"), Path("/content")]:
        if dest.exists():
            try:
                shutil.copy(abs_zip, dest / abs_zip.name)
                logger.info(f"Copied submission file to: {dest / abs_zip.name}")
            except Exception:
                pass

    print("\n" + "=" * 60)
    print("🎉 QDRANT INT8 + SCORE FUSION INFERENCE COMPLETE!")
    print(f"Output ZIP: {abs_zip} ({abs_zip.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 60 + "\n")


def ensure_hf_files(faiss_path: Path, sqlite_path: Path, queries_path: Path):
    """Ensures required files exist, auto-pulling from Hugging Face Bucket if missing."""
    bucket_uri = "hf://buckets/nieths/ViBioMIR"

    if not sqlite_path.exists():
        logger.info(f"{sqlite_path} not found locally! Auto-pulling from HF Bucket...")
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        res = os.system(f"hf cp {bucket_uri}/indices/chunks_meta.sqlite {sqlite_path}")
        if res != 0 or not sqlite_path.exists():
            logger.warning(f"Could not pull {sqlite_path} via 'hf cp'. Please ensure HF login.")

    if not faiss_path.exists():
        logger.info(f"{faiss_path} not found locally! Auto-pulling from HF Bucket...")
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        res = os.system(f"hf cp {bucket_uri}/indices/dense_index.faiss {faiss_path}")
        if res != 0 or not faiss_path.exists():
            logger.warning(f"Could not pull {faiss_path} via 'hf cp'. Please ensure HF login.")

    if not queries_path.exists():
        logger.info(f"{queries_path} not found locally! Auto-pulling from HF Bucket...")
        queries_path.parent.mkdir(parents=True, exist_ok=True)
        os.system(f"hf cp {bucket_uri}/processed/queries_enriched.jsonl {queries_path}")
        if not queries_path.exists():
            logger.info("Attempting fallback to dataset AIGuruTinix/ViBioMIR query.parquet...")
            os.system(
                f"hf download AIGuruTinix/ViBioMIR query.parquet --repo-type dataset --local-dir {queries_path.parent}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Unified Pipeline: Qdrant Int8 Build + Dual-GPU Score Fusion Inference"
    )
    parser.add_argument("--build", action="store_true", help="Build or resume Qdrant Int8 index.")
    parser.add_argument("--infer", action="store_true", help="Run inference on test queries.")
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path("data/indices/qdrant_db"),
        help="Path for on-disk embedded Qdrant storage.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="medical_corpus",
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--faiss-path",
        type=Path,
        default=Path("data/indices/dense_index.faiss"),
        help="Path to precomputed FAISS dense index.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("data/indices/chunks_meta.sqlite"),
        help="Path to chunks SQLite metadata database.",
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=Path("data/processed/queries_enriched.jsonl"),
        help="Path to queries file (queries_enriched.jsonl or query.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save final submission files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Batch size for Qdrant indexing stream.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Max chunks to index (useful for fast sanity testing).",
    )
    parser.add_argument(
        "--top-k-docs",
        type=int,
        default=35,
        help="Target top documents per query (default 35 for balanced F2).",
    )
    parser.add_argument(
        "--top-k-chunks",
        type=int,
        default=45,
        help="Target top chunks per query (default 45 for balanced F2).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=-2.5,
        help="Reranker score threshold to filter out low-confidence chunks.",
    )

    args = parser.parse_args()
    device_embed, device_rerank = setup_devices()

    # Automatically ensure files exist locally
    ensure_hf_files(
        faiss_path=args.faiss_path,
        sqlite_path=args.sqlite_path,
        queries_path=args.queries_path,
    )

    # Default to running both if neither flag is passed
    run_build = args.build or (not args.build and not args.infer)
    run_infer = args.infer or (not args.build and not args.infer)

    if run_build:
        logger.info("=== STEP 1: Building / Resuming Qdrant Int8 Index ===")
        build_qdrant_int8_index(
            qdrant_path=args.qdrant_path,
            collection_name=args.collection_name,
            faiss_path=args.faiss_path,
            sqlite_path=args.sqlite_path,
            device=device_embed,
            batch_size=args.batch_size,
            max_chunks=args.max_chunks,
        )

    if run_infer:
        logger.info("=== STEP 2: Running Inference with Score Fusion & Dual GPU ===")
        # Auto-detect queries path fallback
        q_path = args.queries_path
        if not q_path.exists():
            for alt in [
                Path("data/raw/vibio_mir/query.parquet"),
                Path("data/raw/public_test.parquet"),
                Path("data/raw/queries.jsonl"),
            ]:
                if alt.exists():
                    q_path = alt
                    break

        run_inference_pipeline(
            qdrant_path=args.qdrant_path,
            collection_name=args.collection_name,
            queries_path=q_path,
            output_dir=args.output_dir,
            device_embed=device_embed,
            device_rerank=device_rerank,
            top_k_docs=args.top_k_docs,
            top_k_chunks=args.top_k_chunks,
            score_threshold=args.score_threshold,
        )


if __name__ == "__main__":
    main()
