"""Incremental Index Updater for Medical Document Retrieval (R2AI 2026).

Appends new crawled articles/shards directly to existing FAISS dense index,
SQLite metadata, and BM25s sparse index WITHOUT re-indexing from scratch:

1. Deduplication: Checks existing doc_ids in SQLite to avoid duplicate indexing.
2. Chunking: Chunks only the new articles using DocumentChunker.
3. SQLite Append: Inserts new chunk metadata starting from (max_idx + 1).
4. FAISS Append: Embeds only the new chunks using BGE-M3 (GPU fp16) and calls index.add().
5. BM25s Update: Ultra-fast streaming rebuild of lexical BM25s from SQLite (~2 mins on CPU).
6. Optional Sync: Uploads updated index back to Hugging Face Storage Bucket.
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import faiss
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from src.config import load_config
from src.embedding.bge_m3 import BGEM3Embedder
from src.ingestion.chunker import DocumentChunker
from src.ingestion.cleaner import normalize_text

RE_WORDS = re.compile(r"\w+")


class SqliteTokenStream:
    """Zero-RAM streaming iterator over chunks_meta.sqlite for bm25s indexing."""

    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)

    def __iter__(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for row in cur.execute("SELECT chunk_text, title, lang FROM chunks ORDER BY idx"):
            raw_c = row[0] or ""
            title = row[1] or ""
            lang = row[2] or "vi"
            text = f"Tiêu đề: {title}\nNội dung: {raw_c}" if title else raw_c
            t_lower = text.lower()
            if lang == "zh" or any("\u4e00" <= c <= "\u9fff" for c in text[:50]):
                chars = [c for c in t_lower if not c.isspace()]
                yield chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
            else:
                yield RE_WORDS.findall(t_lower)
        conn.close()


def load_new_articles(files: list[Path]) -> list[dict[str, Any]]:
    """Loads raw articles from JSONL or Parquet files."""
    articles = []
    for f in files:
        logger.info(f"Reading new articles from {f}...")
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(f, columns=["doc_id", "title", "text", "lang", "url"])
            articles.extend(table.to_pylist())
        else:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        articles.append(json.loads(line))
    return articles


def rebuild_bm25s(sqlite_path: Path, output_bm25s_dir: Path):
    """Rebuilds BM25s index in ~2 mins using streaming token iterator."""
    import bm25s

    logger.info(f"Rebuilding BM25s sparse index from {sqlite_path}...")
    t0 = time.time()
    stream = SqliteTokenStream(sqlite_path)

    retriever = bm25s.BM25(k1=1.5, b=0.75)
    retriever.index(stream)

    output_bm25s_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output_bm25s_dir))
    logger.info(f"BM25s sparse index rebuild completed in {time.time() - t0:.1f}s -> {output_bm25s_dir}")


def load_hf_token(cli_token: str | None = None) -> str | None:
    """Loads HF token from CLI argument, environment variable, or .env file."""
    if cli_token and cli_token.strip():
        return cli_token.strip()
    env_token = os.environ.get("HF_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    env_file = Path(".env")
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Incremental Index Appender for ViBioMIR.")
    parser.add_argument(
        "--new-data",
        type=Path,
        nargs="+",
        required=True,
        help="Path to new JSONL or Parquet files containing newly crawled articles.",
    )
    parser.add_argument(
        "--indices-dir",
        type=Path,
        default=Path("data/indices"),
        help="Directory containing existing dense_index.faiss, chunks_meta.sqlite, and bm25s_index.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Configuration YAML file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size for GPU (default: 64).",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip rebuilding BM25s sparse index (only update FAISS & SQLite).",
    )
    parser.add_argument(
        "--sync-to-bucket",
        action="store_true",
        default=True,
        help="Auto-upload updated index to Hugging Face Storage Bucket after completion (default: True).",
    )
    parser.add_argument(
        "--no-sync",
        dest="sync_to_bucket",
        action="store_false",
        help="Disable automatic upload to Hugging Face Storage Bucket.",
    )
    parser.add_argument(
        "--download-index-first",
        action="store_true",
        default=True,
        help="Auto-download existing index from HF Bucket if missing locally (default: True).",
    )
    parser.add_argument(
        "--no-download-index",
        dest="download_index_first",
        action="store_false",
        help="Disable auto-downloading existing index from HF Bucket.",
    )
    parser.add_argument(
        "--bucket-uri",
        type=str,
        default="hf://buckets/nieths/ViBioMIR/indices",
        help="HF Bucket URI to sync indices.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face Token (or set HF_TOKEN environment variable).",
    )

    args = parser.parse_args()

    # Configure HF Token and auto-login if available
    hf_token = load_hf_token(args.hf_token)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Successfully authenticated with Hugging Face Hub.")
        except Exception as e:
            logger.debug(f"HF login notice: {e}")

    hf_bin = shutil.which("hf") or str(Path(sys.executable).parent / "hf") or "hf"

    # Verify existing indices
    faiss_path = args.indices_dir / "dense_index.faiss"
    sqlite_path = args.indices_dir / "chunks_meta.sqlite"
    bm25s_path = args.indices_dir / "bm25s_index"

    # Auto-download from HF Bucket if missing on Kaggle
    if not faiss_path.exists() or not sqlite_path.exists():
        if args.download_index_first:
            logger.info(f"Existing index not found locally in {args.indices_dir}. Auto-downloading from {args.bucket_uri} ...")
            args.indices_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if hf_token:
                env["HF_TOKEN"] = hf_token
            res = subprocess.run([hf_bin, "sync", args.bucket_uri, str(args.indices_dir)], env=env)
            if res.returncode == 0:
                logger.info("Successfully synced baseline index from HF Bucket!")
            else:
                logger.warning(f"Download returned exit code {res.returncode}. Proceeding to verify local files...")

        if not faiss_path.exists() or not sqlite_path.exists():
            raise FileNotFoundError(
                f"Existing index files not found in {args.indices_dir}! "
                f"Expected {faiss_path} and {sqlite_path}. "
                "Please check your internet connection or HF_TOKEN."
            )

    config = load_config(args.config)
    chunker = DocumentChunker(
        max_chunk_size=config.chunking.max_chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        min_chunk_size=config.chunking.min_chunk_size,
        split_by_sentences=config.chunking.split_by_sentences,
        enable_contextual=config.chunking.enable_contextual,
    )

    # 1. Connect to SQLite and check existing counts
    logger.info(f"Connecting to SQLite metadata index: {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*), COALESCE(MAX(idx), -1) FROM chunks")
    current_count, current_max_idx = cur.fetchone()
    logger.info(f"Existing chunks in SQLite: {current_count:,} (max idx: {current_max_idx})")

    # Load existing doc_ids to prevent duplicate insertion
    cur.execute("SELECT DISTINCT doc_id FROM chunks")
    existing_doc_ids = {str(r[0]) for r in cur.fetchall()}
    logger.info(f"Loaded {len(existing_doc_ids):,} existing unique doc_ids.")

    # 2. Load and filter new articles
    raw_articles = load_new_articles(args.new_data)
    logger.info(f"Loaded total {len(raw_articles):,} candidate articles from input files.")

    new_articles = []
    for art in raw_articles:
        did = str(art.get("doc_id") or art.get("id") or "")
        if not did:
            continue
        if did in existing_doc_ids:
            continue
        text = art.get("text", "")
        if len(text.strip()) < config.chunking.min_chunk_size:
            continue
        new_articles.append(art)
        existing_doc_ids.add(did)  # prevent duplicates within input files

    logger.info(f"New unique articles to chunk & index: {len(new_articles):,}")
    if not new_articles:
        logger.info("No new articles to add! All articles already exist in index.")
        return

    # 3. Chunk new articles
    logger.info("Chunking new articles...")
    new_chunks = []
    for art in tqdm(new_articles, desc="Chunking"):
        did = str(art.get("doc_id") or art.get("id"))
        meta = {
            "title": art.get("title", ""),
            "url": art.get("url", ""),
            "source": art.get("source", "crawled"),
        }
        chunks = chunker.chunk_document(
            doc_id=did,
            text=art.get("text", ""),
            metadata=meta,
            lang=art.get("lang"),
        )
        new_chunks.extend(chunks)

    logger.info(f"Generated {len(new_chunks):,} new chunks.")
    if not new_chunks:
        logger.info("No valid chunks produced. Exiting.")
        return

    # 4. Append new chunks to SQLite metadata
    logger.info(f"Appending {len(new_chunks):,} chunks to SQLite...")
    start_idx = current_max_idx + 1
    rows_to_insert = []
    for offset, c in enumerate(new_chunks):
        chunk_idx = start_idx + offset
        rows_to_insert.append(
            (
                chunk_idx,
                str(c.doc_id),
                str(c.chunk_id),
                c.chunk_text,
                c.metadata.get("title", "") if c.metadata else "",
                c.lang or "vi",
            )
        )

    cur.executemany(
        "INSERT INTO chunks (idx, doc_id, chunk_id, chunk_text, title, lang) VALUES (?, ?, ?, ?, ?, ?)",
        rows_to_insert,
    )
    conn.commit()
    conn.close()
    logger.info(f"SQLite successfully updated. New total chunks: {start_idx + len(new_chunks):,}")

    # 5. Encode new chunks and append to FAISS
    logger.info("Loading existing FAISS index (read-write mode)...")
    faiss_index = faiss.read_index(str(faiss_path))
    logger.info(f"Existing FAISS vector count: {faiss_index.ntotal:,}")

    if faiss_index.ntotal != current_count:
        logger.warning(
            f"Count mismatch: FAISS has {faiss_index.ntotal:,} vectors, "
            f"but SQLite had {current_count:,} chunks! Proceeding with append."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing BGE-M3 Embedder on {device} (fp16=True)...")
    embedder = BGEM3Embedder(
        model_name=config.embedding.model_name,
        device=device,
        batch_size=args.batch_size,
        use_fp16=True,
    )

    texts_to_embed = [c.contextual_text or c.chunk_text for c in new_chunks]
    logger.info(f"Encoding {len(texts_to_embed):,} new chunks with BGE-M3...")
    t_emb = time.time()
    new_embeddings = embedder.encode(texts_to_embed, show_progress=True)
    logger.info(f"Encoding completed in {time.time() - t_emb:.1f}s.")

    # Ensure float32 for FAISS IndexFlatIP
    if new_embeddings.dtype != np.float32:
        new_embeddings = new_embeddings.astype(np.float32)

    logger.info(f"Adding {len(new_embeddings):,} vectors to FAISS index...")
    faiss_index.add(new_embeddings)
    logger.info(f"Updated FAISS index ntotal: {faiss_index.ntotal:,}")

    logger.info(f"Writing updated FAISS index to {faiss_path}...")
    faiss.write_index(faiss_index, str(faiss_path))
    logger.info("FAISS dense index successfully updated on disk.")

    # 6. Rebuild BM25s sparse index (fast streaming from updated SQLite)
    if not args.skip_bm25:
        rebuild_bm25s(sqlite_path=sqlite_path, output_bm25s_dir=bm25s_path)
    else:
        logger.info("Skipping BM25s rebuild as requested (--skip-bm25).")

    # 7. Optional Sync to HF Storage Bucket
    if args.sync_to_bucket:
        logger.info(f"Syncing updated indices to HF Bucket: {args.bucket_uri} ...")
        hf_bin = shutil.which("hf") or str(Path(sys.executable).parent / "hf") or "hf"
        env = os.environ.copy()
        if hf_token:
            env["HF_TOKEN"] = hf_token
        res = subprocess.run([hf_bin, "sync", str(args.indices_dir), args.bucket_uri], env=env)
        if res.returncode == 0:
            logger.info("Hugging Face Bucket sync completed successfully!")
        else:
            logger.warning(f"HF Bucket sync returned code {res.returncode}. Please check HF_TOKEN.")

    print("\n" + "=" * 60)
    print("✅ INCREMENTAL INDEX UPDATE COMPLETED SUCCESSFULLY!")
    print(f"Added: {len(new_articles):,} articles | {len(new_chunks):,} chunks")
    print(f"Total FAISS Vectors: {faiss_index.ntotal:,}")
    print(f"Index Directory: {args.indices_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
