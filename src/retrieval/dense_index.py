"""FAISS dense vector index for fast similarity search."""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from loguru import logger


class DenseIndex:
    """Vector index using FAISS IndexFlatIP (Inner Product on normalized vectors = Cosine Sim)."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_ids: list[str] = []
        self.chunk_metadata: list[dict[str, Any]] = []

    def add(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        chunk_metadata: list[dict[str, Any]],
    ):
        """Adds normalized dense embeddings and associated chunk metadata to FAISS index."""
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError("Number of embeddings must match number of chunk_ids.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}."
            )

        # Ensure float32 and C-contiguous
        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vecs)

        self.index.add(vecs)
        self.chunk_ids.extend(chunk_ids)
        self.chunk_metadata.extend(chunk_metadata)
        logger.info(f"Added {len(chunk_ids)} vectors to FAISS index. Total: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Searches FAISS index for top_k nearest chunks.

        Returns:
            List of tuples (chunk_id, similarity_score, metadata)
        """
        if self.index.ntotal == 0:
            return []

        # Prepare 2D query vector
        if query_embedding.ndim == 1:
            q_vec = query_embedding.reshape(1, -1)
        else:
            q_vec = query_embedding

        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)
        faiss.normalize_L2(q_vec)

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_vec, k)

        results = []
        if hasattr(self, "_sqlite_conn") and self._sqlite_conn is not None:
            cur = self._sqlite_conn.cursor()
            for dist, idx_val in zip(distances[0], indices[0]):
                if idx_val < 0 or idx_val >= self.index.ntotal:
                    continue
                row = cur.execute(
                    "SELECT doc_id, chunk_id, chunk_text, title, lang FROM chunks WHERE idx = ?",
                    (int(idx_val),),
                ).fetchone()
                if row:
                    meta = {
                        "doc_id": row[0],
                        "chunk_id": row[1],
                        "chunk_text": row[2],
                        "title": row[3],
                        "lang": row[4],
                    }
                    results.append((row[1], float(dist), meta))
        else:
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.chunk_ids):
                    continue
                results.append((self.chunk_ids[idx], float(dist), self.chunk_metadata[idx]))

        return results

    def save(self, directory: Path | str):
        """Saves FAISS index and metadata to disk."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss_file = save_dir / "dense_index.faiss"
        faiss.write_index(self.index, str(faiss_file))

        meta_file = save_dir / "dense_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": self.dimension,
                    "chunk_ids": self.chunk_ids,
                    "chunk_metadata": self.chunk_metadata,
                },
                f,
                ensure_ascii=False,
            )
        logger.info(f"Saved FAISS index ({self.index.ntotal} items) to {save_dir}")

    @classmethod
    def load(cls, directory: Path | str) -> "DenseIndex":
        """Loads FAISS index and metadata from disk using zero-RAM memory mapping and SQLite."""
        import sqlite3

        load_dir = Path(directory)
        faiss_file = load_dir / "dense_index.faiss"
        sqlite_file = load_dir / "chunks_meta.sqlite"
        meta_file = load_dir / "dense_metadata.json"
        chunks_jsonl = Path("data/processed/chunks.jsonl")

        idx = cls(dimension=1024)
        try:
            idx.index = faiss.read_index(str(faiss_file), faiss.IO_FLAG_MMAP)
            logger.info("Loaded FAISS index via zero-copy memory mapping (IO_FLAG_MMAP).")
        except Exception:
            idx.index = faiss.read_index(str(faiss_file))

        # Auto-create lightweight SQLite metadata if not already present
        if not sqlite_file.exists():
            if chunks_jsonl.exists():
                logger.info(f"Building fast SQLite metadata index from {chunks_jsonl} (RAM-safe)...")
                conn = sqlite3.connect(sqlite_file)
                cur = conn.cursor()
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS chunks (idx INTEGER PRIMARY KEY, doc_id TEXT, chunk_id TEXT, chunk_text TEXT, title TEXT, lang TEXT)"
                )
                batch = []
                with open(chunks_jsonl, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if not line.strip():
                            continue
                        c = json.loads(line)
                        batch.append(
                            (
                                i,
                                str(c.get("doc_id", "")),
                                str(c.get("chunk_id", "")),
                                c.get("contextual_text") or c.get("chunk_text", ""),
                                c.get("title", ""),
                                c.get("lang", "en"),
                            )
                        )
                        if len(batch) >= 10000:
                            cur.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", batch)
                            conn.commit()
                            batch.clear()
                if batch:
                    cur.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", batch)
                    conn.commit()
                conn.close()
                logger.info(f"Created SQLite metadata index at {sqlite_file}")
            elif meta_file.exists():
                logger.info(f"Converting {meta_file} into SQLite to prevent RAM exhaustion...")
                try:
                    import ijson
                except ImportError:
                    import subprocess, sys

                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ijson"], check=False)
                    import ijson

                conn = sqlite3.connect(sqlite_file)
                cur = conn.cursor()
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS chunks (idx INTEGER PRIMARY KEY, doc_id TEXT, chunk_id TEXT, chunk_text TEXT, title TEXT, lang TEXT)"
                )
                batch = []
                with open(meta_file, "rb") as f:
                    for i, c in enumerate(ijson.items(f, "chunk_metadata.item")):
                        batch.append(
                            (
                                i,
                                str(c.get("doc_id", "")),
                                str(c.get("chunk_id", "")),
                                c.get("contextual_text") or c.get("chunk_text", ""),
                                c.get("title", ""),
                                c.get("lang", "en"),
                            )
                        )
                        if len(batch) >= 10000:
                            cur.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", batch)
                            conn.commit()
                            batch.clear()
                if batch:
                    cur.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", batch)
                    conn.commit()
                conn.close()
                logger.info(f"Created SQLite metadata index from JSON at {sqlite_file}")

        if sqlite_file.exists():
            idx._sqlite_conn = sqlite3.connect(sqlite_file)
            logger.info(
                f"Loaded FAISS dense index ({idx.index.ntotal:,} vectors) with SQLite metadata from {load_dir}"
            )
        elif meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            idx.dimension = meta.get("dimension", 1024)
            idx.chunk_ids = meta.get("chunk_ids", [])
            idx.chunk_metadata = meta.get("chunk_metadata", [])
            logger.info(
                f"Loaded FAISS dense index with {idx.index.ntotal:,} vectors from {load_dir}"
            )
        else:
            logger.warning(f"No metadata found for FAISS index in {load_dir}")
        return idx

