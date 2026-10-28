"""FAISS dense vector index for fast similarity search."""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from loguru import logger

from src.retrieval.chunk_meta import ChunkMetaStore


class DenseIndex:
    """Vector index using FAISS IndexFlatIP (Inner Product on normalized vectors = Cosine Sim)."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_ids: list[str] = []
        self.chunk_metadata: list[dict[str, Any]] = []
        self.meta_store: ChunkMetaStore | None = None

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
        if self.meta_store is not None:
            valid = [
                (float(dist), int(idx_val))
                for dist, idx_val in zip(distances[0], indices[0])
                if 0 <= idx_val < self.index.ntotal
            ]
            by_pos = self.meta_store.get_by_positions({pos for _, pos in valid})
            for dist, pos in valid:
                meta = by_pos.get(pos)
                if meta is not None:
                    results.append((meta["chunk_id"], dist, meta))
        else:
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.chunk_ids):
                    continue
                m = dict(self.chunk_metadata[idx])
                raw_chunk = m.get("chunk_text", "")
                if raw_chunk.startswith("Tiêu đề:") and "\nNội dung: " in raw_chunk:
                    m["contextual_text"] = raw_chunk
                    m["chunk_text"] = raw_chunk.split("\nNội dung: ", 1)[-1].strip()
                results.append((self.chunk_ids[idx], float(dist), m))

        return results

    def get_doc_chunks(
        self,
        doc_ids: list[str],
        query: str | None = None,
        max_per_doc: int | None = None,
    ) -> list[dict[str, Any]]:
        """Returns every indexed chunk of the given documents."""
        if self.meta_store is None:
            return []
        return self.meta_store.get_doc_chunks(doc_ids, query=query, max_per_doc=max_per_doc)

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
    def load(
        cls,
        directory: Path | str,
        fallback_meta_dir: Path | str | None = None,
        chunks_meta_path: Path | str | None = None,
    ) -> "DenseIndex":
        """Loads FAISS index and metadata from disk using zero-RAM memory mapping and SQLite.
        
        Supports separate directory/file for FAISS index, custom metadata file path, and fallback metadata directory.
        """
        import sqlite3

        load_path = Path(directory)
        if load_path.is_file():
            faiss_file = load_path
            load_dir = load_path.parent
        else:
            load_dir = load_path
            faiss_candidates = [
                load_dir / "dense_index_sq8.faiss",
                load_dir / "dense_index.faiss",
            ]
            faiss_file = None
            for fc in faiss_candidates:
                if fc.exists():
                    faiss_file = fc
                    break
            if faiss_file is None:
                all_faiss = list(load_dir.glob("*.faiss"))
                if all_faiss:
                    faiss_file = all_faiss[0]
                else:
                    faiss_file = load_dir / "dense_index.faiss"

        sqlite_file = None
        if chunks_meta_path:
            p = Path(chunks_meta_path)
            if p.exists() and p.stat().st_size > 0 and p.suffix in [".sqlite", ".db"]:
                sqlite_file = p

        if sqlite_file is None:
            slim = load_dir / "chunks_meta_slim.sqlite"
            full = load_dir / "chunks_meta.sqlite"
            sqlite_file = slim if (slim.exists() and slim.stat().st_size > 0) else full
            if not (sqlite_file.exists() and sqlite_file.stat().st_size > 0) and fallback_meta_dir:
                fb = Path(fallback_meta_dir)
                for cand in (fb / "chunks_meta_slim.sqlite", fb / "chunks_meta.sqlite"):
                    if cand.exists() and cand.stat().st_size > 0:
                        sqlite_file = cand
                        break

        meta_file = (
            load_dir / "dense_metadata_sq8.json"
            if (load_dir / "dense_metadata_sq8.json").exists()
            else (load_dir / "dense_metadata.json")
        )
        if not meta_file.exists() and fallback_meta_dir:
            fb = Path(fallback_meta_dir)
            if (fb / "dense_metadata_sq8.json").exists():
                meta_file = fb / "dense_metadata_sq8.json"
            elif (fb / "dense_metadata.json").exists():
                meta_file = fb / "dense_metadata.json"

        chunks_jsonl = Path("data/processed/chunks.jsonl")
        parquet_chunks = load_dir / "chunks_meta.parquet"
        if not parquet_chunks.exists() and fallback_meta_dir:
            fb = Path(fallback_meta_dir)
            if (fb / "chunks_meta.parquet").exists():
                parquet_chunks = fb / "chunks_meta.parquet"

        idx = cls(dimension=1024)
        try:
            idx.index = faiss.read_index(str(faiss_file), faiss.IO_FLAG_MMAP)
            logger.info("Loaded FAISS index via zero-copy memory mapping (IO_FLAG_MMAP).")
        except Exception:
            idx.index = faiss.read_index(str(faiss_file))

        # Auto-create lightweight SQLite metadata if not already present
        if not sqlite_file.exists():
            if parquet_chunks.exists():
                logger.info(f"Converting {parquet_chunks} into fast SQLite metadata index (RAM-safe)...")
                import pyarrow.parquet as pq
                conn = sqlite3.connect(sqlite_file, check_same_thread=False)
                cur = conn.cursor()
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS chunks (idx INTEGER PRIMARY KEY, doc_id TEXT, chunk_id TEXT, chunk_text TEXT, title TEXT, lang TEXT)"
                )
                tbl = pq.read_table(str(parquet_chunks))
                pylist = tbl.to_pylist()
                batch = []
                for i, r in enumerate(pylist):
                    raw_c = r.get("contextual_text") or r.get("chunk_text") or ""
                    batch.append(
                        (
                            i,
                            str(r.get("doc_id", "")),
                            str(r.get("chunk_id", i)),
                            raw_c,
                            r.get("title", ""),
                            r.get("lang", "en"),
                        )
                    )
                    if len(batch) >= 20000:
                        cur.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", batch)
                        conn.commit()
                        batch.clear()
                if batch:
                    cur.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", batch)
                    conn.commit()
                conn.close()
                logger.info(f"Created SQLite metadata index from parquet at {sqlite_file}")
            elif chunks_jsonl.exists():
                logger.info(f"Building fast SQLite metadata index from {chunks_jsonl} (RAM-safe)...")
                conn = sqlite3.connect(sqlite_file, check_same_thread=False)
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
                        raw_c = c.get("chunk_text", "")
                        if raw_c.startswith("Tiêu đề:") and "\nNội dung: " in raw_c:
                            raw_c = raw_c.split("\nNội dung: ", 1)[-1].strip()
                        batch.append(
                            (
                                i,
                                str(c.get("doc_id", "")),
                                str(c.get("chunk_id", "")),
                                raw_c,
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

                conn = sqlite3.connect(sqlite_file, check_same_thread=False)
                cur = conn.cursor()
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS chunks (idx INTEGER PRIMARY KEY, doc_id TEXT, chunk_id TEXT, chunk_text TEXT, title TEXT, lang TEXT)"
                )
                batch = []
                with open(meta_file, "rb") as f:
                    for i, c in enumerate(ijson.items(f, "chunk_metadata.item")):
                        raw_c = c.get("chunk_text", "")
                        if raw_c.startswith("Tiêu đề:") and "\nNội dung: " in raw_c:
                            raw_c = raw_c.split("\nNội dung: ", 1)[-1].strip()
                        batch.append(
                            (
                                i,
                                str(c.get("doc_id", "")),
                                str(c.get("chunk_id", "")),
                                raw_c,
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
            idx._sqlite_conn = sqlite3.connect(sqlite_file, check_same_thread=False)
            idx.meta_store = ChunkMetaStore(idx._sqlite_conn)
            idx._id_col = idx.meta_store.pos_col
            idx._has_url = idx.meta_store.has_url
            logger.info(
                f"Loaded FAISS dense index ({idx.index.ntotal:,} vectors) with SQLite metadata from {load_dir} (key column: {idx._id_col}, has_url: {idx._has_url})"
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

