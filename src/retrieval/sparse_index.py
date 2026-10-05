"""Multilingual BM25 sparse index using ultra-fast, memory-mapped bm25s."""

import json
import pickle
import re
from pathlib import Path
from typing import Any

import bm25s
import jieba
from loguru import logger
from pyvi import ViTokenizer

from src.ingestion.cleaner import detect_language


def tokenize_multilingual(text: str, lang: str = "auto") -> list[str]:
    """Tokenizes text based on detected or specified language."""
    if not text:
        return []

    if lang == "auto":
        lang = detect_language(text)

    text_lower = text.lower()

    if lang == "zh":
        tokens = [t.strip() for t in jieba.lcut(text_lower) if t.strip()]
    elif lang == "vi":
        try:
            tokenized_vi = ViTokenizer.tokenize(text_lower)
            tokens = tokenized_vi.split()
        except Exception:
            tokens = re.findall(r"\w+", text_lower)
    else:  # en or other
        tokens = re.findall(r"\w+", text_lower)

    return tokens


class SparseIndex:
    """High-performance BM25 index over document chunks with zero-copy memory mapping."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.retriever: bm25s.BM25 | None = None
        self.chunk_ids: list[str] = []
        self.chunk_metadata: list[dict[str, Any]] = []
        self._sqlite_conn = None

    def build(self, chunks: list[dict[str, Any]], text_key: str = "chunk_text"):
        """Builds bm25s index from a list of chunk dicts in seconds."""
        logger.info(f"Building fast bm25s sparse index over {len(chunks):,} chunks...")
        corpus_tokens = []
        self.chunk_ids = []
        self.chunk_metadata = []

        for chunk in chunks:
            text = chunk.get("contextual_text") or chunk.get(text_key, "")
            lang = chunk.get("lang", "auto")
            tokens = tokenize_multilingual(text, lang=lang)
            corpus_tokens.append(tokens)
            self.chunk_ids.append(chunk["chunk_id"])

            clean_chunk = dict(chunk)
            raw_c = clean_chunk.get("chunk_text", "")
            if raw_c.startswith("Tiêu đề:") and "\nNội dung: " in raw_c:
                clean_chunk["chunk_text"] = raw_c.split("\nNội dung: ", 1)[-1].strip()
            self.chunk_metadata.append(clean_chunk)

        self.retriever = bm25s.BM25(k1=self.k1, b=self.b)
        self.retriever.index(corpus_tokens)
        logger.info("bm25s index construction completed.")

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float, dict[str, Any]]]:
        """Searches BM25 index with a query string.

        Returns:
            List of tuples (chunk_id, bm25_score, chunk_metadata)
        """
        if self.retriever is None or not self.chunk_ids:
            return []

        query_tokens = tokenize_multilingual(query, lang="auto")
        if not query_tokens:
            return []

        k = min(top_k, len(self.chunk_ids))
        res_indices, scores = self.retriever.retrieve([query_tokens], k=k)

        results = []
        top_indices = res_indices[0].tolist()
        top_scores = scores[0].tolist()

        if self._sqlite_conn is not None:
            cur = self._sqlite_conn.cursor()
            for idx, score in zip(top_indices, top_scores):
                if score <= 0 or idx < 0 or idx >= len(self.chunk_ids):
                    continue
                row = cur.execute(
                    "SELECT doc_id, chunk_id, chunk_text, title, lang FROM chunks WHERE idx = ?",
                    (int(idx),),
                ).fetchone()
                if row:
                    raw_c = row[2] or ""
                    if raw_c.startswith("Tiêu đề:") and "\nNội dung: " in raw_c:
                        raw_c = raw_c.split("\nNội dung: ", 1)[-1].strip()
                    meta = {
                        "doc_id": row[0],
                        "chunk_id": row[1],
                        "chunk_text": raw_c,
                        "title": row[3],
                        "lang": row[4],
                    }
                    results.append((row[1], float(score), meta))
        else:
            for idx, score in zip(top_indices, top_scores):
                if score <= 0 or idx < 0 or idx >= len(self.chunk_ids):
                    continue
                results.append((self.chunk_ids[idx], float(score), self.chunk_metadata[idx]))

        return results

    def save(self, directory: Path | str):
        """Saves bm25s index and lightweight chunk metadata to disk."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        bm25s_dir = save_dir / "bm25s_index"
        if self.retriever is not None:
            self.retriever.save(str(bm25s_dir))

        meta_file = save_dir / "bm25s_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "chunk_ids": self.chunk_ids,
                    "chunk_metadata": self.chunk_metadata,
                },
                f,
                ensure_ascii=False,
            )
        # Create a small marker file so code expecting bm25_index.pkl knows sparse index is ready
        (save_dir / "bm25_index.pkl").touch()
        logger.info(f"Saved memory-efficient bm25s sparse index to {bm25s_dir}")

    @classmethod
    def load(cls, directory: Path | str) -> "SparseIndex":
        """Loads sparse index from disk using zero-RAM memory mapping."""
        import sqlite3

        load_dir = Path(directory)
        bm25s_dir = load_dir / "bm25s_index"
        meta_file = load_dir / "bm25s_metadata.json"
        sqlite_file = load_dir / "chunks_meta.sqlite"
        pkl_file = load_dir / "bm25_index.pkl"

        if bm25s_dir.exists():
            idx = cls()
            idx.retriever = bm25s.BM25.load(str(bm25s_dir), mmap=True)
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                idx.k1 = meta.get("k1", 1.5)
                idx.b = meta.get("b", 0.75)
                idx.chunk_ids = meta.get("chunk_ids", [])
                idx.chunk_metadata = meta.get("chunk_metadata", [])
            elif sqlite_file.exists():
                idx._sqlite_conn = sqlite3.connect(sqlite_file)
                cur = idx._sqlite_conn.cursor()
                rows = cur.execute("SELECT chunk_id FROM chunks ORDER BY idx").fetchall()
                idx.chunk_ids = [r[0] for r in rows]

            if sqlite_file.exists():
                idx._sqlite_conn = sqlite3.connect(sqlite_file)

            logger.info(f"Loaded bm25s sparse index ({len(idx.chunk_ids):,} chunks) via zero-copy mmap.")
            return idx

        # Backwards compatibility: fallback to rank_bm25 pickle if bm25s not found
        if pkl_file.exists() and pkl_file.stat().st_size > 0:
            with open(pkl_file, "rb") as f:
                payload = pickle.load(f)
            idx = cls(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))

            class RankBM25Adapter:
                def __init__(self, bm25_obj):
                    self.bm25_obj = bm25_obj

                def retrieve(self, token_batches, k):
                    import numpy as np

                    scores = self.bm25_obj.get_scores(token_batches[0])
                    top_idx = scores.argsort()[::-1][:k]
                    return [np.array(top_idx)], [np.array(scores[top_idx])]

            idx.retriever = RankBM25Adapter(payload["bm25"])
            idx.chunk_ids = payload.get("chunk_ids", [])
            idx.chunk_metadata = payload.get("chunk_metadata", [])
            logger.info(f"Loaded legacy BM25 sparse index with {len(idx.chunk_ids)} chunks.")
            return idx

        raise FileNotFoundError(f"No valid BM25 index found in {load_dir}")
