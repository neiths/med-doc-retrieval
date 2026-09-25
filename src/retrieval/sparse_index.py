"""Multilingual BM25 sparse index with language-specific tokenization."""

import pickle
import re
from pathlib import Path
from typing import Any

import jieba
from loguru import logger
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi

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
    """In-memory BM25 index over document chunks with metadata preservation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.chunk_metadata: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]], text_key: str = "chunk_text"):
        """Builds BM25 index from a list of chunk dicts."""
        logger.info(f"Building Sparse BM25 index over {len(chunks)} chunks...")
        corpus_tokens = []
        self.chunk_ids = []
        self.chunk_metadata = []

        for chunk in chunks:
            text = chunk.get(text_key, "")
            lang = chunk.get("lang", "auto")
            tokens = tokenize_multilingual(text, lang=lang)
            corpus_tokens.append(tokens)
            self.chunk_ids.append(chunk["chunk_id"])
            self.chunk_metadata.append(chunk)

        self.bm25 = BM25Okapi(corpus_tokens, k1=self.k1, b=self.b)
        logger.info("BM25 index construction completed.")

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float, dict[str, Any]]]:
        """Searches BM25 index with a query string.

        Returns:
            List of tuples (chunk_id, bm25_score, chunk_metadata)
        """
        if self.bm25 is None or not self.chunk_ids:
            return []

        query_tokens = tokenize_multilingual(query, lang="auto")
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append((self.chunk_ids[idx], score, self.chunk_metadata[idx]))

        return results

    def save(self, directory: Path | str):
        """Saves sparse index and metadata to disk."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "k1": self.k1,
            "b": self.b,
            "bm25": self.bm25,
            "chunk_ids": self.chunk_ids,
            "chunk_metadata": self.chunk_metadata,
        }
        with open(save_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Saved BM25 sparse index to {save_dir / 'bm25_index.pkl'}")

    @classmethod
    def load(cls, directory: Path | str) -> "SparseIndex":
        """Loads sparse index and metadata from disk."""
        load_dir = Path(directory)
        with open(load_dir / "bm25_index.pkl", "rb") as f:
            payload = pickle.load(f)

        idx = cls(k1=payload["k1"], b=payload["b"])
        idx.bm25 = payload["bm25"]
        idx.chunk_ids = payload["chunk_ids"]
        idx.chunk_metadata = payload["chunk_metadata"]
        logger.info(f"Loaded BM25 sparse index with {len(idx.chunk_ids)} chunks.")
        return idx
