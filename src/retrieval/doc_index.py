"""Document-level FAISS vector index for global article retrieval."""

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow.parquet as pq
from loguru import logger


class DocIndex:
    """Document-level vector index using FAISS (IndexScalarQuantizer SQ8 or IndexFlatIP).

    Stores 878,943 article-level embeddings with metadata (doc_id, title, url, domain).
    """

    def __init__(self, index_path: str | Path, metadata_path: str | Path):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: faiss.Index | None = None
        self.doc_ids: list[int | str] = []
        self.titles: list[str] = []
        self._loaded: bool = False

    def load(self):
        """Loads FAISS index via zero-copy mmap and reads parquet metadata."""
        if self._loaded:
            return

        if not self.index_path.exists():
            raise FileNotFoundError(f"DocIndex FAISS file not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Doc Metadata file not found: {self.metadata_path}")

        logger.info(f"Loading DocIndex from {self.index_path} via IO_FLAG_MMAP...")
        self.index = faiss.read_index(str(self.index_path), faiss.IO_FLAG_MMAP)

        logger.info(f"Loading Doc Metadata from {self.metadata_path}...")
        tbl = pq.read_table(str(self.metadata_path), columns=["doc_id", "title"])
        self.doc_ids = tbl["doc_id"].to_pylist()
        self.titles = tbl["title"].to_pylist()
        self._loaded = True
        logger.info(f"DocIndex loaded successfully ({self.index.ntotal:,} articles).")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 400,
    ) -> list[dict[str, Any]]:
        """Searches doc index for top_k nearest documents.

        Args:
            query_embedding: 1024-dim embedding vector.
            top_k: Number of nearest document candidates to return.

        Returns:
            List of dicts: [{"doc_id": ..., "score": ..., "title": ..., "rank": ...}]
        """
        if not self._loaded:
            self.load()

        q_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            results.append({
                "doc_id": self.doc_ids[idx],
                "score": float(score),
                "title": self.titles[idx],
                "rank": rank,
            })
        return results
