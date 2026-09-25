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
        """Loads FAISS index and metadata from disk."""
        load_dir = Path(directory)
        faiss_file = load_dir / "dense_index.faiss"
        meta_file = load_dir / "dense_metadata.json"

        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

        idx = cls(dimension=meta["dimension"])
        idx.index = faiss.read_index(str(faiss_file))
        idx.chunk_ids = meta["chunk_ids"]
        idx.chunk_metadata = meta["chunk_metadata"]
        logger.info(f"Loaded FAISS dense index with {idx.index.ntotal} vectors from {load_dir}")
        return idx
