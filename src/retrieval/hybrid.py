"""Hybrid Retrieval engine combining Dense (FAISS) and Sparse (BM25) search."""

from typing import Any

import numpy as np

from src.retrieval.dense_index import DenseIndex
from src.retrieval.sparse_index import SparseIndex


class HybridRetriever:
    """Orchestrates multi-modal retrieval using BGE-M3 dense embeddings and BM25."""

    def __init__(
        self,
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 30,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Executes dense + sparse search and fuses results.

        Returns:
            List of candidate chunk dicts with fused scores, sorted descending.
        """
        # 1. Dense search
        dense_results = self.dense_index.search(query_embedding, top_k=dense_top_k)

        # 2. Sparse search
        sparse_results = self.sparse_index.search(query_text, top_k=sparse_top_k)

        # 3. Fusion
        chunk_info: dict[str, dict[str, Any]] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}

        for rank, (cid, score, meta) in enumerate(dense_results):
            dense_ranks[cid] = rank + 1
            dense_scores[cid] = score
            if cid not in chunk_info:
                chunk_info[cid] = meta

        for rank, (cid, score, meta) in enumerate(sparse_results):
            sparse_ranks[cid] = rank + 1
            sparse_scores[cid] = score
            if cid not in chunk_info:
                chunk_info[cid] = meta

        all_chunk_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        fused_scores: dict[str, float] = {}

        if self.fusion_method == "rrf":
            for cid in all_chunk_ids:
                rrf_score = 0.0
                if cid in dense_ranks:
                    rrf_score += self.dense_weight / (self.rrf_k + dense_ranks[cid])
                if cid in sparse_ranks:
                    rrf_score += self.sparse_weight / (self.rrf_k + sparse_ranks[cid])
                fused_scores[cid] = rrf_score
        else:
            # Weighted Min-Max Normalized Combination
            max_d = max(dense_scores.values()) if dense_scores else 1.0
            min_d = min(dense_scores.values()) if dense_scores else 0.0
            range_d = max(max_d - min_d, 1e-6)

            max_s = max(sparse_scores.values()) if sparse_scores else 1.0
            min_s = min(sparse_scores.values()) if sparse_scores else 0.0
            range_s = max(max_s - min_s, 1e-6)

            for cid in all_chunk_ids:
                d_norm = (dense_scores.get(cid, min_d) - min_d) / range_d if cid in dense_scores else 0.0
                s_norm = (sparse_scores.get(cid, min_s) - min_s) / range_s if cid in sparse_scores else 0.0
                fused_scores[cid] = (self.dense_weight * d_norm) + (self.sparse_weight * s_norm)

        # Sort candidate chunks
        sorted_chunks = sorted(all_chunk_ids, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

        candidates = []
        for cid in sorted_chunks:
            meta = chunk_info[cid].copy()
            meta["score"] = fused_scores[cid]
            candidates.append(meta)

        return candidates
