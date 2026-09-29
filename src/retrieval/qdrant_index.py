"""Local Embedded Qdrant Vector Index supporting Native Hybrid Search (Dense + Sparse + RRF)."""

import hashlib
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient, models

from src.retrieval.sparse_index import tokenize_multilingual


def lexical_weights_to_sparse_vector(weights: dict[str | int, float]) -> models.SparseVector:
    """Converts BGE-M3 lexical weights dictionary to Qdrant models.SparseVector.

    BGE-M3 lexical head outputs {token_id: weight}.
    Token IDs are non-negative integers (0 to ~250,000 for XLM-RoBERTa),
    which directly map to Qdrant sparse vector indices.
    """
    if not weights:
        return models.SparseVector(indices=[], values=[])

    indices = []
    values = []
    for k, v in weights.items():
        if isinstance(k, str) and not k.isdigit():
            token_hash = hashlib.md5(k.encode("utf-8")).hexdigest()[:8]
            idx = int(token_hash, 16) % (2**31 - 1)
        else:
            idx = int(k)
        indices.append(idx)
        values.append(float(v))

    return models.SparseVector(indices=indices, values=values)


def text_to_sparse_vector(text: str, lang: str = "auto") -> models.SparseVector:
    """Converts multilingual text into a deterministic sparse term-frequency vector."""
    tokens = tokenize_multilingual(text, lang=lang)
    if not tokens:
        return models.SparseVector(indices=[], values=[])

    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1

    indices = []
    values = []
    for token, count in tf.items():
        # Deterministic 31-bit integer hash index for token
        token_hash = hashlib.md5(token.encode("utf-8")).hexdigest()[:8]
        idx = int(token_hash, 16) % (2**31 - 1)
        weight = float(1.0 + np.log(count))
        indices.append(idx)
        values.append(weight)

    return models.SparseVector(indices=indices, values=values)


class QdrantLocalIndex:
    """Embedded Qdrant Vector Store with native Dense + Sparse Hybrid Search and RRF."""

    def __init__(
        self,
        storage_path: Path | str = "data/indices/qdrant_db",
        collection_name: str = "medical_chunks",
        dimension: int = 1024,
    ):
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.dimension = dimension

        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing QdrantClient (Embedded) at '{self.storage_path}'")
        self.client = QdrantClient(path=str(self.storage_path))
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates collection with named dense and sparse vectors if not exists."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' (dim={self.dimension})"
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dimension,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        sparse_weights: list[dict[str | int, float]] | list[models.SparseVector] | None = None,
        batch_size: int = 256,
    ):
        """Indexes chunks with dense embeddings and native lexical sparse vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings.")

        total = len(chunks)
        logger.info(f"Indexing {total} chunks into Qdrant collection '{self.collection_name}'...")

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_sparse = sparse_weights[i : i + batch_size] if sparse_weights is not None else None

            points = []
            for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
                chunk_id = str(chunk.get("chunk_id", ""))
                # Deterministic UUID from chunk_id
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

                if batch_sparse is not None and j < len(batch_sparse):
                    sp = batch_sparse[j]
                    if isinstance(sp, models.SparseVector):
                        sparse_vec = sp
                    elif isinstance(sp, dict):
                        sparse_vec = lexical_weights_to_sparse_vector(sp)
                    else:
                        sparse_vec = text_to_sparse_vector(
                            text=chunk.get("contextual_text") or chunk.get("chunk_text", ""),
                            lang=chunk.get("lang", "auto"),
                        )
                else:
                    sparse_text = chunk.get("contextual_text") or chunk.get("chunk_text", "")
                    sparse_vec = text_to_sparse_vector(
                        text=sparse_text,
                        lang=chunk.get("lang", "auto"),
                    )

                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": emb.tolist(),
                        "sparse": sparse_vec,
                    },
                    payload={
                        "doc_id": str(chunk.get("doc_id", "")),
                        "chunk_id": chunk_id,
                        "chunk_text": chunk.get("chunk_text", ""),
                        "contextual_text": chunk.get("contextual_text")
                        or chunk.get("chunk_text", ""),
                        "lang": chunk.get("lang", "en"),
                        "title": chunk.get("metadata", {}).get("title", ""),
                        "source": chunk.get("metadata", {}).get("source", ""),
                        "char_start": chunk.get("char_start", 0),
                        "char_end": chunk.get("char_end", 0),
                    },
                )
                points.append(point)

            self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"Successfully indexed {total} chunks into Qdrant.")

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        query_sparse: models.SparseVector | dict[str | int, float] | None = None,
        top_k: int = 50,
        lang_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Executes native Hybrid Search (Dense + Sparse) with Reciprocal Rank Fusion (RRF).

        Returns:
            List of candidate chunk metadata dicts compatible with downstream reranker.
        """
        # Prepare 1D query vector
        if query_embedding.ndim > 1:
            q_vec = query_embedding.flatten().tolist()
        else:
            q_vec = query_embedding.tolist()

        if query_sparse is not None:
            if isinstance(query_sparse, models.SparseVector):
                sparse_query = query_sparse
            elif isinstance(query_sparse, dict):
                sparse_query = lexical_weights_to_sparse_vector(query_sparse)
            else:
                sparse_query = text_to_sparse_vector(query_text, lang="auto")
        else:
            sparse_query = text_to_sparse_vector(query_text, lang="auto")

        # Optional payload filtering (e.g. by language or source)
        query_filter = None
        if lang_filter:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="lang", match=models.MatchValue(value=lang_filter))]
            )

        try:
            # Native Hybrid Search with RRF Fusion
            prefetch_list = [
                models.Prefetch(
                    query=q_vec,
                    using="dense",
                    limit=top_k,
                    filter=query_filter,
                )
            ]
            if sparse_query.indices:
                prefetch_list.append(
                    models.Prefetch(
                        query=sparse_query,
                        using="sparse",
                        limit=top_k,
                        filter=query_filter,
                    )
                )

            res = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch_list,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
            )
            points = res.points
        except Exception as e:
            logger.warning(f"Hybrid RRF query failed ({e}), falling back to pure dense search.")
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                using="dense",
                limit=top_k,
                filter=query_filter,
            )
            points = res.points

        candidates = []
        for p in points:
            payload = p.payload or {}
            cand = {
                "chunk_id": payload.get("chunk_id", ""),
                "doc_id": payload.get("doc_id", ""),
                "chunk_text": payload.get("chunk_text", ""),
                "contextual_text": payload.get("contextual_text") or payload.get("chunk_text", ""),
                "lang": payload.get("lang", "en"),
                "score": float(p.score) if p.score is not None else 0.0,
                "metadata": {
                    "title": payload.get("title", ""),
                    "source": payload.get("source", ""),
                    "char_start": payload.get("char_start", 0),
                    "char_end": payload.get("char_end", 0),
                },
            }
            candidates.append(cand)

        return candidates

    def count(self) -> int:
        """Returns the total number of chunks currently stored in Qdrant."""
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception:
            return 0
