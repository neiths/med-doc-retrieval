"""Cross-encoder reranking module using BAAI/bge-reranker-large or bge-reranker-v2-m3."""

from typing import Any

import torch
from loguru import logger


class BGEReranker:
    """Cross-encoder reranker for fine-grained semantic relevance scoring."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "auto",
        batch_size: int = 16,
        use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_fp16 = use_fp16

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Initializing BGEReranker with model='{model_name}' on device='{self.device}', fp16={self.use_fp16}"
        )
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading reranker model weights from {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if (self.use_fp16 and self.device == "cuda") else torch.float32
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            )
            self._model.to(self.device)
            self._model.eval()

    def compute_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Computes cross-encoder relevance scores for (query, document/chunk) pairs."""
        if not pairs:
            return []

        self._load_model()
        scores: list[float] = []

        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]
                inputs = self._tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self._model(**inputs)
                batch_scores = outputs.logits.view(-1).float().cpu().tolist()
                scores.extend(batch_scores)

        return scores

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k_chunks: int = 10,
        top_k_docs: int = 5,
        score_threshold: float = -5.0,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Reranks candidate chunks and extracts top relevant documents and chunks.

        Args:
            query: User search query in Vietnamese.
            candidates: Candidate chunk dictionaries from hybrid retrieval.
            top_k_chunks: Max chunks to select.
            top_k_docs: Max unique documents to select.
            score_threshold: Minimum cross-encoder score filter.

        Returns:
            Tuple of (list of doc_ids, list of chunk dicts {"doc_id": ..., "chunk_text": ...})
        """
        if not candidates:
            return [], []

        pairs = [(query, cand["chunk_text"]) for cand in candidates]
        scores = self.compute_scores(pairs)

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = score

        # Filter and sort by rerank_score descending
        valid_candidates = [c for c in candidates if c.get("rerank_score", -999) >= score_threshold]
        valid_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        selected_chunks = valid_candidates[:top_k_chunks]

        # Extract unique documents preserving order of highest chunk score
        doc_ids = []
        seen_docs = set()
        for c in selected_chunks:
            did = str(c["doc_id"])
            if did not in seen_docs:
                seen_docs.add(did)
                doc_ids.append(did)
            if len(doc_ids) >= top_k_docs:
                break

        formatted_chunks = [
            {"doc_id": str(c["doc_id"]), "chunk_text": c["chunk_text"]} for c in selected_chunks
        ]

        return doc_ids, formatted_chunks
