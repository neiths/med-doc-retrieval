"""Multilingual dense embedding generator using BAAI/bge-m3."""

import numpy as np
import torch
from loguru import logger


class BGEM3Embedder:
    """Wrapper around BGE-M3 for multilingual dense representation."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Initializing BGEM3Embedder with model='{model_name}' on device='{self.device}', fp16={self.use_fp16}"
        )
        self._model = None

    @property
    def model(self):
        """Lazy loader for SentenceTransformer / FlagEmbedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model weights from {self.model_name}...")
            model_kwargs = {}
            if self.use_fp16 and self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs=model_kwargs if model_kwargs else None,
            )
            self._model.max_seq_length = self.max_length
        return self._model

    def encode(
        self,
        texts: str | list[str],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encodes texts into normalized dense embedding matrix of shape (N, dim)."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
