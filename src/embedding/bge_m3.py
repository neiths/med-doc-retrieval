"""Multilingual dense & native lexical sparse embedding generator using BAAI/bge-m3."""

import numpy as np
import torch
from loguru import logger


class BGEM3Embedder:
    """Wrapper around BAAI/bge-m3 for multilingual dense and lexical sparse representations."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        return_sparse: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.return_sparse = return_sparse

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            f"Initializing BGEM3Embedder with model='{model_name}' on device='{self.device}', "
            f"fp16={self.use_fp16}, return_sparse={self.return_sparse}"
        )
        self._model = None
        self._is_flag_model = False

    @property
    def model(self):
        """Lazy loader for BGEM3FlagModel (preferred) or SentenceTransformer fallback."""
        if self._model is None:
            # 1. Try FlagEmbedding BGEM3FlagModel first (provides native dense + lexical sparse)
            try:
                from FlagEmbedding import BGEM3FlagModel

                logger.info(f"Loading native FlagEmbedding BGEM3FlagModel from {self.model_name}...")
                if self.device == "cpu" or not torch.cuda.is_available():
                    target_devices = ["cpu"]
                    use_fp16 = False
                elif self.device in ["cuda", "auto"]:
                    gpu_count = torch.cuda.device_count()
                    target_devices = [f"cuda:{i}" for i in range(gpu_count)] if gpu_count > 0 else ["cuda:0"]
                    use_fp16 = self.use_fp16
                else:
                    target_devices = [self.device]
                    use_fp16 = self.use_fp16

                logger.info(f"Using target device(s): {target_devices} (FP16: {use_fp16})")

                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=use_fp16,
                    devices=target_devices,
                    batch_size=self.batch_size,
                    query_max_length=self.max_length,
                    passage_max_length=self.max_length,
                    return_dense=True,
                    return_sparse=self.return_sparse,
                )
                self._is_flag_model = True
                logger.info("BGEM3FlagModel loaded successfully with native lexical sparse support.")
            except Exception as e:
                logger.warning(
                    f"Could not initialize FlagEmbedding BGEM3FlagModel ({e}). Falling back to SentenceTransformer."
                )
                from sentence_transformers import SentenceTransformer

                model_kwargs = {}
                if self.use_fp16 and self.device == "cuda":
                    model_kwargs["torch_dtype"] = torch.float16

                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    model_kwargs=model_kwargs if model_kwargs else None,
                )
                self._model.max_seq_length = self.max_length
                self._is_flag_model = False

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

        _ = self.model
        if self._is_flag_model:
            out = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            return out["dense_vecs"].astype(np.float32)
        else:
            return self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            ).astype(np.float32)

    def encode_sparse(
        self,
        texts: str | list[str],
    ) -> list[dict[str, float]]:
        """Encodes texts into native BGE-M3 lexical weights."""
        _, sparse_weights = self.encode_both(texts)
        return sparse_weights

    def encode_both(
        self,
        texts: str | list[str],
        show_progress_bar: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Encodes texts into both dense embeddings and native lexical sparse weights.

        Returns:
            Tuple of:
                - dense_embeddings: np.ndarray of shape (N, 1024)
                - sparse_weights: list of dicts {token_id: float_weight}
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 1024), dtype=np.float32), []

        _ = self.model  # Ensure model is initialized

        if self._is_flag_model:
            out = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense_vecs = out["dense_vecs"].astype(np.float32)
            sparse_weights = out.get("lexical_weights", [])
            return dense_vecs, sparse_weights
        else:
            # Fallback when using SentenceTransformer
            dense_vecs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
            ).astype(np.float32)

            # Generate basic token-frequency dictionary as fallback
            from collections import defaultdict

            from src.retrieval.sparse_index import tokenize_multilingual

            sparse_weights = []
            for t in texts:
                tf = defaultdict(float)
                for tok in tokenize_multilingual(t):
                    tf[tok] += 1.0
                sparse_weights.append(dict(tf))

            return dense_vecs, sparse_weights
