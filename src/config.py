"""Configuration loader and schema using Pydantic."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    indices_dir: Path = Path("data/indices")
    outputs_dir: Path = Path("outputs")
    submissions_dir: Path = Path("outputs/submissions")


class CrawlerConfig(BaseModel):
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    timeout_seconds: int = 15
    max_retries: int = 3
    concurrency: int = 10
    ncbi_email: str = "team@r2ai2026.org"
    ncbi_api_key: str | None = None


class ChunkingConfig(BaseModel):
    max_chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    split_by_sentences: bool = True


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 16
    max_length: int = 512
    normalize_embeddings: bool = True
    device: str = "auto"


class RetrievalConfig(BaseModel):
    dense_top_k: int = 50
    sparse_top_k: int = 50
    fusion_method: str = "rrf"  # "rrf" or "weighted"
    rrf_k: int = 60
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    hybrid_top_k: int = 30


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-large"
    batch_size: int = 16
    device: str = "auto"
    top_k_chunks: int = 10
    top_k_docs: int = 5
    score_threshold: float = -5.0


class ProjectConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


def load_config(config_path: str | Path = "configs/config.yaml") -> ProjectConfig:
    """Loads configuration from YAML file, with fallbacks to defaults."""
    config_file = Path(config_path)
    if not config_file.exists():
        return ProjectConfig()

    with open(config_file, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ProjectConfig(**data)
