"""End-to-end Pipeline orchestrator for Medical Document Retrieval."""

import json
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from src.config import ProjectConfig, load_config
from src.crawler.pubmed import PubMedClient
from src.crawler.query_translator import QueryTranslator
from src.embedding.bge_m3 import BGEM3Embedder
from src.ingestion.chunker import DocumentChunker
from src.reranker.bge_reranker import BGEReranker
from src.retrieval.dense_index import DenseIndex
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.qdrant_index import QdrantLocalIndex
from src.retrieval.sparse_index import SparseIndex
from src.submission.formatter import SubmissionPackage


class MedicalRetrievalPipeline:
    """Orchestrates indexing, hybrid retrieval, reranking, and submission generation."""

    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or load_config()
        self.chunker = DocumentChunker(
            max_chunk_size=self.config.chunking.max_chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            min_chunk_size=self.config.chunking.min_chunk_size,
            split_by_sentences=self.config.chunking.split_by_sentences,
            enable_contextual=self.config.chunking.enable_contextual,
        )
        self.embedder = BGEM3Embedder(
            model_name=self.config.embedding.model_name,
            batch_size=self.config.embedding.batch_size,
            max_length=self.config.embedding.max_length,
            normalize_embeddings=self.config.embedding.normalize_embeddings,
            device=self.config.embedding.device,
            use_fp16=self.config.embedding.use_fp16,
        )
        self.reranker = None
        if self.config.reranker.enabled:
            self.reranker = BGEReranker(
                model_name=self.config.reranker.model_name,
                batch_size=self.config.reranker.batch_size,
                device=self.config.reranker.device,
                use_fp16=self.config.reranker.use_fp16,
            )

        self.translator = None
        if self.config.query_translation.enabled:
            self.translator = QueryTranslator(
                model_name=self.config.query_translation.model_name,
                prompt_prefix=getattr(self.config.query_translation, "prompt_prefix", ""),
                device=self.config.query_translation.device,
                lexicon_path=self.config.query_translation.lexicon_path,
                zh_lexicon_path=getattr(
                    self.config.query_translation, "zh_lexicon_path", "data/lexicon/icd10_vi_zh.json"
                ),
                stopwords_path=self.config.query_translation.stopwords_path,
            )

        self.pubmed_client = None
        if self.config.pubmed.enabled:
            self.pubmed_client = PubMedClient(
                email=self.config.crawler.ncbi_email,
                api_key=self.config.crawler.ncbi_api_key,
                timeout_seconds=self.config.crawler.timeout_seconds,
                cache_file=self.config.pubmed.cache_file,
            )

        self.dense_index: DenseIndex | None = None
        self.sparse_index: SparseIndex | None = None
        self.hybrid_retriever: HybridRetriever | None = None
        self.qdrant_index: QdrantLocalIndex | None = None

        if self.config.retrieval.engine == "qdrant":
            self.qdrant_index = QdrantLocalIndex(
                storage_path=self.config.retrieval.qdrant_path,
                collection_name=self.config.retrieval.collection_name,
                dimension=1024,
            )

    def build_indices(
        self,
        articles_file: str | Path = "data/processed/all_articles.jsonl",
        output_indices_dir: str | Path = "data/indices",
    ):
        """Chunks articles, encodes them, and builds both FAISS and BM25 indices."""
        art_path = Path(articles_file)
        if not art_path.exists():
            raise FileNotFoundError(
                f"Articles file {art_path} not found. Please crawl or collect data first."
            )

        # 1. Load articles
        logger.info(f"Loading articles from {art_path}...")
        articles = []
        with open(art_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))

        logger.info(f"Loaded {len(articles)} documents. Starting chunking...")

        # 2. Chunking
        all_chunks: list[dict[str, Any]] = []
        for art in tqdm(articles, desc="Chunking documents"):
            doc_id = str(art.get("doc_id") or art.get("id"))
            text = art.get("text", "")
            lang = art.get("lang")
            meta = {
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": art.get("source", "unknown"),
            }
            chunks = self.chunker.chunk_document(doc_id=doc_id, text=text, metadata=meta, lang=lang)
            for c in chunks:
                all_chunks.append(c.model_dump())

        logger.info(f"Created {len(all_chunks)} chunks from {len(articles)} documents.")

        # Save processed chunks
        chunks_file = Path(self.config.paths.processed_data_dir) / "chunks.jsonl"
        chunks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_file, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        logger.info(f"Saved chunk metadata to {chunks_file}")

        # 3. Dense & Sparse Indexing
        if self.config.retrieval.engine == "qdrant":
            indexing_batch_size = 2000
            total_chunks = len(all_chunks)
            logger.info(
                f"Indexing {total_chunks:,} chunks into Qdrant in batches of {indexing_batch_size:,}..."
            )

            if self.qdrant_index is None:
                self.qdrant_index = QdrantLocalIndex(
                    storage_path=self.config.retrieval.qdrant_path,
                    collection_name=self.config.retrieval.collection_name,
                    dimension=1024,
                )

            for start_idx in range(0, total_chunks, indexing_batch_size):
                end_idx = min(start_idx + indexing_batch_size, total_chunks)
                batch_chunks = all_chunks[start_idx:end_idx]
                chunk_texts = [c.get("contextual_text") or c["chunk_text"] for c in batch_chunks]
                embeddings, sparse_weights = self.embedder.encode_both(chunk_texts)
                self.qdrant_index.add_chunks(batch_chunks, embeddings, sparse_weights=sparse_weights)
                logger.info(
                    f"Indexed {end_idx:,}/{total_chunks:,} chunks into Qdrant (Total in DB: {self.qdrant_index.count():,})."
                )

            logger.info(
                f"Qdrant collection '{self.config.retrieval.collection_name}' ready with {self.qdrant_index.count():,} chunks."
            )
        else:
            logger.info("Computing BGE-M3 dense and native lexical sparse embeddings...")
            chunk_texts = [c.get("contextual_text") or c["chunk_text"] for c in all_chunks]
            embeddings, sparse_weights = self.embedder.encode_both(chunk_texts, show_progress_bar=True)
            chunk_ids = [c["chunk_id"] for c in all_chunks]
            dense_idx = DenseIndex(dimension=embeddings.shape[1])
            dense_idx.add(embeddings=embeddings, chunk_ids=chunk_ids, chunk_metadata=all_chunks)
            dense_idx.save(output_indices_dir)

            sparse_idx = SparseIndex()
            sparse_idx.build(all_chunks)
            sparse_idx.save(output_indices_dir)

        logger.info("Indices successfully built and persisted.")

    def load_indices(self, indices_dir: str | Path = "data/indices"):
        """Loads FAISS and BM25 indices from disk (fallback when engine == 'faiss')."""
        idx_dir = Path(indices_dir)
        try:
            self.dense_index = DenseIndex.load(idx_dir)
        except Exception as e:
            logger.warning(f"Could not load DenseIndex: {e}")

        try:
            self.sparse_index = SparseIndex.load(idx_dir)
            self.hybrid_retriever = HybridRetriever(
                dense_index=self.dense_index,
                sparse_index=self.sparse_index,
                fusion_method=self.config.retrieval.fusion_method,
                rrf_k=self.config.retrieval.rrf_k,
                dense_weight=self.config.retrieval.dense_weight,
                sparse_weight=self.config.retrieval.sparse_weight,
            )
        except Exception:
            logger.debug("SparseIndex not loaded. Operating in dense-only mode.")
        logger.info("Indices successfully loaded into memory.")

    def search_query(self, query: str) -> dict[str, Any]:
        """Performs end-to-end retrieval for a single query across VI, ZH, and EN."""
        if self.config.retrieval.engine == "faiss" and self.dense_index is None and self.hybrid_retriever is None:
            try:
                self.load_indices(self.config.paths.indices_dir)
            except Exception as e:
                logger.warning(
                    f"Could not load pre-built local indices ({e}). Proceeding with PubMed only if enabled."
                )

        # 1. Embed query (dense + native lexical sparse)
        query_text_for_search = query
        if self.translator:
            parts = [query]
            acr_exp = self.translator.expand_acronyms(query)
            if acr_exp.get("vi"):
                parts.append(acr_exp["vi"])
            zh_kw = self.translator.extract_chinese_keywords(query)
            if zh_kw:
                parts.append(zh_kw)
            if len(parts) > 1:
                query_text_for_search = " ".join(parts)
                logger.debug(f"Enriched query for search: '{query_text_for_search}'")

        q_dense, q_sparse = self.embedder.encode_both([query_text_for_search])
        q_emb = q_dense[0]
        q_sparse_dict = q_sparse[0] if q_sparse else None

        # 2. Local Hybrid / Dense Search (VI & ZH from pre-built indices)
        local_candidates: list[dict[str, Any]] = []
        if self.config.retrieval.engine == "qdrant" and self.qdrant_index is not None:
            if self.qdrant_index.count() > 0:
                local_candidates = self.qdrant_index.search(
                    query_text=query_text_for_search,
                    query_embedding=q_emb,
                    query_sparse=q_sparse_dict,
                    top_k=self.config.retrieval.hybrid_top_k,
                )
        elif self.hybrid_retriever is not None:
            local_candidates = self.hybrid_retriever.search(
                query_text=query,
                query_embedding=q_emb,
                top_k=self.config.retrieval.hybrid_top_k,
                dense_top_k=self.config.retrieval.dense_top_k,
                sparse_top_k=self.config.retrieval.sparse_top_k,
            )
        elif self.dense_index is not None:
            dense_results = self.dense_index.search(
                query_embedding=q_emb,
                top_k=self.config.retrieval.hybrid_top_k,
            )
            local_candidates = [
                {
                    "doc_id": r[2].get("doc_id"),
                    "chunk_id": r[0],
                    "chunk_text": r[2].get("chunk_text"),
                    "score": r[1],
                    "lang": r[2].get("lang", "vi"),
                    "metadata": {"title": r[2].get("title", "")},
                }
                for r in dense_results
            ]

        # 3. Dynamic English Retrieval via PubMed (EN)
        pubmed_candidates: list[dict[str, Any]] = []
        if self.pubmed_client and self.config.pubmed.enabled:
            if self.translator and self.config.query_translation.enabled:
                en_query = self.translator.extract_pubmed_keywords(query)
            else:
                en_query = query

            logger.info(f"Querying PubMed with extracted keywords: '{en_query}'")
            articles = self.pubmed_client.search_candidate_articles(
                query=en_query,
                top_k=self.config.pubmed.max_candidates_per_query,
                source=self.config.pubmed.source_api,
            )

            for art in articles:
                chunks = self.chunker.chunk_document(
                    doc_id=str(art["doc_id"]),
                    text=art["text"],
                    metadata={"title": art.get("title", ""), "source": art.get("source", "pubmed")},
                    lang="en",
                )
                for c in chunks:
                    pubmed_candidates.append(c.model_dump())

        # 4. Merge candidates
        all_candidates = local_candidates + pubmed_candidates
        if not all_candidates:
            logger.warning(f"No candidate documents found for query: '{query}'")
            return {"relevant_docs": [], "relevant_chunks": []}

        if self.reranker and self.config.reranker.enabled:
            doc_ids, relevant_chunks = self.reranker.rerank(
                query=query,
                candidates=all_candidates,
                top_k_chunks=self.config.reranker.top_k_chunks,
                top_k_docs=self.config.reranker.top_k_docs,
                score_threshold=self.config.reranker.score_threshold,
                max_chunks_per_doc=getattr(self.config.reranker, "max_chunks_per_doc", 2),
            )

        else:
            seen_docs = set()
            doc_ids = []
            relevant_chunks = []
            for c in all_candidates[: self.config.reranker.top_k_chunks]:
                did = str(c["doc_id"])
                if did not in seen_docs:
                    seen_docs.add(did)
                    doc_ids.append(did)
                relevant_chunks.append({"doc_id": did, "chunk_text": c["chunk_text"]})
            doc_ids = doc_ids[: self.config.reranker.top_k_docs]

        return {
            "relevant_docs": doc_ids,
            "relevant_chunks": relevant_chunks,
        }

    def predict_queries_jsonl(self, queries_file: str | Path) -> list[dict[str, Any]]:
        """Runs predictions for all queries in a JSONL or Parquet file."""
        q_path = Path(queries_file)
        queries = []
        if q_path.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(q_path)
            queries = table.to_pylist()
        else:
            with open(q_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        queries.append(json.loads(line))

        predictions = []
        logger.info(f"Predicting {len(queries)} queries...")
        for q in tqdm(queries, desc="Evaluating queries"):
            qid = int(q["id"])
            query_text = q["query"]
            result = self.search_query(query_text)
            predictions.append(
                {
                    "id": qid,
                    "relevant_docs": result["relevant_docs"],
                    "relevant_chunks": result["relevant_chunks"],
                }
            )

        return predictions

    def generate_submission(
        self,
        queries_file: str | Path,
        submission_filename: str = "submission.json",
        zip_filename: str | None = None,
    ) -> Path:
        """Runs batch prediction and packages official submission ZIP file."""
        preds = self.predict_queries_jsonl(queries_file)
        json_file, zip_file = SubmissionPackage.save_and_package(
            predictions=preds,
            output_dir=self.config.paths.submissions_dir,
            submission_filename=submission_filename,
            zip_filename=zip_filename,
        )
        return zip_file
