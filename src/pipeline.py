"""End-to-end Pipeline orchestrator for Medical Document Retrieval."""

import json
import time
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
from src.retrieval.multi_query import build_subqueries, fuse_candidates
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

        self._enriched_queries: dict[str, dict[str, Any]] = {}
        enriched_file = Path(self.config.paths.processed_data_dir) / "queries_enriched.jsonl"
        if enriched_file.exists():
            try:
                with open(enriched_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            qtext = item.get("original_query", "").strip()
                            if qtext:
                                self._enriched_queries[qtext] = item
                logger.info(
                    f"Loaded {len(self._enriched_queries):,} precomputed enriched queries from {enriched_file} (offline accelerated)."
                )
            except Exception as e:
                logger.warning(f"Could not load precomputed queries: {e}")

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
        if art_path.suffix == ".parquet" or art_path.is_dir():
            import pyarrow.parquet as pq
            if art_path.is_dir():
                parquet_files = sorted(art_path.glob("*.parquet"))
                if not parquet_files:
                    raise FileNotFoundError(f"No .parquet files found in directory {art_path}")
                logger.info(f"Reading {len(parquet_files)} Parquet shards from {art_path}...")
                table = pq.read_table(parquet_files, columns=["doc_id", "url", "title", "text", "lang"])
            else:
                table = pq.read_table(art_path, columns=["doc_id", "url", "title", "text", "lang"])
            articles = table.to_pylist()
        else:
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
            logger.info("Computing BGE-M3 dense embeddings (streamed in batches to prevent RAM overflow)...")
            dense_idx = DenseIndex(dimension=1024)
            total_chunks = len(all_chunks)
            indexing_batch_size = 5000  # 5,000 chunks per batch keeps RAM constant under 2GB

            for start_idx in range(0, total_chunks, indexing_batch_size):
                end_idx = min(start_idx + indexing_batch_size, total_chunks)
                batch_chunks = all_chunks[start_idx:end_idx]
                chunk_texts = [c.get("contextual_text") or c["chunk_text"] for c in batch_chunks]
                chunk_ids = [c["chunk_id"] for c in batch_chunks]

                # Encode dense only at maximum GPU tensor speed (no sparse dictionary overhead)
                batch_embeddings = self.embedder.encode(chunk_texts)
                dense_idx.add(embeddings=batch_embeddings, chunk_ids=chunk_ids, chunk_metadata=batch_chunks)
                logger.info(
                    f"Indexed {end_idx:,}/{total_chunks:,} chunks into FAISS (Progress: {end_idx/total_chunks*100:.1f}%)."
                )

            dense_idx.save(output_indices_dir)

            sparse_idx = SparseIndex()
            sparse_idx.build(all_chunks)
            sparse_idx.save(output_indices_dir)

            # Keep in-memory instances ready for inference immediately
            self.dense_index = dense_idx
            self.sparse_index = sparse_idx
            if self.dense_index is not None and self.sparse_index is not None:
                self.hybrid_retriever = HybridRetriever(
                    dense_index=self.dense_index,
                    sparse_index=self.sparse_index,
                    fusion_method=self.config.retrieval.fusion_method,
                    rrf_k=self.config.retrieval.rrf_k,
                    dense_weight=self.config.retrieval.dense_weight,
                    sparse_weight=self.config.retrieval.sparse_weight,
                )

        logger.info("Indices successfully built, persisted, and initialized in memory.")

    def load_indices(
        self,
        indices_dir: str | Path = "data/indices",
        dense_dir: str | Path | None = None,
        sparse_dir: str | Path | None = None,
        chunks_meta_path: str | Path | None = None,
    ):
        """Loads FAISS and BM25 indices from disk (supports separate paths for dense, sparse, and metadata)."""
        idx_dir = Path(indices_dir)
        d_dir = Path(dense_dir) if dense_dir else idx_dir
        s_dir = Path(sparse_dir) if sparse_dir else idx_dir
        meta_p = Path(chunks_meta_path) if chunks_meta_path else None

        logger.info(f"Loading indices: Dense path='{d_dir}' | Sparse (BM25) path='{s_dir}' | ChunksMeta='{meta_p}'")
        try:
            self.dense_index = DenseIndex.load(d_dir, fallback_meta_dir=s_dir, chunks_meta_path=meta_p)
        except Exception as e:
            logger.warning(f"Could not load DenseIndex: {e}")

        try:
            self.sparse_index = SparseIndex.load(s_dir, fallback_meta_dir=d_dir, chunks_meta_path=meta_p)
            if self.dense_index is not None and self.sparse_index is not None:
                self._share_chunk_meta_store()
                self.hybrid_retriever = HybridRetriever(
                    dense_index=self.dense_index,
                    sparse_index=self.sparse_index,
                    fusion_method=self.config.retrieval.fusion_method,
                    rrf_k=self.config.retrieval.rrf_k,
                    dense_weight=self.config.retrieval.dense_weight,
                    sparse_weight=self.config.retrieval.sparse_weight,
                )
                logger.info("HybridRetriever (Dense + Sparse) successfully initialized.")
        except FileNotFoundError as e:
            logger.warning(f"SparseIndex not found: {e}. Operating in dense-only mode.")
        except Exception as e:
            logger.warning(f"Failed to load SparseIndex ({type(e).__name__}: {e}). Operating in dense-only mode.")
        logger.info("Indices loading process completed.")

    def _share_chunk_meta_store(self):
        """Reuses one metadata store when dense and sparse read the same SQLite file."""
        dense_store = getattr(self.dense_index, "meta_store", None)
        sparse_store = getattr(self.sparse_index, "meta_store", None)
        if dense_store is None or sparse_store is None:
            return
        dense_db = self.dense_index._sqlite_conn.execute("PRAGMA database_list").fetchone()[2]
        sparse_db = self.sparse_index._sqlite_conn.execute("PRAGMA database_list").fetchone()[2]
        if dense_db and dense_db == sparse_db:
            self.sparse_index.meta_store = dense_store
            logger.info("Shared chunk metadata store between dense and sparse indices.")

    def search_query(self, query: str) -> dict[str, Any]:
        """Performs end-to-end retrieval for a single query across VI, ZH, and EN."""
        if self.config.retrieval.engine == "faiss" and self.dense_index is None and self.hybrid_retriever is None:
            try:
                self.load_indices(self.config.paths.indices_dir)
            except Exception as e:
                logger.warning(
                    f"Could not load pre-built local indices ({e}). Proceeding with PubMed only if enabled."
                )

        # Check if precomputed enrichment exists
        enriched = self._enriched_queries.get(query.strip())

        # 1. Embed query (dense + native lexical sparse)
        if enriched and "search_query_text" in enriched:
            query_text_for_search = enriched["search_query_text"]
        else:
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

        mq_config = self.config.retrieval.multi_query
        subqueries: list[str] = []
        if mq_config.enabled and self.hybrid_retriever is not None:
            subqueries = build_subqueries(
                query_text_for_search,
                max_subqueries=mq_config.max_subqueries,
                min_chars=mq_config.min_chars,
            )

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
            search_kwargs = {
                "top_k": self.config.retrieval.hybrid_top_k,
                "dense_top_k": self.config.retrieval.dense_top_k,
                "sparse_top_k": self.config.retrieval.sparse_top_k,
            }
            if subqueries:
                sub_embs = self.embedder.encode(subqueries)
                weighted_lists = [
                    (
                        1.0,
                        self.hybrid_retriever.search(
                            query_text=query_text_for_search,
                            query_embedding=q_emb,
                            **search_kwargs,
                        ),
                    )
                ]
                for subquery, sub_emb in zip(subqueries, sub_embs):
                    weighted_lists.append(
                        (
                            mq_config.subquery_weight,
                            self.hybrid_retriever.search(
                                query_text=subquery,
                                query_embedding=sub_emb,
                                **search_kwargs,
                            ),
                        )
                    )
                local_candidates = fuse_candidates(
                    weighted_lists,
                    top_k=self.config.retrieval.hybrid_top_k,
                    rrf_k=self.config.retrieval.rrf_k,
                )
            else:
                local_candidates = self.hybrid_retriever.search(
                    query_text=query_text_for_search,
                    query_embedding=q_emb,
                    **search_kwargs,
                )
        elif self.dense_index is not None:
            # Multi-lingual Multi-query Dense Retrieval (VI + EN Translation + ZH Keywords)
            dense_queries = [(1.0, q_emb)]

            # 1. English Translated Query Embedding (for matching English PubMed articles)
            en_text = None
            if enriched and enriched.get("translated_en"):
                en_text = enriched["translated_en"]
            elif self.translator and self.config.query_translation.enabled:
                en_text = self.translator.translate_to_english(query)

            if en_text and en_text.strip():
                en_dense = self.embedder.encode([en_text.strip()])
                dense_queries.append((0.85, en_dense[0]))

            # 2. Chinese Medical Keywords Embedding (for matching Chinese clinical corpus)
            zh_text = None
            if enriched and enriched.get("zh_keywords"):
                zh_text = enriched["zh_keywords"]
            elif self.translator:
                zh_text = self.translator.extract_chinese_keywords(query)

            if zh_text and zh_text.strip():
                zh_dense = self.embedder.encode([zh_text.strip()])
                dense_queries.append((0.75, zh_dense[0]))

            # 3. Search and fuse multilingual results using Reciprocal Rank Fusion (RRF)
            candidate_pool: dict[str, dict[str, Any]] = {}
            for weight, emb in dense_queries:
                d_res = self.dense_index.search(
                    query_embedding=emb,
                    top_k=self.config.retrieval.hybrid_top_k,
                )
                for rank, (cid, score, meta) in enumerate(d_res):
                    key = str(cid)
                    rrf_contrib = weight * (1.0 / (60.0 + rank))
                    if key not in candidate_pool:
                        candidate_pool[key] = {
                            "doc_id": meta.get("doc_id"),
                            "chunk_id": cid,
                            "chunk_text": meta.get("chunk_text"),
                            "contextual_text": meta.get("contextual_text") or meta.get("chunk_text"),
                            "score": rrf_contrib,
                            "dense_score": float(score),
                            "lang": meta.get("lang", "vi"),
                            "metadata": {"title": meta.get("title", "")},
                        }
                    else:
                        candidate_pool[key]["score"] += rrf_contrib
                        if float(score) > candidate_pool[key].get("dense_score", -999.0):
                            candidate_pool[key]["dense_score"] = float(score)

            local_candidates = sorted(
                candidate_pool.values(),
                key=lambda x: x["score"],
                reverse=True,
            )[: self.config.retrieval.hybrid_top_k]

        # 3. Dynamic English Retrieval via PubMed (EN)
        pubmed_candidates: list[dict[str, Any]] = []
        if self.pubmed_client and self.config.pubmed.enabled:
            articles = []
            if enriched and "pubmed_candidate_ids" in enriched:
                # Fast offline retrieval from memory cache
                for pmid in enriched["pubmed_candidate_ids"]:
                    cached_art = self.pubmed_client._memory_cache.get(str(pmid))
                    if cached_art:
                        articles.append(cached_art)

            if not articles:
                if enriched and enriched.get("en_keywords"):
                    en_query = enriched["en_keywords"]
                elif self.translator and self.config.query_translation.enabled:
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
            expand_top_docs = getattr(self.config.reranker, "expand_top_docs", 0)
            expand_provider = None
            has_meta = self.dense_index is not None and self.dense_index.meta_store is not None
            if expand_top_docs > 0 and has_meta:
                def expand_provider(doc_ids, max_per_doc=8):
                    return self.dense_index.get_doc_chunks(
                        doc_ids, query=query, max_per_doc=max_per_doc
                    )

            doc_ids, relevant_chunks = self.reranker.rerank(
                query=query,
                candidates=all_candidates,
                top_k_chunks=self.config.reranker.top_k_chunks,
                top_k_docs=self.config.reranker.top_k_docs,
                score_threshold=self.config.reranker.score_threshold,
                max_chunks_per_doc=getattr(self.config.reranker, "max_chunks_per_doc", 2),
                expand_provider=expand_provider,
                expand_top_docs=expand_top_docs,
                expand_max_chunks_per_doc=getattr(
                    self.config.reranker, "expand_max_chunks_per_doc", 8
                ),
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

    def predict_queries_jsonl(
        self,
        queries_file: str | Path,
        offset: int = 0,
        limit: int | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Runs predictions for queries, optionally sharded and checkpointed.

        Args:
            queries_file: JSONL or Parquet file with queries.
            offset: Index of the first query handled by this shard.
            limit: Number of queries in this shard (None = to the end of the file).
            checkpoint_path: Optional JSONL file. Existing predictions are reused and
                every new prediction is flushed immediately, so an interrupted run
                resumes without re-scoring finished queries.
        """
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

        shard = queries[offset:] if limit is None else queries[offset : offset + limit]

        done: dict[int, dict[str, Any]] = {}
        ckpt = Path(checkpoint_path) if checkpoint_path else None
        if ckpt is not None and ckpt.exists():
            with open(ckpt, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        done[int(item["id"])] = item
            logger.info(f"Resuming from {ckpt}: {len(done):,} predictions already done.")

        pending = [q for q in shard if int(q["id"]) not in done]
        logger.info(
            f"Predicting {len(pending):,} queries (offset={offset}, limit={limit}, "
            f"shard={len(shard):,}, reused={len(shard) - len(pending):,})."
        )

        cf = None
        if ckpt is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            cf = open(ckpt, "a", encoding="utf-8")

        try:
            started = time.monotonic()
            for done_count, q in enumerate(tqdm(pending, desc="Evaluating queries"), start=1):
                qid = int(q["id"])
                query_text = q.get("query") or q.get("original_query") or ""
                result = self.search_query(query_text)
                prediction = {
                    "id": qid,
                    "relevant_docs": result["relevant_docs"],
                    "relevant_chunks": result["relevant_chunks"],
                }
                done[qid] = prediction
                if cf is not None:
                    cf.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    cf.flush()

                if done_count % 50 == 0:
                    elapsed = time.monotonic() - started
                    rate = done_count / elapsed
                    eta_min = (len(pending) - done_count) / rate / 60
                    logger.info(
                        f"Progress {done_count:,}/{len(pending):,} | {rate:.3f} q/s | "
                        f"elapsed {elapsed / 60:.1f} min | ETA {eta_min:.1f} min"
                    )
        finally:
            if cf is not None:
                cf.close()

        return [done[int(q["id"])] for q in shard if int(q["id"]) in done]

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
