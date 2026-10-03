"""Precomputes query enrichment and caches PubMed candidate documents for all queries.

Generates data/processed/queries_enriched.jsonl containing:
- query_id
- original_query
- acronym_expansion (vi, en, zh)
- zh_keywords (for Chinese medical corpus matching)
- en_keywords (for PubMed retrieval)
- translated_en
- search_query_text (enriched query for FAISS Dense + BM25 Sparse retrieval)
- pubmed_candidate_ids (list of PMIDs pre-fetched into pubmed_cache.jsonl)

This enables 100% OFFLINE and 10x FASTER inference on Kaggle/Colab without API rate limits or network drops.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from src.crawler.pubmed import PubMedClient
from src.crawler.query_translator import QueryTranslator


def load_queries(file_path: Path) -> list[dict[str, Any]]:
    """Loads queries from JSONL or Parquet file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Queries file not found at {file_path}")

    queries = []
    if file_path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(file_path)
        for row in table.to_pylist():
            qid = row.get("id") or row.get("query_id")
            qtext = row.get("query") or row.get("text")
            queries.append({"id": qid, "query": qtext})
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    qid = item.get("id") or item.get("query_id")
                    qtext = item.get("query") or item.get("text")
                    queries.append({"id": qid, "query": qtext})

    logger.info(f"Loaded {len(queries):,} queries from {file_path}")
    return queries


def precompute_queries(
    queries_file: Path,
    output_file: Path,
    cache_file: Path = Path("data/processed/pubmed_cache.jsonl"),
    fetch_pubmed: bool = True,
    pubmed_top_k: int = 30,
    batch_size: int = 32,
    device: str = "auto",
):
    """Processes all queries: translates, extracts keywords, pre-fetches PubMed candidates, and saves."""
    queries = load_queries(queries_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Translator
    logger.info("Initializing QueryTranslator (Opus-MT / Medical Lexicons)...")
    translator = QueryTranslator(device=device)

    # 2. Batch Translation to English (High speed on GPU)
    raw_texts = [q["query"] for q in queries]
    logger.info(f"Batch translating {len(raw_texts):,} queries to English (batch_size={batch_size})...")
    t0 = time.time()
    translated_texts = translator.translate_batch(raw_texts, batch_size=batch_size)
    logger.info(f"Translation completed in {time.time() - t0:.2f}s.")

    # 3. Initialize PubMed Client if fetching is enabled
    pubmed_client = None
    if fetch_pubmed:
        logger.info(f"Initializing PubMedClient (cache: {cache_file})...")
        pubmed_client = PubMedClient(cache_file=cache_file)

    # 4. Process each query
    logger.info("Extracting keywords and pre-fetching PubMed candidates...")
    enriched_records = []

    # Check if partial output already exists to allow resume
    existing_by_id = {}
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        existing_by_id[item["id"]] = item
            logger.info(f"Found {len(existing_by_id)} already precomputed queries. Will resume/reuse.")
        except Exception:
            pass

    for i, item in enumerate(tqdm(queries, desc="Enriching queries")):
        qid = item["id"]
        qtext = item["query"]

        # If already fully enriched with pubmed_candidate_ids, reuse
        if qid in existing_by_id and ("pubmed_candidate_ids" in existing_by_id[qid] or not fetch_pubmed):
            enriched_records.append(existing_by_id[qid])
            continue

        trans_en = translated_texts[i]
        acr_exp = translator.expand_acronyms(qtext)
        zh_kw = translator.extract_chinese_keywords(qtext)
        en_kw = translator.extract_pubmed_keywords(qtext, translated=trans_en)

        # Build enriched text for local FAISS + BM25 retrieval
        search_parts = [qtext]
        if acr_exp.get("vi"):
            search_parts.append(acr_exp["vi"])
        if zh_kw:
            search_parts.append(zh_kw)
        search_query_text = " ".join(search_parts)

        # Fetch PubMed candidates
        pubmed_candidate_ids = []
        if pubmed_client and en_kw:
            try:
                articles = pubmed_client.search_candidate_articles(
                    query=en_kw,
                    top_k=pubmed_top_k,
                    source="europe_pmc",
                )
                pubmed_candidate_ids = [str(a["doc_id"]) for a in articles]
            except Exception as e:
                logger.warning(f"Failed to fetch PubMed for query #{qid} ('{en_kw}'): {e}")

        record = {
            "id": qid,
            "original_query": qtext,
            "translated_en": trans_en,
            "acronym_expansion": acr_exp,
            "zh_keywords": zh_kw,
            "en_keywords": en_kw,
            "search_query_text": search_query_text,
            "pubmed_candidate_ids": pubmed_candidate_ids,
        }
        enriched_records.append(record)

        # Periodically save intermediate results every 100 queries
        if (i + 1) % 100 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                for r in enriched_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final write
    with open(output_file, "w", encoding="utf-8") as f:
        for r in enriched_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Successfully saved {len(enriched_records):,} enriched queries to {output_file}!")
    if pubmed_client and pubmed_client.cache_file and pubmed_client.cache_file.exists():
        cache_size_mb = pubmed_client.cache_file.stat().st_size / (1024 * 1024)
        logger.info(f"PubMed cache updated: {len(pubmed_client._memory_cache):,} articles ({cache_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Precompute query enrichment and cache PubMed articles.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/raw/queries.jsonl"),
        help="Path to queries JSONL or Parquet file (default: data/raw/queries.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/queries_enriched.jsonl"),
        help="Path to output enriched queries JSONL file.",
    )
    parser.add_argument(
        "--pubmed-cache",
        type=Path,
        default=Path("data/processed/pubmed_cache.jsonl"),
        help="Path to PubMed cache file (default: data/processed/pubmed_cache.jsonl).",
    )
    parser.add_argument(
        "--pubmed-top-k",
        type=int,
        default=30,
        help="Number of PubMed candidate articles to retrieve per query (default: 30).",
    )
    parser.add_argument(
        "--no-fetch-pubmed",
        action="store_true",
        help="Only extract keywords and translations; skip calling PubMed API.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for MarianMT translation (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device ('cuda', 'cpu', or 'auto').",
    )

    args = parser.parse_args()

    precompute_queries(
        queries_file=args.queries,
        output_file=args.output,
        cache_file=args.pubmed_cache,
        fetch_pubmed=not args.no_fetch_pubmed,
        pubmed_top_k=args.pubmed_top_k,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
