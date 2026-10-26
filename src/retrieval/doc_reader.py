"""Document and Chunk Reader from SQLite corpus and PubMed cache."""

from __future__ import annotations

import json
import sqlite3
import textwrap
import threading
from pathlib import Path
from typing import Any

from IPython.display import HTML, display


def get_document_and_chunks(
    doc_id: int | str,
    db_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
    pubmed_cache_path: str | Path = "data/processed/pubmed_cache.jsonl",
) -> dict[str, Any] | None:
    """Retrieves the full document title, language, all constituent chunks, and stitched full text.

    Args:
        doc_id: The document identifier (integer or string).
        db_path: Path to chunks_meta_slim.sqlite.
        pubmed_cache_path: Path to pubmed_cache.jsonl.

    Returns:
        Dict containing doc_id, title, lang, total_chunks, full_text, and list of chunks.
    """
    db_path = Path(db_path)
    if db_path.exists():
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cur = conn.cursor()
        cols = [c[1] for c in cur.execute("PRAGMA table_info(chunks)").fetchall()]
        has_url = "url" in cols
        select_sql = (
            "SELECT chunk_id, title, url, chunk_text, lang FROM chunks WHERE doc_id = ? ORDER BY chunk_id"
            if has_url
            else "SELECT chunk_id, title, chunk_text, lang FROM chunks WHERE doc_id = ? ORDER BY chunk_id"
        )
        rows = []
        try:
            cur.execute(select_sql, (int(doc_id),))
            rows = cur.fetchall()
        except Exception:
            pass

        if not rows:
            cur.execute(select_sql, (str(doc_id),))
            rows = cur.fetchall()
        conn.close()

        if rows:
            title = rows[0][1] or f"Document #{doc_id}"
            if has_url:
                url = rows[0][2] or ""
                chunks = [{"chunk_id": r[0], "chunk_text": r[3]} for r in rows]
                lang = rows[0][4] or "vi"
            else:
                url = ""
                chunks = [{"chunk_id": r[0], "chunk_text": r[2]} for r in rows]
                lang = rows[0][3] or "vi"

            full_text = "\n\n".join([c["chunk_text"] for c in chunks])
            return {
                "doc_id": doc_id,
                "title": title,
                "url": url,
                "lang": lang,
                "source": "sqlite_corpus",
                "total_chunks": len(chunks),
                "full_text": full_text,
                "chunks": chunks,
            }

    # Fallback to PubMed cache
    pm_path = Path(pubmed_cache_path)
    if pm_path.exists():
        with open(pm_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if str(item.get("doc_id")) == str(doc_id) or str(item.get("pmid")) == str(doc_id):
                    text = item.get("full_text") or item.get("abstract") or ""
                    pmid = item.get("pmid") or doc_id
                    url = item.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    return {
                        "doc_id": doc_id,
                        "title": item.get("title", f"PubMed Article #{doc_id}"),
                        "url": url,
                        "lang": "en",
                        "source": "pubmed_cache",
                        "total_chunks": 1,
                        "full_text": text,
                        "chunks": [{"chunk_id": 0, "chunk_text": text}],
                    }

    return None


def display_document(
    doc_id: int | str,
    db_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
    pubmed_cache_path: str | Path = "data/processed/pubmed_cache.jsonl",
    show_chunks: bool = True,
    max_chunk_chars: int | None = None,
) -> None:
    """Renders a beautifully styled, text-wrapped HTML card in Jupyter for a document and its chunks."""
    doc = get_document_and_chunks(doc_id, db_path=db_path, pubmed_cache_path=pubmed_cache_path)
    if not doc:
        print(f"❌ Không tìm thấy tài liệu với doc_id = {doc_id} trong database.")
        return

    lang_badge = f"<span style='background:#0d6efd; color:white; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:bold; margin-left:8px;'>{doc['lang'].upper()}</span>"
    chunks_badge = f"<span style='background:#198754; color:white; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:bold; margin-left:6px;'>{doc['total_chunks']} CHUNKS</span>"

    chunks_html = ""
    if show_chunks and doc["chunks"]:
        chunks_html += "<div style='margin-top:16px;'><h4 style='color:#333; margin-bottom:8px;'>📑 Danh sách các Chunks thuộc Document:</h4>"
        for i, c in enumerate(doc["chunks"], 1):
            text = c["chunk_text"]
            if max_chunk_chars and len(text) > max_chunk_chars:
                text = text[:max_chunk_chars] + "..."
            # Escape HTML to prevent injection issues while preserving line breaks
            safe_text = (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            chunks_html += f"""
            <div style='background:#f8fafc !important; color:#0f172a !important; border:1px solid #cbd5e1; border-left:4px solid #0d6efd; border-radius:6px; padding:12px; margin-bottom:12px; white-space:pre-wrap; word-wrap:break-word; word-break:break-word; line-height:1.6;'>
                <div style='font-size:12px; font-weight:bold; color:#1d4ed8 !important; margin-bottom:6px;'>Chunk #{i}/{doc['total_chunks']} (Chunk ID: {c['chunk_id']})</div>
                <div style='font-size:14px; color:#0f172a !important;'>{safe_text}</div>
            </div>
            """
        chunks_html += "</div>"

    url = doc.get("url") or ""
    url_html = (
        f"<div style='font-size:12px; margin-top:4px;'>🔗 URL Nguồn: <a href='{url}' target='_blank' style='color:#0d6efd; text-decoration:none;'>{url} ↗</a></div>"
        if url
        else ""
    )

    html = f"""
    <div style='border:1px solid #cbd5e1; border-radius:8px; padding:18px; margin:16px 0; background:#ffffff !important; color:#0f172a !important; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
        <div style='border-bottom:1px solid #e2e8f0; padding-bottom:12px; margin-bottom:12px;'>
            <div style='font-size:13px; color:#475569 !important;'>📄 Document ID: <b style='color:#0f172a;'>{doc['doc_id']}</b> {lang_badge} {chunks_badge}</div>
            <h3 style='margin:8px 0 4px 0; color:#0f172a !important; line-height:1.4;'>{doc['title']}</h3>
            {url_html}
            <div style='font-size:12px; color:#64748b !important; margin-top:4px;'>Độ dài toàn văn: {len(doc['full_text']):,} ký tự | Nguồn: {doc['source']}</div>
        </div>
        {chunks_html}
    </div>
    """
    display(HTML(html))


def print_document(
    doc_id: int | str,
    db_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
    width: int = 95,
) -> None:
    """Prints a document and its chunks to the terminal / standard output with clean line wrapping."""
    doc = get_document_and_chunks(doc_id, db_path=db_path)
    if not doc:
        print(f"❌ Không tìm thấy tài liệu với doc_id = {doc_id}")
        return

    separator = "=" * width
    sub_sep = "-" * width
    print(separator)
    print(f"📄 DOC ID: {doc['doc_id']} | NGÔN NGỮ: {doc['lang'].upper()} | SỐ CHUNKS: {doc['total_chunks']}")
    print(f"TIÊU ĐỀ: {doc['title']}")
    if doc.get("url"):
        print(f"URL:     {doc['url']}")
    print(separator)
    for i, c in enumerate(doc["chunks"], 1):
        print(f"\n[Chunk #{i}/{doc['total_chunks']} - ID: {c['chunk_id']}]")
        print(sub_sep)
        for paragraph in c["chunk_text"].split("\n"):
            if paragraph.strip():
                print(textwrap.fill(paragraph, width=width))
            else:
                print()
    print(separator + "\n")


def get_random_crawled_doc(
    raw_dir: str | Path = "data/raw",
    db_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
    target_lang: str | None = None,
) -> dict[str, Any] | None:
    """Randomly samples a document from the raw crawled shards and pulls its constituent chunks.

    Args:
        raw_dir: Directory containing crawled_shard_*.jsonl files.
        db_path: Path to chunks_meta_slim.sqlite.
        target_lang: Optional language code ('vi', 'zh', 'en').

    Returns:
        Dict with doc_id, url, title, lang, raw_text, total_chunks, chunks, char_count, word_count.
    """
    import random

    raw_path = Path(raw_dir)
    shards = list(raw_path.glob("crawled_shard_*.jsonl"))
    data = None
    if shards:
        for _ in range(25):
            shard = random.choice(shards)
            file_size = shard.stat().st_size
            if file_size < 1000:
                continue
            offset = random.randint(0, max(0, file_size - 10000))
            try:
                with open(shard, "rb") as f:
                    f.seek(offset)
                    f.readline()
                    line = f.readline().decode("utf-8", errors="ignore")
                    if line.strip():
                        record = json.loads(line)
                        if target_lang and record.get("lang") != target_lang:
                            continue
                        data = record
                        break
            except Exception:
                continue

    if not data:
        # Fallback to random row in SQLite
        if Path(db_path).exists():
            conn = sqlite3.connect(db_path, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT max(rowid) FROM chunks")
            max_id = cur.fetchone()[0]
            rand_row = random.randint(1, max_id)
            cur.execute("SELECT doc_id FROM chunks WHERE rowid = ?", (rand_row,))
            row = cur.fetchone()
            conn.close()
            if row:
                doc_info = get_document_and_chunks(row[0], db_path=db_path)
                if doc_info:
                    doc_info["raw_text"] = doc_info["full_text"]
                    doc_info["url"] = ""
                    doc_info["char_count"] = len(doc_info["raw_text"])
                    doc_info["word_count"] = len(doc_info["raw_text"].split())
                    return doc_info
        return None

    doc_id = data.get("doc_id")
    raw_text = data.get("text", "")
    title = data.get("title", f"Document #{doc_id}")
    url = data.get("url", "")
    lang = data.get("lang", "vi")

    # Fetch indexed chunks for this document
    chunks = []
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT chunk_id, chunk_text FROM chunks WHERE doc_id = ? ORDER BY chunk_id",
                (int(doc_id),),
            )
            rows = cur.fetchall()
        except Exception:
            rows = []
        if not rows:
            cur.execute(
                "SELECT chunk_id, chunk_text FROM chunks WHERE doc_id = ? ORDER BY chunk_id",
                (str(doc_id),),
            )
            rows = cur.fetchall()
        conn.close()
        chunks = [{"chunk_id": r[0], "chunk_text": r[1]} for r in rows]

    return {
        "doc_id": doc_id,
        "title": title,
        "url": url,
        "lang": lang,
        "raw_text": raw_text,
        "total_chunks": len(chunks),
        "chunks": chunks,
        "char_count": len(raw_text),
        "word_count": len(raw_text.split()),
    }


def display_random_crawled_doc(
    raw_dir: str | Path = "data/raw",
    db_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
    target_lang: str | None = None,
) -> None:
    """Samples a random crawled text and renders it side-by-side with its chunks in Jupyter."""
    doc = get_random_crawled_doc(raw_dir=raw_dir, db_path=db_path, target_lang=target_lang)
    if not doc:
        print("❌ Không thể lấy tài liệu ngẫu nhiên từ data/raw.")
        return

    lang_badge = f"<span style='background:#0d6efd; color:white; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:bold; margin-left:8px;'>{doc['lang'].upper()}</span>"
    chunks_badge = f"<span style='background:#198754; color:white; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:bold; margin-left:6px;'>{doc['total_chunks']} CHUNKS</span>"
    url_link = (
        f"<a href='{doc['url']}' target='_blank' style='color:#0d6efd; text-decoration:none;'>🔗 {doc['url'][:80]}...</a>"
        if doc["url"]
        else "<span style='color:#6c757d;'>Nguồn: Nội bộ</span>"
    )

    chunks_html = ""
    for i, c in enumerate(doc["chunks"], 1):
        text = c["chunk_text"]
        safe_text = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        c_words = len(text.split())
        c_chars = len(text)
        chunks_html += f"""
        <div style='background:#f8fafc !important; color:#0f172a !important; border:1px solid #cbd5e1; border-left:4px solid #198754; border-radius:6px; padding:12px; margin-bottom:12px; white-space:pre-wrap; word-wrap:break-word; word-break:break-word; line-height:1.6;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='font-size:12px; font-weight:bold; color:#15803d !important;'>🧩 Chunk #{i}/{doc['total_chunks']} (ID: {c['chunk_id']})</span>
                <span style='font-size:11px; color:#475569 !important;'>{c_chars} ký tự | {c_words} từ</span>
            </div>
            <div style='font-size:14px; color:#0f172a !important;'>{safe_text}</div>
        </div>
        """

    safe_raw = (
        doc["raw_text"][:2500].replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    if len(doc["raw_text"]) > 2500:
        safe_raw += "<br><i>[...Văn bản còn tiếp tục...]</i>"

    html = f"""
    <div style='border:1px solid #cbd5e1; border-radius:8px; padding:18px; margin:16px 0; background:#ffffff !important; color:#0f172a !important; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
        <div style='border-bottom:1px solid #e2e8f0; padding-bottom:12px; margin-bottom:12px;'>
            <div style='font-size:13px; color:#475569 !important;'>📄 Document ID: <b style='color:#0f172a;'>{doc['doc_id']}</b> {lang_badge} {chunks_badge}</div>
            <h3 style='margin:8px 0 4px 0; color:#0f172a !important; line-height:1.4;'>{doc['title']}</h3>
            <div style='font-size:12px; margin-top:4px;'>{url_link}</div>
            <div style='font-size:12px; color:#475569 !important; margin-top:4px;'>Tổng độ dài văn bản gốc: <b style='color:#0f172a;'>{doc['char_count']:,}</b> ký tự (<b style='color:#0f172a;'>{doc['word_count']:,}</b> từ)</div>
        </div>

        <div style='display:grid; grid-template-columns: 1fr 1fr; gap:16px;'>
            <div>
                <h4 style='color:#1d4ed8 !important; margin-top:0; border-bottom:2px solid #1d4ed8; padding-bottom:4px;'>📜 Văn Bản Gốc Cào Về (Raw Text)</h4>
                <div style='max-height:550px; overflow-y:auto; background:#f8fafc !important; color:#0f172a !important; border:1px solid #cbd5e1; border-radius:6px; padding:14px; font-size:13.5px; line-height:1.7; white-space:pre-wrap; word-wrap:break-word; word-break:break-word;'>
                    {safe_raw}
                </div>
            </div>
            <div>
                <h4 style='color:#15803d !important; margin-top:0; border-bottom:2px solid #15803d; padding-bottom:4px;'>🧩 Các Chunks Được Chia Ra ({doc['total_chunks']})</h4>
                <div style='max-height:550px; overflow-y:auto; padding-right:4px;'>
                    {chunks_html if chunks_html else "<i style='color:#475569;'>Tài liệu này chưa có chunk trong database.</i>"}
                </div>
            </div>
        </div>
    </div>
    """
    display(HTML(html))


_CACHED_BM25_INDEX = None
_BM25_LOCK = threading.Lock()


def get_cached_bm25_index(
    indices_dir: str | Path = "data/indices",
    chunks_meta_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
):
    """Singleton cached loader for BM25s sparse index."""
    global _CACHED_BM25_INDEX
    if _CACHED_BM25_INDEX is None:
        with _BM25_LOCK:
            if _CACHED_BM25_INDEX is None:
                from src.retrieval.sparse_index import SparseIndex

                _CACHED_BM25_INDEX = SparseIndex.load(
                    directory=indices_dir,
                    fallback_meta_dir=indices_dir,
                    chunks_meta_path=chunks_meta_path,
                )
    return _CACHED_BM25_INDEX


def search_bm25_interactive(
    query: str,
    top_k: int = 5,
    indices_dir: str | Path = "data/indices",
    chunks_meta_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
) -> list[dict[str, Any]]:
    """Performs fast BM25 search and renders interactive HTML results in Jupyter.

    Args:
        query: Search keywords or clinical sentence.
        top_k: Number of candidate chunks to return.

    Returns:
        List of candidate dictionaries.
    """
    import time

    t0 = time.time()
    bm25_idx = get_cached_bm25_index(indices_dir=indices_dir, chunks_meta_path=chunks_meta_path)
    raw_results = bm25_idx.search(query=query, top_k=top_k)
    elapsed_ms = (time.time() - t0) * 1000

    results_data = []
    cards_html = ""

    for rank, (cid, score, meta) in enumerate(raw_results, 1):
        did = meta.get("doc_id", "N/A")
        title = meta.get("title", f"Document #{did}")
        lang = meta.get("lang", "vi").upper()
        text = meta.get("chunk_text", "").strip()
        url = meta.get("url", "")

        results_data.append({
            "rank": rank,
            "chunk_id": cid,
            "doc_id": did,
            "score": score,
            "title": title,
            "url": url,
            "lang": lang,
            "chunk_text": text,
        })

        safe_text = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

        url_link = (
            f"<div style='font-size:12px; margin-top:2px; margin-bottom:6px;'>🔗 URL: <a href='{url}' target='_blank' style='color:#0d6efd; text-decoration:none;'>{url} ↗</a></div>"
            if url
            else ""
        )

        cards_html += f"""
        <div style='background:#f8fafc !important; color:#0f172a !important; border:1px solid #cbd5e1; border-left:5px solid #2563eb; border-radius:6px; padding:14px; margin-bottom:14px; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                <div>
                    <span style='background:#2563eb; color:white; font-size:11px; font-weight:bold; padding:2px 8px; border-radius:4px;'>#{rank}</span>
                    <span style='font-size:12px; font-weight:bold; color:#0f172a; margin-left:8px;'>Doc ID: <b>{did}</b></span>
                    <span style='background:#0d6efd; color:white; font-size:11px; padding:1px 6px; border-radius:3px; margin-left:6px;'>{lang}</span>
                </div>
                <div>
                    <span style='background:#fef3c7; color:#92400e; font-size:12px; font-weight:bold; padding:2px 8px; border-radius:4px; border:1px solid #fde68a;'>BM25 Score: {score:.2f}</span>
                    <span style='font-size:11px; color:#64748b; margin-left:6px;'>Chunk ID: {cid}</span>
                </div>
            </div>
            <h4 style='margin:6px 0; color:#1e293b; font-size:14px; font-weight:600;'>{title}</h4>
            {url_link}
            <div style='font-size:13.5px; color:#334155; line-height:1.6; white-space:pre-wrap; word-wrap:break-word; word-break:break-word;'>
                {safe_text}
            </div>
            <div style='margin-top:8px; font-size:11px; color:#64748b;'>
                👉 Để xem toàn văn tài liệu này, chạy: <code style='background:#e2e8f0; padding:2px 6px; border-radius:3px; color:#0f172a;'>display_document({did})</code>
            </div>
        </div>
        """

    header_html = f"""
    <div style='border:1px solid #cbd5e1; border-radius:8px; padding:16px; margin:16px 0; background:#ffffff !important; color:#0f172a !important; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        <div style='border-bottom:1px solid #e2e8f0; padding-bottom:10px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <h3 style='margin:0; color:#1d4ed8;'>🔎 Kết Quả Tìm Kiếm BM25: <span style='color:#0f172a;'>"{query}"</span></h3>
                <div style='font-size:12px; color:#64748b; margin-top:4px;'>Tìm thấy <b>{len(results_data)}</b> chunks phù hợp trong <b>{elapsed_ms:.1f} ms</b> (qua memory-mapped BM25s)</div>
            </div>
        </div>
        {cards_html if cards_html else "<div style='color:#64748b; font-style:italic;'>Không tìm thấy kết quả phù hợp nào cho từ khóa này.</div>"}
    </div>
    """
    display(HTML(header_html))
    return results_data


_CACHED_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()


def get_cached_pipeline(
    indices_dir: str | Path = "data/indices",
    chunks_meta_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
):
    """Singleton cached loader for the MedicalRetrievalPipeline."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        with _PIPELINE_LOCK:
            if _CACHED_PIPELINE is None:
                from src.config import load_config
                from src.pipeline import MedicalRetrievalPipeline

                config = load_config("configs/config.yaml")
                config.pubmed.source_api = "none"
                pipe = MedicalRetrievalPipeline(config=config)
                pipe.load_indices(
                    indices_dir=indices_dir,
                    dense_dir=indices_dir,
                    sparse_dir=indices_dir,
                    chunks_meta_path=chunks_meta_path,
                )
                _CACHED_PIPELINE = pipe
    return _CACHED_PIPELINE


def search_multistage(
    query: str,
    dense_top_k: int = 8,
    sparse_top_k: int = 8,
    rrf_top_k: int = 12,
    top_k_docs: int = 4,
    indices_dir: str | Path = "data/indices",
    chunks_meta_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
) -> dict[str, Any]:
    """Executes search and records step-by-step intermediate results at each stage.

    Stages:
        1A. Dense FAISS SQ8 Retrieval (Semantic Vector)
        1B. BM25s Sparse Retrieval (Exact Lexical)
        2.  Reciprocal Rank Fusion (RRF)
        3.  Cross-Encoder Reranking (BGE-Reranker-v2-m3)
        4.  Final Output Extraction (Top-1 Strategy)
    """
    import time

    pipeline = get_cached_pipeline(indices_dir=indices_dir, chunks_meta_path=chunks_meta_path)
    t_start = time.time()

    # --- STAGE 1A: Dense Retrieval ---
    t0_dense = time.time()
    q_emb = pipeline.embedder.encode([query])[0]
    raw_dense = pipeline.dense_index.search(q_emb, top_k=dense_top_k)
    dense_elapsed = (time.time() - t0_dense) * 1000

    dense_candidates = []
    dense_rank_map = {}
    for rank, (cid, score, meta) in enumerate(raw_dense, 1):
        dense_rank_map[cid] = rank
        dense_candidates.append({
            "rank": rank,
            "chunk_id": cid,
            "doc_id": meta.get("doc_id", "N/A"),
            "score": float(score),
            "title": meta.get("title", f"Document #{meta.get('doc_id')}"),
            "url": meta.get("url", ""),
            "lang": meta.get("lang", "vi").upper(),
            "chunk_text": meta.get("chunk_text", "").strip(),
            "source": "DENSE",
        })

    # --- STAGE 1B: BM25 Sparse Retrieval ---
    t0_sparse = time.time()
    raw_sparse = pipeline.sparse_index.search(query, top_k=sparse_top_k)
    sparse_elapsed = (time.time() - t0_sparse) * 1000

    sparse_candidates = []
    sparse_rank_map = {}
    for rank, (cid, score, meta) in enumerate(raw_sparse, 1):
        sparse_rank_map[cid] = rank
        sparse_candidates.append({
            "rank": rank,
            "chunk_id": cid,
            "doc_id": meta.get("doc_id", "N/A"),
            "score": float(score),
            "title": meta.get("title", f"Document #{meta.get('doc_id')}"),
            "url": meta.get("url", ""),
            "lang": meta.get("lang", "vi").upper(),
            "chunk_text": meta.get("chunk_text", "").strip(),
            "source": "BM25",
        })

    # --- STAGE 2: RRF Fusion ---
    t0_rrf = time.time()
    all_cids = set(dense_rank_map.keys()).union(set(sparse_rank_map.keys()))
    candidate_meta = {}
    for c in dense_candidates:
        candidate_meta[c["chunk_id"]] = c
    for c in sparse_candidates:
        if c["chunk_id"] not in candidate_meta:
            candidate_meta[c["chunk_id"]] = c

    fused_pool = []
    for cid in all_cids:
        d_rank = dense_rank_map.get(cid)
        s_rank = sparse_rank_map.get(cid)
        score_rrf = 0.0
        if d_rank is not None:
            score_rrf += 0.6 / (60.0 + d_rank)
        if s_rank is not None:
            score_rrf += 0.4 / (60.0 + s_rank)

        src_label = "CẢ HAI (BOTH)" if (d_rank and s_rank) else ("CHỈ DENSE" if d_rank else "CHỈ BM25")
        meta = candidate_meta[cid]
        fused_pool.append({
            "chunk_id": cid,
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "url": meta.get("url", ""),
            "lang": meta["lang"],
            "chunk_text": meta["chunk_text"],
            "rrf_score": score_rrf,
            "dense_rank": d_rank,
            "sparse_rank": s_rank,
            "source": src_label,
        })

    fused_pool.sort(key=lambda x: x["rrf_score"], reverse=True)
    rrf_top = fused_pool[:rrf_top_k]
    for r, item in enumerate(rrf_top, 1):
        item["rank"] = r
    rrf_elapsed = (time.time() - t0_rrf) * 1000

    # --- STAGE 3: Cross-Encoder Reranking ---
    t0_rerank = time.time()
    pairs = [(query, item["chunk_text"]) for item in rrf_top]
    rerank_scores = pipeline.reranker.compute_scores(pairs)

    reranked_pool = []
    for item, r_score in zip(rrf_top, rerank_scores):
        entry = dict(item)
        entry["rerank_score"] = float(r_score)
        entry["rrf_rank"] = item["rank"]
        reranked_pool.append(entry)

    reranked_pool.sort(key=lambda x: x["rerank_score"], reverse=True)
    for r, item in enumerate(reranked_pool, 1):
        item["rank"] = r
        item["rank_delta"] = item["rrf_rank"] - r  # Positive means moved up, negative means moved down
    rerank_elapsed = (time.time() - t0_rerank) * 1000

    # --- STAGE 4: Final Extraction (Top-1 Strategy) ---
    seen_docs = set()
    final_docs = []
    final_chunks = []
    for item in reranked_pool:
        did = str(item["doc_id"])
        if did not in seen_docs:
            seen_docs.add(did)
            final_docs.append(did)
            final_chunks.append({
                "doc_id": did,
                "chunk_id": item["chunk_id"],
                "chunk_text": item["chunk_text"],
                "title": item["title"],
                "url": item.get("url", ""),
                "rerank_score": item["rerank_score"],
            })
        if len(final_docs) >= top_k_docs:
            break

    total_elapsed = (time.time() - t_start) * 1000

    return {
        "query": query,
        "timing_ms": {
            "dense": dense_elapsed,
            "sparse": sparse_elapsed,
            "rrf": rrf_elapsed,
            "rerank": rerank_elapsed,
            "total": total_elapsed,
        },
        "stage_1a_dense": dense_candidates,
        "stage_1b_bm25": sparse_candidates,
        "stage_2_rrf": rrf_top,
        "stage_3_rerank": reranked_pool,
        "stage_4_final": {
            "relevant_docs": final_docs,
            "relevant_chunks": final_chunks,
        },
    }


def display_multistage_search(
    query: str,
    dense_top_k: int = 5,
    sparse_top_k: int = 5,
    rrf_top_k: int = 8,
    top_k_docs: int = 4,
    indices_dir: str | Path = "data/indices",
    chunks_meta_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
) -> None:
    """Renders a comprehensive visual breakdown of each stage in Jupyter."""
    data = search_multistage(
        query=query,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        rrf_top_k=rrf_top_k,
        top_k_docs=top_k_docs,
        indices_dir=indices_dir,
        chunks_meta_path=chunks_meta_path,
    )

    t = data["timing_ms"]

    def make_card(rank, did, lang, score_badge, title, text, url="", extra_info=""):
        safe_text = (
            text[:200].replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", " ")
        ) + ("..." if len(text) > 200 else "")
        url_link = (
            f"<div style='font-size:11px; margin-top:2px; margin-bottom:4px;'>🔗 <a href='{url}' target='_blank' style='color:#0d6efd; text-decoration:none;'>{url[:75]}... ↗</a></div>"
            if url
            else ""
        )
        return f"""
        <div style='background:#f8fafc !important; color:#0f172a !important; border:1px solid #cbd5e1; border-radius:6px; padding:10px; margin-bottom:8px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
                <div>
                    <span style='background:#0f172a; color:white; font-size:11px; font-weight:bold; padding:1px 6px; border-radius:3px;'>#{rank}</span>
                    <span style='font-size:12px; font-weight:bold; margin-left:6px;'>Doc {did}</span>
                    <span style='background:#e2e8f0; color:#0f172a; font-size:10px; padding:1px 5px; border-radius:3px; margin-left:4px;'>{lang}</span>
                </div>
                <div>{score_badge}</div>
            </div>
            <div style='font-size:12px; font-weight:600; color:#1e293b; margin-bottom:2px;'>{title}</div>
            {url_link}
            <div style='font-size:12px; color:#475569; line-height:1.5;'>{safe_text}</div>
            {f"<div style='font-size:11px; color:#64748b; margin-top:4px;'>{extra_info}</div>" if extra_info else ""}
        </div>
        """

    # Stage 1A cards
    d_html = ""
    for item in data["stage_1a_dense"][:dense_top_k]:
        score_badge = f"<span style='background:#dbeafe; color:#1e40af; font-size:11px; font-weight:bold; padding:1px 6px; border-radius:3px;'>Cos: {item['score']:.3f}</span>"
        d_html += make_card(item["rank"], item["doc_id"], item["lang"], score_badge, item["title"], item["chunk_text"], url=item.get("url", ""))

    # Stage 1B cards
    s_html = ""
    for item in data["stage_1b_bm25"][:sparse_top_k]:
        score_badge = f"<span style='background:#fef3c7; color:#92400e; font-size:11px; font-weight:bold; padding:1px 6px; border-radius:3px;'>BM25: {item['score']:.2f}</span>"
        s_html += make_card(item["rank"], item["doc_id"], item["lang"], score_badge, item["title"], item["chunk_text"], url=item.get("url", ""))

    # Stage 2 cards
    r_html = ""
    for item in data["stage_2_rrf"][:rrf_top_k]:
        src_color = "#15803d" if "CẢ HAI" in item["source"] else ("#1d4ed8" if "DENSE" in item["source"] else "#b45309")
        src_badge = f"<span style='background:{src_color}; color:white; font-size:10px; font-weight:bold; padding:1px 6px; border-radius:3px;'>{item['source']}</span>"
        score_badge = f"<span style='background:#f1f5f9; color:#0f172a; font-size:11px; font-weight:bold; padding:1px 6px; border-radius:3px;'>RRF: {item['rrf_score']:.4f}</span>"
        extra = f"Vị trí ban đầu: Dense #{item['dense_rank'] or '-'} | BM25 #{item['sparse_rank'] or '-'} | {src_badge}"
        r_html += make_card(item["rank"], item["doc_id"], item["lang"], score_badge, item["title"], item["chunk_text"], url=item.get("url", ""), extra_info=extra)

    # Stage 3 cards
    ce_html = ""
    for item in data["stage_3_rerank"][:rrf_top_k]:
        delta_str = f"+{item['rank_delta']}" if item["rank_delta"] > 0 else (str(item["rank_delta"]) if item["rank_delta"] < 0 else "=")
        delta_color = "#15803d" if item["rank_delta"] > 0 else ("#dc2626" if item["rank_delta"] < 0 else "#64748b")
        delta_badge = f"<span style='color:{delta_color}; font-weight:bold;'>({delta_str})</span>"
        score_badge = f"<span style='background:#dcfce7; color:#15803d; font-size:11px; font-weight:bold; padding:1px 6px; border-radius:3px;'>Logits: {item['rerank_score']:.2f}</span>"
        extra = f"RRF ban đầu: #{item['rrf_rank']} {delta_badge}"
        ce_html += make_card(item["rank"], item["doc_id"], item["lang"], score_badge, item["title"], item["chunk_text"], url=item.get("url", ""), extra_info=extra)

    # Stage 4 summary
    final = data["stage_4_final"]
    docs_badges = " ".join([f"<span style='background:#0f172a; color:white; padding:3px 10px; border-radius:4px; font-weight:bold; font-size:13px;'>Doc {did}</span>" for did in final["relevant_docs"]])

    main_html = f"""
    <div style='border:1px solid #cbd5e1; border-radius:8px; padding:18px; margin:16px 0; background:#ffffff !important; color:#0f172a !important; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        <div style='border-bottom:2px solid #e2e8f0; padding-bottom:12px; margin-bottom:14px;'>
            <div style='font-size:13px; color:#475569;'>🎯 PIPELINE TRUY HỒI ĐA TẦNG (MULTI-STAGE RETRIEVAL)</div>
            <h3 style='margin:6px 0; color:#1d4ed8;'>Query: <span style='color:#0f172a;'>"{query}"</span></h3>
            <div style='font-size:12px; color:#64748b; margin-top:4px;'>
                ⏱️ Tổng thời gian: <b>{t['total']:.1f} ms</b> (Dense: {t['dense']:.1f}ms | BM25: {t['sparse']:.1f}ms | RRF: {t['rrf']:.1f}ms | Rerank: {t['rerank']:.1f}ms)
            </div>
        </div>

        <!-- STAGE 1: FIRST STAGE RETRIEVAL -->
        <div style='background:#f1f5f9; border-radius:6px; padding:12px; margin-bottom:14px;'>
            <h4 style='margin:0 0 10px 0; color:#1e293b;'>1️⃣ GIAI ĐOẠN 1: TÌM KIẾM SƠ BỘ (FIRST-STAGE RETRIEVAL)</h4>
            <div style='display:grid; grid-template-columns: 1fr 1fr; gap:12px;'>
                <div>
                    <div style='font-weight:bold; color:#1e40af; font-size:13px; margin-bottom:6px;'>🔹 1A. Dense FAISS SQ8 (Ngữ nghĩa BGE-M3)</div>
                    {d_html}
                </div>
                <div>
                    <div style='font-weight:bold; color:#b45309; font-size:13px; margin-bottom:6px;'>🔸 1B. BM25s Sparse (Từ khóa chính xác)</div>
                    {s_html}
                </div>
            </div>
        </div>

        <!-- STAGE 2: RRF FUSION -->
        <div style='background:#f1f5f9; border-radius:6px; padding:12px; margin-bottom:14px;'>
            <h4 style='margin:0 0 6px 0; color:#1e293b;'>2️⃣ GIAI ĐOẠN 2: HỢP NHẤT HẠNG RRF (RECIPROCAL RANK FUSION)</h4>
            <div style='font-size:12px; color:#64748b; margin-bottom:10px;'>Công thức: RRF = 0.6 / (60 + Dense_Rank) + 0.4 / (60 + BM25_Rank). Gộp cả 2 nguồn để tăng Recall tối đa.</div>
            {r_html}
        </div>

        <!-- STAGE 3: CROSS-ENCODER RERANKING -->
        <div style='background:#f1f5f9; border-radius:6px; padding:12px; margin-bottom:14px;'>
            <h4 style='margin:0 0 6px 0; color:#15803d;'>3️⃣ GIAI ĐOẠN 3: TÁI CHẤM ĐIỂM CHÉO (CROSS-ENCODER BGE-RERANKER-V2-M3)</h4>
            <div style='font-size:12px; color:#64748b; margin-bottom:10px;'>Mô hình Cross-Attention đọc toàn bộ câu hỏi và ứng viên để lọc sạch kết quả nhiễu và đẩy bằng chứng tốt nhất lên đầu.</div>
            {ce_html}
        </div>

        <!-- STAGE 4: FINAL SELECTION -->
        <div style='background:#ecfdf5; border:1px solid #a7f3d0; border-radius:6px; padding:14px;'>
            <h4 style='margin:0 0 6px 0; color:#065f46;'>4️⃣ GIAI ĐOẠN 4: TRÍCH XUẤT CUỐI CÙNG (CHIẾN LƯỢC TOP 1 F2-MACRO)</h4>
            <div style='font-size:12px; color:#047857; margin-bottom:10px;'>Chọn ra {len(final['relevant_docs'])} tài liệu xuất sắc nhất và 1 chunk bằng chứng tốt nhất cho mỗi tài liệu:</div>
            <div style='margin-bottom:10px;'>{docs_badges}</div>
            <div style='font-size:12px; color:#065f46;'>Số lượng chunks trích xuất: <b>{len(final['relevant_chunks'])}</b> chunks.</div>
        </div>
    </div>
    """
    display(HTML(main_html))


_CACHED_CATALOG = None


def get_domains_catalog(parquet_dir: str | Path = "data/processed/parquet_corpus"):
    """Scans corpus and returns a aggregated DataFrame with domains, language, counts, and length."""
    global _CACHED_CATALOG
    if _CACHED_CATALOG is not None:
        return _CACHED_CATALOG

    import pandas as pd
    import pyarrow.dataset as ds

    p_dir = Path(parquet_dir)
    if not p_dir.exists():
        return pd.DataFrame()

    dataset = ds.dataset(p_dir, format="parquet")
    scanner = dataset.scanner(columns=["domain", "lang", "char_count", "title"])
    table = scanner.to_table()
    df = table.to_pandas()

    cat = (
        df.groupby(["domain", "lang"])
        .agg(
            article_count=("domain", "count"),
            avg_char_length=("char_count", "mean"),
            sample_title=("title", "first"),
        )
        .reset_index()
        .sort_values(by="article_count", ascending=False)
    )
    _CACHED_CATALOG = cat
    return cat


def get_documents_by_domain(
    domain: str,
    limit: int = 20,
    offset: int = 0,
    keyword_filter: str = "",
    parquet_dir: str | Path = "data/processed/parquet_corpus",
) -> list[dict[str, Any]]:
    """Fetches full original raw documents belonging to a specific domain."""
    import pyarrow.dataset as ds

    p_dir = Path(parquet_dir)
    if not p_dir.exists():
        return []

    dataset = ds.dataset(p_dir, format="parquet")
    filter_expr = ds.field("domain") == domain
    scanner = dataset.scanner(
        columns=["doc_id", "title", "text", "url", "lang", "domain", "char_count"],
        filter=filter_expr,
    )

    batch_docs = []
    # Read batches
    count = 0
    matched = 0
    kw = keyword_filter.strip().lower()

    for batch in scanner.to_batches():
        rows = batch.to_pylist()
        for r in rows:
            if kw and (kw not in r.get("title", "").lower() and kw not in r.get("text", "").lower()):
                continue
            if count >= offset:
                batch_docs.append(r)
                matched += 1
                if matched >= limit:
                    return batch_docs
            count += 1

    return batch_docs


def get_random_doc_by_domain(
    domain: str,
    parquet_dir: str | Path = "data/processed/parquet_corpus",
) -> dict[str, Any] | None:
    """Picks a random raw document from the specified domain."""
    import random
    import pyarrow.dataset as ds

    p_dir = Path(parquet_dir)
    if not p_dir.exists():
        return None

    dataset = ds.dataset(p_dir, format="parquet")
    filter_expr = ds.field("domain") == domain
    scanner = dataset.scanner(
        columns=["doc_id", "title", "text", "url", "lang", "domain", "char_count"],
        filter=filter_expr,
    )

    # Fast reservoir sampling (sample 1)
    chosen = None
    seen = 0
    for batch in scanner.to_batches():
        rows = batch.to_pylist()
        for r in rows:
            seen += 1
            if random.random() < (1.0 / seen):
                chosen = r
    return chosen


def display_raw_document(
    doc: dict[str, Any],
    max_height: int = 650,
) -> None:
    """Renders full raw original document in clean HTML with dark/light mode compatibility."""
    if not doc:
        display(HTML("<b style='color:#dc2626;'>Không tìm thấy văn bản!</b>"))
        return

    import html as html_lib

    title = html_lib.escape(doc.get("title") or f"Tài liệu #{doc.get('doc_id')}")
    raw_text = html_lib.escape(doc.get("text") or doc.get("raw_text") or "")
    url = doc.get("url") or ""
    domain = doc.get("domain") or "N/A"
    lang = (doc.get("lang") or "vi").upper()
    doc_id = str(doc.get("doc_id", "N/A"))
    char_count = len(raw_text)
    word_count = len(raw_text.split())

    url_html = (
        f"<a href='{url}' target='_blank' style='color:#2563eb; text-decoration:none;'>{url[:80]}... ↗</a>"
        if url
        else "<span style='color:#64748b;'>Không có URL</span>"
    )

    html_out = f"""
    <div style='border:1px solid #cbd5e1; border-radius:8px; padding:18px; margin:14px 0; background:#ffffff !important; color:#0f172a !important; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
        <!-- Header -->
        <div style='border-bottom:2px solid #e2e8f0; padding-bottom:12px; margin-bottom:14px;'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                <div>
                    <span style='background:#2563eb; color:white; font-size:11px; font-weight:bold; padding:2px 8px; border-radius:4px;'>Domain: {domain}</span>
                    <span style='background:#059669; color:white; font-size:11px; font-weight:bold; padding:2px 8px; border-radius:4px; margin-left:6px;'>{lang}</span>
                    <span style='background:#f1f5f9; color:#0f172a; font-size:11px; font-weight:bold; padding:2px 8px; border-radius:4px; margin-left:6px;'>Doc ID: {doc_id}</span>
                </div>
                <div style='font-size:12px; color:#64748b;'>
                    <b>{char_count:,}</b> ký tự | <b>{word_count:,}</b> từ
                </div>
            </div>
            <h3 style='margin:6px 0; color:#0f172a !important;'>{title}</h3>
            <div style='font-size:12px;'>🔗 Link nguồn: {url_html}</div>
        </div>

        <!-- Document Body -->
        <div>
            <div style='font-size:12px; font-weight:bold; color:#475569; margin-bottom:6px;'>📜 TOÀN VĂN NGUYÊN BẢN (CHƯA PHÂN ĐOẠN CHUNK):</div>
            <div style='background:#f8fafc !important; color:#0f172a !important; border:1px solid #cbd5e1; border-radius:6px; padding:16px; font-size:14px; line-height:1.75; white-space:pre-wrap !important; word-break:break-word !important; word-wrap:break-word !important; max-height:{max_height}px; overflow-y:auto;'>
{raw_text}
            </div>
        </div>
    </div>
    """
    display(HTML(html_out))


def lookup_raw_doc_by_id(
    doc_id: int | str,
    parquet_dir: str | Path = "data/processed/parquet_corpus",
    sqlite_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
) -> dict[str, Any] | None:
    """Looks up full raw document by doc_id from parquet corpus, with fallback to SQLite metadata."""
    import pyarrow.dataset as ds

    try:
        target_id = int(str(doc_id).strip())
    except ValueError:
        return None

    # 1. Look up in Parquet Corpus
    p_dir = Path(parquet_dir)
    if p_dir.exists():
        dataset = ds.dataset(p_dir, format="parquet")
        scanner = dataset.scanner(
            columns=["doc_id", "title", "text", "url", "lang", "domain", "char_count"],
            filter=(ds.field("doc_id") == target_id),
        )
        rows = scanner.to_table().to_pylist()
        if rows:
            return rows[0]

    # 2. Fallback to SQLite Metadata
    s_path = Path(sqlite_path)
    if s_path.exists():
        info = get_document_and_chunks(target_id, db_path=s_path)
        if info:
            return {
                "doc_id": target_id,
                "title": info.get("title", f"Document #{target_id}"),
                "text": info.get("full_text", ""),
                "url": info.get("url", ""),
                "lang": info.get("lang", "vi"),
                "domain": "Internal SQLite DB",
                "char_count": len(info.get("full_text", "")),
            }

    return None





