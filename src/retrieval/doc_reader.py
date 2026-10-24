"""Document and Chunk Reader from SQLite corpus and PubMed cache."""

from __future__ import annotations

import json
import sqlite3
import textwrap
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
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        rows = []
        try:
            cur.execute(
                "SELECT chunk_id, title, chunk_text, lang FROM chunks WHERE doc_id = ? ORDER BY chunk_id",
                (int(doc_id),),
            )
            rows = cur.fetchall()
        except Exception:
            pass

        if not rows:
            cur.execute(
                "SELECT chunk_id, title, chunk_text, lang FROM chunks WHERE doc_id = ? ORDER BY chunk_id",
                (str(doc_id),),
            )
            rows = cur.fetchall()
        conn.close()

        if rows:
            title = rows[0][1] or f"Document #{doc_id}"
            lang = rows[0][3] or "vi"
            chunks = [{"chunk_id": r[0], "chunk_text": r[2]} for r in rows]
            full_text = "\n\n".join([c["chunk_text"] for c in chunks])
            return {
                "doc_id": doc_id,
                "title": title,
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
                    return {
                        "doc_id": doc_id,
                        "title": item.get("title", f"PubMed Article #{doc_id}"),
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

    html = f"""
    <div style='border:1px solid #cbd5e1; border-radius:8px; padding:18px; margin:16px 0; background:#ffffff !important; color:#0f172a !important; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; box-shadow:0 2px 8px rgba(0,0,0,0.08);'>
        <div style='border-bottom:1px solid #e2e8f0; padding-bottom:12px; margin-bottom:12px;'>
            <div style='font-size:13px; color:#475569 !important;'>📄 Document ID: <b style='color:#0f172a;'>{doc['doc_id']}</b> {lang_badge} {chunks_badge}</div>
            <h3 style='margin:8px 0 4px 0; color:#0f172a !important; line-height:1.4;'>{doc['title']}</h3>
            <div style='font-size:12px; color:#64748b !important;'>Độ dài toàn văn: {len(doc['full_text']):,} ký tự | Nguồn: {doc['source']}</div>
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
            conn = sqlite3.connect(db_path)
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
        conn = sqlite3.connect(db_path)
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


def get_cached_bm25_index(
    indices_dir: str | Path = "data/indices",
    chunks_meta_path: str | Path = "data/indices/chunks_meta_slim.sqlite",
):
    """Singleton cached loader for BM25s sparse index."""
    global _CACHED_BM25_INDEX
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

        results_data.append({
            "rank": rank,
            "chunk_id": cid,
            "doc_id": did,
            "score": score,
            "title": title,
            "lang": lang,
            "chunk_text": text,
        })

        safe_text = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
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


