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
            <div style='background:#f8f9fa; border:1px solid #dee2e6; border-left:4px solid #0d6efd; border-radius:6px; padding:12px; margin-bottom:12px; white-space:pre-wrap; word-wrap:break-word; word-break:break-word; line-height:1.6;'>
                <div style='font-size:12px; font-weight:bold; color:#6c757d; margin-bottom:6px;'>Chunk #{i}/{doc['total_chunks']} (Chunk ID: {c['chunk_id']})</div>
                <div style='font-size:14px; color:#212529;'>{safe_text}</div>
            </div>
            """
        chunks_html += "</div>"

    html = f"""
    <div style='border:1px solid #ced4da; border-radius:8px; padding:18px; margin:16px 0; background:white; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
        <div style='border-bottom:1px solid #e9ecef; padding-bottom:12px; margin-bottom:12px;'>
            <div style='font-size:13px; color:#6c757d;'>📄 Document ID: <b>{doc['doc_id']}</b> {lang_badge} {chunks_badge}</div>
            <h3 style='margin:8px 0 4px 0; color:#1a1e21; line-height:1.4;'>{doc['title']}</h3>
            <div style='font-size:12px; color:#adb5bd;'>Độ dài toàn văn: {len(doc['full_text']):,} ký tự | Nguồn: {doc['source']}</div>
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
