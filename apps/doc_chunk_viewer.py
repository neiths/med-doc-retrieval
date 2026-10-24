"""Streamlit Web UI for inspecting crawled medical documents and visualizing chunk splits.

Run locally:
    streamlit run apps/doc_chunk_viewer.py
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.ingestion.chunker import DocumentChunker
from src.retrieval.doc_reader import get_document_and_chunks, get_random_crawled_doc

# Configure Streamlit page
st.set_page_config(
    page_title="ViBioMIR Medical Document & Chunk Inspector",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for word wrapping and clean typography
st.markdown(
    """
<style>
    .reportview-container, .main, div.stMarkdown, div.stText {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
    }
    .chunk-card {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-left: 5px solid #10b981 !important;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .chunk-header {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        font-weight: 600;
        color: #047857 !important;
        margin-bottom: 8px;
    }
    .chunk-body {
        font-size: 14px;
        color: #0f172a !important;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    .raw-box {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px;
        padding: 18px;
        line-height: 1.7;
        font-size: 14px;
        max-height: 700px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
        line-height: 1.7;
        font-size: 14px;
        max-height: 700px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 12px;
        font-weight: 700;
        border-radius: 4px;
        color: white;
    }
    .badge-blue { background-color: #2563eb; }
    .badge-green { background-color: #059669; }
    .badge-amber { background-color: #d97706; }
</style>
""",
    unsafe_allow_html=True,
)


def load_crawled_doc_by_id(doc_id: str | int) -> dict | None:
    """Finds document in crawled shards or sqlite metadata."""
    # 1. Search raw shards
    raw_shards = list(Path("data/raw").glob("crawled_shard_*.jsonl"))
    str_id = str(doc_id).strip()

    for shard in raw_shards:
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                if f'"doc_id": "{str_id}"' in line or f'"doc_id": {str_id}' in line:
                    try:
                        record = json.loads(line)
                        if str(record.get("doc_id")) == str_id:
                            # Also retrieve indexed chunks
                            chunks_info = get_document_and_chunks(str_id)
                            record["chunks"] = chunks_info["chunks"] if chunks_info else []
                            record["total_chunks"] = len(record["chunks"])
                            record["raw_text"] = record.get("text", "")
                            record["char_count"] = len(record["raw_text"])
                            record["word_count"] = len(record["raw_text"].split())
                            return record
                    except Exception:
                        pass

    # 2. Fallback to SQLite
    chunks_info = get_document_and_chunks(str_id)
    if chunks_info:
        chunks_info["raw_text"] = chunks_info["full_text"]
        chunks_info["url"] = ""
        chunks_info["char_count"] = len(chunks_info["raw_text"])
        chunks_info["word_count"] = len(chunks_info["raw_text"].split())
        return chunks_info

    return None


# Initialize session state for current document
if "current_doc" not in st.session_state:
    st.session_state.current_doc = get_random_crawled_doc()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🏥 Điều Khiển Dữ Liệu")

st.sidebar.subheader("🎲 Lấy Văn Bản Ngẫu Nhiên")
lang_choice = st.sidebar.selectbox(
    "Bộ lọc ngôn ngữ:",
    options=["Tất cả", "Tiếng Việt (vi)", "Tiếng Trung (zh)"],
    index=0,
)
lang_code = None
if lang_choice == "Tiếng Việt (vi)":
    lang_code = "vi"
elif lang_choice == "Tiếng Trung (zh)":
    lang_code = "zh"

if st.sidebar.button("🎲 Lấy Ngẫu Nhiên Tài Liệu Mới", use_container_width=True, type="primary"):
    with st.spinner("Đang chọn tài liệu ngẫu nhiên..."):
        st.session_state.current_doc = get_random_crawled_doc(target_lang=lang_code)

st.sidebar.divider()

st.sidebar.subheader("🔍 Tra Cứu Theo Doc ID Cụ Thể")
manual_id = st.sidebar.text_input("Nhập Doc ID (ví dụ: 545347, 733125, 682549):", "")
if st.sidebar.button("Tìm Kiếm Doc ID", use_container_width=True):
    if manual_id.strip():
        with st.spinner(f"Đang tìm Doc ID {manual_id}..."):
            doc = load_crawled_doc_by_id(manual_id.strip())
            if doc:
                st.session_state.current_doc = doc
                st.sidebar.success(f"Đã tìm thấy Doc ID {manual_id}!")
            else:
                st.sidebar.error(f"Không tìm thấy Doc ID {manual_id} trong kho dữ liệu.")

st.sidebar.divider()

st.sidebar.subheader("⚙️ Trình Thử Nghiệm Phân Đoạn (Chunker Simulator)")
sim_max_size = st.sidebar.slider("Max Chunk Size (ký tự):", min_value=150, max_value=1500, value=512, step=32)
sim_overlap = st.sidebar.slider("Chunk Overlap (ký tự):", min_value=0, max_value=300, value=64, step=16)
sim_split_sent = st.sidebar.checkbox("Cắt theo ranh giới câu trọn vẹn", value=True)

# ----------------- MAIN VIEW -----------------
doc = st.session_state.current_doc

if not doc:
    st.error("Chưa có tài liệu nào được nạp. Hãy bấm nút 'Lấy Ngẫu Nhiên Tài Liệu Mới' ở thanh bên!")
    st.stop()

# Header Information
title = doc.get("title") or f"Tài liệu #{doc.get('doc_id')}"
doc_id = doc.get("doc_id")
lang = (doc.get("lang") or "vi").upper()
url = doc.get("url") or ""
raw_text = doc.get("raw_text") or doc.get("text") or doc.get("full_text") or ""
indexed_chunks = doc.get("chunks", [])

st.title(f"📄 {title}")

# Top Info Badges
col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1.2, 2])
with col1:
    st.metric(label="Document ID", value=str(doc_id))
with col2:
    st.metric(label="Ngôn ngữ", value=lang)
with col3:
    st.metric(label="Chunks Đã Index", value=f"{len(indexed_chunks)}")
with col4:
    st.metric(label="Tổng Độ Dài", value=f"{len(raw_text):,} ký tự")
with col5:
    if url:
        st.markdown(f"**🔗 Link Nguồn Cào Về:**\n[{url[:65]}...]({url})")
    else:
        st.markdown("**📁 Nguồn:** Cơ sở dữ liệu nội bộ")

st.divider()

# Tabs for Views
tab_compare, tab_chunks, tab_raw, tab_sim = st.tabs([
    "📊 So Sánh Song Song (Văn Bản Gốc vs Chunks)",
    "🧩 Danh Sách Các Chunks Đã Index",
    "📜 Toàn Văn Tài Liệu Gốc",
    "🔬 Thử Nghiệm Chunking Động",
])

# 1. TAB COMPARE (SIDE-BY-SIDE)
with tab_compare:
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("📜 Văn Bản Gốc Đã Cào Về (Raw Text)")
        st.markdown(
            f"<div class='raw-box'>{raw_text}</div>",
            unsafe_allow_html=True,
        )

    with c_right:
        st.subheader(f"🧩 Các Chunks Được Chia Ra ({len(indexed_chunks)} chunks)")
        if not indexed_chunks:
            st.info("Tài liệu này chưa có chunk được index trong SQLite.")
        else:
            for idx, c in enumerate(indexed_chunks, 1):
                c_text = c.get("chunk_text", "")
                c_id = c.get("chunk_id", idx)
                c_len = len(c_text)
                c_words = len(c_text.split())

                st.markdown(
                    f"""
                    <div class='chunk-card'>
                        <div class='chunk-header'>
                            <span>🧩 Chunk #{idx}/{len(indexed_chunks)} (ID: {c_id})</span>
                            <span>{c_len} ký tự | {c_words} từ</span>
                        </div>
                        <div class='chunk-body'>{c_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# 2. TAB CHUNKS ONLY
with tab_chunks:
    st.subheader(f"🧩 Chi Tiết Toàn Bộ {len(indexed_chunks)} Chunks Của Document #{doc_id}")
    for idx, c in enumerate(indexed_chunks, 1):
        c_text = c.get("chunk_text", "")
        c_id = c.get("chunk_id", idx)
        with st.expander(f"Chunk #{idx}/{len(indexed_chunks)} — ID: {c_id} ({len(c_text)} ký tự, {len(c_text.split())} từ)", expanded=(idx == 1)):
            st.text_area(label=f"Nội dung Chunk #{idx}:", value=c_text, height=180, key=f"chunk_area_{idx}")

# 3. TAB RAW ONLY
with tab_raw:
    st.subheader("📜 Toàn Văn Văn Bản Gốc (Đã Format & Wrap Text)")
    st.text_area("Toàn văn tài liệu cào về:", value=raw_text, height=600)

# 4. TAB SIMULATOR
with tab_sim:
    st.subheader("🔬 Thử Nghiệm Tham Số Chunking Mới Trên Văn Bản Này")
    st.write(f"Đang áp dụng: `max_chunk_size={sim_max_size}`, `chunk_overlap={sim_overlap}`, `split_by_sentences={sim_split_sent}`")

    if st.button("🚀 Chạy Phân Đoạn Thử Nghiệm", type="primary"):
        chunker = DocumentChunker(
            max_chunk_size=sim_max_size,
            chunk_overlap=sim_overlap,
            min_chunk_size=40,
            split_by_sentences=sim_split_sent,
            enable_contextual=False,
        )
        sim_chunks = chunker.chunk_document(
            doc_id=str(doc_id),
            text=raw_text,
            lang=lang.lower(),
            title=title,
        )

        st.success(f"Đã phân đoạn thành công thành **{len(sim_chunks)} chunks** mới (Trước đó: {len(indexed_chunks)} chunks)!")

        sim_cols = st.columns(2)
        with sim_cols[0]:
            st.metric("Số Chunks Mới", len(sim_chunks), delta=len(sim_chunks) - len(indexed_chunks))
        with sim_cols[1]:
            avg_len = sum(len(c.chunk_text) for c in sim_chunks) / max(len(sim_chunks), 1)
            st.metric("Độ Dài Trung Bình (ký tự)", f"{avg_len:.1f}")

        for s_idx, sc in enumerate(sim_chunks, 1):
            st.markdown(
                f"""
                <div class='chunk-card' style='border-left-color: #3b82f6;'>
                    <div class='chunk-header' style='color: #2563eb;'>
                        <span>🔬 Chunk Thử Nghiệm #{s_idx}/{len(sim_chunks)} (Vị trí ký tự: {sc.char_start} - {sc.char_end})</span>
                        <span>{len(sc.chunk_text)} ký tự | {len(sc.chunk_text.split())} từ</span>
                    </div>
                    <div class='chunk-body'>{sc.chunk_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
