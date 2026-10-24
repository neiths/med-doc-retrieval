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
from src.retrieval.doc_reader import (
    get_document_and_chunks,
    get_random_crawled_doc,
    search_multistage,
)

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


@st.cache_resource(show_spinner="Đang nạp BM25s sparse index...")
def get_bm25_index():
    from src.retrieval.sparse_index import SparseIndex

    return SparseIndex.load(
        directory="data/indices",
        fallback_meta_dir="data/indices",
        chunks_meta_path="data/indices/chunks_meta_slim.sqlite",
    )


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
tab_multistage, tab_bm25, tab_compare, tab_chunks, tab_raw, tab_sim = st.tabs([
    "🎯 Pipeline 4 Giai Đoạn (Dense ➔ BM25 ➔ RRF ➔ Rerank)",
    "🔎 Tìm Kiếm BM25 (BM25 Search)",
    "📊 So Sánh Song Song (Văn Bản Gốc vs Chunks)",
    "🧩 Danh Sách Các Chunks Đã Index",
    "📜 Toàn Văn Tài Liệu Gốc",
    "🔬 Thử Nghiệm Chunking Động",
])

# 0. TAB MULTI-STAGE RETRIEVAL EXPLORER
with tab_multistage:
    st.subheader("🎯 Phân Tích Kết Quả Từng Giai Đoạn (Multi-Stage Retrieval Explorer)")
    st.markdown(
        """
        Khám phá dòng chảy dữ liệu thực tế từ lúc nhận câu hỏi đến khi trích xuất bằng chứng:
        `Query` ➔ `1. Dense + BM25` ➔ `2. RRF Fusion` ➔ `3. Cross-Encoder Rerank` ➔ `4. Trích Xuất Cuối Cùng`
        """
    )

    ms_query = st.text_input(
        "Nhập câu hỏi lâm sàng cần phân tích:",
        value="paracetamol hạ sốt cho trẻ em",
        key="ms_search_input",
    )

    c_s1, c_s2, c_btn = st.columns([1, 1, 1.5])
    with c_s1:
        ms_stage1_k = st.slider("Top K Giai đoạn 1 (Dense & BM25):", min_value=3, max_value=12, value=5, step=1)
    with c_s2:
        ms_rrf_k = st.slider("Top K Hợp nhất RRF:", min_value=4, max_value=20, value=8, step=1)
    with c_btn:
        st.write("")
        st.write("")
        run_multistage = st.button("🚀 Chạy Phân Tích 4 Giai Đoạn", type="primary", use_container_width=True)

    if run_multistage or ms_query.strip():
        with st.spinner("Đang chạy toàn bộ Pipeline 4 giai đoạn qua GPU/CPU..."):
            ms_data = search_multistage(
                query=ms_query.strip(),
                dense_top_k=ms_stage1_k,
                sparse_top_k=ms_stage1_k,
                rrf_top_k=ms_rrf_k,
                top_k_docs=4,
            )

        ms_t = ms_data["timing_ms"]

        # Timing KPI Cards
        st.markdown(
            f"""
            <div style='display:flex; gap:12px; margin: 12px 0 20px 0; flex-wrap:wrap;'>
                <div style='background:#f8fafc; border:1px solid #cbd5e1; border-radius:6px; padding:10px 16px; flex:1; min-width:120px;'>
                    <div style='font-size:11px; color:#64748b; font-weight:bold;'>TỔNG THỜI GIAN</div>
                    <div style='font-size:20px; font-weight:bold; color:#0f172a;'>{ms_t['total']:.0f} ms</div>
                </div>
                <div style='background:#eff6ff; border:1px solid #bfdbfe; border-radius:6px; padding:10px 16px; flex:1; min-width:120px;'>
                    <div style='font-size:11px; color:#1d4ed8; font-weight:bold;'>1A. DENSE FAISS</div>
                    <div style='font-size:20px; font-weight:bold; color:#1e40af;'>{ms_t['dense']:.1f} ms</div>
                </div>
                <div style='background:#fefce8; border:1px solid #fef08a; border-radius:6px; padding:10px 16px; flex:1; min-width:120px;'>
                    <div style='font-size:11px; color:#a16207; font-weight:bold;'>1B. BM25S</div>
                    <div style='font-size:20px; font-weight:bold; color:#854d0e;'>{ms_t['sparse']:.1f} ms</div>
                </div>
                <div style='background:#f5f3ff; border:1px solid #ddd6fe; border-radius:6px; padding:10px 16px; flex:1; min-width:120px;'>
                    <div style='font-size:11px; color:#6d28d9; font-weight:bold;'>2. RRF FUSION</div>
                    <div style='font-size:20px; font-weight:bold; color:#5b21b6;'>{ms_t['rrf']:.1f} ms</div>
                </div>
                <div style='background:#ecfdf5; border:1px solid #a7f3d0; border-radius:6px; padding:10px 16px; flex:1; min-width:120px;'>
                    <div style='font-size:11px; color:#047857; font-weight:bold;'>3. RERANKER</div>
                    <div style='font-size:20px; font-weight:bold; color:#065f46;'>{ms_t['rerank']:.1f} ms</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # STAGE 1: Side by Side
        st.markdown("### 1️⃣ Giai Đoạn 1: Tìm Kiếm Sơ Bộ (First-Stage Retrieval)")
        col_dense, col_bm25 = st.columns(2)
        with col_dense:
            st.markdown("#### 🔹 1A. Dense FAISS SQ8 (Ngữ Nghĩa BGE-M3)")
            for item in ms_data["stage_1a_dense"][:ms_stage1_k]:
                with st.container():
                    st.markdown(
                        f"""
                        <div class='chunk-card' style='border-left-color: #3b82f6;'>
                            <div class='chunk-header' style='color: #1d4ed8;'>
                                <span>#{item['rank']} | Doc ID: <b>{item['doc_id']}</b> <span class='badge badge-blue'>{item['lang']}</span></span>
                                <span>Cosine: <b>{item['score']:.3f}</b></span>
                            </div>
                            <div style='font-weight:600; color:#0f172a; margin-bottom:4px;'>{item['title']}</div>
                            <div class='chunk-body'>{item['chunk_text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        with col_bm25:
            st.markdown("#### 🔸 1B. BM25s Sparse (Từ Khóa Chính Xác)")
            for item in ms_data["stage_1b_bm25"][:ms_stage1_k]:
                with st.container():
                    st.markdown(
                        f"""
                        <div class='chunk-card' style='border-left-color: #f59e0b;'>
                            <div class='chunk-header' style='color: #b45309;'>
                                <span>#{item['rank']} | Doc ID: <b>{item['doc_id']}</b> <span class='badge badge-amber'>{item['lang']}</span></span>
                                <span>BM25 Score: <b>{item['score']:.2f}</b></span>
                            </div>
                            <div style='font-weight:600; color:#0f172a; margin-bottom:4px;'>{item['title']}</div>
                            <div class='chunk-body'>{item['chunk_text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.divider()

        # STAGE 2: RRF Fusion
        st.markdown("### 2️⃣ Giai Đoạn 2: Hợp Nhất Hạng RRF (Reciprocal Rank Fusion)")
        st.caption("Công thức: `RRF Score = 0.6 / (60 + Dense_Rank) + 0.4 / (60 + BM25_Rank)`. Giúp gộp tối đa Recall từ cả 2 nguồn.")
        for item in ms_data["stage_2_rrf"][:ms_rrf_k]:
            src_tag = item["source"]
            badge_color = "#15803d" if "CẢ HAI" in src_tag else ("#1d4ed8" if "DENSE" in src_tag else "#b45309")
            with st.container():
                st.markdown(
                    f"""
                    <div class='chunk-card' style='border-left-color: {badge_color};'>
                        <div class='chunk-header' style='color: #334155;'>
                            <span>#{item['rank']} | Doc ID: <b>{item['doc_id']}</b> <span style='background:{badge_color}; color:white; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:bold;'>{src_tag}</span></span>
                            <span>RRF Score: <b>{item['rrf_score']:.4f}</b> (Dense #{item['dense_rank'] or '-'} | BM25 #{item['sparse_rank'] or '-'})</span>
                        </div>
                        <div style='font-weight:600; color:#0f172a; margin-bottom:4px;'>{item['title']}</div>
                        <div class='chunk-body'>{item['chunk_text']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # STAGE 3: Cross-Encoder Reranking
        st.markdown("### 3️⃣ Giai Đoạn 3: Tái Chấm Điểm Chéo (BGE-Reranker-v2-m3)")
        st.caption("Cross-Attention đọc cả câu hỏi và từng chunk để lọc sạch các kết quả rác và xếp lại thứ hạng chính xác nhất.")
        for item in ms_data["stage_3_rerank"][:ms_rrf_k]:
            delta = item["rank_delta"]
            delta_str = f"+{delta} hạng" if delta > 0 else (f"{delta} hạng" if delta < 0 else "Giữ nguyên")
            delta_bg = "#dcfce7" if delta > 0 else ("#fee2e2" if delta < 0 else "#f1f5f9")
            delta_text = "#15803d" if delta > 0 else ("#b91c1c" if delta < 0 else "#64748b")
            with st.container():
                st.markdown(
                    f"""
                    <div class='chunk-card' style='border-left-color: #10b981;'>
                        <div class='chunk-header' style='color: #065f46;'>
                            <span>#{item['rank']} | Doc ID: <b>{item['doc_id']}</b> <span style='background:{delta_bg}; color:{delta_text}; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:bold;'>{delta_str}</span></span>
                            <span>Rerank Logits: <b>{item['rerank_score']:.2f}</b> (RRF ban đầu: #{item['rrf_rank']})</span>
                        </div>
                        <div style='font-weight:600; color:#0f172a; margin-bottom:4px;'>{item['title']}</div>
                        <div class='chunk-body'>{item['chunk_text']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # STAGE 4: Final Selection
        st.markdown("### 4️⃣ Giai Đoạn 4: Trích Xuất Cuối Cùng (Chiến Lược Top-1 F2-Macro)")
        final_info = ms_data["stage_4_final"]
        st.success(f"Tài liệu được chọn nộp bài ({len(final_info['relevant_docs'])} docs): **{', '.join(['Doc #' + d for d in final_info['relevant_docs']])}**")

        c_docs = st.columns(len(final_info["relevant_docs"]))
        for col, did in zip(c_docs, final_info["relevant_docs"]):
            with col:
                st.button(f"👉 Mở Đọc Doc #{did}", key=f"open_ms_final_{did}", on_click=lambda d=did: setattr(st.session_state, "current_doc", load_crawled_doc_by_id(d)))

st.divider()


# 0. TAB BM25 SEARCH
with tab_bm25:
    st.subheader("🔎 Tìm Kiếm Từ Khóa BM25 Trên Toàn Bộ Kho Dữ Liệu")
    bm25_q = st.text_input(
        "Nhập từ khóa y khoa / câu hỏi lâm sàng cần tra cứu:",
        value="xuất huyết giảm tiểu cầu",
        key="bm25_search_query",
    )
    col_k, col_btn = st.columns([1, 2])
    with col_k:
        bm25_top_k = st.slider("Số lượng kết quả (Top K):", min_value=3, max_value=25, value=5, step=1)
    with col_btn:
        st.write("")
        st.write("")
        run_bm25 = st.button("🚀 Tìm Kiếm Bằng BM25s", type="primary")

    if run_bm25 or bm25_q.strip():
        import time

        with st.spinner("Đang tìm kiếm qua memory-mapped BM25s..."):
            bm25_retriever = get_bm25_index()
            t0 = time.time()
            bm25_results = bm25_retriever.search(query=bm25_q.strip(), top_k=bm25_top_k)
            bm25_elapsed = (time.time() - t0) * 1000

        st.success(f"Tìm thấy **{len(bm25_results)} chunks** phù hợp trong **{bm25_elapsed:.1f} ms**!")

        for r_idx, (cid, score, meta) in enumerate(bm25_results, 1):
            did = meta.get("doc_id", "N/A")
            d_title = meta.get("title", f"Document #{did}")
            d_lang = meta.get("lang", "vi").upper()
            d_text = meta.get("chunk_text", "").strip()
            d_words = len(d_text.split())

            with st.container():
                st.markdown(
                    f"""
                    <div class='chunk-card' style='border-left-color: #2563eb;'>
                        <div class='chunk-header' style='color: #1d4ed8;'>
                            <span>#{r_idx} | Doc ID: <b>{did}</b> <span class='badge badge-blue'>{d_lang}</span></span>
                            <span>BM25 Score: <b>{score:.2f}</b> | Chunk ID: {cid} ({len(d_text)} ký tự, {d_words} từ)</span>
                        </div>
                        <h4 style='margin: 4px 0 8px 0; color: #0f172a;'>{d_title}</h4>
                        <div class='chunk-body'>{d_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_load, _ = st.columns([2, 5])
                with col_load:
                    if st.button(f"👉 Mở Toàn Văn Document #{did}", key=f"open_doc_{cid}_{r_idx}"):
                        loaded = load_crawled_doc_by_id(did)
                        if loaded:
                            st.session_state.current_doc = loaded
                            st.rerun()

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
