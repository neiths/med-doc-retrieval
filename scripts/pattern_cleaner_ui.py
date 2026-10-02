"""Interactive Streamlit UI for discovering, inspecting, and testing boilerplate regex patterns on crawled corpus."""

import re
from collections import Counter
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import yaml

from src.ingestion.cleaner import COMPILED_BOILERPLATE, BOILERPLATE_PATTERNS
from src.ingestion.chunker import DocumentChunker

PATTERNS_CONFIG_FILE = Path("configs/boilerplate_patterns.yaml")

st.set_page_config(
    page_title="ViBioMIR Boilerplate Inspector & Pattern Studio",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 ViBioMIR Boilerplate Discovery & Regex Studio")
st.markdown(
    "Khám phá các dòng lặp lại nhiều nhất trong corpus, kiểm thử biểu thức chính quy (Regex) và làm sạch dữ liệu trong thời gian thực."
)


@st.cache_data(show_spinner="Đang đọc dữ liệu Parquet corpus...")
def load_corpus_data(parquet_path: str) -> pd.DataFrame:
    p = Path(parquet_path)
    if not p.exists():
        st.error(f"Không tìm thấy file: {parquet_path}")
        return pd.DataFrame()
    table = pq.read_table(p, columns=["doc_id", "title", "text", "lang", "domain"])
    return table.to_pandas()


@st.cache_data(show_spinner="Đang phân tích tần suất các dòng văn bản...")
def extract_line_frequencies(df: pd.DataFrame, lang_filter: str, domain_filter: str):
    subset = df
    if lang_filter != "all":
        subset = subset[subset["lang"] == lang_filter]
    if domain_filter != "all":
        subset = subset[subset["domain"] == domain_filter]

    line_counter = Counter()
    doc_counter = Counter()

    for text in subset["text"].dropna():
        lines = set()
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            # Ignore very short lines or empty lines
            if len(line) >= 4:
                line_counter[line] += 1
                lines.add(line)
        for line in lines:
            doc_counter[line] += 1

    records = []
    for line, total_cnt in line_counter.most_common(2000):
        records.append({
            "line_text": line,
            "total_count": total_cnt,
            "doc_count": doc_counter[line],
            "char_len": len(line),
        })

    return pd.DataFrame(records), len(subset)


# Sidebar Configuration
st.sidebar.header("⚙️ Nguồn Dữ Liệu & Bộ Lọc")
parquet_input = st.sidebar.text_input(
    "Đường dẫn Parquet:",
    value="data/processed/parquet_corpus/corpus_part_000.parquet",
)

df_corpus = load_corpus_data(parquet_input)

if df_corpus.empty:
    st.warning("Vui lòng kiểm tra lại đường dẫn file Parquet.")
    st.stop()

# Filter controls
available_langs = ["all"] + sorted(df_corpus["lang"].dropna().unique().tolist())
available_domains = ["all"] + sorted(df_corpus["domain"].dropna().unique().tolist())

selected_lang = st.sidebar.selectbox("Ngôn ngữ:", available_langs, index=0)
selected_domain = st.sidebar.selectbox("Tên miền (Domain):", available_domains, index=0)

min_freq = st.sidebar.slider("Tần suất xuất hiện tối thiểu:", min_value=2, max_value=500, value=10)
search_kw = st.sidebar.text_input("Tìm từ khóa trong dòng:", value="")

# Load line frequency
df_lines, total_docs = extract_line_frequencies(df_corpus, selected_lang, selected_domain)

st.sidebar.markdown(f"**Tổng số bài phân tích:** {total_docs:,}")
st.sidebar.markdown(f"**Tổng số dòng khác biệt:** {len(df_lines):,}")

# Tabs: 1. Line Frequencies, 2. Regex Sandbox, 3. Side-by-Side Cleaner
tab1, tab2, tab3 = st.tabs([
    "📊 Tần Suất Dòng (Frequent Lines)",
    "🧪 Thử Nghiệm Regex (Regex Sandbox)",
    "🔍 So Sánh Trước & Sau (Before / After)",
])

# ----------------- TAB 1: LINE FREQUENCIES -----------------
with tab1:
    st.subheader("1. Các dòng xuất hiện nhiều nhất (Dấu hiệu của Boilerplate)")
    st.caption("Các dòng xuất hiện ở nhiều bài viết khác nhau thường là menu, hotline, footer, quảng cáo.")

    filtered_lines = df_lines[df_lines["total_count"] >= min_freq]
    if search_kw:
        filtered_lines = filtered_lines[
            filtered_lines["line_text"].str.contains(search_kw, case=False, na=False)
        ]

    st.dataframe(
        filtered_lines,
        use_container_width=True,
        column_config={
            "line_text": st.column_config.TextColumn("Nội dung dòng", width="large"),
            "total_count": st.column_config.NumberColumn("Tổng lần xuất hiện", width="small"),
            "doc_count": st.column_config.NumberColumn("Số bài chứa dòng này", width="small"),
            "char_len": st.column_config.NumberColumn("Độ dài", width="small"),
        },
        height=500,
    )

    st.markdown("💡 **Mẹo**: Nhìn vào bảng trên, bạn có thể copy các cụm từ lặp lại nhiều để dán vào Tab **Thử Nghiệm Regex** bên cạnh.")

# ----------------- TAB 2: REGEX SANDBOX -----------------
with tab2:
    st.subheader("2. Kiểm tra Regex trên tập dữ liệu")
    col_input, col_info = st.columns([2, 1])

    with col_input:
        custom_regex = st.text_input(
            "Nhập biểu thức Regex cần kiểm tra:",
            value=r"(?i)^.*(hotline|đặt lịch khám|bản quyền|xem thêm:|chỉ mang tính chất tham khảo).*$",
        )

    with col_info:
        st.markdown("**Các mẫu có sẵn trong hệ thống:**")
        st.write(f"Hiện có `{len(BOILERPLATE_PATTERNS)}` pattern đang được áp dụng trong `cleaner.py`.")

    if custom_regex:
        try:
            pattern = re.compile(custom_regex)
            # Find all matched lines in df_lines
            matched_mask = df_lines["line_text"].apply(lambda s: bool(pattern.match(s)))
            df_matched = df_lines[matched_mask]

            total_removed_instances = df_matched["total_count"].sum()
            docs_affected = df_matched["doc_count"].max() if not df_matched.empty else 0

            st.success(
                f"✅ **Khớp thành công!** Pattern này khớp **{len(df_matched):,} dòng khác nhau**, "
                f"giúp loại bỏ **{total_removed_instances:,} lần xuất hiện rác**."
            )

            st.markdown("##### Danh sách các dòng bị loại bỏ bởi Regex này:")
            st.dataframe(
                df_matched[["line_text", "total_count", "doc_count"]],
                use_container_width=True,
                height=350,
            )

        except re.error as e:
            st.error(f"❌ Cú pháp Regex không hợp lệ: {e}")

# ----------------- TAB 3: SIDE BY SIDE INSPECTOR -----------------
with tab3:
    st.subheader("3. So sánh trực quan văn bản và chunk Trước vs Sau khi làm sạch")

    col_select, col_slider = st.columns([2, 2])
    with col_select:
        # Choose article
        sample_doc_ids = df_corpus["doc_id"].tolist()
        selected_doc_id = st.selectbox("Chọn ID bài viết để soi:", sample_doc_ids[:500], index=0)

    with col_slider:
        chunk_size = st.slider("Max chunk size:", min_value=200, max_value=800, value=400, step=50)

    sample_row = df_corpus[df_corpus["doc_id"] == selected_doc_id].iloc[0]
    raw_text = sample_row["text"]
    title = sample_row["title"]
    domain = sample_row["domain"]
    lang = sample_row["lang"]

    st.markdown(f"**Tiêu đề:** {title} | **Nguồn:** `{domain}` | **Ngôn ngữ:** `{lang}`")

    # Apply regex filter live
    cleaned_lines = []
    test_pattern = None
    if custom_regex:
        try:
            test_pattern = re.compile(custom_regex)
        except Exception:
            pass

    for l in raw_text.split("\n"):
        stripped = l.strip()
        if not stripped:
            continue
        # Check standard boilerplate or custom regex
        if any(p.match(stripped) for p in COMPILED_BOILERPLATE):
            continue
        if test_pattern and test_pattern.match(stripped):
            continue
        cleaned_lines.append(stripped)

    clean_text = "\n".join(cleaned_lines)

    col_raw, col_clean = st.columns(2)

    with col_raw:
        st.markdown(f"##### ❌ Văn bản gốc ({len(raw_text):,} ký tự)")
        st.text_area("Raw Text", value=raw_text, height=450, disabled=True)

    with col_clean:
        st.markdown(f"#####  Văn bản đã làm sạch ({len(clean_text):,} ký tự)")
        st.text_area("Cleaned Text", value=clean_text, height=450, disabled=True)

    st.markdown("---")
    st.subheader("🧩 Xem trước kết quả tách Chunk (Sentence-Aware)")
    chunker = DocumentChunker(max_chunk_size=chunk_size, chunk_overlap=80)
    chunks = chunker.chunk_document(doc_id=str(selected_doc_id), text=clean_text, lang=lang)

    st.write(f"Đã tách thành **{len(chunks)} chunks** hoàn chỉnh câu:")
    for idx, c in enumerate(chunks):
        with st.expander(f"Chunk {idx+1} ({len(c.chunk_text)} ký tự) | [{c.char_start}:{c.char_end}]"):
            st.code(c.chunk_text, language="text")
