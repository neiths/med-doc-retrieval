"""Interactive Streamlit UI: Select multiple junk lines at once using st.form to eliminate lag, then auto-generate Regex."""

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
    page_title="ViBioMIR Batch Boilerplate Studio",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ ViBioMIR Batch Click-to-Clean Studio")
st.markdown(
    "**Không bị giật lag!** Bạn có thể tích chọn hàng loạt câu rác cùng một lúc, sau đó bấm nút để hệ thống xử lý 1 lần duy nhất và tự động sinh Regex."
)

# Initialize session state for selected phrases
if "selected_phrases" not in st.session_state:
    st.session_state["selected_phrases"] = []


@st.cache_data(show_spinner="Đang nạp dữ liệu Parquet corpus...")
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
            if len(line) >= 4:
                line_counter[line] += 1
                lines.add(line)
        for line in lines:
            doc_counter[line] += 1

    records = []
    for line, total_cnt in line_counter.most_common(2000):
        already_caught = any(p.match(line) for p in COMPILED_BOILERPLATE)
        records.append({
            "Chọn_Xóa": False,
            "Nội_dung_dòng": line,
            "Số_lần_lặp": total_cnt,
            "Số_bài_chứa": doc_counter[line],
            "Đã_lọc_sẵn": "✅" if already_caught else "",
        })

    return pd.DataFrame(records), len(subset)


def build_regex_from_phrases(phrases: list[str]) -> str:
    """Safely escapes and combines literal phrases into a robust regex pattern."""
    if not phrases:
        return ""
    escaped_patterns = []
    for ph in phrases:
        clean_p = ph.strip()
        if clean_p:
            escaped = re.escape(clean_p)
            escaped_patterns.append(escaped)
    return r"(?i)^.*(" + "|".join(escaped_patterns) + r").*$"


def load_saved_patterns() -> list[str]:
    if PATTERNS_CONFIG_FILE.exists():
        try:
            with open(PATTERNS_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("patterns", [])
        except Exception:
            pass
    return []


def save_patterns(patterns: list[str]):
    PATTERNS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PATTERNS_CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump({"patterns": patterns}, f, allow_unicode=True, sort_keys=False)


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
display_limit = st.sidebar.select_slider("Số lượng dòng hiển thị:", options=[50, 100, 200, 500, 1000], value=100)
min_freq = st.sidebar.slider("Tần suất xuất hiện tối thiểu:", min_value=2, max_value=500, value=5)
search_kw = st.sidebar.text_input("Tìm kiếm từ khóa trong dòng:", value="")

# Load line frequency
df_lines, total_docs = extract_line_frequencies(df_corpus, selected_lang, selected_domain)

st.sidebar.markdown(f"**Tổng số bài phân tích:** {total_docs:,}")
st.sidebar.markdown(f"**Tổng số dòng khác biệt:** {len(df_lines):,}")

# Tabs
tab1, tab2 = st.tabs([
    "📋 Chọn Hàng Loạt Dòng Rác & Sinh Regex",
    "🔍 So Sánh Trước & Sau (Before / After)",
])

# ----------------- TAB 1: BATCH SELECTION FORM -----------------
with tab1:
    st.subheader("1. Tích chọn hàng loạt dòng rác (Không bị tải lại trang)")
    st.caption(
        "💡 **Cách dùng:** Tích chọn thoải mái bao nhiêu dòng tùy ý ở bảng dưới mà không lo bị giật lag. "
        "Khi chọn xong hết, bấm nút **'⚡ XÁC NHẬN & TỰ ĐỘNG SINH REGEX'** ở cuối bảng."
    )

    display_df = df_lines[df_lines["Số_lần_lặp"] >= min_freq].copy()
    if search_kw:
        display_df = display_df[
            display_df["Nội_dung_dòng"].str.contains(search_kw, case=False, na=False)
        ]

    display_df = display_df.head(display_limit)

    # Use st.form to completely prevent page reload on individual checkbox clicks
    with st.form("batch_selection_form"):
        edited_df = st.data_editor(
            display_df,
            column_config={
                "Chọn_Xóa": st.column_config.CheckboxColumn(" Chọn Xóa", help="Tích chọn dòng này là rác", default=False),
                "Nội_dung_dòng": st.column_config.TextColumn("Nội dung dòng văn bản", width="large"),
                "Số_lần_lặp": st.column_config.NumberColumn("Số lần lặp", width="small"),
                "Số_bài_chứa": st.column_config.NumberColumn("Số bài chứa", width="small"),
                "Đã_lọc_sẵn": st.column_config.TextColumn("Đã lọc sẵn", width="small"),
            },
            disabled=["Nội_dung_dòng", "Số_lần_lặp", "Số_bài_chứa", "Đã_lọc_sẵn"],
            hide_index=True,
            use_container_width=True,
            height=500,
        )

        submitted = st.form_submit_button("⚡ XÁC NHẬN & TỰ ĐỘNG SINH REGEX (Bấm sau khi chọn xong)", type="primary")

    if submitted:
        selected_rows = edited_df[edited_df["Chọn_Xóa"] == True]
        st.session_state["selected_phrases"] = selected_rows["Nội_dung_dòng"].tolist()
        st.toast(f"Đã chọn thành công {len(st.session_state['selected_phrases'])} dòng rác!", icon="🎉")

    # Display results
    selected_phrases = st.session_state["selected_phrases"]

    st.markdown("---")
    st.subheader(f"2. Kết Quả Tổng Hợp ({len(selected_phrases)} dòng đã chọn)")

    if selected_phrases:
        col_list, col_regex = st.columns([1, 1])

        with col_list:
            st.markdown("##### 📄 Danh sách các câu đã chọn (1-Click Copy gửi tôi):")
            formatted_list = "\n".join(f"- {p}" for p in selected_phrases)
            st.text_area("Copy danh sách này:", value=formatted_list, height=240)

        with col_regex:
            st.markdown("##### ⚡ Biểu thức Regex tự động sinh (Hệ thống tự tạo):")
            generated_regex = build_regex_from_phrases(selected_phrases)
            st.code(generated_regex, language="python")

            if st.button("💾 Lưu các mẫu này vào configs/boilerplate_patterns.yaml", type="primary"):
                existing = load_saved_patterns()
                updated = list(dict.fromkeys(existing + [generated_regex]))
                save_patterns(updated)
                st.success(f" Đã lưu thành công {len(updated)} pattern vào configs/boilerplate_patterns.yaml!")
    else:
        st.info("👉 Hãy tích chọn các dòng ở bảng trên rồi bấm nút **'⚡ XÁC NHẬN & TỰ ĐỘNG SINH REGEX'** để xem kết quả.")

# ----------------- TAB 2: BEFORE / AFTER INSPECTOR -----------------
with tab2:
    st.subheader("3. So sánh trực quan văn bản Trước vs Sau khi làm sạch")

    sample_doc_ids = df_corpus["doc_id"].tolist()
    col_sel, col_slide = st.columns([2, 2])
    with col_sel:
        selected_doc_id = st.selectbox("Chọn ID bài viết để kiểm tra:", sample_doc_ids[:500], index=0)
    with col_slide:
        chunk_size = st.slider("Max chunk size:", min_value=200, max_value=800, value=400, step=50)

    sample_row = df_corpus[df_corpus["doc_id"] == selected_doc_id].iloc[0]
    raw_text = sample_row["text"]
    title = sample_row["title"]
    domain = sample_row["domain"]
    lang = sample_row["lang"]

    st.markdown(f"**Tiêu đề:** {title} | **Nguồn:** `{domain}` | **Ngôn ngữ:** `{lang}`")

    # Clean text using compiled boilerplate + newly selected phrases
    active_patterns = list(COMPILED_BOILERPLATE)
    if selected_phrases:
        temp_rgx = build_regex_from_phrases(selected_phrases)
        if temp_rgx:
            active_patterns.append(re.compile(temp_rgx))

    cleaned_lines = []
    for l in raw_text.split("\n"):
        stripped = l.strip()
        if not stripped:
            continue
        if any(p.match(stripped) for p in active_patterns):
            continue
        cleaned_lines.append(stripped)

    clean_text = "\n".join(cleaned_lines)

    col_raw, col_clean = st.columns(2)
    with col_raw:
        st.markdown(f"##### ❌ Văn bản gốc ({len(raw_text):,} ký tự)")
        st.text_area("Raw Text", value=raw_text, height=450, disabled=True)

    with col_clean:
        st.markdown(f"#####  Văn bản sau khi làm sạch ({len(clean_text):,} ký tự)")
        st.text_area("Cleaned Text", value=clean_text, height=450, disabled=True)

    st.markdown("---")
    st.subheader("🧩 Xem trước kết quả tách Chunk hoàn chỉnh câu")
    chunker = DocumentChunker(max_chunk_size=chunk_size, chunk_overlap=80)
    chunks = chunker.chunk_document(doc_id=str(selected_doc_id), text=clean_text, lang=lang)

    st.write(f"Đã tách thành **{len(chunks)} chunks** nguyên vẹn câu:")
    for idx, c in enumerate(chunks):
        with st.expander(f"Chunk {idx+1} ({len(c.chunk_text)} ký tự) | [{c.char_start}:{c.char_end}]"):
            st.code(c.chunk_text, language="text")
