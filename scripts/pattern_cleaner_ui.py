"""Interactive Streamlit UI: Select multiple junk lines at once, auto-generate Regex, test in live sandbox, and manage saved YAML patterns."""

import random
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import yaml

from src.ingestion.cleaner import COMPILED_BOILERPLATE, BOILERPLATE_PATTERNS, load_external_boilerplate_patterns
from src.ingestion.chunker import DocumentChunker

PATTERNS_CONFIG_FILE = Path("configs/boilerplate_patterns.yaml")

st.set_page_config(
    page_title="ViBioMIR Batch Boilerplate & Regex Studio",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 ViBioMIR Boilerplate Studio & Regex Sandbox")
st.markdown(
    "Tích chọn câu rác hàng loạt ➔ Tự động sinh Regex ➔ Kiểm thử trực quan (Live Test Sandbox) ➔ Lưu vĩnh viễn vào YAML."
)

# Initialize session states
if "selected_phrases" not in st.session_state:
    st.session_state["selected_phrases"] = []
if "test_regex_input" not in st.session_state:
    st.session_state["test_regex_input"] = ""


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


# Load all active patterns (built-in + external YAML)
saved_yaml_patterns = load_saved_patterns()
active_compiled_patterns = list(COMPILED_BOILERPLATE) + load_external_boilerplate_patterns()


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
    for line, total_cnt in line_counter.most_common(2500):
        # Check against all currently active patterns (built-in + YAML)
        already_caught = any(p.match(line) for p in active_compiled_patterns)
        records.append({
            "Chọn_Xóa": False,
            "Nội_dung_dòng": line,
            "Số_lần_lặp": total_cnt,
            "Số_bài_chứa": doc_counter[line],
            "Đã_lọc_sẵn": "✅ Đã lưu" if already_caught else "",
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
st.sidebar.markdown(f"💾 **Patterns đã lưu trong YAML:** `{len(saved_yaml_patterns)}`")

# 3 Main Tabs
tab1, tab2, tab3 = st.tabs([
    "📋 Chọn Dòng Rác & Sinh Regex",
    "🧪 Sandbox Kiểm Thử Regex (Live Test)",
    "📁 Quản Lý Patterns Đã Lưu Trong YAML",
])

# ----------------- TAB 1: BATCH SELECTION -----------------
with tab1:
    st.subheader("1. Tích chọn hàng loạt dòng rác (Không bị giật lag)")
    st.caption("Các dòng có dấu `✅ Đã lưu` là những dòng đã được tự động loại bỏ bởi các pattern bạn từng lưu trước đó.")

    display_df = df_lines[df_lines["Số_lần_lặp"] >= min_freq].copy()
    if search_kw:
        display_df = display_df[
            display_df["Nội_dung_dòng"].str.contains(search_kw, case=False, na=False)
        ]

    display_df = display_df.head(display_limit)

    with st.form("batch_selection_form"):
        edited_df = st.data_editor(
            display_df,
            column_config={
                "Chọn_Xóa": st.column_config.CheckboxColumn(" Chọn Xóa", help="Tích chọn dòng này là rác", default=False),
                "Nội_dung_dòng": st.column_config.TextColumn("Nội dung dòng văn bản", width="large"),
                "Số_lần_lặp": st.column_config.NumberColumn("Số lần lặp", width="small"),
                "Số_bài_chứa": st.column_config.NumberColumn("Số bài chứa", width="small"),
                "Đã_lọc_sẵn": st.column_config.TextColumn("Trạng thái", width="small"),
            },
            disabled=["Nội_dung_dòng", "Số_lần_lặp", "Số_bài_chứa", "Đã_lọc_sẵn"],
            hide_index=True,
            use_container_width=True,
            height=450,
        )

        submitted = st.form_submit_button("⚡ XÁC NHẬN & TỰ ĐỘNG SINH REGEX (Bấm sau khi chọn xong)", type="primary")

    if submitted:
        selected_rows = edited_df[edited_df["Chọn_Xóa"] == True]
        st.session_state["selected_phrases"] = selected_rows["Nội_dung_dòng"].tolist()
        gen_regex = build_regex_from_phrases(st.session_state["selected_phrases"])
        st.session_state["test_regex_input"] = gen_regex
        st.toast(f"Đã chọn thành công {len(st.session_state['selected_phrases'])} dòng rác!", icon="🎉")

    selected_phrases = st.session_state["selected_phrases"]

    st.markdown("---")
    st.subheader(f"2. Kết Quả Tổng Hợp ({len(selected_phrases)} dòng đã chọn)")

    if selected_phrases:
        col_list, col_regex = st.columns([1, 1])

        with col_list:
            st.markdown("##### 📄 Danh sách các câu đã chọn (1-Click Copy gửi tôi):")
            formatted_list = "\n".join(f"- {p}" for p in selected_phrases)
            st.text_area("Copy danh sách này:", value=formatted_list, height=220)

        with col_regex:
            st.markdown("##### ⚡ Biểu thức Regex tự động sinh (Hệ thống tự tạo):")
            generated_regex = build_regex_from_phrases(selected_phrases)
            st.code(generated_regex, language="python")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🧪 Chuyển sang Tab Sandbox để Test", type="secondary"):
                    st.session_state["test_regex_input"] = generated_regex
                    st.info("👉 Hãy bấm sang Tab **'🧪 Sandbox Kiểm Thử Regex'** ở trên để kiểm tra kết quả!")

            with col_btn2:
                if st.button("💾 Lưu vĩnh viễn vào YAML", type="primary"):
                    existing = load_saved_patterns()
                    updated = list(dict.fromkeys(existing + [generated_regex]))
                    save_patterns(updated)
                    st.success(f" Đã lưu thành công vào configs/boilerplate_patterns.yaml! Lần chạy sau hệ thống sẽ tự nạp lại.")
                    st.rerun()
    else:
        st.info("👉 Hãy tích chọn các dòng ở bảng trên rồi bấm nút **'⚡ XÁC NHẬN & TỰ ĐỘNG SINH REGEX'** để tạo mẫu.")


# ----------------- TAB 2: REGEX TESTING SANDBOX -----------------
with tab2:
    st.subheader("2. Sandbox Kiểm Thử Regex Trực Tiếp (Live Testing)")
    st.caption("Kiểm tra xem biểu thức Regex xóa đúng những câu rác nào và có vô tình xóa nhầm câu y khoa nào không.")

    # Regex input for testing
    regex_to_test = st.text_input(
        "Nhập hoặc dán Regex cần kiểm thử:",
        value=st.session_state.get("test_regex_input", ""),
        help="Mặc định là Regex vừa sinh từ Tab 1. Bạn có thể sửa đổi tùy ý.",
    )

    if not regex_to_test:
        st.warning("Vui lòng sinh Regex từ Tab 1 hoặc nhập một biểu thức Regex vào ô trên để kiểm thử.")
    else:
        try:
            compiled_test_regex = re.compile(regex_to_test)
            st.success("✅ Cú pháp Regex hoàn toàn hợp lệ!")

            mode = st.radio(
                "Chế độ kiểm thử:",
                ["📖 Kiểm thử trên 1 bài viết trong kho", "✍️ Dán đoạn văn bản tự do để thử", "🔬 Quét an toàn trên 100 bài ngẫu nhiên"],
                horizontal=True,
            )

            # MODE A: SINGLE ARTICLE TEST
            if mode == "📖 Kiểm thử trên 1 bài viết trong kho":
                col_pick, col_btn = st.columns([3, 1])
                with col_pick:
                    sample_doc_ids = df_corpus["doc_id"].tolist()
                    selected_doc_id = st.selectbox("Chọn ID bài viết:", sample_doc_ids[:500], index=0)
                with col_btn:
                    st.write("")
                    st.write("")
                    if st.button("🎲 Chọn bài ngẫu nhiên"):
                        selected_doc_id = random.choice(sample_doc_ids[:500])

                sample_row = df_corpus[df_corpus["doc_id"] == selected_doc_id].iloc[0]
                text = sample_row["text"]
                st.markdown(f"**Tiêu đề:** {sample_row['title']} | **Nguồn:** `{sample_row['domain']}`")

                # Test matching on lines
                lines = text.split("\n")
                deleted_lines = []
                kept_lines = []
                for l in lines:
                    stripped = l.strip()
                    if not stripped:
                        continue
                    if compiled_test_regex.match(stripped):
                        deleted_lines.append(stripped)
                    else:
                        kept_lines.append(stripped)

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown(f"##### ❌ Các dòng bị XÓA ({len(deleted_lines)} dòng):")
                    if deleted_lines:
                        for dl in deleted_lines:
                            st.markdown(f"- 🔴 `{dl}`")
                    else:
                        st.info("Không có dòng nào trong bài này khớp với Regex.")

                with col_res2:
                    st.markdown(f"#####  Các dòng được GIỮ LẠI ({len(kept_lines)} dòng):")
                    cleaned_preview = "\n\n".join(kept_lines)
                    st.text_area("Cleaned Preview", value=cleaned_preview, height=350, disabled=True)

            # MODE B: FREE TEXT SANDBOX
            elif mode == "✍️ Dán đoạn văn bản tự do để thử":
                custom_input_text = st.text_area(
                    "Dán đoạn văn bản bạn muốn test vào đây:",
                    value="Bệnh nhân bị viêm xung huyết hang vị dạ dày.\nĐể đặt lịch khám vui lòng liên hệ HOTLINE 1900 2323.\nBác sĩ khuyên uống nhiều nước và ăn nhạt.",
                    height=180,
                )
                if custom_input_text:
                    c_lines = custom_input_text.split("\n")
                    d_lines = [l.strip() for l in c_lines if l.strip() and compiled_test_regex.match(l.strip())]
                    k_lines = [l.strip() for l in c_lines if l.strip() and not compiled_test_regex.match(l.strip())]

                    st.markdown(f"**Kết quả:** Xóa `{len(d_lines)}` dòng rác | Giữ lại `{len(k_lines)}` dòng nội dung.")
                    st.markdown("**Văn bản sau khi làm sạch:**")
                    st.code("\n".join(k_lines), language="text")

            # MODE C: BATCH SAFETY SCAN
            else:
                st.markdown("##### 🔬 Quét kiểm thử an toàn trên 100 bài viết")
                st.caption("Giúp bạn yên tâm kiểm tra xem Regex có vô tình xóa nhầm những câu dài (nội dung y khoa) không.")
                if st.button("🚀 Bắt đầu quét 100 bài", type="primary"):
                    sample_100 = df_corpus.sample(min(100, len(df_corpus)), random_state=42)
                    matched_samples = []
                    total_lines_deleted = 0
                    articles_affected = 0

                    for _, r in sample_100.iterrows():
                        art_deleted = 0
                        for line in str(r["text"]).split("\n"):
                            sl = line.strip()
                            if sl and compiled_test_regex.match(sl):
                                art_deleted += 1
                                total_lines_deleted += 1
                                if len(matched_samples) < 20:
                                    matched_samples.append((sl, r["domain"], len(sl)))
                        if art_deleted > 0:
                            articles_affected += 1

                    st.metric("Tỷ lệ bài viết có rác bị xóa", f"{articles_affected}/100 bài", f"{total_lines_deleted} dòng bị loại bỏ")

                    st.markdown("**Mẫu 20 dòng bị xóa điển hình:**")
                    df_del = pd.DataFrame(matched_samples, columns=["Dòng bị xóa", "Tên miền", "Độ dài ký tự"])
                    st.dataframe(df_del, use_container_width=True)

        except re.error as err:
            st.error(f"❌ Cú pháp Regex không hợp lệ: {err}")


# ----------------- TAB 3: YAML PATTERN MANAGER -----------------
with tab3:
    st.subheader("3. Quản Lý Các Patterns Đã Lưu Trong YAML")
    st.markdown(
        f"File cấu hình: `{PATTERNS_CONFIG_FILE}`. "
        "Mỗi khi bạn khởi động lại ứng dụng, toàn bộ các mẫu này sẽ **tự động được nạp lại 100%** "
        "và áp dụng trực tiếp cho toàn bộ pipeline làm sạch."
    )

    if not saved_yaml_patterns:
        st.info("Chưa có pattern nào được lưu trong YAML.")
    else:
        st.write(f"Hiện có **{len(saved_yaml_patterns)} pattern(s)** đang hoạt động:")
        for idx, pat in enumerate(saved_yaml_patterns):
            with st.expander(f"Pattern #{idx+1} ({len(pat)} ký tự)"):
                st.code(pat, language="python")
                if st.button(f"🗑️ Xóa Pattern #{idx+1}", key=f"del_{idx}"):
                    saved_yaml_patterns.pop(idx)
                    save_patterns(saved_yaml_patterns)
                    st.success(f"Đã xóa pattern #{idx+1}!")
                    st.rerun()

        st.markdown("---")
        if st.button("🗑️ Xóa sạch tất cả patterns trong YAML", type="secondary"):
            save_patterns([])
            st.warning("Đã xóa sạch toàn bộ patterns trong file YAML.")
            st.rerun()
