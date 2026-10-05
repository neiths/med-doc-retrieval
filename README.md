# 🏥 Road to AI 2026 (R2AI) - Multilingual Medical Document Retrieval System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![uv](https://img.shields.io/badge/environment-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Vector DB](https://img.shields.io/badge/Vector_DB-Qdrant_Local-red.svg)](https://qdrant.tech/)
[![Embedding](https://img.shields.io/badge/Embedding-BGE--M3_(FP16)-green.svg)](https://huggingface.co/BAAI/bge-m3)
[![Reranker](https://img.shields.io/badge/Reranker-BGE--Reranker_(FP16)-orange.svg)](https://huggingface.co/BAAI/bge-reranker-large)
[![Tests](https://img.shields.io/badge/tests-37%2F37_passing-brightgreen.svg)](tests/)
[![Mock Validation](https://img.shields.io/badge/Macro_F2-0.8775-success.svg)](scripts/run_mock_eval.py)

Hệ thống truy hồi thông tin y sinh đa ngôn ngữ (**Multilingual Medical Document Retrieval System**) được xây dựng phục vụ cuộc thi **Road to AI 2026 (R2AI)**.

Hệ thống giải quyết bài toán "khoảng cách thông tin y tế" xuyên biên giới ngôn ngữ:
- **Truy vấn đầu vào (Query):** Tiếng Việt (VI) gồm các câu hỏi lâm sàng, triệu chứng, phác đồ điều trị.
- **Kho tri thức truy hồi (Corpus):** Đa ngôn ngữ (**Tiếng Việt - Tiếng Anh - Tiếng Trung**).
- **Mục tiêu đánh giá:** Định vị chính xác tài liệu liên quan (**Document-level**) và trích xuất nguyên vẹn đoạn văn bản chứa bằng chứng y khoa (**Chunk-level**) tối ưu theo độ đo **Macro F2** ($\beta = 2$).

---

## 🏗️ Tổng quan Kiến trúc Hệ thống (System Architecture)

```text
                                  [User Query (VI)]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [Mở rộng Truy vấn Đa ngữ]                       [Truy xuất Trực tiếp PubMed (EN)]
     ├── ICD-10 Tam ngữ (VI-EN-ZH)                   ├── Trích xuất MeSH Keywords (EN)
     ├── Medical Acronyms (STEMI, HFrEF,...)         ├── PubTator 3.0 / Europe PMC / NCBI API
     └── Zh Clinical Keywords                        └── 2-Tier Caching (Memory + Disk)
                  │                                               │
                  ▼                                               ▼
  [Enriched Query (VI + ZH Keywords)]                  [PubMed Candidates (Chunks)]
                  │                                               │
                  ▼                                               │
      [BGE-M3 Dual Encoding]                                      │
      ├── Dense Vector (1024-dim, FP16)                           │
      └── Native Lexical Sparse Tokens                            │
                  │                                               │
                  ▼                                               │
   [Qdrant Local Hybrid Engine]                                   │
   ├── Indexed Clean Corpus (VI & ZH)                             │
   └── Reciprocal Rank Fusion (RRF)                               │
                  │                                               │
                  ▼                                               │
      [Local Candidates (VI & ZH)]                                │
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                         [Gộp Toàn Bộ Ứng Viên Đa Ngôn Ngữ]
                                          │
                                          ▼
                       [BAAI/bge-reranker-large (Cross-Encoder)]
                       ├── Chấm điểm cặp (Query, Contextual Chunk)
                       └── Balanced Per-Doc Selection (max_chunks_per_doc: 2)
                                          │
                                          ▼
                       [Top-K Documents & Chunks Selection]
                       (Bảo toàn 100% chunk_text nguyên bản tài liệu)
                                          │
                                          ▼
                       [Submission Validator & Packager]
                       (Tự động tạo file ZIP phẳng chuẩn thể lệ BTC)
```

---

## 💡 Điểm Sáng Kỹ Thuật & Giải Pháp Cốt Lõi

### 1. Làm sạch Boilerplate Động & Giải mã Mojibake Tiếng Trung
- **Tự động khôi phục font chữ tiếng Trung (`fix_mojibake`):** Phát hiện và đảo ngược lỗi giải mã GB18030 thành UTF-8 nguyên bản cho hàng ngàn bài viết tiếng Trung (ví dụ `鐧界櫆...` $\to$ `白癜风...`).
- **6 Bộ Quy tắc Master Regex ([configs/boilerplate_patterns.yaml](configs/boilerplate_patterns.yaml)):** Thay thế các chuỗi literal đơn lẻ bằng mẫu Regex quy nạp bao quát 100% dấu thời gian (DateTime), tên tòa soạn/đài truyền thanh 63 tỉnh thành, bộ ban ngành, hotline và chân trang trích dẫn nguồn mà không xóa nhầm nội dung y khoa.
- **UI Lọc Rác Trực Quan ([scripts/pattern_cleaner_ui.py](scripts/pattern_cleaner_ui.py)):** Streamlit Dashboard cho phép tích chọn dòng rác theo lô (Batch Selection), hỗ trợ **Quy nạp thông minh** (*Smart Generalization*), **Sandbox Kiểm thử Live Regex** và lưu vĩnh viễn vào YAML.

### 2. Phân đoạn Bảo toàn Ranh giới Câu (Sentence-Aware Boundary Chunking)
- Không cắt ngang từ ngữ (loại bỏ hoàn toàn lỗi ngắt cụt chữ như `ợng vị...`).
- Tự động nhận diện ranh giới câu đa ngữ (VI: `.`, `?`, `!`; ZH: `。`, `！`, `？`; EN: `.` sau từ hoàn chỉnh).
- **Bảo vệ từ viết tắt y tế:** Nhận biết các tiền tố học hàm, chức danh y khoa (`BS.`, `TS.`, `PGS.`, `BV.`, `Dr.`) và số thập phân (`3.5 mg`, `0.9%`) để không bị ngắt câu sai.
- **Contextual Enrichment:** Bổ sung tiêu đề/chuyên mục vào `contextual_text` khi tính vector nhúng, đồng thời giữ nguyên bản tuyệt đối `chunk_text` để nộp bài.

### 3. Lưu trữ Parquet Shard Hiệu Năng Cao & Tối ưu I/O
- Chuyển đổi dữ liệu cào thô sang chuẩn **Apache Parquet nén ZSTD level 9** ([scripts/convert_jsonl_to_parquet.py](scripts/convert_jsonl_to_parquet.py)).
- Giảm dung lượng từ ~265MB JSONL xuống còn **58MB Parquet** (40,273 bài viết hợp lệ), tốc độ nạp dữ liệu tăng hơn **15 lần**.
- Tích hợp đồng bộ tự động lên **Hugging Face Storage Bucket** (`hf sync`).

### 4. Cây Ontology Tam ngữ ICD-10 & Mở rộng Viết tắt Lâm sàng
- Tích hợp danh mục bệnh học Bộ Y tế Việt Nam, WHO ICD-10 và ICD-10-CN:
  - **10.002** thực thể trong ontology hoàn chỉnh ([icd10_ontology.json](data/lexicon/icd10_ontology.json)).
  - **9.076** cặp song ngữ VI-ZH ([icd10_vi_zh.json](data/lexicon/icd10_vi_zh.json)) và **9.365** cặp song ngữ VI-EN ([icd10_vi_en.json](data/lexicon/icd10_vi_en.json)).
  - **Từ điển từ viết tắt y khoa lâm sàng:** Giải mã các từ viết tắt chuyên sâu (STEMI, HFrEF, ARDS, COPD, DKA,...) sang cả tiếng Việt, tiếng Anh và tiếng Trung.

### 5. Native Hybrid Search trên Qdrant Local Engine
- Chạy Qdrant cục bộ nhúng trực tiếp tại `data/indices/qdrant_db`, **không phụ thuộc Docker**.
- Kết hợp đồng thời **Dense Vector (1024 chiều)** và **Learned Lexical Sparse Tokens** từ `BAAI/bge-m3` ở chế độ **FP16**.
- Hợp nhất điểm số tự động bằng thuật toán **Reciprocal Rank Fusion (RRF)**.

### 6. Tái xếp hạng Cân bằng (Balanced Per-Document Chunk Selection)
- Sử dụng mô hình Cross-Encoder `BAAI/bge-reranker-large` (chạy FP16 mượt mà trên GPU $\le 6$GB VRAM).
- Áp dụng cơ chế **giới hạn số chunk tối đa trên mỗi tài liệu** (`max_chunks_per_doc: 2`): Ngăn chặn tình trạng một tài liệu dài độc chiếm toàn bộ danh sách trả về, đẩy mạnh Document-level Recall và Chunk-level Precision.

---

## ⚙️ Tech Stack & Tuân thủ Thể lệ Cuộc thi

| Thành phần | Công nghệ / Mô hình | Đáp ứng Thể lệ Cuộc thi (R2AI Constraints) |
| :--- | :--- | :--- |
| **Quản lý Môi trường** | `uv` + Python 3.11 | Cài đặt cực nhanh, khóa phụ thuộc 100% qua `uv.lock`. |
| **Vector Database** | **Qdrant (Local Embedded)** | Lưu trữ nhúng tại `data/indices/qdrant_db`, hỗ trợ Native Hybrid Search. |
| **Embedding Model** | `BAAI/bge-m3` (**FP16**) | Trích xuất Dense (1024d) + Sparse đa ngữ VI-EN-ZH, $\le 14B$ tham số. |
| **Re-ranker** | `BAAI/bge-reranker-large` (**FP16**) | Cross-Encoder chấm điểm câu hỏi và đoạn trích, hỗ trợ Balanced Selection. |
| **Ontology Đa ngữ** | **ICD-10 & Clinical Acronyms** | Ánh xạ thực thể bệnh học lâm sàng VI $\leftrightarrow$ EN $\leftrightarrow$ ZH. |
| **Dynamic PubMed Client** | Europe PMC, NCBI Entrez | Truy vấn tài liệu tiếng Anh theo thời gian thực, cache đĩa `pubmed_cache.jsonl`. |
| **Độ đo đánh giá** | Macro F2 ($\beta = 2.0$) | Ưu tiên Recall gấp 2 lần Precision; đánh giá chunk theo exact/overlap. |

---

## 📂 Cấu trúc Thư mục Dự án

```text
med-doc-retrieval/
├── configs/
│   ├── config.yaml                  # Cấu hình siêu tham số (Qdrant, FP16, chunk size, top-k, weights)
│   ├── boilerplate_patterns.yaml    # 6 bộ Master Regex làm sạch rác web crawl
│   └── pubmed_stopwords.txt         # Danh sách từ dừng y sinh cho PubMed query
├── data/
│   ├── lexicon/                     # Từ điển & Cây ontology bệnh học
│   │   ├── icd10_ontology.json      # Cây phân loại bệnh học tam ngữ VI-EN-ZH (10.002 thực thể)
│   │   ├── icd10_vi_zh.json         # Từ điển thực thể bệnh lý VI -> ZH (9.076 thực thể)
│   │   ├── icd10_vi_en.json         # Từ điển thực thể bệnh lý VI -> EN (9.365 thực thể)
│   │   ├── medical_terms.json       # Từ điển y khoa mở rộng (9.440 thuật ngữ)
│   │   └── medical_acronyms.json    # Từ điển viết tắt y khoa lâm sàng tam ngữ (STEMI, HFrEF,...)
│   ├── mock/                        # Bộ dữ liệu mock validation phục vụ benchmark
│   │   ├── articles_all.jsonl       # 29 bài viết mẫu (VI, ZH, EN có PMID)
│   │   ├── queries_val.jsonl        # 8 câu hỏi kiểm định thực tế
│   │   └── ground_truth.json        # Nhãn vàng chuẩn xác theo ký tự chuỗi con
│   ├── raw/                         # Dữ liệu cào thô & dữ liệu ban tổ chức
│   │   ├── crawled_shard_1.jsonl    # 18.798 bài cào thô
│   │   ├── crawled_shard_2.jsonl    # 8.354 bài cào thô
│   │   ├── crawled_shard_3.jsonl    # 20.854 bài cào thô
│   │   └── queries.jsonl            # Tập câu hỏi chính thức từ BTC
│   ├── processed/
│   │   └── parquet_corpus/          # Corpus đã làm sạch và đóng gói Parquet ZSTD
│   │       └── corpus_part_000.parquet # 40.273 bài viết sạch (58 MB)
│   └── indices/                     # Qdrant Local Engine database
│       └── qdrant_db/               # Bộ chỉ mục vector nhúng cục bộ
├── src/
│   ├── config.py                    # Pydantic schema quản lý cấu hình hệ thống
│   ├── crawler/
│   │   ├── url_scraper.py           # Web scraper bất đồng bộ có checkpoint resume
│   │   ├── pubmed.py                # Client tra cứu PubMed / Europe PMC (có 2-tier cache)
│   │   └── query_translator.py      # Trích xuất từ khóa y khoa VI -> EN & VI -> ZH (ICD-10)
│   ├── ingestion/
│   │   ├── cleaner.py               # Chuẩn hóa Unicode NFC, Regex boilerplate & sửa Mojibake
│   │   └── chunker.py               # Tách đoạn theo câu (Sentence-Aware) bảo vệ từ viết tắt y tế
│   ├── embedding/
│   │   └── bge_m3.py                # BGE-M3 Embedder (FP16, Dense + Native Lexical Sparse)
│   ├── retrieval/
│   │   ├── qdrant_index.py          # Qdrant Local Hybrid Engine (Dense + Sparse + RRF)
│   │   ├── dense_index.py           # FAISS Dense Index (dự phòng)
│   │   ├── sparse_index.py          # BM25 Sparse Index (dự phòng)
│   │   └── hybrid.py                # RRF Retriever hợp nhất
│   ├── reranker/
│   │   └── bge_reranker.py          # Cross-Encoder Reranker BGE-Reranker-Large (FP16)
│   ├── evaluation/
│   │   └── metrics.py               # Precision, Recall, Macro F2 (Doc & Chunk levels)
│   ├── submission/
│   │   └── formatter.py             # Schema validator & tự động đóng gói ZIP phẳng
│   └── pipeline.py                  # Điều phối toàn bộ luồng End-to-End Pipeline
├── scripts/
│   ├── convert_jsonl_to_parquet.py  # Chuyển đổi, làm sạch và nén JSONL sang Parquet ZSTD
│   ├── pattern_cleaner_ui.py        # Streamlit UI lọc rác & sinh Regex thông minh
│   ├── preload_models.py            # Tải trước và warm-up toàn bộ weights mô hình
│   ├── run_mock_eval.py             # Benchmark Macro F2 end-to-end trên mock validation
│   ├── test_gpu_memory.py           # Script stress test VRAM trên GPU
│   └── sync_to_hf_bucket.sh         # Script đồng bộ dữ liệu lên Hugging Face Bucket
├── tests/                           # Bộ kiểm thử tự động (37/37 tests PASS)
├── outputs/submissions/             # Nơi lưu trữ file nộp bài submission.zip
├── main.py                          # CLI điều khiển toàn diện hệ thống
├── pyproject.toml                   # Quản lý dependencies với uv
└── README.md                        # Tài liệu hướng dẫn dự án
```

---

## 🚀 Hướng Dẫn Khởi Chạy Nhanh (Quickstart)

### Bước 1: Cài đặt môi trường với `uv`
```bash
# 1. Đồng bộ hóa môi trường ảo và thư viện
uv sync --extra dev

# 2. Kích hoạt môi trường
source .venv/bin/activate
```

### Bước 2: Tải trước weights mô hình (Chạy Offline 100%)
```bash
# Tải và warm-up weights BGE-M3, BGE-Reranker-Large
uv run python scripts/preload_models.py
```

### Bước 3: Chạy toàn bộ Unit Tests
```bash
# Đảm bảo toàn bộ 37 bài kiểm thử đều PASS
uv run pytest
```

### Bước 4: Chạy Benchmark Macro F2 trên Tập Mock Validation
```bash
# Chạy chu trình End-to-End: Indexing -> Qdrant Hybrid -> Cross-Encoder Reranker
uv run python scripts/run_mock_eval.py
```

---

## 💻 Cẩm Nang Sử Dụng CLI Thống Nhất (`main.py`)

Hệ thống cung cấp giao diện dòng lệnh duy nhất qua `main.py`:

```bash
uv run python main.py --help
```

### 1. Thu thập dữ liệu từ danh sách URLs (VI & ZH, hỗ trợ Checkpoint Resume)
```bash
uv run python main.py crawl-urls --input data/raw/sample_urls.jsonl --output data/raw/crawled_articles.jsonl --concurrency 15 --resume
```

### 2. Làm sạch, loại bỏ rác Regex & Đóng gói sang Parquet Shards
```bash
uv run python main.py clean-parquet --input data/raw --output-dir data/processed/parquet_corpus --chunk-size 100000
```

### 3. Mở UI Streamlit Lọc Rác & Thử nghiệm Regex Trực Tiếp
```bash
uv run python main.py ui
# hoặc chạy trực tiếp:
uv run streamlit run scripts/pattern_cleaner_ui.py
```

### 4. Thu thập tài liệu tiếng Anh ngoại tuyến từ PubMed
```bash
uv run python main.py fetch-pubmed --query "myocardial infarction treatment" --max-results 100 --output data/processed/pubmed_articles.jsonl
```

### 5. Xây dựng chỉ mục Qdrant Local Engine
```bash
uv run python main.py build-index --input data/processed/parquet_corpus --output-dir data/indices
```
*Chỉ mục sẽ được lưu trực tiếp vào database cục bộ tại `data/indices/qdrant_db`.*

### 6. Tìm kiếm thử nghiệm một câu hỏi (End-to-End Search)
```bash
uv run python main.py search "Cần xử trí như thế nào khi bệnh nhân bị nhồi máu cơ tim cấp STEMI?"
```

### 7. Đánh giá nội bộ trên tập Validation (Macro F2 Score)
```bash
uv run python main.py evaluate --predictions outputs/val_predictions.json --ground-truth data/mock/ground_truth.json
```

### 8. Sinh file đóng gói nộp bài chính thức (Leaderboard Submission)
```bash
uv run python main.py generate-submission --queries data/raw/queries.jsonl --name submission.json
```
*Hệ thống sẽ chạy batch inference, xác thực schema và tạo file ZIP phẳng tại **`outputs/submissions/submission.zip`** sẵn sàng nộp lên hệ thống chấm thi!*

---

## 📊 Phương Pháp Đánh Giá & Kết Quả Thực Nghiệm

Hiệu suất của hệ thống được đánh giá theo độ đo chính thức của cuộc thi: **Macro F2** ($\beta = 2$):

$$F_2 = \frac{5 \times \mathrm{Precision} \times \mathrm{Recall}}{4 \times \mathrm{Precision} + \mathrm{Recall}}$$

Trọng số $\beta = 2$ nhấn mạnh **Recall (Độ bao phủ)** gấp 2 lần **Precision**.

### Kết Quả Benchmark Thực Tế Trên Tập Mock Validation:

| Cấp độ đánh giá | Precision | Recall (Độ bao phủ) | **Macro F2 ($\beta=2$)** |
| :--- | :---: | :---: | :---: |
| **Document Level** (Tài liệu) | 81.46% | **95.83%** | **0.9202** |
| **Chunk Level** (Đoạn trích) | 58.38% | **95.83%** | **0.8349** |
| **Combined Score (Tổng hợp)** | — | — | **0.8775** |

> [!NOTE]  
> Nhờ tính năng **Sentence-Aware Boundary Chunking** kết hợp **Balanced Per-Doc Chunk Selection** (`max_chunks_per_doc: 2`), Chunk-level Recall đạt tới **95.83%**, giúp tổng điểm Macro F2 vượt mốc **0.877**.

---

## 📋 Checklist Trước Khi Nộp Bài Lên Dashboard

1. [ ] **Định dạng file ZIP:** Phải là file ZIP phẳng, chứa **duy nhất 1 file `.json`** (không nằm trong thư mục con).
2. [ ] **Cấu trúc trường JSON:** Gồm trường `id` (int), `relevant_docs` (list string), `relevant_chunks` (list object `{"doc_id": "...", "chunk_text": "..."}`).
3. [ ] **Quy định `doc_id`:** Với VI/ZH là `id` gốc từ BTC; với EN **bắt buộc là mã PMID** của PubMed.
4. [ ] **Quy định `chunk_text`:** Phải là chuỗi con trích xuất nguyên bản từ tài liệu gốc, không tự ý viết lại hay sinh mới. Khuyến nghị mỗi chunk không vượt quá ~1.024 tokens.
5. [ ] **Giới hạn số lần nộp:** Tối đa 10 lần/ngày (Public Phase) và 5 lần tổng cộng (Private Phase).