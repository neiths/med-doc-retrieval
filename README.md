# 🏥 Road to AI 2026 (R2AI) - Multilingual Medical Document Retrieval System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![uv](https://img.shields.io/badge/environment-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Vector DB](https://img.shields.io/badge/Vector_DB-Qdrant_Local-red.svg)](https://qdrant.tech/)
[![Embedding](https://img.shields.io/badge/Embedding-BGE--M3_(FP16)-green.svg)](https://huggingface.co/BAAI/bge-m3)
[![Reranker](https://img.shields.io/badge/Reranker-BGE--Reranker_(FP16)-orange.svg)](https://huggingface.co/BAAI/bge-reranker-large)
[![Tests](https://img.shields.io/badge/tests-25%2F25_passing-brightgreen.svg)](tests/)
[![Mock Validation](https://img.shields.io/badge/Macro_F2-0.8790-success.svg)](scripts/run_mock_eval.py)

Hệ thống truy hồi thông tin y sinh đa ngôn ngữ (**Multilingual Medical Document Retrieval System**) phục vụ cuộc thi **Road to AI 2026 (R2AI)**.

Hệ thống được thiết kế để giải quyết bài toán "khoảng cách thông tin y tế" xuyên ngôn ngữ:
- **Truy vấn đầu vào (Query):** Tiếng Việt (VI).
- **Kho tri thức truy hồi:** Đa ngôn ngữ (**Tiếng Việt - Tiếng Anh - Tiếng Trung**).
- **Mục tiêu:** Định vị chính xác tài liệu liên quan (**Document-level**) và trích xuất nguyên vẹn đoạn văn bản chứa bằng chứng y khoa (**Chunk-level**) theo độ đo **Macro F2** ($\beta = 2$).

---

## 🏗️ Tổng quan Kiến trúc Hệ thống (System Architecture)

```text
                           [User Query (VI)]
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼ (Nhánh Offline: VI & ZH)                          ▼ (Nhánh Online: EN)
┌───────────────────────────────────┐             ┌───────────────────────────────────┐
│ Raw Data (URLs BTC cấp)           │             │ Query Translator (MarianMT)       │
│ └──> Ingestion & Exact Chunking   │             │ └──> Medical Term Extraction (EN) │
│ └──> Contextual Chunk Enrichment  │             │ └──> PubTator 3.0 / Europe PMC    │
│      (Title/Section Metadata)     │             │ └──> Dynamic Fetch & Chunking     │
│ └──> BGE-M3 Dense & Sparse Encode │             │ └──> Contextual Candidate Chunks  │
│ └──> Qdrant Local Engine Storage  │             └─────────────────┬─────────────────┘
└─────────────────┬─────────────────┘                               │
                  │                                                 │
                  ▼                                                 ▼
     [Qdrant Native Hybrid Search]                     [PubMed Candidate Chunks]
     (Dense + Sparse Vector + RRF)                                  │
                  │                                                 │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                           [Gộp Toàn Bộ Ứng Viên Đa Ngôn Ngữ]
                                           │
                                           ▼
                           [Cross-Encoder Reranker (FP16)]
                           (BAAI/bge-reranker-large Scoring)
                                           │
                                           ▼
                           [Top-K Documents & Chunks Selection]
                           (Trích xuất nguyên vẹn chunk_text gốc)
                                           │
                                           ▼
                           [Submission Validator & Packager]
                           (Tự động tạo file ZIP phẳng chuẩn Leaderboard)
```

---

## ⚙️ Tech Stack & Tuân thủ Quy chế Cuộc thi (Constraints Compliance)

| Thành phần | Công nghệ / Mô hình | Đáp ứng Quy chế (Rules & Constraints) |
| :--- | :--- | :--- |
| **Quản lý Môi trường** | `uv` + Python 3.11 | Tối ưu hóa cài đặt cực nhanh, đồng bộ 100% qua `uv.lock`. |
| **Vector Database** | **Qdrant (Local Embedded)** | Lưu trữ nhúng tại `data/indices/qdrant_db`, **không cần Docker**, hỗ trợ Native Hybrid Search (Dense + Sparse) & RRF trực tiếp ở tầng engine. |
| **Contextual Chunking** | Đa ngôn ngữ (`vi`, `zh`, `en`) | Bổ sung tiêu đề/mục (`contextual_text`) khi tính embedding/reranking; **bảo toàn 100% chuỗi con gốc** (`chunk_text`) cho submission. |
| **Embedding Model** | `BAAI/bge-m3` (chế độ **FP16**) | Đa ngôn ngữ VI-EN-ZH, 1024 chiều, $\le 14B$ tham số (dung sai $\le 15B$, tính riêng từng model), phát hành trước 01/08/2026. |
| **Re-ranker** | `BAAI/bge-reranker-large` (chế độ **FP16**) | Cross-Encoder chấm điểm tương quan ngữ nghĩa trực tiếp giữa câu hỏi VI và contextual chunk đa ngôn ngữ (phát hành trước 01/08/2026). |
| **Query Translator** | `Helsinki-NLP/opus-mt-vi-en` / `ndhieu1101` + ICD-10 Ontology | Mô hình dịch y khoa kết hợp từ điển song ngữ chuẩn hóa Bộ Y tế Việt Nam & WHO (>10.000 thực thể bệnh lý, 9.440 thuật ngữ). |
| **External Medical API** | PubTator 3.0, Europe PMC & NCBI Entrez | Tìm kiếm bài báo PubMed theo từ khóa, tải BiocJSON/XML và lưu cache tự động tại `pubmed_cache.jsonl`. |
| **Độ đo đánh giá** | Macro F2 (beta = 2.0) | Ưu tiên Recall gấp 2 lần Precision theo đúng công thức BTC. Đánh giá chunk theo cơ chế overlap/containment. |

---

## 📂 Cấu trúc Thư mục Dự án

```text
med-doc-retrieval/
├── configs/
│   ├── config.yaml             # Cấu hình siêu tham số (Qdrant, FP16, chunk size, top-k, weights)
│   └── pubmed_stopwords.txt    # Danh sách stopwords / filler words khi tìm kiếm y sinh PubMed
├── data/
│   ├── lexicon/
│   │   ├── icd10_ontology.json # Cây phân loại bệnh học song ngữ chính thức Bộ Y tế & WHO (10.002 thực thể)
│   │   ├── icd10_vi_en.json    # Từ điển ánh xạ thực thể bệnh lý VI -> EN phục vụ retrieval
│   │   └── medical_terms.json  # Từ điển y khoa song ngữ VI-EN mở rộng (9.440 thuật ngữ lâm sàng)
│   ├── mock/                   # Bộ dữ liệu mock validation đa ngôn ngữ phục vụ benchmark
│   │   ├── articles_all.jsonl  # 29 bài viết mẫu (VI, ZH, EN có PMID, distractors)
│   │   ├── queries_val.jsonl   # 8 câu hỏi kiểm định thực tế
│   │   └── ground_truth.json   # Nhãn vàng 100% chuẩn xác theo ký tự chuỗi con
│   ├── raw/                    # Dữ liệu thô từ BTC (urls.jsonl, queries.jsonl)
│   ├── processed/              # Chứa chunks.jsonl, pubmed_cache.jsonl
│   └── indices/                # Qdrant Local Engine database (data/indices/qdrant_db)
├── src/
│   ├── config.py               # Pydantic schema quản lý cấu hình hệ thống
│   ├── crawler/                # Thu thập dữ liệu đa ngôn ngữ
│   │   ├── url_scraper.py      # Async scraper cho URLs bài viết VI và ZH (Trafilatura)
│   │   ├── pubmed.py           # Client tra cứu PubTator 3.0 / Europe PMC / NCBI (có disk cache)
│   │   └── query_translator.py # Bộ dịch MarianMT & trích xuất từ khóa y khoa VI -> EN
│   ├── ingestion/              # Tiền xử lý & phân đoạn
│   │   ├── cleaner.py          # Chuẩn hóa Unicode NFC & lọc ngôn ngữ
│   │   └── chunker.py          # Phân đoạn Contextual Chunking & bảo toàn nguyên vẹn chuỗi con
│   ├── embedding/              # Vector hóa
│   │   └── bge_m3.py           # BGE-M3 Embedder (hỗ trợ FP16, tối ưu VRAM)
│   ├── retrieval/              # Tìm kiếm lai (Hybrid Search)
│   │   ├── qdrant_index.py     # Qdrant Local Engine (Native Dense + Sparse + RRF)
│   │   ├── dense_index.py      # FAISS Dense Index (dự phòng)
│   │   ├── sparse_index.py     # BM25 Sparse Index (Jieba & PyVi tokenization)
│   │   └── hybrid.py           # Reciprocal Rank Fusion kết hợp
│   ├── reranker/               # Tinh chỉnh xếp hạng
│   │   └── bge_reranker.py     # Cross-Encoder Reranker (hỗ trợ FP16)
│   ├── evaluation/             # Đánh giá nội bộ
│   │   └── metrics.py          # Precision, Recall, Macro F2 (Doc & Chunk levels, exact/overlap)
│   ├── submission/             # Đóng gói nộp bài
│   │   └── formatter.py        # Schema validator & tự động nén ZIP phẳng
│   └── pipeline.py             # Điều phối End-to-end Pipeline
├── notebooks/
│   └── 01_baseline_exploration.ipynb # Notebook mẫu thử nghiệm từng thành phần
├── scripts/
│   ├── collect_icd10_ontology.py # Thu thập tự động cây ICD-10 Bộ Y tế & ánh xạ WHO
│   ├── preload_models.py       # Tải trước và kiểm tra toàn bộ weights model cho offline inference
│   ├── run_mock_eval.py        # Benchmark đánh giá Macro F2 end-to-end trên tập mock validation
│   └── test_gpu_memory.py      # Script stress test VRAM trên GPU (RTX 3050 6GB)
├── tests/                      # Bộ kiểm thử tự động (32/32 tests passing)
│   ├── test_chunker.py
│   ├── test_icd10_ontology.py
│   ├── test_metrics.py
│   ├── test_pubmed.py
│   ├── test_qdrant.py
│   ├── test_query_translator.py
│   ├── test_submission.py
│   ├── test_tokenization.py
│   └── test_url_scraper.py
├── outputs/submissions/        # Nơi lưu file kết quả submission.zip
├── .env.example                # Template biến môi trường (NCBI, HuggingFace)
├── pyproject.toml              # Quản lý dependencies với uv
├── uv.lock                     # Khóa phiên bản đảm bảo tính đồng nhất 100%
├── main.py                     # CLI điều khiển hệ thống
├── TEAM_GUIDE.md               # Cẩm nang làm việc nhóm và quy ước Git
├── competition_guide.md        # Điều lệ cuộc thi chính thức
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## 🚀 Khởi chạy Nhanh (Quickstart)

### Bước 1: Cài đặt môi trường với `uv`
```bash
# 1. Đồng bộ hóa toàn bộ môi trường và dev dependencies
uv sync --extra dev

# 2. Kích hoạt môi trường ảo
source .venv/bin/activate
```

### Bước 2: Chạy kiểm thử tự động
```bash
# Đảm bảo toàn bộ 25 bài kiểm thử đều PASS
uv run pytest
```

### Bước 3: Đánh giá Benchmark Macro F2 trên Mock Validation
```bash
# Chạy đánh giá toàn diện chu trình Ingestion -> Qdrant Hybrid -> Reranker
uv run python scripts/run_mock_eval.py
```

**Bảng so sánh hiệu năng thực tế trên tập Mock Validation:**

| Cấp độ đánh giá | Chỉ số | Baseline (Trước Contextual) | **Sau Contextual Chunking** | Mức cải thiện |
| :--- | :--- | :---: | :---: | :---: |
| **Document Level** | Precision | 63.66% | **82.71%** | <font color="green">**+19.05%**</font> |
| | Recall | 91.67% | **95.83%** | <font color="green">**+4.16%**</font> |
| | **Macro F2** | 0.8235 | **0.9211** | <font color="green">**+9.76%**</font> |
| **Chunk Level** | Precision | 41.74% | **57.92%** | <font color="green">**+16.18%**</font> |
| | Recall | 100.0% | **95.83%** | *(Duy trì mức rất cao)* |
| | **Macro F2** | 0.7557 | **0.8368** | <font color="green">**+8.11%**</font> |
| **Combined Score** | **Macro F2** | **0.7896** | **0.8790** | <font color="green">**+8.94% (Đột phá)**</font> |

### Bước 4: Benchmark VRAM trên GPU
```bash
# Kiểm tra bộ nhớ VRAM với BGE-M3 và BGE-Reranker (FP16)
uv run python scripts/test_gpu_memory.py
```
> [!NOTE]  
> Trên GPU **RTX 3050 Laptop (6GB VRAM)**, cả 2 mô hình nạp đồng thời ở chế độ **FP16** chỉ chiếm **~2.13 GB VRAM**, dư thừa hơn **3.5 GB VRAM** cho các tác vụ khác.

---

## 💻 Hướng dẫn Sử dụng CLI (`main.py`)

Hệ thống cung cấp giao diện dòng lệnh đồng nhất qua `main.py`:

### 1. Thu thập dữ liệu từ URL (Tiếng Việt & Tiếng Trung)
```bash
python main.py crawl-urls --input data/raw/urls.jsonl --output data/processed/crawled_articles.jsonl --concurrency 10
```

### 2. Thu thập dữ liệu tiếng Anh từ PubMed (Ngoại tuyến)
```bash
python main.py fetch-pubmed --query "kidney stone treatment" --max-results 100 --output data/processed/pubmed_articles.jsonl
```

### 3. Xây dựng chỉ mục Qdrant Local Engine
Gộp các tài liệu vào `data/processed/all_articles.jsonl`, sau đó chạy:
```bash
python main.py build-index --input data/processed/all_articles.jsonl --output-dir data/indices
```
Chỉ mục sẽ được lưu trực tiếp vào database cục bộ tại `data/indices/qdrant_db`.

### 4. Tìm kiếm thử nghiệm một câu hỏi (End-to-End Search)
```bash
python main.py search "Cần làm gì đối với tình trạng tắc nghẽn đường tiết niệu do sỏi thận?"
```
*Hệ thống sẽ tự động:*
1. Tìm kiếm Hybrid trong **Qdrant** (các bài VI & ZH đã index).
2. Dịch câu hỏi sang tiếng Anh & gọi **PubMed API** lấy các bài báo liên quan (nhánh EN).
3. Gộp ứng viên và chạy **BGE-Reranker** để in ra Top-K tài liệu và đoạn văn bản liên quan nhất từ cả 3 ngôn ngữ.

### 5. Đánh giá nội bộ trên tập Validation (Macro F2 Score)
```bash
python main.py evaluate --predictions outputs/val_predictions.json --ground-truth data/raw/val_groundtruth.json
```

### 6. Sinh file nộp bài chính thức (Leaderboard Submission)
```bash
python main.py generate-submission --queries data/raw/queries.jsonl --name submission.json
```
Lệnh sẽ kiểm tra schema và tạo file ZIP phẳng tại **`outputs/submissions/submission.zip`** sẵn sàng nộp thẳng lên Dashboard cuộc thi!

---

## 📊 Phương pháp Đánh giá (Evaluation Metric)

Hiệu suất được đánh giá ở hai cấp độ: **Document** và **Chunk** bằng thang đo **Macro F2** ($\beta = 2$):

$$F_2 = \frac{5 \times \mathrm{Precision} \times \mathrm{Recall}}{4 \times \mathrm{Precision} + \mathrm{Recall}}$$

- Điểm F2 macro được tính bằng trung bình cộng điểm F2 của tất cả các câu hỏi kiểm thử.
- Trọng số $\beta = 2$ ưu tiên **Recall** cao gấp 2 lần **Precision**.

---

## 📋 Checklist Trước Khi Nộp Bài Lên Dashboard
1. [ ] **Định dạng file ZIP:** Phải là file ZIP phẳng, chỉ chứa **duy nhất 1 file `.json`** (không nằm trong thư mục con).
2. [ ] **Cấu trúc trường:** Trường `id` (int), `relevant_docs` (list string), `relevant_chunks` (list object `{"doc_id": "...", "chunk_text": "..."}`).
3. [ ] **Quy định `doc_id`:** Với VI/ZH là `id` gốc từ BTC; với EN **bắt buộc là mã PMID** của PubMed.
4. [ ] **Quy định `chunk_text`:** Phải là chuỗi trích xuất nguyên bản từ tài liệu gốc, không tự ý viết lại hay sinh mới. Khuyến nghị mỗi chunk không vượt quá ~1.024 tokens.
5. [ ] **Giới hạn số lần nộp:** Tối đa 10 lần/ngày (Public Phase) và 5 lần tổng cộng (Private Phase).

---

## 👥 Làm Việc Nhóm

Xem quy định chi tiết về phân nhánh Git (`feature/*`, `exp/*`), quy tắc chia sẻ dữ liệu và phân chia vai trò trong đội tại **[TEAM_GUIDE.md](TEAM_GUIDE.md)**.