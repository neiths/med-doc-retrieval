# 🏥 Road to AI 2026 (R2AI) - Multilingual Medical Document Retrieval System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![uv](https://img.shields.io/badge/environment-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Vector DB](https://img.shields.io/badge/Vector_DB-Qdrant_Local-red.svg)](https://qdrant.tech/)
[![Embedding](https://img.shields.io/badge/Embedding-BGE--M3_(FP16)-green.svg)](https://huggingface.co/BAAI/bge-m3)
[![Reranker](https://img.shields.io/badge/Reranker-BGE--Reranker_(FP16)-orange.svg)](https://huggingface.co/BAAI/bge-reranker-large)

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
│ └──> BGE-M3 Dense & Sparse Encode │             │ └──> PubMed / PubTator / Europe PMC│
│ └──> Qdrant Local Engine Storage  │             │ └──> Fetch Abstracts & On-the-fly │
└─────────────────┬─────────────────┘             └─────────────────┬─────────────────┘
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
                                           │
                                           ▼
                           [Submission Validator & Packager]
                           (Tự động tạo file ZIP chuẩn Leaderboard)
```

---

## ⚙️ Tech Stack & Tuân thủ Quy chế Cuộc thi (Constraints Compliance)

| Thành phần | Công nghệ / Mô hình | Đáp ứng Quy chế (Rules & Constraints) |
| :--- | :--- | :--- |
| **Quản lý Môi trường** | `uv` + Python 3.11 | Tối ưu hóa cài đặt cực nhanh, đồng bộ 100% qua `uv.lock`. |
| **Vector Database** | **Qdrant (Local Embedded)** | Lưu trữ nhúng tại `data/indices/qdrant_db`, **không cần Docker**, hỗ trợ Native Hybrid Search (Dense + Sparse) & RRF trực tiếp ở tầng engine. |
| **Embedding Model** | `BAAI/bge-m3` (chế độ **FP16**) | Đa ngôn ngữ VI-EN-ZH, 1024 chiều, $\le 14B$ tham số, phát hành trước 06/2026. |
| **Re-ranker** | `BAAI/bge-reranker-large` (chế độ **FP16**) | Cross-Encoder chấm điểm tương quan ngữ nghĩa trực tiếp giữa câu hỏi VI và chunk đa ngôn ngữ. |
| **Query Translator** | `Helsinki-NLP/opus-mt-vi-en` + Bilingual Lexicon | Mô hình dịch mở ~289MB kết hợp `data/lexicon/medical_terms.json` và `configs/pubmed_stopwords.txt`. |
| **External Medical API** | PubTator 3.0, Europe PMC & NCBI Entrez | Tìm kiếm bài báo PubMed theo từ khóa, tải BiocJSON/XML và lưu cache tự động tại `pubmed_cache.jsonl`. |
| **Độ đo đánh giá** | Macro F2 (beta = 2.0) | Ưu tiên Recall gấp 2 lần Precision theo đúng công thức BTC. |

---

## 📂 Cấu trúc Thư mục Dự án

```text
med-doc-retrieval/
├── configs/
│   ├── config.yaml             # Cấu hình siêu tham số (Qdrant, FP16, chunk size, top-k, weights)
│   └── pubmed_stopwords.txt    # Danh sách stopwords / filler words khi tìm kiếm y sinh PubMed
├── data/
│   ├── lexicon/
│   │   └── medical_terms.json  # Từ điển y khoa song ngữ VI-EN mở rộng (>100 thuật ngữ, lưu trên Git)
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
│   │   └── chunker.py          # Phân đoạn bảo toàn nguyên vẹn chuỗi con & doc_id
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
│   │   └── metrics.py          # Precision, Recall, Macro F2 (Doc & Chunk levels)
│   ├── submission/             # Đóng gói nộp bài
│   │   └── formatter.py        # Schema validator & tự động nén ZIP phẳng
│   └── pipeline.py             # Điều phối End-to-end Pipeline
├── notebooks/
│   └── 01_baseline_exploration.ipynb # Notebook mẫu thử nghiệm từng thành phần
├── scripts/
│   └── test_gpu_memory.py      # Script stress test VRAM trên GPU (RTX 3050 6GB)
├── tests/                      # Bộ kiểm thử tự động (14/14 tests passing)
│   ├── test_chunker.py
│   ├── test_metrics.py
│   ├── test_qdrant.py
│   ├── test_query_translator.py
│   ├── test_submission.py
│   └── test_tokenization.py
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
# Đảm bảo 14 bài kiểm thử đều PASS
uv run pytest
```

### Bước 3: Benchmark VRAM trên GPU
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
4. [ ] **Quy định `chunk_text`:** Phải là chuỗi trích xuất nguyên bản từ tài liệu gốc, không tự ý viết lại hay sinh mới.
5. [ ] **Giới hạn số lần nộp:** Tối đa 10 lần/ngày (Public Phase) và 5 lần tổng cộng (Private Phase).

---

## 👥 Làm Việc Nhóm

Xem quy định chi tiết về phân nhánh Git (`feature/*`, `exp/*`), quy tắc chia sẻ dữ liệu và phân chia vai trò trong đội tại **[TEAM_GUIDE.md](file:///home/thienhb/Workspace/med-doc-retrieval/TEAM_GUIDE.md)**.