# 🏥 Road to AI 2026 (R2AI) - Multilingual Medical Document Retrieval System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![uv](https://img.shields.io/badge/environment-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Embedding](https://img.shields.io/badge/Embedding-BGE--M3-green.svg)](https://huggingface.co/BAAI/bge-m3)
[![Reranker](https://img.shields.io/badge/Reranker-BGE--Reranker-orange.svg)](https://huggingface.co/BAAI/bge-reranker-large)

Hệ thống truy hồi thông tin y tế đa ngôn ngữ (**Multilingual Medical Document Retrieval System**) phục vụ cuộc thi **Road to AI 2026 (R2AI)**.

Hệ thống giải quyết bài toán "khoảng cách thông tin y tế" xuyên ngôn ngữ:
- **Truy vấn đầu vào:** Tiếng Việt (VI).
- **Kho tri thức truy hồi:** Đa ngôn ngữ (Tiếng Việt - Tiếng Anh - Tiếng Trung).
- **Mục tiêu:** Xác định chính xác tài liệu liên quan (**Document-level**) và trích xuất nguyên vẹn đoạn văn bản chứa thông tin liên quan (**Chunk-level**) theo thang đo **Macro F2** ($\beta = 2$).

---

## 🏗️ Kiến trúc Hệ thống (System Architecture)

```text
               ┌───────────────────────────┐
               │    User Query (VI)        │
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ Query Processing          │
               └─────────────┬─────────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
             ▼                               ▼
  ┌──────────────────────┐       ┌──────────────────────┐
  │ Dense Retrieval      │       │ Sparse Retrieval     │
  │ BGE-M3 + FAISS       │       │ BM25 (Jieba/PyVi)    │
  │ (Semantic Search)    │       │ (Keyword Matching)   │
  └──────────┬───────────┘       └──────────┬───────────┘
             │                              │
             └───────────────┬──────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ Reciprocal Rank Fusion    │
               │ (RRF Coarse Selection)    │
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ Cross-Encoder Re-ranking  │
               │ (BGE-Reranker-Large)      │
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ Formatter & Packager      │
               │ (Official ZIP Submission) │
               └───────────────────────────┘
```

---

## ⚙️ Tech Stack & Tuân thủ Quy chế Cuộc thi

| Thành phần | Công nghệ / Mô hình | Đáp ứng Quy chế (Constraints Compliance) |
| :--- | :--- | :--- |
| **Quản lý Môi trường** | `uv` + Python 3.11 | Cực nhanh, tái lập 100% môi trường (`uv.lock`) |
| **Embedding Model** | `BAAI/bge-m3` | Mã nguồn mở, đa ngôn ngữ VI/EN/ZH, $\le 14B$, phát hành trước 06/2026 |
| **Re-ranker** | `BAAI/bge-reranker-large` / `v2-m3` | Trọng số mở, cross-encoder tối ưu độ nhạy y khoa |
| **Vector Index** | FAISS (`IndexFlatIP`) | Chạy hoàn toàn cục bộ (offline / local), tốc độ cao |
| **Sparse Index** | BM25Okapi (`rank-bm25`) | Tokenizer ngôn ngữ: `pyvi` (VI), `jieba` (ZH) |
| **Crawler & Data** | `trafilatura` + `httpx` + NCBI/Europe PMC | Trích xuất sạch web VI/ZH và tìm kiếm abstracts PubMed |
| **Độ đo đánh giá** | Macro F2 (Doc & Chunk) | Tối ưu hóa cho Recall ($\beta = 2$) theo công thức BTC |

---

## 📂 Cấu trúc Thư mục Dự án

```text
med-doc-retrieval/
├── configs/
│   └── config.yaml             # Cấu hình siêu tham số (chunk size, weights, models, top-k)
├── data/
│   ├── raw/                    # Dữ liệu thô từ BTC (urls.jsonl, queries.jsonl)
│   ├── processed/              # Dữ liệu đã cào & làm sạch (chunks.jsonl, all_articles.jsonl)
│   └── indices/                # Chỉ mục lưu trữ FAISS dense & BM25 sparse
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic Settings & Config Loader
│   ├── crawler/                # Thu thập dữ liệu
│   │   ├── url_scraper.py      # Async scraper cho URL bài báo VI và ZH (Trafilatura)
│   │   └── pubmed.py           # Client tra cứu PubMed / Europe PMC cho EN
│   ├── ingestion/              # Tiền xử lý & cắt đoạn
│   │   ├── cleaner.py          # Chuẩn hóa Unicode NFC & ngôn ngữ
│   │   └── chunker.py          # Chunking bảo toàn nguyên vẹn chuỗi con (doc_id & text)
│   ├── embedding/              # Vector hóa
│   │   └── bge_m3.py           # BGE-M3 Dense Embedder (CUDA/CPU)
│   ├── retrieval/              # Tìm kiếm lai (Hybrid Search)
│   │   ├── dense_index.py      # FAISS Dense Index
│   │   ├── sparse_index.py     # BM25 Sparse Index (Jieba & PyVi tokenization)
│   │   └── hybrid.py           # Reciprocal Rank Fusion (RRF) & Convex Combination
│   ├── reranker/               # Tinh chỉnh xếp hạng
│   │   └── bge_reranker.py     # Cross-Encoder Reranker
│   ├── evaluation/             # Đánh giá nội bộ
│   │   └── metrics.py          # Precision, Recall, Macro F2 (Doc & Chunk level)
│   ├── submission/             # Đóng gói nộp bài
│   │   └── formatter.py        # Validate JSON schema & tạo ZIP nộp bài chuẩn
│   └── pipeline.py             # Điều phối End-to-end Pipeline
├── notebooks/                  # Jupyter notebooks thử nghiệm và EDA
│   └── 01_baseline_exploration.ipynb
├── tests/                      # Unit tests (metrics, chunker, submission, tokenizer)
│   ├── test_metrics.py
│   ├── test_chunker.py
│   ├── test_submission.py
│   └── test_tokenization.py
├── outputs/
│   └── submissions/            # Chứa các file kết quả submission.zip
├── .env.example                # File mẫu biến môi trường (NCBI, HuggingFace)
├── .gitignore                  # Bỏ qua data nặng, indices và checkpoint
├── .python-version             # Khóa Python 3.11
├── pyproject.toml              # Quản lý dependencies với uv
├── main.py                     # CLI entrypoint điều khiển hệ thống
├── TEAM_GUIDE.md               # Hướng dẫn chi tiết cho thành viên làm việc nhóm
├── competition_guide.md        # Hướng dẫn và thể lệ cuộc thi
└── README.md
```

---

## 🚀 Khởi chạy Nhanh (Quickstart)

### 1. Cài đặt môi trường với `uv`
```bash
# Đồng bộ hóa dependencies và kích hoạt môi trường ảo
uv sync --extra dev
source .venv/bin/activate
```

### 2. Thiết lập cấu hình môi trường
```bash
cp .env.example .env
```

### 3. Kiểm tra unit test
```bash
uv run pytest
```

---

## 💻 Hướng dẫn Sử dụng CLI (`main.py`)

Hệ thống cung cấp giao diện dòng lệnh đồng nhất qua `main.py`:

### 📥 Bước 1: Thu thập dữ liệu
- **Cào bài viết từ URL BTC cung cấp (Tiếng Việt & Tiếng Trung):**
  ```bash
  python main.py crawl-urls --input data/raw/urls.jsonl --output data/processed/crawled_articles.jsonl --concurrency 10
  ```
- **Tìm kiếm & tải bài báo PubMed tiếng Anh:**
  ```bash
  python main.py fetch-pubmed --query "kidney stone treatment" --max-results 100 --output data/processed/pubmed_articles.jsonl
  ```

### 🔨 Bước 2: Xây dựng chỉ mục (Index)
Gộp các tài liệu vào `data/processed/all_articles.jsonl`, sau đó chạy:
```bash
python main.py build-index --input data/processed/all_articles.jsonl --output-dir data/indices/
```

### 🔍 Bước 3: Tìm kiếm thử nghiệm một câu hỏi
```bash
python main.py search "Cần làm gì đối với tình trạng tắc nghẽn đường tiết niệu do sỏi thận?"
```

### 📊 Bước 4: Đánh giá mô hình (Macro F2 Score)
```bash
python main.py evaluate --predictions outputs/val_predictions.json --ground-truth data/raw/val_groundtruth.json
```

### 📦 Bước 5: Sinh file nộp bài (Leaderboard Submission)
```bash
python main.py generate-submission --queries data/raw/queries.jsonl --name submission.json
```
Lệnh sẽ kiểm tra schema và tạo file ZIP chuẩn tại `outputs/submissions/submission.zip`. Tải trực tiếp file này lên Dashboard cuộc thi tại http://leaderboard.aiguru.com.vn/.

---

## 👥 Làm việc Nhóm

Xem chi tiết quy trình làm việc nhóm, chia sẻ dữ liệu và phân chia nhánh Git tại **[TEAM_GUIDE.md](TEAM_GUIDE.md)**.