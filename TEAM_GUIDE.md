# 🤝 Hướng Dẫn Làm Việc Nhóm (Team Collaboration Guide)

Chào mừng các thành viên trong đội tham gia dự án **Road to AI 2026 (R2AI) - Multilingual Medical Document Retrieval**! Tài liệu này quy định quy trình làm việc, cách thiết lập môi trường, chia sẻ dữ liệu và các quy chuẩn thử nghiệm.

---

## 🚀 1. Thiết lập Môi trường Nhanh (Setup)

Dự án sử dụng **[uv](https://github.com/astral-sh/uv)** làm package & environment manager để đảm bảo tốc độ cài đặt tối đa và tính đồng nhất 100% giữa tất cả các máy (Linux, macOS, Windows).

### Bước 1: Cài đặt `uv` (nếu chưa có)
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Bước 2: Clone repository & cài đặt dependencies
```bash
git clone <repo-url>
cd med-doc-retrieval

# Tự động tạo venv với Python 3.11 và cài đặt toàn bộ thư viện (bao gồm dev tools)
uv sync --extra dev
```

### Bước 3: Kích hoạt môi trường
```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Bước 4: Thiết lập biến môi trường
Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
```
*(Điền API key NCBI hoặc Hugging Face token nếu cần).*

---

## 🗂️ 2. Quy ước Quản lý Dữ liệu

> [!WARNING]
> Tuyệt đối **KHÔNG commit file dữ liệu nặng** (`.jsonl`, `.faiss`, `.pkl`, `.zip`, `.pt`, `.safetensors`) lên Git repo. Thư mục `data/` và `outputs/` đã được cấu hình trong `.gitignore`.

### Cấu trúc dữ liệu cục bộ:
- `data/raw/`: Chứa file ban đầu từ BTC:
  - `urls_vi.jsonl`, `urls_zh.jsonl` (Danh sách URL bài viết VI và ZH)
  - `queries_test.jsonl` (Tập câu hỏi kiểm thử tiếng Việt)
- `data/processed/`:
  - `crawled_articles.jsonl`: Kết quả cào văn bản sạch từ URL
  - `pubmed_articles.jsonl`: Dữ liệu tiếng Anh tải từ PubMed/Europe PMC
  - `all_articles.jsonl`: Dữ liệu gộp tất cả ngôn ngữ
  - `chunks.jsonl`: Toàn bộ các chunk kèm metadata
- `data/indices/`:
  - `dense_index.faiss` + `dense_metadata.json`: FAISS vector index
  - `bm25_index.pkl`: BM25 sparse index

---

## ⚙️ 3. Quy trình Vận hành Pipeline (CLI Workflow)

Mọi thao tác đều có thể chạy qua CLI `python main.py --help`:

### 1. Thu thập dữ liệu từ URL (Tiếng Việt & Tiếng Trung)
```bash
python main.py crawl-urls --input data/raw/urls.jsonl --output data/processed/crawled_articles.jsonl --concurrency 15
```

### 2. Thu thập dữ liệu tiếng Anh từ PubMed
```bash
python main.py fetch-pubmed --query "kidney stone treatment" --max-results 100 --output data/processed/pubmed_articles.jsonl
```

### 3. Gộp dữ liệu & Xây dựng Index (Chunking -> Embedding -> FAISS + BM25)
Gộp các file bài báo vào `data/processed/all_articles.jsonl`, sau đó chạy:
```bash
python main.py build-index --input data/processed/all_articles.jsonl --output-dir data/indices/
```

### 4. Thử nghiệm truy vấn nhanh
```bash
python main.py search "Cần làm gì đối với tình trạng tắc nghẽn đường tiết niệu do sỏi thận?"
```

### 5. Đánh giá mô hình trên tập validation cục bộ (Macro F2)
```bash
python main.py evaluate --predictions outputs/val_pred.json --ground-truth data/raw/val_groundtruth.json
```

### 6. Sinh file nộp bài chính thức
```bash
python main.py generate-submission --queries data/raw/test_queries.jsonl --name submission.json
```
File ZIP chuẩn sẽ được tạo tại `outputs/submissions/submission.zip` để nộp thẳng lên Dashboard!

---

## 🧪 4. Quy chuẩn Thử nghiệm & Siêu tham số (Hyperparameters)

Mọi siêu tham số được đặt tại `configs/config.yaml`:
- `chunking`: `max_chunk_size` (mặc định 512), `chunk_overlap` (mặc định 64)
- `retrieval`:
  - `dense_top_k`, `sparse_top_k` (mặc định 50)
  - `fusion_method`: `"rrf"` hoặc `"weighted"`
  - `rrf_k`: hệ số làm mượt (mặc định 60)
  - `dense_weight`, `sparse_weight`: tỷ trọng kết hợp
- `reranker`:
  - `model_name`: `BAAI/bge-reranker-large` hoặc `BAAI/bge-reranker-v2-m3`
  - `top_k_chunks`: số đoạn văn bản chọn lọc cho bài nộp
  - `top_k_docs`: số tài liệu chọn lọc cho bài nộp

**Quy tắc:** Khi thử nghiệm phương pháp mới (prompt expansion, chunking strategy, re-ranker threshold), hãy tạo nhánh Git riêng hoặc cấu hình file yaml mới (ví dụ `configs/exp1_chunk256.yaml`).

---

## 🌿 5. Quy ước Git & Branching

- **`main`**: Nhánh ổn định, luôn chạy được, dùng để nộp bài.
- **`feature/<ten-tinh-nang>`**: Phát triển tính năng mới (ví dụ: `feature/query-expansion`, `feature/pubmed-enrichment`).
- **`exp/<ten-thu-nghiem>`**: Thử nghiệm model, tuning metric (ví dụ: `exp/bge-reranker-v2`, `exp/rrf-tuning`).
- **Pull Request (PR)**: Tạo PR vào `main` kèm log kết quả metric F2 nếu có cải thiện.

---

## 📋 6. Checklist trước khi nộp bài lên Dashboard
1. [ ] Kiểm tra định dạng JSON: `id` (int), `relevant_docs` (list string), `relevant_chunks` (list object `{"doc_id": "...", "chunk_text": "..."}`).
2. [ ] `chunk_text` là đoạn trích xuất **chính xác** từ văn bản gốc (không sinh mới / không bịa đặt).
3. [ ] `doc_id` tiếng Anh giữ nguyên mã **PMID**.
4. [ ] File ZIP chỉ chứa duy nhất file `.json`, không chứa thư mục lồng nhau.
5. [ ] Số lượng nộp: Tối đa 10 lần/ngày (Public) và 5 lần tổng cộng (Private).
