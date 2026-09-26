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
- `data/lexicon/medical_terms.json`: Từ điển y khoa song ngữ VI-EN (hơn 100+ thuật ngữ chuyên khoa, **được lưu trên Git**).
- `configs/pubmed_stopwords.txt`: Danh sách stopwords / filler words cho truy vấn y sinh PubMed (**được lưu trên Git**).
- `data/raw/`: Chứa file ban đầu từ BTC:
  - `urls_vi.jsonl`, `urls_zh.jsonl` (Danh sách URL bài viết VI và ZH)
  - `queries_test.jsonl` (Tập câu hỏi kiểm thử tiếng Việt)
- `data/processed/`:
  - `crawled_articles.jsonl`: Kết quả cào văn bản sạch từ URL (Trafilatura)
  - `pubmed_cache.jsonl`: Cache các bài báo tiếng Anh tải từ Europe PMC / PubTator 3.0 / NCBI
  - `all_articles.jsonl`: Dữ liệu gộp tất cả ngôn ngữ
  - `chunks.jsonl`: Toàn bộ các chunk kèm metadata (doc_id, chunk_text, lang)
- `data/indices/`:
  - `qdrant_db/`: Cơ sở dữ liệu vector Qdrant Local Embedded (lưu cả Dense BGE-M3 + Neural Sparse vectors)

---

## ⚙️ 3. Quy trình Vận hành Pipeline (CLI Workflow)

Mọi thao tác đều có thể chạy qua CLI `python main.py --help`:

### 1. Thu thập dữ liệu từ URL (Tiếng Việt & Tiếng Trung)
```bash
python main.py crawl-urls --input data/raw/urls.jsonl --output data/processed/crawled_articles.jsonl --concurrency 15
```

### 2. Thu thập dữ liệu tiếng Anh từ PubMed / PubTator 3.0
```bash
python main.py fetch-pubmed --query "kidney stone treatment" --max-results 100 --output data/processed/pubmed_articles.jsonl
```

### 3. Gộp dữ liệu & Lập Chỉ Mục vào Qdrant (Chunking -> Embedding BGE-M3 -> Qdrant Local)
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

## 📅 4. Kịch bản Ngày 01/10/2026 (Game Day Runbook)

Vào ngày **01/10/2026**, BTC sẽ công bố danh sách URL bài viết và bộ câu hỏi kiểm thử:

1. **Nhận dữ liệu thô từ BTC**:
   - Tải file URLs tiếng Việt và tiếng Trung vào `data/raw/urls_vi.jsonl` và `data/raw/urls_zh.jsonl`.
   - Tải tập câu hỏi truy vấn vào `data/raw/queries_test.jsonl`.
2. **Cào dữ liệu (Crawl Phase)**:
   ```bash
   python main.py crawl-urls --input data/raw/urls_vi.jsonl --output data/processed/vi_articles.jsonl
   python main.py crawl-urls --input data/raw/urls_zh.jsonl --output data/processed/zh_articles.jsonl
   ```
3. **Lập chỉ mục Qdrant (Indexing Phase)**:
   ```bash
   cat data/processed/vi_articles.jsonl data/processed/zh_articles.jsonl > data/processed/all_articles.jsonl
   python main.py build-index --input data/processed/all_articles.jsonl
   ```
4. **Chạy Pipeline Suy luận Đa ngôn ngữ (Inference Phase)**:
   - Pipeline tự động dịch câu hỏi sang tiếng Anh bằng `opus-mt-vi-en` + `medical_terms.json`.
   - Tìm kiếm bài báo PubMed / PubTator 3.0 / Europe PMC.
   - Tìm kiếm Qdrant Hybrid Search (Dense + Sparse + RRF).
   - Rerank toàn bộ bằng `bge-reranker-large` (FP16).
   ```bash
   python main.py generate-submission --queries data/raw/queries_test.jsonl --name submission_v1.json
   ```
5. **Kiểm tra và nộp bài (Submit Phase)**:
   - Tải file `outputs/submissions/submission_v1.zip` lên mục *My Submissions* tại [AIGuru Leaderboard](http://leaderboard.aiguru.com.vn/).

---

## 🧪 5. Quy chuẩn Thử nghiệm & Siêu tham số (Hyperparameters)

Mọi siêu tham số được đặt tại `configs/config.yaml`:
- `query_translation`:
  - `lexicon_path`: `data/lexicon/medical_terms.json` (từ điển mở rộng)
  - `stopwords_path`: `configs/pubmed_stopwords.txt` (loại bỏ từ nối)
- `pubmed`:
  - `source_api`: `"europe_pmc"`, `"pubtator"`, `"ncbi"`, hoặc `"hybrid"`
  - `max_candidates_per_query`: 30
- `retrieval`:
  - `engine`: `"qdrant"`
  - `dense_top_k`: 50, `sparse_top_k`: 50, `hybrid_top_k`: 30
  - `fusion_method`: `"rrf"` (hệ số `rrf_k: 60`)
- `reranker`:
  - `model_name`: `BAAI/bge-reranker-large` (FP16)
  - `top_k_chunks`: 10
  - `top_k_docs`: 5
  - `score_threshold`: -5.0

---

## 🌿 6. Quy ước Git & Branching

- **`main`**: Nhánh ổn định, luôn chạy được, dùng để nộp bài.
- **`feature/<ten-tinh-nang>`**: Phát triển tính năng mới (ví dụ: `feature/pubtator-integration`, `feature/query-expansion`).
- **`exp/<ten-thu-nghiem>`**: Thử nghiệm model, tuning metric (ví dụ: `exp/bge-reranker-v2`, `exp/rrf-tuning`).
- **Pull Request (PR)**: Tạo PR vào `main` kèm log kết quả metric F2 nếu có cải thiện.

---

## 📋 7. Checklist trước khi nộp bài lên Dashboard
1. [ ] Kiểm tra định dạng JSON: `id` (int), `relevant_docs` (list string), `relevant_chunks` (list object `{"doc_id": "...", "chunk_text": "..."}`).
2. [ ] `chunk_text` là đoạn trích xuất **chính xác** từ văn bản gốc (không sinh mới / không bịa đặt).
3. [ ] `doc_id` tiếng Anh giữ nguyên mã **PMID**.
4. [ ] File ZIP chỉ chứa duy nhất file `.json`, không chứa thư mục lồng nhau.
5. [ ] Số lượng nộp: Tối đa 10 lần/ngày (Public) và 5 lần tổng cộng (Private).
