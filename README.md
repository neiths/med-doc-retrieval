# Road to AI 2026 (R2AI) - Team Solution

Repository dành cho giải pháp **Medical Retrieval System (Hệ thống truy hồi y khoa đa ngôn ngữ)** trong khuôn khổ cuộc thi Road to AI 2026. 

Hệ thống được thiết kế để giải quyết bài toán "khoảng cách thông tin y tế" xuyên ngôn ngữ: **Nhận diện truy vấn tiếng Việt** để truy hồi chính xác các tài liệu và đoạn nội dung (chunks) liên quan từ kho dữ liệu đa ngôn ngữ (**Việt - Anh - Trung**).

---

## 🏗️ Tổng quan Kiến trúc Hệ thống

```text
[User Query (VI)] 
       │
       ▼
[1. Query Processing & Expansion] ──> Xử lý thuật ngữ y khoa & từ viết tắt
       │
       ▼
[2. Hybrid Retrieval (Coarse Search)]
       ├──> Dense Retrieval (BGE-M3 Multilingual Vector Search via Qdrant/FAISS)
       └──> Sparse Retrieval (BM25 Keyword Search)
       │
       ▼
[3. Reciprocal Rank Fusion (RRF)] ──> Tổng hợp và trộn kết quả
       │
       ▼
[4. Cross-Encoder Re-ranking (Fine Search)] ──> Sắp xếp lại với BGE-Reranker
       │
       ▼
[Final Output: Top-K Documents & Chunks]
```

## ⚙️ Tech Stack & Constraints Compliance
- Ngôn ngữ: Python 3.10+
- Environment Manager: uv (siêu nhanh, tối ưu hóa quản lý dependency)
- Embedding Model: bge-m3 (Hỗ trợ tối ưu đa ngôn ngữ VI - EN - ZH, kích thước $\le$ 14B, phát hành trước 06/2026)
- Re-ranker: bge-reranker-large
- Vector Database: Qdrant / FAISS (chạy local)


## 📂 Cấu trúc Thư mục Dự án

```text
med-doc-retrieval/
├── data/                  # Thứa mục chứa dữ liệu thô và dữ liệu đã tiền xử lý
├── src/
│   ├── __init__.py
│   ├── ingestion.py       # Pipeline tiền xử lý và chunking tài liệu đa ngôn ngữ
│   ├── embedding.py       # Cấu hình mô hình embedding (BGE-M3)
│   ├── retrieval.py       # Triển khai Hybrid Search (Dense + Sparse + RRF)
│   └── reranker.py        # Tích hợp Cross-Encoder Reranker
├── notebooks/             # Jupyter notebooks dùng để EDA và thử nghiệm
├── outputs/               # Kết quả dự đoán (prediction files) nộp bài
├── .python-version
├── pyproject.toml         # Quản lý dependency bằng uv
├── main.py                # Điểm khởi chạy chính (Entry point)
└── README.md
```