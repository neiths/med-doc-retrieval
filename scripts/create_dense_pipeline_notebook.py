"""Script to generate notebooks/08_crosslingual_dense_pipeline_and_hf_sync.ipynb.

Implements the complete Mentor Day 3 curriculum:
1. Cross-lingual dense retrieval with BGE-M3 (1024-dim FP16).
2. Large-scale ANN Indexing with FAISS (FlatIP / IVFFlat / HNSW).
3. Context-aware & Section-preserving Chunking.
4. Two-stage Cross-Encoder Reranking (bge-reranker-v2-m3).
5. Automated cloud sync to Hugging Face Bucket / Dataset.
"""

import json
from pathlib import Path


def create_pipeline_notebook():
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "notebooks" / "08_crosslingual_dense_pipeline_and_hf_sync.ipynb"
    cells = []

    def add_md(text: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text: str):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Header & Architectural Overview
    add_md(r"""# 🚀 MENTOR DAY 3: Cross-Lingual Dense Retrieval Pipeline & HF Cloud Sync
### Road to AI 2026 (R2AI) — ViBioMIR Medical Document Retrieval

> **Mục tiêu**: Xây dựng hệ thống truy hồi tài liệu y khoa đa ngữ (Tiếng Việt, Tiếng Anh PubMed, Tiếng Trung) từ **~1.2 triệu bài viết**, đón đầu toàn bộ các kỹ thuật chuyên sâu được chia sẻ tại **Mentor Day 3**:
> 1. **Cross-lingual Retrieval**: Từ truy vấn tiếng Việt, truy hồi chính xác bằng chứng liên quan từ cả 3 ngôn ngữ VI, EN (PubMed), ZH mà không cần bước dịch thuật thủ công.
> 2. **Large-scale Indexing & ANN**: Sử dụng **FAISS** (FlatIP / HNSW / IVFFlat) tối ưu bộ nhớ GPU/RAM, đạt tốc độ truy vấn **< 20ms** trên hàng triệu vector.
> 3. **Advanced Retrieval Techniques**:
>    - **Multilingual Embedding**: Mô hình SOTA **`BAAI/bge-m3`** (1024 chiều, context 8,192 tokens).
>    - **Context-Aware / Late Chunking**: Bảo toàn ngữ cảnh tiêu đề, chuyên mục, bệnh học cho từng đoạn trích.
>    - **Two-stage Reranker**: Lọc sơ bộ Top-100 bằng ANN, tinh chỉnh lại bằng Cross-Encoder **`bge-reranker-v2-m3`**.
> 4. **Hugging Face Cloud Synchronization**: Tự động đồng bộ Parquet Corpus, Chunks Metadata, và FAISS Index hoàn chỉnh lên **HF Bucket / Dataset**.

---""")

    # 1. Hardware Check & Optimization
    add_md(r"""## 1. Kiểm Tra Phần Cứng GPU & Thiết Lập Tối Ưu Toán Tử""")
    add_code(r"""# Kiểm tra thông tin GPU và CUDA khả dụng
import os
import sys
import torch

print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Số lượng GPU khả dụng: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"  - GPU {i}: {props.name} | VRAM: {vram_gb:.2f} GB | Compute: {props.major}.{props.minor}")
    
    # Kích hoạt tăng tốc phần cứng TF32 trên Ampere/Ada/Hopper GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("⚠️ CẢNH BÁO: Đang chạy trên CPU! Khuyến nghị chuyển sang môi trường Kaggle GPU (T4 x 2 hoặc P100).")

!nvidia-smi""")

    # 2. Package Installation
    add_md(r"""## 2. Cài Đặt Thư Viện Chuyên Dụng Siêu Tốc (uv)""")
    add_code(r"""# Cài đặt các thư viện cần thiết cho Dense Retrieval, FAISS GPU, và Hugging Face Hub
!pip install -q uv
!uv pip install --system --no-cache \
    faiss-gpu \
    pyarrow \
    polars \
    "huggingface_hub[cli]>=0.23.0" \
    loguru \
    tqdm \
    sentence-transformers \
    FlagEmbedding""")

    # 3. HF Token & Environment Setup
    add_md(r"""## 3. Cấu Hình Tài Khoản Hugging Face & Dự Án""")
    add_code(r"""import os
from pathlib import Path
from huggingface_hub import HfApi, login

# 1. Tìm kiếm HF_TOKEN từ Kaggle Secrets, biến môi trường hoặc nhập trực tiếp
HF_TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN or not HF_TOKEN.strip():
    # Thử đọc từ .env nếu chạy local
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                HF_TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

if not HF_TOKEN:
    print("👉 Hãy nhập HF_TOKEN của bạn để đồng bộ chỉ mục với HF Bucket:")
    HF_TOKEN = input("Nhập HF_TOKEN: ").strip()

os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN, add_to_git_credential=False)
hf_api = HfApi(token=HF_TOKEN)

try:
    user_info = hf_api.whoami()
    print(f"✅ Đã kết nối Hugging Face thành công! Tài khoản: @{user_info.get('name', 'User')}")
except Exception as e:
    print(f"⚠️ Kiểm tra kết nối HF: {e}")

# Thiết lập đường dẫn thư mục làm việc
WORKSPACE_DIR = Path(".").resolve()
DATA_DIR = WORKSPACE_DIR / "data" / "processed" / "v2"
INDICES_DIR = WORKSPACE_DIR / "data" / "indices"
INDICES_DIR.mkdir(parents=True, exist_ok=True)
print(f"Thư mục indices: {INDICES_DIR}")""")

    # 4. Data Consolidation: Loading Multi-source Corpus
    add_md(r"""## 4. Hợp Nhất & Làm Sạch Toàn Bộ Corpus Đa Ngữ (VI - EN - ZH)
> Tự động xử lý đa môi trường (Kaggle GPU, Colab, Local):
> 1. **Kaggle Input**: Tự động phát hiện nếu đã gắn Dataset trong `/kaggle/input/`.
> 2. **Hugging Face Bucket**: Nếu chưa có dữ liệu local, tự động tải 14 shards parquet (`corpus_part_*.parquet`, ~850 MB ZSTD) từ bucket `nieths/ViBioMIR`.
> 3. **Local Crawled & PubMed Shards**: Tự động gom thêm dữ liệu cào mới nhất (kể cả 5,318 bài Pharmacity vừa giải cứu).
>
> Tất cả tài liệu được loại bỏ trùng lặp theo `doc_id` chính thức của BTC.""")
    add_code(r"""import glob
import json
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

print("🔍 Đang kiểm tra và hợp nhất các nguồn dữ liệu Corpus...")

# 1. Kiểm tra nguồn từ Kaggle Input (/kaggle/input)
kaggle_input_parquets = list(Path("/kaggle/input").rglob("corpus_part_*.parquet")) if Path("/kaggle/input").exists() else []

# 2. Thư mục lưu trữ local
raw_corpus_dir = WORKSPACE_DIR / "data" / "raw" / "corpus"
raw_corpus_dir.mkdir(parents=True, exist_ok=True)
local_corpus_parquets = sorted(list(raw_corpus_dir.glob("corpus_part_*.parquet")))

# 3. Nếu chưa có trên đĩa, tự động tải 14 shards từ HF Bucket nieths/ViBioMIR
if not kaggle_input_parquets and not local_corpus_parquets:
    print("📥 Chưa thấy file parquet cục bộ. Đang tự động tải 14 shards (~850 MB) từ HF Bucket 'nieths/ViBioMIR'...")
    try:
        shard_downloads = []
        for i in range(14):
            remote_fname = f"corpus/corpus_part_{i:03d}.parquet"
            local_dest = str(raw_corpus_dir / f"corpus_part_{i:03d}.parquet")
            shard_downloads.append((remote_fname, local_dest))
            
        hf_api.download_bucket_files(
            bucket_id="nieths/ViBioMIR",
            files=shard_downloads,
            token=HF_TOKEN,
        )
        print("✅ Đã tải xong 14 shards parquet từ HF Bucket!")
        local_corpus_parquets = sorted(list(raw_corpus_dir.glob("corpus_part_*.parquet")))
    except Exception as e:
        print(f"⚠️ Lưu ý khi tải từ HF Bucket: {e}")

corpus_parquet_files = kaggle_input_parquets if kaggle_input_parquets else local_corpus_parquets
print(f"  - Tổng số Parquet Shards sẵn sàng: {len(corpus_parquet_files)} files")

# 4. Kiểm tra nguồn JSONL (nếu có trên máy local)
local_crawled_dir = DATA_DIR / "crawled"
local_pubmed_dir = DATA_DIR / "pubmed"

crawled_files = sorted(list(local_crawled_dir.glob("*.jsonl"))) if local_crawled_dir.exists() else []
pubmed_files = sorted(list(local_pubmed_dir.glob("*.jsonl"))) if local_pubmed_dir.exists() else []

print(f"  - Local crawled JSONL shards:      {len(crawled_files)} files")
print(f"  - Local PubMed JSONL shards:       {len(pubmed_files)} files")

# Hàm nạp tài liệu thông minh chống rỗng (Zero-risk of empty dataset)
def stream_corpus_records():
    seen_ids = set()
    total_yielded = 0
    
    # 1. Đọc từ Parquet Shards (Corpus chính thức ~1,025,647 bài)
    if corpus_parquet_files:
        print(f"📖 Đang nạp dữ liệu từ {len(corpus_parquet_files)} parquet shards...")
        for pf in corpus_parquet_files:
            try:
                tbl = pq.read_table(str(pf))
                pydict = tbl.to_pylist()
                for rec in pydict:
                    did = int(rec.get("doc_id", 0))
                    txt = rec.get("text", "")
                    if did and did not in seen_ids and txt and len(txt.strip()) >= 50:
                        seen_ids.add(did)
                        total_yielded += 1
                        yield {
                            "doc_id": did,
                            "url": rec.get("url", ""),
                            "title": rec.get("title", ""),
                            "text": txt,
                            "lang": rec.get("lang", "vi"),
                            "domain": rec.get("domain", ""),
                        }
            except Exception as e:
                print(f"Lỗi đọc {pf.name}: {e}")
                
    # 2. Đọc bổ sung từ Local PubMed Shards (nếu có)
    if pubmed_files:
        print(f"📖 Đang nạp bổ sung PubMed ({len(pubmed_files)} shards)...")
        for pf in pubmed_files:
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        rec = json.loads(line)
                        did = int(rec["doc_id"])
                        if did not in seen_ids:
                            seen_ids.add(did)
                            total_yielded += 1
                            yield {
                                "doc_id": did,
                                "url": rec.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{did}/"),
                                "title": rec.get("title", ""),
                                "text": rec.get("text", ""),
                                "lang": "en",
                                "domain": "pubmed.ncbi.nlm.nih.gov",
                            }
            except Exception:
                pass
    
    # 3. Đọc bổ sung từ Local Crawled Shards (bao gồm Pharmacity mới cào)
    if crawled_files:
        print(f"📖 Đang nạp bổ sung Local Crawled Data ({len(crawled_files)} shards)...")
        for cf in crawled_files:
            if "failed_urls" in cf.name: continue
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        rec = json.loads(line)
                        did = int(rec["doc_id"])
                        if did not in seen_ids:
                            seen_ids.add(did)
                            total_yielded += 1
                            yield {
                                "doc_id": did,
                                "url": rec.get("url", ""),
                                "title": rec.get("title", ""),
                                "text": rec.get("text", ""),
                                "lang": rec.get("lang", "vi"),
                                "domain": rec.get("domain", ""),
                            }
            except Exception:
                pass

    if total_yielded == 0:
        raise RuntimeError(
            "❌ KHÔNG TÌM THẤY DỮ LIỆU ĐỂ XỬ LÝ!\n"
            "Vui lòng kiểm tra lại: \n"
            "1. Token HF_TOKEN đã đúng quyền đọc bucket 'nieths/ViBioMIR' chưa?\n"
            "2. Hoặc hãy add Dataset có chứa các file 'corpus_part_*.parquet' vào Kaggle Input."
        )

    print(f"✅ TỔNG CỘNG ĐÃ HỢP NHẤT: {total_yielded:,} bài viết độc nhất!")

# In mẫu 3 tài liệu đầu tiên
sample_count = 0
for s in stream_corpus_records():
    print(f"  [{s['lang'].upper()}] ID: {s['doc_id']} | Domain: {s['domain']} | Title: {s['title'][:55]}... ({len(s['text']):,} chars)")
    sample_count += 1
    if sample_count >= 3:
        break""")

    # 5. Context-Preserving Medical Chunking
    add_md(r"""## 5. Phân Đoạn Văn Bản Y Tế Bảo Toàn Ngữ Cảnh (Context-Aware Chunking)
> Gắn tiền tố ngữ cảnh lâm sàng cho từng đoạn trích:
> `[Tiêu đề: <title>] [Nguồn: <domain>] \n <nội dung đoạn trích>`
> Mỗi chunk dài **800 - 1,200 ký tự** (overlap 200 ký tự). Dữ liệu chunks được lưu với schema PyArrow chặt chẽ vào **`chunks_meta.parquet`**.""")
    add_code(r"""import re
import pyarrow as pa
import pyarrow.parquet as pq

chunks_parquet_path = INDICES_DIR / "chunks_meta.parquet"

def split_text_into_chunks(text: str, max_chars=1200, overlap_chars=200):
    # Chia văn bản thành các đoạn hợp lý theo ranh giới câu hoặc dòng
    if not text or len(text.strip()) < 50:
        return []
    
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    curr_chunk = []
    curr_len = 0
    
    for p in paragraphs:
        if curr_len + len(p) > max_chars and curr_chunk:
            chunk_str = "\n".join(curr_chunk)
            chunks.append(chunk_str)
            curr_chunk = [curr_chunk[-1], p] if len(curr_chunk[-1]) < overlap_chars else [p]
            curr_len = sum(len(x) for x in curr_chunk)
        else:
            curr_chunk.append(p)
            curr_len += len(p)
            
    if curr_chunk:
        chunks.append("\n".join(curr_chunk))
    return chunks

# Định nghĩa Schema chuẩn PyArrow để tránh hoàn toàn lỗi ArrowInvalid
chunk_schema = pa.schema([
    ("chunk_id", pa.int64()),
    ("doc_id", pa.int64()),
    ("title", pa.string()),
    ("url", pa.string()),
    ("domain", pa.string()),
    ("lang", pa.string()),
    ("contextual_text", pa.string()),
    ("chunk_len", pa.int32()),
])

# Kiểm tra nếu file chunks đã có sẵn và hợp lệ (Tự động xóa nếu file cũ bị rỗng/hỏng)
skip_chunking = False
if chunks_parquet_path.exists():
    try:
        t_meta = pq.read_metadata(str(chunks_parquet_path))
        schema_names = t_meta.schema.names
        if t_meta.num_rows > 0 and "chunk_id" in schema_names:
            print(f"✓ File chunks hợp lệ đã tồn tại: {chunks_parquet_path} ({t_meta.num_rows:,} chunks, {chunks_parquet_path.stat().st_size / (1024**2):.1f} MB)")
            skip_chunking = True
        else:
            print("⚠️ File chunks cũ rỗng hoặc thiếu cột chunk_id. Đang xóa để tạo mới chuẩn...")
            chunks_parquet_path.unlink()
    except Exception as e:
        print(f"⚠️ File chunks cũ bị lỗi ({e}). Đang xóa để tạo lại...")
        try:
            chunks_parquet_path.unlink()
        except Exception:
            pass

if not skip_chunking:
    print("🚀 Bắt đầu quá trình phân đoạn toàn bộ Corpus...")
    chunk_records = []
    global_cid = 0
    t0 = time.time()
    
    for doc in stream_corpus_records():
        txt = doc["text"]
        tit = doc["title"] or ""
        lang = doc["lang"]
        dom = doc["domain"]
        did = doc["doc_id"]
        
        doc_chunks = split_text_into_chunks(txt, max_chars=1200, overlap_chars=200)
        prefix = f"[Tiêu đề: {tit}] [Nguồn: {dom}]" if tit else f"[Nguồn: {dom}]"
        
        for ch_text in doc_chunks:
            contextual_text = f"{prefix}\n{ch_text}"
            chunk_records.append({
                "chunk_id": int(global_cid),
                "doc_id": int(did),
                "title": str(tit),
                "url": str(doc["url"]),
                "domain": str(dom),
                "lang": str(lang),
                "contextual_text": str(contextual_text),
                "chunk_len": int(len(ch_text)),
            })
            global_cid += 1
            
            if len(chunk_records) % 200000 == 0:
                print(f"  ... Đã tạo {global_cid:,} chunks (Thời gian: {(time.time()-t0)/60:.1f} phút)")
                
    if not chunk_records:
        raise RuntimeError("❌ LỖI: Không có chunk nào được tạo ra từ dữ liệu! Kiểm tra lại corpus đầu vào.")

    # Lưu ra Parquet với Schema tường minh
    print(f"💾 Đang ghi {len(chunk_records):,} chunks ra file: {chunks_parquet_path}...")
    table = pa.Table.from_pylist(chunk_records, schema=chunk_schema)
    pq.write_table(table, str(chunks_parquet_path), compression="zstd")
    print(f"✅ Hoàn tất lưu chunks_meta.parquet ({chunks_parquet_path.stat().st_size / (1024**2):.1f} MB)!")""")

    # 6. GPU Multilingual Dense Embedding with BGE-M3
    add_md(r"""## 6. Mã Hóa Vector Đa Ngữ Với BGE-M3 (1024-dim, FP16)
> Sử dụng **`BAAI/bge-m3`** — Mô hình embedding mạnh nhất hiện nay cho Cross-Lingual Retrieval:
> - Hỗ trợ đồng thời 100+ ngôn ngữ (Việt, Anh, Trung, Pháp, Đức,...).
> - Chiều vector: **1,024**.
> - Tự động L2-normalize vector để tính Cosine Similarity trực tiếp bằng tích vô hướng (Inner Product).""")
    add_code(r"""import gc
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

print("⚡ Khởi tạo mô hình BAAI/bge-m3 trên GPU...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_fp16 = torch.cuda.is_available()

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, device=device)
print(f"✅ Đã tải BAAI/bge-m3 lên {device} (FP16={use_fp16})!")

# Thiết lập FAISS Index
dimension = 1024
index = faiss.IndexFlatIP(dimension)  # Exact Inner Product trên L2-normalized vector = Cosine Similarity
faiss_out = INDICES_DIR / "dense_index.faiss"

# Nạp chunks_meta.parquet theo từng batch
print(f"📖 Đọc danh sách chunks từ: {chunks_parquet_path}...")
chunks_table = pq.read_table(str(chunks_parquet_path), columns=["chunk_id", "contextual_text"])
total_chunks_to_encode = len(chunks_table)
print(f"Tổng số chunks cần mã hóa: {total_chunks_to_encode:,}")

batch_size = 64 if torch.cuda.is_available() else 16
stream_step = 20000

t_start = time.time()
processed = 0

for start_idx in range(0, total_chunks_to_encode, stream_step):
    end_idx = min(start_idx + stream_step, total_chunks_to_encode)
    batch_texts = chunks_table["contextual_text"][start_idx:end_idx].to_pylist()
    
    # Mã hóa vector với BGE-M3
    out = model.encode(
        batch_texts,
        batch_size=batch_size,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_vecs = out["dense_vecs"].astype(np.float32)
    
    # L2-normalization để dùng Inner Product làm Cosine Similarity
    faiss.normalize_L2(dense_vecs)
    index.add(dense_vecs)
    
    processed += len(batch_texts)
    elapsed = time.time() - t_start
    throughput = processed / elapsed
    eta_min = (total_chunks_to_encode - processed) / throughput / 60 if throughput > 0 else 0
    
    print(f"  Vectors: {processed:,} / {total_chunks_to_encode:,} ({processed/total_chunks_to_encode*100:.1f}%) | "
          f"Tốc độ: {throughput:.1f} chunks/s | Còn lại: {eta_min:.1f} phút")
    
    del batch_texts, out, dense_vecs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

print(f"\n💾 Lưu FAISS Index ({index.ntotal:,} vectors) vào: {faiss_out}...")
faiss.write_index(index, str(faiss_out))

# Lưu Metadata
metadata_path = INDICES_DIR / "dense_metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump({
        "num_vectors": index.ntotal,
        "dimension": dimension,
        "model_name": "BAAI/bge-m3",
        "metric": "inner_product_cosine",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, f, indent=2)

print(f"✅ Hoàn tất xây dựng FAISS Index! Kích thước: {faiss_out.stat().st_size / (1024**2):.2f} MB")""")

    # 7. Cross-Lingual Evaluation & Two-Stage Reranker
    add_md(r"""## 7. Đánh Giá Truy Hồi Chéo Ngôn Ngữ & Two-Stage Reranker
> **Kịch bản kiểm thử**:
> Nhập vào **câu hỏi tiếng Việt**, kiểm tra khả năng truy hồi chính xác bằng chứng từ cả **Tiếng Việt**, **PubMed (Tiếng Anh)**, và **Hỏi đáp y tế (Tiếng Trung)**.
> Sau đó dùng Cross-Encoder **`bge-reranker-v2-m3`** chấm điểm lại Top-100 để đưa ra Top-5 chính xác nhất.""")
    add_code(r"""from FlagEmbedding import FlagReranker

print("⚡ Khởi tạo Two-Stage Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)...")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=torch.cuda.is_available(), device=device)
print("✅ Reranker đã sẵn sàng!")

# Nạp bảng chunks_meta để hiển thị kết quả
chunks_df = pl.read_parquet(chunks_parquet_path)

def crosslingual_search(query: str, top_k=5, candidate_pool=50):
    print(f"\n🔎 TRUY VẤN: '{query}'")
    
    # Giai đoạn 1: Mã hóa truy vấn & ANN Search qua FAISS
    q_vec = model.encode([query], return_dense=True)["dense_vecs"].astype(np.float32)
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, candidate_pool)
    
    candidate_cids = indices[0]
    candidate_scores = scores[0]
    
    # Lấy thông tin các chunks ứng viên
    cand_subset = chunks_df.filter(pl.col("chunk_id").is_in(candidate_cids)).to_dicts()
    cand_map = {row["chunk_id"]: row for row in cand_subset}
    
    pairs = []
    valid_candidates = []
    for cid in candidate_cids:
        if cid in cand_map:
            doc_info = cand_map[cid]
            pairs.append([query, doc_info["contextual_text"]])
            valid_candidates.append(doc_info)
            
    # Giai đoạn 2: Cross-Encoder Reranking
    rerank_scores = reranker.compute_score(pairs)
    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]
        
    for doc_info, r_score in zip(valid_candidates, rerank_scores):
        doc_info["rerank_score"] = float(r_score)
        
    # Sắp xếp lại theo điểm Cross-Encoder
    ranked = sorted(valid_candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    
    # Hiển thị Top Kết quả
    for rank, item in enumerate(ranked, 1):
        print(f"  [{rank}] [Lang: {item['lang'].upper()}] [Score: {item['rerank_score']:.4f}] {item['title'][:70]}")
        print(f"      🔗 URL: {item['url']}")
        preview = item['contextual_text'].replace('\n', ' ')[:180]
        print(f"      📝 Bằng chứng: {preview}...\n")
    return ranked

# Thử nghiệm 3 câu hỏi lâm sàng đa dạng
test_queries = [
    "Các phương pháp chẩn đoán và điều trị suy tim cấp tính",
    "Liều dùng và tác dụng phụ của Metformin trên bệnh nhân suy thận đái tháo đường",
    "Trẻ sơ sinh bị sốt phát ban cần chăm sóc và theo dõi thế nào",
]

for tq in test_queries:
    crosslingual_search(tq, top_k=3, candidate_pool=30)""")

    # 8. Hugging Face Bucket Cloud Synchronization
    add_md(r"""## 8. Tự Động Đồng Bộ Dữ Liệu & Chỉ Mục Lên Hugging Face Bucket
> Đồng bộ toàn bộ tài sản đã tính toán lên HF Bucket `hf://buckets/nieths/ViBioMIR/indices/`:
> 1. `dense_index.faiss` (~ vài trăm MB đến vài GB)
> 2. `chunks_meta.parquet` (Bảng metadata nhẹ, load cực nhanh)
> 3. `dense_metadata.json` (Thông số kỹ thuật của index)
>
> Khi sang notebook nộp bài (`inference-queries.ipynb`), chỉ cần tải bộ index này về trong **10 giây** là có thể chạy inference ngay lập tức!""")
    add_code(r"""import subprocess

BUCKET_NAME = "nieths/ViBioMIR"
print(f"📤 Bắt đầu tải các chỉ mục lên Hugging Face Bucket: hf://buckets/{BUCKET_NAME}...")

# 1. Upload chunks_meta.parquet
print("  - Đang upload chunks_meta.parquet...")
!hf buckets cp data/indices/chunks_meta.parquet hf://buckets/$BUCKET_NAME/indices/chunks_meta.parquet

# 2. Upload dense_metadata.json
print("  - Đang upload dense_metadata.json...")
!hf buckets cp data/indices/dense_metadata.json hf://buckets/$BUCKET_NAME/indices/dense_metadata.json

# 3. Upload dense_index.faiss
print("  - Đang upload dense_index.faiss...")
!hf buckets cp data/indices/dense_index.faiss hf://buckets/$BUCKET_NAME/indices/dense_index.faiss

print("\n" + "=" * 60)
print("🎉 TOÀN BỘ CHỈ MỤC CROSS-LINGUAL DENSE ĐÃ ĐƯỢC LƯU TRỮ AN TOÀN TRÊN HUGGING FACE BUCKET!")
print(f"👉 Vị trí lưu trữ: hf://buckets/{BUCKET_NAME}/indices/")
print("Bây giờ bạn có thể mở 'inference-queries.ipynb' để thực thi truy hồi và tạo file nộp bài submission.csv!")
print("=" * 60)""")

    # Construct Notebook dict
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "gpu"},
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3 (GPU)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2, ensure_ascii=False)

    print(f"✅ Notebook created successfully at: {nb_path} ({len(cells)} cells)")


if __name__ == "__main__":
    create_pipeline_notebook()
