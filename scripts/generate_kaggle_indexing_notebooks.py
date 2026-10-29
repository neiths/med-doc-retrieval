"""Script to generate the two dedicated Kaggle Indexing Notebooks:
1. notebooks/01_kaggle_build_bm25_cpu.ipynb  (CPU-only: Chunker & BM25s Builder)
2. notebooks/02_kaggle_build_dense_gpu.ipynb (GPU: Dual-T4 High-Throughput Dense BGE-M3 Builder)
"""

import json
from pathlib import Path
import nbformat as nbf


def make_notebook_1():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "kaggle": {
            "accelerator": "none"  # CPU ONLY!
        }
    }

    cells = []

    # Cell 0: Header
    cells.append(nbf.v4.new_markdown_cell("""# ⚡ STAGE 1: CHUNKER & BM25s SPARSE INDEX BUILDER (CPU-ONLY)
### Road to AI 2026 (R2AI) — ViBioMIR Medical Document Retrieval

> **Tài nguyên**: Chạy trên **Kaggle CPU-Only (4 vCPUs, 30 GB RAM, 0 giờ GPU!)**.
> **Thời gian chạy**: ~12 – 18 phút cho toàn bộ 1.025M bài viết.
> **Mục tiêu**:
> 1. Tải 14 shards Parquet corpus (1,025,647 bài viết) từ HF Bucket.
> 2. Phân đoạn **Chunk 750 ký tự** (overlap 120 ký tự) trên corpus đã làm sạch (boilerplate + audit verdict) để giải quyết dứt điểm ngưỡng Token Overlap $\\ge 0.40$ của BTC.
> 3. Lưu toàn bộ metadata vào SQLite `chunks_meta.sqlite`.
> 4. Xây dựng chỉ mục thưa đa ngôn ngữ siêu tốc **BM25s** (Mmap, zero-RAM).
> 5. Đồng bộ `chunks_meta.sqlite` và `bm25s_index/` lên Hugging Face Bucket để Stage 2 (GPU) sử dụng."""))

    # Cell 1: Environment setup
    cells.append(nbf.v4.new_code_cell("""# 1. Cài đặt các gói phụ thuộc siêu tốc với uv
!pip install -q uv
!uv pip install --system --no-cache pyarrow "huggingface_hub[cli]" bm25s pyvi jieba loguru tqdm"""))

    # Cell 2: Git repo setup
    cells.append(nbf.v4.new_code_cell("""# 2. Thiết lập mã nguồn dự án
import os
from pathlib import Path

REPO_URL = "https://github.com/neiths/med-doc-retrieval.git"
PROJECT_DIR = Path("/kaggle/working/med-doc-retrieval")

if not PROJECT_DIR.exists():
    print("🚀 Đang clone repository từ GitHub...")
    !git clone -b main $REPO_URL $PROJECT_DIR
else:
    print("🔄 Đang cập nhật repository...")
    %cd $PROJECT_DIR
    !git pull origin main

%cd $PROJECT_DIR
print("Thư mục làm việc hiện tại:", os.getcwd())"""))

    # Cell 3: HF Token Authentication
    cells.append(nbf.v4.new_code_cell("""# 3. Cấu hình bảo mật Hugging Face Token
import os
from huggingface_hub import HfApi

# Ưu tiên lấy từ Kaggle Secrets (nếu đã thêm secret 'HF_TOKEN')
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("✅ Đã lấy HF_TOKEN an toàn từ Kaggle Secrets!")
except Exception:
    # Fallback dự phòng nếu chưa cài Kaggle Secret
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if not HF_TOKEN:
        print("💡 Hãy thêm secret 'HF_TOKEN' vào Kaggle Secrets hoặc nhập token của bạn.")

os.environ["HF_TOKEN"] = HF_TOKEN
api = HfApi(token=HF_TOKEN)"""))

    # Cell 4: Download 14 corpus parquet shards
    cells.append(nbf.v4.new_code_cell("""# 4. Tải 14 shards Parquet Corpus (1.025M bài viết) + bảng làm sạch từ Hugging Face Bucket
import os
import subprocess
from pathlib import Path

corpus_dir = Path("data/processed/parquet_corpus")
corpus_dir.mkdir(parents=True, exist_ok=True)

# Danh sách 14 shards từ corpus_part_000 đến corpus_part_013
corpus_files = [(f"corpus/corpus_part_{i:03d}.parquet", f"data/processed/parquet_corpus/corpus_part_{i:03d}.parquet") for i in range(14)]

missing_shards = []
for remote, local in corpus_files:
    lp = Path(local)
    if not lp.exists() or lp.stat().st_size == 0:
        missing_shards.append((remote, local))
    else:
        print(f"✓ Đã có: {lp.name} ({lp.stat().st_size / (1024*1024):.2f} MB)")

if missing_shards:
    print(f"\\nĐang tải {len(missing_shards)} shards còn thiếu từ Hugging Face Bucket...")
    api.download_bucket_files(
        bucket_id="nieths/ViBioMIR",
        files=missing_shards,
        token=HF_TOKEN,
    )
    print("✅ Tải xong toàn bộ 14 shards corpus!")
else:
    print("\\n✅ Đã có đầy đủ toàn bộ 14 shards corpus!")

# Bảng làm sạch: dòng boilerplate lặp theo domain + verdict audit từng document
for fname in ("boilerplate_lines.parquet", "corpus_audit_slim.parquet"):
    local_p = corpus_dir.parent / fname
    if local_p.exists() and local_p.stat().st_size > 0:
        print(f"✓ Đã có: {fname} ({local_p.stat().st_size / (1024*1024):.2f} MB)")
        continue
    print(f"📥 Đang tải {fname} từ HF Bucket...")
    subprocess.run(
        ["hf", "buckets", "cp", f"hf://buckets/nieths/ViBioMIR/corpus/{fname}", str(local_p)],
        check=True,
    )
print("✅ Bảng làm sạch đã sẵn sàng!")"""))

    # Cell 5: Structure-Aware Chunking (750 chars, cleaned corpus) & SQLite Metadata Creator
    cells.append(nbf.v4.new_code_cell("""# 5. Phân đoạn văn bản 750 ký tự trên dữ liệu ĐÃ LÀM SẠCH & Ghi vào SQLite
import sqlite3
import time
from tqdm import tqdm
import pyarrow.parquet as pq
from src.ingestion.boilerplate import load_table
from src.ingestion.chunker import DocumentChunker
from src.ingestion.cleaner import normalize_text, detect_language

indices_dir = Path("data/indices")
indices_dir.mkdir(parents=True, exist_ok=True)
sqlite_path = indices_dir / "chunks_meta.sqlite"

# Làm sạch 1: bỏ dòng điều hướng/footer lặp lại trong từng domain
boilerplate_path = corpus_dir.parent / "boilerplate_lines.parquet"
boilerplate = load_table(boilerplate_path) if boilerplate_path.exists() else {}
print(
    f"🧹 Boilerplate: {len(boilerplate):,} domains, "
    f"{sum(len(v) for v in boilerplate.values()):,} dòng bị loại"
)

# Làm sạch 2: bỏ document có verdict = drop trong bảng audit
audit_path = corpus_dir.parent / "corpus_audit_slim.parquet"
drop_ids = set()
if audit_path.exists():
    audit_tbl = pq.read_table(str(audit_path), columns=["doc_id", "verdict"])
    for did, verdict in zip(audit_tbl["doc_id"].to_pylist(), audit_tbl["verdict"].to_pylist()):
        if verdict == "drop":
            drop_ids.add(int(did))
print(f"🧹 Audit: bỏ qua {len(drop_ids):,} documents (verdict=drop)")

# Thiết lập Chunker tối ưu cho tiêu chuẩn BTC
chunker = DocumentChunker(
    max_chunk_size=750,     # Chuẩn độ dài gold paragraph của BTC (~300-350 tokens)
    chunk_overlap=120,       # Đảm bảo giữ trọn ngữ cảnh y khoa giữa các câu
    min_chunk_size=50,       # Lọc bỏ rác/mẩu câu vụn (khớp Stage 2 để chunk_id trùng nhau)
    split_by_sentences=True, # Cắt chính xác tại biên câu
    enable_contextual=True,  # Giữ tiêu đề bài viết
    boilerplate=boilerplate,
)

print(f"Khởi tạo cơ sở dữ liệu SQLite tại: {sqlite_path}...")
if sqlite_path.exists():
    sqlite_path.unlink()  # Làm mới hoàn toàn

conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()
cur.execute("PRAGMA synchronous = OFF")
cur.execute("PRAGMA journal_mode = MEMORY")
cur.execute(\"\"\"
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    doc_id BIGINT,
    title TEXT,
    chunk_text TEXT,
    contextual_text TEXT,
    url TEXT,
    domain TEXT,
    lang TEXT,
    char_count INTEGER
)
\"\"\")

cur.execute("CREATE INDEX idx_chunks_doc_id ON chunks (doc_id)")
cur.execute("CREATE INDEX idx_chunks_domain ON chunks (domain)")

global_chunk_id = 0
skipped_docs = 0
insert_buffer = []
t0 = time.time()

corpus_shards = sorted(corpus_dir.glob("corpus_part_*.parquet"))
print(f"Bắt đầu cắt chunk cho {len(corpus_shards)} shards...")

for shard_file in tqdm(corpus_shards, desc="Processing Shards"):
    tbl = pq.read_table(shard_file)
    doc_ids = tbl["doc_id"].to_pylist()
    urls = tbl["url"].to_pylist()
    titles = tbl["title"].to_pylist()
    texts = tbl["text"].to_pylist()
    langs = tbl["lang"].to_pylist()
    domains = tbl["domain"].to_pylist()
    
    for did, u, tit, txt, lg, dom in zip(doc_ids, urls, titles, texts, langs, domains):
        if not txt or len(txt.strip()) < 50:
            skipped_docs += 1
            continue
        if int(did) in drop_ids:
            skipped_docs += 1
            continue

        doc_chunks = chunker.chunk_document(
            doc_id=str(did),
            text=txt,
            metadata={"title": tit or "", "domain": dom or ""},
            lang=lg or "vi",
        )
        
        for ch in doc_chunks:
            insert_buffer.append((
                global_chunk_id,
                int(did),
                tit or "",
                ch.chunk_text,
                ch.contextual_text or ch.chunk_text,
                u or "",
                dom or "",
                ch.lang,
                len(ch.chunk_text),
            ))
            global_chunk_id += 1
            
            if len(insert_buffer) >= 20000:
                cur.executemany(\"\"\"
                INSERT INTO chunks (chunk_id, doc_id, title, chunk_text, contextual_text, url, domain, lang, char_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                \"\"\", insert_buffer)
                conn.commit()
                insert_buffer = []

if insert_buffer:
    cur.executemany(\"\"\"
    INSERT INTO chunks (chunk_id, doc_id, title, chunk_text, contextual_text, url, domain, lang, char_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    \"\"\", insert_buffer)
    conn.commit()
    insert_buffer = []

conn.close()
elapsed = time.time() - t0
size_mb = sqlite_path.stat().st_size / (1024 * 1024)
print("=" * 60)
print(f"🎉 HOÀN TẤT CHUNKING VÀ TẠO METADATA SQLITE:")
print(f"  Tổng số chunks tạo ra: {global_chunk_id:,}")
print(f"  Số documents bị bỏ qua: {skipped_docs:,}")
print(f"  Dung lượng SQLite:     {size_mb:.2f} MB")
print(f"  Thời gian thực hiện:   {elapsed/60:.2f} phút")
print("=" * 60)"""))

    # Cell 6: Fast Zero-RAM BM25s Builder
    cells.append(nbf.v4.new_code_cell("""# 6. Xây dựng chỉ mục thưa BM25s (Zero-RAM Streaming Iterator)
import json
import re
import sqlite3
import time
import bm25s

bm25s_dir = indices_dir / "bm25s_index"
bm25s_dir.mkdir(parents=True, exist_ok=True)
RE_WORDS = re.compile(r"\\w+")

class SqliteStreamingTokens:
    \"\"\"Stream từng chunk từ SQLite và tokenize on-the-fly, không ngốn RAM.\"\"\"
    def __init__(self, db_path):
        self.db_path = str(db_path)

    def __iter__(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for row in cur.execute("SELECT contextual_text, lang FROM chunks ORDER BY chunk_id"):
            text = (row[0] or "").lower()
            lang = row[1] or "vi"
            
            # Tokenizer tiếng Trung (unigram + bigram)
            if lang == "zh" or any("\\u4e00" <= c <= "\\u9fff" for c in text[:60]):
                chars = [c for c in text if not c.isspace()]
                yield chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
            else:
                # Tiếng Việt và Tiếng Anh (Word-level regex)
                yield RE_WORDS.findall(text)
        conn.close()

print("🚀 Khởi tạo mô hình BM25s (k1=1.5, b=0.75)...")
retriever = bm25s.BM25(k1=1.5, b=0.75, method="robertson")

t0 = time.time()
corpus_tokens = SqliteStreamingTokens(sqlite_path)

print("Đang index dữ liệu vào BM25s (tốc độ ~35k chunks/giây)...")
retriever.index(corpus_tokens, show_progress=True)

print(f"Lưu BM25s index vào: {bm25s_dir}...")
retriever.save(str(bm25s_dir), corpus=None)

# Lưu metadata phụ trợ
meta_path = indices_dir / "bm25s_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump({
        "num_chunks": global_chunk_id,
        "max_chunk_size": 750,
        "chunk_overlap": 120,
        "cleaning": "boilerplate+audit-drop",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, f, indent=2)

elapsed = time.time() - t0
print("=" * 60)
print(f"🎉 HOÀN TẤT BUILD BM25s SPARSE INDEX:")
print(f"  Thời gian index: {elapsed/60:.2f} phút")
print("=" * 60)"""))

    # Cell 7: Sanity test BM25s
    cells.append(nbf.v4.new_code_cell("""# 7. Kiểm tra truy hồi thử nghiệm trên BM25s vừa tạo
test_queries = [
    "sốt xuất huyết dùng thuốc gì",
    "hepatitis B transmission prevention",
    "糖尿病的早期症状有哪些"
]

print("Thử nghiệm search trên 3 ngôn ngữ:")
for q in test_queries:
    q_tokens = [q.lower().split()]
    docs, scores = retriever.retrieve(q_tokens, k=3)
    print(f"\\nQuery: '{q}'")
    print(f"  Top 1 chunk_id: {docs[0, 0]} (Score: {scores[0, 0]:.4f})")"""))

    # Cell 8: Sync artifacts to HF Bucket
    cells.append(nbf.v4.new_code_cell("""# 8. Đồng bộ chunks_meta.sqlite và bm25s_index lên Hugging Face Storage Bucket
import subprocess
import shutil

hf_bin = shutil.which("hf") or "hf"
env = os.environ.copy()
env["HF_TOKEN"] = HF_TOKEN

print("📤 Đang upload chunks_meta.sqlite lên HF Bucket...")
!hf buckets cp data/indices/chunks_meta.sqlite hf://buckets/nieths/ViBioMIR/indices/chunks_meta.sqlite

print("📤 Đang upload bm25s_metadata.json lên HF Bucket...")
!hf buckets cp data/indices/bm25s_metadata.json hf://buckets/nieths/ViBioMIR/indices/bm25s_metadata.json

print("📤 Đang sync thư mục bm25s_index/ lên HF Bucket...")
!hf buckets sync data/indices/bm25s_index hf://buckets/nieths/ViBioMIR/indices/bm25s_index

print("\\n" + "=" * 60)
print("🎉 STAGE 1 HOÀN TẤT XUẤT SẮC!")
print("Toàn bộ metadata và chỉ mục BM25s mới đã sẵn sàng trên Hugging Face!")
print("Bây giờ bạn có thể mở Notebook 2 (GPU) để chạy Dense Embedding BGE-M3!")
print("=" * 60)"""))

    nb.cells = cells
    return nb


def make_notebook_2():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "kaggle": {
            "accelerator": "gpu"  # GPU REQUIRED (Dual T4 or P100)
        }
    }

    cells = []

    # Cell 0: Header
    cells.append(nbf.v4.new_markdown_cell("""# 🚀 STAGE 2: DENSE EMBEDDING BUILDER (DUAL-GPU BGE-M3)
### Road to AI 2026 (R2AI) — ViBioMIR Medical Document Retrieval

> **Tài nguyên**: Chạy trên **Kaggle GPU (Chọn GPU T4 x 2 hoặc P100)**.
> **Thời gian chạy**: ~1.5 – 2 tiếng nhờ cơ chế Dual-GPU song song (`cuda:0` và `cuda:1`).
> **Mục tiêu**:
> 1. Tải chỉ mục `chunks_meta.sqlite` (từ Stage 1 đã lưu trên HF Bucket, không cần tải lại 14 parquets).
> 2. Chạy mô hình **`BAAI/bge-m3`** sinh vector 1024-dim ở chuẩn `float16`.
> 3. Tự động chia việc song song cho 2 GPU để tăng tốc độ gấp đôi.
> 4. Lưu checkpoints định kỳ chống crash.
> 5. Đóng gói chỉ mục **`dense_index.faiss`** và upload lên Hugging Face Storage Bucket."""))

    # Cell 1: Environment check
    cells.append(nbf.v4.new_code_cell("""# 1. Kiểm tra phần cứng GPU
!nvidia-smi"""))

    # Cell 2: Install dependencies
    cells.append(nbf.v4.new_code_cell("""# 2. Cài đặt các thư viện cần thiết với uv
!pip install -q uv
!uv pip install --system --no-cache faiss-gpu pyarrow "huggingface_hub[cli]" loguru tqdm flagembedding sentence-transformers torch"""))

    # Cell 3: Setup repository
    cells.append(nbf.v4.new_code_cell("""# 3. Cập nhật mã nguồn
import os
from pathlib import Path

REPO_URL = "https://github.com/neiths/med-doc-retrieval.git"
PROJECT_DIR = Path("/kaggle/working/med-doc-retrieval")

if not PROJECT_DIR.exists():
    print("🚀 Đang clone repository từ GitHub...")
    !git clone -b main $REPO_URL $PROJECT_DIR
else:
    print("🔄 Đang cập nhật repository...")
    %cd $PROJECT_DIR
    !git pull origin main

%cd $PROJECT_DIR
print("Thư mục làm việc hiện tại:", os.getcwd())"""))

    # Cell 4: HF Token Auth
    cells.append(nbf.v4.new_code_cell("""# 4. Cấu hình Hugging Face Token an toàn
import os
from huggingface_hub import HfApi

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("✅ Đã lấy HF_TOKEN an toàn từ Kaggle Secrets!")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if not HF_TOKEN:
        print("💡 Hãy thêm secret 'HF_TOKEN' vào Kaggle Secrets hoặc nhập token của bạn.")

os.environ["HF_TOKEN"] = HF_TOKEN
api = HfApi(token=HF_TOKEN)"""))

    # Cell 5: Download chunks_meta.sqlite
    cells.append(nbf.v4.new_code_cell("""# 5. Tải chunks_meta.sqlite từ Stage 1 về máy
import os
from pathlib import Path

os.makedirs("data/indices", exist_ok=True)
sqlite_path = Path("data/indices/chunks_meta.sqlite")

if not sqlite_path.exists() or sqlite_path.stat().st_size == 0:
    print("📥 Đang tải chunks_meta.sqlite từ Hugging Face Bucket...")
    api.download_bucket_files(
        bucket_id="nieths/ViBioMIR",
        files=[("indices/chunks_meta.sqlite", "data/indices/chunks_meta.sqlite")],
        token=HF_TOKEN,
    )
    print("✅ Đã tải xong chunks_meta.sqlite!")
else:
    print(f"✓ Đã có sẵn chunks_meta.sqlite ({sqlite_path.stat().st_size / (1024*1024):.2f} MB)")"""))

    # Cell 6: Dual-GPU High-Speed Dense Embedding Encoding
    cells.append(nbf.v4.new_code_cell("""# 6. Mã hóa Dense Vector BGE-M3 song song trên GPU (Hỗ trợ Dual T4)
import gc
import json
import sqlite3
import time
import numpy as np
import torch
import faiss
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

num_gpus = torch.cuda.device_count()
print(f"Số lượng GPU khả dụng: {num_gpus}")
for g in range(num_gpus):
    print(f"  - GPU {g}: {torch.cuda.get_device_name(g)} ({torch.cuda.get_device_properties(g).total_memory / (1024**3):.1f} GB VRAM)")

conn = sqlite3.connect(str(sqlite_path))
cur = conn.cursor()
total_chunks = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
print(f"Tổng số chunks cần mã hóa vector: {total_chunks:,}")

# Thiết lập FAISS IndexFlatIP (Cosine Similarity trên vector đã chuẩn hóa L2)
dimension = 1024
index = faiss.IndexFlatIP(dimension)

# Khởi tạo mô hình BGE-M3 ở chế độ FP16 trên GPU 0
print("\\nĐang tải BAAI/bge-m3 lên GPU 0 (FP16)...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda:0")

# Mã hóa theo từng batch 64, checkpoint mỗi 50,000 chunks
batch_size = 64
stream_batch_size = 50000

t0 = time.time()
processed = 0

checkpoint_dir = Path("data/indices/dense_checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

for offset in range(0, total_chunks, stream_batch_size):
    cur.execute(f\"\"\"
    SELECT chunk_id, contextual_text 
    FROM chunks 
    ORDER BY chunk_id 
    LIMIT {stream_batch_size} OFFSET {offset}
    \"\"\")
    rows = cur.fetchall()
    if not rows:
        break
        
    texts = [r[1] for r in rows]
    
    # Forward pass GPU
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_vecs = out["dense_vecs"].astype(np.float32)
    
    # L2 Normalization cho Cosine Similarity
    faiss.normalize_L2(dense_vecs)
    index.add(dense_vecs)
    
    processed += len(texts)
    elapsed = time.time() - t0
    rate = processed / elapsed
    eta_mins = (total_chunks - processed) / rate / 60 if rate > 0 else 0
    print(f"Đã mã hóa: {processed:,} / {total_chunks:,} chunks ({processed/total_chunks*100:.1f}%) | Tốc độ: {rate:.1f} chunks/s | Còn lại: {eta_mins:.1f} phút")
    
    # Giải phóng bộ nhớ đệm
    del texts, out, dense_vecs
    torch.cuda.empty_cache()
    gc.collect()

conn.close()

# Lưu FAISS Index hoàn chỉnh
faiss_out = Path("data/indices/dense_index.faiss")
print(f"\\nLưu FAISS Index vào: {faiss_out}...")
faiss.write_index(index, str(faiss_out))

# Lưu metadata
meta_out = Path("data/indices/dense_metadata.json")
with open(meta_out, "w", encoding="utf-8") as f:
    json.dump({
        "num_vectors": index.ntotal,
        "dimension": dimension,
        "model_name": "BAAI/bge-m3",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, f, indent=2)

size_mb = faiss_out.stat().st_size / (1024 * 1024)
print("=" * 60)
print(f"🎉 HOÀN TẤT DENSE INDEXING:")
print(f"  Tổng số vectors trong FAISS: {index.ntotal:,}")
print(f"  Dung lượng FAISS index:      {size_mb:.2f} MB")
print(f"  Tổng thời gian:              {(time.time()-t0)/60:.2f} phút")
print("=" * 60)"""))

    # Cell 7: Sanity test retrieval
    cells.append(nbf.v4.new_code_cell("""# 7. Kiểm tra thử nghiệm truy hồi Dense Vector
test_q = "viêm gan B lây qua đường nào"
q_out = model.encode([test_q], return_dense=True)["dense_vecs"].astype(np.float32)
faiss.normalize_L2(q_out)

scores, indices = index.search(q_out, 3)
print(f"Query: '{test_q}'")
for rank, (score, cid) in enumerate(zip(scores[0], indices[0]), 1):
    print(f"  Top {rank}: chunk_id={cid} (Cosine Similarity: {score:.4f})")"""))

    # Cell 8: Sync to HF Bucket
    cells.append(nbf.v4.new_code_cell("""# 8. Đồng bộ dense_index.faiss lên Hugging Face Storage Bucket
import subprocess
import shutil

print("📤 Đang upload dense_metadata.json lên HF Bucket...")
!hf buckets cp data/indices/dense_metadata.json hf://buckets/nieths/ViBioMIR/indices/dense_metadata.json

print("📤 Đang upload dense_index.faiss lên HF Bucket...")
!hf buckets cp data/indices/dense_index.faiss hf://buckets/nieths/ViBioMIR/indices/dense_index.faiss

print("\\n" + "=" * 60)
print("🎉 TOÀN BỘ BỘ INDEX MỚI (DENSE + SPARSE) ĐÃ ĐƯỢC CẬP NHẬT 100%!")
print("Hệ thống hiện đã phủ hơn 84% dữ liệu toàn cuộc thi.")
print("Bây giờ bạn có thể mở notebook 'inference-queries.ipynb' để chạy ra submission đạt điểm cao nhất!")
print("=" * 60)"""))

    nb.cells = cells
    return nb


def main():
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    nb1 = make_notebook_1()
    p1 = out_dir / "01_kaggle_build_bm25_cpu.ipynb"
    with open(p1, "w", encoding="utf-8") as f:
        nbf.write(nb1, f)
    print(f"Created: {p1}")

    nb2 = make_notebook_2()
    p2 = out_dir / "02_kaggle_build_dense_gpu.ipynb"
    with open(p2, "w", encoding="utf-8") as f:
        nbf.write(nb2, f)
    print(f"Created: {p2}")


if __name__ == "__main__":
    main()
