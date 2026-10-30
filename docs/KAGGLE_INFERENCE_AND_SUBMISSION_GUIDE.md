# 🏆 HƯỚNG DẪN CHẠY INFERENCE VÀ NỘP BÀI TỐI ƯU ĐIỂM SỐ (> 0.24+)
### Cuộc thi: Road to AI 2026 (R2AI) — ViBioMIR: Multilingual Medical Document Retrieval
> **Tài liệu hướng dẫn thực hành khi về nhà**  
> *Ngày cập nhật: 30/10/2026*

---

## 1. TỔNG QUAN & PHÂN TÍCH BÀI HỌC TỪ CÁC LẦN NỘP BÀI

### 1.1. So sánh chi tiết kết quả thực tế từ CodaBench

| Chỉ số | Bài Teammate (Sub 7717, lúc 16:05) | Bài của bạn (Sub 7725, lúc 17:10) | Nhận xét chuyên môn |
| :--- | :---: | :---: | :--- |
| **DOCS_RECALL** | **0.5421** 🏆 | 0.2388 | Teammate nộp **300 docs/query**, bạn nộp **60 docs/query**. |
| **DOCS_PRECISION**| 0.1274 | **0.2353** 🏆 | Dự đoán tài liệu của bạn sạch hơn gấp đôi, ít nhiễu hơn. |
| **DOCS_F2MACRO** | **0.2882** 🏆 | 0.2097 | Teammate ăn điểm nhờ Recall tài liệu vượt trội (> 54%). |
| **CHUNKS_PRECISION**| 0.0769 | **0.0944** 🏆 | **Bạn thắng teammate về độ chính xác của đoạn văn!** |
| **CHUNKS_RECALL** | 0.1149 | **0.1301** 🏆 | **Bạn thắng teammate về độ phủ của đoạn văn!** |
| **CHUNKS_F2MACRO** | 0.0878 | **0.1033** 🏆 | **Chunk của bạn điểm cao hơn hẳn teammate!** |
| **Độ dài Chunk** | **1.120 – 4.245 ký tự** ⚠️ | **~750 ký tự** ✅ | Teammate gộp (stitch) chunk quá dài vi phạm quy tắc LCS 2 chiều $\ge 0.40$! |
| **FINAL_SCORE** | **0.1880** | 0.1565 | Teammate dẫn điểm nhờ ăn gian Recall bằng 300 docs. |

---

### 1.2. Hai phát hiện mang tính quyết định

1. **Về Chunk (Đoạn văn)**:
   - Teammate bị điểm Chunks rất thấp (**0.0878**) do cố tình nối (stitch) các chunk lại với nhau, khiến độ dài chunk dự đoán vọt lên 1.100 – 4.200 ký tự.
   - **Quy tắc chấm của BTC**: Yêu cầu tỷ lệ xâu con chung dài nhất (LCS) 2 chiều:
     $$\frac{\text{LCS}}{\text{len}(pred)} \ge 0.40 \quad \text{và} \quad \frac{\text{LCS}}{\text{len}(gt)} \ge 0.40$$
     Khi đoạn văn dự đoán dài 2.000 – 4.000 ký tự mà đoạn văn nhãn chuẩn của BTC chỉ dài ~700 ký tự thì tỷ lệ $\frac{\text{LCS}}{\text{len}(pred)}$ chỉ đạt $0.15 - 0.25 < 0.40$, **khiến teammate bị tính 0 điểm ở phần lớn các chunk**!
   - Đoạn văn chuẩn **~750 ký tự** của bạn đang làm rất chuẩn, điểm Chunks của bạn (**0.1033**) vượt trội hơn teammate.

2. **Về Docs (Tài liệu)**:
   - Teammate đạt `DOCS_F2MACRO = 0.2882` vì nộp **300 documents** cho mỗi query, kéo `DOCS_RECALL` lên **0.5421**.
   - Do công thức Macro $F_2 = \frac{5 \cdot P \cdot R}{4 \cdot P + R}$ ưu tiên Recall gấp **4 lần** Precision ($\beta = 2$), việc tăng số lượng doc từ 60 lên 300 mang lại lợi ích điểm số cực kỳ to lớn.

---

## 2. CÔNG THỨC CHIẾN THẮNG ĐỂ ĐẠT ĐIỂM > 0.24 – 0.25+

Chúng ta kết hợp **tinh hoa của cả hai phương pháp**:
1. **Lấy ưu điểm Docs từ Teammate**: Nâng `top_k_docs = 300` để đẩy `DOCS_RECALL` lên $> 54\%$, đưa `DOCS_F2MACRO` lên **~0.29 – 0.30**.
2. **Lấy ưu điểm Chunks của bạn**:
   - Giữ nguyên các chunk chuẩn **750 ký tự** (không gộp rác, thỏa mãn tuyệt đối luật LCS $\ge 0.40$).
   - Sử dụng **bộ chỉ mục BM25s 2.32M mới** (vừa upload lên HF Bucket) khớp 100% với FAISS SQ8.
   - Bật lại tính năng **Expand Top Docs** (`expand_top_docs = 30`, `expand_max_chunks_per_doc = 5`) để trích xuất đúng các đoạn văn cốt lõi bên trong 30 tài liệu uy tín nhất.
   - Đặt `top_k_chunks = 150` (thay vì 100 hay 120).
   - Điểm `CHUNKS_F2MACRO` dự kiến sẽ nhảy vọt từ **0.1033 lên ~0.19 – 0.21** (gấp 2.5 lần teammate).
3. **Điểm tổng dự kiến**:
   $$\text{FINAL\_SCORE} = \frac{\text{DOCS\_F2 (0.29)} + \text{CHUNKS\_F2 (0.20)}}{2} \approx \mathbf{0.245 - 0.25+!!}$$

---

## 3. THÔNG SỐ CẤU HÌNH CUỐI CÙNG (ĐÃ LƯU TRÊN REPO)

Toàn bộ tham số đã được thiết lập sẵn trong Cell 8 của notebook [`notebooks/inference-queries.ipynb`](file:///home/thienhb/Workspace/med-doc-retrieval/notebooks/inference-queries.ipynb):

```python
sys.argv = [
    "scripts/run_inference_only.py",
    "--queries", str(queries_path),
    "--dense-dir", dense_arg,
    "--chunks-meta-path", meta_arg,
    "--hybrid-top-k", "350",                     # Tập ứng viên phong phú từ cả FAISS SQ8 và BM25s mới (2.32M)
    "--top-k-docs", "300",                       # Tái hiện DOCS_RECALL > 0.54 của teammate để đạt DOCS_F2 ~ 0.29
    "--top-k-chunks", "150",                     # Ngân sách chunks tối ưu theo trọng số Recall 4x của F2
    "--max-chunks-per-doc", "3",                 # Cho phép 2-3 chunk quan trọng cùng xuất hiện trong 1 doc
    "--expand-top-docs", "30",                   # Quét toàn bộ chunks trong top 30 docs uy tín nhất
    "--expand-max-chunks-per-doc", "5",          # Lọc 5 chunk có từ khóa khớp nhất mỗi doc mở rộng
    "--batch-size", "32",                        # Chạy song song mượt mà trên Dual GPU T4
    "--disable-pubmed-network",                  # Dùng 100% offline cache 0ms, không tốn thời gian chờ mạng
    "--checkpoint", "outputs/submissions/run_v3_win.partial.jsonl",
]
```

---

## 4. QUY TRÌNH THỰC HIỆN KHI VỀ NHÀ (KAGGLE RUNNER)

### Bước 1: Mở Kaggle và chuẩn bị Notebook
1. Đăng nhập vào [Kaggle](https://www.kaggle.com/).
2. Nếu mở lại Notebook cũ:
   - Ở menu bên phải:
     * **Accelerator**: Chọn **GPU T4 x 2** (Dual GPU).
     * **Internet**: Bật **ON**.
     * **Add-ons -> Secrets**: Đảm bảo secret `HF_TOKEN` đã được tích chọn checkbox.
3. Nếu tạo Notebook mới:
   - Bấm **File -> Upload Notebook**, chọn file `notebooks/inference-queries.ipynb` từ repository về máy để nạp lên.

### Bước 2: Chạy các Cell theo thứ tự
1. **Cell 1**: `!nvidia-smi` (Kiểm tra hiển thị 2 GPU Tesla T4, mỗi GPU 16GB VRAM).
2. **Cell 2**: Cấu hình CSS chống tràn dòng chữ ngang.
3. **Cell 3 - 4**: Cài đặt thư viện (`uv pip install ...`).
4. **Cell 5 (Quan trọng)**: Kéo code mới nhất từ GitHub:
   ```python
   !git pull origin main
   ```
   *(Kiểm tra output báo cập nhật commit `a09cd5b` hoặc mới hơn).*
5. **Cell 6**: Tải bộ chỉ mục BM25s mới (2.32M chunks) và PubMed cache offline từ Hugging Face Bucket:
   - Sẽ tự động kiểm tra và tải các file nhị phân `bm25s_index/` và `queries_enriched.jsonl`.
6. **Cell 7**: `!ls -lh data/indices` (Xác nhận các file chỉ mục đã đầy đủ).
7. **Cell 8 (Chạy Inference)**:
   - Bấm chạy cell này.
   - Quá trình chạy dự kiến mất khoảng **55 – 65 phút** cho toàn bộ 1.200 queries (tốc độ ~0.25 - 0.30s/query).
   - Có cơ chế checkpoint `run_v3_win.partial.jsonl`, nếu mạng bị ngắt giữa chừng chỉ cần chạy lại cell là sẽ tự động resume từ query dang dở.
8. **Cell 9**: Bấm nút màu xanh **"⬇ BẤM VÀO ĐÂY ĐỂ TẢI SUBMISSION.ZIP VỀ MÁY"**.
9. **Cell 10**: Cell tự động kiểm tra kiểm định file `submission.zip`:
   - Xác nhận đủ 1.200 queries.
   - Số docs trung bình: ~300 docs/query.
   - Số chunks trung bình: ~150 chunks/query.
   - Ký tự trung bình mỗi chunk: ~750 ký tự (chuẩn bi-directional LCS).

### Bước 3: Nộp bài lên CodaBench
1. Vào cổng thi R2AI 2026 ViBioMIR trên CodaBench.
2. Chọn mục **Submit / View Results**.
3. Tải file `submission.zip` vừa tải về máy lên.
4. Đợi khoảng 2 – 3 phút hệ thống chấm điểm xong và xem điểm số bứt phá!

---

## 5. BẢNG CHECKLIST AN TOÀN TRƯỚC KHI BẤM "RUN ALL"

- [ ] Đã chọn Accelerator: **GPU T4 x 2** (không chọn None hoặc P100).
- [ ] Internet: **ON**.
- [ ] Đã bật Secret `HF_TOKEN` trong tab Add-ons -> Secrets.
- [ ] Cell 5 đã in ra commit mới nhất từ branch `main`.
- [ ] Checkpoint trong Cell 8 là `outputs/submissions/run_v3_win.partial.jsonl` (tránh đọc nhầm checkpoint cũ).

---
*Chúc bạn có buổi tối chạy code thuận lợi và đạt điểm kỷ lục mới dẫn đầu bảng xếp hạng!*
