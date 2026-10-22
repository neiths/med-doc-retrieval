"""Comprehensive Audit & Gap Analysis:
Compares the Hugging Face dataset (tmh1008/ViBioMIR-crawled-corpus), local crawled shards,
and local PubMed corpus against the official competition registry (links_corpus_vi_en_zh.parquet).
"""

import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm


def assign_lang(host: str) -> str:
    if "pubmed" in host:
        return "en"
    zh_domains = [
        "120ask.com", "cnkang.com", "39.net", "familydoctor.com.cn",
        "zysjonline.com", "zhongyibaodian.net", "a-hospital.com", "youlai.cn",
        "chunyuyisheng.com", "haodf.com", "msdmanuals.cn", "iiyi.com"
    ]
    if any(zh in host for zh in zh_domains):
        return "zh"
    return "vi"


def run_audit():
    logger.info("=" * 70)
    logger.info("STARTING COMPREHENSIVE CORPUS AUDIT")
    logger.info("=" * 70)

    # 1. Load official registry
    registry_path = Path("data/raw/vibio_mir/links_corpus_vi_en_zh.parquet")
    logger.info(f"Loading official competition registry from {registry_path}...")
    tbl_reg = pq.read_table(registry_path)
    df_reg = tbl_reg.to_pandas()
    df_reg["host"] = df_reg["url"].apply(lambda u: urlparse(str(u)).netloc.lower())
    df_reg["lang"] = df_reg["host"].apply(assign_lang)

    total_official = len(df_reg)
    official_ids = set(df_reg["id"])
    logger.info(f"Official registry URLs: {total_official:,}")

    official_lang_counts = df_reg["lang"].value_counts().to_dict()
    official_host_counts = df_reg["host"].value_counts().to_dict()

    # 2. Scan Local PubMed (English)
    pubmed_dir = Path("data/processed/v2/pubmed")
    pubmed_ids = set()
    logger.info("Scanning local PubMed directory...")
    for f in pubmed_dir.glob("*.jsonl"):
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    try:
                        pubmed_ids.add(json.loads(line)["doc_id"])
                    except Exception:
                        pass
    logger.info(f"Local PubMed unique articles: {len(pubmed_ids):,}")

    # 3. Scan Local Crawled Shards (JSONL)
    crawled_dir = Path("data/processed/v2/crawled")
    local_crawled_ids = set()
    local_host_counts = Counter()
    logger.info("Scanning local JSONL crawled shards...")
    for f in crawled_dir.glob("crawled_shard_*.jsonl"):
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        doc_id = obj["doc_id"]
                        local_crawled_ids.add(doc_id)
                        local_host_counts[obj.get("domain", "unknown")] += 1
                    except Exception:
                        pass
    logger.info(f"Local JSONL unique crawled articles: {len(local_crawled_ids):,}")

    # 4. Scan Hugging Face Downloaded Shards (Parquet)
    hf_dir = Path("data/processed/v2/hf_crawled_corpus/data")
    hf_shards = sorted(list(hf_dir.glob("*.parquet"))) if hf_dir.exists() else []
    logger.info(f"Found {len(hf_shards)} downloaded parquet shards from Hugging Face...")

    hf_success_ids = set()
    hf_failed_ids = set()
    hf_host_counts = Counter()

    for sf in tqdm(hf_shards, desc="Reading HF Parquet Shards"):
        try:
            tbl = pq.read_table(sf, columns=["id", "status_code", "success", "domain"])
            pyd = tbl.to_pydict()
            ids = pyd["id"]
            statuses = pyd.get("status_code", [200] * len(ids))
            successes = pyd.get("success", [True] * len(ids))
            domains = pyd.get("domain", [""] * len(ids))

            for doc_id, sc, succ, dom in zip(ids, statuses, successes, domains):
                if succ and sc == 200:
                    hf_success_ids.add(doc_id)
                    hf_host_counts[dom] += 1
                else:
                    hf_failed_ids.add(doc_id)
        except Exception as e:
            logger.warning(f"Error reading {sf.name}: {e}")

    # Also load the master scraped_ids list from HF bucket
    hf_bucket_file = Path("data/processed/v2/hf_scraped_ids.parquet")
    if hf_bucket_file.exists():
        hf_bucket_ids = set(pq.read_table(hf_bucket_file)["doc_id"].to_pylist())
        logger.info(f"Loaded {len(hf_bucket_ids):,} IDs from HF master scraped_ids.parquet")
        hf_success_ids.update(hf_bucket_ids)

    logger.info(f"HF Parquet Shards Successful articles: {len(hf_success_ids):,}")
    logger.info(f"HF Parquet Shards Failed/Blocked items: {len(hf_failed_ids):,}")

    # 5. Combined Synthesis
    # All successful articles across all 3 storage sources
    all_success_ids = pubmed_ids.union(local_crawled_ids).union(hf_success_ids)
    all_success_in_registry = all_success_ids.intersection(official_ids)

    missing_ids = official_ids - all_success_in_registry
    df_missing = df_reg[df_reg["id"].isin(missing_ids)]
    missing_host_counts = df_missing["host"].value_counts().to_dict()
    missing_lang_counts = df_missing["lang"].value_counts().to_dict()

    # Calculate per-language completion
    report_lines = []
    report_lines.append("# 📊 BÁO CÁO KIỂM KÊ & ĐỐI SOÁT TOÀN DIỆN DỮ LIỆU CÀO (AUDIT REPORT)")
    report_lines.append(f"\n*Thời gian thực hiện: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    report_lines.append(f"\n---\n")

    report_lines.append("## 1. TỔNG QUAN TIẾN ĐỘ THU THẬP")
    pct_total = len(all_success_in_registry) / total_official * 100
    report_lines.append(f"- **Tổng số URL trong Registry cuộc thi**: **{total_official:,}** URLs")
    report_lines.append(f"- **ĐÃ THU THẬP THÀNH CÔNG (HỢP NHẤT)**: **{len(all_success_in_registry):,}** bài viết (**{pct_total:.2f}%**)")
    report_lines.append(f"- **Số URL còn thiếu (Missing)**: **{len(missing_ids):,}** URLs (**{100 - pct_total:.2f}%**)")

    report_lines.append("\n### Nguồn đóng góp vào kho dữ liệu:")
    report_lines.append(f"| Nguồn Dữ Liệu | Loại File | Số lượng Thành công | Mô tả |")
    report_lines.append(f"| :--- | :--- | :--- | :--- |")
    report_lines.append(f"| **Hugging Face (`tmh1008/ViBioMIR-crawled-corpus`)** | Parquet | **{len(hf_success_ids):,}** bài | Dữ liệu cào từ Kaggle/Colab |")
    report_lines.append(f"| **Local Crawled Shards (`data/processed/v2/crawled/`)** | JSONL | **{len(local_crawled_ids):,}** bài | Dữ liệu cào tại máy local |")
    report_lines.append(f"| **PubMed E-utilities (`data/processed/v2/pubmed/`)** | JSONL | **{len(pubmed_ids):,}** bài | 100% tài liệu y khoa tiếng Anh |")
    report_lines.append(f"| **HỢP NHẤT DUY NHẤT (UNIQUE IDS)** | - | **{len(all_success_in_registry):,}** bài | Không trùng lặp |")

    report_lines.append("\n## 2. TIẾN ĐỘ THEO TỪNG NGÔN NGỮ")
    report_lines.append("| Ngôn ngữ | Mục tiêu BTC | Đã cào thành công | Còn thiếu | Tỷ lệ hoàn thành |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- |")

    for l_code, l_name in [("vi", "Tiếng Việt"), ("en", "Tiếng Anh (PubMed)"), ("zh", "Tiếng Trung")]:
        target = official_lang_counts.get(l_code, 0)
        missing = missing_lang_counts.get(l_code, 0)
        done = target - missing
        pct = (done / target * 100) if target > 0 else 0.0
        status_icon = "✅" if pct >= 99.0 else ("⚠️" if pct >= 50.0 else "❌")
        report_lines.append(f"| {status_icon} **{l_name}** | {target:,} | **{done:,}** | {missing:,} | **{pct:.2f}%** |")

    report_lines.append("\n## 3. CHI TIẾT THEO CÁC DOMAIN CÒN THIẾU")
    report_lines.append("Dưới đây là danh sách các domain còn thiếu nhiều URL nhất:\n")
    report_lines.append("| Tên Miền (Host) | Ngôn ngữ | Mục tiêu BTC | Đã có | Còn thiếu | Tỷ lệ hoàn thành |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for host, missing_cnt in sorted(missing_host_counts.items(), key=lambda x: -x[1]):
        target = official_host_counts.get(host, missing_cnt)
        done = target - missing_cnt
        pct = (done / target * 100) if target > 0 else 0.0
        lang = assign_lang(host).upper()
        report_lines.append(f"| `{host}` | {lang} | {target:,} | {done:,} | **{missing_cnt:,}** | {pct:.1f}% |")

    report_text = "\n".join(report_lines)
    out_report_path = Path("data/processed/v2/hf_and_corpus_audit.md")
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Also save the exact missing URLs to parquet for targeted continuation
    out_missing_parquet = Path("data/processed/v2/exact_missing_urls.parquet")
    pq.write_table(pa.Table.from_pandas(df_missing[["id", "url", "host", "lang"]]), out_missing_parquet)

    logger.success(f"Audit completed! Report written to {out_report_path}")
    logger.success(f"Exact missing URLs ({len(df_missing):,}) saved to {out_missing_parquet}")
    print("\n" + report_text)


if __name__ == "__main__":
    run_audit()
