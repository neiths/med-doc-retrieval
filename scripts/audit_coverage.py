"""B0 - Coverage audit & corpus filtering against the official URL list.

1. Loads the official URL list (links_corpus_vi_en_zh.parquet).
2. Keeps only crawled docs whose doc_id is still in the official list
   (drops ZH docs removed by the organizers).
3. Flags low-quality docs (error/login/home pages, too short, exact duplicates).
   Duplicates are NOT deleted: they are kept with `dup_of` so every id stays mapped.
4. Writes:
   - <out-dir>/corpus_v2/part_XXX.parquet   filtered corpus (+ qc_flag, dup_of)
   - <out-dir>/missing_urls.parquet         official ids not yet crawled (id, url, host)
   - <out-dir>/coverage_report.md           coverage by language/host + QC stats

Usage:
    python scripts/audit_coverage.py \
        --links data/raw/vibio_mir/links_corpus_vi_en_zh.parquet \
        --corpus-dir data/processed/parquet_corpus \
        --out-dir data/processed/v2
"""

import argparse
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

# Titles / text that indicate the page did not contain a real article
ERROR_TITLE_PATTERNS = [
    r"^trang chủ$",
    r"^đăng nhập",
    r"^home$",
    r"^login",
    r"^sign in",
    r"404",
    r"not found",
    r"页面不存在",
    r"抱歉",
    r"^首页$",
    r"^登录",
    r"error",
]
ERROR_TEXT_PATTERNS = [
    r"trang bạn tìm kiếm không tồn tại",
    r"không tìm thấy trang",
    r"page (you requested )?(was )?not found",
    r"this content is not available in english",
    r"您访问的页面不存在",
    r"enable javascript",
    r"access denied",
    r"just a moment\.\.\.",  # Cloudflare challenge
]
RE_ERR_TITLE = re.compile("|".join(ERROR_TITLE_PATTERNS), re.IGNORECASE)
RE_ERR_TEXT = re.compile("|".join(ERROR_TEXT_PATTERNS), re.IGNORECASE)


def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower() or "unknown"
    except Exception:
        return "unknown"


def lang_of_host(host: str) -> str:
    """Best-effort language guess for URLs not crawled yet (used only for reporting)."""
    if "pubmed" in host or "ncbi" in host:
        return "en"
    if host.endswith(".vn") or ".vn" in host or host in {"hellobacsi.com", "youmed.vn"}:
        return "vi"
    if host.endswith(".cn") or host.endswith(".com.cn") or host in {
        "www.120ask.com", "www.cnkang.com", "ask.39.net", "zysjonline.com",
        "zhongyibaodian.net", "www.a-hospital.com", "jb39.com", "test.pmphai.com",
    }:
        return "zh"
    return "?"


def text_fingerprint(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def qc_flag(title: str, text: str, min_chars: int) -> str:
    title = (title or "").strip()
    text = text or ""
    if len(text.strip()) < min_chars:
        return "too_short"
    if title and RE_ERR_TITLE.search(title) and len(text) < 3000:
        return "error_title"
    if RE_ERR_TEXT.search(text[:1500]) and len(text) < 3000:
        return "error_text"
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--links", type=Path, default=Path("data/raw/vibio_mir/links_corpus_vi_en_zh.parquet"))
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/processed/parquet_corpus"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/v2"))
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--rows-per-part", type=int, default=100_000)
    args = ap.parse_args()

    out_corpus = args.out_dir / "corpus_v2"
    out_corpus.mkdir(parents=True, exist_ok=True)
    for old in out_corpus.glob("*.parquet"):
        old.unlink()

    # 1. Official list
    links = pq.read_table(args.links, columns=["id", "url"])
    official = dict(zip(links["id"].to_pylist(), links["url"].to_pylist()))
    logger.info(f"Official URLs: {len(official):,}")

    # 2. Stream crawled corpus
    files = sorted(args.corpus_dir.glob("*.parquet"))
    logger.info(f"Crawled shards: {[f.name for f in files]}")

    seen_ids: set[int] = set()
    fp_first: dict[str, int] = {}
    stats = Counter()
    lang_kept = Counter()
    lang_dropped = Counter()
    qc_counter = Counter()
    qc_by_host: dict[str, Counter] = defaultdict(Counter)

    buffer: list[dict] = []
    part_idx = 0
    schema = None

    def flush():
        nonlocal buffer, part_idx, schema
        if not buffer:
            return
        tbl = pa.Table.from_pylist(buffer, schema=schema)
        pq.write_table(tbl, out_corpus / f"part_{part_idx:03d}.parquet", compression="zstd")
        logger.info(f"Wrote part_{part_idx:03d}.parquet ({len(buffer):,} rows)")
        part_idx += 1
        buffer = []

    for f in files:
        pf = pq.ParquetFile(f)
        if schema is None:
            schema = pf.schema_arrow.append(pa.field("qc_flag", pa.string())).append(pa.field("dup_of", pa.int64()))
        for batch in pf.iter_batches(batch_size=20_000):
            for row in batch.to_pylist():
                stats["crawled_rows"] += 1
                did = row["doc_id"]
                if did in seen_ids:
                    stats["dup_doc_id_rows"] += 1
                    continue
                seen_ids.add(did)
                if did not in official:
                    lang_dropped[row.get("lang") or "?"] += 1
                    continue
                lang_kept[row.get("lang") or "?"] += 1

                flag = qc_flag(row.get("title"), row.get("text"), args.min_chars)
                dup_of = None
                if flag == "ok":
                    fp = text_fingerprint(row.get("text"))
                    if fp in fp_first:
                        flag = "duplicate"
                        dup_of = fp_first[fp]
                    else:
                        fp_first[fp] = did
                qc_counter[flag] += 1
                qc_by_host[row.get("domain") or host_of(row.get("url", ""))][flag] += 1
                row["qc_flag"] = flag
                row["dup_of"] = dup_of
                buffer.append(row)
                if len(buffer) >= args.rows_per_part:
                    flush()
    flush()

    # 3. Missing URLs
    missing_ids = [i for i in official if i not in seen_ids]
    missing_urls = [official[i] for i in missing_ids]
    missing_hosts = [host_of(u) for u in missing_urls]
    pq.write_table(
        pa.table({"id": pa.array(missing_ids, pa.int64()), "url": missing_urls, "host": missing_hosts}),
        args.out_dir / "missing_urls.parquet",
        compression="zstd",
    )

    # 4. Report
    total_by_host = Counter(host_of(u) for u in official.values())
    miss_by_host = Counter(missing_hosts)
    total_by_lang = Counter(lang_of_host(h) for h in total_by_host.elements())
    kept_total = sum(lang_kept.values())
    usable = qc_counter["ok"]

    lines = [
        "# Coverage report (B0)",
        "",
        f"- Official URLs: **{len(official):,}**",
        f"- Crawled rows read: {stats['crawled_rows']:,} (duplicate doc_id rows skipped: {stats['dup_doc_id_rows']:,})",
        f"- Crawled & still official: **{kept_total:,}** ({kept_total / len(official):.1%})",
        f"- Usable after QC (`qc_flag == ok`): **{usable:,}** ({usable / len(official):.1%})",
        f"- Missing (never crawled): **{len(missing_ids):,}** -> `missing_urls.parquet`",
        "",
        "## By language",
        "",
        "| Lang | Official (by host guess) | Crawled & kept | Dropped (not in list) |",
        "|---|---|---|---|",
    ]
    for lg in ["vi", "zh", "en", "?"]:
        lines.append(f"| {lg} | {total_by_lang.get(lg, 0):,} | {lang_kept.get(lg, 0):,} | {lang_dropped.get(lg, 0):,} |")
    lines += ["", "## QC flags (kept docs)", "", "| Flag | Count |", "|---|---|"]
    for k, v in qc_counter.most_common():
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        "## By host (top 40 by official count)",
        "",
        "| Host | Official | Missing | Coverage | Usable | Bad (QC) |",
        "|---|---|---|---|---|---|",
    ]
    for h, n in total_by_host.most_common(40):
        q = qc_by_host.get(h, Counter())
        bad = sum(v for k, v in q.items() if k != "ok")
        cov = 1 - miss_by_host.get(h, 0) / n
        lines.append(f"| {h} | {n:,} | {miss_by_host.get(h, 0):,} | {cov:.0%} | {q.get('ok', 0):,} | {bad:,} |")

    report = args.out_dir / "coverage_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Report written to {report}")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
