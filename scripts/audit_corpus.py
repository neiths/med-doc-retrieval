"""Builds a per-document quality table for the crawled corpus.

Emits one row per document with quality signals and a keep/review/drop verdict, so a
rebuild can filter on a reviewed table instead of ad-hoc greps.

Boilerplate is measured first (one pass) and stripped before length, language and density
signals are computed, so a 150-char navigation tail cannot inflate a 200-char Q&A answer.
The resulting table doubles as the boilerplate table the chunker strips with.

Usage:
    python scripts/audit_corpus.py
    python scripts/audit_corpus.py --corpus data/processed/parquet_corpus \
        --out data/processed/corpus_audit.parquet --min-chars 400
    python scripts/audit_corpus.py --rebuild-boilerplate   # after a corpus change
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from loguru import logger  # noqa: E402

from src.ingestion.boilerplate import (  # noqa: E402
    BoilerplateCounter,
    load_table,
    save_table,
    strip_boilerplate,
)
from src.ingestion.corpus_audit import (  # noqa: E402
    decide_verdict,
    head_fingerprint,
    is_binary,
    language_mismatch,
    load_medical_vocabularies,
    looks_like_payload,
    medical_hits,
    normalize_text,
    resolve_duplicates,
    short_line_char_ratio,
    template_title_kind,
)

SIGNAL_COLUMNS = [
    "doc_id",
    "url",
    "domain",
    "lang",
    "title",
    "char_count",
    "raw_char_count",
    "boilerplate_share",
    "is_binary",
    "payload_like",
    "template_title",
    "lang_mismatch",
    "med_hits",
    "med_density",
    "short_line_ratio",
    "fingerprint",
    "dup_of",
    "verdict",
    "reasons",
]


def corpus_files(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No .parquet corpus files found at {path}")
    return files


def iter_text_batches(files: list[Path], batch_size: int, wanted=("text", "domain")):
    """Yields (file, batch_dict, row_count, available_columns) so both passes share one walk."""
    for file in files:
        parquet_file = pq.ParquetFile(file)
        available = set(parquet_file.schema_arrow.names)
        columns = [c for c in wanted if c in available]
        if "text" not in columns:
            logger.warning(f"{file.name}: no text column, skipped")
            continue
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            yield file, batch.to_pydict(), batch.num_rows, available


def build_boilerplate(files, batch_size, min_share, min_docs) -> dict[str, set[int]]:
    counter = BoilerplateCounter(min_share=min_share, min_docs=min_docs)
    for index, (file, data, n_rows, _) in enumerate(iter_text_batches(files, batch_size), 1):
        domains = data.get("domain", [""] * n_rows)
        for text, domain in zip(data["text"], domains):
            counter.add(domain or "", text or "")
        if index % 2 == 0:
            logger.info(f"Boilerplate pass: {counter.n_docs.total():,} docs counted")
    table = counter.selected()
    logger.info(
        f"Boilerplate table: {len(table):,} domains, {sum(len(v) for v in table.values()):,} lines"
    )
    return table


def scan_corpus(
    files: list[Path],
    vocab: dict[str, set[str]],
    batch_size: int,
    boilerplate: dict[str, set[int]],
) -> dict[str, list]:
    rows: dict[str, list] = {column: [] for column in SIGNAL_COLUMNS}
    statuses: list[str] = []

    for file, data, n_rows, available in iter_text_batches(
        files, batch_size, wanted=("doc_id", "url", "title", "text", "lang", "domain", "status")
    ):
        for i in range(n_rows):
            raw = data["text"][i] or ""
            domain = (data.get("domain") or [""] * n_rows)[i] or ""
            lang = (data.get("lang") or ["vi"] * n_rows)[i] or "vi"
            title = (data.get("title") or [""] * n_rows)[i] or ""

            stripped = strip_boilerplate(raw, boilerplate.get(domain))
            raw_len = len(normalize_text(raw))
            text_len = len(normalize_text(stripped))
            hits = medical_hits(stripped, lang, vocab)

            rows["doc_id"].append(str(data["doc_id"][i]))
            rows["url"].append((data.get("url") or [""] * n_rows)[i] or "")
            rows["domain"].append(domain)
            rows["lang"].append(lang)
            rows["title"].append(title)
            rows["char_count"].append(text_len)
            rows["raw_char_count"].append(raw_len)
            rows["boilerplate_share"].append(
                round(1 - text_len / raw_len, 3) if raw_len else 0.0
            )
            rows["is_binary"].append(is_binary(raw))
            rows["payload_like"].append(looks_like_payload(raw))
            rows["template_title"].append(template_title_kind(title))
            rows["lang_mismatch"].append(language_mismatch(stripped or raw, lang))
            rows["med_hits"].append(hits)
            rows["med_density"].append(round(hits / (text_len / 1000), 3) if text_len else 0.0)
            rows["short_line_ratio"].append(round(short_line_char_ratio(stripped), 3))
            rows["fingerprint"].append(head_fingerprint(stripped))
            rows["dup_of"].append(None)
            rows["verdict"].append("keep")
            rows["reasons"].append([])
            statuses.append((data.get("status") or [None] * n_rows)[i])
        logger.info(f"Scanned {file.name}: total docs so far {len(rows['doc_id']):,}")

    rows["_status"] = statuses
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate a per-document corpus quality table.")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/parquet_corpus"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/corpus_audit.parquet"))
    parser.add_argument(
        "--slim-out",
        type=Path,
        default=Path("data/processed/corpus_audit_slim.parquet"),
        help="Compact doc_id/verdict/reasons table; the index build filters on this.",
    )
    parser.add_argument("--lexicon-dir", type=Path, default=Path("data/lexicon"))
    parser.add_argument(
        "--boilerplate-table",
        type=Path,
        default=Path("data/processed/boilerplate_lines.parquet"),
    )
    parser.add_argument(
        "--rebuild-boilerplate",
        action="store_true",
        help="Recount boilerplate instead of reusing the cached table.",
    )
    parser.add_argument("--no-boilerplate", action="store_true", help="Skip stripping entirely.")
    parser.add_argument(
        "--boilerplate-min-share",
        type=float,
        default=0.01,
        help="Share of a domain's documents a line must appear in to count as chrome.",
    )
    parser.add_argument("--boilerplate-min-docs", type=int, default=20)
    parser.add_argument(
        "--boilerplate-only",
        action="store_true",
        help="Count boilerplate, write the table, and stop before the quality pass.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=400,
        help="Below this a document is flagged for review (too_short), not dropped.",
    )
    parser.add_argument(
        "--min-chars-hard",
        type=int,
        default=60,
        help="Below this a document is chrome or an empty shell and is dropped (fragment). "
        "Structured 方剂 entries are complete in ~100 chars, so this stays low.",
    )
    parser.add_argument(
        "--min-med-density",
        type=float,
        default=0.0,
        help="Review documents whose medical-term hits per 1k chars fall at or below this.",
    )
    parser.add_argument(
        "--max-short-line-ratio",
        type=float,
        default=0.6,
        help="Review documents whose characters are mostly on short (link-like) lines.",
    )
    parser.add_argument("--batch-size", type=int, default=20000)
    args = parser.parse_args()

    files = corpus_files(args.corpus)
    vocab = load_medical_vocabularies(args.lexicon_dir)
    logger.info(
        f"Auditing {len(files)} shard(s) with vocabularies vi={len(vocab['vi']):,} "
        f"en={len(vocab['en']):,}"
    )

    boilerplate: dict[str, set[int]] = {}
    if not args.no_boilerplate:
        if args.boilerplate_table.exists() and not args.rebuild_boilerplate:
            boilerplate = load_table(args.boilerplate_table)
            logger.info(
                f"Loaded boilerplate table {args.boilerplate_table} "
                f"({len(boilerplate):,} domains, {sum(len(v) for v in boilerplate.values()):,} lines)"
            )
        else:
            boilerplate = build_boilerplate(
                files, args.batch_size, args.boilerplate_min_share, args.boilerplate_min_docs
            )
            save_table(boilerplate, args.boilerplate_table)
            logger.info(f"Wrote boilerplate table to {args.boilerplate_table}")

    if args.boilerplate_only:
        ranked = sorted(boilerplate.items(), key=lambda kv: -len(kv[1]))
        print(f"\nboilerplate table: {len(boilerplate):,} domains, "
              f"{sum(len(v) for v in boilerplate.values()):,} lines")
        for domain, hashes in ranked[:25]:
            print(f"  {domain:34s} {len(hashes):>6,} lines")
        return

    rows = scan_corpus(files, vocab, args.batch_size, boilerplate)
    doc_ids, sizes, fingerprints = rows["doc_id"], rows["char_count"], rows["fingerprint"]
    canonical = resolve_duplicates(doc_ids, sizes, fingerprints)

    verdicts, reason_lists = [], []
    for i, doc_id in enumerate(doc_ids):
        signals = {
            "doc_id": doc_id,
            "status": rows["_status"][i],
            "is_binary": rows["is_binary"][i],
            "payload_like": rows["payload_like"][i],
            "char_count": rows["char_count"][i],
            "template_title": rows["template_title"][i],
            "lang_mismatch": rows["lang_mismatch"][i],
            "med_hits": rows["med_hits"][i],
            "med_density": rows["med_density"][i],
            "short_line_ratio": rows["short_line_ratio"][i],
        }
        dup_of = canonical.get(doc_id)
        verdict, reasons = decide_verdict(
            signals,
            dup_of,
            min_chars=args.min_chars,
            min_chars_hard=args.min_chars_hard,
            min_med_density=args.min_med_density,
            max_short_line_ratio=args.max_short_line_ratio,
        )
        rows["dup_of"][i] = dup_of if dup_of != doc_id else None
        verdicts.append(verdict)
        reason_lists.append(reasons)

    rows["verdict"] = verdicts
    rows["reasons"] = reason_lists
    rows.pop("_status")

    table = pa.table(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out, compression="zstd")
    logger.info(f"Wrote {table.num_rows:,} rows to {args.out}")

    slim = pa.table({"doc_id": doc_ids, "verdict": verdicts, "reasons": reason_lists})
    pq.write_table(slim, args.slim_out, compression="zstd")
    logger.info(f"Wrote {slim.num_rows:,} rows to {args.slim_out}")

    total = table.num_rows
    counts = Counter(verdicts)
    print("\nverdicts")
    for verdict in ("keep", "review", "drop"):
        n = counts.get(verdict, 0)
        print(f"  {verdict:7s} {n:>9,}  {n / total:6.2%}")

    print("\nreasons")
    reason_counts = Counter(reason.split(":")[0] for reasons in reason_lists for reason in reasons)
    for reason, n in reason_counts.most_common():
        print(f"  {reason:28s} {n:>9,}  {n / total:6.2%}")

    if boilerplate:
        shares = rows["boilerplate_share"]
        touched = sum(1 for share in shares if share > 0)
        trimmed = sum(int(rows["raw_char_count"][i] - rows["char_count"][i]) for i in range(total))
        print("\nboilerplate")
        print(
            f"  {len(boilerplate):,} domains, {sum(len(v) for v in boilerplate.values()):,} lines, "
            f"{touched:,} docs touched ({touched / total:.1%}), {trimmed:,} chars removed"
        )
        domain_trim: dict[str, list[float]] = {}
        for domain, share in zip(rows["domain"], shares):
            domain_trim.setdefault(domain, []).append(share)
        ranked_trim = sorted(
            ((d, sum(v) / len(v)) for d, v in domain_trim.items() if len(v) >= 100),
            key=lambda kv: -kv[1],
        )[:10]
        for domain, mean_share in ranked_trim:
            print(f"  {domain:34s} mean share removed {mean_share:5.1%}")

    print("\nworst domains by drop+review share")
    per_domain: dict[str, Counter] = {}
    for domain, verdict in zip(rows["domain"], verdicts):
        per_domain.setdefault(domain, Counter())[verdict] += 1
    ranked = sorted(
        per_domain.items(),
        key=lambda kv: -(kv[1]["drop"] + kv[1]["review"]) / max(sum(kv[1].values()), 1),
    )[:15]
    for domain, counter in ranked:
        n = sum(counter.values())
        bad = counter["drop"] + counter["review"]
        print(
            f"  {domain:34s} {bad:>7,}/{n:<7,} {bad / n:6.1%}  (drop {counter['drop']:,}, review {counter['review']:,})"
        )


if __name__ == "__main__":
    main()

