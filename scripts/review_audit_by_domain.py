"""Reviews the corpus audit table domain by domain.

Shows how keep/review/drop landed per site, plus sample dropped documents so the rules
can be spot-checked instead of trusted.

Usage:
    python scripts/review_audit_by_domain.py
    python scripts/review_audit_by_domain.py --min-docs 100 --samples 3
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq  # noqa: E402


def primary_reason(reasons: list[str]) -> str:
    """First reason that actually removes content; duplicate links are reported separately."""
    for reason in reasons or []:
        if not reason.startswith("duplicate_of"):
            return reason.split(":")[0]
    return "duplicate_of" if reasons else ""


def main():
    parser = argparse.ArgumentParser(description="Per-domain review of the corpus audit.")
    parser.add_argument("--audit", type=Path, default=Path("data/processed/corpus_audit.parquet"))
    parser.add_argument("--min-docs", type=int, default=50, help="Hide domains smaller than this.")
    parser.add_argument("--top", type=int, default=40, help="Rows to print, largest domains first.")
    parser.add_argument("--samples", type=int, default=0, help="Sample dropped docs per domain.")
    args = parser.parse_args()

    columns = ["domain", "verdict", "char_count", "reasons"]
    if args.samples:
        columns += ["title", "url"]
    rows = pq.read_table(args.audit, columns=columns).to_pydict()

    total = len(rows["domain"])
    stats: dict[str, Counter] = defaultdict(Counter)
    reasons: dict[str, Counter] = defaultdict(Counter)
    chars: dict[str, list[int]] = defaultdict(list)
    samples: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for i, domain in enumerate(rows["domain"]):
        verdict = rows["verdict"][i]
        stats[domain][verdict] += 1
        stats[domain]["docs"] += 1
        chars[domain].append(rows["char_count"][i])
        if verdict != "drop":
            continue
        reason = primary_reason(rows["reasons"][i])
        reasons[domain][reason] += 1
        if args.samples and len(samples[domain]) < args.samples:
            samples[domain].append((reason, rows["title"][i][:70], rows["url"][i][:90]))

    kept = sum(c["keep"] for c in stats.values())
    print(f"docs {total:,} | keep {kept:,} ({kept / total:.2%}) | domains {len(stats)}")
    print(
        f"{'domain':34s} {'docs':>8s} {'keep%':>6s} {'review%':>8s} {'drop%':>6s} "
        f"{'med chars':>9s}  top drop reason"
    )
    ranked = sorted(
        ((d, c) for d, c in stats.items() if c["docs"] >= args.min_docs),
        key=lambda kv: -kv[1]["docs"],
    )
    for domain, counter in ranked[: args.top]:
        n = counter["docs"]
        median = sorted(chars[domain])[len(chars[domain]) // 2]
        top_reason = reasons[domain].most_common(1)
        reason_text = f"{top_reason[0][0]} ({top_reason[0][1]:,})" if top_reason else ""
        print(
            f"{domain:34s} {n:>8,} {counter['keep'] / n:>6.1%} {counter['review'] / n:>8.1%} "
            f"{counter['drop'] / n:>6.1%} {median:>9,}  {reason_text}"
        )

    if args.samples:
        print("\ndropped samples (spot-check these):")
        for domain, items in sorted(samples.items(), key=lambda kv: -stats[kv[0]]["docs"])[:8]:
            print(f"\n  {domain} ({stats[domain]['docs']:,} docs, {stats[domain]['drop']:,} dropped)")
            for reason, title, url in items:
                print(f"    [{reason}] {title!r}  {url}")


if __name__ == "__main__":
    main()
