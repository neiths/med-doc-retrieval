"""Master CLI crawler for all missing Vietnamese medical websites.

Pre-configured host groups based on content archetype and priority:
- clinical:   Bệnh viện & Chuyên khoa (medlatec, tamanhhospital, hellobacsi, youmed, tiemchunglongchau)
- doisong:    Bách khoa bệnh học Bộ Y Tế (suckhoedoisong.vn - 78k URLs)
- news:       Báo chí y tế chính thống (thanhnien.vn, laodong.vn, vietnamnet.vn, vov.vn - 115k URLs)
- community:  Sức Khỏe Cộng Đồng (suckhoecongdongonline.vn - 147k URLs)
- provincial: Báo địa phương (baonghean, baohaiphong, baodanang, baocantho, baoangiang...)
- all:        Toàn bộ các trang tiếng Việt còn thiếu (~527k URLs)

Usage:
    # 1. Chạy nhóm bệnh viện & chuyên khoa (test 50 bài):
    python scripts/crawl_vietnamese.py --group clinical --limit 50

    # 2. Chạy Sức Khỏe Đời Sống (mỗi 5 phút cào 200 bài):
    python scripts/crawl_vietnamese.py --group doisong --limit 200 --interval-minutes 5

    # 3. Chạy nhóm báo chí y tế nền:
    python scripts/crawl_vietnamese.py --group news --concurrency 15
"""

import argparse
import subprocess
import sys
from pathlib import Path

GROUPS = {
    "clinical": [
        "medlatec.vn",
        "tamanhhospital.vn",
        "hellobacsi.com",
        "youmed.vn",
        "tiemchunglongchau.com.vn",
        "bachmai.gov.vn",
    ],
    "doisong": [
        "suckhoedoisong.vn",
    ],
    "longchau": [
        "nhathuoclongchau.com.vn",
    ],
    "news": [
        "thanhnien.vn",
        "laodong.vn",
        "vietnamnet.vn",
        "vov.vn",
        "vov2.vov.vn",
        "tuoitre.vn",
        "nhandan.vn",
        "tienphong.vn",
    ],
    "community": [
        "suckhoecongdongonline.vn",
        "phunusuckhoe.giadinhonline.vn",
        "giadinhonline.vn",
        "suckhoeviet.org.vn",
    ],
    "provincial": [
        "baohaiphong.vn",
        "baonghean.vn",
        "baodanang.vn",
        "baocantho.com.vn",
        "baoangiang.com.vn",
        "baogialai.com.vn",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--group",
        type=str,
        default="clinical",
        choices=["clinical", "doisong", "longchau", "news", "community", "provincial", "all"],
        help="Target Vietnamese host group to crawl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of URLs per batch (0 for all).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help="Global concurrency (default: 12).",
    )
    parser.add_argument(
        "--domain-concurrency",
        type=int,
        default=2,
        help="Max concurrency per domain (default: 2).",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=45.0,
        help="Circuit-breaker pause duration when a domain triggers 429/403 (default: 45s).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries before marking an article permanently failed (default: 3).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Number of articles per JSONL output file before rotating (default: 5000).",
    )
    parser.add_argument(
        "--interval-minutes",
        type=float,
        default=0.0,
        help="[Deprecated] Translate to domain cooldown seconds if specified.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/v2/crawled"),
        help="Output directory for crawled JSONL shards.",
    )

    args = parser.parse_args()

    if args.group == "all":
        target_hosts = []
        for h_list in GROUPS.values():
            target_hosts.extend(h_list)
    else:
        target_hosts = GROUPS[args.group]

    cooldown = args.cooldown_seconds
    if args.interval_minutes > 0:
        cooldown = args.interval_minutes * 60.0

    print(f"=== Crawling Vietnamese Group: '{args.group}' ===")
    print(f"Target hosts ({len(target_hosts)}): {', '.join(target_hosts)}")

    cmd = [
        sys.executable,
        "scripts/crawl_missing_urls.py",
        "--hosts", *target_hosts,
        "--concurrency", str(args.concurrency),
        "--domain-concurrency", str(args.domain_concurrency),
        "--cooldown-seconds", str(cooldown),
        "--max-retries", str(args.max_retries),
        "--shard-size", str(args.shard_size),
        "--out-dir", str(args.out_dir),
    ]

    if args.limit > 0:
        cmd.extend(["--limit", str(args.limit)])

    # Forward execution to crawl_missing_urls.py
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
