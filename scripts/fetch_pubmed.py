"""B1 - Fetch the 240k official PubMed documents via NCBI E-utilities (efetch XML).

No HTML crawling: PMIDs are parsed from the official URLs and fetched in batches.
Output text keeps the abstract structure (BACKGROUND / METHODS / ...) as headings,
so structure-aware chunking can split by section later.

Resumable: already-fetched ids in the output dir are skipped.
Rate limits: 3 req/s without key, 10 req/s with NCBI_API_KEY (free at ncbi.nlm.nih.gov/account).

Usage:
    export NCBI_API_KEY=...   # optional but recommended
    python scripts/fetch_pubmed.py \
        --links data/raw/vibio_mir/links_corpus_vi_en_zh.parquet \
        --out-dir data/processed/v2/pubmed
"""

import argparse
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
RE_PMID = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


class RateLimiter:
    def __init__(self, rate_per_sec: float):
        self.interval = 1.0 / rate_per_sec
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_t:
                time.sleep(self.next_t - now)
            self.next_t = max(now, self.next_t) + self.interval


def itertext(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def parse_date(article_el) -> str:
    """Returns ISO date (YYYY-MM-DD / YYYY-MM / YYYY) or ''. Never guesses missing parts."""
    for path in [".//ArticleDate", ".//JournalIssue/PubDate"]:
        d = article_el.find(path)
        if d is None:
            continue
        y = (d.findtext("Year") or "").strip()
        if not y:
            md = (d.findtext("MedlineDate") or "").strip()
            m = re.match(r"(\d{4})", md)
            return m.group(1) if m else ""
        mo = (d.findtext("Month") or "").strip()
        dd = (d.findtext("Day") or "").strip()
        if mo and not mo.isdigit():
            mo = str(MONTHS.get(mo[:3].lower(), "")) if mo[:3].lower() in MONTHS else ""
        if mo:
            return f"{y}-{int(mo):02d}" + (f"-{int(dd):02d}" if dd.isdigit() else "")
        return y
    return ""


def parse_article(pa_el) -> dict | None:
    cit = pa_el.find("MedlineCitation")
    if cit is None:
        return None
    pmid = (cit.findtext("PMID") or "").strip()
    art = cit.find("Article")
    if art is None or not pmid:
        return None
    title = itertext(art.find("ArticleTitle"))

    sections = []
    for ab in art.findall("Abstract/AbstractText"):
        label = (ab.get("Label") or "").strip()
        txt = itertext(ab)
        if txt:
            sections.append((label, txt))
    # Some records only have OtherAbstract
    if not sections:
        for ab in cit.findall("OtherAbstract/AbstractText"):
            txt = itertext(ab)
            if txt:
                sections.append(((ab.get("Label") or "").strip(), txt))

    mesh = [itertext(m.find("DescriptorName")) for m in cit.findall("MeshHeadingList/MeshHeading")]
    keywords = [itertext(k) for k in cit.findall("KeywordList/Keyword")]
    journal = itertext(art.find("Journal/Title"))
    lang = (art.findtext("Language") or "").strip()

    # Body text: title line + abstract with section headings preserved (markdown-like)
    parts = [title] if title else []
    for label, txt in sections:
        parts.append(f"## {label}\n{txt}" if label else txt)
    text = "\n\n".join(parts)

    return {
        "pmid": pmid,
        "title": title,
        "text": text,
        "has_abstract": bool(sections),
        "journal": journal,
        "published": parse_date(art),
        "article_lang": lang,
        "mesh": mesh,
        "keywords": keywords,
    }


def fetch_batch(client: httpx.Client, limiter: RateLimiter, pmids: list[str], api_key: str | None,
                retries: int = 5) -> list[dict]:
    data = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": "vibiomir", "email": "team@r2ai2026.org"}
    if api_key:
        data["api_key"] = api_key
    for attempt in range(retries):
        limiter.wait()
        try:
            r = client.post(EFETCH_URL, data=data)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            out = []
            for pa_el in root.findall("PubmedArticle"):
                rec = parse_article(pa_el)
                if rec:
                    out.append(rec)
            return out
        except Exception as e:
            sleep = 2 ** attempt
            logger.warning(f"Batch of {len(pmids)} failed ({e}); retry {attempt + 1}/{retries} in {sleep}s")
            time.sleep(sleep)
    logger.error(f"Giving up on batch starting with PMID {pmids[0]}")
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--links", type=Path, default=Path("data/raw/vibio_mir/links_corpus_vi_en_zh.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/v2/pubmed"))
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="Only fetch first N ids (for testing).")
    args = ap.parse_args()

    api_key = os.environ.get("NCBI_API_KEY")
    rate = 9.0 if api_key else 2.8
    logger.info(f"NCBI API key: {'yes' if api_key else 'no'} -> {rate} req/s")

    links = pq.read_table(args.links, columns=["id", "url"])
    pmid_to_id: dict[str, int] = {}
    for did, url in zip(links["id"].to_pylist(), links["url"].to_pylist()):
        m = RE_PMID.search(url or "")
        if m:
            pmid_to_id[m.group(1)] = did
    logger.info(f"PubMed URLs in official list: {len(pmid_to_id):,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    for f in args.out_dir.glob("pubmed_*.jsonl"):
        with open(f, encoding="utf-8") as fp:
            for line in fp:
                try:
                    done.add(json.loads(line)["pmid"])
                except Exception:
                    continue
    todo = [p for p in pmid_to_id if p not in done]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(f"Already fetched: {len(done):,} | To fetch: {len(todo):,}")
    if not todo:
        return

    batches = [todo[i: i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    out_path = args.out_dir / f"pubmed_{int(time.time())}.jsonl"
    limiter = RateLimiter(rate)
    write_lock = threading.Lock()
    n_ok = n_noabs = 0

    with httpx.Client(timeout=httpx.Timeout(60.0)) as client, open(out_path, "a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch_batch, client, limiter, b, api_key) for b in batches]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="efetch batches"):
            recs = fut.result()
            with write_lock:
                for rec in recs:
                    did = pmid_to_id.get(rec["pmid"])
                    if did is None:
                        continue
                    row = {
                        "doc_id": did,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{rec['pmid']}/",
                        "lang": "en",
                        "domain": "pubmed.ncbi.nlm.nih.gov",
                        "doc_type": "abstract",
                        **rec,
                    }
                    row["char_count"] = len(row["text"])
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_ok += 1
                    n_noabs += 0 if rec["has_abstract"] else 1
                fout.flush()

    logger.info(f"Done. Wrote {n_ok:,} records to {out_path} ({n_noabs:,} title-only, no abstract).")
    logger.info(f"Missing after this run: {len(todo) - n_ok:,} (re-run to retry).")


if __name__ == "__main__":
    main()
