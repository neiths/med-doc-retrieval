"""Corpus-level boilerplate lines: chrome a site repeats on most of its documents.

The regex patterns in the cleaner catch known ad and footer phrasings. This catches the
rest by frequency: a short line that shows up in a large share of a single domain's
documents is navigation, a breadcrumb tail or an app-download block, not content — no
per-domain rules needed.

Boilerplate is identified in one streaming pass (``BoilerplateCounter``), persisted as a
domain -> line-hash parquet table, and stripped before any length or density signal is
measured.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path

# The cap admits footer paragraphs (a fixed byline or "related articles" block is one long
# line) while still excluding article body text, which rarely repeats verbatim.
MAX_LINE_CHARS = 400
MIN_LINE_CHARS = 2
# Sites serve several templates, so one chrome variant can sit on a tenth of a domain's
# pages (120ask: nav on 10.9%). The share is deliberately low; min_docs keeps tiny domains out.
MIN_SHARE = 0.01
MIN_DOCS = 20
# A line seen once can never reach MIN_DOCS; dropping those singles keeps big domains flat.
PRUNE_AT_DOCS = 5000
PRUNE_MIN_ENTRIES = 200_000

_BLANK_RE = re.compile(r"[^\S\n]+")


def normalize_line(line: str) -> str:
    return _BLANK_RE.sub(" ", line or "").strip()


def line_hash(line: str) -> int:
    return int.from_bytes(hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest(), "big")


def doc_line_hashes(text: str) -> set[int]:
    """Hashes of a document's short lines; nav chrome is short, prose paragraphs are not."""
    hashes: set[int] = set()
    for line in (text or "").split("\n"):
        normalized = normalize_line(line)
        if MIN_LINE_CHARS <= len(normalized) <= MAX_LINE_CHARS:
            hashes.add(line_hash(normalized))
    return hashes


class BoilerplateCounter:
    """Per-domain document frequency of short lines, accumulated in one streaming pass."""

    def __init__(self, min_share: float = MIN_SHARE, min_docs: int = MIN_DOCS):
        self.min_share = min_share
        self.min_docs = min_docs
        self.counts: dict[str, Counter[int]] = {}
        self.n_docs: Counter[str] = Counter()

    def add(self, domain: str, text: str) -> None:
        domain = domain or ""
        counts = self.counts.setdefault(domain, Counter())
        for digest in doc_line_hashes(text):
            counts[digest] += 1
        self.n_docs[domain] += 1
        if len(counts) > PRUNE_MIN_ENTRIES and self.n_docs[domain] % PRUNE_AT_DOCS == 0:
            self.counts[domain] = Counter({h: c for h, c in counts.items() if c > 1})

    def selected(self) -> dict[str, set[int]]:
        table: dict[str, set[int]] = {}
        for domain, counts in self.counts.items():
            n_docs = self.n_docs[domain]
            threshold = max(self.min_docs, int(self.min_share * n_docs))
            lines = {digest for digest, seen in counts.items() if seen >= threshold}
            if lines:
                table[domain] = lines
        return table

    def stats(self) -> dict[str, tuple[int, int]]:
        """domain -> (documents seen, selected lines), for logging."""
        table = self.selected()
        return {d: (self.n_docs[d], len(h)) for d, h in table.items()}


def strip_boilerplate(text: str, hashes: set[int] | None) -> str:
    """Drops lines whose hash is in the domain's boilerplate set; blank lines survive."""
    if not text or not hashes:
        return text or ""
    kept = []
    for line in text.split("\n"):
        normalized = normalize_line(line)
        if MIN_LINE_CHARS <= len(normalized) <= MAX_LINE_CHARS and line_hash(normalized) in hashes:
            continue
        kept.append(line)
    return "\n".join(kept)


def save_table(table: dict[str, set[int]], path: str | Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    domains: list[str] = []
    digests: list[int] = []
    for domain in sorted(table):
        hashes = sorted(table[domain])
        domains.extend([domain] * len(hashes))
        digests.extend(hashes)
    table_arrow = pa.table(
        {
            "domain": pa.array(domains, pa.string()),
            "line_hash": pa.array(digests, pa.uint64()),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table_arrow, path, compression="zstd")


def load_table(path: str | Path) -> dict[str, set[int]]:
    import pyarrow.parquet as pq

    columns = pq.read_table(Path(path), columns=["domain", "line_hash"]).to_pydict()
    table: dict[str, set[int]] = {}
    for domain, digest in zip(columns["domain"], columns["line_hash"]):
        table.setdefault(domain, set()).add(int(digest))
    return table
