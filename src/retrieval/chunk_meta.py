"""Shared SQLite chunk metadata accessor used by the dense and sparse indices.

Two layouts exist:
* Current builds store a 0-based chunk counter as ``chunk_id INTEGER PRIMARY KEY``
  and index ``doc_id``, so a document's chunks are fetched with one indexed query.
* Legacy builds use ``idx`` as the row position and textual ``{doc}__c{n}`` ids.
  There, chunks are still contiguous per document, so document boundaries are
  resolved once into position ranges and fetched with primary-key range scans.
"""

import sqlite3
from collections.abc import Iterable, Sequence
from typing import Any

from loguru import logger

from src.ingestion.chunker import build_contextual_text

_TOKEN = r"\w+"


def _tokenize(text: str) -> set[str]:
    import re

    return set(re.findall(_TOKEN, text.lower(), flags=re.UNICODE))


class ChunkMetaStore:
    """Read-only accessor over the ``chunks`` metadata table."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.cur = conn.cursor()
        tables = {
            row[0]
            for row in self.cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "chunks" not in tables:
            raise ValueError("SQLite metadata database has no 'chunks' table")
        self.columns = [r[1] for r in self.cur.execute("PRAGMA table_info(chunks)").fetchall()]
        self.has_url = "url" in self.columns
        self.has_contextual = "contextual_text" in self.columns
        self.pos_col = self._detect_pos_col()
        self._doc_indexed = self._has_doc_index()
        self._ranges: dict[str, list[tuple[int, int]]] | None = None
        self._chunk_id_positions: dict[str, int] | None = None

    def _has_doc_index(self) -> bool:
        """True when SQLite can resolve a doc_id without scanning the chunk table."""
        try:
            for _, idx_name, *_ in self.cur.execute("PRAGMA index_list(chunks)").fetchall():
                cols = [r[2] for r in self.cur.execute(f'PRAGMA index_info("{idx_name}")')]
                if cols and cols[0] == "doc_id":
                    return True
        except sqlite3.Error:
            return False
        return False

    def _detect_pos_col(self) -> str:
        """Finds the column that FAISS/BM25 row positions map to."""
        for col in ("idx", "position"):
            if col in self.columns:
                try:
                    if self.cur.execute(f"SELECT 1 FROM chunks WHERE {col} = 0 LIMIT 1").fetchone():
                        return col
                except sqlite3.Error:
                    continue
        # Some slim builds store the numeric position in chunk_id.
        if "chunk_id" in self.columns:
            try:
                if self.cur.execute("SELECT 1 FROM chunks WHERE chunk_id = 0 LIMIT 1").fetchone():
                    return "chunk_id"
            except sqlite3.Error:
                pass
        return "idx" if "idx" in self.columns else "chunk_id"

    def _select_sql(self) -> str:
        url = ", url" if self.has_url else ""
        ctx = ", contextual_text" if self.has_contextual else ""
        return f"SELECT {self.pos_col}, doc_id, chunk_id, chunk_text, title, lang{ctx}{url} FROM chunks"

    def _row_to_meta(self, row: Sequence[Any]) -> dict[str, Any]:
        pos, doc_id, chunk_id, raw_chunk, title, lang = row[:6]
        raw_chunk = raw_chunk or ""
        next_col = 6
        ctx_text = None
        if self.has_contextual:
            ctx_text = row[next_col] or None
            next_col += 1
        url = (row[next_col] if self.has_url and len(row) > next_col else "") or ""
        if ctx_text is None:
            if raw_chunk.startswith("Tiêu đề:") and "\nNội dung: " in raw_chunk:
                ctx_text = raw_chunk
                raw_chunk = raw_chunk.split("\nNội dung: ", 1)[-1].strip()
            else:
                ctx_text = build_contextual_text(raw_chunk, title=title, lang=lang or "vi")
        return {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_text": raw_chunk,
            "contextual_text": ctx_text,
            "title": title or "",
            "lang": lang or "vi",
            "url": url,
        }

    def _build_ranges(self) -> None:
        logger.info("Building document -> chunk position ranges from SQLite metadata...")
        last_pos = self.cur.execute(f"SELECT MAX({self.pos_col}) FROM chunks").fetchone()[0]
        if last_pos is None:
            self._ranges = {}
            return

        sql = f"""
            WITH ordered AS (
                SELECT {self.pos_col} AS pos, doc_id,
                       LAG(doc_id) OVER (ORDER BY {self.pos_col}) AS prev_doc
                FROM chunks
            )
            SELECT doc_id, pos FROM ordered
            WHERE prev_doc IS NULL OR prev_doc <> doc_id
            ORDER BY pos
        """
        try:
            boundaries = self.cur.execute(sql).fetchall()
        except sqlite3.OperationalError:
            boundaries = self._scan_boundaries()

        ranges: dict[str, list[tuple[int, int]]] = {}
        for i, (doc_id, start) in enumerate(boundaries):
            end = boundaries[i + 1][1] - 1 if i + 1 < len(boundaries) else int(last_pos)
            ranges.setdefault(str(doc_id), []).append((int(start), int(end)))
        self._ranges = ranges
        logger.info(f"Indexed {len(ranges):,} documents into chunk position ranges.")

    def _scan_boundaries(self) -> list[tuple[Any, int]]:
        """Fallback for SQLite builds without window function support."""
        boundaries: list[tuple[Any, int]] = []
        current: str | None = None
        for pos, doc_id in self.cur.execute(
            f"SELECT {self.pos_col}, doc_id FROM chunks ORDER BY {self.pos_col}"
        ):
            key = str(doc_id)
            if key != current:
                boundaries.append((doc_id, int(pos)))
                current = key
        return boundaries

    def _ensure_ranges(self) -> dict[str, list[tuple[int, int]]]:
        if self._ranges is None:
            self._build_ranges()
        return self._ranges

    def get_by_positions(self, positions: Iterable[int]) -> dict[int, dict[str, Any]]:
        """Batch-fetches chunk metadata keyed by position column value."""
        wanted = [int(p) for p in positions]
        if not wanted:
            return {}
        found: dict[int, dict[str, Any]] = {}
        batch_size = 900
        for i in range(0, len(wanted), batch_size):
            batch = wanted[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            sql = f"{self._select_sql()} WHERE {self.pos_col} IN ({placeholders})"
            for row in self.cur.execute(sql, batch):
                found[int(row[0])] = self._row_to_meta(row)
        return found

    def position_of(self, chunk_id: Any) -> int | None:
        """Resolves a ``{doc_id}__c{n}`` chunk id to its position without scanning."""
        cid = str(chunk_id)
        if self.pos_col == "chunk_id" and cid.isdigit():
            # Current layout: the chunk id is the 0-based row position.
            return int(cid)
        doc_part, sep, offset = cid.rpartition("__c")
        if sep and offset.isdigit():
            remaining = int(offset)
            for start, end in self._ensure_ranges().get(doc_part, ()):
                span_len = end - start + 1
                if remaining < span_len:
                    return start + remaining
                remaining -= span_len
            return None

        if self._chunk_id_positions is None:
            logger.info("Chunk ids are not positional; building chunk_id -> position map...")
            self._chunk_id_positions = {
                str(row[1]): int(row[0])
                for row in self.cur.execute(
                    f"SELECT {self.pos_col}, chunk_id FROM chunks ORDER BY {self.pos_col}"
                )
            }
        return self._chunk_id_positions.get(cid)

    def get_by_chunk_ids(self, chunk_ids: Iterable[Any]) -> dict[str, dict[str, Any]]:
        """Batch-fetches metadata for chunk ids, preserving id order in the result."""
        positions: dict[str, int] = {}
        for cid in chunk_ids:
            pos = self.position_of(cid)
            if pos is not None:
                positions[str(cid)] = pos
        by_pos = self.get_by_positions(positions.values())
        return {cid: by_pos[pos] for cid, pos in positions.items() if pos in by_pos}

    def _doc_chunks_indexed(self, doc_id: Any) -> list[dict[str, Any]]:
        sql = (
            f"{self._select_sql()} WHERE doc_id = ? ORDER BY {self.pos_col}"
        )
        return [self._row_to_meta(row) for row in self.cur.execute(sql, (doc_id,))]

    def get_doc_chunks(
        self,
        doc_ids: Iterable[Any],
        query: str | None = None,
        max_per_doc: int | None = None,
    ) -> list[dict[str, Any]]:
        """Returns metadata for every chunk of each requested document."""
        indexed = self._doc_indexed
        if not indexed:
            ranges = self._ensure_ranges()
        results: list[dict[str, Any]] = []
        for doc_id in doc_ids:
            if indexed:
                doc_chunks = self._doc_chunks_indexed(doc_id)
            else:
                spans = ranges.get(str(doc_id))
                if not spans:
                    continue
                doc_chunks = []
                for start, end in spans:
                    sql = f"{self._select_sql()} WHERE {self.pos_col} BETWEEN ? AND ? ORDER BY {self.pos_col}"
                    doc_chunks.extend(
                        self._row_to_meta(row) for row in self.cur.execute(sql, (start, end))
                    )
            if not doc_chunks:
                continue
            if max_per_doc is not None and len(doc_chunks) > max_per_doc:
                doc_chunks = self._select_by_lexical_overlap(doc_chunks, query, max_per_doc)
            results.extend(doc_chunks)
        return results

    @staticmethod
    def _select_by_lexical_overlap(
        chunks: list[dict[str, Any]], query: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """Keeps the highest query-token-overlap chunks for long documents."""
        if not query:
            return chunks[:limit]
        query_tokens = _tokenize(query)
        if not query_tokens:
            return chunks[:limit]
        scored = [
            (len(query_tokens & _tokenize(c["chunk_text"])), idx)
            for idx, c in enumerate(chunks)
        ]
        keep = sorted(scored, key=lambda x: (-x[0], x[1]))[:limit]
        keep_idx = sorted(idx for _, idx in keep)
        return [chunks[i] for i in keep_idx]
