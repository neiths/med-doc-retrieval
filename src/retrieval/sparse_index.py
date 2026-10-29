"""Multilingual BM25 sparse index using ultra-fast, memory-mapped bm25s."""

import json
import pickle
import re
from pathlib import Path
from typing import Any

import bm25s
import jieba
from loguru import logger
from pyvi import ViTokenizer

from src.ingestion.cleaner import detect_language
from src.retrieval.chunk_meta import ChunkMetaStore


RE_WORDS = re.compile(r"\w+")
RE_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize_multilingual(text: str, lang: str = "auto") -> list[str]:
    """Tokenizes text based on detected or specified language."""
    if not text:
        return []

    if lang == "zh" or (lang == "auto" and RE_CJK.search(text[:50])):
        # split()/join strips every Unicode space in C, and pairing the stripped string with
        # itself avoids materializing an intermediate character list before the bigrams.
        stripped = "".join(text.lower().split())
        tokens = list(stripped)
        tokens.extend([a + b for a, b in zip(stripped, stripped[1:])])
        return tokens
    return RE_WORDS.findall(text.lower())


class SparseIndex:
    """High-performance BM25 index over document chunks with zero-copy memory mapping."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.retriever: bm25s.BM25 | None = None
        self.chunk_ids: list[str] = []
        self.chunk_metadata: list[dict[str, Any]] = []
        self._sqlite_conn = None
        self.meta_store: ChunkMetaStore | None = None

    def build(self, chunks: list[dict[str, Any]], text_key: str = "chunk_text"):
        """Builds bm25s index from a list of chunk dicts in seconds."""
        logger.info(f"Building fast bm25s sparse index over {len(chunks):,} chunks...")
        corpus_tokens = []
        self.chunk_ids = []
        self.chunk_metadata = []

        for chunk in chunks:
            text = chunk.get("contextual_text") or chunk.get(text_key, "")
            lang = chunk.get("lang", "auto")
            tokens = tokenize_multilingual(text, lang=lang)
            corpus_tokens.append(tokens)
            self.chunk_ids.append(chunk["chunk_id"])

        self.retriever = bm25s.BM25(k1=self.k1, b=self.b)
        self.retriever.index(corpus_tokens)
        logger.info("bm25s index construction completed.")

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float, dict[str, Any]]]:
        """Searches BM25 index with a query string.

        Returns:
            List of tuples (chunk_id, bm25_score, chunk_metadata)
        """
        if self.retriever is None or not self.chunk_ids:
            return []

        query_tokens = tokenize_multilingual(query, lang="auto")
        if not query_tokens:
            return []

        k = min(top_k, len(self.chunk_ids))
        res_indices, scores = self.retriever.retrieve([query_tokens], k=k)

        results = []
        top_indices = res_indices[0].tolist()
        top_scores = scores[0].tolist()

        if self.meta_store is not None:
            ranked = [
                (idx, score)
                for idx, score in zip(top_indices, top_scores)
                if score > 0 and 0 <= idx < len(self.chunk_ids)
            ]
            cids = [str(self.chunk_ids[idx]) for idx, _ in ranked]
            by_cid = self.meta_store.get_by_chunk_ids(cids)
            for (idx, score), cid in zip(ranked, cids):
                meta = by_cid.get(cid)
                if meta is None:
                    meta = {
                        "doc_id": cid.split("__c")[0] if "__c" in cid else cid,
                        "chunk_id": cid,
                        "chunk_text": "",
                        "contextual_text": "",
                        "title": "",
                        "lang": "vi",
                        "url": "",
                    }
                results.append((meta["chunk_id"], float(score), meta))
        else:
            for idx, score in zip(top_indices, top_scores):
                if score <= 0 or idx < 0 or idx >= len(self.chunk_ids):
                    continue
                cid = self.chunk_ids[idx] if idx < len(self.chunk_ids) else str(idx)
                if idx < len(self.chunk_metadata) and self.chunk_metadata[idx]:
                    m = self.chunk_metadata[idx]
                    meta = {
                        "doc_id": m.get("doc_id", cid.split("_")[0] if "_" in str(cid) else str(cid)),
                        "chunk_id": m.get("chunk_id", cid),
                        "chunk_text": m.get("chunk_text", ""),
                        "title": m.get("title", ""),
                        "lang": m.get("lang", "vi"),
                    }
                else:
                    meta = {
                        "doc_id": cid.split("_")[0] if "_" in str(cid) else str(cid),
                        "chunk_id": cid,
                        "chunk_text": "",
                        "title": "",
                        "lang": "vi",
                    }
                results.append((cid, float(score), meta))

        return results

    def save(self, directory: Path | str):
        """Saves bm25s index and lightweight chunk metadata to disk."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        bm25s_dir = save_dir / "bm25s_index"
        if self.retriever is not None:
            self.retriever.save(str(bm25s_dir))

        meta_file = save_dir / "bm25s_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "chunk_ids": self.chunk_ids,
                },
                f,
                ensure_ascii=False,
            )
        # Create a small marker file so code expecting bm25_index.pkl knows sparse index is ready
        (save_dir / "bm25_index.pkl").touch()
        logger.info(f"Saved memory-efficient bm25s sparse index to {bm25s_dir}")

    @classmethod
    def load(
        cls,
        directory: Path | str,
        fallback_meta_dir: Path | str | None = None,
        chunks_meta_path: Path | str | None = None,
    ) -> "SparseIndex":
        """Loads sparse index from disk using zero-RAM memory mapping."""
        import sqlite3

        load_dir = Path(directory)
        bm25s_candidates = [
            load_dir,
            load_dir / "bm25s_index",
            load_dir / "bm25s_index" / "bm25s_index",
            load_dir.parent / "indices" / "bm25s_index",
            Path("data/indices/bm25s_index"),
            Path("/kaggle/working/data/indices/bm25s_index"),
            Path("/kaggle/working/med-doc-retrieval/data/indices/bm25s_index"),
        ]
        bm25s_dir = None
        for cand in bm25s_candidates:
            if cand.exists() and ((cand / "vocab.index.json").exists() or any(cand.glob("*.npy"))):
                bm25s_dir = cand
                break

        meta_file = load_dir / "bm25s_metadata.json"
        if not meta_file.exists() and bm25s_dir:
            if (bm25s_dir.parent / "bm25s_metadata.json").exists():
                meta_file = bm25s_dir.parent / "bm25s_metadata.json"
            elif (bm25s_dir / "bm25s_metadata.json").exists():
                meta_file = bm25s_dir / "bm25s_metadata.json"

        # Search for SQLite metadata (priority: explicit path -> chunks_meta_slim -> chunks_meta)
        sqlite_file = None
        if chunks_meta_path:
            p = Path(chunks_meta_path)
            if p.exists() and p.suffix in [".sqlite", ".db"]:
                sqlite_file = p

        if sqlite_file is None:
            sqlite_candidates = [
                load_dir / "chunks_meta_slim.sqlite",
                load_dir / "chunks_meta.sqlite",
            ]
            if bm25s_dir:
                sqlite_candidates.extend([
                    bm25s_dir.parent / "chunks_meta_slim.sqlite",
                    bm25s_dir.parent / "chunks_meta.sqlite",
                ])
            if fallback_meta_dir:
                fb = Path(fallback_meta_dir)
                sqlite_candidates.extend([
                    fb / "chunks_meta_slim.sqlite",
                    fb / "chunks_meta.sqlite",
                    fb.parent / "chunks_meta_slim.sqlite",
                    fb.parent / "chunks_meta.sqlite",
                ])
            sqlite_candidates.extend([
                Path("data/indices/chunks_meta_slim.sqlite"),
                Path("data/indices/chunks_meta.sqlite"),
                Path("/kaggle/working/data/indices/chunks_meta_slim.sqlite"),
                Path("/kaggle/working/data/indices/chunks_meta.sqlite"),
            ])
            if Path("/kaggle/input").exists():
                sqlite_candidates.extend(list(Path("/kaggle/input").rglob("chunks_meta*.sqlite")))

            for cand in sqlite_candidates:
                if cand.exists() and cand.stat().st_size > 0:
                    sqlite_file = cand
                    break

        pkl_file = load_dir / "bm25_index.pkl"

        if bm25s_dir is not None and bm25s_dir.exists():
            idx = cls()
            idx.retriever = bm25s.BM25.load(str(bm25s_dir), mmap=True)
            if meta_file.exists():
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                idx.k1 = meta.get("k1", 1.5)
                idx.b = meta.get("b", 0.75)
                idx.chunk_ids = meta.get("chunk_ids", [])
                idx.chunk_metadata = meta.get("chunk_metadata", [])

            if sqlite_file is not None and sqlite_file.exists():
                idx._sqlite_conn = sqlite3.connect(sqlite_file, check_same_thread=False)
                idx.meta_store = ChunkMetaStore(idx._sqlite_conn)
                idx._id_col = idx.meta_store.pos_col
                idx._has_url = idx.meta_store.has_url
                logger.info(f"Connected to SQLite metadata: {sqlite_file} (key column: {idx._id_col}, has_url: {idx._has_url})")
                if not idx.chunk_ids:
                    rows = idx._sqlite_conn.execute(
                        f"SELECT chunk_id FROM chunks ORDER BY {idx._id_col}"
                    ).fetchall()
                    idx.chunk_ids = [r[0] for r in rows]

            logger.info(f"Loaded bm25s sparse index ({len(idx.chunk_ids):,} chunks) via zero-copy mmap.")
            return idx

        # Backwards compatibility: fallback to rank_bm25 pickle if bm25s not found
        if pkl_file.exists() and pkl_file.stat().st_size > 0:
            with open(pkl_file, "rb") as f:
                payload = pickle.load(f)
            idx = cls(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))

            class RankBM25Adapter:
                def __init__(self, bm25_obj):
                    self.bm25_obj = bm25_obj

                def retrieve(self, token_batches, k):
                    import numpy as np

                    scores = self.bm25_obj.get_scores(token_batches[0])
                    top_idx = scores.argsort()[::-1][:k]
                    return [np.array(top_idx)], [np.array(scores[top_idx])]

            idx.retriever = RankBM25Adapter(payload["bm25"])
            idx.chunk_ids = payload.get("chunk_ids", [])
            idx.chunk_metadata = payload.get("chunk_metadata", [])
            logger.info(f"Loaded legacy BM25 sparse index with {len(idx.chunk_ids)} chunks.")
            return idx

        raise FileNotFoundError(f"No valid BM25 index found in {load_dir}")
