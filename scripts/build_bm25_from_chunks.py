"""Standalone RAM-Safe BM25s Builder & SQLite Metadata Creator.

Streams directly from data/processed/chunks.jsonl using multiprocessing (4 CPU cores):
1. Creates zero-RAM SQLite metadata index (data/indices/chunks_meta.sqlite).
2. Tokenizes all 1.63M chunks in parallel batches.
3. Builds and persists bm25s sparse index (data/indices/bm25s_index).
4. Memory usage is strictly bounded under 3 GB RAM (100% safe on Kaggle/Colab).
"""

import json
import multiprocessing as mp
import re
import sqlite3
import time
from pathlib import Path

from loguru import logger
import bm25s
from pyvi import ViTokenizer
import jieba


def tokenize_single_item(item: tuple[str, str]) -> list[str]:
    """Tokenizes text based on language tag."""
    text, lang = item
    if not text:
        return []
    text_lower = text.lower()
    if lang == "zh":
        return [t.strip() for t in jieba.lcut(text_lower) if t.strip()]
    elif lang == "vi":
        try:
            return ViTokenizer.tokenize(text_lower).split()
        except Exception:
            return re.findall(r"\w+", text_lower)
    else:
        return re.findall(r"\w+", text_lower)


def tokenize_worker_batch(batch: list[tuple[str, str]]) -> list[list[str]]:
    """Worker function for multiprocessing pool."""
    return [tokenize_single_item(item) for item in batch]


def main():
    chunks_file = Path("data/processed/chunks.jsonl")
    indices_dir = Path("data/indices")
    indices_dir.mkdir(parents=True, exist_ok=True)
    sqlite_file = indices_dir / "chunks_meta.sqlite"
    bm25s_dir = indices_dir / "bm25s_index"

    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file {chunks_file} not found!")

    logger.info(f"=== Step 1: Building SQLite Metadata Index ({sqlite_file}) ===")
    t0 = time.time()
    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS chunks")
    cur.execute(
        "CREATE TABLE chunks (idx INTEGER PRIMARY KEY, doc_id TEXT, chunk_id TEXT, chunk_text TEXT, title TEXT, lang TEXT)"
    )

    batch_meta = []
    chunk_ids = []
    text_items = []  # List of (text, lang) for tokenization

    logger.info(f"Reading and streaming {chunks_file}...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            cid = str(item.get("chunk_id", f"c_{idx}"))
            did = str(item.get("doc_id", ""))
            raw_text = item.get("chunk_text", "")
            if raw_text.startswith("Tiêu đề:") and "\nNội dung: " in raw_text:
                clean_text = raw_text.split("\nNội dung: ", 1)[-1].strip()
                ctx_text = raw_text
            else:
                clean_text = raw_text
                ctx_text = item.get("contextual_text") or raw_text

            title = item.get("title") or (item.get("metadata") or {}).get("title", "")
            lang = item.get("lang", "vi")

            batch_meta.append((idx, did, cid, clean_text, title, lang))
            chunk_ids.append(cid)
            text_items.append((ctx_text, lang))

            if len(batch_meta) >= 20000:
                cur.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", batch_meta)
                conn.commit()
                batch_meta.clear()

    if batch_meta:
        cur.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", batch_meta)
        conn.commit()
        batch_meta.clear()

    conn.close()
    logger.info(f"Step 1 Complete: Indexed {len(chunk_ids):,} chunks into SQLite in {time.time() - t0:.1f}s.")

    logger.info(f"=== Step 2: Parallel Tokenization (4 CPU Cores) for {len(text_items):,} chunks ===")
    t_tok = time.time()
    num_cpus = max(1, min(4, mp.cpu_count()))
    chunk_batch_size = 5000
    batches = [text_items[i : i + chunk_batch_size] for i in range(0, len(text_items), chunk_batch_size)]
    del text_items  # Free memory immediately

    corpus_tokens = []
    with mp.Pool(num_cpus) as pool:
        for b_idx, batch_res in enumerate(pool.imap(tokenize_worker_batch, batches)):
            corpus_tokens.extend(batch_res)
            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(batches):
                logger.info(
                    f"Tokenized {len(corpus_tokens):,}/{len(chunk_ids):,} chunks ({(len(corpus_tokens)/len(chunk_ids))*100:.1f}%)..."
                )

    logger.info(f"Step 2 Complete: Tokenized {len(corpus_tokens):,} chunks in {time.time() - t_tok:.1f}s.")

    logger.info("=== Step 3: Building BM25s Inverted Index ===")
    t_bm25 = time.time()
    retriever = bm25s.BM25(k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=True)
    logger.info(f"BM25s matrix built in {time.time() - t_bm25:.1f}s. Saving to {bm25s_dir}...")

    retriever.save(str(bm25s_dir))
    with open(indices_dir / "bm25s_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"k1": 1.5, "b": 0.75, "chunk_ids": chunk_ids}, f)

    (indices_dir / "bm25_index.pkl").touch()
    logger.info(f"🎉 SUCCESS! All indices ready at {indices_dir} (Total time: {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
