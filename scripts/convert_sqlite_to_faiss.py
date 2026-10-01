"""Converts Qdrant SQLite vectors to FAISS IndexFlatIP and a lightweight metadata SQLite db."""

import pickle
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
from loguru import logger


def convert():
    src_db = Path("data/indices/qdrant_db/collection/medical_chunks/storage.sqlite")
    if not src_db.exists():
        logger.error(f"Source sqlite file not found: {src_db}")
        return

    out_dir = Path("data/indices")
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = out_dir / "dense_index.faiss"
    meta_db_path = out_dir / "chunks_meta.sqlite"

    logger.info(f"Opening source database: {src_db}")
    src_con = sqlite3.connect(src_db)
    src_cur = src_con.cursor()

    total = src_cur.execute("SELECT count(*) FROM points;").fetchone()[0]
    logger.info(f"Total points to convert: {total:,}")

    # Setup target metadata DB
    if meta_db_path.exists():
        meta_db_path.unlink()
    dst_con = sqlite3.connect(meta_db_path)
    dst_cur = dst_con.cursor()
    dst_cur.execute("PRAGMA synchronous = OFF;")
    dst_cur.execute("PRAGMA journal_mode = MEMORY;")
    dst_cur.execute("""
        CREATE TABLE chunks (
            idx INTEGER PRIMARY KEY,
            doc_id TEXT,
            chunk_id TEXT,
            chunk_text TEXT,
            title TEXT,
            lang TEXT
        );
    """)

    index = faiss.IndexFlatIP(1024)
    batch_size = 20000
    batch_vecs = []
    batch_meta = []

    t0 = time.time()
    current_idx = 0

    src_cur.execute("SELECT point FROM points;")
    for row in src_cur:
        try:
            p = pickle.loads(row[0])
            batch_vecs.append(p.vector["dense"])
            payload = p.payload or {}
            batch_meta.append((
                current_idx,
                str(payload.get("doc_id", "")),
                str(payload.get("chunk_id", "")),
                payload.get("chunk_text", ""),
                payload.get("title", ""),
                payload.get("lang", "en"),
            ))
            current_idx += 1
        except Exception:
            continue

        if len(batch_vecs) >= batch_size:
            arr = np.array(batch_vecs, dtype=np.float32)
            faiss.normalize_L2(arr)
            index.add(arr)
            dst_cur.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?);", batch_meta)
            dst_con.commit()
            logger.info(
                f"Converted {current_idx:,}/{total:,} points ({current_idx / total * 100:.1f}%)..."
            )
            batch_vecs = []
            batch_meta = []

    if batch_vecs:
        arr = np.array(batch_vecs, dtype=np.float32)
        faiss.normalize_L2(arr)
        index.add(arr)
        dst_cur.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?);", batch_meta)
        dst_con.commit()

    logger.info(f"Saving FAISS index to {faiss_path}...")
    faiss.write_index(index, str(faiss_path))
    dst_con.close()
    src_con.close()

    elapsed = time.time() - t0
    logger.info(f"Done! Converted {index.ntotal:,} vectors in {elapsed:.1f}s.")


if __name__ == "__main__":
    convert()
