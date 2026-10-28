"""Unit tests for the SQLite chunk metadata store."""

import sqlite3

from src.retrieval.chunk_meta import ChunkMetaStore


def _conn_with_idx() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE chunks (idx INTEGER PRIMARY KEY, doc_id TEXT, chunk_id TEXT, "
        "chunk_text TEXT, title TEXT, lang TEXT)"
    )
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        [
            (0, "10", "10__c0", "alpha beta gamma", "T1", "vi"),
            (1, "10", "10__c1", "delta epsilon", "T1", "vi"),
            (2, "11", "11__c0", "zeta eta", "T2", "en"),
        ],
    )
    conn.commit()
    return conn


def test_detects_position_column_and_fetches_by_position():
    store = ChunkMetaStore(_conn_with_idx())
    assert store.pos_col == "idx"

    meta = store.get_by_positions([1])
    assert meta[1]["chunk_id"] == "10__c1"
    assert meta[1]["chunk_text"] == "delta epsilon"


def test_position_of_resolves_chunk_id_via_doc_ranges():
    store = ChunkMetaStore(_conn_with_idx())
    assert store.position_of("10__c0") == 0
    assert store.position_of("10__c1") == 1
    assert store.position_of("11__c0") == 2
    assert store.position_of("99__c0") is None


def test_get_doc_chunks_returns_all_chunks_with_context():
    store = ChunkMetaStore(_conn_with_idx())
    chunks = store.get_doc_chunks(["10"])
    assert [c["chunk_id"] for c in chunks] == ["10__c0", "10__c1"]
    assert chunks[0]["contextual_text"] == "Tiêu đề: T1\nNội dung: alpha beta gamma"


def test_get_doc_chunks_caps_long_docs_by_query_overlap():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE chunks (idx INTEGER PRIMARY KEY, doc_id TEXT, chunk_id TEXT, "
        "chunk_text TEXT, title TEXT, lang TEXT)"
    )
    rows = [(i, "7", f"7__c{i}", f"filler {i}", "T", "vi") for i in range(5)]
    rows.append((5, "7", "7__c5", "kidney stone treatment", "T", "vi"))
    conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()

    store = ChunkMetaStore(conn)
    chunks = store.get_doc_chunks(["7"], query="kidney stone", max_per_doc=1)
    assert len(chunks) == 1
    assert chunks[0]["chunk_text"] == "kidney stone treatment"


def test_supports_schema_with_positions_in_chunk_id():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, doc_id TEXT, "
        "chunk_text TEXT, title TEXT, lang TEXT)"
    )
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
        [
            (0, "20", "first passage", "T", "vi"),
            (1, "20", "second passage", "T", "vi"),
        ],
    )
    conn.commit()

    store = ChunkMetaStore(conn)
    assert store.pos_col == "chunk_id"
    assert [c["chunk_text"] for c in store.get_by_positions([0, 1]).values()] == [
        "first passage",
        "second passage",
    ]
    assert [c["chunk_text"] for c in store.get_doc_chunks(["20"])] == [
        "first passage",
        "second passage",
    ]


def _conn_slim() -> sqlite3.Connection:
    """Mirrors chunks_meta_slim.sqlite: numeric chunk ids, integer docs, doc_id index."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, doc_id INTEGER, title TEXT, "
        "url TEXT, chunk_text TEXT, lang TEXT)"
    )
    conn.execute("CREATE INDEX idx_chunks_doc_id ON chunks (doc_id)")
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        [
            (0, 500, "T1", "http://a", "first passage", "vi"),
            (1, 500, "T1", "http://a", "second passage", "vi"),
            (2, 501, "T2", "http://b", "other doc", "en"),
        ],
    )
    conn.commit()
    return conn


def test_slim_layout_uses_doc_index_without_building_ranges():
    store = ChunkMetaStore(_conn_slim())
    assert store.pos_col == "chunk_id"
    assert store._doc_indexed is True

    chunks = store.get_doc_chunks([500, "500"])
    assert [c["chunk_text"] for c in chunks] == [
        "first passage",
        "second passage",
        "first passage",
        "second passage",
    ]
    assert chunks[0]["url"] == "http://a"
    assert chunks[0]["contextual_text"] == "Tiêu đề: T1\nNội dung: first passage"
    assert store._ranges is None


def test_slim_layout_position_of_numeric_ids_needs_no_scan():
    store = ChunkMetaStore(_conn_slim())
    assert store.position_of(2) == 2
    assert store.position_of("1") == 1
    assert store._chunk_id_positions is None
    by_id = store.get_by_chunk_ids([2, 0])
    assert by_id["2"]["chunk_text"] == "other doc"
    assert by_id["0"]["chunk_text"] == "first passage"
