# doc_qa/metadata/logger.py
"""Durable SQLite audit log for every Q&A query."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMNS = [
    "query", "answer", "filename", "doc_id", "confidence_level",
    "top_chunk_page", "top_chunk_section", "top_reranker_score",
    "prompt_tokens", "completion_tokens", "latency_seconds",
    "timestamp", "model_deployment", "extraction_methods_used",
    "chunk_ids_used", "chunks_json",
]

_CREATE_SQL = (
    "CREATE TABLE IF NOT EXISTS query_log ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    + ", ".join(f"{c} TEXT" for c in _COLUMNS)
    + ")"
)


class QueryLogger:
    """Write and read Q&A audit records from a local SQLite database.

    SQLite is used rather than a flat file because it supports concurrent
    writes from multiple Streamlit sessions and allows future filtering of
    the audit trail without loading everything into memory.
    """

    def __init__(self, db_path: str = "data/query_log.db") -> None:
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_SQL)
            conn.commit()

    def log(self, record: Dict[str, Any]) -> None:
        """Insert one answer record into the audit log."""
        values = [str(record.get(c, "")) for c in _COLUMNS]
        placeholders = ", ".join("?" * len(_COLUMNS))
        sql = f"INSERT INTO query_log ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(sql, values)
                conn.commit()
        except Exception as exc:
            logger.error("Failed to log query: %s", exc)

    def get_recent(self, n: int = 50) -> pd.DataFrame:
        """Return the n most recent log entries as a DataFrame."""
        sql = f"SELECT * FROM query_log ORDER BY id DESC LIMIT {int(n)}"
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(sql, conn)
