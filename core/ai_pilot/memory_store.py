"""
Memory Store - Memoria strutturata SQLite con FTS5 per il Pilot

Tre collezioni principali:
  - facts     → coppie chiave/valore che l'AI apprende dall'utente
  - tasks     → attività tracciate
  - documents → chunk di documenti indicizzati per retrieval
"""

import atexit
import sqlite3
import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from .config_loader import PilotConfig


class MemoryStore:
    """Memoria persistente SQLite con ricerca full-text (FTS5)"""

    def __init__(self, cfg: PilotConfig):
        self.cfg = cfg

        # Risolvi il percorso relativo rispetto alla root del progetto
        base_dir = Path(__file__).resolve().parent.parent.parent
        db_path = base_dir / cfg.memory_storage_path.lstrip("./")
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()  # P1-13: RLock for safe nested reads
        self._connect()
        self._init_tables()
        # P2: Use weak reference to avoid preventing GC
        import weakref
        _weak_self = weakref.ref(self)
        def _atexit_close():
            obj = _weak_self()
            if obj is not None:
                obj.close()
        atexit.register(_atexit_close)

    # ------------------------------------------------------------------
    # Connessione
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _init_tables(self) -> None:
        """Crea le tabelle e gli indici FTS5"""
        c = self._conn

        # --- facts ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                key       TEXT    NOT NULL,
                value     TEXT    NOT NULL,
                source    TEXT    DEFAULT '',
                score     REAL    DEFAULT 1.0,
                created_at TEXT   NOT NULL,
                updated_at TEXT   NOT NULL
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key)
        """)

        # FTS5 virtuale per facts
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
            USING fts5(key, value, content=facts, content_rowid=id)
        """)
        # Trigger per sincronizzare FTS5
        c.executescript("""
            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, key, value)
                VALUES (new.id, new.key, new.value);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, key, value)
                VALUES ('delete', old.id, old.key, old.value);
            END;
            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, key, value)
                VALUES ('delete', old.id, old.key, old.value);
                INSERT INTO facts_fts(rowid, key, value)
                VALUES (new.id, new.key, new.value);
            END;
        """)

        # --- tasks ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                title     TEXT    NOT NULL,
                status    TEXT    NOT NULL DEFAULT 'open',
                due_at    TEXT,
                payload   TEXT    DEFAULT '{}',
                created_at TEXT   NOT NULL,
                updated_at TEXT   NOT NULL
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
        """)

        # --- documents (chunks) ---
        c.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                path       TEXT    NOT NULL,
                chunk_idx  INTEGER NOT NULL DEFAULT 0,
                content    TEXT    NOT NULL,
                tags       TEXT    DEFAULT '[]',
                created_at TEXT    NOT NULL
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_docs_path ON documents(path)
        """)

        # FTS5 per documents
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
            USING fts5(path, content, content=documents, content_rowid=id)
        """)
        c.executescript("""
            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, path, content)
                VALUES (new.id, new.path, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, path, content)
                VALUES ('delete', old.id, old.path, old.content);
            END;
        """)

        c.commit()

    # ==================================================================
    # FACTS
    # ==================================================================

    def add_fact(self, key: str, value: str, source: str = "") -> int:
        """Aggiunge o aggiorna un fatto. Restituisce l'ID."""
        now = datetime.now().isoformat()

        with self._lock:
            # Se esiste già un fatto con la stessa chiave, aggiorna
            existing = self._conn.execute(
                "SELECT id FROM facts WHERE key = ?", (key,)
            ).fetchone()

            if existing:
                self._conn.execute(
                    "UPDATE facts SET value=?, source=?, updated_at=? WHERE id=?",
                    (value, source, now, existing["id"])
                )
                self._conn.commit()
                return existing["id"]

            cur = self._conn.execute(
                "INSERT INTO facts (key, value, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, value, source, now, now)
            )
            self._conn.commit()
            return cur.lastrowid

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Escape special FTS5 characters to prevent query injection.
        
        P1-14/P1-15: FTS5 operators like *, OR, NOT, NEAR, + can cause
        OperationalError. We quote each term to disable operators.
        """
        if not query or not query.strip():
            return '""'
        # Remove FTS5 special chars and wrap each word in double quotes
        words = query.split()
        # Escape double quotes within words, then quote each word
        safe_words = []
        for w in words:
            w = w.replace('"', '')
            w = w.strip('*+-~^')
            if w:
                safe_words.append(f'"{w}"')
        return ' '.join(safe_words) if safe_words else '""'

    def get_fact(self, key: str) -> Optional[Dict]:
        """Recupera un fatto per chiave esatta"""
        with self._lock:  # P1-13: lock reads too
            row = self._conn.execute(
                "SELECT * FROM facts WHERE key = ?", (key,)
            ).fetchone()
        return dict(row) if row else None

    def search_facts(self, query: str, limit: int = None) -> List[Dict]:
        """Ricerca full-text tra i fatti"""
        limit = limit or self.cfg.retrieval_top_k
        safe_query = self._sanitize_fts_query(query)
        with self._lock:  # P1-13: lock reads
            try:
                rows = self._conn.execute(
                    "SELECT f.*, rank FROM facts_fts "
                    "JOIN facts f ON facts_fts.rowid = f.id "
                    "WHERE facts_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (safe_query, limit)
                ).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                # Fallback: ricerca LIKE
                return self._search_facts_like(query, limit)

    def _search_facts_like(self, query: str, limit: int) -> List[Dict]:
        pattern = f"%{query}%"
        with self._lock:  # P1-13: lock reads
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE key LIKE ? OR value LIKE ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (pattern, pattern, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_facts(self) -> List[Dict]:
        """Restituisce tutti i fatti"""
        with self._lock:  # P1-13: lock reads
            rows = self._conn.execute(
                "SELECT * FROM facts ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_fact(self, fact_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            self._conn.commit()
        return cur.rowcount > 0

    # ==================================================================
    # TASKS
    # ==================================================================

    def add_task(self, title: str, due_at: str = None, payload: dict = None) -> int:
        """Crea un nuovo task"""
        now = datetime.now().isoformat()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tasks (title, status, due_at, payload, created_at, updated_at) "
                "VALUES (?, 'open', ?, ?, ?, ?)",
                (title, due_at, json.dumps(payload or {}), now, now)
            )
            self._conn.commit()
        return cur.lastrowid

    def update_task_status(self, task_id: int, status: str) -> bool:
        now = datetime.now().isoformat()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE id=?",
                (status, now, task_id)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def get_open_tasks(self) -> List[Dict]:
        with self._lock:  # P1-13: lock reads
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE status = 'open' ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_tasks(self) -> List[Dict]:
        with self._lock:  # P1-13: lock reads
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_task(self, task_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            self._conn.commit()
        return cur.rowcount > 0

    # ==================================================================
    # DOCUMENTS
    # ==================================================================

    def add_document(self, path: str, content: str, tags: list = None) -> List[int]:
        """
        Indicizza un documento suddividendolo in chunk.
        Restituisce gli ID dei chunk inseriti.
        Usa transazione esplicita per evitare inserimenti parziali.
        """
        chunks = self._chunk_text(content)
        ids = []
        now = datetime.now().isoformat()

        with self._lock:
            try:
                self._conn.execute("BEGIN")
                # Rimuovi chunk precedenti dello stesso documento
                self._conn.execute("DELETE FROM documents WHERE path = ?", (path,))

                for idx, chunk in enumerate(chunks):
                    cur = self._conn.execute(
                        "INSERT INTO documents (path, chunk_idx, content, tags, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (path, idx, chunk, json.dumps(tags or []), now)
                    )
                    ids.append(cur.lastrowid)

                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return ids

    def search_documents(self, query: str, limit: int = None) -> List[Dict]:
        """Ricerca full-text nei documenti"""
        limit = limit or self.cfg.retrieval_top_k
        safe_query = self._sanitize_fts_query(query)
        with self._lock:  # P1-13: lock reads
            try:
                rows = self._conn.execute(
                    "SELECT d.*, rank FROM documents_fts "
                    "JOIN documents d ON documents_fts.rowid = d.id "
                    "WHERE documents_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (safe_query, limit)
                ).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                pattern = f"%{query}%"
                rows = self._conn.execute(
                    "SELECT * FROM documents WHERE content LIKE ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (pattern, limit)
                ).fetchall()
                return [dict(r) for r in rows]

    def delete_document(self, path: str) -> int:
        """Rimuove tutti i chunk di un documento"""
        with self._lock:
            cur = self._conn.execute("DELETE FROM documents WHERE path = ?", (path,))
            self._conn.commit()
        return cur.rowcount

    # ==================================================================
    # RETRIEVAL UNIFICATO
    # ==================================================================

    def retrieve(self, query: str, top_k: int = None) -> str:
        """
        Retrieval unificato: cerca in facts + documents, restituisce
        un blocco di testo formattato da iniettare nel system prompt.
        """
        top_k = top_k or self.cfg.retrieval_top_k

        parts: List[str] = []

        # Cerca nei fatti
        facts = self.search_facts(query, limit=top_k // 2 or 4)
        if facts:
            facts_text = "\n".join(
                f"  • {f['key']}: {f['value']}" for f in facts
            )
            parts.append(f"Fatti noti:\n{facts_text}")

        # Cerca nei documenti
        docs = self.search_documents(query, limit=top_k // 2 or 4)
        if docs:
            docs_text = "\n".join(
                f"  [{d['path']} chunk {d['chunk_idx']}] {d['content'][:300]}"
                for d in docs
            )
            parts.append(f"Documenti rilevanti:\n{docs_text}")

        # Task aperti (sempre inclusi se presenti)
        tasks = self.get_open_tasks()
        if tasks:
            tasks_text = "\n".join(
                f"  [{t['id']}] {t['title']} (stato: {t['status']})"
                for t in tasks[:5]
            )
            parts.append(f"Task aperti:\n{tasks_text}")

        return "\n\n".join(parts) if parts else ""

    # ==================================================================
    # Utility
    # ==================================================================

    def _chunk_text(self, text: str) -> List[str]:
        """Suddivide il testo in chunk con overlap"""
        max_chars = self.cfg.chunking_max_chars
        overlap = self.cfg.chunking_overlap

        # P2: Guard against max_chars=0 or invalid values
        if max_chars <= 0:
            max_chars = 2000

        # Guardia contro loop infinito
        # P2-6 fix: cap overlap a metà chunk per evitare progresso troppo lento
        if overlap >= max_chars:
            overlap = max(0, max_chars // 2)

        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunk = text[start:end]
            # P2: Try to split at word boundary instead of mid-word
            if end < len(text) and chunk and not chunk[-1].isspace() and not text[end:end+1].isspace():
                # Look back for a space in last 20% of chunk
                boundary = chunk.rfind(' ', int(max_chars * 0.8))
                if boundary > 0:
                    chunk = chunk[:boundary]
                    end = start + boundary
            chunks.append(chunk)
            start = end - overlap

        return chunks

    def get_stats(self) -> Dict:
        """Statistiche sulla memoria"""
        with self._lock:  # P1-13: lock reads
            facts_count = self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            tasks_count = self._conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            docs_count = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return {
            "facts": facts_count,
            "tasks": tasks_count,
            "document_chunks": docs_count,
            "db_path": self._db_path,
        }

    def close(self) -> None:
        with self._lock:  # P3: lock close
            if self._conn:
                self._conn.close()
                self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
