"""
Audit Logger - Logging strutturato JSONL per eventi, conversazioni e audit

Scrive su file JSONL (una riga JSON per evento) per efficienza e facilità di analisi.
"""

import json
import os
import atexit
import logging
import shutil
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_loader import PilotConfig


# Mappa livelli stringa → logging
_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}


class AuditLogger:
    """Logger strutturato JSONL per il Pilot con buffer e rotazione log"""

    _MAX_LOG_SIZE_MB = 10  # Rotazione automatica sopra questa soglia
    _BUFFER_SIZE = 20      # Righe accumulate prima di flush su disco

    def __init__(self, cfg: PilotConfig):
        self.cfg = cfg
        self._enabled = cfg.audit_enabled
        self._log_prompts = cfg.audit_log_prompts
        self._log_tool_io = cfg.audit_log_tool_io
        self._level = _LEVEL_MAP.get(cfg.log_level, logging.INFO)

        # Risolvi percorsi relativi alla root del progetto
        base_dir = Path(__file__).resolve().parent.parent.parent
        self._events_path = base_dir / cfg.log_events_path.lstrip("./")
        self._conversations_path = base_dir / cfg.log_conversations_path.lstrip("./")

        # Crea directory
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        self._conversations_path.parent.mkdir(parents=True, exist_ok=True)

        # Buffer di scrittura (protetto da RLock per thread safety)
        # P1-12: Use RLock instead of Lock to prevent deadlock when
        # _flush_buffer is called from flush() which already holds the lock
        self._buf_lock = threading.RLock()
        self._buffers: Dict[Path, List[str]] = {
            self._events_path: [],
            self._conversations_path: [],
        }

        # Flush automatico alla chiusura del processo
        # P2: Use weak reference approach to avoid preventing GC
        import weakref
        _weak_self = weakref.ref(self)
        def _atexit_flush():
            obj = _weak_self()
            if obj is not None:
                obj.flush()
        atexit.register(_atexit_flush)

        # Setup logger Python standard (per console)
        self._logger = logging.getLogger("ai_pilot")
        self._logger.setLevel(self._level)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(self._level)
            fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, data: Dict = None, level: str = "info") -> None:
        """Scrive un evento generico nel log eventi"""
        if not self._enabled:
            return

        event = {
            "ts": datetime.now().isoformat(),
            "type": event_type,
            "level": level,
            "data": data or {},
        }

        self._write_jsonl(self._events_path, event)
        self._console_log(level, f"[{event_type}] {json.dumps(data or {}, ensure_ascii=False)[:200]}")

    def log_conversation_turn(
        self,
        conv_id: str,
        role: str,
        content: str,
        metadata: Dict = None,
    ) -> None:
        """Logga un turno di conversazione"""
        if not self._enabled:
            return

        entry = {
            "ts": datetime.now().isoformat(),
            "conv_id": conv_id,
            "role": role,
            "metadata": metadata or {},
        }

        # Logga il contenuto solo se log_prompts è abilitato
        if self._log_prompts:
            entry["content"] = content
        else:
            entry["content_length"] = len(content)

        self._write_jsonl(self._conversations_path, entry)

    def log_tool_call(
        self,
        tool_id: str,
        params: Dict,
        result_success: bool,
        result_output: str = "",
        result_error: str = "",
    ) -> None:
        """Logga l'invocazione di un tool"""
        if not self._enabled or not self._log_tool_io:
            return

        entry = {
            "ts": datetime.now().isoformat(),
            "type": "tool_call",
            "tool_id": tool_id,
            "params": params,
            "success": result_success,
        }

        if self._log_tool_io:
            entry["output_preview"] = result_output[:500] if result_output else ""
            entry["error"] = result_error

        self._write_jsonl(self._events_path, entry)
        self._console_log(
            "info" if result_success else "warn",
            f"Tool [{tool_id}] → {'OK' if result_success else 'ERRORE: ' + result_error[:100]}"
        )

    def log_plan_step(self, step_data: Dict) -> None:
        """Logga un passo del planner ReAct"""
        if not self._enabled:
            return

        entry = {
            "ts": datetime.now().isoformat(),
            "type": "plan_step",
            "step": step_data,
        }
        self._write_jsonl(self._events_path, entry)

    def log_memory_op(self, operation: str, details: Dict) -> None:
        """Logga un'operazione sulla memoria"""
        self.log_event(f"memory_{operation}", details, level="debug")

    def log_error(self, message: str, exception: Exception = None) -> None:
        """Logga un errore"""
        data = {"message": message}
        if exception:
            data["exception"] = str(exception)
            data["type"] = type(exception).__name__
        self.log_event("error", data, level="error")

    def log_startup(self, config_summary: Dict) -> None:
        """Logga l'avvio del Pilot"""
        self.log_event("pilot_startup", config_summary, level="info")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_jsonl(self, path: Path, entry: Dict) -> None:
        """Accoda una riga al buffer; flush su disco quando il buffer è pieno"""
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with self._buf_lock:
            buf = self._buffers.setdefault(path, [])
            buf.append(line)
            if len(buf) >= self._BUFFER_SIZE:
                self._flush_buffer(path)

    def _flush_buffer(self, path: Path) -> None:
        """Scrive il buffer accumulato su disco in un'unica operazione I/O"""
        buf = self._buffers.get(path)
        if not buf:
            return
        # Sposta le righe fuori dal buffer prima di scrivere;
        # in caso di errore le rimettiamo al loro posto.
        pending = list(buf)
        buf.clear()
        try:
            self._maybe_rotate(path)
            with open(path, "a", encoding="utf-8") as f:
                f.writelines(pending)
        except Exception as e:
            # Ripristina le righe non scritte nel buffer
            buf.extend(pending)
            self._logger.error("Errore scrittura log %s: %s", path, e)

    def flush(self) -> None:
        """Forza il flush di tutti i buffer (chiamare prima di shutdown)"""
        with self._buf_lock:
            for path in list(self._buffers):
                self._flush_buffer(path)

    def _maybe_rotate(self, path: Path) -> None:
        """Ruota il file di log se supera _MAX_LOG_SIZE_MB"""
        if not path.exists():
            return
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < self._MAX_LOG_SIZE_MB:
            return
        rotated = path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        try:
            shutil.move(str(path), str(rotated))
            self._logger.info("Log ruotato: %s -> %s", path.name, rotated.name)
        except Exception as e:
            self._logger.error("Errore rotazione log: %s", e)

    def _console_log(self, level: str, message: str) -> None:
        """Log su console via logging standard"""
        log_level = _LEVEL_MAP.get(level, logging.INFO)
        if log_level >= self._level:
            self._logger.log(log_level, message)

    # ------------------------------------------------------------------
    # Utilità
    # ------------------------------------------------------------------

    def read_recent_events(self, n: int = 50) -> list:
        """Legge gli ultimi N eventi dal log"""
        return self._read_tail(self._events_path, n)

    def read_recent_conversations(self, n: int = 50) -> list:
        """Legge gli ultimi N turni di conversazione dal log"""
        return self._read_tail(self._conversations_path, n)

    def _read_tail(self, path: Path, n: int) -> list:
        """Legge le ultime N righe di un file JSONL in modo efficiente"""
        if not path.exists():
            return []

        try:
            # Usa deque con maxlen per leggere solo le ultime N righe
            # senza tenere l'intero file in memoria
            with open(path, "r", encoding="utf-8") as f:
                last_lines = deque(f, maxlen=n)

            entries = []
            for line in last_lines:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return entries
        except Exception:
            return []

    def get_stats(self) -> Dict:
        """Statistiche sui log"""
        def _count_lines(path: Path) -> int:
            if not path.exists():
                return 0
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return sum(1 for _ in f)
            except Exception:
                return 0

        return {
            "events_count": _count_lines(self._events_path),
            "conversations_count": _count_lines(self._conversations_path),
            "events_path": str(self._events_path),
            "conversations_path": str(self._conversations_path),
        }
