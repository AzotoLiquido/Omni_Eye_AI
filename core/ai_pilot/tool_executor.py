"""
Tool Executor - Esecuzione sandboxed dei tool definiti nel registry

Tipi supportati:
  - filesystem  → lettura/lista file nel fs_root
  - python      → esecuzione snippet Python con timeout
  - shell       → comandi shell (disabilitato di default)
  - db          → query sulla memoria SQLite
"""

import os
import re
import shlex
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_loader import PilotConfig


class ToolResult:
    """Risultato dell'esecuzione di un tool"""

    def __init__(self, tool_id: str, success: bool, output: str, error: str = ""):
        self.tool_id = tool_id
        self.success = success
        self.output = output
        self.error = error
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "tool_id": self.tool_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"ERRORE [{self.tool_id}]: {self.error}"


class ToolExecutor:
    """Registry e esecuzione sandboxed dei tool"""

    def __init__(self, cfg: PilotConfig):
        self.cfg = cfg

        # Risolvi fs_root relativo alla root del progetto
        base_dir = Path(__file__).resolve().parent.parent.parent
        self._fs_root = (base_dir / cfg.sandbox_fs_root.lstrip("./")).resolve()
        self._fs_root.mkdir(parents=True, exist_ok=True)

        self._timeout_s = cfg.tool_timeout_ms / 1000.0

        # Registro dei handler (tool_type → metodo)
        self._handlers = {
            "filesystem": self._exec_filesystem,
            "python": self._exec_python,
            "shell": self._exec_shell,
            "db": self._exec_db,
        }

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def get_available_tools(self) -> List[Dict]:
        """Restituisce i tool abilitati con policy valida"""
        tools = []
        for t in self.cfg.get_enabled_tools():
            policy = self.cfg.get_tool_policy(t["id"])
            if policy != "never":
                tools.append(t)
        return tools

    def execute(self, tool_id: str, params: Dict) -> ToolResult:
        """
        Esegue un tool per ID con i parametri dati.

        Args:
            tool_id:  ID del tool (es. "fs", "py")
            params:   Parametri specifici del tool

        Returns:
            ToolResult con output o errore
        """
        tool_cfg = self.cfg.get_tool_config(tool_id)
        if not tool_cfg:
            return ToolResult(tool_id, False, "", f"Tool '{tool_id}' non trovato nel registry")

        if not tool_cfg.get("enabled", True):
            return ToolResult(tool_id, False, "", f"Tool '{tool_id}' è disabilitato")

        policy = self.cfg.get_tool_policy(tool_id)
        if policy == "never":
            return ToolResult(tool_id, False, "", f"Tool '{tool_id}' bloccato dalla policy")

        # Mappa ID config → tipo handler (il config può non avere "type")
        _ID_TYPE = {"fs": "filesystem", "py": "python", "sh": "shell", "db": "db"}
        tool_type = _ID_TYPE.get(tool_id, tool_cfg.get("type", tool_id))
        handler = self._handlers.get(tool_type)
        if not handler:
            return ToolResult(tool_id, False, "", f"Tipo tool '{tool_type}' non supportato")

        try:
            return handler(tool_id, tool_cfg, params)
        except Exception as e:
            return ToolResult(tool_id, False, "", f"Eccezione: {str(e)}")

    # ------------------------------------------------------------------
    # Handler per tipo
    # ------------------------------------------------------------------

    def _exec_filesystem(self, tool_id: str, tool_cfg: Dict, params: Dict) -> ToolResult:
        """
        Operazioni filesystem sandboxed.
        Azioni: read, list, write (se allow_write)
        """
        action = params.get("action", "read")
        target = params.get("path", ".")

        # Sanitizza il percorso: deve stare dentro fs_root
        resolved = self._resolve_safe_path(target)
        if resolved is None:
            return ToolResult(tool_id, False, "",
                              f"Percorso '{target}' fuori dalla sandbox ({self._fs_root})")

        if action == "list":
            return self._fs_list(tool_id, resolved)
        elif action == "read":
            return self._fs_read(tool_id, resolved)
        elif action == "write":
            pol = tool_cfg.get("policy")
            allow_write = pol.get("allow_write", False) if isinstance(pol, dict) else False
            if not allow_write:
                return ToolResult(tool_id, False, "", "Scrittura non consentita dalla policy")
            content = params.get("content", "")
            return self._fs_write(tool_id, resolved, content)
        else:
            return ToolResult(tool_id, False, "", f"Azione filesystem sconosciuta: {action}")

    # Pattern import/builtin pericolosi bloccati nella sandbox Python
    _DANGEROUS_PY = re.compile(
        r'\b(?:import|from)\s+(?:os|subprocess|shutil|socket|ctypes|signal|'
        r'multiprocessing|webbrowser|importlib)\b|'
        r'\b__import__\s*\(|'
        r'\bexec\s*\(|\beval\s*\(|'
        r'\bimportlib\b|\bgetattr\s*\(\s*__builtins__'
    )

    def _exec_python(self, tool_id: str, tool_cfg: Dict, params: Dict) -> ToolResult:
        """Esegue un frammento Python in subprocess isolato con timeout"""
        code = params.get("code", "")
        if not code.strip():
            return ToolResult(tool_id, False, "", "Nessun codice fornito")

        # Blocca import pericolosi
        if self._DANGEROUS_PY.search(code):
            return ToolResult(tool_id, False, "",
                              "Codice contiene import non consentiti nella sandbox")

        max_output = tool_cfg.get("parameters", {}).get("max_output_chars", 10000)

        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code],  # -I = isolated mode
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                cwd=str(self._fs_root),
                env=self._sandboxed_env(),
            )
            output = result.stdout[:max_output]
            if result.returncode != 0:
                return ToolResult(tool_id, False, output,
                                  self._sanitize_stderr(result.stderr))
            return ToolResult(tool_id, True, output)

        except subprocess.TimeoutExpired:
            return ToolResult(tool_id, False, "",
                              f"Timeout ({self._timeout_s}s) superato")
        except Exception as e:
            return ToolResult(tool_id, False, "", str(e))

    def _exec_shell(self, tool_id: str, tool_cfg: Dict, params: Dict) -> ToolResult:
        """Esecuzione comandi shell (richiede abilitazione esplicita)"""
        if not self.cfg.allow_shell_write:
            return ToolResult(tool_id, False, "",
                              "Comandi shell disabilitati dalla policy di sicurezza")

        command = params.get("command", "")
        if not command.strip():
            return ToolResult(tool_id, False, "", "Nessun comando fornito")

        # Blocca metacaratteri shell pericolosi
        _SHELL_META = re.compile(r'[;&|`$(){}\[\]<>!\\]')
        if _SHELL_META.search(command):
            return ToolResult(tool_id, False, "",
                              "Comando contiene metacaratteri shell non consentiti")

        # Allowlist: solo comandi esplicitamente autorizzati
        ALLOWED_COMMANDS = {
            "ls", "dir", "cat", "type", "echo", "pwd", "cd",
            "head", "tail", "wc", "find", "grep", "sort",
        }

        try:
            args = shlex.split(command)
        except ValueError as e:
            return ToolResult(tool_id, False, "", f"Errore parsing comando: {e}")

        if not args:
            return ToolResult(tool_id, False, "", "Nessun comando fornito")

        base_cmd = args[0].lower().rstrip('.exe')
        if base_cmd not in ALLOWED_COMMANDS:
            return ToolResult(tool_id, False, "",
                              f"Comando '{base_cmd}' non in allowlist. "
                              f"Comandi consentiti: {', '.join(sorted(ALLOWED_COMMANDS))}")

        try:
            result = subprocess.run(
                args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                cwd=str(self._fs_root),
                env=self._sandboxed_env(),
            )
            output = result.stdout
            if result.returncode != 0:
                return ToolResult(tool_id, False, output,
                                  self._sanitize_stderr(result.stderr))
            return ToolResult(tool_id, True, output)

        except subprocess.TimeoutExpired:
            return ToolResult(tool_id, False, "", f"Timeout ({self._timeout_s}s)")
        except Exception as e:
            return ToolResult(tool_id, False, "", str(e))

    def _exec_db(self, tool_id: str, tool_cfg: Dict, params: Dict) -> ToolResult:
        """
        Query sulla memoria SQLite (sola lettura).
        Usato internamente dal planner per cercare nella memoria.
        """
        query = params.get("query", "")
        action = params.get("action", "search")

        # Delega alla memory store se disponibile (verrà iniettata dal Pilot)
        if hasattr(self, "_memory_store") and self._memory_store:
            if action == "search":
                results = self._memory_store.retrieve(query)
                return ToolResult(tool_id, True, results or "Nessun risultato trovato")
            elif action == "add_fact":
                key = params.get("key", "")
                value = params.get("value", "")
                if key and value:
                    fid = self._memory_store.add_fact(key, value, source="ai_tool")
                    return ToolResult(tool_id, True, f"Fatto salvato con ID {fid}")
                return ToolResult(tool_id, False, "", "Chiave e valore richiesti")
        return ToolResult(tool_id, False, "", "Memory store non disponibile")

    def set_memory_store(self, store) -> None:
        """Inietta il riferimento alla MemoryStore per il tool db"""
        self._memory_store = store

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    def _resolve_safe_path(self, target: str) -> Optional[Path]:
        """Risolve un percorso assicurandosi che resti nella sandbox"""
        try:
            resolved = (self._fs_root / target).resolve()
            # Verifica che sia dentro fs_root
            if self._fs_root in resolved.parents or resolved == self._fs_root:
                return resolved
            return None
        except (ValueError, OSError):
            return None

    def _fs_list(self, tool_id: str, path: Path) -> ToolResult:
        if not path.exists():
            return ToolResult(tool_id, False, "", f"Percorso non trovato: {path}")
        if not path.is_dir():
            return ToolResult(tool_id, False, "", f"Non è una directory: {path}")

        entries = []
        for entry in sorted(path.iterdir()):
            rel = entry.relative_to(self._fs_root)
            suffix = "/" if entry.is_dir() else ""
            size = entry.stat().st_size if entry.is_file() else 0
            entries.append(f"{rel}{suffix}  ({size} bytes)" if not suffix else f"{rel}{suffix}")

        output = "\n".join(entries) if entries else "(directory vuota)"
        return ToolResult(tool_id, True, output)

    def _fs_read(self, tool_id: str, path: Path) -> ToolResult:
        if not path.exists():
            return ToolResult(tool_id, False, "", f"File non trovato: {path}")
        if not path.is_file():
            return ToolResult(tool_id, False, "", f"Non è un file: {path}")

        max_size = 50_000  # 50KB max lettura
        size = path.stat().st_size
        if size > max_size:
            return ToolResult(tool_id, False, "",
                              f"File troppo grande ({size} bytes, max {max_size})")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            return ToolResult(tool_id, True, content)
        except Exception as e:
            return ToolResult(tool_id, False, "", f"Errore lettura: {e}")

    def _fs_write(self, tool_id: str, path: Path, content: str) -> ToolResult:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(tool_id, True, f"Scritto {len(content)} bytes in {path.name}")
        except Exception as e:
            return ToolResult(tool_id, False, "", f"Errore scrittura: {e}")

    # ------------------------------------------------------------------
    # Sandbox env
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_stderr(stderr: str) -> str:
        """Tronca e rimuove path di sistema da stderr"""
        if not stderr:
            return ""
        msg = stderr.strip()
        msg = re.sub(r'(?i)[a-z]:\\[^\s"\']+', '<path>', msg)
        msg = re.sub(r'/(?:home|usr|tmp|var|etc)[^\s"\']*', '<path>', msg)
        return msg[:500]

    def _sandboxed_env(self) -> Dict[str, str]:
        """Ambiente di esecuzione ridotto per subprocess (approccio allowlist)"""
        _ALLOWED_ENV_KEYS = {
            "PATH", "HOME", "USERPROFILE", "LANG", "LC_ALL",
            "TMP", "TEMP", "TMPDIR", "SYSTEMROOT", "COMSPEC",
            "PYTHONPATH", "VIRTUAL_ENV", "NODE_PATH",
        }
        return {k: v for k, v in os.environ.items() if k.upper() in _ALLOWED_ENV_KEYS}
